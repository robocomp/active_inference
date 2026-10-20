// Task 2 (local-motion policy): rollout + swept-body contact probability.
// Review fix round 1: rollouts ramp from the current velocity (v0, w0); cells carry their hits' scatter and are
//   swept as support segments (jamb_tip_not_overconfident, turn_in_place_from_speed_sees_wall_ahead).
// Review fix round 1 (Task 6): returns are DENOISED one by one (denoise_returns: each moved onto its local line along the
//   normal, anisotropic sigma); the shared-offset integral is adaptive (`epsilon_integral_accuracy` vs a 4001-node
//   reference, sigma_j/sigma_s in [0.05, 5]).
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_policy_contact.cpp src/local_policy/contact.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tpc && /tmp/tpc
#include "../src/local_policy/contact.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>

using namespace rc::policy;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static constexpr float kHalfPi = 1.57079633f;
static constexpr float kGeom = 0.01f;   // body_geometry_sigma_m (default at the call sites)
static BaseLimits limits() { BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 1.f; b.a_rot = 2.f; b.d_rot = 2.f; return b; }
static rc::RobotFootprint body(float margin = 0.05f) { auto f = rc::RobotFootprint::shadow(); f.set_safety_margin(margin); return f; }
// Wall of returns on the line x = x0 (room frame), y in [y0, y1]; robot yaw = pi/2 puts room +x on its right.
static std::vector<Return> wall_x(float x0, float y0, float y1, float sig = 0.01f)
{ std::vector<Return> r; for (float y = y0; y <= y1; y += 0.05f) r.push_back({{x0, y}, sig, 1.f}); return r; }
static std::vector<Return> wall_y(float y0, float x0, float x1, float sig = 0.01f)
{ std::vector<Return> r; for (float x = x0; x <= x1; x += 0.05f) r.push_back({{x, y0}, sig, 1.f}); return r; }

int main()
{
    const auto b = limits();
    const Pose2 origin{{0.f, 0.f}, kHalfPi};
    const float sxy = 0.005f, syaw = 0.005f;

    { // Live fix 2 ruling 1 — WHEEL-COUPLED: tourP3's reversal, back-left (v -0.26, w +0.25) -> forward-right (v 0.24, w -1.0),
      // at Shadow's limits (axle 0.518 m, a_lin 0.5 m/s^2). The two wheels ramp at the per-wheel rates: the left one
      // -0.325 -> +0.499 m/s through zero. Independent axes reached w = -1 in ~0.74 s; the base took 1.6-1.8 s.
        auto time_to_w = [&](const BaseLimits& lim, float w_target) {
            const auto s = rollout(origin, -0.26f, 0.25f, 0.24f, -1.0f, 3.f, lim, 0.01f);
            for (std::size_t i = 1; i < s.size(); ++i)
                if ((s[i].yaw - s[i - 1].yaw) / 0.01f <= 0.97f * w_target) return 0.01f * float(i);
            return 99.f; };
        BaseLimits sh; sh.v_max = 0.5f; sh.w_max = 1.5f; sh.a_lin = 0.5f; sh.d_lin = 1.0f; sh.track_m = 0.518f;
        sh.a_rot = sh.a_lin / 0.259f; sh.d_rot = sh.d_lin / 0.259f;
        const float t_cap = time_to_w(sh, -1.0f);
        BaseLimits sh_measured = sh; sh_measured.d_lin = 0.5f; sh_measured.d_rot = 0.5f / 0.259f;   // tourP3's body decel ~0.56
        const float t_meas = time_to_w(sh_measured, -1.0f);
        std::printf("  reversal to w=-1: %.2f s at the capability's limits (d_lin 1.0), %.2f s with the measured decel (0.5)\n", t_cap, t_meas);
        CHECK("wheel-coupled reversal: ~1.32 s at the capability's limits (independent axes: ~0.74 s)", std::abs(t_cap - 1.32f) < 0.06f);
        CHECK("wheel-coupled reversal: within the measured 1.6-1.8 s at the measured decel", t_meas >= 1.6f and t_meas <= 1.8f);
        // pure axes are unchanged: straight motion ramps at a_lin, an in-place turn at a_lin / (L/2)
        const auto st = rollout(origin, 0.f, 0.f, 0.3f, 0.f, 1.f, sh, 0.01f);
        CHECK("wheel-coupled: a straight command still ramps at a_lin (0.6 s to 0.3 m/s, then 0.4 s, + brake)",
              std::abs(st.back().p.y() - (0.09f + 0.12f + 0.045f)) < 0.005f);
    }
    { // rollout_straight: from rest, 0.3 m/s commanded for 1 s, brake at 1.0 m/s^2.
      // ★ review ruling: rollouts ramp from the current velocity — ramp 0.3 s (0.045 m) + 0.7 s at 0.3 (0.21 m) +
      //   brake (0.045 m) = 0.300 m (was 0.345 with an instant jump to the command).
        const auto s = rollout(origin, 0.f, 0.f, 0.3f, 0.f, 1.f, b, 0.05f);
        CHECK("rollout_straight advance", std::abs((s.back().p.y() - 0.f) - 0.300f) < 0.005f and std::abs(s.back().p.x()) < 1e-4f);
        CHECK("rollout_straight starts at start", (s.front().p - origin.p).norm() < 1e-6f);
        // final speed 0: the last step must not still be moving
        const float last = (s.back().p - s[s.size() - 2].p).norm();
        CHECK("rollout_straight final speed 0", last < 0.3f * 0.05f * 0.5f);
    }
    { // rollout_turn_in_place: 1 rad/s commanded for 1 s from rest, a_rot = d_rot = 2.
      // ★ ramp 0.5 s (0.25 rad) + 0.5 s at 1 rad/s (0.5) + brake (0.25) = 1.00 rad (was 1.25 with an instant jump).
        const auto s = rollout(origin, 0.f, 0.f, 0.f, 1.f, 1.f, b, 0.05f);
        const float dyaw = s.back().yaw - origin.yaw;
        CHECK("rollout_turn_in_place yaw", std::abs(dyaw - 1.0f) < 0.02f);
        CHECK("rollout_turn_in_place no translation", (s.back().p - origin.p).norm() < 1e-5f);
    }
    { // arc closes: v=0.3,w=0.5 commanded for 2 s lies on the circle of radius v/w once both WHEELS have ramped up
      // (Live fix 2: wheels v -+ w L/2 with L = 2 a_lin / a_rot = 1.0 m: 0.05 m/s at 0.05 s, 0.55 m/s at 0.55 s; dt 0.1 =>
      // from pose 6 to the end of the command at pose 20).
        const auto s = rollout(origin, 0.f, 0.f, 0.3f, 0.5f, 2.f, {b}, 0.1f);
        const Pose2& r0 = s[6];
        const Eigen::Vector2f c = r0.p + (0.3f / 0.5f) * Eigen::Vector2f(-std::sin(r0.yaw), std::cos(r0.yaw));
        // ★ only the steady part of the command phase is one circle: the two axes brake at different times afterwards
        float worst = 0; for (std::size_t i = 6; i < 21; ++i) { const auto& q = s[i]; worst = std::max(worst, std::abs((q.p - c).norm() - 0.6f)); }
        CHECK("rollout_arc stays on circle", worst < 1e-4f);
    }
    { // rollout_from_speed: already at 0.5 m/s, commanded 0.2 for 1 s: decelerate 0.3 s (0.105 m) + 0.7 s at 0.2
      // (0.14) + brake (0.02) = 0.265 m; and a sign change 0.5 -> -0.3: 0.5 s to stop (0.125) then 0.3 s to -0.3
      // (-0.045) + 0.2 s at -0.3 (-0.06) + brake (-0.045) = -0.025 m.
        const auto a = rollout(origin, 0.5f, 0.f, 0.2f, 0.f, 1.f, b, 0.05f), r = rollout(origin, 0.5f, 0.f, -0.3f, 0.f, 1.f, b, 0.05f);
        std::printf("  from speed: 0.5->0.2 advance %.4f (0.265) ; 0.5->-0.3 advance %.4f (-0.025)\n", a.back().p.y(), r.back().p.y());
        CHECK("rollout_from_speed decelerates into the command", std::abs(a.back().p.y() - 0.265f) < 0.003f);
        CHECK("rollout_from_speed through zero on a sign change", std::abs(r.back().p.y() + 0.025f) < 0.003f);
        const Pose2 adv = advance(origin, 0.5f, 0.f, 0.2f);
        CHECK("advance straight", std::abs(adv.p.y() - 0.1f) < 1e-5f and std::abs(adv.p.x()) < 1e-6f);
    }
    { // signed_distance agrees with RobotFootprint::contains_yaw (quarter-turn convention)
        // margin 0: exact agreement. margin > 0: ours is a ROUNDED (Euclidean) grow, contains() mitres the corners,
        // so ours must be inside-or-equal (sd < 0 implies contains).
        const auto f = body(0.f), fm = body(0.05f); std::mt19937 g(7); std::uniform_real_distribution<float> u(-0.6f, 0.6f), ang(-3.1f, 3.1f);
        int bad = 0, badm = 0;
        for (int i = 0; i < 2000; ++i)
        {
            const Pose2 p{{u(g), u(g)}, ang(g)}; const Eigen::Vector2f q{u(g), u(g)};
            const float sd = signed_distance(f, p, q);
            if (std::abs(sd) > 1e-4f and (sd < 0.f) != f.contains_yaw(q, p.p, p.yaw)) ++bad;
            if (signed_distance(fm, p, q) < 0.f and not fm.contains_yaw(q, p.p, p.yaw)) ++badm;
        }
        CHECK("signed_distance (margin) inside implies contains_yaw", badm == 0);
        CHECK("signed_distance matches contains_yaw", bad == 0);
        CHECK("signed_distance flank value", std::abs(signed_distance(fm, origin, {0.40f, 0.f}) - (0.40f - 0.284f - 0.05f)) < 1e-4f);
    }
    { // wall_beside_parallel_free
        const auto sw = rollout(origin, 0.f, 0.f, 0.3f, 0.f, 1.f, b, 0.05f);
        const auto r = wall_x(0.40f, -1.f, 2.f);
        const float lp = log_p_no_contact(body(), sw, r, sxy, syaw, kGeom);
        std::printf("  parallel lp=%.5f\n", lp);
        CHECK("wall_beside_parallel_free", lp > std::log(0.99f));
    }
    { // wall_beside_turn_hits
        const auto sw = rollout(origin, 0.f, 0.f, 0.f, 1.f, 1.f, b, 0.05f);
        const auto r = wall_x(0.34f, -1.f, 1.f);
        const float lp = log_p_no_contact(body(), sw, r, sxy, syaw, kGeom);
        std::printf("  turn lp=%.5f\n", lp);
        // ★ fix round 3: contact is against the BARE hull, whose corner reach (0.318 m) misses a 0.34 m wall by 2.2 cm
        //   (~1.5 sigma total). The turn is still CHARGED — ineligible at risk_eps 0.01 — but no longer near-certain,
        //   so the bar moves from P > 0.5 to P > 0.01 (the policy's chance constraint), same intent.
        CHECK("wall_beside_turn_hits", lp < std::log(1.f - 0.01f));
    }
    { // wall_ahead_fast_vs_slow
        const auto r = wall_y(0.8f, -1.f, 1.f);
        const float fast = log_p_no_contact(body(), rollout(origin, 0.f, 0.f, 0.35f, 0.f, 2.f, b, 0.05f), r, sxy, syaw, kGeom);
        const float slow = log_p_no_contact(body(), rollout(origin, 0.f, 0.f, 0.10f, 0.f, 2.f, b, 0.05f), r, sxy, syaw, kGeom);
        std::printf("  fast=%.4f slow=%.4f\n", fast, slow);
        CHECK("wall_ahead_fast_vs_slow", fast < slow and fast < std::log(0.5f) and slow > std::log(0.99f));
    }
    { // start_inside_only_separating: return 0.02 m inside the right flank (x = 0.264)
        const std::vector<Return> r{{{0.264f, 0.f}, 0.01f, 1.f}};
        const std::vector<Pose2> away{origin, {{-0.2f, 0.f}, kHalfPi}};   // body moves room -x = its left: flank leaves the point
        const std::vector<Pose2> into{origin, {{0.2f, 0.f}, kHalfPi}};    // body moves right: point sinks deeper
        CHECK("start_inside separating is free", log_p_no_contact(body(), away, r, sxy, syaw, kGeom) > std::log(0.99f));
        CHECK("start_inside deeper hits", log_p_no_contact(body(), into, r, sxy, syaw, kGeom) < std::log(0.5f));
    }
    { // conditional_continuous_at_surface: a return on the BARE nose line (y = 0.23) at sd0 = -1 mm, 0, +1 mm; the
      // body advances 5 mm into it (small enough to keep P mid-range, where a step would show).
      // ★ fix round 3: distances are to the BARE hull (was the margin-grown nose, y = 0.28).
        const std::vector<Pose2> sw{origin, {{0.f, 0.005f}, kHalfPi}};
        float P[3]; int i = 0;
        for (float d : {-0.001f, 0.f, 0.001f})
        {
            const std::vector<Return> r{{{0.f, 0.23f + d}, 0.01f, 1.f}};
            P[i++] = 1.f - std::exp(log_p_no_contact(body(), sw, r, sxy, syaw, kGeom));
        }
        std::printf("  P(-1mm)=%.4f P(0)=%.4f P(+1mm)=%.4f\n", P[0], P[1], P[2]);
        CHECK("conditional_continuous_at_surface", std::abs(P[0] - P[1]) < 0.05f and std::abs(P[1] - P[2]) < 0.05f and P[1] > 0.1f and P[1] < 0.9f);
    }
    { // backing_away_free: a wall exactly at the grown nose (y = 0.23 + 0.05), straight back
        const auto r = wall_y(0.28f, -0.4f, 0.4f);
        const float P = 1.f - std::exp(log_p_no_contact(body(), rollout(origin, 0.f, 0.f, -0.3f, 0.f, 1.f, b, 0.05f), r, sxy, syaw, kGeom));
        std::printf("  backing away from a wall at the margin edge: P=%.2e\n", P);
        CHECK("backing_away_free", P < 1e-3f);
    }
    { // turn_in_place_from_speed_sees_wall_ahead (review ruling): moving at 0.5 m/s, the command "turn in place"
      // still coasts 0.125 m (d_lin = 1) while turning, so the corner (reach 0.318) sweeps out to 0.443 m from the
      // start. ★ the wall is 0.15 m ahead of the nose (0.38 m from the centre): clear from rest by 6 cm (5 sigma_s),
      //   hit while coasting. (The ruling's 0.35 m ahead of the nose = 0.58 m is beyond the coast's reach at this
      //   deceleration: 0.125 + 0.318 = 0.443 < 0.58, so that geometry charges nothing and would not discriminate.)
        const auto r = wall_y(0.38f, -1.f, 1.f);
        const float moving = 1.f - std::exp(log_p_no_contact(body(), rollout(origin, 0.5f, 0.f, 0.f, 1.f, 1.f, b, 0.05f), r, sxy, syaw, kGeom));
        const float rest = 1.f - std::exp(log_p_no_contact(body(), rollout(origin, 0.f, 0.f, 0.f, 1.f, 1.f, b, 0.05f), r, sxy, syaw, kGeom));
        std::printf("  turn in place with a wall 0.35 m ahead: P(moving at 0.5)=%.4f P(from rest)=%.4f\n", moving, rest);
        CHECK("turn_in_place_from_speed_sees_wall_ahead", moving > 0.01f and rest < 0.01f);
    }
    { // jamb_tip_not_overconfident (review ruling): four hits spread along a jamb tip END (x = 0.30..0.33 at
      // y = 0.60, 1 cm apart — two 0.04 m cells), body threading forward past the tip with its flank (0.284) 1.6 cm
      // from the extreme hit. The cells must not be surer than that extreme hit alone with sigma_r.
      // ★ review fix round 1: denoising moves each hit along the fitted line's NORMAL only (y here), so the extreme hit
      //   stays at x = 0.30 and keeps its raw sigma ALONG the line (x = the clearance direction for the flank).
        std::vector<Return> tip; for (float x : {0.30f, 0.31f, 0.32f, 0.33f}) tip.push_back({{x, 0.60f}, 0.01f, 1.f});
        const std::vector<Return> extreme{{{0.30f, 0.60f}, 0.01f, 1.f}};
        const auto sw = rollout(origin, 0.f, 0.f, 0.5f, 0.f, 1.f, b, 0.05f);
        const float Pt = 1.f - std::exp(log_p_no_contact(body(), sw, tip, sxy, syaw, kGeom));
        const float Pe = 1.f - std::exp(log_p_no_contact(body(), sw, extreme, sxy, syaw, kGeom));
        const auto c = denoise_returns(tip);
        const auto ext = std::ranges::min_element(c, {}, [](const Return& q) { return q.seg.norm() > 0.f ? 9.f : q.p.x(); });
        std::printf("  jamb tip: P(4 hits denoised)=%.4f P(extreme hit alone)=%.4f ; %zu obstacles; end: (%.4f,%.4f) sigma_perp=%.4f sxx=%.2e syy=%.2e\n", Pt, Pe, c.size(), ext->p.x(), ext->p.y(), ext->sigma, ext->sxx, ext->syy);
        CHECK("jamb tip: the extreme hit is kept as an END, where it was (no pull along the surface)", ext->seg.norm() == 0.f and std::abs(ext->p.x() - 0.30f) < 1e-5f);
        CHECK("jamb_tip_not_overconfident", Pt >= Pe - 1e-3f);
        // and inside a flat wall the obstacle is tight and isotropic (the surface continues on both sides: its along-line
        // error does not matter); only its ENDS keep the raw sigma along the wall
        std::vector<Return> dense; for (float y = -1.f; y <= 2.f; y += 0.01f) dense.push_back({{0.40f, y}, 0.01f, 1.f});
        const auto flat = denoise_returns(dense);
        const auto& fc = *std::ranges::min_element(flat, {}, [](const Return& q) { return std::abs(q.p.y()) + (q.seg.norm() > 0.f ? 0.f : 9.f); });
        std::printf("  flat wall (1 cm hits, 3 m): %zu obstacles; a sample: (%.4f,%.4f) half-length %.3f sigma=%.4f\n", flat.size(), fc.p.x(), fc.p.y(), fc.seg.norm(), fc.sigma);
        const auto& end = *std::ranges::max_element(flat, {}, [](const Return& q) { return q.seg.norm() > 0.f ? -9.f : q.p.y(); });
        std::printf("  flat wall END: (%.4f,%.4f) sigma_perp=%.4f sxx=%.2e syy=%.2e\n", end.p.x(), end.p.y(), end.sigma, end.sxx, end.syy);
        CHECK("flat wall: samples on the wall, tight and isotropic; the ends keep sigma_r along it",
              std::abs(fc.p.x() - 0.40f) < 1e-4f and fc.sigma < 0.005f and fc.sxx == 0.f and std::abs(end.p.y() - 2.f) < 0.011f
              and std::abs(end.syy + end.sigma * end.sigma - 1e-4f) < 1e-7f and end.sxx < 1e-9f
              and flat.size() < dense.size() / 4);
    }
    { // p_exist scales the hit; a far return is ignored
        const std::vector<Return> hit{{{0.f, 0.3f}, 0.01f, 1.f}}, ghost{{{0.f, 0.3f}, 0.01f, 0.1f}}, far{{{50.f, 0.f}, 0.01f, 1.f}};
        const auto sw = rollout(origin, 0.f, 0.f, 0.3f, 0.f, 1.f, b, 0.05f);
        const float a = log_p_no_contact(body(), sw, hit, sxy, syaw, kGeom), g = log_p_no_contact(body(), sw, ghost, sxy, syaw, kGeom);
        CHECK("p_exist=0.1 hits ~10x less", g > a and std::abs(g - std::log(0.9f)) < 0.02f);
        CHECK("far return ignored", log_p_no_contact(body(), sw, far, sxy, syaw, kGeom) == 0.f);
    }
    { // denoise_returns (review fix round 1): every LiDAR return stays its own obstacle, moved along its local line's
      // normal; isolated ones untouched; another source (aggregate = false) passed through.
        std::vector<Return> r;
        for (int i = 0; i < 6; ++i) r.push_back({{0.300f + (i % 2 ? 0.004f : -0.004f), 0.01f * i}, 0.01f, 1.f});   // a noisy line x = 0.30
        r.push_back({{0.f, 0.50f}, 0.01f, 1.f});                                                                    // isolated
        r.push_back({{0.f, 0.235f}, 0.005f, 0.001f, 0.f, 0.f, 0.f, false});                                         // a hypothesis
        const auto c = denoise_returns(r);
        std::printf("  denoised: %zu of %zu\n", c.size(), r.size());
        for (const auto& q : c) std::printf("    (%.4f,%.4f) sigma=%.4f sxx=%.2e sxy=%.2e syy=%.2e p=%.3f agg=%d\n", q.p.x(), q.p.y(), q.sigma, q.sxx, q.sxy, q.syy, q.p_exist, int(q.aggregate));
        // every raw LiDAR return lies within c/2 of an obstacle's support (nothing observed is dropped)
        auto dist_to = [](const Return& o, const Eigen::Vector2f& x) { const Eigen::Vector2f a = o.p - o.seg, b = o.p + o.seg, ab = b - a;
            const float t = ab.squaredNorm() > 0.f ? std::clamp((x - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f) : 0.f; return (a + t * ab - x).norm(); };
        bool covered = true;
        for (const auto& q : r) if (q.aggregate) covered = covered and std::ranges::any_of(c, [&](const Return& o) { return o.aggregate and dist_to(o, q.p) <= 0.5f * kDenoiseRadiusM; });
        CHECK("denoise: every return lies within c/2 of an obstacle", covered);
        bool on_line = true;
        for (const auto& o : c) if (o.aggregate and o.p.y() < 0.06f) on_line = on_line and std::abs(o.p.x() - 0.300f) < 0.0035f;
        CHECK("denoise: the line's obstacles lie on it", on_line);
        const auto ends = std::ranges::count_if(c, [](const Return& o) { return o.aggregate and o.seg.norm() == 0.f and o.p.y() < 0.06f and (o.p.y() < 0.001f or o.p.y() > 0.049f); });
        CHECK("denoise: both ends of the line are kept where they were (y 0 and 0.05)", ends == 2);
        const auto iso = std::ranges::find_if(c, [](const Return& q) { return q.aggregate and std::abs(q.p.y() - 0.50f) < 1e-6f; });
        CHECK("denoise: an isolated return keeps its position and sigma_r", iso != c.end() and iso->p.x() == 0.f and iso->sigma == 0.01f and iso->sxx == 0.f);
        CHECK("denoise: a hypothesis (aggregate = false) passes untouched", std::ranges::any_of(c, [](const Return& q) { return not q.aggregate and q.p == Eigen::Vector2f(0.f, 0.235f) and q.p_exist == 0.001f; }));
        // a noisy wall (sigma_r 0.02) 0.309 m away: no denoised return inside the face by more than its own sigma x 3
        std::mt19937 g(1); std::normal_distribution<float> n(0.f, 0.02f);
        std::vector<Return> wall;
        for (int k = 0; k < 1440; ++k)
        {
            const float a = 6.2831853f * k / 1440.f; const Eigen::Vector2f d(std::cos(a), std::sin(a));
            if (d.x() <= 0.05f) continue;
            const float t = 0.309f / d.x();
            if (t > 3.f) continue;
            wall.push_back({d * (t + n(g)), 0.02f, 1.f});
        }
        float worst = 1.f, worst_z = 0.f;
        for (const auto& q : denoise_returns(wall))
            if (std::abs(q.p.y()) < 0.5f)
                for (const float s2 : {-1.f, 1.f})
                {
                    const float x = (q.p + s2 * q.seg).x();
                    worst = std::min(worst, x - 0.309f); worst_z = std::max(worst_z, (0.309f - x) / q.sigma);
                }
        std::printf("  noisy wall (sigma_r 0.02, 0.309 m): worst obstacle point %.4f m inside the face, %.2f of its sigma\n", -worst, worst_z);
        CHECK("denoise: no noise-tail return inside a noisy wall", -worst < 0.01f and worst_z < 4.f);
    }
    { // long_wall_no_gaps (review C1): a parallel noisy wall x = 0.309 with hits to 3 m. Every 0.1 m of wall that holds a
      // raw return holds a denoised obstacle (the representation loses no observed wall; the bearing cells left a 0.58 m
      // gap at y 1.04-1.62), and within the look-ahead (|y| <= 1.6 m) every 0.1 m is covered.
        std::mt19937 g(2); std::normal_distribution<float> n(0.f, 0.02f);
        std::vector<Return> wall;
        for (int k = 0; k < 1440; ++k)
        {
            const float a = 6.2831853f * k / 1440.f; const Eigen::Vector2f d(std::cos(a), std::sin(a));
            if (d.x() <= 0.f) continue;
            const float t = 0.309f / d.x();
            if (t > 3.f) continue;
            wall.push_back({d * (t + n(g)), 0.02f, 1.f});
        }
        const auto c = denoise_returns(wall);
        int raw_bins = 0, lost = 0, gaps_near = 0;
        for (float y0 = -3.f; y0 < 3.f; y0 += 0.1f)
        {
            auto in = [&](const Return& q) { return q.p.y() >= y0 and q.p.y() < y0 + 0.1f; };
            // an obstacle covers the bin if its SUPPORT (segment or point) overlaps it along the wall
            auto sup = [&](const Return& q) { const float a = q.p.y() - std::abs(q.seg.y()), b = q.p.y() + std::abs(q.seg.y()); return b >= y0 and a < y0 + 0.1f; };
            const bool raw = std::ranges::any_of(wall, in), den = std::ranges::any_of(c, sup);
            raw_bins += raw; lost += raw and not den; gaps_near += std::abs(y0 + 0.05f) <= 1.6f and not den;
        }
        std::printf("  long wall: %zu returns -> %zu obstacles; 0.1 m bins with returns %d, lost %d, uncovered within 1.6 m %d\n", wall.size(), c.size(), raw_bins, lost, gaps_near);
        CHECK("long_wall_no_gaps", lost == 0 and gaps_near == 0);
    }
    { // sensor_offset_irrelevant (review C2): the same wall scanned from the helios mount (y -0.155) and the bpearl mount
      // (y +0.140), robot at the origin facing +y sliding forward beside it: P within 0.01.
        auto scan_from = [](float sy, unsigned seed)
        {
            std::mt19937 g(seed); std::normal_distribution<float> n(0.f, 0.02f);
            std::vector<Return> w; const Eigen::Vector2f o(0.f, sy);
            for (int k = 0; k < 1440; ++k)
            {
                const float a = 6.2831853f * k / 1440.f; const Eigen::Vector2f d(std::cos(a), std::sin(a));
                if (d.x() <= 0.f) continue;
                const float t = 0.34f / d.x();
                if (t > 4.f) continue;
                w.push_back({o + d * (t + n(g)), 0.02f, 1.f});
            }
            return w;
        };
        float worst = 0.f;
        for (auto [v, w] : {std::pair{0.25f, 0.f}, std::pair{0.5f, 0.3f}, std::pair{-0.25f, 0.f}})
        {
            const auto sw = rollout(origin, 0.f, 0.f, v, w, 1.f, b, 0.05f);
            const float Ph = 1.f - std::exp(log_p_no_contact(body(), sw, scan_from(-0.155f, 3u), sxy, syaw, kGeom));
            const float Pb = 1.f - std::exp(log_p_no_contact(body(), sw, scan_from(+0.140f, 4u), sxy, syaw, kGeom));
            std::printf("  sensor offset v=%+.2f w=%+.2f : P(helios)=%.4f P(bpearl)=%.4f\n", v, w, Ph, Pb);
            worst = std::max(worst, std::abs(Ph - Pb));
        }
        CHECK("sensor_offset_irrelevant (|dP| <= 0.01)", worst <= 0.01f);
    }
    { // epsilon_integral_accuracy: the adaptive shared-offset integral vs a brute-force 4001-node trapezoid, over
      // sigma_j / sigma_s in [0.05, 5] (random clearance sets, with and without p_exist < 1), and on real sweeps:
      // the pressed wall (x = 0.30) slid along at several sigma_r. Bar: |P - P_ref| <= 0.005 (controller ruling).
        const double ss = std::sqrt(double(sxy) * sxy + double(kGeom) * kGeom);
        std::mt19937 g(11); double worst = 0; int cases = 0;
        for (double ratio : {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0})
        {
            double wr = 0;
            for (int trial = 0; trial < 150; ++trial)
            {
                std::uniform_int_distribution<int> nt(1, 30);
                std::uniform_real_distribution<double> sd0d(-0.01, 0.08), dd(0.0, 0.06), pe(0.3, 1.0);
                std::vector<ContactTerm> t;
                double star = 0.08;
                for (int i = 0, m = nt(g); i < m; ++i) { const double s0 = sd0d(g); star = std::min(star, s0); t.push_back({s0, s0 - dd(g), ratio * ss, trial % 3 == 0 ? pe(g) : 1.0}); }
                // the closest cell now may be one the motion never deepens (not in t): truncate a little closer
                if (trial % 2 == 0) star = std::max(-0.02, star - 0.01);
                wr = std::max(wr, std::abs(std::exp(log_ratio_shared_offset(t, ss, star)) - std::exp(log_ratio_shared_offset(t, ss, star, 4001))));
                ++cases;
            }
            std::printf("  eps integral, sigma_j/sigma_s=%.2f: max |P - P_ref| = %.5f\n", ratio, wr);
            worst = std::max(worst, wr);
        }
        // real sweeps: terms exactly as log_p_no_contact builds them (cells, bare hull, per-cell yaw lever arm)
        const auto f = body();
        for (float sr : {0.0006f, 0.003f, 0.01f, 0.05f})
            for (auto [v, w] : {std::pair{-0.25f, 0.f}, std::pair{-0.5f, -0.3f}, std::pair{0.f, 1.0f}})
            {
                const auto sw = rollout(origin, 0.f, 0.f, v, w, 1.f, b, 0.05f);
                std::vector<ContactTerm> t; double star = 1e9;
                // ★ D1 (Task 6): bearing cells carry extent (support segments) the mean-only reference below does not
                //   model; this check is about the INTEGRAL, so the library is fed the same cells with the scatter
                //   stripped and both sides build identical terms.
                auto cells = denoise_returns(wall_x(0.30f, -2.f, 2.f, sr));
                for (auto& q : cells) { q.sxx = q.sxy = q.syy = 0.f; q.seg.setZero(); }   // points on both sides
                for (const auto& q : cells)
                {
                    const float rho = (q.p - origin.p).norm();
                    const double sj = std::sqrt(double(q.sigma) * q.sigma + double(rho) * rho * syaw * syaw);
                    float sd0 = hull_distance(f, origin, q.p), sd = sd0;
                    for (std::size_t i = 1; i < sw.size(); ++i) sd = std::min(sd, hull_distance(f, sw[i], q.p));
                    if (rho < 1.5f) star = std::min(star, double(sd0));
                    if (not (sd < sd0) or sd - 6.0 * ss > 7.0 * sj) continue;
                    t.push_back({sd0, sd, sj, q.p_exist});
                }
                const double P = 1.0 - std::exp(log_ratio_shared_offset(t, ss, star)), Pref = 1.0 - std::exp(log_ratio_shared_offset(t, ss, star, 4001));
                const double Plib = 1.0 - std::exp(log_p_no_contact_cells(f, sw, cells, sxy, syaw, kGeom));
                std::printf("  wall 0.30 sigma_r=%.4f v=%+.2f w=%+.2f : P=%.4f P_ref=%.4f (log_p_no_contact %.4f)\n", sr, v, w, P, Pref, Plib);
                worst = std::max({worst, std::abs(P - Pref), std::abs(Plib - Pref)}); ++cases;
            }
        std::printf("  epsilon_integral_accuracy: worst |P - P_ref| = %.5f over %d cases\n", worst, cases);
        CHECK("epsilon_integral_accuracy", worst <= 0.005);
    }
    { // timing: 21-pose sweep vs 300 returns
        std::vector<Pose2> sw; for (int i = 0; i < 21; ++i) sw.push_back({{0.f, 0.02f * i}, kHalfPi});
        std::mt19937 g(3); std::uniform_real_distribution<float> u(-1.5f, 1.5f);
        std::vector<Return> r; for (int i = 0; i < 300; ++i) r.push_back({{u(g), u(g)}, 0.02f, 1.f});
        const auto f = body(); std::vector<double> ms; volatile float sink = 0;
        for (int k = 0; k < 201; ++k)
        {
            const auto t0 = std::chrono::steady_clock::now();
            sink = sink + log_p_no_contact(f, sw, r, sxy, syaw, kGeom);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        std::printf("TIMING median log_p_no_contact (21 poses x 300 returns): %.4f ms\n", ms[ms.size() / 2]);
    }
    std::printf(fails ? "FAILED (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
