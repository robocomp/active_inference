// Task 2 (local-motion policy): rollout + swept-body contact probability.
// Review fix round 1: rollouts ramp from the current velocity (v0, w0); cells carry their hits' scatter and are
//   swept as support segments (jamb_tip_not_overconfident, turn_in_place_from_speed_sees_wall_ahead).
// Fix round 4: returns are merged into surface cells (kSurfaceCellM) and the shared-offset integral is adaptive;
//   `surface_cells` and `epsilon_integral_accuracy` (vs a 4001-node reference, sigma_j/sigma_s in [0.05, 5]) cover them.
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
    { // arc closes: v=0.3,w=0.5 commanded for 2 s lies on the circle of radius v/w once both axes have ramped up
      // (v at t = 0.3 s, w at 0.25 s; dt 0.1 => from pose 3 to the end of the command at pose 20).
        const auto s = rollout(origin, 0.f, 0.f, 0.3f, 0.5f, 2.f, {b}, 0.1f);
        const Pose2& r0 = s[3];
        const Eigen::Vector2f c = r0.p + (0.3f / 0.5f) * Eigen::Vector2f(-std::sin(r0.yaw), std::cos(r0.yaw));
        // ★ only the steady part of the command phase is one circle: the two axes brake at different times afterwards
        float worst = 0; for (std::size_t i = 3; i < 21; ++i) { const auto& q = s[i]; worst = std::max(worst, std::abs((q.p - c).norm() - 0.6f)); }
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
      // ★ a mean-only cell at 0.305 with sigma_r/sqrt(2) reads 2.4 sigma where the extreme hit reads 1.6: that is
      //   what the support segment + projected scatter correct.
        std::vector<Return> tip; for (float x : {0.30f, 0.31f, 0.32f, 0.33f}) tip.push_back({{x, 0.60f}, 0.01f, 1.f});
        const std::vector<Return> extreme{{{0.30f, 0.60f}, 0.01f, 1.f}};
        const auto sw = rollout(origin, 0.f, 0.f, 0.5f, 0.f, 1.f, b, 0.05f);
        const float Pt = 1.f - std::exp(log_p_no_contact(body(), sw, tip, sxy, syaw, kGeom));
        const float Pe = 1.f - std::exp(log_p_no_contact(body(), sw, extreme, sxy, syaw, kGeom));
        const auto c = surface_cells(tip);
        std::printf("  jamb tip: P(4 hits as cells)=%.4f P(extreme hit alone)=%.4f ; cells %zu, cell0 sigma=%.4f sxx=%.2e syy=%.2e\n", Pt, Pe, c.size(), c[0].sigma, c[0].sxx, c[0].syy);
        CHECK("jamb_tip_not_overconfident", Pt >= Pe - 1e-3f);
        // and on a flat wall seen along its normal the scatter is orthogonal to the clearance: nothing changes
        std::vector<Return> dense; for (float y = -1.f; y <= 2.f; y += 0.01f) dense.push_back({{0.40f, y}, 0.01f, 1.f});
        const auto flat = surface_cells(dense);
        std::printf("  flat wall cell (1 cm hits): sigma=%.4f sxx=%.2e syy=%.2e\n", flat[5].sigma, flat[5].sxx, flat[5].syy);
        CHECK("flat wall scatter is along the wall", flat[5].sxx < 1e-9f and flat[5].syy > 1e-5f);
    }
    { // p_exist scales the hit; a far return is ignored
        const std::vector<Return> hit{{{0.f, 0.3f}, 0.01f, 1.f}}, ghost{{{0.f, 0.3f}, 0.01f, 0.1f}}, far{{{50.f, 0.f}, 0.01f, 1.f}};
        const auto sw = rollout(origin, 0.f, 0.f, 0.3f, 0.f, 1.f, b, 0.05f);
        const float a = log_p_no_contact(body(), sw, hit, sxy, syaw, kGeom), g = log_p_no_contact(body(), sw, ghost, sxy, syaw, kGeom);
        CHECK("p_exist=0.1 hits ~10x less", g > a and std::abs(g - std::log(0.9f)) < 0.02f);
        CHECK("far return ignored", log_p_no_contact(body(), sw, far, sxy, syaw, kGeom) == 0.f);
    }
    { // surface_cells: n hits of one surface element are ONE obstacle at their mean with sigma_r / sqrt(n)
        std::vector<Return> r;
        for (int i = 0; i < 4; ++i) r.push_back({{0.301f + 0.002f * i, 0.011f + 0.005f * i}, 0.01f, 1.f});   // one 0.04 cell
        r.push_back({{0.30f, 0.11f}, 0.01f, 1.f});                                                            // alone in its cell
        r.push_back({{0.30f, 0.21f}, 0.01f, 0.5f}); r.push_back({{0.31f, 0.22f}, 0.01f, 0.5f});              // two half-sure hits
        const auto c = surface_cells(r);
        std::printf("  cells: %zu ; cell0 p=(%.4f,%.4f) sigma=%.4f ; cell1 sigma=%.4f ; cell2 p_exist=%.3f\n", c.size(),
                    c[0].p.x(), c[0].p.y(), c[0].sigma, c[1].sigma, c[2].p_exist);
        CHECK("surface_cells count", c.size() == 3);
        CHECK("surface_cells mean + sigma/sqrt(n)", std::abs(c[0].p.x() - 0.304f) < 1e-4f and std::abs(c[0].p.y() - 0.0185f) < 1e-4f and std::abs(c[0].sigma - 0.005f) < 1e-5f);
        CHECK("surface_cells isolated hit keeps sigma_r", std::abs(c[1].sigma - 0.01f) < 1e-6f and c[1].p_exist == 1.f);
        CHECK("surface_cells existence combines", std::abs(c[2].p_exist - 0.75f) < 1e-5f);
        CHECK("surface_cells idempotent", surface_cells(c).size() == 3 and std::abs(surface_cells(c)[0].sigma - 0.005f) < 1e-5f);
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
                for (const auto& q : surface_cells(wall_x(0.30f, -2.f, 2.f, sr)))
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
                const double Plib = 1.0 - std::exp(log_p_no_contact(f, sw, wall_x(0.30f, -2.f, 2.f, sr), sxy, syaw, kGeom));
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
