// Task 4 (local-motion policy): the look-ahead decision (continuability, release, blind, arrived).
// Build+run (from controller/):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_local_policy.cpp
//     src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tlp && /tmp/tlp
//
// ★ FRAMES. The brief gives every snapshot in the ROBOT frame (x right, y forward). The robot stands at
//   Pose2{(0,0), yaw = pi/2}, which makes the room frame coincide with it: room +x = robot right, room +y = forward.
//   A bearing beta (CCW from forward, + = left) is the room direction pi/2 + beta.
// Body: RobotFootprint::shadow() + 0.05 m margin (|x| 0.284 -> 0.334 grown, |y| 0.23 -> 0.28, corner reach 0.368).
// Returns: sigma_r 0.01 m, p_exist 1, spaced 0.05 m. Pose sigma 0.005 m / 0.005 rad.
// Fields: 8 x 8 m grid, 0.05 m cells, centred on the robot. Where a test says "map lag" the field's free predicate
//   does NOT contain the wall (a newly seen obstacle the map has not absorbed — the case release exists for).
//
// ★ Fix round 5 (controller ruling): Release is NOT held to risk_eps — when no eligible candidate makes progress it
//   takes, among ALL v <= 0 motions, those that gain continuability (else all) and of them the least likely contact;
//   Stuck only for returns inside the bare hull or a release with P(contact) >= 1 - risk_eps. pressed_into_wall
//   (parallel wall at x = 0.30, 1.6 cm gap) therefore RELEASES at P = 0.078 (best of the 0.04 m surface cells with one
//   return each at this file's 0.05 m sampling; 0.045 at 1 cm sampling). Drive decisions are untouched.
// ★ Review fix round 1: rollouts ramp from the current velocity and start from the latency-advanced pose
//   (latency_advances_start); cells are swept as support segments (jamb tips); Release scans only releasable
//   candidates (P < 1 - risk_eps) and every sweep stops once its result is provably below what its caller needs.
// ★ Review fix round 3: Stuck iff no release has P(contact) < release_risk_max (0.5) — the wedge is Stuck (best 0.978).
// ★ Review fix round 2: the shared offset is TRUNCATED once by "nothing touches now" (eps > -sd0*) and the
//   integrand is the product of per-cell ratios (contact.h). P values rose where a motion deepens cells (the old
//   per-cell conditioning was optimistic): pressed 0.093 -> 0.170, dense-wall slide 0.069 -> 0.159; no verdict moved.
// Timing (-O2, 300 returns, median; this machine): drive 0.6 ms; release worst case (field +inf, 76 eligible)
//   1.1 ms; the reviewer's no-gain WEDGE (sides +-0.30, nose/tail +-0.26, 56 cells, every candidate P ~ 1) 2.6 ms
//   (65 ms before review round 1: 47 x 14 continuation sweeps, un-pruned). n_cont_* left at the brief's 3 x 5.
#include "../src/local_policy/local_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static constexpr float kHalfPi = 1.57079633f, kPi = 3.14159265f;
static BaseLimits limits()
{ BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 1.f; b.a_rot = 2.f; b.d_rot = 2.f;
  b.v_min_eff = 0.03f; b.w_min_eff = 0.08f; return b; }
static rc::RobotFootprint body() { auto f = rc::RobotFootprint::shadow(); f.set_safety_margin(0.05f); return f; }
static std::vector<Return> wall_x(float x0, float y0, float y1, float spacing = 0.05f)
{ std::vector<Return> r; for (float y = y0; y <= y1 + 1e-4f; y += spacing) r.push_back({{x0, y}, 0.01f, 1.f}); return r; }
static std::vector<Return> wall_y(float y0, float x0, float x1)
{ std::vector<Return> r; for (float x = x0; x <= x1 + 1e-4f; x += 0.05f) r.push_back({{x, y0}, 0.01f, 1.f}); return r; }
static const CostToGo::Grid kGrid{{-4.f, -4.f}, 0.05f, 160, 160};
static const auto no_extra = [](const V&) { return 0.f; };

struct Scene
{
    rc::RobotFootprint b = body();
    std::vector<Return> ret;
    Goal goal{.kind = Goal::Kind::Place, .yaw = {}};
    CostToGo field;
    PolicyInput in;
    PolicyParams pp;
    void finish()
    {
        in.pose = {{0.f, 0.f}, kHalfPi};
        in.sigma_rel_xy = 0.005f; in.sigma_rel_yaw = 0.005f;
        in.returns = ret; in.body = &b; in.base = limits(); in.goal = &goal; in.field = &field;
    }
};

static void print(const char* name, const Decision& d)
{
    std::printf("  [%s] mode=%s v=%+.3f w=%+.3f log_p_safe=%.4f log_p_cont=%.3f progress=%+.3f eligible=%d why=\"%s\"\n",
                name, mode_name(d.mode), d.v, d.w, d.log_p_safe, d.log_p_cont, d.progress_m, d.eligible, d.why.c_str());
}
// Independent re-evaluation of the chosen command's sweep (does not trust the Decision's own numbers).
static std::vector<Pose2> sweep_of(const Scene& s, float v, float w)
{ return rollout(policy_start(s.in), s.in.v_now, s.in.w_now, v, w, s.pp.horizon_s, s.in.base, s.pp.dt); }
static float p_contact_of(const Scene& s, float v, float w)
{ return 1.f - std::exp(log_p_no_contact(s.b, sweep_of(s, v, w), s.ret, s.in.sigma_rel_xy, s.in.sigma_rel_yaw, s.in.body_geometry_sigma_m)); }
static Pose2 end_of(const Scene& s, float v, float w) { return sweep_of(s, v, w).back(); }
static float min_sd(const Scene& s, const Pose2& at)
{ float m = 1e9f; for (const auto& r : s.ret) m = std::min(m, signed_distance(s.b, at, r.p)); return m; }

int main()
{
    { // open_space_goal_ahead
        Scene s; s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("open_space_goal_ahead", d);
        CHECK("open_space_goal_ahead", d.mode == Decision::Mode::Drive and d.v >= 0.9f * 0.5f and std::fabs(d.w) < 0.1f);
    }
    { // side_on_wall_goal_behind_left (tour3_1 leg 3): wall x = +0.37 (right), goal 0.7 m at bearing +150 deg.
        Scene s; s.ret = wall_x(0.37f, -2.f, 2.f);
        const float a = kHalfPi + 150.f * kPi / 180.f; s.goal.p = 0.7f * V(std::cos(a), std::sin(a));
        V g = s.goal.p;
        s.field.build(kGrid, [](const V& q) { return std::fabs(q.x() - 0.37f) > 0.05f or std::fabs(q.y()) > 2.f; },
                      no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("side_on_wall_goal_behind_left", d);
        const bool moving = d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release;
        const float pc = p_contact_of(s, d.v, d.w);
        const float lpc_end = log_p_continuable(s.in, s.pp, end_of(s, d.v, d.w));
        std::printf("  recheck: P(contact)=%.5f log_p_cont(end)=%.3f (>= log(1/3)=%.3f) now=%.3f\n", pc, lpc_end,
                    std::log(1.f / 3.f), log_p_continuable(s.in, s.pp, s.in.pose));
        CHECK("side_on_wall drive_or_release", moving);
        CHECK("side_on_wall end continuable", lpc_end >= std::log(1.f / 3.f));
        CHECK("side_on_wall sweep clear (no wall-hitting turn)", pc <= s.pp.risk_eps);
        // the pure in-place turn toward the goal IS evaluated and refused
        std::printf("  in-place turn left 1.5 rad/s: P(contact)=%.3f\n", p_contact_of(s, 0.f, 1.5f));
    }
    { // pressed_into_wall_releases (fix round 2 geometry): the wall lies inside the old margin shell but OUTSIDE the
      // bare hull — flank x = +0.30 (bare half-width 0.284). Goal straight through it (map lag).
        Scene s; s.ret = wall_x(0.30f, -2.f, 2.f); s.goal.p = {1.5f, 0.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("pressed_into_wall_releases", d);
        float h0 = 1e9f, h1 = 1e9f; const Pose2 e = end_of(s, d.v, d.w);
        for (const auto& r : s.ret) { h0 = std::min(h0, hull_distance(s.b, s.in.pose, r.p)); h1 = std::min(h1, hull_distance(s.b, e, r.p)); }
        std::printf("  bare-hull clearance start %.3f -> end %.3f ; P(contact)=%.5f\n", h0, h1, p_contact_of(s, d.v, d.w));
        // ★ fix round 5: Release with v <= 0 (not gated by risk_eps); its P(contact) is printed above
        CHECK("pressed_into_wall release", d.mode == Decision::Mode::Release and d.v <= 0.f);
        CHECK("pressed_into_wall opens the clearance", h1 > h0);
        for (float v : {-0.5f, -0.25f, 0.25f}) for (float w : {-0.6f, 0.f, 0.6f})
            std::printf("  diag x=0.30 P(contact) v=%+.2f w=%+.2f : %.3f\n", v, w, p_contact_of(s, v, w));
    }
    { // dense_wall_not_riskier (fix round 4): the pressed wall (x = 0.30, parallel to the flank) sampled every 1 cm vs
      // every 4 cm. Returns are measurements of ONE surface: more of them must not raise P(contact) (round 3 read
      // 0.18 -> 0.28 going 5 cm -> 2 cm). ★ MONOTONE, not equal (fix round 5 ruling): with sigma_cell = sigma_r/sqrt(n)
      // four hits per 0.04 m cell pin the wall down twice as well as one, so the denser wall reads LOWER
      // (0.075 vs 0.22 for the slide) — equality would mean the extra hits taught nothing.
        Scene d1, d4; d1.ret = wall_x(0.30f, -2.f, 2.f, 0.01f); d4.ret = wall_x(0.30f, -2.f, 2.f, 0.04f);
        d1.goal.p = d4.goal.p = {1.5f, 0.f};
        V g = d1.goal.p; d1.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        d4.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        d1.finish(); d4.finish();
        float worst_gap = 0.f, worst_rise = 0.f;
        for (auto [v, w] : {std::pair{-0.25f, 0.f}, std::pair{0.25f, 0.f}, std::pair{-0.5f, -0.3f}})
        {
            const float p1 = p_contact_of(d1, v, w), p4 = p_contact_of(d4, v, w);
            std::printf("  dense wall v=%+.2f w=%+.2f : P(1 cm)=%.4f P(4 cm)=%.4f  (obstacles %zu vs %zu)\n", v, w, p1, p4,
                        denoise_returns(d1.ret).size(), denoise_returns(d4.ret).size());
            worst_gap = std::max(worst_gap, std::fabs(p1 - p4)); worst_rise = std::max(worst_rise, p1 - p4);
        }
        const auto e1 = decide(d1.in, d1.pp), e4 = decide(d4.in, d4.pp); print("dense 1 cm", e1); print("dense 4 cm", e4);
        std::printf("  dense wall: worst rise %.4f, largest |difference| %.4f\n", worst_rise, worst_gap);
        CHECK("dense_wall_not_riskier (denser never raises P)", worst_rise <= 1e-3f);
    }
    { // noisy_wall_not_inside (D1 ruling, Task 6): a wall PARALLEL to the right flank, 2.5 cm from the bare hull (x = 0.309),
      // seen by 1440 beams with sigma_r 0.02 as the sensor really reports it. Square-grid cells put single noise-tail hits
      // 4-5 cm inside the face and the robot read "returns inside the body" (tour3_1_leg3 Stuck at t = 0). Bearing cells:
      // no inside verdict, and sliding along the wall is below risk_eps.
        Scene s; s.goal.p = {0.f, 3.f};
        std::mt19937 g(5); std::normal_distribution<float> n(0.f, 0.02f);
        for (int k = 0; k < 1440; ++k)
        {
            const float a = 6.2831853f * float(k) / 1440.f; const V d(std::cos(a), std::sin(a));
            if (d.x() <= 0.05f) continue;
            const float t = 0.309f / d.x();
            if (t > 4.f) continue;
            s.ret.push_back({d * (t + n(g)), 0.02f, 1.f});
        }
        V gp = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&gp, 1));
        s.finish();
        int raw_inside = 0; for (const auto& r : s.ret) raw_inside += hull_distance(s.b, s.in.pose, r.p) < 0.f;
        const auto d = decide(s.in, s.pp); print("noisy wall 2.5 cm", d);
        const float Pf = p_contact_of(s, 0.25f, 0.f), Pb = p_contact_of(s, -0.25f, 0.f);
        std::printf("  noisy wall: %d raw hits inside the hull; P(forward 0.25)=%.4f P(back 0.25)=%.4f\n", raw_inside, Pf, Pb);
        // ★ R3 (round 3): NOT "a parallel slide has P < risk_eps". With the shared offset truncated at "the closest cell may
        //   be just touching now" (eps > -0.025 = -2.3 sigma_s) every cell the slide brings to the same nominal clearance
        //   carries its own sigma_j, so even a PERFECT straight wall (sigma_cell 0.5 mm) reads P ~ 1.1 % here (round-2
        //   table); ~1 % is the honest number for this geometry. The oracle is what the robot does: it must not call the
        //   wall a contradiction, and it must not freeze — it drives or curves away.
        CHECK("noisy_wall_not_inside: no inside-the-body verdict", d.why.find("inside the body") == std::string::npos);
        CHECK("noisy_wall_not_inside: not Stuck (drives or curves away)", d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release);
    }
    { // wall_inside_bare_hull_is_stuck: wall at x = +0.274 penetrates the bare hull — physically impossible
        Scene s; s.ret = wall_x(0.274f, -2.f, 2.f); s.goal.p = {1.5f, 0.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("wall_inside_bare_hull_is_stuck", d);
        CHECK("wall_inside_bare_hull_is_stuck", d.mode == Decision::Mode::Stuck and d.v == 0.f and d.w == 0.f
                                                  and d.why.find("returns inside the body") != std::string::npos);
    }
    { // dead_end_releases: 0.8 m wide cul-de-sac (walls x = +-0.4), nose 0.05 m from the end wall (y = 0.23 + 0.05).
      // The end wall spans the whole grid in the field, so the goal beyond it is walled off (field = +inf here).
        Scene s; s.ret = wall_x(-0.4f, -1.2f, 0.28f);
        for (const auto& r : wall_x(0.4f, -1.2f, 0.28f)) s.ret.push_back(r);
        for (const auto& r : wall_y(0.28f, -0.4f, 0.4f)) s.ret.push_back(r);
        s.goal.p = {0.f, 1.5f};
        V g = s.goal.p;
        s.field.build(kGrid, [](const V& q) { return std::fabs(q.y() - 0.28f) > 0.05f and
                                                     not (std::fabs(std::fabs(q.x()) - 0.4f) < 0.05f and q.y() > -1.2f and q.y() < 0.28f); },
                      no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("dead_end_releases", d);
        std::printf("  recheck P(contact)=%.5f\n", p_contact_of(s, d.v, d.w));
        CHECK("dead_end release reverse", d.mode == Decision::Mode::Release and d.v < 0.f);
        CHECK("dead_end release clear", p_contact_of(s, d.v, d.w) <= s.pp.risk_eps);
    }
    { // release_from_side_on_wall_goal_through_it: the pressed scene with the flank 0.036 m CLEAR of the grown body
      // (wall x = 0.37), goal through the wall (map lag). Exercises the release path where eligibility is possible.
        Scene s; s.ret = wall_x(0.37f, -2.f, 2.f); s.goal.p = {1.5f, 0.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("release_side_on_goal_through_wall", d);
        const Pose2 e = end_of(s, d.v, d.w);
        const float sd0 = min_sd(s, s.in.pose), sd1 = min_sd(s, e);
        std::printf("  clearance start %.3f -> end %.3f ; log_p_cont now %.3f -> end %.3f ; P(contact)=%.5f\n", sd0, sd1,
                    log_p_continuable(s.in, s.pp, s.in.pose), log_p_continuable(s.in, s.pp, e), p_contact_of(s, d.v, d.w));
        CHECK("release_side_on release, v<=0", d.mode == Decision::Mode::Release and d.v <= 0.f);
        CHECK("release_side_on buys room", sd1 > sd0 + 1e-3f and d.log_p_cont > log_p_continuable(s.in, s.pp, s.in.pose));
        CHECK("release_side_on clear", p_contact_of(s, d.v, d.w) <= s.pp.risk_eps);
    }
    { // dead_end with the nose 0.05 m from the end wall measured from the MARGIN-GROWN nose (wall y = 0.33).
        Scene s; s.ret = wall_x(-0.4f, -1.2f, 0.33f);
        for (const auto& r : wall_x(0.4f, -1.2f, 0.33f)) s.ret.push_back(r);
        for (const auto& r : wall_y(0.33f, -0.4f, 0.4f)) s.ret.push_back(r);
        s.goal.p = {0.f, 1.5f};
        V g = s.goal.p;
        s.field.build(kGrid, [](const V& q) { return std::fabs(q.y() - 0.33f) > 0.05f and
                                                     not (std::fabs(std::fabs(q.x()) - 0.4f) < 0.05f and q.y() > -1.2f and q.y() < 0.33f); },
                      no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("dead_end_margin_clear", d);
        std::printf("  recheck P(contact)=%.5f\n", p_contact_of(s, d.v, d.w));
        CHECK("dead_end_margin_clear release reverse", d.mode == Decision::Mode::Release and d.v < 0.f);
        CHECK("dead_end_margin_clear clear", p_contact_of(s, d.v, d.w) <= s.pp.risk_eps);
    }
    { // stale_returns_stop
        Scene s; s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish(); s.in.returns_age_s = 1.0f;
        const auto d = decide(s.in, s.pp); print("stale_returns_stop", d);
        CHECK("stale_returns_stop", d.mode == Decision::Mode::Blind and d.v == 0.f and d.w == 0.f);
    }
    { // arrived
        Scene s; s.goal.p = {0.1f, 0.05f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("arrived", d);
        CHECK("arrived", d.mode == Decision::Mode::Arrived and d.v == 0.f and d.w == 0.f);
    }
    { // heading_goal_beside_wall: face 90 deg right (room yaw 0) with a wall at x = +0.34.
        Scene s; s.ret = wall_x(0.34f, -2.f, 2.f); s.goal.kind = Goal::Kind::Heading; s.goal.yaw = 0.f;
        s.finish(); s.in.field = nullptr;
        const auto d = decide(s.in, s.pp); print("heading_goal_beside_wall", d);
        const float pc = p_contact_of(s, d.v, d.w);
        std::printf("  recheck P(contact)=%.5f ; in-place right turn 0.75 rad/s P(contact)=%.3f\n", pc, p_contact_of(s, 0.f, -0.75f));
        CHECK("heading_goal_beside_wall never risky", (d.mode == Decision::Mode::Stuck and d.v == 0.f and d.w == 0.f) or pc <= s.pp.risk_eps);
        // and with room to turn the same goal is driven by turning right
        Scene o; o.goal.kind = Goal::Kind::Heading; o.goal.yaw = 0.f; o.finish(); o.in.field = nullptr;
        const auto e = decide(o.in, o.pp); print("heading_goal_open(diag)", e);
        CHECK("heading_goal_open turns right", e.mode == Decision::Mode::Drive and e.w < 0.f);
        const Pose2 ee = end_of(o, e.v, e.w);
        std::printf("  heading open: end displacement %.4f m\n", (ee.p - o.in.pose.p).norm());
        CHECK("heading_goal_open_space_turns_in_place", std::fabs(e.v) < 0.05f and (ee.p - o.in.pose.p).norm() < 0.05f);
    }
    { // heading_release_holds_position (Live fix 1, tourP1: an orient goal's Release "back up straight" 0.26 m took the robot
      // out of door_2's reach band). A HEADING goal near a door frame where the in-place turn is risky: among the releases
      // that gain continuability, the one whose end pose is CLOSEST to the hold position — not "back up straight".
        Scene s;
        // a door frame's jamb beside the right flank (x = +0.31, y -0.30 .. +0.25): the right turn sweeps the front-right
        // corner into it (P ~0.75). The pre-fix choice here was "back up straight" at v_max (0.50 m); the arcs that also
        // gain continuability end ~0.16 m from the hold position.
        s.ret = wall_x(0.31f, -0.30f, 0.25f, 0.02f);
        s.goal.kind = Goal::Kind::Heading; s.goal.yaw = 0.f; s.goal.p = {0.f, 0.f};   // face east, here
        s.finish(); s.in.field = nullptr;
        const auto d = decide(s.in, s.pp); print("heading_release_holds_position", d);
        const float disp = (end_of(s, d.v, d.w).p - s.goal.p).norm();
        const float straight = (end_of(s, -s.in.base.v_max, 0.f).p - s.goal.p).norm();
        std::printf("  chosen end displacement %.3f m ; back up straight at v_max %.3f m ; in-place right turn P(contact)=%.3f\n",
                    disp, straight, p_contact_of(s, 0.f, -0.75f));
        CHECK("heading_release: releases (the in-place turn is too risky here)", d.mode == Decision::Mode::Release);
        CHECK("heading_release: not 'back up straight' at speed", not (std::fabs(d.w) < 1e-3f and d.v <= -0.5f * s.in.base.v_max));
        CHECK("heading_release: the least-displacing release (well under a full-speed back-up)", disp < 0.5f * straight);
    }
    { // inside_body_is_a_sensing_contradiction (Live fix 2): returns inside the bare hull (tourP3: old-frame returns read
      // against the new room frame's pose) are a sensing/pose error — flagged, so the caller holds instead of handing back
        Scene s; s.ret = wall_x(0.10f, -0.2f, 0.2f, 0.02f);              // a wall THROUGH the body
        s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("inside_body", d);
        CHECK("inside_body: Stuck, flagged as a sensing contradiction", d.mode == Decision::Mode::Stuck and d.sensing_contradiction);
        Scene o; o.ret = wall_x(0.30f, -2.f, 2.f); o.goal.p = {1.5f, 0.f};   // pressed_into_wall: not a contradiction
        V h = o.goal.p; o.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&h, 1));
        o.finish();
        CHECK("inside_body: an ordinary decision is not flagged", not decide(o.in, o.pp).sensing_contradiction);
    }
    { // latency_advances_start (review ruling): moving forward at 0.5 m/s toward a wall 0.45 m ahead of the nose
      // (y = 0.68), goal beyond it (map lag). With latency 0 the policy can keep 0.33 m/s (0.40 m to rest); with
      // 0.3 s of latency the base has coasted 0.15 m before the command bites, and only a slower speed stops short.
        Scene s; s.ret = wall_y(0.68f, -1.f, 1.f); s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish(); s.in.v_now = 0.5f;
        const auto d0 = decide(s.in, s.pp); print("latency 0", d0);
        const float P0 = p_contact_of(s, d0.v, d0.w);
        s.in.latency_s = 0.3f;
        const Pose2 st = policy_start(s.in);
        const auto d1 = decide(s.in, s.pp); print("latency 0.3", d1);
        std::printf("  start advanced by %.3f m (0.150) ; chosen v %.3f -> %.3f ; P(contact) %.4f -> %.4f\n", (st.p - s.in.pose.p).norm(), d0.v, d1.v,
                    P0, p_contact_of(s, d1.v, d1.w));
        CHECK("latency_advances_start pose", std::fabs((st.p - s.in.pose.p).norm() - 0.15f) < 1e-4f and std::fabs(st.yaw - s.in.pose.yaw) < 1e-6f);
        CHECK("latency_advances_start slows the drive", d0.mode == Decision::Mode::Drive and d1.mode == Decision::Mode::Drive and d1.v < d0.v - 0.05f);
    }
    { // non-finite input -> Blind
        Scene s; s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish(); s.in.v_now = std::numeric_limits<float>::quiet_NaN();
        const auto d = decide(s.in, s.pp); print("nan input", d);
        CHECK("non_finite_input_is_blind", d.mode == Decision::Mode::Blind and d.v == 0.f and d.w == 0.f and d.why == "non-finite input");
    }
    { // wedge timing (reviewer's scene): a box so tight nothing can move or gain — sides x = +-0.30, nose/tail
      // y = +-0.26, 300 returns at ~7 mm. Every candidate reads P ~ 1, so no continuation is ever rolled out.
        Scene s;
        for (int i = 0; i < 75; ++i)
        {
            const float t = -0.6f + 1.2f * float(i) / 74.f;
            s.ret.push_back({{0.30f, t * 0.45f}, 0.01f, 1.f}); s.ret.push_back({{-0.30f, t * 0.45f}, 0.01f, 1.f});
            s.ret.push_back({{t * 0.5f, 0.26f}, 0.01f, 1.f}); s.ret.push_back({{t * 0.5f, -0.26f}, 0.01f, 1.f});
        }
        for (auto& q : s.ret) if (std::fabs(q.p.x()) < 0.29f and std::fabs(q.p.y()) < 0.25f) q.p.y() = q.p.y() > 0.f ? 0.26f : -0.26f;
        s.goal.p = {0.f, 3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        std::vector<double> ms; Decision d{};
        for (int i = 0; i < 40; ++i)
        {
            const auto t0 = std::chrono::steady_clock::now();
            d = decide(s.in, s.pp);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        print("wedge", d);
        std::printf("  wedge per-call median %.2f ms, p90 %.2f ms, max %.2f ms (300 returns, -O2)\n", ms[20], ms[36], ms[39]);
        // ★ review fix round 3: Stuck — the least-P motion (the 0.30 rad in-place turn, corner 2 cm into the nose
        //   wall) reads P = 0.978, nowhere near release_risk_max = 0.5; the best P is in `why`.
        std::printf("  wedge: %s ; %s\n", mode_name(d.mode), d.why.c_str());
        CHECK("wedge is stuck", d.mode == Decision::Mode::Stuck and d.v == 0.f and d.w == 0.f and d.why.find("release_risk_max") != std::string::npos);
        CHECK("wedge timing median <= 15 ms", ms[20] <= 15.0);
    }
    { // timing: 300 returns (a 2 x 3 m box around the robot, perimeter 10 m), goal 2 m ahead
        Scene s;
        for (const auto& r : wall_x(-1.f, -1.5f, 1.5f)) s.ret.push_back(r);
        for (const auto& r : wall_x(1.f, -1.5f, 1.5f)) s.ret.push_back(r);
        for (const auto& r : wall_y(-1.5f, -0.95f, 0.95f)) s.ret.push_back(r);
        for (const auto& r : wall_y(1.5f, -0.95f, 0.95f)) s.ret.push_back(r);
        while (s.ret.size() < 300) s.ret.push_back({{0.9f, -1.4f + 0.01f * float(s.ret.size() % 280)}, 0.01f, 1.f});
        s.ret.resize(300);
        s.goal.p = {0.f, 2.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        std::vector<double> ms;
        Decision d{};
        for (int i = 0; i < 100; ++i)
        {
            const auto t0 = std::chrono::steady_clock::now();
            d = decide(s.in, s.pp);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        print("timing", d);
        std::printf("  per-call median %.2f ms, p90 %.2f ms, max %.2f ms (300 returns, -O2)\n", ms[50], ms[90], ms[99]);
        CHECK("timing median <= 15 ms", ms[50] <= 15.0);
        // worst case for the release path: same box, goal walled off (field +inf) => every v <= 0 eligible
        // candidate's continuability is rolled out
        V far{3.5f, 3.5f}; s.field.build(kGrid, [](const V& q) { return q.x() < 3.f; }, no_extra, std::span<const V>(&far, 1));
        ms.clear();
        for (int i = 0; i < 100; ++i)
        {
            const auto t0 = std::chrono::steady_clock::now();
            d = decide(s.in, s.pp);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        print("timing_release", d);
        std::printf("  release per-call median %.2f ms, p90 %.2f ms, max %.2f ms\n", ms[50], ms[90], ms[99]);
        CHECK("timing release median <= 15 ms", ms[50] <= 15.0);
    }
    std::printf("%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
