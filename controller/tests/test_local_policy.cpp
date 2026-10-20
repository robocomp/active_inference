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
// ★ KNOWN FAIL (4 lines, reported to the controller in task-4-report.md, NOT worked around here):
//   pressed_into_wall_releases (x2) and dead_end_releases (x2) end Stuck with eligible = 0. Cause is the contact
//   model (contact.cpp, Task 2), not decide(): every sweep includes the START pose, so a return within ~3 sigma
//   OUTSIDE the grown body is charged to every motion, even one moving straight away (dead end, wall at the grown
//   nose: P = 0.41 standing still, 0.41 backing out); and the "inside at start" exemption is a step (sd0 = -0.001
//   -> 0.446, sd0 = +0.001 -> 0.998). The pressed case additionally needs a motion that brings no NEW wall point
//   into the body, which a differential base sliding along a wall it already penetrates does not have.
//   The *_margin_clear / release_side_on variants are the same scenes with the wall clear of the grown body and
//   show the release path itself working.
// Timing (-O2, 300 returns, median of 100): drive 4.5 ms; release worst case (field +inf, 76 eligible) 9.8 ms
//   (19.3 ms before the exact early stop in the release scan). n_cont_* left at the brief's 3 x 5.
#include "../src/local_policy/local_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
static std::vector<Return> wall_x(float x0, float y0, float y1)
{ std::vector<Return> r; for (float y = y0; y <= y1 + 1e-4f; y += 0.05f) r.push_back({{x0, y}, 0.01f, 1.f}); return r; }
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
        in.sigma_xy = 0.005f; in.sigma_yaw = 0.005f;
        in.returns = ret; in.body = &b; in.base = limits(); in.goal = &goal; in.field = &field;
    }
};

static void print(const char* name, const Decision& d)
{
    std::printf("  [%s] mode=%s v=%+.3f w=%+.3f log_p_safe=%.4f log_p_cont=%.3f progress=%+.3f eligible=%d why=\"%s\"\n",
                name, mode_name(d.mode), d.v, d.w, d.log_p_safe, d.log_p_cont, d.progress_m, d.eligible, d.why.c_str());
}
// Independent re-evaluation of the chosen command's sweep (does not trust the Decision's own numbers).
static float p_contact_of(const Scene& s, float v, float w)
{
    const auto sw = rollout(s.in.pose, v, w, s.pp.horizon_s, s.in.base, s.pp.dt);
    return 1.f - std::exp(log_p_no_contact(s.b, sw, s.ret, s.in.sigma_xy, s.in.sigma_yaw));
}
static Pose2 end_of(const Scene& s, float v, float w) { return rollout(s.in.pose, v, w, s.pp.horizon_s, s.in.base, s.pp.dt).back(); }
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
    { // pressed_into_wall_releases: wall 0.01 m inside the right flank (|x| 0.284 -> x = 0.274), goal through it.
        Scene s; s.ret = wall_x(0.274f, -2.f, 2.f); s.goal.p = {1.5f, 0.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));   // map lag
        s.finish();
        const auto d = decide(s.in, s.pp); print("pressed_into_wall_releases", d);
        const float sd0 = min_sd(s, s.in.pose), sd1 = min_sd(s, end_of(s, d.v, d.w));
        std::printf("  clearance start %.3f -> end %.3f\n", sd0, sd1);
        CHECK("pressed_into_wall release", d.mode == Decision::Mode::Release);
        CHECK("pressed_into_wall moves away", sd1 > sd0 + 1e-3f);
        // diagnostic for the report: the same with the wall 0.01 m inside the MARGIN-grown flank (x = 0.324)
        Scene t; t.ret = wall_x(0.324f, -2.f, 2.f); t.goal.p = {1.5f, 0.f};
        V g2 = t.goal.p; t.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g2, 1));
        t.finish(); print("pressed_into_margin(diag)", decide(t.in, t.pp));
        for (float v : {-0.5f, -0.25f, 0.25f}) for (float w : {-0.6f, 0.f, 0.6f})
            std::printf("  diag x=0.274 P(contact) v=%+.2f w=%+.2f : %.3f | x=0.324 : %.3f\n", v, w, p_contact_of(s, v, w), p_contact_of(t, v, w));
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
