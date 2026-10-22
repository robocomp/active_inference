// Task R: the policy reverses only to unlock / unstall / in a tight manoeuvre. Reversal is blind to the forward camera
// (progress made through space the camera does not see is not credited, in the progress rate's own units) and, for a
// doorway goal, a plan is scored by log P(facing the gap) at its end.
// Build+run (from controller/): g++ -std=c++23 -O2 -I/usr/include/eigen3 -I../common tests/test_policy_reverse.cpp
//   src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp
//   ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tpr && /tmp/tpr
#include "../src/local_policy/local_policy.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace rc::policy;
using V = Eigen::Vector2f;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)
static constexpr float kHalfPi = 1.57079633f;
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
    void finish(bool camera = true)
    {
        in.pose = {{0.f, 0.f}, kHalfPi};            // forward = +y
        in.sigma_rel_xy = 0.005f; in.sigma_rel_yaw = 0.005f;
        in.returns = ret; in.body = &b; in.base = limits(); in.goal = &goal; in.field = &field;
        if (camera) { in.camera_half_fov_rad = 0.96f; in.affordance_facing = true; }   // ZED2i 110 deg HFOV
    }
};
static void print(const char* n, const Decision& d)
{ std::printf("  [%s] mode=%s v=%+.3f w=%+.3f why=\"%s\"\n", n, mode_name(d.mode), d.v, d.w, d.why.c_str()); }

int main()
{
    { // (a) goal directly behind, open space: turn (or arc forward), never reverse
        Scene s; s.goal.p = {0.f, -3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("a", d);
        CHECK("a open space, goal behind: no reverse", d.mode == Decision::Mode::Drive and d.v >= 0.f);
    }
    { // (b) doorway goal behind (gap at y = -1.5, normal pointing -y): turn first, do not back through it
        Scene s;
        s.ret = wall_y(-1.5f, -3.f, -0.45f); for (auto& r : wall_y(-1.5f, 0.45f, 3.f)) s.ret.push_back(r);
        s.goal = Goal{.kind = Goal::Kind::Doorway, .p = {0.f, -1.5f}, .yaw = {}, .normal = {0.f, -1.f}, .width = 0.9f};
        std::vector<V> pts; for (float t = -0.1f; t <= 0.1f + 1e-4f; t += 0.05f) pts.push_back({t, -2.0f});
        s.field.build(kGrid, [](const V&) { return true; }, no_extra, pts);
        s.finish();
        const auto d = decide(s.in, s.pp); print("b", d);
        CHECK("b doorway behind: turn, no reverse", d.mode == Decision::Mode::Drive and d.v >= 0.f);
    }
    { // (c) dead end: walls ahead and both flanks (a turn in place touches), goal behind -> reverse is the only way out
        Scene s; s.ret = wall_x(-0.36f, -2.f, 0.6f);
        for (auto& r : wall_x(0.36f, -2.f, 0.6f)) s.ret.push_back(r);
        for (auto& r : wall_y(0.5f, -0.36f, 0.36f)) s.ret.push_back(r);
        s.goal.p = {0.f, -3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("c", d);
        CHECK("c dead end: reverse", d.v < 0.f and (d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release));
    }
    { // (d) stalled against an obstacle: goal ahead through a wall the map has not absorbed -> release backs up
        Scene s; s.ret = wall_y(0.30f, -2.f, 2.f); s.goal.p = {0.f, 1.5f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("d", d);
        CHECK("d stalled: reverse to unlock", d.mode == Decision::Mode::Release and d.v < 0.f);
    }
    { // (e) tight corridor (flanks 0.32 m: a turn in place touches), goal behind, front open: reverse out is allowed
        Scene s; s.ret = wall_x(-0.32f, -2.f, 2.f);
        for (auto& r : wall_x(0.32f, -2.f, 2.f)) s.ret.push_back(r);
        s.goal.p = {0.f, -3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("e", d);
        CHECK("e tight manoeuvre: reverse", d.v < 0.f);
    }
    std::printf("%s (%d failed)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
