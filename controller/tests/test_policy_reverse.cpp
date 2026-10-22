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
    SeenMap seen;   // what the forward camera has looked at
    void finish(bool camera = true)
    {
        in.pose = {{0.f, 0.f}, kHalfPi};            // forward = +y
        in.sigma_rel_xy = 0.005f; in.sigma_rel_yaw = 0.005f;
        in.returns = ret; in.body = &b; in.base = limits(); in.goal = &goal; in.field = &field;
        in.seen = &seen;
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
        s.goal.p = {0.f, -3.f}; s.seen.mark({{0.f, -2.f}, kHalfPi}, 0.96f, 4.f);   // it drove in along this corridor: the way back was seen
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
        s.goal.p = {0.f, -3.f}; s.seen.mark({{0.f, -2.f}, kHalfPi}, 0.96f, 4.f);   // it drove in along this corridor: the way back was seen
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("e", d);
        CHECK("e tight manoeuvre: reverse", d.v < 0.f);
    }
    { // (g) I3(i): clutter BEHIND the robot, unseen by the camera, goal behind -> turn and go forward, not back into it.
      // Camera-off control: the same scene reverses (clutter leaves a 12 cm lane the contact model clears at risk_eps).
        Scene s;
        for (float y : {-0.55f, -0.95f, -1.35f})
        { s.ret.push_back({{-0.45f, y}, 0.01f, 1.f}); s.ret.push_back({{0.45f, y}, 0.01f, 1.f}); }
        s.goal.p = {0.f, -3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        s.finish();
        const auto d = decide(s.in, s.pp); print("g", d);
        Scene c = s; c.finish(false); c.in.returns = c.ret; c.in.body = &c.b; c.in.goal = &c.goal; c.in.field = &c.field; c.in.seen = nullptr;
        c.in.camera_half_fov_rad = -1.f;
        const auto d0 = decide(c.in, c.pp); print("g-camera-off", d0);
        CHECK("g clutter behind (unseen): no reversal", d.mode == Decision::Mode::Drive and d.v >= 0.f);
        CHECK("g control: without the camera model the same scene reverses (the test discriminates)", d0.v < 0.f);
    }
    { // (h) I3(ii) + P1: with the reverse DISPREFERENCE, a forward arc that is eligible and descends beats reversing even in a tight
      // spot (wall 0.26 m past the nose, goal to the left, a safe slow left arc exists: (h1) reverse over the SEEN way back loses);
      // reversal wins only when nothing forward is eligible: flanks 0.32 m (no turn fits) AND a wall 0.30 m ahead (h2), seen or not.
        auto scene = [&](bool boxed, bool trail) {
            Scene s;
            if (boxed)
            {
                for (auto& r : wall_x(-0.32f, -2.f, 2.f)) s.ret.push_back(r);
                for (auto& r : wall_x(0.32f, -2.f, 2.f)) s.ret.push_back(r);
                for (auto& r : wall_y(0.58f, -0.32f, 0.32f)) s.ret.push_back(r);
                s.goal.p = {0.f, -3.f};
            }
            else
            {
                for (float x = -1.2f; x <= 0.1f + 1e-4f; x += 0.05f) s.ret.push_back({{x, 0.61f}, 0.01f, 1.f});
                s.goal.p = {-1.5f, 0.f};
            }
            V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
            if (trail) s.seen.mark({{0.f, -2.f}, kHalfPi}, 0.96f, 4.f);
            s.finish();
            return decide(s.in, s.pp);
        };
        const auto h1 = scene(false, true), h2s = scene(true, true), h2u = scene(true, false);
        print("h1 wall ahead, goal left, way back seen", h1); print("h2 boxed in, seen", h2s); print("h2 boxed in, unseen", h2u);
        CHECK("h1 an eligible forward arc beats a seen reversal (dispreference)", h1.v > 0.f);
        CHECK("h2 boxed in (no turn, wall ahead): reverse over the seen way back", h2s.v < 0.f);
        // unseen way back: the reversal sweeps unseen space (prior ~ risk_eps) so it is not a Drive plan; the Release machinery answers
        // (it takes the move that gains continuability - a turn at P(contact) 0.39 < release_risk_max 0.5 - which is the pre-existing
        // Release order, flagged in the report). What must hold: the unseen reversal is NOT chosen as a Drive.
        CHECK("h2 boxed in, way back unseen: not a blind Drive reversal (the Release machinery answers)", h2u.mode == Decision::Mode::Release);
    }
    { // (i) P2: goal behind, open space, EVERYTHING SEEN (the whole circle already looked at): the dispreference alone keeps the
      // robot from reversing - it turns and goes forward
        Scene s; s.goal.p = {0.f, -3.f};
        V g = s.goal.p; s.field.build(kGrid, [](const V&) { return true; }, no_extra, std::span<const V>(&g, 1));
        for (float a = 0.f; a < 6.2831853f; a += 0.5f) s.seen.mark({{0.f, 0.f}, a}, 0.96f, 4.f);
        s.finish();
        const auto d = decide(s.in, s.pp); print("i all-seen", d);
        CHECK("i goal behind, open, everything seen: no reverse", d.mode == Decision::Mode::Drive and d.v >= 0.f);
    }
    { // (f) R4: the facing term matters only inside the door's reach - a sideways approach arc far from the gap is not penalised
        Scene s;
        s.goal = Goal{.kind = Goal::Kind::Doorway, .p = {1.5f, 0.f}, .yaw = {}, .normal = {1.f, 0.f}, .width = 0.9f};
        std::vector<V> pts; for (float t = -0.1f; t <= 0.1f + 1e-4f; t += 0.05f) pts.push_back({2.0f, t});
        s.field.build(kGrid, [](const V&) { return true; }, no_extra, pts);
        s.finish();
        const Pose2 sideways_far{{-1.5f, 0.f}, kHalfPi};      // 3 m from the gap, heading +y (90 deg off the gap direction)
        const Pose2 sideways_near{{0.7f, 0.f}, kHalfPi};      // 0.8 m from the gap, same heading
        const Pose2 facing_near{{0.7f, 0.f}, 0.f};            // 0.8 m, heading at the gap
        const Pose2 backwards_near{{0.7f, 0.f}, 3.14159265f};  // 0.8 m, gap behind
        const float far = affordance_log_p_facing(s.in, sideways_far), near = affordance_log_p_facing(s.in, sideways_near);
        const float ok = affordance_log_p_facing(s.in, facing_near), back = affordance_log_p_facing(s.in, backwards_near);
        std::printf("  [f] log P(facing): far sideways %.4f | near sideways %.2f | near facing %.4f | near backwards %.1f\n", far, near, ok, back);
        CHECK("f far from the gap the facing term is ~0 (sideways approach not penalised)", far > -0.25f);
        CHECK("f near the gap, facing it costs ~0", ok > -0.05f);
        CHECK("f near the gap, sideways or backwards costs nats", near < -1.f and back < -1.f and near < 10.f * far);
    }
    std::printf("%s (%d failed)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
