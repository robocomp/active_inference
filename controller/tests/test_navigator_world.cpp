// Task 8 (local-motion policy): Navigator::update_world — the planner's map changes under a goal; the change is applied
// between plans (never interrupting the committed one), traps kept, the monitor's baseline re-read on the new field.
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_navigator_world.cpp
//     src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp src/local_policy/navigator.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tnw && /tmp/tnw
#include "../src/local_policy/navigator.h"
#include "../src/local_policy/policy_tick_gate.h"

#include <cmath>
#include <cstdio>

using namespace rc::policy;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

int main()
{
    const rc::RobotFootprint body = rc::RobotFootprint::shadow();
    BaseLimits b; b.v_max = 0.5f; b.w_max = 1.0f; b.a_lin = 0.5f; b.d_lin = 1.0f; b.a_rot = 1.9f; b.d_rot = 3.8f; b.v_min_eff = 0.04f; b.w_min_eff = 0.1f;
    const CostToGo::Grid grid{{0.f, 0.f}, 0.05f, 161, 81};   // 8 x 4 m
    const float r_ins = body.inscribed_radius();
    const auto open_room = [r_ins](const Eigen::Vector2f& p) { return p.x() > r_ins and p.x() < 8.f - r_ins and p.y() > r_ins and p.y() < 4.f - r_ins; };
    // a partition at x 4 .. 4.2 with a gap only at the top (y > 3.0): the straight line is cut, the field detours
    const auto partitioned = [open_room](const Eigen::Vector2f& p) { return open_room(p) and not (p.x() > 3.7f and p.x() < 4.5f and p.y() < 3.0f); };
    const auto zero = [](const Eigen::Vector2f&) { return 0.f; };
    Navigator nav;
    const Goal goal{.kind = Goal::Kind::Place, .p = {7.f, 1.f}, .yaw = {}};
    nav.set_goal(goal, grid, open_room, zero, {goal.p}, body.circumscribed_radius());
    const Eigen::Vector2f start(1.f, 1.f);
    const float f_open = nav.field().at(start);
    CHECK("open room: field = straight distance", std::abs(f_open - 6.f) < 0.1f);

    PolicyParams pp;
    PolicyInput in;
    in.pose = {start, 0.f}; in.body = &body; in.base = b; in.now_s = 0.0;
    const std::vector<Return> none;
    in.returns = none;
    auto st = nav.step(in, pp);
    CHECK("first step decides a Drive plan", st.d.mode == Decision::Mode::Drive and nav.plan().active);

    nav.update_world(partitioned, zero);
    CHECK("update_world: pending", nav.world_pending());
    in.now_s = 0.1; in.v_now = st.d.v; in.w_now = st.d.w;
    st = nav.step(in, pp);
    CHECK("the committed plan continues: the new world is not applied mid-plan", not st.replanned and nav.world_pending()
          and std::abs(nav.field().at(start) - f_open) < 1e-4f);
    // run until the plan ends and a new one is decided
    bool applied = false;
    for (int k = 2; k < 80 and not applied; ++k)
    {
        // the robot executes what it is told (a perfect base): no failed prediction can lay a trap here
        in.pose.p += 0.05f * st.d.v * Eigen::Vector2f(std::cos(in.pose.yaw), std::sin(in.pose.yaw));
        in.pose.yaw += 0.05f * st.d.w;
        in.v_now = st.d.v; in.w_now = st.d.w;
        in.now_s = 0.05 * k;
        st = nav.step(in, pp);
        applied = st.world_applied;
        if (applied) CHECK("applied exactly when a new plan is decided", st.replanned);
    }
    CHECK("the new world is applied at the next plan decision", applied and not nav.world_pending());
    CHECK("the field now detours around the partition", nav.field().at(start) > f_open + 0.5f);
    std::printf("   field at the start: %.2f m open, %.2f m partitioned\n", f_open, nav.field().at(start));
    CHECK("no trap was invented by the world change", nav.traps().empty());

    // ---- I1: an INTERRUPTION is not evidence — the committed plan and its prediction are dropped unjudged
    {
        Navigator n2;
        n2.set_goal(goal, grid, open_room, zero, {goal.p}, body.circumscribed_radius());
        PolicyInput q = in; q.pose = {start, 0.f}; q.v_now = q.w_now = 0.f; q.now_s = 100.0;
        const auto a = n2.step(q, pp);
        CHECK("I1: a Drive plan is committed", a.d.mode == Decision::Mode::Drive and n2.plan().active);
        // the base is paused for 5 s: nothing moved. Without interrupt() the plan completes and is judged a FAILED prediction.
        Navigator n3 = n2;
        q.now_s = 105.0;
        const auto judged = n3.step(q, pp);
        CHECK("I1 (control): judged after a pause, the unexecuted plan reads as a failed prediction", judged.d.prediction_failed or n3.failed_predictions() > 0);
        n2.interrupt();
        CHECK("I1: interrupt() drops the plan", not n2.plan().active);
        const auto b = n2.step(q, pp);
        CHECK("I1: after interrupt() the resumed decision is not a failed prediction, no trap", not b.d.prediction_failed and not b.trap
              and n2.failed_predictions() == 0 and n2.traps().empty());
    }
    // ---- review fix 2, item 1: a 0.1 s hold at the end of a plan (odometry stale for two ticks) is an INTERRUPTION. Ticks
    //   run every 50 ms (no gap rule applies); the base has not moved. The session's gate turns the hold into interrupt();
    //   the control run without it judges the plan and lays a trap.
    {
        auto run = [&](bool with_gate) {
            Navigator n5;
            n5.set_goal(goal, grid, open_room, zero, {goal.p}, body.circumscribed_radius());
            PolicyInput q = in; q.pose = {start, 0.f}; q.v_now = q.w_now = 0.f;
            PolicyTickGate gate;
            Navigator::Step last;
            double t = 400.0;
            for (; t < 400.0 + pp.horizon_s - 0.1 + 1e-9; t += 0.05)
            { q.now_s = t; if (gate.on_tick(t) and with_gate) n5.interrupt(); last = n5.step(q, pp); }
            gate.on_hold(); gate.on_hold();                            // two held ticks (0.1 s): no step
            t += 0.1;
            q.now_s = t;
            if (gate.on_tick(t) and with_gate) n5.interrupt();
            last = n5.step(q, pp);
            q.now_s = t + 0.05;
            if (gate.on_tick(q.now_s) and with_gate) n5.interrupt();
            const auto after = n5.step(q, pp);
            return n5.failed_predictions() == 0 and n5.traps().empty() and not last.d.prediction_failed and not after.d.prediction_failed;
        };
        CHECK("hold (control): without the gate the plan is judged after the hold — a failed prediction", not run(false));
        CHECK("hold: with the gate a 0.1 s hold lays no trap and judges nothing", run(true));
    }
    // ---- I2: set_goal clears a world queued for the previous goal (Place -> Doorway keeps the corridor)
    {
        Navigator n4;
        n4.set_goal(goal, grid, open_room, zero, {goal.p}, body.circumscribed_radius());
        const auto walled = [](const Eigen::Vector2f& p) { return p.x() < 4.f and p.x() > 0.3f and p.y() > 0.3f and p.y() < 3.7f; };
        n4.update_world(walled, zero);                       // queued for the Place goal: the right half is closed
        const auto with_corridor = [walled](const Eigen::Vector2f& p) { return walled(p) or (p.y() > 1.7f and p.y() < 2.3f and p.x() < 7.8f); };
        const Goal door{.kind = Goal::Kind::Doorway, .p = {4.f, 2.f}, .yaw = {}, .normal = {1.f, 0.f}, .width = 0.9f};
        n4.set_goal(door, grid, with_corridor, zero, {Eigen::Vector2f(6.f, 2.f)}, body.circumscribed_radius());
        CHECK("I2: set_goal clears the pending world", not n4.world_pending());
        PolicyInput q = in; q.pose = {Eigen::Vector2f(2.f, 2.f), 0.f}; q.v_now = q.w_now = 0.f; q.now_s = 200.0;
        n4.step(q, pp);
        CHECK("I2: the doorway's corridor survives the next decision (goal still reachable)", std::isfinite(n4.field().at({2.f, 2.f})));
        // ---- I3: a Doorway is never Arrived by decide(): standing ON the goal line, the session's protocol decides
        q.pose = {Eigen::Vector2f(6.f, 2.f), 0.f}; q.now_s = 300.0;
        n4.interrupt();
        const auto on_line = n4.step(q, pp);
        CHECK("I3: on a doorway's goal line decide() does not report Arrived", on_line.d.mode != Decision::Mode::Arrived);
    }

    std::printf(fails == 0 ? "ALL PASS\n" : "FAILURES (%d failing)\n", fails);
    return fails == 0 ? 0 : 1;
}
