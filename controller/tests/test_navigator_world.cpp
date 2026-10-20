// Task 8 (local-motion policy): Navigator::update_world — the planner's map changes under a goal; the change is applied
// between plans (never interrupting the committed one), traps kept, the monitor's baseline re-read on the new field.
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -I../common tests/test_navigator_world.cpp
//     src/local_policy/local_policy.cpp src/local_policy/contact.cpp src/local_policy/cost_to_go.cpp src/local_policy/navigator.cpp
//     ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp -o /tmp/tnw && /tmp/tnw
#include "../src/local_policy/navigator.h"

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

    std::printf(fails == 0 ? "ALL PASS\n" : "FAILURES (%d failing)\n", fails);
    return fails == 0 ? 0 : 1;
}
