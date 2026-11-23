// AuRo journal item B (2026-11-23): the known-room planner with the change belief on, in an L-shaped room.
// Links epistemic_planner.cpp (see src/CMakeLists.txt change_planner_test).
#include "epistemic_planner.h"
#include <cmath>
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char *what)
{ if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

int main()
{
    // L room: 6 x 3 bottom arm, 3 x 3 upper arm on the left. Robot deep in the bottom-right arm.
    const std::vector<Eigen::Vector2f> L = {{0, 0}, {6, 0}, {6, 3}, {3, 3}, {3, 6}, {0, 6}};
    for (const bool on : {false, true})
    {
        rc::EpistemicPlanner::Params p;
        p.use_change_belief = on;
        rc::EpistemicPlanner pl(p);
        pl.set_room_bounds({0.f, 0.f}, {6.f, 6.f});
        pl.set_room_polygon(L);
        Eigen::Affine2f pose = Eigen::Affine2f::Identity(); pose.translate(Eigen::Vector2f(5.f, 1.5f));
        pl.set_robot_state(pose, Eigen::Matrix3f::Identity() * 1e-4f);
        pl.change_enter_room("room_L", {0.f, 0.f}, {6.f, 6.f}, Eigen::Affine2f::Identity());   // published = internal here
        pl.set_change_unknown_nats(0.1);
        const auto t = pl.update_target();   // marks the robot's surroundings and runs one sweep
        std::printf("change belief %s\n", on ? "ON" : "OFF");
        const Eigen::Vector2f hidden(1.5f, 5.f), seen(4.5f, 1.5f);
        if (not on)
        {
            check(pl.change_info(hidden) == 0.f, "off: no change information");
            continue;
        }
        const auto vis = pl.visible_cells(pose.translation());
        int in_view = 0;
        for (int k = 0; k < pl.change_belief().size(); ++k) in_view += pl.change_belief().cell(k).in_view;
        check(in_view == static_cast<int>(vis.size()) and in_view > 0, "the sweep opened an episode on every visible cell");
        const float g_hidden = pl.change_info(hidden), g_seen = pl.change_info(seen);
        std::printf("     change information: from the hidden arm %.3f nats, from a visible spot %.3f nats (%d cells in view)\n",
                    g_hidden, g_seen, in_view);
        check(g_hidden > g_seen, "the hidden arm offers more than a spot whose view is already in view");
        check(g_hidden > 0.f, "cells with no observation on record carry their unknown state's information");
        if (t.has_value())
        {
            std::printf("     target (%.2f, %.2f)\n", t->position.x(), t->position.y());
            check(t->position.y() > 3.f or t->position.x() < 3.f, "the target lies where the unseen cells are visible from");
        }
        else check(false, "a target is chosen");
    }
    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
