// objects_eps1 leg 2 (2026-11-20): the robot ended a policy episode where the planner's raster said its body did not fit.
// The planner's start escape then walked to the NEAREST footprint-feasible cell, which lay inside a sealed pocket
// (flood fill over 284 cells), so every route — the look-around's and then the door's — failed: cross_unreachable.
// The escape must prefer, within the body's reach, a cell from which the goal is reachable.
// Build+run (from controller/; pure Eigen + std):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_grid_planner_escape.cpp src/grid_planner.cpp
//       src/local_policy/cost_to_go.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tgpe && /tmp/tgpe   (one line)

#include "grid_planner.h"

#include <cstdio>
#include <string>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, const char *what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (not ok) ++failures;
}

const std::vector<Eigen::Vector2f> kRoom{{0.f, 0.f}, {6.f, 0.f}, {6.f, 6.f}, {0.f, 6.f}};
constexpr float kRes = 0.05f;

// A closed box of hard residual cells, two cells thick: x in [x0, x1], y in [y0, y1]. Its inside is a sealed pocket
// the body fits in.
std::vector<Eigen::Vector2f> box(float x0, float x1, float y0, float y1)
{
    std::vector<Eigen::Vector2f> c;
    for (float x = x0; x <= x1 + 1e-4f; x += kRes)
        for (const float y : {y0, y0 + kRes, y1 - kRes, y1}) c.emplace_back(x, y);
    for (float y = y0; y <= y1 + 1e-4f; y += kRes)
        for (const float x : {x0, x0 + kRes, x1 - kRes, x1}) c.emplace_back(x, y);
    return c;
}

rc::GridPlanner planner_with(const std::vector<Eigen::Vector2f> &cells)
{
    rc::GridPlanner p;
    p.params.cell_size_m = 0.10f;
    p.params.safety_margin_m = 0.05f;
    p.params.allow_start_in_collision = true;
    p.set_world(kRoom, {}, rc::OccupiedCells{.centres = cells, .cell_size_m = kRes, .risk_nats = {}, .live_hard = {}});
    return p;
}
}  // namespace

int main()
{
    // The box's east wall is at x = 3.0 (cells 2.95 .. 3.0); the goal is out in the open at (5, 3).
    const auto cells = box(1.0f, 3.0f, 1.0f, 5.0f);
    const Eigen::Vector2f goal{5.0f, 3.0f};

    std::printf("1. start overlapping the pocket's wall, just INSIDE it: the goal side is within the body's reach\n");
    {
        auto p = planner_with(cells);
        const auto path = p.plan({2.90f, 3.0f}, goal);
        check(path.has_value(), "a route exists: the escape takes the goal-connected side of the wall");
        if (not path.has_value()) std::printf("    failure: %s\n", p.last_failure().c_str());
    }

    std::printf("\n2. start deep inside the sealed pocket, body free: no route (unchanged)\n");
    {
        auto p = planner_with(cells);
        const auto path = p.plan({2.0f, 3.0f}, goal);
        check(not path.has_value(), "no route out of a truly sealed pocket");
        check(p.last_failure().find("not connected") != std::string::npos, "reported as not connected");
    }

    std::printf("\n3. start overlapping the wall from OUTSIDE: still routes (unchanged)\n");
    {
        auto p = planner_with(cells);
        check(p.plan({3.10f, 3.0f}, goal).has_value(), "a route exists");
    }

    std::printf("\n4. the same pocket walled by an OBSTACLE POLYGON (matter, not residual evidence): never escaped through\n");
    {
        rc::GridPlanner p;
        p.params.cell_size_m = 0.10f;
        p.params.safety_margin_m = 0.05f;
        p.params.allow_start_in_collision = true;
        const std::vector<std::vector<Eigen::Vector2f>> walls{
            {{0.95f, 0.95f}, {3.05f, 0.95f}, {3.05f, 1.05f}, {0.95f, 1.05f}},
            {{0.95f, 4.95f}, {3.05f, 4.95f}, {3.05f, 5.05f}, {0.95f, 5.05f}},
            {{0.95f, 0.95f}, {1.05f, 0.95f}, {1.05f, 5.05f}, {0.95f, 5.05f}},
            {{2.95f, 0.95f}, {3.05f, 0.95f}, {3.05f, 5.05f}, {2.95f, 5.05f}}};
        p.set_world(kRoom, walls);
        const auto path = p.plan({2.90f, 3.0f}, goal);
        check(not path.has_value(), "no route: a wall is not crossed to reach the goal side");
    }

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
