// Defect A (controller face-the-door, 2026-11-15): an orient turns the body through an ARC, not all the way round.
// onemodeloff01: aff_door_3_open, orient 62 deg off, refused "4/8 headings blocked, nearest 0.40 m, bearing 176 deg" -- cells
// BEHIND the robot that the 62 deg arc never reaches. GridPlanner::why_cannot_sweep asks about the swept arc only.
// Build+run (from controller/; pure Eigen + std):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -Isrc tests/test_orient_sweep.cpp src/grid_planner.cpp
//       src/local_policy/cost_to_go.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tos && /tmp/tos   (one line)
#include "grid_planner.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace
{
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (not ok) ++failures;
}
constexpr float kPi = std::numbers::pi_v<float>;
constexpr float kDeg = kPi / 180.f;
const std::vector<Eigen::Vector2f> kRoom{{0.f, 0.f}, {6.f, 0.f}, {6.f, 6.f}, {0.f, 6.f}};

// a short wall of occupied cells (0.05 m) centred `off` from the robot along `bearing_rad`, `n` cells long, across the bearing
rc::GridPlanner planner_with_wall(const Eigen::Vector2f& robot, float bearing_rad, float dist, int n)
{
    std::vector<Eigen::Vector2f> cells;
    const Eigen::Vector2f u(std::cos(bearing_rad), std::sin(bearing_rad)), t(-u.y(), u.x());
    for (int i = -n; i <= n; ++i) cells.push_back(robot + dist * u + 0.05f * static_cast<float>(i) * t);
    rc::GridPlanner p;
    p.params.cell_size_m = 0.05f;
    p.params.safety_margin_m = 0.05f;
    p.set_world(kRoom, {}, rc::OccupiedCells{.centres = cells, .cell_size_m = 0.05f, .risk_nats = {}});
    return p;
}
}  // namespace

int main()
{
    const Eigen::Vector2f robot(3.f, 3.f);
    // obstacle 0.40 m from the centre, straight behind the robot (the robot faces yaw 0 = +x, so behind = bearing 180 deg)
    auto p = planner_with_wall(robot, 180.f * kDeg, 0.40f, 3);
    const auto full = p.why_cannot_turn(robot);
    std::printf("  full circle: %d/8 headings blocked, %d cell(s)\n", full.headings_blocked, full.cells_blocked);
    check(full.blocked, "the OLD question (all the way round) refuses: the cells behind block some heading");

    // The onemodeloff01 case: cells 0.40 m behind (bearing 180) block the DIAGONAL headings 1,3,5,7 of the 8-heading grid ("4/8
    // headings blocked, nearest 0.40 m, bearing 176"). The robot stands at yaw -56 deg (bucket 7, a blocked one: the grid's footprint
    // has a margin and the robot is physically there), the door asks for +6 deg: a 62 deg turn through buckets 7 -> 0, away from them.
    const auto a = p.why_cannot_sweep(robot, -56.f * kDeg, 6.f * kDeg);
    check(not a.blocked, "A1: the 62 deg orient (the cells it never reaches are behind it) is feasible");
    check(not p.why_cannot_sweep(robot, 0.f, 0.f).blocked, "A1': no turn at all is feasible");
    check(not p.why_cannot_sweep(robot, -56.f * kDeg, -56.f * kDeg).blocked,
          "A1'': a body standing in a margin-blocked bucket may stay there (cells under its own footprint are not newly swept)");

    // B: the SAME cells, a turn whose arc runs into another blocked bucket: refused, with the obstruction named
    const auto b = p.why_cannot_sweep(robot, -56.f * kDeg, 150.f * kDeg);   // short way: -56 -> -154 through -90, -135 (bucket 5)
    std::printf("  206 deg arc: %d heading(s) blocked, nearest %.2f m at bearing %.0f deg\n", b.headings_blocked, b.nearest_m,
                b.nearest_bearing_deg);
    check(b.blocked and b.headings_blocked >= 1, "A2: an arc that sweeps into the cells is refused (Infeasible kept)");
    check(p.why_cannot_sweep(robot, 0.f, 62.f * kDeg).blocked, "A2': from yaw 0 the 62 deg arc enters bucket 1, where the cells are: refused");

    // the short way is the one measured: from 100 deg to -100 deg is 160 deg THROUGH 180 (the cells), not 200 the other way
    check(p.why_cannot_sweep(robot, 100.f * kDeg, -100.f * kDeg).blocked, "A3: the arc is the SHORT way round (through 180)");
    // ...and the same two yaws seen from a robot with the cells on the OTHER side are free
    auto q = planner_with_wall(robot, 0.f, 0.40f, 3);
    check(q.why_cannot_sweep(robot, 100.f * kDeg, -100.f * kDeg).blocked == q.why_cannot_sweep(robot, -100.f * kDeg, 100.f * kDeg).blocked,
          "A3': the short way round is symmetric in its endpoints");

    // a body hemmed in closely is refused any arc that reaches cells its own footprint does not already cover
    auto boxed = planner_with_wall(robot, 45.f * kDeg, 0.33f, 6);
    check(boxed.why_cannot_sweep(robot, 0.f, 90.f * kDeg).blocked, "A4: a body hemmed in at 0.33 m cannot swing into new cells");

    std::printf("test_orient_sweep: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
