// Defect A (controller face-the-door, 2026-11-15, fix round 2): an orient turns the body through an ARC, and the question is
// EXACT GEOMETRY -- does the bare footprint polygon, rotating about the robot's position, intersect an occupied cell's square?
// (Not the planner's 8 heading buckets + safety margin: onemodeloff01 refused a 62 deg turn "4/8 headings blocked, nearest 0.40 m,
// bearing 176" for two cells the real body may never touch.) A cell the body already overlaps is forgiven only while the overlap
// AREA does not grow.
// Build+run (from controller/; pure Eigen + std):
//   g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc tests/test_orient_sweep.cpp src/grid_planner.cpp
//       src/local_policy/cost_to_go.cpp ../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp
//       -o /tmp/tos && /tmp/tos   (one line)
#include "grid_planner.h"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string_view>
#include <utility>
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

rc::GridPlanner planner_with(const std::vector<Eigen::Vector2f>& room, const std::vector<Eigen::Vector2f>& cells, float cell,
                             std::optional<rc::RobotFootprint> fp = std::nullopt)
{
    rc::GridPlanner p;
    p.params.cell_size_m = cell;
    p.params.safety_margin_m = 0.05f;
    if (fp) p.set_footprint(*fp);
    p.set_world(room, {}, rc::OccupiedCells{.centres = cells, .cell_size_m = cell, .risk_nats = {}});
    return p;
}
const std::vector<Eigen::Vector2f> kRoom{{0.f, 0.f}, {6.f, 0.f}, {6.f, 6.f}, {0.f, 6.f}};
// a short wall of occupied cells centred `dist` from the robot along `bearing`, 2n+1 cells across
std::vector<Eigen::Vector2f> wall(const Eigen::Vector2f& robot, float bearing, float dist, int n, float cell)
{
    std::vector<Eigen::Vector2f> c;
    const Eigen::Vector2f u(std::cos(bearing), std::sin(bearing)), t(-u.y(), u.x());
    for (int i = -n; i <= n; ++i) c.push_back(robot + dist * u + cell * static_cast<float>(i) * t);
    return c;
}
float reach_of(const rc::RobotFootprint& f)
{
    float r = 0.f;
    for (const auto& v : f.at_pose_yaw({0.f, 0.f}, 0.f, false)) r = std::max(r, v.norm());
    return r;
}
}  // namespace

int main()
{
    const Eigen::Vector2f robot(3.f, 3.f);
    const rc::RobotFootprint shadow = rc::RobotFootprint::shadow();
    std::printf("  shadow hull reach (farthest vertex) %.3f m\n", reach_of(shadow));

    // cells beyond the body's reach (0.45 m behind): no turn of any size touches them -- the old 8-heading test refused these
    {
        auto p = planner_with(kRoom, wall(robot, 180.f * kDeg, 0.45f, 1, 0.05f), 0.05f);
        check(not p.why_cannot_sweep(robot, 0.f, 170.f * kDeg).blocked, "A1: a turn of 170 deg past cells outside the hull's reach is free");
        check(not p.why_cannot_sweep(robot, -56.f * kDeg, 150.f * kDeg).blocked, "A1': so is the long -56 -> 150 sweep");
    }
    // a cell INSIDE the reach: the body's lateral wheel face (0.284) swings onto it
    {
        auto p = planner_with(kRoom, wall(robot, 180.f * kDeg, 0.30f, 0, 0.05f), 0.05f);
        check(not p.why_cannot_sweep(robot, 0.f, 0.f).blocked, "A2: no turn: free");
        check(p.why_cannot_sweep(robot, 0.f, 90.f * kDeg).blocked, "A2': yaw 0 -> 90 swings the wheel onto the cell 0.30 m behind: refused");
    }
    // a cell the body ALREADY overlaps at the start yaw (found by search: at yaw 90 deg the cell 0.20 m behind / 0.24 m to the side
    // of the robot's centre overlaps the hull by 0.0013 m2): forgiven while the overlap does not grow, refused when it does
    {
        const Eigen::Vector2f r(3.025f, 3.025f), c = r + Eigen::Vector2f(-0.20f, -0.24f);
        auto p = planner_with(kRoom, {c}, 0.05f);
        check(p.body_cell_overlap_m2(r, 90.f * kDeg, c) > 0.f, "R2a: the start pose already overlaps that cell");
        check(not p.why_cannot_sweep(r, 90.f * kDeg, 90.f * kDeg).blocked, "R2b: not turning: allowed");
        check(not p.why_cannot_sweep(r, 90.f * kDeg, 80.f * kDeg).blocked, "R2c: turning the way the overlap does not grow: allowed");
        check(p.why_cannot_sweep(r, 90.f * kDeg, 100.f * kDeg).blocked, "R2d: turning the way the overlap grows: refused");
    }

    // NEGATIVE, at the live lattice (0.10 m): a 0.05 m cell exactly where the hull's lateral wheel face (0.284 m) ends the arc
    // (start yaw 1.18, end 0.10: the wheel axis ends along 0.10 + 90 deg): the arc really reaches it -> refused
    {
        const std::vector<Eigen::Vector2f> room{{0.f, -4.f}, {5.f, -4.f}, {5.f, 0.5f}, {0.f, 0.5f}};
        const Eigen::Vector2f r(2.07f, -1.59f);
        const float wy = 0.10f + 0.5f * kPi;
        const Eigen::Vector2f c = r + 0.28f * Eigen::Vector2f(std::cos(wy), std::sin(wy));
        rc::GridPlanner p;
        p.params.cell_size_m = 0.10f;
        p.params.safety_margin_m = 0.05f;
        p.set_world(room, {}, rc::OccupiedCells{.centres = {c}, .cell_size_m = 0.05f, .risk_nats = {}});
        check(p.why_cannot_sweep(r, 0.10f + 1.08f, 0.10f).blocked, "N1: a cell where the hull's wheel face really sweeps: refused (live lattice)");
        check(not p.why_cannot_sweep(r, 0.10f, 0.10f).blocked, "N2: the same cell, no turn at all: free");
    }

    // THE REAL onemodeloff01 GEOMETRY (controller.log: robot (2.07,-1.59); two cells around (1.67,-1.56), 0.40 m, bearing 176 deg;
    // aff_door_3_open target yaw +0.10 rad, the robot 62 deg off it). The blocking cells are residual's 0.05 m cells; the planner
    // stamps them into ITS lattice (0.10 m cells in the live grid), so each verdict is given on both lattices. The start yaw is
    // unknown from the log, so BOTH 0.10 + 1.08 and 0.10 - 1.08; and for the shadow hull AND the 0.46 x 0.65 m rectangle
    // (half-diagonal 0.398 m).
    {
        const std::vector<Eigen::Vector2f> room{{0.f, -4.f}, {5.f, -4.f}, {5.f, 0.5f}, {0.f, 0.5f}};
        const Eigen::Vector2f r(2.07f, -1.59f);
        const std::vector<Eigen::Vector2f> cells{{1.67f, -1.56f}, {1.67f, -1.61f}};   // residual cells, 0.05 m
        const auto rect = rc::RobotFootprint::from_polygon({{-0.23f, -0.325f}, {0.23f, -0.325f}, {0.23f, 0.325f}, {-0.23f, 0.325f}});
        check(rect.has_value(), "the 0.46 x 0.65 rectangle is a valid footprint");
        for (const auto& [name, fp] : {std::pair{"shadow hull", shadow}, std::pair{"0.46x0.65 rectangle", *rect}})
            for (const float lattice : {0.05f, 0.10f})
            {
                rc::GridPlanner p;
                p.params.cell_size_m = lattice;
                p.params.safety_margin_m = 0.05f;
                p.set_footprint(fp);
                p.set_world(room, {}, rc::OccupiedCells{.centres = cells, .cell_size_m = 0.05f, .risk_nats = {}});
                std::printf("  %s (reach %.3f m), planner lattice %.2f m:\n", name, reach_of(fp), lattice);
                for (const float start : {0.10f + 1.08f, 0.10f - 1.08f})
                {
                    const auto v = p.why_cannot_sweep(r, start, 0.10f);
                    // the nearest hull-to-cell-edge geometry, by sampling the arc against the cells' own squares
                    float amax = 0.f;
                    for (const auto& c : cells)
                        for (int i = 0; i <= 400; ++i)
                            amax = std::max(amax, p.body_cell_overlap_m2(r, start + (0.10f - start) * static_cast<float>(i) / 400.f, c));
                    std::printf("    start yaw %+.2f -> +0.10: %s (%d cell(s); max overlap on the cells' lattice squares %.5f m2)\n", start,
                                v.blocked ? "REFUSED" : "free", v.cells_blocked, amax);
                    // on a lattice equal to residual's 0.05 m the grid holds the cells' true squares: the hull (reach 0.318 m) cannot touch them.
                    // On 0.10 m the stamp enlarges each 0.05 m cell into the enclosing lattice squares (reported, not asserted).
                    // fix round 3: the sweep tests each residual cell at its TRUE 0.05 m extent, whatever the lattice -- so the live 0.10 m
                    // lattice gives the same verdict as the 0.05 m one
                    if (std::string_view(name) == "shadow hull")
                        check(not v.blocked, "onemodeloff01 (the planner's shadow hull, 0.05 m residual cells, lattice as printed): free from either start yaw");
                    else
                        check(v.blocked, "(for the record) a 0.46x0.65 rectangle's corner (0.398 m) would reach them -- refused");
                }
            }
    }

    std::printf("test_orient_sweep: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
