#pragma once
/*
 * door_passage_corridor.h — the free corridor through a doorway being crossed, as a polygon in the room frame.
 *
 * DSR-free on purpose: DoorPassage (controller_door_passage.h) builds it from the graph, and GridPlanner::self_test
 * builds the SAME polygon from the live run's numbers, so the test cannot drift from what the controller plans in.
 *
 * ★ANCHORED ON THE CROSS STANDPOINT AND ITS HEADING, NOT ON THE DOOR NODE. door_concept publishes the through-pose as
 * aperture centre − side·standoff·n with yaw = the travel direction (door_pragmatics.h through_pose), so walking back
 * along that heading lands on the aperture centre by construction. The door node's pose is the LEAF's, and an open
 * leaf has swung beside the doorway: live 2026-10-11 the leaf sat at (-1.05,5.12), half a metre PAST the wall line at
 * y=4.61, so a corridor anchored there never reached the room polygon — the goal was free and unreachable
 * ("no route: 72335 expansions over 16413 free cells"). GridPlanner::self_test (12) reproduces both.
 */

#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

/// Rectangle `width_m` wide along `heading` (room yaw, +x at 0): from `back_m` behind `goal` (through the aperture and
/// on into the room just left, so the corridor OVERLAPS the room polygon instead of merely touching it) to `ahead_m`
/// past it (so the body fits wholly inside when it stands on the goal).
inline std::vector<Eigen::Vector2f> door_passage_corridor(const Eigen::Vector2f& goal, float heading, float width_m,
                                                          float back_m, float ahead_m)
{
    const Eigen::Vector2f d(std::cos(heading), std::sin(heading)), n(-d.y(), d.x());
    const Eigen::Vector2f a = goal - back_m * d, b = goal + ahead_m * d;
    const float h = 0.5f * width_m;
    return {a - h * n, b - h * n, b + h * n, a + h * n};
}

}   // namespace rc
