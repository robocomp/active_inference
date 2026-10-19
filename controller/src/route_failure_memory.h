/*
 * route_failure_memory.h — "routing to THIS target already failed from HERE", remembered across cycles.
 *
 * ── WHY IT IS ITS OWN TYPE (2026-10-19, try96 leg 3) ─────────────────────────────────────────────────
 * The controller's reachability fallback (GridPlanner::nearest_reachable: drive to the closest pose that DOES
 * route) runs only on the cycle after a plan() failed for the same target. That memory used to be keyed on
 * (goal, robot, world_hash()) — and the world hash changes on EVERY set_world, because residual's cells
 * change every cycle. So the record written by a failure never matched the next cycle's world, the fallback
 * never fired, the robot HELD for 5 s and the approach was reported Unreachable while a reachable pose
 * 0.6 m from the goal existed. A failed route is not a fact about one bitmap; it is a fact about a TARGET and
 * where the robot stood. A later bitmap that DOES route is caught for free: plan() is asked first every time,
 * and nearest_reachable returns the goal itself, exactly, the moment it is reachable.
 *
 * ── THE KEY ──────────────────────────────────────────────────────────────────────────────────────────
 *   name   — the target's node name (an affordance; a manual click has its own)
 *   goal   — the PRODUCER's raw pose, before any repair moves it. The repaired pose is ours and moves with
 *            the repair; keying on it is what once made this fallback dead code (see controller_session.cpp).
 *   robot  — where the robot stood when the route failed.
 * failed_here() holds when name and goal match and the robot has not moved materially since the failure —
 * OR when the fallback is already engaged for this target: driving toward the fallback pose moves the robot by
 * construction, and dropping the record then would put the unroutable goal straight back and re-solve from
 * wherever the robot had got to, i.e. a target that chases the robot.
 * "Materially" is the 5 cm the rest of the target identity already uses (same_target_instance), not a new
 * number.
 *
 * Pure std + Eigen — tested in tests/test_grid_planner_curtain.cpp.
 */

#pragma once

#include <optional>
#include <string>
#include <Eigen/Dense>

namespace rc
{

class RouteFailureMemory
{
public:
    static constexpr float kSameM = 0.05f;   // the 5 cm target identity used across the controller

    void note_failure(const std::string &name, const Eigen::Vector2f &raw_goal, const Eigen::Vector2f &robot)
    {
        rec_ = Record{.name = name, .goal = raw_goal, .robot = robot};
    }
    void clear() { rec_.reset(); }
    [[nodiscard]] bool has_record() const { return rec_.has_value(); }

    // Did routing to this target fail from (about) here? See the header for the stickiness.
    [[nodiscard]] bool failed_here(const std::string &name, const Eigen::Vector2f &raw_goal,
                                   const Eigen::Vector2f &robot, bool fallback_engaged) const
    {
        if (not rec_.has_value() or rec_->name != name) return false;
        if ((rec_->goal - raw_goal).squaredNorm() >= kSameM * kSameM) return false;
        return fallback_engaged or (rec_->robot - robot).squaredNorm() < kSameM * kSameM;
    }

private:
    struct Record
    {
        std::string     name;
        Eigen::Vector2f goal  = Eigen::Vector2f::Zero();
        Eigen::Vector2f robot = Eigen::Vector2f::Zero();
    };
    std::optional<Record> rec_;
};

}  // namespace rc
