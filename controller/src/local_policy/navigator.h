// navigator.h — the stateful layer around decide(): the two cost-to-go fields, the ProgressMonitor, trap laying and
// the Unreachable verdict. Pure (Eigen + std, no Qt/DSR): the Task-6 harness and the controller call the SAME code.
//
// API (one Navigator per robot; the caller owns the world model and the clock):
//   set_goal(goal, grid, free, extra, goal_points, trap_radius_m)
//       builds the field for this goal, clears every trap and resets the monitor — a new goal is a new plan. `free` /
//       `extra` / `goal_points` are kept: the field is rebuilt from them whenever a trap is laid.
//       trap_radius_m = the body's circumscribed radius.
//   step(in, params) -> Step
//       fills in.goal / in.field (ONE field: selection, prediction and judgement — review fix round 2), calls
//       decide(in, params, &monitor) and, ONLY on a Release that the monitor caused (Decision::prediction_failed), lays
//       a trap (below), rebuilds the field and reports Unreachable when the goal is no longer reachable from the robot.
//       The caller sends Step::d's (v, w) and stops on Arrived / Stuck / Blind / unreachable.
//   field(), traps(), goal_points(): read-only views (display, oracles).
//
// ★ TRAPS ARE BLOCKED REGIONS (review fix round 2): a trap asserts "the body cannot get through from here", and only a
//   FAILED PREDICTION is evidence of that. The disc (centre = the believed pose + trap_radius downhill, radius =
//   trap_radius) is made NON-FREE in the field (CostToGo::block_disc): the field becomes the true detour distance around
//   it. Never over a goal point: the goal cannot be routed around — the Release itself is the response and the caller's
//   time budget ends a persistent failure.
// ★ UNREACHABLE (literal now): the goal is not reachable from the robot on the trapped field (CostToGo::reachable).
// ★ MONITOR CONTRACT: reset on set_goal and when a trap rebuilds the field (old field values are not comparable); any
//   Release has already cleared the pending prediction inside decide() (review M1).
#pragma once
#include <functional>
#include <optional>
#include <vector>
#include "local_policy.h"

namespace rc::policy
{
class Navigator
{
public:
    struct Trap { Eigen::Vector2f c = Eigen::Vector2f::Zero(), dir = Eigen::Vector2f::Zero(); float r = 0.f; };
    struct Step { Decision d{.mode = Decision::Mode::Stuck, .why = {}}; std::optional<Trap> trap; bool trap_suppressed_by_goal = false; bool unreachable = false; };

    void set_goal(const Goal& goal, const CostToGo::Grid& grid, std::function<bool(const Eigen::Vector2f&)> free,
                  std::function<float(const Eigen::Vector2f&)> extra, std::vector<Eigen::Vector2f> goal_points, float trap_radius_m);
    Step step(PolicyInput in, const PolicyParams& p);

    const CostToGo& field() const { return field_; }
    const std::vector<Trap>& traps() const { return traps_; }
    const std::vector<Eigen::Vector2f>& goal_points() const { return goals_; }
    const Goal& goal() const { return goal_; }
    // downhill direction (-grad, unit) of the field at p
    Eigen::Vector2f downhill(const Eigen::Vector2f& p) const;
    double last_build_ms() const { return last_build_ms_; }

private:
    void build();
    Goal goal_{.kind = Goal::Kind::Place, .yaw = {}};
    CostToGo::Grid grid_{{0.f, 0.f}, 0.05f, 0, 0};
    std::function<bool(const Eigen::Vector2f&)> free_;
    std::function<float(const Eigen::Vector2f&)> extra_;
    std::vector<Eigen::Vector2f> goals_;
    float trap_r_ = 0.f;
    CostToGo field_;
    std::vector<Trap> traps_;
    ProgressMonitor monitor_;
    double last_build_ms_ = 0.0;
};
}  // namespace rc::policy
