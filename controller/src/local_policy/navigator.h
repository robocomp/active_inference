// navigator.h — the stateful layer around decide(): the two cost-to-go fields, the ProgressMonitor, trap laying and
// the Unreachable verdict. Pure (Eigen + std, no Qt/DSR): the Task-6 harness and the controller call the SAME code.
//
// API (one Navigator per robot; the caller owns the world model and the clock):
//   set_goal(goal, grid, free, extra, goal_points, trap_radius_m)
//       builds the TRAP-FREE field (progress_field) for this goal and copies it as the trapped field, clears every
//       trap and resets the monitor — a new goal is a new plan. `free` / `extra` / `goal_points` are kept: the trapped
//       field is rebuilt from them whenever a trap is laid. trap_radius_m = the body's circumscribed radius.
//   step(in, params) -> Step
//       fills in.goal / in.field / in.progress_field, calls decide(in, params, &monitor) and, ONLY on a Release that
//       the monitor caused (Decision::prediction_failed), lays a trap (below); then reports Unreachable when no plan
//       avoids the traps from the robot's pose. The caller sends Step::d's (v, w) and stops on Arrived / Stuck /
//       Blind / unreachable.
//   field(), progress_field(), traps(), goal_points(): read-only views (display, oracles).
//
// ★ TRAPS (Task-6 rulings, consolidated): a trap asserts "the body cannot get through from here", and only a FAILED
//   PREDICTION is evidence of that. Centre = the believed pose + trap_radius downhill on the TRAPPED field (the field the
//   failed plan was descending), radius = trap_radius, cost_m = 2 x the largest finite value of the TRAP-FREE field
//   (crossing it costs more than any route that avoids it; taken from the trap-free field so successive traps do not
//   compound each other's cost). Never over a goal point: the goal cannot be routed around — the Release itself is the
//   response and the caller's time budget ends a persistent failure.
// ★ UNREACHABLE: "no plan without the traps" — a flood from the goal points over the free cells minus every trap's
//   FORWARD half-disc (the side the failed plan was pulling toward) must reach a cell within kReachRadiusM of the robot.
//   DEVIATION from the review's wording ("the start is unreachable on the trapped field"): traps are COSTS, so the
//   trapped field never makes a cell unreachable and that test could never fire; the flood is the same question asked
//   of the traps' meaning. kReachRadiusM = 0.25 m ~ the inscribed radius (0.23) rounded up to whole cells: the cells
//   the body's centre can step into from where it stands.
// ★ MONITOR CONTRACT: reset on set_goal (a new trap-free field) only; laying a trap does not touch the trap-free field,
//   and any Release has already cleared the pending prediction inside decide() (review M1).
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
    struct Trap { Eigen::Vector2f c = Eigen::Vector2f::Zero(), dir = Eigen::Vector2f::Zero(); float r = 0.f, cost = 0.f; };
    struct Step { Decision d{.mode = Decision::Mode::Stuck, .why = {}}; std::optional<Trap> trap; bool trap_suppressed_by_goal = false; bool unreachable = false; };
    static constexpr float kReachRadiusM = 0.25f;

    void set_goal(const Goal& goal, const CostToGo::Grid& grid, std::function<bool(const Eigen::Vector2f&)> free,
                  std::function<float(const Eigen::Vector2f&)> extra, std::vector<Eigen::Vector2f> goal_points, float trap_radius_m);
    Step step(PolicyInput in, const PolicyParams& p);

    const CostToGo& field() const { return trapped_; }
    const CostToGo& progress_field() const { return free_field_; }
    const std::vector<Trap>& traps() const { return traps_; }
    const std::vector<Eigen::Vector2f>& goal_points() const { return goals_; }
    const Goal& goal() const { return goal_; }
    // flood without the traps' forward half-discs (public for tests and display)
    bool plan_exists_from(const Eigen::Vector2f& p) const;
    // downhill direction (-grad, unit) of the trapped field at p
    Eigen::Vector2f downhill(const Eigen::Vector2f& p) const;
    double last_build_ms() const { return last_build_ms_; }

private:
    void rebuild_trapped();
    float max_finite(const CostToGo& f) const;
    Goal goal_{.kind = Goal::Kind::Place, .yaw = {}};
    CostToGo::Grid grid_{{0.f, 0.f}, 0.05f, 0, 0};
    std::function<bool(const Eigen::Vector2f&)> free_;
    std::function<float(const Eigen::Vector2f&)> extra_;
    std::vector<Eigen::Vector2f> goals_;
    float trap_r_ = 0.f;
    CostToGo free_field_, trapped_;
    std::vector<Trap> traps_;
    ProgressMonitor monitor_;
    double last_build_ms_ = 0.0;
};
}  // namespace rc::policy
