// navigator.h — the stateful layer around decide(): the two cost-to-go fields, the ProgressMonitor, trap laying and
// the Unreachable verdict. Pure (Eigen + std, no Qt/DSR): the Task-6 harness and the controller call the SAME code.
//
// API (one Navigator per robot; the caller owns the world model and the clock):
//   set_goal(goal, grid, free, extra, goal_points, trap_radius_m)
//       builds the field for this goal, clears every trap and resets the monitor — a new goal is a new plan. `free` /
//       `extra` / `goal_points` are kept: the field is rebuilt from them whenever a trap is laid.
//       trap_radius_m = the body's circumscribed radius.
//   step(in, params) -> Step
//       fills in.goal / in.field (ONE field: selection, prediction and judgement — review fix round 2), denoises the
//       returns ONCE, and either CONTINUES the committed plan or asks decide(in, params, &monitor) for a new one (below).
//       ONLY on a Release that the monitor caused (Decision::prediction_failed) it lays a trap, rebuilds the field and
//       reports Unreachable when the goal is no longer reachable from the robot. The caller sends Step::d's (v, w) —
//       the plan's command at this instant — and stops on Arrived / Stuck / Blind / unreachable. in.now_s must be set.
//
// ★ PLAN COMMITMENT (review fix round 3: the remaining failures all came from re-deciding every tick — Drive/Release
//   alternation and +-w flips that never let a prediction mature). decide() is the EVALUATOR; the Navigator commits to
//   what it chose: the candidate (v, w) held for horizon_s then braked (Decision::plan_dur1 in all) and, for a Drive
//   whose rate came from a continuation, that continuation from rest (plan_v2, plan_w2) held for horizon_s then braked
//   (plan_dur2) — for a Release, the release motion. Every tick it re-evaluates ONLY the plan's REMAINING sweep (from
//   the latency-advanced pose and the current velocity) against the fresh returns, and replaces the plan iff
//     (a) that sweep's P(contact) > risk_eps (> release_risk_max for a Release plan),
//     (b) the plan completed (its own duration elapsed),
//     (c) the goal or the field changed (a trap laid, set_goal) or the input is Blind / the goal is reached.
//   No other switching. Round 4: a plan COMPLETES at the end of its HOLDS (the brake tail is a safety tail, executed only
//   when no new plan passes its bar) and the next is decided while moving. EVERY completed plan is judged by the monitor
//   (a Drive on the field descent, a Release on the continuability it bought); a plan replaced by (a) or (c) is not
//   judged — the monitor is reset first.
//   State: plan_ (active, mode, the two phases, start time) — see Plan below.
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
    struct Step { Decision d{.mode = Decision::Mode::Stuck, .why = {}}; std::optional<Trap> trap; bool trap_suppressed_by_goal = false; bool unreachable = false;
                  bool replanned = false; float plan_remaining_p = 0.f; };
    // The committed plan: phase 1 = (v1, w1) for horizon_s then brake, dur1 in all; phase 2 (if has2) = (v2, w2) from rest
    // for horizon_s then brake, dur2 in all. t0 = the decision time it started.
    struct Plan { bool active = false; Decision::Mode mode = Decision::Mode::Drive; float v1 = 0, w1 = 0, dur1 = 0, v2 = 0, w2 = 0, dur2 = 0;
                  bool has2 = false; double t0 = 0.0; Decision d{.mode = Decision::Mode::Drive, .why = {}}; };

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
    const Plan& plan() const { return plan_; }

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
    Plan plan_;
    // the plan's command at elapsed time e, and its remaining sweep from (start, v0, w0)
    std::pair<float, float> command_at(float e, const PolicyParams& p) const;
    float plan_length(const PolicyParams& p) const;   // the plan's holds: horizon_s, x2 for a two-step plan
    std::vector<Pose2> remaining_sweep(const Pose2& start, float v0, float w0, float e, const BaseLimits& b, const PolicyParams& p) const;
    double last_build_ms_ = 0.0;
};
}  // namespace rc::policy
