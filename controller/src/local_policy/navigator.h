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
// ★ round 5: the trap marks where the body would go NEXT, beyond its current footprint — centre = pose + 2 trap_radius
//   downhill, so the disc never contains the robot (asserted); NO trap when the field at the robot is non-finite or its
//   downhill direction is undefined (|grad F| over two cells below the field's resolution there): the failure still
//   forces the Release, it carries no route evidence.
// ★ TRAPS ARE BLOCKED REGIONS (review fix round 2): a trap asserts "the body cannot get through from here", and only a
//   FAILED PREDICTION is evidence of that. The disc (centre = the believed pose + trap_radius downhill, radius =
//   trap_radius) is made NON-FREE in the field (CostToGo::block_disc): the field becomes the true detour distance around
//   it. Never over a goal point: the goal cannot be routed around — the Release itself is the response and the caller's
//   time budget ends a persistent failure.
// ★ UNREACHABLE (literal now): the goal is not reachable from the robot on the trapped field (CostToGo::reachable).
// ★ F1 — UNREACHABLE FROM A PLACE (reachability_belief.h): every step() folds its decision into a per-goal belief that no motion
//   escapes from the robot's place (a decision that found no escape — Stuck, a no-gain release, a stop — is evidence for it; one
//   that found an escape, against; motion leaves it behind on the field's cells) and reports Step::hand_back when holding one
//   more tick costs more than the goal is worth under that belief. A Stuck is no longer a verdict: the caller holds and asks again.
//   A hand back resets the belief (the same goal asked again is a new attempt); set_goal resets it; interrupt() keeps it.
// ★ MONITOR CONTRACT: reset on set_goal and when a trap rebuilds the field (old field values are not comparable); any
//   Release has already cleared the pending prediction inside decide() (review M1).
#pragma once
#include <functional>
#include <optional>
#include <vector>
#include "local_policy.h"
#include "reachability_belief.h"

namespace rc::policy
{
class Navigator
{
public:
    struct Trap { Eigen::Vector2f c = Eigen::Vector2f::Zero(), dir = Eigen::Vector2f::Zero(); float r = 0.f; };
    struct Step { Decision d{.mode = Decision::Mode::Stuck, .why = {}}; std::optional<Trap> trap; bool trap_suppressed_by_goal = false; bool unreachable = false;
                  bool world_applied = false;            // Task 8: a pending update_world() was applied this step
                  bool trap_vetoed = false;              // Live fix 2: the trap met the caller's veto region (measured corridor)
                  bool replanned = false; float plan_remaining_p = 0.f;
                  bool completed_early = false;          // the previous plan reached its predicted outcome early (success)
                  bool trap_no_route_evidence = false;   // a failed prediction where the field could not point the way
                  // ★ F1 (reachability_belief.h): the accumulated belief that no motion escapes from this place — its log-odds
                  //   at the robot, the hand-back bar log(route / (v_max dt)), and the verdict: hand the goal back iff L > bar.
                  //   This, not a Stuck tick, is the place-level "Unreachable" the caller hands back on.
                  float unreach_L = 0.f, unreach_bar = INFINITY; bool hand_back = false;
                  float unreach_dt_obs = 0.f; };                 // F1 round 3: the measured interval between observations (s)
    // The committed plan: phase 1 = (v1, w1) for horizon_s then brake, dur1 in all; phase 2 (if has2) = (v2, w2) from rest
    // for horizon_s then brake, dur2 in all. t0 = the decision time it started.
    // round 5: the plan runs PHASE by phase; a phase ends when its hold elapses OR its predicted end state is reached
    //   (heading-changing phase: the predicted turn reached, or the goal's heading within tolerance; translating phase:
    //   its predicted descent realized within the field's resolution; F2: a Release(stop) phase, the base at rest — below
    //   its deadband). phase_t0 / phase_yaw0 / phase_f0 / phase_p0 =
    //   where the current phase started.
    struct Plan { bool active = false; Decision::Mode mode = Decision::Mode::Drive; float v1 = 0, w1 = 0, dur1 = 0, v2 = 0, w2 = 0, dur2 = 0;
                  bool has2 = false; double t0 = 0.0; Decision d{.mode = Decision::Mode::Drive, .why = {}};
                  int phase = 1; double phase_t0 = 0.0; float phase_yaw0 = 0.f, phase_f0 = 0.f; Eigen::Vector2f phase_p0 = Eigen::Vector2f::Zero(); };

    void set_goal(const Goal& goal, const CostToGo::Grid& grid, std::function<bool(const Eigen::Vector2f&)> free,
                  std::function<float(const Eigen::Vector2f&)> extra, std::vector<Eigen::Vector2f> goal_points, float trap_radius_m);
    Step step(PolicyInput in, const PolicyParams& p);
    // ★ Task 8 — THE WORLD CHANGES UNDER A GOAL (the controller's planner map: furniture beliefs, residual cells). A new
    //   free/extra is NOT applied at once — that would replace the committed plan every cycle the map twitches (plan
    //   commitment, review fix round 3) — but at the next point step() decides a new plan anyway (the plan completed, was
    //   replaced, or there was none), AFTER the completed plan was judged on the field it was predicted on. Traps are
    //   kept (re-applied as blocked discs); a pending Drive prediction and the running plan's phase baseline are
    //   re-read on the new field at their own start points (a descent is a difference of field values: the baseline
    //   must be on the field the realized value will be read from); a pending Release prediction is dropped.
    //   The goal and its points are unchanged — a new goal is set_goal.
    void update_world(std::function<bool(const Eigen::Vector2f&)> free, std::function<float(const Eigen::Vector2f&)> extra)
    { pending_free_ = std::move(free); pending_extra_ = std::move(extra); world_pending_ = true; }
    bool world_pending() const { return world_pending_; }
    // ★ Task 8 ruling I1 — AN INTERRUPTION IS NOT EVIDENCE. After a Pause / Halt / hold / LiDAR stall the committed plan and
    //   the pending prediction are dropped WITHOUT being judged (the base did not execute them, so their outcome says
    //   nothing about the world): no failed prediction, no trap. Goal, field and traps are kept.
    //   F1: the reachability belief is KEPT (it is about places, and an interruption gathered nothing), but the interval to the
    //   next tick is a gap, not a tick period: the hand-back bar is not priced on it (ReachabilityBelief::interrupt).
    void interrupt() { plan_ = Plan{}; monitor_.reset(); belief_.interrupt(); claim_ = Claim{}; if (hold_since_ < 0.0) hold_since_ = last_now_; }
    // ★ Live fix 2 ruling 2 — MEASURED EVIDENCE OVERRIDES A TRAP. veto(c, r) true = a trap disc there contradicts direct
    //   evidence that the body fits (the active Doorway's MEASURED corridor, an opening at least the body's width): the
    //   failed prediction still forces its Release, but no trap is laid (Step::trap_vetoed). Cleared by set_goal.
    void set_trap_veto(std::function<bool(const Eigen::Vector2f&, float)> veto) { trap_veto_ = std::move(veto); }

    const CostToGo& field() const { return field_; }
    const std::vector<Trap>& traps() const { return traps_; }
    const std::vector<Eigen::Vector2f>& goal_points() const { return goals_; }
    const Goal& goal() const { return goal_; }
    // downhill direction (-grad, unit) of the field at p
    Eigen::Vector2f downhill(const Eigen::Vector2f& p) const;
    double last_build_ms() const { return last_build_ms_; }
    int failed_predictions() const { return failed_predictions_; }   // since set_goal
    const ReachabilityBelief& reachability() const { return belief_; }
    // F1: the planner's residual_risk_weight. ★ PARKED (fix round 2, live rtD1c): the route-price prior it fed summed correlated
    //   residual log-odds as independent nats and handed goals back on tick 1; the belief now starts at the base rate. Kept for
    //   a per-blob existence prior.
    void set_route_risk_weight(float w) { route_risk_weight_ = w; }
    const Plan& plan() const { return plan_; }

private:
    Step step_core(PolicyInput in, const PolicyParams& p);   // the plan / decide / trap logic; step() adds the F1 belief
    ReachabilityBelief belief_;
    // F1: the committed Drive's claim — from pose `at` (remaining time T0 there) it shortens the goal's remaining time by D;
    //   judged against its own rollout when its plan ends (navigator.cpp step())
    struct Claim { bool pending = false; Pose2 at{}; float T0 = 0.f; double t0 = 0.0;
                   float v0 = 0.f, w0 = 0.f, lat = 0.f, v1 = 0.f, w1 = 0.f, v2 = 0.f, w2 = 0.f; bool has2 = false;
                   float hold = 1.f, dt = 0.05f; BaseLimits base{}; float tau = 0.f; };
    double hold_since_ = -1.0, last_now_ = 0.0;   // F1 round 3: when the current hold began (-1: none)
    Claim claim_;
    Pose2 predicted_pose(const Claim& c, float el) const;
    static float no_shortening(float D, float sig, float T);
    float route_risk_weight_ = 1.f;
    std::optional<float> goal_heading(const Pose2& at) const;
    float remaining_time(const Pose2& at, const BaseLimits& b) const;
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
    // the remaining sweep of the current phase (and phase 2 if still to come) from (start, v0, w0); e = time in the phase
    std::vector<Pose2> remaining_sweep(const Pose2& start, float v0, float w0, float e, const BaseLimits& b, const PolicyParams& p) const;
    // has the current phase reached its predicted end state? (early completion, round 5)
    bool phase_outcome_reached(const PolicyInput& in) const;
    double last_build_ms_ = 0.0;
    int failed_predictions_ = 0;
    std::function<bool(const Eigen::Vector2f&)> pending_free_;
    std::function<float(const Eigen::Vector2f&)> pending_extra_;
    bool world_pending_ = false;
    std::function<bool(const Eigen::Vector2f&, float)> trap_veto_;
    void rebuild_on_new_world();   // build() + re-baseline the monitor and the plan phase (update_world)
};
}  // namespace rc::policy
