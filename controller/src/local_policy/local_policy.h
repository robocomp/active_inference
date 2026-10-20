// local_policy.h — the look-ahead decision: which (v, w) to send this cycle, or why none.
// Eigen + std only (no Qt/DSR). Room frame; yaw = FORWARD heading (policy_types.h).
#pragma once
#include <optional>
#include <span>
#include <string>
#include "policy_types.h"
#include "contact.h"
#include "cost_to_go.h"

namespace rc::policy
{
// The attractor the task layer hands the policy.
//   Place:   reach p (within tol_xy); if `yaw` is set, also face it (within tol_yaw) once near.
//   Doorway: the CALLER builds the field with goal points past the gap; decide() treats it like Place (field
//            decrease) and calls it arrived when the field at the pose is within tol_xy of zero.
//   Heading: face `yaw` (or, if unset, face the point p). No field is needed.
struct Goal { enum class Kind { Place, Doorway, Heading } kind; Eigen::Vector2f p{0,0};
              std::optional<float> yaw; Eigen::Vector2f normal{0,1}; float width = 0.f;
              float tol_xy = 0.25f, tol_yaw = 0.06f; };

// ★ The policy's two stated risk levels (chance constraints), nothing else gates a motion:
//   risk_eps — acceptable P(contact) per DRIVE decision (normal motion toward the goal);
//   release_risk_max — a RELEASE (back up / turn a bit out of a wedge) is attempted only if it is more likely
//   contact-free than not (P(contact) < 0.5): the robot never commits to a motion that will probably bump; when no
//   release clears that bar the base is Stuck and the task layer is told (review ruling: the former bar 1 - risk_eps
//   let a release run at P = 0.978 — a near-certain bump).
//   progress_scale_m is not a gate: it is the distance of progress worth one nat of safety/continuability over one
//   horizon; decide() scores progress as a RATE (round 5) with the scale progress_scale_m / horizon_s (m/s per nat).
struct PolicyParams { float horizon_s = 1.0f, dt = 0.05f; int n_v = 7, n_w = 11, n_cont_v = 3, n_cont_w = 5;
                      float risk_eps = 0.01f, release_risk_max = 0.5f, progress_scale_m = 0.10f, returns_max_age_s = 0.5f; };

// ★ ROUND 8 — pose uncertainty split by WHAT IT AFFECTS:
//   sigma_rel_xy / sigma_rel_yaw: the RELATIVE body-vs-returns error accumulated over a rollout (odometry / motion noise
//     over the horizon + latency). Returns are re-expressed on the believed pose, so absolute localisation error moves
//     body and returns together and cancels: only this enters the CONTACT model.
//   pose_drift_sigma_m / pose_drift_time_s: the absolute pose error as an Ornstein-Uhlenbeck process (stationary std,
//     correlation time). The ProgressMonitor needs the std of its CHANGE over a prediction's elapsed time tau,
//     sigma_drift(tau) = sigma * sqrt(2 (1 - exp(-tau / T))); it computes that itself from these two numbers (0 = no
//     drift). A random-walk localiser with rate q m/sqrt(s) is the limit T -> inf, sigma = q sqrt(T/2).
struct PolicyInput { Pose2 pose; float v_now = 0, w_now = 0, sigma_rel_xy = 0, sigma_rel_yaw = 0, returns_age_s = 0;
                     float pose_drift_sigma_m = 0, pose_drift_time_s = 0;
                     std::span<const Return> returns; const rc::RobotFootprint* body = nullptr; BaseLimits base;
                     const Goal* goal = nullptr; const CostToGo* field = nullptr;
                     // residual error of the proto-derived body (tessellation, hull simplification, beam footprint)
                     float body_geometry_sigma_m = 0.01f;
                     // pose age + command latency: the base keeps (v_now, w_now) for this long before the decided
                     // command takes effect; candidates are rolled out from the pose reached then (policy_start)
                     float latency_s = 0.f;
                     // decision time (s, any monotonic clock): the ProgressMonitor judges a plan when its own
                     // duration has elapsed (round 6)
                     double now_s = 0.0;
                     // round 8: the TRAP-FREE cost-to-go (the same goal, no trap costs) on which the ProgressMonitor
                     // measures predicted and realized descent; nullptr = use `field`. Candidates are chosen on `field`.
                     const CostToGo* progress_field = nullptr;
                     // review fix round 3: `returns` are ALREADY denoise_returns() output (the Navigator denoises once
                     // per tick for its plan check and decide() both)
                     bool returns_denoised = false; };

// log_p_safe / log_p_cont / progress_m describe the CHOSEN candidate (zeros when none is chosen); progress_m is its
// TWO-STEP progress (round 4: the better of its end and its best eligible continuation, measured from the start).
// eligible = number of candidates whose P(contact) <= risk_eps (Drive draws from them; Release from ALL v <= 0 ones).
// Stuck only when returns lie inside the bare hull, or no release candidate has P(contact) < release_risk_max.
// Blind also for a non-finite input (NaN/inf anywhere): no prediction can be made.
struct Decision { enum class Mode { Drive, Release, Arrived, Stuck, Blind } mode; float v = 0, w = 0;
                  float log_p_safe = 0, log_p_cont = 0, progress_m = 0; int eligible = 0; std::string why;
                  // round 7: this Release was caused by the ProgressMonitor (a failed prediction) — the only evidence
                  // that "the body cannot get through from here"; a caller lays a trap on it and on nothing else
                  bool prediction_failed = false;
                  // review fix round 3 — the PLAN this decision commits to (Navigator executes it): the candidate (v, w)
                  // held for horizon_s then braked, plan_dur1 in all; for a Drive whose rate came from a continuation,
                  // that continuation from rest (plan_v2, plan_w2) held for horizon_s then braked, plan_dur2 in all
                  float plan_dur1 = 0.f, plan_v2 = 0.f, plan_w2 = 0.f, plan_dur2 = 0.f; bool plan_has_cont = false; };

// ★ ROUND 6 — STUCK IS A FAILED PREDICTION, judged on what happened. Each Drive decision predicts a two-step field
//   descent dF_pred over its own duration tau_pred (the chosen candidate's duration + the continuation that set its
//   rate). The monitor holds ONE pending prediction (made by the first Drive after a reset): once tau_pred has elapsed
//   it compares the REALIZED descent f(pose then) - f(pose now) with the instrument's FULL resolution (round 7)
//   res = sqrt(res_field^2 + (|grad F| sigma_drift(tau))^2): res_field = CostToGo::resolution_m over the realized displacement,
//   |grad F| = the field's slope at the pose (central differences over one cell; 1, the distance-field slope, where a
//   neighbour is not finite), sigma_drift(tau) from PolicyInput::pose_drift_* (round 8). Round 8: predicted AND realized
//   descents are measured on PolicyInput::progress_field (trap-free) — a trap's own cost is not a failure to progress: realized <= resolution while dF_pred > resolution -> FAILED, and the next decide()
//   is a Release whatever the candidates promise; realized > resolution -> success, the record resets. A prediction
//   no larger than the resolution says nothing and is dropped. No timer or constant beyond the plan's own tau and the
//   field's resolution.
//   API: the caller owns one ProgressMonitor per goal and passes it to every decide(); it calls reset() on a goal change
//   or a rebuild of the TRAP-FREE progress_field (the only field the monitor reads; laying a trap does not touch it).
//   Any Release clears the pending prediction (review M1). Navigator (navigator.h) does all of this. Heading goals (no field) are not
//   monitored. decide() without a monitor behaves as before round 6.
// ★ REVIEW FIX ROUND 4 — EVERY COMPLETED PLAN IS JUDGED. A Drive plan predicts a field descent (above); a RELEASE plan
//   predicts a CONTINUABILITY gain, log_p_cont(release end) - log_p_cont(start), at decision time. When it completes
//   (tau = its hold, horizon_s) the realized gain is log_p_continuable of the actual pose (same coarse set) minus the
//   start's: realized <= 0, or a predicted gain <= 0 (a release that promised nothing), is a FAILED prediction, and the
//   next decision is a Release with prediction_failed (the Navigator lays a trap). A Drive plan's tau is its HOLD
//   (horizon_s, x2 for a two-step plan): the brake is a safety tail, not part of the plan (round 4).
struct ProgressMonitor
{
    bool pending = false, release = false;
    double t0 = 0.0; float tau = 0.f, dF_pred = 0.f, f0 = 0.f, lpc0 = 0.f, gain_pred = 0.f; Pose2 p0{};
    void reset() { *this = ProgressMonitor{}; }
};

Decision decide(const PolicyInput& in, const PolicyParams& p, ProgressMonitor* monitor = nullptr);

// Exposed for tests/diagnostics: log((k+1)/(N+2)) of the coarse continuation set from `from` (a pose at rest).
float log_p_continuable(const PolicyInput& in, const PolicyParams& p, const Pose2& from);
// The pose candidates are rolled out from: `pose` advanced by (v_now, w_now) for latency_s.
Pose2 policy_start(const PolicyInput& in);
// Inside the goal's tolerances at in.pose (what decide() calls Arrived).
bool goal_reached(const PolicyInput& in);
const char* mode_name(Decision::Mode m);
}  // namespace rc::policy
