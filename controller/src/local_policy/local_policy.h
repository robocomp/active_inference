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

struct PolicyInput { Pose2 pose; float v_now = 0, w_now = 0, sigma_xy = 0, sigma_yaw = 0, returns_age_s = 0;
                     std::span<const Return> returns; const rc::RobotFootprint* body = nullptr; BaseLimits base;
                     const Goal* goal = nullptr; const CostToGo* field = nullptr;
                     // residual error of the proto-derived body (tessellation, hull simplification, beam footprint)
                     float body_geometry_sigma_m = 0.01f;
                     // pose age + command latency: the base keeps (v_now, w_now) for this long before the decided
                     // command takes effect; candidates are rolled out from the pose reached then (policy_start)
                     float latency_s = 0.f;
                     // decision time (s, any monotonic clock): the ProgressMonitor judges a plan when its own
                     // duration has elapsed (round 6)
                     double now_s = 0.0; };

// log_p_safe / log_p_cont / progress_m describe the CHOSEN candidate (zeros when none is chosen); progress_m is its
// TWO-STEP progress (round 4: the better of its end and its best eligible continuation, measured from the start).
// eligible = number of candidates whose P(contact) <= risk_eps (Drive draws from them; Release from ALL v <= 0 ones).
// Stuck only when returns lie inside the bare hull, or no release candidate has P(contact) < release_risk_max.
// Blind also for a non-finite input (NaN/inf anywhere): no prediction can be made.
struct Decision { enum class Mode { Drive, Release, Arrived, Stuck, Blind } mode; float v = 0, w = 0;
                  float log_p_safe = 0, log_p_cont = 0, progress_m = 0; int eligible = 0; std::string why;
                  // round 7: this Release was caused by the ProgressMonitor (a failed prediction) — the only evidence
                  // that "the body cannot get through from here"; a caller lays a trap on it and on nothing else
                  bool prediction_failed = false; };

// ★ ROUND 6 — STUCK IS A FAILED PREDICTION, judged on what happened. Each Drive decision predicts a two-step field
//   descent dF_pred over its own duration tau_pred (the chosen candidate's duration + the continuation that set its
//   rate). The monitor holds ONE pending prediction (made by the first Drive after a reset): once tau_pred has elapsed
//   it compares the REALIZED descent f(pose then) - f(pose now) with the instrument's FULL resolution (round 7)
//   res = sqrt(res_field^2 + (|grad F| sigma_xy)^2): res_field = CostToGo::resolution_m over the realized displacement,
//   |grad F| = the field's slope at the pose (central differences over one cell; 1, the distance-field slope, where a
//   neighbour is not finite), sigma_xy = PolicyInput::sigma_xy (the pose std): realized <= resolution while dF_pred > resolution -> FAILED, and the next decide()
//   is a Release whatever the candidates promise; realized > resolution -> success, the record resets. A prediction
//   no larger than the resolution says nothing and is dropped. No timer or constant beyond the plan's own tau and the
//   field's resolution.
//   API: the caller owns one ProgressMonitor per goal and passes it to every decide(); it calls reset() whenever the
//   field is rebuilt (a trap, a new goal: old field values are not comparable). Heading goals (no field) are not
//   monitored. decide() without a monitor behaves as before round 6.
struct ProgressMonitor
{
    bool pending = false, failed = false;
    double t0 = 0.0; float tau = 0.f, dF_pred = 0.f, f0 = 0.f; Pose2 p0{};
    void reset() { *this = ProgressMonitor{}; }
};

Decision decide(const PolicyInput& in, const PolicyParams& p, ProgressMonitor* monitor = nullptr);

// Exposed for tests/diagnostics: log((k+1)/(N+2)) of the coarse continuation set from `from` (a pose at rest).
float log_p_continuable(const PolicyInput& in, const PolicyParams& p, const Pose2& from);
// The pose candidates are rolled out from: `pose` advanced by (v_now, w_now) for latency_s.
Pose2 policy_start(const PolicyInput& in);
const char* mode_name(Decision::Mode m);
}  // namespace rc::policy
