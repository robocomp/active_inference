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

// ★ risk_eps is the ONE threshold (acceptable P(contact) per decision, a chance constraint). progress_scale_m is
//   not a gate: it is the distance of progress worth one nat of safety/continuability in the score.
struct PolicyParams { float horizon_s = 1.0f, dt = 0.05f; int n_v = 7, n_w = 11, n_cont_v = 3, n_cont_w = 5;
                      float risk_eps = 0.01f, progress_scale_m = 0.10f, returns_max_age_s = 0.5f; };

struct PolicyInput { Pose2 pose; float v_now = 0, w_now = 0, sigma_xy = 0, sigma_yaw = 0, returns_age_s = 0;
                     std::span<const Return> returns; const rc::RobotFootprint* body = nullptr; BaseLimits base;
                     const Goal* goal = nullptr; const CostToGo* field = nullptr;
                     // residual error of the proto-derived body (tessellation, hull simplification, beam footprint)
                     float body_geometry_sigma_m = 0.01f; };

// log_p_safe / log_p_cont / progress_m describe the CHOSEN candidate (zeros when none is chosen).
// eligible = number of candidates whose P(contact) <= risk_eps (Drive draws from them; Release from ALL v <= 0 ones).
// Stuck only when returns lie inside the bare hull, or the best release has P(contact) >= 1 - risk_eps.
struct Decision { enum class Mode { Drive, Release, Arrived, Stuck, Blind } mode; float v = 0, w = 0;
                  float log_p_safe = 0, log_p_cont = 0, progress_m = 0; int eligible = 0; std::string why; };

Decision decide(const PolicyInput& in, const PolicyParams& p);

// Exposed for tests/diagnostics: log((k+1)/(N+2)) of the coarse continuation set from `from` (a pose at rest).
float log_p_continuable(const PolicyInput& in, const PolicyParams& p, const Pose2& from);
const char* mode_name(Decision::Mode m);
}  // namespace rc::policy
