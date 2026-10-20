// contact.h — rollout of a (v, w) command and the swept-body probability of touching nothing.
#pragma once
#include <algorithm>
#include <limits>
#include <span>
#include <vector>
#include "policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::policy
{
// ★ Rollouts start from the CURRENT velocity (review ruling). Each axis ramps from v0 / w0 to the command at a_lin /
//   a_rot when its speed grows and at d_lin / d_rot when it shrinks (through zero on a sign change), the command is
//   applied for hold_s (the ramp lies inside it: a command at the edge of the dynamic window is reached exactly at
//   its end), then both axes brake to rest at d_lin / d_rot. Poses every dt (start pose first, final rest pose
//   last). Differential drive, exact arc integration (straight-line limit as |w| -> 0). A rate <= 0 is instantaneous.
std::vector<Pose2> rollout(const Pose2& start, float v0, float w0, float v, float w, float hold_s, const BaseLimits& b, float dt);
// Where a body moving at constant (v, w) is after dt (exact arc): the latency advance of the start pose.
Pose2 advance(const Pose2& start, float v, float w, float dt);

// Signed distance (m) from a room-frame point to the body at `pose`, margin included: negative inside the
// margin-grown polygon. Same quarter-turn convention as RobotFootprint::contains_yaw (tested against it).
// (Geometry/diagnostics only — the contact probability below does NOT use the grown polygon.)
float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q);
// Same against the BARE hull (margin 0): negative = the point is inside the robot itself.
float hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q);

// ★ SURFACE CELLS — BEARING aggregation (controller ruling D1, Task 6; replaces the square-grid cells of fix round 4).
//   LiDAR returns are MEASUREMENTS of surfaces: n hits of one surface pin it down better, not worse. The range noise
//   lies ALONG THE RAY, so the grid cells of fix round 4 cut each surface's noise band across 2-3 bins whenever sigma_r
//   is comparable to the cell (measured in the Task-6 closed loop at sigma_r 0.02: single-hit "tail" cells 4-5 cm
//   inside a wall, treated as independent obstacles — tour3_1_leg3 Stuck at t = 0, P(contact) 0.04-0.10 for straight
//   driving 8 cm from a wall). Now:
//   · BINS are bearings about the SENSOR ORIGIN `origin` (the caller passes where the scan was taken: decide() uses the
//     robot centre at the decision pose, log_p_no_contact the sweep's first pose). Their width is bin_rad =
//     kSurfaceCellM / circumscribed radius (bearing_bin_rad(): 0.04 / 0.318 = 0.126 rad = 7.2 deg for Shadow), i.e. a
//     bin's arc at the body's own reach is the 0.04 m surface element. One surface's hits on one bearing average
//     together however the bin is aligned.
//   · Within a bin only the NEAREST surface can be touched first: hits are sorted by range and grouped, a new group
//     starting when a hit lies more than kSplitSigmas sigma beyond the current group's running mean (sigma of that
//     difference: sqrt(sigma_hit^2 + sigma_mean^2)). Groups are kept from the nearest outward up to and including the
//     first CERTAIN one (combined existence 1): a surface behind a certain one is hidden by it, behind an uncertain one
//     (a ghost, an unseen-contact hypothesis) it still matters.
//   · Each kept group is ONE obstacle: position = precision-weighted mean (var = 1 / sum 1/sigma_i^2 = sigma_r^2 / n),
//     existence = 1 - prod (1 - p_exist_i), and its SCATTER with the known range noise removed: the weighted covariance
//     of the hits (+ their own carried scatter) minus the expected noise sigma_bar^2 (n-1)/n along the mean ray,
//     clamped positive semi-definite. What remains is surface EXTENT: along the surface for a face seen square-on (so
//     dense sampling still pins the wall down), along the ray for a jamb side seen end-on. The existing model then
//     sweeps it as its support segment (half-extent sqrt(3 lambda_1) on the principal axis) and adds the residual
//     lambda_2 projected on the clearance direction to sigma_j^2 (see fix round 2 below).
//     ★ DEVIATION from the ruling's letter ("its tangential scatter"): the tangential part alone drops the extent along
//       the ray, which is exactly the jamb side seen end-on (the support would stop at the group's mean range, inside
//       the jamb's near end). Removing the noise instead keeps both: equal to the tangential scatter for a face seen
//       square-on, the full extent otherwise.
//   kSplitSigmas = 4 is the only grouping constant: a STATED statistical level, two-sided P ~ 6e-5 of wrongly splitting
//   one surface's hits (and a split costs only a duplicate obstacle, never a hidden one: the nearer group is kept).
//   Fix-round-2 note kept for the support model: segment + residual projection keeps the jamb test and does not
//   double-count a flat wall seen obliquely by a corner (ACCEPTED by the controller on that variant table).
constexpr float kSurfaceCellM = 0.04f;
constexpr float kSplitSigmas = 4.f;
inline float bearing_bin_rad(const rc::RobotFootprint& b) { return kSurfaceCellM / std::max(b.circumscribed_radius() - b.safety_margin(), 1e-3f); }   // BARE reach
std::vector<Return> surface_cells(std::span<const Return> r, const Eigen::Vector2f& origin, float bin_rad);

// One obstacle's clearances along a sweep: signed distance to the bare hull at the start (sd0) and the minimum over
// the sweep (sd_min <= sd0), its independent sigma and its existence. Exposed for the integral-accuracy test.
struct ContactTerm { double sd0, sd_min, sigma, p_exist; };
// log E_{eps | eps > -sd0_star}[ prod_j F_j(sd_min_j + eps) / F_j(sd0_j + eps) ], eps ~ N(0, sigma_s^2) truncated by
// the ONE fact "nothing touches now" (sd0_star = the closest cell's clearance now, over ALL cells, not only those
// in `t`), F_j(x) = 1 - p_exist_j * (1 - Phi(x / sigma_j)); every factor is <= 1 and a cell never brought closer
// contributes exactly 1. n_ref = 0: the adaptive rule log_p_no_contact uses (accurate to < 0.005 in P for
// sigma_j / sigma_s in [0.05, 5], tested); n_ref > 0: brute-force trapezoid with n_ref nodes over [-sd0_star, +6
// sigma_s] (the test's reference). Returns <= 0, or -inf when the truncation leaves no offset in range.
// ★ The 0.005 accuracy is ABSOLUTE on the truncated average (the integrand is in [0, 1]) — there is no longer a
//   denominator to divide by; the truncation mass Phi(6) - Phi(u_lo) is exact.
double log_ratio_shared_offset(std::span<const ContactTerm> t, double sigma_s, double sd0_star, int n_ref = 0);

// Log-probability that the swept body touches none of the returns, CONDITIONED ON TOUCHING NONE AT THE START POSE.
// Returns are first merged into surface cells (above). Distances to the BARE hull (the safety margin is the planner's
// route standoff, not part of this probability). Error model: a SHARED clearance offset
// eps ~ N(0, sigma_pose_xy^2 + body_geometry_sigma_m^2) common to every cell, truncated to eps > -sd0* by "nothing
// touches now" (sd0* = the closest cell now), plus an independent per-cell
// sigma_j^2 = sigma_cell^2 + (rho_j*sigma_pose_yaw)^2 + residual scatter (rho_j = range from the pose centre):
//   P = E_{eps > -sd0*}[prod_j F_j(sd_min_j + eps) / F_j(sd0_j + eps)],  F_j(x) = 1 - p_exist*(1 - Phi(x/sigma_j)).
// body_geometry_sigma_m: residual error of the proto-derived body (tessellation, hull simplification, LiDAR beam
// footprint) — default 0.01 m at the call sites.
// lps_floor: the sweep may stop early and return an UPPER BOUND (<= lps_floor) once the result is provably below
// exp(lps_floor) — P(no contact) <= F_j(sd_min_j + 6 sigma_s) / F_j(sd0_j + 6 sigma_s) for any deepened cell with
// p_exist = 1 (its ratio rises with eps); -inf (default) keeps it exact.
float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m,
                       float lps_floor = -std::numeric_limits<float>::infinity());
// Same, for `cells` ALREADY merged by surface_cells() (decide() merges once per cycle, not once per candidate).
float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m,
                             float lps_floor = -std::numeric_limits<float>::infinity());
}  // namespace rc::policy
