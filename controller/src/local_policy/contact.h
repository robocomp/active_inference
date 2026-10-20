// contact.h — rollout of a (v, w) command and the swept-body probability of touching nothing.
#pragma once
#include <span>
#include <vector>
#include "policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::policy
{
// Hold (v, w) for hold_s, then brake both axes independently to rest at d_lin / d_rot. Poses every dt (start pose
// first, final rest pose last). Differential drive, exact arc integration (straight-line limit as |w| -> 0).
std::vector<Pose2> rollout(const Pose2& start, float v, float w, float hold_s, const BaseLimits& b, float dt);

// Signed distance (m) from a room-frame point to the body at `pose`, margin included: negative inside the
// margin-grown polygon. Same quarter-turn convention as RobotFootprint::contains_yaw (tested against it).
// (Geometry/diagnostics only — the contact probability below does NOT use the grown polygon.)
float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q);
// Same against the BARE hull (margin 0): negative = the point is inside the robot itself.
float hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q);

// ★ SURFACE CELLS (controller ruling, fix round 4). LiDAR returns are MEASUREMENTS of surfaces, not independent
//   obstacles: n hits of one wall pin it down better, not worse. Returns are binned into square room-frame cells of
//   side kSurfaceCellM and each cell becomes ONE obstacle: position = precision-weighted mean of its returns
//   (var = 1 / sum 1/sigma_i^2, i.e. sigma_r^2 / n for equal sigma_r — a single isolated hit keeps the full
//   sigma_r), existence = 1 - prod (1 - p_exist_i). Idempotent: a cell's mean lies in its own cell.
//   The side, 0.04 m, is the along-surface scale over which a surface is ONE element: a wall, a door leaf, a table
//   edge or a chair leg (~0.03-0.05 m) does not change its distance to the body within it, and the sensor itself
//   blurs any structure finer than ~2-4 sigma_r (0.01 m range noise). It must never merge returns across a gap the
//   body could pass: the narrowest opening the 0.46 x 0.57 m hull can thread is ~0.5 m, ten cells, so no passable gap
//   is ever hidden. It is a discretisation (like dt), not a threshold: nothing is gated on it.
constexpr float kSurfaceCellM = 0.04f;
std::vector<Return> surface_cells(std::span<const Return> r, float cell_m = kSurfaceCellM);

// One obstacle's clearances along a sweep: signed distance to the bare hull at the start (sd0) and the minimum over
// the sweep (sd_min <= sd0), its independent sigma and its existence. Exposed for the integral-accuracy test.
struct ContactTerm { double sd0, sd_min, sigma, p_exist; };
// log E_eps[prod_j F_j(sd_min_j + eps)] - log E_eps[prod_j F_j(sd0_j + eps)], eps ~ N(0, sigma_s^2),
// F_j(x) = 1 - p_exist_j * (1 - Phi(x / sigma_j)). n_ref = 0: the adaptive rule log_p_no_contact uses (accurate to
// < 0.005 in P for sigma_j / sigma_s in [0.05, 5], tested); n_ref > 0: brute-force trapezoid with n_ref nodes over
// +-6 sigma_s (the test's reference). Returns <= 0, or -inf when the denominator carries no mass.
double log_ratio_shared_offset(std::span<const ContactTerm> t, double sigma_s, int n_ref = 0);

// Log-probability that the swept body touches none of the returns, CONDITIONED ON TOUCHING NONE AT THE START POSE.
// Returns are first merged into surface cells (above). Distances to the BARE hull (the safety margin is the planner's
// route standoff, not part of this probability). Error model: a SHARED clearance offset
// eps ~ N(0, sigma_pose_xy^2 + body_geometry_sigma_m^2) common to every cell, plus an independent per-cell
// sigma_j^2 = sigma_cell^2 + (rho_j*sigma_pose_yaw)^2 (rho_j = range from the pose centre):
//   P = E_eps[prod_j F_j(sd_min_j + eps)] / E_eps[prod_j F_j(sd0_j + eps)],  F_j(x) = 1 - p_exist*(1 - Phi(x/sigma_j)).
// body_geometry_sigma_m: residual error of the proto-derived body (tessellation, hull simplification, LiDAR beam
// footprint) — default 0.01 m at the call sites.
float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m);
// Same, for `cells` ALREADY merged by surface_cells() (decide() merges once per cycle, not once per candidate).
float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m);
}  // namespace rc::policy
