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

// ★ LOCAL SURFACE MODEL (review fix rounds 1-2). LiDAR returns are MEASUREMENTS of surfaces. Each one's neighbours
//   within kDenoiseRadiusM (hash grid) are fitted with a line (PCA), and the returns become:
//   · SURFACE SAMPLES — segments ON the fitted line, one per span of ~c (greedy: a return not yet within c/2 of a sample
//     starts one over [a - c/2, a + c/2] along its line, clipped to where the data are). Each carries the line's sigma
//     at the farther end of its span:  v_line(a) = max(s^2, sigma_r^2) (1/n + a^2 / (n lambda_t)) + max(0, s^2 - sigma_r^2)
//     (s^2 = residual variance about the line, unbiased; a = along-line offset from the neighbours' centroid; lambda_t
//     = their variance along it; sigma_r^2 = the hits' own mean variance), never wider than one raw hit.
//     ★ WHY SAMPLES AT SPACING c (review fix round 2): the returns of one neighbourhood share the fitted line's error —
//       they are NOT independent obstacles (one obstacle per return made a denser wall read riskier: arc 0.151 at 1 cm
//       vs 0.125 at 4 cm, and cost 55-343 ms per decision at ~4900 returns). c is the fit's correlation length: two
//       samples one c apart share no hit. This is the RESOLUTION of the surface model, not the occlusion grouping
//       removed in round 1: nothing observed is discarded — every return lies within c/2 of a sample's span, and the
//       samples are segments, so the surface between them is represented (points at spacing c would let a hull corner
//       pass between them).
//   · the surfaces' ENDS — an extreme hit of a span with no neighbour beyond it along the line (a jamb tip, a leaf's
//     edge) is ALWAYS kept as its own point obstacle: fused with the line along the normal only (moved by
//     v_r / (v_r + v_line) of its offset, sigma_perp^2 = v_r v_line / (v_r + v_line)), its raw sigma kept ALONG the line
//     ((sxx, sxy, syy) = (sigma_r^2 - sigma_perp^2) t t', projected by the contact model on the clearance direction);
//   · isolated hits (fewer than kDenoiseMinNeighbours neighbours, itself included): the raw return, sigma_r, isotropic.
//   Returns with aggregate == false (unseen-contact hypotheses) are passed through untouched.
//   kDenoiseRadiusM = 0.15 m: the neighbourhood must hold a surface's WHOLE noise band around any of its hits (+-3
//   sigma_r), or the fit inherits the truncation bias the grid cells had (at 0.05 m a tail hit stayed 2.9 cm, 6 sigma,
//   inside a wall); it stays below the ~0.2 m straight edges of the hull, so a corner inside the disc shows up as
//   misfit (s^2 > sigma_r^2): a WIDER sigma, never a tighter one.
//   kDenoiseMinNeighbours = 4: a line has 2 parameters; with fewer than 2 residual degrees of freedom its residual says
//   nothing.
constexpr float kDenoiseRadiusM = 0.15f;
constexpr int kDenoiseMinNeighbours = 4;
std::vector<Return> denoise_returns(std::span<const Return> r, float radius_m = kDenoiseRadiusM);
// Signed distance from an obstacle (a point, or a surface sample's segment p +- seg) to the BARE hull at `pose`.
float obstacle_hull_distance(const rc::RobotFootprint& body, const Pose2& pose, const Return& q);

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
// Returns are first denoised (above). Distances to the BARE hull (the safety margin is the planner's
// route standoff, not part of this probability). Error model: a SHARED clearance offset
// eps ~ N(0, sigma_pose_xy^2 + body_geometry_sigma_m^2) common to every cell, truncated to eps > -sd0* by "nothing
// touches now" (sd0* = the closest cell now), plus an independent per-cell
// sigma_j^2 = sigma^2 + (rho_j*sigma_pose_yaw)^2 + n' S n (S = the obstacle's anisotropic part, n = the clearance
// direction; rho_j = range from the pose centre):
//   P = E_{eps > -sd0*}[prod_j F_j(sd_min_j + eps) / F_j(sd0_j + eps)],  F_j(x) = 1 - p_exist*(1 - Phi(x/sigma_j)).
// body_geometry_sigma_m: residual error of the proto-derived body (tessellation, hull simplification, LiDAR beam
// footprint) — default 0.01 m at the call sites.
// lps_floor: the sweep may stop early and return an UPPER BOUND (<= lps_floor) once the result is provably below
// exp(lps_floor) — P(no contact) <= F_j(sd_min_j + 6 sigma_s) / F_j(sd0_j + 6 sigma_s) for any deepened cell with
// p_exist = 1 (its ratio rises with eps); -inf (default) keeps it exact.
float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m,
                       float lps_floor = -std::numeric_limits<float>::infinity());
// Same, for `cells` ALREADY denoised by denoise_returns() (decide() denoises once per cycle, not once per candidate).
float log_p_no_contact_cells(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> cells,
                             float sigma_pose_xy, float sigma_pose_yaw, float body_geometry_sigma_m,
                             float lps_floor = -std::numeric_limits<float>::infinity());
}  // namespace rc::policy
