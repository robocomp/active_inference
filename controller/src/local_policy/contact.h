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

// ★ LOCAL SURFACE DENOISING (review fix round 1; replaces the bearing cells of Task-6 round 2, which discarded observed
//   wall — within ONE scan each beam returns one range, so nothing in a bearing bin occludes anything: the "nearest
//   surface only" rule left a 0.58 m gap inside the look-ahead on a parallel noisy wall — and assumed the sensor at the
//   robot centre, while Shadow's helios sits at y -0.155, its bpearl at +0.140 and live returns come from a buffer of
//   poses). Each LiDAR return (aggregate == true) stays ITS OWN obstacle — no grouping, no occlusion rule:
//   · its neighbours within kDenoiseRadiusM (hash grid of that side) are fitted with a line (PCA);
//   · the return moves toward that line ALONG THE LINE'S NORMAL ONLY — an endpoint stays an endpoint (a jamb tip is
//     never pulled inward along the surface). Across the line the raw hit (variance sigma_r^2) is FUSED with the line's
//     prediction at that point, whose variance is the regression's plus the misfit:
//       v_line = max(s^2, sigma_r^2) (1/n + a^2 / (n lambda_t)) + max(0, s^2 - sigma_r^2)
//     (s^2 = residual variance about the line, unbiased; a = the return's offset along the line from the neighbours'
//     centroid; lambda_t = their variance along it; sigma_r^2 = the hits' own mean variance):
//       moved by v_r / (v_r + v_line) of its offset, sigma_perp^2 = v_r v_line / (v_r + v_line) <= min(v_r, v_line).
//     A clean wall pulls a noise-tail hit onto it; at a corner or a jamb tip the misfit leaves the hit near its place;
//   · ALONG the line it keeps its raw position. Inside a surface (neighbours beyond it on both sides) that position's
//     error does not matter — the surface is there either way — so the obstacle is isotropic at sigma_perp. At an END
//     of the observed surface (no neighbour beyond it on one side) the along-line error moves where the surface stops:
//     (sxx, sxy, syy) carry the extra variance (sigma_r^2 - sigma_perp^2) t t' and the contact model projects it on the
//     clearance direction;
//   · fewer than kDenoiseMinNeighbours neighbours (itself included): the raw return, sigma_r, isotropic.
//   kDenoiseRadiusM = 0.15 m: the neighbourhood must hold a surface's WHOLE noise band around any of its hits, or the
//   fit inherits the truncation bias the grid cells had — a noise-tail hit sits up to ~3 sigma_r (6 cm at sigma_r 0.02)
//   off its surface and the band is +-3 sigma_r wide, so the disc must reach ~2 x 3 sigma_r across plus a few sigma_r
//   along for the PCA axis to be the surface's (measured at 0.05 m: a tail hit 2.9 cm inside a wall, 6 sigma off; at
//   0.15 m: see test_policy_contact); and it stays below the ~0.2 m straight edges of the hull, so a corner or a jamb
//   tip inside the disc shows up as misfit (s^2 > sigma_r^2) — a WIDER sigma and a weaker pull, never a tighter one.
//   kDenoiseMinNeighbours = 4: a line has 2 parameters; with fewer than 2 residual degrees of freedom its residual says
//   nothing. Support segments are GONE: each return is its own obstacle, so a surface's extent is its returns.
//   Returns with aggregate == false (unseen-contact hypotheses) are passed through untouched.
constexpr float kDenoiseRadiusM = 0.15f;
constexpr int kDenoiseMinNeighbours = 4;
std::vector<Return> denoise_returns(std::span<const Return> r, float radius_m = kDenoiseRadiusM);

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
