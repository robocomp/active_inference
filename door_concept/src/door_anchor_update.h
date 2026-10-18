/*
 * door_anchor_update.h — the door's measurement update of the INTERACTION ANCHOR, as a PURE function: the
 * opening's pose relative to the robot (predicted by odometry, common/interaction_anchor/relative_pose.h) is
 * corrected by the robot's OWN LiDAR rays and the camera's door-mask points, all in the robot frame. No room
 * frame is read here, so a room-frame door fit that is 0.45 m off (live: the robot drove into the wall beside
 * the doorway) cannot reach the crossing.
 *
 * ─── FRAMES ──────────────────────────────────────────────────────────────────────────────────────────
 * Robot frame: forward = +y, side = +x (ROBOT_GEOMETRY.md). The anchor is the OPENING: its origin is the
 * opening's centre on the wall's mid-plane, its x-axis runs ALONG the wall, its y-axis is the wall normal.
 * RelPose::x = (x, y, yaw) places it in the robot frame: p_robot = R(yaw) p_anchor + (x, y). When approaching,
 * the robot is on the anchor's −y side, so its rays travel toward +y in anchor coordinates. (The depth below
 * is signed along the ray's travel, so a robot already on the +y side is handled by the same model.)
 *
 * ─── THE GENERATIVE MODEL ─────────────────────────────────────────────────────────────────────────────
 * Each ray (origin o, endpoint e, projected to the floor plane) is judged where its LINE crosses the wall line:
 *   u = the crossing's abscissa along the wall,  d = the endpoint's signed depth past the mid-plane.
 * A ray crossing INSIDE the opening must end beyond the far face; OUTSIDE, it must end on the wall:
 *   L_through(d) = logistic((d − t/2)/σ) / R      (R = the scan's max range: a return beyond the plane is
 *                                                   equally likely anywhere out to it — makes L a DENSITY,
 *                                                   commensurable with L_wall)
 *   L_wall(d)    = N(d; −t/2, σ²)                 (a beam stops on the wall's NEAR face, the one facing it)
 *   L_leaf(d|δ)  = N(d; δ, σ²)                     (a shut leaf at depth δ ∈ [−t/2, t/2]: flush with either
 *                                                   face or between)
 *   p(ray | x, open)    = m(u) L_through(d)   + (1 − m(u)) L_wall(d),   m(u) = logistic((w/2 − |u|_τ)/τ)
 *   p(ray | x, shut, δ) = m(u) L_leaf(d | δ)  + (1 − m(u)) L_wall(d)   (the leaf closes the opening)
 *   every branch is then mixed with two CLUTTER components (ε = 0.05, the one flagged constant):
 *     p_branch(ray) = (1 − 2ε)·[the above] + ε·U(d; d_origin, −t/2) + ε/R
 *   — an OCCLUDER in front of the plane (uniform depth between the beam's origin and the near face: the leaf swung
 *   toward the robot, a person) and a BACKGROUND return anywhere in range (a second doorway in the same wall, the
 *   wall's end, glass). They bound every ray's pull: a far misfit costs at most −log ε, not a quadratic.
 *   p(rays | x)         = π Π p(ray | x, open) + (1 − π) E_δ[ Π p(ray | x, shut, δ) ],   π = 0.5, δ uniform
 * The open/shut latent AND the leaf depth δ are per FRAME (one leaf, one state, one depth for every ray) and are
 * marginalised, not decided. δ is a grid over [−t/2, t/2] at step ≤ σ/2, fine enough that the sum is the integral.
 * ★δ must be marginalised per frame, not per ray: spread per ray (N(d; 0, σ² + (t/2)²)) it charged each ray on a
 * leaf flush with the near face ~0.6 nats for the width, the opening was again pushed off the rays, and the
 * Hessian at the end was indefinite (test shut_door: the covariance fell back to the prior's).
 * ★Without the shut branch a shut door's wall returns inside the hypothesised opening could only mean "no
 * opening here", and the anchor ran off the end of the wall to where no ray landed (test shut_door: x = 4.49 m).
 * With it, a flat wall is explained equally at every lateral shift and says nothing along itself (a leaf
 * RECESSED from the near face still says a little: L_leaf and L_wall then differ, which is real evidence).
 * ★The wall term is centred on the NEAR face, not the mid-plane: centred at 0 it put the posterior's depth on
 * the face the robot sees, 5.5 cm (t/2) short of the opening's centre, on every frame.
 * σ = 0.05 m is the beam's range noise (the same value select_doorway_rays in door_phi_posterior.h uses);
 * τ = σ is the lateral blur of where a beam crosses the jamb line — a 5 cm range error at the grazing angles
 * that define a jamb is a 5 cm error along the wall. |u|_τ = sqrt(u² + τ²) is |u| with the same blur, so the
 * membership is smooth at the centre too. The mixture is the PROPER log-sum (not m·log L_t + (1−m)·log L_w):
 * a through-ray that falls just outside the hypothesised opening is explained only by m, so it PULLS the
 * opening toward it; a wall-ray just inside pushes it away. That is how the jambs constrain the anchor along
 * the wall, while the wall's rays constrain its depth and yaw — and a flat (shut) wall, whose rays look the
 * same at every lateral shift, constrains nothing along itself.
 *
 * Mask: the along-wall centroid ū of the camera's door-mask points (projected into anchor coordinates) is a
 * Student-t (ν = 3) observation of 0, the opening's centre, with σ_m = spread(u)/√min(N, 30) — the 30 caps the
 * √N gain because neighbouring mask pixels are correlated (one segmentation, one boundary error). A misplaced
 * mask loses its pull through the t's tail, not through a gate; camera_nis = ū²/σ_m² at the optimum.
 *
 * Free energy: F(x) = ½ (x − x̄)ᵀ P̄⁻¹ (x − x̄) − log p(rays | x) − log p(mask | x). NO RAY IS EVER REMOVED FROM F.
 *   - PREDICTIVE SPREAD. Each ray's depth σ and jamb blur τ are widened by the spread the current belief P
 *     predicts for that ray (J P Jᵀ, the Kalman S = HPHᵀ + R); the mask's σ likewise. Two stages: the prior's P,
 *     then the Laplace P stage 1 converged to; then a third at the sensor's own σ (Task 7, see the loop). This is what gives the update its reach: against the sensor's 5 cm
 *     alone, wall returns from a prior 0.3 m off in depth are outliers the occluder absorbs, and a 0.45 m lateral
 *     error is beyond the jamb's pull.
 *   - OPTIMISER. Per stage, at most 5 accepted Levenberg-Marquardt steps in the prior's metric, accepted on the
 *     gain ratio. Derivatives are central finite differences of F in double precision (F is smooth by
 *     construction); the curvature used is P̄⁻¹ + [I_L]₊, the data's information projected onto its PSD part.
 *   - LAPLACE + COMMON MODE. At the sensor's own σ: I_L from the exact finite-difference Hessian (a Gauss-
 *     Newton/Fisher one would count each jamb-adjacent wall ray as lateral information on a featureless shut wall),
 *     then the frame's common offset (σ_c = 0.02 m on x, y: one LiDAR extrinsic, one wall that is not a perfect
 *     plane) marginalised by Woodbury, I_eff = (I + I_L C)⁻¹ I_L — so ~900 correlated rays cannot claim a 2 mm σ.
 *
 * COST: ~60 ms on a full scan (469 returns), ~33 ms on the 224 within 2 m of the anchor (-O2; Task 7: the third
 * stage and the visibility pass roughly doubled the Task-3 29 / 15 ms; rays off the door's wall run are skipped). The caller
 * should pass the DOORWAY NEIGHBOURHOOD's rays (Task 4 preselects); rays far from the opening add cost, not
 * information.
 *
 * ─── THIS DOOR'S WALL ONLY (Task 7, fix 2) ────────────────────────────────────────────────────────────
 * The wall line is not infinite: the door's wall RUN spans [u_min, u_max] in anchor coordinates (the caller
 * passes it from the room polygon's wall; unknown ⇒ the aperture ± one width). A ray whose line crosses
 * OUTSIDE the run is explained by the background alone (1/R, independent of x): it carries no wall evidence.
 * Membership is soft at the run's ends (e(u) = σ((u − u_min)/τ)·σ((u_max − u)/τ), the same blur τ as the jambs):
 *   p_branch(ray) = e(u)·[the door/wall/clutter mixture above] + (1 − e(u))·(1/R).
 * ★try64: with the line infinite, a room corner 1.5 m along a SHUT door's wall made the first update slide the
 *  opening 0.9 m along the wall toward the corner (test turning_away_does_not_drag_anchor, k = 0: 0.904 m), and a
 *  gap beyond the run's end captured the opening 1.0 m away (other_wall_outside_extent_ignored).
 * Which rays belong to the run is decided ONCE per frame at the PRIOR mean (under the prior's spread), and a beam
 * travelling AWAY from the line is never on it (see Problem::run_log): a run re-evaluated at each hypothesis made
 * its ends lateral evidence and let the template slide to shed inconvenient rays.
 *
 * ─── VISIBILITY: P(door in view) TEMPERS the sensor evidence (Task 7, fix 1) ────────────────────────────
 * The sensors are evidence about THIS opening only on frames where it is actually observed. At the PRIOR mean,
 * with the prior's predictive spread:
 *   LiDAR   each passed ray's crossing membership of the doorway NEIGHBOURHOOD n(u) = σ((w − |u|_τ)/τ) (the opening
 *           and half a width of wall each side, i.e. the jambs and the wall that frames them) times the run
 *           membership e(u) — the beams the geometry PREDICTS on the doorway, v_lidar = Σ n·e — and the
 *           responsibility r of the doorway's own structure (wall / through / leaf) against the clutter components,
 *           which is where a return inconsistent with the predicted depth (an occluder, a wall elsewhere) loses its
 *           vote: o_lidar = Σ n·e·r (diagnostics). p_lidar = the mean COVERAGE of the doorway's cells — the fraction
 *           of the doorway region some beam entered (crossed, hit, or travelled along): see lidar_view. It falls to
 *           0 when no sampled beam reaches the doorway (out of the sector, or every beam occluded).
 *   Camera  the soft frustum membership of BOTH jambs (azimuth within the half-FOV, range within the depth range),
 *           blurred by the jambs' predictive spread: p_cam_geom. The mask's centroid is evidence of the opening's
 *           centre only if the whole opening is in the image.
 * The ray log-likelihood is multiplied by p_lidar and the mask's by p_cam_geom (a power likelihood: the
 * tempered form of "rays | door in view" mixed with "rays | door not in view", the latter independent of x).
 * p_view = 1 − (1 − p_lidar)(1 − p_cam), p_cam = p_cam_geom when a mask was consumed, else 0. When p_view → 0 the
 * update reduces to the prior, i.e. the odometry prediction, whose covariance grows with the motion.
 * (The neighbourhood's half a width of wall each side is a DEFINITION of "the doorway", soft at its edges.)
 *
 * ─── THE ONLY DECISION LEVELS (deliberate, not model terms) ──────────────────────────────────────────
 *   Both are COUNTS at the posterior mean, for the log and for p_open. Neither feeds back into the pose.
 *   1. SHORT rays — endpoint d < −(3σ + t/2): ended in front of the near face beyond range noise (the same level
 *      as select_doorway_rays). ★This used to DROP the ray from F at the current pose. With a prior 0.15 m+ beyond
 *      the true near face every wall return was "short", nothing pulled the depth back, and the anchor ran 1.1 m
 *      off in depth / 1.3 m along the wall (review, fix round 1). The occluder component does the job continuously.
 *   2. p_open — rays crossing inside the span (|u| < w/2) are THROUGH if d > 3σ + t/2, BLOCKED if |d| ≤ 3σ + t/2;
 *      p_open = (n_through + 1)/(n_through + n_blocked + 2), a Beta(1,1) posterior on "rays pass".
 *
 * The caller selects the rays (the wall band in z: above the floor, below the lintel); this file uses x, y.
 * Header-only, no DSR / Qt / OpenCV of its own (relative_pose.h brings rt_twist.h).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "interaction_anchor/relative_pose.h"

namespace rc::door
{

struct AnchorEvidence
{
    int n_through = 0, n_blocked = 0, n_short = 0;   // ray classes against the predicted opening
    float p_open = 0.5f;                              // (β·n_through+1)/(β·(n_through+n_blocked)+2), rays inside the span
    float camera_nis = -1.f;                          // mask centroid's normalised innovation; -1 = no mask
    // ─── Task 7: is the door being OBSERVED this frame (see "VISIBILITY" in the header) ─────────────────────
    float v_lidar = 0.f;                              // predicted beams on the doorway neighbourhood (Σ membership)
    float o_lidar = 0.f;                              // of those, explained by the doorway's own structure (Σ m·r)
    float p_lidar = 0.f;                              // mean coverage of the doorway's cells (lidar_view)
    float p_cam   = 0.f;                              // P(opening in the zed frustum) when a mask was consumed, else 0
    float p_view  = 0.f;                              // P(door in view) = 1 − (1 − p_lidar)(1 − p_cam)
};

// The door's own belief about its SWUNG leaf, in ANCHOR coordinates (fix round 3). The leaf is a segment of length w
// from the hinge (hinge_u, 0) along (dir_u·cos φ, swing_y·sin φ): at φ = 0 it closes the opening. `known` false = no
// swung-leaf component (an open door's leaf is then left to the clutter components, as before).
struct LeafPrior
{
    bool  known = false;
    float hinge_u = 0.f;     // ±w/2: the hinge jamb
    float dir_u = 1.f;       // along-wall direction from the hinge across the opening (−sign(hinge_u))
    float swing_y = -1.f;    // the side it opens toward (−1 = the robot's side at creation)
    float phi = 0.f;         // opening angle (rad), the door's estimate
    float phi_sigma = 0.f;   // its spread (rad); 0 = taken as exact
};

// The zed frustum in the ROBOT frame, for the camera's visibility term. `known` false = no frustum (the mask is then
// not tempered by geometry — the pre-Task-7 behaviour).
struct CameraFrustum
{
    bool known = false;
    Eigen::Vector2f origin{0.f, 0.f};     // optical centre, robot frame (x, y)
    Eigen::Vector2f forward{0.f, 1.f};    // viewing direction, robot frame, unit (zed: its y axis = DEPTH)
    float half_fov  = 0.f;                // horizontal half field of view (rad)
    float max_range = 0.f;                // depth range (m)
};

// ─── THE INITIAL BELIEF (from_room) ─────────────────────────────────────────────────────────────────────────
// The anchor is initialised once from the room frames, robot←room ∘ room←opening. Its covariance is the sum of
// THREE INDEPENDENT sources, all in the ANCHOR's axes (along the wall, across it, yaw) — independent, so they add:
// no max, no floor.
//   belief      the door belief's own variance of the opening (door_concept's AI2 belief: the along-wall centre);
//   robot_pose  the robot←room localisation covariance, projected onto the anchor (robot_pose_cov_on_anchor_*);
//   σ_model     the room-frame-vs-sensor DISCREPANCY the anchor exists to absorb: a room-frame door fit that is
//               internally confident and still wrong (try63 leg 2: σ_s 0.04 m with the door 0.45 m off), the room
//               polygon line vs the wall's mid-plane, a wall direction from a short polygon edge. Neither of the
//               other two terms can see that error — it is a MODEL error of the room-frame chain, not noise in it.
// ★Without σ_model the room's confidence becomes the anchor's: a 0.04 m prior on a door 0.45 m off is an 11σ
//  error, the jamb rays are outliers the clutter components absorb, and the anchor never moves (test
//  room_frame_error_is_correctable, RED with the belief-only prior).
inline Eigen::Matrix3f anchor_prior_cov(const Eigen::Matrix3f& belief, const Eigen::Matrix3f& robot_pose,
                                        const Eigen::Vector3f& sigma_model)
{
    return belief + robot_pose + Eigen::Matrix3f(sigma_model.cwiseAbs2().asDiagonal());
}

// Anchor-axes covariance → RelPose coordinates (x, y of the anchor origin IN THE ROBOT FRAME, yaw).
inline Eigen::Matrix3f anchor_axes_to_robot(const Eigen::Matrix3f& P_anchor, float anchor_yaw)
{
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    R.topLeftCorner<2, 2>() << std::cos(anchor_yaw), -std::sin(anchor_yaw), std::sin(anchor_yaw), std::cos(anchor_yaw);
    return R * P_anchor * R.transpose();
}
inline Eigen::Matrix3f robot_to_anchor_axes(const Eigen::Matrix3f& P_robot, float anchor_yaw)
{
    return anchor_axes_to_robot(P_robot, -anchor_yaw);
}

// The localisation covariance projected onto the anchor, for the two ways the localisation edge can be anchored.
// `anchor` = the opening in the ROBOT frame (x, y, yaw); C = the edge's (x, y, yaw) pose covariance, in the
// edge's PARENT frame (dsr_rt_api.h: "the pose is written in the PARENT frame, so its covariance is too").
//
// Edge robot → room (parent = robot; this fleet's room_concept): the edge is the room's pose (t, θ) in the robot
// frame, and the anchor is p = R(θ)·p_room + t ⇒ ∂p/∂t = I, ∂p/∂θ = S·(p − t) = (−(p_y − t_y), p_x − t_x), ∂yaw/∂θ = 1.
inline Eigen::Matrix3f robot_pose_cov_on_anchor_from_room_in_robot(const Eigen::Matrix3f& C, const Eigen::Vector2f& room_in_robot,
                                                                   const Eigen::Vector3f& anchor)
{
    Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
    J(0, 2) = -(anchor.y() - room_in_robot.y());
    J(1, 2) =  (anchor.x() - room_in_robot.x());
    return robot_to_anchor_axes(J * C * J.transpose(), anchor.z());
}
// Edge room → robot (parent = room): the edge is the robot's pose (t_r, θ_r) in the room, and the anchor is
// p = R(θ_r)ᵀ(p_room − t_r) ⇒ ∂p/∂t_r = −R(θ_r)ᵀ, ∂p/∂θ_r = (p_y, −p_x), ∂yaw/∂θ_r = −1.
inline Eigen::Matrix3f robot_pose_cov_on_anchor_from_robot_in_room(const Eigen::Matrix3f& C, float robot_yaw_in_room,
                                                                   const Eigen::Vector3f& anchor)
{
    const float c = std::cos(robot_yaw_in_room), s = std::sin(robot_yaw_in_room);
    Eigen::Matrix3f J = Eigen::Matrix3f::Zero();
    J.topLeftCorner<2, 2>() << -c, -s, s, -c;   // −R(θ_r)ᵀ
    J(0, 2) = anchor.y();
    J(1, 2) = -anchor.x();
    J(2, 2) = -1.f;
    return robot_to_anchor_axes(J * C * J.transpose(), anchor.z());
}

namespace anchor_detail
{
constexpr double kRangeSigma  = 0.05;   // beam range noise (m), as in door_phi_posterior.h
constexpr double kMaxRange    = 8.0;    // the scan's max range (m), as in select_doorway_rays
constexpr double kMaskNu      = 3.0;    // Student-t dof of the mask centroid
constexpr int    kMaskNCap    = 30;     // correlated mask pixels: √N gain capped here
constexpr int    kMaxIter     = 5;
constexpr double kPriorOpen   = 0.5;    // P(door open) before this frame: uninformed (the interface carries no leaf belief)
// ⚑ FLAGGED CONSTANT — the clutter fraction: the share of beams explained by something that is not this doorway's
//   wall/opening/leaf. It weights BOTH extra components: an occluder in front of the plane (the leaf swung toward the
//   robot, a person) and a background return anywhere in range (another doorway in the same wall, the wall's end,
//   glass). No model term sets it from data yet; 0.05 is "one beam in twenty".
constexpr double kClutter     = 0.05;
// The per-frame COMMON MODE: every ray of one frame shares the LiDAR's extrinsic error and the wall's departure
// from a plane, so their evidence about the anchor's lateral/depth position is correlated at this σ (m).
constexpr double kCommonSigma = 0.02;

using V3d = Eigen::Vector3d;
using M3d = Eigen::Matrix3d;

inline double log_sigmoid(double z) { return z >= 0 ? -std::log1p(std::exp(-z)) : z - std::log1p(std::exp(z)); }
inline double log_add_exp(double a, double b)
{
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    const double m = std::max(a, b);
    return m + std::log1p(std::exp(-std::abs(a - b)));
}
inline double log_normal(double r, double var) { return -0.5 * r * r / var - 0.5 * std::log(2.0 * M_PI * var); }
// log(1 − e^a) for a ≤ 0, accurate at both ends.
inline double log1m_exp(double a)
{
    if (a >= 0.0) return -std::numeric_limits<double>::infinity();
    return a > -M_LN2 ? std::log(-std::expm1(a)) : std::log1p(-std::exp(a));
}

// A 2-D ray in ROBOT coordinates.
struct Ray2 { Eigen::Vector2d o, e; };

// The ray in anchor coordinates: along-wall crossing u, the endpoint's signed depth d and the ORIGIN's signed
// depth d_o (both along the ray's travel; d_o < 0 while the beam starts in front of the plane).
struct UD { double u, d, d_o; };
inline UD to_anchor(const Ray2& r, const V3d& x)
{
    const double c = std::cos(x.z()), s = std::sin(x.z());
    const Eigen::Vector2d t(x.x(), x.y());
    auto inv = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - t; return Eigen::Vector2d(c * q.x() + s * q.y(), -s * q.x() + c * q.y()); };
    const Eigen::Vector2d oa = inv(r.o), ea = inv(r.e);
    const double sgn = oa.y() <= 0 ? 1.0 : -1.0;   // which side the beam starts on: its travel is "past the plane"
    const double dy = ea.y() - oa.y();
    // Where the ray's LINE crosses y = 0. A ray parallel to the wall never crosses it: put its crossing far along
    // the wall (1e9 m, finite so differences stay finite), where m = 0 and only the depth terms speak.
    const double u = std::abs(dy) > 1e-12 ? oa.x() - oa.y() * (ea.x() - oa.x()) / dy : std::copysign(1e9, ea.x() - oa.x());
    return {u, sgn * ea.y(), sgn * oa.y()};
}

// The PREDICTIVE spread of one ray's (u, d) under the current pose belief P: J P Jᵀ, J by finite differences.
// The likelihood is evaluated against the innovation's spread, not the sensor's alone — the Kalman S = HPHᵀ + R.
struct Spread { double su2 = 0, sd2 = 0; };
inline Spread predictive_spread(const Ray2& r, const V3d& x, const M3d& P)
{
    Eigen::Matrix<double, 2, 3> J;
    for (int i = 0; i < 3; ++i)
    {
        V3d e = V3d::Zero(); e(i) = 1e-4;
        const UD p = to_anchor(r, x + e), m = to_anchor(r, x - e);
        J(0, i) = (p.u - m.u) / 2e-4;
        J(1, i) = (p.d - m.d) / 2e-4;
    }
    const Eigen::Matrix2d S = J * P * J.transpose();
    Spread sp{S(0, 0), S(1, 1)};
    if (not std::isfinite(sp.su2)) sp.su2 = 0;   // a ray parallel to the wall: its crossing is "far" either way
    if (not std::isfinite(sp.sd2)) sp.sd2 = 0;
    return sp;
}

struct Model
{
    double w, t;                       // opening width, wall thickness (m)
    double u_min, u_max;               // the door's wall run along the wall, anchor coordinates (m)
    std::vector<double> leaf_depths;   // the δ grid
    double short_level, count_level;   // both 3σ + t/2 (see the header: counting only)
    Model(double w_, double t_, double u_min_, double u_max_)
        : w(std::max(0.0, w_)), t(std::max(0.0, t_)),
          // the run always contains the opening itself (a run that excludes its own door is not this door's wall)
          u_min(std::min(u_min_, -0.5 * std::max(0.0, w_))), u_max(std::max(u_max_, 0.5 * std::max(0.0, w_))),
          short_level(3.0 * kRangeSigma + 0.5 * t), count_level(3.0 * kRangeSigma + 0.5 * t)
    {
        const int n = 1 + static_cast<int>(std::ceil(t / (0.5 * kRangeSigma)));
        for (int k = 0; k < n; ++k)
            leaf_depths.push_back(n == 1 ? 0.0 : -0.5 * t + t * double(k) / double(n - 1));
    }

    // Per-ray quantities shared by every branch (open, shut at each δ).
    struct Terms { double log_in, log_out, log_wall, log_through, log_clutter, sd2, log_run, log_off_run, log_nbhd; };
    // log_run: the ray's membership of THIS door's wall run, fixed per frame at the prior (run_membership below).
    Terms terms(const UD& a, const Spread& sp, double log_run) const
    {
        const double sd2 = kRangeSigma * kRangeSigma + sp.sd2;               // depth: range noise + pose spread
        const double sd  = std::sqrt(sd2);
        const double tau = std::sqrt(kRangeSigma * kRangeSigma + sp.su2);    // jamb blur: beam + pose spread
        const double z   = (0.5 * w - std::hypot(a.u, tau)) / tau;           // |u| blurred by τ
        // Occluder: the return is anywhere between the origin and the near face, uniform in depth (soft edges
        // at the depth spread). Background: anywhere in range.
        const double near = -0.5 * t;
        const double log_occ = log_sigmoid((near - a.d) / sd) + log_sigmoid((a.d - a.d_o) / sd)
                               - std::log(std::hypot(near - a.d_o, sd));
        const double log_bg  = -std::log(kMaxRange);
        // The doorway NEIGHBOURHOOD (visibility only): the opening and half a width of wall each side, |u| < w.
        const double log_nbhd = log_sigmoid((w - std::hypot(a.u, tau)) / tau);
        return {log_sigmoid(z), log_sigmoid(-z),
                log_normal(a.d - near, sd2),                                  // stops on the NEAR face
                log_sigmoid((a.d - 0.5 * t) / sd) - std::log(kMaxRange),     // beyond the far face
                log_add_exp(std::log(kClutter) + log_occ, std::log(kClutter) + log_bg),
                sd2, log_run, log1m_exp(log_run), log_nbhd};
    }
    // Every branch, on the run: (1 − 2ε)·[m·L_inside + (1 − m)·L_wall] + ε·L_occluder + ε·L_background;
    // off the run: the background alone (fix 2: no wall evidence).
    static double mix(const Terms& k, double log_inside)
    {
        const double core = log_add_exp(k.log_in + log_inside, k.log_out + k.log_wall);
        const double on   = log_add_exp(std::log(1.0 - 2.0 * kClutter) + core, k.log_clutter);
        return log_add_exp(k.log_run + on, k.log_off_run - std::log(kMaxRange));
    }
    // The doorway structure's responsibility for this ray against the clutter components (visibility only), the
    // open/shut latent at its prior and δ averaged — per ray, which is what "is THIS beam explained by the door" asks.
    double responsibility(const Terms& k, const UD& a) const
    {
        double l_leaf = -std::numeric_limits<double>::infinity();
        for (const double dl : leaf_depths) l_leaf = log_add_exp(l_leaf, log_normal(a.d - dl, k.sd2) - std::log(double(leaf_depths.size())));
        const double log_inside = log_add_exp(std::log(kPriorOpen) + k.log_through, std::log(1.0 - kPriorOpen) + l_leaf);
        const double core = std::log(1.0 - 2.0 * kClutter) + log_add_exp(k.log_in + log_inside, k.log_out + k.log_wall);
        return std::exp(core - log_add_exp(core, k.log_clutter));
    }
    // The door's wall RUN (fix 2): soft membership of [u_min, u_max] at the jamb blur, times "the beam travels TOWARD
    // the wall line" (its crossing lies ahead of the origin, d > d_o): a beam heading away never reaches this wall, so
    // its line's crossing is not a measurement of it. ★Without that factor, beams fired AWAY from the door crossed
    // the line (extended backwards) near the run's end and each paid log(1/ε) on the run: shifting the run to drop
    // them slid the opening along a shut wall (test turning_away_does_not_drag_anchor).
    double run_membership(const UD& a, const Spread& sp) const
    {
        const double sd  = std::sqrt(kRangeSigma * kRangeSigma + sp.sd2);
        const double tau = std::sqrt(kRangeSigma * kRangeSigma + sp.su2);
        return log_sigmoid((a.u - u_min) / tau) + log_sigmoid((u_max - a.u) / tau) + log_sigmoid((a.d - a.d_o) / sd);
    }
    // OPEN with a swung leaf: (1 − 2ε)·[h·L_leaf + (1 − h)·(m·L_through + (1 − m)·L_wall)] + clutter, on the run.
    static double loglik_open_swung(const Terms& k, double log_h, double log_l_leaf)
    {
        const double rest = log_add_exp(k.log_in + k.log_through, k.log_out + k.log_wall);
        const double core = log_add_exp(log_h + log_l_leaf, log1m_exp(std::min(log_h, -1e-12)) + rest);
        const double on   = log_add_exp(std::log(1.0 - 2.0 * kClutter) + core, k.log_clutter);
        return log_add_exp(k.log_run + on, k.log_off_run - std::log(kMaxRange));
    }
    double loglik_open(const Terms& k) const { return mix(k, k.log_through); }
    double loglik_shut(const Terms& k, const UD& a, double delta) const { return mix(k, log_normal(a.d - delta, k.sd2)); }
};

// The swung leaf as the model sees it: a φ grid (±2σ in σ steps, Gaussian weights) marginalised per FRAME, like δ.
struct SwingLeaf
{
    double hu = 0, du = 1, sy = -1, len = 0;
    std::vector<double> phis, log_w;   // empty = no swung-leaf component
};
// Where this ray's segment-line meets the leaf segment at angle φ: log membership (soft at the leaf's ends, the hit
// AHEAD of the origin) and the hit's signed depth in the ray's own convention (as UD::d).
struct LeafHit { double log_h = -std::numeric_limits<double>::infinity(), d_hit = 0; };
// oa, ea: the ray's origin and endpoint in ANCHOR coordinates.
inline LeafHit leaf_hit(const Eigen::Vector2d& oa, const Eigen::Vector2d& ea, const SwingLeaf& L, double phi, double tau, double sd)
{
    const Eigen::Vector2d v = ea - oa;
    const Eigen::Vector2d H(L.hu, 0.0), D(L.du * std::cos(phi), L.sy * std::sin(phi));
    const double den = v.x() * D.y() - v.y() * D.x();
    LeafHit h;
    if (std::abs(den) < 1e-12) return h;                       // parallel: it never meets the leaf
    const Eigen::Vector2d q = H - oa;
    const double t   = (q.x() * D.y() - q.y() * D.x()) / den;  // along the ray (1 = its endpoint)
    const double lam = (q.x() * v.y() - q.y() * v.x()) / den;  // along the leaf (0 = hinge, len = free edge)
    const double vlen = v.norm();
    h.log_h = log_sigmoid(lam / tau) + log_sigmoid((L.len - lam) / tau) + log_sigmoid(t * vlen / sd);
    const double sgn = oa.y() <= 0 ? 1.0 : -1.0;
    h.d_hit = sgn * (oa.y() + t * v.y());
    return h;
}

struct Mask
{
    std::vector<Eigen::Vector2d> pts;   // robot frame, x-y
    double sigma2 = 0;                  // σ_m², fixed at the prior (the spread is a property of the mask, not of x)
    bool   valid = false;
    double centroid_u(const V3d& x) const
    {
        const double c = std::cos(x.z()), s = std::sin(x.z());
        double acc = 0;
        for (const auto& p : pts) acc += c * (p.x() - x.x()) + s * (p.y() - x.y());
        return acc / double(pts.size());
    }
    double predictive_var(const V3d& x, const M3d& P) const   // σ_m² + J P Jᵀ
    {
        Eigen::RowVector3d J;
        for (int i = 0; i < 3; ++i) { V3d e = V3d::Zero(); e(i) = 1e-4; J(i) = (centroid_u(x + e) - centroid_u(x - e)) / 2e-4; }
        return sigma2 + J * P * J.transpose();
    }
    // −log Student-t(ū; 0, √var, ν), up to a constant
    static double neg_loglik(double r, double var) { return 0.5 * (kMaskNu + 1.0) * std::log1p(r * r / (kMaskNu * var)); }
};

struct Problem
{
    V3d xbar; M3d Pinv;
    Model model;
    std::vector<Ray2> rays;            // ALL rays: none is ever removed from F
    const Mask* mask = nullptr;
    std::vector<Spread> spread;        // per-ray predictive spread, fixed for one iteration (empty = sensor only)
    // Per-ray log membership of the door's wall run, decided ONCE per frame at the prior mean under the prior's spread.
    // ★Fixed, not re-evaluated at x: a run that moved with the hypothesis made its two ENDS lateral evidence — the shut
    //  door "localised" along its wall to the room polygon's wall ends (σ_lat 0.30 → 0.04 on a flat wall) — and a
    //  template that can shed inconvenient rays by sliding is a template that slides. Which rays belong to this wall
    //  is a data-association decision taken under the prior; the pose is then inferred from those rays only.
    std::vector<double> run_log;
    double mask_var = 0;
    double beta_lidar = 1.0, beta_cam = 1.0;   // visibility tempering (fix 1): P(door in view) per sensor
    SwingLeaf leaf;                            // the open branch's swung leaf (fix round 3); empty = none

    // Fix the predictive spreads at the linearisation point x under belief P (P = 0: the measurement model alone).
    void linearise(const V3d& x, const M3d& P)
    {
        spread.assign(rays.size(), Spread{});
        if (not P.isZero())
            for (std::size_t i = 0; i < rays.size(); ++i) spread[i] = predictive_spread(rays[i], x, P);
        if (mask != nullptr and mask->valid) mask_var = mask->predictive_var(x, P);
    }

    double F(const V3d& x) const
    {
        V3d dx = x - xbar;
        dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
        double f = 0.5 * dx.dot(Pinv * dx);
        const std::size_t nd = model.leaf_depths.size();
        double l_open = 0;
        std::vector<double> l_leaf(nd, 0.0);
        const std::size_t np = leaf.phis.size();
        std::vector<double> l_open_phi(np, 0.0);
        for (std::size_t i = 0; i < rays.size(); ++i)
        {
            const UD a = to_anchor(rays[i], x);
            const auto k = model.terms(a, spread[i], run_log[i]);
            if (np == 0) l_open += model.loglik_open(k);
            else
            {
                // OPEN with the leaf swung to φ: a beam whose line meets the leaf ahead of the sensor returns on it,
                // N(d − d_hit) — the 90° leaf beside the jamb is otherwise ~2000 "short" 3-D returns that a slightly
                // rotated wall line half-explains (test crossing_with_leaf_at_moderate_pi: yaw −0.17 rad, 0.11 m).
                const double tau = std::sqrt(kRangeSigma * kRangeSigma + spread[i].su2), sd = std::sqrt(k.sd2);
                const double cz = std::cos(x.z()), sz = std::sin(x.z());
                const Eigen::Vector2d qo = rays[i].o - x.head<2>(), qe = rays[i].e - x.head<2>();
                const Eigen::Vector2d oa(cz * qo.x() + sz * qo.y(), -sz * qo.x() + cz * qo.y());
                const Eigen::Vector2d ea(cz * qe.x() + sz * qe.y(), -sz * qe.x() + cz * qe.y());
                for (std::size_t j = 0; j < np; ++j)
                {
                    const LeafHit h = leaf_hit(oa, ea, leaf, leaf.phis[j], tau, sd);
                    l_open_phi[j] += model.loglik_open_swung(k, h.log_h, log_normal(a.d - h.d_hit, k.sd2));
                }
            }
            for (std::size_t j = 0; j < nd; ++j) l_leaf[j] += model.loglik_shut(k, a, model.leaf_depths[j]);
        }
        if (np > 0)
        {
            l_open = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < np; ++j) l_open = log_add_exp(l_open, leaf.log_w[j] + l_open_phi[j]);
        }
        double l_shut = -std::numeric_limits<double>::infinity();   // E_δ over the uniform grid
        for (const double l : l_leaf) l_shut = log_add_exp(l_shut, l - std::log(double(nd)));
        // The door is open or shut for the WHOLE frame: marginalise that latent at the frame level.
        if (not rays.empty())
            f -= beta_lidar * log_add_exp(std::log(kPriorOpen) + l_open, std::log(1.0 - kPriorOpen) + l_shut);
        if (mask != nullptr and mask->valid) f += beta_cam * Mask::neg_loglik(mask->centroid_u(x), mask_var);
        return f;
    }
    // Central finite differences. Steps: 1 mm / 1 mrad — well below τ = 5 cm, the shortest scale in F.
    void derivatives(const V3d& x, V3d& g, M3d& H) const
    {
        const V3d h(1e-3, 1e-3, 1e-3);
        const double f0 = F(x);
        for (int i = 0; i < 3; ++i)
        {
            V3d ei = V3d::Zero(); ei(i) = h(i);
            const double fp = F(x + ei), fm = F(x - ei);
            g(i) = (fp - fm) / (2.0 * h(i));
            H(i, i) = (fp - 2.0 * f0 + fm) / (h(i) * h(i));
            for (int j = 0; j < i; ++j)
            {
                V3d ej = V3d::Zero(); ej(j) = h(j);
                H(i, j) = H(j, i) = (F(x + ei + ej) - F(x + ei - ej) - F(x - ei + ej) + F(x - ei - ej)) / (4.0 * h(i) * h(j));
            }
        }
    }
};
// The data's information I_L = H − P̄⁻¹, projected onto its positive-semidefinite part. Along a flat (shut) wall the
// exact curvature of the data term is ~0 and can come out slightly NEGATIVE (the soft jamb edges); taken at face
// value that makes the Newton step a jump of metres and a "belief" WIDER than the prior (σ_lat 0.5 → 2.2 m), which
// then widened the next stage's predictive spread. A Gaussian measurement cannot add uncertainty: negative
// directions carry no information. (This is the projection that keeps the Laplace belief a proper Gaussian, not a
// tuned floor.)
inline M3d psd_information(const M3d& H, const M3d& Pinv)
{
    Eigen::SelfAdjointEigenSolver<M3d> es(0.5 * ((H - Pinv) + (H - Pinv).transpose()));
    return es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() * es.eigenvectors().transpose();
}

// ─── VISIBILITY at the prior (fix 1) ──────────────────────────────────────────────────────────────────────────
// LiDAR: v = Σ n·e (beams on the doorway neighbourhood), o = Σ n·e·r (explained by its structure) — diagnostics — and
// p = how much of the doorway was observed AS PREDICTED.
//   GEOMETRY. The doorway region is the wall slab |y| ≤ t/2 (+ the beam's depth spread) over |u| < w (the opening and
//   half a width of wall each side), cut into cells of the model's lateral resolution (kRangeSigma). A beam observes
//   the cells its segment origin → endpoint ENTERS: it ENDS in one (on a wall face, a jamb face, a leaf) or PASSES
//   through them (across the doorway, or along it from inside the doorway). A return on a jamb face also observes
//   the wall cells beside that jamb (the wall's END; from inside the doorway they are behind the jamb). A beam that
//   stopped SHORT of the slab "ends" at the cell where its line enters, with weight g = exp(−½ gap²/sd²) under the
//   prior's depth spread (a prior 0.35 m behind the true wall still sees it; an occluder a metre short does not).
//   THE WALLS (cells weighted by 1 − m, m = σ((w/2 − |u|)/τ) the opening membership): a beam ENDING in a cell is
//   consistent with the predicted wall (κ = 1 − m), one PASSING through it is not (κ = m) — passing through a
//   predicted WALL is evidence the wall is not there. Per cell N = Σ g, C = Σ g·κ, coverage 1 − (2ε)^N (each beam may
//   be clutter), score = coverage · C/N; the wall factor is the (1 − m)-weighted mean score.
//   THE OPENING is scored as a MODEL POSTERIOR (fix round 3): H_door — a doorway here, open with π = P(open) (the
//   door's own belief, passed in) or shut with 1 − π, ONE state for the whole frame — against H_wall — solid wall
//   here — at equal prior odds, from the beams that observed the opening cells (each observation weighted g·m):
//     log r = Σ g·m·[pass ? +λ : −λ],  λ = log((1 − 2ε)/2ε)   (r = L(open)/L(wall); a shut leaf IS a wall to a beam)
//     P(H_door) = (π·r + 1 − π) / (π·r + 1 − π + 1).
//   Pass-throughs favour the doorway for ANY π > 0; an ending on the predicted opening counts against it only as much
//   as π says the door is open (a stale π = 0.05 on a wall: ½·(0.95/1.0) ≈ 0.49 — the shut-door unobservability
//   below). The opening factor is P(H_door) times the opening's m-weighted coverage (an unobserved opening is not seen).
//   p = opening factor × wall factor: the OPENING and its WALLS both have to be seen as predicted.
// ★History. (1) "≥ 1 beam struck the doorway" saturated at 1 with the doorway half out of the sector, where a shut
//  span slid (turning_away_does_not_drag_anchor). (2) Credit only where a beam's LINE crossed y = 0 went unobserved
//  MID-CROSSING (P(view) 0.41–0.55 on the door line; mid_crossing_360_keeps_view). (3) Segment-entry credit WITHOUT
//  the outcome check read 1.00 for a prior 1.2 m in free space, rotated 90°, or 1.5 m along onto solid wall with the
//  door open (review probe; tests pview_low_*). "Seen" must mean seen AS PREDICTED. (4) Scoring the opening by
//  κ = m·π made p ∝ π: a correctly anchored open door read ≈ 0.86·π, under the publication level mid-cross at π ≤ 0.8,
//  and a STALE π = 0.05 made a wrong pose on solid wall outscore the truth (review round 3; tests
//  crossing_with_leaf_at_moderate_pi, true_pose_outscores_wrong_wall). Hence the posterior.
// ⚠ ACCEPTED LIMIT: a SHUT door offset ALONG its wall is unobservable by LiDAR — a flush leaf is the wall — so p stays
//   high there and the anchor's lateral error is not corrected by the rays. Only the camera's door mask can resolve
//   it, and the crossing needs the door open anyway (an open door's jambs pull the anchor into place).
struct LidarView { double v = 0, o = 0, p = 0; };
// Under the spreads pb.linearise(x, P) staged (the prior's). pi_open = P(open) from the door's belief.
inline LidarView lidar_view(const Problem& pb, const V3d& x, double pi_open)
{
    LidarView lv;
    const double w = pb.model.w;
    const int n_cells = std::max(1, int(std::ceil(2.0 * w / kRangeSigma)));
    const double cell = 2.0 * w / double(n_cells);
    std::vector<double> N(std::size_t(n_cells), 0.0), C(std::size_t(n_cells), 0.0), m(std::size_t(n_cells), 0.0);
    const double lambda = std::log((1.0 - 2.0 * kClutter) / (2.0 * kClutter));
    double log_r = 0.0;   // log L(open) / L(wall) over the opening's observations
    for (int cc = 0; cc < n_cells; ++cc)
        m[std::size_t(cc)] = std::exp(log_sigmoid((0.5 * w - std::abs(-w + (cc + 0.5) * cell)) / kRangeSigma));
    auto observe = [&](int cc, double g, bool ends)
    {
        const double mm = m[std::size_t(cc)];
        const double kappa = ends ? 1.0 - mm : mm;   // the WALL hypothesis' consistency (wall factor)
        N[std::size_t(cc)] += g;
        C[std::size_t(cc)] += g * kappa;
        log_r += g * mm * (ends ? -lambda : lambda);
    };
    auto cell_of = [&](double u) { return std::clamp(int(std::floor((u + w) / cell)), 0, n_cells - 1); };
    const double cz = std::cos(x.z()), sz = std::sin(x.z());
    auto inv = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - x.head<2>(); return Eigen::Vector2d(cz * q.x() + sz * q.y(), -sz * q.x() + cz * q.y()); };
    for (std::size_t i = 0; i < pb.rays.size(); ++i)
    {
        const UD a = to_anchor(pb.rays[i], x);
        const auto k = pb.model.terms(a, pb.spread[i], pb.run_log[i]);
        const double ne = std::exp(k.log_nbhd + k.log_run);
        lv.v += ne;
        lv.o += ne * pb.model.responsibility(k, a);
        // The segment's part inside the slab |y| ≤ hs, as a parameter interval [t0, t1] ⊂ [0, 1].
        const Eigen::Vector2d oa = inv(pb.rays[i].o), ea = inv(pb.rays[i].e);
        const double hs = 0.5 * pb.model.t + std::sqrt(k.sd2);
        const double dy = ea.y() - oa.y();
        double t0 = 0.0, t1 = 1.0;
        if (std::abs(dy) < 1e-12) { if (std::abs(oa.y()) > hs) continue; }
        else
        {
            double ta = (-hs - oa.y()) / dy, tb = (hs - oa.y()) / dy;
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(0.0, ta); t1 = std::min(1.0, tb);
            if (t0 > t1)
            {
                if (ta < 0.0) continue;                                    // the slab is behind the beam's origin
                const double u_in = oa.x() + ta * (ea.x() - oa.x());
                if (std::abs(u_in) >= w) continue;
                const double gap = std::abs(ea.y()) - hs;                  // stopped SHORT: "ends" there, softly
                observe(cell_of(u_in), std::exp(-0.5 * gap * gap / k.sd2), true);
                continue;
            }
        }
        const bool ends_inside = t1 >= 1.0;                                 // the endpoint lies in the (outer) slab
        auto pass_cells = [&](double pa_, double pb_)
        {
            if (pa_ > pb_) return;
            const double ua = oa.x() + pa_ * (ea.x() - oa.x()), ub = oa.x() + pb_ * (ea.x() - oa.x());
            const double lo = std::min(ua, ub), hi = std::max(ua, ub);
            if (hi > -w and lo < w)
                for (int cc = cell_of(std::max(lo, -w)); cc <= cell_of(std::min(hi, w)); ++cc) observe(cc, 1.0, false);
        };
        if (ends_inside)
        {
            if (std::abs(ea.x()) < w) observe(cell_of(ea.x()), 1.0, true);
            // Its path before the return, where it ran INSIDE the structure's own band |y| ≤ t/2 (along the opening from
            // inside the doorway, say), passed free space — except its last range-σ, which is the return's own noise.
            // The depth-spread band in front of a face is NOT counted: a beam ending on a wall face crosses it, and that
            // says nothing about absence.
            const double hc = 0.5 * pb.model.t;
            const double len = (ea - oa).norm();
            const double t_stop = len > kRangeSigma ? 1.0 - kRangeSigma / len : 0.0;
            if (std::abs(dy) < 1e-12) { if (std::abs(oa.y()) <= hc) pass_cells(0.0, t_stop); }
            else
            {
                double ta = (-hc - oa.y()) / dy, tb = (hc - oa.y()) / dy;
                if (ta > tb) std::swap(ta, tb);
                pass_cells(std::max(0.0, ta), std::min(t_stop, tb));
            }
        }
        else
            pass_cells(t0, t1);   // went THROUGH the slab: every cell it traversed was free
        // A return ON A JAMB FACE observes the wall's END: the wall cells beside that jamb.
        if (ends_inside)
        {
            const double tau = std::sqrt(kRangeSigma * kRangeSigma + pb.spread[i].su2);
            if (std::abs(std::abs(ea.x()) - 0.5 * w) <= tau)
                for (int cc = 0; cc < n_cells; ++cc)
                {
                    const double uc = -w + (cc + 0.5) * cell;
                    if (std::abs(uc) > 0.5 * w and (uc > 0) == (ea.x() > 0)) observe(cc, 1.0, true);
                }
        }
    }
    double so = 0, wo = 0, sw = 0, ww = 0;
    for (int cc = 0; cc < n_cells; ++cc)
    {
        const std::size_t c = std::size_t(cc);
        const double cover = 1.0 - std::pow(2.0 * kClutter, N[c]);
        so += m[c] * cover; wo += m[c];
        sw += (1.0 - m[c]) * (N[c] > 0 ? cover * (C[c] / N[c]) : 0.0); ww += (1.0 - m[c]);
    }
    // P(H_door | beams) with r = exp(log_r), in a form that cannot overflow.
    double p_door;
    if (log_r > 0.0)
    {
        const double ir = std::exp(-log_r);                                   // 1/r
        p_door = (pi_open + (1.0 - pi_open) * ir) / (pi_open + (1.0 - pi_open) * ir + ir);
    }
    else
    {
        const double r = std::exp(log_r);
        p_door = (pi_open * r + 1.0 - pi_open) / (pi_open * r + 1.0 - pi_open + 1.0);
    }
    lv.p = (wo > 0 ? so / wo : 0.0) * p_door * (ww > 0 ? sw / ww : 0.0);
    return lv;
}
// Camera: both jambs inside the frustum (azimuth and range), blurred by their predictive spread under P.
inline double camera_view(const CameraFrustum& cam, double w, const V3d& x, const M3d& P)
{
    if (not cam.known) return 1.0;
    const Eigen::Vector2d o = cam.origin.cast<double>(), f = cam.forward.cast<double>().normalized();
    const Eigen::Vector2d right(f.y(), -f.x());
    const Eigen::Vector2d a(std::cos(x.z()), std::sin(x.z())), da(-std::sin(x.z()), std::cos(x.z()));
    double log_p = 0.0;
    for (const double sgn : {-1.0, 1.0})
    {
        const Eigen::Vector2d J = x.head<2>() + sgn * 0.5 * w * a;
        Eigen::Matrix<double, 2, 3> Jac; Jac << 1, 0, sgn * 0.5 * w * da.x(), 0, 1, sgn * 0.5 * w * da.y();
        const Eigen::Matrix2d S = Jac * P * Jac.transpose();
        const Eigen::Vector2d q = J - o;
        const double range = std::max(q.norm(), 1e-6);
        const Eigen::Vector2d u = q / range, n(-u.y(), u.x());
        const double az = std::atan2(q.dot(right), q.dot(f));
        const double s_az = std::sqrt(kRangeSigma * kRangeSigma + n.dot(S * n)) / range;
        const double s_r  = std::sqrt(kRangeSigma * kRangeSigma + u.dot(S * u));
        log_p += log_sigmoid((double(cam.half_fov) - std::abs(az)) / s_az) + log_sigmoid((double(cam.max_range) - range) / s_r);
    }
    return std::exp(log_p);
}
}  // namespace anchor_detail

// ─── RAY SELECTION for the update (the caller's half, kept here so it is testable) ────────────────────────────────
// A FLOOR return (z below the wall band) is kept iff its beam went THROUGH THIS OPENING: it crossed the predicted
// door line between the sensor and the endpoint, at a height inside the doorway's band [z_lo, z_hi], within the
// opening (|u| < w/2 + τ at the prior, τ the jamb blur), and it landed beyond the predicted FAR face by more than the
// prior's own depth σ. The downward bpearl dome sees an open doorway almost only this way (its through-beams end on
// the next room's floor). ★The depth clause is what keeps a mis-predicted SHUT door shut: with the prior 0.25 m in
// front of the true wall, real floor returns between the predicted line and the wall crossed the line inside the band
// and read as "through" (review, fix round 1; test shut_door_reads_shut_with_floor). Beyond the far face by more than
// the prior's depth σ, the floor is behind the wall wherever the wall can be.
inline bool keep_floor_return(const rc::interaction::RelPose& prior, float w, float t, float z_lo, float z_hi,
                              const Eigen::Vector3f& origin, const Eigen::Vector3f& p)
{
    const Eigen::Vector2f c = prior.x.head<2>();
    const Eigen::Vector2f a(std::cos(prior.x.z()), std::sin(prior.x.z())), n(-a.y(), a.x());
    const float s_o = n.dot(origin.head<2>() - c), s_p = n.dot(p.head<2>() - c);
    if (not (s_o * s_p < 0.f)) return false;                                  // never crossed the line
    const float k = s_o / (s_o - s_p);
    const float z_cross = origin.z() + k * (p.z() - origin.z());
    if (z_cross < z_lo or z_cross > z_hi) return false;                       // under the floor band / over the lintel
    const float u_cross = a.dot(origin.head<2>() + k * (p.head<2>() - origin.head<2>()) - c);
    if (std::abs(u_cross) >= 0.5f * w + float(anchor_detail::kRangeSigma)) return false;   // not through the opening
    const float sigma_depth = std::sqrt(std::max(0.f, n.dot(prior.P.topLeftCorner<2, 2>() * n)));
    return std::abs(s_p) > 0.5f * t + sigma_depth;                            // beyond the far face by > the prior's σ
}

// The update's rays from one robot-frame sweep: the wall band [z_lo, z_hi], plus the floor returns that went through
// the opening (keep_floor_return), restricted to beams whose segment passes within `reach` of the predicted opening
// (COST only: farther beams cannot cross the opening or its jambs).
inline std::vector<Eigen::Vector3f> select_anchor_rays(const rc::interaction::RelPose& prior, float w, float t,
                                                       float z_lo, float z_hi, const Eigen::Vector3f& origin,
                                                       const std::vector<Eigen::Vector3f>& sweep, float reach)
{
    std::vector<Eigen::Vector3f> out;
    const Eigen::Vector2f c = prior.x.head<2>(), o = origin.head<2>();
    for (const auto& p : sweep)
    {
        if (p.z() > z_hi) continue;
        if (p.z() < z_lo and not keep_floor_return(prior, w, t, z_lo, z_hi, origin, p)) continue;
        const Eigen::Vector2f d = p.head<2>() - o;
        const float len2 = d.squaredNorm();
        if (len2 < 1e-6f) continue;
        const float u = std::clamp((c - o).dot(d) / len2, 0.0f, 1.0f);
        if ((o + u * d - c).norm() <= reach) out.push_back(p);
    }
    return out;
}

// One free-energy minimisation: prior = the predicted RelPose; likelihood = rays (+ mask). Levenberg-Marquardt on
// (x, y, yaw) against the predictive spread (two stages); Laplace covariance at the sensor's σ with the frame's
// common mode marginalised (Woodbury).
// wall_u_min/max: the door's wall run along the wall in ANCHOR coordinates (fix 2); `cam`: the zed frustum in the
// robot frame for the camera's visibility (fix 1; default unknown = the mask is not tempered by geometry);
// p_open_prior: the door's own P(open) (door_concept's leaf-angle belief), the structure the visibility check
// predicts in the opening (0.5 = unknown: a leaf and a gap are equally expected there).
inline std::pair<rc::interaction::RelPose, AnchorEvidence>
update_door_anchor(const rc::interaction::RelPose& prior, float width, float thickness, float wall_u_min, float wall_u_max,
                   const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                   const std::vector<Eigen::Vector3f>& mask_pts_robot, const CameraFrustum& cam = {},
                   float p_open_prior = 0.5f, const LeafPrior& leaf = {})
{
    using namespace anchor_detail;
    AnchorEvidence ev;

    // Mask: the spread needs two points; one point has no measured σ and claims nothing.
    Mask mask;
    if (mask_pts_robot.size() >= 2)
    {
        for (const auto& p : mask_pts_robot) mask.pts.push_back(p.head<2>().cast<double>());
        const V3d x0 = prior.x.cast<double>();
        const double c = std::cos(x0.z()), s = std::sin(x0.z());
        double mu = 0, m2 = 0;
        for (const auto& q : mask.pts) mu += c * (q.x() - x0.x()) + s * (q.y() - x0.y());
        mu /= double(mask.pts.size());
        for (const auto& q : mask.pts) { const double du = c * (q.x() - x0.x()) + s * (q.y() - x0.y()) - mu; m2 += du * du; }
        const double spread2 = m2 / double(mask.pts.size() - 1);
        mask.sigma2 = spread2 / double(std::min<std::size_t>(mask.pts.size(), kMaskNCap));
        mask.valid = mask.sigma2 > 0;   // all points at one abscissa: no spread ⇒ no measured σ
    }

    Problem pb{prior.x.cast<double>(), prior.P.cast<double>().inverse(), Model(width, thickness, wall_u_min, wall_u_max),
               {}, &mask, {}, {}, 0};
    if (leaf.known)
    {
        pb.leaf.hu = leaf.hinge_u; pb.leaf.du = leaf.dir_u; pb.leaf.sy = leaf.swing_y; pb.leaf.len = pb.model.w;
        const int nz = leaf.phi_sigma > 0.f ? 2 : 0;
        for (int z = -nz; z <= nz; ++z)
        {
            const double phi = std::clamp(double(leaf.phi) + z * double(leaf.phi_sigma), 0.0, M_PI_2);
            if (not pb.leaf.phis.empty() and pb.leaf.phis.back() == phi)   // clamped onto the same angle: one node
                pb.leaf.log_w.back() = log_add_exp(pb.leaf.log_w.back(), -0.5 * z * z);
            else { pb.leaf.phis.push_back(phi); pb.leaf.log_w.push_back(-0.5 * z * z); }
        }
        double norm = -std::numeric_limits<double>::infinity();
        for (const double lw : pb.leaf.log_w) norm = log_add_exp(norm, lw);
        for (double& lw : pb.leaf.log_w) lw -= norm;
    }
    const Eigen::Vector2d o = lidar_origin_robot.head<2>().cast<double>();
    pb.rays.reserve(rays_robot.size());
    for (const auto& p : rays_robot)
        if (const Eigen::Vector2d e = p.head<2>().cast<double>(); (e - o).squaredNorm() > 1e-12)   // a zero-length ray has no direction
            pb.rays.push_back({o, e});

    if (pb.rays.empty() and not mask.valid)
        return {prior, ev};   // nothing measured: the prior IS the posterior, bit for bit

    V3d x = pb.xbar;
    M3d P = prior.P.cast<double>();
    // ─── THIS DOOR'S WALL RUN, decided at the prior (fix 2) ────────────────────────────────────────────────────
    pb.linearise(x, P);   // the prior's predictive spreads: the run, the visibility and stage 0 all use these
    pb.run_log.resize(pb.rays.size());
    for (std::size_t i = 0; i < pb.rays.size(); ++i)
        pb.run_log[i] = pb.model.run_membership(to_anchor(pb.rays[i], x), pb.spread[i]);
    // ─── VISIBILITY at the prior: how much of this frame's evidence is about THIS doorway (fix 1) ─────────────
    {
        const LidarView lv = lidar_view(pb, x, std::clamp(double(p_open_prior), 0.0, 1.0));
        const double pc = camera_view(cam, pb.model.w, x, P);
        pb.beta_lidar = lv.p;
        pb.beta_cam   = pc;
        ev.v_lidar = float(lv.v);
        ev.o_lidar = float(lv.o);
        ev.p_lidar = float(lv.p);
        ev.p_cam   = mask.valid ? float(pc) : 0.f;
        ev.p_view  = float(1.0 - (1.0 - lv.p) * (1.0 - double(ev.p_cam)));
    }
    // A ray whose run membership is below 1e-9 contributes a CONSTANT to F (the background, x-independent) to within
    // 1e-9 relative: it is left out of the optimisation as a matter of arithmetic, not of modelling (COST only — the
    // counts below still see every ray).
    const std::vector<Ray2> all_rays = pb.rays;
    {
        std::size_t n = 0;
        for (std::size_t i = 0; i < pb.rays.size(); ++i)
            if (pb.run_log[i] > std::log(1e-9))
            {
                pb.rays[n] = pb.rays[i]; pb.run_log[n] = pb.run_log[i]; pb.spread[n] = pb.spread[i]; ++n;
            }
        pb.rays.resize(n); pb.run_log.resize(n); pb.spread.resize(n);
    }
    M3d H = pb.Pinv;
    V3d g;
    // ─── two STAGES of predictive spread, then the sensor alone (a third LM stage, then the Laplace) ─────────────────────────────────────────────
    // Each stage scores the rays against the spread a BELIEF predicts for them (the Kalman S = HPHᵀ + R): stage 1
    // the prior's, stage 2 the Laplace belief stage 1 converged to. From a prior 0.35 m off in depth the wall
    // returns are then a broad pull rather than 7σ outliers the occluder component would absorb, and a 0.45 m
    // lateral error sits inside the jamb's blurred reach. The spread is updated only at a stage's OPTIMUM — the
    // curvature at a point still 0.5 m off is not a belief, and using it collapsed the reach mid-way (dy = +0.35:
    // σ_lat 0.5 → 0.12 after one step, anchor stranded 0.66 m along the wall). The final Laplace (below) is at the
    // sensor's own σ. Per stage: at most kMaxIter accepted Levenberg-Marquardt steps.
    // ★Task 7: a THIRD stage at the sensor's own σ. The two spread stages optimise a WIDENED objective, and on a shut
    //  wall (lateral unobservable) its optimum wanders along the wall — the blurred jamb/leaf terms' Occam push toward
    //  fewer in-span rays — by ~0.1 m; the closing Laplace mean is one Newton step and kept most of that wander wherever
    //  the true F ripples at the jamb's ray spacing. Converging on the true F first: frame-0 lateral wander 0.092 →
    //  0.037 m, worst over the turn 0.100 → 0.043 m (test turning_away_does_not_drag_anchor). Cost ~+30 %.
    for (int stage = 0; stage < 3; ++stage)
    {
        if (stage > 0) pb.linearise(x, stage < 2 ? P : M3d::Zero());   // stage 0: staged above, at the prior
        // Levenberg-Marquardt in the PRIOR's metric (damping μ P̄⁻¹), steps accepted on the gain ratio ρ = actual /
        // predicted decrease (Nielsen's μ update). ★A plain Newton step + Armijo line search failed here: along a
        // shut wall the lateral direction has only the prior's curvature, so the step's lateral part was metres long,
        // and the large depth/yaw gain in the same step made the line search ACCEPT it (dy = +0.35: the anchor was
        // dragged 1.3 m along the wall). The gain ratio checks the quadratic model's promise per step, so a
        // direction the model mispredicts is shrunk instead of carried along. (ρ > 0, ×2, ⅓ are the textbook
        // optimiser constants, not model levels.)
        double mu = 1e-2, nu = 2.0;
        for (int it = 0; it < kMaxIter; ++it)
        {
            pb.derivatives(x, g, H);
            const M3d Hp = pb.Pinv + psd_information(H, pb.Pinv);   // positive definite by construction
            const double f0 = pb.F(x);
            bool moved = false;
            for (int k = 0; k < 30 and not moved; ++k)
            {
                const V3d step = -(Hp + mu * pb.Pinv).llt().solve(g);
                const double predicted = -(g.dot(step) + 0.5 * step.dot(Hp * step));
                const double actual    = f0 - pb.F(x + step);
                if (predicted > 0 and actual > 0)
                {
                    const double rho = actual / predicted;
                    x += step;
                    mu *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3));
                    nu = 2.0;
                    moved = true;
                }
                else { mu *= nu; nu *= 2.0; }
            }
            x.z() = std::remainder(x.z(), 2.0 * M_PI);
            if (not moved) break;   // no step decreases F: at this stage's optimum (to the optimiser's resolution)
        }
        pb.derivatives(x, g, H);
        P = (pb.Pinv + psd_information(H, pb.Pinv)).llt().solve(M3d::Identity());
    }

    // ─── Laplace at the measurement model's own σ, with the frame's common mode marginalised ─────────────────
    // H = P̄⁻¹ + I_L (I_L = the data's curvature, PSD part). The rays share one LiDAR extrinsic and one wall, so their
    // lateral/depth evidence carries a common offset b ~ N(0, C), C = σ_c² on (x, y): the data then inform the
    // anchor through I_eff = (I_L⁻¹ + C)⁻¹ = (I + I_L C)⁻¹ I_L (Woodbury: no inverse of I_L, which is singular
    // along a shut wall). Without it a frame of ~900 rays claimed a 2 mm depth σ. The mean is the same Gaussian
    // algebra: x = x̄ + P_post (I + I_L C)⁻¹ I_L (x_L − x̄), with I_L (x_L − x̄) = H (x − x̄) − g (one Newton step
    // on the true F, which also absorbs the iteration's residual gradient).
    pb.linearise(x, M3d::Zero());
    pb.derivatives(x, g, H);
    V3d dx = x - pb.xbar;
    dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
    const M3d I_L = psd_information(H, pb.Pinv);
    H = pb.Pinv + I_L;
    if (pb.F(x + H.llt().solve(-g)) > pb.F(x))
        g.setZero();   // the true-F Newton step would not descend: x already is its optimum to our resolution
    const M3d C   = V3d(kCommonSigma * kCommonSigma, kCommonSigma * kCommonSigma, 0.0).asDiagonal();
    const M3d A   = M3d::Identity() + I_L * C;
    const Eigen::PartialPivLU<M3d> Alu(A);
    M3d I_eff = Alu.solve(I_L);
    I_eff = 0.5 * (I_eff + I_eff.transpose());

    rc::interaction::RelPose post = prior;
    if (Eigen::LLT<M3d> llt(pb.Pinv + I_eff); llt.info() == Eigen::Success)
    {
        const M3d S = llt.solve(M3d::Identity());
        V3d xp = pb.xbar + S * Alu.solve(H * dx - g);
        xp.z() = std::remainder(xp.z(), 2.0 * M_PI);
        post.x = xp.cast<float>();
        post.P = (0.5 * (S + S.transpose())).cast<float>();
    }
    else
    {
        // Not positive definite: the data did not localise the anchor here — keep the last iteration's belief.
        post.x = x.cast<float>();
        post.P = P.cast<float>();
    }
    x = post.x.cast<double>();

    // ─── Evidence at the posterior mean (COUNTING ONLY — see the header) ─────────────────────────────────────
    for (const auto& r : all_rays)
    {
        const UD a = to_anchor(r, x);
        if (a.d < -pb.model.short_level) { ++ev.n_short; continue; }   // ended in front of the near face
        if (std::abs(a.u) >= 0.5 * pb.model.w) continue;               // crosses outside the span
        if (a.d > pb.model.count_level) ++ev.n_through;
        else ++ev.n_blocked;                                            // |d| ≤ 3σ + t/2
    }
    // Tempered like the pose evidence: a door out of view says nothing about whether it is open.
    ev.p_open = float((pb.beta_lidar * ev.n_through + 1.0) / (pb.beta_lidar * (ev.n_through + ev.n_blocked) + 2.0));
    if (mask.valid)
    {
        const double r = mask.centroid_u(x);
        ev.camera_nis = float(r * r / mask.sigma2);
    }
    return {post, ev};
}

}  // namespace rc::door
