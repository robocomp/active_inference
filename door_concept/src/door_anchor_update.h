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
 * Free energy: F(x) = ½ (x − x̄)ᵀ P̄⁻¹ (x − x̄) − log p(rays | x) − log p(mask | x). No ray is removed from F by a
 * misfit; the frame is SUBSAMPLED to a compute budget (stratified, each kept ray weighted by the beams it stands for —
 * see THE RAY BUDGET), which estimates the whole frame's log-likelihood rather than dropping any of it.
 *   - PREDICTIVE SPREAD. Each ray's depth σ and jamb blur τ are widened by the spread the current belief P
 *     predicts for that ray (J P Jᵀ, the Kalman S = HPHᵀ + R); the mask's σ likewise. Two stages: the prior's P,
 *     then the Laplace P stage 1 converged to; then a third at the sensor's own σ (Task 7, see the loop). This is what gives the update its reach: against the sensor's 5 cm
 *     alone, wall returns from a prior 0.3 m off in depth are outliers the occluder absorbs, and a 0.45 m lateral
 *     error is beyond the jamb's pull.
 *   - OPTIMISER. Per stage, at most 5 accepted Levenberg-Marquardt steps in the prior's metric, accepted on the
 *     gain ratio, stopping when the undamped step promises < 1e-3 nats. Derivatives are finite differences of F in
 *     double precision (F is smooth by construction); the curvature used is P̄⁻¹ + [I_L]₊, the data's information
 *     projected onto its PSD part.
 *   - LAPLACE + PERSISTENT BIAS (Task 10). At the sensor's own σ: I_L from the exact finite-difference Hessian (a Gauss-
 *     Newton/Fisher one would count each jamb-adjacent wall ray as lateral information on a featureless shut wall).
 *     What the rays share — the mount (σ_c on x, y; its yaw ε about the LiDAR, a lateral term growing with range), the
 *     wall's straightness, and the misfit of the structure this model does not describe (a cluster-robust sandwich of
 *     the rays' scores) — is ONE error from frame to frame, not fresh noise: it is carried as a latent factor L beside
 *     the pose (AnchorBias), the update uses the Schmidt (consider) gain, and the published P = P_z + L Lᵀ never falls
 *     below one view's systematic error. (Task 7–9's per-frame Woodbury common mode, I_eff = (I + I_L C)⁻¹ I_L, was
 *     divided by √N over a chain: try66–69 published σ_yaw 0.1–0.2° against 2–28° errors.)
 *
 * COST (fix round 4, -O2): ≤ ~30 ms per update on a 7,800-ray 3-D bpearl frame with the leaf configurations
 * marginalised (was 1.4–1.9 s: every ray within 2 m entered F, ~400 evaluations of F per update). Bounded by THE RAY
 * BUDGET (≤ 400 rays in F, whatever the frame size; the selection pass over the full frame is ~0.7 ms per 7,000 rays),
 * the leaf configurations (all ~130 scored once per frame, ≤ 4 in F) and the optimiser's tolerance. The harness
 * test_door_anchor_update times the crossing frames and fails above 50 ms.
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
 * tempered form of "rays | door in view" mixed with "rays | door not in view", the latter independent of x) — both
 * evaluated at the PRIOR, which is what the evidence is weighed against. When they → 0 the update reduces to the
 * prior, i.e. the odometry prediction, whose covariance grows with the motion.
 * REPORTED (fix round 4): p_view = 1 − (1 − p_lidar)(1 − p_cam), p_cam = p_cam_geom when a mask was consumed, else 0,
 * evaluated at the POSTERIOR (door_view): the pose the caller publishes is the pose whose visibility it reads.
 * The CALLER then holds the last mask frame's p_cam while that frame is younger than kAnchorStaleMs (Task 8 fix 1:
 * held_camera_view / combine_view): a cycle without a new frame is not "the camera sees nothing".
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
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
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
    float p_lidar = 0.f;                              // = lidar_open · lidar_door · lidar_wall (lidar_view)
    float lidar_open = 0.f, lidar_door = 0.f, lidar_wall = 0.f;   // its three factors (diagnostics: the replay prints them)
    int   n_budget = 0;                               // rays that entered F (after the ray budget), for the frame record
    float az_step = 0.f;                              // the frame's measured azimuth step (rad), for the log
    float p_cam   = 0.f;                              // P(opening in the zed frustum) when a mask was consumed, else 0
    float p_view  = 0.f;                              // P(door in view) = 1 − (1 − p_lidar)(1 − p_cam)
    float sig_bias_yaw = 0.f;                         // Task 10: the persistent bias's yaw σ in the published P (rad)
    // Task 10 diagnostics (the replay prints them): the frame's model information I_L, its score-space misfit J_s, the gain.
    Eigen::Matrix3f info = Eigen::Matrix3f::Zero(), misfit = Eigen::Matrix3f::Zero(), gain = Eigen::Matrix3f::Zero();
    Eigen::Vector3f info_innov = Eigen::Vector3f::Zero();   // I_L ν: the data-only optimum's pull on the prior mean
};

// The door's own belief about its SWUNG leaf, in ANCHOR coordinates (fix round 3). The leaf is a segment of length w
// from the hinge (hinge_u, 0) along (dir_u·cos φ, swing_y·sin φ): at φ = 0 it closes the opening. `known` false = no
// swung-leaf component (an open door's leaf is then left to the clutter components, as before).
// ★Fix round 4: the configuration is a BELIEF, not a fact. update_door_anchor marginalises, per frame, the hinge side
//  (stated / flipped, weights p_hinge / 1 − p_hinge), the swing side (p_swing), φ over the door's whole range
//  [0, phi_max] (weights N(φ; phi, phi_sigma) when phi_known, else uniform), and a "leaf NOT as believed" branch of
//  prior mass p_not_as_believed (half on "no leaf in the model": the old open mixture; half spread uniformly over every
//  hinge × swing × φ). A belief taken as certain cost 0.52 m / 1.57 rad with the hinge flipped (round-4 probe).
struct LeafPrior
{
    bool  known = false;
    float hinge_u = 0.f;     // ±w/2: the hinge jamb (as believed)
    float dir_u = 1.f;       // along-wall direction from the hinge across the opening (−sign(hinge_u))
    float swing_y = -1.f;    // the side it opens toward (−1 = the robot's side at creation)
    float phi = 0.f;         // opening angle (rad), the door's estimate
    float phi_sigma = 0.f;   // its spread (rad); 0 = as sharp as the φ grid resolves
    bool  phi_known = true;  // false = no φ estimate: uniform over [0, phi_max]
    float phi_max = float(M_PI_2);   // the hinge's travel (rad): the door's real range, up to π
    float p_hinge = 0.5f;    // P(the hinge is on the stated jamb); 0.5 = no belief (door_concept has none yet)
    float p_swing = 0.5f;    // P(it swings to the stated side); 0.5 = no belief
    // P(the leaf is NOT as believed). ⚑ 0.2, flagged: no logged rate exists. door_concept's φ right after an `open`
    // command IS the command (phi_from_command), its φ curve is often flat exactly while the door is open, and hinge /
    // swing are config pins, so its belief is a prediction that is wrong whenever the door did not do what was asked.
    // "One in five" is that guess; the value is NOT load-bearing: with hundreds of leaf returns per frame the data's
    // log-likelihood ratio between configurations is tens to hundreds of nats, so any mass in [0.05, 0.5] gives the
    // same posterior (fix round 4 report: the wrong-belief table at 0.05 / 0.2 / 0.5).
    float p_not_as_believed = 0.2f;
    // Task 9 fix round 1: the leaf's length (m); 0 = the opening's width. A door's leaf spans its CLEAR width between the
    // casings (hinged on a casing's inner face), narrower than the wall gap the opening model is fitted to.
    float len = 0.f;
};

// ─── THE TWO WIDTHS OF A DOORWAY (Task 9 fix round 1) ──────────────────────────────────────────────────────────────
// door_concept's aperture w is the WALL GAP: the hole in the wall plane its belief is fitted to (DOOR_1: 1.06 m). The
// passage is narrower: a casing board of width c lines each side (DoorConcept.FrameCasingM — Webots' Door.proto: 0.05 m
// frames, the walls ending 3 cm short of them; the config's 0.06 m), and the leaf hangs on a casing's inner face and spans
// the CLEAR width between them (DOOR_1: 0.9 m clear, a 0.95 m leaf overlapping its stops). The anchor's opening model is
// the thin-wall hole (wall gap); its LEAF and the controller's servo (published width_m) are the clear width.
// ★try67 (replay, current model): the update's opening at the clear width 0.93 dragged the anchor 45° at the mouth (the
//  casings and the slots beside them do not fit a thin wall), while the leaf hinged at the wall gap's half-width dragged
//  it 30° (the leaf's real hinge 6 cm further in). Hinge + length at the clear width, opening at the wall gap: 94 % of the
//  approach-to-mouth frames ≥ 0.6, within 0.06 m of the truth.
[[nodiscard]] inline float anchor_clear_width(float wall_gap, float casing) { return std::max(0.f, wall_gap - 2.f * std::max(0.f, casing)); }

// ─── WHAT TO PUBLISH THIS CYCLE (Task 9 fix round 1, user-approved) ────────────────────────────────────────────────
// Seen      — this measured cycle saw the doorway (P(door in view) ≥ the publication level): the posterior.
// Predicted — not seen now, but the door's CROSS is claimed, the anchor was seen in this passage, and the last sighting is
//             at most predict_ms old (interaction_anchor.h kAnchorPredictMs, ⚑ flagged): the last seen posterior carried
//             forward by odometry, its covariance grown by the motion — the controller's P(body fits) slows on it.
// None      — otherwise (the strict rule): nothing is written, the edge ages out.
enum class AnchorPublish : std::uint8_t { None, Seen, Predicted };
[[nodiscard]] inline AnchorPublish anchor_publish_decision(bool seen_now, bool cross_claimed, bool seen_in_passage,
                                                           std::int64_t since_seen_ms, std::int64_t predict_ms)
{
    if (seen_now) return AnchorPublish::Seen;
    if (cross_claimed and seen_in_passage and since_seen_ms >= 0 and since_seen_ms <= predict_ms) return AnchorPublish::Predicted;
    return AnchorPublish::None;
}

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
constexpr double kFTol        = 1e-3;   // optimiser tolerance (nats): a step promising less is not taken — F's posterior scale is 1 nat
constexpr double kPriorOpen   = 0.5;    // P(door open) before this frame: uninformed (the interface carries no leaf belief)
// ⚑ FLAGGED CONSTANT — the clutter fraction: the share of beams explained by something that is not this doorway's
//   wall/opening/leaf. It weights BOTH extra components: an occluder in front of the plane (the leaf swung toward the
//   robot, a person) and a background return anywhere in range (another doorway in the same wall, the wall's end,
//   glass). No model term sets it from data yet; 0.05 is "one beam in twenty".
constexpr double kClutter     = 0.05;
// The per-frame COMMON MODE: every ray of one frame shares the LiDAR's extrinsic error and the wall's departure
// from a plane, so their evidence about the anchor's lateral/depth position is correlated at this σ (m).
constexpr double kCommonSigma = 0.02;
// ⚑ COMPUTE CUTOFF (not a model level): a term below 1e-6 of what it is added to is dropped (inside_negligible,
//   leaf_hit). The likelihood it changes moves by ≤ 1e-6 per ray; its x-gradient by less.
constexpr double kLogNegligible = -13.815510557964274;   // log(1e-6)
// The door LEAF's half thickness (m): a standard interior leaf is ~40 mm. Model geometry (see leaf_hit): a beam stops on
// the leaf's FACE, and the leaf turns about a hinge on the jamb's swing-side face. With a thin leaf hinged on the
// mid-plane (round 3), the 90-degree-open leaf's face sat 4 cm off the model line; once phi was marginalised (fix round 4)
// the fit traded that misfit for phi = 1.47 and 0.08 rad of anchor yaw mid-doorway (pinning phi = pi/2 gave 0.027).
constexpr double kLeafHalfThick = 0.02;

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
// log(1 − e^a) for a ≤ 0, accurate at both ends.
inline double log1m_exp(double a)
{
    if (a >= 0.0) return -std::numeric_limits<double>::infinity();
    return a > -M_LN2 ? std::log(-std::expm1(a)) : std::log1p(-std::exp(a));
}

// A 2-D ray in ROBOT coordinates.
// floor: the return is on the FLOOR (below the caller's wall band): it hit no wall (Task 8, visibility only).
struct Ray2 { Eigen::Vector2d o, e; bool floor = false; std::uint32_t src = 0; };   // src: index into the caller's rays

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
struct Spread
{
    double su2 = 0, sd2 = 0;   // the pose's predictive variance of (u, d)
    // ...and of the ENDPOINT's own along-wall coordinate (fix round 1: the jamb-face test in view_visit asks where the
    // return IS, not where its line crosses the wall — for a beam nearly parallel to the wall the crossing's spread is
    // metres, and a far-wall return 3.5 m away passed as "on the jamb face").
    double se2 = 0;
    // Derived per ray once per stage (fix round 4, COST): the depth variance with the range noise, its σ, the jamb blur
    // τ and the Gaussian's log normaliser — constant while the spread is.
    double var_d = kRangeSigma * kRangeSigma, sd = kRangeSigma, tau = kRangeSigma,
           log_norm = 0.5 * std::log(2.0 * M_PI * kRangeSigma * kRangeSigma);
};
inline Spread derive(Spread s)
{
    s.var_d = kRangeSigma * kRangeSigma + s.sd2;
    s.sd = std::sqrt(s.var_d);
    s.tau = std::sqrt(kRangeSigma * kRangeSigma + s.su2);
    s.log_norm = 0.5 * std::log(2.0 * M_PI * s.var_d);
    return s;
}
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
    {
        // ea_u = R(−yaw)(e − t)·x̂ ⇒ ∂/∂t = (−cos yaw, −sin yaw), ∂/∂yaw = the endpoint's anchor y.
        const double c = std::cos(x.z()), s = std::sin(x.z());
        const Eigen::Vector2d q = r.e - x.head<2>();
        const Eigen::RowVector3d Je(-c, -s, -s * q.x() + c * q.y());
        sp.se2 = std::max(0.0, double(Je * P * Je.transpose()));
    }
    return derive(sp);
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
    //   In LINEAR terms every branch is  p = α·L_inside + β  with  α = run·(1 − 2ε)·m  and
    //   β = run·[(1 − 2ε)(1 − m)·L_wall + clutter] + (1 − run)/R  — so a branch costs ONE log-add once α, β are known
    //   (fix round 4, COST: the δ loop was three nested log-adds per node). log_K, log_O serve the swung leaf:
    //   p = run·(1 − 2ε)·h·L_leaf + (1 − h)·O + K,  O = run·(1 − 2ε)·(m·L_through + (1 − m)·L_wall),  K = run·clutter + (1 − run)/R.
    struct Terms
    {
        double log_in, log_out, log_wall, log_through, log_clutter, sd2, log_run, log_off_run;
        double log_alpha, log_beta, log_K, log_O, log_norm;
    };
    // log_run: the ray's membership of THIS door's wall run, fixed per frame at the prior (run_membership below).
    Terms terms(const UD& a, const Spread& sp, double log_run) const
    {
        const double sd2 = sp.var_d, sd = sp.sd, tau = sp.tau;                // depth: range noise + pose spread; jamb blur
        const double z   = (0.5 * w - std::hypot(a.u, tau)) / tau;           // |u| blurred by τ
        // Occluder: the return is anywhere between the origin and the near face, uniform in depth (soft edges
        // at the depth spread). Background: anywhere in range.
        const double near = -0.5 * t;
        const double log_occ = log_sigmoid((near - a.d) / sd) + log_sigmoid((a.d - a.d_o) / sd)
                               - std::log(std::hypot(near - a.d_o, sd));
        static const double log_bg = -std::log(kMaxRange), log_eps = std::log(kClutter), log_1m2e = std::log(1.0 - 2.0 * kClutter);
        Terms k;
        k.log_in      = log_sigmoid(z);
        k.log_out     = k.log_in - z;                                         // log σ(−z) = log σ(z) − z
        k.log_wall    = -0.5 * (a.d - near) * (a.d - near) / sd2 - sp.log_norm; // stops on the NEAR face
        k.log_through = log_sigmoid((a.d - 0.5 * t) / sd) + log_bg;           // beyond the far face
        k.log_clutter = log_eps + log_add_exp(log_occ, log_bg);
        k.sd2 = sd2; k.log_norm = sp.log_norm;
        k.log_run = log_run; k.log_off_run = log1m_exp(log_run);
        const double off = k.log_off_run + log_bg;
        k.log_alpha = log_run + log_1m2e + k.log_in;
        k.log_beta  = log_add_exp(log_run + log_add_exp(log_1m2e + k.log_out + k.log_wall, k.log_clutter), off);
        k.log_K     = log_add_exp(log_run + k.log_clutter, off);
        k.log_O     = log_run + log_1m2e + log_add_exp(k.log_in + k.log_through, k.log_out + k.log_wall);
        return k;
    }
    double log_gauss(double r, const Terms& k) const { return -0.5 * r * r / k.sd2 - k.log_norm; }
    // Every branch, on the run: (1 − 2ε)·[m·L_inside + (1 − m)·L_wall] + ε·L_occluder + ε·L_background;
    // off the run: the background alone (fix 2: no wall evidence).
    static double mix(const Terms& k, double log_inside) { return log_add_exp(k.log_alpha + log_inside, k.log_beta); }
    // The doorway structure's responsibility for this ray against the clutter components (visibility only), the
    // open/shut latent at its prior and δ averaged — per ray, which is what "is THIS beam explained by the door" asks.
    double responsibility(const Terms& k, const UD& a) const
    {
        double l_leaf = -std::numeric_limits<double>::infinity();
        for (const double dl : leaf_depths) l_leaf = log_add_exp(l_leaf, log_gauss(a.d - dl, k) - std::log(double(leaf_depths.size())));
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
        const double sd = sp.sd, tau = sp.tau;
        return log_sigmoid((a.u - u_min) / tau) + log_sigmoid((u_max - a.u) / tau) + log_sigmoid((a.d - a.d_o) / sd);
    }
    // OPEN with a swung leaf: (1 − 2ε)·[h·L_leaf + (1 − h)·(m·L_through + (1 − m)·L_wall)] + clutter, on the run.
    static double loglik_open_swung(const Terms& k, double log_h, double log_l_leaf)
    {
        static const double log_1m2e = std::log(1.0 - 2.0 * kClutter);
        return log_add_exp(log_add_exp(k.log_run + log_1m2e + log_h + log_l_leaf, log1m_exp(std::min(log_h, -1e-12)) + k.log_O), k.log_K);
    }
    double loglik_open(const Terms& k) const { return mix(k, k.log_through); }
    // ⚑ COMPUTE CUTOFF (not a model level): whether the hypothesis INSIDE the opening (the gap, or a shut leaf at any δ)
    //  can matter for this ray. When log m + the largest inside likelihood lies below the rest of its mixture by more
    //  than log 1e6, every inside hypothesis gives the ray the same value to 1e-6 and the δ loop is skipped.
    bool inside_negligible(const Terms& k) const
    {
        const double l_max = std::max(-k.log_norm, -std::log(kMaxRange));
        return k.log_alpha + l_max < k.log_beta + kLogNegligible;
    }
    double loglik_shut(const Terms& k, const UD& a, double delta) const { return mix(k, log_gauss(a.d - delta, k)); }
};

// The swung leaf as the model sees it (fix round 4): every CONFIGURATION hinge × swing × φ is one hypothesis with its
// prior weight, plus the "no leaf" branch; one configuration holds for the whole frame (one leaf), so they are
// marginalised per FRAME, like δ. φ runs over the door's whole range on a grid whose step moves the leaf's free edge by
// one range σ (Δφ = σ / w): finer cannot be resolved by a beam, coarser would alias the edge between nodes.
struct LeafCfg
{
    double hu = 0, du = 1, sy = -1, phi = 0, log_prior = 0, Dx = 1, Dy = 0;   // D = the leaf's unit direction
    int block = 0, k = 0;                                                     // hinge × swing block, φ node
};
struct LeafModel
{
    double len = 0;
    std::vector<LeafCfg> cfgs;                 // empty = no swung-leaf component (LeafPrior::known false)
    double log_prior_none = -std::numeric_limits<double>::infinity();   // the "no leaf in the model" branch
    // The configurations this stage's F marginalises: those carrying the posterior at the stage's start (see
    // select_leaf_configs); index -1 = the no-leaf branch.
    std::vector<int> active;
    std::vector<double> active_w;   // their posterior weights, normalised over the active set (select_leaf_configs)
};
// Where this ray's segment-line meets the leaf segment at angle φ: log membership (soft at the leaf's ends, the hit
// AHEAD of the origin) and the hit's signed depth in the ray's own convention (as UD::d).
struct LeafHit { double log_h = -std::numeric_limits<double>::infinity(), d_hit = 0; };
// ⚑ COMPUTE CUTOFF (kLogNegligible): a leaf membership below 1e-6 is treated as 0, i.e. a ray whose line passes
//   outside the leaf's reach by more than ~14 jamb blurs is scored by the plain open mixture. The bound is the
//   logistic's own (log σ(z) ≤ min(0, z)), so the test costs no transcendental and the leaf's likelihood is evaluated
//   only for rays whose line passes near the leaf.
// oa, ea: the ray's origin and endpoint in ANCHOR coordinates.
inline LeafHit leaf_hit(const Eigen::Vector2d& oa, const Eigen::Vector2d& ea, const LeafCfg& L, double len, double half_t,
                        double tau, double sd)
{
    const Eigen::Vector2d v = ea - oa;
    const Eigen::Vector2d D(L.Dx, L.Dy);
    // The leaf is a SLAB of thickness 2·kLeafHalfThick turning about a hinge on the jamb's SWING-side face
    // (hu, sy·t/2), the slab on the side n_off of its hinge line — toward the opening when swung 90°, into the wall's
    // thickness when shut — and a beam meets the face turned toward its origin.
    const Eigen::Vector2d n_off = (L.du * L.sy) * Eigen::Vector2d(L.Dy, -L.Dx);
    const Eigen::Vector2d Hc = Eigen::Vector2d(L.hu, L.sy * half_t) + kLeafHalfThick * n_off;   // the slab's centre line
    const double side = n_off.dot(oa - Hc) >= 0.0 ? 1.0 : -1.0;
    const Eigen::Vector2d H = Hc + side * kLeafHalfThick * n_off;
    const double den = v.x() * D.y() - v.y() * D.x();
    LeafHit h;
    if (std::abs(den) < 1e-12) return h;                       // parallel: it never meets the leaf
    const Eigen::Vector2d q = H - oa;
    const double t   = (q.x() * D.y() - q.y() * D.x()) / den;  // along the ray (1 = its endpoint)
    const double lam = (q.x() * v.y() - q.y() * v.x()) / den;  // along the leaf (0 = hinge, len = free edge)
    const double vlen = v.norm();
    const double z1 = lam / tau, z2 = (len - lam) / tau, z3 = t * vlen / sd;
    if (std::min(0.0, z1) + std::min(0.0, z2) + std::min(0.0, z3) < kLogNegligible) return h;   // see the cutoff
    h.log_h = log_sigmoid(z1) + log_sigmoid(z2) + log_sigmoid(z3);
    const double sgn = oa.y() <= 0 ? 1.0 : -1.0;
    h.d_hit = sgn * (oa.y() + t * v.y());
    return h;
}
// P(the line of sight origin → target is BLOCKED by configuration L's leaf) (Task 8, visibility only): the sight line
// meets the leaf's centre line between its hinge and free edge, ahead of the origin and before the target — each of
// the four soft at the jamb blur τ, as leaf_hit is. oa, target: anchor coordinates.
inline double leaf_occlusion(const Eigen::Vector2d& oa, const Eigen::Vector2d& target, const LeafCfg& L, double len, double half_t,
                             double tau)
{
    const Eigen::Vector2d v = target - oa;
    const Eigen::Vector2d D(L.Dx, L.Dy);
    const Eigen::Vector2d n_off = (L.du * L.sy) * Eigen::Vector2d(L.Dy, -L.Dx);
    const Eigen::Vector2d H = Eigen::Vector2d(L.hu, L.sy * half_t) + kLeafHalfThick * n_off;   // the slab's centre line
    const double den = v.x() * D.y() - v.y() * D.x();
    if (std::abs(den) < 1e-12) return 0.0;
    const Eigen::Vector2d q = H - oa;
    const double t   = (q.x() * D.y() - q.y() * D.x()) / den;
    const double lam = (q.x() * v.y() - q.y() * v.x()) / den;
    const double vl  = v.norm();
    return std::exp(log_sigmoid(lam / tau) + log_sigmoid((len - lam) / tau) + log_sigmoid(t * vl / tau) + log_sigmoid((1.0 - t) * vl / tau));
}
// The configurations and their prior weights from the door's belief (see LeafPrior).
inline LeafModel make_leaf_model(const LeafPrior& lp, double w)
{
    LeafModel L;
    if (not lp.known or w <= 0.0) return L;
    L.len = lp.len > 0.f ? double(lp.len) : w;
    const double phi_max = std::clamp(double(lp.phi_max), 0.0, M_PI);
    const int n_phi = 1 + std::max(1, int(std::ceil(phi_max / (kRangeSigma / L.len))));
    const double rho = std::clamp(double(lp.p_not_as_believed), 0.0, 1.0);
    const double ph = std::clamp(double(lp.p_hinge), 0.0, 1.0), ps = std::clamp(double(lp.p_swing), 0.0, 1.0);
    // φ weights under the belief, normalised over the grid (a σ below the grid step is the grid's resolution).
    std::vector<double> wphi(std::size_t(n_phi), 1.0);
    if (lp.phi_known)
    {
        const double step = phi_max / double(n_phi - 1);
        const double s = std::max(double(lp.phi_sigma), 0.5 * step);
        for (int k = 0; k < n_phi; ++k) { const double z = (step * k - double(lp.phi)) / s; wphi[std::size_t(k)] = std::exp(-0.5 * z * z); }
    }
    double zphi = 0.0;
    for (const double v : wphi) zphi += v;
    if (not (zphi > 0.0)) { std::ranges::fill(wphi, 1.0); zphi = double(n_phi); }   // a belief outside the range: uniform
    const double n_cfg = 4.0 * double(n_phi);
    L.log_prior_none = rho > 0.0 ? std::log(0.5 * rho) : -std::numeric_limits<double>::infinity();
    for (const int hs : {1, -1})
        for (const int ss : {1, -1})
        {
            const double w_hs = (hs > 0 ? ph : 1.0 - ph) * (ss > 0 ? ps : 1.0 - ps);
            for (int k = 0; k < n_phi; ++k)
            {
                LeafCfg c;
                c.hu = hs * double(lp.hinge_u); c.du = hs * double(lp.dir_u); c.sy = ss * double(lp.swing_y);
                c.phi = phi_max * double(k) / double(n_phi - 1);
                c.Dx = c.du * std::cos(c.phi); c.Dy = c.sy * std::sin(c.phi);
                c.block = (hs > 0 ? 0 : 2) + (ss > 0 ? 0 : 1); c.k = k;
                const double p = (1.0 - rho) * w_hs * wphi[std::size_t(k)] / zphi + 0.5 * rho / n_cfg;
                if (p <= 0.0) continue;
                c.log_prior = std::log(p);
                L.cfgs.push_back(c);
            }
        }
    return L;
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

// ⚑ COMPUTE BUDGET (not a model level): at most this many leaf configurations are marginalised inside one stage's F —
//   the most probable at the stage's start, which with hundreds of leaf returns carry all but ~1e-6 of the posterior
//   (select_leaf_configs). A flat posterior (no beam reaches any leaf) keeps the first ones by prior weight, all of
//   which then score alike.
constexpr int kMaxLeafConfigs = 4;

struct Problem
{
    V3d xbar; M3d Pinv;
    Model model;
    std::vector<Ray2> rays;            // the frame's rays (after the ray budget: see budget_rays)
    const Mask* mask = nullptr;
    std::vector<Spread> spread;        // per-ray predictive spread, fixed for one iteration (empty = sensor only)
    // Per-ray log membership of the door's wall run, decided ONCE per frame at the prior mean under the prior's spread.
    // ★Fixed, not re-evaluated at x: a run that moved with the hypothesis made its two ENDS lateral evidence — the shut
    //  door "localised" along its wall to the room polygon's wall ends (σ_lat 0.30 → 0.04 on a flat wall) — and a
    //  template that can shed inconvenient rays by sliding is a template that slides. Which rays belong to this wall
    //  is a data-association decision taken under the prior; the pose is then inferred from those rays only.
    std::vector<double> run_log;
    std::vector<double> wt;            // per-ray weight: how many of the frame's beams this ray stands for (budget_rays)
    double mask_var = 0;
    double beta_lidar = 1.0, beta_cam = 1.0;   // visibility tempering (fix 1): P(door in view) per sensor
    LeafModel leaf;                            // the open branch's swung leaf configurations (fix round 4)
    double az_step = 0.0;                      // the frame's azimuth step (rad), measured from its rays (stage_rays); 0 = unknown

    // Fix the predictive spreads at the linearisation point x under belief P (P = 0: the measurement model alone).
    void linearise(const V3d& x, const M3d& P)
    {
        spread.assign(rays.size(), Spread{});
        if (not P.isZero())
            for (std::size_t i = 0; i < rays.size(); ++i) spread[i] = predictive_spread(rays[i], x, P);
        if (mask != nullptr and mask->valid) mask_var = mask->predictive_var(x, P);
    }

    // One ray at x, in anchor coordinates (cos/sin of the yaw hoisted out of the ray loop).
    struct RayAt { Eigen::Vector2d oa, ea; UD a; };
    static RayAt at(const Ray2& r, const V3d& x, double cz, double sz)
    {
        const Eigen::Vector2d qo = r.o - x.head<2>(), qe = r.e - x.head<2>();
        RayAt q{{cz * qo.x() + sz * qo.y(), -sz * qo.x() + cz * qo.y()}, {cz * qe.x() + sz * qe.y(), -sz * qe.x() + cz * qe.y()}, {}};
        const double sgn = q.oa.y() <= 0 ? 1.0 : -1.0;
        const double dy = q.ea.y() - q.oa.y();
        const double u = std::abs(dy) > 1e-12 ? q.oa.x() - q.oa.y() * (q.ea.x() - q.oa.x()) / dy : std::copysign(1e9, q.ea.x() - q.oa.x());
        q.a = {u, sgn * q.ea.y(), sgn * q.oa.y()};
        return q;
    }
    // The open branch with configuration c's leaf, minus the plain open mixture, for one ray (0 when its line misses
    // the leaf by more than the compute cutoff — see leaf_hit).
    double leaf_delta(const RayAt& r, const Model::Terms& k, std::size_t i, const LeafCfg& c, double l_open) const
    {
        const LeafHit h = leaf_hit(r.oa, r.ea, c, leaf.len, 0.5 * model.t, spread[i].tau, spread[i].sd);
        if (h.log_h == -std::numeric_limits<double>::infinity()) return 0.0;
        return Model::loglik_open_swung(k, h.log_h, model.log_gauss(r.a.d - h.d_hit, k)) - l_open;
    }

    // log p(rays | x): the open/shut latent, the shut leaf's depth δ and the active leaf configurations marginalised
    // per FRAME; each ray weighted by how many beams it stands for.
    double frame_loglik(const V3d& x) const
    {
        const std::size_t nd = model.leaf_depths.size(), na = leaf.active.size();
        double c_open = 0, c_shut = 0;                                   // the parts every hypothesis shares
        std::vector<double> d_shut(nd, 0.0), d_open(na, 0.0);           // the parts that differ
        const double cz = std::cos(x.z()), sz = std::sin(x.z());
        for (std::size_t i = 0; i < rays.size(); ++i)
        {
            const RayAt r = at(rays[i], x, cz, sz);
            const auto k = model.terms(r.a, spread[i], run_log[i]);
            const double wi = wt[i];
            const double l_open = model.loglik_open(k);
            c_open += wi * l_open;
            if (model.inside_negligible(k)) c_shut += wi * l_open;     // every δ (and the gap) scores this ray alike
            else
                for (std::size_t j = 0; j < nd; ++j) d_shut[j] += wi * model.loglik_shut(k, r.a, model.leaf_depths[j]);
            for (std::size_t c = 0; c < na; ++c)
                if (leaf.active[c] >= 0) d_open[c] += wi * leaf_delta(r, k, i, leaf.cfgs[std::size_t(leaf.active[c])], l_open);
        }
        double l_open = c_open;
        if (not leaf.cfgs.empty())
        {
            l_open = -std::numeric_limits<double>::infinity();
            for (std::size_t c = 0; c < na; ++c)
            {
                const double lp = leaf.active[c] < 0 ? leaf.log_prior_none : leaf.cfgs[std::size_t(leaf.active[c])].log_prior;
                l_open = log_add_exp(l_open, lp + c_open + d_open[c]);
            }
        }
        double l_shut = -std::numeric_limits<double>::infinity();   // E_δ over the uniform grid
        for (const double l : d_shut) l_shut = log_add_exp(l_shut, l - std::log(double(nd)));
        l_shut += c_shut;
        // The door is open or shut for the WHOLE frame: marginalise that latent at the frame level.
        return log_add_exp(std::log(kPriorOpen) + l_open, std::log(1.0 - kPriorOpen) + l_shut);
    }

    // ─── PER-RAY SCORES (Task 10) ────────────────────────────────────────────────────────────────────────────────
    // frame_loglik is a FRAME-level mixture over its hypotheses h (open × the active leaf configurations, shut × δ):
    //   log p(rays | x) = log Σ_h π_h exp(Σ_i w_i ℓ_i(h, x)).
    // Its gradient splits EXACTLY into per-ray terms (Fisher's identity): ∇ log p = Σ_i w_i Σ_h r_h ∇ℓ_i(h), r_h the frame's
    // posterior responsibility of h at x. ray_scores returns, per kept ray, the score of ONE of the beams it stands for,
    // tempered as F tempers it: s_i = β_lidar Σ_h r_h ∇ℓ_i(h). (The shut branch is scored exactly, without
    // inside_negligible's shortcut — the two agree to 1e-6 per ray.) Central differences at F's own 1e-3 step.
    std::vector<V3d> ray_scores(const V3d& x) const
    {
        const std::size_t n = rays.size(), nd = model.leaf_depths.size();
        const bool leafy = not leaf.cfgs.empty();
        const std::size_t no = leafy ? leaf.active.size() : 1, nh = no + nd;
        std::vector<double> lp(nh);
        for (std::size_t c = 0; c < no; ++c)
            lp[c] = std::log(kPriorOpen) + (leafy ? (leaf.active[c] < 0 ? leaf.log_prior_none : leaf.cfgs[std::size_t(leaf.active[c])].log_prior) : 0.0);
        for (std::size_t j = 0; j < nd; ++j) lp[no + j] = std::log(1.0 - kPriorOpen) - std::log(double(nd));
        auto per_ray = [&](const V3d& xx, std::size_t i, double* out)
        {
            const RayAt r = at(rays[i], xx, std::cos(xx.z()), std::sin(xx.z()));
            const auto k = model.terms(r.a, spread[i], run_log[i]);
            const double l_open = model.loglik_open(k);
            for (std::size_t c = 0; c < no; ++c)
                out[c] = l_open + ((leafy and leaf.active[c] >= 0) ? leaf_delta(r, k, i, leaf.cfgs[std::size_t(leaf.active[c])], l_open) : 0.0);
            for (std::size_t j = 0; j < nd; ++j) out[no + j] = model.loglik_shut(k, r.a, model.leaf_depths[j]);
            // −∞ − (−∞) inside a branch (a leaf the ray cannot meet) is that branch's impossibility, not a number.
            for (std::size_t h = 0; h < nh; ++h) if (std::isnan(out[h])) out[h] = -std::numeric_limits<double>::infinity();
        };
        std::vector<double> l0(n * nh);
        std::vector<double> Lh(lp);
        for (std::size_t i = 0; i < n; ++i)
        {
            per_ray(x, i, &l0[i * nh]);
            for (std::size_t h = 0; h < nh; ++h) Lh[h] += wt[i] * l0[i * nh + h];
        }
        double z = -std::numeric_limits<double>::infinity();
        for (const double v : Lh) z = log_add_exp(z, v);
        std::vector<double> resp(nh);
        for (std::size_t h = 0; h < nh; ++h) resp[h] = std::exp(Lh[h] - z);
        std::vector<V3d> s(n, V3d::Zero());
        std::vector<double> lpb(nh), lmb(nh);
        constexpr double h_ = 1e-3;
        for (std::size_t i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
            {
                V3d e = V3d::Zero(); e(a) = h_;
                per_ray(x + e, i, lpb.data());
                per_ray(x - e, i, lmb.data());
                double g = 0.0;
                for (std::size_t h = 0; h < nh; ++h)
                    if (const double d = lpb[h] - lmb[h]; resp[h] > 0.0 and std::isfinite(d)) g += resp[h] * d / (2.0 * h_);
                s[i](a) = beta_lidar * g;
            }
        return s;
    }

    // Which leaf configurations carry the posterior at x (under the spreads staged): all of them scored once, the most
    // probable kept until they hold 1 − 1e-6 of it (at most kMaxLeafConfigs). A configuration none of whose beams
    // reach its leaf scores like the no-leaf branch; the data separate the rest by tens of nats per frame.
    // global: score every configuration (the frame's first stage, under the prior's wide spreads, where the likelihood
    // in φ is broad enough that the whole grid can be searched without aliasing); otherwise only the neighbours (±2 φ
    // nodes, same hinge and swing) of the configurations active so far, and the no-leaf branch — a later stage refines
    // what the first found (fix round 4: COST, the full grid is ~130 configurations).
    void select_leaf_configs(const V3d& x, bool global)
    {
        if (leaf.cfgs.empty()) { leaf.active.clear(); leaf.active_w.clear(); return; }
        std::vector<int> cand;
        for (std::size_t c = 0; c < leaf.cfgs.size(); ++c)
        {
            bool take = global;
            for (const int a : leaf.active)
                if (not take and a >= 0 and leaf.cfgs[std::size_t(a)].block == leaf.cfgs[c].block
                    and std::abs(leaf.cfgs[std::size_t(a)].k - leaf.cfgs[c].k) <= 2)
                    take = true;
            if (take) cand.push_back(int(c));
        }
        leaf.active.clear();
        leaf.active_w.clear();
        const std::size_t nc = cand.size();
        std::vector<double> d(nc, 0.0);
        const double cz = std::cos(x.z()), sz = std::sin(x.z());
        for (std::size_t i = 0; i < rays.size(); ++i)
        {
            const RayAt r = at(rays[i], x, cz, sz);
            const auto k = model.terms(r.a, spread[i], run_log[i]);
            const double l_open = model.loglik_open(k);
            for (std::size_t c = 0; c < nc; ++c) d[c] += wt[i] * leaf_delta(r, k, i, leaf.cfgs[std::size_t(cand[c])], l_open);
        }
        std::vector<std::pair<double, int>> lw;
        lw.reserve(nc + 1);
        lw.emplace_back(leaf.log_prior_none, -1);
        for (std::size_t c = 0; c < nc; ++c) lw.emplace_back(leaf.cfgs[std::size_t(cand[c])].log_prior + d[c], cand[c]);
        std::ranges::stable_sort(lw, [](const auto& p, const auto& q) { return p.first > q.first; });
        double z = -std::numeric_limits<double>::infinity();
        for (const auto& [l, c] : lw) z = log_add_exp(z, l);
        double kept = 0.0;
        for (const auto& [l, c] : lw)
        {
            if (int(leaf.active.size()) >= kMaxLeafConfigs or kept >= 1.0 - 1e-6) break;
            leaf.active.push_back(c);
            leaf.active_w.push_back(std::exp(l - z));
            kept += std::exp(l - z);
        }
        for (double& w : leaf.active_w) w /= std::max(kept, 1e-300);
    }
    double F(const V3d& x) const
    {
        V3d dx = x - xbar;
        dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
        double f = 0.5 * dx.dot(Pinv * dx);
        if (not rays.empty()) f -= beta_lidar * frame_loglik(x);
        if (mask != nullptr and mask->valid) f += beta_cam * Mask::neg_loglik(mask->centroid_u(x), mask_var);
        return f;
    }
    // Finite differences. Steps: 1 mm / 1 mrad — well below τ = 5 cm, the shortest scale in F. Gradient and diagonal
    // central; the off-diagonal from the one extra corner F(x + eᵢ + eⱼ) (fix round 4, COST: 10 evaluations instead of
    // 19 — its O(h) error is h × F's third derivative, 1e-3 against a 5 cm scale).
    void derivatives(const V3d& x, V3d& g, M3d& H, double* f_at_x = nullptr) const
    {
        const V3d h(1e-3, 1e-3, 1e-3);
        const double f0 = F(x);
        if (f_at_x != nullptr) *f_at_x = f0;
        V3d fp, fm;
        for (int i = 0; i < 3; ++i)
        {
            V3d ei = V3d::Zero(); ei(i) = h(i);
            fp(i) = F(x + ei); fm(i) = F(x - ei);
            g(i) = (fp(i) - fm(i)) / (2.0 * h(i));
            H(i, i) = (fp(i) - 2.0 * f0 + fm(i)) / (h(i) * h(i));
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < i; ++j)
            {
                V3d e = V3d::Zero(); e(i) = h(i); e(j) = h(j);
                H(i, j) = H(j, i) = (F(x + e) - fp(i) - fp(j) + f0) / (h(i) * h(j));
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
//   be clutter), score = coverage · C/N. Task 8: per SIDE, factor = 1 − Σ(1 − m)·[vis·(1 − coverage) + coverage·
//   (1 − C/N)] / Σ(1 − m), vis = the cell's PREDICTED visibility (P(detect) from the angle its face subtends over the
//   frame's measured azimuth step, × P(no predicted leaf in the line of sight)); the wall factor is the geometric mean
//   of the two sides. A FLOOR return never ends in a cell (view_visit), and an ending beam's path through the structure
//   band counts as passing only if the beam started inside it.
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
//  crossing_with_leaf_at_moderate_pi, true_pose_outscores_wrong_wall). Hence the posterior. (5) Task 8, try65: at the
//  CORRECT pose p_lidar read 0.44–0.47 live and 0.43–0.49 reproduced (the Webots door, the live caller configuration):
//  the leaf swung 90° toward the robot HID the wall on its hinge side, those cells scored 0, and the wall factor sat at
//  0.46 — half the doorway "unseen". Now 0.82–0.92 on the approach (test try65_webots_door_crossing).
// ⚠ ACCEPTED LIMIT: a SHUT door offset ALONG its wall is unobservable by LiDAR — a flush leaf is the wall — so p stays
//   high there and the anchor's lateral error is not corrected by the rays. Only the camera's door mask can resolve
//   it, and the crossing needs the door open anyway (an open door's jambs pull the anchor into place).
struct LidarView { double v = 0, o = 0, p = 0, open_cover = 0, p_door = 0, wall = 0; };   // p = open_cover · p_door · wall
// Under the spreads pb.linearise(x, P) staged (the prior's). pi_open = P(open) from the door's belief.
// The doorway region's cells: |u| < w in steps of the model's lateral resolution.
struct ViewGrid
{
    double w = 0;
    int n = 1;
    double cell = 0;
    explicit ViewGrid(double w_) : w(w_), n(std::max(1, int(std::ceil(2.0 * w_ / kRangeSigma)))), cell(2.0 * w_ / double(n)) {}
    int of(double u) const { return std::clamp(int(std::floor((u + w) / cell)), 0, n - 1); }
    double centre(int cc) const { return -w + (cc + 0.5) * cell; }
};
// The cells ONE beam observes (oa, ea: origin and endpoint in anchor coordinates; sd2: its depth spread; se2: its
// endpoint's along-wall variance under the pose), reported as observe(cell, weight, ends). The geometry described above; returns whether any cell was
// observed (a beam that observes none says nothing about the doorway's visibility).
// ★Task 8 — a FLOOR return (floor = true) never ENDS in a cell: the floor under a wall is hidden by the wall, so a beam
//  that reached the floor inside the slab PASSED every cell it crossed there, its last one included. Projected to the
//  floor plane and scored as an ending, a floor return inside a wall cell read as "the wall seen as predicted"; a pose
//  dragged 90° onto the swung leaf collected 35 % consistent endings on a wall that was not there from them.
template <class Observe>
bool view_visit(const Model& model, const ViewGrid& G, const Eigen::Vector2d& oa, const Eigen::Vector2d& ea, double sd2,
                double se2, bool floor, Observe&& observe, double sd2_path = -1.0)
{
    const double w = G.w;
    bool any = false;
    auto obs = [&](int cc, double g, bool ends) { any = true; observe(cc, g, ends); };
    // The segment's part inside the slab |y| ≤ hs, as a parameter interval [t0, t1] ⊂ [0, 1].
    const double hs = 0.5 * model.t + std::sqrt(sd2);
    const double dy = ea.y() - oa.y();
    double t0 = 0.0, t1 = 1.0;
    if (std::abs(dy) < 1e-12) { if (std::abs(oa.y()) > hs) return false; }
    else
    {
        double ta = (-hs - oa.y()) / dy, tb = (hs - oa.y()) / dy;
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(0.0, ta); t1 = std::min(1.0, tb);
        if (t0 > t1)
        {
            if (ta < 0.0 or floor) return false;                           // the slab is behind the beam's origin / floor in front
            const double u_in = oa.x() + ta * (ea.x() - oa.x());
            if (std::abs(u_in) >= w) return false;
            const double gap = std::abs(ea.y()) - hs;                      // stopped SHORT: "ends" there, softly
            obs(G.of(u_in), std::exp(-0.5 * gap * gap / sd2), true);
            return any;
        }
    }
    const bool ends_inside = t1 >= 1.0 and not floor;                       // the endpoint lies in the (outer) slab
    // near = the segment ran in the depth band beside the structure, not inside it: it observes the OPENING there (a gap is
    // seen free from just in front of it) but not a wall (passing 3 cm in front of a wall is not passing through it), so
    // each cell's observation is weighted by its opening membership m — continuous, the same m lidar_view scores with.
    auto pass_cells = [&](double pa_, double pb_, bool near = false)
    {
        if (pa_ > pb_) return;
        const double ua = oa.x() + pa_ * (ea.x() - oa.x()), ub = oa.x() + pb_ * (ea.x() - oa.x());
        const double lo = std::min(ua, ub), hi = std::max(ua, ub);
        if (hi > -w and lo < w)
            for (int cc = G.of(std::max(lo, -w)); cc <= G.of(std::min(hi, w)); ++cc)
                obs(cc, near ? std::exp(log_sigmoid((0.5 * model.w - std::abs(G.centre(cc))) / kRangeSigma)) : 1.0, false);
    };
    if (ends_inside)
    {
        if (std::abs(ea.x()) < w) obs(G.of(ea.x()), 1.0, true);
        // Its path before the return, where it ran INSIDE the structure's own band |y| ≤ t/2 (along the opening from
        // inside the doorway, say), passed free space — except its last range-σ, which is the return's own noise.
        // The depth-spread band in front of a face is NOT counted: a beam ending on a wall face crosses it, and that
        // says nothing about absence.
        const double hc = 0.5 * model.t;
        const double len = (ea - oa).norm();
        const double t_stop = len > kRangeSigma ? 1.0 - kRangeSigma / len : 0.0;
        // ★Task 8: only a beam that STARTED inside the band (the sensor in the doorway) has such a path. A beam that
        //  ENTERED the band from in front and ended in it was approaching its own return: on the try65 door (frame 20 cm
        //  deep, the model's wall 5 cm thick) a grazing beam ending on the real wall face 1.5 cm behind the model's near
        //  face ran ~9 cm "inside the wall" by the model, and those passes read the walls as absent at the doorway's mouth.
        if (std::abs(oa.y()) <= hc)
        {
            if (std::abs(dy) < 1e-12) pass_cells(0.0, t_stop);
            else pass_cells(0.0, std::min(t_stop, std::max((-hc - oa.y()) / dy, (hc - oa.y()) / dy)));
        }
        // A return ON A JAMB FACE observes the wall's END: the wall cells beside that jamb.
        const double tau = std::sqrt(kRangeSigma * kRangeSigma + se2);
        if (std::abs(std::abs(ea.x()) - 0.5 * w) <= tau)
            for (int cc = 0; cc < G.n; ++cc)
            {
                const double uc = G.centre(cc);
                if (std::abs(uc) > 0.5 * w and (uc > 0) == (ea.x() > 0)) obs(cc, 1.0, true);
            }
    }
    else
    {
        // Went THROUGH the slab: every cell it traversed INSIDE THE STRUCTURE was free. ★Task 9 fix round 1 (try67, the
        // mouth): "inside" is |y| ≤ t/2 + the pose's own depth σ (sd2_path), NOT the endpoint's depth spread hs, whose
        // range-noise part (kRangeSigma = 5 cm) describes where a RETURN lands, not where a beam's PATH runs. With hs, a
        // sensor 3.5 cm in front of the near face (the robot at the doorway's mouth) started every beam INSIDE the slab,
        // and each beam running sideways in front of the wall "passed through" it: the walls factor fell 0.83 → 0.48 at a
        // pose 2–4 cm off the truth.
        const double hp = 0.5 * model.t + std::sqrt(std::max(0.0, sd2_path >= 0.0 ? sd2_path : sd2 - kRangeSigma * kRangeSigma));
        double p0 = t0, p1 = t1;
        if (std::abs(dy) < 1e-12) { if (std::abs(oa.y()) > hp) p1 = p0 - 1.0; }
        else
        {
            double pa_ = (-hp - oa.y()) / dy, pb_ = (hp - oa.y()) / dy;
            if (pa_ > pb_) std::swap(pa_, pb_);
            p0 = std::max(t0, pa_); p1 = std::min(t1, pb_);
        }
        if (p0 > p1) pass_cells(t0, t1, true);            // only ever in the band beside the structure
        else { pass_cells(p0, p1); pass_cells(t0, p0, true); pass_cells(p1, t1, true); }
    }
    return any;
}

inline LidarView lidar_view(const Problem& pb, const V3d& x, double pi_open)
{
    LidarView lv;
    const double w = pb.model.w;
    const ViewGrid G(w);
    const std::size_t nc = std::size_t(G.n);
    std::vector<double> N(nc, 0.0), C(nc, 0.0), m(nc, 0.0);
    const double lambda = std::log((1.0 - 2.0 * kClutter) / (2.0 * kClutter));
    double log_r = 0.0;   // log L(open) / L(wall) over the opening's observations
    for (int cc = 0; cc < G.n; ++cc)
        m[std::size_t(cc)] = std::exp(log_sigmoid((0.5 * w - std::abs(G.centre(cc))) / kRangeSigma));
    const double cz = std::cos(x.z()), sz = std::sin(x.z());
    // ─── PREDICTED VISIBILITY of each cell (Task 8) = P(detect) × P(no leaf stands between the sensor and the cell), the
    //     latter under the leaf configurations this frame's beams hold (select_leaf_configs; no leaf model ⇒ 1). A wall cell
    //     the door's OWN swung leaf hides, or that the scanner's beams are not expected to reach (grazing, far), is not
    //     expected to be observed, so its silence is not "the doorway is not seen": it does not count as MISSED (below).
    //     It cannot excuse a contradiction: a cell some beam PASSED through is scored by its beams, hidden or not.
    std::vector<double> vis(nc, 1.0);
    if (not pb.rays.empty())
    {
        const Eigen::Vector2d qo = pb.rays.front().o - x.head<2>();
        const Eigen::Vector2d oa(cz * qo.x() + sz * qo.y(), -sz * qo.x() + cz * qo.y());   // the LiDAR origin, anchor coords
        const double side = oa.y() <= 0.0 ? -1.0 : 1.0;
        for (int cc = 0; cc < G.n; ++cc)
        {
            // What the cell shows the sensor: the wall face turned toward it, one cell long.
            const double y_face = side * 0.5 * pb.model.t;
            const Eigen::Vector2d a(G.centre(cc) - 0.5 * G.cell, y_face), b(G.centre(cc) + 0.5 * G.cell, y_face);
            // P(detect): the beams the scanner puts on the face, its subtended angle over the measured azimuth step —
            // a face seen at grazing incidence, or far, is expected to catch none (Poisson, mean Δθ / step).
            double v = 1.0;
            if (pb.az_step > 0.0)
            {
                const Eigen::Vector2d da = a - oa, db = b - oa;
                const double dtheta = std::abs(std::atan2(da.x() * db.y() - da.y() * db.x(), da.dot(db)));
                v = -std::expm1(-dtheta / pb.az_step);
            }
            // ...times P(no swung leaf stands in the way), under the configurations this frame's beams hold.
            if (not pb.leaf.active.empty())
            {
                double lv_ = 0.0;
                const Eigen::Vector2d target(G.centre(cc), y_face);
                for (std::size_t q = 0; q < pb.leaf.active.size(); ++q)
                    lv_ += pb.leaf.active_w[q] * (pb.leaf.active[q] < 0 ? 1.0
                           : 1.0 - leaf_occlusion(oa, target, pb.leaf.cfgs[std::size_t(pb.leaf.active[q])], pb.leaf.len, 0.5 * pb.model.t, kRangeSigma));
                v *= lv_;
            }
            vis[std::size_t(cc)] = v;
        }
    }
    for (std::size_t i = 0; i < pb.rays.size(); ++i)
    {
        const auto r = Problem::at(pb.rays[i], x, cz, sz);
        const auto k = pb.model.terms(r.a, pb.spread[i], pb.run_log[i]);
        const double wi = pb.wt[i];
        // The doorway NEIGHBOURHOOD (visibility only): the opening and half a width of wall each side, |u| < w.
        const double log_nbhd = log_sigmoid((w - std::hypot(r.a.u, pb.spread[i].tau)) / pb.spread[i].tau);
        const double ne = std::exp(log_nbhd + k.log_run);
        lv.v += wi * ne;
        lv.o += wi * ne * pb.model.responsibility(k, r.a);
        // Each observation counts for the beams this ray stands for (budget_rays): N and log r are SUMS over beams.
        view_visit(pb.model, G, r.oa, r.ea, k.sd2, pb.spread[i].se2, pb.rays[i].floor, [&](int cc, double g, bool ends)
        {
            const double mm = m[std::size_t(cc)];
            const double kappa = ends ? 1.0 - mm : mm;   // the WALL hypothesis' consistency (wall factor)
            N[std::size_t(cc)] += wi * g;
            C[std::size_t(cc)] += wi * g * kappa;
            log_r += wi * g * mm * (ends ? -lambda : lambda);
        }, pb.spread[i].sd2);
    }
    // THE WALLS, one per side of the opening (Task 8). Per cell: score = coverage 1 − (2ε)^N times consistency C/N (as
    // before); a side's factor is 1 − (its MISSED mass)/(its mass), the missed mass Σ (1 − m)·vis·(1 − score): a cell the
    // predicted leaf hides is not missed, a cell seen as predicted is not missed, a visible cell nobody saw or a beam
    // passed through is. With every cell visible this is the side's mean score, the pre-Task-8 wall factor.
    // The doorway needs BOTH walls: the factor is the GEOMETRIC mean of the sides — equal sides give the side's value (the
    // old mean), one contradicted side pulls the pair down by its own factor. ★The arithmetic mean gave a pose dragged 90°
    // onto the swung leaf (one side all passes, the other all endings) 0.5–0.55, on the publication level: 24 wrong frames
    // published (dragged_posterior_not_published). The plain PRODUCT ruled those out too but squared every side
    // (0.92 → 0.85) and failed the leaf crossing (0.44); the minimum failed it at 0.55 — both measured.
    // ★Task 9: a wall cell is a cell of THIS DOOR'S WALL RUN. The doorway neighbourhood (|u| < w) can reach past the run's
    //  end (a room corner 0.64 m from the opening's centre, try66 leg 1): there the model predicts NO wall, so a beam that
    //  passes the cell is consistent and a cell nobody saw is not missed. Each cell's wall mass is (1 − m)·e(u), e the run
    //  membership at the cell (Model::run_membership's ends, soft at the model's lateral resolution). ★Unweighted, the +u
    //  side of try66's door counted 0.36 m of "visible wall nobody saw" beyond the corner and its factor fell to 0.2 at the
    //  TRUE pose (p_view 0.42 at s = −2 m, door open).
    double so = 0, wo = 0, miss[2] = {0, 0}, mass[2] = {0, 0};
    for (std::size_t c = 0; c < nc; ++c)
    {
        const double cover = 1.0 - std::pow(2.0 * kClutter, N[c]);
        so += m[c] * cover; wo += m[c];
        const double consistent = N[c] > 0 ? C[c] / N[c] : 1.0;
        const double uc = G.centre(int(c));
        const int side = uc < 0.0 ? 0 : 1;
        const double wall = (1.0 - m[c]) * std::exp(log_sigmoid((uc - pb.model.u_min) / kRangeSigma)
                                                    + log_sigmoid((pb.model.u_max - uc) / kRangeSigma));
        miss[side] += wall * (vis[c] * (1.0 - cover) + cover * (1.0 - consistent));
        mass[side] += wall;
    }
    // P(H_door | beams) = A / (A + 1), A = π·r + 1 − π: in LOG form, log A = log(π·e^{log r} + (1 − π)), so it can neither
    // overflow nor divide 0 by 0. ★Task 9 fix round 1 (try67): the old form divided by (π + (1 − π)·e^{−log r} + e^{−log r}),
    //  which is 0/0 = NaN at π = 0 (the door's own belief read it SHUT at the mouth) with log r ≈ 10⁴ (thousands of passes):
    //  the NaN tempered F, the posterior came out NaN, and the anchor stayed NaN — unpublished — for the rest of the run.
    {
        const double ninf = -std::numeric_limits<double>::infinity();
        const double pi = std::clamp(pi_open, 0.0, 1.0);
        const double log_A = log_add_exp(pi > 0.0 ? std::log(pi) + log_r : ninf, pi < 1.0 ? std::log1p(-pi) : ninf);
        lv.p_door = std::exp(log_sigmoid(log_A));   // A / (A + 1)
    }
    lv.open_cover = wo > 0 ? so / wo : 0.0;
    {
        const double a = std::max(0.0, mass[0] > 0 ? 1.0 - miss[0] / mass[0] : 1.0), b = std::max(0.0, mass[1] > 0 ? 1.0 - miss[1] / mass[1] : 1.0);
        lv.wall = std::sqrt(a * b);
    }
    lv.p = lv.open_cover * lv.p_door * lv.wall;
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

// ─── THE RAY BUDGET (fix round 4) ───────────────────────────────────────────────────────────────────────────────
// ⚑ FLAGGED COMPUTE BUDGET, not a model level: at most kRayBudget rays enter F per update. Within 2 m of the opening a
//   3-D bpearl frame is 5,000–8,000 beams; all of them cost 1.0–1.9 s per update at -O2 on door_concept's 100 ms main
//   thread (round-4 review). The beams of one azimuth are the dome's elevation layers, which in the floor plane lie on
//   ONE line: they are near-copies, not independent views of the doorway.
//   STRATIFIED: strata = azimuth (kAzBin) × the doorway cell the beam's line crosses (ViewGrid; one more stratum for
//   "crosses elsewhere"), so the jambs, the opening and every bearing keep their share however dense the dome is.
//   Quotas by water-filling (every stratum gets min(n_s, L), the largest L that fits; the remainder one more each,
//   spread by stride over the strata in key order). DETERMINISTIC: within a stratum the rays are ordered by range and
//   taken at evenly spaced positions — no RNG, the same frame gives the same subset on every run.
//   WEIGHTED: each kept ray stands for n_s / q_s beams of its stratum (×(strata in the pool / strata sampled) when the
//   budget cannot reach every stratum), so log p(rays | x) and the visibility's beam counts keep the full frame's
//   scale (a Horvitz-Thompson estimate), and the frame's common mode still caps what they claim.
constexpr std::size_t kRayBudget = 400;
constexpr double      kAzBin     = 2.0 * M_PI / 180.0;
struct Budget { std::vector<std::size_t> idx; std::vector<double> wt; };
inline Budget budget_rays(const std::vector<Ray2>& rays, const std::vector<std::size_t>& candidates, const V3d& x, double w,
                          std::size_t budget = kRayBudget)
{
    Budget B;
    if (candidates.size() <= budget)
    {
        B.idx = candidates;
        B.wt.assign(candidates.size(), 1.0);
        return B;
    }
    const ViewGrid G(w);
    const double cz = std::cos(x.z()), sz = std::sin(x.z());
    std::map<std::int64_t, std::vector<std::size_t>> strata;
    for (const std::size_t i : candidates)
    {
        const Eigen::Vector2d v = rays[i].e - rays[i].o;
        const auto az = std::int64_t(std::floor((std::atan2(v.y(), v.x()) + M_PI) / kAzBin));
        const UD a = Problem::at(rays[i], x, cz, sz).a;
        const int cell = (a.d > a.d_o and std::abs(a.u) < w) ? 1 + G.of(a.u) : 0;
        strata[az * (G.n + 1) + cell].push_back(i);
    }
    std::vector<std::vector<std::size_t>*> S;
    S.reserve(strata.size());
    for (auto& [key, v] : strata)
    {
        std::ranges::stable_sort(v, [&](std::size_t p, std::size_t q)
                                 { return (rays[p].e - rays[p].o).squaredNorm() < (rays[q].e - rays[q].o).squaredNorm(); });
        S.push_back(&v);
    }
    // Water level L: the largest with Σ min(n_s, L) ≤ budget.
    auto filled = [&](std::size_t L) { std::size_t t = 0; for (const auto* v : S) t += std::min(v->size(), L); return t; };
    std::size_t lo = 0, hi = budget;
    while (lo < hi) { const std::size_t mid = (lo + hi + 1) / 2; if (filled(mid) <= budget) lo = mid; else hi = mid - 1; }
    const std::size_t L = lo;
    std::vector<std::size_t> pool;   // strata with rays left above the level
    for (std::size_t k = 0; k < S.size(); ++k) if (S[k]->size() > L) pool.push_back(k);
    const std::size_t R = std::min(budget - filled(L), pool.size());
    std::vector<std::size_t> q(S.size());
    for (std::size_t k = 0; k < S.size(); ++k) q[k] = std::min(S[k]->size(), L);
    for (std::size_t j = 0; j < R; ++j) ++q[pool[(2 * j + 1) * pool.size() / (2 * R)]];
    // Below a level of one some strata go unsampled: the sampled ones of the pool then also stand for those.
    const double ht = (L == 0 and R > 0) ? double(pool.size()) / double(R) : 1.0;
    for (std::size_t k = 0; k < S.size(); ++k)
    {
        const auto& v = *S[k];
        if (q[k] == 0) continue;
        const double wk = double(v.size()) / double(q[k]) * (L == 0 ? ht : 1.0);
        for (std::size_t j = 0; j < q[k]; ++j)
        {
            B.idx.push_back(v[(2 * j + 1) * v.size() / (2 * q[k])]);
            B.wt.push_back(wk);
        }
    }
    return B;
}

// ─── One frame's inputs, shared by the update and by door_view ─────────────────────────────────────────────────
// Mask: the spread needs two points; one point has no measured σ and claims nothing. σ_m is measured along the wall of
// the pose x0 (the spread is a property of the mask, not of the hypothesis).
inline Mask make_mask(const std::vector<Eigen::Vector3f>& mask_pts_robot, const V3d& x0)
{
    Mask mask;
    if (mask_pts_robot.size() < 2) return mask;
    for (const auto& p : mask_pts_robot) mask.pts.push_back(p.head<2>().cast<double>());
    const double c = std::cos(x0.z()), s = std::sin(x0.z());
    double mu = 0, m2 = 0;
    for (const auto& q : mask.pts) mu += c * (q.x() - x0.x()) + s * (q.y() - x0.y());
    mu /= double(mask.pts.size());
    for (const auto& q : mask.pts) { const double du = c * (q.x() - x0.x()) + s * (q.y() - x0.y()) - mu; m2 += du * du; }
    const double spread2 = m2 / double(mask.pts.size() - 1);
    mask.sigma2 = spread2 / double(std::min<std::size_t>(mask.pts.size(), kMaskNCap));
    mask.valid = mask.sigma2 > 0;   // all points at one abscissa: no spread ⇒ no measured σ
    return mask;
}
inline std::vector<Ray2> make_rays(const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                                   float floor_below = -std::numeric_limits<float>::infinity())
{
    const Eigen::Vector2d o = lidar_origin_robot.head<2>().cast<double>();
    std::vector<Ray2> all_rays;
    all_rays.reserve(rays_robot.size());
    for (std::size_t i = 0; i < rays_robot.size(); ++i)
        if (const Eigen::Vector2d e = rays_robot[i].head<2>().cast<double>(); (e - o).squaredNorm() > 1e-12)   // a zero-length ray has no direction
            all_rays.push_back({o, e, rays_robot[i].z() < floor_below, std::uint32_t(i)});
    return all_rays;
}
// The scanner's azimuth step, MEASURED from the frame (Task 8): the median gap between its distinct beam azimuths. The
// elevation layers of one azimuth share it to float rounding (the same rigid transform moved every point), so azimuths
// closer than ⚑ 1e-5 rad (ARITHMETIC: float rounding of a direction at metres of range is ~1e-7 rad) are one azimuth.
inline double azimuth_step(const std::vector<Ray2>& rays)
{
    std::vector<double> az;
    az.reserve(rays.size());
    for (const auto& r : rays) { const Eigen::Vector2d v = r.e - r.o; az.push_back(std::atan2(v.y(), v.x())); }
    std::ranges::sort(az);
    std::vector<double> gaps;
    for (std::size_t i = 1; i < az.size(); ++i)
        if (const double g = az[i] - az[i - 1]; g > 1e-5) gaps.push_back(g);
    if (gaps.empty()) return 0.0;
    std::ranges::nth_element(gaps, gaps.begin() + std::ptrdiff_t(gaps.size() / 2));
    return gaps[gaps.size() / 2];
}
// The frame's rays staged on pb at belief (x, P): this door's wall RUN decided at x (fix 2) and the RAY BUDGET.
// A ray is a CANDIDATE when it is on this door's wall run (membership above 1e-9: below it the ray adds the
// background's x-independent constant to F to within 1e-9 relative — arithmetic, not modelling) OR it observes a
// doorway cell (view_visit: a beam travelling along the opening is off the run but is how the doorway is seen from
// inside it). Every other ray contributes nothing to F or to the visibility. The candidates are then cut to the
// budget by stratified, weighted, deterministic subsampling (budget_rays). Spreads: x's predictive ones under P.
inline void stage_rays(Problem& pb, const std::vector<Ray2>& all_rays, const V3d& x, const M3d& P)
{
    const ViewGrid G(pb.model.w);
    const double cz = std::cos(x.z()), sz = std::sin(x.z());
    std::vector<std::size_t> candidates;
    std::vector<Spread> sp_all(all_rays.size());
    std::vector<double> run_all(all_rays.size(), -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < all_rays.size(); ++i)
    {
        sp_all[i] = predictive_spread(all_rays[i], x, P);
        const auto r = Problem::at(all_rays[i], x, cz, sz);
        run_all[i] = pb.model.run_membership(r.a, sp_all[i]);
        if (run_all[i] > std::log(1e-9) or view_visit(pb.model, G, r.oa, r.ea, sp_all[i].var_d, sp_all[i].se2, all_rays[i].floor, [](int, double, bool) {}))
            candidates.push_back(i);
    }
    pb.az_step = azimuth_step(all_rays);
    const Budget bud = budget_rays(all_rays, candidates, x, pb.model.w);
    pb.rays.clear(); pb.run_log.clear(); pb.spread.clear();
    pb.rays.reserve(bud.idx.size());
    for (const std::size_t i : bud.idx) { pb.rays.push_back(all_rays[i]); pb.run_log.push_back(run_all[i]); pb.spread.push_back(sp_all[i]); }
    pb.wt = bud.wt;
    if (pb.mask != nullptr and pb.mask->valid) pb.mask_var = pb.mask->predictive_var(x, P);
}
// The same staging from an ALREADY-budgeted set and its weights (the frame record's replay, fix round 1): per ray the
// spread and run membership stage_rays computes for the rays it keeps, so the two stagings are identical.
inline void stage_given(Problem& pb, const std::vector<Ray2>& rays, const std::vector<double>& wt, double az_step, const V3d& x, const M3d& P)
{
    const double cz = std::cos(x.z()), sz = std::sin(x.z());
    pb.rays = rays; pb.wt = wt; pb.az_step = az_step;
    pb.run_log.clear(); pb.spread.clear();
    for (const auto& r : rays)
    {
        pb.spread.push_back(predictive_spread(r, x, P));
        pb.run_log.push_back(pb.model.run_membership(Problem::at(r, x, cz, sz).a, pb.spread.back()));
    }
    if (pb.mask != nullptr and pb.mask->valid) pb.mask_var = pb.mask->predictive_var(x, P);
}

// ─── THE CAMERA TERM (fix round 1): the mask SEEN AS PREDICTED, not only the jambs in the frustum ───────────────────
// p_cam = P(both jambs in the frustum) × the agreement of the mask with the predicted opening, as seen from the camera:
// the intersection-over-union of two azimuth intervals — the mask's extent (its points' mean azimuth ± √3·sd, the
// extent of a uniform spread with that sd) and the predicted jambs' azimuths. Azimuth, not depth: an open doorway's mask
// pixels lie at the far room's depth, but every one of them is seen THROUGH the opening. ★The frustum alone (Task 7)
// read ≈ 1 for any pose whose jambs fell in the zed's 110° field — a prior yawed 90°, or 1.2 m in free space in front of
// the wall — so a held or fresh mask frame published a wrong pose whatever the LiDAR said (test held_mask_wrong_poses).
inline double camera_term(const CameraFrustum& cam, double w, const V3d& x, const M3d& P, const std::vector<Eigen::Vector2d>& pts)
{
    if (pts.size() < 2) return 0.0;
    const Eigen::Vector2d o = cam.known ? Eigen::Vector2d(cam.origin.cast<double>()) : Eigen::Vector2d(Eigen::Vector2d::Zero());
    const Eigen::Vector2d f = cam.known ? Eigen::Vector2d(cam.forward.cast<double>().normalized()) : Eigen::Vector2d(0.0, 1.0);
    const Eigen::Vector2d right(f.y(), -f.x());
    auto az = [&](const Eigen::Vector2d& p) { const Eigen::Vector2d q = p - o; return std::atan2(q.dot(right), q.dot(f)); };
    double mu = 0.0, m2 = 0.0;
    for (const auto& p : pts) mu += az(p);
    mu /= double(pts.size());
    for (const auto& p : pts) { const double d = az(p) - mu; m2 += d * d; }
    const double half = std::sqrt(3.0 * m2 / double(pts.size() - 1));
    const Eigen::Vector2d a(std::cos(x.z()), std::sin(x.z()));
    const double j1 = az(x.head<2>() - 0.5 * w * a), j2 = az(x.head<2>() + 0.5 * w * a);
    const double plo = std::min(j1, j2), phi = std::max(j1, j2), mlo = mu - half, mhi = mu + half;
    const double inter = std::max(0.0, std::min(phi, mhi) - std::max(plo, mlo));
    const double uni = std::max(phi, mhi) - std::min(plo, mlo);
    return camera_view(cam, w, x, P) * (uni > 0.0 ? inter / uni : 0.0);
}
}  // namespace anchor_detail

// The budgeted rays one staging consumed (robot frame, z kept: floor returns are told by it), their Horvitz-Thompson
// weights, the frame's measured azimuth step and the PRE-budget count (fix round 1: what the frame record stores).
struct StagedRays { std::vector<Eigen::Vector3f> rays; std::vector<double> wt; std::uint32_t n_pre = 0; double az_step = 0.0; };
// The update's two stagings — at the prior (the fit) and at the posterior (door_view: what is published). record: filled
// by the update. replay: consumed INSTEAD of staging rays_robot (the record's replay; rays_robot is then ignored).
struct AnchorStaging { bool replay = false; StagedRays prior, post; };
namespace anchor_detail
{
inline StagedRays export_staged(const Problem& pb, const std::vector<Eigen::Vector3f>& src, std::size_t n_pre)
{
    StagedRays s;
    s.n_pre = std::uint32_t(n_pre); s.az_step = pb.az_step; s.wt = pb.wt;
    for (const auto& r : pb.rays) s.rays.push_back(src[r.src]);
    return s;
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

// ─── THE CAMERA'S VIEW EVIDENCE PERSISTS (Task 8, fix 1; fix round 1) ──────────────────────────────────────────────
// A cycle without a NEW mask frame is a cycle the camera did not report on, not a cycle it saw nothing: the mask stream
// runs slower than the LiDAR and each frame is consumed once. ★try65: p_cam read 0 on every cycle without a new frame, so
// p_view alternated between ≈ 1.00 and p_lidar ≈ 0.45 — 21 PAUSED/RESUMED transitions on a correct anchor.
// What is HELD is the last mask frame — its points, carried to the robot frame of now by the body's own motion
// (carry_held_mask, the odometry the anchor is predicted by, never the anchor's posterior) — while that frame is younger
// than the controller's staleness window (kAnchorStaleMs, passed in: no new constant). The camera TERM is recomputed every
// cycle at the CURRENT pose (camera_seen): ★fix round 1 — holding the number itself published a pose that LiDAR-only
// updates had since dragged, with the visibility of the pose the frame was taken at. A frame with no door slice holds no
// points: the camera looked and saw none (term 0).
struct CameraViewHold { std::vector<Eigen::Vector3f> mask; std::int64_t stamp_ms = 0; bool valid = false; };
inline void hold_mask_frame(CameraViewHold& h, const std::vector<Eigen::Vector3f>& pts, std::int64_t frame_stamp_ms)
{
    h.mask = pts; h.stamp_ms = frame_stamp_ms; h.valid = true;
}
// The robot moved by (dx, dy, dyaw) in its own frame since the last cycle: the held points move by the inverse (the same
// algebra as relative_pose.h move_mean).
inline void carry_held_mask(CameraViewHold& h, float dx, float dy, float dyaw)
{
    for (auto& p : h.mask)
    {
        const Eigen::Vector3f q = rc::interaction::move_mean(Eigen::Vector3f(p.x(), p.y(), 0.f), dx, dy, dyaw);
        p = {q.x(), q.y(), p.z()};
    }
}
// The camera term of pose `at` from robot-frame mask points (0 with fewer than two): frustum × agreement (camera_term).
inline float camera_seen(const CameraFrustum& cam, float w, const rc::interaction::RelPose& at, const std::vector<Eigen::Vector3f>& mask_pts)
{
    std::vector<Eigen::Vector2d> pts;
    for (const auto& p : mask_pts) pts.push_back(p.head<2>().cast<double>());
    return float(anchor_detail::camera_term(cam, double(w), at.x.cast<double>(), at.P.cast<double>(), pts));
}
// The held frame's camera term at `at`, while the frame is younger than window_ms; else 0.
inline float held_camera_view(const CameraViewHold& h, std::int64_t now_ms, std::int64_t window_ms, const CameraFrustum& cam, float w,
                              const rc::interaction::RelPose& at)
{
    return (h.valid and now_ms - h.stamp_ms < window_ms) ? camera_seen(cam, w, at, h.mask) : 0.f;
}
// P(door in view) from the two sensors, either of which may see it: 1 − (1 − p_lidar)(1 − p_cam).
inline float combine_view(float p_lidar, float p_cam) { return 1.f - (1.f - p_lidar) * (1.f - p_cam); }

// ─── P(door in view) OF A GIVEN POSE (fix round 4) ─────────────────────────────────────────────────────────────
// The visibility of the doorway as predicted by the belief `at` (its mean and its covariance's predictive spread), from
// one frame's rays (+ mask): the v_lidar / o_lidar / p_lidar / p_cam / p_view fields of AnchorEvidence (the rest stay
// at their defaults). update_door_anchor reports THIS function of its POSTERIOR — the pose it returns for publication
// — so the published p_view and the published pose always belong together.
inline AnchorEvidence door_view(const rc::interaction::RelPose& at, float width, float thickness, float wall_u_min, float wall_u_max,
                                const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                                const std::vector<Eigen::Vector3f>& mask_pts_robot, const CameraFrustum& cam = {},
                                float p_open_prior = 0.5f, const LeafPrior& leaf = {},
                                float floor_below = -std::numeric_limits<float>::infinity(),
                                const StagedRays* pre = nullptr, StagedRays* staged_out = nullptr)
{
    using namespace anchor_detail;
    AnchorEvidence ev;
    const V3d x = at.x.cast<double>();
    const M3d P = at.P.cast<double>();
    const Mask mask = make_mask(mask_pts_robot, x);
    // pre: an already-staged set (the record's replay) instead of staging rays_robot here.
    const std::vector<Ray2> all_rays = make_rays(lidar_origin_robot, pre != nullptr ? pre->rays : rays_robot, floor_below);
    if (all_rays.empty() and not mask.valid) return ev;
    Problem pb{x, P.inverse(), Model(width, thickness, wall_u_min, wall_u_max)};
    pb.mask = &mask;
    pb.leaf = make_leaf_model(leaf, pb.model.w);
    if (pre != nullptr) stage_given(pb, all_rays, pre->wt, pre->az_step, x, P);
    else
    {
        stage_rays(pb, all_rays, x, P);
        if (staged_out != nullptr) *staged_out = export_staged(pb, rays_robot, rays_robot.size());
    }
    ev.az_step = float(pb.az_step);
    pb.select_leaf_configs(x, true);   // the leaf configurations this frame's beams hold at x: who hides what (Task 8)
    const LidarView lv = lidar_view(pb, x, std::clamp(double(p_open_prior), 0.0, 1.0));
    const double pc = camera_term(cam, pb.model.w, x, P, mask.pts);
    ev.v_lidar = float(lv.v);
    ev.o_lidar = float(lv.o);
    ev.p_lidar = float(lv.p);
    ev.lidar_open = float(lv.open_cover); ev.lidar_door = float(lv.p_door); ev.lidar_wall = float(lv.wall);
    ev.p_cam   = mask.valid ? float(pc) : 0.f;
    ev.p_view  = combine_view(ev.p_lidar, ev.p_cam);
    return ev;
}

// ─── THE PERSISTENT BIAS (Task 10) ──────────────────────────────────────────────────────────────────────────────
// Every frame's rays share errors no number of rays can average out, and — the point of Task 10 — so does the NEXT
// frame's: 0.1 s later the same mount sees the same wall, and the same structure the model does not describe (the casing
// boards, a swung leaf off its believed hinge, the slot between wall and frame) pulls the fit the same way. Treated as
// fresh noise on every frame, the old per-frame Woodbury common mode was divided by √N over a chain: try66–69 published
// σ_yaw 0.1–0.2° against errors of 2–28° (median |err|/σ 10–43).
// The model: the rays measure x + b_k (+ the beams' independent noise, I_L⁻¹), with b_k = Γ_k ξ this view's systematic
// error and ξ ~ N(0, I) ONE latent for the whole passage (fully correlated from frame to frame; its magnitude Γ_k
// re-evaluated for the current view, bias_terms). The posterior's error is n + L ξ, n independent of ξ (covariance P_z,
// the part the beams' noise legitimately averages), and for ANY gain K
//   L_k = (I − K) J L_{k−1} + K Γ_k,   P_z ← (I − K) P_z (I − K)ᵀ + K I_L⁻¹ Kᵀ,   published P = P_z + L Lᵀ,
// J the odometry's motion Jacobian. K is the Schmidt (consider) gain, the one that minimises P with b in the innovation:
//   K = (P_z − L (Γ − L)ᵀ) (P_z + (Γ − L)(Γ − L)ᵀ + I_L⁻¹)⁻¹,
// evaluated in I_L's range with every I_L⁻¹ cancelled (I_L Γ is bounded even where Γ's misfit block is not). A frame that
// can only repeat the bias already in L adds no confidence; repeated identical frames drive L → Γ: the posterior never
// claims less than one view's systematic error. The caller keeps L (AnchorBias) beside the RelPose, carries it by the same motion Jacobian
// (transport_bias), and passes it back: the update recovers P_z = P − L Lᵀ.
// L has two blocks of three latent columns, both oriented on the robot's axes (symmetric square roots), so the same
// latent means the same direction from frame to frame:
//   0–2  the PHYSICAL errors the rays share (mount, wall) — the same for the whole passage;
//   3–5  the model's MISFIT — a property of the VIEW, so it decorrelates as the view changes (below).
inline constexpr int kBiasCols = 6;
using BiasFactor = Eigen::Matrix<float, 3, kBiasCols>;
struct AnchorBias { BiasFactor L = BiasFactor::Zero(); };
// The odometry step that moves the anchor's mean (relative_pose.h move_mean, at the PRE-move mean x) moves its error the
// same way: L ← J L (linear, so P − L Lᵀ predicts exactly as P_z would).
// The PHYSICAL latent is the same for the whole passage. The MISFIT is a property of the VIEW — which beams land on which
// casing, leaf edge or stretch of wall — so it decorrelates as the view changes: ξ_m ← ρ ξ_m + √(1 − ρ²) η,
// ρ = exp(−Δ / ℓ), with Δ how far the doorway moved in the robot's frame this step (translation, and the lever of a
// turn) and ℓ the misfit's own scale, half the opening — the size of the clusters it is measured on (bias_terms). The
// part that decorrelates is no longer shared with the next frame: it leaves L and stays in P (P − L Lᵀ grows by
// (1 − ρ²) L_m L_mᵀ), where the next frames' evidence can average it.
// ★Fully persistent (ρ = 1), a lateral misfit picked up on the far approach could never be corrected once the open
//  door's jambs were in view (try67 replay: the lateral error held at −0.1…−0.27 m for 25 s). Fully fresh (ρ = 0: the
//  misfit as one more per-frame noise) the doorway band's yaw drag came back: 5–18° median error at |s| < 1 m on try66,
//  67 and 69, against 1–4° with the view's persistence (task-10-report.md, the band table).
inline void transport_bias(AnchorBias& b, const Eigen::Vector3f& x_before, float side, float adv, float dth, float width)
{
    Eigen::Matrix3f J, G;
    rc::interaction::predict_jacobians(x_before, side, adv, dth, J, G);
    const Eigen::Vector3f moved = rc::interaction::move_mean(x_before, side, adv, dth);
    const float ell = 0.5f * std::max(width, 1e-3f);
    const float rho = std::exp(-(moved.head<2>() - x_before.head<2>()).norm() / ell);
    b.L = J * b.L;
    b.L.rightCols<3>() *= rho;
}
// ⚑ FLAGGED CONSTANT — the LiDAR mount's yaw uncertainty (rad). Nobody has measured the LiDAR's extrinsic yaw; this is the
//   codebase's stated mount-yaw prior for the zed/ricoh (ImageEdge.mountYawSigma = 0.0035 rad), whose cluster-honest
//   solve came out at ±0.216° = 0.0038 rad (room_concept/VALIDATION_THREE_DEVICE_CORNERS.md). Same bracket class.
constexpr double kMountYawSigma = 0.0035;

// This view's systematic error b, in the robot frame on the anchor's (x, y, yaw): b = C^{1/2} ξ_p + I_L⁺ J_s^{1/2} ξ_m.
//   PHYSICS, C. (i) The mount's translation and the wall's departure from a plane: σ_c on x, y (kCommonSigma, as before).
//   (ii) The mount's yaw ε rotates the whole scan about the LiDAR origin o: the anchor turns by ε AND its position
//   moves by ε × (p − o) — a lateral common mode that GROWS WITH RANGE: g = (−(p_y − o_y), p_x − o_x, 1), σ_ε² g gᵀ.
//   (iii) The wall's own straightness: σ_c of departure at each end of the run [u_min, u_max] tilts the line by
//   √2 σ_c / (u_max − u_min).
//   MISFIT, J_s (cluster-robust sandwich, in SCORE space). The model's information I_L is what the rays WOULD carry were
//   the model right; the dispersion of their scores says what they do carry. Scores are summed per STRUCTURAL CLUSTER —
//   beams ending on one element share its misfit (one casing board, one leaf, one stretch of wall): the endpoint's
//   along-wall position in bins of half the opening (edges at the jambs and the centre) × its depth class (in front of
//   the wall / on it / beyond it, at the model's own count level 3σ + t/2); the mask is one more cluster. With S_c the
//   clusters' score sums, centred and × G/(G − 1) (CR1), J = Σ (S_c − S̄)(S_c − S̄)ᵀ / β_c (the power likelihood's own
//   scale: a well-specified frame gives J ≈ I_L). The data-only sandwich I_L⁻¹ J I_L⁻¹ exceeds the model's I_L⁻¹ by
//   I_L⁻¹ (J − I_L) I_L⁻¹; J_s = the PSD part of (J − I_L) in I_L's range (eigenvalues above 1e-9 of the largest:
//   arithmetic, the numerical rank) is that excess in score space — bounded, where its pose-space image is not (a
//   direction the frame barely constrains has an unbounded data-only misfit, and the update never forms it: see the
//   gain below). A model that OVER-states the scatter in some direction earns no extra precision there.
//   ⚑ The cluster size (half the opening) is a modelling definition: the structures the thin-wall model misdescribes are
//   the jambs' casings, the leaf (one clear width) and the wall either side. Replays: try66–69 (task-10-report.md).
struct BiasTerms { Eigen::Matrix3d C = Eigen::Matrix3d::Zero(), Js = Eigen::Matrix3d::Zero(); };
inline BiasTerms bias_terms(const anchor_detail::Problem& pb, const Eigen::Vector3d& x, const Eigen::Matrix3d& I_L,
                            const Eigen::Vector2d& lidar_o)
{
    using namespace anchor_detail;
    BiasTerms bt;
    M3d& B = bt.C;
    B(0, 0) = B(1, 1) = kCommonSigma * kCommonSigma;
    const V3d gy(-(x.y() - lidar_o.y()), x.x() - lidar_o.x(), 1.0);
    B += kMountYawSigma * kMountYawSigma * gy * gy.transpose();
    const double run = std::max(pb.model.u_max - pb.model.u_min, 1e-9);
    B(2, 2) += 2.0 * kCommonSigma * kCommonSigma / (run * run);

    std::map<std::int64_t, V3d> cl;
    if (not pb.rays.empty() and pb.beta_lidar > 1e-12)
    {
        const auto sc = pb.ray_scores(x);
        const double cz = std::cos(x.z()), sz = std::sin(x.z()), half = 0.5 * std::max(pb.model.w, 1e-9);
        for (std::size_t i = 0; i < sc.size(); ++i)
        {
            const auto q = Problem::at(pb.rays[i], x, cz, sz);
            const auto ub = std::int64_t(std::floor(std::clamp(q.ea.x() / half, -1e6, 1e6)));   // (clamp: integer range only)
            const int dc = q.a.d < -pb.model.count_level ? 0 : (q.a.d > pb.model.count_level ? 2 : 1);
            cl.try_emplace(ub * 3 + dc, V3d::Zero()).first->second += pb.wt[i] * sc[i] / std::sqrt(pb.beta_lidar);   // ÷√β per factor: J carries 1/β
        }
    }
    if (pb.mask != nullptr and pb.mask->valid and pb.beta_cam > 1e-12)
    {
        V3d sm;
        for (int a = 0; a < 3; ++a)
        {
            V3d e = V3d::Zero(); e(a) = 1e-3;
            sm(a) = -pb.beta_cam * (Mask::neg_loglik(pb.mask->centroid_u(x + e), pb.mask_var) - Mask::neg_loglik(pb.mask->centroid_u(x - e), pb.mask_var)) / 2e-3;
        }
        cl[std::numeric_limits<std::int64_t>::max()] = sm / std::sqrt(pb.beta_cam);
    }
    const std::size_t G = cl.size();
    if (G < 2) return bt;   // one cluster: its scatter is not measurable
    V3d mean = V3d::Zero();
    for (const auto& [k, v] : cl) mean += v;
    mean /= double(G);
    M3d J = M3d::Zero();
    for (const auto& [k, v] : cl) J += (v - mean) * (v - mean).transpose();
    J *= double(G) / double(G - 1);
    Eigen::SelfAdjointEigenSolver<M3d> es(0.5 * (I_L + I_L.transpose()));
    const double lmax = es.eigenvalues().maxCoeff();
    if (not (lmax > 0.0)) return bt;
    std::vector<int> r;
    for (int q = 0; q < 3; ++q) if (es.eigenvalues()(q) > 1e-9 * lmax) r.push_back(q);
    const int nr = int(r.size());
    Eigen::MatrixXd U(3, nr);
    for (int a = 0; a < nr; ++a) U.col(a) = es.eigenvectors().col(r[std::size_t(a)]);
    const Eigen::MatrixXd M = U.transpose() * (J - I_L) * U;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> em(0.5 * (M + M.transpose()));
    bt.Js = U * em.eigenvectors() * em.eigenvalues().cwiseMax(0.0).asDiagonal() * em.eigenvectors().transpose() * U.transpose();
    return bt;
}
inline Eigen::Matrix3d sym_sqrt(const Eigen::Matrix3d& A)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> e(0.5 * (A + A.transpose()));
    return e.eigenvectors() * e.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() * e.eigenvectors().transpose();
}

// One free-energy minimisation: prior = the predicted RelPose; likelihood = rays (+ mask). Levenberg-Marquardt on
// (x, y, yaw) against the predictive spread (two stages); Laplace at the sensor's σ; the frame's systematic error
// carried as a PERSISTENT bias (Task 10, AnchorBias above): `bias` in/out, nullptr = a single frame (L = 0 before it).
// wall_u_min/max: the door's wall run along the wall in ANCHOR coordinates (fix 2); `cam`: the zed frustum in the
// robot frame for the camera's visibility (fix 1; default unknown = the mask is not tempered by geometry);
// p_open_prior: the door's own P(open) (door_concept's leaf-angle belief), the structure the visibility check
// predicts in the opening (0.5 = unknown: a leaf and a gap are equally expected there).
inline std::pair<rc::interaction::RelPose, AnchorEvidence>
update_door_anchor(const rc::interaction::RelPose& prior, float width, float thickness, float wall_u_min, float wall_u_max,
                   const Eigen::Vector3f& lidar_origin_robot, const std::vector<Eigen::Vector3f>& rays_robot,
                   const std::vector<Eigen::Vector3f>& mask_pts_robot, const CameraFrustum& cam = {},
                   float p_open_prior = 0.5f, const LeafPrior& leaf = {},
                   float floor_below = -std::numeric_limits<float>::infinity(), AnchorStaging* staging = nullptr,
                   AnchorBias* bias = nullptr)
{
    using namespace anchor_detail;
    AnchorEvidence ev;
    // Task 10: the data-space prior P_z = P − L Lᵀ. When the carried factor no longer fits under P (the caller re-set
    // or fused the belief without it), the bookkeeping restarts: L = 0, P is the data-space belief.
    Eigen::Matrix<double, 3, kBiasCols> L0 = Eigen::Matrix<double, 3, kBiasCols>::Zero();
    M3d Pz0 = prior.P.cast<double>();
    if (bias != nullptr)
    {
        L0 = bias->L.cast<double>();
        const M3d t = Pz0 - L0 * L0.transpose();
        if (Eigen::LLT<M3d>(0.5 * (t + t.transpose())).info() == Eigen::Success) Pz0 = 0.5 * (t + t.transpose());
        else L0.setZero();
    }

    const Mask mask = make_mask(mask_pts_robot, prior.x.cast<double>());

    // The FIT — its prior, predictive spreads, ray budget, tempering — runs under the whole prior P: where the beams can
    // land is set by everything the anchor does not know, the bias included (a fit under P_z alone could not reach jambs
    // a persistent error had put 0.3 m off: try67 replay, 0.68 m lateral error). P_z enters only the gain.
    Problem pb{prior.x.cast<double>(), prior.P.cast<double>().inverse(), Model(width, thickness, wall_u_min, wall_u_max)};
    pb.mask = &mask;
    pb.leaf = make_leaf_model(leaf, pb.model.w);
    // Replay (staging->replay): the record's prior-staged set stands for the frame (the counts below then see only it).
    const bool replay = staging != nullptr and staging->replay;
    const std::vector<Ray2> all_rays = make_rays(lidar_origin_robot, replay ? staging->prior.rays : rays_robot, floor_below);

    if (all_rays.empty() and not mask.valid)
        return {prior, ev};   // nothing measured: the prior IS the posterior, bit for bit

    V3d x = pb.xbar;
    M3d P = prior.P.cast<double>();
    // ─── THIS DOOR'S WALL RUN, decided at the prior (fix 2), and the RAY BUDGET (fix round 4): stage_rays ───────────
    if (replay) stage_given(pb, all_rays, staging->prior.wt, staging->prior.az_step, x, P);
    else
    {
        stage_rays(pb, all_rays, x, P);
        if (staging != nullptr) staging->prior = export_staged(pb, rays_robot, rays_robot.size());
    }
    ev.n_budget = int(pb.rays.size());
    ev.az_step = float(pb.az_step);
    pb.select_leaf_configs(x, true);
    // ─── VISIBILITY at the prior: how much of this frame's evidence is about THIS doorway (fix 1) — it TEMPERS the
    //     evidence. What is PUBLISHED (ev.p_view) is re-evaluated at the posterior below (fix round 4).
    {
        const LidarView lv = lidar_view(pb, x, std::clamp(double(p_open_prior), 0.0, 1.0));
        pb.beta_lidar = lv.p;
        pb.beta_cam   = camera_view(cam, pb.model.w, x, P);
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
        if (stage > 0)   // stage 0: staged above, at the prior
        {
            pb.linearise(x, stage < 2 ? P : M3d::Zero());
            pb.select_leaf_configs(x, false);   // the leaf configurations that carry the posterior HERE (fix round 4)
        }
        // Levenberg-Marquardt in the PRIOR's metric (damping μ P̄⁻¹), steps accepted on the gain ratio ρ = actual /
        // predicted decrease (Nielsen's μ update). ★A plain Newton step + Armijo line search failed here: along a
        // shut wall the lateral direction has only the prior's curvature, so the step's lateral part was metres long,
        // and the large depth/yaw gain in the same step made the line search ACCEPT it (dy = +0.35: the anchor was
        // dragged 1.3 m along the wall). The gain ratio checks the quadratic model's promise per step, so a
        // direction the model mispredicts is shrunk instead of carried along. (ρ > 0, ×2, ⅓ are the textbook
        // optimiser constants, not model levels.)
        double mu = 1e-2, nu = 2.0;
        bool fresh = false;   // g, H were computed at the current x under the current spreads
        for (int it = 0; it < kMaxIter; ++it)
        {
            double f0 = 0;
            pb.derivatives(x, g, H, &f0);
            fresh = true;
            const M3d Hp = pb.Pinv + psd_information(H, pb.Pinv);   // positive definite by construction
            // Converged when even the undamped Newton step promises less than kFTol (fix round 4: a stage used to end
            // by failing up to 30 damped trials, ~30 F evaluations each, a quarter of the update's cost).
            {
                const V3d step = -Hp.llt().solve(g);
                if (-(g.dot(step) + 0.5 * step.dot(Hp * step)) < kFTol) break;
            }
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
                    fresh = false;
                }
                else { mu *= nu; nu *= 2.0; }
            }
            x.z() = std::remainder(x.z(), 2.0 * M_PI);
            if (not moved) break;   // no step decreases F: at this stage's optimum (to the optimiser's resolution)
        }
        if (not fresh) pb.derivatives(x, g, H);
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
    pb.select_leaf_configs(x, false);
    pb.derivatives(x, g, H);
    V3d dx = x - pb.xbar;
    dx.z() = std::remainder(dx.z(), 2.0 * M_PI);
    const M3d I_L = psd_information(H, pb.Pinv);
    H = pb.Pinv + I_L;
    if (pb.F(x + H.llt().solve(-g)) > pb.F(x))
        g.setZero();   // the true-F Newton step would not descend: x already is its optimum to our resolution
    // ─── Task 10: the PERSISTENT bias, and the covariance that is published ───────────────────────────────────────
    // The beams' own noise is what I_L describes, and it is independent from frame to frame: it accumulates in the
    // data-space belief (P_z). What does NOT average out is the frame's SYSTEMATIC error b — the LiDAR's extrinsic, the
    // wall's departure from a plane, and the structure this model does not describe (casings, a swung leaf off its
    // believed hinge) — because the next frame, 0.1 s later, sees the same wall through the same mount (see
    // AnchorBias). Its factor Γ is this view's (bias_terms); the posterior's sensitivity to the latent is L, updated with
    // the Schmidt gain K (see AnchorBias):  L ← (I − K) L + K Γ,  P_z ← (I − K) P_z (I − K)ᵀ + K I_L⁻¹ Kᵀ,  P = P_z + L Lᵀ.
    // Everything is formed in I_L's range, so no I_L⁻¹ appears. The mean: x̄ + K ν (b has zero mean).
    rc::interaction::RelPose post = prior;
    rc::interaction::RelPose post_z = prior;   // the data-space belief: what the beams predict the rays from
    using M3L = Eigen::Matrix<double, 3, kBiasCols>;
    M3L L_new = L0;
    {
        const M3d& Pz = Pz0;
        const BiasTerms bt = bias_terms(pb, x, I_L, lidar_origin_robot.head<2>().cast<double>());
        const M3d Ch = sym_sqrt(bt.C), Jh = sym_sqrt(bt.Js);
        Eigen::SelfAdjointEigenSolver<M3d> es(0.5 * (I_L + I_L.transpose()));
        const double lmax = es.eigenvalues().maxCoeff();
        std::vector<int> rr;
        for (int q = 0; q < 3; ++q) if (lmax > 0.0 and es.eigenvalues()(q) > 1e-9 * lmax) rr.push_back(q);
        const int nr = int(rr.size());
        M3d K = M3d::Zero(), KIp = M3d::Zero();   // the gain and K I_L⁺
        V3d Kv = V3d::Zero();
        M3L KG = M3L::Zero();
        if (nr > 0)
        {
            Eigen::MatrixXd U(3, nr);
            Eigen::VectorXd lam(nr);
            for (int a2 = 0; a2 < nr; ++a2) { U.col(a2) = es.eigenvectors().col(rr[std::size_t(a2)]); lam(a2) = es.eigenvalues()(rr[std::size_t(a2)]); }
            M3L GI;                                         // I_L Γ = [I_L C^{1/2}, J_s^{1/2}]
            GI.setZero();
            GI.leftCols<3>() = I_L * Ch;
            GI.rightCols<3>() = Jh;
            const M3L DI = GI - I_L * L0;                   // I_L (Γ − L)
            const Eigen::MatrixXd Nt = U.transpose() * (I_L * Pz * I_L + DI * DI.transpose()) * U;
            const Eigen::MatrixXd Q = (Nt + Eigen::MatrixXd(lam.asDiagonal())).llt().solve(Eigen::MatrixXd::Identity(nr, nr));
            const M3d T = Pz * I_L - L0 * DI.transpose();   // (P_z − L (Γ − L)ᵀ) I_L
            KIp = T * U * Q * U.transpose();
            K = KIp * I_L;
            Kv = KIp * (H * dx - g);                        // K ν, from I_L ν = H (x − x̄) − g (one Newton step on the true F)
            KG.leftCols<3>() = K * Ch;
            KG.rightCols<3>() = KIp * Jh;                   // K I_L⁺ J_s^{1/2}: Jh lies in I_L's range
        }
        const M3d IK = M3d::Identity() - K;
        const M3d Pz_new = IK * Pz * IK.transpose() + KIp * I_L * KIp.transpose();
        L_new = IK * L0 + KG;
        ev.info = I_L.cast<float>(); ev.misfit = bt.Js.cast<float>(); ev.gain = K.cast<float>(); ev.info_innov = (H * dx - g).cast<float>();
        V3d xp = pb.xbar + Kv;
        xp.z() = std::remainder(xp.z(), 2.0 * M_PI);
        post.x = xp.cast<float>();
        post_z.x = post.x;
        post_z.P = (0.5 * (Pz_new + Pz_new.transpose())).cast<float>();
        const M3d Pp = Pz_new + L_new * L_new.transpose();
        post.P = (0.5 * (Pp + Pp.transpose())).cast<float>();
        ev.sig_bias_yaw = float(std::sqrt(std::max(0.0, (L_new * L_new.transpose())(2, 2))));
    }
    if (bias != nullptr) bias->L = L_new.cast<float>();
    // A belief is never NaN: a non-finite posterior (any cause) is no update — the prior stands, and this frame's evidence
    // reports nothing in view. (try67: one NaN posterior silenced the anchor for 120 s; predicting a NaN keeps it NaN.)
    if (not post.x.allFinite() or not post.P.allFinite() or not L_new.allFinite())
    {
        if (bias != nullptr) bias->L = L0.cast<float>();
        return {prior, AnchorEvidence{}};
    }
    x = post.x.cast<double>();

    // ─── P(door in view) at the POSTERIOR — the pose that is written to the edge (fix round 4) ─────────────────────
    // ★Evaluated at the prior, a within-frame drag was published: k = 0 of the round-4 probe read p_view 0.871 at the
    //  prior while the posterior it published was already 1.29 rad off. Asked of the posterior (its mean and its own
    //  covariance), the same frame says whether the pose being PUBLISHED is the one the beams see.
    {
        const AnchorEvidence vw = door_view(post, width, thickness, wall_u_min, wall_u_max, lidar_origin_robot, rays_robot,
                                            mask_pts_robot, cam, p_open_prior, leaf, floor_below,
                                            replay ? &staging->post : nullptr,
                                            (staging != nullptr and not replay) ? &staging->post : nullptr);
        ev.v_lidar = vw.v_lidar; ev.o_lidar = vw.o_lidar; ev.p_lidar = vw.p_lidar; ev.p_cam = vw.p_cam; ev.p_view = vw.p_view;
        ev.lidar_open = vw.lidar_open; ev.lidar_door = vw.lidar_door; ev.lidar_wall = vw.lidar_wall;
    }

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
