/*
 * countable_viewpoint.h — how much a sensor's SHARED error has decorrelated since the channel last updated the
 * instance (spec 2026-11-15 lidar-measures §7c.2). Pure Eigen, no OpenCV/DSR/Qt.
 *
 * A sensor's systematic error (a bias in where the camera sees the edge, a ring-gap aliasing on the LiDAR) is
 * the same on consecutive frames and only decorrelates when the VIEWPOINT changes. rho in [0,1] is the
 * remaining correlation: 1 = parked (the shared error is unchanged, so re-measuring adds nothing), 0 = fully
 * decorrelated. It is a continuous function of the odometry path relative to the object — no gate.
 * Rotation in place moves neither the sensor's bearing to the object nor its range, so it leaves rho = 1.
 */
#pragma once

#include <optional>
#include <span>

#include <Eigen/Dense>

namespace rc {

// Accumulated since the channel's last update of the instance; the caller resets it after an update.
struct ViewpointPath
{
    float dphi_rad = 0.f;   // |bearing change| as seen from the object, summed (tangential path / range)
    float dz_m     = 0.f;   // |range change| to the object, summed (radial path)
};

// Adds |step·t̂|/z to dphi and |step·r̂| to dz, r̂ = unit(sensor − object), t̂ ⊥ r̂, z = |sensor − object|.
void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy);

// Room-frame displacement of the robot over dt. Robot frame: +x right, +y FORWARD (controller/FRAMES.md,
// ROBOT_GEOMETRY.md), CCW-positive yaw; v_adv is along +y, v_side along +x; rotated into the room by yaw_room.
Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s);

// exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z): e-fold when the camera has moved by the object's angular half-width.
float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m);
// exp(−|Δφ|/g), g = ring gap (rad). g ≤ 0 (SensorModel absent) ⇒ 1.
float rho_lidar_lateral(const ViewpointPath& p, float ring_gap_rad);
// exp(−|Δz||tan ε|/(g z)), ε = elevation of the beam. g ≤ 0 ⇒ 1.
float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float ring_gap_rad);
// ── The localiser's correction for ρ_chain (§7c.2), as testable bookkeeping. ─────────────────────────────────────────
// c = (pose at the last odometry-consistent sample − ref) − odometry integrated since ref. It must FAIL CLOSED like the
// camera/LiDAR paths: a stretch whose odometry cannot be integrated (no speed, a stale stamp, an unknown interval) is
// NOT a localiser correction, so the reference absorbs that stretch's pose increment (ref += Δpose) — the corrections
// accumulated before it are kept. ref and odo restart at the last consistent sample when the chain latent is updated.
struct ChainTracker
{
    Eigen::Vector3f ref  = Eigen::Vector3f::Zero();   // (x, y, psi) the correction is measured from
    Eigen::Vector3f prev = Eigen::Vector3f::Zero();   // pose at the last observed (odometry-consistent) sample
    Eigen::Vector3f odo  = Eigen::Vector3f::Zero();   // odometry (dx, dy, dpsi) integrated since ref
    bool have = false;
};
// SE(2) odometry step: (x, y) as odometry_step_room, psi = omega·dt.
Eigen::Vector3f odometry_step_room_se2(float v_adv, float v_side, float omega, float yaw_room, float dt_s);
// One sample: pose (x, y, psi) now, and the odometry step over the interval since the previous sample (nullopt ⇔ it
// cannot be integrated ⇒ the stretch is absorbed into ref). The first sample initialises the tracker.
void chain_observe(ChainTracker& t, const Eigen::Vector3f& pose, std::optional<Eigen::Vector3f> odo_step);
// c, the localiser's correction since ref (zero before the first sample); psi wrapped to (−pi, pi].
Eigen::Vector3f chain_correction(const ChainTracker& t);
// After the chain latent's update: measure from the last consistent sample.
void chain_rebase(ChainTracker& t);

// exp(−½ cᵀΣ⁻¹c) on the non-zero subspace of Σ (pseudo-inverse). A zero-variance direction with c ≠ 0 ⇒ 0; c = 0 ⇒ 1.
float rho_chain(const Eigen::Vector3f& c, const Eigen::Matrix3f& Sigma_pose);

// ── The robot's pose error in the room frame (spec 7c.7 b) ──────────────────────────────────────────────────────────
// Σ_pose (x, y, psi) in the room frame from the robot→room edge's 6×6 row-major rt_covariance block (one 36-slot block,
// ordered [x y z rx ry rz], in the edge's parent frame): slots {0, 1, 5}, pushed through rc::se2_inverse_jacobian
// evaluated at the pose of the edge (the pose being inverted). Empty span ⇒ zero.
Eigen::Matrix3f pose_cov_in_room(std::span<const float> rt_cov36, const Eigen::Vector3f& edge_pose);

// F_t = [[1,0,−Δr_y],[0,1,Δr_x],[0,0,1]]: how the pose error e = (δx, δy, δψ) is carried by a robot translation Δr.
Eigen::Matrix3d chain_transport(const Eigen::Vector2f& dr);

// The lever ℓ (object centre − robot position, room frame) through which a robot yaw error moves the object.
struct ChainLever { Eigen::Vector2f l = Eigen::Vector2f::Zero(); bool set = false; };
void lever_start(ChainLever& L, const Eigen::Vector2f& seed_c, const Eigen::Vector2f& robot_r);
// ℓ ← ρ(ℓ − Δr) + (1 − ρ)(c − r).
void lever_step(ChainLever& L, const Eigen::Vector2f& dr, float rho_chain, const Eigen::Vector2f& belief_c,
                const Eigen::Vector2f& robot_r);
// Θ = (cx, cy, H, w, h, yaw): rows cx = [1,0,−ℓy], cy = [0,1,ℓx], yaw = [0,0,1]; H, w, h zero.
Eigen::Matrix<double, 6, 3> chain_B(const Eigen::Vector2f& l);

}  // namespace rc
