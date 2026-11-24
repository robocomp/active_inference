/*
 * common/ai_belief/viewpoint.h — (moved from countable_concept/src/countable_viewpoint.{h,cpp}; header-only) how much a sensor's SHARED error has decorrelated since the channel last updated the
 * instance (spec 2026-11-15 lidar-measures §7c.2). Pure Eigen, no OpenCV/DSR/Qt.
 *
 * A sensor's systematic error (a bias in where the camera sees the edge, a ring-gap aliasing on the LiDAR) is
 * the same on consecutive frames and only decorrelates when the VIEWPOINT changes. rho in [0,1] is the
 * remaining correlation: 1 = parked (the shared error is unchanged, so re-measuring adds nothing), 0 = fully
 * decorrelated. It is a continuous function of the odometry path relative to the object — no gate.
 * Rotation in place moves neither the sensor's bearing to the object nor its range, so it leaves rho = 1.
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>

#include <Eigen/Dense>

#include "../place_memory/place_map.h"   // place::se2_inverse_jacobian

namespace rc::ai::view {

// Accumulated since the channel's last update of the instance; the caller resets it after an update.
struct ViewpointPath
{
    float dphi_rad = 0.f;   // |bearing change| as seen from the object, summed (tangential path / range)
    float dz_m     = 0.f;   // |range change| to the object, summed (radial path)
};

// Adds |step·t̂|/z to dphi and |step·r̂| to dz, r̂ = unit(sensor − object), t̂ ⊥ r̂, z = |sensor − object|.
inline void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy)
{
    const Eigen::Vector2f r = sensor_xy_room - object_xy;
    const float z = r.norm();
    // Degenerate geometry (the sensor sits at the object centre: r̂ undefined), not a tuning threshold.
    if (z < 1e-3f)
        return;
    const Eigen::Vector2f rhat = r / z;
    const Eigen::Vector2f that(-rhat.y(), rhat.x());
    p.dphi_rad += std::abs(step_xy_room.dot(that)) / z;
    p.dz_m     += std::abs(step_xy_room.dot(rhat));
}

// Room-frame displacement of the robot over dt. Robot frame: +x right, +y FORWARD (controller/FRAMES.md,
// ROBOT_GEOMETRY.md), CCW-positive yaw; v_adv is along +y, v_side along +x; rotated into the room by yaw_room.
inline Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s)
{
    const float c = std::cos(yaw_room), s = std::sin(yaw_room);
    const Eigen::Vector2f robot(v_side * dt_s, v_adv * dt_s);   // x right, y forward
    return {c * robot.x() - s * robot.y(), s * robot.x() + c * robot.y()};
}

namespace detail {
// |Δφ| / half_angle, with the 0/0 limit defined as 0: no bearing change decorrelates nothing, whatever the object's
// width (degenerate geometry w⊥ = 0, not a tuning guard). Δφ > 0 with half_angle = 0 stays +inf ⇒ ρ = 0.
inline float bearing_term(float dphi, float half_angle)
{
    if (dphi == 0.f) return 0.f;
    return std::abs(dphi) / half_angle;
}
}  // namespace detail

// exp(−|Δφ|/atan(w⊥/2z) − |Δz|/z): e-fold when the camera has moved by the object's angular half-width.
inline float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m)
{
    const float half_angle = std::atan(w_perp_m / (2.f * z_m));
    return std::exp(-detail::bearing_term(p.dphi_rad, half_angle) - std::abs(p.dz_m) / z_m);
}
// exp(−|Δφ|/g), g = ring gap (rad). g ≤ 0 (SensorModel absent) ⇒ 1.
inline float rho_lidar_lateral(const ViewpointPath& p, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dphi_rad) / g);
}
// exp(−|Δz||tan ε|/(g z)), ε = elevation of the beam. g ≤ 0 ⇒ 1.
inline float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dz_m) * std::abs(std::tan(eps_rad)) / (g * z_m));
}
namespace detail {
inline float wrap_pi(float a)
{
    a = std::fmod(a + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
    if (a <= 0.f) a += 2.f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
}
}  // namespace detail

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
inline Eigen::Vector3f odometry_step_room_se2(float v_adv, float v_side, float omega, float yaw_room, float dt_s)
{
    const Eigen::Vector2f d = odometry_step_room(v_adv, v_side, yaw_room, dt_s);
    return {d.x(), d.y(), omega * dt_s};
}
// One sample: pose (x, y, psi) now, and the odometry step over the interval since the previous sample (nullopt ⇔ it
// cannot be integrated ⇒ the stretch is absorbed into ref). The first sample initialises the tracker.
inline void chain_observe(ChainTracker& t, const Eigen::Vector3f& pose, std::optional<Eigen::Vector3f> odo_step)
{
    if (not t.have)
    {
        t.ref = t.prev = pose;
        t.odo.setZero();
        t.have = true;
        return;
    }
    if (odo_step.has_value()) t.odo += *odo_step;
    else                      t.ref += pose - t.prev;   // unknown odometry: not a correction (fail closed)
    t.prev = pose;
}
// c, the localiser's correction since ref (zero before the first sample); psi wrapped to (−pi, pi].
inline Eigen::Vector3f chain_correction(const ChainTracker& t)
{
    if (not t.have) return Eigen::Vector3f::Zero();
    Eigen::Vector3f c = (t.prev - t.ref) - t.odo;
    c.z() = detail::wrap_pi(c.z());
    return c;
}
// After the chain latent's update: measure from the last consistent sample.
inline void chain_rebase(ChainTracker& t)
{
    t.ref = t.prev;
    t.odo.setZero();
}

// exp(−½ cᵀΣ⁻¹c) on the non-zero subspace of Σ (pseudo-inverse). A zero-variance direction with c ≠ 0 ⇒ 0; c = 0 ⇒ 1.
inline double rho_chain(const Eigen::Vector3d& c, const Eigen::Matrix3d& S)
{
    // A coordinate whose Σ diagonal is exactly zero is declared by the data to have no uncertainty (PSD ⇒ its row and
    // column are zero): any non-zero c there cannot be explained at all ⇒ 0. No tolerance anywhere. The remaining
    // principal sub-block (symmetrised) is decomposed; eigen-directions with λ ≤ 0 (sign test) contribute nothing.
    std::array<int, 3> keep{};
    int n = 0;
    for (int i = 0; i < 3; ++i)
    {
        if (S(i, i) == 0.0) { if (c(i) != 0.0) return 0.0; }
        else keep[static_cast<std::size_t>(n++)] = i;
    }
    if (n == 0) return 1.0;
    Eigen::MatrixXd Sub(n, n);
    Eigen::VectorXd cs(n);
    for (int i = 0; i < n; ++i)
    {
        cs(i) = c(keep[static_cast<std::size_t>(i)]);
        for (int j = 0; j < n; ++j)
            Sub(i, j) = 0.5 * (S(keep[static_cast<std::size_t>(i)], keep[static_cast<std::size_t>(j)])
                             + S(keep[static_cast<std::size_t>(j)], keep[static_cast<std::size_t>(i)]));
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Sub);
    double maha = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double lam = es.eigenvalues()(i);
        if (lam > 0.0)
        {
            const double p = es.eigenvectors().col(i).dot(cs);
            maha += p * p / lam;
        }
    }
    return std::exp(-0.5 * maha);
}

// ── The robot's pose error in the room frame (spec 7c.7 b) ──────────────────────────────────────────────────────────
// Σ_pose (x, y, psi) in the room frame from the robot→room edge's 6×6 row-major rt_covariance block (one 36-slot block,
// ordered [x y z rx ry rz], in the edge's parent frame): slots {0, 1, 5}, pushed through rc::se2_inverse_jacobian
// evaluated at the pose of the edge (the pose being inverted). Empty span ⇒ zero.
inline Eigen::Matrix3d pose_cov_in_room(std::span<const float> rt_cov36, const Eigen::Vector3f& edge_pose)
{
    if (rt_cov36.size() < 36) return Eigen::Matrix3d::Zero();
    constexpr int idx[3] = {0, 1, 5};
    Eigen::Matrix3d S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S(i, j) = rt_cov36[static_cast<std::size_t>(idx[i] * 6 + idx[j])];
    const Eigen::Matrix3d J = place::se2_inverse_jacobian(edge_pose).cast<double>();
    Eigen::Matrix3d R = J * S * J.transpose();   // double: float J Σ Jᵀ would put ~1e-7 noise on a null direction
    R = 0.5 * (R + R.transpose());
    return R;
}

// F_t = [[1,0,−Δr_y],[0,1,Δr_x],[0,0,1]]: how the pose error e = (δx, δy, δψ) is carried by a robot translation Δr.
inline Eigen::Matrix3d chain_transport(const Eigen::Vector2f& dr)
{
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -dr.y();
    F(1, 2) =  dr.x();
    return F;
}

// The lever ℓ (object centre − robot position, room frame) through which a robot yaw error moves the object.
struct ChainLever { Eigen::Vector2f l = Eigen::Vector2f::Zero(); bool set = false; };
inline void lever_start(ChainLever& L, const Eigen::Vector2f& seed_c, const Eigen::Vector2f& robot_r)
{
    L.l = seed_c - robot_r;
    L.set = true;
}
// ℓ ← ρ(ℓ − Δr) + (1 − ρ)(c − r).
inline void lever_step(ChainLever& L, const Eigen::Vector2f& dr, float rho, const Eigen::Vector2f& c, const Eigen::Vector2f& r)
{
    L.l = rho * (L.l - dr) + (1.f - rho) * (c - r);
}
// Θ = (cx, cy, H, w, h, yaw): rows cx = [1,0,−ℓy], cy = [0,1,ℓx], yaw = [0,0,1]; H, w, h zero.
inline Eigen::Matrix<double, 6, 3> chain_B(const Eigen::Vector2f& l)
{
    Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
    B(0, 0) = 1.0; B(0, 2) = -l.y();
    B(1, 1) = 1.0; B(1, 2) =  l.x();
    B(5, 2) = 1.0;
    return B;
}

}  // namespace rc::ai::view
