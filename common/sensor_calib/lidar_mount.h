/*  lidar_mount.h — the planar correction (and sim-only injection) applied to helios points after the
 *  DSR device->body transform. Pure. Plan docs/superpowers/plans/2026-10-05-joint-calibration.md Task 5.
 *
 *  ONE CONVENTION (Fable review 1c). The helios MOUNT POSE ERROR is M = Planar{lever_x, lever_y, psi_h}:
 *  the sensor's displacement and CCW yaw from its nominal mount, in the body frame. Transforming through
 *  the NOMINAL mount yields p_nom = M^-1(p_true), so
 *    - correction (LidarMountApply):  T_corr   = M_est              (restores p_true)
 *    - injection  (sim only):         T_inject = inverse(M_inj)     (plants M_inj)
 *  and a correct estimate gives compose(T_corr, T_inject) = identity. Use inverse() wherever an inverse
 *  is meant, never hand-negated fields.
 */
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace rc::lidar_mount
{
    struct Planar { float x = 0.f, y = 0.f, yaw = 0.f; };   ///< body frame, metres / rad

    [[nodiscard]] inline Eigen::Vector3f apply(const Planar &t, const Eigen::Vector3f &p)
    {
        const float c = std::cos(t.yaw), s = std::sin(t.yaw);
        return {c * p.x() - s * p.y() + t.x, s * p.x() + c * p.y() + t.y, p.z()};
    }
    /// a ∘ b : apply b, then a.
    [[nodiscard]] inline Planar compose(const Planar &a, const Planar &b)
    {
        const float c = std::cos(a.yaw), s = std::sin(a.yaw);
        return {c * b.x - s * b.y + a.x, s * b.x + c * b.y + a.y, a.yaw + b.yaw};
    }
    [[nodiscard]] inline Planar inverse(const Planar &a)
    {
        const float c = std::cos(a.yaw), s = std::sin(a.yaw);
        return {-(c * a.x + s * a.y), -(-s * a.x + c * a.y), -a.yaw};
    }
    /// Exactly the identity (all three fields exactly zero): lets the ingestor skip the pass when
    /// nothing is configured. An exact equality, not a tolerance.
    [[nodiscard]] inline bool is_identity(const Planar &a)
    {
        return a.x == 0.f and a.y == 0.f and a.yaw == 0.f;
    }

    /* ── THE 6-DoF MOUNT (plan docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md Task 4) ──
     *  Pose6 M = {t, rpy}: a small mount error in the BODY frame, rotation R = Rz(yaw) Ry(pitch) Rx(roll)
     *  ABOUT THE SENSOR ORIGIN s (the sensor's nominal origin in the same frame), so
     *        apply(M, s, p) = R (p - s) + s + t        (a point AT the sensor origin moves only by t).
     *  SAME convention as Planar: correction T_corr = M_est restores p_true, injection T_inject = inverse(M_inj)
     *  plants M_inj, compose(M, inverse(M)) = identity. To first order rpy IS the rotation vector omega the
     *  estimators use (mount_factors.h: p' = p + omega x (p - s) + t).
     *  ⚠ Planar rotates about the BODY origin, Pose6 about the SENSOR origin: for the same physical error the
     *  translations differ by (I - R) s (helios, 1 deg: 2.7 mm in x). joint_calibration.h solve_mounts maps the
     *  kinematic factor's body-origin lever onto Pose6 explicitly.
     *  ⚠ WRITE-BACK: helios and bpearl hang INVERTED (shadow.json R = Ry(pi)). The SENSOR-frame increment of a
     *  body-frame rotation omega is Ry(pi)^T omega = (-omega_x, omega_y, -omega_z) -- roll and yaw flip, pitch
     *  does not (sensor_frame_increment_inverted). A write-back must COMPOSE rotations, never add euler angles.
     */
    struct Pose6
    {
        Eigen::Vector3f t   = Eigen::Vector3f::Zero();   ///< m, body frame
        Eigen::Vector3f rpy = Eigen::Vector3f::Zero();   ///< rad: roll (x), pitch (y), yaw (z), about the sensor origin
    };
    [[nodiscard]] inline Eigen::Matrix3f rotation(const Pose6 &m)
    {
        return (Eigen::AngleAxisf(m.rpy.z(), Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(m.rpy.y(), Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(m.rpy.x(), Eigen::Vector3f::UnitX())).toRotationMatrix();
    }
    [[nodiscard]] inline Eigen::Vector3f rpy_of(const Eigen::Matrix3f &R)
    {
        return {std::atan2(R(2, 1), R(2, 2)), -std::asin(std::clamp(R(2, 0), -1.f, 1.f)), std::atan2(R(1, 0), R(0, 0))};
    }
    [[nodiscard]] inline Eigen::Vector3f apply(const Pose6 &m, const Eigen::Vector3f &s, const Eigen::Vector3f &p)
    {
        return rotation(m) * (p - s) + s + m.t;
    }
    /// a ∘ b : apply b, then a (both about the same sensor origin, so s drops out).
    [[nodiscard]] inline Pose6 compose(const Pose6 &a, const Pose6 &b)
    {
        const Eigen::Matrix3f Ra = rotation(a);
        return {Ra * b.t + a.t, rpy_of(Ra * rotation(b))};
    }
    [[nodiscard]] inline Pose6 inverse(const Pose6 &a)
    {
        const Eigen::Matrix3f Rt = rotation(a).transpose();
        return {-(Rt * a.t), rpy_of(Rt)};
    }
    /// Exactly the identity (every field exactly zero), so the ingestor can skip the pass. Not a tolerance.
    [[nodiscard]] inline bool is_identity(const Pose6 &a)
    {
        return (a.t.array() == 0.f).all() and (a.rpy.array() == 0.f).all();
    }
    /// The sensor-frame increment of a BODY-frame rotation omega for a sensor mounted INVERTED (R_nom = Ry(pi)).
    [[nodiscard]] inline Eigen::Vector3f sensor_frame_increment_inverted(const Eigen::Vector3f &omega_body)
    {
        return {-omega_body.x(), omega_body.y(), -omega_body.z()};
    }
}
