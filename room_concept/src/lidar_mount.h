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
}
