#include "efe_gradient.h"

#include <algorithm>
#include <cmath>

namespace
{
    /** Quadratic repulsion gradient on a single joint with bounded limits.
     *  Returns the additional velocity contribution that pushes q_i away from
     *  whichever wall it is approaching, zero when far enough inside.
     *  Continuous joints (±inf limits) trivially return 0. */
    double limit_repulsion_velocity(double q_i, double lo, double hi,
                                    double margin, double gain)
    {
        if (not std::isfinite(lo) or not std::isfinite(hi))
            return 0.0;
        const double d_lo = q_i - lo;
        const double d_hi = hi - q_i;
        if (d_lo < margin)
        {
            const double e = (margin - d_lo) / margin;   // ∈ (0, 1]
            return +gain * e * e;                        // push toward positive (away from lo)
        }
        if (d_hi < margin)
        {
            const double e = (margin - d_hi) / margin;
            return -gain * e * e;                        // push toward negative (away from hi)
        }
        return 0.0;
    }
}

std::array<double, Kinematics::N_ARM_JOINTS> efe_gradient_step(
    Kinematics& kin,
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& x_target,
    const EFEParams& params)
{
    // 1. Position error in world frame.
    const Eigen::Vector3d x_ee = kin.forward_kinematics(q);
    const Eigen::Vector3d err  = x_ee - x_target;

    // 2. Instrumental gradient: q̇ = −α · J_linᵀ · err
    const Eigen::Matrix<double, 3, Kinematics::N_ARM_JOINTS> J = kin.arm_jacobian_linear(q);
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> q_dot = -params.gain_pos * J.transpose() * err;

    // 3. Joint-limit repulsion (only on revolute joints; continuous → no-op).
    const auto lims = kin.arm_joint_position_limits();
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        q_dot(i) += limit_repulsion_velocity(q[i], lims[i].first, lims[i].second,
                                              params.limit_margin, params.limit_gain);

    // 4. Per-joint velocity clip — never exceed the URDF-declared limit
    //    nor the user-configured ceiling.
    const auto vlim = kin.arm_joint_velocity_limits();
    std::array<double, Kinematics::N_ARM_JOINTS> out{};
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
    {
        const double cap = std::min(params.max_joint_vel,
                                    std::isfinite(vlim[i]) ? vlim[i] : params.max_joint_vel);
        out[i] = std::clamp(q_dot(i), -cap, +cap);
    }
    return out;
}
