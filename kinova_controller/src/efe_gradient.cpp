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

    /// Yoshikawa manipulability μ = √det(J · Jᵀ). Clamped at 0 to survive
    /// the tiny negative determinants that occur near singularities due
    /// to floating-point cancellation.
    inline double yoshikawa_mu(const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS>& J)
    {
        const double det = (J * J.transpose()).determinant();
        return det > 0.0 ? std::sqrt(det) : 0.0;
    }
}

std::array<double, Kinematics::N_ARM_JOINTS> efe_gradient_step(
    Kinematics& kin,
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& x_target,
    const EFEParams& params)
{
    // 1. Pose + 6×7 Jacobian (linear rows 0-2, angular rows 3-5).
    const auto pose = kin.tool_pose(q);
    const auto J    = kin.arm_jacobian_full(q);
    const auto J_lin = J.template topRows<3>();      // 3×7  d(pos)/dq
    const auto J_ang = J.template bottomRows<3>();   // 3×7  d(ω)/dq

    // 2. Damped-least-squares position step (Corke RVC §8.4).
    //    Q     = J_lin · J_linᵀ + λ²I_3                    (3×3, SPD).
    //    q̇_p  = −J_linᵀ · Q⁻¹ · (C_pos ⊙ err).
    //    λ does double duty: bounds q̇ in singular directions AND folds
    //    the velocity-effort penalty (Corke's L2 on ‖q̇‖) into the same
    //    solve. C_pos keeps its anisotropic-precision semantics.
    //    Q's LDLT factorisation is reused below for the null-space
    //    projector if manipulability is active.
    const Eigen::Vector3d err_pos = pose.position - x_target;
    const double lambda_sq = params.dls_lambda * params.dls_lambda;
    const Eigen::Matrix3d Q =
        J_lin * J_lin.transpose() + lambda_sq * Eigen::Matrix3d::Identity();
    const Eigen::LDLT<Eigen::Matrix3d> Q_ldlt(Q);
    const Eigen::Vector3d weighted_err = params.C_pos.cwiseProduct(err_pos);
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> q_dot =
        -J_lin.transpose() * Q_ldlt.solve(weighted_err);

    // 3. Orientation gradient (partial alignment of tool-z with z_des).
    //    From C_orient(q) = 1 − z_tool(q) · z_des, the chain rule yields
    //        ∂C/∂q = −J_angᵀ · (z_tool × z_des),
    //    so gradient-descent gives q̇_o = +α_o · J_angᵀ · (z_tool × z_des).
    //    Cross product vanishes both at perfect alignment (good) and at the
    //    antipodal 180° case (singular); the position term perturbs us off
    //    the singular point if we ever start there.
    if (params.gain_orient > 0.0)
    {
        const Eigen::Vector3d z_tool = pose.rotation.col(2);
        const Eigen::Vector3d cross  = z_tool.cross(params.desired_approach);
        q_dot += params.gain_orient * J_ang.transpose() * cross;
    }

    // 3b. Secondary-axis alignment for roll control (tool +X ↔ x_des).
    //     Same form as the z-axis cost — the cross-product is the gradient
    //     of (1 − x_tool · x_des). Lets us pin the gripper's open axis to,
    //     e.g., the bottle's vertical axis for a side grasp.
    if (params.gain_secondary > 0.0)
    {
        const Eigen::Vector3d x_tool = pose.rotation.col(0);
        const Eigen::Vector3d cross  = x_tool.cross(params.desired_secondary);
        q_dot += params.gain_secondary * J_ang.transpose() * cross;
    }

    // 3c. Manipulability ascent in the null space of the DLS position task
    //     (Corke RVC §8.4 redundancy resolution). ∂μ/∂q via central
    //     differences — 14 extra arm_jacobian_full calls, ≈ 0.5 ms total at
    //     20 ms cadence. The soft null-space projector
    //         N = I − J_linᵀ Q⁻¹ J_lin
    //     reuses the LDLT factorisation built above. With this projection
    //     the manipulability term cannot disturb the EE position task; it
    //     only spends the 4 redundant DOFs of the 7-DOF arm.
    //     N.B. arm_jacobian_full(q±h) mutates kin.data_, so this MUST run
    //     after every place that still reads J/J_lin/J_ang (they are local
    //     copies, so fine), and we don't depend on kin's internal state
    //     after this block.
    if (params.gain_mu > 0.0)
    {
        constexpr double h = 1e-4;
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> grad_mu;
        auto q_pert = q;
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q_pert[i]  = q[i] + h;
            const double mu_p = yoshikawa_mu(kin.arm_jacobian_full(q_pert));
            q_pert[i]  = q[i] - h;
            const double mu_m = yoshikawa_mu(kin.arm_jacobian_full(q_pert));
            q_pert[i]  = q[i];                          // restore for next axis
            grad_mu(i) = (mu_p - mu_m) / (2.0 * h);
        }
        const Eigen::Vector3d J_grad_mu = J_lin * grad_mu;
        const Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> projected =
            J_lin.transpose() * Q_ldlt.solve(J_grad_mu);
        q_dot += params.gain_mu * (grad_mu - projected);
    }

    // 4. Uniform global scaling to respect velocity limits while preserving the
    //    gradient DIRECTION (ratio of position vs orientation contributions).
    //
    //    Per-joint clipping was: clip(q̇_i, ±cap_i) — truncates small
    //    orientation terms when position saturates joints, so orientation
    //    only converges after position is nearly done.
    //
    //    Uniform scaling instead: scale = min_i(cap_i / |q̇_i|), then
    //    q̇_out = scale · q̇. The direction is preserved, so orientation
    //    is always represented proportionally to its gradient magnitude
    //    throughout the whole trajectory, not just at the end.
    const auto vlim = kin.arm_joint_velocity_limits();
    double scale = 1.0;
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
    {
        const double cap = std::min(params.max_joint_vel,
                                    std::isfinite(vlim[i]) ? vlim[i] : params.max_joint_vel);
        const double qi_abs = std::abs(q_dot(i));
        if (qi_abs > cap)
            scale = std::min(scale, cap / qi_abs);
    }
    q_dot *= scale;

    // 5. Joint-limit repulsion added AFTER scaling so it's not diluted by
    //    the scale factor — it's a safety push that must act at full strength.
    const auto lims = kin.arm_joint_position_limits();
    for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        q_dot(i) += limit_repulsion_velocity(q[i], lims[i].first, lims[i].second,
                                              params.limit_margin, params.limit_gain);

    std::array<double, Kinematics::N_ARM_JOINTS> out{};
    Eigen::Map<Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>>(out.data()) = q_dot;
    return out;
}
