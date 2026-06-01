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

    // 2. Preferred TWIST: a bounded Cartesian velocity (linear + angular) that
    //    carries the whole tool frame toward the goal. Both parts use the same
    //    saturated-ramp idea so neither can saturate the joints and drag the
    //    other — the lesson from the first runs, where a separate Jᵀ orientation
    //    gradient (geodesic angle up to π) flung the EE up to reconfigure while
    //    the position term crawled at a fraction of v_approach.
    //
    //  2a. Linear: a constant-deceleration profile toward the target,
    //        v_mag = min(v_approach, √(2·a_approach·(‖err‖−d_dead)₊)),  ẋ* = −v_mag·ê.
    //      The √ ramp decelerates to ZERO in finite time (natural ease-in),
    //      unlike a proportional v∝‖err‖ ramp which crawls forever. ê is the
    //      C_pos-weighted error direction (isotropic ⇒ straight line).
    //      The arrive_deadband (d_dead) SHIFTS the √ so the speed reaches zero
    //      at radius d_dead and stays zero inside it. Without it the √ has
    //      infinite slope at the origin (dv/dd = √(2a)/(2√d) → ∞), so a 2 mm
    //      error still commands ~5 cm/s — in discrete time that overshoots,
    //      reverses, and limit-cycles, and any target-pose noise feeds the
    //      shake. The shifted √ is C⁰ (v=0 at the band edge, no jerk) and gives
    //      a quiet hold zone for the next (grasp) step. AIF reading: the
    //      preference has finite width — no gradient pressure inside tolerance.
    const Eigen::Vector3d err_pos      = pose.position - x_target;
    const Eigen::Vector3d weighted_dir = params.C_pos.cwiseProduct(err_pos);
    const double dist   = err_pos.norm();
    const double w_norm = weighted_dir.norm();
    Eigen::Vector3d v_des = Eigen::Vector3d::Zero();
    if (dist > params.arrive_deadband and w_norm > 1e-9)
    {
        const double reach = dist - params.arrive_deadband;
        const double v_mag = std::min(params.v_approach,
                                      std::sqrt(2.0 * params.a_approach * reach));
        v_des = -v_mag * (weighted_dir / w_norm);
    }

    //  2b. Angular: rotate the tool frame toward the desired orientation along
    //      the SO(3) geodesic (axis·angle of the shortest rotation), with the
    //      angular SPEED saturated at omega_max so a 90° "camera-up" roll is a
    //      bounded, coordinated turn rather than a joint-saturating lunge.
    //        gain_secondary ≤ 0 → pin only tool +Z to z_des (roll free).
    //        gain_secondary > 0 → pin the full frame R_des=[x⟂, z×x, z_des];
    //                             one consistent geodesic, no axis-fighting.
    Eigen::Vector3d omega_des = Eigen::Vector3d::Zero();
    if (params.gain_orient > 0.0)
    {
        const Eigen::Vector3d z_des = params.desired_approach.normalized();
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        double          angle = 0.0;

        Eigen::Vector3d x_des = params.desired_secondary
                                - params.desired_secondary.dot(z_des) * z_des;
        if (params.gain_secondary <= 0.0 or x_des.norm() < 1e-6)
        {
            // Approach-axis only (or secondary unusable): align tool +Z → z_des.
            const Eigen::Vector3d z_tool = pose.rotation.col(2);
            const Eigen::Vector3d c = z_tool.cross(z_des);
            const double s = c.norm();
            if (s > 1e-9)
            {
                axis  = c / s;
                angle = std::atan2(s, z_tool.dot(z_des));   // ∈ [0, π]
            }
        }
        else
        {
            // Full tool frame via the single shortest-path rotation.
            x_des.normalize();
            Eigen::Matrix3d R_des;
            R_des.col(0) = x_des;
            R_des.col(1) = z_des.cross(x_des);
            R_des.col(2) = z_des;
            const Eigen::AngleAxisd err_rot(R_des * pose.rotation.transpose());
            axis  = err_rot.axis();
            angle = err_rot.angle();                        // ∈ [0, π]
        }
        // orient_deadband mirrors the linear deadband: hold (zero angular flow)
        // within a small angle of the goal so sub-tolerance orientation noise
        // doesn't keep twisting the wrist while hovering.
        const double turn  = std::max(0.0, angle - params.orient_deadband);
        const double w_mag = std::min(params.omega_max, params.gain_orient * turn);
        omega_des = w_mag * axis;
    }

    //  2c. Coordinated 6-DOF damped-least-squares resolved-rate (Corke RVC
    //      §8.4). Stack the linear and angular Jacobians and the desired twist
    //      and solve ONCE: q̇ = J6ᵀ (J6 J6ᵀ + λ²I₆)⁻¹ · [v_des; ω_des]. Because
    //      both tasks come out of the same operator, position and orientation
    //      descend together — the orientation joints no longer drag the EE off
    //      target and the position term no longer fights them. λ bounds q̇ near
    //      singularities and folds in Corke's velocity-effort penalty. The
    //      factorisation is reused for the null-space projector below.
    Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS> J6;
    J6.topRows<3>()    = J_lin;
    J6.bottomRows<3>() = J_ang;
    Eigen::Matrix<double, 6, 1> twist;
    twist.head<3>() = v_des;
    twist.tail<3>() = omega_des;
    const double lambda_sq = params.dls_lambda * params.dls_lambda;
    const Eigen::Matrix<double, 6, 6> Q6 =
        J6 * J6.transpose() + lambda_sq * Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> Q6_ldlt(Q6);
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> q_dot =
        J6.transpose() * Q6_ldlt.solve(twist);

    // 3. Manipulability ascent in the null space of the 6-DOF pose task (Corke
    //    RVC §8.4 redundancy resolution). A 7-DOF arm on a 6-DOF task has one
    //    redundant DOF; the soft projector N = I − J6ᵀ Q6⁻¹ J6 confines this
    //    term to it so it cannot disturb the pose. ∂μ/∂q by central differences.
    //    N.B. arm_jacobian_full(q±h) mutates kin.data_, so this MUST run after
    //    every read of J/J_lin/J_ang (all local copies, so fine).
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
        const Eigen::Matrix<double, 6, 1> J6_grad_mu = J6 * grad_mu;
        const Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> projected =
            J6.transpose() * Q6_ldlt.solve(J6_grad_mu);
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
