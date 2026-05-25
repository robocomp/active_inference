#pragma once

#include "kinematics.h"
#include <Eigen/Dense>
#include <array>

/**
 * Continuous-action EFE-gradient controller for the Kinova arm.
 *
 * Under a Gaussian preference p̃(o) on the end-effector position with mean
 * x_target and precision C_inv = (1/σ²) I_3, the instrumental EFE term is
 *
 *   G_instr(q) = ½ (f(q) − x*)ᵀ C_inv (f(q) − x*)            (+ const)
 *
 * and its gradient w.r.t. the seven arm joints is
 *
 *   ∂G_instr/∂q_arm = J_lin(q)ᵀ C_inv (f(q) − x*)
 *
 * where J_lin is the 3×7 linear part of the tool_frame Jacobian, restricted
 * to the arm columns. Action selection becomes gradient descent:
 *
 *   q̇*_arm = −α · J_linᵀ C_inv (f(q) − x*)
 *
 * This file adds a quadratic joint-limit repulsion term for the three
 * revolute arm joints (joint_2, joint_4, joint_6); continuous joints are
 * ignored. Final command is clipped to the URDF's per-joint velocity limits.
 */
struct EFEParams
{
    /** α — scalar position gain. Equivalent to a diagonal C_inv = α I. */
    double gain_pos = 2.0;

    /** Cap per-joint |q̇| (rad/s). Defaults below URDF's ~1.4 rad/s for safety. */
    double max_joint_vel = 0.5;

    /** Distance (rad) from a joint limit at which repulsion activates. */
    double limit_margin = 0.10;

    /** Repulsion strength. Higher → stronger push away from limits. */
    double limit_gain = 5.0;
};

/**
 * One step of EFE-gradient on the arm.
 *
 * @param kin      Kinematics wrapper (mutated internally by FK/Jacobian).
 * @param q        Current 7 arm joint angles, rad.
 * @param x_target Desired tool_frame position, world frame, m.
 * @param params   Gain/limit parameters.
 *
 * @return         7-vector of commanded arm joint velocities, rad/s, clipped.
 *                 Index order matches q (joint_1..joint_7).
 */
std::array<double, Kinematics::N_ARM_JOINTS> efe_gradient_step(
    Kinematics& kin,
    const std::array<double, Kinematics::N_ARM_JOINTS>& q,
    const Eigen::Vector3d& x_target,
    const EFEParams& params = {});
