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
    /** Per-axis position precision (the C⁻¹ diagonal in AIF terms).
     *  Different weights per axis let orientation converge without fighting
     *  the position gradient: relax x/y (arm reaches there easily) and
     *  tighten z (approach direction matters most for table-surface tasks).
     *  This is the principled fix for slow orientation convergence: once
     *  x/y precision is relaxed, the orientation gradient dominates and
     *  drives the tool to pointing down without being cancelled by position.
     *  Default: x=4, y=4, z=8 — twice as precise on approach axis. */
    Eigen::Vector3d C_pos{4.0, 4.0, 8.0};

    /** α_o — scalar orientation gain on the partial alignment cost
     *  C_orient = 1 − z_tool · z_des. Restored to 1.0 (was lowered to 0.3
     *  to fight chatter caused by the old scalar gain_pos at 100 ms period;
     *  with anisotropic C_pos and 20 ms period the conflict is resolved). */
    double gain_orient = 1.0;

    /** Desired approach direction in world frame. The tool's local z-axis
     *  (its "approach" direction) is driven to align with this. Default
     *  (0, 0, −1): tool points straight down (perpendicular to a horizontal
     *  table surface). */
    Eigen::Vector3d desired_approach{0.0, 0.0, -1.0};

    /** Cap per-joint |q̇| (rad/s). Webots proto maxVelocity = 0.8727 rad/s;
     *  bridge homing uses 0.8 safely. Setting to 0.8 here matches that and
     *  roughly halves convergence time vs 0.5 (joints were already saturated). */
    double max_joint_vel = 0.87;

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
