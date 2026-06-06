#pragma once

#include "kinematics.h"
#include <Eigen/Dense>
#include <array>
#include <optional>

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

    /** Cap (rad/s) on the desired tool angular speed in the coordinated 6-DOF
     *  twist. Bounds the orientation half of the resolved-rate solve so a large
     *  reorientation (the camera-up roll has geodesic angle ≈ π) is a smooth,
     *  bounded turn that descends alongside the position approach rather than a
     *  joint-saturating lunge. ω_des = min(omega_max, gain_orient·angle)·axis.
     *  ~1.0 lets a 90° roll finish in ≈1.5 s. */
    double omega_max = 1.0;

    /** Desired approach direction in world frame. The tool's local z-axis
     *  (its "approach" direction) is driven to align with this. Default
     *  (0, 0, −1): tool points straight down (perpendicular to a horizontal
     *  table surface). */
    Eigen::Vector3d desired_approach{0.0, 0.0, -1.0};

    /** Selects the orientation regime (the actual gain is gain_orient):
     *   == 0  →  pin only the approach axis (tool +Z), roll free — top-down
     *           grasps. Uses the single-axis cross-product gradient.
     *   >  0  →  pin the full tool frame via a single SO(3) geodesic error
     *           toward R_des = [desired_secondary⟂, Z×X, desired_approach].
     *           The geodesic is one consistent shortest-path rotation, so a
     *           large roll correction (e.g. a 90° side-grasp) is spread across
     *           the whole reach instead of fighting the approach term and
     *           spinning the wrist at the end. */
    double gain_secondary = 0.0;

    /** Target for the tool's local +X axis (the Robotiq 2F-85 open/close
     *  direction on this chain). For a side grasp of an upright object,
     *  set this to the object's vertical axis expressed in world frame
     *  so the fingers wrap around it. */
    Eigen::Vector3d desired_secondary{1.0, 0.0, 0.0};

    /** Cap per-joint |q̇| (rad/s). Webots proto maxVelocity = 0.8727 rad/s;
     *  bridge homing uses 0.8 safely. Setting to 0.8 here matches that and
     *  roughly halves convergence time vs 0.5 (joints were already saturated). */
    double max_joint_vel = 0.87;

    /** Distance (rad) from a joint limit at which repulsion activates. */
    double limit_margin = 0.10;

    /** Repulsion strength. Higher → stronger push away from limits. */
    double limit_gain = 5.0;

    /** Damped-least-squares λ for the position step:
     *      q̇_p = −J_linᵀ (J_lin J_linᵀ + λ²I)⁻¹ · (C_pos ⊙ err).
     *  Plays two roles at once — Corke's "velocity effort" penalty (an
     *  implicit L2 cost on ‖q̇‖) and the singular-direction regulariser
     *  that keeps q̇ bounded near singularities. 0.05 is a typical starting
     *  value; raise toward 0.1–0.2 if you see twitchy near-singular q̇. */
    double dls_lambda = 0.05;

    /** Resolve the pragmatic twist via a QP (proxQP) instead of the closed-form
     *  DLS. With no inequality constraints the QP is the SAME problem
     *  (min ½q̇ᵀ(J6ᵀJ6+λ²I)q̇ − (J6ᵀξ)ᵀq̇), so behaviour is identical — this is the
     *  scaffold for migrating joint-limit / obstacle terms from soft penalties to
     *  hard QP constraints. Set from Controller.solver = "dls" | "qp". */
    bool use_qp = false;

    /** QP path only: max APPROACH speed (m/s) permitted at the edge of an obstacle's
     *  influence band, for the collision velocity-dampers (mast/table). The admissible
     *  approach speed ramps linearly from this at the band edge to 0 at the surface,
     *  a HARD non-penetration constraint replacing the soft repulsion. */
    double obs_damper_xi = 0.5;

    /** QP path only: weight on the μ/elbow LINEAR term, g_lin = −w·Σ gain·∇. The free-
     *  space redundancy contribution is (w/λ²)·gain·N·∇, so w = λ² reproduces the old
     *  null-space projection (redundancy strictly off the task), while w > λ² lets μ/elbow
     *  genuinely assert in the objective and the task slack δ yields — the real NEO trade.
     *  ≤ 0 means "use λ²" (projection-equivalent default). */
    double redundancy_weight = 0.0;

    /** Preferred-flow attractor — the continuous-active-inference upgrade from
     *  "preference over end-effector POSITION only" to a preference over
     *  position AND velocity, with the attractor at the goal (x*, ẋ=0).
     *
     *  The raw-error gradient drives q̇ ∝ −(C_pos ⊙ err), whose magnitude
     *  saturates the joints, so the EE cruises at full speed until the very
     *  end and overshoots — shoving whatever it is reaching for. Instead the
     *  model prefers a Cartesian velocity toward the target whose SPEED is a
     *  constant-deceleration profile:
     *      ẋ* = −min(v_approach, √(2·a_approach·‖err‖)) · ê
     *  i.e. cruise at v_approach far away, then decelerate at a_approach to
     *  reach ZERO speed exactly at the target — in FINITE time, the natural
     *  ease-in feel. (A proportional ramp v∝‖err‖ instead decays
     *  exponentially and crawls forever near the goal — avoid it.) The decel
     *  zone spans v_approach²/(2·a_approach).
     *
     *  ê is the unit error direction. With an ISOTROPIC C_pos this is a
     *  straight line to the target; the old {4,4,8} anisotropy (relax x/y so
     *  orientation dominates the approach axis) is no longer needed because
     *  the speed cap already keeps the position term from drowning the
     *  orientation term, and it produced an unnatural "vertical-leg-first"
     *  L-path. Set C_pos anisotropically only if you deliberately want a
     *  curved, axis-biased approach.
     *
     *  a_approach→∞, v_approach→∞ recovers the old raw-error gradient. */
    double v_approach = 0.15;  ///< Cartesian cruise-speed cap [m/s]
    double a_approach = 0.15;  ///< deceleration [m/s²]; decel zone = v_approach²/(2·a_approach)

    /** Look-ahead coarticulation. When `blend_next` is set and the EE is within
     *  `blend_radius` of the current target (a transit via-point), the COMMANDED velocity
     *  is steered continuously from "toward this via" to "toward the next waypoint" at
     *  cruise — so the EE rounds the corner instead of decelerating to 0 here and reversing
     *  direction at the next leg's handoff. (A scalar terminal speed pointed at THIS via,
     *  as an earlier carry-through did, leaves a velocity the next leg must cancel — adding
     *  the corner it meant to remove; steering toward the NEXT waypoint removes it.)
     *  No `blend_next` or `blend_radius`=0 ⇒ the original √-to-zero stop (precise terminal
     *  goal). Skill-gated by the caller (inert for a novice), so the A/B baseline is intact. */
    std::optional<Eigen::Vector3d> blend_next;
    double blend_radius = 0.0;

    /** Preference-field mode (option (c) prototype, steps 1–2). When `use_field` and
     *  `blend_next` are set, v_des is the natural-gradient resultant of a 2-via Gaussian
     *  mixture preference over {current via μ_c=x_target with precision `prec_current`,
     *  next via μ_n=blend_next with precision `prec_next`}. ONE precision per via expresses
     *  BOTH stop-vs-pass and straight-vs-bend, via λ_c = exp(−prec_current/prec_ref) ∈ (0,1]
     *  ("pass-throughness"):
     *    • high prec_current ⇒ λ_c→0 ⇒ terminal speed→0 and heading→straight at μ_c
     *      = the √-to-zero STOP (parity with the discrete FSM leg, step 1);
     *    • low  prec_current ⇒ λ_c→~1 ⇒ cruise terminal speed and heading bends toward μ_n
     *      = the look-ahead BLEND, emergent (step 2).
     *  `field_overlap` [m] localizes how near μ_c the next-via pull ramps in (the RBF width
     *  standing in for the phase variable τ of step 3). use_field=false ⇒ the blend/√ paths. */
    bool   use_field    = false;
    double prec_current = 30.0;   ///< Π_c of the current via (high ⇒ stop, low ⇒ pass through)
    double prec_next    = 30.0;   ///< Π_n of the next via (the attractor steered toward)
    double prec_ref     = 6.0;    ///< sets λ_c=exp(−Π_c/prec_ref): the precision→terminal-speed map
    double field_overlap = 0.06;  ///< [m] RBF width: how near μ_c the next-via pull ramps in

    /** Hold zone. The √ deceleration profile has INFINITE slope at the origin,
     *  so without a deadband a sub-mm error still commands several cm/s and the
     *  arm dithers/limit-cycles at the goal (amplified by target-pose noise) —
     *  bad for a stable hand-off to the next approach/grasp step. These shift
     *  the linear/angular speed ramps so they reach zero at the band edge and
     *  stay zero inside: the tool settles and holds dead still. Equivalent to
     *  giving the preference its finite tolerance width (no gradient pressure
     *  inside). arrive_deadband should be ≲ the grasp position tolerance. */
    double arrive_deadband = 0.005;  ///< m;   hold (zero linear flow) within this radius
    double orient_deadband = 0.03;   ///< rad; hold (zero angular flow) within this angle (~1.7°)

    /** α_μ — Yoshikawa-manipulability ascent gain. Adds
     *      q̇ += α_μ · N · ∂μ/∂q
     *  where μ = √det(J Jᵀ) and N = I − J_linᵀ Q⁻¹ J_lin is the soft
     *  null-space projector of the DLS position task — so the secondary
     *  objective cannot disturb the EE position. ∂μ/∂q is computed by
     *  central differences (≈ 0.5 ms / cycle). 0 disables (default).
     *  Useful range 0.3 – 1.0. */
    /** Orientation regime override for a symmetric (cylinder) grasp: pin only the
     *  tool **+Y** axis to `desired_tool_y` (the bottle's long axis) and leave the
     *  YAW about it free. Fingers stay perpendicular to the bottle (a valid grasp)
     *  while the approach azimuth is a free DOF the arm spends to stay clear of the
     *  column. Takes precedence over gain_secondary when true. */
    bool            align_tool_y = false;
    Eigen::Vector3d desired_tool_y{0.0, 0.0, 1.0};

    double gain_mu = 0.3;

    /** α_elbow — null-space elbow-placement gain. Adds
     *      q̇ += α_elbow · N · J_elbowᵀ · (p_elbow* − p_elbow)
     *  where J_elbow is the 3×7 elbow (joint_4) linear Jacobian and N is the
     *  6-DOF pose-task null-space projector (same N as the μ term), so parking
     *  the elbow cannot disturb the tool pose. Only the HORIZONTAL error is used
     *  (the term zeroes Δz), to bias the elbow "behind the mast" without forcing
     *  a height. elbow_target is the desired elbow position in world frame.
     *  0 disables (default). Mutually exclusive in practice with gain_mu (both
     *  spend the single redundant DOF). */
    double          gain_elbow = 0.0;
    Eigen::Vector3d elbow_target{0.0, 0.0, 0.0};  // null-space pull target for the elbow (world); horizontal error only

    /** Soft obstacle repulsions (potential field). Each is a quadratic ramp that
     *  is ZERO beyond its margin and grows as the obstacle is approached; added
     *  to q̇ BEFORE the velocity scaling so the command stays within limits and
     *  the constraint negotiates smoothly with the task (a soft, not hard, wall).
     *  - Column: WHOLE-ARM repulsion — every movable arm joint (j3..j7) is pushed
     *    away from the vertical column (a finite cylinder: axis col_xy, radius
     *    col_radius, z∈[col_z_lo,col_z_hi]) when its clearance to the surface
     *    drops below col_margin. Covers the wrist/forearm, not just the elbow.
     *  - Table: push the HAND (tool) up (+Z world) when its height above table_z
     *    drops below table_safe — keeps the hand from diving into the table. */
    double          gain_mast  = 0.0;
    Eigen::Vector2d col_xy{0.0, 0.0};
    double          col_radius = 0.05;
    double          col_z_lo   = -0.10;
    double          col_z_hi   =  1.30;
    double          col_margin = 0.10;
    double          gain_table = 0.0;
    double          table_z    = 0.0;
    double          table_safe = 0.05;

    /** Bottle-as-obstacle (approach safety). A finite vertical cylinder (axis bottle_xy,
     *  radius bottle_radius, z∈[bottle_z_lo,bottle_z_hi]) that bounds the TOOL's radial
     *  approach speed so the gripper can't clip/tip the target bottle. The caller enables
     *  it (gain_bottle>0) ONLY in the first/approach phase, where the tool goes to the
     *  standoff (outside the cylinder); it is OFF from Inserting on, when the gripper must
     *  move into the bottle to grasp. */
    double          gain_bottle  = 0.0;
    Eigen::Vector2d bottle_xy{0.0, 0.0};
    double          bottle_radius = 0.04;
    double          bottle_z_lo   = 0.0;
    double          bottle_z_hi   = 0.25;
    double          bottle_margin = 0.04;
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
