#include "efe_gradient.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <proxsuite/proxqp/dense/dense.hpp>

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

    using QpRow = Eigen::Matrix<double, 1, Kinematics::N_ARM_JOINTS>;

    /// Append one inequality row  lo ≤ aᵀq̇ ≤ hi  to a growing (C, l, u) block.
    void append_row(Eigen::MatrixXd& C, Eigen::VectorXd& l, Eigen::VectorXd& u,
                    const QpRow& a, double lo, double hi)
    {
        const int r = static_cast<int>(C.rows());
        C.conservativeResize(r + 1, Kinematics::N_ARM_JOINTS); C.row(r) = a;
        l.conservativeResize(r + 1); l(r) = lo;
        u.conservativeResize(r + 1); u(r) = hi;
    }

    /// Joint-limit velocity dampers (Haviland & Corke). For a finite-limit joint inside
    /// the influence band d_infl: q̇_i ≤ η·(q^max−q_i)/d_infl and q̇_i ≥ −η·(q_i−q^min)/d_infl,
    /// ramping the admissible velocity toward the wall to 0 (η = full joint speed at the
    /// band edge, d_s = 0). Rows added ONLY for joints inside the band ⇒ far from limits
    /// no constraint = identical to DLS. Continuous joints (±inf) are never constrained.
    void add_joint_limit_dampers(const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                                 const std::array<std::pair<double, double>, Kinematics::N_ARM_JOINTS>& lims,
                                 double d_infl, double eta,
                                 Eigen::MatrixXd& C, Eigen::VectorXd& l, Eigen::VectorXd& u)
    {
        constexpr int N = Kinematics::N_ARM_JOINTS;
        constexpr double INF = 1e20;
        for (int i = 0; i < N; ++i)
        {
            if (not std::isfinite(lims[i].first) or not std::isfinite(lims[i].second))
                continue;                                  // continuous joint: no wall
            double li = -INF, ui = INF;
            const double d_up = lims[i].second - q[i];
            const double d_lo = q[i] - lims[i].first;
            if (d_up < d_infl) ui =  eta * (d_up / d_infl); // → 0 at wall, < 0 if overshot
            if (d_lo < d_infl) li = -eta * (d_lo / d_infl);
            if (ui < INF or li > -INF)
            {
                QpRow a = QpRow::Zero(); a(i) = 1.0;
                append_row(C, l, u, a, li, ui);
            }
        }
    }

    /// Obstacle collision velocity-dampers (Haviland & Corke), replacing the soft
    /// potential-field repulsions on the QP path. For a robot point near an obstacle,
    /// bound its APPROACH speed:  −n̂ᵀ J_point q̇ ≤ ξ·clr/d_infl  (clamped ≥ 0 so q̇=0 is
    /// always feasible — a "no deeper penetration" guarantee, not active retreat), where
    /// n̂ is the outward surface normal and clr the clearance. Admissible approach speed
    /// ramps to 0 at the surface.
    void add_obstacle_dampers(Kinematics& kin,
                              const std::array<double, Kinematics::N_ARM_JOINTS>& q,
                              const Kinematics::ToolPose& pose,
                              const Eigen::Matrix<double, 3, Kinematics::N_ARM_JOINTS>& J_lin,
                              const EFEParams& p,
                              Eigen::MatrixXd& C, Eigen::VectorXd& l, Eigen::VectorXd& u)
    {
        constexpr int N = Kinematics::N_ARM_JOINTS;
        constexpr double INF = 1e20;
        // Mast: every movable joint (j3..j7, idx 2..6) vs the finite vertical cylinder.
        if (p.gain_mast > 0.0)
            for (int j = 2; j <= 6; ++j)
            {
                const Eigen::Vector3d pj = kin.joint_position(q, j);
                const double cz = std::clamp(pj.z(), p.col_z_lo, p.col_z_hi);
                const Eigen::Vector3d colpt(p.col_xy.x(), p.col_xy.y(), cz);
                const Eigen::Vector3d diff = pj - colpt;
                const double dist = diff.norm();
                const double clr  = dist - p.col_radius;                 // clearance to surface
                if (clr < p.col_margin and dist > 1e-6)
                {
                    const Eigen::Vector3d n = diff / dist;               // outward normal
                    const Eigen::Matrix<double, 3, N> Jj = kin.arm_jacobian_joint_linear(q, j);
                    const QpRow a = -(n.transpose() * Jj);               // approach speed = a·q̇
                    const double zeta = std::max(0.0, p.obs_damper_xi * (clr / p.col_margin));
                    append_row(C, l, u, a, -INF, zeta);
                }
            }
        // Table: keep the arm above the horizontal plane (approach = downward, +Z normal).
        if (p.gain_table > 0.0)
        {
            // Tool point.
            const double d = pose.position.z() - p.table_z;
            if (d < p.table_safe)
            {
                const QpRow a = -(Eigen::RowVector3d(0, 0, 1) * J_lin);
                const double zeta = std::max(0.0, p.obs_damper_xi * (d / p.table_safe));
                append_row(C, l, u, a, -INF, zeta);
            }
            // ELBOW-only table damper. A blanket per-joint plane constraint froze the grasp
            // descent (the WRIST legitimately nears the table on a low reach), but the elbow
            // (joint_4) never needs to touch the table even while grasping — the null-space
            // elbow-placement pull can swing it down into the surface. Bound just the elbow
            // point's downward speed near the plane; q̇=0 stays feasible (no penetration), and
            // the tool/wrist are untouched so this does not stall the descent.
            //
            // CRITICAL: a SMALL dedicated margin. The elbow legitimately holds only ~6 cm
            // above the table at the converged standoff, so using the tool's table_safe (6 cm)
            // makes the damper bind at the HOLD pose — and since position is only a soft slack
            // in the NEO QP, it then yields to the hard constraint and the EE drifts off. A
            // 3 cm band sits below the hold clearance (no fight at convergence) yet still
            // arrests a genuine dive toward the surface.
            constexpr double ELBOW_TABLE_SAFE = 0.03;
            const double de = kin.elbow_position(q).z() - p.table_z;
            if (de < ELBOW_TABLE_SAFE)
            {
                const Eigen::Matrix<double, 3, N> Je = kin.arm_jacobian_elbow_linear(q);
                const QpRow a = -(Eigen::RowVector3d(0, 0, 1) * Je);   // elbow downward speed
                const double zeta = std::max(0.0, p.obs_damper_xi * (de / ELBOW_TABLE_SAFE));
                append_row(C, l, u, a, -INF, zeta);
            }
        }
        // Bottle (approach phase only): TOOL point vs the target's finite vertical cylinder.
        // Bounds the tool's radial approach speed so the gripper rounds to the standoff
        // without clipping/tipping the bottle; clr→0 still lets it reach the surface (the
        // grasp itself runs with this OFF). Same velocity-damper form as the mast.
        if (p.gain_bottle > 0.0)
        {
            const double cz = std::clamp(pose.position.z(), p.bottle_z_lo, p.bottle_z_hi);
            const Eigen::Vector3d c(p.bottle_xy.x(), p.bottle_xy.y(), cz);
            const Eigen::Vector3d diff = pose.position - c;     // ≈ horizontal (radial)
            const double dist = diff.norm();
            const double clr  = dist - p.bottle_radius;
            if (clr < p.bottle_margin and dist > 1e-6)
            {
                const Eigen::Vector3d n = diff / dist;
                const QpRow a = -(n.transpose() * J_lin);
                const double zeta = std::max(0.0, p.obs_damper_xi * (clr / p.bottle_margin));
                append_row(C, l, u, a, -INF, zeta);
            }
        }
    }

    /// Canonical NEO (Haviland & Corke) reactive QP. Decision vars x = [q̇; δ], δ∈ℝ⁶ the
    /// task slack:
    ///   min ½(λ² ‖q̇‖² + ‖δ‖²) + g_linᵀ q̇   s.t.  J6 q̇ − δ = ξ,  l ≤ C_damp q̇ ≤ u.
    /// - Q = diag(λ²I₇, I₆): joint-velocity effort + slack regularisers (λ_δ = 1).
    /// - g_lin = −λ²·Σ gain·∇(redundancy): μ-ascent + elbow as the LINEAR objective term.
    ///   With this weight and NO active inequality the solution is exactly the DLS task
    ///   plus the null-space-projected redundancy (since (J6ᵀJ6+λ²I)⁻¹ = (1/λ²)·N), so it
    ///   reproduces the pre-QP controller in free space. When a damper binds, δ and the
    ///   redundancy terms yield optimally to respect the HARD constraint — now on the full
    ///   q̇, which is what makes the joint-limit / collision guarantees exact.
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>
    solve_neo_qp(const Eigen::Matrix<double, 6, Kinematics::N_ARM_JOINTS>& J6,
                 const Eigen::Matrix<double, 6, 1>& twist,
                 double lambda_sq,
                 const Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>& g_lin,
                 const Eigen::MatrixXd& C_damp, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                 double orient_slack = 1.0)
    {
        constexpr int N = Kinematics::N_ARM_JOINTS;
        const int dim  = N + 6;                            // q̇ (7) + slack δ (6)
        const int n_in = static_cast<int>(C_damp.rows());

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        H.topLeftCorner(N, N) = lambda_sq * Eigen::MatrixXd::Identity(N, N);      // λ² I on q̇
        // Slack weight: 1 on the 3 position rows, orient_slack on the 3 orientation
        // rows. orient_slack < 1 makes leaving orientation error CHEAP, so the QP
        // gives the rotation slack instead of driving q̇ into a singularity to hit it.
        Eigen::Matrix<double, 6, 1> w_delta;
        w_delta << 1.0, 1.0, 1.0, orient_slack, orient_slack, orient_slack;
        H.bottomRightCorner(6, 6) = w_delta.asDiagonal();                         // W_δ on δ
        Eigen::VectorXd g = Eigen::VectorXd::Zero(dim);
        g.head(N) = g_lin;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, dim);                        // [J6 | −I₆]
        A.leftCols(N)  = J6;
        A.rightCols(6) = -Eigen::MatrixXd::Identity(6, 6);
        const Eigen::VectorXd b = twist;

        proxsuite::proxqp::dense::QP<double> qp(dim, 6, n_in);
        qp.settings.eps_abs = 1e-10;
        qp.settings.eps_rel = 0.0;
        qp.settings.verbose = false;
        if (n_in == 0)
            qp.init(H, g, A, b, proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt);
        else
        {
            Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_in, dim);                 // [C_damp | 0]
            C.leftCols(N) = C_damp;
            qp.init(H, g, A, b, C, l, u);
        }
        qp.solve();
        return qp.results.x.head(N);
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
    if (params.use_field and params.blend_next.has_value())
    {
        // ── Preference-field twist (option (c), steps 1–2) ──────────────────────────────
        // A 2-via Gaussian-mixture preference: μ_c=x_target (Π_c), μ_n=blend_next (Π_n). The
        // precision-weighted natural-gradient resultant gives the HEADING, and λ_c =
        // exp(−Π_c/prec_ref) ∈ (0,1] ("pass-throughness" of μ_c) sets the TERMINAL speed.
        //   • Π_c high ⇒ λ_c→0 ⇒ v_term→0 and the next-via term vanishes ⇒ straight to μ_c,
        //     √-decelerate to a stop = PARITY with the discrete leg (step 1).
        //   • Π_c low  ⇒ λ_c→~1 ⇒ cruise v_term and the heading bends toward μ_n near μ_c
        //     ⇒ the look-ahead blend, EMERGENT (step 2).
        // w_n (RBF in dist, width field_overlap) localizes the bend near μ_c — the Euclidean
        // stand-in for the phase variable τ (step 3). Both limits reduce to known-good code.
        const Eigen::Vector3d to_c = -err_pos;                                  // EE → μ_c
        const Eigen::Vector3d to_n = params.blend_next.value() - pose.position; // EE → μ_n
        const double lambda_c = std::exp(-params.prec_current / std::max(1e-9, params.prec_ref));
        const double w_n = std::exp(-(dist * dist) /
                                    (params.field_overlap * params.field_overlap));
        const Eigen::Vector3d g = params.prec_current * to_c
                                + lambda_c * w_n * params.prec_next * to_n;
        const double gn = g.norm();
        const double v_term = params.v_approach * lambda_c;                     // terminal speed at μ_c
        const double reach  = dist - params.arrive_deadband;
        const double v_mag  = std::min(params.v_approach,
            std::sqrt(v_term * v_term + 2.0 * params.a_approach * std::max(0.0, reach)));
        if (gn > 1e-9)
            v_des = v_mag * (g / gn);
    }
    else if (params.blend_next.has_value() and params.blend_radius > 0.0 and dist < params.blend_radius)
    {
        // Coarticulation: inside the blend zone rotate the commanded velocity from "into
        // this via" toward "toward the next waypoint" at cruise, so the direction is
        // CONTINUOUS through the corner — the EE rounds it instead of stopping. s goes 1 at
        // the zone edge → 0 at the via, so v_des goes dir_in → dir_out (tangent at the via).
        const Eigen::Vector3d dir_out = (params.blend_next.value() - x_target).normalized();  // via → next
        const Eigen::Vector3d dir_in  = (dist > 1e-9) ? (-err_pos / dist) : dir_out;          // EE → via
        const double s = std::clamp(dist / params.blend_radius, 0.0, 1.0);
        const Eigen::Vector3d blended = s * dir_in + (1.0 - s) * dir_out;
        const double bn = blended.norm();
        if (bn > 1e-9)
            v_des = params.v_approach * (blended / bn);
    }
    else if (dist > params.arrive_deadband and w_norm > 1e-9)
    {
        const double reach = dist - params.arrive_deadband;
        // Shifted √ ramp decelerating to ZERO at the deadband edge: a precise terminal stop.
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
        if (params.align_tool_y)
        {
            // Pin tool +Y to the bottle axis (fingers ⟂ bottle), YAW FREE: single-
            // axis alignment on tool +Y, so the approach azimuth is left to the
            // redundancy (the arm spends it to stay clear of the column).
            const Eigen::Vector3d y_des  = params.desired_tool_y.normalized();
            const Eigen::Vector3d y_tool = pose.rotation.col(1);
            const Eigen::Vector3d c = y_tool.cross(y_des);
            const double s = c.norm();
            if (s > 1e-9)
            {
                axis  = c / s;
                angle = std::atan2(s, y_tool.dot(y_des));   // ∈ [0, π]
            }
        }
        else if (params.gain_secondary <= 0.0 or x_des.norm() < 1e-6)
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
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> Q6_ldlt(Q6);   // also reused below for N
    const auto pos_lims = kin.arm_joint_position_limits();

    // Redundancy gradients (RAW, unprojected): manipulability μ-ascent and the elbow
    // posture pull. Computed once; the DLS path projects them into the task null space
    // and adds them post-solve, while the QP path folds them into the NEO linear term
    // g_lin. ∂μ/∂q by central differences. N.B. arm_jacobian_full(q±h) mutates kin.data_,
    // so this runs after every read of J/J_lin/J_ang (all local copies, so fine).
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> grad_mu =
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>::Zero();
    if (params.gain_mu > 0.0)
    {
        constexpr double h = 1e-4;
        auto q_pert = q;
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
        {
            q_pert[i] = q[i] + h; const double mu_p = yoshikawa_mu(kin.arm_jacobian_full(q_pert));
            q_pert[i] = q[i] - h; const double mu_m = yoshikawa_mu(kin.arm_jacobian_full(q_pert));
            q_pert[i] = q[i];                              // restore for next axis
            grad_mu(i) = (mu_p - mu_m) / (2.0 * h);
        }
    }
    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> grad_el =
        Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>::Zero();
    if (params.gain_elbow > 0.0)
    {
        const Eigen::Matrix<double, 3, Kinematics::N_ARM_JOINTS> J_el = kin.arm_jacobian_elbow_linear(q);
        Eigen::Vector3d err = params.elbow_target - kin.elbow_position(q);
        err.z() = 0.0;                                     // horizontal placement (back-right)
        grad_el = J_el.transpose() * err;
    }

    Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> q_dot;
    if (params.use_qp)
    {
        // NEO QP (step 4): task as a slacked equality, μ + elbow as the linear objective
        // term, joint-limit + obstacle velocity-dampers as hard inequalities. In free
        // space (no damper active) g_lin = −λ²·Σ gain·∇ reproduces DLS-plus-projection
        // exactly; when a damper binds, the slack + redundancy yield to the HARD
        // constraint on the full q̇.
        Eigen::MatrixXd C(0, Kinematics::N_ARM_JOINTS);
        Eigen::VectorXd l(0), u(0);
        add_joint_limit_dampers(q, pos_lims, params.limit_margin, params.max_joint_vel, C, l, u);
        add_obstacle_dampers(kin, q, pose, J_lin, params, C, l, u);
        // w = λ² reproduces the null-space projection; w > λ² lets μ/elbow assert and the
        // task slack yield (genuine NEO). ≤ 0 ⇒ projection-equivalent default.
        const double w = params.redundancy_weight > 0.0 ? params.redundancy_weight : lambda_sq;
        const Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1> g_lin =
            -w * (params.gain_mu * grad_mu + params.gain_elbow * grad_el);
        q_dot = solve_neo_qp(J6, twist, lambda_sq, g_lin, C, l, u, params.orient_slack);
    }
    else
    {
        // DLS path (fallback): closed-form pragmatic term + null-space-projected
        // redundancy. The soft projector N = I − J6ᵀ Q6⁻¹ J6 confines μ/elbow to the
        // redundant DOF so they cannot disturb the tool pose.
        q_dot = (J6.transpose() * Q6_ldlt.solve(twist)).eval();
        if (params.gain_mu > 0.0)
            q_dot += params.gain_mu * (grad_mu - J6.transpose() * Q6_ldlt.solve(J6 * grad_mu));
        if (params.gain_elbow > 0.0)
            q_dot += params.gain_elbow * (grad_el - J6.transpose() * Q6_ldlt.solve(J6 * grad_el));
    }

    // 3c. Soft obstacle repulsions (potential field), added before scaling so the
    //     command stays velocity-bounded and the constraint negotiates with the
    //     task. Quadratic ramp, exactly zero outside the margin.
    //     DLS path only: the QP path (step 3) enforces these as hard collision
    //     velocity-damper inequalities inside the solve (add_obstacle_dampers), so the
    //     soft push would double-count here.
    if (not params.use_qp and params.gain_mast > 0.0)  // WHOLE-ARM ↔ vertical column
    {
        // Repel every movable arm joint (j3..j7, idx 2..6 — skip the shoulder/
        // mount which is always at the column) away from the finite cylinder.
        // Applied directly (not null-space): if the wrist nears the column the
        // hand must back off, which is the intended soft-wall behaviour.
        for (int j = 2; j <= 6; ++j)
        {
            const Eigen::Vector3d p_j = kin.joint_position(q, j);
            const double cz = std::clamp(p_j.z(), params.col_z_lo, params.col_z_hi);
            const Eigen::Vector3d colpt(params.col_xy.x(), params.col_xy.y(), cz);
            const Eigen::Vector3d diff = p_j - colpt;
            const double dist = diff.norm();
            const double clr  = dist - params.col_radius;        // clearance to the surface
            if (clr < params.col_margin and dist > 1e-6)
            {
                const auto J_j = kin.arm_jacobian_joint_linear(q, j);
                const double e = (params.col_margin - clr) / params.col_margin;   // (0,1]
                const Eigen::Vector3d v = diff / dist;                            // push away
                q_dot += params.gain_mast * e * e * (J_j.transpose() * v);
            }
        }
    }
    if (not params.use_qp and params.gain_table > 0.0)  // hand ↔ table top
    {
        const double d = pose.position.z() - params.table_z;
        if (d < params.table_safe)
        {
            const double e = std::clamp((params.table_safe - d) / params.table_safe, 0.0, 1.0);
            q_dot += params.gain_table * e * e * (J_lin.transpose() * Eigen::Vector3d(0, 0, 1));
        }
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

    // 5. Joint-limit repulsion added AFTER scaling so it's not diluted by the scale
    //    factor — a safety push that must act at full strength. DLS path only: the QP
    //    path (step 2) enforces joint limits as hard velocity-damper inequalities
    //    inside the solve instead, so the soft push would double-count here.
    if (not params.use_qp)
        for (int i = 0; i < Kinematics::N_ARM_JOINTS; ++i)
            q_dot(i) += limit_repulsion_velocity(q[i], pos_lims[i].first, pos_lims[i].second,
                                                 params.limit_margin, params.limit_gain);

    std::array<double, Kinematics::N_ARM_JOINTS> out{};
    Eigen::Map<Eigen::Matrix<double, Kinematics::N_ARM_JOINTS, 1>>(out.data()) = q_dot;
    return out;
}
