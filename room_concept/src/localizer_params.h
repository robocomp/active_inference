#pragma once
// Split out of room_concept.h (refactor step 3, docs/refactor_proposal_2026-10-13.md). The types live in
// rc::loc and RoomConcept re-exports each under its old name (RoomConcept::Params), so no caller changed.
// Contents: Params.

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "object_anchor_factor.h"
#include "image_edge_factor.h"
#include "se2_preintegration.h"
#include "motion_calibration.h"    // rc::calib::Config
#include "room_boxes_channel.h"   // rc::boxch::Params
#include "wall_map.h"             // wallmap::Params
#include "wall_segmenter.h"       // wallseg::Params

namespace rc::loc
{
    struct Params
    {
        float learning_rate_pos = 0.05f;  // Uniform LR for all pose DoF (x, y, θ)

        float wall_thickness = 0.1f;
        float wall_height = 2.4f;        // meters
        int max_lidar_points = 1000;      // Subsample for speed

        // ===== Sliding Window (RFE) =====
        // Paper: "Total-Time Active Inference" - Realized Free Energy over a past window
        // W=1 degenerates to the old single-step behaviour
        int rfe_window_size = 10;             // Number of past timesteps to retain
        int rfe_max_lidar_per_old_slot = 300;  // Subsample older slots to save compute
        float rfe_obs_sigma = 0.05f;           // σ_obs for SDF observation noise (m)
        float rfe_huber_delta = 0.15f;         // Huber threshold (m)

        // ===== Boundary Prior Quality Gate =====
        // When the previous frame's localization was poor (sdf_mse_prev > sigma_sdf),
        // the boundary prior anchors ADAM to a bad pose, preventing convergence.
        // This gate scales the boundary prior weight by:
        //   w = min(1, (sigma_sdf / sqrt(sdf_mse_prev))^2)
        // so a well-localised prior (sdf_mse ≈ 0) has full weight (w≈1),
        // and a bad prior (sdf_mse >> sigma_sdf) is suppressed (w→0).
        // Set to false to use fixed weight=1 (legacy behaviour).
        bool  rfe_boundary_quality_gate = true;

        // ===== Hierarchical precision on the boundary prior (HIERARCHICAL_PRECISION.md) =====
        // When enabled, boundary_weight is no longer min(1, σ_sdf²/sdf_mse_prev). Instead the boundary
        // precision scale π=exp(u_b) is INFERRED (Gamma/log-precision hyperprior) from the boundary
        // factor's own residual r_b, and a slow in-process hyper-state v (map_trust, Option A) predicts
        // u via g(v)=u0+g_gain·v. Default OFF ⇒ identical behaviour to the legacy quality gate.
        bool  hier_prec_boundary_enabled = false;
        float hier_prec_u0        = 0.0f;   // log-precision prior mean (exp(0)=1 → full prior)
        float hier_prec_sigma_u2  = 1.0f;   // how far r_b may pull u from g(v)
        float hier_prec_lr_u      = 0.1f;   // π-step size (fast, per-frame)
        float hier_prec_g_gain    = 1.0f;   // k in g(v)=u0+k·v (top-down map)
        float hier_prec_lr_v      = 0.02f;  // v-step size (slow hyper-state)
        float hier_prec_sigma_v2  = 1.0f;   // prior scale on the map_trust state v


        // ===== Far-point distance weighting =====
        // Far points have a longer lever arm for orientation correction and are
        // under-represented relative to their informational value (lidar point
        // density decreases with range).  When enabled, each point's SDF residual
        // is scaled by  w_i = dist_i / mean(dist)  before averaging, so distant
        // points contribute proportionally more to the gradient.
        // The normalisation keeps the total loss magnitude unchanged.
        bool  far_points_weight = false;       // enable distance-proportional weighting
        // w_i = (dist_i^α / mean(dist^α)), clamped to [far_points_min_weight, ∞), re-normalised.
        // α=1: linear (original); α=2: quadratic boost; α>2: aggressive emphasis on far points.
        // Normalisation preserves total loss magnitude regardless of α.
        float far_points_exponent   = 1.0f;  // α — exponent of the distance weight
        float far_points_min_weight = 0.1f;  // floor weight to avoid silencing near points

        // ===== Incidence-angle weighting =====
        // Grazing hits are more ambiguous against the nearest-face model, so reduce
        // their contribution using the cosine of the angle between the ray and the
        // closest wall normal. The sign of the normal is irrelevant because |dot| is used.
        bool  incidence_angle_weight = false;
        float incidence_angle_exponent = 1.0f;   // β — exponent of the incidence weight
        float incidence_angle_min_weight = 0.2f; // floor weight for grazing rays

        // GPU/CPU selection
        // Note: For small tensors (~200 points), CPU is faster due to GPU transfer overhead
        bool optimizer_timing_csv = true;   // lean per-update timing CSV (etc/optimizer_timing.csv)

        // ===== Prediction-based Early Exit =====
        // If predicted pose already has low SDF error, skip optimization entirely
        // This saves CPU when motion model is accurate (smooth motion)
        bool prediction_early_exit = true;
        float sigma_sdf = 0.15f;              // SDF observation noise (15cm)
        float prediction_trust_factor = 0.5f; // Base threshold = sigma_sdf * factor (~7.5cm)
        int min_tracking_steps = 20;          // Wait for system to stabilize before early exit

        // ===== Rotation-adaptive Early Exit =====
        // During rotation, a small theta error (e.g. 0.02 rad) produces large SDF displacements
        // at room scale (5m × 0.02 rad ≈ 10 cm).  The base threshold (7.5 cm) is too tight.
        // This coupling factor widens the trust threshold proportionally to the measured rotation:
        //   threshold = sigma_sdf * trust_factor + rotation_sdf_coupling * |delta_theta|
        // Set to 0 to disable (reverts to fixed threshold).
        float rotation_sdf_coupling = 0.5f;   // m/rad — extra SDF tolerance per radian of rotation

        // ===== Differential Test (A/B comparison) =====

        // ===== Velocity-Adaptive Gradient Weights =====
        // Adjust optimization emphasis based on current motion profile
        bool velocity_adaptive_weights = true;
        float linear_velocity_threshold = 0.05f;   // m/s - below this = "not moving linearly"
        float angular_velocity_threshold = 0.1f;  // rad/s - used for velocity-adaptive gradient weights
                                                   // (boost theta gradient when rotating).
                                                   // NOTE: keep well below max robot rotation speed (~1 rad/s).
                                                   // Previously 1.0 rad/s which was never triggered in practice.
        float weight_boost_factor = 2.0f;          // Multiplier for emphasized parameters
        float weight_reduction_factor = 0.5f;      // Multiplier for de-emphasized parameters
        float weight_smoothing_alpha = 0.3f;       // EMA smoothing for weight transitions

        // ===== Dual-Prior Fusion (command + odometry) =====
        // Master switch for the COMMAND (joystick / controller) channel as a motion prior. On, the
        // selection is the historical one: Fused when both channels are fresh, else
        // Measured, else Command, else FallbackZero. Off, the command channel is still COMPUTED and
        // LOGGED (cmd_* columns, cmd_valid/cmd_fresh keep their real values) but never enters the
        // prediction, so the selection collapses to Measured / FallbackZero and `motion_prior_source`
        // in ai2_log.csv reads "measured" instead of "fused" — that column is how you check the flag
        // actually took, since the cmd_* columns deliberately keep reporting what the channel WOULD
        // have said. Computing it regardless is not waste: compute_odometry_prior() also advances
        // last_lidar_timestamp, and skipping the call would break that bookkeeping.
        // Purpose: an open-loop command is a statement of INTENT, not of motion — the wheels may not
        // have obeyed it (slip, stall, a blocked robot). Turning it off asks what the encoder channel
        // alone is worth, which is the A/B the fusion weights were never measured against.
        // DEFAULT OFF since 2026-09-18. Measured on Shadow in Webots room 2, with the LiDAR scans as
        // the witness (range-vs-bearing profile alignment, which needs neither odometry nor the
        // supervisor): over a 34 deg pivot the fused prediction turned 49.3 deg, a gain of 1.41 with
        // 1.67 deg rms about it, and left 2.5 deg of permanent heading error after the pivot returned.
        // Translation over the same recording was right to 2%, so this is the rotation-only signature
        // of following INTENT rather than motion. The config's own note (etc/config.toml:918) had
        // already flagged this as the test to run before blaming the wheels; P3Bot measured +1.70% and
        // set it false. The base is well calibrated — a prior that asserts the commanded turn against
        // a base that does not execute it is subtracting information, not adding it.
        bool use_command_velocity_prior = false;

        // ===== Prior covariance model =====
        // Process noise for commanded velocity prior (open-loop, less reliable)
        float cmd_noise_trans = 0.20f;   // Fractional position noise per meter of motion
        // Fractional rotation noise per radian of COMMANDED rotation. 0.18, not 0.10, kept on
        // measured outcome: it shifts fusion weight to the encoder (the better per-frame predictor),
        // and early exit while rotating went 91.6% -> 95.6% with over-gate frames 3.6% -> 0.5% on
        // matched populations. It is NOT the command's measured error -- properly integrated (the
        // localizer's own cmd_dth/meas_dth, not a re-integration of the velocity columns, which
        // mangles the sparse stepwise command signal) the command runs ~10% below the encoder and
        // ~3% below the room-anchored posterior. See config.toml for the full retraction.
        float cmd_noise_rot   = 0.18f;
        float cmd_noise_base  = 0.05f;   // Base position noise even when stationary (m)
        float stationary_noise_damping = 0.7f;  // Multiplier applied to base noise when near-stationary

        // Process noise for measured odometry prior (encoder/IMU, more reliable)
        float odom_noise_trans = 0.08f;  // Fractional position noise per meter of motion
        float odom_noise_rot   = 0.04f;  // Fractional rotation noise per radian of rotation
        float odom_noise_base  = 0.01f;  // Base position noise even when stationary (m)
        float odom_noise_scale = 1.0f;   // Multiplier on all odom noise params (>1 simulates worse odometry)

        // ===== Encoder angular slip model =====
        // Fraction of the rotation increment that slips, so the measured-odometry rotation variance
        // gains
        //   rot_var_extra = (encoder_rot_slip_k * |delta_theta|)^2
        // Was (encoder_rot_slip_k * |vel_rot|)^2 — a rate used as an angle, inflating sigma by 1/dt
        // (~16x at the 62 ms update interval); see build_motion_covariance for the measurement.
        //
        // NOTE the premise above this line was also backwards. It read "encoders under-report
        // rotation at high angular speed, so make the fusion trust the COMMAND more". The command is
        // the WORSE per-frame predictor of the two, not the better one: with this term inflated the
        // prior collapsed onto it and the SDF had to push heading further into the turn on 100% of
        // frames. So if either prior deserves an inflation term growing with |omega| it is the
        // command (CmdNoiseRot), not this one. Set to 0 to disable.
        // (An earlier version of this note quoted "encoder 1.068, command 0.747 of true rotation".
        // Those ratios were measurement artefacts — corrected figures are cmd/post 0.967 and
        // meas/post 1.072; see the retraction in config.toml. Note also that under
        // webots-bridge the reported angular speed is the supervisor's GROUND TRUTH
        // (specificworker.cpp: pose_data.rot = shadow_velocity[5]), not wheel odometry, so in sim
        // this term is modelling noise that does not exist on the input.)
        float encoder_rot_slip_k = 0.15f;  // dimensionless: fraction of the rotation that slips

        // ===== Online Motion Model Learning =====
        // Continuously adapt noise parameters from post-optimisation residuals.
        // Uses three EMA estimators:
        //   • learned_slip_k       — replaces encoder_rot_slip_k (rot-slip model)
        //   • learned_odom_noise_trans — replaces odom_noise_trans (translational noise fraction)
        //   • odom_bias            — systematic odometry bias subtracted per step
        // All updates are gated on localization quality < boundary_hessian_quality_threshold.
        // (LearnMotionModel and the MotionLearn* EMA knobs were deleted 2026-08-26 with the
        //  learner they configured — A/B'd and rejected, see compute_motion_covariance.)
                                                      // (more permissive than boundary_hessian_quality_threshold)

        int manual_reset_skip_frames = 5;      // Frames to skip optimization after manual pose set

        // ===== Relocalisation (reloc_search.h) =====
        // ONE mechanism replaces the RecoveryManager (10 bad frames + 60-frame cooldown), the map-trust
        // relocaliser and the periodic symmetry check — three relocalisers that fought each other: the
        // symmetry check flipped a LOST robot on 74 of 80 checks (09-12) and reset the recovery counter
        // every time, and it fired on a door OPENING with the robot 1.6 m inside (09-16).
        //
        // LOST BELIEF. A two-state HMM filter on s = median |SDF| (the worse of the post-fit residual and
        // the prediction error, as before). Tracking emission: log-normal in s, LEARNT online from the
        // frames it explains; lost emission: log-uniform over [reloc_lost_s_min, reloc_lost_s_max]. The
        // per-frame transition probability is the kidnap hazard. Search when P(lost) ≥ ½.
        // RIVAL MODES. The search returns a mixture; the non-committed modes are carried, moved with the
        // robot's own motion, and scored against the committed pose on every scan. A rival takes over only
        // when its accumulated log-likelihood ratio passes Wald's SPRT bound log((1−α)/α) with α = the
        // hazard — so a symmetric twin (ratio ≡ 0) can never flip-flop, and a genuinely better mode wins
        // in as few scans as its evidence allows.
        float reloc_outlier_frac       = 0.30f;  // ε — clutter / through-door fraction ⚠ modelling constant
        float reloc_wall_offset_sigma  = 0.03f;  // m — per-wall common-mode error marginalised from mode Σ
        float reloc_body_clearance     = 0.30f;  // m — robot body radius; no pose closer to a wall
        int   reloc_max_points         = 150;    // wall-band subsample for the lattice
        int   reloc_max_modes          = 4;      // mixture budget
        float reloc_hazard             = 1e-3f;  // per-frame P(kidnap / silent loss) — the HMM transition AND α
        float reloc_track_s_median     = 0.034f; // m — prior median of s when tracking (measured p50, 09-16)
        float reloc_track_log_sigma    = 0.30f;  // prior spread of ln s when tracking (p99/p50 = 0.069/0.034)
        int   reloc_track_memory_frames= 200;    // forgetting timescale of the learnt tracking emission
        float reloc_lost_s_min         = 0.01f;  // m — support of the lost emission (log-uniform)
        float reloc_lost_s_max         = 5.0f;   // m
        bool  reloc_enabled            = true;   // false = relocalisation never moves the pose (lost, mode_switch, seed_validation)
        int orientation_search_max_samples = 100;      // Lidar subsample for orientation candidates

        // ===== Gauss-Newton / Levenberg-Marquardt (room_gn_solver.h) =====
        // The only pose backend. The Adam and L-BFGS backends, the GN-vs-them shadow and the single-step
        // differential test were deleted with torch (2026-10-17); GN had been the default since its
        // 72.7 -> ~4 ms switch, and its fallback to Adam fired 0 times in try42-45.
        // Room optimises each object anchor's position as a PRIVATE variable (birth prior = the
        // producing agent's own belief) instead of pinning it to a frozen snapshot. Never written back
        // to the graph: the producer stays the sole authority for the published pose, so the gap
        // between the two estimates is a diagnostic instead of a silent merge. GN-only — the autograd
        // backends have no landmark variables. Default off.
        bool  object_anchor_optimize_landmark = false;

        int   gn_max_iters    = 10;
        float gn_lambda_init  = 1e-3f;   // Levenberg damping relative to diag(H)
        float gn_step_tol     = 1e-5f;   // ‖δ‖∞ (m / rad) convergence test
        float gn_loss_rel_tol = 1e-4f;   // relative loss-improvement convergence test

        // ===== Covariance Numerics =====
        float eigenvalue_clamp_posterior = 1e-4f;      // Min eigenvalue for posterior precision
        float eigenvalue_clamp_boundary = 1e-3f;       // Min eigenvalue for boundary-prior Hessian
        float eigenvalue_clamp_boundary_max = 500.0f;  // Max eigenvalue — prevents over-confident prior from contaminated scans
        // ===== Boundary Prior Quality Gate (Solutions B & C) =====
        // Prevents the boundary prior from being poisoned by bad localization frames
        // (e.g. after a forced displacement or while an obstacle occludes the scan).
        //
        // Solution B — quality-gated Hessian:
        //   When the oldest slot's sdf_mse_final > boundary_hessian_quality_threshold,
        //   the boundary prior precision is computed from the motion factor alone (kinematic
        //   precision only) instead of the full H_obs + H_motion Hessian. This prevents a
        //   contaminated scan from generating a high-confidence prior at a wrong pose.
        //
        // Solution C — quality-gated mu update:
        //   When the oldest slot's sdf_mse_final > boundary_mu_quality_threshold, the
        //   boundary prior mu (the pose anchor) is NOT updated. The prior continues to point
        //   to the last known good pose instead of following the confused estimate.
        float boundary_hessian_quality_threshold = 0.08f; // m — above this: motion-only Hessian
        float boundary_mu_quality_threshold = 0.10f;      // m — above this: keep previous mu

        // ===== FEJ + Schur marginalization boundary prior (replaces Solutions B/C when ON) =====
        // The legacy boundary prior above is a rough approximation to sliding-window
        // marginalization: it anchors the surviving oldest slot to the DROPPED slot's pose with a
        // Hessian that double-counts the survivor's own obs, and it re-anchors mu to drifting slots
        // every slide (a non-FEJ moving linearization point) — the "erroneous information gain" that
        // poisons the prior into the loss_boundary ratchet (0→559) diagnosed 2026-07-15.
        //
        // When boundary_fej_schur is ON, recompute_boundary_prior/append are replaced by
        // marginalize_oldest(), which forms the EXACT Schur complement of the dropped slot over its
        // Markov blanket (the next slot, via the motion factor) — folding the previous marginal, the
        // dropped slot's obs Hessian, and the motion factor into a marginal prior on the survivor —
        // and FREEZES its linearization point (First-Estimates Jacobians). The prior then carries a
        // linear term g (absent in the legacy form). No re-linearization ⇒ no information-gain ratchet.
        bool  boundary_fej_schur = false;
        // Continuous quality-as-precision weight on the dropped slot's obs contribution to the Schur
        // fold: w = 1/(1+(sdf_mse_final/σ_q)²). Replaces the hard 0.08/0.10 gates (no threshold) — a
        // poorly-fit dropped slot contributes weak, high-covariance information that cannot anchor.
        float boundary_quality_sigma = 0.10f;             // m — σ_q for the soft obs-quality weight
        float covariance_regularization = 1e-4f;       // λ added to posterior precision diagonal
        // covariance_det_min RETIRED 2026-08-29. It gated the posterior update on
        // det(P) > 1e-10, which for a 3x3 pose covariance means sigma > 21.5 mm on equal axes — so
        // it refused 89 of 102 updates and the published sigma never absorbed an observation.
        // A determinant cannot distinguish a SINGULAR covariance from a PRECISE one. Replaced by
        // the positive-definiteness and condition-number tests already computed alongside it.
        // float covariance_det_min = 1e-10f;
        float condition_number_max = 1e6f;             // Max condition number for valid covariance

        // ===== Motion Model =====
        float lateral_noise_fraction = 0.3f;           // Lateral noise as fraction of forward noise
        float base_rotation_noise_fraction = 0.5f;     // Base rotation noise relative to base translation noise
        float stationary_motion_threshold = 0.001f;    // meters; below this = stationary for covariance
        float stationary_speed_threshold = 0.03f;      // m/s; below this = near-stationary for process noise
        // ZUPT gate for the gyro heading override in integrate_odometry_over_window(): below these
        // WHEEL-reported rates the segment is treated as truly stopped, so dtheta stays on the wheel
        // channel (reads ~0 exactly there, no scrubbing to correct for) instead of the gyro. Without
        // this, a parked robot integrates pure gyro random walk as if it were rotation -- measured
        // ~0.011 rad accumulated per 5 s window with the robot stationary (see the [ImuInject] guard
        // comment a few lines below imu_dtheta_sum_'s use).
        bool  zupt_enabled       = true;
        float zupt_wheel_rot_eps = 0.002f;             // rad/s
        float zupt_wheel_lin_eps = 0.001f;             // m/s (adv, side)
        float rotation_position_coupling = 0.15f;      // meters of position uncertainty per radian of rotation
        float rotation_noise_base = 0.01f;             // Base rotation std when no commanded motion
        // Fallback diagonal for the slot motion covariance, used ONLY when no odometry prior is valid.
        // Measured over a 71k-frame run: that never happened (prior source = fused 70865 / measured 150 /
        // fallback_zero 1, sel_valid==0 on zero frames), so this value is effectively inert. The prior
        // actually in use is ~2.3e-4 (sigma 15 mm/frame). Do not tune this expecting an effect.
        float default_slot_motion_cov = 0.01f;

        // ===== Preintegrated motion covariance (see se2_preintegration.h) =====
        // When true, the motion factor's covariance is PROPAGATED through the interval's samples
        // instead of being asserted by compute_motion_covariance(). The MEAN is unchanged (same
        // midpoint-θ integration, same arithmetic), so this is a covariance-only change and both
        // optimizer backends pick it up with no factor edit — MotionFactor and the torch motion term
        // already consume a full 3×3 precision.
        //
        // What it changes, concretely: the covariance gains the θ↔xy and x↔y cross terms that a
        // heading error rotating the subsequent translation actually produces; dt enters where the
        // physics puts it, so the constants stop being update-rate dependent; and the strided window
        // chains intervals with the transport term instead of summing covariances.
        //
        // ★ WHERE THESE NUMBERS COME FROM. They are not hard-coded: room_config.cpp DERIVES them from
        // the legacy constants (StationaryMotionThreshold / OdomNoiseBase / RotationNoiseBase /
        // OdomNoiseTrans / OdomNoiseRot / EncoderRotSlipK and the Cmd* equivalents) right after those
        // load, so the two models cannot drift apart and the A/B tests the covariance's SHAPE rather
        // than a simultaneous re-tuning. The values below are only the fallback for code that builds a
        // Params without the loader (gn_selftest). An explicit Preint* config key overrides either.
        //
        // ★ MEASURED side by side at the live constants (StationaryMotionThreshold 0.02, OdomNoiseBase
        // 0.01, RotationNoiseBase 0.01, OdomNoiseTrans 0.08, OdomNoiseRot 0.04, EncoderRotSlipK 0.05),
        // Legacy / Propagated, sigma in m and rad, for the MEASURED-ODOMETRY channel. STRIDED rows
        // compare against the per-frame legacy covariance SUMMED over the interval, which is what
        // stride_cov_accum_ actually builds:
        //
        //   parked, 50 ms                sigma_xy 0.0200 / 0.0200   sigma_th 0.0100 / 0.0100   <- IDENTICAL
        //   straight 0.5 m/s, 50 ms               0.0120 / 0.0200            0.0100 / 0.0100
        //   pivot 1.0 rad/s, 50 ms                0.0214 / 0.0200            0.0123 / 0.0105
        //   arc 0.5 m/s + 1.0 rad/s               0.0142 / 0.0201            0.0123 / 0.0105
        //   STRIDED arc, 0.5 s (10 fr)            0.0447 / 0.0637            0.0388 / 0.0450
        //   STRIDED straight, 0.5 s               0.0379 / 0.0634            0.0316 / 0.0316
        //
        // ⚠⚠ THE COMMAND CHANNEL IS NOT IDENTICAL WHILE PARKED, and the table above does not show it.
        // compute_motion_covariance()'s stationary branch overrides BOTH channels' bases with
        // StationaryMotionThreshold, so legacy parked gives the command 0.02 as well — verified live,
        // cmd_cov_xx and meas_cov_xx both read exactly 4e-4. The derivation below takes
        // max(StationaryMotionThreshold, CmdNoiseBase) = 0.05, which is 2.5x LOOSER, so parked:
        //     fused cov_xx  2.00e-4 -> 3.45e-4  (1.7x looser)
        //     fusion weight on the ODOMETRY channel  0.50 -> 0.86
        // That is KEPT DELIBERATELY, not left as an oversight: while parked, an encoder reporting zero
        // is worth far more than an open-loop command reporting zero, and legacy's stationary branch
        // discards that distinction precisely where it is strongest. Shifting weight onto the encoder is
        // also the direction cf26167 established. But it IS a change in the regime the robot occupies
        // >99% of the time, so it belongs in the A/B, and the honest reason it cannot simply be matched
        // is that legacy's floor is NON-MONOTONIC in speed and in opposite directions per channel
        // (odometry 0.02 parked -> 0.01 moving, command 0.02 -> 0.05); no single density reproduces both
        // branches. Set PreintCmdSigmaVLat/Long = 0.0894 to force the legacy parked value if the A/B
        // needs the two runs to differ in one thing only.
        //
        // So: for the ODOMETRY channel the stationary regime matches exactly, and everywhere else the
        // propagated form is broadly 1.3-1.7x LOOSER in position rather than tighter. Two mechanisms,
        // both of them the legacy shape being wrong rather than the new one:
        //   • Legacy's floor DROPS discontinuously from 0.02 m to 0.01 m the moment |Δp| exceeds
        //     StationaryMotionThreshold, so it asserts MORE position uncertainty when barely moving
        //     than when moving fast. The propagated form has no switch and that non-monotonicity goes.
        //   • Legacy's rotation_position_coupling inflates position by 0.15·|Δθ| even for a pivot in
        //     place, where a heading error has no translation to rotate. The propagated form gets 0
        //     there because the geometry says 0 — which is why pivot rows come out tighter.
        //
        // ⚠ AND THE HONEST CAVEAT: at these constants the cross terms are nearly invisible
        // (rho(y,theta) = -0.028 on the strided arc, against -0.19 in the gn_selftest trajectory).
        // The derived floor is 0.02 m per 50 ms = 0.089 m/√s of position random walk for a PARKED
        // wheeled robot, which is not a physical number — it is a stabiliser added to stop loss_motion
        // spiking to 6000 on a 3.4 cm parked pose residual — and it swamps every other term including
        // the structure this change exists to introduce. So the algebra being right (validated against
        // Monte Carlo, see gn_selftest) is NOT yet the same as the covariance being right. Calibrating
        // sigma_* against the real stream is the load-bearing step, not an optional refinement: park
        // the robot and take the sample variance of the odometry stream, then one constant-velocity leg
        // for scale_*. Until then these are a faithful translation of the old constants, not a
        // measurement, and a MOVING A/B (early exit while translating, innov_norm, loss_motion) is the
        // only thing that settles whether the looser prior helps or hurts.
        /// Apply the rest hypothesis as a FACTOR on the state instead of as covariance shaping, so
        /// it can arrest a parked robot's drift rather than only shrinking the covariance around it.
        /// Mutually exclusive with the per-sample form; see NoiseModel::zupt_as_factor.
        bool zupt_as_factor = false;
        /// Apply the rest hypothesis to the PREDICTED INCREMENT, at prior selection — the only place
        /// it reaches the ~99% of cycles that publish without the solver running. See the block in
        /// select_motion_prior(). Independent of the two covariance forms above; it acts on the mean,
        /// they act on the covariance.
        /// One damped SDF gradient step on cycles that early-exit — the only thing tried so far that
        /// attacks the CAUSE of parked drift rather than its step size. See try_prediction_early_exit.
        bool  sdf_polish_enabled = false;
        bool  zupt_on_prediction = false;
        float zupt_pred_v_max    = 1.0f;   ///< m/s   — width of the "moving" hypothesis, not a limit
        float zupt_pred_w_max    = 2.0f;
        /// Decide rest vs moving with RestMotionChannel (rest_density_learner.h): a velocity state per
        /// body-frame channel whose noise is learnt from its own innovations; PreintZuptDensity* become
        /// only a one-cycle prior. false = the legacy per-cycle mixture that failed the creep test.
        bool  zupt_pred_learn_rest = false;
   ///< rad/s — the base's plausible range, not a clamp
        /// Where the calibration WINDOW (evidence, never parameters) is kept between runs.
        std::string calib_state_file = "etc/motion_calib_state.csv";
        bool motion_preintegration = false;
        /// Learn the motion-noise COMPONENT coefficients (per metre / per radian / per second) from the
        /// localiser's corrections (motion_noise_vc.h). Needs odom_preint_noise.motion_proportional.
        /// RoomConcept.MotionNoiseLearn / MotionNoiseMemory.
        bool   motion_noise_learn  = false;
        double motion_noise_memory = 20000.0;   ///< corrections remembered (~20 runs of ~1k)
        /// Measure the SCAN-TO-SCAN innovation every cycle and learn the noise components from it
        /// (motion_noise_innov.h, Fable window memo 2026-10-05). Always learns and logs
        /// (tmp/noise_innov/innov_<ts>.csv); the learnt coefficients reach the prediction ONLY when
        /// motion_noise_learn is also true. RoomConcept.MotionNoiseInnov.
        bool   motion_noise_innov  = true;
        /// Learn the slow POSE-FIELD bias of the scan-only pose without ground truth (pose_field_bias.h) and
        /// log it (heading CSV pf_*). pose_field_publish additionally ADDS its covariance to the PUBLISHED pose
        /// covariance (never to the solver's). RoomConcept.PoseFieldBias / PoseFieldPublish / PoseFieldSigma0.
        /// Gauss-Newton iterations for the scan-only pose z (scan_only_pose). 1 = the old single step, which
        /// under-stepped (g = 0.81-0.88). RoomConcept.ScanOnlyIters.
        int    scan_only_iters     = 3;
        bool   pose_field_bias     = true;
        bool   pose_field_publish  = false;
        double pose_field_sigma0   = 0.02;
        /// r2 mount factor (plan 2026-10-08): feed the KINEMATIC factor (helios dx, dy, psi) from the same
        /// innovation stream (log only: tmp/mount_factors/kin_*.csv). RoomConcept.MountFactors.
        bool   mount_factors       = true;
        rc::preint::NoiseModel odom_preint_noise{};  // measured-odometry channel
        // Command channel. Its floor stays deliberately looser than the encoder's (cmd_noise_base
        // 0.05 m vs odom_noise_base 0.01 m) because an open-loop command really can be wrong while the
        // robot stands still — the wheels may simply not have obeyed it.
        rc::preint::NoiseModel cmd_preint_noise{ .sigma_v_lat  = 0.224f,   // cmd_noise_base 0.05 / √0.05
                                                 .sigma_v_long = 0.224f,
                                                 .sigma_omega  = 0.0447f,
                                                 .scale_v      = 0.20f,    // cmd_noise_trans
                                                 .scale_omega  = 0.18f };  // cmd_noise_rot

        // ===== Strided RFE window =====
        // The window holds rfe_window_size poses, but at the lidar rate they are all essentially the
        // SAME pose: measured while moving, the max pairwise separation across the 5 slots is 76 mm
        // (p50) in a ~9 m room, and 75.8% of all frames span under 20 mm. Five near-coincident
        // viewpoints buy five nearly-identical SDF evaluations and almost no independent geometric
        // leverage on lateral position.
        //
        // With striding, a new slot is ADMITTED only once the robot has actually moved; until then the
        // newest slot is REPLACED in place, so the current frame is always represented but does not
        // consume the window's span. Predicted baselines from the same log: 0.10 m spacing -> ~222 mm
        // p50, 0.25 m -> ~448 mm, against 76 mm today.
        //
        // ★ The motion factor between slots must then span the whole skipped interval, so the odometry
        // delta and its covariance are ACCUMULATED across replaced frames (stride_delta_accum_).
        // Getting that wrong would under-constrain the very DOF this is meant to fix.
        // ★ Watch loss_boundary for a ratchet when enabling: room-fast-rotation-track-lag traced an
        // earlier failure to the marginalization prior being poisoned by admitted early-exit poses, and
        // a longer stride means each admitted slot carries more accumulated drift.
        bool  window_stride_enabled  = false;
        float window_min_travel_m    = 0.10f;   // admit a new slot after this much travel...
        float window_min_turn_rad    = 0.15f;   // ...or this much turn, whichever comes first

        // ===== Innovation-based adaptive covariance =====
        // The published sigma is otherwise inert: measured over a 71k-frame run, a 26x change in
        // innovation (2.4 -> 62.7 mm) moves sigma_xy by 6%, and the whole p10..p90 band is
        // 0.0415..0.0508 m. Sigma comes from the Laplace curvature — how SHARP the minimum is, which
        // says nothing about whether it is the RIGHT minimum — so a confident wrong pose is published
        // as confident. Downstream, the controller's speed governor can never engage and all five
        // concept agents fold a constant into their common-mode variance.
        //
        // Two attempts to fix this by accumulating process noise while coasting BOTH failed, measured:
        //   - accumulating the full Q:  mean sigma 0.0443 -> 0.0748 (+69%), 29.7% of frames above the
        //     controller's 0.080 throttle knee, innovation ratio only 1.059 -> 1.122. Q is dominated by
        //     a per-frame constant floor (odom_noise_base is 86% of the term at 0.35 m/s), so that was
        //     really accumulating TIME — and time has r=+0.001 against the correction that follows.
        //   - accumulating only the motion-proportional part: mean preserved (40.7 vs 40.4 mm) but
        //     ratio 1.000, because that term is ~1.6 mm/frame against a 40 mm base. Useless.
        // The base sigma itself is the problem, so no process-noise scheme can repair it.
        //
        // INSTEAD: the innovation (optimised pose - predicted pose) is a direct SAMPLE of the estimate's
        // own error — two independent estimates of the same pose, differenced. Its running second moment
        // is therefore a lower bound on the error covariance that the filter cannot argue with. Publish
        //     sigma_ii = max(laplace_ii, EMA[innovation_i^2])
        // so the covariance can never claim more precision than the estimator's own disagreements
        // demonstrate. This is standard innovation-based adaptive estimation.
        //
        // Why this passes where the others failed: at the median innovation (12 mm) the Laplace term
        // (44 mm) dominates and the mean is untouched — protecting the 0.020 m of headroom under
        // PoseXYStdSlow that a validated -11.2% lap-time tuning depends on — while a sustained run of
        // large innovations (p99 97 mm, i.e. the corridor) lifts sigma above it. Mean unchanged,
        // dynamic range restored, which is exactly the acceptance test.
        // Online motion-model learning: slow parameters (forward scale, mount yaw, gyro scale)
        // estimated from the optimizer's own corrections. See motion_calibration.h. Precision-only
        // adaptation (adaptive_cov below, HierPrec*) cannot do this job: it widens the posterior but
        // never moves the mean, so a biased channel still walks -- the tooth gets longer, not smaller.
        rc::calib::Config motion_calib{};

                // Add the accelerometer's within-segment double integration to the wheel displacement.
        // integrate_odometry_over_window holds a 10 Hz wheel velocity FLAT across each segment; the
        // true displacement is v0*dt + integral(integral(a)), and odom.adv IS v0 (the sample's stamp
        // opens the segment), so the IMU supplies exactly the term the zero-order hold drops. Zero at
        // constant speed, largest under acceleration -- which is when the prediction error grows.
        //
        // It CANNOT replace the wheels the way the gyro replaced them for heading: an accelerometer
        // observes only the CHANGE in velocity, never velocity itself, so the wheels still supply the
        // absolute v this is a change to. It also cannot fix wheel scale (k_v does that) or wheel
        // noise. Bounded by construction: the term is confined to one segment and never chained, so
        // it cannot drift however wrong the accelerometer is.
        bool imu_linear_injection = false;
        // ── HEADING = PRODUCT OF TWO FACTORS, not a switch ──────────────────────────────────────────
        // true: every segment's heading is the precision-weighted mean of the wheel factor and the
        // gyro factor, each corrected by ITS OWN calibration (wheels: k_omega_w + dk_wheel; gyro:
        // k_omega + b_omega). The weights come from each channel's stated noise density plus its
        // scale uncertainty, so the gyro dominates a fast turn and the wheels a stop -- with no gate.
        // false: the legacy 0/1 switch (gyro whenever it brackets the segment and the wheels are not
        // stationary), one shared scale, kept for A/B. ★The bias sign fix applies to BOTH.
        bool heading_fusion = false;
        // Learn the wheels' heading-noise density as d_w^2 = c0 + c1|v| + c2|omega| from the
        // wheel-minus-gyro disagreement (truth cancels), anchored on the gyro's stated density. Used
        // for the wheel factor's weight in the fusion above. See wheel_noise_learner.h.
        bool heading_noise_learning = false;
        // ── The WHEELS' own per-sample velocity variance, as a preintegration density ──────────────
        // OFF by default, like every channel that can move the published pose. When on, each odometry
        // sample's stated velCov (forwarded by robot_concept onto robot_current_speed_variance) is
        // converted to a density -- sigma = sqrt(var_sample * dt_sample), with dt_sample MEASURED --
        // and ADDED IN QUADRATURE to odom_preint_noise for that segment.
        //
        // It can only ever LOOSEN the motion prior, never tighten it: the model constants are lumped
        // MODEL error (slip, timing, unmodelled dynamics) and a sensor's noise is a separate, extra
        // contribution, so the two add. Replacing one with the other was tried on 2026-08-23 and made
        // the prior 135x too confident in heading. What this buys is a prior that widens on its own
        // when a producer says its wheels are degrading, with nobody editing a config file.
        bool odom_variance_injection = false;

        /// Measure the published sigma against what the window's own normal equations say about the
        /// newest pose. Writes etc/hessian_check.csv on optimised frames.
        /// ★ It answers a specific question: compute_posterior_covariance() uses the FILTERING form
        ///   (Λ_prev + H_newest_observation), whose prediction step is gated on the SDF polish and
        ///   therefore off — so Λ can only grow and sigma can only shrink. This says by how much,
        ///   with no ground truth required, by comparing the ratio over time since the last solve.
        bool  hessian_check = true;
        /// Take the published covariance from the GN solver's own normal equations (the newest
        /// pose's Schur-complemented marginal) instead of a double-backward autograd Hessian folded
        /// into a recursion. See compute_posterior_covariance for the three measured reasons.
        /// GN only; with any other optimizer the old path runs regardless.
        bool  covariance_from_solver = true;
        bool  adaptive_cov_enabled = false;
        // EMA rate for the innovation second moment. 0.02 ~= a 50-frame memory: long enough that one
        // bad frame cannot spike the published sigma (which would hit the speed governor), short enough
        // to track entering and leaving a feature-poor stretch.
        float adaptive_cov_lambda  = 0.02f;

        // ===== Velocity-Adaptive =====
        float combined_motion_weight = 1.2f;           // Weight when both translating and rotating

        // ===== Corner Detection =====
        bool  enable_corner_tracking = true;      // Master switch for corner factors in Adam loss
        bool  sdf_current_slot_only = false;           // If true, SDF obs evaluated only for the newest slot;
                                                       // older slots contribute via motion + corner factors only.
        float corner_obs_sigma = 0.04f;                // Corner measurement noise (meters)
        int   min_tracking_steps_for_corners = 5;      // Require this many tracking steps before adding corner factors
        int   corner_max_slots = 5;                    // Only apply corner factors to the newest N slots
        float corner_huber_delta = 0.3f;               // (legacy, unused) old isotropic Huber in meters
        // Graded anisotropic corner factor: loss = 0.5·gain·huber(m)·rᵀΛ_det r,  m = sqrt(gain·rᵀΛ_det r).
        float corner_precision_gain = 1.0f;            // global scale on corner precision vs the SDF term
        float corner_huber_sigma    = 3.0f;            // Huber saturation in WHITENED (σ) units, not meters
        // CornerDetector geometry/grading (applied to corner_detector_.params() at model init).
        float corner_wall_band      = 0.35f;           // perpendicular gather band (m) — MUST exceed model misfit
        float corner_base_sigma     = 0.04f;           // detection noise floor σ0 (m) per wall
        float corner_orient_tau_deg = 20.0f;           // smooth orientation-trust scale (deg)
        // Exclusion ("two corners cannot occupy the same physical space"): coincident detections are
        // fused when their separation is inside a χ²₂ region of their OWN combined covariance.
        float corner_merge_chi2 = 5.991f;              // χ²₂ @95%; 0 disables the exclusion test
        float corner_merge_prior_sigma = 0.0f;         // m; 0 ⇒ use the detector's search_radius
        // Landmark admissibility in units of the detector's map_sigma — a vertex whose shorter adjacent
        // wall is below this is not a feature the traced layout can assert, so it never becomes a point
        // landmark (it stays in the polygon and in the SDF). 0 disables. See CornerDetector::Params.
        float corner_min_wall_map_sigmas = 3.0f;
        // Landmark retirement by observed information — a corner whose weakest-axis σ never gets below
        // this many map_sigmas is one the robot cannot measure here, whatever the layout claims. Leaky,
        // so it recovers by itself. 0 disables. See CornerDetector::Params::min_yield_map_sigmas.
        float corner_min_yield_map_sigmas = 5.0f;
        float corner_yield_leak           = 0.02f;   // per matched frame (~2.5 s at 20 Hz)
        float corner_yield_release_factor = 2.0f;    // hysteresis: release above this × the retire bar
        int   corner_yield_warmup         = 10;      // MATCHED frames tolerated before retiring — an
                                                     // evidence budget in "uninformative votes", not a
                                                     // time delay. Keep small; see CornerDetector.
        // Per-model-corner attribution CSV (etc/corner_stats.csv), rewritten periodically: which
        // vertices actually earn their keep as landmarks vs cost work and return nothing.
        bool  corner_stats_csv = true;
        // Corner-consistency gate on the prediction early-exit. Without it, the early-exit validates the
        // predicted pose on SDF ALONE — and a rot180-flipped pose is SDF-ambiguous, so a flip slips through
        // while the corners (which DO disagree) are bypassed. When on, a predicted pose whose worst corner
        // whitened residual m=√(rᵀΛ_det r) exceeds corner_early_exit_sigma is rejected → Adam runs and the
        // corner factor pulls it back. Flag-gated (A/B) — the anchor early-exit gate destabilized before,
        // but corners are re-detected per-frame + graded, so a bad detection can't force a bad correction.
        bool  corner_early_exit_check = false;         // OFF by default — flip on to A/B
        float corner_early_exit_sigma = 4.0f;          // per-corner whitened-residual (σ) disagreement tolerance
        // CONSENSUS, not worst-case: a lone mismatched corner (a flaky notch detection / one misassociation)
        // must NOT force a full optimization when the others corroborate the prediction. Reject the
        // early-exit only when at least this many corners disagree (m>σ). A true rot180 flip trips ALL
        // corners, so consensus still catches it; a single outlier is outvoted.
        int   corner_early_exit_min_bad = 2;           // # of disagreeing corners required to force Adam

        // ===== Object anchors (validated modelled objects as SE(2) pose landmarks) =====
        // As the room fills with confidently-localised objects (tables, …) their poses
        // become interior factors that tighten the robot↔room link.  Precision-weighted by
        // each object's own belief covariance — no threshold (see object_anchor_source.h).
        ObjectAnchorFactor::Params object_anchor;      // .enable defaults false (OFF)

        // RGB edge alignment: structural contours (wall-wall corners, floor-wall junction) against
        // the image gradient. OFF by default at three levels — see room_config.h. The precision is
        // DERIVED (Cramer-Rao from the measured gradient profile) and the shared-nuisance cap is
        // physical, so there is deliberately no weight knob here either.
        ImageEdgeFactor::Params image_edge;            // .enable / .drive default false (OFF)
        int         image_edge_max_slots = 1;          // newest slot only, like corners
        bool        image_edge_shadow    = false;      // evaluate + log, do NOT move the pose
        std::string image_edge_csv       = "etc/image_edge.csv";
        int   object_anchor_max_slots = 3;             // Only apply object factors to the newest N slots
        float object_anchor_early_exit_sigma = 2.0f;   // prediction early-exit gate: whitened anchor-residual
                                                       // σ-cutoff above which the optimizer is forced to run

        // ===== Wall-SLAM: estimate the room layout instead of loading it (RoomShape.MapMode) =====
        // The ONE switch for the whole feature. Given ⇒ today's agent, byte for byte: nothing below is
        // read. Estimate ⇒ no layout is loaded; walls are landmarks (wall_map.h), the wall point factors
        // replace the SDF factor, the first pose is the origin until the polygon closes and is
        // re-anchored to the bbox-centred frame the fleet expects (reanchor_map_frame). No second gate
        // inside Estimate mode: freezing is by accumulated precision, structure change by ΔF.
        enum class MapMode { Given, Estimate };
        MapMode map_mode = MapMode::Given;
        wallseg::Params wall_seg;              // segmenter: sensor σ, χ² levels, RANSAC budget
        /// The box layout sidecar. Off by default; see room_boxes_channel.h for why it is a
        /// sidecar and not a replacement.
        rc::boxch::Params box;
        wallmap::Params wall_map;              // association, birth, Manhattan prior, publish bar
        float wall_gauge_sigma_xy = 1e-3f;     // m — first-pose gauge fix (a gauge, not a model term)
        // RoomShape.RecordWallInput — DIAGNOSTIC RECORDER (2026-09-17), default OFF. Writes, per frame, exactly
        // what the wall step receives (robot-frame points + z + observation weights, slot pose, pose covariance,
        // map-frame odometry delta, stride flag, timestamp) to tmp/wall_input_<unix_ms>.bin, so a live run can
        // be replayed through the offline harness core. Read-only: changes no estimate.
        bool  record_wall_input = false;
        /// RoomShape.PlainSlam (2026-10-09): while SEARCHING, the pose comes from plain scan-to-occupancy SLAM
        /// (scan_slam.h), NOT from the provisional layout. The wall map observes at that pose and never feeds it.
        bool  plain_slam = false;
        float wall_gauge_sigma_theta = 1e-3f;  // rad
        int   wall_max_slots = 0;              // wall factors on the newest N slots (0 ⇒ every slot)


        // ===== Debug Logging =====
        bool debug_log_enabled = false;      // Write per-frame CSV to tmp/sdf_localizer/log.csv

        // ===== Rerun streaming =====
        bool rerun_enabled = false;
        std::string rerun_host = "127.0.0.1";
        int rerun_port = 9877;
        int rerun_sdf_every_n = 20;
        int rerun_sdf_resolution = 150;
        int rerun_max_queue = 30;
    };
} // namespace rc::loc
