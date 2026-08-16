#pragma once

#include <memory>
#include <vector>
#include <deque>
#include <map>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <variant>
#include <optional>
#include <string>
#include <fstream>
#include <functional>

// ---- PyTorch vs Qt macros (slots/signals/emit) ----
// Qt uses 'slots' as a macro. PyTorch/libtorch has methods named slots(), which breaks compilation.
// We temporarily undefine Qt macros *only* while including torch headers, then restore them.
#ifdef slots
  #define RC_QT_SLOTS_WAS_DEFINED
  #undef slots
#endif
#ifdef signals
  #define RC_QT_SIGNALS_WAS_DEFINED
  #undef signals
#endif
#ifdef emit
  #define RC_QT_EMIT_WAS_DEFINED
  #undef emit
#endif

#include <torch/torch.h>

#ifdef RC_QT_SLOTS_WAS_DEFINED
  #define slots Q_SLOTS
  #undef RC_QT_SLOTS_WAS_DEFINED
#endif
#ifdef RC_QT_SIGNALS_WAS_DEFINED
  #define signals Q_SIGNALS
  #undef RC_QT_SIGNALS_WAS_DEFINED
#endif
#ifdef RC_QT_EMIT_WAS_DEFINED
  #define emit Q_EMIT
  #undef RC_QT_EMIT_WAS_DEFINED
#endif

#include <Eigen/Dense>
#include "common_types.h"
#include "buffer_types.h"
#include "room_model.h"
#include "corner_detector.h"
#include "rerun_logger.h"
#include "object_anchor_types.h"
#include "object_anchor_factor.h"
#include "se2_preintegration.h"

namespace rc
{
/**
 * RoomConcept
 * =============
 * Implementación mínima para estimar (o refinar) el estado conjunto robot-habitación usando SDF.
 *
 * Estado (5): [width, length, x, y, phi]
 *  - width/length: dimensiones completas (m)
 *  - x,y,phi: pose del robot respecto al centro de la habitación (habitación centrada en (0,0))
 *
 * Flujo previsto en esta fase:
 *  - set_initial_state(...) con el GT / hipótesis inicial de la habitación y pose aproximada.
 *  - update(...) optimiza esos 5 parámetros minimizando mean(SDF^2) vía Adam.
 */

class RoomConcept
{
public:
    struct Params
    {
        int num_iterations = 25;          // Balance between speed and convergence
        float learning_rate_pos = 0.05f;  // Uniform LR for all pose DoF (x, y, θ)
        float min_loss_threshold = 0.1f;  // Early exit threshold

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

        // ===== FE-native relocalization when map-trust collapses (HIERARCHICAL_PRECISION.md) =====
        // When the higher-level belief exp(u_b_) drops below a floor for a few frames, the map no longer
        // explains the robot → run the existing hierarchical grid search (same action as RecoveryManager,
        // which is kept as an independent raw-sdf backstop). Requires hier_prec_boundary_enabled. OFF by default.
        bool  hier_prec_reloc_enabled        = false;
        float hier_prec_reloc_floor          = 0.10f; // exp(u_b_) below this = map-trust collapse
        int   hier_prec_reloc_consecutive    = 3;     // frames below floor before firing (debounce)
        int   hier_prec_reloc_cooldown_frames= 30;    // frames suppressed after a fire
        float hier_prec_ee_dtheta_min        = 0.15f; // rad (|Δθ| over window) gating the early-exit nudge

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
        bool use_cuda = false;
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
        bool differential_test_enabled = false;  // Enable shadow single-step evaluator for RFE vs baseline comparison

        // ===== Recovery Detection ===
        // Trigger grid search when full Adam keeps returning high loss
        float recovery_loss_threshold = 0.3f;  // final_loss above this = bad localization
        int recovery_consecutive_count = 3;    // How many bad frames before triggering recovery

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
        // ===== Prior covariance model =====
        // Process noise for commanded velocity prior (open-loop, less reliable)
        float cmd_noise_trans = 0.20f;   // Fractional position noise per meter of motion
        // Fractional rotation noise per radian of COMMANDED rotation. 0.18, not 0.10, kept on
        // measured outcome: it shifts fusion weight to the encoder (the better per-frame predictor),
        // and early exit while rotating went 91.6% -> 95.6% with over-gate frames 3.6% -> 0.5% on
        // matched populations. It is NOT the command's measured error -- properly integrated (the
        // localizer's own cmd_dth/meas_dth, not a re-integration of the velocity columns, which
        // mangles the sparse stepwise command signal) the command runs ~10% below the encoder and
        // ~3% below the room-anchored posterior. See config_apartamento.toml for the full retraction.
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
        // meas/post 1.072; see the retraction in config_apartamento.toml. Note also that under
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
        bool  learn_motion_model           = false;   // Master switch
        float motion_learn_alpha           = 0.05f;   // EMA rate for slip-k and trans noise
        float motion_learn_beta            = 0.02f;   // EMA rate for bias vector (slower)
        float motion_learn_min_omega       = 0.05f;   // Min angular speed (rad/s) to update slip-k
        float motion_learn_min_trans       = 0.05f;   // Min translation (m) per slot to update trans-noise
        int   motion_learn_min_frames      = 50;      // Warmup frames before using learned values
        float motion_learn_quality_threshold = 0.12f; // Per-slot SDF MSE gate for motion learning
                                                      // (more permissive than boundary_hessian_quality_threshold)

        // ===== Recovery =====
        int recovery_cooldown_frames = 30;     // Frames to skip detection after recovery
        int manual_reset_skip_frames = 5;      // Frames to skip optimization after manual pose set

        // ===== Periodic 180° symmetry flip check =====
        // Every N optimizer-frames test whether rotating the robot by 180° gives
        // lower SDF FE than the current pose.  Disabled once the room is stable.
        // Set to 0 to disable.
        int   symmetry_check_interval       = 5;     // optimizer-frames between checks (small → can
                                                     // integrate evidence across a momentary turn)
        float symmetry_flip_min_improvement = 0.005f; // BASE per-check loss advantage to count as evidence
        // Leaky evidence accumulation + confidence-scaled threshold so a long-correct track requires
        // SUSTAINED contrary evidence to flip (not a momentary tight-turn degeneracy). Each check adds
        // (loss_cur - best_loss - min_improvement) to flip_evidence with a leak; flip when it exceeds
        // threshold = base × (1 + confidence_gain × min(good_fit_streak, cap)). good_fit_streak counts
        // consecutive frames whose SDF is below symmetry_good_fit_mse (the track is "established").
        float symmetry_evidence_leak        = 0.6f;   // per-check retention (<1 → momentary blips decay)
        float symmetry_flip_evidence_thresh = 0.02f;  // base evidence to cross before flipping
        float symmetry_confidence_gain      = 0.05f;  // threshold growth per established frame
        int   symmetry_confidence_cap       = 200;    // streak cap (~10-20 s) so threshold can't run away
        float symmetry_good_fit_mse         = 0.02f;  // SDF below this ⇒ a "good fit" frame (streak++)
        // Streak DECAYS (not resets) on a bad frame so a momentary tight-turn degradation only dents the
        // established confidence; only sustained bad fit (truly lost) erodes it. cap/decay ≈ bad frames
        // to fully lose confidence (200/8 ≈ 25 frames ≈ ~1.3 s @19 Hz).
        int   symmetry_confidence_decay     = 8;
        // Trial CSV: one row per symmetry check (not just when a flip fires), so a flip event
        // can be traced back to the evidence/threshold trajectory that led to it. Independent
        // of debug_log_enabled — cheap (1 row / symmetry_check_interval frames).
        bool  symmetry_debug_csv            = true;

        // ===== Grid Search / Orientation Search =====
        float grid_search_wall_margin = 0.3f;        // meters from room walls
        int grid_search_max_samples = 150;            // Lidar subsample for grid evaluation
        // Success bar as a FRACTION of recovery_loss_threshold, not an absolute. A search that reports
        // success must leave a pose recovery will not instantly call lost again, or the two loop
        // forever; deriving the bar makes that un-driftable. 0.5 ⇒ the search must halve the error
        // recovery considers "lost" before it claims to have solved anything.
        float grid_search_good_factor = 0.5f;
        int orientation_search_max_samples = 100;      // Lidar subsample for orientation candidates

        // ===== Optimizer Selection =====
        // "ADAM"  — adaptive moment estimation (current default)
        // "LBFGS" — limited-memory BFGS with Wolfe line search (faster convergence)
        // "GN"    — Levenberg-Marquardt on analytic Jacobians (room_gn_solver), no autograd
        // "CAVI"  — coordinate-ascent variational inference (reserved, not yet implemented)
        std::string optimizer_type = "LBFGS";

        // ===== Gauss-Newton / Levenberg-Marquardt backend (room_gn_solver.h) =====
        // Same objective as compute_rfe_loss, solved as a 15×15 normal-equation system instead of
        // ~29 torch forward+backward passes. OFF by default in every sense: `optimizer_type` must say
        // "GN" for it to drive the pose, and gn_shadow must be set for it to run at all otherwise.
        //
        // gn_shadow runs BOTH backends on the SAME starting window state, keeps the authoritative
        // one's answer, and logs the pair to gn_shadow_csv_path. It is a numerical comparison on
        // identical inputs, not a behavioural A/B — the live pose is untouched.
        // Room optimises each object anchor's position as a PRIVATE variable (birth prior = the
        // producing agent's own belief) instead of pinning it to a frozen snapshot. Never written back
        // to the graph: the producer stays the sole authority for the published pose, so the gap
        // between the two estimates is a diagnostic instead of a silent merge. GN-only — the autograd
        // backends have no landmark variables. Default off.
        bool  object_anchor_optimize_landmark = false;

        bool  gn_shadow = false;
        // Finite-difference check of the analytic Jacobian, logged as grad_relerr in the shadow CSV.
        // Costs 2·15 extra loss evaluations per optimized frame, so keep it on only for a validation
        // run, not for a timing run — it inflates ms_gn.
        bool  gn_grad_check = false;
        std::string gn_shadow_csv_path = "etc/gn_shadow.csv";
        int   gn_max_iters    = 10;
        float gn_lambda_init  = 1e-3f;   // Levenberg damping relative to diag(H)
        float gn_step_tol     = 1e-5f;   // ‖δ‖∞ (m / rad) convergence test
        float gn_loss_rel_tol = 1e-4f;   // relative loss-improvement convergence test

        // ===== Adam Convergence =====
        float convergence_relative_tol = 0.01f;       // Relative loss-change stopping criterion
        int convergence_min_iters = 8;                 // Minimum iterations before convergence check

        // ===== L-BFGS parameters =====
        // Used only when optimizer_type == "LBFGS".
        // lr=1 is the standard initial Newton step; strong_wolfe line search adjusts it.
        float  lbfgs_lr               = 1.0f;   // Initial step size
        int    lbfgs_history_size     = 5;      // (s,y) pairs kept for H^-1 approximation
        double lbfgs_tolerance_grad   = 1e-5;   // Stop when ||grad||_inf < tol
        double lbfgs_tolerance_change = 1e-7;   // Stop when |Δloss| < tol

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
        float covariance_det_min = 1e-10f;             // Min determinant for valid covariance
        float condition_number_max = 1e6f;             // Max condition number for valid covariance

        // ===== Motion Model =====
        float lateral_noise_fraction = 0.3f;           // Lateral noise as fraction of forward noise
        float base_rotation_noise_fraction = 0.5f;     // Base rotation noise relative to base translation noise
        float stationary_motion_threshold = 0.001f;    // meters; below this = stationary for covariance
        float stationary_speed_threshold = 0.03f;      // m/s; below this = near-stationary for process noise
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
        bool motion_preintegration = false;
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
        int   object_anchor_max_slots = 3;             // Only apply object factors to the newest N slots
        float object_anchor_early_exit_sigma = 2.0f;   // prediction early-exit gate: whitened anchor-residual
                                                       // σ-cutoff above which the optimizer is forced to run

        // Torch threading configuration. The window solve is a tiny problem (3 DOF × ~5 slots ×
        // ~400 pts), so intra-op parallelism mostly buys thread-dispatch overhead, not speed, while
        // pinning that many cores. Applied in start() (set_num_threads is runtime-safe), config key
        // RoomConcept.TorchNumThreads. Lower this to shed CPU cores; A/B against t_adam_ms.
        int torch_num_threads = 2;          // intra-op CPU threads (was hard-coded 5 = the 4-core budget)
        int torch_num_interop_threads = 2;  // inter-op threads (set once at static init; not runtime-tunable)

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

    // Get the torch device based on params
    torch::Device get_device() const
    {
        if (params.use_cuda && torch::cuda::is_available())
            return torch::kCUDA;
        return torch::kCPU;
    }

    struct UpdateResult
    {
        bool ok = false;
        // Set when the SDF optimization produced a non-finite pose and this result is a dead-reckoned
        // FALLBACK, not a fix. Consumers must treat it as such (its covariance is inflated to match).
        bool diverged = false;
        float final_loss = 0.f;      // Scaled loss (for optimization)
        // MEDIAN |SDF| IN METRES — despite the name, which is legacy and kept only because two public
        // config keys (SymmetryGoodFitMse, StableSdfMseMax) are named after it. It is NOT a mean and
        // NOT squared, so do NOT sqrt it: that produced the square root of a length at the recovery
        // trigger and made the grid search's bar ~5x looser than recovery's, so recovery fired forever
        // on a state the search then declared fine. Compare it only against other metres, and produce
        // it only via median_abs_sdf().
        float sdf_mse = 0.f;
        // Early-exit decision variable: mean |SDF| at the odometry-predicted pose (meters). The
        // optimizer is SKIPPED when this falls below sigma_sdf*prediction_trust_factor (+rotation
        // boost). NaN when the early-exit gate wasn't evaluated this frame (warmup / no odometry /
        // prior not ok / manual-reset settle). See try_prediction_early_exit().
        float early_exit_metric = std::numeric_limits<float>::quiet_NaN();
        /// Median |SDF| at the predicted pose — the median-valued twin of early_exit_metric.
        float pred_sdf_median = std::numeric_limits<float>::quiet_NaN();
        Eigen::Matrix<float,5,1> state = Eigen::Matrix<float,5,1>::Zero();
        Eigen::Affine2f robot_pose = Eigen::Affine2f::Identity();
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
        float condition_number = 0.f;
        int iterations_used = 0;

        // Timestamp (system-clock epoch ms) of the lidar scan used for this result.
        // Consumers can measure the age of this result as (now_ms - timestamp_ms).
        std::int64_t timestamp_ms = 0;

        // Innovation: difference between optimized pose and prediction (Kalman innovation)
        Eigen::Vector3f innovation = Eigen::Vector3f::Zero();  // [dx, dy, dtheta]
        float innovation_norm = 0.f;  // ||innovation|| for quick health check

        // Corner detections for this frame (world frame, for drawing)
        std::vector<CornerDetector::CornerMatch> corner_matches;
        int corners_in_fov = 0;

        // The lidar scan used for this result (robot frame, synchronized with robot_pose)
        std::vector<Eigen::Vector3f> lidar_scan;
    };

    struct OdometryPrior
    {
        bool valid = false;
        bool fresh = false;
        bool is_measured = false;
        Eigen::Vector3f delta_pose;      // [dx, dy, dtheta] in meters & radians
        torch::Tensor covariance;        // 3x3 covariance matrix
        Eigen::Matrix3f covariance_eigen = Eigen::Matrix3f::Identity();
        VelocityCommand velocity_cmd;    // The actual velocity command
        float dt;                        // Time delta
        float prior_weight = 1.0f;      // How much to trust this prior
        // Preintegrated summary of the same interval, filled only when Params::motion_preintegration
        // is on. `covariance_eigen` above is then this interval's covariance(); the Interval itself is
        // kept because the strided window has to CHAIN intervals (rc::preint::chain), which needs the
        // scale Jacobians, not just the covariance they contribute to.
        rc::preint::Interval preint{};
        bool has_preint = false;

        OdometryPrior()
            : delta_pose(Eigen::Vector3f::Zero())
            , covariance(torch::zeros({3,3}, torch::kFloat32))
            , covariance_eigen(Eigen::Matrix3f::Identity())
            , dt(0.0f)
        {}
    };

    enum class MotionPriorSource
    {
        None,
        Command,
        Measured,
        Fused,
        FallbackZero
    };

    struct MotionPriorSelection
    {
        OdometryPrior command_prior;
        OdometryPrior measured_prior;
        OdometryPrior selected_prior;
        MotionPriorSource source = MotionPriorSource::None;
        Eigen::Vector2f predicted_pos = Eigen::Vector2f::Zero();
        float predicted_theta = 0.f;
        Eigen::Matrix3f prediction_precision = Eigen::Matrix3f::Identity();
    };

    struct MotionIngressDebug
    {
        std::string command_source = "na";
        std::string odom_source = "na";
        float command_rot_raw = 0.f;
        float command_rot_normalized = 0.f;
        float command_adv_raw = 0.f;
        float command_adv_normalized = 0.f;
        float odom_rot_raw = 0.f;
        float odom_rot_normalized = 0.f;
        float odom_adv_raw = 0.f;
        float odom_adv_normalized = 0.f;
        std::int64_t command_ts_ms = 0;
        std::int64_t odom_ts_ms = 0;
    };

    // ===== Threading / Run Context =====
    /// External buffers that the localization thread reads from directly.
    struct RunContext
    {
        HighLidarBuffer* high_lidar_buffer   = nullptr;  // lidar + GT pose
        VelocityBuffer* velocity_buffer = nullptr;  // joystick / controller commands
        OdometryBuffer* odometry_buffer = nullptr;  // measured odometry (encoders/IMU)
        ImuBuffer* imu_buffer = nullptr;            // inertial samples, ~10x the odometry rate
        const SimClockMap* sim_clock = nullptr;     // producer sim clock <- local wall clock
    };

    /// Thread-safe command variants (pushed from UI thread, drained in run loop)
    struct CmdSetPolygon  { std::vector<Eigen::Vector2f> vertices; };
    struct CmdSetPose     { float x; float y; float theta; };
    struct CmdGridSearch  { std::vector<Eigen::Vector3f> lidar_points; float grid_res; float angle_res; };
    using Command = std::variant<CmdSetPolygon, CmdSetPose, CmdGridSearch>;

    RoomConcept() = default;
    ~RoomConcept();

    /// Set the run context (buffer pointers).  Must be called before start().
    void set_run_context(const RunContext& ctx) { run_ctx_ = ctx; }

    /// Start the internal localization thread.  Requires set_run_context() first.
    void start();

    /// Request the localization thread to stop and join it.
    void stop();

    /// True while the localization thread is running.
    bool is_running() const { return loc_running_.load(); }

    /// True once the first successful UpdateResult has been published.
    bool is_loc_initialized() const { return loc_initialized_.load(); }

    /// Thread-safe: get the latest UpdateResult (nullopt if not yet available).
    std::optional<UpdateResult> get_last_result() const;

    /// Register a callback fired from the LOCALIZER THREAD the instant a fresh (ok) result is stored,
    /// so the owner can publish it to the DSR graph WITHOUT waiting for the next compute() tick. The
    /// callback must be cheap and thread-safe (SpecificWorker marshals to the main thread via a
    /// Qt::QueuedConnection); it must NOT touch the DSR graph directly. Removes ~one compute-period of
    /// lidar→RT-publish latency.
    void set_on_result_ready(std::function<void()> cb) { on_result_ready_ = std::move(cb); }

    /// Dead-reckon a room←robot pose forward by a constant body velocity (adv=forward, side=lateral,
    /// rot=CCW, robot frame) over dt seconds, using the same midpoint-SE2 convention as
    /// integrate_velocity_over_window. Pure/const — used for high-rate predict-publish between lidar
    /// corrections. Returns the input pose unchanged if dt <= 0.
    Eigen::Affine2f predict_pose_forward(const Eigen::Affine2f& pose,
                                         float adv, float side, float rot, float dt) const;

    /// Optimizer-timing telemetry accumulated on the localization thread since the last call.
    struct OptTiming
    {
        std::uint32_t count       = 0;   // updates that ran (new lidar frames processed)
        std::uint32_t early_exits = 0;   // of those, how many skipped Adam/LBFGS (prediction good)
        double        avg_update_ms = 0.0;
    };
    /// Thread-safe: read AND RESET the optimizer-timing accumulators (call ~1 Hz from the UI).
    OptTiming take_optimizer_timing();

    /// Thread-safe convenience: returns [half_w, half_h, x, y, theta] or zeros.
    Eigen::Matrix<float,5,1> get_loc_state() const;

    /// Thread-safe: record the latest command sample entering the motion pipeline.
    void record_command_ingress(const std::string& source,
                                float raw_adv,
                                float raw_rot,
                                float normalized_adv,
                                float normalized_rot,
                                std::int64_t ts_ms);

    /// Thread-safe: record the latest measured odometry sample entering the motion pipeline.
    void record_odometry_ingress(const std::string& source,
                                 float raw_adv,
                                 float raw_rot,
                                 float normalized_adv,
                                 float normalized_rot,
                                 std::int64_t ts_ms);

    /// Thread-safe: push a command to be executed on the localization thread.
    void push_command(Command cmd);

    /// Wake the localization thread when a newer lidar scan is available.
    void notify_new_lidar(std::int64_t lidar_timestamp_ms);

    // ----- Initialization configuration -----
    void configure_room_from_polygon(const std::vector<Eigen::Vector2f>& polygon_vertices);
    void configure_room_from_rect(float width, float length);
    const std::vector<Eigen::Vector2f>& polygon_vertices() const { return init_polygon_vertices_; }
    std::vector<Eigen::Vector2f> nominal_room_polygon() const
    {
        if (init_use_polygon_ && init_polygon_vertices_.size() >= 3)
            return init_polygon_vertices_;

        const float half_w = init_room_width_ * 0.5f;
        const float half_l = init_room_length_ * 0.5f;
        return {
            {-half_w, -half_l},
            { half_w, -half_l},
            { half_w,  half_l},
            {-half_w,  half_l}
        };
    }
    void set_seed_pose_file(const std::string& pose_file_path);

    void set_initial_state(float width, float length, float x, float y, float phi);
    void set_polygon_room(const std::vector<Eigen::Vector2f>& polygon_vertices);
    void set_robot_pose(float x, float y, float theta, bool manual_reset = true);  // Set robot pose manually (e.g., from UI click)
    float evaluate_pose_fit(const std::vector<Eigen::Vector3f>& lidar_points,
                                     int max_samples = 300) const;
    bool is_initialized() const { return model_ != nullptr; }

    /// True while a global grid search is running, and for `hold_ms` after one finishes. The hold
    /// exists because the GUI polls at ~15-30 Hz from another thread: a search that completes between
    /// two polls would otherwise never be seen, and the operator would have no way to tell a search
    /// from a freeze. Safe to call from any thread (two atomics, no lock).
    bool is_grid_searching(int hold_ms = 1500) const
    {
        if (grid_search_active_.load(std::memory_order_relaxed))
            return true;
        const auto ended = grid_search_end_ms_.load(std::memory_order_relaxed);
        if (ended == 0) return false;
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return (now - ended) < hold_ms;
    }

    // Grid search for initial pose (solves kidnapping problem)
    // Returns true if a good pose was found, false otherwise
    bool grid_search_initial_pose(const std::vector<Eigen::Vector3f>& lidar_points,
                                   float grid_resolution = 0.5f,  // meters
                                   float angle_resolution = M_PI_4);  // 45 degrees

    Eigen::Matrix<float,5,1> get_current_state() const
    {
        if (model_) return model_->get_state();
        return Eigen::Matrix<float,5,1>::Zero();
    }

    // ===== Sliding Window (RFE) types =====
    struct WindowSlot
    {
        torch::Tensor pose;             // [3] = {x, y, theta}, requires_grad=true
        torch::Tensor lidar_points;     // [N, 3] stored observation (no grad)
        Eigen::Vector3f odometry_delta = Eigen::Vector3f::Zero(); // delta from prev slot
        Eigen::Matrix3f motion_cov = Eigen::Matrix3f::Identity(); // Σ_dyn
        int64_t timestamp_ms = 0;

        // Cached tensors (computed once at append-time, reused every Adam iteration)
        torch::Tensor odom_delta_tensor;   // [3], on device
        torch::Tensor motion_prec_tensor;  // [3,3], Σ_dyn^{-1} on device
        bool subsampled = false;           // true once old-slot subsampling has been applied

        // Quality of this slot's localization (set after Adam/early-exit, read when slot
        // becomes the oldest and its Hessian is used for the boundary prior).
        float sdf_mse_final = 0.0f;

        // Corner observations for this slot (robot frame)
        struct CornerObs {
            Eigen::Vector2f model_corner_world;  // world-frame model corner
            Eigen::Vector2f detected_robot;      // detected position in robot frame
            // Graded per-detection precision Λ_det (robot frame) from CornerDetector: the loss applies
            // this anisotropically (rᵀΛ_det r), so shallow/marginal corners contribute weakly along
            // ill-determined directions and clean asymmetric corners contribute strongly. Rank-1 for a
            // near-parallel (aperture-ambiguous) corner. Replaces the old scalar corner_obs_sigma·I.
            Eigen::Matrix2f information = Eigen::Matrix2f::Identity();
        };
        std::vector<CornerObs> corner_obs;

        // Batched corner constants (built ONCE from corner_obs, reused every optimizer closure).
        // The per-corner world corner / robot-frame detection / anisotropic precision Λ_det do NOT
        // depend on the pose being optimized, so building them per-iteration (as tiny torch::tensor
        // allocations) was pure dispatch overhead that scaled O(corners × closures). Stacked here they
        // let compute_rfe_loss evaluate the whole corner factor as a handful of batched ops.
        // Defined (non-empty) iff corner_obs is non-empty; rebuilt by rebuild_corner_batch().
        torch::Tensor corner_cw;           // [N,2] world-frame model corners
        torch::Tensor corner_detected;     // [N,2] detected positions in robot frame
        torch::Tensor corner_information;  // [N,2,2] graded precision Λ_det (robot frame)

        // Rebuild the batched corner tensors from corner_obs (call after populating corner_obs).
        void rebuild_corner_batch(torch::Device device)
        {
            const int n = static_cast<int>(corner_obs.size());
            if (n == 0)
            {
                corner_cw = corner_detected = corner_information = torch::Tensor{};
                return;
            }
            auto cw   = torch::empty({n, 2}, torch::kFloat32);
            auto det  = torch::empty({n, 2}, torch::kFloat32);
            auto info = torch::empty({n, 2, 2}, torch::kFloat32);
            auto acw = cw.accessor<float, 2>();
            auto adet = det.accessor<float, 2>();
            auto ainfo = info.accessor<float, 3>();
            for (int i = 0; i < n; ++i)
            {
                const auto& o = corner_obs[i];
                acw[i][0] = o.model_corner_world.x(); acw[i][1] = o.model_corner_world.y();
                adet[i][0] = o.detected_robot.x();    adet[i][1] = o.detected_robot.y();
                ainfo[i][0][0] = o.information(0,0);   ainfo[i][0][1] = o.information(0,1);
                ainfo[i][1][0] = o.information(1,0);   ainfo[i][1][1] = o.information(1,1);
            }
            corner_cw = cw.to(device);
            corner_detected = det.to(device);
            corner_information = info.to(device);
        }

        // Object-anchor observations for this slot (validated modelled objects as SE(2) landmarks).
        std::vector<ObjectAnchorObs> object_anchors;
    };

    struct BoundaryPrior
    {
        // Legacy (boundary_fej_schur=false): mu = dropped state's MAP pose, precision = quality-gated
        // Hessian, grad unused. Boundary loss = 0.5·(x_front−mu)ᵀ·precision·(x_front−mu).
        //
        // FEJ+Schur (boundary_fej_schur=true): mu = FROZEN marginal MODE (mean form) of the surviving
        // slot, computed from the unclamped Schur complement as mu = x₁* − Λ_marg⁻¹·g_marg (≈ x₀*+odom);
        // precision = Λ_marg = Ω − ΩΛ₀₀⁻¹Ω, eigenvalue-clamped AFTER the mode is fixed (so clamping only
        // softens confidence, never moves the anchor). Same pure-quadratic loss 0.5·Δᵀ·precision·Δ,
        // Δ = wrap(x_front − mu). mu is NOT updated between marginalizations (the FEJ discipline).
        Eigen::Vector3f mu = Eigen::Vector3f::Zero();            // dropped-state pose (legacy) / frozen marginal mode (Schur)
        Eigen::Matrix3f precision = Eigen::Matrix3f::Identity(); // Σ^{-1} (legacy Hessian) / Λ_marg (Schur)
        Eigen::Vector3f grad = Eigen::Vector3f::Zero();          // vestigial (mean form uses no linear term)
        bool valid = false;
    };

    UpdateResult update(const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar,
                        const std::vector<rc::VelocityCommand> &velocity_history,
                        const std::vector<rc::OdometryReading> &odometry_history);

    Params params;

    /// Hand the latest validated object anchors (gathered on the MAIN thread from the DSR graph)
    /// to the localizer.  Copied into the newest window slot at the next update().  Thread-safe.
    /// Room's private landmark estimates (node id → room-frame position) — DISPLAY only. Empty unless
    /// object_anchor_optimize_landmark is on. Thread-safe copy under the anchor lock.
    std::map<std::uint64_t, Eigen::Vector2f> object_landmarks() const
    {
        std::scoped_lock lk(object_anchors_mutex_);
        std::map<std::uint64_t, Eigen::Vector2f> out;
        for (const auto& [id, e] : landmark_estimates_) out.emplace(id, e.p);
        return out;
    }

    /// The anchors the localizer is currently using — for DISPLAY only (the 2D canvas overlay).
    /// Thread-safe; returns a copy, so the caller never holds the lock while drawing.
    std::vector<ObjectAnchorObs> object_anchors() const
    {
        std::scoped_lock lk(object_anchors_mutex_);
        return latest_object_anchors_;
    }

    void set_object_anchors(std::vector<ObjectAnchorObs> anchors)
    {
        std::scoped_lock lk(object_anchors_mutex_);
        latest_object_anchors_ = std::move(anchors);
    }

    // Process noise covariance (diagonal [x, y, theta])
    Eigen::Vector3f process_noise = {0.01f, 0.01f, 0.01f};

    // Current covariance estimate [3x3]
    Eigen::Matrix3f current_covariance = Eigen::Matrix3f::Identity() * 0.1f;
    // Last pose the optimizer produced that was finite — the bottom rung of the divergence-recovery
    // cascade (corners -> lidar/SDF -> odometry -> this).
    Eigen::Vector3f last_good_pose_ = Eigen::Vector3f::Zero();
    bool            last_good_pose_valid_ = false;

private:
   // ===== Threading internals =====
   RunContext run_ctx_;
   std::thread loc_thread_;
   std::atomic<bool> stop_requested_{false};
   std::atomic<bool> loc_running_{false};
   std::atomic<bool> loc_initialized_{false};

   mutable std::mutex result_mutex_;
   std::optional<UpdateResult> last_result_;

   // Latest object anchors from the graph (set on main thread, consumed by the localizer thread).
   mutable std::mutex object_anchors_mutex_;
   std::vector<ObjectAnchorObs> latest_object_anchors_;

   /// Room's PRIVATE landmark estimates, keyed by graph node id: room-frame position + the precision it
   /// has accumulated. Persist across frames — that accumulation over diverse viewing azimuths is the
   /// whole point (a position-only observation gives 2 constraints, so one viewpoint never determines a
   /// landmark; see ray_anisotropic_cov's bearing-only limit). Born from the producer's belief, then
   /// refined by room's own observations alone.
   struct LandmarkEstimate
   {
       Eigen::Vector2f p = Eigen::Vector2f::Zero();
       Eigen::Matrix2f information = Eigen::Matrix2f::Zero();
   };
   std::map<std::uint64_t, LandmarkEstimate> landmark_estimates_;
   std::function<void()> on_result_ready_;   // fired (localizer thread) after a fresh ok result is stored

   std::mutex cmd_mutex_;
   std::vector<Command> pending_commands_;
    std::atomic<bool> commands_pending_{false};
    std::mutex wake_mutex_;
    std::condition_variable wake_cv_;
    std::int64_t latest_notified_lidar_ts_ = std::numeric_limits<std::int64_t>::min();

   /// The localization loop body (runs on loc_thread_)
   void run();

   std::shared_ptr<Model> model_;
   std::int64_t last_lidar_timestamp = 0;
   UpdateResult last_update_result;
   bool needs_orientation_search_ = true;  // Search for best orientation on first update
   // Set when bootstrap seeded the model from the saved pose before any sweep had arrived, so the
   // seed could not be checked against the room yet. The first new frame runs validate_seed_pose.
   // Localizer-thread only (bootstrap and run() are both on loc_thread_) — no atomic needed.
   bool pending_seed_validation_ = false;
   bool validate_seed_pose(const std::vector<Eigen::Vector3f>& pts);

   // Grid-search liveness for the UI (written on the localizer thread, read on the GUI thread).
   std::atomic<bool>         grid_search_active_{false};
   std::atomic<std::int64_t> grid_search_end_ms_{0};

   // Manual pose reset - skip optimization for a few frames
   int manual_reset_frames_ = 0;  // Counter to skip optimization after manual reset

   // Smoothed pose to reduce jitter (legacy, used only when rfe_window_size == 1)
   Eigen::Vector3f smoothed_pose_ = Eigen::Vector3f::Zero();  // [x, y, theta]
   bool has_smoothed_pose_ = false;

   // Flip instrumentation: last OPTIMIZED pose, to detect a ~180° yaw jump and dump why (see [FLIP] log).
   float flip_prev_x_ = 0.f, flip_prev_y_ = 0.f, flip_prev_th_ = 0.f;
   bool  flip_prev_valid_ = false;

   // ===== Recovery Manager =====
   struct RecoveryManager
   {
       int consecutive_bad_frames = 0;
       int cooldown = 0;

       /// Returns true if recovery should be triggered now. `avg_sdf_err` must be the WORST of the
       /// prediction error and the post-fit residual — see the call site for why watching only the
       /// post-fit number misses the case the operator actually sees.
       ///
       /// NOTE there is deliberately no `iterations_used <= 0` guard any more. It used to skip every
       /// early-exit frame on the assumption that sdf_mse was stale then; it is not — the early-exit
       /// path sets res.sdf_mse itself (room_concept.cpp, `res.sdf_mse = mean_sdf_pred`). The guard
       /// made recovery structurally blind in exactly the regime where a silent mislocalization lives:
       /// a 180°-flipped pose in a symmetric room is SDF-ambiguous, so it early-exits every frame, so
       /// recovery never accumulated a single bad frame and could never fire.
       bool check(float avg_sdf_err, int /*iterations_used*/,
                  float threshold, int consecutive_count)
       {
           if (cooldown > 0) { --cooldown; return false; }
           if (avg_sdf_err > threshold)
           {
               ++consecutive_bad_frames;
               return consecutive_bad_frames >= consecutive_count;
           }
           consecutive_bad_frames = 0;
           return false;
       }
       void on_recovery_done(int cooldown_frames)
       { consecutive_bad_frames = 0; cooldown = cooldown_frames; }
       void reset() { consecutive_bad_frames = 0; cooldown = 0; }
   };
   RecoveryManager recovery_;

   // ===== Loss/recovery episode log =====
   // One row per grid_search_initial_pose() call, written wherever it returns. The per-frame debug log
   // could only ever show recovery INDIRECTLY — I had to infer episodes from window_size resets and
   // proxy the early-exit metric through final_loss, which is exactly the kind of inference that has
   // been wrong twice in this work. This records what the search actually did, so the algorithm can be
   // improved from measurement instead of from a story about it.
   //
   // The load-bearing column is ESS (effective sample size of the softmax weights over the evaluated
   // poses): ~1 means one pose explains the scan and the fix is decisive; large means a flat valley of
   // near-equal candidates, i.e. the scan genuinely does not determine the pose and ANY winner drawn
   // from it is arbitrary. That distinction is invisible in a success/fail bit, and it is the one that
   // decides whether the answer is "search harder" or "go somewhere the geometry is informative".
   struct SearchEpisode
   {
       std::int64_t ts_ms = 0;
       const char*  trigger = "unknown";   // recovery | map_trust | seed | manual
       int          n_lidar = 0;
       float        incumbent_x = 0.f, incumbent_y = 0.f, incumbent_theta = 0.f;
       float        incumbent_loss = std::numeric_limits<float>::quiet_NaN();
       float        good_thr = 0.f;
       int          stage = -1;            // 0 = yaw sweep, 1 = local lattice, 2 = coarse global, 3 = fine
       float        best_x = 0.f, best_y = 0.f, best_theta = 0.f;
       float        best_loss = std::numeric_limits<float>::quiet_NaN();
       float        topk_best = std::numeric_limits<float>::quiet_NaN();   // Stage-2 candidate spread:
       float        topk_worst = std::numeric_limits<float>::quiet_NaN();  // how flat is the valley
       float        ess = std::numeric_limits<float>::quiet_NaN();
       float        cov_xx = std::numeric_limits<float>::quiet_NaN();
       float        cov_yy = std::numeric_limits<float>::quiet_NaN();
       float        cov_tt = std::numeric_limits<float>::quiet_NaN();
       int          n_evals = 0;
       float        jump_m = 0.f;          // distance from the incumbent to the committed pose
       float        jump_rad = 0.f;
       bool         success = false;
       float        duration_ms = 0.f;
       float        beta = std::numeric_limits<float>::quiet_NaN();   // softmax temperature actually used
       int          n_points = 0;                                     // N behind the median's std error
   };
   // Why the next grid search was invoked. Set by the caller immediately before the call; the search
   // itself has no way to know, and "which trigger fires most and which of them actually resolve" is
   // the first question the episode log has to answer.
   const char*   search_trigger_ = "unknown";
   std::ofstream recovery_log_;
   std::string   recovery_log_path_;
   void write_search_episode(const SearchEpisode& e);

   // ===== Window Manager (RFE sliding window) =====
   struct WindowManager
   {
       std::deque<WindowSlot> window;
       BoundaryPrior boundary_prior;

       bool empty() const { return window.empty(); }
       size_t size() const { return window.size(); }
       void clear() { window.clear(); boundary_prior.valid = false; }
       WindowSlot& newest() { return window.back(); }
       const WindowSlot& newest() const { return window.back(); }

       /// Slide if full, append new slot.  Returns true if window was slid.
       /// mu_quality_threshold: only update boundary_prior.mu from the dropped slot if its
       /// sdf_mse_final is below this value (Solution C — legacy path only).
       /// fej_schur: when true, do NOT touch boundary_prior here — marginalize_oldest() (called
       /// BEFORE append) has already folded the dropped slot into the FEJ+Schur prior.
       bool append(WindowSlot slot, int max_window_size,
                   float mu_quality_threshold = std::numeric_limits<float>::max(),
                   bool fej_schur = false);

       /// Subsample lidar in all slots except the newest.
       void subsample_old_slots(int max_pts_per_slot);

       /// Collect all pose tensors for the optimizer.
       std::vector<torch::Tensor> collect_params() const;

       /// Build the full RFE loss over the current window (Eq. 27).
       /// boundary_weight scales the boundary prior term (1.0 = full, 0.0 = disabled).
       torch::Tensor compute_rfe_loss(const Model& model, const Params& params,
                                       torch::Device device,
                                       float boundary_weight = 1.0f) const;

       /// Per-term loss breakdown — called once after Adam for diagnostic logging.
       struct LossBreakdown {
           float boundary = 0.f;
           float obs      = 0.f;
           float motion   = 0.f;
           float corner   = 0.f;
           float object   = 0.f;
       };
       LossBreakdown compute_rfe_loss_breakdown(const Model& model, const Params& params,
                                                 torch::Device device) const;

       /// Recompute boundary prior Hessian from oldest surviving slot (LEGACY path).
       void recompute_boundary_prior(const Model& model, const Params& params,
                                      torch::Device device);

       /// FEJ + Schur marginalization (boundary_fej_schur path). Call BEFORE append() pops, while
       /// both the dropping slot (window.front() = x₀) and its blanket (window[1] = x₁) are live.
       /// Forms the exact Schur complement of x₀ over x₁ from {previous marginal on x₀, x₀'s obs
       /// factor (soft quality-weighted), x₀↔x₁ motion factor}, and stores a FEJ-frozen prior on x₁.
       /// No-op unless the window is full (>= max_window_size) with size > 1.
       void marginalize_oldest(const Model& model, const Params& params, torch::Device device);
   };
   WindowManager window_mgr_;

   // ---- Strided-window state (see Params::window_stride_enabled) ----
   // Pose at which the last slot was ADMITTED, and the odometry delta + covariance accumulated since.
   // While the newest slot is being replaced rather than appended, the motion factor it carries has to
   // describe the whole interval back to the last admitted slot, not the last frame.
   Eigen::Vector3f stride_last_admitted_{0.f, 0.f, 0.f};
   bool            stride_has_admitted_ = false;
   Eigen::Vector3f stride_delta_accum_  = Eigen::Vector3f::Zero();
   Eigen::Matrix3f stride_cov_accum_    = Eigen::Matrix3f::Zero();
   // Preintegrated form of the same accumulation (Params::motion_preintegration). Chained with
   // rc::preint::chain() rather than summed: an error in the heading accumulated so far rotates all
   // the translation that follows, and `stride_cov_accum_ += ...` drops exactly that term.
   rc::preint::Interval stride_preint_accum_{};
   bool preint_announced_ = false;   // one-shot "the propagated covariance is really in force" log

   // ---- Debug-log mirrors, set where the newest slot is BUILT so BOTH writer paths can emit them ----
   // The early-exit writer lives in a different function from the slot build, so anything it must log
   // has to be a member. Getting this wrong is how columns end up hardcoded to zero on one path only
   // (n_lidar still is), which then silently halves the usable population of every analysis.
   /// The appended tail of a debug-log row, emitted by BOTH writers so they cannot drift apart. One
   /// function rather than two copies precisely because two copies is how this file's columns got out
   /// of step before.
   void write_debug_tail();
   Eigen::Matrix3f last_slot_motion_cov_  = Eigen::Matrix3f::Zero();  // full 3x3, for the off-diagonals
   float           last_imu_cover_        = -1.f;  // fraction of segments whose dtheta came from the gyro
   int             last_preint_samples_   = 0;    // odometry samples summarised into this slot's factor
   float           last_preint_duration_s_ = 0.f; // and over how long — reveals strided chaining
   bool imu_injection_announced_ = false;  // one-shot "dtheta really is coming from the gyro" log
   // Rolling stats for the periodic [ImuInject] line. The one-shot above proves the path bound at
   // all; these say whether it is still binding, how much of each interval the IMU actually covers,
   // and -- the number that matters -- how much heading the gyro is taking OUT of the wheel estimate.
   // A silently degrading channel (IMU stalls, buffer too shallow, clock map lost) reads as normal in
   // every outcome metric until it has already cost accuracy.
   int          imu_seg_used_ = 0, imu_seg_total_ = 0;
   double       imu_dtheta_sum_ = 0.0, wheel_dtheta_sum_ = 0.0;  // rad, over IMU-covered segments only
   bool         imu_stats_sim_clock_ = false;
   std::int64_t imu_stats_last_log_ms_ = 0;

   // Running second moment of the innovation, per axis (x, y, theta). See Params::adaptive_cov_enabled.
   Eigen::Vector3f innov_m2_ = Eigen::Vector3f::Zero();
   /// Raise the published covariance to at least the error the innovations demonstrate. Call ONLY on
   /// frames that actually had a prediction — a zero innovation otherwise means "no information", not
   /// "perfect agreement", and folding those in would drag the estimate back down.
   void apply_adaptive_covariance(UpdateResult& res);

   /// Drop the strided-window bookkeeping. MUST accompany every window_mgr_.clear(): after a recovery
   /// or relocalization the last-admitted pose refers to a trajectory that no longer exists, so
   /// comparing against it would suppress admissions until the robot happened to travel far from a
   /// stale reference.
   void reset_stride_state()
   {
       stride_has_admitted_ = false;
       stride_delta_accum_.setZero();
       stride_cov_accum_.setZero();
       stride_preint_accum_ = rc::preint::Interval{};
   }

   // Prediction-based early exit tracking
   int tracking_step_count_ = 0;
   int prediction_early_exits_ = 0;
   // Last mean |SDF| at the predicted pose evaluated by try_prediction_early_exit() this frame
   // (the early-exit decision variable). NaN when the gate's pre-conditions weren't met. Consumed
   // by update() to expose it on the Adam path too (the value that TRIGGERED optimization).
   float last_early_exit_metric_ = std::numeric_limits<float>::quiet_NaN();
   /// Median |SDF| at the SAME predicted pose and points as last_early_exit_metric_ (which is
   /// the mean). Kept so that recovery can compare like with like: its threshold is calibrated
   /// on medians (see RecoveryLossThreshold and compute_seed_error), and the mean runs ~15-20%
   /// higher. Logging both also makes the median/mean ratio measurable from any run.
   float last_pred_sdf_median_  = std::numeric_limits<float>::quiet_NaN();

   // Velocity-adaptive gradient weights [x, y, theta]
   Eigen::Vector3f current_velocity_weights_ = Eigen::Vector3f::Ones();

    // Corner detector
    CornerDetector corner_detector_;

    // Startup initialization configuration
    bool init_use_polygon_ = false;
    std::vector<Eigen::Vector2f> init_polygon_vertices_;
    float init_room_width_ = 10.f;
    float init_room_length_ = 10.f;
    std::string seed_pose_file_path_;

   // ===== Debug Logging (localization thread only — no mutex needed) =====
   std::ofstream      debug_log_;
   // Lean per-update optimizer-timing CSV (loc-thread only; no lock). Captures the cost distribution
   // and CUDA warmup curve. Opened lazily on the first update; gated by params.optimizer_timing_csv.
   std::ofstream      opt_csv_;
   bool               opt_csv_open_attempted_ = false;
   // Per-symmetry-check trial CSV (loc-thread only; no lock). Gated by params.symmetry_debug_csv.
   std::ofstream      symmetry_csv_;
   bool               symmetry_csv_open_attempted_ = false;

   std::ofstream      flip_csv_;   // one row per detected ~180° flip (tmp/sdf_localizer/flips_<ts>.csv)
   bool               flip_csv_open_attempted_ = false;
   // Hierarchical boundary-precision A/B trace (etc/hier_prec.csv): one row per converged frame while
   // params.hier_prec_boundary_enabled. Lazy-opened on first use; loc-thread only, no lock.
   std::ofstream      hier_prec_csv_;
   bool               hier_prec_csv_open_attempted_ = false;

   // ── Per-model-corner attribution (etc/corner_stats.csv) ────────────────────────────────────────
   // Accumulated over the run and rewritten whole every kCornerStatsRewritePeriod corner frames, so
   // the file is always a current snapshot rather than an append log to be re-aggregated. Answers the
   // question the aggregate rej_* counters cannot: WHICH vertices are worth keeping as landmarks.
   // Loc-thread only, no lock — same discipline as opt_csv_/hier_prec_csv_.
   struct CornerVertexStats
   {
       int    in_fov = 0;        // visible, detection attempted
       int    occluded = 0;      // in range but occluded — not the corner's own fault
       int    accepted = 0;      // survived to a match that entered the loss
       int    rival_n = 0;       // accepted samples that had a competing model corner in gate
       double assoc_prob_sum = 0.0;
       double runnerup_chi2_sum = 0.0;   // over rival_n only (INFEASIBLE rivals excluded)
       double resid_sum = 0.0;           // ‖detected − predicted‖ (m)
       double lambda_min_sum = 0.0;      // smallest eigenvalue of Λ_det — the degenerate direction
       double lambda_max_sum = 0.0;
       int    suppressed = 0;            // CUMULATIVE frames retired by the information-yield rule —
                                         // "has been retired at some point", NOT the current state
       bool   retired_now = false;       // state at the most recent match. Read THIS for "is it
                                         // retired right now"; with hysteresis the two differ often
       float  last_yield = 0.f;          // most recent running λ_min estimate behind that decision
   };
   static constexpr int kCornerStatsRewritePeriod = 200;
   std::map<int, CornerVertexStats> corner_vertex_stats_;
   int                corner_stats_frames_ = 0;
   std::ofstream      corner_stats_csv_;
   void accumulate_corner_stats(const CornerDetector::DetectionResult& det);
   void write_corner_stats_csv();
   RerunLogger        rerun_logger_;
   int                rerun_frame_counter_ = 0;
   int                symmetry_check_counter_ = 0;
   float              symmetry_flip_evidence_ = 0.f;  // leaky accumulator of contrary-orientation evidence
   int                good_fit_streak_        = 0;    // consecutive good-SDF frames (track establishment)
   bool               rerun_room_polygon_sent_ = false;
   std::vector<float> last_adam_losses_;    // per-iteration losses from last Adam/LBFGS run
   float              last_loss_init_  = 0.f;  // loss before first step
   float              prev_sdf_mse_    = 0.f;  // sdf_mse from previous frame (for boundary quality gate)
   // ===== Hierarchical boundary precision (HIERARCHICAL_PRECISION.md), persisted across frames =====
   float              u_b_             = 0.f;  // boundary log-precision posterior; exp(u_b_) = ⟨π⟩ = boundary_weight
   bool               u_b_init_        = false;// lazily seeded to g(v) on first use / after a recovery reset
   float              map_trust_v_     = 0.f;  // Option-A slow hyper-state v; g(v)=u0+g_gain·v predicts u_b_
   int                map_trust_low_streak_     = 0;  // consecutive frames exp(u_b_) < hier_prec_reloc_floor
   int                map_trust_reloc_cooldown_ = 0;  // frames left suppressing a map-trust relocalization
   /// Stage-1 boundary-precision inference: one fast step on u_b_ (from the boundary residual r_b, using
   /// the post-optimization posterior covariance sigma_x for the tr(Λ_b Σ) term) plus one slow step on the
   /// map_trust hyper-state v. No-op unless params.hier_prec_boundary_enabled. Call once per frame after
   /// the window pose has converged.
   void               update_boundary_hyperprecision(const Eigen::Matrix3f &sigma_x);
   /// Close the rotation early-exit gap: on an early-exit frame with large |dtheta|, nudge u_b_ down using a
   /// SURROGATE residual r_ee=(mean_sdf_pred/sigma_sdf)² (no boundary factor is evaluated on early-exit), so a
   /// degrading rotation that keeps early-exiting can still collapse map-trust. Fast-only (leaves v to the
   /// optimized path). No-op unless hier_prec_boundary_enabled && hier_prec_reloc_enabled.
   void               nudge_map_trust_early_exit(float mean_sdf_pred, float dtheta);
   /// Append one row to etc/hier_prec.csv (lazy-opened). src ∈ {opt, ee, reloc}. Loc-thread only, no lock.
   void               log_hier_prec_row(const char *src, float r_b, float quad, float trace, bool reloc_fired);
   WindowManager::LossBreakdown last_loss_breakdown_;  // FE term breakdown after last Adam
   float              last_lbfgs_grad_norm_ = 0.f; // final gradient inf-norm after L-BFGS (0 for Adam/early-exit)
   // Per-frame timing — t_update_start_ set at entry of update(), shared with early-exit path
   std::chrono::high_resolution_clock::time_point t_update_start_;
   float              last_t_adam_ms_      = 0.f;
   float              last_t_cov_ms_       = 0.f;
   float              last_t_breakdown_ms_ = 0.f;
   // Optimizer-timing telemetry (written on loc_thread_, drained by the worker via take_optimizer_timing).
   std::atomic<std::uint32_t> opt_total_count_{0};
   std::atomic<std::uint32_t> opt_earlyexit_count_{0};
   std::atomic<std::uint64_t> opt_update_us_sum_{0};
   OdometryPrior      last_measured_prior_; // saved by apply_dual_prior_fusion for logging
    OdometryPrior      last_selected_prior_;
    MotionPriorSource  last_motion_prior_source_ = MotionPriorSource::None;
   Eigen::Matrix3f    last_cmd_cov_ = Eigen::Matrix3f::Identity();
    bool               last_command_prior_fresh_ = false;
    bool               last_measured_prior_fresh_ = false;
   std::int64_t       prev_lidar_ts_for_log_ = 0;
   std::string        slot_poses_pre_;    // packed slot poses before Adam  (debug log)
   std::string        slot_poses_post_;   // packed slot poses after Adam
   std::string        slot_sdf_mse_str_;  // packed sdf_mse_final per slot
   std::string        debug_log_path_;    // full path of the current log file (includes timestamp)
    mutable std::mutex motion_ingress_debug_mutex_;
    MotionIngressDebug motion_ingress_debug_;
   void init_debug_log();
    MotionIngressDebug get_motion_ingress_debug() const;

   // ===== Online motion model learned state =====
   // Initialised to -1 (sentinel = "warmup not done; use static params").
   // After motion_learn_min_frames quality updates they replace the static Params values.
   float           learned_slip_k_              = -1.f;  // learned encoder_rot_slip_k
   float           learned_odom_noise_trans_    = -1.f;  // learned odom_noise_trans fraction
   Eigen::Vector3f learned_odom_bias_           = Eigen::Vector3f::Zero(); // [dx,dy,dtheta] bias
   int             motion_learn_good_frames_    = 0;     // quality frames accumulated
   // Robust buffer for trans-noise samples: collect N then take median for one EMA step.
   std::vector<float> trans_noise_sample_buf_;
   void            adapt_motion_model();

   // ===== Differential test: RFE vs single-step =====
   struct DiffTestStats
   {
       double pred_sdf_sum = 0;     // sum of prediction-only SDF median
       double single_sdf_sum = 0;   // sum of single-step Adam SDF median
       double rfe_sdf_sum = 0;      // sum of full-RFE SDF median
       int count = 0;
       int adam_frames = 0;         // frames where Adam actually ran
       int rfe_wins = 0;            // frames where RFE < single-step (SDF)
       int single_wins = 0;         // frames where single-step < RFE (SDF)
       int early_exit_seen = 0;     // total early-exit frames seen (for sampling)

       // Pose jitter: residual motion after subtracting odometry prediction.
       // Lower jitter = more temporally consistent.
       double rfe_jitter_sum = 0;       // sum of ||RFE_pose - predicted_pose||₂
       double single_jitter_sum = 0;    // sum of ||single_pose - predicted_pose||₂
       double rfe_theta_jitter_sum = 0; // sum of |wrap(RFE_theta - pred_theta)|
       double single_theta_jitter_sum = 0;

       // Correction consistency: how much each method's correction vector
       // changes between consecutive frames. Lower = smoother.
       // correction[t] = optimised_pose[t] - predicted_pose[t]
       float prev_rfe_cx = 0, prev_rfe_cy = 0, prev_rfe_ctheta = 0;
       float prev_single_cx = 0, prev_single_cy = 0, prev_single_ctheta = 0;
       bool has_prev = false;
       double rfe_corr_consistency_sum = 0;     // sum ||rfe_corr[t] - rfe_corr[t-1]||₂
       double single_corr_consistency_sum = 0;  // sum ||single_corr[t] - single_corr[t-1]||₂
       double rfe_theta_consistency_sum = 0;    // sum |wrap(rfe_θcorr[t] - rfe_θcorr[t-1])|
       double single_theta_consistency_sum = 0;
   };
   DiffTestStats diff_test_;

   /// Run a single-step Adam (no window) on the given scan at the predicted pose.
   /// Returns the SDF median-absolute-error after optimisation.
   float shadow_single_step_adam(const torch::Tensor& points_tensor,
                                 float pred_x, float pred_y, float pred_theta) const;

   // Compute velocity-adaptive weights based on motion profile
   Eigen::Vector3f compute_velocity_adaptive_weights(const OdometryPrior& odometry_prior);
    float estimate_orientation_from_points(const std::vector<Eigen::Vector3f>& pts) const;
    void resolve_initial_yaw_ambiguity(const std::vector<Eigen::Vector3f>& lidar_points, float prior_phi);
    bool bootstrap_initialization_from_lidar();

    struct PredictionState
    {
        torch::Tensor propagated_cov;  // Predicted covariance
        bool have_propagated = false;   // Whether prediction was performed
        std::vector<float> previous_pose;  // Robot pose BEFORE prediction (for prior loss)
        std::vector<float> predicted_pose; // Robot pose after prediction
    };

    // Pre-allocated tensors for predict_step (avoid per-frame allocation)
    torch::Tensor predict_F_;   // [dim x dim] Jacobian
    torch::Tensor predict_Q_;   // [dim x dim] process noise
    int predict_alloc_dim_ = 0; // dimension they were allocated for

    Eigen::Matrix3f compute_motion_covariance(const OdometryPrior &odometry_prior,
                                              bool is_measured_odometry = false);
    RoomConcept::OdometryPrior compute_odometry_prior(
                    const std::vector<VelocityCommand>& velocity_history,
                    const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar);
    /// `preint_out`, when non-null, additionally receives the interval PREINTEGRATED over the same
    /// samples — covariance and scale Jacobians propagated step by step (se2_preintegration.h). The
    /// returned mean is computed by the same lines either way, so passing it cannot change the mean.
    Eigen::Vector3f integrate_velocity_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<VelocityCommand> &velocity_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms,
                                                   rc::preint::Interval *preint_out = nullptr);

    /// Integrate measured odometry (adv, side, rot) over the same time window.
    /// `imu_history` is optional; when it covers a segment, that segment's heading change comes from
    /// the gyro instead of the wheel-derived rot. Bounds are WALL epoch-ms and are converted
    /// internally to whatever clock the readings are measured in -- see the note at the definition.
    Eigen::Vector3f integrate_odometry_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<OdometryReading> &odometry_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms,
                                                   rc::preint::Interval *preint_out = nullptr,
                                                   const std::vector<ImuReading> *imu_history = nullptr,
                                                   const SimClockMap *sim_clock = nullptr);

    /// Compute odometry prior from measured velocities (encoder/IMU feedback)
    OdometryPrior compute_measured_odometry_prior(
                    const std::vector<OdometryReading>& odometry_history,
                    const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar);

    /// Fuse command prior and measured odometry prior into a single Gaussian
    /// Returns: (fused_mean, fused_precision) where mean is [x, y, theta]
    std::pair<Eigen::Vector3f, Eigen::Matrix3f> fuse_priors(
        const Eigen::Vector3f &pred_cmd, const Eigen::Matrix3f &cov_cmd,
        const Eigen::Vector3f &pred_odom, const Eigen::Matrix3f &cov_odom) const;

    PredictionState predict_step(std::shared_ptr<Model> &room,
                                  const OdometryPrior &odometry_prior,
                                  bool is_localized);

    // ===== update() helper methods =====
    static std::string_view motion_prior_source_name(MotionPriorSource source);

    /// Compute command and measured priors, then select a single motion prior.
    MotionPriorSelection build_motion_prior_selection(
        const std::vector<VelocityCommand>& velocity_history,
        const std::vector<OdometryReading>& odometry_history,
        const std::pair<std::vector<Eigen::Vector3f>, std::int64_t>& lidar);

    /// Check if the predicted pose already has low SDF error and can skip Adam.
    /// Returns an UpdateResult if early exit is taken, nullopt otherwise.
    std::optional<UpdateResult> try_prediction_early_exit(
        const torch::Tensor& points_tensor,
        const Eigen::Vector3f& slot_odom_delta,
        const OdometryPrior& odometry_prior,
        std::int64_t lidar_timestamp_ms);

    /// Run the Adam optimisation loop over the sliding window.
    /// Returns {last_loss, iterations_used}.
    std::pair<float, int> run_adam_loop(const OdometryPrior& odometry_prior);

    /// Run the L-BFGS optimisation loop over the sliding window.
    /// Returns {last_loss, func_evaluations}.
    std::pair<float, int> run_lbfgs_loop(const OdometryPrior& odometry_prior);

    /// Run the Levenberg-Marquardt backend (room_gn_solver) over the sliding window, writing the
    /// solution back into the slot pose tensors. Returns {last_loss, iterations}. On failure the
    /// window is left exactly as it was and the loss is NaN.
    std::pair<float, int> run_gn_loop(const OdometryPrior& odometry_prior);

    /// Run the GN backend on the CURRENT window WITHOUT keeping its answer, and log it beside the
    /// authoritative backend's. Call after the authority has run; poses_before is the state both
    /// started from, poses_after the authority's solution (restored on return).
    void run_gn_shadow(const std::vector<Eigen::Vector3f>& poses_before,
                       const std::vector<Eigen::Vector3f>& poses_after,
                       float authority_loss, int authority_iters, float authority_ms,
                       std::int64_t timestamp_ms);

    /// The boundary-prior precision scale for this frame (hierarchical π=exp(u_b) when enabled, else
    /// the legacy quality gate). One definition, so every backend minimises the same objective.
    float boundary_weight_now() const;

    /// Slot poses as plain [x, y, θ], newest last — the exchange format with the GN backend.
    std::vector<Eigen::Vector3f> read_window_poses() const;
    void write_window_poses(const std::vector<Eigen::Vector3f>& poses);

    std::ofstream gn_shadow_csv_;

    /// Compute posterior covariance via autograd Hessian.
    /// Updates current_covariance and returns {covariance, condition_number}.
    std::pair<Eigen::Matrix3f, float> compute_posterior_covariance(
        const torch::Tensor& points_tensor);

    // Compute the exact 3×3 Hessian of a scalar loss w.r.t. a 3-vector via double-backprop.
    static Eigen::Matrix3f autograd_hessian_3x3(const torch::Tensor& loss,
                                                 const torch::Tensor& param);

    // Find best initial orientation by testing multiple candidates (0°, 90°, 180°, 270°)
    float find_best_initial_orientation(const std::vector<Eigen::Vector3f>& lidar_points,
                                        float x, float y, float base_phi);

    // points to tensor [N,3] - overload for Eigen::Vector3f
    static torch::Tensor points_to_tensor_xyz(const std::vector<Eigen::Vector3f> &points,
                                               torch::Device device = torch::kCPU)
    {
        const auto N = static_cast<long>(points.size());
        // Build on CPU first: data_ptr() of a CUDA tensor is a DEVICE pointer, and a host std::memcpy
        // into it SIGSEGVs (the UseCuda=true crash). Fill host-side, then move to the target device.
        auto tensor = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        auto ptr = tensor.data_ptr<float>();
        // Eigen::Vector3f is 3 contiguous floats — bulk-copy each point
        for (long i = 0; i < N; ++i)
            std::memcpy(ptr + i * 3, points[i].data(), 3 * sizeof(float));
        return device.is_cpu() ? tensor : tensor.to(device);
    }

    /// THE fit metric for this class: median |SDF| in METRES, robust to the outliers that furniture
    /// (absent from the polygon) always contributes. Every pose-quality number must be reduced with
    /// this one function so they stay comparable — recovery's trigger, the grid search's success bar,
    /// the symmetry check's candidate scores and the stability gate are all judged against each other,
    /// and each was previously computing its own variant (mean-squared m², sqrt-of-a-length) while
    /// being compared as though the units matched. See UpdateResult::sdf_mse.
    static float median_abs_sdf(const torch::Tensor &sdf_vals)
    {
        return torch::median(torch::abs(sdf_vals)).item<float>();
    }

    // Fit metric at the model's CURRENT pose.
    static float compute_sdf_median_abs(const torch::Tensor &points_xyz, const Model &m)
    {
        return median_abs_sdf(m.sdf(points_xyz));
    }
};

} // namespace rc

