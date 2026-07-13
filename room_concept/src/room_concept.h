#pragma once

#include <memory>
#include <vector>
#include <deque>
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
        float cmd_noise_rot   = 0.10f;   // Fractional rotation noise per radian of rotation
        float cmd_noise_base  = 0.05f;   // Base position noise even when stationary (m)
        float stationary_noise_damping = 0.7f;  // Multiplier applied to base noise when near-stationary

        // Process noise for measured odometry prior (encoder/IMU, more reliable)
        float odom_noise_trans = 0.08f;  // Fractional position noise per meter of motion
        float odom_noise_rot   = 0.04f;  // Fractional rotation noise per radian of rotation
        float odom_noise_base  = 0.01f;  // Base position noise even when stationary (m)
        float odom_noise_scale = 1.0f;   // Multiplier on all odom noise params (>1 simulates worse odometry)

        // ===== Encoder angular slip model =====
        // At high angular speeds, wheel encoders under-report rotation due to wheel slip.
        // The rotation covariance for the measured odometry prior is inflated by:
        //   rot_var_extra = (encoder_rot_slip_k * |vel_rot|)^2
        // so that the Bayesian fusion trusts the command (cmd_dth) more than the encoder
        // at high angular speeds.  Set to 0 to disable.
        float encoder_rot_slip_k = 0.15f;  // rad uncertainty per rad/s of angular speed

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
        float grid_search_good_threshold = 1.0f;      // SDF loss < this = acceptable pose
        int orientation_search_max_samples = 100;      // Lidar subsample for orientation candidates

        // ===== Optimizer Selection =====
        // "ADAM"  — adaptive moment estimation (current default)
        // "LBFGS" — limited-memory BFGS with Wolfe line search (faster convergence)
        // "CAVI"  — coordinate-ascent variational inference (reserved, not yet implemented)
        std::string optimizer_type = "LBFGS";

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
        float default_slot_motion_cov = 0.01f;         // Default diagonal for slot motion covariance

        // ===== Velocity-Adaptive =====
        float combined_motion_weight = 1.2f;           // Weight when both translating and rotating

        // ===== Corner Detection =====
        bool  enable_corner_tracking = true;      // Master switch for corner factors in Adam loss
        bool  sdf_current_slot_only = false;           // If true, SDF obs evaluated only for the newest slot;
                                                       // older slots contribute via motion + corner factors only.
        float corner_obs_sigma = 0.04f;                // Corner measurement noise (meters)
        int   min_tracking_steps_for_corners = 5;      // Require this many tracking steps before adding corner factors
        int   corner_max_slots = 5;                    // Only apply corner factors to the newest N slots
        float corner_huber_delta = 0.3f;               // Huber saturation for corner residuals (meters)

        // ===== Object anchors (validated modelled objects as SE(2) pose landmarks) =====
        // As the room fills with confidently-localised objects (tables, …) their poses
        // become interior factors that tighten the robot↔room link.  Precision-weighted by
        // each object's own belief covariance — no threshold (see object_anchor_source.h).
        ObjectAnchorFactor::Params object_anchor;      // .enable defaults false (OFF)
        int   object_anchor_max_slots = 3;             // Only apply object factors to the newest N slots
        float object_anchor_early_exit_sigma = 2.0f;   // prediction early-exit gate: whitened anchor-residual
                                                       // σ-cutoff above which the optimizer is forced to run

        // Torch threading configuration
        int torch_num_threads = 5;          // Limit CPU threads to avoid overload
        int torch_num_interop_threads = 2;  // Limit inter-op threads for better latency

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
        float final_loss = 0.f;      // Scaled loss (for optimization)
        float sdf_mse = 0.f;         // Unscaled SDF MSE (for display: sqrt gives avg error in meters)
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
    float evaluate_pose_mean_abs_sdf(const std::vector<Eigen::Vector3f>& lidar_points,
                                     int max_samples = 300) const;
    bool is_initialized() const { return model_ != nullptr; }

    /// Disable relocalization (symmetry check + recovery) once the room is stable.
    void set_relocalization_enabled(bool enabled) { relocalization_enabled_.store(enabled); }
    bool is_relocalization_enabled() const        { return relocalization_enabled_.load(); }

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
            // Note: directional precision (Σ_c^{-1}) is not used in compute_rfe_loss();
            // the loss applies a scalar corner_obs_sigma uniformly in all directions.
        };
        std::vector<CornerObs> corner_obs;

        // Object-anchor observations for this slot (validated modelled objects as SE(2) landmarks).
        std::vector<ObjectAnchorObs> object_anchors;
    };

    struct BoundaryPrior
    {
        Eigen::Vector3f mu = Eigen::Vector3f::Zero();        // MAP pose of dropped state
        Eigen::Matrix3f precision = Eigen::Matrix3f::Identity(); // Σ^{-1} from diag Hessian
        bool valid = false;
    };

    UpdateResult update(const std::pair<std::vector<Eigen::Vector3f>, std::int64_t> &lidar,
                        const std::vector<rc::VelocityCommand> &velocity_history,
                        const std::vector<rc::OdometryReading> &odometry_history);

    Params params;

    /// Hand the latest validated object anchors (gathered on the MAIN thread from the DSR graph)
    /// to the localizer.  Copied into the newest window slot at the next update().  Thread-safe.
    void set_object_anchors(std::vector<ObjectAnchorObs> anchors)
    {
        std::scoped_lock lk(object_anchors_mutex_);
        latest_object_anchors_ = std::move(anchors);
    }

    // Process noise covariance (diagonal [x, y, theta])
    Eigen::Vector3f process_noise = {0.01f, 0.01f, 0.01f};

    // Current covariance estimate [3x3]
    Eigen::Matrix3f current_covariance = Eigen::Matrix3f::Identity() * 0.1f;

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
   std::mutex object_anchors_mutex_;
   std::vector<ObjectAnchorObs> latest_object_anchors_;
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

       /// Returns true if recovery should be triggered now.
       bool check(float avg_sdf_err, int iterations_used,
                  float threshold, int consecutive_count)
       {
           if (cooldown > 0) { --cooldown; return false; }
           if (iterations_used <= 0) return false;
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
       /// sdf_mse_final is below this value (Solution C).
       bool append(WindowSlot slot, int max_window_size, float mu_quality_threshold = std::numeric_limits<float>::max());

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

       /// Recompute boundary prior Hessian from oldest surviving slot.
       void recompute_boundary_prior(const Model& model, const Params& params,
                                      torch::Device device);
   };
   WindowManager window_mgr_;

   // Prediction-based early exit tracking
   int tracking_step_count_ = 0;
   int prediction_early_exits_ = 0;

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
   RerunLogger        rerun_logger_;
   int                rerun_frame_counter_ = 0;
   int                symmetry_check_counter_ = 0;
   float              symmetry_flip_evidence_ = 0.f;  // leaky accumulator of contrary-orientation evidence
   int                good_fit_streak_        = 0;    // consecutive good-SDF frames (track establishment)
   std::atomic<bool>  relocalization_enabled_{true};
   bool               rerun_room_polygon_sent_ = false;
   std::vector<float> last_adam_losses_;    // per-iteration losses from last Adam/LBFGS run
   float              last_loss_init_  = 0.f;  // loss before first step
   float              prev_sdf_mse_    = 0.f;  // sdf_mse from previous frame (for boundary quality gate)
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
    Eigen::Vector3f integrate_velocity_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<VelocityCommand> &velocity_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms);

    /// Integrate measured odometry (adv, side, rot) over the same time window
    Eigen::Vector3f integrate_odometry_over_window(const Eigen::Affine2f &robot_pose,
                                                   const std::vector<OdometryReading> &odometry_history,
                                                   const int64_t &t_start_ms, const int64_t &t_end_ms);

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

    // Returns median absolute SDF error for UI display (robust to outliers)
    static float compute_sdf_mse_unscaled(const torch::Tensor &points_xyz, const Model &m)
    {
        const auto sdf_vals = m.sdf(points_xyz);
        const auto abs_sdf = torch::abs(sdf_vals);
        return torch::median(abs_sdf).item<float>();
    }
};

} // namespace rc

