#pragma once
// Split out of room_concept.h (refactor step 3, docs/refactor_proposal_2026-10-13.md). The types live in
// rc::loc and RoomConcept re-exports each under its old name (RoomConcept::<same name>), so no caller changed.
// Contents: UpdateResult, OdometryPrior, MotionPriorSource, MotionPriorSelection, MotionIngressDebug.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "common_types.h"
#include "corner_detector.h"
#include "wall_map.h"
#include "wall_segmenter.h"
#include "se2_preintegration.h"
#include "motion_calibration.h"
#include "mount_factors.h"
#include "surprise.h"

namespace rc::loc
{
    struct UpdateResult
    {
        bool ok = false;
        /// Relocalisation epoch this result belongs to. It advances each time relocalisation COMMITS a
        /// discrete pose jump (a search that moved the pose, a rival-mode switch). The pose publisher lets
        /// the first result of a new epoch through its kinematic clamp, so the jump lands in one frame
        /// instead of being slewed at the robot's speed limit, and clamps again from the next frame on.
        std::uint64_t reloc_epoch = 0;
        // Set when the SDF optimization produced a non-finite pose and this result is a dead-reckoned
        // FALLBACK, not a fix. Consumers must treat it as such (its covariance is inflated to match).
        bool diverged = false;
        /// The LOCALISER's own estimate, before the publisher's kinematic clamp rewrote robot_pose/covariance in this
        /// copy (pose_publisher.cpp). The clamp's carry is publication LAG folded into σ — measured live (rtD1g2) at
        /// 1.9–4.3 m while the estimate's own σ was ~3 cm — so a consumer asking "where is the robot" (the crossing
        /// test) reads these, not the published pair. Valid only when the publisher filled them.
        bool            estimate_valid = false;
        Eigen::Affine2f estimate_pose = Eigen::Affine2f::Identity();
        Eigen::Matrix3f estimate_covariance = Eigen::Matrix3f::Identity();
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
        /// The same criterion WITHOUT the open-door weighting, and the fraction of the scan's weight
        /// those doors removed. The gate decides on early_exit_metric; these two exist so that
        /// decision can be second-guessed from a log — a gate audited only by the statistic it
        /// already truncated cannot be audited at all.
        float early_exit_metric_unweighted = std::numeric_limits<float>::quiet_NaN();
        float door_discount = 0.f;
        /// True when the SDF polish moved the pose on this cycle. The calibrator reads it as "a
        /// correction happened", which on early-exit cycles it now is.
        bool  sdf_polished = false;
        /// Median |SDF| at the predicted pose — the median-valued twin of early_exit_metric.
        float pred_sdf_median = std::numeric_limits<float>::quiet_NaN();

        // ── Heading-channel attribution for THIS cycle (radians) ───────────────────────────────
        // The prediction's heading is a MIXTURE: the gyro on segments it fully brackets, the wheels
        // on the rest. A scale error in the prediction can therefore belong to either channel, and
        // the mixture weight moves with IMU coverage, so a single pooled dtheta cannot attribute it.
        // imu_dtheta + wheel_dtheta is what actually entered the motion prior; wheel_shadow_dtheta is
        // what the wheels claimed on the segments the gyro overrode, i.e. the direct wheel-vs-gyro
        // comparison on identical intervals.
        // The ODOMETRY-PREDICTED pose, before the optimizer touched it. (corrected - predicted) is the
        // accumulated prediction error the optimizer just removed -- i.e. one tooth of the sawtooth,
        // decomposed into x/y/theta in the ROOM frame. It needs no ground truth and no frame
        // alignment, so it measures the same thing on the real robot as it does here. On early-exit
        // cycles it equals robot_pose exactly; that identity is a free self-check on this plumbing.
        float pred_x = 0.f, pred_y = 0.f, pred_theta = 0.f;
        bool  slot_appended = false;   ///< strided window: this frame was APPENDED (not a replace of the newest)
        /// KL(posterior || motion prediction) for this cycle, in nats -- what the room evidence had to tell
        /// the motion model (see surprise.h). Zero on an early exit; the open-loop stretch is scored whole
        /// at the next correction. Logged per cycle in the heading CSV, drawn on the pred |SDF| plot.
        rc::surprise::Surprise surprise;

        // Body-frame displacement the WHEELS claimed this cycle: dx_local = side*dt (lateral),
        // dy_local = adv*dt (forward). Heading has a second opinion (the gyro) so its errors are
        // catchable; translation has NONE -- the wheels are integrated unconditionally. On a mecanum
        // base the lateral term is the one roller slip corrupts, and a lateral error is invisible to
        // both an arc-length ratio and a rotation gain, so it has to be read directly.
        float dx_local = 0.f, dy_local = 0.f;
        // Linear channel from the accelerometer, per cycle. imu_dv* is the velocity CHANGE the IMU
        // saw; wheel_dv* is the change the WHEELS claim over the same segments. Their disagreement is
        // translation's first independent witness -- the analogue of imu_dtheta vs
        // wheel_shadow_dtheta, which is what exonerated the wheels for heading. imu_dp* is the
        // within-interval double integration (the 0.5*a*T^2 correction to constant velocity), which
        // is sub-millimetre at 50 ms and must never be chained across intervals.
        float imu_dvx = 0.f, imu_dvy = 0.f;
        float imu_dpx = 0.f, imu_dpy = 0.f;
        float wheel_dvx = 0.f, wheel_dvy = 0.f;
        int   imu_lin_segs = 0;

        // Learned motion-model parameters at this cycle, so convergence can be watched in the CSV.
        float calib_k_v = 1.f, calib_k_w = 1.f, calib_yaw = 0.f;
        int   calib_episodes = 0;
        /// Spans that reached the episode trigger with NO correction in them, and were CARRIED
        /// forward rather than emitted as an observation of zero. Logged beside calib_episodes
        /// because the ratio is the honest measure of how much the localiser is actually teaching:
        /// on 2026-08-30, 86% of the window was such spans, emitted as confident zeros, and they
        /// carried 94% of the information (CALIB_UNMEASURED_EPISODES.md). A rising carried count
        /// with a flat episode count means the teacher has gone quiet -- which is exactly what was
        /// invisible before.
        int   calib_carried = 0;
        /// Carried spans abandoned at the linearisation cap: the pose went uncorrected too far.
        int   calib_dropped = 0;
        // 1-sigma uncertainty of each learned parameter. These are what say whether a parameter has
        // CONVERGED or merely stopped moving: a value that sits still with a wide sigma has not been
        // taught anything, it just has not been asked. Units differ (yaw in rad, the scales
        // dimensionless), so compare each against its own history, not against the others.
        /// The joint solve's full parameter vector and its 1-sigma, carried whole rather than
        /// unpacked field by field -- see the note at the viewer's call site. Six parameters was
        /// where hand-copying stopped being tenable.
        Eigen::Matrix<float, rc::calib::P_COUNT, 1> calib_value =
            Eigen::Matrix<float, rc::calib::P_COUNT, 1>::Zero();
        Eigen::Matrix<float, rc::calib::P_COUNT, 1> calib_sigma =
            Eigen::Matrix<float, rc::calib::P_COUNT, 1>::Zero();
        float calib_b_omega = 0.f;      ///< rad/s, the gyro bias the joint solve can now separate
        /// The motion block's normal equations as of its last re-solve (plan 2026-10-05 Task 4), for
        /// the joint calibration monitor on the MAIN thread. A copy, because motion_calib_ is fed on
        /// the localizer thread and its window must not be iterated from another one.
        rc::calib::BatchEstimator::Information calib_information{};
        /// r2 KINEMATIC mount factor (mount_factors.h), a COPY made on the localizer thread every cycle (r1
        /// thread rule): helios (dx, dy, psi) with the odometry nuisance already Schur-marginalised.
        rc::mountf::Info3 mount_kin_info{};
        /// The motion calibrator's ACTING eps on the odometry (yaw_offset()): must read 0 while the mount
        /// factors own the helios yaw (r2.2 items 4/9). Logged in the mounts CSV.
        float mount_yaw_offset = 0.f;
        /// Bitmask over rc::calib::Param: which parameters this window actually TAUGHT (posterior
        /// shrank against the prior). A parameter the driving never excited reads 0 here and sits at
        /// its previous value -- which a bare value cannot be distinguished from convergence.
        int   calib_informed = 0;
        int   calib_applied = 0;        ///< bitmask: parameter i is correcting the odometry (informed AND allowed)
        float calib_condition = 0.f;    ///< of the CORRELATION-normalised information matrix
        float calib_sigma_yaw = 0.f, calib_sigma_k_v = 0.f, calib_sigma_k_w = 0.f;
        float calib_sigma_b_omega = 0.f;   ///< never leave this at 0: a displayed 0 reads as certainty
        float calib_pos_var = 0.f;   // the R actually used for the position channel, for auditing

        float imu_dtheta          = 0.f;
        float wheel_dtheta        = 0.f;
        rc::calib::HeadingCovariates heading_cov{};   ///< per-channel heading Jacobian, this cycle
        float gyro_weight         = -1.f;   ///< time-weighted mean w_g this cycle, <0 = no segment
        /// Per-cycle heading diagnostics for tmp/heading/heading_*.csv. RAW channel rotations (before
        /// any calibration) so each channel's scale can be regressed against truth offline; the
        /// densities are the ones the weights were actually computed from (time-weighted means,
        /// <0 = not computed this cycle, e.g. the legacy switch).
        struct HeadingDiag
        {
            float raw_wheel = 0.f;    ///< rad, sum rot*dt over ALL segments
            float raw_gyro  = 0.f;    ///< rad, integrated gyro over the segments it bracketed
            float gyro_dt   = 0.f;    ///< s bracketed by the gyro
            float total_dt  = 0.f;    ///< s integrated
            float zupt_dt   = 0.f;    ///< s on which the wheels read stationary
            float dens_w    = -1.f;   ///< rad/sqrt(s) used for the wheel factor
            float dens_g    = -1.f;   ///< rad/sqrt(s) used for the gyro factor
            float nl_c0 = 0.f, nl_c1 = 0.f, nl_c2 = 0.f;   ///< learned wheel density^2 coefficients
            float rest_gain_tr = -1.f, rest_gain_ro = -1.f;  ///< P(moving) applied to the prediction; <0 = off
            float rest_dens_v = 0.f, rest_dens_w = 0.f;      ///< the mixture's learnt rest densities
            long  nl_samples = 0;                          ///< segments the learner has seen
        } heading_diag;
        float wheel_shadow_dtheta = 0.f;
        int   imu_segs = 0, wheel_segs = 0;
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
        // The shadow plain-SLAM pose (internal frame) once the map layout is the GIVEN room — the model-independent
        // pose the refiner's evidence is accumulated at (layout_refiner.h). Set by record_slam_shadow on both paths.
        Eigen::Vector3f slam_pose = Eigen::Vector3f::Zero();
        bool slam_pose_valid = false;

        // Wall-SLAM snapshot for the viewer (Estimate mode only; empty otherwise).
        struct WallView
        {
            std::vector<wallseg::WallSegment>  segments;    // this frame, robot frame
            std::vector<wallmap::WallLandmark> walls;       // map frame
            wallmap::Polygon polygon;
            std::vector<int> seg_to_wall;
            bool  theta0_born = false;
            float theta0 = 0.f;
            int   candidates = 0;
            int   births = 0;
            bool  map_ready = false;
        };
        WallView wall_view;
    };

    struct OdometryPrior
    {
        bool valid = false;
        bool fresh = false;
        bool is_measured = false;
        Eigen::Vector3f delta_pose;      // [dx, dy, dtheta] in meters & radians
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
} // namespace rc::loc
