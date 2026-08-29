/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_config.h.
 */

#include "room_config.h"
#include <limits>
#include <cmath>

#include <ConfigLoader/ConfigLoader.h>
#include <QDebug>

#include "room_concept.h"
#include "epistemic_controller.h"

namespace rc
{

void load_room_config(const ConfigLoader& cl, RoomConfig& p,
                      rc::RoomConcept& room_concept, rc::EpistemicController& epistemic)
{
    // ── RoomConcept params ─────────────────────────────────────────────────
    rc::ConfigLoaderUtils::load_required<bool>(cl, "RoomConcept.PredictionEarlyExit", p.PREDICTION_EARLY_EXIT);
    rc::ConfigLoaderUtils::load_required<int>(cl, "RoomConcept.NumIterations", room_concept.params.num_iterations);
    rc::ConfigLoaderUtils::load_required<int>(cl, "RoomConcept.WindowSize", room_concept.params.rfe_window_size);
    rc::ConfigLoaderUtils::load_required<int>(cl, "RoomConcept.MaxLidarPoints", room_concept.params.max_lidar_points);
    rc::ConfigLoaderUtils::load_required<int>(cl, "RoomConcept.MaxLidarOldSlot", room_concept.params.rfe_max_lidar_per_old_slot);
    rc::ConfigLoaderUtils::load_required<float, double>(cl, "RoomConcept.RecoveryLossThreshold", room_concept.params.recovery_loss_threshold);
    rc::ConfigLoaderUtils::load_required<int>(cl, "RoomConcept.RecoveryConsecutiveCount", room_concept.params.recovery_consecutive_count);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.GridSearchGoodFactor", room_concept.params.grid_search_good_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.GridSearchWallMargin", room_concept.params.grid_search_wall_margin);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.GridSearchMaxSamples", room_concept.params.grid_search_max_samples);

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdometryNoiseFactor", p.ODOMETRY_NOISE_FACTOR);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.OdomSampleLog", p.ODOM_SAMPLE_LOG);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "RoomConcept.CalibStateFile", p.CALIB_STATE_FILE);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseScale", room_concept.params.odom_noise_scale);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.DifferentialTest", room_concept.params.differential_test_enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.SdfCurrentSlotOnly", room_concept.params.sdf_current_slot_only);
    // Raise the published covariance to at least what the innovations demonstrate, so the sigma stops
    // being a constant. Default OFF — see Params::adaptive_cov_enabled for the acceptance test and for
    // the two measured process-noise approaches that failed before it.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.HessianCheck", room_concept.params.hessian_check);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.AdaptiveCovEnabled", room_concept.params.adaptive_cov_enabled);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.AdaptiveCovLambda", room_concept.params.adaptive_cov_lambda);
    // Strided RFE window — see Params::window_stride_enabled.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.WindowStrideEnabled", room_concept.params.window_stride_enabled);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.WindowMinTravel", room_concept.params.window_min_travel_m);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.WindowMinTurn", room_concept.params.window_min_turn_rad);
    // Kinematic clamp on the published pose — see RoomConfig::POSE_CLAMP_ENABLED.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.PoseClampEnabled", p.POSE_CLAMP_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PoseClampVMax", p.POSE_CLAMP_V_MAX);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PoseClampWMax", p.POSE_CLAMP_W_MAX);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PoseClampMaxDt", p.POSE_CLAMP_MAX_DT_S);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.PreserveBootstrapRoom", p.PRESERVE_BOOTSTRAP_ROOM);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.RobotVelCovAdv", p.ROBOT_VEL_COV_ADV);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.RobotVelCovSide", p.ROBOT_VEL_COV_SIDE);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.RobotVelCovRot", p.ROBOT_VEL_COV_ROT);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "PredictPublish.enabled", p.PREDICT_PUBLISH_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.max_coast_s", p.PREDICT_PUBLISH_MAX_COAST_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.process_noise_xy", p.PREDICT_PROCESS_NOISE_XY);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.process_noise_theta", p.PREDICT_PROCESS_NOISE_THETA);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.blend_gain", p.PREDICT_BLEND_GAIN);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.max_dt_s", p.PREDICT_MAX_DT_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.max_blend_step_m", p.PREDICT_MAX_BLEND_STEP_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.max_blend_step_rad", p.PREDICT_MAX_BLEND_STEP_RAD);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.snap_thresh_m", p.PREDICT_SNAP_THRESH_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "PredictPublish.snap_thresh_rad", p.PREDICT_SNAP_THRESH_RAD);
    rc::ConfigLoaderUtils::load_optional_apply<std::string>(cl, "RoomConcept.OptimizerType", [&](const std::string& optimizer_type)
    {
        p.OptimizerType = optimizer_type;
        room_concept.params.optimizer_type = optimizer_type;
    });
    rc::ConfigLoaderUtils::load_optional_apply<std::string>(cl, "RoomConcept.RoomLayoutSvg", [&](const std::string& svg_file)
    {
        p.ROOM_LAYOUT_SVG = svg_file;
    });
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "RoomConcept.LayoutDir", p.LAYOUT_DIR);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.RecenterRoomPolygon", p.RECENTER_ROOM_POLYGON);
    // Ceiling height (m). Sets the room DSR node's room_height attribute (walls/ceiling overlay) and
    // the EXPECTED ceiling location for the LiDAR startup geometry check. Set it explicitly for rooms
    // whose ceiling is above the LiDAR's vertical reach (e.g. 3 m), where the check can't detect it.
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.RoomHeight", p.room_height);

    // Media plane (RGB for the camera window + LiDAR for LidarIngestor). DDS domain +
    // topics are read from the producer's media descriptor on the graph, not config.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "Media.lidar_use_media", p.LIDAR_USE_MEDIA);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "Media.lidar_stall_timeout_ms", p.LIDAR_STALL_TIMEOUT_MS);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "Media.lidar_wait_log_period_ms", p.LIDAR_WAIT_LOG_PERIOD_MS);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "Media.lidar_helios_name", p.LIDAR_HELIOS_NAME);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "Media.lidar_robot_frame", p.LIDAR_ROBOT_FRAME);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "Media.lidar_high_min_height", p.LIDAR_HIGH_MIN_HEIGHT);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "Media.lidar_high_max_height", p.LIDAR_HIGH_MAX_HEIGHT);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "Media.lidar_startup_geometry_check", p.LIDAR_STARTUP_GEOMETRY_CHECK);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "Media.lidar_startup_check_sweeps", p.LIDAR_STARTUP_CHECK_SWEEPS);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "Media.lidar_floor_tolerance", p.LIDAR_FLOOR_TOLERANCE);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "Media.lidar_ceiling_margin", p.LIDAR_CEILING_MARGIN);

    // Camera-overlay object projection: comma-separated DSR node types (e.g. "object,table,cylinder,chair").
    rc::ConfigLoaderUtils::load_optional_apply<std::string>(cl, "Overlay.ObjectTypes", [&](const std::string& csv)
    {
        std::vector<std::string> types;
        std::size_t start = 0;
        while (start <= csv.size())
        {
            const std::size_t comma = csv.find(',', start);
            const std::size_t end = (comma == std::string::npos) ? csv.size() : comma;
            std::string tok = csv.substr(start, end - start);
            const auto l = tok.find_first_not_of(" \t");
            const auto r = tok.find_last_not_of(" \t");
            if (l != std::string::npos)
                types.push_back(tok.substr(l, r - l + 1));
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
        if (not types.empty())
            p.OVERLAY_OBJECT_TYPES = std::move(types);
    });

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.MotionCalibEnabled", room_concept.params.motion_calib.enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.ImuLinearInjection", room_concept.params.imu_linear_injection);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.OdomVarianceInjection", room_concept.params.odom_variance_injection);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibYawP0", room_concept.params.motion_calib.yaw_p0);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibYawQ", room_concept.params.motion_calib.yaw_q);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibScaleP0", room_concept.params.motion_calib.scale_p0);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibScaleQ", room_concept.params.motion_calib.scale_q);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibRotModelSigma", room_concept.params.motion_calib.rot_model_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionCalibFitModelGain", room_concept.params.motion_calib.fit_model_gain);

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SigmaSdf", room_concept.params.sigma_sdf);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PredictionTrustFactor", room_concept.params.prediction_trust_factor);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.MinTrackingSteps", room_concept.params.min_tracking_steps);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.RotationSdfCoupling", room_concept.params.rotation_sdf_coupling);

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.LbfgsLr", room_concept.params.lbfgs_lr);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.LbfgsHistorySize", room_concept.params.lbfgs_history_size);
    rc::ConfigLoaderUtils::load_optional<double>(cl, "RoomConcept.LbfgsToleranceGrad", room_concept.params.lbfgs_tolerance_grad);
    rc::ConfigLoaderUtils::load_optional<double>(cl, "RoomConcept.LbfgsToleranceChange", room_concept.params.lbfgs_tolerance_change);

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.LearningRatePos", room_concept.params.learning_rate_pos);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.ObsSigma", room_concept.params.rfe_obs_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HuberDelta", room_concept.params.rfe_huber_delta);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.ConvergenceRelTol", room_concept.params.convergence_relative_tol);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.ConvergenceMinIters", room_concept.params.convergence_min_iters);

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.BoundaryQualityGate", room_concept.params.rfe_boundary_quality_gate);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.BoundaryHessianQualityThreshold", room_concept.params.boundary_hessian_quality_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.BoundaryMuQualityThreshold", room_concept.params.boundary_mu_quality_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.EigenvalueClampBoundaryMax", room_concept.params.eigenvalue_clamp_boundary_max);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.TorchNumThreads", room_concept.params.torch_num_threads);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.BoundaryFejSchur", room_concept.params.boundary_fej_schur);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.BoundaryQualitySigma", room_concept.params.boundary_quality_sigma);

    // Hierarchical precision on the boundary prior (HIERARCHICAL_PRECISION.md) — default OFF.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.HierPrecBoundaryEnabled", room_concept.params.hier_prec_boundary_enabled);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecU0", room_concept.params.hier_prec_u0);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecSigmaU2", room_concept.params.hier_prec_sigma_u2);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecLrU", room_concept.params.hier_prec_lr_u);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecGGain", room_concept.params.hier_prec_g_gain);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecLrV", room_concept.params.hier_prec_lr_v);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecSigmaV2", room_concept.params.hier_prec_sigma_v2);

    // FE-native relocalization on map-trust collapse (HIERARCHICAL_PRECISION.md) — default OFF.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.HierPrecRelocEnabled", room_concept.params.hier_prec_reloc_enabled);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecRelocFloor", room_concept.params.hier_prec_reloc_floor);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.HierPrecRelocConsecutive", room_concept.params.hier_prec_reloc_consecutive);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.HierPrecRelocCooldownFrames", room_concept.params.hier_prec_reloc_cooldown_frames);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.HierPrecEeDthetaMin", room_concept.params.hier_prec_ee_dtheta_min);

    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.RecoveryCooldownFrames", room_concept.params.recovery_cooldown_frames);

    // 180° symmetry-flip robustness (leaky evidence + confidence-scaled threshold).
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.SymmetryCheckInterval", room_concept.params.symmetry_check_interval);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SymmetryFlipMinImprovement", room_concept.params.symmetry_flip_min_improvement);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SymmetryEvidenceLeak", room_concept.params.symmetry_evidence_leak);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SymmetryFlipEvidenceThresh", room_concept.params.symmetry_flip_evidence_thresh);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SymmetryConfidenceGain", room_concept.params.symmetry_confidence_gain);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.SymmetryConfidenceCap", room_concept.params.symmetry_confidence_cap);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.SymmetryGoodFitMse", room_concept.params.symmetry_good_fit_mse);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.SymmetryConfidenceDecay", room_concept.params.symmetry_confidence_decay);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.SymmetryDebugCsv", room_concept.params.symmetry_debug_csv);

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.VelocityAdaptiveWeights", room_concept.params.velocity_adaptive_weights);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.LinearVelocityThreshold", room_concept.params.linear_velocity_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.AngularVelocityThreshold", room_concept.params.angular_velocity_threshold);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.WeightBoostFactor", room_concept.params.weight_boost_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.WeightReductionFactor", room_concept.params.weight_reduction_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.WeightSmoothingAlpha", room_concept.params.weight_smoothing_alpha);

    // Master switch for the command (joystick / controller) motion prior — see
    // Params::use_command_velocity_prior. Off ⇒ the channel is still computed and logged but never
    // enters the prediction, so motion_prior_source collapses to measured / fallback_zero.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.UseCommandVelocityPrior", room_concept.params.use_command_velocity_prior);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseTrans", room_concept.params.cmd_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseRot", room_concept.params.cmd_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseBase", room_concept.params.cmd_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseTrans", room_concept.params.odom_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseRot", room_concept.params.odom_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseBase", room_concept.params.odom_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.EncoderRotSlipK", room_concept.params.encoder_rot_slip_k);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.StationaryMotionThreshold", room_concept.params.stationary_motion_threshold);

    // ── Preintegrated motion covariance — see Params::motion_preintegration, se2_preintegration.h ──
    // The densities are DERIVED from the legacy constants just loaded, not hard-coded, so the two
    // models cannot silently drift apart: changing EncoderRotSlipK or OdomNoiseBase moves both. That
    // matters because the whole point of the A/B is that the flag changes the SHAPE of the covariance,
    // not its magnitude at the operating point — and a stale hard-coded default would quietly turn it
    // into a re-tuning as well, which is the confound that makes an A/B unreadable.
    //
    //   sigma = (legacy per-frame std) / sqrt(dt_ref)   — a per-frame constant becomes a density
    //   scale = the legacy FRACTIONAL terms, which were always describing a correlated scale error
    //
    // dt_ref is the update interval the legacy constants were tuned at (measured median 42-50 ms).
    // Any explicit Preint* key below overrides the derived value, so a calibrated measurement of the
    // real odometry stream can replace this translation without touching code.
    {
        constexpr float dt_ref = 0.05f;                      // s — the interval the legacy tuning assumed
        const float inv_sqrt_dt = 1.f / std::sqrt(dt_ref);
        auto& po = room_concept.params.odom_preint_noise;
        auto& pc = room_concept.params.cmd_preint_noise;
        const auto& p0 = room_concept.params;

        // Translation floor: the legacy value actually in force when the robot is barely moving is
        // StationaryMotionThreshold (0.02 m live — raised from 0.001 to stop loss_motion spiking to
        // 6000 on a 3.4 cm parked residual), which is the branch the robot is in for >99% of frames.
        // Take the LOOSER of it and OdomNoiseBase so the derived floor is never tighter than either.
        const float odom_floor = std::max(p0.stationary_motion_threshold, p0.odom_noise_base);
        po.sigma_v_lat  = odom_floor * inv_sqrt_dt;
        po.sigma_v_long = odom_floor * inv_sqrt_dt;
        po.sigma_omega  = p0.rotation_noise_base * inv_sqrt_dt;
        po.scale_v      = p0.odom_noise_trans;
        // OdomNoiseRot and EncoderRotSlipK are BOTH fractions of the rotation increment and the legacy
        // model combines them in quadrature, so the equivalent single scale sigma is their hypot.
        po.scale_omega  = std::hypot(p0.odom_noise_rot, p0.encoder_rot_slip_k);

        const float cmd_floor = std::max(p0.stationary_motion_threshold, p0.cmd_noise_base);
        pc.sigma_v_lat  = cmd_floor * inv_sqrt_dt;
        pc.sigma_v_long = cmd_floor * inv_sqrt_dt;
        pc.sigma_omega  = p0.rotation_noise_base * inv_sqrt_dt;
        pc.scale_v      = p0.cmd_noise_trans;
        pc.scale_omega  = p0.cmd_noise_rot;
    }
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.MotionPreintegration", room_concept.params.motion_preintegration);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintOdomSigmaVLat",  room_concept.params.odom_preint_noise.sigma_v_lat);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintOdomSigmaVLong", room_concept.params.odom_preint_noise.sigma_v_long);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintOdomSigmaOmega", room_concept.params.odom_preint_noise.sigma_omega);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintOdomScaleV",     room_concept.params.odom_preint_noise.scale_v);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintOdomScaleOmega", room_concept.params.odom_preint_noise.scale_omega);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintCmdSigmaVLat",   room_concept.params.cmd_preint_noise.sigma_v_lat);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintCmdSigmaVLong",  room_concept.params.cmd_preint_noise.sigma_v_long);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintCmdSigmaOmega",  room_concept.params.cmd_preint_noise.sigma_omega);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintCmdScaleV",      room_concept.params.cmd_preint_noise.scale_v);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintCmdScaleOmega",  room_concept.params.cmd_preint_noise.scale_omega);

    // ── ZUPT: the rest hypothesis as a factor (see the NoiseModel block in se2_preintegration.h) ──
    // One set of keys drives BOTH channels: "the body is at rest" is a statement about the robot, not
    // about which stream happened to report it, and letting the two disagree would mean the prior's
    // parked stiffness depended on which channel won selection that frame.
    {
        auto& po = room_concept.params.odom_preint_noise;
        auto& pc = room_concept.params.cmd_preint_noise;
        rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.PreintZupt", po.zupt_enabled);
        rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.PreintZuptAsFactor",
                                                   room_concept.params.zupt_as_factor);
        // The two forms are EXCLUSIVE: running both counts one hypothesis twice, once shaping the
        // prior's covariance and once as its own factor. The flag therefore turns the per-sample
        // shaping off rather than leaving the caller to remember.
        rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.SdfPolishOnEarlyExit",
                                                   room_concept.params.sdf_polish_enabled);
        rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.PreintZuptOnPrediction",
                                                   room_concept.params.zupt_on_prediction);
        rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptPredVMax",
                                                            room_concept.params.zupt_pred_v_max);
        rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptPredWMax",
                                                            room_concept.params.zupt_pred_w_max);
        po.zupt_as_factor = room_concept.params.zupt_as_factor;
        pc.zupt_as_factor = room_concept.params.zupt_as_factor;
        // ★ The keys were RENAMED with their units on 2026-08-26 (per-sample sigma -> density). The
        // old names are read into a sentinel purely so a config still carrying them FAILS LOUDLY
        // instead of having them silently ignored — which would leave that platform on the header
        // default while its config file appeared to say otherwise. A silently orphaned key is worse
        // than a missing one: the file documents an intent that nothing implements.
        {
            constexpr float kSentinel = -12345.f;
            float legacy_v = kSentinel, legacy_w = kSentinel;
            rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptSigmaV",     legacy_v);
            rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptSigmaOmega", legacy_w);
            if (legacy_v != kSentinel or legacy_w != kSentinel)
                throw std::runtime_error(
                    "config uses PreintZuptSigmaV/Omega, which are PER-SAMPLE standard deviations and "
                    "no longer exist. They are now PreintZuptDensityV/Omega, in m/sqrt(s) and "
                    "rad/sqrt(s). Convert with density = sigma * sqrt(publish_period_s) to preserve "
                    "today's behaviour at today's rate, or re-measure with tools/odom_whiteness.py.");
        }
        rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptDensityV",     po.zupt_density_v);
        rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptDensityOmega", po.zupt_density_omega);
        rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.PreintZuptLeverM",     po.zupt_lever_m);
        pc.zupt_enabled     = po.zupt_enabled;
        pc.zupt_density_v   = po.zupt_density_v;
        pc.zupt_density_omega = po.zupt_density_omega;
        pc.zupt_lever_m     = po.zupt_lever_m;
    }


    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.EnableCornerTracking", room_concept.params.enable_corner_tracking);
    // Graded-covariance corner factor (replaces the old hard rej_angle/rej_orient gates).
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerWallBand", room_concept.params.corner_wall_band);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerBaseSigma", room_concept.params.corner_base_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerOrientTauDeg", room_concept.params.corner_orient_tau_deg);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerMergeChi2", room_concept.params.corner_merge_chi2);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerMergePriorSigma", room_concept.params.corner_merge_prior_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerMinWallMapSigmas", room_concept.params.corner_min_wall_map_sigmas);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.CornerStatsCsv", room_concept.params.corner_stats_csv);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerMinYieldMapSigmas", room_concept.params.corner_min_yield_map_sigmas);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerYieldLeak", room_concept.params.corner_yield_leak);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.CornerYieldWarmup", room_concept.params.corner_yield_warmup);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerYieldReleaseFactor", room_concept.params.corner_yield_release_factor);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerPrecisionGain", room_concept.params.corner_precision_gain);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerHuberSigma", room_concept.params.corner_huber_sigma);
    // Gauss-Newton / Levenberg-Marquardt backend (room_gn_solver.h). All OFF by default: GnShadow
    // only LOGS, and driving the pose with it additionally requires OptimizerType = "GN".
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.GnShadow", room_concept.params.gn_shadow);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.GnGradCheck", room_concept.params.gn_grad_check);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "RoomConcept.GnShadowCsv", room_concept.params.gn_shadow_csv_path);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.GnMaxIters", room_concept.params.gn_max_iters);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.GnLambdaInit", room_concept.params.gn_lambda_init);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.GnStepTol", room_concept.params.gn_step_tol);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.GnLossRelTol", room_concept.params.gn_loss_rel_tol);

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.CornerEarlyExitCheck", room_concept.params.corner_early_exit_check);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CornerEarlyExitSigma", room_concept.params.corner_early_exit_sigma);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.CornerEarlyExitMinBad", room_concept.params.corner_early_exit_min_bad);

    // Object anchors (validated modelled objects as SE(2) pose landmarks). Loaded into BOTH the
    // shared config (read by RoomSceneGraph's graph-side gather) and the localizer params.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ObjectAnchor.enable", p.OBJECT_ANCHOR_ENABLE);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ObjectAnchor.optimizeLandmark", p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK);
    // Guarded by exists(): ConfigLoader throws on `key = []`, and a silently-empty list would disable
    // every landmark while the enable flag still read true — a confusing way to get nothing.
    if (cl.exists("ObjectAnchor.subtypes"))
    {
        auto v = cl.get<std::vector<std::string>>("ObjectAnchor.subtypes");
        if (not v.empty()) p.OBJECT_ANCHOR_SUBTYPES = std::move(v);
    }
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.weight", p.OBJECT_ANCHOR_WEIGHT);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.huber", p.OBJECT_ANCHOR_HUBER);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "ObjectAnchor.maxSlots", p.OBJECT_ANCHOR_MAX_SLOTS);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.measSigmaXY", p.OBJECT_ANCHOR_MEAS_SIG_XY);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.measSigmaYaw", p.OBJECT_ANCHOR_MEAS_SIG_YAW);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.earlyExitSigma", p.OBJECT_ANCHOR_EARLY_EXIT_SIGMA);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.validateSigma", p.OBJECT_ANCHOR_VALIDATE_SIGMA);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ObjectAnchor.freshnessEnable", p.OBJECT_ANCHOR_FRESHNESS_ENABLE);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ObjectAnchor.freshnessAgeScale", p.OBJECT_ANCHOR_FRESHNESS_AGE_SCALE);
    room_concept.params.object_anchor.enable      = p.OBJECT_ANCHOR_ENABLE;
    // GN-only: the autograd backends have no landmark variables, so silently honouring this under
    // LBFGS/ADAM would mean the flag reads true while nothing optimises. Refuse loudly instead.
    room_concept.params.object_anchor_optimize_landmark = p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK;
    if (p.OBJECT_ANCHOR_OPTIMIZE_LANDMARK and room_concept.params.optimizer_type != "GN")
    {
        qWarning() << "[room] ObjectAnchor.optimizeLandmark needs OptimizerType = \"GN\" (have"
                   << QString::fromStdString(room_concept.params.optimizer_type) << ") — landmark "
                      "optimisation DISABLED, anchors stay pinned";
        room_concept.params.object_anchor_optimize_landmark = false;
    }
    room_concept.params.object_anchor.weight      = p.OBJECT_ANCHOR_WEIGHT;
    room_concept.params.object_anchor.huber_delta = p.OBJECT_ANCHOR_HUBER;
    room_concept.params.object_anchor_max_slots   = p.OBJECT_ANCHOR_MAX_SLOTS;
    room_concept.params.object_anchor_early_exit_sigma = p.OBJECT_ANCHOR_EARLY_EXIT_SIGMA;

    // ── RGB edge alignment ────────────────────────────────────────────────────────────────────
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.enable", p.IMAGE_EDGE_ENABLE);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.shadow", p.IMAGE_EDGE_SHADOW);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.drive",  p.IMAGE_EDGE_DRIVE);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "ImageEdge.camera", p.IMAGE_EDGE_CAMERA);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.useWallCorners", p.IMAGE_EDGE_USE_WALL_CORNERS);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.useFloorJunction", p.IMAGE_EDGE_USE_FLOOR_JUNCTION);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.useWallCeiling", p.IMAGE_EDGE_USE_WALL_CEILING);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.sampleSpacingM", p.IMAGE_EDGE_SAMPLE_SPACING_M);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.searchSigmas", p.IMAGE_EDGE_SEARCH_SIGMAS);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "ImageEdge.maxSearchPx", p.IMAGE_EDGE_MAX_SEARCH_PX);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "ImageEdge.maxSlots", p.IMAGE_EDGE_MAX_SLOTS);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.mountPitchSigma", p.IMAGE_EDGE_MOUNT_PITCH_SIGMA);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.mountHeightSigma", p.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.mountYawSigma", p.IMAGE_EDGE_MOUNT_YAW_SIGMA);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.mountYawCorrection", p.IMAGE_EDGE_MOUNT_YAW_CORR);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "ImageEdge.wallPositionSigma", p.IMAGE_EDGE_WALL_POS_SIGMA);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "ImageEdge.useTriplePoints", p.IMAGE_EDGE_USE_TRIPLE_POINTS);

    // ── Platform overlays ────────────────────────────────────────────────────────────────────────
    // Parsed here because the ConfigLoader does not outlive this call; APPLIED later, once the
    // robot's own name is known from the graph. See RoomConfig::apply_platform.
    // The names come from Platform.names so the loader never has to enumerate sections — and an
    // absent key means "no overlays", which is the ordinary single-robot case rather than an error.
    // ⚠ ConfigLoader THROWS on an empty array (memory: configloader-empty-array-throws), so omit
    //   Platform.names rather than writing [].
    {
        std::vector<std::string> names;
        try { rc::ConfigLoaderUtils::load_optional<std::vector<std::string>>(cl, "Platform.names", names); }
        catch (const std::exception& e) { qWarning() << "[cfg] Platform.names ignored:" << e.what(); }
        for (const auto& n : names)
        {
            RoomConfig::PlatformOverlay ov;
            // load_optional returns void and leaves the destination ALONE when the key is absent,
            // so presence is detected with a sentinel rather than a return value. NaN is the right
            // sentinel here: no legitimate config value can collide with it, whereas 0 or -1 could.
            const auto f = [&](const char* key, std::optional<float>& dst)
            {
                float v = std::numeric_limits<float>::quiet_NaN();
                try { rc::ConfigLoaderUtils::load_optional<float, double>(
                          cl, ("Platform." + n + "." + key).c_str(), v); }
                catch (...) { return; }
                if (std::isfinite(v)) dst = v;
            };
            f("mountPitchSigma",    ov.mount_pitch_sigma);
            f("mountHeightSigma",   ov.mount_height_sigma);
            f("mountYawSigma",      ov.mount_yaw_sigma);
            f("mountYawCorrection", ov.mount_yaw_correction);
            f("wallPositionSigma",  ov.wall_position_sigma);
            f("CmdNoiseRot",        ov.cmd_noise_rot);
            f("CmdNoiseTrans",      ov.cmd_noise_trans);
            f("PreintZuptDensityV",     ov.zupt_density_v);
            f("PreintZuptDensityOmega", ov.zupt_density_omega);
            f("PreintOdomSigmaOmega",   ov.odom_preint_sigma_omega);
            f("PreintOdomScaleOmega",   ov.odom_preint_scale_omega);
            f("SdfSafe",                ov.sdf_safe);
            f("SdfDanger",              ov.sdf_danger);
            // An int and a bool have no spare sentinel the way a float has NaN, so each is read
            // twice from opposite seeds: only a key that is actually present makes the two agree.
            const auto i = [&](const char* key, std::optional<int>& dst)
            {
                int lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
                try { rc::ConfigLoaderUtils::load_optional<int>(cl, ("Platform." + n + "." + key).c_str(), lo);
                      rc::ConfigLoaderUtils::load_optional<int>(cl, ("Platform." + n + "." + key).c_str(), hi); }
                catch (...) { return; }
                if (lo == hi) dst = lo;
            };
            const auto b = [&](const char* key, std::optional<bool>& dst)
            {
                bool t = true, fa = false;
                try { rc::ConfigLoaderUtils::load_optional<bool>(cl, ("Platform." + n + "." + key).c_str(), t);
                      rc::ConfigLoaderUtils::load_optional<bool>(cl, ("Platform." + n + "." + key).c_str(), fa); }
                catch (...) { return; }
                if (t == fa) dst = t;
            };
            i("Period",                     ov.period_compute);
            b("UseCommandVelocityPrior",    ov.use_command_velocity_prior);
            b("CalibPivotEnabled",          ov.calib_pivot_enabled);
            b("OdomSampleLog",              ov.odom_sample_log);
            // Drift keys — see PlatformOverlay in the header for why they are here.
            f("RecoveryLossThreshold",             ov.recovery_loss_threshold);
            f("PredictionTrustFactor",             ov.prediction_trust_factor);
            f("RotationSdfCoupling",               ov.rotation_sdf_coupling);
            f("BoundaryHessianQualityThreshold",   ov.boundary_hessian_quality_threshold);
            f("BoundaryMuQualityThreshold",        ov.boundary_mu_quality_threshold);
            f("SymmetryGoodFitMse",                ov.symmetry_good_fit_mse);
            f("GnLossRelTol",                      ov.gn_loss_rel_tol);
            i("GnMaxIters",                        ov.gn_max_iters);
            i("TorchNumThreads",                   ov.torch_num_threads);
            b("AdaptiveCovEnabled",                ov.adaptive_cov_enabled);
            b("WindowStrideEnabled",               ov.window_stride_enabled);
            b("BoundaryFejSchur",                  ov.boundary_fej_schur);
            b("HierPrecBoundaryEnabled",           ov.hier_prec_boundary_enabled);
            b("CornerEarlyExitCheck",              ov.corner_early_exit_check);
            f("WIor",                              ov.w_ior);
            f("BeliefForgetTime",                  ov.belief_forget_time);
            f("ObjectAnchorMeasSigmaXY",           ov.object_anchor_meas_sigma_xy);
            f("StableSdfMseMax",                   ov.stable_sdf_mse_max);
            std::string cam;
            try { rc::ConfigLoaderUtils::load_optional<std::string>(
                      cl, ("Platform." + n + ".camera").c_str(), cam); }
            catch (...) {}
            if (not cam.empty()) ov.image_edge_camera = cam;
            p.platform_overlays[n] = ov;
        }
        if (not names.empty())
            qInfo() << "[cfg] platform overlays parsed for" << static_cast<int>(names.size())
                    << "robot(s); the matching one is applied once the graph names this robot";
    }
    {
        std::vector<std::string> names;
        try { rc::ConfigLoaderUtils::load_optional<std::vector<std::string>>(cl, "Scenario.names", names); }
        catch (const std::exception& e) { qWarning() << "[cfg] Scenario.names ignored:" << e.what(); }
        for (const auto& n : names)
        {
            RoomConfig::ScenarioOverlay ov;
            std::string svg;
            try { rc::ConfigLoaderUtils::load_optional<std::string>(
                      cl, ("Scenario." + n + ".RoomLayoutSvg").c_str(), svg); } catch (...) {}
            if (not svg.empty()) ov.room_layout_svg = svg;
            float h = std::numeric_limits<float>::quiet_NaN();
            try { rc::ConfigLoaderUtils::load_optional<float, double>(
                      cl, ("Scenario." + n + ".RoomHeight").c_str(), h); } catch (...) {}
            if (std::isfinite(h)) ov.room_height = h;
            // A bool has no spare sentinel, so it is read twice from opposite starting points: if
            // the key is absent both reads keep their seed and disagree, and only a key that is
            // actually present makes them agree. Cheaper than adding an API for one flag.
            bool b_t = true, b_f = false;
            try { rc::ConfigLoaderUtils::load_optional<bool>(
                      cl, ("Scenario." + n + ".RecenterRoomPolygon").c_str(), b_t);
                  rc::ConfigLoaderUtils::load_optional<bool>(
                      cl, ("Scenario." + n + ".RecenterRoomPolygon").c_str(), b_f); } catch (...) {}
            if (b_t == b_f) ov.recenter_room_polygon = b_t;
            const auto sf = [&](const char* key, std::optional<float>& dst)
            {
                float v = std::numeric_limits<float>::quiet_NaN();
                try { rc::ConfigLoaderUtils::load_optional<float, double>(
                          cl, ("Scenario." + n + "." + key).c_str(), v); }
                catch (...) { return; }
                if (std::isfinite(v)) dst = v;
            };
            sf("LidarHighMaxHeight", ov.lidar_high_max_height);
            sf("TargetWallMargin",   ov.target_wall_margin);
            {   // ⚠ ConfigLoader throws on an EMPTY array, so absent is how to say "leave it".
                std::vector<std::string> subs;
                try { rc::ConfigLoaderUtils::load_optional<std::vector<std::string>>(
                          cl, ("Scenario." + n + ".ObjectAnchorSubtypes").c_str(), subs); }
                catch (...) {}
                if (not subs.empty()) ov.object_anchor_subtypes = std::move(subs);
            }
            p.scenario_overlays[n] = ov;
        }
        if (not names.empty())
            qInfo() << "[cfg] scenario overlays parsed for" << static_cast<int>(names.size())
                    << "scenario(s); the matching one is applied once the graph names this place";
    }
    // ConfigLoader throws on an EMPTY array (memory: configloader-empty-array-throws), so an absent
    // key is the way to say "no extra calibration cameras", not `calibCameras = []`.
    try { rc::ConfigLoaderUtils::load_optional<std::vector<std::string>>(cl, "ImageEdge.calibCameras", p.CALIB_CAMERAS); }
    catch (const std::exception& e) { qWarning() << "[cfg] ImageEdge.calibCameras ignored:" << e.what(); }
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "ImageEdge.csv", p.IMAGE_EDGE_CSV);

    room_concept.params.image_edge.enable             = p.IMAGE_EDGE_ENABLE;
    room_concept.params.image_edge.drive              = p.IMAGE_EDGE_DRIVE;
    room_concept.params.image_edge.search_sigmas      = p.IMAGE_EDGE_SEARCH_SIGMAS;
    room_concept.params.image_edge.mount_pitch_sigma  = p.IMAGE_EDGE_MOUNT_PITCH_SIGMA;
    room_concept.params.image_edge.mount_height_sigma = p.IMAGE_EDGE_MOUNT_HEIGHT_SIGMA;
    room_concept.params.image_edge.mount_yaw_sigma    = p.IMAGE_EDGE_MOUNT_YAW_SIGMA;
    room_concept.params.image_edge.wall_position_sigma = p.IMAGE_EDGE_WALL_POS_SIGMA;
    room_concept.params.image_edge.use_triple_points  = p.IMAGE_EDGE_USE_TRIPLE_POINTS;
    room_concept.params.image_edge_max_slots          = p.IMAGE_EDGE_MAX_SLOTS;
    room_concept.params.image_edge_shadow             = p.IMAGE_EDGE_SHADOW;
    room_concept.params.image_edge_csv                = p.IMAGE_EDGE_CSV;
    room_concept.params.calib_state_file              = p.CALIB_STATE_FILE;
    // GN-only, and refused LOUDLY rather than honoured silently: the autograd backends evaluate the
    // term (the torch mirror exists) but only the GN factor list can be driven by it, so under
    // LBFGS/ADAM the flag would read true while nothing moved. Same rule, same reason, as
    // ObjectAnchor.optimizeLandmark above.
    if (p.IMAGE_EDGE_DRIVE and room_concept.params.optimizer_type != "GN")
    {
        qWarning() << "[room] ImageEdge.drive needs OptimizerType = \"GN\" (have"
                   << QString::fromStdString(room_concept.params.optimizer_type)
                   << ") — RGB edge term DEMOTED to shadow (evaluated + logged, does not move the pose)";
        room_concept.params.image_edge.drive  = false;
        room_concept.params.image_edge_shadow = true;
    }
    if (p.IMAGE_EDGE_DRIVE and not p.IMAGE_EDGE_ENABLE)
        qWarning() << "[room] ImageEdge.drive = true but ImageEdge.enable = false — nothing is built, "
                      "the term is INERT. Set enable = true.";
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.FarPointsWeight", room_concept.params.far_points_weight);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.FarPointsExponent", room_concept.params.far_points_exponent);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.FarPointsMinWeight", room_concept.params.far_points_min_weight);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.IncidenceAngleWeight", room_concept.params.incidence_angle_weight);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.IncidenceAngleExponent", room_concept.params.incidence_angle_exponent);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.IncidenceAngleMinWeight", room_concept.params.incidence_angle_min_weight);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.UseCuda", room_concept.params.use_cuda);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.DebugLog", room_concept.params.debug_log_enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.OptimizerTimingCsv", room_concept.params.optimizer_timing_csv);

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.RerunEnabled", room_concept.params.rerun_enabled);
    rc::ConfigLoaderUtils::load_optional<std::string>(cl, "RoomConcept.RerunHost", room_concept.params.rerun_host);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.RerunPort", room_concept.params.rerun_port);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.RerunSdfEveryN", room_concept.params.rerun_sdf_every_n);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.RerunSdfResolution", room_concept.params.rerun_sdf_resolution);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.RerunMaxQueue", room_concept.params.rerun_max_queue);

    // ── DSR stabilization thresholds ──────────────────────────────────────
    rc::ConfigLoaderUtils::load_optional<int>(cl, "DSR.StableFramesRequired", p.STABLE_FRAMES_REQUIRED);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.StableSdfMseMax", p.STABLE_SDF_MSE_MAX);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.StableCovTtMax", p.STABLE_COV_TT_MAX);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "DSR.BootstrapTableEnabled", p.BOOTSTRAP_TABLE_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableX", p.BOOTSTRAP_TABLE_X);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableY", p.BOOTSTRAP_TABLE_Y);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableYaw", p.BOOTSTRAP_TABLE_YAW);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableWidth", p.BOOTSTRAP_TABLE_WIDTH);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableDepth", p.BOOTSTRAP_TABLE_DEPTH);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "DSR.BootstrapTableHeight", p.BOOTSTRAP_TABLE_HEIGHT);

    // ── EpistemicController params ─────────────────────────────────────────
    auto& ec = epistemic.params;
    auto& ep = epistemic.epistemic_planner().params;
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "EpistemicController.PublishAffordance", p.PUBLISH_AFFORDANCE);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.CalibPivotEnabled", p.CALIB_PIVOT_ENABLED);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CalibForcedGainNats", p.CALIB_FORCED_GAIN_NATS);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ExecStallTimeout", p.EXEC_STALL_TIMEOUT_S);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ExecStallProgress", p.EXEC_STALL_PROGRESS_M);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "EpistemicController.NumArcCurvatures", ec.num_arc_curvatures);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "EpistemicController.HorizonSteps", ec.horizon_steps);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.Dt", ec.dt);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.MaxAdvSpeed", ec.max_adv_speed);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.MaxRotSpeed", ec.max_rot_speed);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WEpistemic", ec.w_epistemic);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WPragmatic", ec.w_pragmatic);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WHeading", ec.w_heading);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WBoundary", ec.w_boundary);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.KRot", ec.k_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.GaussianSigma", ec.gaussian_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.SpeedHorizonS", ec.speed_horizon_s);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ObstacleRadius", ec.obstacle_radius);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ObstacleK", ec.obstacle_k);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ObstacleStepCap", ec.obstacle_step_cap);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WObstacle", ec.w_obstacle);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WallFilterMargin", ec.wall_filter_margin);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.BandwidthCoupling", ec.bandwidth_coupling);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.SdfSafe", ec.sdf_safe);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.SdfDanger", ec.sdf_danger);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.GovernorAlphaMin", ec.governor_alpha_min);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimCornerSigma", ec.fim_corner_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimMaxRange", ec.fim_max_range);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.GridResolution", ep.grid_resolution);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.MinDistance", ep.min_distance);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "EpistemicController.MaxCandidates", ep.max_candidates);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.TargetWallMargin", ep.target_wall_margin);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.TargetObstacleClearance", ep.target_obstacle_clearance);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.AngularDominanceRatio", ep.angular_dominance_ratio);
    // WExploration (a far-is-better distance BONUS) is gone — the sign was wrong and produced a
    // corner-to-corner oscillation. Travel distance is a cost; see Params::w_travel_cost.
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WTravelCost", ep.w_travel_cost);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.IorCellSize", ep.ior_cell_size);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.IorDecayTime", ep.ior_decay_time);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WIor", ep.w_ior);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WIorDrive", ep.w_ior_drive);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WPathInterest", ep.w_path_interest);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.IorPathRadius", ep.ior_path_radius);
    // PatrolEnabled / PatrolGainFloor / InfoExhaustedGain were the second-level "patrol mode"
    // switch. Removed: the epistemic term is now MARGINAL (it extinguishes itself when there is
    // nothing left to see) and the IoR drive is unbounded in neglect age (it never flatlines), so
    // the hand-off happens continuously and there is nothing left to trigger or floor.
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimCornerSigma", ep.fim_corner_sigma);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimMaxRange", ep.fim_max_range);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimPriorPrecisionFloor", ep.fim_prior_precision_floor);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "EpistemicController.FimUseWalls", ep.fim_use_walls);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimWallSigma", ep.fim_wall_sigma);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "EpistemicController.FimWallRays", ep.fim_wall_rays);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "EpistemicController.FimWallIncidenceWeight", ep.fim_wall_incidence_weight);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.FimWallIncidenceMin", ep.fim_wall_incidence_min);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.ArrivalDistance", ep.arrival_distance);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.DwellTime", ep.dwell_time);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.BeliefForgetTime", ep.belief_forget_time);
    epistemic.set_robot_footprint(p.ROBOT_WIDTH, p.ROBOT_LENGTH);
}


std::vector<std::string> RoomConfig::apply_platform(const std::string& robot)
{
    std::vector<std::string> changed;
    const auto it = platform_overlays.find(robot);
    if (it == platform_overlays.end()) return changed;   // no section for this robot: keep the shared defaults
    const auto& ov = it->second;
    const auto set = [&](const std::optional<float>& src, float& dst, const char* name)
    { if (src.has_value() and *src != dst) { changed.emplace_back(name); dst = *src; } };
    set(ov.mount_pitch_sigma,    IMAGE_EDGE_MOUNT_PITCH_SIGMA,  "mountPitchSigma");
    set(ov.mount_height_sigma,   IMAGE_EDGE_MOUNT_HEIGHT_SIGMA, "mountHeightSigma");
    set(ov.mount_yaw_sigma,      IMAGE_EDGE_MOUNT_YAW_SIGMA,    "mountYawSigma");
    set(ov.mount_yaw_correction, IMAGE_EDGE_MOUNT_YAW_CORR,     "mountYawCorrection");
    set(ov.wall_position_sigma,  IMAGE_EDGE_WALL_POS_SIGMA,     "wallPositionSigma");
    if (ov.image_edge_camera.has_value() and *ov.image_edge_camera != IMAGE_EDGE_CAMERA)
    { changed.emplace_back("camera"); IMAGE_EDGE_CAMERA = *ov.image_edge_camera; }
    if (ov.calib_pivot_enabled.has_value() and *ov.calib_pivot_enabled != CALIB_PIVOT_ENABLED)
    { changed.emplace_back("CalibPivotEnabled"); CALIB_PIVOT_ENABLED = *ov.calib_pivot_enabled; }
    if (ov.odom_sample_log.has_value() and *ov.odom_sample_log != ODOM_SAMPLE_LOG)
    { changed.emplace_back("OdomSampleLog"); ODOM_SAMPLE_LOG = *ov.odom_sample_log; }
    if (ov.object_anchor_meas_sigma_xy.has_value() and *ov.object_anchor_meas_sigma_xy != OBJECT_ANCHOR_MEAS_SIG_XY)
    { changed.emplace_back("ObjectAnchorMeasSigmaXY"); OBJECT_ANCHOR_MEAS_SIG_XY = *ov.object_anchor_meas_sigma_xy; }
    if (ov.stable_sdf_mse_max.has_value() and *ov.stable_sdf_mse_max != STABLE_SDF_MSE_MAX)
    { changed.emplace_back("StableSdfMseMax"); STABLE_SDF_MSE_MAX = *ov.stable_sdf_mse_max; }
    return changed;
}

// The overlay values whose destinations live in RoomConcept / EpistemicController rather than here.
// ⚠ CmdNoiseRot/Trans were PARSED by apply_platform and applied by nothing — harmless while each
//   robot had its own file (the top-level key did the work) and a silent regression the moment the
//   files merged. They are applied here.
std::vector<std::string> RoomConfig::apply_platform_to(const std::string& robot,
                                                       rc::RoomConcept& room_concept,
                                                       rc::EpistemicController& epistemic)
{
    std::vector<std::string> changed;
    const auto it = platform_overlays.find(robot);
    if (it == platform_overlays.end()) return changed;
    const auto& ov = it->second;
    const auto set = [&](const auto& src, auto& dst, const char* name)
    { if (src.has_value() and *src != dst) { changed.emplace_back(name); dst = *src; } };
    auto& rp = room_concept.params;
    set(ov.cmd_noise_rot,           rp.cmd_noise_rot,                     "CmdNoiseRot");
    set(ov.cmd_noise_trans,         rp.cmd_noise_trans,                   "CmdNoiseTrans");
    set(ov.use_command_velocity_prior, rp.use_command_velocity_prior,     "UseCommandVelocityPrior");
    set(ov.zupt_density_v,          rp.odom_preint_noise.zupt_density_v,     "PreintZuptDensityV");
    set(ov.zupt_density_omega,      rp.odom_preint_noise.zupt_density_omega, "PreintZuptDensityOmega");
    set(ov.odom_preint_sigma_omega, rp.odom_preint_noise.sigma_omega,        "PreintOdomSigmaOmega");
    set(ov.odom_preint_scale_omega, rp.odom_preint_noise.scale_omega,        "PreintOdomScaleOmega");
    set(ov.sdf_safe,                epistemic.params.sdf_safe,            "SdfSafe");
    set(ov.sdf_danger,              epistemic.params.sdf_danger,          "SdfDanger");
    // Drift keys — same mechanism, different reason. See PlatformOverlay in the header.
    set(ov.recovery_loss_threshold,         rp.recovery_loss_threshold,               "RecoveryLossThreshold");
    set(ov.prediction_trust_factor,         rp.prediction_trust_factor,               "PredictionTrustFactor");
    set(ov.rotation_sdf_coupling,           rp.rotation_sdf_coupling,                 "RotationSdfCoupling");
    set(ov.boundary_hessian_quality_threshold, rp.boundary_hessian_quality_threshold,    "BoundaryHessianQualityThreshold");
    set(ov.boundary_mu_quality_threshold,   rp.boundary_mu_quality_threshold,         "BoundaryMuQualityThreshold");
    set(ov.symmetry_good_fit_mse,           rp.symmetry_good_fit_mse,                 "SymmetryGoodFitMse");
    set(ov.gn_loss_rel_tol,                 rp.gn_loss_rel_tol,                       "GnLossRelTol");
    set(ov.gn_max_iters,                    rp.gn_max_iters,                          "GnMaxIters");
    set(ov.torch_num_threads,               rp.torch_num_threads,                     "TorchNumThreads");
    set(ov.adaptive_cov_enabled,            rp.adaptive_cov_enabled,                  "AdaptiveCovEnabled");
    set(ov.window_stride_enabled,           rp.window_stride_enabled,                 "WindowStrideEnabled");
    set(ov.boundary_fej_schur,              rp.boundary_fej_schur,                    "BoundaryFejSchur");
    set(ov.hier_prec_boundary_enabled,      rp.hier_prec_boundary_enabled,            "HierPrecBoundaryEnabled");
    set(ov.corner_early_exit_check,         rp.corner_early_exit_check,               "CornerEarlyExitCheck");
    set(ov.w_ior, epistemic.epistemic_planner().params.w_ior, "WIor");
    set(ov.belief_forget_time, epistemic.epistemic_planner().params.belief_forget_time, "BeliefForgetTime");
    return changed;
}

std::vector<std::string> RoomConfig::apply_scenario_to(const std::string& scenario,
                                                       rc::EpistemicController& epistemic)
{
    std::vector<std::string> changed;
    const auto it = scenario_overlays.find(scenario);
    if (it == scenario_overlays.end()) return changed;
    const auto& ov = it->second;
    if (ov.target_wall_margin.has_value() and
        *ov.target_wall_margin != epistemic.epistemic_planner().params.target_wall_margin)
    { changed.emplace_back("TargetWallMargin");
      epistemic.epistemic_planner().params.target_wall_margin = *ov.target_wall_margin; }
    return changed;
}

std::vector<std::string> RoomConfig::apply_scenario(const std::string& scenario)
{
    std::vector<std::string> changed;
    const auto it = scenario_overlays.find(scenario);
    if (it == scenario_overlays.end()) return changed;
    const auto& ov = it->second;
    if (ov.room_layout_svg.has_value() and *ov.room_layout_svg != ROOM_LAYOUT_SVG)
    { changed.emplace_back("RoomLayoutSvg"); ROOM_LAYOUT_SVG = *ov.room_layout_svg; }
    if (ov.room_height.has_value() and *ov.room_height != room_height)
    { changed.emplace_back("RoomHeight"); room_height = *ov.room_height; }
    if (ov.recenter_room_polygon.has_value() and *ov.recenter_room_polygon != RECENTER_ROOM_POLYGON)
    { changed.emplace_back("RecenterRoomPolygon"); RECENTER_ROOM_POLYGON = *ov.recenter_room_polygon; }
    if (ov.lidar_high_max_height.has_value() and *ov.lidar_high_max_height != LIDAR_HIGH_MAX_HEIGHT)
    { changed.emplace_back("LidarHighMaxHeight"); LIDAR_HIGH_MAX_HEIGHT = *ov.lidar_high_max_height; }
    if (ov.object_anchor_subtypes.has_value() and *ov.object_anchor_subtypes != OBJECT_ANCHOR_SUBTYPES)
    { changed.emplace_back("ObjectAnchorSubtypes"); OBJECT_ANCHOR_SUBTYPES = *ov.object_anchor_subtypes; }

    // ── A WALL BAND MAY NOT REACH THE CEILING ───────────────────────────────────────────────────
    // The high band feeds a 2-D WALL polygon SDF. A ceiling return sits at an arbitrary INTERIOR
    // xy, so its distance to the nearest wall is large by construction: let the ceiling plane into
    // the band and the localiser is fitting a wall model to points that are not on a wall.
    // ★ MEASURED 2026-08-29: with LidarHighMaxHeight raised to 3.0 in a room whose RoomHeight is
    //   3.0, the startup check found a 51450-point z-peak at 3.01 m and the SDF residual sat at
    //   0.164 m RMS after optimisation, gate pinned, 0% early exit — while the pose stayed
    //   confident and corner association healthy, because the pose was not the thing that was wrong.
    // ★ NOT a tuned threshold: it is the same LIDAR_CEILING_MARGIN the startup ceiling check already
    //   applies, enforced against the ceiling the SCENARIO states rather than only against the one
    //   the check manages to detect. The check can be fooled — it was, here — and a stated ceiling
    //   cannot be. Announced, never silent: a value that is quietly reduced is a value nobody
    //   revisits.
    if (const float cap = room_height - LIDAR_CEILING_MARGIN;
        room_height > 0.f and LIDAR_HIGH_MAX_HEIGHT > cap)
    {
        qWarning() << "[cfg] LidarHighMaxHeight" << LIDAR_HIGH_MAX_HEIGHT
                   << "m reaches the stated" << room_height << "m ceiling; the high band feeds a 2-D"
                   << "WALL SDF and a ceiling return has no wall to be near. Clamped to" << cap
                   << "m (ceiling -" << LIDAR_CEILING_MARGIN << "m).";
        LIDAR_HIGH_MAX_HEIGHT = cap;
        changed.emplace_back("LidarHighMaxHeight(clamped below the ceiling)");
    }
    return changed;
}

}  // namespace rc
