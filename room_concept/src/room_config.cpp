/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_config.h.
 */

#include "room_config.h"

#include <ConfigLoader/ConfigLoader.h>

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
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseScale", room_concept.params.odom_noise_scale);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.DifferentialTest", room_concept.params.differential_test_enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.SdfCurrentSlotOnly", room_concept.params.sdf_current_slot_only);
    // Raise the published covariance to at least what the innovations demonstrate, so the sigma stops
    // being a constant. Default OFF — see Params::adaptive_cov_enabled for the acceptance test and for
    // the two measured process-noise approaches that failed before it.
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

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseTrans", room_concept.params.cmd_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseRot", room_concept.params.cmd_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.CmdNoiseBase", room_concept.params.cmd_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseTrans", room_concept.params.odom_noise_trans);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseRot", room_concept.params.odom_noise_rot);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseBase", room_concept.params.odom_noise_base);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.EncoderRotSlipK", room_concept.params.encoder_rot_slip_k);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.StationaryMotionThreshold", room_concept.params.stationary_motion_threshold);

    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.LearnMotionModel", room_concept.params.learn_motion_model);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionLearnAlpha", room_concept.params.motion_learn_alpha);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionLearnBeta", room_concept.params.motion_learn_beta);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionLearnMinOmega", room_concept.params.motion_learn_min_omega);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionLearnMinTrans", room_concept.params.motion_learn_min_trans);
    rc::ConfigLoaderUtils::load_optional<int>(cl, "RoomConcept.MotionLearnMinFrames", room_concept.params.motion_learn_min_frames);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.MotionLearnQualityThreshold", room_concept.params.motion_learn_quality_threshold);

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
    room_concept.params.object_anchor.weight      = p.OBJECT_ANCHOR_WEIGHT;
    room_concept.params.object_anchor.huber_delta = p.OBJECT_ANCHOR_HUBER;
    room_concept.params.object_anchor_max_slots   = p.OBJECT_ANCHOR_MAX_SLOTS;
    room_concept.params.object_anchor_early_exit_sigma = p.OBJECT_ANCHOR_EARLY_EXIT_SIGMA;
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

}  // namespace rc
