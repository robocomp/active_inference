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

    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdometryNoiseFactor", p.ODOMETRY_NOISE_FACTOR);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "RoomConcept.OdomNoiseScale", room_concept.params.odom_noise_scale);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.DifferentialTest", room_concept.params.differential_test_enabled);
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "RoomConcept.SdfCurrentSlotOnly", room_concept.params.sdf_current_slot_only);
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

    // Media plane (RGB for the camera window + LiDAR for LidarIngestor). DDS domain +
    // topics are read from the producer's media descriptor on the graph, not config.
    rc::ConfigLoaderUtils::load_optional<bool>(cl, "Media.lidar_use_media", p.LIDAR_USE_MEDIA);

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
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.AngularDominanceRatio", ep.angular_dominance_ratio);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WExploration", ep.w_exploration);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.IorCellSize", ep.ior_cell_size);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.IorDecayTime", ep.ior_decay_time);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WIor", ep.w_ior);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WIorDrive", ep.w_ior_drive);
    rc::ConfigLoaderUtils::load_optional<float, double>(cl, "EpistemicController.WPathInterest", ep.w_path_interest);
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
