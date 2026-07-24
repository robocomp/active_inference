/*
 * cabinet_config.cpp  —  fill CabinetConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in cabinet_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "cabinet_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

CabinetConfig load_cabinet_config(const ConfigLoader& cfg)
{
    CabinetConfig out;

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };
    auto getb = [&](const std::string& k, bool def) -> bool {
        return cfg.exists(k) ? cfg.get<bool>(k) : def;
    };

    // ─── Agent convergence & cadence ───────────────────────────────────────────
    out.state_eps                = getf("CabinetConcept.StateEps",               0.04f);
    out.K_stable                 = geti("CabinetConcept.KStable",                30);
    out.detection_alive_max_frames = geti("CabinetConcept.DetectionAliveMaxFrames", 40);
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.obs_distance             = getf("CabinetConcept.ObsDistance",            1.8f);
    out.epistemic_cooldown_cycles= geti("CabinetConcept.EpistemicCooldownCycles", 200);
    out.cabinet_log_period_frames  = geti("CabinetConcept.CabinetLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("CabinetConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("CabinetConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("CabinetConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("CabinetConcept.VoxelSelectHeightMarginM", 0.25f);

    // ─── CabinetModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("CabinetModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("CabinetModel.SdfThresholdForStorage", 0.08f);

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("CabinetModel.AI2SigmaBaseM",       0.03f);
    out.ai2_clutter_frac     = getf("CabinetModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("CabinetModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("CabinetModel.AI2PriorSizeStd",     0.30f);
    out.ai2_process_std_m    = getf("CabinetModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("CabinetModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_age_nominal_dt_s = getf("CabinetModel.AI2AgeNominalDtS",    0.0f);
    out.ai2_common_mode_pos_std  = getf("CabinetModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("CabinetModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("CabinetModel.AI2CommonModeYawStd",  0.03f);
    out.ai2_range_noise_lat_per_m = getf("CabinetModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("CabinetModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_range_noise_size_per_m = getf("CabinetModel.AI2RangeNoiseSizePerM", 0.08f);
    out.ai2_trunc_gate_frac    = getf("CabinetModel.AI2TruncGateFrac",   0.10f);
    out.ai2_gn_iters         = geti("CabinetModel.AI2GnIters",          4);
    out.ai2_csv_path         = gets("CabinetModel.AI2CsvPath",          "");
    out.birth_surprise_probe = getb("CabinetModel.BirthSurpriseProbe",  false);
    out.pixel_sigma_over_f     = getf("CabinetModel.PixelSigmaOverF",       0.0015f);
    out.depth_sigma0_m         = getf("CabinetModel.DepthSigma0M",          0.006f);
    out.depth_sigma_range_coef = getf("CabinetModel.DepthSigmaRangeCoef",   0.004f);
    out.model_sigma_m          = getf("CabinetModel.ModelSigmaM",           0.010f);
    out.footprint_residual     = getb("CabinetModel.FootprintResidual",     false);
    out.quotient_chart         = getb("CabinetModel.QuotientChart",          false);
    out.erosion_px_std = getf("CabinetModel.ErosionPxStd", 2.0f);
    out.depth_bias_std         = getf("CabinetModel.DepthBiasStd",          0.015f);
    out.depth_scale_std        = getf("CabinetModel.DepthScaleStd",         0.010f);

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("CabinetConcept.RtCovScale",           1.0f);
    out.publish_object_obs            = getb("CabinetConcept.PublishObjectObs",   false);
    out.object_obs_frame              = gets("CabinetConcept.ObjectObsFrame",     "body");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.50f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.birth_fusion             = getb("Tracker.BirthFusion",       false);
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain",   6.0f);
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef",8.0f);
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM",0.50f);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",      300);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.60f);
    out.tracker_z_gate_m         = getf("Tracker.ZGateM",           0.60f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.05f);
    out.merge_n_sigma            = getf("Tracker.MergeNSigma",       3.0f);
    out.merge_gap_floor_m        = getf("Tracker.MergeGapFloorM",    0.30f);
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM",      1.0f);
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM",      0.6f);
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM",     0.75f);
    out.residual_birth_enabled   = getb("Tracker.ResidualBirthEnabled", true);
    out.residual_birth_frames    = geti("Tracker.ResidualBirthFrames",  4);
    out.residual_birth_match_m   = getf("Tracker.ResidualBirthMatchM",  0.40f);
    out.residual_birth_min_pts   = geti("Tracker.ResidualBirthMinPts",  600);
    out.residual_birth_sep_m     = getf("Tracker.ResidualBirthSepM",    0.60f);
    out.residual_claim_frac      = getf("Tracker.ResidualClaimFrac",    0.15f);
    out.residual_claim_margin_m  = getf("Tracker.ResidualClaimMarginM", 0.20f);
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f);
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f);
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM",  1.0f);

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("CabinetModel.LidarPrecision",      0.0f);
    out.lidar_bpearl_precision = getf("CabinetModel.LidarBpearlPrecision", 0.0f);
    out.lidar_robust_c_m     = getf("CabinetModel.LidarRobustCM",       0.05f);
    out.lidar_select_margin_m = getf("CabinetModel.LidarSelectMarginM", 0.10f);
    out.lidar_coverage_n0     = getf("CabinetModel.LidarCoverageN0",     60.0f);
    out.lidar_coverage_ang_power = getf("CabinetModel.LidarCoverageAngPower", 1.0f);
    out.max_step_m            = getf("CabinetModel.MaxStepM",            1.0f);
    out.coverage_precision    = getf("CabinetModel.CoveragePrecision",  0.0f);
    out.coverage_robust_c_m   = getf("CabinetModel.CoverageRobustCM",   0.15f);
    out.free_space_precision  = getf("CabinetModel.FreeSpacePrecision", 0.0f);
    out.footprint_moment_precision = getf("CabinetModel.FootprintMomentPrecision", 0.0f);
    out.footprint_moment_range_per_m = getf("CabinetModel.FootprintMomentRangePerM", 0.03f);
    out.fe_baseline_adapt_down       = getf("CabinetModel.FeBaselineAdaptDown", 0.05f);
    out.fe_baseline_adapt_up         = getf("CabinetModel.FeBaselineAdaptUp",   0.005f);
    out.fe_surprise_smooth           = getf("CabinetModel.FeSurpriseSmooth",    0.10f);
    out.footprint_moment_motion_gain = getf("CabinetModel.FootprintMomentMotionGain", 0.30f);
    out.orientation_motion_ref       = getf("CabinetModel.OrientationMotionRef", 0.50f);
    out.obliquity_moment_gain        = getf("CabinetModel.ObliquityMomentGain", 0.0f);
    out.footprint_moment_completeness_gain = getf("CabinetModel.FootprintMomentCompletenessGain", 0.0f);
    out.footprint_moment_min_completeness  = getf("CabinetModel.FootprintMomentMinCompleteness",  0.02f);

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("CabinetModel.ExistenceRemovalEnabled", false);
    out.existence_removal_prob    = getf("CabinetModel.ExistenceRemovalProb",    0.12f);
    out.existence_logodds_max     = getf("CabinetModel.ExistenceLogoddsMax",     4.0f);
    out.existence_detection_prob  = getf("CabinetModel.ExistenceDetectionProb",  0.85f);
    out.existence_clutter_prob    = getf("CabinetModel.ExistenceClutterProb",    0.05f);
    out.existence_sensor_sigma_m  = getf("CabinetModel.ExistenceSensorSigmaM",   0.03f);
    out.existence_remove_frames   = geti("CabinetModel.ExistenceRemoveFrames",   15);
    out.existence_absence_range_ref_m = getf("CabinetModel.ExistenceAbsenceRangeRefM", 2.5f);
    out.existence_absence_range_power = getf("CabinetModel.ExistenceAbsenceRangePower", 2.0f);
    out.existence_verify_surprise     = getf("CabinetModel.ExistenceVerifySurprise",   20.0f);
    out.existence_verify_gain         = getf("CabinetModel.ExistenceVerifyGain",       5.0f);
    out.wall_precision                = getf("CabinetModel.WallPrecision", 400.0f);
    out.wall_reach_m                  = getf("CabinetModel.WallReachM", 0.35f);
    out.wall_sigma_m                  = getf("CabinetModel.WallSigmaM", 0.02f);
    out.wall_parallel_precision       = getf("CabinetModel.WallParallelPrecision", 200.0f);
    out.room_axis_precision           = getf("CabinetModel.RoomAxisPrecision", 300.0f);
    out.room_axis_capture_rad         = getf("CabinetModel.RoomAxisCaptureRad", 0.0f);
    out.seed_room_axis_snap           = getb("CabinetModel.SeedRoomAxisSnap", true);
    out.counter_evidence_enabled      = getb("CabinetModel.CounterEvidence", true);
    out.lshape_split_enabled          = getb("CabinetModel.LShapeSplitEnabled", true);
    out.lshape_min_arm_pts            = geti("CabinetModel.LShapeMinArmPts", 500);
    out.lshape_bin_m                  = getf("CabinetModel.LShapeBinM", 0.15f);
    out.lshape_arm_halfwidth_m        = getf("CabinetModel.LShapeArmHalfwidthM", 0.45f);
    out.extent_precision              = getf("CabinetModel.ExtentPrecision", 800.0f);
    out.existence_lidar_absence       = getb("CabinetModel.ExistenceLidarAbsence", false);

    std::print("cabinet_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
