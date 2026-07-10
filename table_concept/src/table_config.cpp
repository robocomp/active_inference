/*
 * table_config.cpp — fill TableConfig from a RoboComp ConfigLoader.
 */

#include "table_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

TableConfig load_table_config(const ConfigLoader& cfg)
{
    TableConfig out;

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

    // Agent convergence
    out.state_eps                = getf("TableConcept.StateEps",               0.04f);
    out.K_stable                 = geti("TableConcept.KStable",                30);
    out.detection_alive_max_frames = geti("TableConcept.DetectionAliveMaxFrames", 40);
    out.obs_distance             = getf("TableConcept.ObsDistance",            1.8f);
    out.epistemic_cooldown_cycles= geti("TableConcept.EpistemicCooldownCycles", 200);
    out.table_log_period_frames  = geti("TableConcept.TableLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("TableConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("TableConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("TableConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("TableConcept.VoxelSelectHeightMarginM", 0.25f);

    // TableModel geometry / mask split
    out.sigma_obs          = getf("TableModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("TableModel.SdfThresholdForStorage", 0.08f);

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("TableModel.AI2SigmaBaseM",       0.03f);
    out.ai2_clutter_frac     = getf("TableModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("TableModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("TableModel.AI2PriorSizeStd",     0.30f);
    out.ai2_process_std_m    = getf("TableModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("TableModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_age_nominal_dt_s = getf("TableModel.AI2AgeNominalDtS",    0.0f);
    out.ai2_common_mode_pos_std  = getf("TableModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("TableModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("TableModel.AI2CommonModeYawStd",  0.03f);
    out.ai2_range_noise_lat_per_m = getf("TableModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("TableModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_range_noise_size_per_m = getf("TableModel.AI2RangeNoiseSizePerM", 0.08f);
    out.ai2_trunc_gate_frac    = getf("TableModel.AI2TruncGateFrac",   0.10f);
    out.ai2_gn_iters         = geti("TableModel.AI2GnIters",          4);
    out.ai2_csv_path         = gets("TableModel.AI2CsvPath",          "");

    out.rt_cov_scale                  = getf("TableConcept.RtCovScale",           1.0f);
    out.rt_cov_add_chain              = getb("TableConcept.RtCovAddChain",       true);

    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.50f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",      300);
    out.tracker_death_enabled    = getb("Tracker.DeathEnabled",     false);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.60f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.05f);
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM",      1.0f);
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM",      0.6f);
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM",     0.75f);
    out.tracker_birth_size_std   = getf("Tracker.BirthSizeStd",     0.15f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",          false);
    out.ricoh_birth_conf         = getf("Tracker.RicohBirthConf",   0.60f);
    out.ricoh_birth_max_var      = getf("Tracker.RicohBirthMaxVar", 0.005f);
    out.use_ricoh_slices         = getb("Tracker.UseRicohSlices",   true);
    out.ricoh_anchor_sigma_m     = getf("Tracker.RicohAnchorSigmaM", 0.50f);

    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("TableModel.LidarPrecision",      0.0f);
    out.lidar_robust_c_m     = getf("TableModel.LidarRobustCM",       0.05f);
    out.lidar_select_margin_m = getf("TableModel.LidarSelectMarginM", 0.10f);
    out.lidar_frame_node      = gets("TableModel.LidarFrameNode",     "lidar3D");
    out.lidar_coverage_n0     = getf("TableModel.LidarCoverageN0",     60.0f);
    out.lidar_coverage_ang_power = getf("TableModel.LidarCoverageAngPower", 1.0f);
    out.max_step_m            = getf("TableModel.MaxStepM",            1.0f);
    out.coverage_precision    = getf("TableModel.CoveragePrecision",  0.0f);
    out.coverage_robust_c_m   = getf("TableModel.CoverageRobustCM",   0.15f);
    out.free_space_precision  = getf("TableModel.FreeSpacePrecision", 0.0f);
    out.footprint_moment_precision = getf("TableModel.FootprintMomentPrecision", 0.0f);
    out.footprint_moment_range_per_m = getf("TableModel.FootprintMomentRangePerM", 0.03f);
    out.fe_baseline_adapt_down       = getf("TableModel.FeBaselineAdaptDown", 0.05f);
    out.fe_baseline_adapt_up         = getf("TableModel.FeBaselineAdaptUp",   0.005f);
    out.fe_surprise_smooth           = getf("TableModel.FeSurpriseSmooth",    0.10f);
    out.footprint_moment_motion_gain = getf("TableModel.FootprintMomentMotionGain", 0.30f);
    out.orientation_motion_ref       = getf("TableModel.OrientationMotionRef", 0.50f);

    out.existence_removal_enabled = getb("TableModel.ExistenceRemovalEnabled", false);
    out.existence_removal_prob    = getf("TableModel.ExistenceRemovalProb",    0.12f);
    out.existence_logodds_max     = getf("TableModel.ExistenceLogoddsMax",     4.0f);
    out.existence_detection_prob  = getf("TableModel.ExistenceDetectionProb",  0.85f);
    out.existence_clutter_prob    = getf("TableModel.ExistenceClutterProb",    0.05f);
    out.existence_sensor_sigma_m  = getf("TableModel.ExistenceSensorSigmaM",   0.03f);
    out.existence_remove_frames   = geti("TableModel.ExistenceRemoveFrames",   15);
    out.existence_absence_range_ref_m = getf("TableModel.ExistenceAbsenceRangeRefM", 2.5f);
    out.existence_absence_range_power = getf("TableModel.ExistenceAbsenceRangePower", 2.0f);
    out.existence_lidar_absence       = getb("TableModel.ExistenceLidarAbsence", false);

    std::print("table_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
