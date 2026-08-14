/*
 * table_config.cpp  —  fill TableConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in table_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "table_config.h"

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

TableConfig load_table_config(const ConfigLoader& cfg)
{
    TableConfig out;

    // The one place this path is written. It is relative to the agent's CWD (<agent>/), not to src/.
    static constexpr const char* kManifestPath = "../common/concept_manifest/table.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL. The manifest has carried `from = measured|fitted|nominal|inherited`
    // since 2026-08-11, with a note saying "inherited is worse than absent — an absent value gets asked
    // about, an inherited one gets trusted". A note stopped nothing: hood_concept shipped TEN cloned defects
    // that week, several of them DECLARED as inherited, in writing, in this very file. The declaration is
    // now the enforcement — see rc::manifest::provenance_ok. Refusing to start is the whole point: a comment
    // cannot fail a build, and everything that only warned was fixed around rather than fixed.
    if (not rc::manifest::provenance_ok(kManifestPath, "table"))
        std::exit(EXIT_FAILURE);

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
    out.state_eps                = getf("TableConcept.StateEps",               0.04f);
    out.K_stable                 = geti("TableConcept.KStable",                30);
    out.detection_alive_max_frames = geti("TableConcept.DetectionAliveMaxFrames", 40);
    out.matched_frames_before_aging = geti("TableConcept.MatchedFramesBeforeAging", 5);
    out.central_region_frac      = getf("TableConcept.CentralRegionFrac",     0.25f);
    out.obs_distance             = getf("TableConcept.ObsDistance",            1.8f);
    out.epistemic_cooldown_cycles= geti("TableConcept.EpistemicCooldownCycles", 200);
    out.table_log_period_frames  = geti("TableConcept.TableLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("TableConcept.VoxelBankMaxPoints",     4000);
    out.publish_voxel_bank        = getb("TableConcept.PublishVoxelBank",       false);
    out.voxel_bank_quantization_m= getf("TableConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("TableConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("TableConcept.VoxelSelectHeightMarginM", 0.25f);

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.show_dashboard           = getb("TableConcept.ShowDashboard",          true);
    out.shape_eval_period        = geti("TableConcept.ShapeEvalPeriod",        30);
    out.shape_eval_min_points    = geti("TableConcept.ShapeEvalMinPoints",     300);
    out.shape_evidence_clamp     = getf("TableConcept.ShapeEvidenceClamp",     8.0f);
    out.shape_evidence_ema_alpha = getf("TableModel.ShapeEvidenceEmaAlpha", 0.0f);
    out.dump_cloud_path          = gets("TableConcept.DumpCloudPath",          "");

    // ─── TableModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("TableModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("TableModel.SdfThresholdForStorage", 0.08f);

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("TableModel.AI2SigmaBaseM",       0.03f);
    out.detect_min_fill      = getf("TableModel.DetectMinFill",       0.10f);
    out.detect_max_fill      = getf("TableModel.DetectMaxFill",       0.60f);
    out.detect_soft          = getf("TableModel.DetectSoft",          0.06f);
    out.ai2_clutter_frac     = getf("TableModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("TableModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("TableModel.AI2PriorSizeStd",     0.30f);
    out.ai2_process_std_m    = getf("TableModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("TableModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_process_std_extent_m  = getf("TableModel.AI2ProcessStdExtentM",  0.0f);
    out.ai2_clamp_sigma_to_prior  = getb("TableModel.AI2ClampSigmaToPrior",  true);
    out.ai2_age_nominal_dt_s = getf("TableModel.AI2AgeNominalDtS",    0.0f);
    out.ai2_common_mode_pos_std  = getf("TableModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("TableModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("TableModel.AI2CommonModeYawStd",  0.03f);
    out.motion_cm_pos_gain       = getf("TableModel.MotionCmPosGain",      0.10f);
    out.motion_cm_size_gain      = getf("TableModel.MotionCmSizeGain",     0.20f);
    out.motion_cm_yaw_gain       = getf("TableModel.MotionCmYawGain",      0.12f);
    out.ai2_range_noise_lat_per_m = getf("TableModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("TableModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_range_noise_size_per_m = getf("TableModel.AI2RangeNoiseSizePerM", 0.08f);
    out.ai2_coverage_size_gain     = getf("TableModel.AI2CoverageSizeGain",   0.30f);
    out.ai2_coverage_yaw_gain      = getf("TableModel.AI2CoverageYawGain",    0.15f);
    out.ai2_coverage_pos_gain      = getf("TableModel.AI2CoveragePosGain",    0.05f);
    out.ai2_coverage_min           = getf("TableModel.AI2CoverageMin",        0.02f);
    out.ai2_trunc_gate_frac    = getf("TableModel.AI2TruncGateFrac",   0.10f);
    out.fixation_enabled       = getb("TableModel.FixationEnabled",     true);
    out.fixation_min_pts       = geti("TableModel.FixationMinPts",      300);
    out.fixation_max_trunc     = getf("TableModel.FixationMaxTrunc",    0.10f);
    out.fixation_range_m       = getf("TableModel.FixationRangeM",      0.0f);   // retired as a gate
    out.fixation_centre_frac   = getf("TableModel.FixationCentreFrac",  0.60f);
    out.fixation_still_dotd    = getf("TableModel.FixationStillDotd",   0.05f);
    out.ai2_gn_iters         = geti("TableModel.AI2GnIters",          4);
    out.ai2_csv_path         = gets("TableModel.AI2CsvPath",          "");
    out.birth_surprise_probe = getb("TableModel.BirthSurpriseProbe",  false);
    out.pixel_sigma_over_f     = getf("TableModel.PixelSigmaOverF",       0.0015f);
    out.depth_sigma0_m         = getf("TableModel.DepthSigma0M",          0.006f);
    out.depth_sigma_range_coef = getf("TableModel.DepthSigmaRangeCoef",   0.004f);
    out.model_sigma_m          = getf("TableModel.ModelSigmaM",           0.010f);
    out.footprint_residual     = getb("TableModel.FootprintResidual",     false);
    out.quotient_chart         = getb("TableModel.QuotientChart",          false);
    out.depth_tilt_std         = getf("TableModel.DepthTiltStd",          0.020f);
    out.depth_bias_std         = getf("TableModel.DepthBiasStd",          0.015f);
    out.depth_scale_std        = getf("TableModel.DepthScaleStd",         0.010f);

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("TableConcept.RtCovScale",           1.0f);
    out.publish_object_obs            = getb("TableConcept.PublishObjectObs",   false);
    out.object_obs_frame              = gets("TableConcept.ObjectObsFrame",     "body");

    // ─── Multi-instance tracker + ricoh attention ──────────────────────────────
    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.50f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.35f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.birth_fusion             = getb("Tracker.BirthFusion",       false);
    out.birth_fusion_gain        = getf("Tracker.BirthFusionGain",   6.0f);
    out.birth_fusion_mass_ref    = getf("Tracker.BirthFusionMassRef",8.0f);
    out.birth_fusion_radius_m    = getf("Tracker.BirthFusionRadiusM",0.50f);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.60f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.05f);
    out.tracker_birth_width_m    = getf("Tracker.BirthWidthM",      1.0f);
    out.tracker_birth_depth_m    = getf("Tracker.BirthDepthM",      0.6f);
    out.tracker_birth_height_m   = getf("Tracker.BirthHeightM",     0.75f);
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f);
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f);
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM",  1.0f);

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("TableModel.LidarPrecision",      0.0f);
    out.lidar_bpearl_precision = getf("TableModel.LidarBpearlPrecision", 0.0f);
    out.lidar_robust_c_m     = getf("TableModel.LidarRobustCM",       0.05f);
    out.lidar_select_margin_m = getf("TableModel.LidarSelectMarginM", 0.10f);
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
    out.obliquity_moment_gain        = getf("TableModel.ObliquityMomentGain", 0.0f);
    out.footprint_moment_completeness_gain = getf("TableModel.FootprintMomentCompletenessGain", 0.0f);
    out.footprint_moment_min_completeness  = getf("TableModel.FootprintMomentMinCompleteness",  0.02f);

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("TableModel.ExistenceRemovalEnabled", false);
    out.existence_removal_prob    = getf("TableModel.ExistenceRemovalProb",    0.12f);
    out.existence_logodds_max     = getf("TableModel.ExistenceLogoddsMax",     4.0f);
    out.existence_frame_correlation = getf("TableModel.ExistenceFrameCorrelation", 0.0f);
    out.existence_detection_prob  = getf("TableModel.ExistenceDetectionProb",  0.85f);
    out.existence_clutter_prob    = getf("TableModel.ExistenceClutterProb",    0.05f);
    out.existence_sensor_sigma_m  = getf("TableModel.ExistenceSensorSigmaM",   0.03f);
    out.existence_remove_frames   = geti("TableModel.ExistenceRemoveFrames",   15);
    out.existence_absence_range_ref_m = getf("TableModel.ExistenceAbsenceRangeRefM", 2.5f);
    out.existence_absence_range_power = getf("TableModel.ExistenceAbsenceRangePower", 2.0f);
    out.existence_los_margin_m        = getf("TableModel.ExistenceLosMarginM",       0.25f);
    out.existence_los_azim_bins       = geti("TableModel.ExistenceLosAzimBins",        0);
    out.existence_verify_surprise     = getf("TableModel.ExistenceVerifySurprise",   20.0f);
    out.verify_surprise_smooth        = getf("TableModel.VerifySurpriseSmooth",       0.10f);
    out.existence_verify_gain         = getf("TableModel.ExistenceVerifyGain",       5.0f);
    out.existence_leg_occupancy       = getb("TableModel.ExistenceLegOccupancy", true);
    out.existence_local_clutter       = getb("TableModel.ExistenceLocalClutter", true);
    out.existence_lidar_confirms      = getb("TableModel.ExistenceLidarConfirms", false);
    out.existence_absence_envelope    = getb("TableModel.ExistenceAbsenceEnvelope", true);

    std::print("table_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
