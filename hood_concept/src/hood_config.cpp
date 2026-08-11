/*
 * hood_config.cpp  —  fill HoodConfig from a RoboComp ConfigLoader.
 *
 * Every key is optional: a missing TOML key keeps the default declared in hood_config.h. The typed
 * getf/geti/gets/getb helpers below just wrap ConfigLoader (which has no defaulted get overload).
 */

#include "hood_config.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <string>

#include <genericworker.h>   // ConfigLoader

namespace rc {

HoodConfig load_hood_config(const ConfigLoader& cfg)
{
    HoodConfig out;

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
    out.state_eps                = getf("HoodConcept.StateEps",               0.04f);
    out.K_stable                 = geti("HoodConcept.KStable",                30);
    out.detection_alive_max_frames = geti("HoodConcept.DetectionAliveMaxFrames", 40);
    out.matched_frames_before_aging = geti("HoodConcept.MatchedFramesBeforeAging", 5);
    out.central_region_frac      = getf("HoodConcept.CentralRegionFrac",     0.25f);
    out.epistemic_cooldown_cycles= geti("HoodConcept.EpistemicCooldownCycles", 200);
    out.hood_log_period_frames  = geti("HoodConcept.HoodLogPeriodFrames",   30);
    out.voxel_bank_max_points    = geti("HoodConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("HoodConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("HoodConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("HoodConcept.VoxelSelectHeightMarginM", 0.25f);

    // ─── Primary-input (masks) stream gate — lifecycle liveness ────────────────
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.show_dashboard           = getb("HoodConcept.ShowDashboard",          true);
    out.shape_eval_period        = geti("HoodConcept.ShapeEvalPeriod",        30);
    out.shape_eval_min_points    = geti("HoodConcept.ShapeEvalMinPoints",     300);
    out.shape_evidence_clamp     = getf("HoodConcept.ShapeEvidenceClamp",     8.0f);
    out.dump_cloud_path          = gets("HoodConcept.DumpCloudPath",          "");

    // ─── HoodModel geometry / mask split ──────────────────────────────────────
    out.sigma_obs          = getf("HoodModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("HoodModel.SdfThresholdForStorage", 0.08f);

    // ─── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m     = getf("HoodModel.AI2SigmaBaseM",       0.03f);
    out.ai2_clutter_frac     = getf("HoodModel.AI2ClutterFrac",      0.10f);
    out.ai2_clutter_scale_m  = getf("HoodModel.AI2ClutterScaleM",    0.12f);
    out.ai2_prior_size_std   = getf("HoodModel.AI2PriorSizeStd",     0.30f);
    out.ai2_prior_footprint_m   = getf("HoodModel.AI2PriorFootprintM",   0.60f);
    out.ai2_prior_footprint_std = getf("HoodModel.AI2PriorFootprintStd", 0.08f);
    out.ai2_prior_height_m      = getf("HoodModel.AI2PriorHeightM",      1.70f);
    out.vertical_extent_m = getf("HoodModel.VerticalExtentM", 0.50f);
    out.ai2_prior_height_std    = getf("HoodModel.AI2PriorHeightStd",    0.50f);
    out.ai2_depth_unobs_precision = getf("HoodModel.AI2DepthUnobsPrecision", 1500.0f);
    out.ai2_depth_obs_band_m      = getf("HoodModel.AI2DepthObsBandM",       0.10f);
    out.ai2_top_no_float_precision = getf("HoodModel.AI2TopNoFloatPrecision", 10000.0f);
    out.ai2_top_no_float_margin_m  = getf("HoodModel.AI2TopNoFloatMarginM",   0.02f);
    out.ai2_top_overseg_sigma_per_m = getf("HoodModel.AI2TopOversegSigmaPerM", 2.0f);
    out.ai2_wall_precision          = getf("HoodModel.AI2WallPrecision",         400.0f);
    out.ai2_wall_parallel_precision = getf("HoodModel.AI2WallParallelPrecision", 200.0f);
    out.ai2_wall_reach_m            = getf("HoodModel.AI2WallReachM",             0.15f);
    out.ai2_door_clearance_gain     = getf("HoodModel.AI2DoorClearanceGain",       3.0f);
    out.detect_min_fill             = getf("HoodModel.DetectMinFill",              0.10f);
    out.detect_max_fill             = getf("HoodModel.DetectMaxFill",              0.60f);
    out.detect_soft                 = getf("HoodModel.DetectSoft",                 0.06f);
    out.ai2_volatility_infer        = getb("HoodModel.AI2VolatilityInfer",        false);
    out.ai2_volatility_lr           = getf("HoodModel.AI2VolatilityLr",           0.02f);
    out.ai2_volatility_sigma        = getf("HoodModel.AI2VolatilitySigma",        2.0f);
    out.ai2_wall_explain_frac       = getf("HoodModel.AI2WallExplainFrac",        0.25f);
    out.ai2_wall_explain_sigma_m    = getf("HoodModel.AI2WallExplainSigmaM",      0.05f);
    out.ai2_wall_no_cross_precision = getf("HoodModel.WallNoCrossPrecision",     2000.0f);
    out.ai2_wall_no_cross_margin_m  = getf("HoodModel.WallNoCrossMarginM",       0.0f);
    out.ai2_process_std_m    = getf("HoodModel.AI2ProcessStdM",      0.005f);
    out.ai2_process_std_yaw  = getf("HoodModel.AI2ProcessStdYaw",    0.01f);
    out.ai2_age_nominal_dt_s = getf("HoodModel.AI2AgeNominalDtS",    0.0f);
    out.ai2_common_mode_pos_std  = getf("HoodModel.AI2CommonModePosStd",  0.03f);
    out.ai2_common_mode_size_std = getf("HoodModel.AI2CommonModeSizeStd", 0.02f);
    out.ai2_common_mode_yaw_std  = getf("HoodModel.AI2CommonModeYawStd",  0.03f);
    out.motion_cm_pos_gain       = getf("HoodModel.MotionCmPosGain",      0.10f);
    out.motion_cm_size_gain      = getf("HoodModel.MotionCmSizeGain",     0.20f);
    out.motion_cm_yaw_gain       = getf("HoodModel.MotionCmYawGain",      0.12f);
    out.ai2_ang_lever_m           = getf("HoodModel.AI2AngLeverM",           2.0f);
    out.ai2_periph_ref            = getf("HoodModel.AI2PeriphRef",           0.50f);
    out.ai2_motion_ref_mps        = getf("HoodModel.AI2MotionRefMps",        0.60f);
    out.ai2_motion_confirm_only   = getb("HoodModel.AI2MotionConfirmOnly",   true);
    out.ai2_still_lin_mps         = getf("HoodModel.AI2StillLinMps",         0.05f);
    out.ai2_still_ang_radps       = getf("HoodModel.AI2StillAngRadps",       0.10f);
    out.ai2_still_dotd            = getf("HoodModel.AI2StillDotd",           0.05f);
    out.ai2_moving_update_center_radius = getf("HoodModel.AI2MovingUpdateCenterRadius", 0.35f);
    out.ai2_range_noise_lat_per_m = getf("HoodModel.AI2RangeNoiseLatPerM", 0.02f);
    out.ai2_range_noise_yaw_per_m = getf("HoodModel.AI2RangeNoiseYawPerM", 0.03f);
    out.ai2_range_noise_size_per_m = getf("HoodModel.AI2RangeNoiseSizePerM", 0.08f);
    out.ai2_trunc_gate_frac    = getf("HoodModel.AI2TruncGateFrac",   0.10f);
    out.ai2_gn_iters         = geti("HoodModel.AI2GnIters",          4);
    out.ai2_csv_path         = gets("HoodModel.AI2CsvPath",          "");
    out.birth_surprise_probe = getb("HoodModel.BirthSurpriseProbe",  false);
    out.pixel_sigma_over_f     = getf("HoodModel.PixelSigmaOverF",       0.0015f);
    out.depth_sigma0_m         = getf("HoodModel.DepthSigma0M",          0.006f);
    out.depth_sigma_range_coef = getf("HoodModel.DepthSigmaRangeCoef",   0.004f);
    out.model_sigma_m          = getf("HoodModel.ModelSigmaM",           0.010f);
    out.footprint_residual     = getb("HoodModel.FootprintResidual",     false);
    out.quotient_chart         = getb("HoodModel.QuotientChart",          false);
    out.depth_tilt_std         = getf("HoodModel.DepthTiltStd",          0.020f);
    out.depth_bias_std         = getf("HoodModel.DepthBiasStd",          0.015f);
    out.depth_scale_std        = getf("HoodModel.DepthScaleStd",         0.010f);

    // ─── "Is this really a fridge?" plausibility filter + soft singleton ───────
    out.fridge_filter_enabled   = getb("HoodConcept.FridgeFilterEnabled",   true);
    out.plaus_aspect_scale      = getf("HoodModel.AspectScale",             0.15f);
    out.plaus_size_scale        = getf("HoodModel.SizeScale",               0.15f);
    out.plaus_alt_size_scale    = getf("HoodModel.AltSizeScale",            0.60f);
    out.plaus_height_min        = getf("HoodModel.HeightPlausibleMin",      1.20f);
    out.plaus_height_soft       = getf("HoodModel.HeightSoft",              0.15f);
    out.plaus_fe_ref            = getf("HoodModel.FeRef",                    2.0f);
    out.plaus_fe_scale          = getf("HoodModel.FeScale",                 1.0f);
    out.plaus_clamp             = getf("HoodModel.PlausClamp",              8.0f);
    out.plaus_height_prior_gain = getf("HoodModel.PlausHeightPriorGain",    2000.0f);
    out.plaus_to_existence_gain = getf("HoodModel.PlausToExistenceGain",    1.5f);
    out.singleton_inhibition    = getf("HoodModel.SingletonInhibition",     1.0f);
    out.fridge_filter_log       = getb("HoodConcept.FridgeFilterLog",       false);

    // ─── RT-edge covariance upload ─────────────────────────────────────────────
    out.rt_cov_scale                  = getf("HoodConcept.RtCovScale",           1.0f);
    out.publish_object_obs            = getb("HoodConcept.PublishObjectObs",   false);
    out.object_obs_frame              = gets("HoodConcept.ObjectObsFrame",     "body");

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
    // Birth fragment: keep the probation burst and admit the birth on it (see hood_config.h).
    out.birth_frag_enabled       = getb("Tracker.BirthFragment",          true);
    out.birth_frag_voxel_m       = getf("Tracker.BirthFragmentVoxelM",    0.03f);
    out.birth_frag_max_pts       = geti("Tracker.BirthFragmentMaxPts",    20000);
    out.birth_frag_delta_ms      = static_cast<std::uint64_t>(
                                       std::max(0, geti("Tracker.BirthFragmentDeltaMs", 4000)));
    out.birth_admit_plausibility = getf("Tracker.BirthAdmitPlausibility", 0.35f);
    out.ricoh_attention_conf     = getf("Tracker.RicohAttentionConf", 0.60f);
    out.ricoh_attention_angle_margin_rad = getf("Tracker.RicohAttentionAngleMargin", 0.05f);
    out.ricoh_attention_range_band_m     = getf("Tracker.RicohAttentionRangeBandM",  1.0f);

    // ─── LiDAR range factor · coverage · free-space · footprint moment · FE ────
    // YOLO-independent LiDAR first-hit range factor (common/ai_belief/lidar_ray_factor.h). OFF by default.
    out.lidar_precision      = getf("HoodModel.LidarPrecision",      0.0f);
    out.lidar_bpearl_precision = getf("HoodModel.LidarBpearlPrecision", 0.0f);
    out.lidar_robust_c_m     = getf("HoodModel.LidarRobustCM",       0.05f);
    out.lidar_select_margin_m = getf("HoodModel.LidarSelectMarginM", 0.10f);
    out.lidar_coverage_n0     = getf("HoodModel.LidarCoverageN0",     60.0f);
    out.lidar_coverage_ang_power = getf("HoodModel.LidarCoverageAngPower", 1.0f);
    out.max_step_m            = getf("HoodModel.MaxStepM",            1.0f);
    out.coverage_precision    = getf("HoodModel.CoveragePrecision",  0.0f);
    out.coverage_robust_c_m   = getf("HoodModel.CoverageRobustCM",   0.15f);
    out.free_space_precision  = getf("HoodModel.FreeSpacePrecision", 0.0f);
    out.footprint_moment_precision = getf("HoodModel.FootprintMomentPrecision", 0.0f);
    out.footprint_moment_range_per_m = getf("HoodModel.FootprintMomentRangePerM", 0.03f);
    out.fe_baseline_adapt_down       = getf("HoodModel.FeBaselineAdaptDown", 0.05f);
    out.fe_baseline_adapt_up         = getf("HoodModel.FeBaselineAdaptUp",   0.005f);
    out.fe_surprise_smooth           = getf("HoodModel.FeSurpriseSmooth",    0.10f);
    out.footprint_moment_motion_gain = getf("HoodModel.FootprintMomentMotionGain", 0.30f);
    out.orientation_motion_ref       = getf("HoodModel.OrientationMotionRef", 0.50f);

    // ─── Appearance-based FRONT (door) detection + yaw resolver ────────────────
    out.front_detect_enabled   = getb("HoodConcept.FrontDetectEnabled",   true);
    out.front_min_face_area_px = getf("HoodConcept.FrontMinFaceAreaPx",    900.0f);
    out.front_min_confidence   = getf("HoodConcept.FrontMinConfidence",    0.10f);
    out.front_log              = getb("HoodConcept.FrontLog",              false);
    out.obliquity_moment_gain        = getf("HoodModel.ObliquityMomentGain", 0.0f);
    out.footprint_moment_completeness_gain = getf("HoodModel.FootprintMomentCompletenessGain", 0.0f);
    out.footprint_moment_min_completeness  = getf("HoodModel.FootprintMomentMinCompleteness",  0.02f);

    // ─── Existence / removal ───────────────────────────────────────────────────
    out.existence_removal_enabled = getb("HoodModel.ExistenceRemovalEnabled", false);
    out.existence_removal_prob    = getf("HoodModel.ExistenceRemovalProb",    0.12f);
    out.existence_frame_correlation = getf("HoodModel.ExistenceFrameCorrelation", 0.0f);
    out.existence_logodds_max     = getf("HoodModel.ExistenceLogoddsMax",     4.0f);
    out.existence_detection_prob  = getf("HoodModel.ExistenceDetectionProb",  0.85f);
    out.existence_clutter_prob    = getf("HoodModel.ExistenceClutterProb",    0.05f);
    out.existence_sensor_sigma_m  = getf("HoodModel.ExistenceSensorSigmaM",   0.03f);
    out.existence_remove_frames   = geti("HoodModel.ExistenceRemoveFrames",   15);
    out.existence_absence_range_ref_m = getf("HoodModel.ExistenceAbsenceRangeRefM", 2.5f);
    out.existence_absence_range_power = getf("HoodModel.ExistenceAbsenceRangePower", 2.0f);
    out.existence_verify_surprise     = getf("HoodModel.ExistenceVerifySurprise",   20.0f);
    out.verify_surprise_smooth        = getf("HoodModel.VerifySurpriseSmooth",       0.10f);
    out.existence_verify_gain         = getf("HoodModel.ExistenceVerifyGain",       5.0f);

    std::print("hood_concept: configuration loaded.\n");
    return out;
}

// ─── Concept-manifest cross-check (declarative-priors experiment, step 1½) ────────────────────────
//
// common/concept_manifest/<concept>.concept.toml declares WHAT a hood IS — its priors as world facts,
// separate from the lifecycle knobs. It is not authoritative yet. This compares it against the priors the live
// etc/config.toml actually produced, so we learn whether a manifest can reproduce the running agent BEFORE
// anything is generated from it. Startup-only, read-only: it never changes a value, it only reports.
//
// A DIFFERS line is a finding, not a failure — it means the manifest and the running config disagree about a
// world fact, and one of them is wrong. A MISSING line means the manifest does not yet describe that prior.
bool verify_hood_manifest(const HoodConfig& out, const std::string& path)
{
    ConfigLoader man;
    try { man.load(path); }
    catch (...) { std::print("[manifest] not loaded ({}) — cross-check skipped\n", path); return false; }

    int agree = 0, differ = 0, missing = 0;
    const auto chk = [&](const char* key, float live, const char* what) {
        if (not man.exists(key)) { ++missing;
            std::print("[manifest] MISSING  {:<42} live={:<10.4g} ({})\n", key, live, what); return; }
        const float m = static_cast<float>(man.get<double>(key));
        const float tol = 1e-4f * std::max(1.0f, std::abs(live));
        if (std::abs(m - live) <= tol) { ++agree; }
        else { ++differ;
            std::print("[manifest] DIFFERS  {:<42} manifest={:<10.4g} live={:<10.4g} ({})\n", key, m, live, what); }
    };

    chk("prior.footprint.mean_m",              out.ai2_prior_footprint_m,      "fridge footprint mean");
    chk("prior.footprint.std_m",               out.ai2_prior_footprint_std,    "footprint prior std");
    chk("prior.height.mean_m",                 out.ai2_prior_height_m,         "height anchor mean");
    chk("prior.height.std_m",                  out.ai2_prior_height_std,       "height anchor std");
    chk("prior.depth_observability.precision", out.ai2_depth_unobs_precision,  "depth-unobserved precision");
    chk("prior.depth_observability.observed_band_m", out.ai2_depth_obs_band_m, "depth observed band");
    chk("prior.top.precision",                 out.ai2_top_no_float_precision, "top no-float anchor");
    chk("prior.top.margin_m",                  out.ai2_top_no_float_margin_m,  "top anchor margin");
    chk("prior.attachment.precision",          out.ai2_wall_precision,         "wall flush");
    chk("prior.attachment.parallel_precision", out.ai2_wall_parallel_precision,"wall parallel");
    chk("prior.attachment.reach_m",            out.ai2_wall_reach_m,           "flush reach");
    chk("prior.attachment.no_cross_precision", out.ai2_wall_no_cross_precision,"wall no-cross");
    chk("prior.identity.aspect_scale",         out.plaus_aspect_scale,         "identity aspect");
    chk("prior.identity.size_scale",           out.plaus_size_scale,           "identity size");
    chk("prior.identity.alt_size_scale",       out.plaus_alt_size_scale,       "identity alternative");
    chk("prior.identity.height_min_m",         out.plaus_height_min,           "identity height centre");
    chk("prior.identity.height_soft_m",        out.plaus_height_soft,          "identity height softness");
    chk("prior.clearance.gain_nats",           out.ai2_door_clearance_gain,    "door clearance prior");
    chk("prior.explaining_away.weight",        out.ai2_wall_explain_frac,      "wall explain-away weight");
    chk("prior.explaining_away.sigma_m",       out.ai2_wall_explain_sigma_m,   "wall explain-away sigma");
    chk("cue.door_seam.min_face_area_px",      out.front_min_face_area_px,     "door cue min face area");
    chk("cue.door_seam.min_confidence",        out.front_min_confidence,       "door cue min confidence");

    std::print("[manifest] {} — {} agree, {} DIFFER, {} missing\n",
               (differ == 0 and missing == 0) ? "reproduces the live config" : "does NOT yet reproduce the live config",
               agree, differ, missing);
    return differ == 0 and missing == 0;
}

}  // namespace rc
