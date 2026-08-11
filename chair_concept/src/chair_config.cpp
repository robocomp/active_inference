/*
 * chair_config.cpp — fill ChairConfig from a RoboComp ConfigLoader.
 */

#include "chair_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

ChairConfig load_chair_config(const ConfigLoader& cfg)
{
    ChairConfig out;

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
    out.state_eps                = getf("ChairConcept.StateEps",               0.04f);
    out.K_stable                 = geti("ChairConcept.KStable",                30);
    out.detection_alive_max_frames = geti("ChairConcept.DetectionAliveMaxFrames", 40);
    out.obs_distance             = getf("ChairConcept.ObsDistance",            1.8f);
    out.min_standoff_m           = getf("ChairConcept.MinStandOffM",           1.8f);
    out.epistemic_cooldown_cycles= geti("ChairConcept.EpistemicCooldownCycles", 200);
    out.chair_log_period_frames  = geti("ChairConcept.ChairLogPeriodFrames",   30);
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.voxel_bank_max_points    = geti("ChairConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("ChairConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("ChairConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("ChairConcept.VoxelSelectHeightMarginM", 0.25f);

    // ChairModel geometry / mask split
    out.sigma_obs          = getf("ChairModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("ChairModel.SdfThresholdForStorage", 0.08f);

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m         = getf("ChairModel.AI2SigmaBaseM",           0.03f);
    out.detect_min_fill          = getf("ChairModel.DetectMinFill",           0.10f);
    out.detect_max_fill          = getf("ChairModel.DetectMaxFill",           0.60f);
    out.detect_soft              = getf("ChairModel.DetectSoft",              0.06f);
    out.ai2_clutter_frac         = getf("ChairModel.AI2ClutterFrac",          0.10f);
    out.ai2_clutter_scale_m      = getf("ChairModel.AI2ClutterScaleM",        0.12f);
    out.ai2_clutter_structure_gain = getf("ChairModel.AI2ClutterStructureGain", 1.0f);
    out.ai2_prior_size_std       = getf("ChairModel.AI2PriorSizeStd",         0.15f);
    out.ai2_process_std_m        = getf("ChairModel.AI2ProcessStdM",          0.005f);
    out.ai2_process_std_yaw      = getf("ChairModel.AI2ProcessStdYaw",        0.01f);
    out.ai2_process_std_size     = getf("ChairModel.AI2ProcessStdSize",       0.0005f);
    out.ai2_age_nominal_dt_s     = getf("ChairModel.AI2AgeNominalDtS",        0.0f);
    out.ai2_floor_z              = getf("ChairModel.AI2FloorZ",               0.0f);
    out.ai2_floor_std            = getf("ChairModel.AI2FloorStd",             0.03f);
    out.ai2_seat_anchor_std      = getf("ChairModel.AI2SeatAnchorStd",        0.04f);
    out.ai2_seat_anchor_band     = getf("ChairModel.AI2SeatAnchorBand",       0.12f);
    out.ai2_seat_extent_std      = getf("ChairModel.AI2SeatExtentStd",        0.02f);
    out.ai2_common_mode_pos_std  = getf("ChairModel.AI2CommonModePosStd",     0.03f);
    out.ai2_common_mode_size_std = getf("ChairModel.AI2CommonModeSizeStd",    0.02f);
    out.ai2_common_mode_yaw_std  = getf("ChairModel.AI2CommonModeYawStd",     0.03f);
    out.ai2_range_noise_lat_per_m = getf("ChairModel.AI2RangeNoiseLatPerM",   0.02f);
    out.ai2_range_noise_yaw_per_m = getf("ChairModel.AI2RangeNoiseYawPerM",   0.03f);
    out.motion_cm_pos_gain        = getf("ChairModel.MotionCmPosGain",        0.10f);
    out.motion_cm_yaw_gain        = getf("ChairModel.MotionCmYawGain",        0.12f);
    out.ai2_ang_lever_m           = getf("ChairModel.AI2AngLeverM",           2.0f);
    out.ai2_periph_ref            = getf("ChairModel.AI2PeriphRef",           0.50f);
    out.ai2_motion_ref_mps        = getf("ChairModel.AI2MotionRefMps",        0.60f);
    out.ai2_motion_confirm_only   = getb("ChairModel.AI2MotionConfirmOnly",   false);
    out.fixation_enabled          = getb("ChairModel.FixationEnabled",         true);
    out.fixation_min_pts          = geti("ChairModel.FixationMinPts",          150);
    out.fixation_max_clutter      = getf("ChairModel.FixationMaxClutter",      0.35f);
    out.fixation_range_m          = getf("ChairModel.FixationRangeM",          0.0f);   // retired as a gate
    out.fixation_centre_frac      = getf("ChairModel.FixationCentreFrac",      0.60f);
    out.fixation_centre_precision = getb("ChairModel.FixationCentrePrecision", false);
    out.fixation_periph_floor     = getf("ChairModel.FixationPeriphFloor",     0.05f);
    out.fixation_still_dotd       = getf("ChairModel.FixationStillDotd",       0.05f);
    out.fixation_still_lin_mps    = getf("ChairModel.FixationStillLinMps",     0.05f);
    out.fixation_still_ang_radps  = getf("ChairModel.FixationStillAngRadps",   0.10f);
    out.ai2_still_lin_mps         = getf("ChairModel.AI2StillLinMps",         0.05f);
    out.ai2_still_ang_radps       = getf("ChairModel.AI2StillAngRadps",       0.10f);
    out.ai2_still_dotd            = getf("ChairModel.AI2StillDotd",           0.05f);
    out.ai2_moving_update_center_radius = getf("ChairModel.AI2MovingUpdateCenterRadius", 0.35f);
    out.ai2_obliquity_yaw_gain    = getf("ChairModel.AI2ObliquityYawGain",    0.05f);
    out.ai2_orientation_motion_ref = getf("ChairModel.AI2OrientationMotionRef", 0.50f);
    out.ai2_fe_baseline_adapt_down = getf("ChairModel.AI2FeBaselineAdaptDown", 0.05f);
    out.ai2_fe_baseline_adapt_up   = getf("ChairModel.AI2FeBaselineAdaptUp",   0.005f);
    out.ai2_fe_surprise_smooth     = getf("ChairModel.AI2FeSurpriseSmooth",    0.10f);
    out.ai2_trunc_gate_frac      = getf("ChairModel.AI2TruncGateFrac",        0.10f);
    out.ai2_gn_iters             = geti("ChairModel.AI2GnIters",              4);
    out.ai2_extent_std           = getf("ChairModel.AI2ExtentStd",            0.05f);
    out.ai2_csv_path             = gets("ChairModel.AI2CsvPath",              "");

    out.rt_cov_upload                 = getb("ChairConcept.RtCovUpload",         true);
    out.rt_cov_scale                  = getf("ChairConcept.RtCovScale",          1.0f);
    out.rt_cov_add_chain              = getb("ChairConcept.RtCovAddChain",       true);

    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis",  9.0f);
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM",    0.40f);
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.20f);
    out.tracker_birth_frames     = geti("Tracker.BirthFrames",      8);
    out.tracker_death_frames     = geti("Tracker.DeathFrames",      300);
    out.tracker_death_enabled    = getb("Tracker.DeathEnabled",     false);
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM",     0.70f);
    out.birth_fragment_frac    = getf("Tracker.BirthFragmentFrac",   0.25f);
    out.birth_fragment_reach_m = getf("Tracker.BirthFragmentReachM", 1.00f);
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.20f);
    out.tracker_prune_enabled         = getb("Tracker.PruneEnabled",        true);
    out.tracker_prune_maturity_cycles = geti("Tracker.PruneMaturityCycles", 90);
    out.tracker_prune_patience        = geti("Tracker.PrunePatience",       30);
    out.exist_enabled            = getb("Existence.Enabled",          true);

    // Level-2 arrangement prior (ring_metaconcept). Off ⇒ the belief is bit-for-bit pre-rig.
    out.rig_yaw_prior_enabled    = getb("RigPrior.Enabled",           true);
    out.rig_yaw_kappa_max        = getf("RigPrior.YawKappaMax",       1.5f);
    out.rig_prior_stale_ms       = geti("RigPrior.StaleMs",           5000);
    out.exist_birth_logodds      = getf("Existence.BirthLogodds",     1.0f);
    out.exist_remove_logodds     = getf("Existence.RemoveLogodds",   -3.0f);
    out.exist_max_logodds        = getf("Existence.MaxLogodds",       4.0f);
    out.exist_evidence_gain      = getf("Existence.EvidenceGain",     0.15f);
    out.exist_expected_support_c = getf("Existence.ExpectedSupportC", 7000.f);
    out.exist_adequacy_ref       = getf("Existence.AdequacyRef",      0.30f);
    out.exist_adequacy_cap       = getf("Existence.AdequacyCap",      1.5f);
    out.exist_calib_adapt        = getf("Existence.CalibAdapt",       0.02f);
    out.exist_vacate_confident_frames = geti("Existence.VacateConfidentFrames", 45);
    out.exist_occlusion_check    = getb("Existence.OcclusionCheck",   true);
    out.exist_occlusion_margin_m = getf("Existence.OcclusionMarginM", 0.30f);
    out.exist_room_prior         = getb("Existence.RoomPrior",        true);
    out.exist_room_margin_m      = getf("Existence.RoomMarginM",      0.40f);
    out.exist_out_of_room_gain   = getf("Existence.OutOfRoomGain",    1.5f);
    out.exist_zed_edge_offset    = getf("Existence.ZedEdgeOffset",    1.0f);
    out.exist_zed_range_full     = getf("Existence.ZedRangeFull",     4.0f);
    out.exist_zed_range_ref      = getf("Existence.ZedRangeRef",      7.0f);
    out.exist_zed_clear_los_floor = getf("Existence.ZedClearLosFloor", 0.0f);   // 0 = pure pd (see chair_config.h)
    out.tracker_birth_seat_w     = getf("Tracker.BirthSeatW",       0.60f);
    out.tracker_birth_seat_d     = getf("Tracker.BirthSeatD",       0.52f);
    out.tracker_birth_seat_h     = getf("Tracker.BirthSeatH",       0.595f);
    out.tracker_birth_back_h     = getf("Tracker.BirthBackH",       0.655f);
    out.tracker_birth_seat_thick = getf("Tracker.BirthSeatThick",   0.075f);
    out.tracker_birth_leg_half   = getf("Tracker.BirthLegHalf",     0.0375f);
    out.ai2_mode_obs_weighting   = getb("ChairModel.AI2ModeObsWeighting", true);
    out.ai2_mode_sat_back_pts    = getf("ChairModel.AI2ModeSatBackPts",   60.0f);
    out.ai2_view_budget          = getf("ChairModel.AI2ViewBudget",        3.0f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",          false);
    out.ricoh_birth_enabled      = getb("Tracker.RicohBirthEnabled", false);
    out.ricoh_birth_conf         = getf("Tracker.RicohBirthConf",    0.60f);
    out.ricoh_birth_max_var      = getf("Tracker.RicohBirthMaxVar",  0.005f);
    out.bearing_birth_enabled    = getb("Bearing.BirthEnabled",     false);
    out.bearing_confirm_gate_rad = getf("Bearing.ConfirmGateRad",   0.17f);
    out.bearing_birth_frames     = geti("Bearing.BirthFrames",      8);
    out.bearing_match_rad        = getf("Bearing.MatchRad",         0.17f);
    out.bearing_max_miss         = geti("Bearing.MaxMiss",          4);
    out.bearing_nominal_range_m  = getf("Bearing.NominalRangeM",    2.0f);
    out.bearing_along_std_m      = getf("Bearing.AlongStdM",        3.0f);
    out.bearing_across_std_m     = getf("Bearing.AcrossStdM",       0.30f);
    out.bearing_yaw_std_rad      = getf("Bearing.YawStdRad",        3.14f);

    std::print("chair_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
