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
    out.voxel_bank_max_points    = geti("ChairConcept.VoxelBankMaxPoints",     4000);
    out.voxel_bank_quantization_m= getf("ChairConcept.VoxelBankQuantizationM", 0.02f);
    out.voxel_select_radius_margin_m = getf("ChairConcept.VoxelSelectRadiusMarginM", 0.50f);
    out.voxel_select_height_margin_m = getf("ChairConcept.VoxelSelectHeightMarginM", 0.25f);

    // ChairModel geometry / mask split
    out.sigma_obs          = getf("ChairModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("ChairModel.SdfThresholdForStorage", 0.08f);

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m         = getf("ChairModel.AI2SigmaBaseM",           0.03f);
    out.ai2_clutter_frac         = getf("ChairModel.AI2ClutterFrac",          0.10f);
    out.ai2_clutter_scale_m      = getf("ChairModel.AI2ClutterScaleM",        0.12f);
    out.ai2_clutter_structure_gain = getf("ChairModel.AI2ClutterStructureGain", 1.0f);
    out.ai2_prior_size_std       = getf("ChairModel.AI2PriorSizeStd",         0.15f);
    out.ai2_process_std_m        = getf("ChairModel.AI2ProcessStdM",          0.005f);
    out.ai2_process_std_yaw      = getf("ChairModel.AI2ProcessStdYaw",        0.01f);
    out.ai2_process_std_size     = getf("ChairModel.AI2ProcessStdSize",       0.0005f);
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
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap",     0.20f);
    out.tracker_prune_enabled         = getb("Tracker.PruneEnabled",        true);
    out.tracker_prune_maturity_cycles = geti("Tracker.PruneMaturityCycles", 90);
    out.tracker_prune_patience        = geti("Tracker.PrunePatience",       30);
    out.tracker_birth_seat_w     = getf("Tracker.BirthSeatW",       0.45f);
    out.tracker_birth_seat_d     = getf("Tracker.BirthSeatD",       0.45f);
    out.tracker_birth_seat_h     = getf("Tracker.BirthSeatH",       0.45f);
    out.tracker_birth_back_h     = getf("Tracker.BirthBackH",       0.45f);
    out.tracker_nll_cost         = getb("Tracker.NllCost",          false);
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
