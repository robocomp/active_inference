/*
 * door_config.cpp — fill DoorConfig from a RoboComp ConfigLoader.
 */

#include "door_config.h"

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

DoorConfig load_door_config(const ConfigLoader& cfg)
{
    DoorConfig out;


    // The one place this path is written. Relative to the agent's CWD (<agent>/), not to src/ — the same
    // string was right in one place and wrong in the other for a week, and the manifest was inert the whole time.
    static constexpr const char* kManifestPath = "../common/concept_manifest/door.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL — rc::manifest::provenance_ok. `from = "inherited"` means the number
    // arrived by a rename and nobody chose it for THIS object; hood shipped ten such defects in a week with
    // several of them declared, in writing, in its manifest. A note stopped nothing, so this stops the agent.
    if (not rc::manifest::provenance_ok(kManifestPath, "door"))
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

    // Agent convergence
    out.state_eps                = getf("DoorConcept.StateEps",               0.04f);
    out.K_stable                 = geti("DoorConcept.KStable",                30);
    out.detection_alive_max_frames = geti("DoorConcept.DetectionAliveMaxFrames", 40);
    out.obs_distance             = getf("DoorConcept.ObsDistance",            1.8f);
    out.min_standoff_m           = getf("DoorConcept.MinStandOffM",           1.8f);
    out.epistemic_cooldown_cycles= geti("DoorConcept.EpistemicCooldownCycles", 200);
    out.door_log_period_frames  = geti("DoorConcept.DoorLogPeriodFrames",   30);
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs",           3000);
    out.support_bank_max_points    = geti("DoorConcept.SupportBankMaxPoints",     4000);
    out.support_bank_quantization_m= getf("DoorConcept.SupportBankQuantizationM", 0.02f);
    out.support_select_radius_margin_m = getf("DoorConcept.SupportSelectRadiusMarginM", 0.50f);
    out.support_select_height_margin_m = getf("DoorConcept.SupportSelectHeightMarginM", 0.25f);

    // DoorModel geometry / mask split
    out.sigma_obs          = getf("DoorModel.SigmaObs",          0.05f);
    out.sdf_threshold_for_storage = getf("DoorModel.SdfThresholdForStorage", 0.08f);

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m         = getf("DoorModel.AI2SigmaBaseM",           0.03f);
    out.detect_min_fill          = getf("DoorModel.DetectMinFill",           0.10f);
    out.detect_max_fill          = getf("DoorModel.DetectMaxFill",           0.60f);
    out.detect_soft              = getf("DoorModel.DetectSoft",              0.06f);
    out.ai2_clutter_frac         = getf("DoorModel.AI2ClutterFrac",          0.10f);
    out.ai2_clutter_scale_m      = getf("DoorModel.AI2ClutterScaleM",        0.12f);
    out.ai2_clutter_structure_gain = getf("DoorModel.AI2ClutterStructureGain", 1.0f);
    out.ai2_prior_size_std       = getf("DoorModel.AI2PriorSizeStd",         0.15f);
    out.ai2_process_std_m        = getf("DoorModel.AI2ProcessStdM",          0.005f);
    out.ai2_process_std_yaw      = getf("DoorModel.AI2ProcessStdYaw",        0.01f);
    out.ai2_process_std_size     = getf("DoorModel.AI2ProcessStdSize",       0.0005f);
    out.ai2_age_nominal_dt_s     = getf("DoorModel.AI2AgeNominalDtS",        0.0f);
    out.ai2_floor_z              = getf("DoorModel.AI2FloorZ",               0.0f);
    out.ai2_floor_std            = getf("DoorModel.AI2FloorStd",             0.03f);
    out.ai2_seat_anchor_std      = getf("DoorModel.AI2SeatAnchorStd",        0.04f);
    out.ai2_seat_anchor_band     = getf("DoorModel.AI2SeatAnchorBand",       0.12f);
    out.ai2_seat_extent_std      = getf("DoorModel.AI2SeatExtentStd",        0.02f);
    out.ai2_common_mode_pos_std  = getf("DoorModel.AI2CommonModePosStd",     0.03f);
    out.ai2_common_mode_size_std = getf("DoorModel.AI2CommonModeSizeStd",    0.35f);
    out.ai2_common_mode_yaw_std  = getf("DoorModel.AI2CommonModeYawStd",     0.03f);
    out.ai2_range_noise_lat_per_m = getf("DoorModel.AI2RangeNoiseLatPerM",   0.02f);
    out.ai2_range_noise_yaw_per_m = getf("DoorModel.AI2RangeNoiseYawPerM",   0.03f);
    out.motion_cm_pos_gain        = getf("DoorModel.MotionCmPosGain",        0.10f);
    out.motion_cm_yaw_gain        = getf("DoorModel.MotionCmYawGain",        0.12f);
    out.ai2_ang_lever_m           = getf("DoorModel.AI2AngLeverM",           2.0f);
    out.ai2_periph_ref            = getf("DoorModel.AI2PeriphRef",           0.50f);
    out.ai2_motion_ref_mps        = getf("DoorModel.AI2MotionRefMps",        0.60f);
    out.ai2_motion_confirm_only   = getb("DoorModel.AI2MotionConfirmOnly",   false);
    out.ai2_still_lin_mps         = getf("DoorModel.AI2StillLinMps",         0.05f);
    out.ai2_still_ang_radps       = getf("DoorModel.AI2StillAngRadps",       0.10f);
    out.ai2_still_dotd            = getf("DoorModel.AI2StillDotd",           0.05f);
    out.ai2_moving_update_center_radius = getf("DoorModel.AI2MovingUpdateCenterRadius", 0.35f);
    out.ai2_obliquity_yaw_gain    = getf("DoorModel.AI2ObliquityYawGain",    0.05f);
    out.ai2_orientation_motion_ref = getf("DoorModel.AI2OrientationMotionRef", 0.50f);
    out.ai2_fe_baseline_adapt_down = getf("DoorModel.AI2FeBaselineAdaptDown", 0.05f);
    out.ai2_fe_baseline_adapt_up   = getf("DoorModel.AI2FeBaselineAdaptUp",   0.005f);
    out.ai2_fe_surprise_smooth     = getf("DoorModel.AI2FeSurpriseSmooth",    0.10f);
    out.ai2_trunc_gate_frac      = getf("DoorModel.AI2TruncGateFrac",        0.10f);
    out.ai2_gn_iters             = geti("DoorModel.AI2GnIters",              4);
    out.ai2_extent_std           = getf("DoorModel.AI2ExtentStd",            0.05f);
    out.ai2_csv_path             = gets("DoorModel.AI2CsvPath",              "");

    out.rt_cov_upload                 = getb("DoorConcept.RtCovUpload",         true);
    out.rt_cov_scale                  = getf("DoorConcept.RtCovScale",          1.0f);
    out.rt_cov_add_chain              = getb("DoorConcept.RtCovAddChain",       true);

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
    out.exist_enabled            = getb("Existence.Enabled",          true);
    out.exist_birth_logodds      = getf("Existence.BirthLogodds",     1.0f);
    out.exist_max_logodds        = getf("Existence.MaxLogodds",       4.0f);
    out.exist_removal_prob       = getf("Existence.RemovalProb",      0.12f);
    out.exist_remove_frames      = geti("Existence.RemoveFrames",     15);
    out.exist_detection_prob     = getf("Existence.DetectionProb",    0.85f);
    out.exist_clutter_prob       = getf("Existence.ClutterProb",      0.05f);
    out.exist_sensor_sigma_m     = getf("Existence.SensorSigmaM",     0.03f);
    out.exist_central_region_frac = getf("Existence.CentralRegionFrac", 0.25f);
    out.exist_occlusion_margin_m = getf("Existence.OcclusionMarginM", 0.30f);
    out.exist_room_prior         = getb("Existence.RoomPrior",        true);
    out.exist_room_margin_m      = getf("Existence.RoomMarginM",      0.40f);
    out.exist_out_of_room_gain   = getf("Existence.OutOfRoomGain",    1.5f);
    out.exist_min_height_prior   = getb("Existence.MinHeightPrior",   true);
    out.exist_min_height_m       = getf("Existence.MinHeightM",       1.80f);
    out.exist_min_height_conf    = getf("Existence.MinHeightConf",    0.30f);
    out.exist_short_gain         = getf("Existence.ShortGain",        1.5f);
    out.exist_reacquire_radius_m = getf("Existence.ReacquireRadiusM", 0.60f);
    out.exist_ghost_max          = geti("Existence.GhostMax",         8);
    // [Openable] — deliberately ABSENT from etc/config.toml (same as [Bearing]): a dormant, opt-in feature
    // whose defaults keep the leaf pinned flush. See the block comment in door_config.h.
    out.openable_enabled     = getb("Openable.Enabled",     false);
    out.openable_phi_init    = getf("Openable.PhiInitRad",  0.0f);
    out.openable_hinge_side  = geti("Openable.HingeSide",   0);
    out.openable_swing_dir   = getf("Openable.SwingDir",    1.0f);
    out.openable_phi_max_rad = getf("Openable.PhiMaxRad",   1.5707963f);
    out.door_prior_w_m           = getf("Door.PriorWidthM",         0.70f);
    out.door_prior_w_std         = getf("Door.PriorWidthStd",       0.06f);
    out.door_prior_h_m           = getf("Door.PriorHeightM",        2.00f);
    out.door_prior_h_std         = getf("Door.PriorHeightStd",      0.08f);
    out.door_prior_s_std         = getf("Door.PriorAlongWallStd",   0.60f);
    out.door_thickness_m         = getf("Door.ThicknessM",          0.05f);
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

    std::print("door_concept: configuration loaded.\n");
    return out;
}

}  // namespace rc
