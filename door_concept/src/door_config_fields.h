// door_config_fields.h — EVERY DoorConfig key read, as one template over the four getters (getf/geti/gets/getb).
//
// ★WHY A TEMPLATE (door_concept DoorCore, 2026-10-21). load_door_config (door_config.cpp) reads the agent's config
// through RoboComp's ConfigLoader. The replay tool (tools/door_replay.cpp) must rebuild the SAME DoorConfig from a
// recording's header, which carries etc/config_effective.csv — and it is a standalone g++ build with no ConfigLoader.
// A second key→field table in the tool would drift from this one the first time a key is added; so the key→field
// mapping lives here ONCE and both readers instantiate it with their own getters. Every getter is called as
// get(key, code_default, description[, rc::cfg::Opts]) and returns the effective value.
// Pure: no DSR, no Qt, no ConfigLoader (the Opts argument, when a read passes one, is the caller's type).
#pragma once
#include <string>
#include "door_config.h"

namespace rc
{

template <class GetF, class GetI, class GetS, class GetB>
void read_door_config_fields(DoorConfig& out, GetF&& getf, GetI&& geti, GetS&& gets, GetB&& getb)
{
    // Agent convergence
    out.state_eps                = getf("DoorConcept.StateEps", 0.04f,
            "Σ|Δstate| threshold between cycles for convergence (m+rad)");
    out.K_stable                 = geti("DoorConcept.KStable", 30,
            "consecutive stable frames before model_stable");
    out.detection_alive_max_frames = geti("DoorConcept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh door mask before detection_alive=false");
    out.obs_distance             = getf("DoorConcept.ObsDistance", 1.8f,
            "d_obs for epistemic planner");
    out.min_standoff_m           = getf("DoorConcept.MinStandOffM", 1.8f,
            "min stand-off floor for epistemic viewpoints (YOLO misses too-close doors)");
    out.epistemic_cooldown_cycles= geti("DoorConcept.EpistemicCooldownCycles", 200,
            "min cycles withdrawn after satisfaction");
    out.door_log_period_frames  = geti("DoorConcept.DoorLogPeriodFrames", 30,
            "per-door log throttle");
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "Primary-input (masks) stream gate: no NEW masks frame for this many ms while Operating ⇒ demote out of Operating rather than integrate stale…");
    out.support_bank_max_points    = geti("DoorConcept.SupportBankMaxPoints", 4000,
            "max number of door-owned support points kept in memory");
    out.support_bank_quantization_m= getf("DoorConcept.SupportBankQuantizationM", 0.02f,
            "dedup grid (m) for support-bank hashing");
    out.support_select_radius_margin_m = getf("DoorConcept.SupportSelectRadiusMarginM", 0.50f,
            "per-door XY ownership gate margin for mixed proto-doors");
    out.support_select_height_margin_m = getf("DoorConcept.SupportSelectHeightMarginM", 0.25f,
            "extra Z margin above door height when selecting owned points");

    // DoorModel geometry / mask split
    out.sigma_obs          = getf("DoorModel.SigmaObs", 0.05f,
            "DoorModel geometry / mask split: on-surface membership for the candidate/residual split in DoorFitter::observe (a mask point within…");
    out.sdf_threshold_for_storage = getf("DoorModel.SdfThresholdForStorage", 0.08f,
            "on-surface membership for the candidate/residual split");

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m         = getf("DoorModel.AI2SigmaBaseM", 0.03f,
            "");
    out.detect_min_fill          = getf("DoorModel.DetectMinFill", 0.10f,
            "ONE model, two consumers: the epistemic planner puts the stand-off at its argmax, and the removal channel weights absence by it, so a missing mask…");
    out.detect_max_fill          = getf("DoorModel.DetectMaxFill", 0.60f,
            "");
    out.detect_soft              = getf("DoorModel.DetectSoft", 0.06f,
            "");
    out.ai2_clutter_frac         = getf("DoorModel.AI2ClutterFrac", 0.10f,
            "");
    out.ai2_clutter_scale_m      = getf("DoorModel.AI2ClutterScaleM", 0.12f,
            "");
    out.ai2_clutter_structure_gain = getf("DoorModel.AI2ClutterStructureGain", 1.0f,
            "density-aware clutter: shrink the clutter prior for seat-coplanar points (closes the escape valve → fixes seat_d collapse + its overconfidence). 0 → f");
    out.ai2_prior_size_std       = getf("DoorModel.AI2PriorSizeStd", 0.15f,
            "");
    out.ai2_process_std_m        = getf("DoorModel.AI2ProcessStdM", 0.005f,
            "");
    out.ai2_process_std_yaw      = getf("DoorModel.AI2ProcessStdYaw", 0.01f,
            "");
    out.ai2_process_std_size     = getf("DoorModel.AI2ProcessStdSize", 0.0005f,
            "rigid size DOFs ≪ pose (Tier-1: kills the vertical random walk)");
    out.ai2_age_nominal_dt_s     = getf("DoorModel.AI2AgeNominalDtS", 0.0f,
            "Stale-belief aging (measurement-age → covariance)");
    out.ai2_floor_z              = getf("DoorModel.AI2FloorZ", 0.0f,
            "room-frame floor; cz pinned here (Tier-1: removes cz gauge freedom)");
    out.ai2_floor_std            = getf("DoorModel.AI2FloorStd", 0.03f,
            "floor-height uncertainty (m) → common-mode z");
    out.ai2_seat_anchor_std      = getf("DoorModel.AI2SeatAnchorStd", 0.04f,
            "seat-layer height anchor obs noise (m); 0 → off. Fixes seat_h gauge runaway");
    out.ai2_seat_anchor_band     = getf("DoorModel.AI2SeatAnchorBand", 0.12f,
            "seat-layer mean-shift bandwidth (m) = seat vertical scale");
    out.ai2_seat_extent_std      = getf("DoorModel.AI2SeatExtentStd", 0.02f,
            "seat-layer footprint (seat_w/seat_d) span anchor obs noise (m); 0 → off. Fixes seat_d collapse");
    out.ai2_common_mode_pos_std  = getf("DoorModel.AI2CommonModePosStd", 0.03f,
            "");
    out.ai2_common_mode_size_std = getf("DoorModel.AI2CommonModeSizeStd", 0.35f,
            "per-frame SHARED size (w,h) error → caps how far one mask moves them (anti width-drift)");
    out.ai2_common_mode_yaw_std  = getf("DoorModel.AI2CommonModeYawStd", 0.03f,
            "");
    out.ai2_range_noise_lat_per_m = getf("DoorModel.AI2RangeNoiseLatPerM", 0.02f,
            "static range → R + position common-mode (m per m)");
    out.ai2_range_noise_yaw_per_m = getf("DoorModel.AI2RangeNoiseYawPerM", 0.03f,
            "static range → yaw common-mode (rad per m)");
    out.motion_cm_pos_gain        = getf("DoorModel.MotionCmPosGain", 0.10f,
            "position (cx,cy) shared-error std per m/s — anti-DRIFT (0.10→0.30)");
    out.motion_cm_yaw_gain        = getf("DoorModel.MotionCmYawGain", 0.12f,
            "yaw shared-error std (rad) per m/s — anti-ROTATE (0.12→0.50)");
    out.ai2_ang_lever_m           = getf("DoorModel.AI2AngLeverM", 2.0f,
            "rad/s → m/s lever (tangential speed of a door ~this far away) for ego-motion");
    out.ai2_periph_ref            = getf("DoorModel.AI2PeriphRef", 0.50f,
            "centroid radius (focal-norm, tan of off-axis angle) at which the periphery");
    out.ai2_motion_ref_mps        = getf("DoorModel.AI2MotionRefMps", 0.60f,
            "motion magnitude (m/s) at which a fully-peripheral frame becomes fully");
    out.ai2_motion_confirm_only   = getb("DoorModel.AI2MotionConfirmOnly", false,
            "unreliable (existence weight → 0)");
    out.ai2_still_lin_mps         = getf("DoorModel.AI2StillLinMps", 0.05f,
            "(hard gate) camera linear speed (m/s) below which robot counts as 'still'");
    out.ai2_still_ang_radps       = getf("DoorModel.AI2StillAngRadps", 0.10f,
            "(hard gate) camera angular speed (rad/s) below which robot counts as 'still'");
    out.ai2_still_dotd            = getf("DoorModel.AI2StillDotd", 0.05f,
            "(hard gate) per-mask ego-motion corruption speed (m/s) still-level");
    out.ai2_moving_update_center_radius = getf("DoorModel.AI2MovingUpdateCenterRadius", 0.35f,
            "(hard gate) mask centroid radius below which a moving update is allowed");
    out.ai2_obliquity_yaw_gain    = getf("DoorModel.AI2ObliquityYawGain", 0.05f,
            "Obliquity yaw cap (TABLE.md §6): the backrest (the door's yaw-carrying surface) is a vertical plate, so ⚠★DEAD KEY (confirmed 2026-07-31) — loaded by…");
    out.ai2_orientation_motion_ref = getf("DoorModel.AI2OrientationMotionRef", 0.50f,
            "Ego-motion reliability of the discrete orientation vote: w = 1/(1+(dotd/ref)²)");
    out.ai2_fe_baseline_adapt_down = getf("DoorModel.AI2FeBaselineAdaptDown", 0.05f,
            "FE-surprise attention baseline (TABLE.md §9): asymmetric EMA (down fast = consolidate a better fit; up slow = a sustained rise, the door moved, stays…");
    out.ai2_fe_baseline_adapt_up   = getf("DoorModel.AI2FeBaselineAdaptUp", 0.005f,
            "");
    out.ai2_fe_surprise_smooth     = getf("DoorModel.AI2FeSurpriseSmooth", 0.10f,
            "");
    out.ai2_trunc_gate_frac      = getf("DoorModel.AI2TruncGateFrac", 0.10f,
            "skip the geometric update (predict only) when this fraction of the mask is on the image border");
    out.lidar_bpearl_precision   = getf("DoorModel.LidarBpearlPrecision", out.lidar_bpearl_precision,
            "★90 DEGREES WAS A HARD CEILING AND REAL DOORS GO PAST IT. estimate_phi searched [0, pi/2] and CLAMPED the estimate to it, so a door standing open at…");
    out.phi_curve_csv_path       = gets("DoorModel.PhiCurveCsvPath", out.phi_curve_csv_path,
            "★TWO FILES THAT TOGETHER RECONSTRUCT THE DECISION, not summarise it");
    out.phi_points_csv_path      = gets("DoorModel.PhiPointsCsvPath", out.phi_points_csv_path,
            "");
    out.phi_debug_every_n        = geti("DoorModel.PhiDebugEveryN", out.phi_debug_every_n,
            "0 disables both");
    out.phi_max_rad              = getf("DoorModel.PhiMaxRad", out.phi_max_rad,
            "120 deg");
    out.phi_step_rad             = getf("DoorModel.PhiStepRad", out.phi_step_rad,
            "5 deg");
    out.ai2_gn_iters             = geti("DoorModel.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_extent_std           = getf("DoorModel.AI2ExtentStd", 0.05f,
            "extent-observation noise (m) for the coverage/extent likelihood");
    out.ai2_csv_path             = gets("DoorModel.AI2CsvPath", "",
            "per-cycle AI2 log (state + Σ diag + range/motion); empty disables");
    out.detect_probe_csv_path    = gets("DoorConcept.DetectProbeCsvPath", out.detect_probe_csv_path,
            "Detector truth table (rc::probe, shared schema — common/detect_probe/detect_probe.h): one row per cycle per live door recording WHERE the robot…");
    out.rgb_contour_check        = getb("DoorConcept.RgbContourCheck", out.rgb_contour_check,
            "RGB contour check: score the door's own projected silhouette against image edges (common/contour_edge)");
    out.contour_depth_check      = getb("DoorConcept.ContourDepthCheck", out.contour_depth_check,
            "The ZED-depth test asks the absolute question the RGB gradient cannot: is there a surface at the distance the belief predicts, with space receding…");
    out.frame_contour_check      = getb("DoorConcept.FrameContourCheck", out.frame_contour_check,
            "Doorway-FRAME existence evidence: score the aperture's jambs + lintel (which do not move when the leaf swings) against the RGB image, relative to the frame slid along the wall");
    out.frame_depth_check        = getb("DoorConcept.FrameDepthCheck", out.frame_depth_check,
            "Depth half of the frame channel: does the hole recede behind the wall across each jamb/lintel edge (relative, so blind to a common-mode ZED bias)");
    out.frame_casing_m           = getf("DoorConcept.FrameCasingM", out.frame_casing_m,
            "Width of the casing (frame boards) lining a doorway: the frame's depth half reads the hole side past it, since the casing stands proud of the wall");
    out.door_control_endpoint    = gets("DoorConcept.DoorControlEndpoint", out.door_control_endpoint,
            "RoboCompDoorControl provider endpoint");

    out.rt_cov_upload                 = getb("DoorConcept.RtCovUpload", true,
            "Upload the door pose covariance onto the room→door RT edge (rt_covariance_att, 6×6 SE3), built from the belief's full Σ over [cx,cy,cz,yaw,...]:…");
    out.rt_cov_scale                  = getf("DoorConcept.RtCovScale", 1.0f,
            "var = RtCovScale · Σ_ii; calibrate toward NEES≈1");
    out.rt_cov_add_chain              = getb("DoorConcept.RtCovAddChain", true,
            "Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov");
    out.anchor_frame_record           = getb("DoorConcept.AnchorFrameRecord", false,
            "Record every measured interaction-anchor cycle to etc/anchor_frames.bin (replay_anchor_frames; ≤ 200 MB/run). Development instrument, default off");
    out.decision_log                  = getb("DoorConcept.DecisionLog", false,
            "Write etc/door_decisions.bin: the binary per-cycle decision log (decision_log.h) the DoorCore replay is checked against. Default off");
    out.record_inputs                 = getb("DoorConcept.RecordInputs", false,
            "Record every compute() cycle's DoorCycleInput (DCIN v1) for the DoorCore replay (input_snapshot.h). Default off");
    out.record_inputs_path            = gets("DoorConcept.RecordInputsPath", out.record_inputs_path,
            "Where RecordInputs writes (rotated per run, the 3 previous kept)");
    out.record_max_mb                 = geti("DoorConcept.RecordMaxMB", out.record_max_mb,
            "RecordInputs size cap (MB): at the cap the recording STOPS, with a log line");
    out.anchor_log                    = getb("DoorConcept.AnchorLog", false,
            "Write etc/anchor_log.csv, one row per cycle while an interaction anchor exists. Development instrument, default off");

    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "χ²₂ gate (~3σ) for a mask↔instance match once it has a cov");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.40f,
            "metric XY gate (m) before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.20f,
            "R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)");
    out.tracker_birth_frames     = getf("Tracker.BirthFrames", out.tracker_birth_frames,
            "IDEAL OBSERVATIONS of unexplained evidence before a birth");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.70f,
            "a birth must be ≥ this (m) from every existing door (anti-dup)");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.20f,
            "merge two instances whose seat footprints overlap ≥ this");
    out.exist_enabled            = getb("Existence.Enabled", true,
            "Existence.Enabled — use the log-odds belief (else the old prune)");
    out.exist_birth_logodds      = getf("Existence.BirthLogodds", 1.0f,
            "Existence.BirthLogodds — L seeded at birth (a birth already needed birth_frames of evidence)");
    out.exist_max_logodds        = getf("Existence.MaxLogodds", 4.0f,
            "Existence.MaxLogodds — saturation cap (a real door can't earn infinite immunity)");
    out.exist_removal_prob       = getf("Existence.RemovalProb", 0.12f,
            "Existence.RemovalProb");
    out.exist_remove_frames      = geti("Existence.RemoveFrames", 15,
            "Existence.RemoveFrames — consecutive EVIDENCE cycles the decision must hold");
    out.exist_detection_prob     = getf("Existence.DetectionProb", 0.85f,
            "Existence.DetectionProb — P(mask lights a predicted pixel | door present & observable)");
    out.exist_clutter_prob       = getf("Existence.ClutterProb", 0.05f,
            "Existence.ClutterProb — P(mask lights a predicted pixel | no door)");
    out.exist_sensor_sigma_m     = getf("Existence.SensorSigmaM", 0.03f,
            "Existence.SensorSigmaM — range/localisation noise σ");
    out.exist_central_region_frac = getf("Existence.CentralRegionFrac", 0.25f,
            "Existence.CentralRegionFrac");
    out.exist_occlusion_margin_m = getf("Existence.OcclusionMarginM", 0.30f,
            "Existence.OcclusionMarginM — an occluder must be ≥ this much CLOSER to count");
    out.exist_room_prior         = getb("Existence.RoomPrior", true,
            "Existence.RoomPrior — enforce the room-containment pose prior");
    out.exist_room_margin_m      = getf("Existence.RoomMarginM", 0.40f,
            "Existence.RoomMarginM — tolerance a door centre may sit OUTSIDE the walls");
    out.exist_out_of_room_gain   = getf("Existence.OutOfRoomGain", 1.5f,
            "Existence.OutOfRoomGain — |ΔL| per frame while outside (debounces a 1-frame glitch)");
    out.exist_wall_thickness_m   = getf("Existence.WallThicknessM", 0.20f,
            "Existence.WallThicknessM — wall thickness T: an aperture lies in the band [0,T] beyond the room-side polygon face");
    out.exist_boundary_sigma_m   = getf("Existence.BoundarySigmaM", 0.10f,
            "Existence.BoundarySigmaM — σ of the door's registration against the room polygon face (wall-band prior + leaf-beyond weight)");
    out.exist_min_height_prior   = getb("Existence.MinHeightPrior", true,
            "Existence.MinHeightPrior");
    out.exist_min_height_m       = getf("Existence.MinHeightM", 1.80f,
            "Existence.MinHeightM — a door's support must reach this (m)");
    out.exist_min_height_conf    = getf("Existence.MinHeightConf", 0.30f,
            "Existence.MinHeightConf — untruncated-evidence weight required");
    out.exist_short_gain         = getf("Existence.ShortGain", 1.5f,
            "Existence.ShortGain — |ΔL| per frame while confidently short");
    out.exist_reacquire_radius_m = getf("Existence.ReacquireRadiusM", 0.60f,
            "Existence.ReacquireRadiusM (0 disables re-acquisition)");
    out.exist_ghost_max          = geti("Existence.GhostMax", 8,
            "Existence.GhostMax — most recent removals retained");
    // [Openable] — deliberately ABSENT from etc/config.toml (same as [Bearing]): a dormant, opt-in feature
    // whose defaults keep the leaf pinned flush. See the block comment in door_config.h.
    out.openable_enabled     = getb("Openable.Enabled", false,
            "Openable.Enabled — false ⇒ phi is the literal 0.0f everywhere");
    out.openable_phi_init    = getf("Openable.PhiInitRad", 0.0f,
            "Openable.PhiInitRad — constant opening angle (rad)");
    out.openable_hinge_side  = geti("Openable.HingeSide", 0,
            "Openable.HingeSide — 0 = near (s) edge, 1 = far (s+w) edge");
    out.openable_swing_dir   = getf("Openable.SwingDir", 1.0f,
            "Openable.SwingDir — +1 / −1: side of the wall it opens toward");
    out.openable_phi_max_rad = getf("Openable.PhiMaxRad", 1.5707963f,
            "Openable.PhiMaxRad — physical hinge travel limit (M1 uses it)");
    out.door_prior_w_m           = getf("Door.PriorWidthM", out.door_prior_w_m,
            "width prior mean (m)");
    out.door_prior_w_std         = getf("Door.PriorWidthStd", 0.06f,
            "width prior std (m) — strong");
    out.door_prior_h_m           = getf("Door.PriorHeightM", out.door_prior_h_m,
            "height prior mean (m)");
    out.door_prior_h_std         = getf("Door.PriorHeightStd", 0.08f,
            "height prior std (m) — strong");
    out.door_prior_s_std         = getf("Door.PriorAlongWallStd", 0.60f,
            "along-wall offset prior std (m) — broad (the fit localises s)");
    out.door_recess_prior_std    = getf("Door.RecessPriorStd", -1.0f,
            "prior std (m) of the leaf plane's depth off the wall line; < 0 = derived, sqrt(T^2/12 + sigma_b^2) (Task 5b)");
    out.door_thickness_m         = getf("Door.ThicknessM", 0.05f,
            "fixed panel thickness (m, across the wall)");
    out.tracker_nll_cost         = getb("Tracker.NllCost", false,
            "association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker");
    out.bearing_birth_enabled    = getb("Bearing.BirthEnabled", false,
            "Bearing.BirthEnabled");
    out.bearing_confirm_gate_rad = getf("Bearing.ConfirmGateRad", 0.17f,
            "Bearing.ConfirmGateRad — bearing within this of a live door's azimuth = 'explained'");
    out.bearing_birth_frames     = geti("Bearing.BirthFrames", 8,
            "Bearing.BirthFrames — unmatched-bearing streak before promotion");
    out.bearing_match_rad        = getf("Bearing.MatchRad", 0.17f,
            "Bearing.MatchRad — candidate↔bearing azimuth match tolerance");
    out.bearing_max_miss         = geti("Bearing.MaxMiss", 4,
            "Bearing.MaxMiss — streak gap tolerance (intermittent 360 detection)");
    out.bearing_nominal_range_m  = getf("Bearing.NominalRangeM", 2.0f,
            "Bearing.NominalRangeM — where the mean starts on the ray (Σ carries the real uncertainty)");
    out.bearing_along_std_m      = getf("Bearing.AlongStdM", 3.0f,
            "Bearing.AlongStdM — Σ std ALONG the ray (unknown range)");
    out.bearing_across_std_m     = getf("Bearing.AcrossStdM", 0.30f,
            "Bearing.AcrossStdM — Σ std ACROSS the ray (bearing known)");
    out.bearing_yaw_std_rad      = getf("Bearing.YawStdRad", 3.14f,
            "Bearing.YawStdRad — orientation fully unknown at birth");
}

}  // namespace rc
