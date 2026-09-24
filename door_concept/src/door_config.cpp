/*
 * door_config.cpp — fill DoorConfig from a RoboComp ConfigLoader.
 */

#include "door_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

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
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "door_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

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
    out.door_control_endpoint    = gets("DoorConcept.DoorControlEndpoint", out.door_control_endpoint,
            "RoboCompDoorControl provider endpoint");

    out.rt_cov_upload                 = getb("DoorConcept.RtCovUpload", true,
            "Upload the door pose covariance onto the room→door RT edge (rt_covariance_att, 6×6 SE3), built from the belief's full Σ over [cx,cy,cz,yaw,...]:…");
    out.rt_cov_scale                  = getf("DoorConcept.RtCovScale", 1.0f,
            "var = RtCovScale · Σ_ii; calibrate toward NEES≈1");
    out.rt_cov_add_chain              = getb("DoorConcept.RtCovAddChain", true,
            "Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov");

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
    out.door_thickness_m         = getf("Door.ThicknessM", 0.05f,
            "fixed panel thickness (m, across the wall)");
    out.tracker_nll_cost         = getb("Tracker.NllCost", false,
            "association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker");
    out.ricoh_birth_enabled      = getb("Tracker.RicohBirthEnabled", false,
            "Tracker.RicohBirthEnabled — allow a confident ricoh-depth slice to birth");
    out.ricoh_birth_conf         = getf("Tracker.RicohBirthConf", 0.60f,
            "Tracker.RicohBirthConf — min YOLO confidence for a ricoh birth");
    out.ricoh_birth_max_var      = getf("Tracker.RicohBirthMaxVar", 0.005f,
            "Tracker.RicohBirthMaxVar — max depth_var (m²) for a ricoh birth");
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

    std::print("door_concept: configuration loaded.\n");
    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★z_top is the door-leaf height prior (2.00), matching the manifest. A door has no LiDAR sweep, so
    // there is no selection band to check — only point ownership.
    {
        const auto span = rc::manifest::adopt_span(kManifestPath, "door",
                                                  rc::manifest::Support::floor_anchored,
                                                  out.door_prior_h_m, out.door_prior_h_m);
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = out.door_prior_h_m;
        decl.extent_m = out.door_prior_h_m;
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body("door ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] door ✓ every derived z-band contains the declared body\n");
    }

    // ── the three pragmatic affordances: approach / open / cross ([DoorAffordance]) ───────────────
    {
        auto& pg = out.pragmatic;
        pg.enabled               = getb("DoorAffordance.Enabled", pg.enabled,
                "distance or an angle");
        pg.actuation_reach_m     = getf("DoorAffordance.ActuationReachM", pg.actuation_reach_m,
                "⚠ApproachStandoffM IS GONE, not renamed");
        pg.cross_standoff_m      = getf("DoorAffordance.CrossStandoffM", pg.cross_standoff_m,
                "how far PAST the aperture the cross target sits");
        pg.cross_clearance_m     = getf("DoorAffordance.CrossClearanceM", pg.cross_clearance_m,
                "distance past the aperture plane that counts as fully through");
        pg.robot_passage_width_m = getf("DoorAffordance.RobotPassageWidthM", pg.robot_passage_width_m,
                "★A MEASUREMENT OF THE ROBOT, not a tuning knob — and the one direction in which a doorway closes");
        pg.passage_margin_m      = getf("DoorAffordance.PassageMarginM", pg.passage_margin_m,
                "clearance demanded on top of the body width");
        pg.pose_sigma_floor_m    = getf("DoorAffordance.PoseSigmaFloorM", pg.pose_sigma_floor_m,
                "Pose-uncertainty floor for door_reach_prob");
        pg.swing_clearance_m     = getf("DoorAffordance.SwingClearanceM", pg.swing_clearance_m,
                "★CLEARANCE FOR THE LEAF'S OWN SWEPT ARC — a SAFETY quantity, not a preference");
        pg.actuation_swing_rate  = getf("DoorAffordance.ActuationSwingRate", pg.actuation_swing_rate,
                "rad/s; 90 deg in ~3.1 s");
        pg.phi_sigma_rad         = getf("DoorAffordance.PhiSigmaRad", pg.phi_sigma_rad,
                "★THE WIDTH OF THE LEAF-ANGLE BELIEF WHEN THE IMAGE IS NOT RESOLVING IT. estimate_phi() refills phi_curve only from a fresh mask, and this agent's own…");
        pg.value_approach        = getf("DoorAffordance.ValueApproach", pg.value_approach,
                "★An epistemic gain is a measured entropy reduction in nats");
        pg.value_open            = getf("DoorAffordance.ValueOpen", pg.value_open,
                "");
        pg.value_cross           = getf("DoorAffordance.ValueCross", pg.value_cross,
                "");
        pg.approach_timeout_s    = getf("DoorAffordance.ApproachTimeoutS", pg.approach_timeout_s,
                "");
        pg.open_timeout_s        = getf("DoorAffordance.OpenTimeoutS", pg.open_timeout_s,
                "provider latency + swing time, not a navigation budget");
        pg.cross_timeout_s       = getf("DoorAffordance.CrossTimeoutS", pg.cross_timeout_s,
                "");
        pg.offer_prob            = getf("DoorAffordance.OfferProb", pg.offer_prob,
                "Offer policy (rc::pragmatic::PragmaticAffordance::Policy): offer above `offer_prob`, withdraw below `withdraw_prob`, and only after the decision has…");
        pg.withdraw_prob         = getf("DoorAffordance.WithdrawProb", pg.withdraw_prob,
                "… and withdraw once it falls below THIS (must be < OfferProb; refused at load)");
        pg.stable_cycles         = geti("DoorAffordance.StableCycles", pg.stable_cycles,
                "… and only after the decision has held this many MEASURED cycles");
        pg.transitable_prob      = getf("DoorAffordance.TransitableProb", pg.transitable_prob,
                "assert once P(passable) rises above this …");
        pg.not_transitable_prob  = getf("DoorAffordance.NotTransitableProb", pg.not_transitable_prob,
                "… and retract once it falls below THIS (must be < the above)");
        pg.transitable_stable_cycles = geti("DoorAffordance.TransitableStableCycles", pg.transitable_stable_cycles,
                "… and only after the decision has held this many MEASURED cycles");
        pg.autonomous_actuation  = getb("DoorAffordance.AutonomousActuation", pg.autonomous_actuation,
                "★MAY THIS AGENT ASK A PROVIDER TO MOVE A DOOR WITHOUT A HUMAN? The `open` affordance's whole point is that it can, and the consumer's claim is the…");
        pg.actuation_provider_id = gets("DoorAffordance.ActuationProviderId", pg.actuation_provider_id,
                "Which door the PROVIDER should move, by its own advertised id");

        // ★REFUSE AN INVERTED SCHMITT BAND AT LOAD. withdraw_prob >= offer_prob is not a band, it is a
        // single line with the two decisions on the wrong sides of it: the affordance would be offered and
        // withdrawn on the same probability and chatter at the compute rate, writing a node into the shared
        // graph every few cycles. Caught here because the symptom (CRDT churn in somebody else's agent) is
        // nowhere near the cause. See [[dsr-crdt-dot-cloud-unbounded]].
        // The `transitable` band gets the same refusal as the offer band above, and for the same reason:
        // an inverted band asserts and retracts on one probability, which would flap a PUBLIC edge in the
        // shared graph at the compute rate — CRDT churn in somebody else's agent, nowhere near the cause.
        if (pg.not_transitable_prob >= pg.transitable_prob)
        {
            std::print("[config] DoorAffordance.NotTransitableProb ({:.2f}) must be BELOW TransitableProb "
                       "({:.2f}) — that is not a hysteresis band. Falling back to 0.35/0.60.\n",
                       pg.not_transitable_prob, pg.transitable_prob);
            pg.not_transitable_prob = 0.35f;
            pg.transitable_prob     = 0.60f;
        }
        if (pg.withdraw_prob >= pg.offer_prob)
        {
            std::print("[config] DoorAffordance.WithdrawProb ({:.2f}) must be BELOW OfferProb ({:.2f}) — "
                       "that is not a hysteresis band. Falling back to 0.35/0.60.\n",
                       pg.withdraw_prob, pg.offer_prob);
            pg.withdraw_prob = 0.35f;
            pg.offer_prob    = 0.60f;
        }
        // A passage the robot cannot fit through can never be crossed, and a body width of 0 says the
        // robot is a point — both are configuration mistakes worth a line rather than a silent behaviour.
        if (pg.robot_passage_width_m <= 0.0f)
            std::print("[config] DoorAffordance.RobotPassageWidthM is {:.2f} m — the robot is not a point; "
                       "the `cross` affordance will be offered for ANY ajar door\n", pg.robot_passage_width_m);
    }

    return out;
}

}  // namespace rc
