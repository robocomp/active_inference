/*
 * chair_config.cpp — fill ChairConfig from a RoboComp ConfigLoader.
 */

#include "chair_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

ChairConfig load_chair_config(const ConfigLoader& cfg)
{
    ChairConfig out;


    // The one place this path is written. Relative to the agent's CWD (<agent>/), not to src/ — the same
    // string was right in one place and wrong in the other for a week, and the manifest was inert the whole time.
    static constexpr const char* kManifestPath = "../common/concept_manifest/chair.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL — rc::manifest::provenance_ok. `from = "inherited"` means the number
    // arrived by a rename and nobody chose it for THIS object; hood shipped ten such defects in a week with
    // several of them declared, in writing, in its manifest. A note stopped nothing, so this stops the agent.
    if (not rc::manifest::provenance_ok(kManifestPath, "chair"))
        std::exit(EXIT_FAILURE);

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "chair_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // Agent convergence
    out.state_eps                = getf("ChairConcept.StateEps", 0.04f,
            "Σ|Δstate| threshold between cycles for convergence (m+rad)");
    out.K_stable                 = geti("ChairConcept.KStable", 30,
            "consecutive stable frames before model_stable");
    out.detection_alive_max_frames = geti("ChairConcept.DetectionAliveMaxFrames", 40,
            "cycles without a fresh chair mask before detection_alive=false");
    out.obs_distance             = getf("ChairConcept.ObsDistance", 1.8f,
            "d_obs for epistemic planner");
    out.min_standoff_m           = getf("ChairConcept.MinStandOffM", 1.8f,
            "min stand-off floor for epistemic viewpoints (YOLO misses too-close chairs)");
    out.epistemic_cooldown_cycles= geti("ChairConcept.EpistemicCooldownCycles", 200,
            "min cycles withdrawn after satisfaction");
    out.chair_log_period_frames  = geti("ChairConcept.ChairLogPeriodFrames", 30,
            "per-chair log throttle");
    out.masks_stall_timeout_ms   = geti("Media.MasksStallTimeoutMs", 3000,
            "Primary-input (masks) stream gate: no NEW masks frame for this many ms while Operating ⇒ demote out of Operating rather than integrate stale…");
    out.support_bank_max_points    = geti("ChairConcept.SupportBankMaxPoints", 4000,
            "max number of chair-owned support points kept in memory");
    out.support_bank_quantization_m= getf("ChairConcept.SupportBankQuantizationM", 0.02f,
            "dedup grid (m) for support-bank hashing");
    out.support_select_radius_margin_m = getf("ChairConcept.SupportSelectRadiusMarginM", 0.50f,
            "per-chair XY ownership gate margin for mixed proto-chairs");
    out.support_select_height_margin_m = getf("ChairConcept.SupportSelectHeightMarginM", 0.25f,
            "extra Z margin above chair height when selecting owned points");

    // ChairModel geometry / mask split
    out.sigma_obs          = getf("ChairModel.SigmaObs", 0.05f,
            "ChairModel geometry / mask split: on-surface membership for the candidate/residual split in ChairFitter::observe (a mask point within…");
    out.sdf_threshold_for_storage = getf("ChairModel.SdfThresholdForStorage", 0.08f,
            "on-surface membership for the candidate/residual split");

    // ── AI2 belief ────────────────────────────────────────────────────────────
    out.ai2_sigma_base_m         = getf("ChairModel.AI2SigmaBaseM", 0.03f,
            "");
    out.detect_min_fill          = getf("ChairModel.DetectMinFill", 0.10f,
            "ONE model, two consumers: the epistemic planner puts the stand-off at its argmax, and the removal channel weights absence by it, so a missing mask…");
    out.detect_max_fill          = getf("ChairModel.DetectMaxFill", 0.60f,
            "");
    out.detect_soft              = getf("ChairModel.DetectSoft", 0.06f,
            "");
    out.ai2_clutter_frac         = getf("ChairModel.AI2ClutterFrac", 0.10f,
            "");
    out.ai2_clutter_scale_m      = getf("ChairModel.AI2ClutterScaleM", 0.12f,
            "");
    out.ai2_clutter_structure_gain = getf("ChairModel.AI2ClutterStructureGain", 1.0f,
            "density-aware clutter: shrink the clutter prior for seat-coplanar points (closes the escape valve → fixes seat_d collapse + its overconfidence). 0 → f");
    out.ai2_prior_size_std       = getf("ChairModel.AI2PriorSizeStd", 0.15f,
            "");
    out.ai2_process_std_m        = getf("ChairModel.AI2ProcessStdM", 0.005f,
            "");
    out.ai2_process_std_yaw      = getf("ChairModel.AI2ProcessStdYaw", 0.01f,
            "");
    out.ai2_process_std_size     = getf("ChairModel.AI2ProcessStdSize", 0.0005f,
            "rigid size DOFs ≪ pose (Tier-1: kills the vertical random walk)");
    out.ai2_age_nominal_dt_s     = getf("ChairModel.AI2AgeNominalDtS", 0.0f,
            "Stale-belief aging (measurement-age → covariance)");
    out.ai2_floor_z              = getf("ChairModel.AI2FloorZ", 0.0f,
            "room-frame floor; cz pinned here (Tier-1: removes cz gauge freedom)");
    out.ai2_floor_std            = getf("ChairModel.AI2FloorStd", 0.03f,
            "floor-height uncertainty (m) → common-mode z");
    out.ai2_seat_anchor_std      = getf("ChairModel.AI2SeatAnchorStd", 0.04f,
            "seat-layer height anchor obs noise (m); 0 → off. Fixes seat_h gauge runaway");
    out.ai2_seat_anchor_band     = getf("ChairModel.AI2SeatAnchorBand", 0.12f,
            "seat-layer mean-shift bandwidth (m) = seat vertical scale");
    out.ai2_seat_extent_std      = getf("ChairModel.AI2SeatExtentStd", 0.02f,
            "seat-layer footprint (seat_w/seat_d) span anchor obs noise (m); 0 → off. Fixes seat_d collapse");
    out.ai2_common_mode_pos_std  = getf("ChairModel.AI2CommonModePosStd", 0.03f,
            "");
    out.ai2_common_mode_size_std = getf("ChairModel.AI2CommonModeSizeStd", 0.02f,
            "");
    out.ai2_common_mode_yaw_std  = getf("ChairModel.AI2CommonModeYawStd", 0.03f,
            "");
    out.ai2_range_noise_lat_per_m = getf("ChairModel.AI2RangeNoiseLatPerM", 0.02f,
            "static range → R + position common-mode (m per m)");
    out.ai2_range_noise_yaw_per_m = getf("ChairModel.AI2RangeNoiseYawPerM", 0.03f,
            "static range → yaw common-mode (rad per m)");
    out.motion_cm_pos_gain        = getf("ChairModel.MotionCmPosGain", 0.10f,
            "position (cx,cy) shared-error std per m/s — anti-DRIFT (0.10→0.30)");
    out.motion_cm_yaw_gain        = getf("ChairModel.MotionCmYawGain", 0.12f,
            "yaw shared-error std (rad) per m/s — anti-ROTATE (0.12→0.50)");
    out.ai2_ang_lever_m           = getf("ChairModel.AI2AngLeverM", 2.0f,
            "rad/s → m/s lever (tangential speed of a chair ~this far away) for ego-motion");
    out.ai2_periph_ref            = getf("ChairModel.AI2PeriphRef", 0.50f,
            "centroid radius (focal-norm, tan of off-axis angle) at which the periphery");
    out.ai2_motion_ref_mps        = getf("ChairModel.AI2MotionRefMps", 0.60f,
            "motion magnitude (m/s) at which a fully-peripheral frame becomes fully");
    out.ai2_motion_confirm_only   = getb("ChairModel.AI2MotionConfirmOnly", false,
            "unreliable (existence weight → 0)");
    out.fixation_enabled          = getb("ChairModel.FixationEnabled", true,
            "Same rationale as table_concept's (see TableConfig::fixation_enabled): every graded lever acts through the per-frame common-mode Σc, and the engine…");
    out.fixation_min_pts          = geti("ChairModel.FixationMinPts", 150,
            "min mask points for the frame to say anything about the pose");
    out.fixation_max_clutter      = getf("ChairModel.FixationMaxClutter", 0.35f,
            "max clutter fraction (surface the model cannot explain)");
    out.fixation_range_m          = getf("ChairModel.FixationRangeM", 0.0f,
            "RETIRED as a gate (0 = off). Kept so an A/B revert is one edit");   // retired as a gate
    out.fixation_centre_frac      = getf("ChairModel.FixationCentreFrac", 0.60f,
            "CENTRED: max mask-centroid radius (focal-norm) — the 'fovea'");
    out.fixation_centre_precision = getb("ChairModel.FixationCentrePrecision", false,
            "false (default) = today: outside the fovea the geometry update is INHIBITED entirely. true = opportunistic: an off-fovea object is still observed, at…");
    out.fixation_periph_floor     = getf("ChairModel.FixationPeriphFloor", 0.05f,
            "Floor on the peripheral precision factor, so a fully off-axis look is worth LITTLE rather than NOTHING. 1 = no peripheral attenuation at all; 0 = a…");
    out.fixation_still_dotd       = getf("ChairModel.FixationStillDotd", 0.05f,
            "STILL: max |motion_dotd| = Z·‖ṡ‖ (m/s) ego-motion mask smear");
    out.fixation_still_lin_mps    = getf("ChairModel.FixationStillLinMps", 0.05f,
            "STILL: max robot linear speed (m/s)");
    out.fixation_still_ang_radps  = getf("ChairModel.FixationStillAngRadps", 0.10f,
            "STILL: max robot angular speed (rad/s) — the turn case");
    out.ai2_still_lin_mps         = getf("ChairModel.AI2StillLinMps", 0.05f,
            "(hard gate) camera linear speed (m/s) below which robot counts as 'still'");
    out.ai2_still_ang_radps       = getf("ChairModel.AI2StillAngRadps", 0.10f,
            "(hard gate) camera angular speed (rad/s) below which robot counts as 'still'");
    out.ai2_still_dotd            = getf("ChairModel.AI2StillDotd", 0.05f,
            "(hard gate) per-mask ego-motion corruption speed (m/s) still-level");
    out.ai2_moving_update_center_radius = getf("ChairModel.AI2MovingUpdateCenterRadius", 0.35f,
            "(hard gate) mask centroid radius below which a moving update is allowed");
    out.ai2_obliquity_yaw_gain    = getf("ChairModel.AI2ObliquityYawGain", 0.05f,
            "Obliquity yaw cap (TABLE.md §6): the backrest (the chair's yaw-carrying surface) is a vertical plate, so a view that grazes it edge-on can barely…");
    out.ai2_orientation_motion_ref = getf("ChairModel.AI2OrientationMotionRef", 0.50f,
            "Ego-motion reliability of the discrete orientation vote: w = 1/(1+(dotd/ref)²)");
    out.ai2_fe_baseline_adapt_down = getf("ChairModel.AI2FeBaselineAdaptDown", 0.05f,
            "FE-surprise attention baseline (TABLE.md §9): asymmetric EMA (down fast = consolidate a better fit; up slow = a sustained rise, the chair moved,…");
    out.ai2_fe_baseline_adapt_up   = getf("ChairModel.AI2FeBaselineAdaptUp", 0.005f,
            "");
    out.ai2_fe_surprise_smooth     = getf("ChairModel.AI2FeSurpriseSmooth", 0.10f,
            "");
    out.ai2_trunc_gate_frac      = getf("ChairModel.AI2TruncGateFrac", 0.10f,
            "skip the geometric update (predict only) when this fraction of the mask is on the image border");
    out.ai2_gn_iters             = geti("ChairModel.AI2GnIters", 4,
            "Gauss-Newton iterations per frame");
    out.ai2_extent_std           = getf("ChairModel.AI2ExtentStd", 0.05f,
            "extent-observation noise (m) for the coverage/extent likelihood");
    out.ai2_csv_path             = gets("ChairModel.AI2CsvPath", "",
            "per-cycle AI2 log (state + Σ diag + range/motion); empty disables");

    out.rt_cov_upload                 = getb("ChairConcept.RtCovUpload", true,
            "Upload the chair pose covariance onto the room→chair RT edge (rt_covariance_att, 6×6 SE3), built from the belief's full Σ over [cx,cy,cz,yaw,...]:…");
    out.rt_cov_scale                  = getf("ChairConcept.RtCovScale", 1.0f,
            "var = RtCovScale · Σ_ii; calibrate toward NEES≈1");
    out.rt_cov_add_chain              = getb("ChairConcept.RtCovAddChain", true,
            "Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov");

    out.tracker_gate_mahalanobis = getf("Tracker.GateMahalanobis", 9.0f,
            "χ²₂ gate (~3σ) for a mask↔instance match once it has a cov");
    out.tracker_gate_fallback_m  = getf("Tracker.GateFallbackM", 0.40f,
            "metric XY gate (m) before an instance has a usable covariance");
    out.tracker_detection_noise_m = getf("Tracker.DetectionNoiseM", 0.20f,
            "R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)");
    out.tracker_birth_frames     = geti("Tracker.BirthFrames", 8,
            "frames a mask must stay unexplained before spawning a chair");
    out.tracker_birth_min_sep_m  = getf("Tracker.BirthMinSepM", 0.70f,
            "a birth must be ≥ this (m) from every existing chair (anti-dup)");
    out.birth_fragment_frac    = getf("Tracker.BirthFragmentFrac", 0.25f,
            "A birth candidate carrying less than this FRACTION of a nearby, much larger same-label mask is a FRAGMENT of it, not a new chair (YOLO splits one…");
    out.birth_fragment_reach_m = getf("Tracker.BirthFragmentReachM", 1.00f,
            "");
    out.tracker_merge_overlap    = getf("Tracker.MergeOverlap", 0.20f,
            "merge two instances whose seat footprints overlap ≥ this");
    out.exist_enabled            = getb("Existence.Enabled", true,
            "Existence.Enabled — use the log-odds belief (else the old prune)");

    // Level-2 arrangement prior (ring_metaconcept). Off ⇒ the belief is bit-for-bit pre-rig.
    out.rig_yaw_prior_enabled    = getb("RigPrior.Enabled", true,
            "master A/B switch");
    out.rig_yaw_kappa_max        = getf("RigPrior.YawKappaMax", 1.5f,
            "consumer cap; ChairBelief::kRigKappaMax enforces the same bound");
    out.rig_prior_stale_ms       = geti("RigPrior.StaleMs", 5000,
            "ignore a message no rig is refreshing; 0 disables");
    out.exist_birth_logodds      = getf("Existence.BirthLogodds", 1.0f,
            "Existence.BirthLogodds — L seeded at birth (a birth already needed birth_frames of evidence)");
    out.exist_remove_logodds     = getf("Existence.RemoveLogodds", -3.0f,
            "Existence.RemoveLogodds — remove when L falls below this");
    out.exist_remove_frames      = geti("Existence.RemoveFrames", 15,
            "Existence.RemoveFrames");
    out.exist_max_logodds        = getf("Existence.MaxLogodds", 4.0f,
            "Existence.MaxLogodds — saturation cap (a real chair can't earn infinite immunity)");
    out.exist_evidence_gain      = getf("Existence.EvidenceGain", 0.15f,
            "Existence.EvidenceGain — per-frame |ΔL| scale (occlusion tolerance = span/gain frames)");
    out.exist_expected_support_c = getf("Existence.ExpectedSupportC", 7000.f,
            "Existence.ExpectedSupportC — expected support scale: E[npts]=C/range² (boundary ≈ a handful of pts)");
    out.exist_adequacy_ref       = getf("Existence.AdequacyRef", 0.30f,
            "Existence.AdequacyRef — WON-mask support/expected below this → negative evidence");
    out.exist_adequacy_cap       = getf("Existence.AdequacyCap", 1.5f,
            "Existence.AdequacyCap — clamp so one dense frame can't over-confirm");
    out.exist_calib_adapt        = getf("Existence.CalibAdapt", 0.02f,
            "Existence.CalibAdapt — EWMA adapting C (0=OFF; a dense chair would poison a sparse one)");
    out.exist_vacate_confident_frames = geti("Existence.VacateConfidentFrames", 45,
            "Existence.VacateConfidentFrames — frames_since_detection at which an in-view");
    out.exist_occlusion_check    = getb("Existence.OcclusionCheck", true,
            "Existence.OcclusionCheck — suppress the vacate negative when the chair is hidden");
    out.exist_occlusion_margin_m = getf("Existence.OcclusionMarginM", 0.30f,
            "Existence.OcclusionMarginM — an occluder must be ≥ this much CLOSER to count");
    out.exist_room_prior         = getb("Existence.RoomPrior", true,
            "Existence.RoomPrior — enforce the room-containment pose prior");
    out.exist_room_margin_m      = getf("Existence.RoomMarginM", 0.40f,
            "Existence.RoomMarginM — tolerance a chair centre may sit OUTSIDE the walls");
    out.exist_out_of_room_gain   = getf("Existence.OutOfRoomGain", 1.5f,
            "Existence.OutOfRoomGain — |ΔL| per frame while outside (debounces a 1-frame glitch)");
    out.exist_zed_edge_offset    = getf("Existence.ZedEdgeOffset", 1.0f,
            "Existence.ZedEdgeOffset — normalised ROI offset at which ZED detectability→0");
    out.exist_zed_range_full     = getf("Existence.ZedRangeFull", 4.0f,
            "Existence.ZedRangeFull — within this range (m) ZED detects reliably (pd=1)");
    out.exist_zed_range_ref      = getf("Existence.ZedRangeRef", 7.0f,
            "Existence.ZedRangeRef — beyond this range (m) ZED absence is uninformative (pd=0)");
    out.exist_zed_clear_los_floor = getf("Existence.ZedClearLosFloor", 0.0f,
            "Existence.ZedClearLosFloor");   // 0 = pure pd (see chair_config.h)
    out.tracker_birth_seat_w     = getf("Tracker.BirthSeatW", 0.60f,
            "x extent — the span the backrest covers");
    out.tracker_birth_seat_d     = getf("Tracker.BirthSeatD", 0.52f,
            "y extent — front-to-back; backrest sits on the -y edge");
    out.tracker_birth_seat_h     = getf("Tracker.BirthSeatH", 0.595f,
            "seat TOP above the floor");
    out.tracker_birth_back_h     = getf("Tracker.BirthBackH", 0.655f,
            "backrest height above the seat top (→ top at 1.25)");
    out.tracker_birth_seat_thick = getf("Tracker.BirthSeatThick", 0.075f,
            "seat slab thickness (also sets the leg/seat z-band split)");
    out.tracker_birth_leg_half   = getf("Tracker.BirthLegHalf", 0.0375f,
            "square leg half-side");
    out.ai2_mode_obs_weighting   = getb("ChairModel.AI2ModeObsWeighting", true,
            "Weight each frame's yaw-mode vote by the information it carries (backrest mass × viewpoint novelty) instead of counting every frame equally. false =…");
    out.ai2_mode_sat_back_pts    = getf("ChairModel.AI2ModeSatBackPts", 60.0f,
            "backrest mass at which a frame is half-informative");
    out.ai2_view_budget          = getf("ChairModel.AI2ViewBudget", 3.0f,
            "Total mode-evidence weight one bearing bin may ever contribute (see ChairBeliefParams::view_budget)");
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
            "Bearing.ConfirmGateRad — bearing within this of a live chair's azimuth = 'explained'");
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

    std::print("chair_concept: configuration loaded.\n");
    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★z_top = seat top + backrest = 0.595 + 0.655 = 1.25 m, from the BIRTH template that actually runs.
    // ⚠The manifest declares 0.90 (a 0.45/0.45 split from chair_model.h). Those disagree, and adopt_span
    // PRINTS it rather than one of them winning quietly — which is the whole reason this call exists. Resolve
    // it by measuring these chairs, then set `from = "measured"`; do not silently align one to the other.
    {
        const auto span = rc::manifest::adopt_span(kManifestPath, "chair",
                                                  rc::manifest::Support::floor_anchored,
                                                  (out.tracker_birth_seat_h + out.tracker_birth_back_h), (out.tracker_birth_seat_h + out.tracker_birth_back_h));
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = (out.tracker_birth_seat_h + out.tracker_birth_back_h);
        decl.extent_m = (out.tracker_birth_seat_h + out.tracker_birth_back_h);
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body("chair ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] chair ✓ every derived z-band contains the declared body\n");
    }

    return out;
}

}  // namespace rc
