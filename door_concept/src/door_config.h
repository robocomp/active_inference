/*
 * door_config.h
 *
 * Plain-data configuration for the door_concept agent, plus a loader that fills
 * it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a new
 * concept agent can copy this file and edit only the keys it needs (mirrors
 * bottle_concept/bottle_config.h).
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct DoorConfig
{
    // Agent convergence
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   detection_alive_max_frames = 40; // cycles without a fresh door mask before detection_alive=false
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float min_standoff_m    = 1.8f;    // min stand-off floor for epistemic viewpoints (YOLO misses too-close doors)
    int   epistemic_cooldown_cycles = 200;    // min cycles withdrawn after satisfaction
    int   door_log_period_frames = 30;

    // Primary-input (masks) stream gate: no NEW masks frame for this many ms while Operating ⇒ demote
    // out of Operating rather than integrate stale evidence. 0 disables the gate. Mirrors table_concept.
    int   masks_stall_timeout_ms  = 3000;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;

    // DoorModel geometry / mask split: on-surface membership for the candidate/residual split in
    // DoorFitter::observe (a mask point within sdf_threshold_for_storage of the compound SDF is a candidate).
    float sigma_obs         = 0.05f;
    float sdf_threshold_for_storage = 0.08f;

    // ── AI2 belief (mirrors table_concept [TableModel].AI2*): recursive-Laplace full-covariance filter ──
    float ai2_sigma_base_m       = 0.03f;
    float ai2_clutter_frac       = 0.10f;
    float ai2_clutter_scale_m    = 0.12f;
    float ai2_clutter_structure_gain = 1.0f;  // density-aware clutter: shrink the clutter prior for seat-coplanar points (closes the escape valve → fixes seat_d collapse + its overconfidence). 0 → flat clutter
    float ai2_prior_size_std     = 0.15f;
    float ai2_process_std_m      = 0.005f;
    float ai2_process_std_yaw    = 0.01f;
    float ai2_process_std_size   = 0.0005f;  // rigid size DOFs ≪ pose (Tier-1: kills the vertical random walk)
    // Stale-belief aging (measurement-age → covariance). Nominal mask-stream period (s): with >0, when a door
    // is unseen Σ inflates by Q·(dt/this) on the agent's own clock (rc::ai::inflate_for_age) so a dead feed reads
    // downstream as growing uncertainty. <=0 DISABLES it → freeze-on-stale (historic). Mirrors table_concept.
    float ai2_age_nominal_dt_s   = 0.0f;
    float ai2_floor_z            = 0.0f;     // room-frame floor; cz pinned here (Tier-1: removes cz gauge freedom)
    float ai2_floor_std          = 0.03f;    // floor-height uncertainty (m) → common-mode z
    float ai2_seat_anchor_std    = 0.04f;    // seat-layer height anchor obs noise (m); 0 → off. Fixes seat_h gauge runaway
    float ai2_seat_anchor_band   = 0.12f;    // seat-layer mean-shift bandwidth (m) = seat vertical scale
    float ai2_seat_extent_std    = 0.02f;    // seat-layer footprint (seat_w/seat_d) span anchor obs noise (m); 0 → off. Fixes seat_d collapse
    float ai2_common_mode_pos_std  = 0.03f;
    float ai2_common_mode_size_std = 0.35f;   // per-frame SHARED size (w,h) error → caps how far one mask moves them (anti width-drift)
    float ai2_common_mode_yaw_std  = 0.03f;
    float ai2_range_noise_lat_per_m = 0.02f;   // static range → R + position common-mode (m per m)
    float ai2_range_noise_yaw_per_m = 0.03f;   // static range → yaw common-mode (rad per m)
    // Ego-motion "be-still-to-update" fixation (ported from table_concept motion_cm_*_gain; CONCEPT_AGENT_RECIPE
    // §"Belief invariants" → ego-motion common-mode). A moving frame's mask is ONE shared smear, so route it into
    // the per-frame COMMON-MODE (frame.chain_cov_*), NOT per-point R — per-point R averages the shared error away,
    // whereas the common-mode caps the frame's authority to MOVE the mean. Effect: geometry updates concentrate at
    // stillness; a frame captured while the robot moves only CONFIRMS. Continuous, no gate (0 at stillness).
    // Door is pose-only N=3 [cx,cy,yaw] → no size gain. std growth per m/s of motion_dotd; 0 disables.
    float motion_cm_pos_gain = 0.30f;   // position (cx,cy) shared-error std per m/s — anti-DRIFT  (0.10→0.30)
    // yaw is the one that BIT: "passing by the door rotated it". This term is 0 at stillness, so a large gain has
    // NO cost on a stationary/correctly-seen door — it only denies a moving frame the authority to rotate the mean.
    float motion_cm_yaw_gain = 0.50f;   // yaw shared-error std (rad) per m/s      — anti-ROTATE (0.12→0.50)
    // ── "Be-still-to-update" as CONTINUOUS PRECISION (AIF-aligned; replaces the old hard gate) ──────────────
    // A frame's authority to MOVE the mean falls off smoothly with an UNRELIABILITY = motion × off-axis-position:
    // a still OR well-centred mask keeps authority (u→0); a moving AND peripheral mask loses it (u→1) so it only
    // CONFIRMS. "Confirmation-only" is the limit precision→0, not a branch — no threshold. The motion magnitude
    // combines the per-mask corruption speed (motion_dotd) with the robot's own measured ego-speed (transform
    // chain), and the off-axis penalty grows with the mask's centroid radius². Both enter the per-frame
    // common-mode (mot_*_var below) and the existence reliability weight.
    float ai2_ang_lever_m   = 2.0f;    // rad/s → m/s lever (tangential speed of a door ~this far away) for ego-motion
    float ai2_periph_ref    = 0.50f;   // centroid radius (focal-norm, tan of off-axis angle) at which the periphery
                                       // penalty saturates to 1; 0.50 ≈ tan(27°). Smaller = only near-axis is "central".
    float ai2_motion_ref_mps = 0.60f;  // motion magnitude (m/s) at which a fully-peripheral frame becomes fully
                                       // unreliable (existence weight → 0). Larger = more tolerant of motion.
    // A/B FALLBACK — the OLD hard gate (predict-only when moving & off-centre). Default OFF; the continuous
    // precision above is the live path. Set true to compare. still_*/moving_update_center_radius feed only this.
    bool  ai2_motion_confirm_only = false;
    float ai2_still_lin_mps       = 0.05f;  // (hard gate) camera linear speed (m/s) below which robot counts as "still"
    float ai2_still_ang_radps     = 0.10f;  // (hard gate) camera angular speed (rad/s) below which robot counts as "still"
    float ai2_still_dotd          = 0.05f;  // (hard gate) per-mask ego-motion corruption speed (m/s) still-level
    float ai2_moving_update_center_radius = 0.35f;  // (hard gate) mask centroid radius below which a moving update is allowed
    // Obliquity yaw cap (TABLE.md §6): the backrest (the door's yaw-carrying surface) is a vertical plate, so
    // a view that grazes it edge-on can barely observe yaw. Grow the SHARED yaw variance as 1/obliquity_cos−1
    // so a grazing frame confirms the door but can't rotate a converged one. Continuous covariance, no gate.
    // ⚠ Starts at table's value; the surface geometry differs (vertical backrest vs horizontal top) so treat
    // it as UNVALIDATED for door and re-tune from the logged obliquity_cos before trusting it. 0 = OFF.
    float ai2_obliquity_yaw_gain = 0.05f;
    // Ego-motion reliability of the discrete orientation vote: w = 1/(1+(dotd/ref)²). A smeared/moving frame
    // (large motion_dotd) barely votes on the 4-way mode — the observed 180° flips arrived on motion frames.
    float ai2_orientation_motion_ref = 0.50f;
    // FE-surprise attention baseline (TABLE.md §9): asymmetric EMA (down fast = consolidate a better fit; up
    // slow = a sustained rise, the door moved, stays surprising) + a smoothed positive gap = the surprise.
    float ai2_fe_baseline_adapt_down = 0.05f;
    float ai2_fe_baseline_adapt_up   = 0.005f;
    float ai2_fe_surprise_smooth     = 0.10f;
    float ai2_trunc_gate_frac    = 0.10f;
    int   ai2_gn_iters           = 4;
    float ai2_extent_std         = 0.05f;   // extent-observation noise (m) for the coverage/extent likelihood
    std::string ai2_csv_path     = "";

    // Upload the door pose covariance onto the room→door RT edge (rt_covariance_att, 6×6 SE3), built
    // from the belief's full Σ over [cx,cy,cz,yaw,...]: x←cx, y←cy, z←cz, yaw←ψ; roll/pitch are
    // unobservable (large). rt_cov_scale calibrates the raw variance toward NEES≈1.
    bool  rt_cov_upload = true;
    float rt_cov_scale  = 1.0f;
    bool  rt_cov_add_chain = true;   // Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov

    // ── Multi-instance birth/associate/death tracker (shared rc::InstanceTracker) ──────────────────
    // "door" masks are associated to instances by a covariance-gated 1-to-1 (cov from the belief Σ),
    // a persistently-unexplained mask spawns a new door, and (if death_enabled) an unobserved instance
    // is retired. Doors are persistent furniture: death OFF by default; birth_min_sep wide. Mirrors table.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.40f;   // metric XY gate (m) before an instance has a usable covariance
    float tracker_detection_noise_m = 0.20f;  // R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)
    int   tracker_birth_frames     = 8;       // frames a mask must stay unexplained before spawning a door
    int   tracker_death_frames     = 300;     // frames an instance may go unobserved before retirement
    bool  tracker_death_enabled    = false;   // OFF: a door is persistent furniture — removed only by MERGE
    float tracker_birth_min_sep_m  = 0.70f;   // a birth must be ≥ this (m) from every existing door (anti-dup)
    float tracker_merge_overlap    = 0.20f;   // merge two instances whose seat footprints overlap ≥ this
                                              // fraction of the smaller, keeping the more-observed. 0 disables.
    bool  tracker_prune_enabled        = true; // stillbirth prune of phantom duplicates born from churn
    int   tracker_prune_maturity_cycles = 90;  // probation window; older instances are permanent furniture
    int   tracker_prune_patience       = 30;   // consecutive tracker-unassigned cycles in probation → prune
    // ── Existence belief (shared rc::exist log-odds channel — the same one table/chair use) ─────────────
    // Evidence is the PIXEL-LEVEL silhouette (DoorFitter::compute_silhouette_existence): occupancy confirms,
    // absence removes, occlusion and out-of-frustum HOLD. The parameters below are physical sensor RATES and
    // one decision boundary — not gains, ramps or detectability curves. Removed with the old scheme:
    // RemoveLogodds / EvidenceGain / ExpectedSupportC / AdequacyRef / AdequacyCap / CalibAdapt /
    // VacateConfidentFrames / OcclusionCheck / ZedEdgeOffset / ZedRangeFull / ZedRangeRef / ZedClearLosFloor
    // (the last of which was parsed but never read). Detectability is now measured from the projection itself.
    bool  exist_enabled            = true;   // Existence.Enabled — use the log-odds belief (else the old prune)
    float exist_birth_logodds      = 1.0f;   // Existence.BirthLogodds — L seeded at birth (a birth already needed birth_frames of evidence)
    float exist_max_logodds        =  4.0f;  // Existence.MaxLogodds — saturation cap (a real door can't earn infinite immunity)
    // The ONE honest decision boundary: remove when P(exists) drops below this. Replaces RemoveLogodds — same
    // quantity, stated as a probability instead of a log-odds so it reads as the decision it is.
    float exist_removal_prob       = 0.12f;  // Existence.RemovalProb
    int   exist_remove_frames      = 15;     // Existence.RemoveFrames — consecutive EVIDENCE cycles the decision must hold
    // Physical sensor rates for the log-likelihood ratio (rc::exist::SensorModel), NOT gates.
    float exist_detection_prob     = 0.85f;  // Existence.DetectionProb — P(mask lights a predicted pixel | door present & observable)
    float exist_clutter_prob       = 0.05f;  // Existence.ClutterProb   — P(mask lights a predicted pixel | no door)
    float exist_sensor_sigma_m     = 0.03f;  // Existence.SensorSigmaM  — range/localisation noise σ
    // Central-image box [f, 1-f]²: a detectable silhouette sample inside it counts as the robot LOOKING at the
    // door. A door merely clipping the wide frustum edge is not a verifying view, so its absence barely removes.
    float exist_central_region_frac = 0.25f; // Existence.CentralRegionFrac
    float exist_occlusion_margin_m = 0.30f;  // Existence.OcclusionMarginM — an occluder must be ≥ this much CLOSER to count
    // Room-containment pose prior: P(a door outside the room walls) ≈ 0. Applies even when the instance is out
    // of view / behind a wall (a localization glitch can birth a door outside; the sensor then can't reach it
    // to vacate it). An out-of-room instance draws this STRONG negative log-odds each frame → removed in a few.
    bool  exist_room_prior         = true;   // Existence.RoomPrior — enforce the room-containment pose prior
    float exist_room_margin_m      = 0.40f;  // Existence.RoomMarginM — tolerance a door centre may sit OUTSIDE the walls
    float exist_out_of_room_gain   = 1.5f;   // Existence.OutOfRoomGain — |ΔL| per frame while outside (debounces a 1-frame glitch)
    // ── MINIMUM-HEIGHT prior ───────────────────────────────────────────────────────────────────────
    // A door is an aperture a person walks THROUGH: P(door | it tops out below ~1.8 m) ≈ 0. Same kind of
    // categorical prior as RoomPrior above (a fact about what a door IS, not a tuned detectability curve),
    // and applied the same way: suppress the BIRTH, and draw a strong negative on an instance that is
    // confidently too short. It MUST be judged on the observed support top, never on the fitted h — the
    // template anchor pins h at 2.0 m regardless of evidence, so a test on h can never fire. See
    // DoorInstance::obs_top_z. MinConf guards against acting on a handful of clipped views.
    bool  exist_min_height_prior = true;     // Existence.MinHeightPrior
    float exist_min_height_m     = 1.80f;    // Existence.MinHeightM — a door's support must reach this (m)
    float exist_min_height_conf  = 0.30f;    // Existence.MinHeightConf — untruncated-evidence weight required
                                             // before the prior may act (0 = act on the first clean view)
    float exist_short_gain       = 1.5f;     // Existence.ShortGain — |ΔL| per frame while confidently short
    // ── Identity re-acquisition ────────────────────────────────────────────────────────────────────
    // A removed door is remembered as a GHOST (name + converged belief). A later detection landing within
    // ReacquireRadiusM of a ghost is the SAME door coming back: it resumes that identity and its accumulated
    // geometry instead of being re-born as door_N+1. Without this, one flicker cost the live run three node
    // identities for a single physical door (door_1 → door_3, same 2 cm spot) and every downstream consumer
    // saw a brand-new object each time.
    float exist_reacquire_radius_m = 0.60f;  // Existence.ReacquireRadiusM (0 disables re-acquisition)
    int   exist_ghost_max          = 8;      // Existence.GhostMax — most recent removals retained
    // ── Openable door: APERTURE / LEAF decomposition (M0 = structure only) ─────────────────────────
    // A door is an APERTURE (a static hole in a wall) plus a LEAF (a rigid panel hinged on one of its
    // vertical edges) — see door_geometry.h. M0 introduces the decomposition with phi PINNED at 0, so the
    // leaf is flush in the aperture and behaviour is unchanged; phi becomes a fitted DOF in M1 and the
    // hinge side / swing direction become discrete hypotheses in M2. This flag does NOT enable estimation:
    // with it ON, phi is a CONSTANT read from PhiInitRad — a structural smoke test (open the door in the
    // model and confirm the existence channel no longer deletes it), not an inference.
    // Read in exactly one place, DoorFitter::make_belief_params, so the M0 pin cannot drift.
    bool  openable_enabled     = false;      // Openable.Enabled — false ⇒ phi is the literal 0.0f everywhere
    float openable_phi_init    = 0.0f;       // Openable.PhiInitRad — constant opening angle (rad)
    int   openable_hinge_side  = 0;          // Openable.HingeSide — 0 = near (s) edge, 1 = far (s+w) edge
    float openable_swing_dir   = 1.0f;       // Openable.SwingDir — +1 / −1: side of the wall it opens toward
    float openable_phi_max_rad = 1.5707963f; // Openable.PhiMaxRad — physical hinge travel limit (M1 uses it)
    // ── Door panel priors (wall-frame belief θ=[s,w,h]) ────────────────────────────────────────────
    // The door is a thin panel IN a wall: s (along-wall offset) is localised by the fit (BROAD prior),
    // while w,h are STRONG template priors — a standard leaf ≈ 0.70 m × 2.00 m. Realised as tight seed
    // Σ + tiny process noise (rc::ai's prior is temporal), so w,h stay near template unless sustained
    // consistent evidence moves them. thickness is fixed (across-wall extent, not a DOF). No gate.
    float door_prior_w_m   = 0.70f;   // width prior mean (m)
    float door_prior_w_std = 0.06f;   // width prior std (m) — strong
    float door_prior_h_m   = 2.00f;   // height prior mean (m)
    float door_prior_h_std = 0.08f;   // height prior std (m) — strong
    float door_prior_s_std = 0.60f;   // along-wall offset prior std (m) — broad (the fit localises s)
    float door_thickness_m = 0.05f;   // fixed panel thickness (m, across the wall)
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
    // ZED-only BIRTH gate: only a ZED slice (depth_var==0) may SPAWN a door; a ricoh LiDAR-depth slice
    // (depth_var>0, unreliable depth/extent) may associate/confirm an existing door but never birth a phantom.
    // A confident-ricoh escape hatch (OFF by default) permits a very-confident, low-variance ricoh birth.
    bool  ricoh_birth_enabled = false;   // Tracker.RicohBirthEnabled — allow a confident ricoh-depth slice to birth
    float ricoh_birth_conf    = 0.60f;   // Tracker.RicohBirthConf — min YOLO confidence for a ricoh birth
    float ricoh_birth_max_var = 0.005f;  // Tracker.RicohBirthMaxVar — max depth_var (m²) for a ricoh birth
    // ── RGB-360 bearing-only hypothesis birth (Part C-birth; RICOH_360_PERIPHERAL_DETECTION.md) ──────
    // A peripheral 360 "door" bearing (a no-depth mask slice, azimuth calibrated 2026-07-04) that matches
    // no live door and PERSISTS births a BROAD-Σ hypothesis: the mean is placed at a nominal range on the
    // ray, but Σ is huge ALONG the ray (range unknown) and tight ACROSS it (bearing known). The hypothesis
    // authors an Orient affordance (rotate to look); a depth mask then collapses Σ, or it dies unobserved.
    // Default OFF — the whole glance→orient loop is opt-in.
    bool  bearing_birth_enabled    = false;   // Bearing.BirthEnabled
    float bearing_confirm_gate_rad = 0.17f;   // Bearing.ConfirmGateRad — bearing within this of a live door's azimuth = "explained"
    int   bearing_birth_frames     = 8;       // Bearing.BirthFrames — unmatched-bearing streak before promotion
    float bearing_match_rad        = 0.17f;   // Bearing.MatchRad — candidate↔bearing azimuth match tolerance
    int   bearing_max_miss         = 4;       // Bearing.MaxMiss — streak gap tolerance (intermittent 360 detection)
    float bearing_nominal_range_m  = 2.0f;    // Bearing.NominalRangeM — where the mean starts on the ray (Σ carries the real uncertainty)
    float bearing_along_std_m      = 3.0f;    // Bearing.AlongStdM — Σ std ALONG the ray (unknown range)
    float bearing_across_std_m     = 0.30f;   // Bearing.AcrossStdM — Σ std ACROSS the ray (bearing known)
    float bearing_yaw_std_rad      = 3.14f;   // Bearing.YawStdRad — orientation fully unknown at birth
};

// Fill a DoorConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
DoorConfig load_door_config(const ConfigLoader& cfg);

}  // namespace rc
