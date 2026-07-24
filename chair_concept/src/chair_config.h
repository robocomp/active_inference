/*
 * chair_config.h
 *
 * Plain-data configuration for the chair_concept agent, plus a loader that fills
 * it from a RoboComp ConfigLoader. Kept separate from SpecificWorker so a new
 * concept agent can copy this file and edit only the keys it needs (mirrors
 * bottle_concept/bottle_config.h).
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct ChairConfig
{
    // Agent convergence
    float state_eps         = 0.04f;   // Σ|Δstate| threshold between cycles for convergence (m+rad)
    int   K_stable          = 30;
    int   detection_alive_max_frames = 40; // cycles without a fresh chair mask before detection_alive=false
    float obs_distance      = 1.8f;    // d_obs for epistemic planner
    float min_standoff_m    = 1.8f;    // min stand-off floor for epistemic viewpoints (YOLO misses too-close chairs)
    int   epistemic_cooldown_cycles = 200;    // min cycles withdrawn after satisfaction
    int   chair_log_period_frames = 30;

    // Primary-input (masks) stream gate: no NEW masks frame for this many ms while Operating ⇒ demote
    // out of Operating rather than integrate stale evidence. 0 disables the gate. Mirrors table_concept.
    int   masks_stall_timeout_ms  = 3000;
    int   voxel_bank_max_points = 4000;
    float voxel_bank_quantization_m = 0.02f;
    float voxel_select_radius_margin_m = 0.50f;
    float voxel_select_height_margin_m = 0.25f;

    // ChairModel geometry / mask split: on-surface membership for the candidate/residual split in
    // ChairFitter::observe (a mask point within sdf_threshold_for_storage of the compound SDF is a candidate).
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
    // Stale-belief aging (measurement-age → covariance). Nominal mask-stream period (s): with >0, when a chair
    // is unseen Σ inflates by Q·(dt/this) on the agent's own clock (rc::ai::inflate_for_age) so a dead feed reads
    // downstream as growing uncertainty. <=0 DISABLES it → freeze-on-stale (historic). Mirrors table_concept.
    float ai2_age_nominal_dt_s   = 0.0f;
    float ai2_floor_z            = 0.0f;     // room-frame floor; cz pinned here (Tier-1: removes cz gauge freedom)
    float ai2_floor_std          = 0.03f;    // floor-height uncertainty (m) → common-mode z
    float ai2_seat_anchor_std    = 0.04f;    // seat-layer height anchor obs noise (m); 0 → off. Fixes seat_h gauge runaway
    float ai2_seat_anchor_band   = 0.12f;    // seat-layer mean-shift bandwidth (m) = seat vertical scale
    float ai2_seat_extent_std    = 0.02f;    // seat-layer footprint (seat_w/seat_d) span anchor obs noise (m); 0 → off. Fixes seat_d collapse
    float ai2_common_mode_pos_std  = 0.03f;
    float ai2_common_mode_size_std = 0.02f;
    float ai2_common_mode_yaw_std  = 0.03f;
    float ai2_range_noise_lat_per_m = 0.02f;   // static range → R + position common-mode (m per m)
    float ai2_range_noise_yaw_per_m = 0.03f;   // static range → yaw common-mode (rad per m)
    // Ego-motion "be-still-to-update" fixation (ported from table_concept motion_cm_*_gain; CONCEPT_AGENT_RECIPE
    // §"Belief invariants" → ego-motion common-mode). A moving frame's mask is ONE shared smear, so route it into
    // the per-frame COMMON-MODE (frame.chain_cov_*), NOT per-point R — per-point R averages the shared error away,
    // whereas the common-mode caps the frame's authority to MOVE the mean. Effect: geometry updates concentrate at
    // stillness; a frame captured while the robot moves only CONFIRMS. Continuous, no gate (0 at stillness).
    // Chair is pose-only N=3 [cx,cy,yaw] → no size gain. std growth per m/s of motion_dotd; 0 disables.
    float motion_cm_pos_gain = 0.30f;   // position (cx,cy) shared-error std per m/s — anti-DRIFT  (0.10→0.30)
    // yaw is the one that BIT: "passing by the chair rotated it". This term is 0 at stillness, so a large gain has
    // NO cost on a stationary/correctly-seen chair — it only denies a moving frame the authority to rotate the mean.
    float motion_cm_yaw_gain = 0.50f;   // yaw shared-error std (rad) per m/s      — anti-ROTATE (0.12→0.50)
    // HARD "be-still-to-update" gate (the agreed invariant): ONLY a still/almost-still robot with a fresh ZED YOLO
    // mask may alter pose/shape; every other frame is CONFIRMATION-of-existence only (predict, no mean/shape move).
    // The common-mode above softens the near-still regime; this makes the moving regime EXACT (zero authority).
    bool  ai2_motion_confirm_only = true;   // moving robot → predict-only (no pose/shape update)
    float ai2_still_lin_mps       = 0.05f;  // camera linear speed (m/s) below which the robot counts as "still"
    float ai2_still_ang_radps     = 0.10f;  // camera angular speed (rad/s) below which the robot counts as "still"
    float ai2_still_dotd          = 0.05f;  // per-mask ego-motion corruption speed (m/s) still-level (OR'd in)
    // Moving-robot EXCEPTION: a mask WELL-CENTRED in the image (near the principal point) has minimal motion smear
    // and no peripheral distortion, so it is trustworthy enough to update pose/shape EVEN while moving. Allow the
    // update when the mask centroid radius is below this. UNITS = focal-normalised (tan of the ray angle off the
    // optical axis): 0.35 ≈ tan(19°), i.e. the centroid within ~19° of the axis; ~1.0 ≈ the edge of a ~90° FoV.
    // Lower = stricter (only near-axis masks may update while moving). <0 disables the exception.
    float ai2_moving_update_center_radius = 0.35f;
    // Obliquity yaw cap (TABLE.md §6): the backrest (the chair's yaw-carrying surface) is a vertical plate, so
    // a view that grazes it edge-on can barely observe yaw. Grow the SHARED yaw variance as 1/obliquity_cos−1
    // so a grazing frame confirms the chair but can't rotate a converged one. Continuous covariance, no gate.
    // ⚠ Starts at table's value; the surface geometry differs (vertical backrest vs horizontal top) so treat
    // it as UNVALIDATED for chair and re-tune from the logged obliquity_cos before trusting it. 0 = OFF.
    float ai2_obliquity_yaw_gain = 0.05f;
    // Ego-motion reliability of the discrete orientation vote: w = 1/(1+(dotd/ref)²). A smeared/moving frame
    // (large motion_dotd) barely votes on the 4-way mode — the observed 180° flips arrived on motion frames.
    float ai2_orientation_motion_ref = 0.50f;
    // FE-surprise attention baseline (TABLE.md §9): asymmetric EMA (down fast = consolidate a better fit; up
    // slow = a sustained rise, the chair moved, stays surprising) + a smoothed positive gap = the surprise.
    float ai2_fe_baseline_adapt_down = 0.05f;
    float ai2_fe_baseline_adapt_up   = 0.005f;
    float ai2_fe_surprise_smooth     = 0.10f;
    float ai2_trunc_gate_frac    = 0.10f;
    int   ai2_gn_iters           = 4;
    float ai2_extent_std         = 0.05f;   // extent-observation noise (m) for the coverage/extent likelihood
    std::string ai2_csv_path     = "";

    // Upload the chair pose covariance onto the room→chair RT edge (rt_covariance_att, 6×6 SE3), built
    // from the belief's full Σ over [cx,cy,cz,yaw,...]: x←cx, y←cy, z←cz, yaw←ψ; roll/pitch are
    // unobservable (large). rt_cov_scale calibrates the raw variance toward NEES≈1.
    bool  rt_cov_upload = true;
    float rt_cov_scale  = 1.0f;
    bool  rt_cov_add_chain = true;   // Part B: add the localization/chain cov J·Σ_chain·Jᵀ to the published RT cov

    // ── Multi-instance birth/associate/death tracker (shared rc::InstanceTracker) ──────────────────
    // "chair" masks are associated to instances by a covariance-gated 1-to-1 (cov from the belief Σ),
    // a persistently-unexplained mask spawns a new chair, and (if death_enabled) an unobserved instance
    // is retired. Chairs are persistent furniture: death OFF by default; birth_min_sep wide. Mirrors table.
    float tracker_gate_mahalanobis = 9.0f;    // χ²₂ gate (~3σ) for a mask↔instance match once it has a cov
    float tracker_gate_fallback_m  = 0.40f;   // metric XY gate (m) before an instance has a usable covariance
    float tracker_detection_noise_m = 0.20f;  // R in the association innovation cov S=P+R²I (≥ centroid-vs-fit offset)
    int   tracker_birth_frames     = 8;       // frames a mask must stay unexplained before spawning a chair
    int   tracker_death_frames     = 300;     // frames an instance may go unobserved before retirement
    bool  tracker_death_enabled    = false;   // OFF: a chair is persistent furniture — removed only by MERGE
    float tracker_birth_min_sep_m  = 0.70f;   // a birth must be ≥ this (m) from every existing chair (anti-dup)
    float tracker_merge_overlap    = 0.20f;   // merge two instances whose seat footprints overlap ≥ this
                                              // fraction of the smaller, keeping the more-observed. 0 disables.
    bool  tracker_prune_enabled        = true; // stillbirth prune of phantom duplicates born from churn
    int   tracker_prune_maturity_cycles = 90;  // probation window; older instances are permanent furniture
    int   tracker_prune_patience       = 30;   // consecutive tracker-unassigned cycles in probation → prune
    // ── Existence belief (continuous log-odds removal; replaces the wall-clock stillbirth prune above) ──
    // A phantom accumulates NEGATIVE existence evidence every SENSOR frame it is expected-visible yet
    // under-supported (silhouette-contradiction / free-space), and is removed when its log-odds drops below
    // the floor — regardless of age. A real chair keeps its log-odds pinned at the cap by being explained.
    // Expected support scales as E[npts] = ExpectedSupportC / range² (a chair's silhouette solid angle);
    // ExpectedSupportC ≈ npts·range² of a well-seen chair (measured ~7000 for the webots dining set).
    bool  exist_enabled            = true;   // Existence.Enabled — use the log-odds belief (else the old prune)
    float exist_birth_logodds      = 1.0f;   // Existence.BirthLogodds — L seeded at birth (a birth already needed birth_frames of evidence)
    float exist_remove_logodds     = -3.0f;  // Existence.RemoveLogodds — remove when L falls below this
    float exist_max_logodds        =  4.0f;  // Existence.MaxLogodds — saturation cap (a real chair can't earn infinite immunity)
    float exist_evidence_gain      = 0.15f;  // Existence.EvidenceGain — per-frame |ΔL| scale (occlusion tolerance = span/gain frames)
    float exist_expected_support_c = 2500.f; // Existence.ExpectedSupportC — expected support scale: E[npts]=C/range² (boundary ≈ a handful of pts)
    float exist_adequacy_ref       = 0.25f;  // Existence.AdequacyRef — WON-mask support/expected below this → negative evidence
    float exist_adequacy_cap       = 1.5f;   // Existence.AdequacyCap — clamp so one dense frame can't over-confirm
    float exist_calib_adapt        = 0.0f;   // Existence.CalibAdapt — EWMA adapting C (0=OFF; a dense chair would poison a sparse one)
    int   exist_vacate_confident_frames = 45; // Existence.VacateConfidentFrames — frames_since_detection at which an in-view,
                                              // unexplained instance draws FULL vacate negative (ramps 0→1; grace vs death-spiral)
    bool  exist_occlusion_check    = true;   // Existence.OcclusionCheck — suppress the vacate negative when the chair is hidden
    float exist_occlusion_margin_m = 0.30f;  // Existence.OcclusionMarginM — an occluder must be ≥ this much CLOSER to count
    // Room-containment pose prior: P(a chair outside the room walls) ≈ 0. Applies even when the instance is out
    // of view / behind a wall (a localization glitch can birth a chair outside; the sensor then can't reach it
    // to vacate it). An out-of-room instance draws this STRONG negative log-odds each frame → removed in a few.
    bool  exist_room_prior         = true;   // Existence.RoomPrior — enforce the room-containment pose prior
    float exist_room_margin_m      = 0.40f;  // Existence.RoomMarginM — tolerance a chair centre may sit OUTSIDE the walls
    float exist_out_of_room_gain   = 1.5f;   // Existence.OutOfRoomGain — |ΔL| per frame while outside (debounces a 1-frame glitch)
    float tracker_birth_seat_w     = 0.45f;   // seed seat width/depth/heights for a freshly born chair node
    float tracker_birth_seat_d     = 0.45f;
    float tracker_birth_seat_h     = 0.45f;
    float tracker_birth_back_h     = 0.45f;
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
    // ── RGB-360 bearing-only hypothesis birth (Part C-birth; RICOH_360_PERIPHERAL_DETECTION.md) ──────
    // A peripheral 360 "chair" bearing (a no-depth mask slice, azimuth calibrated 2026-07-04) that matches
    // no live chair and PERSISTS births a BROAD-Σ hypothesis: the mean is placed at a nominal range on the
    // ray, but Σ is huge ALONG the ray (range unknown) and tight ACROSS it (bearing known). The hypothesis
    // authors an Orient affordance (rotate to look); a depth mask then collapses Σ, or it dies unobserved.
    // Default OFF — the whole glance→orient loop is opt-in.
    bool  bearing_birth_enabled    = false;   // Bearing.BirthEnabled
    float bearing_confirm_gate_rad = 0.17f;   // Bearing.ConfirmGateRad — bearing within this of a live chair's azimuth = "explained"
    int   bearing_birth_frames     = 8;       // Bearing.BirthFrames — unmatched-bearing streak before promotion
    float bearing_match_rad        = 0.17f;   // Bearing.MatchRad — candidate↔bearing azimuth match tolerance
    int   bearing_max_miss         = 4;       // Bearing.MaxMiss — streak gap tolerance (intermittent 360 detection)
    float bearing_nominal_range_m  = 2.0f;    // Bearing.NominalRangeM — where the mean starts on the ray (Σ carries the real uncertainty)
    float bearing_along_std_m      = 3.0f;    // Bearing.AlongStdM — Σ std ALONG the ray (unknown range)
    float bearing_across_std_m     = 0.30f;   // Bearing.AcrossStdM — Σ std ACROSS the ray (bearing known)
    float bearing_yaw_std_rad      = 3.14f;   // Bearing.YawStdRad — orientation fully unknown at birth
};

// Fill a ChairConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
ChairConfig load_chair_config(const ConfigLoader& cfg);

}  // namespace rc
