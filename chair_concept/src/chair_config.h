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
    // ── Level-2 arrangement prior (ring_metaconcept → chair, via the group_member edge) ──────────
    // A square seat makes yaw 4-fold ambiguous; the thin backrest often cannot break it, and a wrong
    // mode is STICKY (observed 2026-07-26: 570 cycles wrong, cleared only by restarting the agent).
    // The rig supplies the missing structural evidence ("a chair at this spot faces the table").
    bool  rig_yaw_prior_enabled = true;    // master A/B switch
    float rig_yaw_kappa_max     = 1.5f;    // consumer cap; ChairBelief::kRigKappaMax enforces the same bound
    int   rig_prior_stale_ms    = 5000;    // ignore a message no rig is refreshing; 0 disables

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
    // ── DETECTOR ENVELOPE (common/detectability) — the YOLO inverse model ─────────────────────────
    // ONE model, two consumers: the epistemic planner puts the stand-off at its argmax, and the removal
    // channel weights absence by it, so a missing mask from a pose the detector could never fire from
    // reads as EXPECTED rather than as evidence the object is gone. Defaults are the fleet PRIOR, not a
    // measurement, so behaviour is unchanged until etc/config.toml sets them. The envelope is genuinely
    // object-dependent — measured max_fill 1.32 (refrigerator) vs 0.677 (table) — so every agent needs
    // its own, fitted from its ai2_log with common/detectability/tools/fit_envelope (--label fresh).
    float detect_min_fill        = 0.10f;
    float detect_max_fill        = 0.60f;
    float detect_soft            = 0.06f;
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
    // ── "Be-still-to-update" as CONTINUOUS PRECISION (AIF-aligned; replaces the old hard gate) ──────────────
    // A frame's authority to MOVE the mean falls off smoothly with an UNRELIABILITY = motion × off-axis-position:
    // a still OR well-centred mask keeps authority (u→0); a moving AND peripheral mask loses it (u→1) so it only
    // CONFIRMS. "Confirmation-only" is the limit precision→0, not a branch — no threshold. The motion magnitude
    // combines the per-mask corruption speed (motion_dotd) with the robot's own measured ego-speed (transform
    // chain), and the off-axis penalty grows with the mask's centroid radius². Both enter the per-frame
    // common-mode (mot_*_var below) and the existence reliability weight.
    float ai2_ang_lever_m   = 2.0f;    // rad/s → m/s lever (tangential speed of a chair ~this far away) for ego-motion
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

    // ── FIXATION gate: the ATTENTION mechanism (a DELIBERATE, flagged threshold) ──────────────────────────
    // Same rationale as table_concept's (see TableConfig::fixation_enabled): every graded lever acts through
    // the per-frame common-mode Σc, and the engine saturates a frame's information at Σc⁻¹ — a NONZERO
    // asymptote — so a bad frame is attenuated but still moves the GN mean, and accumulation over hundreds of
    // frames beats attenuation. Live: chairs still repositioned from 6 m after the graded terms were added.
    // The model is EYE FIXATION: primates take in detail only during fixations and actively suppress intake
    // otherwise. Attention is precision, and an unattended channel's precision is ZERO — the observation is
    // not integrated. With no pan-tilt, a fixation is the whole-body condition: CLOSE + CENTRED + STILL.
    // Outside a fixation the cycle is predict-only (mean HELD, Σ inflates). Association and existence do not
    // read this, so the chair is still CONFIRMED and tracked — only its POSE is frozen.
    // Distinct from ai2_motion_confirm_only above, which required moving AND off-centre and had no range
    // condition at all (so a still robot 6 m away updated at full authority — the actual live failure).
    // CAVEAT by design: a chair never approached, centred and held still for keeps its birth pose.
    bool  fixation_enabled       = true;
    // RESOLVABLE (replaced the hard RANGE cut on 2026-07-30 — see ChairFitter::fixated). Range is only a
    // proxy; what decides whether a frame can resolve the pose is how much UN-CLUTTERED surface it puts on
    // the chair. The range cut rejected a dense 3.7 m stare while accepting a starved 2.4 m glance, and
    // blocked 400/400 cycles on all three real chairs while the robot was parked and staring.
    int   fixation_min_pts       = 150;    // min mask points for the frame to say anything about the pose
    float fixation_max_clutter   = 0.35f;  // max clutter fraction (surface the model cannot explain)
    float fixation_range_m       = 0.0f;   // RETIRED as a gate (0 = off). Kept so an A/B revert is one edit.
    float fixation_centre_frac   = 0.60f;  // CENTRED: max mask-centroid radius (focal-norm) — the "fovea".
    // ── A/B: is CENTRED a GATE or a PRECISION term? ───────────────────────────────────────────────
    // false (default) = today: outside the fovea the geometry update is INHIBITED entirely.
    // true            = opportunistic: an off-fovea object is still observed, at REDUCED precision
    //                   (R inflated by the existing periphery_penalty), so a fixation completed on one
    //                   object pays partial evidence to everything else in the frame — the table and the
    //                   chairs while the robot stands at the fridge. STILL and RESOLVABLE still gate:
    //                   they are the conditions a peripheral look shares, and CENTRED is the only one
    //                   that is genuinely target-specific.
    // ⚠WHY THE GATE EXISTS, and what this A/B is really testing: attenuation ALONE previously lost to
    // ACCUMULATION — "the graded common-mode terms saturate at a nonzero asymptote, so attenuation loses
    // over hundreds of frames (chairs still repositioned from 6 m)". The metric belief has no per-viewpoint
    // novelty budget across frames the way flip_acc_ does, so a peripheral frame repeated hundreds of times
    // can still walk the pose. Run this ON only with that failure in mind: the thing to watch is a chair
    // being repositioned from far away, not whether peripheral evidence arrives.
    bool  fixation_centre_precision = false;
    // Floor on the peripheral precision factor, so a fully off-axis look is worth LITTLE rather than
    // NOTHING. 1 = no peripheral attenuation at all; 0 = a fully peripheral frame carries no weight.
    float fixation_periph_floor = 0.05f;
                                           // 0.60 matches AI2PeriphRef (0.50 ≈ tan 27°, where the codebase's
                                           // own periphery penalty saturates); 0.35 was tighter than that.
    float fixation_still_dotd    = 0.05f;  // STILL: max |motion_dotd| = Z·‖ṡ‖ (m/s) ego-motion mask smear
    float fixation_still_lin_mps = 0.05f;  // STILL: max robot linear speed (m/s)
    float fixation_still_ang_radps = 0.10f;// STILL: max robot angular speed (rad/s) — the turn case

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
    // A birth candidate carrying less than this FRACTION of a nearby, much larger same-label mask is a
    // FRAGMENT of it, not a new chair (YOLO splits one chair into backrest+seat routinely). Compared
    // within one frame, so range cancels. reach = one chair-length: a piece lies within that of its
    // parent. 0 disables. See the note at the call site — the separation gate cannot catch this.
    float birth_fragment_frac    = 0.25f;
    float birth_fragment_reach_m = 1.00f;
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
    // ★Existence.RemoveFrames — the DEBOUNCE chair never had, in IDEAL OBSERVATIONS (Sum p_vis), the fleet
    // unit. chair was the only agent in the fleet that could delete an instance on a SINGLE frame: removal
    // was a bare `L < RemoveLogodds` with nothing behind it. That is not merely a missing safety margin —
    // OutOfRoomGain draws 1.5 nats EVERY frame regardless of visibility, so +4 -> -3 is under 5 frames, and
    // its own comment ("debounces a 1-frame glitch") shows the GAIN was being asked to do a debounce's job.
    // A localization wobble that puts a real chair outside the room polygon for five frames deleted it.
    // 15 is the fleet value; the decision now goes through rc::exist::decide_removal like every sibling.
    int   exist_remove_frames      = 15;     // Existence.RemoveFrames
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
    // "ZED removes, ricoh confirms": a ricoh WIN confirms existence (resets staleness) but a ricoh mask's ABSENCE
    // never removes — only ZED absence does, and only to the degree ZED would RELIABLY have detected a present
    // chair (falls off toward the image edge + with range). Stops removal of a far/peripheral, only-ricoh-visible
    // chair whose 360 view is momentarily occluded (the reported bug).
    float exist_zed_edge_offset = 1.0f;   // Existence.ZedEdgeOffset — normalised ROI offset at which ZED detectability→0
    float exist_zed_range_full  = 4.0f;   // Existence.ZedRangeFull — within this range (m) ZED detects reliably (pd=1)
    float exist_zed_range_ref   = 7.0f;   // Existence.ZedRangeRef — beyond this range (m) ZED absence is uninformative (pd=0)
    // pd FLOOR for an UNOCCLUDED, in-frustum chair. ★DEFAULT 0 SINCE 2026-07-29 — this floor was the live cause
    // of "chairs disappear when the robot isn't looking at them". It overrides zed_detectability()==0 (chair past
    // ZedRangeRef or at the image edge, i.e. the ZED provably CANNOT resolve it) and vacates anyway at the floor
    // rate; with conf saturated (since_det ≥ VacateConfidentFrames) that is a constant −g·floor every sensor frame
    // and L walks to RemoveLogodds with ZERO informative observations. Measured in chair_existence_log.csv: 53 of
    // 69 vacate-branch rows had zed_pd < 0.15 (51 of them exactly 0), and chair_2/chair_3 fell 3.09→1.74 and
    // 2.44→1.09 purely on the floor before being deleted — the real chairs the user reported.
    // It is also a hard threshold of exactly the kind CLAUDE.md forbids: "absence removes only to the degree the
    // sensor could have detected" IS the model, and the floor contradicts it. The glitch-stranded phantom it was
    // aimed at is already handled by the room-containment prior above (a mislocalised chair lands outside the
    // walls and draws exist_out_of_room_gain every frame regardless of visibility). A phantom that is genuinely
    // in-room and never resolvable should drive the robot to GO LOOK (table_concept's wants_verification pull),
    // not be deleted unseen. Set > 0 only to reinstate the old unseen-decay.
    float exist_zed_clear_los_floor = 0.0f;  // Existence.ZedClearLosFloor
    // ★TEMPLATE GEOMETRY — this is the chair the belief actually believes in. The pose-only belief does
    // NOT fit size, so any error here is a systematic model-vs-world mismatch that the clutter component
    // silently absorbs. 2026-07-27: these were 0.45 across the board against a real Webots SimpleChair of
    // seat 0.60x0.52 with its top at 0.595 and a backrest reaching 1.25. Two consequences, both measured:
    //   · responsibilities() gates the parts by z about tpl_seat_h, so with z_high=0.45 against a real seat
    //     top of 0.595, back_g = 1/(1+e^((0.45-0.595)/0.03)) = 0.992 — 99% of SEAT points were attributed to
    //     the BACKREST, turning it into a line detector across the seat plane and voting for a 90°-rotated
    //     mode. Fixing tpl_seat_h alone moved a chair's continuous yaw error from 98.1° to 1.0°.
    //   · 30-56% of each mask sat above the template's backrest top (0.90) and was dumped into flat clutter
    //     — the tallest, most yaw-informative surface in the scene was invisible to the model.
    // Also note the real seat is RECTANGULAR: the 4-fold yaw ambiguity resolve_orientation exists to fight
    // was largely an artefact of the square 0.45x0.45 template. Re-derive these for any new chair model.
    float tracker_birth_seat_w     = 0.60f;   // x extent — the span the backrest covers
    float tracker_birth_seat_d     = 0.52f;   // y extent — front-to-back; backrest sits on the -y edge
    float tracker_birth_seat_h     = 0.595f;  // seat TOP above the floor
    float tracker_birth_back_h     = 0.655f;  // backrest height above the seat top (→ top at 1.25)
    float tracker_birth_seat_thick = 0.075f;  // seat slab thickness (also sets the leg/seat z-band split)
    float tracker_birth_leg_half   = 0.0375f; // square leg half-side

    // ── Discrete yaw-mode evidence weighting (see ChairBeliefParams::mode_obs_weighting) ──────────
    // Weight each frame's yaw-mode vote by the information it carries (backrest mass × viewpoint
    // novelty) instead of counting every frame equally. false = the pre-2026-07-27 behaviour, kept
    // for A/B: that path let 21-point frames at 6.5 m rotate a chair resolved on 1700 points at 3.5 m.
    bool  ai2_mode_obs_weighting = true;
    float ai2_mode_sat_back_pts  = 60.0f;   // backrest mass at which a frame is half-informative
    // Total mode-evidence weight one bearing bin may ever contribute (see ChairBeliefParams::view_budget).
    // Replaces the old 1/(1+visits) novelty divisor so a deliberate fixation can actually settle the yaw.
    float ai2_view_budget        = 3.0f;
    bool  tracker_nll_cost         = false;   // association cost = ½(m²+ln|S|) NLL (vs raw m²); see InstanceTracker
    // ZED-only BIRTH gate: only a ZED slice (depth_var==0) may SPAWN a chair; a ricoh LiDAR-depth slice
    // (depth_var>0, unreliable depth/extent) may associate/confirm an existing chair but never birth a phantom.
    // A confident-ricoh escape hatch (OFF by default) permits a very-confident, low-variance ricoh birth.
    bool  ricoh_birth_enabled = false;   // Tracker.RicohBirthEnabled — allow a confident ricoh-depth slice to birth
    float ricoh_birth_conf    = 0.60f;   // Tracker.RicohBirthConf — min YOLO confidence for a ricoh birth
    float ricoh_birth_max_var = 0.005f;  // Tracker.RicohBirthMaxVar — max depth_var (m²) for a ricoh birth
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
