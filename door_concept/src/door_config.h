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


// ─── the three PRAGMATIC door affordances: approach / open / cross ────────────────────────────────
// The tunables, all PHYSICAL: a body width, a reach, two stand-offs, plus the OFFER POLICY (the one
// decision boundary, stated as the probability it actually is, with a Schmitt band). The model that
// turns these into preconditions is door_pragmatics.h; read its header before changing any of them.
struct DoorPragmaticCfg
{
    bool  enabled              = true;
    // ⚠ApproachStandoffM IS GONE, not renamed. Where to stand is no longer a configured distance: it is
    // the MIDDLE of the actuation band, whose near edge is derived from this door's own width (the leaf's
    // swept arc) and this robot's body. A fixed 1.00 m was inside the arc of a 1.00 m leaf — the robot
    // would have been struck by the door it asked to be opened — and no single number can be right for
    // the next door of a different width. A key that is read and ignored looks exactly like one that
    // works, so it is deleted rather than left inert.
    // FAR edge of the actuation band, from the aperture plane. The NEAR edge is derived (safe_standoff).
    float actuation_reach_m    = 2.00f;
    float cross_standoff_m     = 1.20f;  // how far PAST the aperture the cross target sits
    float cross_clearance_m    = 0.80f;  // distance past the aperture plane that counts as fully through
    // ★A MEASUREMENT OF THE ROBOT, not a tuning knob — and the one direction in which a doorway closes.
    // Shadow's lateral half-width measured off the robocomp meshes is 0.2938 m per side (0.588 m across);
    // the hull compiled into rc::RobotFootprint::shadow() is 22.2 mm NARROWER per side than the mesh it
    // claims to come from, and on P3Bot it is a different body altogether. So this belongs in a
    // [Platform.<robot>] overlay per the fleet config convention, NOT in a shared default.
    // See common/robot_footprint/mesh_hull.h and ROBOT_GEOMETRY.md.
    // ⚠The default below is Shadow's and is WRONG for any other base.
    float robot_passage_width_m = 0.60f;
    float passage_margin_m      = 0.10f; // clearance demanded on top of the body width
    // Pose-uncertainty floor for door_reach_prob. Used when the pose-chain covariance is unavailable
    // (RtCovAddChain off, or no mask fitted yet): a probability computed with σ == 0 would be a step
    // function, i.e. exactly the distance threshold this design exists to avoid.
    float pose_sigma_floor_m    = 0.10f;
    // ★CLEARANCE FOR THE LEAF'S OWN SWEPT ARC — a SAFETY quantity, not a preference.
    // A leaf hinged on a vertical aperture edge sweeps a circle of radius equal to its own width, so at
    // 90 degrees it reaches a full `w` out from the wall plane. If it opens TOWARD the robot, anything
    // standing closer than that is inside the arc and gets struck by the door it just asked to be opened.
    // ★AND WE DO NOT KNOW WHICH WAY IT OPENS. `LeafState::swing` is never fitted — door_fitter.cpp sets
    // it once from config and it stays there — so "it opens away from me" is an assumption with nothing
    // behind it. The only safe reading of an unestimated direction is that it could come at us.
    // Hence the minimum stand-off is DERIVED, not configured: aperture width + half the robot's body +
    // this margin. Configuring it as a distance would be silently wrong the first time a wider door
    // appeared; deriving it cannot be.
    float swing_clearance_m     = 0.20f;
    // ★THE PROVIDER'S SWING RATE, rad/s — and it is DUPLICATED, which is a defect this comment exists to
    // make visible rather than hide. When this agent asks for a door to be opened it predicts the leaf
    // angle as (rate x elapsed) so it is not surprised by its own action; if that prediction is FASTER
    // than the real hinge, the agent believes the door open while it is still swinging. It used to be a
    // literal 1.2f in request_door_actuation with a comment saying "matches the bridge's SwingRate" —
    // which is a coupling held together by somebody remembering.
    // ⚠MUST MATCH webots-bridge [DoorControl] SwingRate. The real fix is for the provider to ADVERTISE it:
    // DoorControl.idsl's Capabilities carries typicalLatencySec and nothing about rate, so there is
    // currently nothing to read. Extending the interface means regenerating the Ice stubs on both sides.
    float actuation_swing_rate = 0.5f;   // rad/s; 90 deg in ~3.1 s
    // ★THE WIDTH OF THE LEAF-ANGLE BELIEF WHEN THE IMAGE IS NOT RESOLVING IT. estimate_phi() refills
    // phi_curve only from a fresh mask, and this agent's own [DoorConcept] notes record that the ADE20K
    // posterior over a plainly visible closed door collapses 0.995 → 0.048 as the robot CLOSES on it —
    // so the curve is typically absent at exactly the range from which `open` has to be taken. On those
    // cycles P(passable) is marginalised over N(phi_est, this) instead of over the curve: same question,
    // same absence of any angle threshold, weights from the belief rather than from the image. It is a
    // PRIOR WIDTH, not a measurement — 0 disables the fallback and `open`/`cross` then appear only while
    // the door is actively segmented. Published as door_open_prob_from_curve so the two never blur.
    float phi_sigma_rad         = 0.25f;

    // ── VALUE: what the consumer ranks these on. POLICY PLACEHOLDERS, and deliberately LOW. ────────
    // ★An epistemic gain is a measured entropy reduction in nats. A pragmatic value is the expected free
    // energy reduction from reaching a GOAL STATE — and nothing in this system currently wants to be on
    // the other side of a door. So the honest value of "cross" is near zero until a mission layer asks,
    // and these are set low on purpose: the door's offers stay VISIBLE and selectable when nothing else
    // bids, without out-bidding the fleet's exploration. When a mission layer arrives it should WRITE
    // these, and they should stop being config.
    //
    // ★★★WHOEVER REPLACES THESE: DO NOT MAKE THE PRICE A FUNCTION OF THE MEAN ALONE. IT LATCHES.
    // The intended source is the passage history ltsm_agent accumulates per doorway — a Beta posterior
    // over p(get through | I ask), folded from claimed crossings. That belief then GOVERNS ITS OWN
    // SAMPLING: the price decides whether the controller claims `cross`, and a claim is the only thing
    // that produces evidence. So a run of bad luck — two Timeouts from a door that happened to be shut —
    // drops the price, claims stop, and NO EVIDENCE CAN EVER ARRIVE TO CORRECT IT. The belief freezes
    // pessimistic with every instrument reading healthy, and the only cure is a human with a config
    // override. It is the bootstrap problem again, in the form that appears AFTER everything works.
    // The escape is the term a mean cannot express: a WIDE Beta means an uncertain doorway, and an
    // uncertain doorway is worth trying precisely BECAUSE trying it resolves the uncertainty. Price it
    // with the expected information gain alongside the pragmatic value — the same machinery as
    // room_concept's information-gain explorer and the controller's epistemic planner. This is also why
    // the history channel publishes a likelihood ratio and a conjugate posterior rather than a bare
    // ratio: the VARIANCE is not decoration beside the mean, it is the quantity the epistemic term
    // consumes, and a channel that published only a point estimate could not have an escape at all.
    // (Second, weaker answer: an inferred-volatility forgetting factor — MODEL_HISTORY §3 — decays an
    // old pessimistic posterior back toward the prior, so willingness to retry returns with time. Which
    // is also the truth about doors: one that was shut last month says little about today. α and β are
    // floats precisely so that slots in with no schema change.)
    // Agreed with ltsm_agent 2026-09-12; see [[door-pragmatic-affordances]].
    float value_approach = 0.5f;
    float value_open     = 0.3f;
    float value_cross    = 0.3f;

    float approach_timeout_s = 60.0f;
    float open_timeout_s     = 12.0f;   // provider latency + swing time, not a navigation budget
    float cross_timeout_s    = 40.0f;

    // Offer policy (rc::pragmatic::PragmaticAffordance::Policy): offer above `offer_prob`, withdraw below
    // `withdraw_prob`, and only after the decision has held `stable_cycles` MEASURED cycles.
    float offer_prob     = 0.60f;
    float withdraw_prob  = 0.35f;
    int   stable_cycles  = 3;

    // ── THE `transitable` SELF-EDGE: door --[transitable]--> door ────────────────────────────────
    // Asserted on the door's own node exactly while this agent believes the robot can get through it,
    // so a consumer can ask the GRAPH rather than re-deriving passability from geometry it would have to
    // fetch, transform and interpret for itself.
    //
    // ★"transitable", NOT "open", AND THE DISTINCTION IS REAL RATHER THAN COSMETIC. The belief behind
    // this edge is `door_open_prob` = P(clear span >= the ROBOT'S passage width), so it does not say the
    // leaf has moved — it says THIS BODY FITS. A door ajar by 20 cm is open and not transitable, and a
    // wide doorway may be transitable for this robot and not for a larger one. Naming the edge after the
    // quantity actually computed stops a consumer reading it as a statement about the door's mechanism.
    // (It is also already a registered cortex edge type, so this needs no reinstall.)
    //
    // ★SAME DECISION SHAPE AS EVERY OTHER BOUNDARY IN THIS AGENT: a Bayesian decision on a probability
    // with a Schmitt band and a debounce in MEASURED cycles — never a bare threshold on phi. Separate
    // keys from the affordance offer band above because this is a different act: an affordance is an
    // offer this agent makes, while this edge is a public ASSERTION ABOUT THE WORLD that other agents
    // may route on, and the two do not have to become confident at the same rate.
    float transitable_prob     = 0.60f;   // assert once P(passable) rises above this …
    float not_transitable_prob = 0.35f;   // … and retract once it falls below THIS (must be < the above)
    int   transitable_stable_cycles = 3;  // … and only after the decision has held this many MEASURED cycles

    // ★MAY THIS AGENT ASK A PROVIDER TO MOVE A DOOR WITHOUT A HUMAN? The `open` affordance's whole point
    // is that it can, and the consumer's claim is the trigger. Left as a switch because the request is an
    // act in the world with a real cost when the provider is a PERSON (DoorActuator::Caps::requires_human).
    bool  autonomous_actuation = true;
    // Which door the PROVIDER should move, by its own advertised id. Empty ⇒ resolve BY PLACE from the
    // pose we send, which requires the room↔world registration to be solved (door_world_registration) —
    // measured live, root←room is NOT that transform (6.45 m and ~90.3° out), so a by-place request
    // without a solved registration is refused locally rather than sent to open some other door.
    std::string actuation_provider_id;
};

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
    int   support_bank_max_points = 4000;
    float support_bank_quantization_m = 0.02f;
    float support_select_radius_margin_m = 0.50f;
    float support_select_height_margin_m = 0.25f;

    // DoorModel geometry / mask split: on-surface membership for the candidate/residual split in
    // DoorFitter::observe (a mask point within sdf_threshold_for_storage of the compound SDF is a candidate).
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
    // ⚠★DEAD KEY (confirmed 2026-07-31) — loaded by door_config.cpp but READ BY NOTHING. It was inherited
    // from the chair/table lineage, where a grazing view of a yaw-carrying surface inflates the shared yaw
    // variance. A door has NO free yaw DOF: the wall fixes yaw, lateral and floor, so the state is [s,w,h]
    // and door_fitter.cpp hard-sets dbg_obliquity_cos = 1.0f with the comment "there is no orientation to
    // observe — the backrest-obliquity yaw cap (chair) is gone". DoorFitter::door_view_obliquity() exists but
    // is explicitly diagnostic-only (a log column).
    // Kept, not deleted, because the leaf angle phi is a fitted DOF on the open-door path and a future
    // leaf-obliquity term would land exactly here. Do not tune it expecting an effect; wire it first.
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
    // ── HOW FAR THE LEAF CAN OPEN ────────────────────────────────────────────────────────────────
    // ★90 DEGREES WAS A HARD CEILING AND REAL DOORS GO PAST IT. estimate_phi searched [0, pi/2] and
    // CLAMPED the estimate to it, so a door standing open at ~100 degrees — swung back toward the wall,
    // which is what a door held open actually looks like — had NO representable hypothesis, and the
    // nearest state the model could offer was "nearly shut". Measured live 2026-09-13: a plainly open
    // door read 10 deg. The estimator was not wrong about that door; it had nothing to be right with.
    // 120 deg is the usual travel before a leaf meets a wall or a stop. A MODEL RANGE, not a tuning
    // knob: widening it adds hypotheses, it does not bias the ones already there.
    // ── LiDAR POINTS FOR THE HINGE BRANCH (bpearl) ───────────────────────────────────────────────
    // ★THE LEAF IS GEOMETRICALLY OBVIOUS AND SEMANTICALLY INVISIBLE. Everything this agent fits comes
    // from mask-deprojected points — i.e. only what the segmenter chose to call "door" — and measured
    // 2026-09-13, once the door stands open that is a sliver of JAMB: overlap with the predicted leaf
    // fell to 0.013 at every candidate angle, so the leaf-angle estimate had no evidence and parked near
    // flush on a door open past 90 deg. A leaf swung toward the robot at ~1 m is nonetheless a large flat
    // surface that the low bpearl dome returns cleanly, whatever any classifier calls it.
    // Precision, not a boolean: 0 leaves the ingestor entirely dormant (no DDS participant), and a larger
    // value weighs those rays more against the mask points in the same free energy.
    float lidar_bpearl_precision = 1.0f;

    // ── LEAF-TRACKER FORENSICS ───────────────────────────────────────────────────────────────────
    // ★TWO FILES THAT TOGETHER RECONSTRUCT THE DECISION, not summarise it. A whole day was lost to
    // changing the estimator and measuring afterwards: each summary number (phi, then w_sdf, then
    // leaf_pts) answered the previous question and raised a new one, because a single scalar cannot show
    // the SHAPE of a likelihood or the GEOMETRY of a point selection — and every failure so far has been
    // one of those two.
    //   phi_curve  : one row per HYPOTHESIS — every channel's raw value and the final weight, so the
    //                curve can be plotted and its argmin/flatness/monotonicity read directly.
    //   phi_points : one row per cycle — the selected cloud's geometry relative to the WALL PLANE, which
    //                is the quantity that decides whether the LiDAR branch is looking at a leaf or at a
    //                wall. A closed leaf lies IN that plane and is indistinguishable from the wall, so
    //                the signed-distance distribution is the whole story.
    // Throttled by cycle, not by event: an estimator that is wrong INTERMITTENTLY has to be sampled
    // uniformly or the log records only the cycles somebody already suspected.
    std::string phi_curve_csv_path  = "etc/door_phi_curve.csv";
    std::string phi_points_csv_path = "etc/door_phi_points.csv";
    int         phi_debug_every_n   = 5;    // 0 disables both
    float phi_max_rad            = 2.0944f;   // 120 deg
    // Angular resolution of the hypothesis grid; the step COUNT is derived from range/resolution so that
    // widening the range cannot silently coarsen the search — that would trade the ceiling for a blur,
    // and resolving which angle the leaf is at is the entire point.
    float phi_step_rad           = 0.0873f;   // 5 deg
    int   ai2_gn_iters           = 4;
    float ai2_extent_std         = 0.05f;   // extent-observation noise (m) for the coverage/extent likelihood
    std::string ai2_csv_path     = "";
    // Detector truth table (rc::probe, shared schema — common/detect_probe/detect_probe.h): one row per
    // cycle per live door recording WHERE the robot stood, HOW the door projected, and what the detector
    // did. door had none of this: the ai2 log can say "no mask this cycle" but not from how many DISTINCT
    // viewpoints, so the correlated-absence question could not even be posed for a door.
    // ★Default ON. A diagnostic that defaults off produces a file nobody has on the day it is needed.
    std::string detect_probe_csv_path = "etc/detect_probe.csv";

    // RGB contour check: score the door's own projected silhouette against image edges (common/contour_edge).
    // Independent of every classifier, which is the point — the semantic posterior collapses from 0.995 to
    // 0.048 on a plainly visible closed door as the robot closes, while its jamb and lintel stay in the RGB.
    bool  rgb_contour_check = true;

    // ── THE METRIC HALF OF THE CONTOUR CHANNEL, separately switchable ────────────────────────────
    // The ZED-depth test asks the absolute question the RGB gradient cannot: is there a surface at the
    // distance the belief predicts, with space receding behind its boundary? A photograph of a door has
    // the borders and none of the step, so this is the half a poster cannot pass — which is why it
    // exists and why turning it off has a real cost: a door that has genuinely GONE (carried away, or a
    // leaf swung fully clear) loses the one channel that can say "we are looking straight through the
    // place it is supposed to be", and removal then rests on mask absence alone, which is slower.
    //
    // ⚠MEASURED 2026-09-13 AND TURNED OFF IN etc/config.toml FOR THIS DEPLOYMENT. On a door the user
    // confirmed correctly fitted (hanging from its wall, corroborated in viewer3d) this half voted to
    // REMOVE on 489 of 489 cycles moving and 40 of 40 parked — never once positive — at a near-constant
    // verdict of -0.52 to -0.58, driven by a stable `dbg_depth_bias_m` of -0.20 m parked / -0.25 m
    // moving, in a band only 8 cm wide. A systematic offset that tight is a calibration disagreement,
    // not an absence, and `DoorInstance::dbg_depth_bias_m` says so in its own comment: "a fit error, not
    // an absence". Spending it as existence evidence made the door's survival depend entirely on the RGB
    // contour outrunning a permanent penalty: parked the contour reads +2.65 and wins, but on the frames
    // where YOLO goes blind it falls to +0.53, the sum turns negative, and the door is deleted. The
    // deaths looked like YOLO blindness; blindness only removed what was MASKING the constant negative.
    // ★So this is OFF pending a measurement of where the 20 cm lives — ZED depth bias at this range
    // (retina already has a LiDAR-anchored depth-correction dataset for exactly this) or the check's own
    // sampling. It is NOT a repudiation of the channel, and the default stays true so no other
    // deployment changes behaviour underneath itself.
    bool  contour_depth_check = true;

    // RoboCompDoorControl provider endpoint. Empty ⇒ the feature is off and the UI says so.
    // ⚠The webots-bridge listens on 10017 (its etc/config.toml, which is the file it is run with).
    // An older note recorded 10008; the bridge's own config comment documents that disagreement and
    // says etc/config's 10017 wins.
    std::string door_control_endpoint = "doorcontrol:tcp -h localhost -p 10017";

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
    float tracker_birth_frames     = 6.6f;    // IDEAL OBSERVATIONS of unexplained evidence before a birth
                                              // (8 rescaled by the graded model's median conf 0.826 — see config.toml)
    float tracker_birth_min_sep_m  = 0.70f;   // a birth must be ≥ this (m) from every existing door (anti-dup)
    float tracker_merge_overlap    = 0.20f;   // merge two instances whose seat footprints overlap ≥ this
                                              // fraction of the smaller, keeping the more-observed. 0 disables.
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
    // ★MEASURED AGAINST THE DOORS THAT ARE ACTUALLY HERE, not a generic guess. piso.wbt's Door PROTO
    // is `size 0.1 1 2.1` and `frameHeight 2.1`: 1.00 m wide, 2.10 m tall. The old 0.70 +- 0.06 put the
    // real door FIVE SIGMA out on a prior its own comment calls "strong", so the fit crawled to ~0.75
    // and was held there for ever — the projected leaf stayed ~25% narrower than the door it was
    // looking at, which is visible as the model box being smaller than the mask.
    // ⚠That also capped the phi measurement: IoU between a 1.52 m^2 model and a 2.10 m^2 door cannot
    // exceed 0.72 however right phi is, so the score was flattened before phi even entered. The width
    // was upstream of the phi instability, not a separate cosmetic complaint.
    // ⚠These are apartment facts, and they belong in etc/config.toml per deployment (Door.PriorWidthM /
    // Door.PriorHeightM) rather than compiled in. The defaults are moved because a default that is 5
    // sigma from every door in the only world we run is not a useful prior.
    float door_prior_w_m   = 1.00f;   // width prior mean (m)
    float door_prior_w_std = 0.06f;   // width prior std (m) — strong
    float door_prior_h_m   = 2.10f;   // height prior mean (m)
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

    // The three pragmatic affordances (approach / open / cross) — see DoorPragmaticCfg above.
    DoorPragmaticCfg pragmatic{};
};

// Fill a DoorConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
DoorConfig load_door_config(const ConfigLoader& cfg);

}  // namespace rc
