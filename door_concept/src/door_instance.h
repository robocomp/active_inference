/*
 * door_instance.h
 *
 * Per-door runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the geometry/state container + the AI2 full-covariance belief, convergence bookkeeping, the
 * door-owned support-point memory bank, and the epistemic affordance request.
 */

#pragma once

#include <chrono>
#include <utility>
#include "../../common/exclusion/exclusion.h"        // rc::exclusion::Seniority (SHARED)
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "door_model.h"        // DoorModel / DoorState
#include "door_belief.h"       // rc::DoorBelief (AI2 recursive-Laplace belief)
#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
#include "../../common/pragmatic_affordance/pragmatic_affordance.h" // rc::pragmatic (SHARED) — approach/open/cross
#include "door_pragmatics.h"   // rc::door::InteractionState — the three preconditions
#include "../../common/existence_belief/existence_belief.h"   // rc::exist::ExistenceBelief (shared w/ table/chair)

namespace rc {

struct DoorInstance
{
    uint64_t    node_id;
    std::string node_name;
    // DSR RT parent = the WALL node this door hangs from (0 = not yet resolved / hanging from room). A door
    // belongs to a wall, so its room→door pose is published as a wall→door RT edge (pose in the wall frame);
    // set by DoorSceneGraph::write_rt_pose once the door's centre resolves to a "wall_*" node.
    std::uint64_t wall_node_id = 0;
    // Which node this door was last POSITIONED NEXT TO on the 2-D graph canvas. Not the same question
    // as who its RT parent is, and that is exactly the bug it exists to close: the canvas position was
    // computed from the ROOM while the RT edge came from a WALL, so with 30 walls laid out across the
    // canvas the door was drawn floating beside the room with its one edge running off-screen to a
    // distant parent. Read as "the door is disconnected from the room" — twice, by the person whose
    // system it is — when the graph was correct the whole time. A view that contradicts the structure
    // costs more than no view.
    std::uint64_t canvas_anchor_id = 0;

    // Geometry / state container: the accepted room-frame pose + dims (leaf AND aperture).
    DoorModel  model;

    // ── Cached geometry, authored ONLY by DoorFitter::refresh_geometry ──────────────────────────────
    // The one place every consumer reads the door's shape from. `aperture` is the static hole in the wall
    // (what the DSR RT edge, wall association, merge, ghost identity and the room prior key on); `leaf` is
    // its articulation; `leaf_pose` is where the panel physically is right now. With phi pinned at 0 (M0)
    // leaf_pose is exactly the aperture rectangle, so nothing observable changes. See door_geometry.h for
    // why this is centralised — two independent SDFs and ~10 scattered cos/sin(yaw) rectangles used to
    // disagree, and the disagreement deleted doors.
    door::Aperture aperture{};
    door::LeafState leaf{};

    // ── PHI: the leaf's opening angle, now estimated (M1) ─────────────────────────────────────────
    // ★ANTICIPATE WHAT WE OURSELVES CAUSED. When this agent asks a provider to open a door, the change
    // that follows is the one thing it should predict perfectly — the command, the target angle and the
    // provider's swing rate are all known to the requester. Treating it as a surprise is what made the
    // existence channel read "the door opened" as "the door is gone": with phi pinned at 0 the predicted
    // silhouette stayed in the doorway the leaf had just left, all 420 samples went dark, and the door
    // was deleted BECAUSE it obeyed. The command therefore enters as a PRIOR on phi, not as a licence to
    // suppress evidence — the estimate is still scored against the image every cycle and can refute it.
    float phi_est      = 0.0f;    // current estimate (rad)
    float phi_cmd_from = 0.0f;    // where the commanded swing started
    float phi_cmd_to   = 0.0f;    // where it was asked to end
    float phi_cmd_rate = 0.0f;    // rad/s the provider advertised; 0 ⇒ no swing model, jump to target
    std::chrono::steady_clock::time_point phi_cmd_t0{};
    bool  phi_cmd_active = false; // a commanded swing is in flight (or just finished and unconfirmed)
    // Provenance of the silhouette occ/free split, copied from DoorSilhouette for the log: was the
    // absence marginalised over the opening angle, and how concentrated that posterior was. w_max near
    // 1/n_hyp means the frame did not identify phi at all and the absence was averaged away almost
    // entirely — which is the intended behaviour, and must be visible as such rather than inferred.
    bool  dbg_phi_marg  = false;
    int   dbg_phi_nhyp  = 0;
    float dbg_phi_wmax  = 0.0f;
    // When phi was last estimated, so the persistence prior's width is the swing the leaf could
    // ACTUALLY have made in the elapsed time rather than a per-cycle constant. Zero = never.
    std::chrono::steady_clock::time_point phi_last_t{};
    // Depth half of the contour channel. ★n == 0 is NOT MEASURED, a different state from a verdict of 0.
    float dbg_depth_verdict = 0.0f;   // signed, [-1,+1]
    float dbg_depth_bias_m  = 0.0f;   // mean (observed − predicted) depth, SIGNED: a fit error, not an absence
    int   dbg_depth_n       = 0;
    // ★THE WIDTH OF THE LEAF-ANGLE POSTERIOR, and the flag that says whether ANY channel spoke this
    // cycle. Without these, phi is a number that cannot admit ignorance: the estimator emitted an angle
    // whether or not anything had measured one, and every consumer read it as a measurement. Measured
    // live 2026-09-13 — the cycle reporting phi = 0 deg on a (coincidentally) shut door had zero LiDAR
    // points and all three channel weights exactly zero. sigma near the full range means NOT MEASURED.
    float phi_sigma    = 2.0944f;   // rad; seeded at the full range = "nothing known yet"
    // ★THE MEASUREMENT THAT DECIDES WHETHER THE MODEL OR THE SELECTION IS AT FAULT. At phi = 0 (a shut
    // leaf, flush in the aperture) a correct model + correct selection should EXPLAIN nearly every ray in
    // the doorway column. If it explains almost none, the rays are not landing on the modelled leaf and
    // no change to the likelihood can help. dbg_ray_signed0 says which way they miss: negative = stopping
    // SHORT of the leaf (floor, step, frame in the way), positive = flying PAST it (the leaf is not where
    // the model puts it, or is not there at all).
    int   dbg_ray_hit0 = 0;        // rays for which the model at phi=0 predicts ANY hit
    int   dbg_ray_expl0 = 0;       // …of which the measured range matches, within tolerance
    float dbg_ray_signed0 = 0.0f;  // mean signed (predicted - measured) over the predicted hits, m
    float dbg_phi_cv   = 0.0f;   // spread of the combined phi likelihood; 0 = perfectly flat
    bool  phi_measured = false;     // did any channel contribute a likelihood this cycle?
    float phi_support  = -1.0f;   // was the camera mask overlap at phi_est; -1 since the camera stopped scoring phi (2026-09-14)
    // ★IS phi_est A MEASUREMENT, OR OUR OWN REQUEST ECHOED BACK? True when the last thing to move it was
    // the PRIOR — the commanded swing — because the image offered no resolvable hypothesis at all.
    // This exists because of a live defect it closes. Asking a provider to open a door arms an
    // anticipation (phi_cmd_*) so the agent is not surprised by its own action; with no curve that
    // anticipation IS phi_est. The passability fallback then marginalises over N(phi_est, sigma), reads
    // "wide open" at the commanded 90 degrees, and door_open_prob hits 1.0 — so the `open` affordance's
    // completion predicate (door_open_prob >= 0.70) is satisfied BY THE REQUEST, and reports Satisfied
    // for a door that may never have moved. That is "a protocol event is not evidence" arriving through
    // the back door: not the provider's Delivered, but our own anticipation of it.
    // The anticipation stays — it is right, and it is what stops the existence channel deleting a door
    // for obeying. What must not happen is it standing in as EVIDENCE of passability. A door is open
    // when we SEE it open.
    bool  phi_from_command = false;
    // ── THE HINGE BRANCH'S OWN INSTRUMENTS ──────────────────────────────────────────────────────
    // ★WITHOUT THESE THE ESTIMATE CANNOT BE DIAGNOSED, ONLY GUESSED AT. Measured 2026-09-13: phi came out
    // strongly bimodal — 476 rows near 0 deg, 106 piled against the 120 deg clamp — and from the logged
    // columns alone there was no way to tell "the LiDAR branch found the leaf and it really is wide open"
    // from "the LiDAR branch was empty and the prior jumped to the ceiling". Those two call for opposite
    // fixes, so shipping the branch without recording which one is happening was the actual mistake.
    int   dbg_leaf_pts = 0;     // LiDAR points selected for the hinge branch; ★0 = NOT MEASURED
    int   dbg_leaf_off_plane = 0;  // …of which stand OFF the wall plane. A CLOSED leaf is in the
                                   // wall plane and therefore invisible to LiDAR: only these can
                                   // ever say the leaf is open. Near 0 with a large phi = a
                                   // contradiction, and the angle came from somewhere else.
    float dbg_w_sdf  = 0.0f;    // normalised weight each channel put on the WINNING angle. All three
    float dbg_w_mask = 0.0f;    // sum to the post-prior weight, so their ratio says which channel
    float dbg_w_edge = 0.0f;    // actually decided this cycle — the question the bimodality poses.
    // ★THE WHOLE LIKELIHOOD CURVE, NOT ONLY ITS ARGMAX. estimate_phi scores every candidate angle and
    // then throws all but the best away; the silhouette channel — the ONLY channel allowed to remove a
    // door — was then rendered at that single angle. Measured 2026-09-10 the peak support is 0.007-0.245,
    // i.e. the curve is nearly FLAT and the argmax is noise: the leaf was projected out into the room,
    // occ went 207 -> 0, free_eff 0 -> 33, and L fell +4.00 -> -4.00 in 13 cycles twice on one approach.
    // A parameter the data does not identify must not be able to drive a removal. Keeping the curve lets
    // the absence be MARGINALISED over phi instead of conditioned on a point estimate, so a flat curve
    // charges almost no absence and a peaked one charges it in full — without any gate on "flatness".
    // (phi rad, unnormalised POSTERIOR weight = image support x persistence/command prior). It carries
    // the prior because both consumers need the same thing: the angle we actually believe, not the one
    // the image alone would pick off a curve with no peak in it.
    std::vector<std::pair<float, float>> phi_curve;
    door::LeafPose  leaf_pose{};

    // ── AI2 belief ────────────────────────────────────────────────────────────────
    // Full-covariance recursive filter over θ=[cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h]. Lazily
    // initialised from the model state on the first cycle; its result is written back into `model`
    // so downstream publish/viewer code is unchanged.
    DoorBelief ai2_belief;
    bool  ai2_initialized = false;
    // RGB-360 bearing-only hypothesis (Part C-birth): born from a peripheral 360 bearing with a broad
    // along-ray Σ and NO depth yet. Authors an Orient affordance (rotate to look) instead of the normal
    // one; cleared the first time a real depth mask is observed (the glance paid off → normal instance).
    bool  is_bearing_hypothesis = false;
    float hypothesis_azimuth    = 0.0f;   // room-frame bearing to look toward (the Orient affordance's target yaw)
    // ★Does this hypothesis have a RANGE, or only a bearing? Set when the birth path resolved one from the
    // precision cascade in door_bearing_range.h (today: the residual grid; soon: dense ricoh depth). With a
    // range the hypothesis has a real (x,y), so its affordance becomes "GO THERE and check" (Servo) instead
    // of "glance that way" (Orient). Without one it stays a glance — the honest answer, not a fallback.
    bool  hypothesis_range_known = false;
    float hypothesis_range_sigma = 0.0f;   // 1σ along the ray (m), as reported by whichever rung answered
    // ★The observation that BIRTHED this hypothesis: where the robot stood and what bearing it saw. Kept so a
    // later sighting from somewhere else can be triangulated against it (rc::door::triangulate_bearings) —
    // the robot's own travel is the baseline. Without this the proto could only ever be re-confirmed, never
    // ranged, and an unchecked proto would sit at its guessed range forever.
    Eigen::Vector2f hypothesis_obs_from{0.0f, 0.0f};
    int   hypothesis_fixes = 0;            // how many bearings have been fused into it (diagnostic)
    float last_motion_var  = 0.0f;   // ego-motion downweight (added to R)
    float last_motion_dotd = 0.0f;   // motion-corruption speed (diagnostic)
    float last_trunc_frac  = 0.0f;   // silhouette truncation (predict-only gate)
    float last_range       = 0.0f;   // mean camera→mask depth Z (m): static range weighting
    float last_centroid_radius = 1.0f; // normalised mask-centroid radius from the principal point (0=centred, →1 edge)
    float last_depth_var   = 0.0f;   // σ_range² (m²) of the assigned slice: 0 for a ZED slice, >0 for a ricoh
                                     // LiDAR-depth slice → added to R so ricoh's unreliable depth barely moves the fit
    float last_clutter_frac = 0.0f;  // mean clutter responsibility (fraction of the mask the model can't explain)
    float dbg_obliquity_cos = 1.0f;  // |cos| of camera→door horizontal ray vs backrest normal (1=face-on, →0 grazing)

    // Free-energy readout + attention baseline (clutter-inclusive F; TABLE.md §9). dbg_energy HOLDS the last
    // FE across a gated/rejected cycle (which took no measurement). fe_baseline tracks DOWN fast / UP slow so a
    // sustained rise (the door moved) surfaces as fe_surprise before the baseline accepts it.
    float dbg_energy   = 0.0f;
    // ★HELD like dbg_energy beside it (2026-08-16): this frame's off-surface (residual) support-point
    // count, persisted so the dashboard trace keeps its last reading between mask frames instead of
    // adding no point at all and vanishing — which is what it did while the count lived only on the
    // local observation. The other four concept agents have held it all along.
    int   dbg_resid_pts = 0;
    float fe_baseline  = -1.0f;   // <0 = uninitialised (seed to the first accepted FE)
    float fe_surprise  = 0.0f;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged   = 0;      // consecutive frames with |Δstate| < state_eps
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask (chain-cov pinning)
    // Agent-clock stamp of the last belief touch (set EVERY inference cycle) so a stale cycle inflates Σ by
    // the real elapsed time (measurement-age → covariance); mirrors table_concept.
    std::chrono::steady_clock::time_point last_belief_touch{};
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f, chain_cov_yaw = 0.0f;  // localization/chain cov (m²,rad²)
    int  assigned_mask_idx  = -1;     // tracker's gated mask-slice assignment (-1 = use greedy nearest)
    // ── Existence belief (shared rc::exist channel — same one table/chair use) ──────────────────────
    // L = log P(exists)/P(¬exists), integrated per SENSOR frame from the PIXEL-LEVEL silhouette evidence
    // (DoorFitter::compute_silhouette_existence): the model's panel silhouette is projected into the ZED and
    // each predicted sample votes
    //   lit by a "door" mask   ⇒ OCCUPANCY (still there),
    //   lit by a NON-door mask ⇒ OCCLUDED  ⇒ excluded from the detectable footprint (no vote),
    //   lit by nothing         ⇒ ABSENCE   (predicted-but-not-there — the "gone" signal, which fires even on a
    //                                       frame where YOLO produced no door mask at all).
    // The decisive property, and the reason this replaced the old pd·conf·obliquity scheme: a sample OUTSIDE
    // the real camera frustum is not detectable, so a door behind the robot yields n_detectable==0 ⇒ rc::exist
    // HOLDs. "Not looked at" is structurally distinct from "looked at and empty" — the old scheme charged
    // absence evidence through a bearing-free range term and deleted a real door whenever the robot turned
    // around. Occlusion is likewise a continuous shrinkage of the detectable footprint, never an indefinite
    // freeze. Removal is a Bayesian decision on P(exists) (should_remove), debounced over consecutive
    // EVIDENCE cycles — no age immunity, and no removal from a view that could not have resolved the door.
    rc::exist::ExistenceBelief existence{};
    bool existence_seeded = false;            // false until seeded with cfg.exist_birth_logodds on first visit
    // Accumulated LOOKS (Σ p_detect) whose decision says "remove" — NOT a count of cycles. A cycle in which
    // the door could not have been resolved (p_detect → 0) correctly leaves L untouched, and must leave the
    // debounce untouched too, or the door is condemned by evidence gathered once and executed later while
    // the robot looks elsewhere. Float for the same reason table/bottle use one: partial looks partially count.
    // Debounce state, SHARED (rc::exist::RemovalDebounce): the streak in ideal observations plus the

    // SHARED mutual exclusion (common/exclusion): was another concept's object already standing
    // here when this instance was BORN? Resolved once, at creation. A junior instance stops
    // counting its senior's returns as evidence that IT exists. See exclusion.h.
    rc::exclusion::Seniority exclusion;
    // consecutive-starved count that makes a condemned-but-unexecutable instance visible instead of frozen.
    rc::exist::RemovalDebounce existence_debounce;
    // Silhouette diagnostics (last evidence cycle) — the columns of etc/door_existence_log.csv.
    float dbg_sil_occ = 0.0f, dbg_sil_free = 0.0f, dbg_sil_free_eff = 0.0f;
    int   dbg_sil_ndet = 0, dbg_sil_ntotal = 0, dbg_sil_noccl = 0, dbg_sil_ncells = 0;
    float dbg_sil_pdetect = 0.0f, dbg_sil_central = 0.0f, dbg_sil_resolv = 0.0f;
    // Contour check against retina's graded posterior (see specificworker.cpp). Logged so the channel
    // can be audited from outside: dbg_field_n == 0 means the field was unavailable and this cycle behaved
    // exactly as it did before the channel existed — which is a different fact from "no support found".
    float dbg_field_support = 0.5f;   // 0.5 = the contour is no more door-like than the rest of the frame
    // RGB contour check (common/contour_edge): support = across-boundary gradient on the believed
    // contour vs the same shape displaced along the wall. 0.5 = indistinguishable from a displaced copy.
    // ★dbg_edge_n == 0 means NOT MEASURED (no frame, contour behind the camera) — not "no support".
    float dbg_edge_support = 0.5f;
    float dbg_edge_true = 0.0f, dbg_edge_ctrl = 0.0f;
    int   dbg_edge_n = 0;
    float dbg_edge_delta = 0.0f;   // log-odds this channel contributed this cycle (signed)
    float dbg_edge_excess = 0.0f;  // (s_true - s_ctrl) / frame mean gradient — THE evidence quantity
    int   dbg_edge_nctl = 0;       // control placements that survived; 0 ⇒ nothing to compare against,
                                   // which happens at close range as the controls fall off-frame
    float dbg_field_mean = 0.0f, dbg_field_bg = 0.0f;
    int   dbg_field_n = 0;
    // The fit's own admissibility verdict for the last processed frame (truncated mask, or the robot moving
    // with the mask off-centre ⇒ predict-only). Read by the existence channel: a frame that may not MOVE the
    // geometry may not DESTROY the door either. See specificworker.cpp's absence term.
    bool  dbg_gated = false;
    // …split by MECHANISM, and stamped with whether it was computed from THIS cycle's mask. The existence
    // channel must read the gate's REASON and its FRESHNESS, never the bare verdict: the fit returns early
    // when no mask reaches it, so on exactly the cycles the absence guard fires, `dbg_gated` is STALE — the
    // verdict of whenever the door was last seen. A phantom is by definition never seen again, so a single
    // truncated birth frame pinned dbg_gated=true forever and its absence evidence was zeroed for good.
    // See specificworker.cpp's `view_untrustworthy`, and [[table-phantom-immortality]] for the same defect.
    bool  dbg_trunc_gated  = false;   // the mask was clipped by the image border this frame
    bool  dbg_motion_gated = false;   // robot moving with the mask off-centre (AI2MotionConfirmOnly path)
    bool  dbg_gate_fresh   = false;   // the two flags above were computed THIS cycle. false ⇒ they mean nothing

    // ── Observed vertical extent (the MINIMUM-HEIGHT prior) ─────────────────────────────────────────
    // A door is an aperture a person walks THROUGH, so P(door | its support tops out well below a door's
    // height) ≈ 0. This has to be judged on what the sensor actually SAW, not on the fitted h: the template
    // anchor (prior_h_std 0.08 ⇒ precision 156/frame vs the 8.2 per-frame cap) pins h at 2.0 m whatever the
    // data says, so a phantom fed by a low blob still reports a confident 2.0 m — a test on fitted h can
    // never fire. Measured live: the real door's h moved 2.000 → 2.063, the phantom's never left 2.000.
    //
    // obs_top_z is an EWMA of the support bbox top (room frame, floor = 0) over UNTRUNCATED observations
    // only. Truncation matters: a mask clipped by the image border has an unobserved top, so its bbox top
    // is a LOWER BOUND, not a measurement — the real door is routinely clipped (n_detectable 270/420 live),
    // and counting those views would delete it. Confidence accumulates with (1 − trunc_frac).
    float obs_top_z    = std::numeric_limits<float>::quiet_NaN();   // NaN = never measured untruncated
    float obs_top_conf = 0.0f;   // ∈[0,1] EWMA weight of untruncated evidence behind obs_top_z
    float obs_top_last = std::numeric_limits<float>::quiet_NaN();   // last raw measurement (diagnostic)
    int  processed_cycles   = 0;      // per-door compute cycles for log throttling
    bool model_stable       = false;
    int  model_generation   = 0;
    DoorState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;

    bool epistemic_pending  = false;
    // Schmitt-trigger hysteresis for the epistemic affordance (anti-oscillation).
    bool epistemic_satisfied = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied door may re-arm

    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a
    // settled door stops jittering the retina mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening door
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // Door-owned support-point memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> support_bank_pts;
    std::unordered_set<std::uint64_t> support_bank_keys;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    ObjectAffordance affordance;

    // ── THE THREE PRAGMATIC AFFORDANCES: approach / open / cross ────────────────────────────────────
    // Siblings of `affordance` above under the same door node, and a different KIND of offer: that one
    // asks the controller for a LOOK that shrinks Σ, these three ask it to do something to the door.
    // Each is preconditioned — on the wire only while this agent believes the action is possible at all
    // (rc::pragmatic, see its header for the three rules that keeps honest). `approach` is effectively
    // unconditional for a located door; `open` needs the robot inside the actuation zone with the
    // aperture believed shut; `cross` needs the aperture believed passable for this robot's body.
    rc::pragmatic::PragmaticAffordance aff_approach;
    rc::pragmatic::PragmaticAffordance aff_open;
    rc::pragmatic::PragmaticAffordance aff_cross;
    bool pragmatic_inited = false;   // init() takes the node id/name, so it waits for the DSR node

    // This cycle's preconditions, recomputed once and then (a) published on the door node as the
    // affordances' completion predicates and (b) used to decide what to offer. Held on the instance so
    // the dashboard and the CSV logs read the SAME numbers the decisions were taken on.
    rc::door::InteractionState interaction{};

    // ── THE `transitable` SELF-EDGE's decision state ────────────────────────────────────────────
    // door --[transitable]--> door is asserted while this agent believes the ROBOT CAN GET THROUGH (see
    // DoorPragmaticCfg: it is P(clear span >= this body's passage width), not a claim about the leaf).
    // Two streaks rather than one so the Schmitt band has somewhere to live, counted in MEASURED cycles:
    // a cycle that learned nothing about phi must not advance either of them, or a door nobody looked at
    // would eventually assert or retract a public edge on the strength of not having been observed.
    int  transitable_streak     = 0;
    int  not_transitable_streak = 0;
    bool transitable_asserted   = false;   // what we last WROTE, so a transition can be logged once

    // ★THE SIDE THE ROBOT WAS ON WHEN THE `cross` CLAIM WAS MADE. "Crossed" is a CHANGE of side, not a
    // position, so no static predicate on a coordinate can express it — door_crossing_progress is signed
    // against this. Latched on the claim and only then: re-latching per cycle would make the progress
    // measure its own input and read 0 for ever. NaN = no claim outstanding.
    float cross_side0 = std::numeric_limits<float>::quiet_NaN();

    // Door actuation, as far as the `open` affordance knows. ★`asked` is NOT `opened`: a provider's
    // Delivered says it did what it was asked, never that the world changed (door_actuator.h). Whether
    // the leaf moved is answered by door_open_prob, from the image, like everything else here.
    int  actuation_requests = 0;      // how many times we have asked, this door, this run
    bool actuation_refused  = false;  // the provider said never-ask-again (UnknownDoor / NotActuable)
    std::string actuation_note;       // last provider state or local refusal reason, for the UI/log

    // ── Active-perception aids for the controller's local lock-on search ──────────
    // Detection aliveness: how recently YOLO produced a "door" mask for this instance, and the
    // confidence of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh door mask (0 = just detected)
    float last_mask_confidence   = 0.0f;      // YOLO confidence of the last door detection
    bool  detection_alive        = false;     // frames_since_detection < threshold

    // Predicted in-image door ROI from projecting the current model through the camera extrinsic.
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the door in the
    // frame) and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image
    // The two axes the max() above collapses. Kept because the detector envelope's two shoulders test
    // DIFFERENT axes — the short axis decides whether there are enough pixels to segment, the long axis
    // whether it still fits with context — so a single max() cannot express either. common/detectability's
    // fit_envelope REQUIRES both, which is why door could not be fitted at all until now.
    float roi_fill_h   = 0.0f;   // (max_col-min_col)/W
    float roi_fill_v   = 0.0f;   // (max_row-min_row)/H
};

}  // namespace rc
