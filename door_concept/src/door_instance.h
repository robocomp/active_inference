/*
 * door_instance.h
 *
 * Per-door runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the geometry/state container + the AI2 full-covariance belief, convergence bookkeeping, the
 * door-owned voxel memory bank, and the epistemic affordance request.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "door_model.h"        // DoorModel / DoorState
#include "door_belief.h"       // rc::DoorBelief (AI2 recursive-Laplace belief)
#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
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
    int  unassigned_streak  = 0;      // consecutive tracker cycles with no mask assignment (stillbirth prune)
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
    float existence_remove_streak = 0.0f;
    // Silhouette diagnostics (last evidence cycle) — the columns of etc/door_existence_log.csv.
    float dbg_sil_occ = 0.0f, dbg_sil_free = 0.0f, dbg_sil_free_eff = 0.0f;
    int   dbg_sil_ndet = 0, dbg_sil_ntotal = 0, dbg_sil_noccl = 0, dbg_sil_ncells = 0;
    float dbg_sil_pdetect = 0.0f, dbg_sil_central = 0.0f, dbg_sil_resolv = 0.0f;
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
    // settled door stops jittering the voxelizer mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening door
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // Door-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    ObjectAffordance affordance;

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
};

}  // namespace rc
