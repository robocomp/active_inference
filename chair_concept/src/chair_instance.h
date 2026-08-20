/*
 * chair_instance.h
 *
 * Per-chair runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the geometry/state container + the AI2 full-covariance belief, convergence bookkeeping, the
 * chair-owned support-point memory bank, and the epistemic affordance request.
 */

#pragma once

#include <chrono>
#include "../../common/exclusion/exclusion.h"        // rc::exclusion::Seniority (SHARED)
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "chair_model.h"
#include "../../common/existence_belief/existence_belief.h"   // rc::exist::ExistenceBelief        // ChairModel / ChairState
#include "chair_belief.h"       // rc::ChairBelief (AI2 recursive-Laplace belief)
#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
#include "../../common/appearance_belief/appearance_belief.h" // per-instance albedo chromaticity (DISPLAY only)

namespace rc {

struct ChairInstance
{
    uint64_t    node_id;
    std::string node_name;

    // Geometry / state container: holds the accepted pose+dims and the compound SDF used to split
    // the mask's support points into on-surface (candidate) vs off-surface (residual) sets.
    ChairModel  model;

    // ── AI2 belief ────────────────────────────────────────────────────────────────
    // Full-covariance recursive filter over θ=[cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h]. Lazily
    // initialised from the model state on the first cycle; its result is written back into `model`
    // so downstream publish/viewer code is unchanged.
    ChairBelief ai2_belief;
    bool  ai2_initialized = false;
    // RGB-360 bearing-only hypothesis (Part C-birth): born from a peripheral 360 bearing with a broad
    // along-ray Σ and NO depth yet. Authors an Orient affordance (rotate to look) instead of the normal
    // one; cleared the first time a real depth mask is observed (the glance paid off → normal instance).
    bool  is_bearing_hypothesis = false;
    float hypothesis_azimuth    = 0.0f;   // room-frame bearing to look toward (the Orient affordance's target yaw)
    float last_motion_var  = 0.0f;   // ego-motion downweight (added to R)
    float last_motion_dotd = 0.0f;   // motion-corruption speed (diagnostic)
    float last_trunc_frac  = 0.0f;   // silhouette truncation (predict-only gate)
    float last_range       = 0.0f;   // mean camera→mask depth Z (m): static range weighting
    float last_centroid_radius = 1.0f; // normalised mask-centroid radius from the principal point (0=centred, →1 edge)
    float last_depth_var   = 0.0f;   // σ_range² (m²) of the assigned slice: 0 for a ZED slice, >0 for a ricoh
                                     // LiDAR-depth slice → added to R so ricoh's unreliable depth barely moves the fit
    float last_clutter_frac = 0.0f;  // mean clutter responsibility (fraction of the mask the model can't explain)
    float dbg_obliquity_cos = 1.0f;  // |cos| of camera→chair horizontal ray vs backrest normal (1=face-on, →0 grazing)
    // Obliquity MARGINALISED over the yaw posterior — this, not the point estimate above, feeds the yaw
    // common-mode. The point estimate is CIRCULAR (built from the belief's own yaw, so a wrong yaw declares
    // its own error unobservable and protects it); this one may only claim "edge-on" to the degree the yaw is
    // genuinely known, and tightens back to the point estimate as σ_yaw converges. See chair_fitter.cpp.
    float dbg_obliquity_eff = 1.0f;
    bool  dbg_fixated       = true;  // did the FIXATION (attention) gate admit a geometry update this cycle?

    // Free-energy readout + attention baseline (clutter-inclusive F; TABLE.md §9). dbg_energy HOLDS the last
    // FE across a gated/rejected cycle (which took no measurement). fe_baseline tracks DOWN fast / UP slow so a
    // sustained rise (the chair moved) surfaces as fe_surprise before the baseline accepts it.
    float dbg_energy   = 0.0f;
    // ★HELD like dbg_energy beside it (2026-08-16): this frame's off-surface (residual) support-point
    // count, persisted so the dashboard trace keeps its last reading between mask frames instead of
    // adding no point at all and vanishing — which is what it did while the count lived only on the
    // local observation. The other four concept agents have held it all along.
    int   dbg_resid_pts = 0;
    float fe_baseline  = -1.0f;   // <0 = uninitialised (seed to the first accepted FE)
    float fe_surprise  = 0.0f;
    // ★The PUBLISHED epistemic_gain — the number the controller's affordance selection ranks on
    // (efe_score = gain − lambda_cost*dist, lambda_cost = 0.2/m, switch_margin = 0.5). Logged because
    // cross-agent selection was UNAUDITABLE: door was the only agent recording its gain, so "why does the
    // robot never visit a door" could not be answered against what a table or a fridge actually offers.
    // ★TWO GAINS, TWO COLUMNS — the single `dbg_nbv_gain` that used to be here meant the RAW planner gain in
    // hood/refrigerator/table and the PUBLISHED one in bottle/cabinet/chair, because the mirror sat on
    // opposite sides of the cooldown suppression. The same column therefore answered two different questions
    // depending on which log you opened, and during a cooldown hold the two differ by the whole gain.
    float dbg_nbv_gain_raw = 0.0f;   // ΔH the planner computed — "how uncertain is the belief?"
    float dbg_nbv_gain_pub = 0.0f;   // after the cooldown suppression and the verification floor: the number
                                     // the controller actually ranks on — "why did/didn't the robot come?"

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
    // ── Existence belief (continuous log-odds; supersedes the wall-clock stillbirth prune) ──────────
    // L = log P(exists)/P(¬exists). Integrated once per SENSOR frame while the instance is in the camera
    // frustum (roi_valid), via two channels: (1) WON a mask → POSITIVE/NEGATIVE by how much of the expected
    // chair silhouette the model EXPLAINS = (support/expected_at_range)·(1−clutter): a real chair explains a
    // good fraction (+); a far-too-sparse OR a big ~all-clutter won mask explains ≈nothing (−). (2) WON NOTHING
    // → absence evidence whose confidence RAMPS with frames_since_detection (freshness-as-precision): a brief
    // occlusion / lost slice barely moves L and recovers on the next win (this ramp prevents the death-spiral
    // churn), but a spot left unexplained for seconds accrues the full negative. HELD out of frustum / stale /
    // bearing-only / un-initialised. Removed when L < cfg.exist_remove_logodds — evidence-based, NO age
    // immunity; a real chair keeps L pinned at the saturation cap by being explained. NaN = un-seeded.
    // ★THE SHARED POLICY OWNS THE ARITHMETIC. chair used to keep only this float and hand-roll the
    // update, which is why it silently missed three fixes made in rc::exist: the occupancy-only floor
    // (a ratchet), the frame-correlation weighting, and occlusion-as-strength. It could not use the
    // shared class before because that class was reachable only through carve_box()/mask_evidence(),
    // which need LiDAR beams or pixel silhouettes — and chair has neither, only a point-count adequacy
    // proxy. The (p_vis, log_ratio) seam removed that coupling: an agent chooses its EVIDENCE, never
    // its ARITHMETIC. See CONCEPT_AGENT_INVARIANTS.md.
    rc::exist::ExistenceBelief existence{0.0f, 4.0f};
    // Mirror of existence.logodds(), kept because the dashboards, the strip and the CSVs read it. NaN
    // until the channel is seeded, which is what those readers test for.
    // Removal debounce, SHARED (rc::exist::RemovalDebounce) — chair had none at all and could delete on a

    // SHARED mutual exclusion (common/exclusion): was another concept's object already standing
    // here when this instance was BORN? Resolved once, at creation. A junior instance stops
    // counting its senior's returns as evidence that IT exists. See exclusion.h.
    rc::exclusion::Seniority exclusion;
    // single frame. streak = ideal observations of sustained condemnation; starved = consecutive condemned
    // cycles that resolved nothing, which is what makes a frozen decision visible instead of silent.
    rc::exist::RemovalDebounce existence_debounce;
    float exist_logodds = std::numeric_limits<float>::quiet_NaN();
    int  processed_cycles   = 0;      // per-chair compute cycles for log throttling
    bool model_stable       = false;
    int  model_generation   = 0;
    ChairState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;

    bool epistemic_pending  = false;
    // Schmitt-trigger hysteresis for the epistemic affordance (anti-oscillation).
    bool epistemic_satisfied = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied chair may re-arm

    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a
    // settled chair stops jittering the retina mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening chair
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // Chair-owned support-point memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> support_bank_pts;
    std::unordered_set<std::uint64_t> support_bank_keys;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    ObjectAffordance affordance;

    // ── Active-perception aids for the controller's local lock-on search ──────────
    // Detection aliveness: how recently YOLO produced a "chair" mask for this instance, and the
    // confidence of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh chair mask (0 = just detected)
    float last_mask_confidence   = 0.0f;      // YOLO confidence of the last chair detection
    bool  detection_alive        = false;     // frames_since_detection < threshold

    // Predicted in-image chair ROI from projecting the current model through the camera extrinsic.
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the chair in the
    // frame) and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image

    // ── Level-2 rig prior actually received this cycle (diagnostics; see refresh_rig_yaw_prior) ──
    // Room-frame bearing chair→camera (rad), NaN until an extrinsic is available. Identifies the
    // VIEWPOINT so the discrete yaw-mode test can discount repeated looks from the same bearing.
    float dbg_view_azimuth = std::numeric_limits<float>::quiet_NaN();

    bool  rig_edge_found = false;   // an incoming group_member edge was present
    float rig_kappa      = 0.0f;    // precision handed to the belief AFTER both caps
    float rig_prior_yaw  = 0.0f;    // rad, member convention

    // ── Appearance (DISPLAY ONLY) ─────────────────────────────────────────────────────────────────
    // 3-DOF Gaussian over this chair's albedo CHROMATICITY, fed by the retina's per-mask colour
    // summary and consumed only by chair_scene_graph, which publishes its MAP as `mesh_color_rgb` for
    // the 3D viewer's mesh tint. Nothing in the AI2 belief, the association gate, the existence
    // log-odds or the yaw-mode test reads it — a channel built from tens of thousands of correlated
    // mask pixels is kept out of the fit on purpose. See common/appearance_belief/appearance_belief.h.
    //
    // Note for a future reader tempted to use it: colour would be a plausible cue for the 90°/180° yaw
    // ambiguity, but NOT from this belief — it is one chromaticity for the whole instance, so it
    // cannot distinguish seat from backrest. That needs the per-PART render deferred in
    // appearance-belief-mesh-tint.
    rc::appearance::AppearanceBelief appearance;
    // Wall stamp of the last appearance drift step, so the drift tracks real elapsed time rather than a
    // cycle count — an agent running slow (or briefly stalled) must not under-inflate. Zero until first use.
    std::chrono::steady_clock::time_point last_appearance_tp{};
};

}  // namespace rc
