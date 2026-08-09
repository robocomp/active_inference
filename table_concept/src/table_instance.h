/*
 * table_instance.h  —  per-table runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h).
 *
 * Bundles the geometry/state container, the AI2 full-covariance belief, convergence bookkeeping, the
 * per-frame sensor/diagnostics snapshots, the table-owned voxel memory bank, the existence log-odds, and
 * the epistemic affordance request — everything TableFitter carries per tracked table.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "table_model.h"        // TableModel / TableState
#include "table_belief.h"       // AI2 full-covariance belief (TABLE.md)
#include "table_affordance.h"   // TableAffordance
#include "../../common/existence_belief/existence_belief.h"   // per-instance existence log-odds (removal)
#include "../../common/appearance_belief/appearance_belief.h" // per-instance albedo chromaticity (DISPLAY only)

namespace rc {

struct TableInstance
{
    // ── Identity ──────────────────────────────────────────────────────────────────────────────────
    uint64_t    node_id;
    std::string node_name;

    // ── Geometry / state container ────────────────────────────────────────────────────────────────
    // Holds the accepted pose+dims and the compound SDF used to split the mask's support points into
    // on-surface (candidate) vs off-surface (residual) sets.
    TableModel  model;

    // ── AI2 belief (TABLE.md) ─────────────────────────────────────────────────────────────
    // Full-covariance recursive filter over θ=[cx,cy,H,w,h,yaw]. Lazily initialised from the model state on
    // the first cycle; its result is written back into `model` so downstream publish/viewer code is unchanged.
    TableBelief ai2_belief;
    bool        ai2_initialized = false;

    // ── Per-frame mask corruption / range (from the selected mask slice) ──────────────────────────
    // motion_var inflates the observation precision R (downweight); trunc_frac gates the update (truncated →
    // biased mask); dotd is the motion-corruption speed, kept for diagnostics (see MASK_MOTION_CORRUPTION.md).
    float last_motion_var      = 0.0f;
    float last_motion_dotd     = 0.0f;
    float last_trunc_frac      = 0.0f;
    float last_centroid_radius = 0.0f;
    float last_range           = 0.0f;   // mean camera→mask depth Z (m), this frame — static range weighting
    // Depth-uncertainty of the mask (common/depth_projection): σ_range² (m²) added to R, sibling to
    // last_motion_var. 0 for dense-depth zed masks; the scored range variance for lidar-depth ricoh masks.
    float last_depth_var       = 0.0f;

    // ── Appearance (DISPLAY ONLY) ─────────────────────────────────────────────────────────────────
    // 3-DOF Gaussian over the table's albedo CHROMATICITY, fed by the voxelizer's per-mask colour summary
    // and consumed only by table_scene_graph, which publishes its MAP as `mesh_color_rgb` for the 3D
    // viewer's mesh tint. Nothing in the AI2 belief, the association gate, or the existence log-odds reads
    // it — a channel built from tens of thousands of correlated mask pixels is kept out of the fit on
    // purpose. See common/appearance_belief/appearance_belief.h.
    rc::appearance::AppearanceBelief appearance;
    // Wall stamp of the last appearance drift step, so the drift tracks real elapsed time rather than a
    // cycle count — an agent running slow (or briefly stalled) must not under-inflate. Zero until first use.
    std::chrono::steady_clock::time_point last_appearance_tp{};

    // ── LiDAR range-channel diagnostics (this frame) ──────────────────────────────────────────────
    // #returns fed to the factor, #returns in a generous box, and their mean |dist| to the current model
    // surface. Few rays / large resid ⇒ wrong lidar→room frame.
    int   dbg_lidar_rays     = 0;
    int   dbg_lidar_raw      = 0;
    float dbg_lidar_resid_m  = -1.0f;
    float dbg_lidar_meanz_m  = -1.0f;   // mean z of selected returns (room frame) — vs H detects a z-offset
    float dbg_lidar_topz_m   = -1.0f;   // mean z of the HIGHEST 20% of selected returns ≈ observed tabletop z
    float dbg_lidar_floorz_m = -1.0f;   // mean z of the LOWEST 20% ≈ floor; should read ~0 if z-calib OK
    float dbg_lidar_cov_ang  = -1.0f;   // angular-coverage weight (1−R)^p ∈[0,1]; low ⇒ one-sided sweep
    int   dbg_lidar_bpearl_rays = 0;    // low-bpearl returns staged into the extra ray-set this frame (0 = OFF/none)

    // ── Divergence safety net (mirrors bottle) ────────────────────────────────────────────────────
    // Consecutive frames whose centre GN step was rejected as an outlier (exceeded cfg.max_step_m). Non-zero
    // ⇒ the fit tried to run away and was held.
    int   frames_diverged = 0;

    // ── Convergence / association bookkeeping ─────────────────────────────────────────────────────
    int  matched_frames        = 0;   // frames with fresh sensing data
    int  frames_converged      = 0;   // consecutive frames with |Δstate| < state_eps
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask (chain-cov pinning)
    // Wall-clock (agent's own steady clock) of the last belief touch — measured on EVERY inference cycle, not
    // just fresh ones — so a stale cycle can inflate Σ by the real elapsed time (measurement-age → covariance).
    std::chrono::steady_clock::time_point last_belief_touch{};
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;  // Part B localization/chain cov (m²), added to the RT cov
    // Per-frame ROBOT-frame observation z_o (x,y,yaw) for the room localizer's object-anchor factor.
    // Computed by the fitter (gated), published by the scene-graph writer. See common/object_anchor.
    Eigen::Vector3f obs_robot = Eigen::Vector3f::Zero();
    // Anisotropic 2×2 measurement covariance R_o (ROBOT/body frame) for this single-view centroid: tight
    // ⊥-ray, loose along-ray so the near-face/partial-view bias falls in R_o's null-space and can't drag
    // the robot pose. Built in compute_object_observation via ray_anisotropic_cov; published as
    // obj_obs_robot_cov=[Rxx,Ryy,Rxy]. See TABLE_TRIANGULATION.md.
    Eigen::Matrix2f obs_robot_cov = Eigen::Matrix2f::Identity();
    bool obs_robot_valid = false;
    int  processed_cycles = 0;        // per-table compute cycles for log throttling
    // Tracker's gated mask assignments for THIS frame (indices into the masks packet slices). With
    // multi-detection fusion a table can be seen by BOTH the ZED and the ricoh-360 masks in one cycle, so
    // this is a LIST — process_table_node runs one belief update per slice (sequential = joint likelihood,
    // each sensor keeping its own R + common-mode). Empty = no association this frame. Set by run_instance_tracker.
    std::vector<int>  assigned_mask_idxs;
    bool model_stable     = false;
    int  model_generation = 0;
    TableState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;

    // ── Epistemic affordance state (Schmitt-trigger hysteresis, anti-oscillation) ─────────────────
    bool epistemic_pending   = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied table may re-arm

    // ── write_rt_pose dead-band (suppress tiny pose oscillations) ─────────────────────────────────
    float last_written_cx = std::numeric_limits<float>::max();
    float last_written_cy = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a settled
    // table stops jittering the voxelizer mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening table refreshes its
    // edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // ── Table-owned voxel memory bank (room frame), independent of per-frame uploads ──────────────
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    bool cloud_dumped = false;   // one-shot guard for the diagnostic voxel-bank dump (cfg_.dump_cloud_path)

    // ── Shape model-selection (round vs square), set by TableFitter::evaluate_shape ──────────────
    // Published as the node's object_subtype string (the voxelizer renders the matching mesh). Chosen by
    // free-energy / model evidence, NOT a threshold: shape_evidence is a bounded accumulated log-Bayes-
    // factor (round − square) over periodic voxel-bank comparisons; subtype flips at the zero boundary.
    std::string subtype        = "square";   // "square" | "round" (free-form for future sub-subtypes)
    float       shape_evidence = 0.0f;       // >0 ⇒ round better explains the accumulated cloud
    int         shape_eval_ctr = 0;          // cycles since the last periodic shape evaluation
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    TableAffordance affordance;

    // ── Active-perception aids for the controller's local lock-on search ──────────────────────────
    // Detection aliveness: how recently YOLO produced a "table" mask for this instance, and the confidence
    // of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh table mask (0 = just detected)
    float last_mask_confidence   = 0.0f;     // YOLO confidence of the last table detection
    bool  detection_alive        = false;    // frames_since_detection < threshold

    // ── Existence log-odds (removal) ──────────────────────────────────────────────────────────────
    // LiDAR free-space carve + mask-absence accrue negative evidence when the table is demonstrably gone.
    // Removed only when the removal decision holds for existence_remove_frames consecutive cycles (debounce)
    // — deleting furniture warrants SUSTAINED evidence, not a transient hiccup.
    rc::exist::ExistenceBelief existence;
    // Removal debounce, in units of ONE FULLY-RESOLVING LOOK (not cycles). A cycle contributes its own p_detect:
    // a confident, centred, close view counts ~1, a view that cannot resolve the table counts ~0. Fractional
    // because a timer would delete a table for the passage of uninformative time — see table_existence.cpp.
    float existence_remove_streak = 0.0f;
    // Verification-gated removal (active-inference): a predicted-visible-but-absent observation from a view that
    // CANNOT resolve the table (far / peripheral / edge-on) does NOT vote removal — it raises this decayed
    // go-VERIFY surprise. When it crosses existence_verify_surprise, wants_verification arms the epistemic planner
    // to drive the robot to a good ZED viewpoint; only a HIGH-P(detect) view can then confirm-or-remove.
    float verify_surprise    = 0.0f;
    bool  wants_verification = false;

    // ── EvidenceMonitor per-cycle snapshots (persisted copies of transient fit/existence values) ──
    int   dbg_n_zed_slices = 0;   // ZED mask slices fused this cycle (ricoh is bearing-only, never fused)
    int   dbg_cand_pts     = 0;   // this frame's on-surface (candidate) support point count
    int   dbg_resid_pts    = 0;   // this frame's off-surface (residual/unexplained) support point count
    float dbg_energy       = 0.0f;  // mean per-point data energy (NLL proxy) at the converged state
    float dbg_R            = 0.0f;  // per-point measurement variance used this frame (m²)
    // FE ATTENTION / SURPRISE (active-perception trigger). fe_baseline = the "well-fit" free-energy level this
    // table settles to (slow EMA; tracks DOWN fast to consolidate a better fit, UP slow so a SUSTAINED rise — the
    // world changed, e.g. the table was moved — shows as surprise before the baseline accepts it). fe_surprise =
    // the smoothed positive gap (FE − baseline): ~0 when the belief matches the data, spikes when the table moves
    // and decays as the robot re-observes and the fit re-converges. This is the signal that will raise the table's
    // epistemic priority in the controller's EFE (attend + dwell until FE settles). See active-perception thread.
    float fe_baseline = -1.0f;    // <0 = uninitialised
    float fe_surprise = 0.0f;
    bool  dbg_gated   = false;    // truncation-gated (predict-only, no geometric update) this frame
    // WHY the frame was gated, split by error mechanism — the existence channel must distinguish them.
    // A frozen belief compared against a live image measures REGISTRATION when the freeze came from a bad
    // VIEWPOINT (truncated mask / robot moving / object off-centre), so silhouette absence is untrustworthy
    // there. It does NOT when the freeze came from too few mask points: "no mask on this box" IS the absence
    // observation, and suppressing it made a maskless phantom unremovable by construction (it can never reach
    // FixationMinPts, so it was permanently gated ⇒ permanently immune). See table_existence.cpp.
    bool  dbg_trunc_gated = false;   // the mask was truncation-gated this frame
    bool  dbg_gate_fresh  = false;   // the gate verdict above was computed THIS cycle. false ⇒ no fresh mask
                                     // reached the fit, so the flags are STALE (from whenever it was last
                                     // seen) and carry no information about the current frame.

    // Provenance of the mask packet this fit consumed (instrumentation; NO effect on the fit). The CSV showed
    // long runs of fits whose GEOMETRY was bit-identical (npts/range pinned) while the belief ratcheted — i.e.
    // the same capture re-integrated as independent evidence. These two columns say WHICH identity signal
    // moved: fid = mask_frame_id (a publish counter, advances on every republish), ts = mask_timestamp_ms
    // (the capture stamp; absent ⇒ 0 ⇒ the capture-stamp half of the freshness gate is disabled).
    int           dbg_pkt_frame_id = -1;
    std::uint64_t dbg_pkt_ts_ms    = 0;

    // ── Rogue-mask yaw diagnostics (instrumentation; NO effect on the fit) ─────────────────────────
    // Captured per fresh cycle to CATCH the abrupt-rotation frames (robot turning / entering-leaving FoV /
    // passing by, where an in-frame-but-partial mask snaps yaw). yaw at cycle entry + the per-channel yaw split,
    // plus the two view-geometry covariates the truncation gate does NOT measure: obliquity (how grazing the
    // tabletop view is → foreshortened footprint) and completeness (observed vs believed footprint area).
    float dbg_yaw_pre       = 0.0f;   // belief yaw at cycle start, before the update (rad)
    float dbg_obliquity_cos = 1.0f;   // |cos(incidence)| of camera→table ray vs tabletop normal +z (1=top-down, →0 grazing)
    float dbg_completeness  = 1.0f;   // observed footprint area / believed w·h (<1 = partial/foreshortened mask)
    // PERSISTENT copy of the above (dbg_completeness is reset to 1 at the top of every fresh cycle, so it cannot
    // be read back). Survives cycles — including GATED ones, where no new footprint is measured and the last
    // known view quality is the right carry-forward. Feeds the COVERAGE common-mode in table_fitter.cpp: it is
    // the covariate that says how much of the believed footprint this viewpoint actually spans, which is what
    // licenses a frame to RESHAPE the table. One cycle stale by construction; harmless for a static object.
    float last_completeness = 1.0f;
    // FIXATION gate readouts (see TableConfig::fixation_enabled): which of the three attention conditions
    // held this cycle, and whether the geometry update was therefore admitted. Logged to the CSV so a
    // "why is my table frozen / why did it still move" question is answerable without a rebuild.
    bool  dbg_fixated     = true;
    bool  dbg_fix_close   = true;
    bool  dbg_fix_centred = true;
    bool  dbg_fix_still   = true;
    float dbg_dyaw_points   = 0.0f;   // this cycle's yaw move attributed to the per-point GN channel (rad)
    float dbg_dyaw_moment   = 0.0f;   // ...to the footprint-moment channel (rad)
    float dbg_dyaw_flip     = 0.0f;   // ...to the resolve_orientation 90° flip (rad)
    // Existence channel readouts from the last update_existence_and_remove (per-modality occ/free + counts).
    // *_free is the RAW sensor count; *_free_eff is what actually drove the log-odds after the range/occlusion
    // absence-confidence scaling AND the observed-guard (so raw→eff shows how much the absence was degraded).
    float dbg_ex_lidar_occ = 0.0f, dbg_ex_lidar_free = 0.0f, dbg_ex_lidar_free_eff = 0.0f;  int dbg_ex_lidar_n   = 0;
    // MEASURED clutter prior (rc::exist::measure_clutter): q = P(return | this neighbourhood, no table) from the
    // equal-volume shell around the box, and the ΔL the contrast actually admitted. Read these to see WHY the
    // LiDAR channel is confirming or silent: q ≈ the box's own return rate ⇒ indistinguishable ⇒ llr → 0.
    float dbg_ex_clutter_q = 0.0f;   int dbg_ex_clutter_n = 0;
    float dbg_ex_lidar_llr = 0.0f;   // the LiDAR channel's per-cycle ΔL after the contrast (was always +llr_occ)
    float dbg_ex_sil_occ   = 0.0f, dbg_ex_sil_free   = 0.0f, dbg_ex_sil_free_eff   = 0.0f;  int dbg_ex_sil_ndet  = 0;
    // Silhouette samples ATTEMPTED (SilhouetteExistence::n_total). Kept alongside n_detectable so the real
    // in-FoV FRACTION (ndet/ntotal) is recoverable — the phantom log's attribution needs "how much of the
    // object could the sensor actually have seen", which a bare detectable COUNT cannot answer.
    int   dbg_ex_sil_ntotal = 0;
    // Verification-gated removal diagnostics: p_detect (how resolving this view is), central_frac, and the
    // decayed go-verify surprise. Show why an absence went to REMOVAL (high p_detect) vs VERIFICATION (low).
    float dbg_ex_pdetect = 1.0f, dbg_ex_central = 0.0f;

    // ── Predicted in-image table ROI (current model projected through the camera extrinsic) ───────
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the table in the frame)
    // and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image
    // ★The two AXES behind that max, kept for DETECTOR-ENVELOPE CALIBRATION. p_detect is a function of the
    // max alone, but fitting min_fill/max_fill from (roi_fill, fired) pairs without the axes is a trap: a
    // grazing or edge-on view is a tall thin sliver whose MAX is large, so its misses land in high-fill bins
    // and drag the fitted max_fill shoulder down to explain something that is not a framing effect at all.
    // Logging both lets that bias be TESTED afterwards; it cannot be recovered from the max after the fact.
    // ── NBV emission (what the planner PROPOSED this cycle) ────────────────────────────────────────
    float dbg_nbv_standoff = 0.0f;   // face-relative stand-off chosen (m)
    float dbg_nbv_target_x = 0.0f, dbg_nbv_target_y = 0.0f;
    float dbg_nbv_pdetect  = 0.0f;   // P(detect) there
    float dbg_nbv_fill     = 0.0f;   // predicted roi_fill there
    float dbg_nbv_vfov     = 0.0f;   // camera vfov IN FORCE at proposal time; 0 ⇒ model was incomplete

    float roi_fill_h   = 0.0f;   // Δcol / W
    float roi_fill_v   = 0.0f;   // Δrow / H
};

}  // namespace rc
