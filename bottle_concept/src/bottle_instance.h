/*
 * bottle_instance.h
 *
 * Per-bottle runtime state shared by the bottle_concept collaborators
 * (BottleFitter fits it, BottleSceneGraph publishes it). One BottleInstance per
 * "bottle_N" cylinder node in the DSR graph.
 */

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "bottle_model.h"
#include "bottle_belief.h"           // rc::BottleBelief (AI2 recursive-Laplace belief — the fit)
#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
#include "../../common/existence_belief/existence_belief.h"   // rc::exist::ExistenceBelief (the shared policy)

namespace rc {

struct BottleInstance
{
    uint64_t    node_id = 0;
    std::string node_name;

    // The fit: the recursive-Laplace full-covariance belief on the shared engine (mirrors table/chair).
    // Lazily initialised from the model state on the first fresh frame; run_inference writes its posterior
    // back into `model` (the state carrier) so all downstream publish/viewer/RT code is unchanged. Its Σ
    // drives the tracker gate + NBV + published P_bottle.
    BottleModel  model;
    BottleBelief ai2_belief;
    bool         ai2_initialized = false;

    // Epistemic "go see the hidden face" affordance: a DSR node advertising the far-side viewpoint
    // (EpistemicPlanner) for the mission controller. The node persists and refreshes; its ΔH gain
    // governs selection (low nats → the controller won't pick it).
    ObjectAffordance affordance;
    bool epistemic_pending  = false;
    int  epistemic_cooldown = 0;   // post-completion hold (cycles) during which the gain is suppressed
    float last_epistemic_gain = 0.0f;   // most recent published ΔH (nats) — for the dashboard

    int  matched_frames        = 0;     // frames with fresh sensing data
    bool reseed_requested      = false; // move-experiment: force a fresh cold-start at the next pose

    // Part B diagnostics (published RT-edge covariance, room frame, m²) — set in write_rt_pose: the
    // chain/localization term J·Σ_chain·Jᵀ and the TOTAL published xx,yy (fit+chain).
    float dbg_chain_cov_xx = -1.0f, dbg_chain_cov_yy = -1.0f;
    float dbg_rtcov_xx     = -1.0f, dbg_rtcov_yy     = -1.0f;
    // Self-gate state for the shared RT-cov publisher: bottle used to rewrite the edge EVERY cycle,
    // and an RT edge write is a CRDT delta every peer must merge (see the dot-cloud pathology).
    float last_pub_cov_trace = -1.0f;
    // LiDAR range-channel diagnostics (set in feed_lidar): returns SELECTED for this instance this frame and
    // their mean |SDF| to the current model surface. A healthy frame = tens of rays at a few-cm residual; a
    // wrong LidarFrameNode (mount double-applied) shows ~0 rays or a large systematic residual.
    int   dbg_lidar_rays    = 0;
    int   dbg_lidar_raw     = 0;       // returns in a GENEROUS box (before the tight select) — raw≈selected ⇒
                                       // returns physically absent (occlusion/scan geometry, a VIEWPOINT problem);
                                       // raw≫selected ⇒ they're there but the tight box misses them (offset/calib).
    float dbg_lidar_resid_m = -1.0f;   // -1 = no LiDAR this frame
    // Divergence persistence: consecutive frames whose fit explained NONE of its data (belief energy == 0,
    // i.e. every point fell to the clutter component — a drifted/inflated model). Retired past a config bound.
    int   frames_diverged   = 0;
    // Negative-information persistence: true only when the bottle centre projects INSIDE the camera
    // frustum this cycle. The tracker accrues a death "miss" only when expected_visible — so out-of-FoV
    // the bottle PERSISTS, and it is retired only if it should be seen yet isn't (removed). Default
    // false = don't kill until we confirm it is in view.
    bool          expected_visible = false;
    int  frames_converged      = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  last_masks_frame_seen = -1;    // last masks packet frame consumed
    int  processed_cycles      = 0;     // per-bottle compute cycles for log throttling
    int  model_generation      = 0;
    // ── YOLO detection aliveness (active-perception feedback for the affordance contract) ──────────
    // The producer publishes bottle_detection_alive/_confidence on the bottle node so the controller's
    // servo lock-on completes on "YOLO is firing on this bottle". frames_since_detection ticks every
    // observe() cycle and resets to 0 when a fresh bottle mask is selected; detection_alive is the
    // thresholded form. last_pub_* dead-band the publish so a settled bottle stops rewriting the node.
    // Mask slice assigned to THIS instance by the InstanceTracker this cycle (-1 = none → observe()
    // falls back to greedy nearest). Set by the worker's tracker step; read in observe().
    int   assigned_mask_idx        = -1;
    int   frames_since_detection   = 100000;   // cycles since the last fresh bottle mask (0 = just seen)
    float last_mask_confidence     = 0.0f;     // YOLO confidence of the last selected bottle mask
    float last_motion_dotd         = 0.0f;     // last slice's ego-motion smear speed Z·‖ṡ‖ (m/s) → common-mode fixation
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask frame (chain-cov pinning)
    // Agent-clock stamp of the last belief touch (set EVERY inference cycle) so a stale cycle inflates the
    // position Σ by the real elapsed time (measurement-age → covariance) rather than a fixed per-frame step.
    std::chrono::steady_clock::time_point last_belief_touch{};
    bool  detection_alive          = false;    // frames_since_detection < threshold
    bool  last_pub_detection_alive = false;    // dead-band trace for the published flag
    float last_pub_detection_conf  = -1.0f;    // dead-band trace for the published confidence
    float prev_free_energy     = std::numeric_limits<float>::max();
    // Clutter-inclusive free-energy readout (TABLE.md §3) + attention baseline (§9). last_clutter_frac is the
    // honest "explains none of its data" signal (replaces the old surface-energy==0 divergence sentinel).
    // fe_baseline tracks DOWN fast / UP slow so a sustained rise (the bottle moved) surfaces as fe_surprise.
    float last_clutter_frac    = 0.0f;
    // P(a mover is in contact) this cycle — the CAUSE that licenses the position to be volatile. Logged
    // because a position that suddenly starts moving is only explicable alongside this number.
    float mover_p              = 0.0f;
    float fe_baseline          = -1.0f;   // <0 = uninitialised (seed to the first explained FE)
    // ── EXISTENCE (bottle_existence.cpp; the shared rc::exist policy) ─────────────────────────────
    // L = log P(exists)/P(¬exists). Removal is a decision on THIS, never a miss counter (invariant 5) —
    // bottle was the last object agent still retiring on tracker_death_frames. Seeded once, at the first
    // cycle with a belief to project (a bottle with no geometry has nothing to be absent from).
    rc::exist::ExistenceBelief existence{0.0f, 4.0f};
    bool  existence_seeded        = false;
    // Debounce state, SHARED (rc::exist::RemovalDebounce): the streak in ideal observations plus the
    // consecutive-starved count that makes a condemned-but-unexecutable instance visible instead of frozen.
    rc::exist::RemovalDebounce existence_debounce;
    float exist_logodds           = 0.0f;   // mirror of existence.logodds() for the dashboard / strip / logs
    // Last cycle's camera-channel geometry, for the dashboard and the phantom-event record. These are what
    // make a death ATTRIBUTABLE: "removed at p_detect 0.71, fill 0.28, vis 1.00" is a claim that can be
    // checked; the old "unsupported for 90 frames" was not.
    float dbg_ex_p_detect   = 0.0f;
    float dbg_ex_fill       = 0.0f;
    float dbg_ex_vis        = 0.0f;
    float dbg_ex_lidar_occ  = 0.0f;
    float dbg_ex_lidar_free = 0.0f;
    int   dbg_ex_lidar_n    = 0;
    float dbg_ex_lidar_pres = 0.0f;   // P(the sweep can resolve an object this small) — see bottle_existence.cpp
    float fe_surprise          = 0.0f;
    // ★The PUBLISHED epistemic_gain — the number the controller's affordance selection ranks on
    // (efe_score = gain − lambda_cost*dist, lambda_cost = 0.2/m, switch_margin = 0.5). Logged because
    // cross-agent selection was UNAUDITABLE: door was the only agent recording its gain, so "why does the
    // robot never visit a door" could not be answered against what a table or a fridge actually offers.
    float dbg_nbv_gain = 0.0f;
    // Off-surface (unexplained) support points of the last FRESH frame. HELD between masks on purpose so
    // the dashboard trace keeps its last real value instead of crashing to 0 on every idle cycle.
    int   dbg_resid_pts        = 0;
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx = std::numeric_limits<float>::max();
    float last_written_cy = std::numeric_limits<float>::max();
    // Last model PUBLISHED to the graph — gate node/edge writes to meaningful changes so a
    // stable fit stops rewriting the node (big mesh + model_generation) and the RT edge.
    float last_pub_radius = std::numeric_limits<float>::max();
    float last_pub_height = std::numeric_limits<float>::max();
    float last_pub_cx     = std::numeric_limits<float>::max();
    float last_pub_cy     = std::numeric_limits<float>::max();
    float last_pub_cz     = std::numeric_limits<float>::max();
    // Bottle-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f>      voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Table-top z (room frame) the bottle stands on, when a "table" node is found under it; NaN
    // otherwise (bottle then "hangs from the room", no z anchor / no surface filter). Refreshed
    // each cycle. Anchors cz = table_top_z + height/2 and floors point ingestion at the surface.
    float table_top_z = std::numeric_limits<float>::quiet_NaN();
    // RT-tree parent the bottle "hangs from": the table node when found, else the room. The fit is
    // always in the room frame; the RT edge (parent→bottle) is written in the parent frame.
    std::uint64_t parent_id = 0;
    std::string   parent_name = "room";
    // Support-surface decision hysteresis: a challenger surface must win for support_commit_cycles
    // consecutive cycles before we re-parent (anti-thrash at table edges / the floor↔table boundary).
    std::uint64_t support_challenger_id    = 0;
    int           support_challenger_count = 0;
    bool          support_reparent_pending = false;   // tells write_rt_pose to swap the RT parent edge
};

}  // namespace rc
