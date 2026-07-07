/*
 * table_instance.h
 *
 * Per-table runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the geometry/state container + the AI2 full-covariance belief, convergence bookkeeping, the
 * table-owned voxel memory bank, and the epistemic affordance request.
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
#include "table_belief.h"       // AI2 full-covariance belief (TABLE_FIT_AI2.md)
#include "table_affordance.h"   // TableAffordance
#include "../../common/existence_belief/existence_belief.h"   // per-instance existence log-odds (removal)

namespace rc {

struct TableInstance
{
    uint64_t    node_id;
    std::string node_name;

    // Geometry / state container: holds the accepted pose+dims and the compound SDF used to split
    // the mask's support points into on-surface (candidate) vs off-surface (residual) sets.
    TableModel  model;

    // ── AI2 belief (TABLE_FIT_AI2.md) ─────────────────────────────────────────────
    // Full-covariance recursive filter over θ=[cx,cy,H,w,h,yaw]. Lazily initialised from the model
    // state on the first cycle; its result is written back into `model` so downstream publish/viewer
    // code is unchanged.
    TableBelief ai2_belief;
    bool        ai2_initialized = false;
    // This frame's ego-motion capture-corruption (from the selected mask slice): motion_var inflates the
    // observation precision R (downweight), trunc_frac gates the update (truncated→biased mask).
    // dotd is the motion-corruption speed, kept for diagnostics (see MASK_MOTION_CORRUPTION.md).
    float last_motion_var      = 0.0f;
    float last_motion_dotd     = 0.0f;
    float last_trunc_frac      = 0.0f;
    float last_centroid_radius = 0.0f;
    float last_range           = 0.0f;   // mean camera→mask depth Z (m), this frame — static range weighting

    // LiDAR range-channel diagnostics (this frame): #returns fed to the factor, #returns in a generous box,
    // and their mean |dist| to the current model surface. Few rays / large resid ⇒ wrong LidarFrameNode.
    int   dbg_lidar_rays       = 0;
    int   dbg_lidar_raw        = 0;
    float dbg_lidar_resid_m    = -1.0f;
    float dbg_lidar_meanz_m    = -1.0f;   // mean z of selected returns (room frame) — vs H detects a z-offset
    float dbg_lidar_topz_m     = -1.0f;   // mean z of the HIGHEST 20% of selected returns ≈ observed tabletop z
    float dbg_lidar_floorz_m   = -1.0f;   // mean z of the LOWEST 20% ≈ floor; should read ~0 if z-calib OK
    float dbg_lidar_cov_ang    = -1.0f;   // angular-coverage weight (1−R)^p ∈[0,1]; low ⇒ one-sided sweep

    // Divergence safety net (mirrors bottle): consecutive frames whose centre GN step was rejected as an
    // outlier (exceeded cfg.max_step_m). Non-zero ⇒ the fit tried to run away and was held.
    int   frames_diverged      = 0;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged   = 0;      // consecutive frames with |Δstate| < state_eps
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask (chain-cov pinning)
    // Wall-clock (agent's own steady clock) of the last belief touch — measured on EVERY inference cycle, not
    // just fresh ones — so a stale cycle can inflate Σ by the real elapsed time (measurement-age → covariance).
    std::chrono::steady_clock::time_point last_belief_touch{};
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;  // Part B localization/chain cov (m²), added to the RT cov
    int  processed_cycles   = 0;      // per-table compute cycles for log throttling
    // Tracker's gated mask assignment for THIS frame (index into the masks packet slices), or -1.
    // Set each cycle by run_instance_tracker(); read in TableFitter::observe.
    int  assigned_mask_idx  = -1;
    bool model_stable       = false;
    int  model_generation   = 0;
    TableState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;

    bool epistemic_pending  = false;
    // Schmitt-trigger hysteresis for the epistemic affordance (anti-oscillation).
    bool epistemic_satisfied = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied table may re-arm

    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a
    // settled table stops jittering the voxelizer mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening table
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // Table-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    TableAffordance affordance;

    // ── Active-perception aids for the controller's local lock-on search ──────────
    // Detection aliveness: how recently YOLO produced a "table" mask for this instance, and the
    // confidence of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh table mask (0 = just detected)
    float last_mask_confidence   = 0.0f;      // YOLO confidence of the last table detection
    bool  detection_alive        = false;     // frames_since_detection < threshold

    // Existence log-odds (removal): LiDAR free-space carve + mask-absence accrue negative evidence when the
    // table is demonstrably gone. Removed only when the removal decision holds for existence_remove_frames
    // consecutive cycles (debounce) — deleting furniture warrants SUSTAINED evidence, not a transient hiccup.
    rc::exist::ExistenceBelief existence;
    int existence_remove_streak = 0;

    // Predicted in-image table ROI from projecting the current model through the camera extrinsic.
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the table in the
    // frame) and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image
};

}  // namespace rc
