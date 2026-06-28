/*
 * bottle_instance.h
 *
 * Per-bottle runtime state shared by the bottle_concept collaborators
 * (BottleFitter fits it, BottleSceneGraph publishes it). One BottleInstance per
 * "bottle_N" cylinder node in the DSR graph.
 */

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "bottle_model.h"
#include "sample_queue_geometry.h"   // common SampleQueue<Model> + bottle's geometry policy
#include "bottle_affordance.h"
#include "../../common/belief_stabilizer/belief_stabilizer.h"   // rc::StabilizerState
#include "../../common/motion_filter/cv_filter.h"               // rc::CVFilter (movable-object tracking)

namespace rc {

struct BottleInstance
{
    uint64_t    node_id = 0;
    std::string node_name;

    BottleModel model;
    SampleQueue<BottleModel> queue;

    // Epistemic "go see the hidden face" affordance: a DSR node advertising the far-side viewpoint
    // (EpistemicPlanner) for the mission controller. The node persists and refreshes; its ΔH gain
    // governs selection (low nats → the controller won't pick it).
    BottleAffordance affordance;
    bool epistemic_pending  = false;
    int  epistemic_cooldown = 0;   // post-completion hold (cycles) during which the gain is suppressed
    float last_epistemic_gain = 0.0f;   // most recent published ΔH (nats) — for the dashboard

    int  matched_frames        = 0;     // frames with fresh sensing data
    bool reseed_requested      = false; // move-experiment: force a fresh cold-start at the next pose

    // ── Motion model (movable object) ──────────────────────────────────────────────────────────────
    // A bottle is MOVABLE: its position TRACKS via a constant-velocity filter on (cx,cy) instead of
    // hardening (cz is table-anchored; radius/height harden via the stabiliser). Replaces the
    // maturity-locked position gain that froze the fit and made a moved bottle spawn a new instance.
    CVFilter      motion;                  // CV KF on cx,cy (2 axes); empty until first fit
    std::uint64_t last_motion_ts_ms = 0;   // capture stamp of the last CV update (for dt)
    int           last_motion_frame  = -1; // mask frame id of the last CV update (fresh-frame guard)
    Eigen::Vector3f last_obs_centroid = Eigen::Vector3f::Zero();  // fresh-frame mask centroid = CV measurement
                                           // (FOLLOWS the bottle; the queue-anchored SDF fit lags)
    // Part B diagnostics (published RT-edge covariance, room frame, m²) — set in write_rt_pose, logged in
    // the fisher CSV: the chain/localization term J·Σ_chain·Jᵀ and the TOTAL published xx,yy (fit+chain).
    float dbg_chain_cov_xx = -1.0f, dbg_chain_cov_yy = -1.0f;
    float dbg_rtcov_xx     = -1.0f, dbg_rtcov_yy     = -1.0f;
    // Grasp-mode seam (Part B step 2, not yet wired): when the controller grasps the bottle its motion
    // becomes a KNOWN control input (the arm). The handshake will set this + switch the transition to
    // known-input / re-parent-under-gripper. Left false for the free-movable CV mode.
    bool          grasped = false;
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
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask frame (chain-cov pinning)
    bool  detection_alive          = false;    // frames_since_detection < threshold
    bool  last_pub_detection_alive = false;    // dead-band trace for the published flag
    float last_pub_detection_conf  = -1.0f;    // dead-band trace for the published confidence
    float prev_free_energy     = std::numeric_limits<float>::max();
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
    SampleQueueMetrics      last_queue_metrics;
    FreeEnergyDecomposition last_fe_terms;
    // ── Fisher information filter (per-DOF; currently DIAGNOSTIC) ──────────────────────────────────
    // The principled replacement for a "times viewed" proxy: a per-DOF information filter over
    // [cx,cy,cz,radius,height] (matches BottleState::to_array()). Each fresh mask measures the
    // observation Fisher information (curvature of the SDF data-likelihood); the filter accumulates
    // it across viewpoints so a well-seen DOF hardens while an unobserved one stays plastic — the
    // stabiliser for successive gatherings of evidence as the robot orbits the table. The Q-bleed
    // predict keeps the precision at a finite steady state (the fix for P_bottle overconfidence).
    // Phase 1–2: folded + logged only — not yet driving acceptance or the published covariance.
    // Shared rc::BeliefStabilizer state over the 5 DOFs [cx,cy,cz,radius,height]. Currently only the
    // Fisher accumulators are exercised (diagnostic); the Kalman/CUSUM fields stay zero until bottle's
    // acceptance is wired to compute_acceptance (a future behaviour change).
    StabilizerState<5> stab;
    // Previous fresh-frame state, so the stabiliser's CUSUM can be run DIAGNOSTICALLY (innovation =
    // this fit vs last) to populate stab.counter_evidence for the dashboard — without applying the
    // gate to the fit (bottle still uses its convergence gate, not the Kalman/CUSUM acceptance).
    std::array<float, 5> prev_diag_state{};
    bool has_prev_diag = false;
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
