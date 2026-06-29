/*
 * chair_instance.h
 *
 * Per-chair runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the generative model + sample queue, convergence/dead-band bookkeeping, the chair-owned
 * voxel memory bank, and the epistemic affordance request.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "chair_model.h"        // ChairModel / ChairState / FreeEnergyDecomposition
#include "sample_queue_geometry.h"   // common SampleQueue<Model> + chair's geometry policy
#include "chair_affordance.h"   // ChairAffordance
#include "../../common/belief_stabilizer/belief_stabilizer.h"   // rc::StabilizerState

namespace rc {

struct ChairInstance
{
    uint64_t    node_id;
    std::string node_name;

    ChairModel  model;
    SampleQueue<ChairModel> queue;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged    = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  frames_rising      = 0;      // consecutive frames with F increasing
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask (chain-cov pinning)
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;  // Part B localization/chain cov (m²), added to the RT cov
    int  assigned_mask_idx  = -1;     // tracker's gated mask-slice assignment (-1 = use greedy nearest)
    int  unassigned_streak  = 0;      // consecutive tracker cycles with no mask assignment (stillbirth prune)
    int  processed_cycles   = 0;      // per-chair compute cycles for log throttling
    bool model_stable       = false;
    int  model_generation   = 0;
    ChairState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;
    int        settle_maturity = 0;    // cycles since last genuine new-evidence burst; drives acceptance-gain decay
    // Per-DOF accumulated observation info ("times viewed"): grows with integrated face coverage and
    // STIFFENS the w/h acceptance gain — a well-seen extent hardens (belief→knowledge) while an
    // unobserved face stays plastic until first seen. info_w ← x-faces, info_h ← y-faces.
    float      info_w = 0.0f;
    float      info_h = 0.0f;
    // ── Per-DOF belief stabiliser state ──────────────────────────────────────────
    // The Fisher-information filter + Kalman acceptance + CUSUM/SPRT gate, now the shared
    // rc::BeliefStabilizer over the 8 DOFs [cx,cy,w,h,H,leg,yaw,inset]. The algorithm + params live in
    // ChairFitter's stabilizer; this is just the per-chair state (accumulators, gains, gate). info_w/
    // info_h are mirrors of stab.fisher_info[w/h], feeding the legacy acceptance-gain stiffener.
    StabilizerState<8> stab;
    ExtentDiagnostics    last_extent_diag{};   // footprint-extent vs fitted w/h + top/leg point split (size-bias diagnostic)
    bool epistemic_pending  = false;
    // Schmitt-trigger hysteresis for the epistemic affordance (anti-oscillation): once the expected
    // information gain ΔH falls below the withdraw threshold the chair is "satisfied" and latched; it
    // stays withdrawn until a cooldown elapses AND ΔH climbs back above the (higher) re-arm threshold,
    // so it can't chatter across a single threshold as the belief jitters.
    bool epistemic_satisfied = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied chair may re-arm
    float prev_free_energy  = std::numeric_limits<float>::max();
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a
    // settled chair stops jittering the voxelizer mesh (which renders the mesh attr, NOT the coarsely
    // dead-banded RT pose). Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening chair
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();
    // Last coverage deficit (written by step_convergence, read by plot)
    float last_coverage_deficit = 0.f;
    // Higher-level confidence state driving warm-start precision from above
    float warm_confidence = 0.0f;
    SampleQueueMetrics last_queue_metrics;
    FreeEnergyDecomposition last_fe_terms;
    // Chair-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    ChairAffordance affordance;

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
};

}  // namespace rc
