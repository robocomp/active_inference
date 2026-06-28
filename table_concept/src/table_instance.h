/*
 * table_instance.h
 *
 * Per-table runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h):
 * the generative model + sample queue, convergence/dead-band bookkeeping, the table-owned
 * voxel memory bank, and the epistemic affordance request.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "table_model.h"        // TableModel / TableState / FreeEnergyDecomposition
#include "sample_queue_geometry.h"   // common SampleQueue<Model> + table's geometry policy
#include "table_affordance.h"   // TableAffordance
#include "../../common/belief_stabilizer/belief_stabilizer.h"   // rc::StabilizerState

namespace rc {

struct TableInstance
{
    uint64_t    node_id;
    std::string node_name;

    TableModel  model;
    SampleQueue<TableModel> queue;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged    = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  frames_rising      = 0;      // consecutive frames with F increasing
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    int  processed_cycles   = 0;      // per-table compute cycles for log throttling
    // Tracker's gated mask assignment for THIS frame (index into the masks packet slices), or -1 to fall
    // back to greedy nearest-mask. Set each cycle by run_instance_tracker(); read in TableFitter::observe.
    int  assigned_mask_idx  = -1;
    bool model_stable       = false;
    int  model_generation   = 0;
    TableState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
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
    // TableFitter's stabilizer; this is just the per-table state (accumulators, gains, gate). info_w/
    // info_h are mirrors of stab.fisher_info[w/h], feeding the legacy acceptance-gain stiffener.
    StabilizerState<8> stab;
    ExtentDiagnostics    last_extent_diag{};   // footprint-extent vs fitted w/h + top/leg point split (size-bias diagnostic)
    bool epistemic_pending  = false;
    // Schmitt-trigger hysteresis for the epistemic affordance (anti-oscillation): once the expected
    // information gain ΔH falls below the withdraw threshold the table is "satisfied" and latched; it
    // stays withdrawn until a cooldown elapses AND ΔH climbs back above the (higher) re-arm threshold,
    // so it can't chatter across a single threshold as the belief jitters.
    bool epistemic_satisfied = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied table may re-arm
    float prev_free_energy  = std::numeric_limits<float>::max();
    // Running FE level on ACCEPTED (good) fresh frames (EMA); the FE-spike guard rejects a fresh frame
    // whose raw-fit FE is ≫ this (contaminated point set the model can't explain). NaN until first accept.
    float fe_baseline       = std::numeric_limits<float>::quiet_NaN();
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a
    // settled table stops jittering the voxelizer mesh (which renders the mesh attr, NOT the coarsely
    // dead-banded RT pose). Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening table
    // refreshes its edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();
    // Last coverage deficit (written by step_convergence, read by plot)
    float last_coverage_deficit = 0.f;
    // Higher-level confidence state driving warm-start precision from above
    float warm_confidence = 0.0f;
    SampleQueueMetrics last_queue_metrics;
    FreeEnergyDecomposition last_fe_terms;
    // Table-owned voxel memory bank (room frame), independent of per-frame uploads.
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    TableAffordance affordance;

    // ── Active-perception aids for the controller's local lock-on search ──────────
    // Detection aliveness: how recently YOLO produced a "table" mask for this instance, and the
    // confidence of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh table mask (0 = just detected)
    float last_mask_confidence   = 0.0f;      // YOLO confidence of the last table detection
    bool  detection_alive        = false;     // frames_since_detection < threshold

    // Predicted in-image table ROI from projecting the current model through the camera extrinsic.
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the table in the
    // frame) and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image
};

}  // namespace rc
