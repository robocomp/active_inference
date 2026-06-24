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
#include "sample_queue.h"       // SampleQueue / SampleQueueMetrics
#include "table_affordance.h"   // TableAffordance

namespace rc {

struct TableInstance
{
    uint64_t    node_id;
    std::string node_name;

    TableModel  model;
    SampleQueue queue;

    int  last_frame_seen    = -1;     // last_sensing_frame_att value read
    int  matched_frames     = 0;      // frames with fresh sensing data
    int  frames_converged    = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  frames_rising      = 0;      // consecutive frames with F increasing
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    int  processed_cycles   = 0;      // per-table compute cycles for log throttling
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
    // ── Fisher information filter ─────────────────────────────────────────────────
    // The principled replacement for info_w/info_h: a per-DOF information filter over
    // [cx,cy,w,h,H,leg,yaw,inset]. Each fresh mask measures the observation Fisher information
    // (curvature of the SDF data-likelihood); the filter accumulates it across viewpoints so a
    // well-seen DOF hardens while an unobserved one stays plastic — the stabiliser for successive
    // gatherings of evidence as the robot orbits the table. info_w/info_h are mirrors of the
    // (normalised) accumulators for DOFs w/h, feeding the existing acceptance-gain stiffener.
    std::array<float, 8> fisher_info{};        // normalised "equivalent views" accumulator (drives stiffness)
    std::array<float, 8> fisher_info_raw{};    // raw accumulated Fisher precision Σ⁻¹ (for posterior std / EFE)
    std::array<float, 8> fisher_info_peak{};   // per-DOF adaptive normaliser (best single-view info seen)
    std::array<float, 8> last_obs_info{};      // most recent frame's raw Fisher diagonal
    std::array<float, 8> last_kalman_gain{};   // per-DOF acceptance gain K=obs/(Y_pred+obs) (the calibrated stiffness)
    bool epistemic_pending  = false;
    float prev_free_energy  = std::numeric_limits<float>::max();
    // Dead-band tracking for write_rt_pose — suppress tiny oscillations
    float last_written_cx   = std::numeric_limits<float>::max();
    float last_written_cy   = std::numeric_limits<float>::max();
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
