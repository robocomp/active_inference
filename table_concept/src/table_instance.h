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
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    TableAffordance affordance;
};

}  // namespace rc
