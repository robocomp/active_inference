/*
 * bottle_instance.h
 *
 * Per-bottle runtime state shared by the bottle_concept collaborators
 * (BottleFitter fits it, BottleSceneGraph publishes it). One BottleInstance per
 * "bottle_N" cylinder node in the DSR graph.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "bottle_model.h"
#include "sample_queue.h"

namespace rc {

struct BottleInstance
{
    uint64_t    node_id = 0;
    std::string node_name;

    BottleModel model;
    SampleQueue queue;

    int  matched_frames        = 0;     // frames with fresh sensing data
    bool reseed_requested      = false; // move-experiment: force a fresh cold-start at the next pose
    int  frames_converged      = 0;     // consecutive frames with |ΔFE| < fe_eps
    int  last_masks_frame_seen = -1;    // last masks packet frame consumed
    int  processed_cycles      = 0;     // per-bottle compute cycles for log throttling
    int  model_generation      = 0;
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
};

}  // namespace rc
