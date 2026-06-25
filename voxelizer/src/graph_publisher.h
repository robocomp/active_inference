/*
 * graph_publisher.h
 *
 * All of the voxelizer's DSR "semantic_grid" exports in one place:
 *   - "masks"  — the live product, deprojected YOLO masks (room frame) consumed by
 *                the concept agents. Published every cycle.
 *   - "tracks" — class-agnostic voxel tracks. Config-gated (Voxel.publish_tracks),
 *                default OFF (only the deferred table_concept ever read it).
 *   - "voxels" — the semantic voxel grid. Config-gated (Voxel.publish_voxels),
 *                default OFF (no consumer; the GL viewer is fed directly).
 *
 * Owns the per-node "ready" state and the masks publish sequence; the graph relayout
 * is injected as a callback so this stays decoupled from the GUI (graph_viewers).
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the processors exist.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_eigen_defs.h>

#include "rgbd_data.h"
#include "voxelizer_params.h"

class VoxelProcessor;
class UnifiedVoxelGrid;
struct SegDetection;

class GraphPublisher
{
public:
    GraphPublisher(std::shared_ptr<DSR::DSRGraph> graph,
                   const VoxelizerParams& params,
                   VoxelProcessor* voxel_processor,
                   UnifiedVoxelGrid* voxel_grid,
                   std::function<void()> relayout);

    // Publish every enabled export for this cycle: masks (always), tracks/voxels (config-gated).
    // frame_ts_ms = capture stamp of the rgbd/depth frame; published on the masks node so consumers can
    // pin their pose lookups to capture time (0 = unknown → consumers fall back to latest pose).
    void publish(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                 const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms = 0);

    // Re-publish the (possibly just-cleared) voxel grid — for the "Clear Voxels" button.
    // No-op unless Voxel.publish_voxels is set.
    void refresh_voxels_node();

    // Delete every "semantic_grid" node this agent left in the graph + reset ready flags.
    void cleanup_semantic_grid_nodes();

private:
    // Create the named semantic_grid node under "zed" (+ identity RT edge) if absent. Returns true
    // once the node exists/is ready. `relayout` triggers a graph relayout after creation (masks/voxels
    // do; tracks deliberately don't, to avoid a churn-time viewer repaint of a freed item).
    bool ensure_node(const char* name, const char* color, bool& ready, bool relayout);

    void upload_masks(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                      const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms);
    void publish_tracks();
    void upload_voxels();

    std::shared_ptr<DSR::DSRGraph> G_;
    const VoxelizerParams&         params_;
    VoxelProcessor*                voxel_processor_ = nullptr;
    UnifiedVoxelGrid*              voxel_grid_      = nullptr;
    std::function<void()>          relayout_;

    bool          masks_ready_  = false;
    bool          tracks_ready_ = false;
    bool          voxels_ready_ = false;
    std::uint64_t last_masks_uploaded_frame_ = 0;
    std::uint64_t masks_publish_seq_         = 0;
};
