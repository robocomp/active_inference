/*
 * graph_publisher.h
 *
 * The voxelizer's DSR "semantic_grid" export:
 *   - "masks" — the live product, deprojected YOLO masks (room frame) consumed by
 *               the concept agents. Published every cycle.
 *
 * Owns the per-node "ready" state and the masks publish sequence; the graph relayout
 * is injected as a callback so this stays decoupled from the GUI (graph_viewers).
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G exists.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_eigen_defs.h>

#include "rgbd_data.h"
#include "voxelizer_params.h"

struct SegDetection;
namespace rc::human_pose { struct PoseDetection; }

// A detection from the RGB-360 (ricoh) panorama. Published into the SAME "masks" node. Two flavours:
//   - bearing-only (has_lidar_depth=false): a no-depth slice (has_depth=0) tagged with a room-frame
//     bearing instead of a 3D centroid (Part B legacy).
//   - depth-carrying (has_lidar_depth=true): the lidar reprojected into the panorama gave this mask a
//     3D range, so it's published as a FULL 3D slice (has_depth=1) with room-frame support points and a
//     mask_depth_var (m²) the consumer adds to R — same currency as the interaction-matrix motion_var.
struct BearingDetection
{
    std::string label;
    float       class_id = -1.0f;
    float       confidence = 0.0f;
    float       azimuth_room_rad = 0.0f;   // room-frame bearing to the detection (fallback channel)

    bool                          has_lidar_depth = false;
    float                         range_var = 0.0f;               // σ_range² (m²) → mask_depth_var
    Eigen::Vector3f               centroid_room = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f>  support_room;                   // lidar hits inside the mask (room frame)
};

class GraphPublisher
{
public:
    GraphPublisher(std::shared_ptr<DSR::DSRGraph> graph,
                   const VoxelizerParams& params,
                   std::function<void()> relayout);

    // Publish masks for this cycle. frame_ts_ms = capture stamp of the rgbd/depth frame; published on
    // the masks node so consumers can pin their pose lookups to capture time (0 = unknown → consumers
    // fall back to latest pose).
    void publish(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                 const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms = 0,
                 const std::vector<BearingDetection>& bearing_detections = {});

    // Publish human skeletons on the dedicated 'skeleton' node. Keypoints are BODY_18, deprojected
    // to the CAMERA (zed) frame — NOT room frame — so a consumer interacting with a person can
    // servo robot-relative (re-parent under the robot in the RT tree) without paying localization
    // noise. frame_ts_ms pins the camera-pose lookup at capture time. No-op unless poses present.
    void publish_skeletons(const RGBDData& rgbd,
                           const std::vector<rc::human_pose::PoseDetection>& poses,
                           std::uint64_t frame_ts_ms = 0);

    // Delete every "semantic_grid" node this agent left in the graph + reset ready flags.
    void cleanup_semantic_grid_nodes();

private:
    // Create the named semantic_grid node under "zed" (+ identity RT edge) if absent. Returns true
    // once the node exists/is ready. `relayout` triggers a graph relayout after creation (masks/voxels
    // do; tracks deliberately don't, to avoid a churn-time viewer repaint of a freed item).
    bool ensure_node(const char* name, const char* color, bool& ready, bool relayout);

    void upload_masks(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                      const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms,
                      const std::vector<BearingDetection>& bearing_detections);
    void upload_skeletons(const RGBDData& rgbd,
                          const std::vector<rc::human_pose::PoseDetection>& poses,
                          std::uint64_t frame_ts_ms);

    std::shared_ptr<DSR::DSRGraph> G_;
    const VoxelizerParams&         params_;
    std::function<void()>          relayout_;

    // Previous zed pose + capture stamp, kept to finite-difference the camera twist for the per-mask
    // ego-motion corruption annotation (common/motion_corruption). Updated each masks upload.
    Eigen::Matrix3f last_zed_R_    = Eigen::Matrix3f::Identity();
    Eigen::Vector3f last_zed_t_    = Eigen::Vector3f::Zero();
    std::uint64_t   last_zed_ts_ms_ = 0;
    bool            have_last_zed_  = false;

    // Diagnostic CSV for the ego-motion corruption (MaskMotion.csv_log). Opened lazily on first write.
    std::ofstream   motion_csv_;
    bool            motion_csv_open_attempted_ = false;

    bool          masks_ready_    = false;
    bool          skeleton_ready_ = false;
    std::uint64_t last_masks_uploaded_frame_    = 0;
    std::uint64_t masks_publish_seq_            = 0;
    std::uint64_t last_skeleton_uploaded_frame_ = 0;
    std::uint64_t skeleton_publish_seq_         = 0;
    int           last_logged_skeleton_count_   = -1;   // log only on a count change (0↔N↔M)
};
