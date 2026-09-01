#pragma once

/*
 * scene_feed.h — reads the shared DSR graph (+ the LiDAR media plane) and pushes what it finds into
 * the 3-D metric view. Split out of retina/src/scene_processor.cpp, whose viewer-feeding half this
 * is, verbatim in behaviour.
 *
 * ★WHY THIS IS NOT IN THE RETINA ANY MORE. Nothing it draws is the retina's: the room polygon is
 * room_concept's, the object boxes and display meshes belong to the concept agents, the skeletons to
 * human_concept, the occupancy grid to residual_concept, the LiDAR to the media plane. The single
 * exception is the YOLO mask cloud, and that arrives through the graph like everything else. A
 * perception component was carrying ~2.8 k lines of renderer for other agents' products, and — the part
 * that actually cost something — the room dependency this view genuinely needs was keeping the whole
 * agent, YOLO included, gated behind localization.
 *
 * FRAME. Everything is drawn in the ROOM frame; the GL widget applies its own room→GL axis swap. The
 * mask cloud is the one input that arrives in CAMERA frame (the retina publishes the raw
 * deprojection now), so this class transforms it, pinned to the mask capture stamp — not to "latest",
 * or the cloud lags the robot pose and shimmers under motion.
 *
 * THREADING. Main thread only. get_transformation_matrix with ts==0 walks InnerEigenAPI's UNLOCKED
 * cache (CLAUDE.md), and the refresh is a QTimer on the main thread. No update_node/update_edge signal
 * is connected — those fire on the FastDDS reader threads.
 */

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "graph_object_box.h"

namespace rc { class VoxelOpenGLViewer; }

namespace rc
{

class SceneFeed
{
public:
    SceneFeed(std::shared_ptr<DSR::DSRGraph> graph, VoxelOpenGLViewer* viewer);

    // The two frames EVERY drawing pass is expressed in. Reported by refresh() so the agent can hold
    // in Waiting and say so, instead of asking cortex for a transform that cannot resolve — which is
    // what produced `get_transformation_matrix … origen or dest nodes do not exist` at the refresh
    // rate: a log flood standing in for a state change.
    struct WorldFrames
    {
        std::string room_name;    // empty ⇒ no node of type "room" in the graph right now
        std::string robot_name;   // empty ⇒ no node of type "robot"
        [[nodiscard]] bool ready() const { return not room_name.empty() and not robot_name.empty(); }
    };

    // Names are discovered by TYPE, never hardcoded, and memoized — but the memo is DROPPED as soon as
    // the node behind it leaves the graph (see the definition).
    std::pair<std::string, std::string> room_robot_names();

    // One full refresh: everything below, in the order the view expects. Safe to call with pieces of
    // the graph missing — each step no-ops rather than blanking what it cannot read this tick.
    WorldFrames refresh();

    // LiDAR arrives on the media plane (domain 7), NOT through the graph. Returns false if the
    // descriptor/stream is not up yet; safe to retry every tick.
    bool init_lidar(const std::string& topic);

    void set_verbose(bool v) { verbose_ = v; }

private:
    struct RoomPolygonData
    {
        std::string room_name;
        std::vector<float> polygon_x;
        std::vector<float> polygon_y;
        float room_height = 2.4f;
    };

    std::optional<RoomPolygonData> get_room_polygon_from_graph() const;
    std::optional<GraphObjectBox> build_graph_object_box(const DSR::Node& node,
                                                         const std::string& room_name,
                                                         std::uint64_t timestamp_ms) const;
    std::vector<GraphObjectBox> get_graph_object_boxes(const std::string& room_name,
                                                       std::uint64_t timestamp_ms) const;
    // Forward-extrapolate room←robot over the RT lag, then compose the static robot←zed mount. Same
    // correction the retina applies to its own transforms; without it the mask cloud is drawn
    // against a pose ~90 ms stale, which reads as shimmer under rotation.
    void forward_extrapolate_room_T_robot(Mat::RTMat& room_T_robot, const std::string& room_name,
                                          const std::string& robot_name, std::uint64_t timestamp_ms) const;
    std::optional<Mat::RTMat> room_T_zed_extrapolated(DSR::InnerEigenAPI* eigen,
                                                      const std::string& room_name,
                                                      const std::string& robot_name,
                                                      std::uint64_t stamp) const;

    void update_room_polygon_in_viewers();
    void update_viewer_graph_object_boxes(std::span<const GraphObjectBox> graph_boxes);
    void update_viewer_object_meshes();
    void update_viewer_person_skeletons();
    void update_viewer_table_rfe_points();
    void update_viewer_mask_points();
    void update_viewer_grid();
    void update_viewer_robot_pose(const Mat::RTMat& room_T_robot);
    void refresh_viewer_robot_pose_latest();
    void update_viewer_lidar_points(std::span<const Eigen::Vector3f> lidar_points_room,
                                    std::span<const std::uint8_t> plane_id = {});
    void poll_lidar();

    std::shared_ptr<DSR::DSRGraph>      graph_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_owned_;
    DSR::InnerEigenAPI*                 inner_eigen_api_ = nullptr;
    VoxelOpenGLViewer*                  voxel_viewer_    = nullptr;

    std::string room_node_name_;
    std::string robot_node_name_;
    // DISPLAY-ONLY smoothing of the drawn robot pose. A time-constant EMA, so it is rate-independent:
    // it averages out localization correction snaps without lagging real motion by more than ~tau.
    // Nothing but the glyph reads these — no belief, no transform.
    bool  robot_disp_sm_init_  = false;
    float robot_disp_sm_x_     = 0.0f;
    float robot_disp_sm_y_     = 0.0f;
    float robot_disp_sm_theta_ = 0.0f;
    std::chrono::steady_clock::time_point robot_disp_sm_last_{};

    bool  verbose_ = false;
    bool  transforms_interpolate_rt_ = true;
    bool  mask_pose_extrapolate_ = true;
    float mask_pose_extrap_max_dt_s_ = 0.2f;
    int   polygon_tick_ = 0;   // the room polygon is static-ish; rebuild it 1 tick in 50
};

}   // namespace rc
