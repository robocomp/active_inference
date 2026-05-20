#pragma once

#include <genericworker.h>
#include <fps/fps.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rc
{
    class VoxelOpenGLViewer;
}

class SceneProcessor
{
public:
    SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph,
                   std::mutex& lidar_points_mutex,
                   std::vector<float>& latest_lidar_xs,
                   std::vector<float>& latest_lidar_ys,
                   std::vector<float>& latest_lidar_zs,
                   std::uint64_t& latest_lidar_timestamp_ms,
                   std::atomic<bool>& lidar_stream_seen,
                   std::atomic<bool>& rgbd_stream_seen,
                   std::atomic<bool>& lidar_stream_wait_logged,
                   std::atomic<bool>& rgbd_stream_wait_logged);

    void configure(DSR::InnerEigenAPI* inner_eigen_api,
                   rc::VoxelOpenGLViewer* voxel_viewer,
                   bool transforms_interpolate_rt,
                   bool verbose_debug);

    std::pair<std::string, std::string> get_room_robot_names_for_compute();
    bool ensure_room_and_robot_ready(FPSCounter& compute_fps,
                                     const std::string& room_name,
                                     const std::string& robot_name);
    std::optional<Mat::RTMat> get_room_robot_transform(FPSCounter& compute_fps,
                                                       const std::string& room_name,
                                                       const std::string& robot_name,
                                                       std::uint64_t timestamp_ms);
    std::optional<Mat::RTMat> get_room_zed_transform(FPSCounter& compute_fps,
                                                     const std::string& room_name,
                                                     std::uint64_t timestamp_ms);
    std::uint64_t get_rgbd_frame_timestamp_ms(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const;
    void check_input_stream_startup_status();
    void log_room_robot_pose_periodic(const Mat::RTMat& room_T_robot) const;
    void mark_room_rt_ready();
    void update_room_polygon_periodic();
    void overlay_room_polygon_on_canvas(cv::Mat& canvas,
                                        const RoboCompCameraRGBDSimple::TRGBD& rgbd) const;
    void update_viewer_robot_pose(const Mat::RTMat& room_T_robot);
    void update_viewer_lidar_points(const std::string& room_name,
                                    const std::string& robot_name,
                                    const Mat::RTMat& room_T_robot_fallback);
    void update_viewer_graph_object_boxes(const std::string& room_name,
                                          std::uint64_t timestamp_ms);

private:
    struct RoomPolygonData
    {
        std::string room_name;
        std::vector<float> polygon_x;
        std::vector<float> polygon_y;
        float room_height = 2.4f;
    };

    struct RoomToCameraBasis
    {
        Mat::Vector3d origin{0.0, 0.0, 0.0};
        Mat::Vector3d axis_x{0.0, 0.0, 0.0};
        Mat::Vector3d axis_y{0.0, 0.0, 0.0};
        Mat::Vector3d axis_z{0.0, 0.0, 0.0};
    };

    struct GraphObjectBox
    {
        QVector3D min;
        QVector3D max;
        std::string category;
    };

    bool compute_room_to_camera_basis(const std::string& camera_node_name,
                                      const std::string& room_frame_name,
                                      std::uint64_t rt_timestamp,
                                      RoomToCameraBasis& basis) const;
    std::optional<RoomPolygonData> get_room_polygon_from_graph() const;
    std::optional<GraphObjectBox> build_graph_object_box(const DSR::Node& node,
                                                         const std::string& room_name,
                                                         std::uint64_t timestamp_ms) const;
    void update_room_polygon_in_viewers();

    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI* inner_eigen_api_ = nullptr;
    rc::VoxelOpenGLViewer* voxel_viewer_ = nullptr;
    std::mutex& lidar_points_mutex_;
    std::vector<float>& latest_lidar_xs_;
    std::vector<float>& latest_lidar_ys_;
    std::vector<float>& latest_lidar_zs_;
    std::uint64_t& latest_lidar_timestamp_ms_;
    std::atomic<bool>& lidar_stream_seen_;
    std::atomic<bool>& rgbd_stream_seen_;
    std::atomic<bool>& lidar_stream_wait_logged_;
    std::atomic<bool>& rgbd_stream_wait_logged_;
    bool transforms_interpolate_rt_ = true;
    bool verbose_debug_ = false;
    bool room_ready_logged_ = false;
    bool room_wait_logged_ = false;
    bool room_rt_ready_logged_ = false;
    bool room_rt_wait_logged_ = false;
    int polygon_check_count_ = 0;
    std::chrono::steady_clock::time_point input_stream_watchdog_start_ = std::chrono::steady_clock::now();
    mutable std::mutex node_names_mutex_;
    std::string room_node_name_;
    std::string robot_node_name_;
};