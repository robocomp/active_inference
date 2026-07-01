#pragma once
#include <genericworker.h>
#include <fps/fps.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "graph_object_box.h"
#include "rgbd_data.h"

namespace rc
{
    class VoxelOpenGLViewer;
}

namespace rc::media { class MediaSubscriber; }
namespace rc::media { class Image360Subscriber; }
namespace rc::media { class LidarSubscriber; }

class SceneProcessor
{
public:
    // Data bundles returned from DSR reads
    struct LidarData
    {
        std::vector<float> xs;
        std::vector<float> ys;
        std::vector<float> zs;
        std::uint64_t timestamp_ms = 0;
    };

    explicit SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph);
    ~SceneProcessor();

    void configure(DSR::InnerEigenAPI* inner_eigen_api,
                   rc::VoxelOpenGLViewer* voxel_viewer,
                   bool transforms_interpolate_rt,
                   bool verbose_debug,
                   bool mask_pose_extrapolate,
                   float mask_pose_extrap_max_dt_s);

    // Media plane (zero-copy DDS): RGBD pixels arrive here instead of via the DSR
    // graph. Camera intrinsics (focal) are still read from the static 'zed' node.
    bool init_media_plane(std::uint32_t domain_id,
                          const std::string& rgb_topic,
                          const std::string& depth_topic);

    // LiDAR over the same zero-copy media plane (robot_concept's LidarFrame stream). When use_media is
    // false (or init fails) the DSR graph 'lidar3D' node remains the source. Points share the sensor
    // frame of the graph laser_* attrs, so the downstream room_T_robot transform is unchanged.
    bool init_lidar_media_plane(std::uint32_t domain_id, const std::string& topic, bool use_media);

    // RGBD_360 panorama over the wide Image360Frame plane (display-only for the
    // Ricoh popup). Discovered from the "ricoh" node descriptor; falls back to the
    // configured domain/topic. Independent of the ZED rgb/depth pipeline.
    bool init_ricoh_media_plane(std::uint32_t domain_id, const std::string& topic);
    // Drain the ricoh subscriber; decodes into the latest-frame cache ONLY when
    // ricoh is wanted (window visible), so a hidden popup costs just a poll-discard.
    void poll_ricoh();
    void set_ricoh_wanted(bool on) { ricoh_wanted_.store(on, std::memory_order_relaxed); }
    // Latest decoded panorama (BGR, CV_8UC3); empty until a frame arrives.
    const cv::Mat& ricoh_bgr() const { return media_ricoh_.bgr; }
    // Latest ricoh source stamp (ms); recorded on every poll (even when not decoding) for rate telemetry.
    std::uint64_t ricoh_last_stamp_ms() const { return ricoh_last_stamp_ms_; }

    std::pair<std::string, std::string> get_room_robot_names_for_compute();
    bool ensure_room_and_robot_ready(FPSCounter& compute_fps,
                                     const std::string& room_name,
                                     const std::string& robot_name);
    std::optional<Mat::RTMat> get_room_robot_transform(FPSCounter& compute_fps,
                                                       const std::string& room_name,
                                                       const std::string& robot_name,
                                                       std::uint64_t timestamp_ms);
    // room→zed = room→robot (already resolved at frame time) ∘ robot→zed (static
    // camera extrinsics, queried at latest). Decomposing this way keeps the robot
    // pose time-correct for a dynamic room while the rigid extrinsics — which carry
    // only their bootstrap timestamp — are never pinned to a per-frame timestamp.
    std::optional<Mat::RTMat> get_room_zed_transform(FPSCounter& compute_fps,
                                                     const std::string& robot_name,
                                                     const Mat::RTMat& room_T_robot);
    // DSR-native data accessors (no proxy needed)
    std::uint64_t get_frame_timestamp_ms() const;
    std::optional<cv::Mat> get_rgb_from_dsr() const;
    // Latest LiDAR scan from the media plane (LidarFrame). std::nullopt if disabled or nothing received.
    std::optional<LidarData> get_lidar3D();
    std::optional<RGBDData> get_rgbd_frame_from_dsr() const;

    void check_input_stream_startup_status();
    void log_room_robot_pose_periodic(const Mat::RTMat& room_T_robot) const;
    void mark_room_rt_ready();
    void update_room_polygon_periodic();
    void overlay_room_polygon_on_canvas(cv::Mat& canvas, std::uint64_t frame_ts_ms) const;
    void update_viewer_robot_pose(const Mat::RTMat& room_T_robot);
    // Reads the LATEST robot pose (RT ts=0) and pushes it to the viewer, decoupled from the
    // per-frame camera timestamp. Cheap; called from the render timer so the robot redraws at the
    // render cadence instead of the perception-frame rate. No-op until room/robot/inner-eigen exist.
    void refresh_viewer_robot_pose_latest();
    void update_viewer_lidar_points(const std::string& room_name,
                                    const std::string& robot_name,
                                    const Mat::RTMat& room_T_robot_fallback);
    std::vector<GraphObjectBox> get_graph_object_boxes(const std::string& room_name,
                                                       std::uint64_t timestamp_ms) const;
    void update_viewer_graph_object_boxes(std::span<const GraphObjectBox> graph_boxes);
    void update_viewer_object_meshes();
    // Read human_concept's 'person' nodes (mesh_vertices_att = fitted BODY_18, room frame) and feed
    // them to the GL viewer as 3D skeletons.
    void update_viewer_person_skeletons();
    void update_viewer_table_rfe_points();
    // Feed the YOLO mask support points (room frame) from the "masks" node to the 3D viewer.
    void update_viewer_mask_points();

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
    bool transforms_interpolate_rt_ = true;
    bool verbose_debug_ = false;
    bool mask_pose_extrapolate_ = true;       // extrapolate robot pose to capture stamp via RT-edge velocity
    float mask_pose_extrap_max_dt_s_ = 0.2f;  // clamp the extrapolation horizon
    // Diagnostic CSV (etc/pose_extrap_log.csv): raw vs extrapolated pose + dt/velocity/displacement, so
    // the extrapolation magnitude (= the mask lag-bias being cancelled) can be analysed. Opened lazily.
    std::ofstream pose_extrap_csv_;
    bool          pose_extrap_csv_open_attempted_ = false;
    bool room_ready_logged_ = false;
    bool room_wait_logged_ = false;
    bool room_rt_ready_logged_ = false;
    bool room_rt_wait_logged_ = false;
    int polygon_check_count_ = 0;

    // Viewer-only EMA on the displayed robot pose. The RT edge now carries room_concept's 60 Hz
    // dead-reckoned predict-publish; reading its leading edge (ts=0) at the render-tick rate shows the
    // raw prediction noise + correction snaps → jitter. A short time-constant low-pass smooths the
    // DISPLAY without touching the mask pipeline (which queries its own capture-time pose).
    bool  robot_disp_sm_init_ = false;
    float robot_disp_sm_x_ = 0.f, robot_disp_sm_y_ = 0.f, robot_disp_sm_theta_ = 0.f;
    std::chrono::steady_clock::time_point robot_disp_sm_last_ = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point input_stream_watchdog_start_ = std::chrono::steady_clock::now();
    mutable std::mutex node_names_mutex_;
    std::string room_node_name_;
    std::string robot_node_name_;

    // --- Media plane RGBD source (replaces cam_rgb/cam_depth DSR blobs) ---
    void drain_media_plane() const;  // polls subscribers, refreshes the caches
    std::unique_ptr<rc::media::MediaSubscriber> media_rgb_sub_;
    std::unique_ptr<rc::media::MediaSubscriber> media_depth_sub_;

    // LiDAR media-plane source (preferred over the DSR graph when up). Latest scan cached in
    // sensor frame, refreshed by draining the subscriber inside get_lidar3D().
    std::unique_ptr<rc::media::LidarSubscriber> lidar_sub_;
    bool      lidar_use_media_ = false;
    LidarData media_lidar_;
    bool      media_lidar_valid_ = false;
    struct MediaRgbCache
    {
        bool          valid = false;
        std::uint64_t frame_id = 0;
        std::uint64_t stamp = 0;   // camera alivetime (ms), opaque timestamp
        int           width = 0;
        int           height = 0;
        cv::Mat       bgr;         // CV_8UC3
    };
    struct MediaDepthCache
    {
        bool          valid = false;
        std::uint64_t frame_id = 0;
        std::uint64_t stamp = 0;
        int           width = 0;
        int           height = 0;
        std::vector<float> depth;  // metric, row*width+col
    };
    mutable MediaRgbCache   media_rgb_;
    mutable MediaDepthCache media_depth_;

    // RGBD_360 panorama (display-only). Reuses the RGB cache shape.
    std::unique_ptr<rc::media::Image360Subscriber> media_ricoh_sub_;
    MediaRgbCache      media_ricoh_;
    std::atomic<bool>  ricoh_wanted_{false};   // set by the popup's show/hide toggle
    std::uint64_t      ricoh_last_stamp_ms_ = 0;   // last polled ricoh source stamp (rate telemetry)
};