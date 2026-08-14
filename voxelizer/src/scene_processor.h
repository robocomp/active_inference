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
#include "media_plane_source.h"   // MediaPlaneSource + LidarData (DDS media ingest, owned by this class)

namespace rc
{
}

class SceneProcessor
{
public:
    explicit SceneProcessor(const std::shared_ptr<DSR::DSRGraph>& graph);
    ~SceneProcessor();

    // The shared DSR graph (its API is thread-safe — shared_mutex). Handed to worker threads that do
    // their own graph reads (e.g. RicohYoloWorker resolving room<-ricoh on its own thread).
    std::shared_ptr<DSR::DSRGraph> graph() const { return graph_; }

    void configure(DSR::InnerEigenAPI* inner_eigen_api,
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
    bool ricoh_available() const { return media_ and media_->ricoh_available(); }
    // Drain the ricoh subscriber; decodes into the latest-frame cache when ricoh is wanted (window
    // visible) OR force=true (a one-off decode requested by e.g. rc::RicohYoloWorker's own thread), so
    // a hidden-and-unforced poll costs just a poll-discard. May be called from the ricoh worker thread
    // AND the main thread (on_render_tick, when the worker isn't running) — the cache is mutex-guarded.
    void poll_ricoh(bool force = false);
    void set_ricoh_wanted(bool on) { if (media_) media_->set_ricoh_wanted(on); }
    // Thread-safe snapshot of the latest decoded panorama (BGR, CV_8UC3); empty until a frame arrives.
    // A cv::Mat copy is a cheap shallow (refcounted) copy, not a pixel copy — safe to call from any
    // thread while poll_ricoh() concurrently overwrites the cache on another.
    cv::Mat ricoh_bgr_copy() const;
    // Latest ricoh source stamp (ms); recorded on every poll (even when not decoding) for rate telemetry.
    // Atomic — safe to read from any thread regardless of who last called poll_ricoh().
    std::uint64_t ricoh_last_stamp_ms() const { return media_ ? media_->ricoh_last_stamp_ms() : 0; }

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
    // Worker-thread-callable room_T_zed at `stamp`, using the CALLER's own InnerEigenAPI instance (for the
    // ts==0 static robot→zed extrinsic) and applying the SAME forward pose-extrapolation the voxel path
    // gets — so masks are placed at the capture-instant pose, not the ~100 ms-lagged newest RT block. No
    // CSV/logging (that stays on the main-thread get_room_robot_transform). std::nullopt on a missing hop.
    std::optional<Mat::RTMat> room_T_zed_extrapolated(DSR::InnerEigenAPI* eigen,
                                                      const std::string& room_name,
                                                      const std::string& robot_name,
                                                      std::uint64_t stamp) const;
    // DSR-native data accessors (no proxy needed)
    std::uint64_t get_frame_timestamp_ms() const;
    // Latest LiDAR scan from the media plane (LidarFrame). std::nullopt if disabled or nothing received.
    std::optional<LidarData> get_lidar3D();
    std::optional<RGBDData> get_rgbd_frame_from_dsr() const;

    void check_input_stream_startup_status();
    void mark_room_rt_ready();
    std::vector<GraphObjectBox> get_graph_object_boxes(const std::string& room_name,
                                                       std::uint64_t timestamp_ms) const;
    // Room floor polygon (room frame, metres) + ceiling height, gathered on the MAIN thread so the
    // ZED projection overlay never re-reads the graph itself (its per-frame inner_eigen tree-walks
    // raced DDS-thread node inserts during residual churn → heap corruption). false if no room yet.
    bool get_room_layout(std::vector<float>& polygon_x, std::vector<float>& polygon_y, float& room_height) const;

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

    std::shared_ptr<DSR::DSRGraph> graph_;
    DSR::InnerEigenAPI* inner_eigen_api_ = nullptr;
    bool transforms_interpolate_rt_ = true;
    bool verbose_debug_ = false;
    bool mask_pose_extrapolate_ = true;       // extrapolate robot pose to capture stamp via RT-edge velocity
    float mask_pose_extrap_max_dt_s_ = 0.2f;  // clamp the extrapolation horizon

    // Shared forward pose-extrapolation, used by BOTH the main-thread voxel path (get_room_robot_transform)
    // and the worker-thread masks path (room_T_zed_extrapolated). Reads the robot→room RT edge's body-frame
    // velocity from the graph (thread-safe) and rolls room_T_robot FORWARD to `timestamp_ms`. Mutates in
    // place; fills `diag`. No-op (diag.applied=false) if edge/velocity/attrs missing or ts<=newest block.
    struct PoseExtrapDiag {
        std::uint64_t newest_block_ms = 0;
        float  dt_s = 0.0f;
        double adv = 0.0, side = 0.0, rot = 0.0;
        double raw_x = 0.0, raw_y = 0.0, raw_th = 0.0;   // pose before extrapolation
        double dx = 0.0, dy = 0.0, dth = 0.0;            // applied displacement
        bool   applied = false;
    };
    void forward_extrapolate_room_T_robot(Mat::RTMat& room_T_robot, const std::string& room_name,
                                          const std::string& robot_name, std::uint64_t timestamp_ms,
                                          PoseExtrapDiag& diag) const;
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

    // Zero-copy DDS media ingest (ZED RGBD, shared LiDAR reader, Ricoh-360). Owns all subscriber/cache/
    // thread state; the public media methods above forward to it. Constructed in the ctor.
    std::unique_ptr<MediaPlaneSource> media_;
};