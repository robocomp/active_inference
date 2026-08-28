/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

/*
 *  camera_ingestor.h — media-plane RGB acquisition for the localizer's edge-alignment term.
 *
 *  WHY THIS IS NOT AN ACCESSOR ON CameraVisualizer
 *  -----------------------------------------------
 *  CameraVisualizer already drains the same stream, but it is a QDialog owned by RoomViewer. Taking
 *  the measurement path through it would (a) invert the dependency — inference on a widget — and tie
 *  the term's availability to whether a viewer exists, when room_concept must run headless; (b) tie
 *  it to the ZED node specifically, while the Ricoh path needs a SECOND stream on a SECOND node the
 *  visualizer does not subscribe to. An ingestor parameterised by node name serves both. Two
 *  consumers on media-plane domain 7 is exactly what the plane is for — LidarIngestor and
 *  CameraVisualizer already coexist.
 *
 *  THREADING
 *  ---------
 *    main thread   : construct; bind CameraAPI; calibrate the reduced CameraModel; read + cache the
 *                    STATIC camera<-robot extrinsic (the ts==0 InnerEigen path is main-thread-only
 *                    per instance, CLAUDE.md); start().
 *    ingest thread : owns the subscriber; poll(); RGB8/BGR8 -> GRAY8 here (so the boundary payload is
 *                    a third the size and the localizer never sees three channels); measure sigma_i;
 *                    publish under the cache lock.
 *    compute thread: take_latest() moves a std::vector<uint8_t> out from under the lock.
 *
 *  The crossing type is std::vector<uint8_t>, which has VALUE semantics — so the cv::Mat refcounted
 *  shallow-handle hazard that CLAUDE.md names as the cause of the 2026-07 cores cannot arise here.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <genericworker.h>          // DSR API

#include "image_edge_types.h"

namespace DSR { class InnerEigenAPI; class CameraAPI; }
namespace rc::media { class MediaSubscriber; class Image360Subscriber; }

namespace rc
{

class CameraIngestor
{
public:
    /// `camera_node` is the DSR node name ("zed" | "ricoh").
    CameraIngestor(std::shared_ptr<DSR::DSRGraph> graph, std::string camera_node);
    ~CameraIngestor();
    CameraIngestor(const CameraIngestor&) = delete;
    CameraIngestor& operator=(const CameraIngestor&) = delete;

    /// MAIN THREAD, once the graph is loaded and the agent is Operating. Binds CameraAPI, calibrates
    /// the reduced CameraModel against project(), and reads the static camera<-robot extrinsic.
    /// Returns false (and logs why) if any of those is not yet resolvable — the caller retries.
    ///
    /// ★ `robot_frame` is passed HERE and not to the constructor: RoomConfig::LIDAR_ROBOT_FRAME is
    ///   auto-derived from the type-"robot" node by check_init_graph_is_valid(), which runs AFTER
    ///   the collaborators are constructed. Capturing it at construction time captures the empty
    ///   string, and every bind then fails with an unresolvable RT chain — forever, silently apart
    ///   from a warning. It MUST be the same frame the LiDAR points are in, or the SDF term and this
    ///   one would be constraining different poses.
    bool bind_camera(const std::string& robot_frame);

    /// LOCAL boresight-yaw correction, radians, applied to the extrinsic read from the graph.
    ///
    /// ★ WHY THIS IS HERE AND NOT IN THE ROBOT'S GEOMETRY. The graph's camera<-robot RT is the
    ///   shared, authoritative description every agent reads. This knob lets room_concept test a
    ///   measured mount error WITHOUT editing that shared description, so a wrong sign or a wrong
    ///   magnitude cannot silently propagate to every other consumer of the zed extrinsic. Once the
    ///   value is confirmed here it belongs in the robot geometry, and this should go back to 0.
    ///
    /// MEASURED 2026-08-28: +0.814 +/- 0.022 deg (0.01421 rad, 6.37 px at fy=448), by joining the
    /// image-edge rigid-shift fit against the Webots supervisor across 99 windows and 7 heading
    /// bins. Flat across heading (spread 0.058 deg), which is what makes it a BORESIGHT and not the
    /// localiser's own heading error (that ran ~0.18 deg sd and correlated at +0.816 with the shift).
    void set_mount_yaw_correction(float rad) noexcept { mount_yaw_correction_ = rad; }

    /// Start / stop the ingest thread. START ONLY once Operating (the thread touches the DSR graph for
    /// subscriber discovery, and doing that during the join window corrupts it). Idempotent.
    void start();
    void stop();

    /// Sample the newest DEPTH frame at the given pixels. Fills `depth_m` (< 0 where unavailable)
    /// and `stamp_ms` with that frame's capture time. Returns how many pixels yielded a value.
    ///
    /// ★ COMPUTE THREAD ONLY, and the depth subscriber is created lazily HERE for that reason. The
    ///   RGB subscriber is made and polled on the ingest thread; keeping each subscriber confined to
    ///   one thread is what makes two readers on the same node safe without a lock. Lazy creation
    ///   from an already-Operating worker is the pattern CLAUDE.md sanctions for this component.
    ///
    /// ★ TRUE ZERO-COPY: the loaned sample is a view into the SHM segment and is valid ONLY inside
    ///   the poll callback (media_transport.h:315), so the pixels are read there and nothing else is
    ///   copied. A 1280x720 depth frame is 1.8-3.7 MB; this touches a few dozen bytes of it.
    ///   The corollary is that the pixel list must be known BEFORE polling — which it is, because
    ///   triple points are detected from the RGB frame first.
    int probe_depth(const std::vector<Eigen::Vector2f>& uv, int patch_radius,
                    std::vector<float>& depth_m, std::int64_t& stamp_ms);

    /// polls attempted / polls that returned a depth frame. A subscriber that exists and never
    /// delivers looks exactly like one that was never created; these two numbers separate them.
    [[nodiscard]] std::pair<long, long> depth_stats() const noexcept
    { return {depth_polls_, depth_hits_}; }

    /// Move the newest frame out, if one has arrived since the last call. Returns false if not.
    /// Compute thread. Cheap: a vector move under the lock, no copy, no decode.
    bool take_latest(GrayFrame& out);

    [[nodiscard]] bool ready() const noexcept { return model_.valid and extrinsic_ok_; }
    [[nodiscard]] const CameraModel& model() const noexcept { return model_; }
    /// camera <- robot, the ONLY transform the measurement path takes from the graph. room <- robot
    /// is the STATE VARIABLE and must never be read here (see image_edge_source.h).
    [[nodiscard]] const Eigen::Matrix3f& cam_R_robot() const noexcept { return cam_R_robot_; }
    [[nodiscard]] const Eigen::Vector3f& cam_t_robot() const noexcept { return cam_t_robot_; }
    [[nodiscard]] const std::string& node_name() const noexcept { return camera_node_; }

    /// ms since a frame last reached the cache, or -1 if none ever has. For the stall log.
    [[nodiscard]] std::int64_t ms_since_last_frame() const noexcept;
    [[nodiscard]] std::uint64_t frames_ingested() const noexcept { return frames_.load(std::memory_order_relaxed); }

private:
    void ingest_loop();
    bool ingest_pump();
    bool try_discover();
    /// The one place a decoded frame becomes the cached GrayFrame — both reader types land here, so
    /// the ZED and the Ricoh cannot drift apart in how their noise sigma is measured or stamped.
    void absorb_gray(std::vector<std::uint8_t> gray, int w, int h, std::int64_t stamp_ms);

    std::shared_ptr<DSR::DSRGraph> G_;
    std::string camera_node_;
    std::string robot_frame_;

    std::unique_ptr<DSR::CameraAPI>          camera_api_;
    // ★ TWO SUBSCRIBER TYPES, and exactly one of them is ever non-null. This is not a convenience:
    //   the ZED and the Ricoh are DIFFERENT DDS TYPES on differently-named streams — "rgb" carrying
    //   ImageFrame vs "rgb360" carrying Image360Frame (~5.5 MB inline, its own topic type). A single
    //   MediaSubscriber on "rgb" finds no descriptor on the `ricoh` node at all, so it returns
    //   nullptr and try_discover() retries once a second FOREVER: the agent looks healthy, the
    //   [imgedge] liveness line never prints, and nothing says why. Which one to build is decided by
    //   READING THE NODE'S DESCRIPTOR (CLAUDE.md: domain + topic + streams come only from there),
    //   never by matching the node name against a hard-coded list.
    std::unique_ptr<rc::media::MediaSubscriber>    sub_;
    std::unique_ptr<rc::media::Image360Subscriber> sub360_;

    CameraModel     model_;
    std::unique_ptr<rc::media::MediaSubscriber>   sub_depth_;   ///< COMPUTE thread only
    bool            depth_absent_warned_ = false;
    long            depth_polls_ = 0, depth_hits_ = 0;
    float           mount_yaw_correction_ = 0.f;   ///< rad, applied about the ROBOT VERTICAL at bind
    Eigen::Matrix3f cam_R_robot_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f cam_t_robot_ = Eigen::Vector3f::Zero();
    bool            extrinsic_ok_ = false;

    // Newest frame, ingest thread -> compute thread.
    std::mutex  frame_mtx_;
    GrayFrame   frame_;
    bool        frame_fresh_ = false;
    std::vector<std::uint8_t> scratch_;   // ingest-thread only: the colour staging buffer

    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::mutex              wake_mtx_;
    std::condition_variable wake_cv_;
    std::chrono::steady_clock::time_point last_discovery_{};
    bool no_stream_warned_ = false;   // the "no readable stream" line is permanent news, said once
    std::atomic<std::int64_t>  last_frame_wall_ms_{0};
    std::atomic<std::uint64_t> frames_{0};
};

}  // namespace rc
