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

#include <Eigen/Dense>
#include <genericworker.h>          // DSR API

#include "image_edge_types.h"

namespace DSR { class InnerEigenAPI; class CameraAPI; }
namespace rc::media { class MediaSubscriber; }

namespace rc
{

class CameraIngestor
{
public:
    /// `camera_node` is the DSR node name ("zed" | "ricoh"); `robot_frame` MUST be the same frame the
    /// LiDAR points are expressed in (RoomConfig::LIDAR_ROBOT_FRAME), or the two observation terms
    /// would be constraining different poses.
    CameraIngestor(std::shared_ptr<DSR::DSRGraph> graph, std::string camera_node, std::string robot_frame);
    ~CameraIngestor();
    CameraIngestor(const CameraIngestor&) = delete;
    CameraIngestor& operator=(const CameraIngestor&) = delete;

    /// MAIN THREAD, once the graph is loaded and the agent is Operating. Binds CameraAPI, calibrates
    /// the reduced CameraModel against project(), and reads the static camera<-robot extrinsic.
    /// Returns false (and logs why) if any of those is not yet resolvable — the caller may retry.
    bool bind_camera();

    /// Start / stop the ingest thread. START ONLY once Operating (the thread touches the DSR graph for
    /// subscriber discovery, and doing that during the join window corrupts it). Idempotent.
    void start();
    void stop();

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

    std::shared_ptr<DSR::DSRGraph> G_;
    std::string camera_node_;
    std::string robot_frame_;

    std::unique_ptr<DSR::CameraAPI>          camera_api_;
    std::unique_ptr<rc::media::MediaSubscriber> sub_;

    CameraModel     model_;
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
    std::atomic<std::int64_t>  last_frame_wall_ms_{0};
    std::atomic<std::uint64_t> frames_{0};
};

}  // namespace rc
