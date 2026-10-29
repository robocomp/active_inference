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
 *  visualizer does not subscribe to. An ingestor parameterised by node name serves both.
 *  ★ 2026-10-10: the two still CONSUME independently, but no longer SUBSCRIBE independently. Both
 *  attach to the same CameraStreamHub (one DDS reader per camera node per process), so each frame
 *  is transferred once and fanned out; each consumer keeps its own rate limit and its own copy.
 *
 *  THREADING
 *  ---------
 *    main thread   : construct; bind CameraAPI; calibrate the reduced CameraModel; read + cache the
 *                    STATIC camera<-robot extrinsic (the ts==0 InnerEigen path is main-thread-only
 *                    per instance, CLAUDE.md); start().
 *    hub thread    : (CameraStreamHub — ONE subscriber per camera node, shared with the viewer's
 *                    CameraVisualizer since 2026-10-10) delivers each loaned frame to on_frame();
 *                    RGB8/BGR8 -> GRAY8 here (so the boundary payload is
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
#include "camera_stream_hub.h"

namespace DSR { class InnerEigenAPI; class CameraAPI; }
namespace rc::media { class MediaSubscriber; }

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

    /// MAIN THREAD. Re-reads the graph's camera<-robot RT and, if it differs from what was last read, makes it
    /// the extrinsic.
    /// This is how a body->camera edge that robot_concept COMMITTED at runtime reaches the image-edge term:
    /// bind_camera() used to be the only read, so a committed mount was ignored until restart. The read goes
    /// through inner_ (created on the main thread in bind_camera, so its queued cache invalidations fire).
    /// Returns true when the base changed. No-op before a successful bind_camera().
    bool refresh_graph_extrinsic();


    /// MAIN THREAD. The extrinsic half of bind_camera() on its own (reads cam<-robot, applies it, sets
    /// extrinsic_ok_); bind_camera() calls it. Public so the graph->extrinsic path is testable without a
    /// camera stream (room_concept/tests/lidar_mount_edge_live_test.cpp).
    bool bind_extrinsic(const std::string& robot_frame);


    /// Convert at most one frame per `ms`. 0 = every delivered frame (the old behaviour).
    ///
    /// ★ THIS THROTTLES THE CONVERSION, NEVER THE DRAIN. The reader is RELIABLE: if its SHM pool is
    ///   allowed to back up the producer's loan_sample() eventually fails and it stops publishing,
    ///   freezing every other consumer on the plane (see ingest_loop). So the poll keeps running at
    ///   full rate and the loaned sample is still consumed — what is skipped is gray_from_rgb8 over
    ///   a frame nobody was going to read. On the ricoh that is ~5.5 MB per frame.
    /// ★ Measured 2026-08-29: the corner MATHS is 45 ms/s for the driving camera and 10 ms/s for the
    ///   calibration channel — 5.5% of a core between them. The cost is here, in pulling and
    ///   converting frames, spread over the per-camera ingest threads. Rate-limiting the extraction
    ///   would have saved 1%; rate-limiting this is where the CPU actually is.
    void set_min_convert_interval_ms(int ms) noexcept { min_convert_ms_ = std::max(0, ms); }

    /// Frames delivered vs frames actually converted since the last call, then reset. A throttle
    /// nobody can see is a throttle nobody revisits.
    [[nodiscard]] std::pair<long, long> convert_stats() noexcept
    { const auto r = std::pair{conv_seen_, conv_done_}; conv_seen_ = conv_done_ = 0; return r; }

    /// Attach to / detach from the node's shared CameraStreamHub (which owns the ONE subscriber and
    /// its drain thread). START ONLY once Operating (the hub thread touches the DSR graph for
    /// subscriber discovery). Idempotent. stop() returns only once no frame callback is in flight.
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
    /// Unit ray through a pixel, in CAMERA coordinates (x right, y forward, z up).
    ///
    /// ★ Goes through CameraAPI, which dispatches on the projection model, because the reduced
    ///   CameraModel cannot invert itself — it was calibrated to REPRODUCE project(), not to undo
    ///   it, and declines on a panorama for that reason. Pure arithmetic on stored intrinsics: no
    ///   graph read, so the ts==0 cache cliff does not apply and any thread may call it.
    [[nodiscard]] Eigen::Vector3f ray_from_pixel(double u, double v) const;

    [[nodiscard]] const Eigen::Matrix3f& cam_R_robot() const noexcept { return cam_R_robot_; }
    [[nodiscard]] const Eigen::Vector3f& cam_t_robot() const noexcept { return cam_t_robot_; }
    [[nodiscard]] const std::string& node_name() const noexcept { return camera_node_; }

    /// ms since a frame last reached the cache, or -1 if none ever has. For the stall log.
    [[nodiscard]] std::int64_t ms_since_last_frame() const noexcept;
    [[nodiscard]] std::uint64_t frames_ingested() const noexcept { return frames_.load(std::memory_order_relaxed); }

private:
    /// The hub callback (hub thread): throttle, convert to grey, absorb. The view dies on return.
    void on_frame(const CameraFrameView& f);
    /// The one place a decoded frame becomes the cached GrayFrame — both reader types land here, so
    /// the ZED and the Ricoh cannot drift apart in how their noise sigma is measured or stamped.
    void absorb_gray(std::vector<std::uint8_t> gray, int w, int h, std::int64_t stamp_ms);

    std::shared_ptr<DSR::DSRGraph> G_;
    std::string camera_node_;
    std::string robot_frame_;
    std::unique_ptr<DSR::InnerEigenAPI>      inner_;            // main-thread-owned (see refresh_graph_extrinsic)
    Eigen::Matrix4d                          graph_cam_T_robot_ = Eigen::Matrix4d::Zero();   // last graph read
    /// Graph extrinsic -> cam_R_robot_/cam_t_robot_, shared by bind_camera and refresh_graph_extrinsic.
    void apply_graph_extrinsic_(const Eigen::Matrix4d& M);

    std::unique_ptr<DSR::CameraAPI>          camera_api_;
    // The RGB stream comes from the process-wide hub for this node (camera_stream_hub.h): ONE DDS
    // reader per camera, shared with the viewer, chosen ("rgb" vs "rgb360") from the node's descriptor.
    std::shared_ptr<CameraStreamHub> hub_;
    int                              hub_id_ = -1;

    CameraModel     model_;
    std::unique_ptr<rc::media::MediaSubscriber>   sub_depth_;   ///< COMPUTE thread only
    bool            depth_absent_warned_ = false;
    long            depth_polls_ = 0, depth_hits_ = 0;
    int  min_convert_ms_ = 0;      ///< see set_min_convert_interval_ms
    std::int64_t last_convert_ms_ = 0;
    long conv_seen_ = 0, conv_done_ = 0;
    bool convert_due(std::int64_t stamp_ms) noexcept;
    Eigen::Matrix3f cam_R_robot_ = Eigen::Matrix3f::Identity();
    Eigen::Vector3f cam_t_robot_ = Eigen::Vector3f::Zero();
    bool            extrinsic_ok_ = false;

    // Newest frame, ingest thread -> compute thread.
    std::mutex  frame_mtx_;
    GrayFrame   frame_;
    bool        frame_fresh_ = false;

    std::atomic<std::int64_t>  last_frame_wall_ms_{0};
    std::atomic<std::uint64_t> frames_{0};
};

}  // namespace rc
