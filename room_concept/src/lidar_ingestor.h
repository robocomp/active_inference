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

// LidarIngestor — media-plane LiDAR acquisition for room_concept.
//
// Drains robot_concept's zero-copy LidarFrame stream synchronously via pump(), called once per
// compute cycle from the compute thread: it fills the HighLidarBuffer the localizer reads and wakes
// the localizer (RoomConcept::notify_new_lidar). No dedicated ingest thread and no DSR-graph path —
// a single-caller non-blocking poll() is the one thread-safe procedure every agent uses to read the
// media infrastructure (mirrors the voxelizer's SceneProcessor::get_lidar3D()). The earlier
// CV/signal/watchdog ingest-thread scaffolding was crash-hunting for what turned out to be the Eigen
// alignment ABI bug (now fixed), so it is gone.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <genericworker.h>          // DSR API

#include "buffer_types.h"           // rc::HighLidarBuffer, rc::LidarData (+ Eigen)
#include "room_concept.h"           // rc::RoomConcept (notify_new_lidar)
#include "room_config.h"            // rc::RoomConfig (shared config)

namespace DSR { class InnerEigenAPI; }
namespace rc::media { class LidarPlaneReader; }

namespace rc
{

class LidarIngestor
{
public:
    // Lightweight: stores deps + creates the room-side buffer. The DDS subscriber is
    // NOT created here — it is brought up lazily in pump() once the "lidar3D" node +
    // its media descriptor exist, reading the DDS domain/topic from that JSON (no config).
    LidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, rc::RoomConcept& room_concept,
                  const rc::RoomConfig& params);
    ~LidarIngestor();
    LidarIngestor(const LidarIngestor&) = delete;
    LidarIngestor& operator=(const LidarIngestor&) = delete;

    // Start/stop the dedicated ingest thread. START ONLY once the agent is Operating (post graph-join):
    // the thread reads the DSR graph (subscriber discovery + inner_eigen device→robot transform), and
    // touching the graph during the join window corrupts it — same rule as compute(). Idempotent.
    // stop() joins the thread; it is also called by the destructor, which the owner must invoke while G
    // and RoomConcept are still alive (see SpecificWorker teardown).
    void start();
    void stop();

    // Drain the media plane (non-blocking), push the newest scan to the buffer + wake the localizer.
    // Returns true if a fresh scan was ingested. Runs on the ingest thread; exposed for tests.
    bool pump();

    // The localizer reads its lidar from here (also wired into RoomConcept::RunContext).
    [[nodiscard]] rc::HighLidarBuffer& buffer() noexcept { return high_lidar_buffer_; }

private:
    // Ingest-thread body: tightly paced poll of the reader so a fresh scan reaches the localizer with
    // ~0-2 ms latency instead of waiting for the next ~16 ms compute() tick. Sleeps briefly (woken on
    // stop) when no new frame is available, so it is not a hard spin.
    void ingest_loop();
    void ingest_scan(std::vector<Eigen::Vector3f>&& points_high, std::int64_t src_ts);

    // One-shot startup geometry self-check (ingest thread only). Accumulates a z-histogram of the
    // first LIDAR_STARTUP_CHECK_SWEEPS body-frame sweeps, then: (a) locates the floor plane and warns
    // if it disagrees with the mount geometry (RT root<-body vs body<-helios) -> a wrong mount height;
    // (b) locates the ceiling plane and caps high_max_z_ at ceiling - margin so only upper-wall points
    // reach the localizer. Re-armed by start() so it re-runs on each Operating entry.
    void accumulate_geometry_sample(const std::vector<Eigen::Vector3f>& sweep_body);
    void run_startup_geometry_check();

    std::shared_ptr<DSR::DSRGraph> G_;
    rc::RoomConcept*      room_concept_ = nullptr;
    const rc::RoomConfig* params_       = nullptr;
    // Shared media-plane consumer: prefers the "helios" high plane (DEVICE frame), transformed to the
    // robot base ("body") via the DSR RT tree; falls back to the fused "lidar3D" plane. Same reader
    // every agent uses. inner_eigen_ backs its RT queries and must outlive it.
    std::unique_ptr<DSR::InnerEigenAPI>          inner_eigen_;
    std::unique_ptr<rc::media::LidarPlaneReader> reader_;

    rc::HighLidarBuffer high_lidar_buffer_{3};
    std::int64_t last_ingested_lidar_ts_ = std::numeric_limits<std::int64_t>::min();

    // Runtime upper bound of the high band (m, body frame). Starts at LIDAR_HIGH_MAX_HEIGHT and is
    // lowered to (detected ceiling - margin) by the startup check. Ingest-thread only, so plain float.
    float high_max_z_ = 0.f;
    // Startup geometry-check state (ingest thread only; re-armed by start()).
    bool  geom_check_done_ = false;
    int   geom_sweeps_     = 0;
    std::vector<int> geom_hist_;   // helios z-histogram (walls/ceiling; floor is grazing/high)
    // Joint (z-bin × horizontal-radius-bin) helios histogram, for the ceiling perimeter-vs-interior test:
    // a real ceiling fills the interior (returns CLOSER than the walls), a wall-top ring sits at the wall
    // range. Row-major [z_bin * GEOM_NR + r_bin]. Dropped with geom_hist_ after the one-shot check.
    std::vector<int> geom_rz_hist_;
    // bpearl (downward dome) is the HEAD-ON floor sensor — its own z-histogram is the floor calibration
    // datum. Comes up later than helios → warm-up counter; separate reader dropped after the check.
    int   geom_bpearl_sweeps_ = 0;
    std::vector<int> geom_hist_bpearl_;
    std::unique_ptr<rc::media::LidarPlaneReader> geom_bpearl_reader_;

    // Dedicated ingest thread (started at Operating-enter, joined in stop()/dtor).
    std::thread             thread_;
    std::atomic<bool>       running_{false};
    std::mutex              wake_mutex_;
    std::condition_variable wake_cv_;

    // Source-attribution telemetry (ingest thread only): 5 s "[LidarSrc]" log.
    std::uint64_t fresh_frames_      = 0;   // LidarFrames drained from the plane
    std::uint64_t served_            = 0;   // scans actually pushed to the buffer
    std::int64_t  last_src_report_ms_ = 0;
};

}  // namespace rc
