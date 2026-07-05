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

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <genericworker.h>          // DSR API

#include "buffer_types.h"           // rc::HighLidarBuffer, rc::LidarData (+ Eigen)
#include "room_concept.h"           // rc::RoomConcept (notify_new_lidar)
#include "room_config.h"            // rc::RoomConfig (shared config)

namespace DSR { class InnerEigenAPI; }
namespace rc::media { class LidarSubscriber; }

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

    // Drain the media plane (non-blocking), push the newest scan to the buffer + wake the localizer.
    // Call once per compute cycle from the compute thread. Returns true if a fresh scan was ingested.
    bool pump();

    // The localizer reads its lidar from here (also wired into RoomConcept::RunContext).
    [[nodiscard]] rc::HighLidarBuffer& buffer() noexcept { return high_lidar_buffer_; }

private:
    void ingest_scan(std::vector<Eigen::Vector3f>&& points_high, std::int64_t src_ts);
    // Lazily create the subscriber from a media descriptor (self-throttled). Prefers the
    // per-device "helios" plane (DEVICE frame, metres → transformed device->robot here) and
    // falls back to the legacy fused "lidar3D" plane (already robot-frame). Returns true once
    // it is up. Until a node + descriptor exist, returns false.
    bool ensure_subscriber();

    std::shared_ptr<DSR::DSRGraph> G_;
    rc::RoomConcept*      room_concept_ = nullptr;
    const rc::RoomConfig* params_       = nullptr;
    std::unique_ptr<rc::media::LidarSubscriber> lidar_sub_;
    std::unique_ptr<DSR::InnerEigenAPI>         inner_eigen_;   // device->robot transform (helios plane)
    std::string source_node_;                                   // node we subscribed to ("helios"/"lidar3D")
    bool        needs_transform_ = false;                       // true ⇔ points arrive in the device frame
    std::int64_t last_transform_warn_ms_ = 0;                   // throttle "no RT" warnings
    std::chrono::steady_clock::time_point last_init_attempt_{};

    rc::HighLidarBuffer high_lidar_buffer_{3};
    std::int64_t last_ingested_lidar_ts_ = std::numeric_limits<std::int64_t>::min();

    // Source-attribution telemetry (compute thread only): 5 s "[LidarSrc]" log.
    std::uint64_t fresh_frames_      = 0;   // LidarFrames drained from the plane
    std::uint64_t served_            = 0;   // scans actually pushed to the buffer
    std::int64_t  last_src_report_ms_ = 0;
};

}  // namespace rc
