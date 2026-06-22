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

// LidarIngestor — decoupled LiDAR acquisition for room_concept.
//
// The heavy point-cloud decode used to run inside a queued Qt slot on the GUI
// thread; under rendering load that thread starved and the localization inner
// loop stopped receiving fresh lidar. This class moves the decode + buffer fill
// + RoomConcept notify onto a dedicated thread, woken by a cheap CV trigger from
// the DDS/Qt signal path (with a periodic timeout + a GUI-thread watchdog poll).
// It owns the HighLidarBuffer the localizer reads from, plus all the lidar health
// telemetry. SpecificWorker forwards the DSR laser signals here and calls
// tick_health() once per compute cycle.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <genericworker.h>          // DSR API + att type tags

#include "buffer_types.h"           // rc::HighLidarBuffer, rc::LidarData
#include "room_concept.h"           // rc::RoomConcept (notify_new_lidar)

class QObject;

namespace rc::media { class LidarSubscriber; }

namespace rc
{

class LidarIngestor
{
public:
    struct Config
    {
        std::string lidar_name      = "lidar3D";
        float       high_min_height = 1.5f;   // m, floor/ceiling cutoff for the "high" cloud
        // Media plane (preferred source): subscribe to robot_concept's LidarFrame stream
        // instead of polling the DSR graph. The graph path stays as a watchdog fallback.
        bool          use_media        = true;
        std::uint32_t media_domain_id  = 7;
        std::string   media_lidar_topic = "rc/lidar3d/points";
    };

    struct Deps
    {
        std::shared_ptr<DSR::DSRGraph> graph;
        rc::RoomConcept*               room_concept = nullptr;
        QObject*                       qt_owner     = nullptr;   // QTimer::singleShot target (GUI thread)
    };

    LidarIngestor();    // out-of-line (unique_ptr<LidarSubscriber> incomplete in callers)
    ~LidarIngestor();
    LidarIngestor(const LidarIngestor&) = delete;
    LidarIngestor& operator=(const LidarIngestor&) = delete;

    void init(const Config& cfg, Deps deps);
    void start();   // launch the ingest thread
    void stop();    // signal + join (idempotent)

    // The localizer reads its lidar from here; also wired into RoomConcept::RunContext.
    [[nodiscard]] rc::HighLidarBuffer& buffer() noexcept { return high_lidar_buffer_; }

    // DSR signal entry points (called from SpecificWorker's Qt slots).
    void on_laser_node_signal();                  // update_node_signal, type "laser"
    void on_laser_attrs_signal(std::uint64_t id); // update_node_attr_signal, laser payload

    // Health log + (GUI-thread) watchdog poll; call once per compute tick.
    void tick_health(std::int64_t now_ms, bool on_gui_thread);

private:
    void lidar_ingest_loop();
    void ingest_latest_lidar();                                          // drain graph-fed pending scan
    void ingest_scan(std::vector<Eigen::Vector3f>&& points_high, std::int64_t src_ts);  // shared tail
    void schedule_lidar_copy_to_pending(std::optional<std::uint64_t> node_id, const char* origin, bool count_signal_event);
    bool copy_latest_lidar_to_pending(std::optional<std::uint64_t> node_id, const char* origin, bool count_signal_event);

    Config cfg_;
    std::shared_ptr<DSR::DSRGraph> G_;
    rc::RoomConcept* room_concept_ = nullptr;
    QObject*         qt_owner_     = nullptr;
    std::unique_ptr<rc::media::LidarSubscriber> lidar_sub_;   // primary source when cfg_.use_media

    rc::HighLidarBuffer high_lidar_buffer_{3};

    // ── Health telemetry ───────────────────────────────────────────────────
    std::atomic<std::uint64_t> lidar_signal_events_{0};
    std::atomic<std::uint64_t> lidar_copied_frames_{0};
    std::atomic<std::uint64_t> lidar_ingested_frames_{0};
    std::atomic<std::uint64_t> lidar_watchdog_pulls_{0};
    std::atomic<std::uint64_t> lidar_ingest_wakeups_{0};
    std::atomic<std::uint64_t> lidar_copy_mutex_busy_{0};
    std::atomic<std::uint64_t> lidar_ingest_loop_beats_{0};
    std::atomic<bool>          lidar_copy_queued_{false};
    std::atomic<std::int64_t>  lidar_last_signal_ms_{0};
    std::atomic<std::int64_t>  lidar_last_ingest_ms_{0};
    std::atomic<std::int64_t>  lidar_last_ingest_loop_beat_ms_{0};
    std::atomic<bool>          lidar_notify_inflight_{false};
    std::atomic<std::int64_t>  lidar_notify_inflight_since_ms_{0};
    std::int64_t               last_lidar_health_log_ms_   = 0;
    std::int64_t               last_lidar_watchdog_pull_ms_ = 0;
    std::deque<std::int64_t>   lidar_recent_source_ts_;

    // ── Ingest thread ──────────────────────────────────────────────────────
    std::thread             lidar_ingest_thread_;
    std::atomic<bool>       lidar_ingest_running_{false};
    std::atomic<bool>       lidar_ingest_dirty_{false};
    std::mutex              lidar_ingest_mutex_;
    std::condition_variable lidar_ingest_cv_;
    std::int64_t            last_ingested_lidar_ts_ = std::numeric_limits<std::int64_t>::min();
    struct PendingLidarScan
    {
        std::vector<float> xs, ys, zs;
        std::int64_t source_ts = std::numeric_limits<std::int64_t>::min();
        bool has_data = false;
    };
    PendingLidarScan pending_lidar_scan_;
};

}  // namespace rc
