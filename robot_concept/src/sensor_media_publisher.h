/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// SensorMediaPublisher — orchestration layer for the zero-copy media plane.
//
// It owns the per-stream publishers (from common/media_transport) for every raw
// sensor stream the robot exports out of the DSR/CRDT graph: RGB + depth images,
// the LiDAR scan and the IMU sample. It exposes a one-call publish API per
// payload, advertises a single self-describing MediaDescriptor (with per-stream
// type names) on a graph node, and reports unified per-stream stats. Each frame
// carries its own epoch-ms timestamp so consumers can realign streams upstream.
//
// Adding another IMAGE-like stream (a 2nd camera, a mask) is a one-line StreamSpec
// in Config::image_streams. A new payload family (e.g. wheel odometry) needs a new
// bounded IDL type + a publish_<x>()/view, mirroring the lidar/imu wiring here.
//
// Not thread-safe per stream: each publish_*() for a given stream must come from a
// single producer thread (RGBD thread for images, lidar thread, imu thread).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "../../common/media_transport/media_transport.h"

class SensorMediaPublisher
{
public:
    // One advertised stream: a key ("rgb","depth","lidar","imu"), its DDS topic
    // and the stream_id tag (rc::media::STREAM_*).
    struct StreamSpec
    {
        std::string   key;
        std::string   topic;
        std::uint32_t stream_id = 0;
    };

    struct Config
    {
        std::uint32_t            domain_id     = 0;
        int                      history_depth = 8;
        // Zero-copy SHM data-sharing QoS. Default OFF (churn-safe): opt in only on a
        // static, non-churning topology — see rc::media::PublisherConfig::data_sharing.
        // Advertised in the descriptor so consumers adopt the same setting.
        bool                     data_sharing  = false;
        std::vector<StreamSpec>  image_streams;   // rgb, depth, …
        std::optional<StreamSpec> image360_stream;// nullopt ⇒ ricoh panorama not published (wide Image360Frame)
        std::optional<StreamSpec> lidar_stream;   // nullopt ⇒ lidar not published
        std::optional<StreamSpec> imu_stream;     // nullopt ⇒ imu not published
    };

    // ── per-payload views (carry no DDS types to the call site) ──────────────
    struct ImageFrameView
    {
        std::uint64_t       stamp_ms = 0;
        std::uint32_t       width  = 0, height = 0, step = 0, format = 0;
        const std::uint8_t *data   = nullptr;
        std::size_t         nbytes = 0;
    };

    struct LidarFrameView
    {
        std::uint64_t stamp_ms = 0;
        const float  *points   = nullptr;   // interleaved x,y,z (metres)
        std::uint32_t count    = 0;         // number of points
        std::uint32_t stride   = 3;         // floats per point
        std::uint32_t format   = 0;         // rc::media::LIDAR_FORMAT_*
    };

    struct ImuFrameView
    {
        std::uint64_t stamp_ms = 0;
        float acc[3]  = {0, 0, 0};
        float gyro[3] = {0, 0, 0};
        float mag[3]  = {0, 0, 0};
        float rpy[3]  = {0, 0, 0};          // roll, pitch, yaw
        float temperature = 0.f;
    };

    SensorMediaPublisher() = default;
    ~SensorMediaPublisher();
    SensorMediaPublisher(const SensorMediaPublisher&) = delete;
    SensorMediaPublisher& operator=(const SensorMediaPublisher&) = delete;

    // Create a publisher per configured stream. Returns true iff every one came
    // up. Logs a [Media] summary line.
    bool init(const Config& cfg);

    [[nodiscard]] bool ready() const noexcept { return ready_; }          // images up
    [[nodiscard]] bool image360_ready() const noexcept { return image360_.has_value(); }
    [[nodiscard]] bool lidar_ready() const noexcept { return lidar_.has_value(); }
    [[nodiscard]] bool imu_ready() const noexcept { return imu_.has_value(); }
    [[nodiscard]] bool data_sharing_active() const noexcept;

    // Advertise the plane on a DSR node (templated to keep this header DSR-free).
    // `keys` selects which streams to write: empty ⇒ all (single-node bundle), or
    // a subset so each sensor node carries only its own stream(s) — e.g.
    // {"rgb","depth"} on "zed", {"lidar"} on "lidar3D", {"imu"} on "imu".
    // Returns false if the node is absent (nothing written).
    template <class Graph>
    bool advertise(Graph& graph, const std::string& node_name,
                   const std::vector<std::string>& keys = {},
                   const char* attr_name = rc::media::MEDIA_DESCRIPTOR_ATTR) const;

    // Write the live media throughput (bytes/s, summed over the node's streams) onto the same sensor
    // node as its descriptor, as the `media_bps` attribute. Call ~1 Hz from the graph/main thread.
    // Returns false if the node is absent. Non-const: advances the per-stream sampling window.
    template <class Graph>
    bool advertise_stats(Graph& graph, const std::string& node_name,
                         const std::vector<std::string>& keys = {});

    [[nodiscard]] std::string descriptor_json(const std::vector<std::string>& keys = {}) const;

    // Sum of live bytes/s over the streams selected by `keys` (empty ⇒ all), from the LOCAL publish
    // path (nonzero only while robot_concept produces/bridges the stream). Advances each selected
    // stream's sampling window, so call once per node per period.
    [[nodiscard]] double current_bps(const std::vector<std::string>& keys);

    // Loan → fill → publish for one stream. Returns false (frame dropped) on: not
    // ready, key/stream unknown, empty/oversize payload, loan unavailable, publish
    // failure — folded into the per-stream stats.
    bool publish_image(const std::string& key, const ImageFrameView& view);
    // Publish the wide 360 panorama on the Image360Frame plane (buffer bound
    // MAX_IMAGE360_BYTES, not MAX_IMAGE_BYTES). No-op if no image360 stream configured.
    bool publish_image360(const ImageFrameView& view);
    bool publish_lidar(const LidarFrameView& view);
    bool publish_imu(const ImuFrameView& view);

    // Which producer thread's streams to report. Each group is touched by exactly
    // one thread (images=RGBD, lidar=lidar, imu=imu), so per-group reporting stays
    // lock-free and race-free — never report a group from another thread.
    enum class StatsGroup { Image, Image360, Lidar, Imu };

    // Emit a [Media] stats line for one group's streams at most once per `interval`;
    // self-throttles. Call from that group's own producer thread.
    void maybe_report_stats(StatsGroup group, std::chrono::seconds interval);

    void shutdown();

private:
    struct Stats
    {
        std::uint64_t ok = 0;
        std::uint64_t bytes = 0;
        std::uint64_t loan_fail = 0;
        std::uint64_t publish_fail = 0;
        void reset() { ok = bytes = loan_fail = publish_fail = 0; }
    };

    template <class PubT>
    struct Stream
    {
        std::string   key;
        std::string   topic;
        std::uint32_t stream_id = 0;
        PubT          pub;
        std::uint64_t frame_id = 0;
        Stats         stats;
        // Monotonic byte counter for live throughput (never reset, unlike Stats::bytes). Written from
        // the producer thread, sampled from the graph thread in advertise_stats() → atomic.
        std::atomic<std::uint64_t> cum_bytes{0};
        std::uint64_t             bps_prev_bytes = 0;                 // graph-thread-only bps state
        std::chrono::steady_clock::time_point bps_prev_t{};
    };

    [[nodiscard]] rc::media::MediaDescriptor make_descriptor(
        const std::vector<std::string>& keys = {}) const;
    void report_one(const std::string& key, Stats& s);

    std::map<std::string, Stream<rc::media::MediaPublisher>> images_;
    std::optional<Stream<rc::media::Image360Publisher>>      image360_;
    std::optional<Stream<rc::media::LidarPublisher>>         lidar_;
    std::optional<Stream<rc::media::ImuPublisher>>           imu_;

    std::uint32_t domain_id_          = 0;
    int           history_depth_      = 8;
    bool          shared_memory_only_ = true;   // mirrors PublisherConfig defaults
    bool          data_sharing_       = false;
    bool          ready_              = false;   // image streams up
    // One report clock per group; each touched only by its own producer thread.
    std::chrono::steady_clock::time_point image_report_at_{};
    std::chrono::steady_clock::time_point image360_report_at_{};
    std::chrono::steady_clock::time_point lidar_report_at_{};
    std::chrono::steady_clock::time_point imu_report_at_{};
    // Transport-status clock. Unlike the per-group clocks above, maybe_report_stats()
    // runs on ALL FOUR sensor threads, so this one is atomic: a CAS lets exactly one
    // thread emit the periodic data_sharing line per interval, whichever sensors are on.
    std::atomic<std::int64_t> status_report_next_ns_{0};
};

// ── advertise() is templated on the DSR graph, so it stays in the header ──────
template <class Graph>
bool SensorMediaPublisher::advertise(Graph& graph, const std::string& node_name,
                                     const std::vector<std::string>& keys,
                                     const char* attr_name) const
{
    auto node = graph.get_node(node_name);
    if (not node.has_value())
        return false;
    graph.runtime_checked_add_or_modify_attrib_local(node.value(), attr_name,
                                                      make_descriptor(keys).to_json());
    graph.update_node(node.value());
    return true;
}

template <class Graph>
bool SensorMediaPublisher::advertise_stats(Graph& graph, const std::string& node_name,
                                           const std::vector<std::string>& keys)
{
    auto node = graph.get_node(node_name);
    if (not node.has_value())
        return false;
    // NOTE: runtime_checked (not the typed <media_bps_att> form) on purpose — this header stays
    // DSR-att-header-free (it receives DSRGraph& from its includer and includes no dsr/ headers),
    // so the typed alias isn't in scope here. See CONCEPT_AGENT_RECIPE.md §"Attribute access".
    graph.runtime_checked_add_or_modify_attrib_local(node.value(), "media_bps",
                                                      static_cast<float>(current_bps(keys)));
    graph.update_node(node.value());
    return true;
}
