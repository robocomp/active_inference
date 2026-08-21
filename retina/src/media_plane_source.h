#pragma once

// Zero-copy DDS media-plane ingest for the retina: ZED RGB+depth, the shared multi-plane LiDAR
// reader, and the Ricoh-360 panorama. Split out of SceneProcessor so all DDS subscriber/cache/thread
// state lives in one place; SceneProcessor holds one of these and forwards to it. Subscribers are
// created from the producer's per-node media descriptors in the graph (never from config).

#include "rgbd_data.h"
#include "zed_frame_aligner.h"   // ZedFrameAligner (PIMPL over BufferSync — no doublebuffer/threadpool leak)

#include <opencv2/core.hpp>

#include <atomic>
#include <thread>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace DSR { class DSRGraph; class InnerEigenAPI; }
namespace rc::media { class MediaSubscriber; }
namespace rc::media { class Image360Subscriber; }
namespace rc::media { class LidarPlaneReader; }

// Merged LiDAR sweep in the ROBOT frame (callers apply the dynamic room<-robot pose at the scan stamp).
struct LidarData
{
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
    std::vector<std::uint8_t> plane_id;   // per-point source plane (helios=0, bpearl=1) for viewer colouring
    std::uint64_t timestamp_ms = 0;
};

class MediaPlaneSource
{
public:
    explicit MediaPlaneSource(std::shared_ptr<DSR::DSRGraph> graph);
    ~MediaPlaneSource();   // out-of-line: members hold unique_ptrs to incomplete DDS types

    MediaPlaneSource(const MediaPlaneSource&) = delete;
    MediaPlaneSource& operator=(const MediaPlaneSource&) = delete;

    // --- ZED RGBD plane ---
    bool init_media_plane(std::uint32_t domain_id, const std::string& rgb_topic, const std::string& depth_topic);
    std::uint64_t get_frame_timestamp_ms() const;
    std::optional<RGBDData> get_rgbd_frame_from_dsr() const;
    bool rgb_valid() const { return latest_rgb_stamp_.load(std::memory_order_relaxed) != 0; }

    // ★CHEAP PRE-CHECK for the pull worker: the newest rgb stamp the INGEST thread has seen, readable
    // without assembling (and deep-copying) anything. get_frame_timestamp_ms() cannot serve this: it
    // returns last_frame_ts_, which is SET BY the assemble call, so asking it first would always report
    // the previous frame. One atomic load.
    [[nodiscard]] std::uint64_t pending_rgb_stamp() const
    { return latest_rgb_stamp_.load(std::memory_order_relaxed); }
    bool depth_valid() const { return latest_depth_stamp_.load(std::memory_order_relaxed) != 0; }

    // --- LiDAR plane (shared reader) --- inner_eigen backs the device->robot RT transform.
    bool init_lidar_media_plane(DSR::InnerEigenAPI* inner_eigen, std::uint32_t domain_id,
                                const std::string& topic, bool use_media);
    std::optional<LidarData> get_lidar3D(const std::string& robot_name);

    // --- Ricoh-360 panorama --- may be polled from the ricoh worker thread AND the main thread.
    bool init_ricoh_media_plane(std::uint32_t domain_id, const std::string& topic);
    bool ricoh_available() const { return media_ricoh_sub_ != nullptr; }
    void poll_ricoh(bool force = false);
    void set_ricoh_wanted(bool on) { ricoh_wanted_.store(on, std::memory_order_relaxed); }
    cv::Mat ricoh_bgr_copy() const;
    std::uint64_t ricoh_last_stamp_ms() const { return ricoh_last_stamp_ms_.load(std::memory_order_relaxed); }

private:
    void drain_media_plane() const;   // drains RGB+depth subscribers into the aligner

    // ── EVENT-DRIVEN INGEST ───────────────────────────────────────────────────────────────────────
    // Start/stop the thread that BLOCKS on the media plane and drains it the moment data lands.
    // ★WHY A THREAD AND NOT THE CONSUMER'S LOOP. The drain used to run inline inside
    // get_rgbd_frame_from_dsr(), so the rate at which frames were COLLECTED was the rate at which the
    // perception worker asked for them. That coupling is what made poll granularity cost real frames:
    // at a 15 ms ask against a 25 ms camera it lost roughly half of them, and at 4 ms about 9%. A
    // frame that arrives is now always taken, whatever the consumer is doing.
    // It blocks in wait_and_poll rather than using a DDS listener ON PURPOSE: the callback then runs on
    // THIS thread, not on a FastDDS reader thread, which is the hazard CLAUDE.md opens with.
    void start_ingest();
    void stop_ingest();

    std::shared_ptr<DSR::DSRGraph> graph_;

    std::unique_ptr<rc::media::MediaSubscriber> media_rgb_sub_;
    std::unique_ptr<rc::media::MediaSubscriber> media_depth_sub_;

    std::unique_ptr<rc::media::LidarPlaneReader> lidar_reader_;
    bool      lidar_use_media_ = false;
    LidarData media_lidar_;
    bool      media_lidar_valid_ = false;

    // ZED rgb+depth: timestamp-aligned (rgb frame F always meets depth frame F, no "latest of each" skew).
    ZedFrameAligner zed_buf_{8};
    // ── EXACT ARRIVAL COUNTERS ─────────────────────────────────────────────────────────────────────
    // Monotonic count of samples the SUBSCRIBER received, including the ones perception skips. This is
    // the source rate measured rather than inferred: the previous estimate read the stamps of frames we
    // chose to PROCESS and tried to recover the producer's cadence from their gaps, which can only ever
    // be a bound and was reading ABOVE the source. A delivery count divided by wall time cannot.
    // Written on the drain thread, read on the GUI thread ⇒ atomic, relaxed (a counter, not a fence).
    mutable std::atomic<std::uint64_t> rx_rgb_total_{0};
    mutable std::atomic<std::uint64_t> rx_ricoh_total_{0};
    // LiDAR counts COMPLETE media-plane updates — the INTERSECTION of the contributing planes, not the
    // union. LidarPlaneReader sets the merged stamp to the MAX of the planes' capture stamps, so it
    // advances whenever EITHER plane is fresh; but such a sweep still carries the OTHER plane's cached
    // points, so it is not a new observation of the whole field of view. Counting those inflates the rate
    // toward the sum of the planes (~40 for two at 20 Hz, landing at ~28-30 in practice). Counting the
    // MINIMUM plane stamp instead gives one increment per FULL refresh — 20 Hz for two 20 Hz planes,
    // independent of their relative phase, which is the rate at which genuinely new complete data arrives.
    mutable std::atomic<std::uint64_t> rx_lidar_plane0_total_{0};
    mutable std::vector<std::uint64_t> lidar_last_counted_;   // per-plane stamps at the last counted update

    mutable std::atomic<std::uint64_t> latest_rgb_stamp_{0};    // newest rgb stamp seen (rate telemetry / validity)
    // Arrival-cadence probe (see the note in drain_media_plane). Ingest-thread only, no locking needed.
    mutable std::chrono::steady_clock::time_point probe_prev_wall_{};
    mutable std::uint64_t                         probe_prev_stamp_ = 0;
    mutable std::array<int, 6>                    probe_stamp_hist_{};
    mutable std::array<int, 6>                    probe_wall_hist_{};
    mutable std::atomic<std::uint64_t> latest_depth_stamp_{0};  // newest depth stamp seen
    mutable std::atomic<std::uint64_t> last_frame_ts_{0};       // stamp of the last assembled RGBD (get_frame_timestamp_ms)

    struct MediaRgbCache   // used by the ricoh single-latest cache below
    {
        bool          valid = false;
        std::uint64_t frame_id = 0;
        std::uint64_t stamp = 0;   // camera alivetime (ms), opaque timestamp
        int           width = 0;
        int           height = 0;
        cv::Mat       bgr;         // CV_8UC3
    };

    std::unique_ptr<rc::media::Image360Subscriber> media_ricoh_sub_;
    mutable std::mutex media_ricoh_mutex_;
    MediaRgbCache      media_ricoh_;
public:
    // Exact source rates: arrivals since process start. Caller differentiates against wall time.
    [[nodiscard]] std::uint64_t rx_rgb_total()   const { return rx_rgb_total_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t rx_ricoh_total() const { return rx_ricoh_total_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t rx_lidar_total() const { return rx_lidar_plane0_total_.load(std::memory_order_relaxed); }
private:
    std::jthread       ingest_thread_;
    std::atomic<bool>  ingest_stop_{false};
    std::atomic<bool>  ricoh_wanted_{false};
    std::atomic<std::uint64_t> ricoh_last_stamp_ms_{0};
};
