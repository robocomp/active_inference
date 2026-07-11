#pragma once

// Zero-copy DDS media-plane ingest for the voxelizer: ZED RGB+depth, the shared multi-plane LiDAR
// reader, and the Ricoh-360 panorama. Split out of SceneProcessor so all DDS subscriber/cache/thread
// state lives in one place; SceneProcessor holds one of these and forwards to it. Subscribers are
// created from the producer's per-node media descriptors in the graph (never from config).

#include "rgbd_data.h"
#include "zed_frame_aligner.h"   // ZedFrameAligner (PIMPL over BufferSync — no doublebuffer/threadpool leak)

#include <opencv2/core.hpp>

#include <atomic>
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
    void drain_media_plane() const;   // polls RGB+depth subscribers, refreshes the caches

    std::shared_ptr<DSR::DSRGraph> graph_;

    std::unique_ptr<rc::media::MediaSubscriber> media_rgb_sub_;
    std::unique_ptr<rc::media::MediaSubscriber> media_depth_sub_;

    std::unique_ptr<rc::media::LidarPlaneReader> lidar_reader_;
    bool      lidar_use_media_ = false;
    LidarData media_lidar_;
    bool      media_lidar_valid_ = false;

    // ZED rgb+depth: timestamp-aligned (rgb frame F always meets depth frame F, no "latest of each" skew).
    ZedFrameAligner zed_buf_{8};
    mutable std::atomic<std::uint64_t> latest_rgb_stamp_{0};    // newest rgb stamp seen (rate telemetry / validity)
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
    std::atomic<bool>  ricoh_wanted_{false};
    std::atomic<std::uint64_t> ricoh_last_stamp_ms_{0};
};
