/*
 * hood_rgb_ingestor.h  —  ZED RGB media-plane consumer for appearance-based FRONT (door) detection
 *
 * Sibling of hood_lidar_ingestor.h (the sanctioned in-agent media-plane pattern). Brings up the shared
 * zero-copy ImageFrame subscriber on the "zed"/"rgb" stream (same descriptor-driven factory every agent uses)
 * and, each compute() cycle, decodes the newest frame into a DEEP-COPIED BGR cv::Mat + its capture stamp — the
 * pixels HoodProjection::detect_front scores for door-ness. The fitted 3D box (not YOLO's 2D mask) is what
 * gets projected, so this keeps working when the robot is too close for YOLO to detect the fridge.
 *
 * Crash-safety (mirrors HoodLidarIngestor / room_concept::CameraVisualizer):
 *  - The subscriber is created LAZILY in pump() (called from the Operating compute/main thread), never in a ctor
 *    or a free-running thread, and only once the "zed" node + media descriptor exist AND the feature is enabled
 *    (cfg.front_detect_enabled). Discovery is self-throttled to ~1 Hz.
 *  - The ImageFrame delivered to poll()'s callback is a loaned SHM view valid only for the call, so the frame is
 *    DEEP-COPIED (cv::Mat::clone) out during the callback — never a shallow handle held across cycles.
 *  - The owner (SpecificWorker) must reset this ingestor BEFORE tearing the graph/subscriber down.
 *  - Stays fully dormant (no DDS participant) while cfg.front_detect_enabled is false.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <dsr/api/dsr_api.h>

#include "hood_config.h"

namespace rc::media { class MediaSubscriber; }

namespace rc
{

class HoodRgbIngestor
{
public:
    HoodRgbIngestor(std::shared_ptr<DSR::DSRGraph> graph, const HoodConfig& cfg);
    ~HoodRgbIngestor();

    // Drain the newest ZED RGB frame off the media plane into frame_ (deep-copied BGR) + stamp_ms_. Brings the
    // subscriber up lazily on first call once the "zed" descriptor exists. Returns true iff a fresh frame was
    // decoded this call. Main-thread only (reads the graph for descriptor discovery). No-op while disabled.
    bool pump();

    const cv::Mat&  frame()    const { return frame_; }      // latest BGR frame (empty until the first pump)
    std::uint64_t   stamp_ms() const { return stamp_ms_; }   // its capture stamp (producer's frame stamp)
    bool            fresh()    const { return fresh_; }       // a new frame arrived on the last pump()

private:
    bool try_discover();   // self-throttled lazy subscriber bring-up (descriptor-driven factory)

    std::shared_ptr<DSR::DSRGraph>              G_;
    const HoodConfig*                   cfg_ = nullptr;
    std::unique_ptr<rc::media::MediaSubscriber> sub_;
    std::string                                 camera_node_name_ = "zed";

    cv::Mat        frame_;                     // latest decoded BGR frame (deep copy of the loaned SHM view)
    std::uint64_t  stamp_ms_ = 0;
    bool           fresh_    = false;
    std::chrono::steady_clock::time_point last_discovery_attempt_{};
};

}  // namespace rc
