/*
 * rgb_ingestor.h — ZED RGB media-plane consumer, SHARED.
 *
 * ★WHY THIS FILE EXISTS. hood_concept and refrigerator_concept each carried a private copy of this
 * ingestor — 177 lines, byte-identical after substituting the agent's name. door_concept needed the same
 * thing (to score its projected silhouette against the RGB), and a third verbatim copy is how a fleet
 * ends up fixing a media-plane bug in two places out of three. This is that code with the agent-specific
 * config replaced by two plain arguments.
 * ⚠hood and refrigerator still use their own copies: migrating a running agent is a separate change with
 * its own restart, and this one only claims the NEW caller. They should be pointed here when convenient.
 *
 * Sibling of common/lidar_ingestor (the sanctioned media-plane consumer pattern). Brings up the shared
 * zero-copy ImageFrame subscriber on the "zed"/"rgb" stream and, each cycle, decodes the newest frame
 * into a DEEP-COPIED BGR cv::Mat plus its capture stamp.
 *
 * Crash-safety (unchanged from the original, and all of it load-bearing):
 *  - The subscriber is created LAZILY in pump(), from the Operating compute/main thread — never in a
 *    constructor and never on a free-running thread — and only once the camera node + media descriptor
 *    exist AND the feature is enabled. Discovery self-throttles to ~1 Hz.
 *  - The ImageFrame handed to poll()'s callback is a loaned SHM view valid only for that call, so it is
 *    DEEP-COPIED out during the callback. A shallow cv::Mat handle held across cycles is the documented
 *    cross-thread heap smash in CLAUDE.md.
 *  - The owner must reset this ingestor BEFORE tearing the graph/subscriber down.
 *  - Fully dormant (no DDS participant) while disabled.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <dsr/api/dsr_api.h>


namespace rc::media { class MediaSubscriber; }

namespace rc
{

class RgbIngestor
{
public:
    // `enabled` is read on every pump() so a feature toggle can turn the plane on/off at runtime
    // without reconstructing anything; `camera_node` is the DSR node carrying the descriptor.
    RgbIngestor(std::shared_ptr<DSR::DSRGraph> graph, const bool* enabled,
                std::string camera_node = "zed");
    ~RgbIngestor();

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
    const bool*                                 enabled_ = nullptr;
    std::unique_ptr<rc::media::MediaSubscriber> sub_;
    std::string                                 camera_node_name_;

    cv::Mat        frame_;                     // latest decoded BGR frame (deep copy of the loaned SHM view)
    std::uint64_t  stamp_ms_ = 0;
    bool           fresh_    = false;
    std::chrono::steady_clock::time_point last_discovery_attempt_{};
};

}  // namespace rc
