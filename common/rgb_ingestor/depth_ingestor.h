/*
 * depth_ingestor.h — ZED DEPTH media-plane consumer, SHARED. Sibling of rgb_ingestor.h.
 *
 * ★WHY A SEPARATE INGESTOR RATHER THAN A SECOND STREAM INSIDE RgbIngestor. Six components subscribe to
 * "rgb" today and exactly two want depth; folding a second subscriber into the shared RGB ingestor would
 * make all six create a DDS reader they never read. It is also a different decode with a different
 * failure mode — see the invalid-pixel rule below — and the two do not belong under one `fresh()`.
 *
 * DELIVERS a CV_32F image in METRES, decoded from either wire format the plane carries:
 *   FORMAT_DEPTH_F32 — already metres; may contain inf/nan on a miss
 *   FORMAT_Z16       — millimetres, with 0 meaning NO RETURN
 *
 * ★A MISS IS NaN HERE, NEVER 0.0. Z16's 0 is the sensor saying "I have no reading", and a consumer that
 * receives it as the number zero reads it as a surface 0 m away — in front of everything, agreeing with
 * nothing, and perfectly confident about it. Normalising every miss to NaN once, here, means no consumer
 * can make that mistake, and the consumers' own `isfinite` checks then mean what they appear to mean.
 *
 * ⚠MAX_IMAGE_BYTES IS 3686400 = exactly 1280x720x4, so FORMAT_DEPTH_F32 at that resolution fits with ZERO
 * margin and anything larger is dropped SILENTLY by the media plane — the producer looks healthy while the
 * consumer reads 0.0 Hz for ever. Z16 at the same resolution is 1.84 MB and has room to spare. If depth
 * never arrives, suspect the size before the wiring. See [[media-plane-max-image-bytes]].
 *
 * ⚠THE DEPTH PLANE NEED NOT SHARE THE RGB RESOLUTION, and a consumer that indexes it with RGB pixel
 * coordinates will read the wrong part of the scene without any error. width()/height() are exposed for
 * exactly that reason; rc::edges::ContourDepthParams takes the resulting scale factors.
 *
 * Same crash-safety contract as RgbIngestor: subscriber created LAZILY in pump() on the Operating
 * compute/main thread (never a ctor, never a free-running thread), discovery self-throttled to ~1 Hz, the
 * loaned SHM view DEEP-COPIED out during the callback, fully dormant while disabled.
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

class DepthIngestor
{
public:
    DepthIngestor(std::shared_ptr<DSR::DSRGraph> graph, const bool* enabled,
                  std::string camera_node = "zed");
    ~DepthIngestor();

    // Drain the newest depth frame into frame_ (CV_32F metres, NaN where there is no return) + stamp_ms_.
    // Returns true iff a fresh frame was decoded. Main-thread only (reads the graph to discover). No-op
    // while disabled.
    bool pump();

    const cv::Mat&  frame()    const { return frame_; }      // CV_32F metres; NaN = no return. Empty until first pump
    std::uint64_t   stamp_ms() const { return stamp_ms_; }
    bool            fresh()    const { return fresh_; }
    int             width()    const { return frame_.cols; }
    int             height()   const { return frame_.rows; }

private:
    bool try_discover();

    std::shared_ptr<DSR::DSRGraph>              G_;
    const bool*                                 enabled_ = nullptr;
    std::unique_ptr<rc::media::MediaSubscriber> sub_;
    std::string                                 camera_node_name_;

    cv::Mat        frame_;
    std::uint64_t  stamp_ms_ = 0;
    bool           fresh_    = false;
    bool           absent_warned_ = false;
    std::chrono::steady_clock::time_point last_discovery_attempt_{};
};

}  // namespace rc
