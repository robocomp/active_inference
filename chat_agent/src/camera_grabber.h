#pragma once
/*
 * camera_grabber.h — the newest camera frame, on demand, for the look() tool (chat_session.h).
 *
 *   front = ZED rgb   (node "zed",   stream "rgb",    rc::media::MediaSubscriber)
 *   360   = Ricoh     (node "ricoh", stream "rgb360", rc::media::Image360Subscriber)
 *
 * Media-plane consumer pattern (CLAUDE.md):
 *  - subscribers come ONLY from the shared descriptor-driven factories (common/media_transport/media_transport.h),
 *    created by discover() on the MAIN thread, after the graph is loaded and only once the sensor node exists.
 *    discover() is called from the session's digest timer and self-throttles; a robot without a camera node is a
 *    permanent, quiet state.
 *  - frames are pulled ON DEMAND in grab(): the subscribers deliver latest-only (drain_newest), so one poll() at
 *    look() time yields the newest sample; nothing is decoded or copied while nobody asks.
 *  - the ImageFrame handed to poll()'s callback is a loaned SHM view valid only during the call ⇒ DEEP-COPIED
 *    (clone / cvtColor into an owned buffer) before it returns.
 *  - main thread only; nothing here is shared with another thread.
 *
 * The age of a frame is measured from the DDS source timestamp (system_clock, when the producer wrote it), not from
 * the frame's own stamp_ms, which in simulation may be simulated time.
 */

#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <dsr/api/dsr_api.h>

namespace rc::media { class MediaSubscriber; class Image360Subscriber; }

namespace rc::chat
{

enum class Camera { front, pano };

struct GrabbedFrame
{
    cv::Mat bgr;                 ///< owned, deep-copied BGR
    std::int64_t age_ms = 0;     ///< now − producer write time
    const char* label = "";      ///< "front camera" / "360-degree camera"
};

class CameraGrabber
{
public:
    explicit CameraGrabber(std::shared_ptr<DSR::DSRGraph> G);
    ~CameraGrabber();

    /// Bring up the subscribers whose sensor node + descriptor exist. Self-throttled (retry_period); main thread.
    void discover();
    /// The newest frame of that camera, or why there is none ("camera not available: ...").
    [[nodiscard]] std::expected<GrabbedFrame, std::string> grab(Camera cam);

    [[nodiscard]] static const char* label(Camera cam) { return cam == Camera::front ? "front camera" : "360-degree camera"; }

private:
    struct Cached
    {
        cv::Mat bgr;
        std::int64_t source_ns = 0;   ///< DDS source timestamp of `bgr` (system_clock epoch ns)
    };

    std::shared_ptr<DSR::DSRGraph> G_;
    std::unique_ptr<rc::media::MediaSubscriber> front_;
    std::unique_ptr<rc::media::Image360Subscriber> pano_;
    Cached front_last_, pano_last_;   ///< the last frame grabbed, reused when no newer one has arrived
    std::chrono::steady_clock::time_point next_try_{};
    static constexpr std::chrono::seconds retry_period{5};   ///< descriptor discovery retry (the factory logs each miss)
};

}   // namespace rc::chat
