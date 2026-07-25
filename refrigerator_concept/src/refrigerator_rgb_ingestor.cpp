/*
 *    refrigerator_rgb_ingestor.cpp  —  see refrigerator_rgb_ingestor.h
 */

#include "refrigerator_rgb_ingestor.h"

#include <cstring>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

RefrigeratorRgbIngestor::RefrigeratorRgbIngestor(std::shared_ptr<DSR::DSRGraph> graph, const RefrigeratorConfig& cfg)
    : G_(std::move(graph)), cfg_(&cfg)
{
    // Subscriber comes up lazily inside pump() once the "zed" node + descriptor exist AND the feature is on;
    // nothing touches DDS here (never from a ctor / free-running thread — media-plane consumer pattern).
}

RefrigeratorRgbIngestor::~RefrigeratorRgbIngestor()
{
    sub_.reset();
}

bool RefrigeratorRgbIngestor::try_discover()
{
    if (sub_ or not G_ or not cfg_->front_detect_enabled)
        return false;
    // Self-throttle discovery attempts to ~1 Hz (pump() runs at the compute rate).
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_attempt_ < std::chrono::seconds(1))
        return false;
    last_discovery_attempt_ = now;

    // Shared descriptor-driven factory (identical init code to every other agent): verifies the "zed" node +
    // descriptor exist and reads the DDS domain/topic from the JSON (dedicated media domain, not the Agent domain).
    sub_ = rc::media::make_image_subscriber_from_graph(*G_, camera_node_name_, "rgb");
    return sub_ != nullptr;
}

bool RefrigeratorRgbIngestor::pump()
{
    fresh_ = false;
    if (not cfg_->front_detect_enabled)
        return false;   // feature off ⇒ fully dormant (no DDS participant ever created)
    if (not sub_)
    {
        try_discover();
        return false;
    }

    // Drain to the NEWEST available frame (poll delivers each queued sample; keep the last). The ImageFrame is a
    // loaned SHM view valid only during the callback → DEEP-COPY (clone) the pixels out before it returns.
    const int delivered = sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0)
            return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const std::uint32_t fmt = f.format();
        std::size_t expected = 0;
        switch (fmt)
        {
            case rc::media::FORMAT_BGR8:
            case rc::media::FORMAT_RGB8:  expected = npix * 3; break;
            case rc::media::FORMAT_GRAY8: expected = npix;     break;
            default: return;   // depth / unknown formats are not consumed here
        }
        if (f.size() < expected)
            return;

        const uchar* data = reinterpret_cast<const uchar*>(f.data().data());
        // Normalise every format to a DEEP-COPIED 3-channel BGR frame. The door-ness metric only uses the
        // grayscale gradient, so the R/B convention is immaterial to the score — but storing canonical BGR keeps
        // the frame directly usable by any later overlay. cvtColor / clone allocate the ingestor-owned buffer.
        switch (fmt)
        {
            case rc::media::FORMAT_BGR8:
            {
                const cv::Mat view(h, w, CV_8UC3, const_cast<uchar*>(data));
                frame_ = view.clone();                       // deep copy out of the loaned SHM segment
                break;
            }
            case rc::media::FORMAT_RGB8:
            {
                const cv::Mat view(h, w, CV_8UC3, const_cast<uchar*>(data));
                cv::cvtColor(view, frame_, cv::COLOR_RGB2BGR);  // cvtColor writes a fresh owned buffer
                break;
            }
            case rc::media::FORMAT_GRAY8:
            {
                const cv::Mat view(h, w, CV_8UC1, const_cast<uchar*>(data));
                cv::cvtColor(view, frame_, cv::COLOR_GRAY2BGR);
                break;
            }
            default:
                return;
        }
        stamp_ms_ = f.stamp_ms();
        fresh_    = true;
    });
    return delivered > 0 and fresh_;
}

}  // namespace rc
