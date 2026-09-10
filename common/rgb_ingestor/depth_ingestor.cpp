/*
 *    depth_ingestor.cpp  —  see depth_ingestor.h
 */

#include "depth_ingestor.h"

#include <cmath>
#include <limits>
#include <print>
#include <utility>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

namespace
{
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
}

DepthIngestor::DepthIngestor(std::shared_ptr<DSR::DSRGraph> graph, const bool* enabled, std::string camera_node)
    : G_(std::move(graph)), enabled_(enabled), camera_node_name_(std::move(camera_node))
{
}

DepthIngestor::~DepthIngestor()
{
    sub_.reset();
}

bool DepthIngestor::try_discover()
{
    if (sub_ or not G_ or not (enabled_ != nullptr and *enabled_))
        return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_attempt_ < std::chrono::seconds(1))
        return false;
    last_discovery_attempt_ = now;

    // Say the absence out loud ONCE. A camera that advertises no depth stream is a permanent condition
    // for this run, and a consumer silently scoring nothing looks identical to a consumer scoring zero.
    if (const auto desc = rc::media::descriptor_from_graph(*G_, camera_node_name_);
        desc.has_value() and not desc->streams.contains("depth"))
    {
        if (not absent_warned_)
        {
            absent_warned_ = true;
            std::print("[depth_ingestor] node '{}' advertises no 'depth' stream — the depth contour "
                       "channel will stay silent for this run\n", camera_node_name_);
        }
        return false;
    }

    sub_ = rc::media::make_image_subscriber_from_graph(*G_, camera_node_name_, "depth");
    if (sub_)
        std::print("[depth_ingestor] depth subscriber up on '{}' (Z16 -> 1.84 MB, F32 -> 3.69 MB at "
                   "1280x720; the plane's ceiling is 3.69 MB)\n", camera_node_name_);
    return sub_ != nullptr;
}

bool DepthIngestor::pump()
{
    fresh_ = false;
    if (not (enabled_ != nullptr and *enabled_))
        return false;
    if (not sub_)
    {
        try_discover();
        return false;
    }

    const int delivered = sub_->poll([this](const rc::media::ImageFrame& f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0)
            return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const std::uint32_t fmt = f.format();
        const bool f32 = (fmt == rc::media::FORMAT_DEPTH_F32);
        const bool z16 = (fmt == rc::media::FORMAT_Z16);
        if (not f32 and not z16)
            return;                                  // colour formats are not consumed here
        if (f.size() < npix * (f32 ? 4u : 2u))
            return;

        // Decode into an ingestor-OWNED buffer: the ImageFrame is a loaned SHM view valid only for this
        // callback, so nothing may survive it by reference.
        frame_.create(h, w, CV_32F);
        const auto* p32 = reinterpret_cast<const float*>(f.data().data());
        const auto* p16 = reinterpret_cast<const std::uint16_t*>(f.data().data());
        for (int y = 0; y < h; ++y)
        {
            float* dst = frame_.ptr<float>(y);
            for (int x = 0; x < w; ++x)
            {
                const std::size_t idx = static_cast<std::size_t>(y) * w + x;
                // ★EVERY MISS BECOMES NaN, including Z16's 0 — see the header. A miss that arrives as
                // the number zero is a surface 0 m away that nothing downstream can tell from a reading.
                if (f32)
                {
                    const float d = p32[idx];
                    dst[x] = (std::isfinite(d) and d > 0.0f) ? d : kNaN;
                }
                else
                {
                    const std::uint16_t mm = p16[idx];
                    dst[x] = (mm == 0) ? kNaN : static_cast<float>(mm) * 1e-3f;
                }
            }
        }
        stamp_ms_ = f.stamp_ms();
        fresh_    = true;
    });
    return delivered > 0 and fresh_;
}

}  // namespace rc
