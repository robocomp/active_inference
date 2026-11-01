#include "camera_grabber.h"

#include <format>
#include <print>

#include <opencv2/imgproc.hpp>

#include "../../common/media_transport/media_transport.h"

namespace rc::chat
{

namespace
{
constexpr const char* kFrontNode = "zed";
constexpr const char* kFrontStream = "rgb";
constexpr const char* kPanoNode = "ricoh";
constexpr const char* kPanoStream = "rgb360";

std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

/// Deep copy of a loaned 8-bit view into an owned BGR cv::Mat. `rgb` = the bytes are R,G,B; `grey` = one channel.
cv::Mat owned_bgr(const unsigned char* data, std::size_t size, int w, int h, bool rgb, bool grey)
{
    const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    if (w <= 0 or h <= 0 or size < npix * (grey ? 1u : 3u)) return {};
    cv::Mat out;
    if (grey)
        cv::cvtColor(cv::Mat(h, w, CV_8UC1, const_cast<unsigned char*>(data)), out, cv::COLOR_GRAY2BGR);
    else if (rgb)
        cv::cvtColor(cv::Mat(h, w, CV_8UC3, const_cast<unsigned char*>(data)), out, cv::COLOR_RGB2BGR);
    else
        out = cv::Mat(h, w, CV_8UC3, const_cast<unsigned char*>(data)).clone();   // out of the SHM loan
    return out;
}
}   // namespace

CameraGrabber::CameraGrabber(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G)) {}

CameraGrabber::~CameraGrabber()
{
    front_.reset();
    pano_.reset();
}

void CameraGrabber::discover()
{
    if (not G_ or (front_ and pano_)) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < next_try_) return;
    next_try_ = now + retry_period;
    // Node absent ⇒ this robot has no such camera (or it is not up yet): stay quiet, ask again later.
    if (not front_ and G_->get_node(kFrontNode).has_value())
        front_ = rc::media::make_image_subscriber_from_graph(*G_, kFrontNode, kFrontStream);
    if (not pano_ and G_->get_node(kPanoNode).has_value())
        pano_ = rc::media::make_image360_subscriber_from_graph(*G_, kPanoNode, kPanoStream);
}

void CameraGrabber::refresh(Camera cam)
{
    const bool front = cam == Camera::front;
    if (front ? not front_ : not pano_) return;
    Cached& last = front ? front_last_ : pano_last_;
    if (front)
    {
        front_->poll([&last, this](const rc::media::ImageFrame& f, std::int64_t)
        {
            const std::uint32_t fmt = f.format();
            if (fmt != rc::media::FORMAT_BGR8 and fmt != rc::media::FORMAT_RGB8 and fmt != rc::media::FORMAT_GRAY8) return;
            cv::Mat m = owned_bgr(reinterpret_cast<const unsigned char*>(f.data().data()), f.size(),
                                  static_cast<int>(f.width()), static_cast<int>(f.height()),
                                  fmt == rc::media::FORMAT_RGB8, fmt == rc::media::FORMAT_GRAY8);
            if (m.empty()) return;
            last.bgr = std::move(m);
            last.source_ns = front_->last_source_ns();
            last.rate.push(f.frame_id(), static_cast<std::uint64_t>(last.source_ns));
        });
    }
    else
    {
        pano_->poll([&last, this](const rc::media::Image360Frame& f, std::int64_t)
        {
            const std::uint32_t fmt = f.format();
            cv::Mat m = owned_bgr(reinterpret_cast<const unsigned char*>(f.data().data()), f.size(),
                                  static_cast<int>(f.width()), static_cast<int>(f.height()),
                                  fmt == rc::media::IMG360_FORMAT_RGB8, fmt == rc::media::IMG360_FORMAT_GRAY8);
            if (m.empty()) return;
            last.bgr = std::move(m);
            last.source_ns = pano_->last_source_ns();
            last.rate.push(f.frame_id(), static_cast<std::uint64_t>(last.source_ns));
        });
    }
}

CameraGrabber::StreamStats CameraGrabber::stats(Camera cam) const
{
    const bool front = cam == Camera::front;
    const Cached& last = front ? front_last_ : pano_last_;
    StreamStats s{.subscribed = front ? bool(front_) : bool(pano_)};
    s.fps = last.rate.fps();
    if (last.source_ns > 0) s.frame_age_ms = std::uint64_t(std::max<std::int64_t>(0, now_ns() - last.source_ns) / 1'000'000);
    s.width = last.bgr.cols;
    s.height = last.bgr.rows;
    return s;
}

std::expected<GrabbedFrame, std::string> CameraGrabber::grab(Camera cam)
{
    const bool front = cam == Camera::front;
    if (front ? not front_ : not pano_)
    {
        discover();   // maybe the node appeared since the last timer tick (throttled all the same)
        if (front ? not front_ : not pano_)
            return std::unexpected(std::format("camera not available: no '{}' media stream on node '{}' yet",
                                               front ? kFrontStream : kPanoStream, front ? kFrontNode : kPanoNode));
    }
    refresh(cam);
    const Cached& last = front ? front_last_ : pano_last_;
    if (last.bgr.empty())
        return std::unexpected(std::format("camera not available: the {} has not delivered a frame yet", label(cam)));
    GrabbedFrame g;
    g.bgr = last.bgr.clone();   // the caller owns its copy; the cache stays ours
    g.age_ms = last.source_ns > 0 ? (now_ns() - last.source_ns) / 1'000'000 : -1;
    g.label = label(cam);
    return g;
}

}   // namespace rc::chat
