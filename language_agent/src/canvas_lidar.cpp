#include "canvas_lidar.h"

#include <algorithm>
#include <format>

#include <QMatrix4x4>

#include "../../common/media_transport/media_transport.h"

namespace rc::chat
{

LidarGrabber::LidarGrabber(std::shared_ptr<DSR::DSRGraph> G) : G_(std::move(G))
{
    rings_[0].node = "helios";
    rings_[1].node = "bpearl";
}

LidarGrabber::~LidarGrabber() = default;

void LidarGrabber::want(std::vector<std::string> sensors)
{
    if (sensors.empty()) sensors = {"helios", "bpearl"};
    if (sensors != wanted_)
    {
        wanted_ = std::move(sensors);
        next_try_ = {};
        for (auto& r : rings_)
            if (std::ranges::find(wanted_, r.node) == wanted_.end()) { r.sub.reset(); r.points.clear(); }
    }
    retry();
}

void LidarGrabber::retry()
{
    const auto now = std::chrono::steady_clock::now();
    if (now < next_try_) return;
    next_try_ = now + retry_period;
    for (auto& r : rings_)
        if (not r.sub and std::ranges::find(wanted_, r.node) != wanted_.end())
            r.sub = rc::media::make_lidar_subscriber_from_graph(*G_, r.node, "lidar");
}

void LidarGrabber::release()
{
    for (auto& r : rings_) { r.sub.reset(); r.points.clear(); }
    wanted_.clear();
    stamp_ms_ = 0;
}

bool LidarGrabber::poll(DSR::InnerEigenAPI& inner)
{
    bool changed = false;
    for (auto& r : rings_)
    {
        if (not r.sub) continue;
        // body <- sensor, resolved now (main thread, the main thread's instance); identity when it does not resolve
        QMatrix4x4 xf;
        if (const auto T = inner.get_transformation_matrix("body", r.node); T.has_value())
        {
            const auto& M = T->matrix();
            xf = QMatrix4x4(float(M(0, 0)), float(M(0, 1)), float(M(0, 2)), float(M(0, 3)),
                            float(M(1, 0)), float(M(1, 1)), float(M(1, 2)), float(M(1, 3)),
                            float(M(2, 0)), float(M(2, 1)), float(M(2, 2)), float(M(2, 3)), 0.f, 0.f, 0.f, 1.f);
        }
        r.sub->poll([&](const rc::media::LidarFrame& f, std::int64_t)
        {
            // a loaned view, valid only inside this call: copy the points out (transformed) before returning
            const std::size_t stride = f.stride() > 0 ? f.stride() : 3;
            const auto& pts = f.points();
            const std::size_t n = std::min<std::size_t>(f.count(), pts.size() / stride);
            r.points.clear();
            r.points.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
                r.points.push_back(xf.map(QVector3D(pts[i * stride + 0], pts[i * stride + 1], pts[i * stride + 2])));
            if (&r == &rings_[0] or not rings_[0].sub) stamp_ms_ = f.stamp_ms();   // the primary ring paces the rate
            changed = true;
        });
    }
    return changed;
}

std::string LidarGrabber::status() const
{
    std::string s;
    for (const auto& r : rings_)
    {
        if (std::ranges::find(wanted_, r.node) == wanted_.end()) continue;
        if (not s.empty()) s += " · ";
        s += r.sub ? std::format("{} {} pts", r.node, r.points.size())
                   : std::format("{}: no stream (no node or no media descriptor)", r.node);
    }
    return s;
}

}   // namespace rc::chat
