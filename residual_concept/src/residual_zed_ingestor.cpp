/*
 *    residual_zed_ingestor.cpp  —  see residual_zed_ingestor.h
 */

#include "residual_zed_ingestor.h"

#include <print>
#include <utility>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

ResidualZedIngestor::ResidualZedIngestor(std::shared_ptr<DSR::DSRGraph> graph, const ResidualConfig& cfg)
    : G_(std::move(graph)), cfg_(&cfg)
{
    // Subscriber comes up lazily in pump()/ensure_subscriber() once the "zed" node + descriptor exist.
}

ResidualZedIngestor::~ResidualZedIngestor()
{
    depth_sub_.reset();
}

bool ResidualZedIngestor::ensure_subscriber()
{
    if (depth_sub_)
        return true;
    if (not G_)
        return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_init_ < std::chrono::seconds(1))
        return false;
    last_init_ = now;

    // Descriptor-driven config from the producer's "zed" node (domain/topic/QoS); no config entry needed.
    auto desc = rc::media::descriptor_from_graph(*G_, "zed");
    if (not desc.has_value())
        return false;
    auto cfg = desc->subscriber_config("depth");
    if (not cfg.has_value())
        return false;

    depth_sub_ = std::make_unique<rc::media::MediaSubscriber>();
    if (not depth_sub_->init(*cfg))
    {
        depth_sub_.reset();
        return false;
    }
    std::println("[ResidualZed] depth subscriber up (zed/depth, domain={} '{}')", cfg->domain_id, cfg->topic_name);
    return true;
}

void ResidualZedIngestor::read_intrinsics(int width, int height)
{
    // Focal lengths from the static "zed" node; principal point = image centre (rectified stream).
    K_.cx = 0.5f * static_cast<float>(width);
    K_.cy = 0.5f * static_cast<float>(height);
    if (const auto zed = G_->get_node("zed"); zed.has_value())
    {
        if (const auto fx = G_->get_attrib_by_name<cam_depth_focalx_att>(zed.value()); fx.has_value())
            K_.fx = static_cast<float>(fx.value());
        if (const auto fy = G_->get_attrib_by_name<cam_depth_focaly_att>(zed.value()); fy.has_value())
            K_.fy = static_cast<float>(fy.value());
    }
}

bool ResidualZedIngestor::pump()
{
    if (not ensure_subscriber())
        return false;

    bool fresh = false;
    std::int64_t newest = last_ts_;
    depth_sub_->poll([&](const rc::media::ImageFrame& f, std::int64_t)
    {
        const auto ts = static_cast<std::int64_t>(f.stamp_ms());
        if (ts <= newest)
            return;
        const int w = static_cast<int>(f.width()), h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0)
            return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        if (f.format() == rc::media::FORMAT_DEPTH_F32)
        {
            if (f.size() < npix * sizeof(float)) return;
            const float* p = reinterpret_cast<const float*>(f.data().data());
            depth_.depth.assign(p, p + npix);
        }
        else if (f.format() == rc::media::FORMAT_Z16)
        {
            if (f.size() < npix * sizeof(std::uint16_t)) return;
            const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(f.data().data());
            depth_.depth.resize(npix);
            for (std::size_t i = 0; i < npix; ++i)
                depth_.depth[i] = static_cast<float>(p[i]) * 0.001f;   // mm → m
        }
        else
            return;
        depth_.width = w; depth_.height = h;
        newest = ts;
        fresh = true;
    });

    if (not fresh)
        return false;
    last_ts_ = newest;
    read_intrinsics(depth_.width, depth_.height);
    return true;
}

}  // namespace rc
