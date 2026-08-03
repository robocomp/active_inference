/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see lidar_plane_reader.h.
 */

#include "lidar_plane_reader.h"

#include <cmath>
#include <print>
#include <utility>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "media_transport.h"

namespace rc::media
{

LidarPlaneReader::LidarPlaneReader(std::shared_ptr<DSR::DSRGraph> graph,
                                   DSR::InnerEigenAPI* inner_eigen,
                                   std::vector<std::string> preferred_planes,
                                   std::string fused_fallback,
                                   std::string stream_key)
    : G_(std::move(graph)), inner_eigen_(inner_eigen),
      preferred_(std::move(preferred_planes)), fallback_(std::move(fused_fallback)),
      stream_key_(std::move(stream_key))
{
    preferred_planes_.reserve(preferred_.size());
    for (const auto& node : preferred_)
        preferred_planes_.push_back(Plane{node, nullptr, 0});
    fallback_plane_.node = fallback_;
}

LidarPlaneReader::~LidarPlaneReader()
{
    // Subscribers hold DDS participants; drop them before the reader dies.
    for (auto& p : preferred_planes_)
        p.sub.reset();
    fallback_plane_.sub.reset();
}

bool LidarPlaneReader::any_live() const noexcept
{
    for (const auto& p : preferred_planes_)
        if (p.sub)
            return true;
    return fallback_plane_.sub != nullptr;
}

void LidarPlaneReader::ensure_subscribers()
{
    if (not G_)
        return;

    // Self-throttle discovery to ~1 Hz (poll runs every compute cycle).
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_ < std::chrono::seconds(1))
        return;
    last_discovery_ = now;

    // Bring up each preferred plane as soon as its node + media descriptor exist. The factory is the
    // same descriptor-driven init every agent uses; it returns nullptr until the plane is live.
    bool any_preferred = false;
    for (auto& p : preferred_planes_)
    {
        if (not p.sub)
        {
            p.sub = make_lidar_subscriber_from_graph(*G_, p.node, stream_key_);
            if (p.sub)
            {
                p.sub_since = now;
                std::println("[Lidar] plane '{}' subscriber created (device frame → RT transform)"
                             " — waiting for the publisher", p.node);
            }
        }
        any_preferred = any_preferred or (p.sub != nullptr);
        report_discovery(p);
    }

    if (any_preferred)
    {
        // A per-device plane owns the data now — never double-ingest the fused fallback.
        if (fallback_plane_.sub)
        {
            std::println("[Lidar] per-device planes live — dropping fused '{}' fallback", fallback_);
            fallback_plane_.sub.reset();
        }
        return;
    }

    // No per-device plane yet: fall back to the fused (already-robot-frame) plane while robot_concept
    // is bridging. Its node frame is identity vs. the robot base, so the RT transform is a near no-op.
    if (not fallback_plane_.sub and not fallback_.empty())
    {
        fallback_plane_.sub = make_lidar_subscriber_from_graph(*G_, fallback_plane_.node, stream_key_);
        if (fallback_plane_.sub)
        {
            fallback_plane_.sub_since = now;
            std::println("[Lidar] using fused '{}' plane (robot frame)", fallback_);
        }
    }
    report_discovery(fallback_plane_);
}

void LidarPlaneReader::report_discovery(Plane& p)
{
    // A DataReader that exists but is matched with NO publisher is the signature of a media plane
    // whose discovery is broken — and it is INVISIBLE otherwise, because "no frames" is also what a
    // perfectly healthy idle sensor looks like. That ambiguity is what made the 2026-07-21 stale-SHM
    // outage read as an agent regression for hours, so state which one it is, once, either way.
    // The delay before complaining is a REPORTING grace (discovery is not instantaneous), not a
    // tuning knob: nothing downstream branches on it.
    if (not p.sub)
        return;
    const int matched = p.sub->matched_writers();
    if (matched > 0)
    {
        if (not p.ever_matched)
        {
            p.ever_matched = true;
            std::println("[Lidar] plane '{}' LIVE — matched {} publisher(s)", p.node, matched);
        }
        return;
    }
    if (p.ever_matched or p.warned_unmatched or p.sub_since.time_since_epoch().count() == 0)
        return;
    if (std::chrono::steady_clock::now() - p.sub_since < std::chrono::seconds(3))
        return;
    p.warned_unmatched = true;
    std::println("[Lidar] ★ plane '{}': subscriber up for 3 s with NO publisher matched. The producer "
                 "is running but its announcements are not reaching us, so NO SCAN WILL EVER ARRIVE.",
                 p.node);
    std::println("[Lidar]   This is the media plane's SHM discovery, not this agent. Confirm with "
                 "'utils/dds_preflight/dds_preflight 7'; the cure is to stop every process on domain 7, "
                 "'rm -f /dev/shm/fastdds_*', and relaunch.");
}

std::int64_t LidarPlaneReader::drain_newest(LidarSubscriber& sub, std::int64_t prev_ts,
                                            std::vector<Eigen::Vector3f>& raw) const
{
    std::int64_t newest_ts = prev_ts;
    sub.poll([&](const LidarFrame& f, std::int64_t)
    {
        const auto ts = static_cast<std::int64_t>(f.stamp_ms());
        if (ts <= newest_ts)
            return;   // dedup by source stamp; keep only the newest sweep in this drain
        newest_ts = ts;
        const std::uint32_t stride = f.stride() ? f.stride() : 3u;
        const auto& pts = f.points();
        raw.clear();
        raw.reserve(f.count());
        for (std::uint32_t i = 0; i < f.count(); ++i)
        {
            const std::size_t base = static_cast<std::size_t>(i) * stride;
            if (base + 2 >= pts.size())
                break;
            const float x = pts[base], y = pts[base + 1], z = pts[base + 2];
            if (std::isfinite(x) and std::isfinite(y) and std::isfinite(z))
                raw.emplace_back(x, y, z);
        }
    });
    return newest_ts;
}

bool LidarPlaneReader::append_transformed(const std::string& node_frame, const std::string& target_frame,
                                          std::int64_t stamp_ms, bool interpolate,
                                          const std::vector<Eigen::Vector3f>& raw,
                                          std::vector<Eigen::Vector3f>& out, Eigen::Vector3f& origin_out) const
{
    if (not inner_eigen_ or raw.empty())
        return false;

    const auto time_query = interpolate ? DSR::RT_API::TimeQuery::Interpolated
                                        : DSR::RT_API::TimeQuery::Nearest;
    const auto T = inner_eigen_->get_transformation_matrix(target_frame, node_frame,
                                                           static_cast<std::uint64_t>(std::max<std::int64_t>(0, stamp_ms)),
                                                           "RT", time_query);
    if (not T.has_value())
        return false;

    // Element-wise copy (no aligned Eigen load) — mirrors the concept ingestors.
    Eigen::Matrix4d M;
    { const auto& s = T.value().matrix(); for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) M(i, j) = s(i, j); }

    out.reserve(out.size() + raw.size());
    for (const auto& p : raw)
    {
        const Eigen::Vector4d q = M * Eigen::Vector4d(p.x(), p.y(), p.z(), 1.0);
        out.emplace_back(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
    }
    origin_out = Eigen::Vector3f(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)), static_cast<float>(M(2, 3)));
    return true;
}

std::optional<LidarSweep> LidarPlaneReader::poll(const std::string& target_frame,
                                                 bool interpolate, bool enabled)
{
    if (not enabled or not G_)
        return std::nullopt;

    ensure_subscribers();

    // 1) Refresh each live plane's CACHE from its newest frame (dedup by source stamp). A plane with no
    //    fresh frame this poll keeps its previous cache — it is NOT dropped from the merge.
    std::vector<Eigen::Vector3f> raw;
    bool any_fresh = false;
    auto refresh = [&](Plane& p)
    {
        if (not p.sub)
            return;
        raw.clear();
        const std::int64_t ts = drain_newest(*p.sub, p.last_ts, raw);
        if (ts <= p.last_ts or raw.empty())
            return;   // no new frame → keep the existing cache
        p.last_ts   = ts;
        p.cache_ts  = ts;
        p.raw_cache = std::move(raw);
        any_fresh   = true;
    };

    bool any_preferred_live = false;
    for (auto& p : preferred_planes_)
    {
        if (p.sub) any_preferred_live = true;
        refresh(p);
    }
    if (not any_preferred_live)
        refresh(fallback_plane_);

    // Nothing new this cycle → let the consumer keep its last sweep (no redundant reprocessing, no blink).
    if (not any_fresh)
        return std::nullopt;

    // 2) Build the sweep from EVERY plane's LATEST cache — not only the planes fresh this poll — so an
    //    out-of-phase plane (helios vs bpearl) never drops out and the merged cloud stays whole. Each plane
    //    is transformed at ITS OWN capture stamp (correct per-plane registration on a moving robot).
    LidarSweep sweep;
    bool any = false;
    auto merge_plane = [&](const Plane& p, std::uint8_t pid)   // NOT 'emit' — that is a Qt macro
    {
        if (p.raw_cache.empty())
            return;
        Eigen::Vector3f origin;
        if (not append_transformed(p.node, target_frame, p.cache_ts, interpolate, p.raw_cache, sweep.points, origin))
            return;   // RT edge not ready yet — skip this plane this poll
        sweep.plane_id.resize(sweep.points.size(), pid);   // tag the points just appended with their plane
        sweep.origin   = origin;
        sweep.stamp_ms = std::max(sweep.stamp_ms, p.cache_ts);
        any = true;
    };

    if (any_preferred_live)
    {
        std::uint8_t pid = 0;
        for (const auto& p : preferred_planes_) merge_plane(p, pid++);
    }
    else
        merge_plane(fallback_plane_, 0);

    if (not any or sweep.points.empty())
        return std::nullopt;
    return sweep;
}

}  // namespace rc::media
