/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see lidar_ingestor.h.
 */

#include "lidar_ingestor.h"

#include <algorithm>
#include <cmath>
#include <print>

#include <QDateTime>
#include <QString>
#include <QtCore/qdebug.h>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

LidarIngestor::LidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, rc::RoomConcept& room_concept,
                             const rc::RoomConfig& params)
    : G_(std::move(graph)), room_concept_(&room_concept), params_(&params)
{
    if (!params_->LIDAR_USE_MEDIA)
    {
        qWarning() << "[Lidar] LIDAR_USE_MEDIA=false and the DSR-graph path was removed — no LiDAR source";
        return;
    }

    // Prefer the self-describing descriptor on "zed" (domain/topic authored by the producer);
    // fall back to the configured domain/topic.
    std::uint32_t domain = static_cast<std::uint32_t>(params_->MEDIA_DOMAIN_ID);
    std::string   topic  = params_->MEDIA_LIDAR_TOPIC;
    if (G_)
        if (auto desc = rc::media::descriptor_from_graph(*G_, "zed"); desc.has_value())
            if (auto sub = desc->subscriber_config("lidar"); sub.has_value())
            {
                domain = sub->domain_id;
                topic  = sub->topic_name;
            }

    auto sub = std::make_unique<rc::media::LidarSubscriber>();
    rc::media::SubscriberConfig scfg;
    scfg.domain_id     = domain;
    scfg.topic_name    = topic;
    scfg.history_depth = 8;
    if (sub->init(scfg))
    {
        lidar_sub_ = std::move(sub);
        qInfo() << "[Lidar] media-plane source up domain=" << domain
                << "topic=" << QString::fromStdString(topic);
    }
    else
        qWarning() << "[Lidar] media-plane subscriber init FAILED — no LiDAR source";
}

LidarIngestor::~LidarIngestor()
{
    lidar_sub_.reset();
}

bool LidarIngestor::pump()
{
    if (!lidar_sub_)
        return false;

    const float min_h_m = params_->LIDAR_HIGH_MIN_HEIGHT;
    bool ingested = false;

    // Drain all pending frames (non-blocking); ingest_scan dedups by timestamp, so only the newest
    // scan actually reaches the buffer + localizer.
    const int got = lidar_sub_->poll([this, min_h_m, &ingested](const rc::media::LidarFrame& f, std::int64_t)
    {
        const std::uint32_t stride = f.stride() ? f.stride() : 3u;
        const auto& pts = f.points();
        std::vector<Eigen::Vector3f> points_high;
        points_high.reserve(f.count());
        for (std::uint32_t i = 0; i < f.count(); ++i)
        {
            const std::size_t base = static_cast<std::size_t>(i) * stride;
            if (base + 2 >= pts.size())
                break;
            const float x = pts[base], y = pts[base + 1], z = pts[base + 2];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                continue;
            if (z > min_h_m)
                points_high.emplace_back(x, y, z);
        }
        ingest_scan(std::move(points_high), static_cast<std::int64_t>(f.stamp_ms()));
        ingested = true;
    });
    fresh_frames_ += static_cast<std::uint64_t>(std::max(0, got));

    const auto now = QDateTime::currentMSecsSinceEpoch();
    if (last_src_report_ms_ == 0 || now - last_src_report_ms_ >= 5000)
    {
        std::println("[LidarSrc] 5s media fresh={} served={}", fresh_frames_, served_);
        fresh_frames_ = served_ = 0;
        last_src_report_ms_ = now;
    }
    return ingested;
}

void LidarIngestor::ingest_scan(std::vector<Eigen::Vector3f>&& points_high, std::int64_t src_ts)
{
    if (src_ts <= last_ingested_lidar_ts_)
        return;
    last_ingested_lidar_ts_ = src_ts;

    const std::uint64_t ts = static_cast<std::uint64_t>(std::max<std::int64_t>(0, src_ts));
    high_lidar_buffer_.put<0>(rc::LidarData{std::move(points_high), src_ts}, ts);
    if (room_concept_)
        room_concept_->notify_new_lidar(static_cast<std::int64_t>(ts));
    ++served_;
}

}  // namespace rc
