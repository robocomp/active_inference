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

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

LidarIngestor::LidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, rc::RoomConcept& room_concept,
                             const rc::RoomConfig& params)
    : G_(std::move(graph)), room_concept_(&room_concept), params_(&params)
{
    if (!params_->LIDAR_USE_MEDIA)
        qWarning() << "[Lidar] LIDAR_USE_MEDIA=false and the DSR-graph path was removed — no LiDAR source";
    // The DDS subscriber is created lazily in pump()/ensure_subscriber() once the
    // "lidar3D" node + media descriptor exist; nothing to bring up here.
}

LidarIngestor::~LidarIngestor()
{
    lidar_sub_.reset();
}

bool LidarIngestor::ensure_subscriber()
{
    if (lidar_sub_)
        return true;
    if (!params_->LIDAR_USE_MEDIA || !G_)
        return false;

    // Self-throttle the discovery attempts (compute thread, every cycle).
    const auto now = std::chrono::steady_clock::now();
    if (now - last_init_attempt_ < std::chrono::seconds(1))
        return false;
    last_init_attempt_ = now;

    // Prefer the per-device high LiDAR plane ("helios"): lidar3d_dds publishes it in the DEVICE
    // frame (metres), so points must be transformed device->robot via the DSR RT tree here. When
    // that plane is not live (robot_concept is bridging the fused scan instead) fall back to the
    // legacy "lidar3D" plane, which already arrives in the robot frame (no transform needed).
    // The shared descriptor-driven factory verifies the node + descriptor and returns nullptr
    // until they exist, so this simply retries once per second.
    if ((lidar_sub_ = rc::media::make_lidar_subscriber_from_graph(*G_, params_->LIDAR_HELIOS_NAME, "lidar")))
    {
        source_node_     = params_->LIDAR_HELIOS_NAME;
        needs_transform_ = true;
        inner_eigen_     = G_->get_inner_eigen_api();
    }
    else if ((lidar_sub_ = rc::media::make_lidar_subscriber_from_graph(*G_, params_->LIDAR_NAME, "lidar")))
    {
        source_node_     = params_->LIDAR_NAME;
        needs_transform_ = false;
    }
    if (lidar_sub_)
        std::println("[Lidar] subscribed to '{}' plane (device-frame transform={})",
                     source_node_, needs_transform_);
    return lidar_sub_ != nullptr;
}

bool LidarIngestor::pump()
{
    if (!ensure_subscriber())
        return false;

    const float min_h_m = params_->LIDAR_HIGH_MIN_HEIGHT;
    bool ingested = false;

    // Drain all pending frames (non-blocking); ingest_scan dedups by timestamp, so only the newest
    // scan actually reaches the buffer + localizer.
    const int got = lidar_sub_->poll([this, min_h_m, &ingested](const rc::media::LidarFrame& f, std::int64_t)
    {
        // Points from the "helios" plane are in the DEVICE frame — fetch the device->robot RT once
        // per frame (identity when consuming the already-robot-frame "lidar3D" plane). The height
        // filter below is meaningful only AFTER this transform (z becomes height above the robot base).
        Eigen::Affine3d T_robot_dev = Eigen::Affine3d::Identity();
        if (needs_transform_)
        {
            auto m = inner_eigen_
                ? inner_eigen_->get_transformation_matrix(params_->LIDAR_ROBOT_FRAME, source_node_,
                                                          0, "RT", DSR::RT_API::TimeQuery::Nearest)
                : std::nullopt;
            if (!m.has_value())
            {
                // No RT edge yet (graph still joining): drop this frame rather than feed
                // device-frame points as if they were robot-frame. Throttled warning.
                const auto now_ms = QDateTime::currentMSecsSinceEpoch();
                if (now_ms - last_transform_warn_ms_ >= 5000)
                {
                    qWarning() << "[Lidar] no" << QString::fromStdString(params_->LIDAR_ROBOT_FRAME)
                               << "<-" << QString::fromStdString(source_node_)
                               << "RT edge — dropping device-frame scan until it appears";
                    last_transform_warn_ms_ = now_ms;
                }
                return;
            }
            T_robot_dev = m.value();
        }

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
            const Eigen::Vector3f p = needs_transform_
                ? (T_robot_dev * Eigen::Vector3d(x, y, z)).cast<float>()
                : Eigen::Vector3f(x, y, z);
            if (p.z() > min_h_m)
                points_high.emplace_back(p);
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
