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

#include "../../common/media_transport/lidar_plane_reader.h"

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
    // Shared media-plane reader: the high "helios" plane (DEVICE frame) transformed to the robot base
    // ("body"), with the fused "lidar3D" plane as fallback while robot_concept is bridging. The
    // subscribers themselves are brought up lazily inside reader_->poll() (throttled), once each
    // node + descriptor exists — nothing touches DDS here. inner_eigen_ backs the RT transform.
    inner_eigen_ = G_ ? G_->get_inner_eigen_api() : nullptr;
    reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_.get(),
        std::vector<std::string>{params_->LIDAR_HELIOS_NAME},
        params_->LIDAR_NAME, "lidar");
}

LidarIngestor::~LidarIngestor()
{
    reader_.reset();
}

bool LidarIngestor::pump()
{
    if (!reader_)
        return false;

    // One shared reader call: newest "helios" (or fallback "lidar3D") sweep, transformed DEVICE->robot
    // base ("body") via the DSR RT tree. interpolate=false — helios/lidar3D → body only crosses the
    // static mount edge, so the sweep stamp is irrelevant. The height filter below is meaningful in
    // this robot-base frame (z = height above the base).
    const auto sweep = reader_->poll(params_->LIDAR_ROBOT_FRAME, /*interpolate=*/false);
    if (sweep.has_value())
        ++fresh_frames_;

    bool ingested = false;
    if (sweep.has_value() and not sweep->points.empty())
    {
        const float min_h_m = params_->LIDAR_HIGH_MIN_HEIGHT;
        std::vector<Eigen::Vector3f> points_high;
        points_high.reserve(sweep->points.size());
        for (const auto& p : sweep->points)
            if (p.z() > min_h_m)
                points_high.emplace_back(p);
        ingest_scan(std::move(points_high), sweep->stamp_ms);
        ingested = true;
    }

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
