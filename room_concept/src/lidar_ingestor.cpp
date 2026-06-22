/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see lidar_ingestor.h.
 */

#include "lidar_ingestor.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <QDateTime>
#include <QString>
#include <QTimer>
#include <QtCore/qdebug.h>

#include "../../common/media_transport/media_transport.h"

namespace rc
{

LidarIngestor::LidarIngestor() = default;

LidarIngestor::~LidarIngestor()
{
    stop();
}

void LidarIngestor::init(const Config& cfg, Deps deps)
{
    cfg_          = cfg;
    G_            = deps.graph;
    room_concept_ = deps.room_concept;
    qt_owner_     = deps.qt_owner;

    // Preferred source: the zero-copy media plane (robot_concept's LidarFrame stream).
    // The DSR graph laser_* path remains as a watchdog fallback if this fails to come up.
    if (cfg_.use_media)
    {
        // Prefer the self-describing descriptor on "zed" (domain/topic authored by the
        // producer); fall back to the configured domain/topic.
        std::uint32_t domain = cfg_.media_domain_id;
        std::string   topic  = cfg_.media_lidar_topic;
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
            qWarning() << "[Lidar] media-plane subscriber init FAILED — falling back to DSR graph laser_*";
    }
}

void LidarIngestor::start()
{
    if (lidar_ingest_running_.exchange(true))
        return;
    lidar_ingest_thread_ = std::thread(&LidarIngestor::lidar_ingest_loop, this);
}

void LidarIngestor::stop()
{
    lidar_ingest_running_.store(false);
    lidar_ingest_cv_.notify_all();
    if (lidar_ingest_thread_.joinable())
        lidar_ingest_thread_.join();
    lidar_sub_.reset();   // tear down DDS endpoints after the producer thread is stopped
}

///////////////////////////////////////////////////////////////////////////////
void LidarIngestor::on_laser_node_signal()
{
    schedule_lidar_copy_to_pending(std::nullopt, "update_node_signal", true);
}

void LidarIngestor::on_laser_attrs_signal(std::uint64_t id)
{
    if (!G_)
        return;
    if (const auto node_opt = G_->get_node(id); node_opt.has_value() && node_opt->type() == "laser")
        schedule_lidar_copy_to_pending(id, "update_node_attr_signal", true);
}

void LidarIngestor::schedule_lidar_copy_to_pending(std::optional<std::uint64_t> node_id, const char* origin, bool count_signal_event)
{
    if (count_signal_event)
    {
        lidar_signal_events_.fetch_add(1, std::memory_order_relaxed);
        lidar_last_signal_ms_.store(QDateTime::currentMSecsSinceEpoch(), std::memory_order_relaxed);
    }

    // Coalesce bursts of graph signals into a single queued pull from the Qt thread.
    if (lidar_copy_queued_.exchange(true, std::memory_order_acq_rel))
        return;

    QTimer::singleShot(0, qt_owner_, [this, node_id, origin]()
    {
        lidar_copy_queued_.store(false, std::memory_order_release);
        copy_latest_lidar_to_pending(node_id, origin, false);
    });
}

bool LidarIngestor::copy_latest_lidar_to_pending(std::optional<std::uint64_t> node_id, const char* origin, bool count_signal_event)
{
    (void)origin;

    if (count_signal_event)
    {
        lidar_signal_events_.fetch_add(1, std::memory_order_relaxed);
        lidar_last_signal_ms_.store(QDateTime::currentMSecsSinceEpoch(), std::memory_order_relaxed);
    }

    if (!G_)
        return false;

    std::optional<DSR::Node> node_opt;
    if (node_id.has_value())
        node_opt = G_->get_node(node_id.value());
    if (!node_opt.has_value())
    {
        node_opt = G_->get_node(cfg_.lidar_name);
        if (!node_opt.has_value())
        {
            node_opt = G_->get_node("lidar3d");
            if (!node_opt.has_value())
                node_opt = G_->get_node("lidar3D");
        }
    }
    if (!node_opt.has_value())
        return false;

    const auto &lidar3D = node_opt.value();
    const auto lx = G_->get_attrib_by_name<laser_X_att>(lidar3D);
    const auto ly = G_->get_attrib_by_name<laser_Y_att>(lidar3D);
    const auto lz = G_->get_attrib_by_name<laser_Z_att>(lidar3D);
    const auto laser_ts = G_->get_attrib_by_name<laser_timestamp_att>(lidar3D);
    if (!lx.has_value() || !ly.has_value() || !lz.has_value() || !laser_ts.has_value())
        return false;

    const auto source_ts = static_cast<std::int64_t>(laser_ts.value());

    {
        std::unique_lock<std::mutex> lk(lidar_ingest_mutex_, std::try_to_lock);
        if (!lk.owns_lock())
        {
            // Avoid blocking DDS/update callback threads if ingest thread is inside
            // a long operation while holding the mutex.
            lidar_copy_mutex_busy_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (source_ts <= last_ingested_lidar_ts_)
            return false;
        if (pending_lidar_scan_.has_data && source_ts <= pending_lidar_scan_.source_ts)
            return false;

        pending_lidar_scan_.xs = lx.value().get();
        pending_lidar_scan_.ys = ly.value().get();
        pending_lidar_scan_.zs = lz.value().get();
        pending_lidar_scan_.source_ts = source_ts;
        pending_lidar_scan_.has_data = true;
        lidar_copied_frames_.fetch_add(1, std::memory_order_relaxed);
    }

    lidar_ingest_dirty_.store(true, std::memory_order_release);
    lidar_ingest_cv_.notify_one();
    return true;
}

void LidarIngestor::lidar_ingest_loop()
{
    const float min_h_m = cfg_.high_min_height;
    while (lidar_ingest_running_.load(std::memory_order_acquire))
    {
        lidar_ingest_loop_beats_.fetch_add(1, std::memory_order_relaxed);
        lidar_last_ingest_loop_beat_ms_.store(QDateTime::currentMSecsSinceEpoch(), std::memory_order_relaxed);

        if (lidar_sub_)
        {
            // Primary source: drain the media plane (blocks up to 50 ms for a frame).
            lidar_sub_->wait_and_poll([this, min_h_m](const rc::media::LidarFrame& f, std::int64_t)
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
                lidar_ingest_wakeups_.fetch_add(1, std::memory_order_relaxed);
                ingest_scan(std::move(points_high), static_cast<std::int64_t>(f.stamp_ms()));
            }, 50);
            // Also drain any graph-fed pending scan (watchdog fallback).
            ingest_latest_lidar();
            continue;
        }

        // Fallback source: DSR graph laser_* via the CV-driven pending scan.
        {
            std::unique_lock lk(lidar_ingest_mutex_);
            const bool woke_for_work = lidar_ingest_cv_.wait_for(lk, std::chrono::milliseconds(50), [this]
            {
                return !lidar_ingest_running_.load(std::memory_order_acquire)
                    || lidar_ingest_dirty_.load(std::memory_order_acquire);
            });
            if (woke_for_work)
                lidar_ingest_wakeups_.fetch_add(1, std::memory_order_relaxed);
            lidar_ingest_dirty_.store(false, std::memory_order_release);
        }
        if (!lidar_ingest_running_.load(std::memory_order_acquire))
            break;
        ingest_latest_lidar();
    }
}

void LidarIngestor::ingest_latest_lidar()
{
    std::vector<float> xs, ys, zs;
    std::int64_t src_ts = 0;

    {
        std::scoped_lock lk(lidar_ingest_mutex_);
        if (!pending_lidar_scan_.has_data)
            return;

        xs = std::move(pending_lidar_scan_.xs);
        ys = std::move(pending_lidar_scan_.ys);
        zs = std::move(pending_lidar_scan_.zs);
        src_ts = pending_lidar_scan_.source_ts;
        pending_lidar_scan_.has_data = false;

        if (src_ts <= last_ingested_lidar_ts_)
            return;
    }

    const std::size_t npts = std::min({xs.size(), ys.size(), zs.size()});
    std::vector<Eigen::Vector3f> points_high;
    points_high.reserve(npts);
    const float min_h_m = cfg_.high_min_height;
    for (std::size_t i = 0; i < npts; ++i)
    {
        const float x_raw = xs[i], y_raw = ys[i], z_raw = zs[i];
        if (!std::isfinite(x_raw) || !std::isfinite(y_raw) || !std::isfinite(z_raw))
            continue;
        if (z_raw > min_h_m)
            points_high.emplace_back(x_raw, y_raw, z_raw);
    }

    ingest_scan(std::move(points_high), src_ts);
}

// Shared tail for both sources: dedup by timestamp, fill the buffer, notify the
// localizer. Runs only on the single ingest thread, so last_ingested_lidar_ts_ is
// not contended here (other threads only read it under the mutex).
void LidarIngestor::ingest_scan(std::vector<Eigen::Vector3f>&& points_high, std::int64_t src_ts)
{
    if (src_ts <= last_ingested_lidar_ts_)
        return;
    last_ingested_lidar_ts_ = src_ts;

    const std::uint64_t ts = static_cast<std::uint64_t>(std::max<std::int64_t>(0, src_ts));
    high_lidar_buffer_.put<0>(rc::LidarData{std::move(points_high), src_ts}, ts);
    const auto notify_start_ms = QDateTime::currentMSecsSinceEpoch();
    lidar_notify_inflight_.store(true, std::memory_order_relaxed);
    lidar_notify_inflight_since_ms_.store(notify_start_ms, std::memory_order_relaxed);
    if (room_concept_)
        room_concept_->notify_new_lidar(static_cast<std::int64_t>(ts));
    lidar_notify_inflight_.store(false, std::memory_order_relaxed);
    lidar_notify_inflight_since_ms_.store(0, std::memory_order_relaxed);
    const auto notify_dt_ms = QDateTime::currentMSecsSinceEpoch() - notify_start_ms;
    if (notify_dt_ms > 200)
        qWarning() << "[Lidar][ingest] notify_new_lidar slow" << "dt_ms=" << notify_dt_ms;

    {
        std::scoped_lock lk(lidar_ingest_mutex_);
        lidar_recent_source_ts_.push_back(src_ts);
        while (lidar_recent_source_ts_.size() > 5)
            lidar_recent_source_ts_.pop_front();
    }

    lidar_ingested_frames_.fetch_add(1, std::memory_order_relaxed);
    lidar_last_ingest_ms_.store(QDateTime::currentMSecsSinceEpoch(), std::memory_order_relaxed);
}

///////////////////////////////////////////////////////////////////////////////
void LidarIngestor::tick_health(std::int64_t now_ms, bool on_gui_thread)
{
    const auto last_signal_ms = lidar_last_signal_ms_.load(std::memory_order_relaxed);
    const auto last_ingest_ms = lidar_last_ingest_ms_.load(std::memory_order_relaxed);
    const auto signal_age_ms = (last_signal_ms > 0) ? (now_ms - last_signal_ms) : -1;
    const auto ingest_age_ms = (last_ingest_ms > 0) ? (now_ms - last_ingest_ms) : -1;

    if (last_lidar_health_log_ms_ == 0 || now_ms - last_lidar_health_log_ms_ >= 3000)
    {
        last_lidar_health_log_ms_ = now_ms;
        const auto beat_age = (lidar_last_ingest_loop_beat_ms_.load(std::memory_order_relaxed) > 0)
                              ? (now_ms - lidar_last_ingest_loop_beat_ms_.load(std::memory_order_relaxed)) : -1;
        if (ingest_age_ms < 0 || ingest_age_ms > 2000)
        {
            qWarning() << "[Lidar][health] stale ingest in Operating"
                       << "signal_events=" << lidar_signal_events_.load(std::memory_order_relaxed)
                       << "copied_frames=" << lidar_copied_frames_.load(std::memory_order_relaxed)
                       << "ingested_frames=" << lidar_ingested_frames_.load(std::memory_order_relaxed)
                       << "watchdog_pulls=" << lidar_watchdog_pulls_.load(std::memory_order_relaxed)
                       << "ingest_wakeups=" << lidar_ingest_wakeups_.load(std::memory_order_relaxed)
                       << "copy_mutex_busy=" << lidar_copy_mutex_busy_.load(std::memory_order_relaxed)
                       << "ingest_loop_beats=" << lidar_ingest_loop_beats_.load(std::memory_order_relaxed)
                       << "ingest_loop_beat_age_ms=" << beat_age
                       << "last_signal_age_ms=" << signal_age_ms
                       << "last_ingest_age_ms=" << ingest_age_ms
                       << "dirty=" << lidar_ingest_dirty_.load(std::memory_order_relaxed);
        }
        else
        {
            qInfo() << "[Lidar][health] ok"
                    << "signal_events=" << lidar_signal_events_.load(std::memory_order_relaxed)
                    << "copied_frames=" << lidar_copied_frames_.load(std::memory_order_relaxed)
                    << "ingested_frames=" << lidar_ingested_frames_.load(std::memory_order_relaxed)
                    << "watchdog_pulls=" << lidar_watchdog_pulls_.load(std::memory_order_relaxed)
                    << "ingest_wakeups=" << lidar_ingest_wakeups_.load(std::memory_order_relaxed)
                    << "copy_mutex_busy=" << lidar_copy_mutex_busy_.load(std::memory_order_relaxed)
                    << "ingest_loop_beats=" << lidar_ingest_loop_beats_.load(std::memory_order_relaxed)
                    << "ingest_loop_beat_age_ms=" << beat_age
                    << "last_signal_age_ms=" << signal_age_ms
                    << "last_ingest_age_ms=" << ingest_age_ms;
        }
    }

    if (lidar_notify_inflight_.load(std::memory_order_relaxed))
    {
        const auto since_ms = lidar_notify_inflight_since_ms_.load(std::memory_order_relaxed);
        if (since_ms > 0 && now_ms - since_ms > 2000)
            qWarning() << "[Lidar][health] ingest thread appears blocked in notify_new_lidar"
                       << "blocked_ms=" << (now_ms - since_ms);
    }

    if (on_gui_thread &&
        (signal_age_ms < 0 || signal_age_ms > 2000) &&
        (last_lidar_watchdog_pull_ms_ == 0 || now_ms - last_lidar_watchdog_pull_ms_ >= 1000))
    {
        last_lidar_watchdog_pull_ms_ = now_ms;
        if (copy_latest_lidar_to_pending(std::nullopt, "watchdog", false))
        {
            lidar_watchdog_pulls_.fetch_add(1, std::memory_order_relaxed);
            qWarning() << "[Lidar][watchdog] recovered pending scan from graph poll"
                       << "signal_age_ms=" << signal_age_ms
                       << "ingest_age_ms=" << ingest_age_ms;
        }
    }
}

}  // namespace rc
