/*
 *    Copyright (C) 2026 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "sensor_media_publisher.h"

#include <algorithm>
#include <cstring>

#include <QString>
#include <QtCore/qdebug.h>

SensorMediaPublisher::~SensorMediaPublisher()
{
    shutdown();
}

namespace
{
// Build a PublisherConfig that all streams share (same domain + QoS depth + data-sharing).
rc::media::PublisherConfig stream_config(std::uint32_t domain_id, int history_depth,
                                         const std::string& topic, bool data_sharing)
{
    rc::media::PublisherConfig pc;
    pc.domain_id     = domain_id;
    pc.topic_name    = topic;
    pc.history_depth = history_depth;
    pc.data_sharing  = data_sharing;
    return pc;
}
}  // namespace

bool SensorMediaPublisher::init(const Config& cfg)
{
    domain_id_     = cfg.domain_id;
    history_depth_ = cfg.history_depth;
    // The QoS the publishers actually use, so the advertised descriptor matches.
    // shared_memory_only stays at the lib default (SHM transport, same-board); only
    // the zero-copy data-sharing layer is exposed to config (cfg.data_sharing).
    const rc::media::PublisherConfig defaults;
    shared_memory_only_ = defaults.shared_memory_only;
    data_sharing_       = cfg.data_sharing;

    QString summary;
    // "ready" = every stream we were ASKED to create came up, and at least one was.
    // (An empty image_streams — e.g. zed disabled — is NOT a failure as long as some
    // other requested stream, lidar/imu/rgb360, initialized.)
    bool all_ok     = true;
    bool any_stream = false;

    for (const auto& spec : cfg.image_streams)
    {
        any_stream = true;
        auto [it, inserted] = images_.try_emplace(spec.key);
        auto& st     = it->second;
        st.key       = spec.key;
        st.topic     = spec.topic;
        st.stream_id = spec.stream_id;
        if (not st.pub.init(stream_config(cfg.domain_id, cfg.history_depth, spec.topic, cfg.data_sharing)))
        {
            all_ok = false;
            qWarning() << "[Media] image publisher init FAILED" << QString::fromStdString(spec.key)
                       << "topic=" << QString::fromStdString(spec.topic);
        }
        summary += " " + QString::fromStdString(spec.key) + "=" + QString::fromStdString(spec.topic);
    }

    if (cfg.lidar_stream)
    {
        any_stream = true;
        lidar_.emplace();
        lidar_->key       = cfg.lidar_stream->key;
        lidar_->topic     = cfg.lidar_stream->topic;
        lidar_->stream_id = cfg.lidar_stream->stream_id;
        if (not lidar_->pub.init(stream_config(cfg.domain_id, cfg.history_depth, cfg.lidar_stream->topic, cfg.data_sharing)))
        {
            all_ok = false;
            qWarning() << "[Media] lidar publisher init FAILED topic="
                       << QString::fromStdString(cfg.lidar_stream->topic);
            lidar_.reset();                          // ⇒ lidar_ready()==false, publish no-ops
        }
        else
            summary += " " + QString::fromStdString(cfg.lidar_stream->key)
                     + "=" + QString::fromStdString(cfg.lidar_stream->topic);
    }

    if (cfg.imu_stream)
    {
        any_stream = true;
        imu_.emplace();
        imu_->key       = cfg.imu_stream->key;
        imu_->topic     = cfg.imu_stream->topic;
        imu_->stream_id = cfg.imu_stream->stream_id;
        if (not imu_->pub.init(stream_config(cfg.domain_id, cfg.history_depth, cfg.imu_stream->topic, cfg.data_sharing)))
        {
            all_ok = false;
            qWarning() << "[Media] imu publisher init FAILED topic="
                       << QString::fromStdString(cfg.imu_stream->topic);
            imu_.reset();
        }
        else
            summary += " " + QString::fromStdString(cfg.imu_stream->key)
                     + "=" + QString::fromStdString(cfg.imu_stream->topic);
    }

    if (cfg.image360_stream)
    {
        any_stream = true;
        image360_.emplace();
        image360_->key       = cfg.image360_stream->key;
        image360_->topic     = cfg.image360_stream->topic;
        image360_->stream_id = cfg.image360_stream->stream_id;
        if (not image360_->pub.init(stream_config(cfg.domain_id, cfg.history_depth, cfg.image360_stream->topic, cfg.data_sharing)))
        {
            all_ok = false;
            qWarning() << "[Media] image360 publisher init FAILED topic="
                       << QString::fromStdString(cfg.image360_stream->topic);
            image360_.reset();                       // ⇒ image360_ready()==false, publish no-ops
        }
        else
            summary += " " + QString::fromStdString(cfg.image360_stream->key)
                     + "=" + QString::fromStdString(cfg.image360_stream->topic);
    }

    ready_ = all_ok and any_stream;
    const auto now = std::chrono::steady_clock::now();
    image_report_at_ = image360_report_at_ = lidar_report_at_ = imu_report_at_ = now;

    if (ready_)
        qInfo() << "[Media] publishers ready domain=" << domain_id_
                << "streams:" << summary
                << "data_sharing=" << data_sharing_active();
    else if (not any_stream)
        qWarning() << "[Media] no media streams configured — all sensor gates off; nothing published on the media plane";
    else
        qWarning() << "[Media] one or more media publishers FAILED to init (streams up:" << summary << ")";
    return ready_;
}

bool SensorMediaPublisher::data_sharing_active() const noexcept
{
    if (images_.empty())
        return false;
    for (const auto& [key, st] : images_)
        if (not st.pub.data_sharing_active())
            return false;
    if (lidar_ and not lidar_->pub.data_sharing_active())
        return false;
    if (imu_ and not imu_->pub.data_sharing_active())
        return false;
    return true;
}

rc::media::MediaDescriptor SensorMediaPublisher::make_descriptor(
    const std::vector<std::string>& keys) const
{
    // Empty `keys` ⇒ advertise EVERY configured stream (back-compat single-node
    // bundle). Otherwise emit only the requested keys, so each sensor node carries
    // just its own stream(s): zed→rgb/depth, lidar3D→lidar, imu→imu.
    const bool all = keys.empty();
    auto wanted = [&](const std::string& k)
    { return all or std::find(keys.begin(), keys.end(), k) != keys.end(); };

    rc::media::MediaDescriptor desc;
    desc.domain_id          = domain_id_;
    desc.history_depth      = history_depth_;
    desc.shared_memory_only = shared_memory_only_;
    desc.data_sharing       = data_sharing_;
    desc.ready              = ready_;             // false ⇒ consumers should wait
    desc.type_name          = "ImageFrame";       // global default (image-only consumers)
    desc.type_tag           = rc::media::IMAGE_FRAME_TYPE_TAG;
    for (const auto& [key, st] : images_)
        if (wanted(key))
        {
            desc.streams[key]      = st.topic;
            desc.stream_types[key] = "ImageFrame";
        }
    if (lidar_ and wanted(lidar_->key))
    {
        desc.streams[lidar_->key]      = lidar_->topic;
        desc.stream_types[lidar_->key] = "LidarFrame";
    }
    if (imu_ and wanted(imu_->key))
    {
        desc.streams[imu_->key]      = imu_->topic;
        desc.stream_types[imu_->key] = "ImuFrame";
        // Same treatment as the 360 stream below, and for the same reason: this is a per-node
        // advertise on the "imu" node carrying only the IMU, so leaving the top-level default in
        // place published a descriptor that read
        //     "type_name":"ImageFrame","type_tag":"ImageFrame.v2" ... "imu_type":"ImuFrame"
        // — self-contradictory, and wrong in the field a consumer's zero-copy schema guard reads.
        // It happens not to break subscriber_config(), which uses only the per-stream topic, but a
        // descriptor whose job is to be authoritative should not need a reader to know which half of
        // it to believe.
        const bool only_imu = (images_.empty() ||
            std::none_of(images_.begin(), images_.end(),
                         [&](const auto& kv){ return wanted(kv.first); }))
            and not (lidar_ and wanted(lidar_->key))
            and not (image360_ and wanted(image360_->key));
        if (only_imu)
        {
            desc.type_name = "ImuFrame";
            desc.type_tag  = rc::media::IMU_FRAME_TYPE_TAG;
        }
    }
    if (image360_ and wanted(image360_->key))
    {
        desc.streams[image360_->key]      = image360_->topic;
        desc.stream_types[image360_->key] = "Image360Frame";
        // When THIS descriptor carries only the 360 panorama (per-node advertise
        // on the ricoh node), stamp the top-level tag/type as Image360Frame so a
        // consumer's zero-copy schema guard matches. Mixed bundles keep the
        // ImageFrame default and rely on per-stream stream_types.
        const bool only360 = images_.empty() ||
            std::none_of(images_.begin(), images_.end(),
                         [&](const auto& kv){ return wanted(kv.first); });
        if (only360)
        {
            desc.type_name = "Image360Frame";
            desc.type_tag  = rc::media::IMAGE360_FRAME_TYPE_TAG;
        }
    }
    return desc;
}

std::string SensorMediaPublisher::descriptor_json(const std::vector<std::string>& keys) const
{
    return make_descriptor(keys).to_json();
}

bool SensorMediaPublisher::publish_image(const std::string& key, const ImageFrameView& view)
{
    if (not ready_)
        return false;
    auto it = images_.find(key);
    if (it == images_.end())
        return false;
    auto& st = it->second;
    if (view.data == nullptr or view.nbytes == 0 or view.nbytes > rc::media::MAX_IMAGE_BYTES)
        return false;

    auto* s = st.pub.loan();
    if (s == nullptr)
    {
        ++st.stats.loan_fail;
        return false;
    }
    s->stream_id(st.stream_id);
    s->frame_id(st.frame_id++);
    s->stamp_ms(view.stamp_ms);
    s->width(view.width);
    s->height(view.height);
    s->step(view.step);
    s->format(view.format);
    s->size(static_cast<std::uint32_t>(view.nbytes));
    std::memcpy(s->data().data(), view.data, view.nbytes);

    if (st.pub.publish(s))
    {
        ++st.stats.ok;
        st.stats.bytes += view.nbytes;
        st.cum_bytes.fetch_add(view.nbytes, std::memory_order_relaxed);
        return true;
    }
    ++st.stats.publish_fail;
    return false;
}

bool SensorMediaPublisher::publish_image360(const ImageFrameView& view)
{
    if (not image360_)
        return false;
    auto& st = *image360_;
    if (view.data == nullptr or view.nbytes == 0 or view.nbytes > rc::media::MAX_IMAGE360_BYTES)
        return false;

    auto* s = st.pub.loan();
    if (s == nullptr)
    {
        ++st.stats.loan_fail;
        return false;
    }
    s->stream_id(st.stream_id);
    s->frame_id(st.frame_id++);
    s->stamp_ms(view.stamp_ms);
    s->width(view.width);
    s->height(view.height);
    s->step(view.step);
    s->format(view.format);
    s->size(static_cast<std::uint32_t>(view.nbytes));
    std::memcpy(s->data().data(), view.data, view.nbytes);

    if (st.pub.publish(s))
    {
        ++st.stats.ok;
        st.stats.bytes += view.nbytes;
        st.cum_bytes.fetch_add(view.nbytes, std::memory_order_relaxed);
        return true;
    }
    ++st.stats.publish_fail;
    return false;
}

bool SensorMediaPublisher::publish_lidar(const LidarFrameView& view)
{
    if (not lidar_)
        return false;
    auto& st = *lidar_;
    if (view.points == nullptr or view.count == 0 or view.stride == 0)
        return false;
    const std::size_t nfloats = static_cast<std::size_t>(view.count) * view.stride;
    if (nfloats > static_cast<std::size_t>(rc::media::MAX_LIDAR_POINTS) * 3u)   // array bound
        return false;

    auto* s = st.pub.loan();
    if (s == nullptr)
    {
        ++st.stats.loan_fail;
        return false;
    }
    s->stream_id(st.stream_id);
    s->frame_id(st.frame_id++);
    s->stamp_ms(view.stamp_ms);
    s->count(view.count);
    s->format(view.format);
    s->stride(view.stride);
    std::memcpy(s->points().data(), view.points, nfloats * sizeof(float));

    if (st.pub.publish(s))
    {
        ++st.stats.ok;
        st.stats.bytes += nfloats * sizeof(float);
        st.cum_bytes.fetch_add(nfloats * sizeof(float), std::memory_order_relaxed);
        return true;
    }
    ++st.stats.publish_fail;
    return false;
}

bool SensorMediaPublisher::publish_imu(const ImuFrameView& view)
{
    if (not imu_)
        return false;
    auto& st = *imu_;

    auto* s = st.pub.loan();
    if (s == nullptr)
    {
        ++st.stats.loan_fail;
        return false;
    }
    s->stream_id(st.stream_id);
    s->frame_id(st.frame_id++);
    s->stamp_ms(view.stamp_ms);
    s->acc_x(view.acc[0]);   s->acc_y(view.acc[1]);   s->acc_z(view.acc[2]);
    s->gyro_x(view.gyro[0]); s->gyro_y(view.gyro[1]); s->gyro_z(view.gyro[2]);
    s->mag_x(view.mag[0]);   s->mag_y(view.mag[1]);   s->mag_z(view.mag[2]);
    s->roll(view.rpy[0]);    s->pitch(view.rpy[1]);   s->yaw(view.rpy[2]);
    s->temperature(view.temperature);
    s->sim_stamp_ms(view.sim_stamp_ms);
    s->gyro_var(view.gyro_var);

    if (st.pub.publish(s))
    {
        ++st.stats.ok;
        st.stats.bytes += sizeof(rc::media::ImuFrame);
        st.cum_bytes.fetch_add(sizeof(rc::media::ImuFrame), std::memory_order_relaxed);
        return true;
    }
    ++st.stats.publish_fail;
    return false;
}

void SensorMediaPublisher::report_one(const std::string& key, Stats& s)
{
    // Routine "ok=" heartbeat dropped; warn only when publishing actually fails.
    if (s.loan_fail > 0 or s.publish_fail > 0)
        qWarning() << "[Media] stats" << QString::fromStdString(key)
                   << "ok=" << s.ok
                   << "MB=" << (static_cast<double>(s.bytes) / 1e6)
                   << "loan_fail=" << s.loan_fail
                   << "publish_fail=" << s.publish_fail;
    s.reset();
}

double SensorMediaPublisher::current_bps(const std::vector<std::string>& keys)
{
    const auto now = std::chrono::steady_clock::now();
    const bool all = keys.empty();
    auto wanted = [&](const std::string& k)
    { return all or std::find(keys.begin(), keys.end(), k) != keys.end(); };

    // Sample one stream: (cum_bytes - prev) / dt. The first sample seeds prev and returns 0.
    auto sample = [&](auto& st) -> double
    {
        const std::uint64_t cur = st.cum_bytes.load(std::memory_order_relaxed);
        double bps = 0.0;
        if (st.bps_prev_t.time_since_epoch().count() != 0)
        {
            const double dt = std::chrono::duration<double>(now - st.bps_prev_t).count();
            if (dt > 0.0) bps = static_cast<double>(cur - st.bps_prev_bytes) / dt;
        }
        st.bps_prev_bytes = cur;
        st.bps_prev_t     = now;
        return bps;
    };

    double sum = 0.0;
    for (auto& [key, st] : images_)
        if (wanted(key)) sum += sample(st);
    if (image360_ and wanted(image360_->key)) sum += sample(*image360_);
    if (lidar_ and wanted(lidar_->key))       sum += sample(*lidar_);
    if (imu_ and wanted(imu_->key))           sum += sample(*imu_);
    return sum;
}

void SensorMediaPublisher::maybe_report_stats(StatsGroup group, std::chrono::seconds interval)
{
    const auto now = std::chrono::steady_clock::now();
    auto due = [&](std::chrono::steady_clock::time_point& at) -> bool
    {
        if (now - at < interval)
            return false;
        at = now;
        return true;
    };

    // Periodic transport status so the zero-copy state stays visible in the live log
    // (not just at startup). Called from all four sensor threads, so gate on an atomic
    // CAS — exactly one thread prints per interval. data_sharing_active() reads a flag
    // fixed at init, so it is safe to sample concurrently.
    const std::int64_t now_ns = now.time_since_epoch().count();
    std::int64_t next_ns = status_report_next_ns_.load(std::memory_order_relaxed);
    if (now_ns >= next_ns and
        status_report_next_ns_.compare_exchange_strong(
            next_ns, now_ns + std::chrono::nanoseconds(interval).count(),
            std::memory_order_relaxed))
        qInfo() << "[Media] transport domain=" << domain_id_
                << "data_sharing=" << data_sharing_active()
                << "(1=zero-copy SHM loans, 0=SHM-transport memcpy)";

    switch (group)
    {
        case StatsGroup::Image:
            if (due(image_report_at_))
                for (auto& [key, st] : images_)
                    report_one(key, st.stats);
            break;
        case StatsGroup::Image360:
            if (image360_ and due(image360_report_at_))
                report_one(image360_->key, image360_->stats);
            break;
        case StatsGroup::Lidar:
            if (lidar_ and due(lidar_report_at_))
                report_one(lidar_->key, lidar_->stats);
            break;
        case StatsGroup::Imu:
            if (imu_ and due(imu_report_at_))
                report_one(imu_->key, imu_->stats);
            break;
    }
}

void SensorMediaPublisher::shutdown()
{
    ready_ = false;
    images_.clear();   // ~MediaPublisher tears down each DDS endpoint
    image360_.reset();
    lidar_.reset();
    imu_.reset();
}
