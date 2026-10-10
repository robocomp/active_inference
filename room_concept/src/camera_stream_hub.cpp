/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#include <genericworker.h>   // FIRST: DSR's signal emitter is not self-contained

#include "camera_stream_hub.h"
#include "../../common/status_stream/status_sink.h"   // rc::status::println (routed)

#include <pthread.h>

#include <algorithm>
#include <map>
#include <ranges>

#include <QDebug>
#include <QString>

#include <media_transport/media_transport.h>

namespace rc
{

namespace
{
    std::mutex& registry_mtx() { static std::mutex m; return m; }
    std::map<std::string, std::weak_ptr<CameraStreamHub>>& registry()
    { static std::map<std::string, std::weak_ptr<CameraStreamHub>> r; return r; }
}

std::shared_ptr<CameraStreamHub> CameraStreamHub::acquire(std::shared_ptr<DSR::DSRGraph> graph,
                                                          const std::string& node)
{
    if (not graph) return nullptr;
    std::lock_guard<std::mutex> lk(registry_mtx());
    auto& slot = registry()[node];
    if (auto hub = slot.lock()) return hub;
    std::shared_ptr<CameraStreamHub> hub(new CameraStreamHub(std::move(graph), node));
    slot = hub;
    return hub;
}

CameraStreamHub::CameraStreamHub(std::shared_ptr<DSR::DSRGraph> graph, std::string node)
    : G_(std::move(graph)), node_(std::move(node))
{
    thread_ = std::thread([this]
    {
        // Same name the CameraIngestor thread had, so per-thread CPU samples stay comparable.
        const auto tname = ("cam:" + node_).substr(0, 15);
        pthread_setname_np(pthread_self(), tname.c_str());
        loop();
    });
}

CameraStreamHub::~CameraStreamHub()
{
    running_.store(false, std::memory_order_release);
    wake_cv_.notify_all();
    if (thread_.joinable()) thread_.join();      // the thread closes its own subscriber on the way out
}

int CameraStreamHub::add_consumer(std::string name, Callback cb)
{
    int id = 0, n = 0;
    {
        std::lock_guard<std::mutex> lk(cons_mtx_);
        id = next_id_++;
        consumers_.push_back({id, std::move(name), std::move(cb)});
        n = static_cast<int>(consumers_.size());
        n_consumers_.store(n, std::memory_order_release);
        rc::status::println("[media-hub] {}: consumer '{}' attached ({} total: {})",
                            node_, consumers_.back().name, n, consumer_names_locked());
    }
    wake_cv_.notify_all();
    return id;
}

void CameraStreamHub::remove_consumer(int id)
{
    // Taking cons_mtx_ waits out any in-flight dispatch (it is held across the callbacks).
    std::lock_guard<std::mutex> lk(cons_mtx_);
    const auto it = std::ranges::find(consumers_, id, &Consumer::id);
    if (it == consumers_.end()) return;
    const std::string name = it->name;
    consumers_.erase(it);
    n_consumers_.store(static_cast<int>(consumers_.size()), std::memory_order_release);
    rc::status::println("[media-hub] {}: consumer '{}' detached ({} left{}{})", node_, name,
                        consumers_.size(), consumers_.empty() ? "" : ": ", consumer_names_locked());
    wake_cv_.notify_all();
}

std::string CameraStreamHub::consumer_names_locked() const
{
    std::string s;
    for (const auto& c : consumers_) { if (not s.empty()) s += ", "; s += c.name; }
    return s;
}

void CameraStreamHub::close_subscriber(const char* why)
{
    if (not sub_ and not sub360_) return;
    sub_.reset();
    sub360_.reset();
    subscribed_.store(false, std::memory_order_release);
    discovery_reason_logged_ = false;
    rc::status::println("[media-hub] {}: subscriber CLOSED ({})", node_, why);
}

void CameraStreamHub::loop()
{
    // Tight drain while anyone listens (the reader is RELIABLE: an undrained SHM pool makes the
    // producer's loan_sample() fail and freezes every consumer on the plane). With no consumer the
    // subscriber is closed, so there is nothing to drain and the thread just sleeps.
    while (running_.load(std::memory_order_acquire))
    {
        if (n_consumers_.load(std::memory_order_acquire) == 0)
        {
            close_subscriber("no consumers");
            std::unique_lock<std::mutex> lk(wake_mtx_);
            wake_cv_.wait_for(lk, std::chrono::milliseconds(200), [this]
            {
                return not running_.load(std::memory_order_acquire)
                       or n_consumers_.load(std::memory_order_acquire) > 0;
            });
            continue;
        }
        if (not sub_ and not sub360_)
            try_discover();
        else if (poll_and_dispatch() > 0)
            continue;      // drained a frame — try again at once in case more are queued
        std::unique_lock<std::mutex> lk(wake_mtx_);
        wake_cv_.wait_for(lk, std::chrono::milliseconds(5),
                          [this] { return not running_.load(std::memory_order_acquire); });
    }
    close_subscriber("hub destroyed");
}

bool CameraStreamHub::try_discover()
{
    if (sub_ or sub360_ or not G_) return false;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_discovery_ < std::chrono::seconds(1)) return false;   // self-throttle
    last_discovery_ = now;

    // Descriptor-driven (CLAUDE.md): the producer's descriptor names the stream, and "rgb" (ImageFrame)
    // vs "rgb360" (Image360Frame) are different DDS types. Asking the descriptor first also keeps the
    // log clean — the factories print on a miss, and this retries once a second.
    const auto desc = rc::media::descriptor_from_graph(*G_, node_);
    if (not desc.has_value())
    {
        if (not discovery_reason_logged_)
        {
            discovery_reason_logged_ = true;
            qWarning() << "[media-hub]" << QString::fromStdString(node_)
                       << "has no media descriptor yet — retrying every second";
        }
        return false;
    }
    std::string key;
    if (desc->streams.contains("rgb360"))
    {
        sub360_ = rc::media::make_image360_subscriber_from_graph(*G_, node_, "rgb360");
        key = "rgb360";
    }
    else if (desc->streams.contains("rgb"))
    {
        sub_ = rc::media::make_image_subscriber_from_graph(*G_, node_, "rgb");
        key = "rgb";
    }
    else if (not discovery_reason_logged_)
    {
        discovery_reason_logged_ = true;
        std::string keys;
        for (const auto& [k, v] : desc->streams) { if (not keys.empty()) keys += ", "; keys += k; }
        qWarning() << "[media-hub]" << QString::fromStdString(node_)
                   << "advertises no image stream this process can read (has:"
                   << QString::fromStdString(keys) << ")";
        return false;
    }
    const bool up = sub_ != nullptr or sub360_ != nullptr;
    if (up)
    {
        subscribed_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(cons_mtx_);
        rc::status::println("[media-hub] {}: subscriber UP on '{}' — ONE reader shared by: {}",
                            node_, key, consumer_names_locked());
    }
    else if (not discovery_reason_logged_)
    {
        discovery_reason_logged_ = true;
        qWarning() << "[media-hub]" << QString::fromStdString(node_)
                   << "descriptor found and the stream is advertised, but the reader could not be"
                      " created — check the media plane domain and MAX_IMAGE_BYTES";
    }
    return up;
}

int CameraStreamHub::poll_and_dispatch()
{
    // The view is valid only inside the poll callback, so the consumers run THERE, under the consumer
    // lock (which is what makes remove_consumer() a hard barrier).
    const auto dispatch = [this](const CameraFrameView& v)
    {
        std::lock_guard<std::mutex> lk(cons_mtx_);
        for (const auto& c : consumers_) c.cb(v);
    };
    if (sub360_)
        return sub360_->poll([&](const rc::media::Image360Frame& f, std::int64_t)
        {
            dispatch(CameraFrameView{.data = reinterpret_cast<const std::uint8_t*>(f.data().data()),
                                     .size = f.size(),
                                     .width = static_cast<int>(f.width()),
                                     .height = static_cast<int>(f.height()),
                                     .format = static_cast<std::uint32_t>(f.format()),
                                     .is360 = true,
                                     .stamp_ms = static_cast<std::int64_t>(f.stamp_ms())});
        });
    return sub_->poll([&](const rc::media::ImageFrame& f, std::int64_t)
    {
        dispatch(CameraFrameView{.data = reinterpret_cast<const std::uint8_t*>(f.data().data()),
                                 .size = f.size(),
                                 .width = static_cast<int>(f.width()),
                                 .height = static_cast<int>(f.height()),
                                 .format = static_cast<std::uint32_t>(f.format()),
                                 .is360 = false,
                                 .stamp_ms = static_cast<std::int64_t>(f.stamp_ms())});
    });
}

}  // namespace rc
