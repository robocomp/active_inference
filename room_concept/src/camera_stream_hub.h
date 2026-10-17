/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

/*
 *  camera_stream_hub.h — ONE media-plane RGB subscriber per camera node, per process.
 *
 *  WHY
 *  ---
 *  room_concept used to open TWO independent readers on every camera stream: CameraIngestor (the RGB
 *  image-edge / calibration channels) and CameraVisualizer (the viewer's camera windows). Each frame —
 *  the ricoh panorama is ~5.5 MB at ~30 Hz, 160 MB/s — was transferred and reassembled twice. A hub
 *  owns the single subscriber plus ONE drain thread, and fans each loaned frame out to its consumers.
 *
 *  CONTRACT
 *  --------
 *  - acquire(G, node) returns the process-wide hub for that node (created on first use, destroyed
 *    when the last shared_ptr goes — its destructor joins the drain thread).
 *  - add_consumer(name, cb): cb runs ON THE HUB THREAD, inside the poll, with a VIEW into the loaned
 *    SHM sample that is valid ONLY for the duration of the call. A consumer that keeps anything must
 *    DEEP-COPY it there (CLAUDE.md: never share a mutable buffer across threads). Consumers rate-limit
 *    themselves inside the callback (a skipped frame costs nothing: the loan is still released).
 *  - remove_consumer(id) blocks until no dispatch to that consumer is in flight, so after it returns
 *    the callback is never called again and its captures may die.
 *  - The subscriber EXISTS ONLY WHILE AT LEAST ONE CONSUMER IS REGISTERED. With none, the drain thread
 *    closes it (on the thread that created it) and sleeps; the next add_consumer re-discovers it. This
 *    is what lets a viewer-only stream (the zed, when no calibration channel uses it) be subscribed
 *    exactly while its window is shown.
 *  - Callbacks must not call add_consumer / remove_consumer (they run under the consumer lock).
 *
 *  Discovery is descriptor-driven, as before: the stream key ("rgb" → ImageFrame, "rgb360" →
 *  Image360Frame, different DDS types) is read from the node's media descriptor, never assumed.
 *  The hub NEVER touches data_sharing settings — it calls the same shared factories.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace DSR { class DSRGraph; }
namespace rc::media { class MediaSubscriber; class Image360Subscriber; }

namespace rc
{

/// A loaned frame, valid only inside the consumer callback.
struct CameraFrameView
{
    const std::uint8_t* data = nullptr;
    std::size_t   size     = 0;
    int           width    = 0;
    int           height   = 0;
    /// The frame's OWN format field. For is360 == true this is the IMG360_FORMAT_* numbering, for
    /// is360 == false the plain FORMAT_* numbering — they differ, so check is360 first.
    std::uint32_t format   = 0;
    bool          is360    = false;
    std::int64_t  stamp_ms = 0;
};

class CameraStreamHub
{
public:
    using Callback = std::function<void(const CameraFrameView&)>;

    /// The process-wide hub for `node` (thread-safe). Never null when `graph` is non-null.
    static std::shared_ptr<CameraStreamHub> acquire(std::shared_ptr<DSR::DSRGraph> graph,
                                                    const std::string& node);

    ~CameraStreamHub();
    CameraStreamHub(const CameraStreamHub&) = delete;
    CameraStreamHub& operator=(const CameraStreamHub&) = delete;

    /// Register a consumer; returns its id. Wakes the drain thread, which brings the subscriber up.
    int  add_consumer(std::string name, Callback cb);
    /// Unregister. Blocks until any in-flight dispatch to it has returned. Unknown id = no-op.
    void remove_consumer(int id);

    /// True while a subscriber is live (for "waiting for frames" vs "not initialized" UI text).
    [[nodiscard]] bool subscribed() const noexcept { return subscribed_.load(std::memory_order_acquire); }
    [[nodiscard]] const std::string& node() const noexcept { return node_; }

private:
    CameraStreamHub(std::shared_ptr<DSR::DSRGraph> graph, std::string node);   // via acquire() only
    void loop();
    bool try_discover();
    int  poll_and_dispatch();
    void close_subscriber(const char* why);
    std::string consumer_names_locked() const;

    std::shared_ptr<DSR::DSRGraph> G_;
    std::string node_;

    // Owned EXCLUSIVELY by the drain thread (created, polled and destroyed there).
    std::unique_ptr<rc::media::MediaSubscriber>    sub_;
    std::unique_ptr<rc::media::Image360Subscriber> sub360_;
    std::chrono::steady_clock::time_point last_discovery_{};
    bool discovery_reason_logged_ = false;

    struct Consumer { int id; std::string name; Callback cb; };
    mutable std::mutex     cons_mtx_;          // guards consumers_; HELD during dispatch
    std::vector<Consumer>  consumers_;
    int                    next_id_ = 1;
    std::atomic<int>       n_consumers_{0};

    std::atomic<bool>       running_{true};
    std::atomic<bool>       subscribed_{false};
    std::mutex              wake_mtx_;
    std::condition_variable wake_cv_;
    std::thread             thread_;
};

}  // namespace rc
