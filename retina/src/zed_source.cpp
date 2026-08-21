#include "zed_source.h"

#include <chrono>
#include <print>
#include "scene_processor.h"

#include <genericworker.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include <string>
#include <utility>

namespace rc
{

ZedSource::ZedSource(SceneProcessor* scene, std::shared_ptr<DSR::DSRGraph> graph)
    : scene_(scene), graph_(std::move(graph))
{
    if (graph_)
        inner_eigen_ = graph_->get_inner_eigen_api();
}

ZedSource::~ZedSource() = default;

std::optional<PerceptionFrame> ZedSource::operator()()
{
    if (!scene_)
        return std::nullopt;

    // ★★TEST THE STAMP BEFORE PAYING FOR THE FRAME. The order used to be assemble-then-compare, so
    // every poll that found nothing new still deep-copied a full RGB+depth pair and threw it away.
    // MEASURED 2026-08-20: 317 idle polls per 5 s against 94 hits — with 94x19.2 ms of real work, the
    // idle polls were sharing 3195 ms, i.e. **10.1 ms EACH**, and moving ~2 GB per 5 s to no purpose.
    // That is what made the worker blind: a frame landing during one of those copies is superseded by
    // the next before the loop looks again, so 44 of 138 arriving frames were passed over (26.6 Hz in,
    // 17-18 Hz processed) while the stages themselves only needed 19 ms of a 37.5 ms grab period.
    // The gate is one atomic load; the copy now happens only for a frame we are actually going to use.
    if (const std::uint64_t pending = scene_->pending_rgb_stamp();
        pending == 0 or pending == last_stamp_)
    {
        ++probe_polls_idle_;
        return std::nullopt;
    }

    auto rgbd = scene_->get_rgbd_frame_from_dsr();   // drains the aligned RGBD (worker thread now)
    if (!rgbd.has_value())
        return std::nullopt;

    const std::uint64_t stamp = scene_->get_frame_timestamp_ms();
    if (stamp == 0 || stamp == last_stamp_)
    {
        ++probe_polls_idle_;
        return std::nullopt;   // no new grab since last time
    }
    // ★WHERE THE MISSING THIRD GOES. The aligner is LATEST-WINS: this returns the newest committed grab,
    // so any frame that arrived while the previous one was being processed is passed over. That is by
    // design ("processed < feed means skipped, not lost"), but the AMOUNT was never measured, and it is
    // larger than the timing budget explains — 26.6 Hz arriving (rgb=133/depth=134 per 5 s, matching
    // stamps) against 17.6 Hz processed, with stages at 20 ms inside a 37.5 ms camera period.
    // A jump of more than one grab means frames were passed over; counting them here separates "the
    // worker was busy" from "the buffer never committed them" without guessing from rates.
    if (last_stamp_ != 0 and stamp > last_stamp_)
    {
        const std::uint64_t jump = stamp - last_stamp_;
        probe_jump_sum_ += jump;
        if (jump > 55) ++probe_skipped_;      // > ~1.5 grabs at 26.6 Hz ⇒ at least one frame passed over
        if (jump > probe_jump_max_) probe_jump_max_ = jump;
    }
    ++probe_hits_;
    {
        const auto now = std::chrono::steady_clock::now();
        if (probe_last_.time_since_epoch().count() == 0) probe_last_ = now;
        else if (std::chrono::duration<double>(now - probe_last_).count() >= 5.0)
        {
            const double dt = std::chrono::duration<double>(now - probe_last_).count();
            std::println("[ZedSource] 5s hits={} ({:.1f} Hz) idle_polls={} skipped_grabs={} "
                         "mean_jump={:.1f}ms max_jump={}ms",
                         probe_hits_, probe_hits_ / dt, probe_polls_idle_, probe_skipped_,
                         probe_hits_ ? static_cast<double>(probe_jump_sum_) / probe_hits_ : 0.0,
                         probe_jump_max_);
            probe_hits_ = probe_polls_idle_ = probe_skipped_ = 0;
            probe_jump_sum_ = 0; probe_jump_max_ = 0;
            probe_last_ = now;
        }
    }
    last_stamp_ = stamp;

    PerceptionFrame pf;
    pf.rgbd   = std::move(*rgbd);   // rgb already an owned clone (get_rgbd_frame_from_dsr)
    pf.stamp  = stamp;
    pf.is_360 = false;

    // room<-zed at the capture stamp WITH forward pose-extrapolation (the same correction the voxel path
    // gets), so masks land at the capture-instant robot pose instead of the ~100 ms-lagged newest RT block.
    // Own inner_eigen instance (ts!=0 room<-robot → no cache; ts==0 static robot->zed).
    if (inner_eigen_ && graph_ && scene_)
    {
        std::string room_name, robot_name;
        if (const auto rooms = graph_->get_nodes_by_type("room"); !rooms.empty())
            room_name = rooms.front().name();
        if (const auto robots = graph_->get_nodes_by_type("robot"); !robots.empty())
            robot_name = robots.front().name();
        if (auto T = scene_->room_T_zed_extrapolated(inner_eigen_.get(), room_name, robot_name, stamp);
            T.has_value())
            pf.room_T_sensor = T.value();
    }
    return pf;
}

} // namespace rc
