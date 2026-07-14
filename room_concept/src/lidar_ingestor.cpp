/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see lidar_ingestor.h.
 */

#include "lidar_ingestor.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <print>
#include <utility>

#include <QDateTime>
#include <QString>
#include <QtCore/qdebug.h>

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "../../common/media_transport/lidar_plane_reader.h"

namespace
{
// Startup geometry-check z-histogram (body frame, metres): floor spike near 0, ceiling spike near
// room_height - base. Range covers below the base up to above any real ceiling; 2 cm bins.
constexpr float GEOM_Z_LO  = -0.5f;
constexpr float GEOM_Z_HI  =  3.2f;
constexpr float GEOM_BIN   =  0.02f;
constexpr int   GEOM_NBINS = static_cast<int>((GEOM_Z_HI - GEOM_Z_LO) / GEOM_BIN) + 1;
}  // namespace

namespace rc
{

LidarIngestor::LidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, rc::RoomConcept& room_concept,
                             const rc::RoomConfig& params)
    : G_(std::move(graph)), room_concept_(&room_concept), params_(&params)
{
    high_max_z_ = params_->LIDAR_HIGH_MAX_HEIGHT;   // until the startup check refines it from the ceiling
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
    stop();          // join the ingest thread BEFORE dropping the reader (thread uses it)
    reader_.reset();
}

void LidarIngestor::start()
{
    if (running_.exchange(true))
        return;      // idempotent: already running
    // Re-arm the one-shot startup geometry check for this Operating session (runs before the thread).
    geom_check_done_ = false;
    geom_sweeps_     = 0;
    geom_bpearl_sweeps_ = 0;
    geom_hist_.assign(GEOM_NBINS, 0);
    geom_hist_bpearl_.assign(GEOM_NBINS, 0);
    high_max_z_      = params_ ? params_->LIDAR_HIGH_MAX_HEIGHT : 2.0f;
    thread_ = std::thread(&LidarIngestor::ingest_loop, this);
}

void LidarIngestor::stop()
{
    if (!running_.exchange(false))
        return;      // not running
    wake_cv_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

void LidarIngestor::ingest_loop()
{
    // Tight ingest loop on a dedicated thread: react to a fresh scan with ~0-2 ms latency instead of
    // waiting for the next ~16 ms compute() tick. pump() reads the DSR graph (reader discovery +
    // inner_eigen device→robot transform) — safe here because the thread is started only once the agent
    // is Operating (post graph-join), mirroring the controller's control_thread_ which likewise runs
    // inner_eigen off the main thread. The graph WRITE (RT publish) still happens on the main thread via
    // RoomConcept's on_result_ready → QueuedConnection.
    while (running_.load(std::memory_order_acquire))
    {
        const bool got = pump();
        if (got)
            continue;                        // drain again immediately in case more is queued
        // No fresh frame: brief idle wait (woken instantly by stop()). 2 ms ⇒ ≤2 ms extra latency and a
        // ~500 Hz idle poll of cheap non-blocking DDS takes — negligible CPU (lidar source is ~20 Hz).
        std::unique_lock<std::mutex> lk(wake_mutex_);
        wake_cv_.wait_for(lk, std::chrono::milliseconds(2),
                          [this] { return !running_.load(std::memory_order_acquire); });
    }
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
        // One-shot startup geometry check runs off the FULL sweep (needs the floor + ceiling returns
        // the high-band filter below strips). It warns on a floor/geometry mismatch and lowers
        // high_max_z_ to just under the detected ceiling.
        if (params_->LIDAR_STARTUP_GEOMETRY_CHECK and not geom_check_done_)
        {
            // Accumulate bpearl's own low-z histogram (the head-on floor datum) alongside helios. bpearl's
            // reader comes up later than helios, so gate the trigger on its warm-up (fallback 3× so a
            // missing bpearl can't stall). Source self-filter already removed the robot base/base-plate,
            // so a small near-cut is enough; source floor cut is off, so the real floor is present.
            if (not geom_bpearl_reader_ and G_)
                geom_bpearl_reader_ = std::make_unique<rc::media::LidarPlaneReader>(
                    G_, inner_eigen_.get(), std::vector<std::string>{"bpearl"}, params_->LIDAR_NAME, "lidar");
            if (geom_bpearl_reader_)
                if (const auto bp = geom_bpearl_reader_->poll(params_->LIDAR_ROBOT_FRAME, /*interpolate=*/false);
                    bp.has_value() and not bp->points.empty())
                {
                    ++geom_bpearl_sweeps_;
                    for (const auto& p : bp->points)
                    {
                        if (p.head<2>().norm() < 0.10f) continue;
                        const int b = static_cast<int>(std::floor((p.z() - GEOM_Z_LO) / GEOM_BIN));
                        if (b >= 0 and b < static_cast<int>(geom_hist_bpearl_.size())) ++geom_hist_bpearl_[b];
                    }
                }
            accumulate_geometry_sample(sweep->points);

            const int need = std::max(1, params_->LIDAR_STARTUP_CHECK_SWEEPS);
            if (geom_sweeps_ >= need and (geom_bpearl_sweeps_ >= need or geom_sweeps_ >= 3 * need))
            {
                run_startup_geometry_check();
                geom_check_done_ = true;
                std::vector<int>().swap(geom_hist_);
            }
        }

        const float min_h_m = params_->LIDAR_HIGH_MIN_HEIGHT;
        const float max_h_m = high_max_z_;                 // upper bound excludes the ceiling plane

        // Compute z min/max of raw points for debug
        float z_raw_min = 1e9f, z_raw_max = -1e9f;
        for (const auto& p : sweep->points) {
            if (p.z() < z_raw_min) z_raw_min = p.z();
            if (p.z() > z_raw_max) z_raw_max = p.z();
        }
        std::println("[LidarSrc] z range: min={:.3f} max={:.3f}  high band=[{:.3f}, {:.3f}]  raw_pts={}  filtered_pts=",
                     z_raw_min, z_raw_max, min_h_m, max_h_m, sweep->points.size());

        std::vector<Eigen::Vector3f> points_high;
        points_high.reserve(sweep->points.size());
        for (const auto& p : sweep->points)
            if (p.z() > min_h_m and p.z() < max_h_m)
                points_high.emplace_back(p);
        std::println("[LidarSrc] high band pts={}", points_high.size());
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

void LidarIngestor::accumulate_geometry_sample(const std::vector<Eigen::Vector3f>& pts)
{
    for (const auto& p : pts)
    {
        if (p.head<2>().norm() < 0.8f)       // skip self-hits hugging the robot base
            continue;
        const int b = static_cast<int>(std::floor((p.z() - GEOM_Z_LO) / GEOM_BIN));
        if (b >= 0 and b < static_cast<int>(geom_hist_.size()))
            ++geom_hist_[b];
    }
    ++geom_sweeps_;   // trigger decision lives in pump() (it also gates on bpearl warm-up)
}

void LidarIngestor::run_startup_geometry_check()
{
    // Mode (peak) of the accumulated z-histogram within a [lo,hi] window.
    const auto peak_in = [](const std::vector<int>& hist, float lo, float hi) -> std::pair<float, int>
    {
        int best_bin = -1, best_cnt = 0;
        for (int b = 0; b < static_cast<int>(hist.size()); ++b)
        {
            const float z = GEOM_Z_LO + (b + 0.5f) * GEOM_BIN;
            if (z < lo or z > hi) continue;
            if (hist[b] > best_cnt) { best_cnt = hist[b]; best_bin = b; }
        }
        if (best_bin < 0) return {0.f, 0};
        return {GEOM_Z_LO + (best_bin + 0.5f) * GEOM_BIN, best_cnt};
    };

    // RT geometry: robot-frame origin height above the root/floor (root<-Shadow).
    std::optional<double> base_z;
    if (inner_eigen_)
        if (auto t = inner_eigen_->get_translation_vector("root", params_->LIDAR_ROBOT_FRAME); t.has_value())
            base_z = t->z();

    // (a) Floor-plane CALIBRATION from BPEARL (the downward dome — head-on, dense, near). helios is an
    // upright 360 lidar whose floor is all grazing/at-range (reads several cm high) → reference only, NOT
    // the datum. With the source floor cut removed, the real floor reaches us and must land at world z ≈ 0.
    const auto [he_floor_z, he_cnt] = peak_in(geom_hist_,        GEOM_Z_LO, 0.5f);   // helios (grazing, ref)
    const auto [bp_floor_z, bp_cnt] = peak_in(geom_hist_bpearl_, GEOM_Z_LO, 0.5f);   // bpearl (head-on, datum)
    const int floor_cnt = he_cnt;   // used to scale the ceiling threshold below
    if (base_z.has_value())
    {
        const float bp_root = bp_floor_z + static_cast<float>(*base_z);
        const float he_root = he_floor_z + static_cast<float>(*base_z);
        if (bp_cnt > 0)
        {
            if (std::abs(bp_root) > params_->LIDAR_FLOOR_TOLERANCE)
                std::println("[FloorCheck] WARNING: frame='{}' base_z={:.0f} mm | bpearl floor at world z = {:.0f} mm "
                             "(expected ~0, {} pts, {} sweeps) -> floor datum off by {:.0f} mm — check root height / "
                             "sensor mounts in shadow.json (helios ref {:.0f} mm is grazing, not the datum)",
                             params_->LIDAR_ROBOT_FRAME, base_z.value_or(0.0) * 1000.0, bp_root * 1000.f,
                             bp_cnt, geom_bpearl_sweeps_, bp_root * 1000.f, he_root * 1000.f);
            else
                std::println("[FloorCheck] OK: frame='{}' base_z={:.0f} mm | bpearl floor at world z = {:.0f} mm "
                             "(within {:.0f} mm, {} pts) | helios ref {:.0f} mm (grazing, high by design)",
                             params_->LIDAR_ROBOT_FRAME, base_z.value_or(0.0) * 1000.0, bp_root * 1000.f,
                             params_->LIDAR_FLOOR_TOLERANCE * 1000.f, bp_cnt, he_root * 1000.f);
        }
        else
            std::println("[FloorCheck] bpearl floor unavailable ({} sweeps) — helios floor {:.0f} mm is GRAZING "
                         "(upright lidar, floor >3 m out), not a reliable datum; skipping mount verdict.",
                         geom_bpearl_sweeps_, he_root * 1000.f);
    }
    else
        std::println("[FloorCheck] floor verification skipped (helios_pts={}, bpearl_pts={}, root RT ready={})",
                     he_cnt, bp_cnt, base_z.has_value());

    // (b) Ceiling plane -> cap the high band at (ceiling - margin) so only upper-wall points survive.
    // Search a window centred on the expected ceiling (room_height above the floor) to avoid picking a
    // dense lower-wall/furniture band; fall back to a fixed upper window when the base height is unknown.
    float clo = 1.8f, chi = GEOM_Z_HI;
    if (base_z.has_value())
    {
        const float exp_ceil = params_->room_height - static_cast<float>(*base_z);
        clo = exp_ceil - 0.5f;
        chi = exp_ceil + 0.4f;
    }
    clo = std::max(clo, params_->LIDAR_HIGH_MIN_HEIGHT + 0.2f);
    const auto [ceil_z, ceil_cnt] = peak_in(geom_hist_, clo, chi);
    const float cfg_max = params_->LIDAR_HIGH_MAX_HEIGHT;
    // Absolute threshold: the search window is already location-restricted to the expected ceiling, so a
    // substantial peak there IS the ceiling. (The old `floor_cnt/5` broke once the source floor cut was
    // removed — the un-filtered floor is now huge, making that ratio reject a real ceiling; floor_cnt is
    // helios's grazing floor here anyway.)
    if (ceil_cnt >= 400)
    {
        high_max_z_ = std::clamp(ceil_z - params_->LIDAR_CEILING_MARGIN,
                                 params_->LIDAR_HIGH_MIN_HEIGHT + 0.1f, GEOM_Z_HI);
        std::println("[FloorCheck] ceiling detected at body z = {:.2f} m ({} pts) -> high band capped at "
                     "{:.2f} m (upper walls only; config max was {:.2f} m)",
                     ceil_z, ceil_cnt, high_max_z_, cfg_max);
    }
    else
    {
        high_max_z_ = cfg_max;
        std::println("[FloorCheck] no clear ceiling (best high peak {} pts, floor_ref {}) -> high band max kept at {:.2f} m",
                     ceil_cnt, floor_cnt, cfg_max);
    }

    // Drop the one-shot bpearl probe reader + histogram (diagnostic; frees the extra subscriber).
    geom_bpearl_reader_.reset();
    std::vector<int>().swap(geom_hist_bpearl_);
}

}  // namespace rc
