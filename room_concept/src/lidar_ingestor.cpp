/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see lidar_ingestor.h.
 */

#include "lidar_ingestor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <optional>
#include <print>
#include <utility>

#include <QDateTime>
#include <QString>
#include <QtCore/qdebug.h>

#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "../../common/media_transport/lidar_plane_reader.h"
#include "../../common/media_transport/media_transport.h"   // descriptor_from_graph (graph-only probe)

namespace
{
// Startup geometry-check z-histogram (body frame, metres): floor spike near 0, ceiling spike near
// room_height - base. Range covers below the base up to above any real ceiling; 2 cm bins.
constexpr float GEOM_Z_LO  = -0.5f;
constexpr float GEOM_Z_HI  =  3.2f;
constexpr float GEOM_BIN   =  0.02f;
constexpr int   GEOM_NBINS = static_cast<int>((GEOM_Z_HI - GEOM_Z_LO) / GEOM_BIN) + 1;
// Horizontal-radius binning for the ceiling perimeter-vs-interior test (body frame, metres).
constexpr float GEOM_R_MAX = 10.0f;
constexpr float GEOM_R_BIN = 0.25f;
constexpr int   GEOM_NR    = static_cast<int>(GEOM_R_MAX / GEOM_R_BIN) + 1;
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
    // ("body"). The
    // subscribers themselves are brought up lazily inside reader_->poll() (throttled), once each
    // node + descriptor exists — nothing touches DDS here. inner_eigen_ backs the RT transform.
    inner_eigen_ = G_ ? G_->get_inner_eigen_api() : nullptr;
    reader_ = std::make_unique<rc::media::LidarPlaneReader>(
        G_, inner_eigen_.get(),
        std::vector<std::string>{params_->LIDAR_HELIOS_NAME}, "lidar");
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
    geom_rz_hist_.assign(static_cast<std::size_t>(GEOM_NBINS) * GEOM_NR, 0);
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

    // One shared reader call: newest "helios" sweep, transformed DEVICE->robot
    // base ("body") via the DSR RT tree. interpolate=false — helios → body only crosses the
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
                    G_, inner_eigen_.get(), std::vector<std::string>{"bpearl"}, "lidar");
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
                std::vector<int>().swap(geom_rz_hist_);
            }
        }

        const float min_h_m = params_->LIDAR_HIGH_MIN_HEIGHT;
        const float max_h_m = high_max_z_;                 // upper bound excludes the ceiling plane

        std::vector<Eigen::Vector3f> points_high;
        points_high.reserve(sweep->points.size());
        for (const auto& p : sweep->points)
            if (p.z() > min_h_m and p.z() < max_h_m)
                points_high.emplace_back(p);

        // ── WHAT IS ACTUALLY IN THE BAND ────────────────────────────────────────────────────────
        // The band limits are a property of the SENSOR'S FAN as much as of the room: they were set
        // for a helios hanging the other way up, whose beams reached this height range at long range.
        // Invert the fan and the same two numbers select a completely different population — near
        // walls at steep elevation instead of far walls at shallow — and the localiser is fitting a
        // polygon to whatever survives. That change is invisible in every number downstream, which
        // report the FIT and not what was fitted, so it is reported here at the point of selection.
        {
            band_sweeps_++; band_in_ += static_cast<long>(points_high.size());
            band_total_ += static_cast<long>(sweep->points.size());
            for (const auto& p : points_high)
            {
                band_z_sum_ += p.z();
                band_r_sum_ += std::hypot(p.x(), p.y());
                band_z_lo_ = std::min(band_z_lo_, p.z());
                band_z_hi_ = std::max(band_z_hi_, p.z());
                // Where in the band, in sixths. A CEILING is a spike in the top bin: walls spread
                // roughly evenly over the band, a horizontal surface piles into one slice. The mean
                // alone cannot separate those — it is the same 2.28 either way.
                const int b = std::clamp(static_cast<int>(6.f * (p.z() - min_h_m)
                                                          / std::max(1e-3f, max_h_m - min_h_m)), 0, 5);
                band_zhist_[b]++;
            }
            const auto tnow = QDateTime::currentMSecsSinceEpoch();
            if (band_report_ms_ == 0) band_report_ms_ = tnow;
            else if (tnow - band_report_ms_ > 5000)
            {
                band_report_ms_ = tnow;
                const double n = std::max(1L, band_in_);
                const double hn = std::max(1L, band_in_);
                std::println("[band] {:.1f}-{:.2f} m keeps {}/{} pts/sweep ({:.1f}%) | z mean {:.2f} "
                             "span {:.2f}..{:.2f} | mean ground range {:.2f} m | z sixths "
                             "{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}/{:.0f}% (top bin = ceiling if it spikes)",
                             min_h_m, max_h_m,
                             band_in_ / std::max(1L, band_sweeps_), band_total_ / std::max(1L, band_sweeps_),
                             100.0 * band_in_ / std::max(1L, band_total_),
                             band_z_sum_ / n, band_z_lo_, band_z_hi_, band_r_sum_ / n,
                             100.0 * band_zhist_[0] / hn, 100.0 * band_zhist_[1] / hn,
                             100.0 * band_zhist_[2] / hn, 100.0 * band_zhist_[3] / hn,
                             100.0 * band_zhist_[4] / hn, 100.0 * band_zhist_[5] / hn);
                band_sweeps_ = band_in_ = band_total_ = 0;
                band_z_sum_ = band_r_sum_ = 0.0;
                band_z_lo_ = 1e9f; band_z_hi_ = -1e9f; band_zhist_ = {};
            }
        }
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
    // Wall clock, NOT the source stamp: the state machine asks "did anything arrive recently?", and a
    // producer republishing an old stamp (or a clock skew between hosts) must not read as liveness.
    last_frame_wall_ms_.store(QDateTime::currentMSecsSinceEpoch(), std::memory_order_release);
}

bool LidarIngestor::stream_descriptor_available(std::string* detail) const
{
    if (not params_->LIDAR_USE_MEDIA)
    {
        if (detail) *detail = "LidarUseMedia=false — no LiDAR source configured";
        return false;
    }
    if (not G_)
    {
        if (detail) *detail = "no DSR graph";
        return false;
    }
    // A plane is usable when its sensor node carries a media descriptor advertising a "lidar" stream.
    // Same lookup make_lidar_subscriber_from_graph() does, minus the DDS init — so a true answer here
    // means the subscriber WILL come up, and a false one names exactly what the producer hasn't
    // published yet. Mirrors the reader's preference order.
    for (const auto& node : {params_->LIDAR_HELIOS_NAME})
    {
        const auto desc = rc::media::descriptor_from_graph(*G_, node);
        if (desc.has_value() and desc->subscriber_config("lidar").has_value())
        {
            if (detail) *detail = node;
            return true;
        }
    }
    if (detail)
        *detail = std::format("no 'lidar' media descriptor on node '{}' "
                              "(is robot_concept up and advertising?)",
                              params_->LIDAR_HELIOS_NAME);
    return false;
}

std::int64_t LidarIngestor::ms_since_last_frame() const noexcept
{
    const std::int64_t last = last_frame_wall_ms_.load(std::memory_order_acquire);
    if (last == 0)
        return -1;   // nothing has ever arrived
    return QDateTime::currentMSecsSinceEpoch() - last;
}

void LidarIngestor::accumulate_geometry_sample(const std::vector<Eigen::Vector3f>& pts)
{
    for (const auto& p : pts)
    {
        const float r = p.head<2>().norm();
        if (r < 0.8f)                        // skip self-hits hugging the robot base
            continue;
        const int b = static_cast<int>(std::floor((p.z() - GEOM_Z_LO) / GEOM_BIN));
        if (b >= 0 and b < static_cast<int>(geom_hist_.size()))
        {
            ++geom_hist_[b];
            const int rb = static_cast<int>(r / GEOM_R_BIN);   // horizontal-radius bin
            if (rb >= 0 and rb < GEOM_NR)
                ++geom_rz_hist_[static_cast<std::size_t>(b) * GEOM_NR + rb];
        }
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

    // Median horizontal radius of the returns in a z-window, from the joint (z,r) histogram. This is the
    // spatial discriminant: a ceiling PLANE fills the interior, so its returns land CLOSER than the walls;
    // an upright-LiDAR wall-top locus sits at the wall range. Compares like-for-like (both are helios).
    // Any quantile of the radius distribution in a z slice, not just the median: the ceiling test
    // below needs the LOW end of the peak's radii as well as its middle.
    const auto quantile_radius_in = [this](float zlo, float zhi, float q) -> std::pair<float, int>
    {
        std::array<int, GEOM_NR> racc{};
        int total = 0;
        for (int b = 0; b < GEOM_NBINS; ++b)
        {
            const float z = GEOM_Z_LO + (b + 0.5f) * GEOM_BIN;
            if (z < zlo or z > zhi) continue;
            const int* row = &geom_rz_hist_[static_cast<std::size_t>(b) * GEOM_NR];
            for (int rb = 0; rb < GEOM_NR; ++rb) { racc[rb] += row[rb]; total += row[rb]; }
        }
        if (total == 0) return {0.f, 0};
        const int target = std::max(1, static_cast<int>(q * static_cast<float>(total)));
        int cum = 0;
        for (int rb = 0; rb < GEOM_NR; ++rb)
        {
            cum += racc[rb];
            if (cum >= target) return {(rb + 0.5f) * GEOM_R_BIN, total};
        }
        return {(GEOM_NR - 0.5f) * GEOM_R_BIN, total};
    };
    const auto median_radius_in = [this](float zlo, float zhi) -> std::pair<float, int>
    {
        std::array<int, GEOM_NR> racc{};
        int total = 0;
        for (int b = 0; b < GEOM_NBINS; ++b)
        {
            const float z = GEOM_Z_LO + (b + 0.5f) * GEOM_BIN;
            if (z < zlo or z > zhi) continue;
            const int* row = &geom_rz_hist_[static_cast<std::size_t>(b) * GEOM_NR];
            for (int rb = 0; rb < GEOM_NR; ++rb) { racc[rb] += row[rb]; total += row[rb]; }
        }
        if (total == 0) return {0.f, 0};
        int cum = 0;
        for (int rb = 0; rb < GEOM_NR; ++rb)
        {
            cum += racc[rb];
            if (cum * 2 >= total) return {(rb + 0.5f) * GEOM_R_BIN, total};
        }
        return {(GEOM_NR - 0.5f) * GEOM_R_BIN, total};
    };

    // Perimeter-vs-interior test: a real ceiling's returns are meaningfully CLOSER than the walls. Reference
    // = the UPPER-WALL band just below the peak (same walls, one notch lower — above furniture, apples-to-
    // apples). A wall-top ring has r_peak ≈ r_ref (ratio≈1); a ceiling has r_peak < r_ref. The 0.75 boundary
    // is scale-free (relative range), not an absolute distance cut.
    const float ref_lo = std::max(0.9f, ceil_z - 0.70f);
    const float ref_hi = ceil_z - 0.25f;                       // strictly below the peak window
    const auto [r_ref, ref_n]   = median_radius_in(ref_lo, ref_hi);
    const auto [r_peak, peak_n] = median_radius_in(ceil_z - 0.10f, ceil_z + 0.10f);
    // The INNER edge of the peak's returns, which is what makes the two hypotheses separable.
    const auto [r_in, in_n]     = quantile_radius_in(ceil_z - 0.10f, ceil_z + 0.10f, 0.05f);
    const bool spatial_ok  = (ceil_cnt >= 400) and (ref_hi > ref_lo)
                             and ref_n >= 200 and peak_n >= 200 and r_ref > 0.f;

    // ── CEILING or WALL-TOP: predict r_peak under each, and take the closer ────────────────────
    // ★ The old test was `r_peak < 0.75 * r_ref` — the z-peak's returns being "interior". That
    //   ratio encodes a ceiling seen as a FILLED DISC, whose area-weighted median radius is
    //   0.707*R. It is wrong for any lidar that cannot see straight up, and on 2026-08-29 it
    //   misclassified a 51450-point ceiling as wall-top by 9 cm, leaving the ceiling plane inside
    //   the band that feeds a 2-D WALL SDF (residual 0.164 m RMS, 0% early exit, 180% CPU).
    //   With the helios inverted, a fan reaching +54.5 deg from a 1.075 m mount sees the 3.01 m
    //   ceiling as an ANNULUS from 1.38 m outward, not a disc — median sqrt((1.38^2+4.12^2)/2)
    //   = 3.07 m, against the disc model's 2.91 and the measured 3.38.
    // ★ NO RATIO CAN FIX THAT, because the expected radius depends on the mount height and the
    //   fan's maximum elevation. So predict it instead, from what is OBSERVABLE in the peak itself:
    //     ceiling  -> returns fill an annulus [r_in, r_ref];  area-weighted median = sqrt((r_in^2+r_ref^2)/2)
    //                 (r_in = 0 recovers the filled disc, so the old case is contained, not replaced)
    //     wall-top -> returns lie ON the boundary, at the same radii as the wall band below: r_ref
    //   and choose whichever prediction the measurement is closer to. That is a likelihood ratio
    //   between two shapes, not a cutoff, and it needs no sensor model, no mount height and no
    //   tuning — the geometry it would need is already imprinted on the returns.
    const float pred_ceiling = std::sqrt(0.5f * (r_in * r_in + r_ref * r_ref));
    const float pred_wall    = r_ref;
    const bool  is_ceiling   = spatial_ok and in_n >= 200
                               and std::abs(r_peak - pred_ceiling) < std::abs(r_peak - pred_wall);

    if (ceil_cnt >= 400 and is_ceiling)
    {
        high_max_z_ = std::clamp(ceil_z - params_->LIDAR_CEILING_MARGIN,
                                 params_->LIDAR_HIGH_MIN_HEIGHT + 0.1f, GEOM_Z_HI);
        std::println("[CeilingCheck] CEILING at body z = {:.2f} m ({} pts): r_peak={:.2f} m matches the "
                     "annulus prediction {:.2f} m (inner edge {:.2f} m) better than the wall {:.2f} m "
                     "-> high band capped at {:.2f} m.",
                     ceil_z, ceil_cnt, r_peak, pred_ceiling, r_in, pred_wall, high_max_z_);
    }
    else if (ceil_cnt >= 400 and spatial_ok)   // strong z-peak but at the wall range → wall-top, keep it
    {
        high_max_z_ = cfg_max;
        std::println("[CeilingCheck] z-peak at {:.2f} m ({} pts) is WALL-TOP: r_peak={:.2f} m is closer to "
                     "the wall {:.2f} m than to the annulus prediction {:.2f} m (inner edge {:.2f} m) -> "
                     "high band kept at config max {:.2f} m (top-wall points retained for the SDF).",
                     ceil_z, ceil_cnt, r_peak, pred_wall, pred_ceiling, r_in, cfg_max);
    }
    else if (ceil_cnt >= 400)   // z-peak but spatial test inconclusive (sparse ref/peak) → cut, to be safe
    {
        high_max_z_ = std::clamp(ceil_z - params_->LIDAR_CEILING_MARGIN,
                                 params_->LIDAR_HIGH_MIN_HEIGHT + 0.1f, GEOM_Z_HI);
        std::println("[CeilingCheck] z-peak at {:.2f} m ({} pts) but spatial test inconclusive "
                     "(ref_n={}, peak_n={}) -> conservatively capped at {:.2f} m.",
                     ceil_z, ceil_cnt, ref_n, peak_n, high_max_z_);
    }
    else
    {
        high_max_z_ = cfg_max;
        std::println("[CeilingCheck] no z-density peak in [{:.2f}, {:.2f}] m (best {} pts) -> high band max = "
                     "config cutoff {:.2f} m (NOT an assumed ceiling; just where the band stops).",
                     clo, chi, ceil_cnt, cfg_max);
    }

    // Drop the one-shot bpearl probe reader + histogram (diagnostic; frees the extra subscriber).
    geom_bpearl_reader_.reset();
    std::vector<int>().swap(geom_hist_bpearl_);
}

}  // namespace rc
