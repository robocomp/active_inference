#pragma once
/*
 * rt_query_probe — what did my timestamped RT query actually DO?
 *
 * WHY THIS EXISTS. A timestamped transform query that falls outside the edge's history ring returns
 * the end block and looks exactly like a success: same optional, same RTMat, no warning. For an object
 * fitter that is the difference between placing a detection at the pose the camera actually had and
 * placing it at a pose from another instant — which smears geometry under rotation and reads, from the
 * outside, as a calibration or detector fault. cortex reports it now (RT_API::TimeQueryInfo); this is
 * the cheapest possible way for an agent to LISTEN, before deciding whether it needs to act.
 *
 * ★DELIBERATELY NOT A FIX. It changes no behaviour and no query. It answers one question — "does this
 * call site ever clamp, and by how much" — so the decision to migrate a fitter to
 * TimeQuery::Extrapolated is made from its own measured numbers rather than from the assumption that
 * fresher data must always outrun the pose feed. Measured on this fleet, that assumption is FALSE for
 * the camera paths: retina logged zero extrapolation events in two months because room_concept's ring
 * spans ~1.2 s and camera stamps land inside it. The controller's LiDAR path, on the same graph,
 * extrapolates on 35% of moving cycles. Same graph, opposite regimes — so guessing is not available.
 *
 * ★A CSV, NOT A LOG LINE, AND FOR TWO REASONS. Stdout does not reliably survive the launcher, and the
 * useful question is not "does it clamp" but "does it clamp WHILE TURNING": the controller's clamp
 * rate goes 1-3% parked to 35% moving, so a number without a motion column cannot separate the two.
 * Each row therefore carries the robot's own speed over the same window, sampled from the twist the
 * producer publishes rather than differenced here.
 *
 *   usage:  static rc::rtprobe::Probe probe{"bottle_fitter", "etc/rt_query_probe.csv"};
 *           DSR::RT_API::TimeQueryInfo info;
 *           auto T = inner->get_transformation_matrix(dst, src, ts, "RT",
 *                                                     DSR::RT_API::TimeQuery::Interpolated, &info);
 *           probe.note(info, G);      // G optional: without it the motion columns are empty
 *
 * Appends (never truncates) so a run can be compared against the last one; the file is small — one
 * row per 15 s per call site. Atomic counters, because a fitter may run on a worker thread and a torn
 * count is a lie of exactly the kind this exists to detect.
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <locale>
#include <memory>
#include <mutex>
#include <string>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

namespace rc::rtprobe
{
class Probe
{
public:
    explicit Probe(std::string tag, std::string csv_path = "etc/rt_query_probe.csv",
                   std::int64_t period_ms = 15000)
        : tag_(std::move(tag)), csv_path_(std::move(csv_path)), period_ms_(period_ms) {}

    void note(const DSR::RT_API::TimeQueryInfo &i,
              const std::shared_ptr<DSR::DSRGraph> &G = nullptr)
    {
        using O = DSR::RT_API::TimeQueryInfo::Outcome;
        total_.fetch_add(1, std::memory_order_relaxed);
        switch (i.outcome)
        {
            case O::Exact:        exact_.fetch_add(1, std::memory_order_relaxed); break;
            case O::Interpolated: interp_.fetch_add(1, std::memory_order_relaxed); break;
            case O::Extrapolated: extrap_.fetch_add(1, std::memory_order_relaxed); break;
            case O::Clamped:      clamped_.fetch_add(1, std::memory_order_relaxed); break;
            case O::Stale:        stale_.fetch_add(1, std::memory_order_relaxed); break;
        }
        // Worst |gap| in the window: the clamp's MAGNITUDE is what says whether it matters. A 5 ms
        // clamp on a parked robot is noise; 200 ms mid-turn is centimetres of misplacement.
        const auto g = i.gap_ms < 0 ? -i.gap_ms : i.gap_ms;
        auto prev = max_gap_.load(std::memory_order_relaxed);
        while (g > prev and not max_gap_.compare_exchange_weak(prev, g, std::memory_order_relaxed)) {}
        stale_edges_.store(i.stale_edges, std::memory_order_relaxed);
        ring_span_.store(i.ring_span_ms, std::memory_order_relaxed);
        note_motion(G);
        maybe_write();
    }

private:
    // Peak speed seen in this window, read from the twist the PRODUCER publishes rather than
    // differenced from poses here — differencing picks up the localiser's own corrections and would
    // report motion the robot did not make (measured elsewhere: ~75 mm snaps on ~7% of updates).
    void note_motion(const std::shared_ptr<DSR::DSRGraph> &G)
    {
        if (G == nullptr or G->get_rt_api() == nullptr) return;
        const auto robots = G->get_nodes_by_type("robot");
        const auto rooms  = G->get_nodes_by_type("room");
        if (robots.empty() or rooms.empty()) return;
        auto e = G->get_rt_api()->get_edge_RT(robots.front(), rooms.front().id());
        if (not e.has_value()) e = G->get_rt_api()->get_edge_RT(rooms.front(), robots.front().id());
        if (not e.has_value()) return;
        const auto tv = G->get_attrib_by_name<rt_translation_velocity_att>(e.value());
        const auto rv = G->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(e.value());
        if (tv.has_value() and tv->get().size() >= 2)
        {
            const double v = std::hypot(tv->get()[0], tv->get()[1]);   // magnitude: order-agnostic
            auto p = max_v_.load(std::memory_order_relaxed);
            while (v > p and not max_v_.compare_exchange_weak(p, v, std::memory_order_relaxed)) {}
        }
        if (rv.has_value() and rv->get().size() >= 3)
        {
            const double w = std::abs(rv->get()[2]);
            auto p = max_w_.load(std::memory_order_relaxed);
            while (w > p and not max_w_.compare_exchange_weak(p, w, std::memory_order_relaxed)) {}
        }
    }

    void maybe_write()
    {
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch()).count();
        auto last = last_ms_.load(std::memory_order_relaxed);
        if (last != 0 and now - last < period_ms_) return;
        if (not last_ms_.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
        if (last == 0) return;              // first call only arms the clock; nothing measured yet
        const auto n = total_.exchange(0, std::memory_order_relaxed);
        if (n == 0) return;
        const auto take = [](std::atomic<std::uint64_t> &c)
        { return c.exchange(0, std::memory_order_relaxed); };

        std::scoped_lock lk(io_);
        if (not csv_.is_open())
        {
            const bool fresh = not std::ifstream(csv_path_).good();
            csv_.open(csv_path_, std::ios::out | std::ios::app);
            // Decimal POINT regardless of LANG: these machines run es_ES, where the C locale would
            // emit a COMMA and split every float into two columns (CLAUDE.md).
            csv_.imbue(std::locale::classic());
            if (csv_.is_open() and fresh)
                csv_ << "# One row per call site per window. APPENDS across runs — compare, don't assume.\n"
                        "# outcome shares are % of the window's queries:\n"
                        "#   exact  = the query landed on a block (best case; common when the pose is\n"
                        "#            DERIVED from the same sensor frame, so the stamps coincide)\n"
                        "#   interp = bracketed between two blocks — healthy\n"
                        "#   extrap = past an end, walked along the twist (only if this site asks for it)\n"
                        "#   ★clamped = past an end of a LIVE ring, returned the end block AS-IS. THE ONE\n"
                        "#            TO READ: this site placed a detection at a pose from another instant.\n"
                        "#   stale  = past an end by more than the ring's own span. Usually STRUCTURAL, not\n"
                        "#            a fault — a mount written once at bootstrap sits in most chains and\n"
                        "#            pins the chain's worst-edge verdict; stale_edges counts them.\n"
                        "# max_gap_ms = worst distance outside the ring in the window. Read it WITH\n"
                        "#   max_v/max_w: a clamp only costs geometry while the robot is moving.\n"
                        "t_ms,tag,n,exact,interp,extrap,clamped,stale,max_gap_ms,ring_span_ms,"
                        "stale_edges,max_v_mps,max_w_rps\n";
        }
        if (not csv_.is_open()) return;
        const auto pc = [n](std::uint64_t c) { return 100.0 * static_cast<double>(c) / static_cast<double>(n); };
        csv_ << now << ',' << tag_ << ',' << n << ','
             << pc(take(exact_)) << ',' << pc(take(interp_)) << ',' << pc(take(extrap_)) << ','
             << pc(take(clamped_)) << ',' << pc(take(stale_)) << ','
             << max_gap_.exchange(0, std::memory_order_relaxed) << ','
             << ring_span_.load(std::memory_order_relaxed) << ','
             << stale_edges_.load(std::memory_order_relaxed) << ','
             << max_v_.exchange(0.0, std::memory_order_relaxed) << ','
             << max_w_.exchange(0.0, std::memory_order_relaxed) << '\n';
        csv_.flush();
    }

    std::string tag_, csv_path_;
    std::int64_t period_ms_;
    std::mutex io_;
    std::ofstream csv_;
    std::atomic<std::uint64_t> total_{0}, exact_{0}, interp_{0}, extrap_{0}, clamped_{0}, stale_{0};
    std::atomic<std::int64_t> max_gap_{0}, ring_span_{0}, last_ms_{0};
    std::atomic<int> stale_edges_{0};
    std::atomic<double> max_v_{0.0}, max_w_{0.0};
};
}   // namespace rc::rtprobe
