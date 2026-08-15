#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <string>

namespace rc
{

// Lightweight observer of input-stream delivery rates, keyed by source timestamp.
// Telemetry + stall detection only — NO control (the regulator owns actuation). Rates
// come from distinct-stamp deltas (EMA, robust to the media cache repeating the last
// frame); a stream is "stalled" when no new stamp has arrived for `stall_s` wall
// seconds. Streams are registered lazily on first tick(), so adding one (e.g. ricoh)
// is a single tick() call at the source.
class StreamRateMonitor
{
    static constexpr std::size_t kGapWin = 32;   // recent stamp gaps kept per stream for the quantile
public:
    void tick(const std::string& name, std::uint64_t stamp_ms)
    {
        auto& s = streams_[name];
        const auto now = std::chrono::steady_clock::now();
        if (s.last_stamp != 0 && stamp_ms > s.last_stamp)
        {
            const double dt = static_cast<double>(stamp_ms - s.last_stamp);
            if (dt > 0.0 && dt < 5000.0)
            {
                // ★AVERAGE THE PERIOD, THEN INVERT — never average the rate. An EMA over 1/dt is biased
                // HIGH by Jensen's inequality (E[1/dt] > 1/E[dt]), and the bias grows with jitter, so the
                // reported rate could exceed the source rate — which is impossible and was visible: 29.1 Hz
                // shown for a ZED delivering 26.4. Averaging dt and inverting once has no such bias.
                s.dt_ema = s.dt_ema > 0.0 ? 0.9 * s.dt_ema + 0.1 * dt : dt;
                s.hz     = s.dt_ema > 0.0 ? 1000.0 / s.dt_ema : 0.0;

                // SOURCE CADENCE. We only see the stamps of frames we chose to PROCESS, so gaps are
                // k*T for integer k and the smallest ones are our best evidence of T itself.
                // ★NOT the strict minimum. min over a window is an extreme-value statistic: one jittered
                // stamp sets it for the whole window, and the more samples the lower it drifts — that is
                // what reported 38.5 Hz for both a 26.4 Hz camera and a 31.3 Hz one (both had a single
                // 26 ms gap). A low QUANTILE of the recent gaps needs the short gap to RECUR before it
                // counts, so a lone outlier cannot move it.
                s.gaps[s.gap_i % kGapWin] = dt;
                ++s.gap_i;
                const std::size_t n = std::min<std::size_t>(s.gap_i, kGapWin);
                if (n >= 8)
                {
                    std::array<double, kGapWin> tmp = s.gaps;
                    const std::size_t k = n / 5;                       // 20th percentile
                    std::nth_element(tmp.begin(), tmp.begin() + k, tmp.begin() + n);
                    s.feed_hz = tmp[k] > 0.0 ? 1000.0 / tmp[k] : 0.0;
                }
            }
        }
        if (stamp_ms != s.last_stamp) { s.last_stamp = stamp_ms; s.last_new = now; }
    }

    // EMA delivery rate (Hz) of `name`; -1 if the stream was never seen.
    [[nodiscard]] double rate_hz(const std::string& name) const
    {
        auto it = streams_.find(name);
        return it == streams_.end() ? -1.0 : it->second.hz;
    }

    // SOURCE cadence (Hz) of `name`, from a low quantile of the recent stamp gaps. Compare against
    // rate_hz(): rate_hz is what WE processed, feed_hz is the fastest cadence the source RELIABLY shows.
    // ★It is a LOWER BOUND on the true source rate, and deliberately so: if we never process two
    // consecutive source frames, the smallest gap we can observe is already a multiple of the true
    // period, so feed_hz reads low. It must never read HIGH — a rate above the source is impossible and
    // is the bug this replaced. -1 if never seen, 0 until enough gaps have accumulated.
    [[nodiscard]] double feed_hz(const std::string& name) const
    {
        auto it = streams_.find(name);
        return it == streams_.end() ? -1.0 : it->second.feed_hz;
    }

    // Wall seconds since the last NEW stamp on `name`; -1 if the stream was never seen.
    [[nodiscard]] double idle_s(const std::string& name) const
    {
        auto it = streams_.find(name);
        if (it == streams_.end() || it->second.last_new.time_since_epoch().count() == 0)
            return -1.0;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - it->second.last_new).count();
    }

    // Externally measured SOURCE rate (Hz) for `name` — arrivals counted at the subscriber. Optional:
    // when set (>0) the report prints THIS instead of the tick-derived rate.
    // ★WHY THE REPORT NEEDS IT. This line is "input-stream health", but s.hz is derived from tick(), and
    // a tick is only as fast as whatever samples it. rgb360 is ticked from the 50 ms render timer, so its
    // reported rate was CAPPED AT 20 Hz and read 19.4 for a camera streaming at 31 — the display was
    // measuring its own clock, not the stream. A counted arrival rate has no such ceiling.
    void set_source_hz(const std::string& name, double hz) { streams_[name].source_hz = hz; }

    struct Report { std::string summary; bool any_stall = false; };

    // Returns a report at most every `report_s`; flags streams idle for > `stall_s`.
    std::optional<Report> maybe_report(double report_s, double stall_s)
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_report_.time_since_epoch().count() == 0) last_report_ = now;
        if (std::chrono::duration<double>(now - last_report_).count() < report_s)
            return std::nullopt;
        last_report_ = now;

        Report r;
        for (auto& [name, s] : streams_)
        {
            const double idle = std::chrono::duration<double>(now - s.last_new).count();
            const bool stalled = idle > stall_s;
            if (stalled) r.any_stall = true;
            char buf[64];
            if (stalled) std::snprintf(buf, sizeof buf, "%s STALLED(%.1fs) | ", name.c_str(), idle);
            // Prefer the counted source rate; fall back to the tick-derived one when nobody supplied it
            // (e.g. the readout panel that measures it is disabled).
            else         std::snprintf(buf, sizeof buf, "%s %.1f Hz | ", name.c_str(),
                                       s.source_hz > 0.0 ? s.source_hz : s.hz);
            r.summary += buf;
        }
        return r;
    }

private:
    struct Stat
    {
        double        hz = 0.0;
        double        source_hz = 0.0;   // counted at the subscriber; 0 = not supplied, use hz
        double        dt_ema = 0.0;      // EMA of the stamp PERIOD (ms) — hz is 1000/this, never an EMA of rates
        double        feed_hz = 0.0;     // 1000 / 20th-percentile gap: the fastest cadence we RELIABLY see
        std::array<double, kGapWin> gaps{};
        std::size_t   gap_i = 0;
        std::uint64_t last_stamp = 0;
        std::chrono::steady_clock::time_point last_new{};
    };
    std::map<std::string, Stat> streams_;
    std::chrono::steady_clock::time_point last_report_{};
};

} // namespace rc
