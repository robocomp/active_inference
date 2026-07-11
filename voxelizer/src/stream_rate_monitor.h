#pragma once

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
                const double hz = 1000.0 / dt;
                s.hz = s.hz > 0.0 ? 0.9 * s.hz + 0.1 * hz : hz;
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

    // Wall seconds since the last NEW stamp on `name`; -1 if the stream was never seen.
    [[nodiscard]] double idle_s(const std::string& name) const
    {
        auto it = streams_.find(name);
        if (it == streams_.end() || it->second.last_new.time_since_epoch().count() == 0)
            return -1.0;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - it->second.last_new).count();
    }

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
            else         std::snprintf(buf, sizeof buf, "%s %.1f Hz | ", name.c_str(), s.hz);
            r.summary += buf;
        }
        return r;
    }

private:
    struct Stat
    {
        double        hz = 0.0;
        std::uint64_t last_stamp = 0;
        std::chrono::steady_clock::time_point last_new{};
    };
    std::map<std::string, Stat> streams_;
    std::chrono::steady_clock::time_point last_report_{};
};

} // namespace rc
