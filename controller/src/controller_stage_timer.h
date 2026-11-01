#pragma once
// Per-cycle stage profiler. Cheap: two steady_clock reads and one short mutex per scope; prints ONE line per 5 s.
//   [stage] <name> mean X ms max Y ms n N | ...   (sorted by mean, descending)
// Names must be string literals (they are used as map keys by pointer-stable string_view).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace rc
{
class StageProfile
{
public:
    static StageProfile &instance() { static StageProfile p; return p; }

    void add(std::string_view name, double ms)
    {
        std::lock_guard lk(mx_);
        auto &a = acc_[std::string(name)];
        a.sum += ms; a.max = std::max(a.max, ms); ++a.n;
    }

    // Call from the loop; prints at most once per period_s and resets the window.
    void report_if_due(double period_s = 5.0)
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard lk(mx_);
        if (last_.time_since_epoch().count() == 0) { last_ = now; return; }
        if (std::chrono::duration<double>(now - last_).count() < period_s) return;
        last_ = now;
        std::vector<std::pair<std::string, Acc>> v(acc_.begin(), acc_.end());
        acc_.clear();
        if (v.empty()) return;
        std::ranges::sort(v, [](const auto &a, const auto &b) { return a.second.sum / a.second.n > b.second.sum / b.second.n; });
        std::string line = "[stage]";
        for (const auto &[k, a] : v)
            line += std::format(" {} mean {:.2f} ms max {:.2f} ms n {} |", k, a.sum / a.n, a.max, a.n);
        std::println("{}", line);
    }

private:
    struct Acc { double sum = 0, max = 0; std::uint64_t n = 0; };
    std::mutex mx_;
    std::map<std::string, Acc> acc_;
    std::chrono::steady_clock::time_point last_{};
};

struct ScopedStage
{
    explicit ScopedStage(std::string_view n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~ScopedStage()
    {
        StageProfile::instance().add(name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    ScopedStage(const ScopedStage &) = delete;
    ScopedStage &operator=(const ScopedStage &) = delete;
    std::string_view name;
    std::chrono::steady_clock::time_point t0;
};
}  // namespace rc
#define RC_STAGE_CAT2(a, b) a##b
#define RC_STAGE_CAT(a, b) RC_STAGE_CAT2(a, b)
#define RC_STAGE(name) ::rc::ScopedStage RC_STAGE_CAT(rc_stage_, __LINE__){name}
