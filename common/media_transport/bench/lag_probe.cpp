// lag_probe — measure the END-TO-END AGE of frames on a media plane, headless.
//
// The quantity is wall-clock now minus the frame's SOURCE capture stamp (epoch ms) — exactly what
// the stream viewers display (rc::viewers::LagMeter, common/viewers/frame_lag.h). This is the
// headless twin: no Qt, no window, and it reports the DISTRIBUTION rather than a smoothed level,
// which is what you need when deciding whether a number is a transport cost or a scheduling one.
//
// WHY THE DISTRIBUTION. A mean lag on its own does not say where it comes from. Two shapes settle it:
//   * lag p10..p90 spanning one poll period  -> the cost is the producer's polling PHASE, and the
//     min is the true transport+processing floor.
//   * stamp deltas that are BIMODAL at k and 2k -> the producer is DOWNSAMPLING its own source, i.e.
//     dropping frames, which no latency number alone would have revealed.
// That second signature is how the 50 ms lidar3d_dds poll was caught discarding ~35% of a 32 ms
// source on 2026-09-10 (deltas p25=33, p75=64; 20 Hz published from ~31 Hz in).
//
// Usage:  lag_probe lidar|image|image360|imu <topic> [domain=7] [seconds=5]
//   ./lag_probe lidar rc/lidar3d/points
//   ./lag_probe image rc/zed/rgb 7 8
//   ./lag_probe imu rc/imu/data
//
// A stamp of 0 means the producer did not stamp the frame; those are skipped rather than counted as
// zero-age. The value is signed: negative means the producer's clock leads ours, which on a
// multi-machine setup is clock skew, not latency.
#include "media_transport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
double now_ms()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double quantile(const std::vector<double>& sorted, double q)
{
    return sorted[std::min(sorted.size() - 1, static_cast<std::size_t>(q * sorted.size()))];
}

template <class Sub, class Frame>
int run(const std::string& topic, std::uint32_t domain, double secs)
{
    rc::media::SubscriberConfig cfg;
    cfg.domain_id = domain;
    cfg.topic_name = topic;
    cfg.shared_memory_only = true;
    Sub sub;
    if (not sub.init(cfg)) { std::printf("init failed on %s (domain %u)\n", topic.c_str(), domain); return 1; }

    std::vector<double> lags, deltas;
    std::uint64_t prev = 0;
    std::size_t dups = 0;   // frames republished under a stamp already seen
    const auto t_end = std::chrono::steady_clock::now() + std::chrono::duration<double>(secs);
    while (std::chrono::steady_clock::now() < t_end)
        sub.wait_and_poll([&](const Frame& f, std::int64_t)
        {
            const std::uint64_t stamp = f.stamp_ms();
            if (stamp == 0) return;                  // unstamped: no age can be formed
            lags.push_back(now_ms() - static_cast<double>(stamp));
            if (prev != 0 and stamp == prev) ++dups;   // a producer with no dedup republishes the same scan
            if (prev != 0 and stamp > prev) deltas.push_back(static_cast<double>(stamp - prev));
            prev = stamp;
        }, 200);

    if (lags.empty()) { std::printf("%-22s no stamped frames in %.0f s\n", topic.c_str(), secs); return 2; }

    std::sort(lags.begin(), lags.end());
    std::sort(deltas.begin(), deltas.end());
    double lsum = 0.0; for (double l : lags) lsum += l;
    double dsum = 0.0; for (double d : deltas) dsum += d;

    std::printf("%-22s n=%3zu  lag mean %6.1f  p10 %6.1f  med %6.1f  p90 %6.1f  min %6.1f  max %6.1f ms\n",
                topic.c_str(), lags.size(), lsum / static_cast<double>(lags.size()),
                quantile(lags, 0.10), quantile(lags, 0.50), quantile(lags, 0.90),
                lags.front(), lags.back());
    if (dups > 0)
        std::printf("%-22s   ^ %zu of %zu frames (%.0f%%) REPUBLISH a stamp already sent: the producer is not"
                    " deduping, so consumers pay for and re-time data they already have\n",
                    topic.c_str(), dups, lags.size(), 100.0 * static_cast<double>(dups) / static_cast<double>(lags.size()));
    if (not deltas.empty())
    {
        std::printf("%-22s stamp deltas  p10 %5.1f  p25 %5.1f  med %5.1f  p75 %5.1f  p90 %5.1f ms"
                    "   -> %.1f Hz published\n",
                    topic.c_str(), quantile(deltas, 0.10), quantile(deltas, 0.25), quantile(deltas, 0.50),
                    quantile(deltas, 0.75), quantile(deltas, 0.90),
                    1000.0 * static_cast<double>(deltas.size()) / dsum);
        // Say the bimodality out loud: p75 near 2x p25 is a producer skipping every other source frame.
        if (const double lo = quantile(deltas, 0.25), hi = quantile(deltas, 0.75);
            lo > 0.0 and hi > 1.6 * lo)
            std::printf("%-22s   ^ BIMODAL (%.0f / %.0f ms): the producer is downsampling its source — "
                        "it is dropping frames, not just delaying them\n", topic.c_str(), lo, hi);
    }
    return 0;
}
}   // namespace

int main(int argc, char** argv)
{
    const std::string kind   = argc > 1 ? argv[1] : "lidar";
    const std::string topic  = argc > 2 ? argv[2] : "rc/lidar3d/points";
    const std::uint32_t domain = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 7u;
    const double secs        = argc > 4 ? std::stod(argv[4]) : 5.0;
    if (kind == "image")    return run<rc::media::MediaSubscriber,    rc::media::ImageFrame>(topic, domain, secs);
    if (kind == "image360") return run<rc::media::Image360Subscriber, rc::media::Image360Frame>(topic, domain, secs);
    if (kind == "lidar")    return run<rc::media::LidarSubscriber,    rc::media::LidarFrame>(topic, domain, secs);
    if (kind == "imu")      return run<rc::media::ImuSubscriber,      rc::media::ImuFrame>(topic, domain, secs);
    std::printf("usage: lag_probe lidar|image|image360|imu <topic> [domain=7] [seconds=5]\n");
    return 64;
}
