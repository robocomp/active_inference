// Gate-1 bench for the media transport plane.
//
// Run in two terminals (same board) to exercise real cross-process
// shared-memory data-sharing / zero-copy:
//
//   ./media_bench sub
//   ./media_bench pub --fps 30
//
// The publisher simulates a 1280x720 BGR8 ZED RGB stream (2.76 MB/frame) and
// writes via zero-copy loans. The subscriber measures received fps, frame
// drops (frame_id gaps), payload integrity, and end-to-end latency.

#include "../media_transport.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <fastdds/dds/log/Log.hpp>
#include <regex>
#include "../generated/idl/image_framePubSubTypes.hpp"

using namespace std::chrono;
using rc::media::ImageFrame;

namespace
{
std::atomic_bool g_stop{false};
void on_sigint(int) { g_stop.store(true); }

std::int64_t now_ns()
{
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

constexpr std::uint32_t kWidth  = 1280;
constexpr std::uint32_t kHeight = 720;
constexpr std::uint32_t kBytes  = kWidth * kHeight * 3;  // BGR8 = 2,764,800

struct Args
{
    std::string role;
    std::string topic = "rc/zed/rgb";
    int   domain = 0;
    double fps   = 30.0;
    int   secs   = 0;  // 0 = run until Ctrl-C
    bool  udp    = false;  // true => allow builtin transports (UDP+SHM)
};

Args parse(int argc, char** argv)
{
    Args a;
    if (argc > 1) a.role = argv[1];
    for (int i = 2; i < argc; ++i)
    {
        std::string k = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--topic")       a.topic  = next();
        else if (k == "--domain") a.domain = std::atoi(next());
        else if (k == "--fps")    a.fps    = std::atof(next());
        else if (k == "--secs")   a.secs   = std::atoi(next());
        else if (k == "--udp")    a.udp    = true;
    }
    return a;
}

int run_pub(const Args& a)
{
    rc::media::MediaPublisher pub;
    rc::media::PublisherConfig cfg;
    cfg.domain_id = static_cast<std::uint32_t>(a.domain);
    cfg.topic_name = a.topic;
    cfg.history_depth = 8;
    cfg.shared_memory_only = !a.udp;
    if (!pub.init(cfg))
    {
        std::fprintf(stderr, "[pub] init failed\n");
        return 1;
    }
    std::printf("[pub] topic=%s fps=%.1f frame=%ux%u (%u B) data_sharing=%d\n",
                a.topic.c_str(), a.fps, kWidth, kHeight, kBytes,
                pub.data_sharing_active());

    const auto period = duration_cast<nanoseconds>(duration<double>(1.0 / a.fps));
    auto next_tick = steady_clock::now();
    std::uint64_t fid = 0, sent = 0, skipped = 0;
    auto last_report = steady_clock::now();
    const auto deadline = steady_clock::now() + seconds(a.secs);

    while (!g_stop.load())
    {
        if (a.secs > 0 && steady_clock::now() >= deadline) break;

        ImageFrame* f = pub.loan();
        if (f == nullptr)
        {
            ++skipped;
        }
        else
        {
            ++fid;
            f->stream_id(rc::media::STREAM_ZED_RGB);
            f->frame_id(fid);
            f->stamp_ns(static_cast<std::uint64_t>(now_ns()));
            f->width(kWidth);
            f->height(kHeight);
            f->step(kWidth * 3);
            f->format(rc::media::FORMAT_BGR8);
            f->size(kBytes);
            // Simulate the single per-frame memcpy (memory-bandwidth cost) and
            // embed an integrity sentinel: first & last byte = fid low byte.
            auto& buf = f->data();
            std::memset(buf.data(), static_cast<int>(fid & 0xFF), kBytes);
            if (pub.publish(f)) ++sent;
            else pub.discard(f);
        }

        if (auto now = steady_clock::now(); now - last_report >= seconds(1))
        {
            std::printf("[pub] sent=%llu fps~%llu skipped=%llu\n",
                        (unsigned long long)sent, (unsigned long long)sent,
                        (unsigned long long)skipped);
            sent = 0;
            last_report = now;
        }

        next_tick += period;
        std::this_thread::sleep_until(next_tick);
    }
    std::printf("[pub] done. total frames=%llu skipped=%llu\n",
                (unsigned long long)fid, (unsigned long long)skipped);
    return 0;
}

int run_sub(const Args& a)
{
    rc::media::MediaSubscriber sub;
    rc::media::SubscriberConfig cfg;
    cfg.domain_id = static_cast<std::uint32_t>(a.domain);
    cfg.topic_name = a.topic;
    cfg.history_depth = 8;
    cfg.shared_memory_only = !a.udp;
    if (!sub.init(cfg))
    {
        std::fprintf(stderr, "[sub] init failed\n");
        return 1;
    }
    std::printf("[sub] topic=%s data_sharing=%d  waiting for frames...\n",
                a.topic.c_str(), sub.data_sharing_active());

    std::uint64_t last_fid = 0, recv_win = 0, drops = 0, bad = 0;
    long double lat_sum_us = 0;
    auto last_report = steady_clock::now();
    const auto deadline = steady_clock::now() + seconds(a.secs);

    while (!g_stop.load())
    {
        if (a.secs > 0 && steady_clock::now() >= deadline) break;

        const int n = sub.wait_and_poll(
            [&](const ImageFrame& f, std::int64_t recv)
            {
                ++recv_win;
                const std::uint8_t expect = static_cast<std::uint8_t>(f.frame_id() & 0xFF);
                const auto& buf = f.data();
                if (f.size() == 0 || buf[0] != expect || buf[f.size() - 1] != expect)
                    ++bad;
                if (last_fid != 0 && f.frame_id() > last_fid + 1)
                    drops += (f.frame_id() - last_fid - 1);
                last_fid = f.frame_id();
                lat_sum_us += (recv - static_cast<std::int64_t>(f.stamp_ns())) / 1000.0L;
            },
            200);
        (void)n;

        if (auto now = steady_clock::now(); now - last_report >= seconds(1))
        {
            const double avg_lat = recv_win ? static_cast<double>(lat_sum_us / recv_win) : 0.0;
            const double mbps = recv_win * (kBytes / (1024.0 * 1024.0));
            std::printf("[sub] fps=%llu drops=%llu bad=%llu avg_lat=%.0fus thr=%.0fMB/s\n",
                        (unsigned long long)recv_win, (unsigned long long)drops,
                        (unsigned long long)bad, avg_lat, mbps);
            recv_win = 0; lat_sum_us = 0;
            last_report = now;
        }
    }
    std::printf("[sub] done. total_drops=%llu bad=%llu\n",
                (unsigned long long)drops, (unsigned long long)bad);
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::signal(SIGINT, on_sigint);
    Args a = parse(argc, argv);
    eprosima::fastdds::dds::Log::SetVerbosity(eprosima::fastdds::dds::Log::Info);
    eprosima::fastdds::dds::Log::SetCategoryFilter(std::regex("(DATASHARING|RTPS_WRITER|RTPS_READER)"));
    {
        rc::media::ImageFramePubSubType ts;
        std::printf("[diag] is_plain(XCDRv2)=%d max_serialized=%u\n",
                    ts.is_plain(eprosima::fastdds::dds::DataRepresentationId_t::XCDR2_DATA_REPRESENTATION),
                    ts.max_serialized_type_size);
    }
    if (a.role == "pub") return run_pub(a);
    if (a.role == "sub") return run_sub(a);
    std::fprintf(stderr, "usage: %s pub|sub [--topic T] [--domain D] [--fps N] [--secs S]\n", argv[0]);
    return 2;
}
