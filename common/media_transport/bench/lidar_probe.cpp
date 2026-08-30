// lidar_probe — read ONE frame off a lidar media plane and describe its BEAM GEOMETRY.
//
// WHY. On 2026-08-29 a helios cloud came out wrong and three plausible explanations
// (wrong mount axis, wrong tilt sign, a mirrored azimuth) all predicted "it looks rotated".
// None can be told apart by looking at a 3-D view. This prints the numbers that separate them,
// in the DEVICE frame the plane actually carries:
//   * the elevation span and the per-layer step  -> is the fan the 70 deg the proto declares?
//   * mean elevation PER AZIMUTH SECTOR          -> a PITCH is flat across sectors; a ROLL is a
//                                                   sinusoid. This is the tiltAngle question.
//   * the azimuth of the nearest returns         -> where the near structure sits, to compare
//                                                   two sensors that see the same room.
// Usage: lidar_probe <topic> [domain]
#include "media_transport.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <string>

int main(int argc, char** argv)
{
    const std::string topic = argc > 1 ? argv[1] : "rc/lidar3d/points";
    const std::uint32_t domain = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 7u;

    rc::media::SubscriberConfig cfg;
    cfg.domain_id = domain;
    cfg.topic_name = topic;
    cfg.shared_memory_only = true;
    rc::media::LidarSubscriber sub;
    if (not sub.init(cfg)) { std::printf("init failed on %s\n", topic.c_str()); return 1; }

    std::vector<float> xyz;
    for (int tries = 0; tries < 60 and xyz.empty(); ++tries)
        sub.wait_and_poll([&](const rc::media::LidarFrame& f, std::int64_t)
        {
            const std::size_t n = f.count();
            const std::size_t st = f.stride() ? f.stride() : 3;
            xyz.assign(f.points().data(), f.points().data() + n * st);
        }, 200);
    if (xyz.empty()) { std::printf("no frame on %s (domain %u)\n", topic.c_str(), domain); return 2; }

    const std::size_t n = xyz.size() / 3;
    std::printf("=== %s  (domain %u)   %zu points\n", topic.c_str(), domain, n);
    if (argc > 3)   // optional CSV dump: x,y,z — for offline re-mapping experiments
    {
        if (std::FILE* f = std::fopen(argv[3], "w"))
        {
            for (std::size_t i = 0; i < n; ++i)
                std::fprintf(f, "%.5f,%.5f,%.5f\n", xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            std::fclose(f);
            std::printf("[dump] %zu points -> %s\n", n, argv[3]);
        }
    }

    constexpr int NS = 8;                      // azimuth sectors
    std::vector<double> el_sum(NS, 0.0), r_min(NS, 1e9);
    std::vector<int>    el_n(NS, 0);
    std::vector<float>  elev; elev.reserve(n);
    double zlo = 1e9, zhi = -1e9, rlo = 1e9, rhi = -1e9;

    for (std::size_t i = 0; i < n; ++i)
    {
        const float x = xyz[3*i], y = xyz[3*i+1], z = xyz[3*i+2];
        const double rh = std::hypot(x, y);
        if (rh < 1e-6 and std::fabs(z) < 1e-6) continue;
        const double el = std::atan2(z, rh) * 180.0 / M_PI;
        const double az = std::atan2(y, x) * 180.0 / M_PI;       // -180..180
        const int s = std::clamp(static_cast<int>((az + 180.0) / (360.0 / NS)), 0, NS - 1);
        elev.push_back(static_cast<float>(el));
        el_sum[s] += el; el_n[s]++;
        r_min[s] = std::min(r_min[s], std::hypot(rh, static_cast<double>(z)));
        zlo = std::min(zlo, (double)z); zhi = std::max(zhi, (double)z);
        rlo = std::min(rlo, rh);        rhi = std::max(rhi, rh);
    }
    if (elev.empty()) { std::printf("all points degenerate\n"); return 3; }

    std::sort(elev.begin(), elev.end());
    auto pct = [&](double p){ return elev[std::clamp<std::size_t>((std::size_t)(p*elev.size()), 0, elev.size()-1)]; };
    std::printf("elevation deg: min %.2f  p05 %.2f  p50 %.2f  p95 %.2f  max %.2f   (span %.2f)\n",
                elev.front(), pct(0.05), pct(0.50), pct(0.95), elev.back(), elev.back()-elev.front());
    std::printf("z m: [%.3f .. %.3f]   horiz radius m: [%.3f .. %.3f]\n", zlo, zhi, rlo, rhi);

    // Distinct elevation LEVELS = the layers. Cluster sorted elevations with a 0.5 deg gap.
    int layers = 1; double last = elev.front(), first_l = elev.front(), prev_l = elev.front();
    double step_sum = 0.0; int step_n = 0;
    for (float e : elev)
        if (e - last > 0.5) { layers++; step_sum += e - prev_l; step_n++; prev_l = e; last = e; }
        else last = std::max(last, (double)e);
    std::printf("distinct layers ~%d   mean step %.3f deg   (first %.2f)\n",
                layers, step_n ? step_sum/step_n : 0.0, first_l);

    // Dense horizontal planes. The floor is a physical fact: it MUST show up as a peak at
    // |z| = the mount height. Which SIGN it lands on says whether the device frame is upright
    // or inverted, and a peak at the wrong |z| entirely says the layer->elevation map is broken.
    {
        constexpr double BIN = 0.10;
        std::vector<int> h(121, 0);
        for (std::size_t i = 0; i < n; ++i)
        {
            const float z = xyz[3*i+2];
            const int b = (int)std::lround(z / BIN) + 60;
            if (b >= 0 and b < 121) h[b]++;
        }
        std::printf("\ndensest z planes (0.10 m bins):\n");
        std::vector<int> idx(121); for (int i = 0; i < 121; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return h[a] > h[b]; });
        for (int k = 0; k < 5 and h[idx[k]] > 0; ++k)
            std::printf("   z = %+6.2f m   %6d pts\n", (idx[k] - 60) * BIN, h[idx[k]]);
    }

    // Median SLANT RANGE of the extreme layers. This is the discriminator between two
    // hypotheses that both look like "the cloud is rotated": a reversed layer order and an
    // inverted tilt sign predict very different ranges for the same assigned elevation.
    {
        auto layer_stats = [&](double e_lo, double e_hi, const char* tag)
        {
            std::vector<double> rr;
            for (std::size_t i = 0; i < n; ++i)
            {
                const float x = xyz[3*i], y = xyz[3*i+1], z = xyz[3*i+2];
                const double rh = std::hypot(x, y);
                const double el = std::atan2(z, rh) * 180.0 / M_PI;
                if (el >= e_lo and el <= e_hi) rr.push_back(std::hypot(rh, (double)z));
            }
            if (rr.empty()) { std::printf("   %s: no points\n", tag); return; }
            std::sort(rr.begin(), rr.end());
            std::printf("   %s  n=%5zu  slant range  p10 %.2f  MEDIAN %.2f  p90 %.2f m\n",
                        tag, rr.size(), rr[rr.size()/10], rr[rr.size()/2], rr[rr.size()*9/10]);
        };
        std::printf("\nextreme layers (assigned elevation -> what they actually ranged):\n");
        layer_stats(elev.back() - 1.0, elev.back() + 1.0, "TOP    layer");
        layer_stats(elev.front() - 1.0, elev.front() + 1.0, "BOTTOM layer");
    }

    std::printf("\nazimuth sector    mean elev    n      nearest r\n");
    for (int s = 0; s < NS; ++s)
        std::printf("  [%+4.0f..%+4.0f]      %+7.2f  %6d   %8.3f\n",
                    -180.0 + s*(360.0/NS), -180.0 + (s+1)*(360.0/NS),
                    el_n[s] ? el_sum[s]/el_n[s] : 0.0, el_n[s], el_n[s] ? r_min[s] : 0.0);
    std::printf("\n^ FLAT mean-elev across sectors = a PITCH (tiltAngle behaves as a fan offset).\n"
                "  SINUSOIDAL across sectors      = a ROLL (the sweep plane is tilted).\n");
    return 0;
}
