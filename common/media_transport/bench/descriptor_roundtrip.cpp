// Round-trip the descriptor under the agent's real locale.
#include "media_transport.h"
#include <clocale>
#include <cstdio>
int main()
{
    std::setlocale(LC_ALL, "");                 // Qt does this; without it the test lies
    std::printf("LC_NUMERIC = %s\n\n", std::setlocale(LC_NUMERIC, nullptr));

    rc::media::MediaDescriptor d;
    d.domain_id = 7; d.type_name = "LidarFrame"; d.type_tag = rc::media::LIDAR_FRAME_TYPE_TAG;
    d.ready = true; d.streams["lidar"] = "helios_points"; d.stream_types["lidar"] = "LidarFrame";

    // 1. an EXISTING producer (no model) must be byte-identical to before
    const std::string without = d.to_json();
    std::printf("no model advertised:\n  %s\n\n", without.c_str());

    auto& m = d.model;
    m.version = 1; m.source = "datasheet"; m.ref = "helios-32/rev-b"; m.stamp_ms = 1787914174794;
    m.frame = "helios";
    m.rings = 4; m.ring_elev_deg = {-15.f, -7.5f, 0.f, 7.5f};
    m.azimuth_step_deg = 0.2f; m.range_min_m = 0.2f; m.range_max_m = 120.f; m.rate_hz = 10.f;
    m.range_sigma_floor_m = 0.0005f;            // the value that becomes 0 under strtod
    m.range_k_rel = 0.003f;
    m.gyro_sigma = 1.0e-4f; m.gyro_bias = 5.0e-5f;

    const std::string js = d.to_json();
    std::printf("with model:\n  %s\n\n", js.c_str());

    const auto back = rc::media::MediaDescriptor::from_json(js);
    if (not back) { std::printf("FAIL: from_json returned nullopt\n"); return 1; }
    const auto& b = back->model;
    int bad = 0;
    auto chk = [&bad](const char* n, float got, float want)
    { const bool ok = std::abs(got-want) <= 1e-9f*std::abs(want)+1e-12f;
      std::printf("  %-26s got=%-12g want=%-12g %s\n", n, got, want, ok?"ok":"MISMATCH"); bad += !ok; };

    std::printf("round-trip:\n");
    chk("range_sigma_floor_m", b.range_sigma_floor_m.value_or(-1), 0.0005f);
    chk("range_k_rel",         b.range_k_rel.value_or(-1),         0.003f);
    chk("gyro_sigma",          b.gyro_sigma.value_or(-1),          1.0e-4f);
    chk("gyro_bias",           b.gyro_bias.value_or(-1),           5.0e-5f);
    chk("azimuth_step_deg",    b.azimuth_step_deg.value_or(-1),    0.2f);
    chk("range_max_m",         b.range_max_m.value_or(-1),         120.f);
    std::printf("  %-26s got=%zu want=4\n", "ring_elev_deg.size()", b.ring_elev_deg.size());
    bad += (b.ring_elev_deg.size() != 4);
    for (std::size_t i=0;i<b.ring_elev_deg.size();++i) std::printf("      ring[%zu] = %g\n", i, b.ring_elev_deg[i]);
    std::printf("  %-26s %s / %s / %lld\n", "provenance", b.source.c_str(), b.ref.c_str(), (long long)b.stamp_ms);
    // absent must stay ABSENT, not zero
    std::printf("  %-26s %s\n", "acc_sigma (never set)", b.acc_sigma.has_value()?"PRESENT - BUG":"absent (correct)");
    bad += b.acc_sigma.has_value();
    // streams must not have grown phantom entries from the new keys
    std::printf("  %-26s %zu (want 1)\n", "streams", back->streams.size());
    bad += (back->streams.size()!=1);
    std::printf("\n%s\n", bad ? "FAILURES" : "ALL OK");
    return bad!=0;
}
