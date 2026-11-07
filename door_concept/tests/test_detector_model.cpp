// test_detector_model.cpp — Task 3: detector model + exact Bernoulli LLRs + persisted table.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_detector_model.cpp \
//       -o /tmp/test_detector_model && /tmp/test_detector_model
#include <clocale>
#include <cmath>
#include <cstdio>
#include "../src/core/detector_model.h"
#ifndef DETECTORS_CSV
#define DETECTORS_CSV "etc/door_detectors.csv"
#endif
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static ViewGeometry zed_geom(float d, float aspect)   // 1280x720, f=700, door 0.85 x 2.05
{
    ViewGeometry g; g.projectable = true; g.vis = 1.0f; g.aspect = aspect; g.range_m = d;
    g.fill_h = 700.f * 0.85f * aspect / (d * 1280.f); g.fill_v = 700.f * 2.05f / (d * 720.f);
    return g;
}
int main()
{
    const auto D = default_detectors();
    { // miss_zero_when_unseen
        CHECK(miss_llr(0.f, 0.f, 0.7f) == 0.f);
        CHECK(miss_llr(0.f, 0.05f, 0.f) == 0.f);
        CHECK(miss_llr(0.3f, 0.05f, 0.f) == std::log(0.7f));
        ViewGeometry g = zed_geom(3.f, 1.f); g.vis = 0.f;
        const float q = detect_prob(D.zed, g, 1.f);
        CHECK(q == 0.f);
        CHECK(miss_llr(q, D.zed.false_alarm, g.vis) == 0.f);
    }
    { // miss_monotone
        float prev = 1e9f;
        for (float q = 0.f; q < 0.95f; q += 0.05f) { const float m = miss_llr(q, 0.05f, 1.f); CHECK(m < prev); prev = m; }
    }
    { // aspect_70deg_1m_and_6m (spec test 4)
        for (float d : {1.f, 6.f}) for (float asp : {1.f, std::cos(70.f * 3.14159265f / 180.f)})
        {
            const auto g = zed_geom(d, asp);
            const float q = detect_prob(D.zed, g, 1.f);
            const float h = hit_llr(q, D.zed.false_alarm, g.vis, 1.f), m = miss_llr(q, D.zed.false_alarm, g.vis);
            std::printf("  zed d=%.0f asp=%.2f q=%.4g hit=%.3f miss=%.3f\n", d, asp, q, h, m);
            CHECK(std::abs(m) < std::abs(h));
        }
    }
    { // ricoh_1p4m_defaults_printed (R8: spec's q~0 claim is false; recorded, not forced)
        ViewGeometry g; g.projectable = true; g.vis = 1.f; g.aspect = 1.f; g.fill_h = 0.28145f; g.fill_v = 0.38618f; g.range_m = 1.4f;
        const auto t = load_detectors(DETECTORS_CSV);
        const float q = detect_prob(t.ricoh, g, 1.f), m = miss_llr(q, t.ricoh.false_alarm, g.vis);
        std::printf("  ricoh 1.4 m: q=%.4f miss_llr=%.4f\n", q, m);
        CHECK(std::isfinite(q) and std::isfinite(m) and m < 0.f);
    }
    { // csv_roundtrip_es_locale
        const char* loc = std::setlocale(LC_ALL, "es_ES.UTF-8");
        if (loc == nullptr) std::setlocale(LC_ALL, "");
        std::printf("  locale: %s\n", loc ? loc : "(es_ES unavailable)");
        DetectorTable t = D; t.ricoh.env.min_fill = 0.2345f; t.zed.hit_rate = 0.8125f; t.zed.n_hit_trials = 37.f;
        const std::filesystem::path p = "/tmp/_det_rt.csv";
        save_detectors(p, t, "test");
        const auto r = load_detectors(p);
        CHECK(r.ricoh.env.min_fill == t.ricoh.env.min_fill and r.zed.hit_rate == t.zed.hit_rate and r.zed.n_hit_trials == 37.f);
        CHECK(r.ricoh.env.max_fill == t.ricoh.env.max_fill and r.ricoh.false_alarm == t.ricoh.false_alarm);
        CHECK(load_detectors("/nonexistent.csv").zed.env.min_fill == D.zed.env.min_fill);
    }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails != 0;
}
