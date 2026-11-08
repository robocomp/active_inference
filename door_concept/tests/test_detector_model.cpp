// test_detector_model.cpp — Task 3: detector model + exact Bernoulli LLRs + persisted table.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_detector_model.cpp \
//       -o /tmp/test_detector_model && /tmp/test_detector_model
#include <clocale>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iterator>
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
    { // miss_zero_when_unseen: q == 0 => miss and hit are exactly 0 for any false_alarm
        for (float fa : {0.f, 0.02f, 0.3f})
        {
            DetectorModel m = D.zed; m.false_alarm = fa;
            CHECK(miss_llr(0.f, false_alarm_eff(m, 0.f)) == 0.f);
            CHECK(hit_llr(0.f, m, 1.f) == 0.f);
        }
        ViewGeometry g = zed_geom(3.f, 1.f); g.vis = 0.f;
        const float q = detect_prob(D.zed, g, 1.f);
        CHECK(q == 0.f and miss_llr(q, false_alarm_eff(D.zed, q)) == 0.f and hit_llr(q, D.zed, 1.f) == 0.f);
    }
    { // miss_monotone
        float prev = 1e9f;
        for (float q = 0.f; q < 0.95f; q += 0.05f) { const float m = miss_llr(q, 0.05f * q / 0.9f); CHECK(m <= prev); prev = m; }
        CHECK(miss_llr(0.5f, 0.02f) < miss_llr(0.1f, 0.02f));
    }
    { // aspect_70deg_1m_and_6m (spec test 4)
        for (float d : {1.f, 6.f}) for (float asp : {1.f, std::cos(70.f * 3.14159265f / 180.f)})
        {
            const auto g = zed_geom(d, asp);
            const float q = detect_prob(D.zed, g, 1.f);
            const float h = hit_llr(q, D.zed, 1.f), m = miss_llr(q, false_alarm_eff(D.zed, q));
            std::printf("  zed d=%.0f asp=%.2f q=%.4g hit=%.3f miss=%.3g\n", d, asp, q, h, m);
            CHECK(std::abs(m) < std::abs(h));
        }
    }
    { // overflowing_zed_hit_not_negative
        const auto g = zed_geom(1.f, 1.f);
        const float q = detect_prob(D.zed, g, 1.f);
        const float h = hit_llr(q, D.zed, 1.f), m = miss_llr(q, false_alarm_eff(D.zed, q));
        std::printf("  zed 1 m: q=%.3g hit=%.4f miss=%.3g\n", q, h, m);
        CHECK(h >= 0.f and std::abs(m) < 1e-6f);
    }
    { // ricoh_1p4m_defaults_printed (R8: spec's q~0 claim is false; recorded, not forced)
        ViewGeometry g; g.projectable = true; g.vis = 1.f; g.aspect = 1.f; g.fill_h = 0.28145f; g.fill_v = 0.38618f; g.range_m = 1.4f;
        const auto t = load_detectors(DETECTORS_CSV);
        const float q = detect_prob(t.ricoh, g, 1.f), m = miss_llr(q, false_alarm_eff(t.ricoh, q));
        std::printf("  ricoh 1.4 m: q=%.4f miss_llr=%.4f\n", q, m);
        CHECK(std::isfinite(q) and std::isfinite(m) and m < 0.f);
    }
    { // csv_roundtrip_es_locale
        const char* loc = std::setlocale(LC_ALL, "es_ES.UTF-8");
        if (loc == nullptr) std::setlocale(LC_ALL, "");
        std::printf("  locale: %s\n", loc ? loc : "(es_ES unavailable)");
        DetectorTable t = D; t.ricoh.env.min_fill = 0.2345f; t.zed.hit_rate = 0.8125f; t.zed.n_hit_trials = 37.f;
        const std::filesystem::path p = std::filesystem::temp_directory_path() / std::format("det_rt_{}.csv", ::getpid());
        save_detectors(p, t, "test");
        const auto r = load_detectors(p);
        CHECK(r.ricoh.env.min_fill == t.ricoh.env.min_fill and r.zed.hit_rate == t.zed.hit_rate and r.zed.n_hit_trials == 37.f);
        CHECK(r.ricoh.env.max_fill == t.ricoh.env.max_fill and r.ricoh.false_alarm == t.ricoh.false_alarm);
        CHECK(load_detectors("/nonexistent.csv").zed.env.min_fill == D.zed.env.min_fill);
        std::filesystem::remove(p);
    }
    { // loaded_out_of_domain_falls_back + malformed_field_rejected
        const auto tmp = std::filesystem::temp_directory_path() / std::format("det_dom_{}.csv", ::getpid());
        const auto write = [&](const char* body) { std::ofstream o(tmp); o << "camera,param,value,n,source\n" << body; };
        write("zed,false_alarm,0,0,t\nricoh,min_fill,0.3,0,t\n");
        auto r = load_detectors(tmp);
        CHECK(r.zed.false_alarm == D.zed.false_alarm and r.zed.hit_rate == D.zed.hit_rate);   // zed reverted
        CHECK(r.ricoh.env.min_fill == 0.3f);                                                    // ricoh in domain, kept
        write("ricoh,hit_rate,1,5,t\nzed,soft,0.07,0,t\n");
        r = load_detectors(tmp);
        CHECK(r.ricoh.hit_rate == D.ricoh.hit_rate and r.ricoh.n_hit_trials == 0.f and r.zed.env.soft == 0.07f);
        write("zed,false_alarm,0.5,0,t\nzed,hit_rate,0.4,0,t\n");   // false_alarm >= hit_rate
        CHECK(load_detectors(tmp).zed.false_alarm == D.zed.false_alarm);
        write("zed,min_fill,0.1abc,0,t\nzed,max_fill,0.7,0,t\n");
        r = load_detectors(tmp);
        CHECK(r.zed.env.min_fill == D.zed.env.min_fill and r.zed.env.max_fill == 0.7f);
        std::filesystem::remove(tmp);
    }
    { // support_llr: monotone, continuous on the closed interval, finite at the ends, signs as the model says (spec: no test on x)
        const SupportModel m = D.support;
        float prev = support_llr(0.0f, m), max_step = 0.0f;
        CHECK(std::isfinite(prev));
        for (int i = 1; i <= 2000; ++i)
        {
            const float l = support_llr(static_cast<float>(i) / 2000.0f, m);
            CHECK(std::isfinite(l));
            CHECK(l <= prev + 1e-5f);                 // monotone decreasing in x
            max_step = std::max(max_step, std::abs(l - prev));
            prev = l;
        }
        std::printf("  support: max |dl| per dx=0.0005: %.4f; l(0)=%.2f l(0.05)=%.2f l(0.3)=%.2f l(0.5)=%.2f l(0.7)=%.2f l(0.9)=%.2f l(1)=%.2f\n",
                    max_step, support_llr(0.f, m), support_llr(.05f, m), support_llr(.3f, m), support_llr(.5f, m), support_llr(.7f, m), support_llr(.9f, m), support_llr(1.f, m));
        CHECK(max_step < 0.05f);                      // no step anywhere (a jump would show up as one big sample-to-sample change)
        CHECK(support_llr(0.05f, m) > 0.0f);          // a well-fitted door: a small positive
        CHECK(support_llr(1.0f, m) < -4.0f and support_llr(0.95f, m) < -4.0f);   // a fit explaining nothing of its support: strongly negative
        CHECK(support_llr(1.0000001f, m) == support_llr(1.0f, m));   // the mean of responsibilities rounding past 1 is 1
    }
    { // support_rows_roundtrip_and_missing_falls_back_loudly
        DetectorTable t = D; t.support.present_b = 5.f; t.support.absent_a = 9.f; t.support.present_outlier = 0.01f; t.support.n_trials = 12.f;
        const auto p = std::filesystem::temp_directory_path() / std::format("det_sup_{}.csv", ::getpid());
        save_detectors(p, t, "test");
        auto r = load_detectors(p);
        CHECK(r.support.present_b == 5.f and r.support.absent_a == 9.f and r.support.present_outlier == 0.01f and r.support.n_trials == 12.f);
        CHECK(r.support.present_a == t.support.present_a and r.support.absent_b == t.support.absent_b and r.support.absent_outlier == t.support.absent_outlier);
        const auto tmp = std::filesystem::temp_directory_path() / std::format("det_sup2_{}.csv", ::getpid());
        const auto err = std::filesystem::temp_directory_path() / std::format("det_sup_err_{}.txt", ::getpid());
        const auto loud = [&](const char* body) {
            { std::ofstream o(tmp); o << "camera,param,value,n,source\n" << body; }
            std::fflush(stderr); int keep = dup(2); std::freopen(err.c_str(), "w", stderr);
            const auto m = load_detectors(tmp);
            std::fflush(stderr); dup2(keep, 2); close(keep);
            std::ifstream e(err); std::string s((std::istreambuf_iterator<char>(e)), {});
            return std::pair{m.support, s};
        };
        { // a missing row: compiled defaults, and it says so
          const auto [s, msg] = loud("zed,min_fill,0.1,0,t\nsupport,present_b,7,0,t\nsupport,absent_a,5,0,t\n");
          CHECK(s.present_b == D.support.present_b and s.absent_a == D.support.absent_a);
          CHECK(msg.find("support rows missing") != std::string::npos); }
        { // out of domain (outlier 0 would make the term infinite): defaults, loud
          const auto [s, msg] = loud("support,present_a,1,0,t\nsupport,present_b,4,0,t\nsupport,present_outlier,0,0,t\nsupport,absent_a,12,0,t\nsupport,absent_b,1,0,t\nsupport,absent_outlier,0.8,0,t\n");
          CHECK(s.present_outlier == D.support.present_outlier);
          CHECK(msg.find("out of domain") != std::string::npos); }
        std::filesystem::remove(tmp); std::filesystem::remove(p); std::filesystem::remove(err);
    }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails != 0;
}
