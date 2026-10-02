// Offline validation of the rest mixture's own stationary density (rc::preint::RestDensityLearner) and
// of judging it across cycles (rc::preint::RestWindow) -- the same code the agent runs.
//   g++ -std=c++23 -O2 tools/rest_density_selftest.cpp -o /tmp/rds && /tmp/rds
// Truth is Shadow's measured wheel rest noise, 4.5e-4 m/sqrt(s) on the translation magnitude; the prior
// is the borrowed PreintZuptDensityV (6.39e-3, 14x too wide) -- the configuration that failed the creep test.
#include "../src/rest_density_learner.h"
#include <cstdio>
#include <random>
using rc::preint::RestDensityLearner; using rc::preint::RestWindow;
int failures = 0;
void check(bool c, const char* w) { std::printf("   %-66s %s\n", w, c ? "PASS" : "*** FAIL ***"); if (not c) ++failures; }
int main()
{
    std::mt19937 rng(11); std::normal_distribution<float> N(0.f, 1.f);
    const float T = 0.05f, d_true = 4.5e-4f, prior = 6.39e-3f, vmax = 1.0f, lever = 0.26f;
    const float axis = d_true * std::sqrt(T / 2.f);                // per-axis increment sd at rest
    RestDensityLearner L; L.set_prior(prior); RestWindow W; W.set_window(0.5f);
    // One cycle at forward speed v: returns P(moving) applied to that cycle; learns if asked.
    auto cycle = [&](float v, bool learn) {
        const auto w = W.push(v * T + axis * N(rng), axis * N(rng), 0.f, T, lever);
        const float g = RestDensityLearner::p_moving(w.m_tr, L.density(), w.T, vmax * w.T);
        if (learn) L.add(w.m_tr, w.T, 1.f - g);
        return g; };
    auto mean_gain = [&](float v, int n, bool learn) {
        for (int i = 0; i < 20; ++i) cycle(v, learn);              // let the window fill at this speed
        double s = 0; for (int i = 0; i < n; ++i) s += cycle(v, learn); return s / n; };
    std::printf("1. COLD START: density %.2e (prior %.2e)\n", L.density(), prior);
    check(std::abs(L.density() - prior) < 1e-6f, "starts at the prior, not at zero");
    for (int i = 0; i < 12000; ++i) cycle(0.f, true);
    std::printf("2. AFTER 10 min PARKED: density %.2e (truth %.2e)\n", L.density(), d_true);
    check(std::abs(L.density() - d_true) < 0.2f * d_true, "converges to the true rest density within 20 %");
    std::printf("   parked: mean P(moving) %.3f (the parked pose must hold)\n", mean_gain(0.f, 2000, false));
    check(mean_gain(0.f, 2000, false) < 0.1, "parked increments are suppressed (gain < 0.1)");
    std::printf("3. GAIN through the windowed mixture once learnt:\n");
    for (float v : {0.002f, 0.005f, 0.01f, 0.02f, 0.05f}) std::printf("   %4.1f cm/s  %.3f\n", 100 * v, mean_gain(v, 2000, false));
    check(mean_gain(0.005f, 2000, false) > 0.9, "0.5 cm/s creep passes (gain > 0.9)");
    check(mean_gain(0.01f, 2000, false) > 0.99, "1 cm/s creep passes (gain > 0.99)");
    const float before = L.density();
    mean_gain(0.007f, 6000, true);                                 // 5 min creeping at 0.7 cm/s, learning
    std::printf("4. AFTER 5 min CREEPING at 0.7 cm/s while learning: density %.2e -> %.2e\n", before, L.density());
    check(L.density() < 1.3f * before, "a long slow creep does not widen the rest density (< 1.3x)");
    check(mean_gain(0.01f, 2000, false) > 0.99, "and 1 cm/s still passes afterwards");
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
