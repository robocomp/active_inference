// Offline validation of RestMotionChannel (one per body-frame channel: forward, lateral, rotation): a
// velocity Kalman filter whose odometry noise is learnt from its own innovations, and a Savage-Dickey
// test of v = 0 -- the code the agent runs. Translation gain = 1 - (1-p_fwd)(1-p_lat); rotation = p_rot.
//   g++ -std=c++23 -O2 tools/rest_density_selftest.cpp -o /tmp/rds && /tmp/rds
// It REPLAYS THE LIVE FAILURE of 2026-10-02 20:56: a short parked warm-up (30 s) from the borrowed
// 14x-wide prior, then the registered creep schedule, learning throughout.
#include "../src/rest_density_learner.h"
#include <cstdio>
#include <random>
#include <vector>
using namespace rc::preint;
int failures = 0;
void check(bool c, const char* w) { std::printf("   %-70s %s\n", w, c ? "PASS" : "*** FAIL ***"); if (not c) ++failures; }
int main()
{
    std::mt19937 rng(5); std::normal_distribution<float> N(0.f, 1.f);
    const float T = 0.05f, d_lin = 4.5e-4f, d_rot = 6.4e-4f, prior_lin = 6.39e-3f, prior_rot = 1.02e-2f;
    RestMotionChannel Cf, Cl, Cr;
    { RestMotionChannel::Params pr; pr.v0 = 1.15f; Cr.set_params(pr); }
    // the configured |translation| density spread over two axes, as the agent does
    Cf.set_prior(prior_lin / std::sqrt(2.f)); Cl.set_prior(prior_lin / std::sqrt(2.f)); Cr.set_prior(prior_rot);
    struct Seg { float v, w, secs; };
    // returns mean P(moving) over the segment after its first `skip` seconds
    auto run = [&](float v, float w, float secs, float skip, std::vector<float>* trace = nullptr) {
        double s = 0; int n = 0; const int cycles = int(secs / T);
        for (int i = 0; i < cycles; ++i) {
            const float sx = d_lin * std::sqrt(T / 2.f), sr = d_rot * std::sqrt(T);
            const float dx = v * T + sx * N(rng), dy = sx * N(rng), dth = w * T + sr * N(rng);
            const float pf = Cf.step(dx, T), pl = Cl.step(dy, T), pr = Cr.step(dth, T);
            const float pm = (w != 0.f) ? pr : 1.f - (1.f - pf) * (1.f - pl);   // the gain this leg is judged by
            if (trace) trace->push_back(pm);
            if (i * T >= skip) { s += pm; ++n; }
        }
        return n ? s / n : 0.0; };
    std::printf("1. SHORT WARM-UP: 30 s parked from the 14x-wide prior\n");
    const double park0 = run(0, 0, 30, 5);
    std::printf("   parked P(moving) %.3f   learnt density lin %.2e rot %.2e (truth %.1e, %.1e)\n", park0, (Cf.density()*std::sqrt(2.f)), Cr.density(), d_lin, d_rot);
    std::printf("2. THE REGISTERED CREEP SCHEDULE, learning throughout (mean P(moving) after the first 1 s of each leg)\n");
    float worst_low = 1.f; double worst_density = 0;
    for (float v : {0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.08f}) {
        const float secs = std::min(15.f, 0.6f / v);
        const double g1 = run(v, 0, secs, 1.0); run(0, 0, 5, 0); const double g2 = run(-v, 0, secs, 1.0); run(0, 0, 5, 0);
        std::printf("   %4.1f cm/s  fwd %.3f  back %.3f   density lin %.2e\n", 100 * v, g1, g2, (Cf.density()*std::sqrt(2.f)));
        if (v >= 0.01f) worst_low = std::min<float>(worst_low, std::min(g1, g2));
        worst_density = std::max<double>(worst_density, (Cf.density()*std::sqrt(2.f)));
    }
    for (float w : {0.01f, 0.02f, 0.05f}) {
        const double g1 = run(0, w, 15, 1.0); run(0, 0, 5, 0); const double g2 = run(0, -w, 15, 1.0); run(0, 0, 5, 0);
        std::printf("   %.2f rad/s  ccw %.3f  cw %.3f   density rot %.2e\n", w, g1, g2, Cr.density());
    }
    check(worst_low > 0.9, "every leg at >= 1 cm/s passes after its first second (P(moving) > 0.9)");
    check(worst_density < 3 * d_lin, "the learnt rest density never runs away (< 3x truth at any point)");
    std::printf("3. PARKED AGAIN, 2 min\n");
    const double park1 = run(0, 0, 120, 5);
    std::printf("   parked P(moving) %.3f   density lin %.2e\n", park1, (Cf.density()*std::sqrt(2.f)));
    check(park1 < 0.05, "parked increments are suppressed (P(moving) < 0.05)");
    check(std::abs((Cf.density()*std::sqrt(2.f)) - d_lin) < 0.3f * d_lin, "and the rest density sits within 30 % of the truth");
    std::printf("4. TRANSITIONS: onset at 5 cm/s from rest, then a stop\n");
    std::vector<float> on, off; run(0, 0, 10, 0); run(0.05f, 0, 2, 0, &on); run(0, 0, 3, 0, &off);
    int k_on = 0; while (k_on < int(on.size()) and on[k_on] < 0.5f) ++k_on;
    int k_off = 0; while (k_off < int(off.size()) and off[k_off] > 0.1f) ++k_off;
    std::printf("   P(moving) > 0.5 after %d cycles (%.2f s); < 0.1 after stopping in %d cycles (%.2f s)\n", k_on, k_on * T, k_off, k_off * T);
    check(k_on * T <= 0.25f, "motion is recognised within 0.25 s of onset");
    check(k_off * T <= 1.5f, "rest is recognised within 1.5 s of stopping");
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
