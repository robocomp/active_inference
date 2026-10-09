// Offline validation of rc::calib::WheelNoiseLearner: plant a motion-dependent wheel density, feed
// wheel-minus-gyro differences generated with it, and check the learner recovers it -- and that a cold
// start sits on its prior instead of claiming perfect wheels.
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/wheel_noise_selftest.cpp -o /tmp/wns && /tmp/wns
#include "../src/wheel_noise_learner.h"
#include <cstdio>
#include <random>
using rc::calib::WheelNoiseLearner;
int failures = 0;
void check(bool c, const char* w) { std::printf("   %-62s %s\n", w, c ? "PASS" : "*** FAIL ***"); if (not c) ++failures; }
int main()
{
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double c0 = 0.0007 * 0.0007, c1 = 2.4e-4, c2 = 7.6e-4, sg = 0.0021, dt = 0.02;
    {
        WheelNoiseLearner L; L.set_prior_density(0.0037f);
        std::printf("1. COLD START: density at rest before any data = %.4f (prior 0.0037)\n", L.density(0, 0));
        check(std::abs(L.density(0, 0) - 0.0037f) < 1e-5, "cold start reports the PRIOR density, not zero");
    }
    {
        WheelNoiseLearner L; L.set_prior_density(0.0037f);
        for (int i = 0; i < 25000; ++i)
        {
            const double r = U(rng);
            const double v = r < 0.3 ? 0.0 : (r < 0.7 ? 0.35 : 0.1);
            const double w = U(rng) < 0.5 ? 0.0 : (U(rng) < 0.5 ? 0.15 : 0.9);
            const double var = (sg * sg + c0 + c1 * v + c2 * w) * dt;
            L.add(static_cast<float>(std::sqrt(var) * N(rng)), dt, v, w, sg);
        }
        const auto c = L.coeffs();
        std::printf("2. RECOVERY over %ld segments: c0 %.2e (truth %.2e)  c1 %.2e (%.2e)  c2 %.2e (%.2e)\n",
                    L.samples(), c[0], c0, c[1], c1, c[2], c2);
        std::printf("   density at v=.35 w=0: %.4f (truth %.4f)   at v=0 w=.9: %.4f (truth %.4f)\n",
                    L.density(0.35f, 0.f), std::sqrt(c0 + c1 * 0.35), L.density(0.f, 0.9f), std::sqrt(c0 + c2 * 0.9));
        check(std::abs(c[1] - c1) < 0.15 * c1, "speed term recovered within 15%");
        check(std::abs(c[2] - c2) < 0.15 * c2, "turn-rate term recovered within 15%");
        check(L.density(0.f, 0.f) < 0.0015f, "rest density driven well below the pessimistic prior");
    }
    {
        WheelNoiseLearner L; L.set_prior_density(0.0037f);
        for (int i = 0; i < 2000; ++i) L.add(0.01f, dt, 0.3f, 0.3f, -1.f);   // gyro states no variance
        check(L.samples() == 0, "no anchor (gyro density unstated) => no samples taken");
    }
    {
        // The live failure of 2026-10-01: drive, then stand still for far longer than any window.
        WheelNoiseLearner L; L.set_prior_density(0.0037f); L.set_motion_prior_density(0.0447f);
        auto feed = [&](double v, double w) {
            const double var = (sg * sg + c0 + c1 * v + c2 * w) * dt;
            L.add(static_cast<float>(std::sqrt(var) * N(rng)), dt, v, w, sg);
        };
        for (int i = 0; i < 20000; ++i) { const double r = U(rng); feed(r < 0.5 ? 0.35 : 0.0, r < 0.5 ? 0.0 : 0.9); }
        const auto before = L.coeffs();
        for (int i = 0; i < 120000; ++i) feed(0.0, 0.0);               // 40 min parked at 50 Hz
        const auto after = L.coeffs();
        std::printf("3. LONG STOP after driving: c1 %.2e -> %.2e, c2 %.2e -> %.2e (truth %.2e, %.2e)\n",
                    before[1], after[1], before[2], after[2], c1, c2);
        check(std::abs(after[1] - c1) < 0.2 * c1, "speed term survives a 40-minute stop");
        check(std::abs(after[2] - c2) < 0.2 * c2, "turn-rate term survives a 40-minute stop");
    }
    {
        WheelNoiseLearner L; L.set_prior_density(0.0037f); L.set_motion_prior_density(0.0447f);
        for (int i = 0; i < 30000; ++i) L.add(static_cast<float>(sg * std::sqrt(dt) * N(rng)), dt, 0.0, 0.0, sg);
        std::printf("4. NEVER MOVED: density at v=.3 %.4f, at w=.5 %.4f (motion prior 0.0447)\n",
                    L.density(0.3f, 0.f), L.density(0.f, 0.5f));
        check(L.density(0.3f, 0.f) > 0.02f and L.density(0.f, 0.5f) > 0.02f,
              "unexcited motion terms stay at the conservative prior, not at zero");
    }
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
