// ─────────────────────────────────────────────────────────────────────────────────────────────────
// DOES THE ONLINE ESTIMATOR AGREE WITH THE OFFLINE ONE?
//
// build: g++ -std=c++23 -O2 -o scale_estimator_test scale_estimator_test.cpp && ./scale_estimator_test
//
// room_concept/tools/motion_calib.cpp fits e = s·Δ + N(0,σ²T) over a whole log in one pass, and its
// answers are the ones already quoted in the thesis (gyro 1.0114 / 1.0172, wheel 1.1450). The online
// estimator has to reproduce those on the same data before it is allowed to claim anything, and it
// has to behave sensibly in the three regimes the offline fit never sees: no excitation, drift, and a
// scale that changes under it.
//
// ★A TEST THAT CANNOT FAIL PROVES NOTHING — the lesson this project has paid for three times today
// (an unbounded CHOOSE that generated 0 states, a grep that read TLC's error as a verdict, a TLA+
// model whose producer always armed the same cell). So each case below states the number it expects
// and why, and the exercise-free case asserts the estimator says "I do not know" rather than zero.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "scale_estimator.h"

#include <cstdio>
#include <random>
#include <vector>

using rc::calib::ScaleEstimator;

namespace
{
int failures = 0;

void check(bool ok, const char *what, const char *why = "")
{
    if (not ok) { ++failures; std::printf("  ✗ %s  %s\n", what, why); }
    else        std::printf("  ✓ %s\n", what);
}

/// The batch fit, transcribed from motion_calib.cpp — the reference this must match.
struct Batch { double s, sigma; };
Batch batch_fit(const std::vector<double> &d, const std::vector<double> &e, const std::vector<double> &T)
{
    double sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < d.size(); ++i) { const double w = 1.0 / T[i]; sxx += w*d[i]*d[i]; sxy += w*d[i]*e[i]; }
    const double s = sxy / sxx;
    double ss = 0;
    for (std::size_t i = 0; i < d.size(); ++i) { const double r = e[i] - s*d[i]; ss += r*r / T[i]; }
    return {s, std::sqrt(ss / static_cast<double>(d.size() - 1))};
}
}   // namespace

int main()
{
    std::printf("── the online scale estimator, against the offline fit ──\n\n");
    std::mt19937 rng(20260820);
    std::normal_distribution<double> g(0.0, 1.0);

    // ── 1. SAME DATA, SAME ANSWER ───────────────────────────────────────────────────────────────
    // A run of windows with a true scale of 0.07 and a density of 0.03. The online estimator is fed
    // the windows one at a time, as it would be live; the batch fit sees them all at once.
    {
        const double s_true = 0.07, sigma_true = 0.03;
        std::vector<double> d, e, T;
        for (int i = 0; i < 400; ++i)
        {
            const double Ti = 0.5 + 0.5 * (i % 4);
            const double di = 0.2 + 0.9 * ((i * 37) % 100) / 100.0;      // spread of increments
            d.push_back(di); T.push_back(Ti);
            e.push_back(s_true * di + sigma_true * std::sqrt(Ti) * g(rng));
        }
        const auto b = batch_fit(d, e, T);
        // ★Prior OFF for this comparison (a huge prior_std) — the batch fit has no prior, so leaving
        // ours in would be comparing two different estimators and calling the difference an error.
        ScaleEstimator on({.scale_walk_density = 0.0, .prior_std = 1e6});
        for (std::size_t i = 0; i < d.size(); ++i) on.add(d[i], e[i], T[i]);
        const auto c = on.posterior();
        std::printf("  batch  s = %.5f  sigma = %.5f\n  online s = %.5f  sigma = %.5f  (s_std %.5f, %d windows)\n",
                    b.s, b.sigma, c.s, c.sigma, c.s_std, c.windows);
        check(std::abs(c.s - b.s) < 1e-9, "scale matches the batch fit to numerical precision");
        check(std::abs(c.sigma - b.sigma) < 1e-9, "density matches the batch fit");
        check(std::abs(c.s - s_true) < 3 * c.s_std, "and the truth is inside three posterior sigmas",
              "the fit is right but its own error bar disagrees");
        check(c.identifiable(), "reports itself identifiable when well excited");
    }

    // ── 2. NO EXCITATION IS NOT A ZERO SCALE ────────────────────────────────────────────────────
    // The channel is never exercised: Δ ≈ 0 throughout. The honest answer is "I do not know", which
    // means a wide posterior — NOT a confident 0. This is the case a naive implementation gets wrong
    // and then reports as a perfectly calibrated robot.
    {
        // ★THE NOISE FLOOR DOES NOT SHRINK WITH THE INCREMENT. A first version of this case paired
        // 1e-4 rad of motion with 1e-5 of noise — a sensor 30x quieter than the real gyro — and the
        // slope was then genuinely identifiable, so the estimator was right and the TEST was wrong.
        // sigma_omega is a density: 0.0447 rad/sqrt(s), fixed, whatever the robot does. Against that,
        // 1e-4 rad of excitation carries no information about a scale, which is the whole point.
        ScaleEstimator on({.scale_walk_density = 0.0, .prior_std = 0.15});
        for (int i = 0; i < 200; ++i) on.add(1e-4, 1e-4 * 0.07 + 0.0447 * g(rng), 1.0);
        const auto c = on.posterior();
        std::printf("\n  unexcited: s = %.4f  s_std = %.4f  info = %.3e  span = %.2e\n",
                    c.s, c.s_std, c.info, c.span);
        check(c.s_std > 0.10, "an unexercised channel stays near its PRIOR width",
              "it claims to know a scale it has no evidence for");
        check(not c.identifiable(), "and reports itself NOT identifiable");
    }

    // ── 3. THE SCALE DRIFTS, AND THE ESTIMATOR HAS TO FOLLOW ────────────────────────────────────
    // Half the run at 0.05, half at 0.12 — a payload change, a tyre. With a random-walk density the
    // posterior can move; without one it is anchored by the first half for ever.
    {
        auto run = [&](double walk)
        {
            ScaleEstimator on({.scale_walk_density = walk, .prior_std = 0.15});
            for (int i = 0; i < 600; ++i)
            {
                const double s_true = (i < 300) ? 0.05 : 0.12;
                const double di = 0.5 + 0.5 * ((i * 17) % 10) / 10.0;
                on.predict(1.0);
                on.add(di, s_true * di + 0.01 * g(rng), 1.0);
            }
            return on.posterior().s;
        };
        const double frozen = run(0.0), tracking = run(3e-3);
        std::printf("\n  after a step 0.05 -> 0.12:  no-walk s = %.4f | with-walk s = %.4f\n", frozen, tracking);
        check(std::abs(tracking - 0.12) < std::abs(frozen - 0.12),
              "the random walk lets the estimate follow a scale that really changed");
    }

    // ── 4. TIME WITHOUT EVIDENCE WIDENS THE POSTERIOR ───────────────────────────────────────────
    // This is what makes a calibration affordance ASK: a channel that has not been exercised for a
    // long time is one the robot no longer knows, and the widening is what says so.
    {
        ScaleEstimator on({.scale_walk_density = 1e-3, .prior_std = 0.15});
        for (int i = 0; i < 200; ++i) on.add(0.8, 0.8 * 0.07 + 0.01 * g(rng), 1.0);
        const double sharp = on.posterior().s_std;
        for (int i = 0; i < 600; ++i) on.predict(1.0);       // ten minutes of nothing
        const double stale = on.posterior().s_std;
        std::printf("\n  s_std: %.5f sharp -> %.5f after 600 s idle\n", sharp, stale);
        check(stale > sharp, "an idle channel's posterior widens on its own",
              "nothing would ever ask for a calibration manoeuvre");
    }

    // ── 5. THE GAIN IS MARGINAL, AND IT SELF-EXTINGUISHES ───────────────────────────────────────
    // The affordance must be priced on what a deliberate manoeuvre adds BEYOND the free data the
    // robot's ordinary tours already deliver, and it must stop being worth doing as the posterior
    // sharpens. Both are properties of expected_information_gain, so both are checked here.
    {
        ScaleEstimator fresh({.scale_walk_density = 0.0, .prior_std = 0.15});
        const double g_alone   = fresh.expected_information_gain(1.5, 3.0, 0.0);
        const double g_on_top  = fresh.expected_information_gain(1.5, 3.0, 3.0);
        ScaleEstimator sharp({.scale_walk_density = 0.0, .prior_std = 0.15});
        for (int i = 0; i < 500; ++i) sharp.add(1.0, 0.07 + 0.01 * g(rng), 1.0);
        const double g_sharp = sharp.expected_information_gain(1.5, 3.0, 0.0);
        std::printf("\n  EIG(nats): fresh %.4f | same manoeuvre after busy tours %.4f | when converged %.4f\n",
                    g_alone, g_on_top, g_sharp);
        check(g_on_top < g_alone, "free data from other tours REDUCES what the manoeuvre is worth");
        check(g_sharp < g_alone, "and a converged posterior makes it worth almost nothing",
              "it would keep asking for ever");
    }

    std::printf("\n%d check(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
