// AuRo journal item B (2026-11-23): the room's change belief (change_belief.h).
//   g++ -std=c++23 -O2 -I../src -I/usr/include/eigen3 change_belief_test.cpp -o change_belief_test && ./change_belief_test
#include "change_belief.h"
#include <cmath>
#include <cstdio>
#include <random>

static int failures = 0;
static void check(bool ok, const char *what)
{ if (!ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what); }

int main()
{
    using namespace rc::change;
    const double ln2 = std::log(2.0);
    std::printf("mutual information\n");
    check(mutual_info(0.0, 0.85, 0.05) < 1e-12 and mutual_info(1.0, 0.85, 0.05) < 1e-12, "I = 0 when the change is certain either way");
    double mx = 0.0;
    for (int i = 1; i < 100; ++i) mx = std::max(mx, mutual_info(i / 100.0, 0.85, 0.05));
    check(mx > 0.3 and mx <= ln2 + 1e-12, "maximal inside (0,1), never above ln 2");
    check(std::abs(mutual_info(0.5, 1.0, 0.0) - ln2) < 1e-12, "a perfect test of a fair coin is worth ln 2");
    check(mutual_info(0.3, 0.5, 0.5) < 1e-12, "an uninformative test is worth nothing");

    std::printf("predictive change probability\n");
    ChangeBelief b;
    const double ell = 0.3;
    check(b.p_changed(0.0, ell) == 0.0, "p = 0 at age 0");
    check(b.p_changed(3600.0, ell) < b.p_changed(86400.0, ell) and b.p_changed(86400.0, ell) < b.p_changed(30 * 86400.0, ell),
          "p grows with age");
    check(b.p_changed(1e12, ell) > 0.99, "p -> 1 for a very old observation");
    // the posterior predictive is below the plug-in at the posterior mean (Jensen): E[1-e^{-λx}] ≤ 1-e^{-E[λ]x}
    const double x = ell * ell * 86400.0, plug = 1.0 - std::exp(-b.hazard() * x);
    check(b.p_changed(86400.0, ell) <= plug + 1e-9, "the predictive is no more confident than the plug-in");
    check(std::abs(b.hazard() - 3e-6) / 3e-6 < 0.05, "the grid reproduces the prior mean");

    std::printf("information of a view\n");
    b.init({0.f, 0.f}, 4, 4, 0.5f);
    b.observe(0, 1000.0, false, ell);
    check(b.info(0, 1000.0, ell) < 1e-12, "a cell observed now offers nothing (marginal by construction)");
    check(b.info(0, 1000.0 + 86400.0, ell) > b.info(0, 1000.0 + 3600.0, ell), "older observation, more information");
    check(std::abs(b.info(1, 1000.0, ell) - b.info(0, 1000.0 + b.params().unseen_age_s, ell)) < 1e-12,
          "a never-observed cell is valued at the unseen-age prior");
    const double n = 0.25 / (ell * ell);
    check(b.info(1, 0.0, ell) <= n * ln2 + 1e-12, "at most ln 2 per region");
    std::printf("     a 0.5 m cell last seen 1 h ago: %.4f nats; 1 day ago: %.3f nats (prior λ %.1e /m²/s)\n",
                b.info(0, 1000.0 + 3600.0, ell), b.info(0, 1000.0 + 86400.0, ell), b.hazard());

    std::printf("learning the hazard\n");
    {
        // A home whose true hazard is 10x the prior: cells re-observed every 2 h for 30 days, flips drawn with d, f.
        const double lam_true = 3e-5, d = 0.85, f = 0.05;
        Params p; ChangeBelief lb(p); lb.init({0.f, 0.f}, 10, 10, 0.5f);
        std::mt19937 rng(7); std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<bool> state(100, false);
        for (int k = 0; k < 100; ++k) lb.observe(k, 0.0, false, ell);
        const double dt = 7200.0, regions = 0.25 / (ell * ell);
        for (double t = dt; t <= 30 * 86400.0; t += dt)
            for (int k = 0; k < 100; ++k)
            {
                const bool changed = U(rng) < 1.0 - std::exp(-lam_true * 0.25 * dt);   // a change somewhere in the cell
                const bool shows = changed ? U(rng) < d : U(rng) < f;
                if (shows) state[k] = not state[k];
                lb.observe(k, t, state[k], ell);
            }
        (void)regions;
        std::printf("     prior λ %.1e, true %.1e, learnt %.2e\n", p.hazard_mean, lam_true, lb.hazard());
        check(lb.hazard() > 0.5 * lam_true and lb.hazard() < 2.0 * lam_true, "learnt λ within a factor 2 of the truth");
    }
    {
        // A home where nothing ever changes: only false alarms. λ must fall well below the prior.
        Params p; ChangeBelief lb(p); lb.init({0.f, 0.f}, 10, 10, 0.5f);
        std::mt19937 rng(3); std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<bool> state(100, false);
        for (int k = 0; k < 100; ++k) lb.observe(k, 0.0, false, ell);
        for (double t = 7200.0; t <= 30 * 86400.0; t += 7200.0)
            for (int k = 0; k < 100; ++k)
            {
                if (U(rng) < p.false_alarm) state[k] = not state[k];
                lb.observe(k, t, state[k], ell);
            }
        std::printf("     static home: prior λ %.1e, learnt %.2e\n", p.hazard_mean, lb.hazard());
        check(lb.hazard() < 0.5 * p.hazard_mean, "false alarms alone do not teach a high hazard");
    }
    ChangeBelief c; c.init({0.f, 0.f}, 2, 2, 0.5f);
    const double h0 = c.hazard();
    c.observe(0, 0.0, true, ell); c.observe(1, 0.0, false, ell);
    check(c.hazard() == h0 and c.tests() == 0, "a first observation teaches λ nothing");
    c.observe(0, 3600.0, false, ell);
    const double h1 = c.hazard();
    check(h1 > h0 and c.tests() == 1, "a flip raises λ");
    c.clear_cells();
    check(c.cell(0).t_obs < 0.0 and c.hazard() == h1, "a new room forgets the cells, keeps λ");

    std::printf("observation episodes\n");
    {
        ChangeBelief e; e.init({0.f, 0.f}, 2, 1, 0.5f);
        // cell 0 seen for 10 sweeps (matter in 7), then out of view: one episode, matter = majority
        for (int i = 0; i < 10; ++i) e.sweep({0}, [&](int) { return i < 7; }, 100.0 + i);
        check(e.cell(0).in_view and e.info(0, 105.0, ell) == 0.0, "a cell in view offers nothing");
        e.sweep({1}, [](int) { return false; }, 111.0);
        check(not e.cell(0).in_view and e.cell(0).matter == 1 and e.cell(0).t_obs == 111.0, "leaving view closes one episode, majority state");
        check(e.tests() == 0, "first episodes are not tests");
        // flicker while in view is never a test
        for (int i = 0; i < 50; ++i) e.sweep({1}, [&](int) { return i % 2 == 0; }, 112.0 + i);
        check(e.tests() == 0, "flicker inside one episode is not counted");
        // cell 0 back in view an hour later, now empty: one test, a flip
        const double h = e.hazard();
        e.sweep({0}, [](int) { return false; }, 3700.0);
        e.sweep({}, [](int) { return false; }, 3701.0);
        check(e.tests() == 1 and e.hazard() > h, "a re-observation after leaving view is one test (cell 1's first episode is none); its flip raises λ");
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILED", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
