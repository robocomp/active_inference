// trajectory_harvester_test -- synthetic, no DDS/DSR. Trajectory: straight 5 m, a pi hairpin (pivot), a 30 s stop.
//   make -C build trajectory_harvester_test && ../bin/trajectory_harvester_test
#include "../../common/sensor_calib/trajectory_harvester.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace sc::calib;

static int g_fail = 0;
#define CHECK(c, ...) do { if (not (c)) { ++g_fail; std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static constexpr std::int64_t T0 = 1'000'000;   // ms; never 0

struct Phase { double t0, t1, adv, rot; };
// straight 10 s @ 0.5 m/s = 5 m; hairpin pi @ 1 rad/s; stop 30 s
static const double kPi = std::numbers::pi;
static std::vector<Phase> phases()
{ return { {0.0, 10.0, 0.5, 0.0}, {10.0, 10.0 + kPi, 0.0, 1.0}, {10.0 + kPi, 40.0 + kPi, 0.0, 0.0} }; }

struct Run { std::vector<Pair> pairs; EpisodeSummary ep; long kf = 0; };

// stalled_from/to (s): keyframes offered inside are flagged stalled. Odometry keeps flowing.
static Run run(double spacing, double stall_from = -1, double stall_to = -1)
{
    HarvesterConfig cfg; cfg.keyframe_spacing_s = spacing;
    TrajectoryHarvester h(cfg);
    const auto ph = phases();
    const double total = ph.back().t1;
    for (int ms = 0; ms <= int(std::llround(total * 1000.0)); ms += 50)   // 20 Hz odometry, 10 Hz sweeps (integer ms)
    {
        const double t = ms * 1e-3;
        const auto it = std::find_if(ph.begin(), ph.end(), [&](const Phase& p) { return t >= p.t0 - 1e-9 and t < p.t1; });
        const Phase p = it == ph.end() ? ph.back() : *it;
        h.add_odometry(T0 + ms, t < p.t1 ? p.adv : 0.0, 0.0, p.rot);
        if (ms % 100 == 0)
            h.add_keyframe(T0 + ms, stall_from >= 0 and t >= stall_from and t <= stall_to);
    }
    h.finish();
    Run r; r.pairs = h.take_pairs(); r.ep = h.close_episode();
    return r;
}

static double mid_s(const Pair& p) { return 0.5 * double(p.stamp_a_ms + p.stamp_b_ms - 2 * T0) / 1000.0; }

static Eigen::Vector3d plant() { return {0.03, -0.02, 1.84 * kPi / 180.0}; }

static MountEvidence evidence(const std::vector<Pair>& ps, bool noisy, unsigned seed = 7)
{
    MountEvidence ev{HarvesterConfig{}};
    std::mt19937 g(seed); std::normal_distribution<double> n(0.0, 1.0);
    for (const auto& p : ps)
    {
        Eigen::Vector2d y = p.J * plant();
        if (noisy) { y(0) += std::sqrt(p.var(0)) * n(g); y(1) += std::sqrt(p.var(1)) * n(g); }
        ev.add(p, y(0), y(1));
    }
    return ev;
}

int main()
{
    const Run base = run(0.5);
    std::printf("base: %zu pairs, %ld keyframes\n", base.pairs.size(), base.ep.keyframes);
    for (const auto& p : base.pairs)
        std::printf("  pair kf %llu-%llu  t %.1f..%.1f  dth %.3f d_fwd %.3f  psi-info %.3g  lever-info %.3g\n",
                    (unsigned long long)p.id_a, (unsigned long long)p.id_b, double(p.stamp_a_ms - T0) / 1e3,
                    double(p.stamp_b_ms - T0) / 1e3, p.dth, p.d_fwd, p.info(2, 2), p.info(0, 0) + p.info(1, 1));
    CHECK(not base.pairs.empty(), "no pairs");

    // 1. information shares
    double psi_tot = 0, psi_str = 0, lev_tot = 0, lev_hp = 0, lev_max = 0;
    for (const auto& p : base.pairs)
    {
        const double m = mid_s(p), lev = p.info(0, 0) + p.info(1, 1);
        psi_tot += p.info(2, 2); lev_tot += lev; lev_max = std::max(lev_max, lev);
        if (m < 10.0) psi_str += p.info(2, 2);
        if (m >= 10.0 and m < 10.0 + kPi) lev_hp += lev;
    }
    std::printf("psi share from straight %.4f   lever share from hairpin %.4f   biggest single lever pair %.4f\n",
                psi_str / psi_tot, lev_hp / lev_tot, lev_max / lev_tot);
    CHECK(psi_tot > 0 and psi_str / psi_tot >= 0.9, "yaw share %.3f", psi_str / psi_tot);
    CHECK(lev_tot > 0 and lev_hp / lev_tot >= 0.9, "lever share %.3f", lev_hp / lev_tot);
    CHECK(lev_tot > 0 and lev_max / lev_tot >= 0.9, "hairpin not one pair: biggest %.3f", lev_max / lev_tot);

    // no keyframe in two pairs
    for (std::size_t i = 0; i < base.pairs.size(); ++i)
        for (std::size_t j = i + 1; j < base.pairs.size(); ++j)
        {
            const auto &a = base.pairs[i], &b = base.pairs[j];
            CHECK(a.id_b < b.id_a or b.id_b < a.id_a, "pairs %zu,%zu share keyframes", i, j);
        }

    // 2. planted mount error recovered within 2 sigma (noisy measurement)
    {
        const auto ev = evidence(base.pairs, true);
        const Eigen::Vector3d m = ev.mean(), s = ev.sigma(), t = plant();
        std::printf("noisy: mean (%.4f %.4f %.4f deg) sigma (%.4f %.4f %.4f deg)\n", m(0), m(1), m(2) * 180 / kPi,
                    s(0), s(1), s(2) * 180 / kPi);
        for (int k = 0; k < 3; ++k) CHECK(std::abs(m(k) - t(k)) < 2.0 * s(k), "dof %d err %.5f sigma %.5f", k, m(k) - t(k), s(k));
    }

    // 3. keyframe spacing changes no posterior mean by 0.1 sigma (noise-free: the mean is then deterministic)
    {
        const auto e0 = evidence(base.pairs, false);
        for (const double sp : {0.25, 1.0})
        {
            const auto r = run(sp);
            const auto e1 = evidence(r.pairs, false);
            const Eigen::Vector3d d = (e1.mean() - e0.mean()).cwiseAbs().cwiseQuotient(e0.sigma());
            std::printf("spacing %.2f: %zu pairs, |dmean|/sigma = (%.4f %.4f %.4f)\n", sp, r.pairs.size(), d(0), d(1), d(2));
            for (int k = 0; k < 3; ++k) CHECK(d(k) < 0.1, "spacing %.2f dof %d shift %.3f sigma", sp, k, d(k));
        }
    }

    // 4. a stalled interval contributes nothing: no pair has an end inside it or spans it
    {
        const double a = 3.0, b = 6.0;
        const auto r = run(0.5, a, b);
        CHECK(r.ep.dropped_stalled > 0, "nothing was dropped");
        for (const auto& p : r.pairs)
        {
            const double ta = double(p.stamp_a_ms - T0) / 1e3, tb = double(p.stamp_b_ms - T0) / 1e3;
            CHECK(tb < a or ta > b, "pair %.1f..%.1f overlaps the stalled interval", ta, tb);
        }
        // the same run with the interval removed from the data can only have less yaw information than the full run
        double psi = 0; for (const auto& p : r.pairs) psi += p.info(2, 2);
        CHECK(psi < psi_tot, "stalled run has as much yaw information as the full run");
    }

    // 5. a stop alone is worth exactly nothing
    {
        HarvesterConfig cfg; TrajectoryHarvester h(cfg);
        for (double t = 0; t <= 40; t += 0.05) { h.add_odometry(T0 + std::llround(t * 1000), 0, 0, 0);
            if (std::fmod(std::llround(t * 1000.0), 100) == 0) h.add_keyframe(T0 + std::llround(t * 1000), false); }
        h.finish();
        CHECK(h.take_pairs().empty() and h.information().norm() == 0.0, "stop produced information");
    }

    std::printf(g_fail ? "trajectory_harvester_test: %d FAILED\n" : "trajectory_harvester_test: ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
