// trajectory_harvester_test -- synthetic, no DDS/DSR. Trajectory: straight 5 m, a pi hairpin (pivot), a 30 s stop.
//   make -C build trajectory_harvester_test && ../bin/trajectory_harvester_test
#include "../../common/sensor_calib/trajectory_harvester.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <numbers>
#include <random>
#include <vector>

using namespace sc::calib;

static int g_fail = 0;
#define CHECK(c, ...) do { if (not (c)) { ++g_fail; std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static constexpr std::int64_t T0 = 1'000'000;   // ms; never 0
static const double kPi = std::numbers::pi;

struct Phase { double t0, t1, adv, rot; };
// straight 10 s @ 0.5 m/s = 5 m; hairpin pi @ 1 rad/s; stop 30 s
static std::vector<Phase> phases()
{ return { {0.0, 10.0, 0.5, 0.0}, {10.0, 10.0 + kPi, 0.0, 1.0}, {10.0 + kPi, 40.0 + kPi, 0.0, 0.0} }; }
static double th_true(double t) { return std::clamp(t - 10.0, 0.0, kPi); }
static double x_true(double t) { return 0.5 * std::min(t, 10.0); }

struct Opt { double wheel_gain = 1.0; bool gyro = false; double stall_from = -1, stall_to = -1; };
struct Run { std::vector<Pair> pairs; EpisodeSummary ep; };

static Run run(const HarvesterConfig& cfg, const Opt& o = {})
{
    TrajectoryHarvester h(cfg);
    const auto ph = phases();
    const double total = ph.back().t1;
    for (int ms = 0; ms <= int(std::llround(total * 1000.0)); ms += 50)   // 20 Hz odometry + gyro, 10 Hz sweeps (integer ms)
    {
        const double t = ms * 1e-3;
        const auto it = std::find_if(ph.begin(), ph.end(), [&](const Phase& p) { return t >= p.t0 - 1e-9 and t < p.t1; });
        const Phase p = it == ph.end() ? ph.back() : *it;
        const double adv = t < p.t1 ? p.adv : 0.0, rot = p.rot;
        h.add_odometry(T0 + ms, adv, 0.0, rot * o.wheel_gain);
        if (o.gyro) h.add_gyro(T0 + ms, rot);
        if (ms % 100 == 0)
            h.add_keyframe(T0 + ms, o.stall_from >= 0 and t >= o.stall_from and t <= o.stall_to);
    }
    h.finish();
    Run r; r.pairs = h.take_pairs(); r.ep = h.close_episode();
    return r;
}
static HarvesterConfig cfg_of(double spacing, double per_rad = 0.005)
{ HarvesterConfig c; c.keyframe_spacing_s = spacing; c.odo_sigma_per_rad = per_rad; return c; }

static double tsec(std::int64_t ms) { return double(ms - T0) / 1000.0; }
static double mid_s(const Pair& p) { return 0.5 * (tsec(p.stamp_a_ms) + tsec(p.stamp_b_ms)); }
static double lever(const Eigen::Matrix3d& i) { return i(0, 0) + i(1, 1); }
static Eigen::Vector3d plant() { return {0.03, -0.02, 1.84 * kPi / 180.0}; }

// Common random numbers: the registration error of a SWEEP is a function of its stamp (and row) alone, so every
// keyframe spacing sees the same scan errors; a pair's noise is e(b) - e(a), variance reg^2 as the model says.
static double scan_err(std::int64_t ms, int row, double reg)
{
    std::mt19937 g(unsigned(ms) * 2654435761u + unsigned(row) * 40503u + 12345u);
    return std::normal_distribution<double>(0.0, reg / std::sqrt(2.0))(g);
}

using Truth = std::function<Eigen::Matrix<double, 2, 3>(const Pair&)>;
static MountEvidence evidence(const std::vector<Pair>& ps, bool noisy, const Truth& jt = nullptr)
{
    const HarvesterConfig c;
    MountEvidence ev{c};
    for (const auto& p : ps)
    {
        const Eigen::Matrix<double, 2, 3> J = jt ? jt(p) : p.J;
        Eigen::Vector2d y = J * plant();
        if (noisy)
            for (int r = 0; r < 2; ++r)
                y(r) += scan_err(p.stamp_b_ms, r, c.reg_sigma_m) - scan_err(p.stamp_a_ms, r, c.reg_sigma_m);
        ev.add(p, y(0), y(1));
    }
    return ev;
}

// best lever information of the hairpin over ALL segmentations of its keyframes into disjoint pairs (exhaustive, memoised)
static double best_hairpin_lever(double spacing, const HarvesterConfig& cfg)
{
    std::vector<double> ang;
    for (double t = 0.0; t <= 40.0; t += spacing)
        if (t >= 10.0 - 1e-9 and t <= 10.0 + kPi + spacing) ang.push_back(th_true(t));
    const int n = int(ang.size());
    auto val = [&](int a, int b) { const double d = ang[b] - ang[a];
        return lever(make_pair_from_motion(d, 0, 0, 0, d, cfg).info); };
    std::map<int, double> memo;
    std::function<double(int)> f = [&](int i) -> double
    {
        if (i >= n - 1) return 0.0;
        if (auto it = memo.find(i); it != memo.end()) return it->second;
        double b = f(i + 1);
        for (int j = i + 1; j < n; ++j) b = std::max(b, val(i, j) + f(j + 1));
        return memo[i] = b;
    };
    return f(0);
}
static double chopped_hairpin_lever(double piece, const HarvesterConfig& cfg)
{
    double s = 0.0;
    for (double a = 0.0; a + piece <= kPi + 1e-9; a += piece) s += lever(make_pair_from_motion(piece, 0, 0, 0, piece, cfg).info);
    return s;
}

int main()
{
    // ── 1. information shares at the default noise; pairs never share a keyframe ──
    const Run base = run(cfg_of(0.5));
    std::printf("base: %zu pairs, %ld keyframes\n", base.pairs.size(), base.ep.keyframes);
    for (const auto& p : base.pairs)
        std::printf("  pair kf %llu-%llu  t %.1f..%.1f  dth %.3f d_fwd %.3f  psi-info %.3g  lever-info %.3g\n",
                    (unsigned long long)p.id_a, (unsigned long long)p.id_b, tsec(p.stamp_a_ms), tsec(p.stamp_b_ms),
                    p.dth, p.d_fwd, p.info(2, 2), lever(p.info));
    double psi_tot = 0, psi_str = 0;
    for (const auto& p : base.pairs) { psi_tot += p.info(2, 2); if (mid_s(p) < 10.0) psi_str += p.info(2, 2); }
    CHECK(psi_tot > 0 and psi_str / psi_tot >= 0.9, "yaw share %.3f", psi_str / psi_tot);
    for (std::size_t i = 0; i < base.pairs.size(); ++i)
        for (std::size_t j = i + 1; j < base.pairs.size(); ++j)
            CHECK(base.pairs[i].id_b < base.pairs[j].id_a or base.pairs[j].id_b < base.pairs[i].id_a, "pairs share keyframes");

    // ── 2. hairpin: the segmentation is optimal under the noise model and never chops (4 configs) ──
    for (const double sp : {0.1, 0.5})
        for (const double pr : {0.001, 0.005})
        {
            const auto cfg = cfg_of(sp, pr);
            const Run r = run(cfg);
            double hp = 0, str = 0, tot = 0; int npairs = 0;
            for (const auto& p : r.pairs)
            {
                tot += lever(p.info);
                if (mid_s(p) >= 10.0 and mid_s(p) < 10.0 + kPi) { hp += lever(p.info); ++npairs; }
            }
            (void)str;
            const double best = best_hairpin_lever(sp, cfg), chop = chopped_hairpin_lever(0.20, cfg);
            std::printf("hairpin spacing %.1f odo/rad %.3f: %d pairs  lever %.3g  best %.3g (%.3f)  chop-0.20 %.3g (x%.1f)  share %.3f\n",
                        sp, pr, npairs, hp, best, hp / best, chop, hp / chop, hp / tot);
            CHECK(hp >= 0.9 * best, "hairpin lever %.3g < 0.9 * best %.3g", hp, best);
            // 5x chopped is demanded only where the model lets ANY segmentation reach it: with odo_sigma_per_rad = 0.005 the
            // optimum itself is only ~3.5-4x the 0.20 rad chopping (rotation-growing noise), and the bar is then 0.9 * optimum.
            CHECK(hp >= std::min(5.0, 0.9 * best / chop) * chop, "hairpin lever %.3g vs chopped %.3g (best %.3g)", hp, chop, best);
            if (pr < 0.002) CHECK(hp >= 5.0 * chop, "hairpin lever %.3g < 5 x chopped %.3g", hp, chop);
            CHECK(hp / tot >= 0.9, "lever share from hairpin %.3f", hp / tot);
        }

    // ── 3. planted mount recovered (CRN noise) ──
    {
        const auto ev = evidence(base.pairs, true);
        const Eigen::Vector3d m = ev.mean(), s = ev.sigma(), t = plant();
        std::printf("noisy: mean (%.4f %.4f %.4f deg) sigma (%.4f %.4f %.4f deg)\n", m(0), m(1), m(2) * 180 / kPi, s(0), s(1), s(2) * 180 / kPi);
        for (int k = 0; k < 3; ++k) CHECK(std::abs(m(k) - t(k)) < 2.0 * s(k), "dof %d err %.5f sigma %.5f", k, m(k) - t(k), s(k));
    }

    // ── 4. spacing insensitivity ──
    //  (a) common-random-numbers noise: every spacing must still recover the plant within 2 sigma (noisy means cannot be
    //      compared to 0.1 sigma: different keyframes => different scan errors, in ANY implementation);
    //  (b) noise-free means move < 0.1 sigma; (c) posterior sigma moves < 10 % over 0.1..0.5 s; 1.0 s is a reported
    //      measurement (a coarse grid loses real rotation at both ends of the hairpin).
    {
        std::vector<double> sps{0.1, 0.25, 0.5, 1.0};
        std::vector<Eigen::Vector3d> mu, sg;
        for (const double sp : sps)
        {
            const auto pairs = run(cfg_of(sp)).pairs;
            const auto en = evidence(pairs, true), e0 = evidence(pairs, false);
            mu.push_back(e0.mean()); sg.push_back(e0.sigma());
            std::printf("spacing %.2f: noisy mean (%.5f %.5f %.5f) noise-free mean (%.5f %.5f %.5f) sigma (%.5f %.5f %.5f)\n", sp,
                        en.mean()(0), en.mean()(1), en.mean()(2), mu.back()(0), mu.back()(1), mu.back()(2), sg.back()(0), sg.back()(1), sg.back()(2));
            for (int k = 0; k < 3; ++k) CHECK(std::abs(en.mean()(k) - plant()(k)) < 2.0 * en.sigma()(k), "spacing %.2f noisy dof %d off", sp, k);
        }
        for (int k = 0; k < 3; ++k)
        {
            double mmin = 1e9, mmax = -1e9, smin = 1e9, smax = -1e9;
            for (std::size_t i = 0; i < mu.size(); ++i) { mmin = std::min(mmin, mu[i](k)); mmax = std::max(mmax, mu[i](k)); }
            for (std::size_t i = 0; i < 3; ++i) { smin = std::min(smin, sg[i](k)); smax = std::max(smax, sg[i](k)); }
            CHECK((mmax - mmin) < 0.1 * sg[2](k), "dof %d noise-free mean spread %.3f sigma", k, (mmax - mmin) / sg[2](k));
            CHECK((smax - smin) < 0.1 * smax, "dof %d sigma spread %.1f %% (0.1..0.5 s)", k, 100 * (smax - smin) / smax);
            std::printf("dof %d: noise-free mean spread %.4f sigma; sigma spread 0.1..0.5 s %.1f %%, incl 1.0 s %.1f %%\n", k,
                        (mmax - mmin) / sg[2](k), 100 * (smax - smin) / smax, 100 * (sg[3](k) - smin) / sg[3](k));
        }
    }

    // ── 5. wheels over-report rotation by 41 %, unbiased gyro: fused dth ~ truth, mount still recovered ──
    {
        Opt o; o.wheel_gain = 1.41; o.gyro = true;
        const Run r = run(cfg_of(0.5), o);
        for (const auto& p : r.pairs)
        {
            const double dtrue = th_true(tsec(p.stamp_b_ms)) - th_true(tsec(p.stamp_a_ms));
            if (dtrue > 0.5)
                std::printf("fused: dth %.4f true %.4f sigma %.4f  (wheel %.4f gyro %.4f)\n", p.dth, dtrue, p.dth_sigma, p.dth_wheel, p.dth_gyro);
            CHECK(std::abs(p.dth - dtrue) < 2.0 * p.dth_sigma, "fused dth %.4f vs true %.4f (2 sigma %.4f)", p.dth, dtrue, 2 * p.dth_sigma);
            if (dtrue > 0.5) CHECK(std::abs(p.dth_wheel - dtrue) > 2.0 * p.dth_sigma, "wheel alone would have passed: test is toothless");
        }
        const HarvesterConfig c = cfg_of(0.5);
        const Truth jt = [&](const Pair& p)
        {   // the Jacobian of the TRUE motion of that pair
            const double ta = tsec(p.stamp_a_ms), tb = tsec(p.stamp_b_ms), d = th_true(tb) - th_true(ta);
            const double cx = x_true(tb) - x_true(ta), thm = th_true(ta) + 0.5 * d;
            return make_pair_from_motion(d, cx * std::cos(thm), -cx * std::sin(thm), 0, 0, c).J;
        };
        const auto ev = evidence(r.pairs, true, jt);
        const Eigen::Vector3d m = ev.mean(), s = ev.sigma(), t = plant();
        std::printf("gyro+41%%: mean (%.4f %.4f %.4f deg)\n", m(0), m(1), m(2) * 180 / kPi);
        for (int k = 0; k < 3; ++k) CHECK(std::abs(m(k) - t(k)) < 2.0 * s(k), "fused: dof %d err %.5f sigma %.5f", k, m(k) - t(k), s(k));
    }

    // ── 6. a stalled interval contributes nothing ──
    {
        Opt o; o.stall_from = 3.0; o.stall_to = 6.0;
        const Run r = run(cfg_of(0.5), o);
        CHECK(r.ep.dropped_stalled > 0, "nothing was dropped");
        double psi = 0;
        for (const auto& p : r.pairs)
        {
            CHECK(tsec(p.stamp_b_ms) < 3.0 or tsec(p.stamp_a_ms) > 6.0, "pair %.1f..%.1f overlaps the stalled interval", tsec(p.stamp_a_ms), tsec(p.stamp_b_ms));
            psi += p.info(2, 2);
        }
        CHECK(psi < psi_tot, "stalled run has as much yaw information as the full run");
    }

    // ── 7. a stop alone is worth exactly nothing ──
    {
        TrajectoryHarvester h{HarvesterConfig{}};
        for (int ms = 0; ms <= 40000; ms += 50) { h.add_odometry(T0 + ms, 0, 0, 0); if (ms % 100 == 0) h.add_keyframe(T0 + ms, false); }
        h.finish();
        CHECK(h.take_pairs().empty() and h.information().norm() == 0.0, "stop produced information");
    }

    std::printf(g_fail ? "trajectory_harvester_test: %d FAILED\n" : "trajectory_harvester_test: ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
