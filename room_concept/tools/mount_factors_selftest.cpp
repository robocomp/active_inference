// Offline checks of rc::mountf (plan docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md,
// Tasks 1, 1b (the KINEMATIC factor; the floor/vertical parts 2, 3 and P moved to
// sensor_calibration/tests/mount_factors_test.cpp on 2026-10-29); the r2.2 CORRECTIONS supersede the task text). Synthetic data in the SAME model the
// estimators assume, plus one replay pin on a logged Webots run (skipped when the log is absent).
//
//   g++ -std=c++23 -O2 -I../common -I/usr/include/eigen3 tools/mount_factors_selftest.cpp -o /tmp/mount_factors_selftest && /tmp/mount_factors_selftest
//
// Run from room_concept/ so the replay pin finds tmp/noise_innov/.
#include "../src/kinematic_mount.h"

#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <array>
#include <random>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc::mountf;
static constexpr double kDegD = M_PI / 180.0;
// ─────────────────────────────── Part 1 / 1b: the kinematic factor ───────────────────────────────
namespace kin
{
    struct Truth { double dx = 0.03, dy = -0.02, psi = 0.0175; };
    enum Motion { Parked, Straight, PivotCCW, PivotCW, ArcCCW, ArcCW };
    struct Gen
    {
        std::mt19937 rng{7};
        double sigma = 0.003, out_p = 0.04, out_k = 20.0;     // white row noise, 4 % gross outliers (flagged)
        double u_kv = 0.0, u_lat = 0.0;                      // odometry residual error: u = theta_true - theta_applied
        /// One cycle in the MIDPOINT model (r2.2 item 1): delta_fwd = 2 sin(dth/2) dx + u_kv d_fwd,
        /// delta_lat = -2 sin(dth/2) dy + psi d_fwd + u_lat d_lat.
        KinematicMount::Cycle cycle(Motion m, const Truth &t)
        {
            std::normal_distribution<double> n01(0.0, 1.0);
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            double dth = 0.0, dfwd = 0.0;
            switch (m)
            {
                case Parked: break;
                case Straight: dfwd = 0.015; break;                  // 0.3 m/s at 20 Hz
                case PivotCCW: dth = 0.02; break;                    // 0.4 rad/s
                case PivotCW:  dth = -0.02; break;
                case ArcCCW:   dth = 0.015; dfwd = 0.015; break;     // v/omega = 1 m
                case ArcCW:    dth = -0.015; dfwd = 0.015; break;
            }
            const double s = 2.0 * std::sin(0.5 * dth);
            KinematicMount::Cycle c;
            c.dth = dth; c.d_fwd = dfwd; c.d_lat = 0.0;
            c.var_fwd = c.var_lat = sigma * sigma;
            const bool out_f = u01(rng) < out_p, out_l = u01(rng) < out_p;
            const double kf = out_f ? std::sqrt(out_k) : 1.0, kl = out_l ? std::sqrt(out_k) : 1.0;
            c.w_fwd = out_f ? 0.02 : 1.0;  c.w_lat = out_l ? 0.02 : 1.0;   // the learner's 1 - r
            c.innov_fwd = s * t.dx + u_kv * dfwd + kf * sigma * n01(rng);
            c.innov_lat = -s * t.dy + t.psi * dfwd + u_lat * c.d_lat + kl * sigma * n01(rng);
            return c;
        }
    };
    static Motion pick(std::mt19937 &rng, const std::vector<std::pair<Motion, double>> &mix)
    {
        double tot = 0.0; for (auto &[m, w] : mix) tot += w;
        double r = std::uniform_real_distribution<double>(0.0, tot)(rng);
        for (auto &[m, w] : mix) { if ((r -= w) <= 0.0) return m; }
        return mix.back().first;
    }
    static void run(KinematicMount &km, Gen &g, const Truth &t, int n, const std::vector<std::pair<Motion, double>> &mix)
    {
        for (int i = 0; i < n; ++i) km.observe(g.cycle(pick(g.rng, mix), t));
    }
    static const std::vector<std::pair<Motion, double>> kMixed{
        {Parked, 0.4}, {Straight, 0.25}, {PivotCCW, 0.08}, {PivotCW, 0.08}, {ArcCCW, 0.095}, {ArcCW, 0.095}};
    static void print(const char *tag, const KinematicMount &km)
    {
        const auto m = km.mean(), s = km.sigma();
        std::printf("    %-34s dx %+6.1f +- %4.1f mm | dy %+6.1f +- %4.1f mm | psi %+.3f +- %.3f deg\n", tag,
                    m[0] * 1e3, s[0] * 1e3, m[1] * 1e3, s[1] * 1e3, m[2] / kDegD, s[2] / kDegD);
    }
    static KinematicMount fresh(double memory = 1e9)
    {
        KinematicMount km; KinematicMount::Params p; p.memory = memory; km.set_params(p); return km;
    }
    /// The calibrator reports the odometry as converged and correct (sigma 0.05 %).
    static void tight(KinematicMount &km, double mu = 0.0, double theta = 0.0, long solve = 0)
    {
        km.set_odometry(Eigen::Vector2d(0.0005, 0.0005).cwiseAbs2().asDiagonal(), Eigen::Vector2d(theta, 0.0),
                        Eigen::Vector2d(mu, 0.0), solve);
    }
    static void wide(KinematicMount &km)
    {
        km.set_odometry(Eigen::Vector2d(0.02, 0.05).cwiseAbs2().asDiagonal(), Eigen::Vector2d::Zero(),
                        Eigen::Vector2d::Zero(), 0);
    }

    // ── replay pin: δ rows from a logged run (innov CSV), through THIS estimator ──
    struct Csv { std::vector<std::string> head; };
    static bool replay(const char *path, KinematicMount &km, long &rows)
    {
        std::ifstream f(path);
        if (not f.is_open()) return false;
        std::string line;
        if (not std::getline(f, line)) return false;
        std::vector<std::string> h;
        for (std::size_t a = 0, b; a <= line.size(); a = b + 1)
        { b = line.find(',', a); if (b == std::string::npos) b = line.size(); h.emplace_back(line.substr(a, b - a)); }
        auto col = [&](std::string_view n) { for (std::size_t i = 0; i < h.size(); ++i) if (h[i] == n) return int(i); return -1; };
        const int cf = col("d_fwd"), cl = col("d_lat"), of = col("odom_fwd"), ol = col("odom_lat"), ot = col("odom_th"),
                  mf = col("m_fwd"), ml = col("m_lat"), rf = col("r_out_fwd"), rl = col("r_out_lat");
        if (cf < 0 or cl < 0 or of < 0 or ot < 0 or mf < 0 or ml < 0 or rf < 0 or rl < 0) return false;
        std::vector<double> v(h.size());
        rows = 0;
        while (std::getline(f, line))
        {
            std::size_t a = 0; int k = 0; bool ok = true;
            for (std::size_t b; a <= line.size() and k < int(v.size()); a = b + 1, ++k)
            {
                b = line.find(',', a); if (b == std::string::npos) b = line.size();
                const char *s = line.data() + a, *e = line.data() + b;
                const auto r = std::from_chars(s, e, v[std::size_t(k)]);   // locale-independent (CLAUDE.md)
                if (r.ec != std::errc{}) v[std::size_t(k)] = NAN;
            }
            if (k < int(v.size())) continue;
            KinematicMount::Cycle c;
            c.innov_fwd = v[cf]; c.innov_lat = v[cl]; c.d_fwd = v[of]; c.d_lat = ol >= 0 ? v[ol] : 0.0; c.dth = v[ot];
            c.var_fwd = v[mf]; c.var_lat = v[ml]; c.w_fwd = 1.0 - v[rf]; c.w_lat = 1.0 - v[rl];
            for (double x : {c.innov_fwd, c.innov_lat, c.d_fwd, c.dth, c.var_fwd, c.var_lat, c.w_fwd, c.w_lat}) ok = ok and std::isfinite(x);
            if (not ok) continue;
            km.observe(c); ++rows;
        }
        return rows > 0;
    }
}

static void part1()
{
    using namespace kin;
    std::printf("Part 1: KinematicMount (helios dx, dy, psi from the innovation stream)\n");
    const Truth t;
    const double pxy = 0.05, pyaw = 1.0 * kDegD;
    {   // A: mixed motion, odometry converged and correct
        Gen g; auto km = fresh(); tight(km);
        run(km, g, t, 40000, kMixed);
        print("A mixed", km);
        const auto m = km.mean(), s = km.sigma();
        check(std::abs(m[0] - t.dx) < 2 * s[0] and std::abs(m[1] - t.dy) < 2 * s[1] and std::abs(m[2] - t.psi) < 2 * s[2],
              "A: planted (30, -20) mm / 1 deg recovered within 2 sigma");
        check(s[0] < 0.005 and s[1] < 0.005 and s[2] < 0.1 * kDegD, "A: sigma < 5 mm / 0.1 deg");
    }
    {   // B: straights (and parking) only -> the lever is not excited
        Gen g; auto km = fresh(); tight(km);
        run(km, g, t, 40000, {{Parked, 0.5}, {Straight, 0.5}});
        print("B straights only", km);
        const auto s = km.sigma();
        check(s[0] >= 0.9 * pxy and s[1] >= 0.9 * pxy, "B: straights only -> dx, dy UNINFORMED (sigma >= 0.9 prior)");
        check(s[2] < 0.2 * pyaw, "B: straights only -> psi informed");
    }
    {   // C: pivots (and parking) only -> the yaw is not excited
        Gen g; auto km = fresh(); tight(km);
        run(km, g, t, 40000, {{Parked, 0.5}, {PivotCCW, 0.25}, {PivotCW, 0.25}});
        print("C pivots only", km);
        const auto s = km.sigma();
        check(s[2] >= 0.9 * pyaw, "C: pivots only -> psi UNINFORMED (sigma >= 0.9 prior)");
        check(s[0] < 0.01 and s[1] < 0.01, "C: pivots only -> dx, dy informed");
    }
    {   // the ma1 rule: every row at weight 1 -- doubling the data halves the variance exactly
        Gen g1, g2; auto a = fresh(), b = fresh(); tight(a); tight(b);
        run(a, g1, t, 10000, kMixed); run(b, g2, t, 20000, kMixed);
        const double r = (a.sigma()[0] / b.sigma()[0]);
        std::printf("    sigma ratio 10k/20k cycles: %.3f (sqrt 2 = 1.414)\n", r);
        check(std::abs(r - std::sqrt(2.0)) < 0.1, "rows at weight 1 (no MA(1) factor): sigma scales as 1/sqrt(n)");
    }
    {   // E (r2.2 item 3): arcs of ONE turning direction (v/omega = 1 m) + a 1 % forward-scale error.
        // dx and k_v are COLLINEAR there (2 sin(dth/2) ~ dth, d_fwd = R dth), so the lever must NOT come out
        // confidently wrong -- it must report the confound as width. Control: a tight odometry posterior
        // (which here is a LIE: the scale is 1 % off) gives the confidently-wrong answer the Schur form avoids.
        Gen g; g.u_kv = 0.01;
        auto km = fresh(); wide(km);
        Gen g2 = g; auto ctl = fresh(); tight(ctl);
        run(km, g, t, 40000, {{Parked, 0.3}, {ArcCCW, 0.7}});
        run(ctl, g2, t, 40000, {{Parked, 0.3}, {ArcCCW, 0.7}});
        print("E one-way arcs, 1% k_v, wide odo", km);
        print("E control: tight (wrong) odo", ctl);
        check(std::abs(km.mean()[0] - t.dx) < 2 * km.sigma()[0], "E: wide odometry posterior -> dx NOT confidently wrong");
        check(std::abs(ctl.mean()[0] - t.dx) > 2 * ctl.sigma()[0], "E control: a tight odometry claim IS confidently wrong (the case bites)");
    }
}

static void part1b()
{
    using namespace kin;
    std::printf("Part 1b: odometry readiness (common-mode Schur over the odometry nuisance)\n");
    {   // (a) Sigma_o at the prior, a 3 % scale error, one-direction arcs only, NO lever: not confidently
        //     wrong, and the width the confound costs is visible (vs. a tight posterior).
        Truth t0{0.0, 0.0, 0.0};
        const std::vector<std::pair<Motion, double>> arcs{{Parked, 0.3}, {ArcCCW, 0.7}};
        Gen g; g.u_kv = 0.03; Gen g2 = g;
        auto km = fresh(); wide(km); auto ctl = fresh(); tight(ctl);
        run(km, g, t0, 40000, arcs);
        run(ctl, g2, t0, 40000, arcs);
        print("(a) 3% k_v, arcs, wide odo", km);
        print("(a) control tight odo", ctl);
        check(std::abs(km.mean()[0]) < 2 * km.sigma()[0], "(a) wide odometry: lever not confidently wrong (|dx| < 2 sigma)");
        check(km.sigma()[0] > 5 * ctl.sigma()[0], "(a) wide odometry: dx sigma carries the confound (> 5x the tight one)");
        // ... and BY WEIGHT, not by a switch: add straights (they measure k_v directly, delta_fwd = u d_fwd)
        // and the same wide prior no longer costs the lever anything.
        Gen g3; g3.u_kv = 0.03; auto km3 = fresh(); wide(km3);
        run(km3, g3, t0, 40000, {{Parked, 0.3}, {ArcCCW, 0.6}, {Straight, 0.1}});
        print("(a') + straights, wide odo", km3);
        check(std::abs(km3.mean()[0]) < 2 * km3.sigma()[0] and km3.sigma()[0] < 0.25 * km.sigma()[0],
              "(a') straights resolve the scale: the lever is informed again with the SAME wide odometry prior");
    }
    {   // (c) the APPLIED k_v jumps by 2 % mid-run: re-referencing (b -= C dtheta, e -= D dtheta) keeps the
        //     estimate put. Exactness is checked with Sigma_drift OFF (a jump IS drift, which widens Sigma_o by
        //     design); the plan's 1-sigma criterion with it ON.
        Truth t;
        const double k_true = 0.02;
        const std::vector<std::pair<Motion, double>> mix{{Parked, 0.3}, {ArcCCW, 0.3}, {PivotCCW, 0.1}, {PivotCW, 0.1}, {Straight, 0.2}};
        for (const bool drift : {false, true})
        {
            KinematicMount km; KinematicMount::Params p; p.memory = 1e9; p.drift = drift; km.set_params(p);
            Gen g;
            g.u_kv = k_true - 0.0;  tight(km, /*mu=*/k_true - 0.0, /*theta=*/0.0, 1);
            run(km, g, t, 20000, mix);
            const auto m1 = km.mean(), s1 = km.sigma();
            print(drift ? "(c) drift on: before the jump" : "(c) drift off: before the jump", km);
            g.u_kv = k_true - 0.02; tight(km, /*mu=*/0.0, /*theta=*/0.02, 2);       // applied now 0.02 = truth
            const auto m_after_jump = km.mean();
            run(km, g, t, 20000, mix);
            print(drift ? "(c) drift on: end" : "(c) drift off: end", km);
            if (not drift)
                check((m_after_jump - m1).norm() < 1e-9, "(c) a change of the applied k_v alone does not move the mount estimate (exact re-reference)");
            else
            {
                check(std::abs(km.mean()[0] - m1[0]) < s1[0] and std::abs(km.mean()[1] - m1[1]) < s1[1],
                      "(c) after the jump the estimate stays within 1 sigma of the pre-jump estimate");
                check(std::abs(km.mean()[0] - t.dx) < 2 * km.sigma()[0], "(c) and agrees with the truth (2 sigma)");
            }
        }
    }
    {   // (d) Sigma_drift: an applied k_v that OSCILLATES +-2 % between re-solves while the calibrator
        //     reports a tiny sigma -- a calibration that is still sliding. No lever: no false detection.
        Truth t0{0.0, 0.0, 0.0};
        const double k_true = 0.01;
        auto drive = [&](bool with_drift)
        {
            KinematicMount km; KinematicMount::Params p; p.memory = 1e9; p.drift = with_drift; km.set_params(p);
            Gen g;
            for (int solve = 0; solve < 40; ++solve)
            {
                const double th = (solve % 2) ? k_true + 0.02 : k_true - 0.02;
                g.u_kv = k_true - th;
                tight(km, 0.0, th, solve);                 // "converged at the applied value": the claim to distrust
                run(km, g, t0, 1000, {{Parked, 0.3}, {ArcCCW, 0.7}});
            }
            return km;
        };
        const auto km = drive(true), ctl = drive(false);
        print("(d) oscillating k_v, drift on", km);
        print("(d) control, drift off", ctl);
        std::printf("    Sigma_drift(k_v) sd %.4f\n", std::sqrt(km.sigma_odometry()(0, 0)));
        check(std::abs(km.mean()[0]) < 2 * km.sigma()[0], "(d) drift on: no false lever detection");
        check(km.sigma()[0] > 3 * ctl.sigma()[0], "(d) drift on: dx sigma inflates against the drift-free control");
        check(std::abs(ctl.mean()[0]) > 2 * ctl.sigma()[0], "(d) control (drift off) IS a false detection (the case bites)");
    }
}

static void part1_replay()
{
    using namespace kin;
    std::printf("Part 1 D: replay pins on logged Webots runs (plant (30, -20) mm, +1 deg)\n");
    struct Run { const char *path; bool known_dx_shortfall; };
    for (const Run r : {Run{"tmp/noise_innov/innov_2026-10-08_10-56-32.csv", false},
                        Run{"tmp/noise_innov/innov_2026-10-08_12-18-41.csv", true}})
    {
        auto km = fresh(); long rows = 0;
        if (not replay(r.path, km, rows)) { std::printf("    (skipped: %s absent)\n", r.path); continue; }
        std::printf("    %s: %ld rows\n", r.path, rows);
        print("replay", km);
        const auto m = km.mean(), s = km.sigma();
        const Truth t;
        check(m[0] > 2 * s[0] and m[1] < -2 * s[1] and m[2] > 2 * s[2], "D: all three DETECTED with the planted signs");
        check(std::abs(m[1] - t.dy) < 2 * s[1] and std::abs(m[2] - t.psi) < 2 * s[2], "D: dy and psi agree with the plant (2 sigma)");
        const bool dx_ok = std::abs(m[0] - t.dx) < 2 * s[0];
        if (r.known_dx_shortfall)
            std::printf("  [%s] D: dx agrees with the plant (2 sigma) -- KNOWN SHORTFALL on this run, r2.2 item 8 "
                        "(see the diagnosis; not asserted)\n", dx_ok ? "PASS" : "INFO");
        else check(dx_ok, "D: dx agrees with the plant (2 sigma)");
    }
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: a harness that parses agent data must run under the agent's locale
    (void)argc; (void)argv;
    part1(); part1b(); part1_replay();
    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
