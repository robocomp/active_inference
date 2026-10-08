// Offline checks of rc::mountf (plan docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md,
// Tasks 1, 1b, 2, 3; the r2.2 CORRECTIONS supersede the task text). Synthetic data in the SAME model the
// estimators assume, plus one replay pin on a logged Webots run (skipped when the log is absent).
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/mount_factors_selftest.cpp -o /tmp/mount_factors_selftest && /tmp/mount_factors_selftest
//
// Run from room_concept/ so the replay pin finds tmp/noise_innov/.
#include "../src/mount_factors.h"

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


// ─────────────────────────────── shared synthetic room + LiDAR ray caster ───────────────────────────────
namespace syn
{
    using V3 = Eigen::Vector3d;
    struct Room { double hx = 2.0, hy = 3.0, h = 2.6; };          // 4 x 6 m, 2.6 m ceiling, world frame
    /// Small 6-DoF mount error about the sensor origin s: p_true = R(w)(p_nom - s) + s + t.
    struct M6 { V3 t = V3::Zero(), w = V3::Zero(); };
    inline Eigen::Matrix3d rot(const V3 &w)
    {
        const double a = w.norm();
        return a > 0 ? Eigen::AngleAxisd(a, w / a).toRotationMatrix() : Eigen::Matrix3d::Identity();
    }
    /// Ray from o along unit d (WORLD frame) to the first of: floor z=0, ceiling z=h, the 4 walls. Returns t.
    inline double cast(const Room &r, const V3 &o, const V3 &d, bool ceiling = true)
    {
        double best = 1e9;
        auto take = [&](double t) { if (t > 1e-6 and t < best) best = t; };
        if (d.z() < -1e-9) take(-o.z() / d.z());
        if (ceiling and d.z() > 1e-9) take((r.h - o.z()) / d.z());
        if (d.x() > 1e-9) take((r.hx - o.x()) / d.x());
        if (d.x() < -1e-9) take((-r.hx - o.x()) / d.x());
        if (d.y() > 1e-9) take((r.hy - o.y()) / d.y());
        if (d.y() < -1e-9) take((-r.hy - o.y()) / d.y());
        return best;
    }
    struct Pose { double x = 0, y = 0, th = 0; };   // body -> world: p_w = R(th) p_b + (x, y), z unchanged
    /// One sweep in the BODY frame as the NOMINAL mount transform produces it (p_nom = M^-1(p_true)).
    /// s = nominal sensor origin (body), elevations (rad, about the horizontal), azimuth step.
    inline std::vector<Eigen::Vector3f> sweep(const Room &room, const Pose &P, const V3 &s, const M6 &M,
                                              const std::vector<double> &elev, double az_step, double noise,
                                              std::mt19937 &rng, bool ceiling = true)
    {
        std::normal_distribution<double> n01(0.0, 1.0);
        const Eigen::Matrix3d R = rot(M.w);
        const double c = std::cos(P.th), sn = std::sin(P.th);
        auto to_w = [&](const V3 &b) { return V3(c * b.x() - sn * b.y() + P.x, sn * b.x() + c * b.y() + P.y, b.z()); };
        auto to_b = [&](const V3 &w) { const V3 q(w.x() - P.x, w.y() - P.y, w.z()); return V3(c * q.x() + sn * q.y(), -sn * q.x() + c * q.y(), q.z()); };
        const V3 s_true = s + M.t;                                   // the sensor origin, truly
        std::vector<Eigen::Vector3f> out;
        for (double az = 0.0; az < 2 * M_PI; az += az_step)
            for (const double el : elev)
            {
                const V3 d_nom(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
                const V3 d_b = R * d_nom;                              // the true ray, body frame
                const V3 o_w = to_w(s_true);
                const V3 d_w(c * d_b.x() - sn * d_b.y(), sn * d_b.x() + c * d_b.y(), d_b.z());
                const double t = cast(room, o_w, d_w, ceiling);
                if (t > 50.0) continue;
                V3 p_true = to_b(o_w + t * d_w) + noise * V3(n01(rng), n01(rng), n01(rng));
                const V3 p_nom = R.transpose() * (p_true - s - M.t) + s;   // M^-1
                out.emplace_back(p_nom.cast<float>());
            }
        return out;
    }
    inline std::vector<double> range(double a, double b, double step)
    { std::vector<double> v; for (double x = a; x <= b + 1e-9; x += step) v.push_back(x * kDegD); return v; }
}

// ─────────────────────────────── Part 2: the floor plane ───────────────────────────────
static void part2()
{
    using namespace syn;
    std::printf("Part 2: FloorPlaneMount (bpearl roll, pitch, z from the floor seen head-on)\n");
    const Room room;
    const V3 s_bp(0.0, 0.14, 0.7025);                 // Shadow frame (r2.2 item 5)
    const auto elev = range(-88.0, -2.0, 3.0);        // inverted dome, pointing down
    M6 M; M.w = V3(0.5 * kDegD, -0.8 * kDegD, 0.0); M.t = V3(0.0, 0.0, 0.01);
    auto drive = [&](FloorPlaneMount &fm, int sweeps, double furniture, double flat_sd, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ux(-1.0, 1.0), uth(-M_PI, M_PI), u01(0.0, 1.0), uz(0.3, 0.6);
        std::normal_distribution<double> n01(0.0, 1.0);
        for (int k = 0; k < sweeps; ++k)
        {
            const Pose P{ux(rng), 1.5 * ux(rng), uth(rng)};
            auto pts = sweep(room, P, s_bp, M, elev, 2.0 * kDegD, 0.005, rng, false);
            const double off = flat_sd * n01(rng);      // the whole sweep's floor offset (common mode)
            for (auto &p : pts)
            {
                const V3 pd = p.cast<double>();
                const V3 pt = rot(M.w) * (pd - s_bp) + s_bp + M.t;   // back to true to edit the scene
                V3 q = pt;
                if (std::abs(pt.z()) < 0.03)
                {
                    q.z() += off;
                    if (pt.head<2>().norm() < 1.5 and u01(rng) < furniture) q.z() = uz(rng);   // a seat, a table top
                }
                p = (rot(M.w).transpose() * (q - s_bp - M.t) + s_bp).cast<float>();
            }
            fm.observe_sweep(pts, s_bp.cast<float>());
        }
    };
    auto show = [](const char *tag, const FloorPlaneMount &fm)
    {
        const auto i = fm.info(); const auto m = i.mean(), s = i.sigma();
        std::printf("    %-30s roll %+.3f +- %.4f deg | pitch %+.3f +- %.4f deg | dz %+6.2f +- %.2f mm | share %.2f sigma_f %.1f mm\n",
                    tag, m[0] / kDegD, s[0] / kDegD, m[1] / kDegD, s[1] / kDegD, m[2] * 1e3, s[2] * 1e3,
                    fm.floor_share(), fm.sigma_floor() * 1e3);
    };
    // 2 sigma PLUS 1 % of the plant: the rows are first order in omega, the synthetic plant is an exact
    // rotation, so an O(omega^2) linearisation residue (~0.002 deg on 0.8 deg) is EXPECTED, and with
    // sigma ~0.0006 deg "within 2 sigma" alone would test it rather than the estimator (review item 9).
    auto agrees = [&](const FloorPlaneMount &fm, bool with_dz)
    {
        const auto i = fm.info(); const auto m = i.mean(), s = i.sigma();
        const double lin = 0.01 * M.w.norm();
        return std::abs(m[0] - M.w.x()) < 2 * s[0] + lin and std::abs(m[1] - M.w.y()) < 2 * s[1] + lin
               and (not with_dz or std::abs(m[2] - M.t.z()) < 2 * s[2] + 0.01 * std::abs(M.t.z()));
    };
    {
        FloorPlaneMount fm; drive(fm, 100, 0.0, 0.0, 11); show("A clean floor", fm);
        check(agrees(fm, true), "A: planted roll 0.5, pitch -0.8 deg, dz 10 mm recovered (2 sigma + 1 % linearisation)");
    }
    {
        FloorPlaneMount fm; drive(fm, 100, 0.2, 0.0, 12); show("B 20% furniture near", fm);
        check(agrees(fm, true), "B: 20 % of near floor points on furniture -> still recovered (mixture, no band)");
    }
    {
        const int S = 100; const double flat = 0.005;
        FloorPlaneMount fm; drive(fm, S, 0.0, flat, 13); show("C per-sweep floor offset 5 mm", fm);
        const auto s = fm.info().sigma();
        std::printf("    sigma_flat/sqrt(sweeps) = %.2f mm\n", flat / std::sqrt(double(S)) * 1e3);
        check(s[2] > 0.8 * flat / std::sqrt(double(S)), "C: dz sigma does not collapse below ~sigma_flat/sqrt(sweeps)");
        check(agrees(fm, true), "C: and the planted values are still recovered");
        check(s[0] < 0.02 * kDegD and s[1] < 0.02 * kDegD, "C: the per-sweep dz nuisance does NOT absorb roll/pitch (tilt stays sharp)");
    }
}

// ─────────────────────────────── Part 3: verticals (walls vertical, ceiling horizontal) ───────────────────────────────
static void part3()
{
    using namespace syn;
    std::printf("Part 3: VerticalMount (helios tilt from walls + ceiling; bpearl planar relative to the map)\n");
    const Room room;
    const V3 s_h(0.0, -0.155, 1.1075), s_bp(0.0, 0.14, 0.7025);     // Shadow frame (r2.2 item 5)
    const auto elev_h  = range(-15.5, 54.5, 2.0);                    // inverted helios: the fan points UP
    const auto elev_bp = range(-88.0, -2.0, 3.0);
    const Eigen::Matrix3f wide = Eigen::Vector3f(0.05f, 0.05f, float(2.0 * kDegD)).cwiseAbs2().asDiagonal();
    // the wall map in the BODY frame, as the localiser's pose estimate puts it
    auto walls_body = [&](const Pose &P)
    {
        const double c = std::cos(P.th), sn = std::sin(P.th);
        auto tb = [&](double x, double y) { const double qx = x - P.x, qy = y - P.y; return Eigen::Vector2f(float(c * qx + sn * qy), float(-sn * qx + c * qy)); };
        const std::array<Eigen::Vector2f, 4> v{tb(-room.hx, -room.hy), tb(room.hx, -room.hy), tb(room.hx, room.hy), tb(-room.hx, room.hy)};
        std::vector<VerticalMount::Wall> w;
        for (int k = 0; k < 4; ++k) w.push_back({v[k], v[(k + 1) % 4]});
        return w;
    };
    struct Arm { VerticalMount::Role role; M6 M; double pose_sd_xy, pose_sd_th; int sweeps; };
    auto drive = [&](const Arm &a, unsigned seed)
    {
        VerticalMount vm; VerticalMount::Params p; p.role = a.role; vm.set_params(p);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ux(-1.0, 1.0), uth(-M_PI, M_PI);
        std::normal_distribution<double> n01(0.0, 1.0);
        const bool helios = a.role == VerticalMount::Role::HeliosTilt;
        const V3 s = helios ? s_h : s_bp;
        for (int k = 0; k < a.sweeps; ++k)
        {
            const Pose P{ux(rng), 1.5 * ux(rng), uth(rng)};
            const auto pts = sweep(room, P, s, a.M, helios ? elev_h : elev_bp, 1.5 * kDegD, 0.01, rng, helios);
            const Pose Pe{P.x + a.pose_sd_xy * n01(rng), P.y + a.pose_sd_xy * n01(rng), P.th + a.pose_sd_th * n01(rng)};
            vm.observe_sweep(pts, s.cast<float>(), walls_body(Pe), wide, helios ? float(room.h) : 0.f);
        }
        return vm;
    };
    auto show = [](const char *tag, const VerticalMount &vm)
    {
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        std::printf("    %-32s dx %+5.1f+-%4.1f dy %+5.1f+-%4.1f dz %+5.1f+-%4.1f mm | roll %+.3f+-%.3f pitch %+.3f+-%.3f yaw %+.3f+-%.3f deg | share %.2f sw %.1f mm\n",
                    tag, m[0] * 1e3, s[0] * 1e3, m[1] * 1e3, s[1] * 1e3, m[2] * 1e3, s[2] * 1e3, m[3] / kDegD, s[3] / kDegD,
                    m[4] / kDegD, s[4] / kDegD, m[5] / kDegD, s[5] / kDegD, vm.wall_share(), vm.sigma_wall() * 1e3);
    };
    const VerticalMount::Params defaults;
    const auto prior_sd = defaults.prior_sigma;
    auto at_prior = [&](const VerticalMount &vm, std::initializer_list<int> idx)
    {
        const auto i = vm.info(); const auto s = i.sigma(), m = i.mean();
        bool ok = true;
        for (int k : idx) ok = ok and std::abs(s[k] - prior_sd[k]) < 1e-9 * prior_sd[k] + 1e-15 and std::abs(m[k]) < 1e-12;
        return ok;
    };
    const double lin = 0.01;   // 1 % of the plant: first-order rows vs an exact synthetic rotation (see part 2)
    {   // A: planted helios tilt, exact pose
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 60};
        a.M.w = V3(0.7 * kDegD, -0.4 * kDegD, 0.0);
        const auto vm = drive(a, 21); show("A helios tilt 0.7/-0.4 deg", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3] - a.M.w.x()) < 2 * s[3] + lin * a.M.w.norm() and std::abs(m[4] - a.M.w.y()) < 2 * s[4] + lin * a.M.w.norm(),
              "A: helios roll/pitch recovered within 2 sigma (+1 % linearisation)");
        check(s[3] < 0.05 * kDegD and s[4] < 0.05 * kDegD, "A: and informed (sigma < 0.05 deg)");
    }
    {   // C: random per-sweep POSE error (3 cm, 1 deg), ZERO tilt -> pose error is never read as tilt (Review Focus #1)
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.03, 1.0 * kDegD, 60};
        const auto vm = drive(a, 23); show("C pose error 3 cm/1 deg, no tilt", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3]) < 2 * s[3] and std::abs(m[4]) < 2 * s[4], "C: per-sweep pose error does NOT leak into tilt (within 2 sigma of 0)");
    }
    {   // D: helios dz planted 2 cm -> unobservable (the ceiling height is a nuisance): stays EXACTLY at its prior
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 30};
        a.M.t = V3(0.0, 0.0, 0.02);
        const auto vm = drive(a, 24); show("D helios dz 2 cm", vm);
        check(at_prior(vm, {2}), "D: helios dz stays at its prior (sigma == prior, mean 0)");
    }
    {   // E (r2.2 item 6): helios with a planted 30 mm lever and 1 deg yaw, CORRECT map -> NO planar information
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 30};
        a.M.t = V3(0.03, -0.02, 0.0); a.M.w = V3(0.0, 0.0, 1.0 * kDegD);
        const auto vm = drive(a, 25); show("E helios lever 30 mm + 1 deg yaw", vm);
        check(at_prior(vm, {0, 1, 5}), "E: the vertical factor gives the helios ZERO planar information (sigma == prior)");
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3]) < 2 * s[3] and std::abs(m[4]) < 2 * s[4], "E: and the lever is not read as tilt");
    }
    {   // B: bpearl planar offset RELATIVE to the map (dx 2 cm, dy -3 cm, dyaw 0.8 deg), small true pose noise
        Arm a{VerticalMount::Role::BpearlPlanar, {}, 0.01, 0.3 * kDegD, 400};
        a.M.t = V3(0.02, -0.03, 0.0); a.M.w = V3(0.0, 0.0, 0.8 * kDegD);
        const auto vm = drive(a, 22); show("B bpearl planar 20/-30 mm, 0.8 deg", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[0] - 0.02) < 2 * s[0] and std::abs(m[1] + 0.03) < 2 * s[1]
              and std::abs(m[5] - a.M.w.z()) < 2 * s[5] + lin * a.M.w.norm(), "B: bpearl planar (relative) recovered within 2 sigma");
        check(std::abs(m[0]) > 2 * s[0] and std::abs(m[1]) > 2 * s[1] and std::abs(m[5]) > 2 * s[5], "B: and detected");
        check(at_prior(vm, {2, 3, 4}), "B: bpearl verticals carry NO tilt/z (the floor owns those)");
    }
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: a harness that parses agent data must run under the agent's locale
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() or only == "1") { part1(); part1b(); part1_replay(); }
    if (only.empty() or only == "2") part2();
    if (only.empty() or only == "3") part3();
    // ── PLACE CELLS (fix A, 2026-10-08): repeated views of one place saturate; new places keep adding ──
    {
        std::printf("\nP. place-level common mode (PlaceCells)\n");
        using PC = rc::mountf::PlaceCells<3>;
        const Eigen::Vector3d psig(0.03, 0.03, 0.0087);           // place error: 3 cm, 3 cm, 0.5 deg
        const Eigen::Matrix3d Hs = Eigen::Vector3d(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01), 1.0 / (0.0035 * 0.0035)).asDiagonal();
        const Eigen::Vector3d place_err(0.048, -0.032, 0.006);    // the parked spot's own error (the live case)
        PC pc; pc.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 2461; ++i) pc.add(42, Hs, Hs * place_err, 1.0);
        for (int i = 0; i < 1000; ++i) pc.add(1000 + i % 200, Hs, Eigen::Vector3d::Zero(), 1.0);
        Eigen::Matrix3d H; Eigen::Vector3d b; pc.effective(H, b);
        const Eigen::Vector3d est = H.ldlt().solve(b), sd = H.inverse().diagonal().cwiseSqrt();
        std::printf("    one parked place + 200 others: est (%+.1f, %+.1f) mm (%+.2f deg), sd (%.1f, %.1f) mm\n",
                    est(0) * 1e3, est(1) * 1e3, est(2) * 180 / M_PI, sd(0) * 1e3, sd(1) * 1e3);
        check(std::abs(est(0)) < 0.002 and std::abs(est(1)) < 0.002, "P1 the parked place counts as ONE place (est ~ 0)");
        PC flat; flat.place_prec = Eigen::Vector3d::Constant(1e12);
        for (int i = 0; i < 2461; ++i) flat.add(42, Hs, Hs * place_err, 1.0);
        for (int i = 0; i < 1000; ++i) flat.add(1000 + i % 200, Hs, Eigen::Vector3d::Zero(), 1.0);
        flat.effective(H, b);
        const Eigen::Vector3d est0 = H.ldlt().solve(b);
        std::printf("    without the place nuisance: est (%+.1f, %+.1f) mm\n", est0(0) * 1e3, est0(1) * 1e3);
        check(est0(0) > 0.025, "P2 control: without it the parked view dominates (the live defect)");
        PC one; one.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 100000; ++i) one.add(7, Hs, Eigen::Vector3d::Zero(), 1.0);
        one.effective(H, b);
        check(H(0, 0) <= 1.0 / (0.03 * 0.03) * 1.0001, "P3 one place saturates at 1/sigma_place^2");
        PC f; f.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 5000; ++i) f.add(i, Hs, Eigen::Vector3d::Zero(), 0.99);
        check(f.size() < 5000, "P4 forgotten cells are pruned");
        check(rc::mountf::place_key(1.1f, 2.2f, 0.1f) == rc::mountf::place_key(1.2f, 2.3f, 0.2f), "P5 same cell, same key");
        check(rc::mountf::place_key(1.1f, 2.2f, 0.1f) != rc::mountf::place_key(1.1f, 2.2f, 1.7f), "P5 other heading octant, other key");
    }

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
