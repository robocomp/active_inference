// Offline validation of rc::preint::MotionNoiseInnov against KNOWN motion-noise coefficients.
//
// Simulates what the agent actually sees (Fable window memo 2026-10-05 §4), NOT a Kalman update:
//   * the robot drives a mix of parked / straight / pivot cycles;
//   * the odometry under-reports by a constant SCALE error and adds white per-cycle noise whose
//     covariance is sum_j k_j U_j (per metre, per radian, per second components);
//   * every cycle the scan gives a scan-only pose z = truth + b(truth) + e, where e is white with a
//     variance s * R and R is the scan's CLAIMED covariance (s ~ 0.01: the claim is 10x the sigma), and
//     b is a smooth bias FIELD over position (40 mm / 1 deg, metres of correlation length);
//   * the learner sees delta = z_n - z_{n-1} - odom_n and the regressors.
// Pass: the per-metre / per-radian coefficients are recovered, the per-second ones stay at zero (the
// bias field must NOT be attributed to time), s comes out near its true value, and the heading scale
// error lands in its own coefficient instead of inflating k_th_turn.
//
// Build:  g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/motion_noise_innov_selftest.cpp -o /tmp/mni && /tmp/mni
#include "../src/motion_noise_innov.h"
#include <cmath>
#include <cstdio>
#include <random>

using rc::preint::MotionNoiseInnov;

namespace
{
int failures = 0;
void check(bool ok, const char *what) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what); failures += not ok; }
std::mt19937 rng(20261005);
double N(double s) { return std::normal_distribution<double>(0.0, s)(rng); }
double U01() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }

// Room convention: body forward = (-sin th, cos th), lateral = (cos th, sin th) in world coordinates.
Eigen::Matrix3d body_axes(double th)
{
    Eigen::Matrix3d T;
    T << -std::sin(th), std::cos(th), 0.0,
          std::cos(th), std::sin(th), 0.0,
          0.0,          0.0,          1.0;
    return T;   // columns: forward, lateral, heading
}

struct Truth
{
    double k[6] = {9.4e-5, 6e-6, 1e-5, 1.9e-4, 0.0, 0.0};   // long, lat, lat_turn, th_turn, t_trans, t_rot
    double s = 0.01;          // white scan variance / claimed
    double eps_fwd = 0.01;    // forward scale error (odometry under-reports 1 %)
    double eps_th  = 0.03;    // heading scale error (3 %)
    double bias_amp_xy = 0.04, bias_amp_th = 0.0175, bias_len = 3.0;
};

Eigen::Vector3d bias(const Truth &t, const Eigen::Vector3d &x)
{
    const double a = std::sin(x.x() / t.bias_len) * std::cos(x.y() / t.bias_len);
    const double b = std::cos(x.x() / t.bias_len + 0.7) * std::sin(x.y() / t.bias_len + 0.3);
    return {t.bias_amp_xy * a, t.bias_amp_xy * b, t.bias_amp_th * a * b};
}

double wrap(double a) { return std::remainder(a, 2.0 * M_PI); }

// Run the simulation and feed the learner. Returns it.
// out_share: fraction of MOVING cycles whose scan-only fix carries an extra gross error of out_sigma (m / rad
// scaled), the live heavy tail. mixture: the learner's outlier component on/off.
MotionNoiseInnov run(const Truth &tr, int cycles, bool with_bias, double out_share = 0.0, double out_sigma = 0.0,
                     bool mixture = true, double move_var = 0.0, bool move_term = true)
{
    MotionNoiseInnov L;
    MotionNoiseInnov::Params p;
    p.memory = 1e7;                                            // no forgetting in the test
    p.k0 = {9.4e-5, 6e-6, 1e-5, 1.9e-4, 1e-6, 1e-7};          // the configured values (agent's priors)
    p.mixture = mixture;
    if (not move_term) p.mv_sd = {1e-12, 1e-12, 1e-12};       // the term pinned at 0 (the old model)
    L.set_params(p);

    Eigen::Vector3d x(0.0, 0.0, 0.0);                          // true pose (world)
    const double dt = 0.05;
    Eigen::Vector3d z_prev = Eigen::Vector3d::Constant(NAN);
    Eigen::Matrix3d R_prev = Eigen::Matrix3d::Zero();
    int mode = 0, mode_left = 0;
    for (int n = 0; n < cycles; ++n)
    {
        if (mode_left-- <= 0) { mode = static_cast<int>(3.0 * U01()); mode_left = 40 + static_cast<int>(200 * U01()); }
        // reported odometry increment, body frame
        double d = 0.0, dth = 0.0;
        if (mode == 1) { d = 0.02 + 0.02 * U01(); dth = N(0.003); }
        if (mode == 2) { dth = (U01() < 0.5 ? -1.0 : 1.0) * (0.02 + 0.04 * U01()); }
        const double ad = std::abs(d), ath = std::abs(dth);
        // the unit components of this cycle, BODY frame then world
        const Eigen::Matrix3d T = body_axes(x.z());
        const Eigen::Vector3d ef = T.col(0), el = T.col(1), eh = T.col(2);
        std::array<Eigen::Matrix3f, 6> U;
        U[0] = (ef * ef.transpose() * ad).cast<float>();
        U[1] = (el * el.transpose() * ad).cast<float>();
        U[2] = (el * el.transpose() * ath).cast<float>();
        U[3] = (eh * eh.transpose() * ath).cast<float>();
        U[4] = ((ef * ef.transpose() + el * el.transpose()) * dt).cast<float>();
        U[5] = (eh * eh.transpose() * dt).cast<float>();
        // true body increment = reported * (1 + eps) + white noise of the stated components
        const double n_f = N(std::sqrt(tr.k[0] * ad + tr.k[4] * dt));
        const double n_l = N(std::sqrt(tr.k[1] * ad + tr.k[2] * ath + tr.k[4] * dt));
        const double n_t = N(std::sqrt(tr.k[3] * ath + tr.k[5] * dt));
        const double d_true = d * (1.0 + tr.eps_fwd) + n_f;
        const double l_true = n_l;
        const double th_true = dth * (1.0 + tr.eps_th) + n_t;
        const Eigen::Vector3d odom_world = Eigen::Vector3d(ef.x() * d, ef.y() * d, dth);
        x += Eigen::Vector3d(ef.x() * d_true + el.x() * l_true, ef.y() * d_true + el.y() * l_true, th_true);
        x.z() = wrap(x.z());
        // the scan: claimed covariance (body diag, varies with geometry) and the true white part
        const Eigen::Matrix3d Tn = body_axes(x.z());
        const double cf = 0.010 * (0.7 + 0.6 * U01()), cl = 0.010 * (0.7 + 0.6 * U01());
        const double ct = 0.00272 * (0.7 + 0.6 * U01());       // 0.156 deg
        const Eigen::Matrix3d Rb = Eigen::Vector3d(cf * cf, cl * cl, ct * ct).asDiagonal();
        const Eigen::Matrix3d R = Tn * Rb * Tn.transpose();     // world
        const Eigen::Vector3d e_b(N(std::sqrt(tr.s) * cf), N(std::sqrt(tr.s) * cl), N(std::sqrt(tr.s) * ct));
        Eigen::Vector3d z = x + Tn * e_b + (with_bias ? bias(tr, x) : Eigen::Vector3d::Zero());
        if (mode != 0 and move_var > 0.0)                       // the live parked->moving variance STEP
            z += Eigen::Vector3d(N(std::sqrt(move_var * dt)), N(std::sqrt(move_var * dt)), N(std::sqrt(0.1 * move_var * dt)));
        if (mode != 0 and U01() < out_share)                    // a bad scan-only fix (heavy tail)
            z += Eigen::Vector3d(N(out_sigma), N(out_sigma), N(out_sigma * 0.25));
        if (std::isfinite(z_prev.x()))
        {
            Eigen::Vector3d delta = z - z_prev - odom_world;
            delta.z() = wrap(delta.z());
            L.observe(delta.cast<float>(), T.cast<float>(), U, (R + R_prev).cast<float>(),
                      Eigen::Vector3f(static_cast<float>(d), 0.f, static_cast<float>(dth)), mode != 0 ? 1.f : 0.f,
                      static_cast<float>(dt));
        }
        z_prev = z;  R_prev = R;
    }
    return L;
}

void report(const MotionNoiseInnov &L, const Truth &tr)
{
    const char *kn[6] = {"k_long", "k_lat", "k_lat_turn", "k_th_turn", "k_t_trans", "k_t_rot"};
    for (int j = 0; j < 6; ++j) std::printf("    %-10s %.3e  (truth %.3e)\n", kn[j], L.k(j), tr.k[j]);
    std::printf("    s fwd/lat/th %.4f %.4f %.4f (truth %.4f)\n", L.s(0), L.s(1), L.s(2), tr.s);
    std::printf("    q fwd/lat/th %.2e %.2e %.2e (truth eps^2 %.2e / 0 / %.2e)\n", L.q(0), L.q(1), L.q(2),
                tr.eps_fwd * tr.eps_fwd, tr.eps_th * tr.eps_th);
}
}   // namespace

int main()
{
    const Truth tr;
    std::printf("1. clean (no bias field), 40k cycles\n");
    const auto A = run(tr, 40000, false);
    report(A, tr);
    const auto rel = [](double v, double t) { return std::abs(v - t) / t; };
    check(rel(A.k(0), tr.k[0]) < 0.15, "k_long within 15 %");
    check(rel(A.k(3), tr.k[3]) < 0.15, "k_th_turn within 15 % (the scale error did NOT shrink it)");
    check(rel(A.k(2), tr.k[2]) < 0.35, "k_lat_turn within 35 %");
    check(rel(A.s(0), tr.s) < 0.30 and rel(A.s(2), tr.s) < 0.30, "scan excess s within 30 % (fwd, th)");
    check(rel(A.q(2), tr.eps_th * tr.eps_th) < 0.5, "heading scale error lands in q_th (within 50 %)");
    check(A.k(4) < 2e-6 and A.k(5) < 2e-7, "per-second terms stay near zero");

    std::printf("\n2. with a smooth bias FIELD over position (40 mm / 1 deg, 3 m)\n");
    const auto B = run(tr, 40000, true);
    report(B, tr);
    check(rel(B.k(0), tr.k[0]) < 0.20, "k_long within 20 % under the bias field");
    check(rel(B.k(3), tr.k[3]) < 0.20, "k_th_turn within 20 % under the bias field");
    check(B.k(4) < 2e-6 and B.k(5) < 2e-7, "the bias field is NOT attributed to per-second noise");

    std::printf("\n4. HEAVY TAIL: 4 %% of moving cycles get a gross scan error (sigma 1.5 cm / 0.21 deg)\n");
    {
        const auto plain = run(tr, 40000, false, 0.04, 0.015, false, 0.0, false);   // neither term: the old model
        const auto mix   = run(tr, 40000, false, 0.04, 0.015, true);
        std::printf("    plain  : k_long %.3e  k_th_turn %.3e\n", plain.k(0), plain.k(3));
        std::printf("    mixture: k_long %.3e  k_th_turn %.3e | pi fwd/lat/th %.3f %.3f %.3f  kappa %.0f %.0f %.0f\n",
                    mix.k(0), mix.k(3), mix.pi(0), mix.pi(1), mix.pi(2), mix.kappa(0), mix.kappa(1), mix.kappa(2));
        check(rel(plain.k(0), tr.k[0]) > 0.5, "the PLAIN fit is inflated by the tail (the live defect, reproduced)");
        check(rel(mix.k(0), tr.k[0]) < 0.20, "mixture: k_long within 20 % despite the tail");
        check(rel(mix.k(3), tr.k[3]) < 0.20, "mixture: k_th_turn within 20 % despite the tail");
    }
    std::printf("    clean data, mixture on: pi fwd %.4f (should stay near its 1 %% prior or below)\n", A.pi(0));
    check(A.pi(0) < 0.02, "clean data: the outlier share does not grow");

    std::printf("\n5. a parked->moving variance STEP (5e-5 m^2/s while moving, no trend in speed)\n");
    {
        const auto old_m = run(tr, 40000, false, 0.0, 0.0, true, 5e-5, false);
        const auto new_m = run(tr, 40000, false, 0.0, 0.0, true, 5e-5, true);
        std::printf("    no move term : k_long %.3e k_th_turn %.3e\n", old_m.k(0), old_m.k(3));
        std::printf("    move term    : k_long %.3e k_th_turn %.3e  mv fwd/lat/th %.2e %.2e %.2e\n",
                    new_m.k(0), new_m.k(3), new_m.mv(0), new_m.mv(1), new_m.mv(2));
        check(rel(old_m.k(0), tr.k[0]) > 0.5, "without it, the step is charged per metre (the live symptom)");
        check(rel(new_m.k(0), tr.k[0]) < 0.25, "with it, k_long within 25 %");
        // Planted on the SCAN: each innovation holds two scans, so the coefficient reads 2v. Planted on the
        // odometry it would read v. The data cannot say which — the header says so.
        check(rel(new_m.mv(0), 2 * 5e-5) < 0.3, "and the step is recovered (fwd; 2v for a scan-side step, 30 %)");
    }

    std::printf("\n3. parked only: nothing about the per-metre terms may be learnt\n");
    {
        MotionNoiseInnov L;
        MotionNoiseInnov::Params p; p.memory = 1e7; p.k0 = {9.4e-5, 6e-6, 1e-5, 1.9e-4, 1e-6, 1e-7};
        L.set_params(p);
        const Eigen::Matrix3f T = body_axes(0.3).cast<float>();
        std::array<Eigen::Matrix3f, 6> U{};
        for (auto &u : U) u.setZero();
        U[4] = (Eigen::Vector3f(1, 1, 0).asDiagonal() * 0.05f).toDenseMatrix();
        U[5] = (Eigen::Vector3f(0, 0, 1).asDiagonal() * 0.05f).toDenseMatrix();
        const Eigen::Matrix3f R = Eigen::Vector3f(2e-4f, 2e-4f, 1.5e-5f).asDiagonal();
        for (int n = 0; n < 5000; ++n)
        {
            const Eigen::Vector3f delta(N(std::sqrt(0.01 * 2e-4)), N(std::sqrt(0.01 * 2e-4)), N(std::sqrt(0.01 * 1.5e-5)));
            L.observe(delta, T, U, R, Eigen::Vector3f::Zero(), 0.f, 0.05f);
        }
        std::printf("    k_long %.3e k_th_turn %.3e (priors 9.4e-5, 1.9e-4)\n", L.k(0), L.k(3));
        check(rel(L.k(0), 9.4e-5) < 0.05 and rel(L.k(3), 1.9e-4) < 0.05, "per-metre/radian stay at their priors");
        check(rel(L.s(0), 0.01) < 0.15, "s learnt from parked cycles alone");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
