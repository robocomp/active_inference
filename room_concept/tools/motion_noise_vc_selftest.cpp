// g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/motion_noise_vc_selftest.cpp -o /tmp/x && /tmp/x
// 1. Preintegrator bookkeeping: with motion-proportional noise and no stated sensor noise, the
//    interval covariance IS Σ k_j·unit[j] — through add() and through chain().
// 2. Closed-loop recovery: parked / driving / pivoting stretches through the REAL Integrator, the
//    prior built with the learner's current coefficients, corrections drawn from TRUE coefficients and
//    a true scan excess. The learner must recover both.
#include "motion_noise_vc.h"
#include "se2_preintegration.h"
#include <cstdio>
#include <random>

using rc::preint::Integrator;
using rc::preint::Interval;
using rc::preint::MotionNoiseVC;
using rc::preint::NoiseModel;

namespace
{
    NoiseModel model(const std::array<double, 6> &k)
    {
        NoiseModel q;
        q.motion_proportional = true;
        q.zupt_enabled = false;
        q.scale_v = q.scale_omega = 0.f;
        q.k_long = float(k[0]); q.k_lat = float(k[1]); q.k_lat_turn = float(k[2]);
        q.k_th_turn = float(k[3]); q.k_t_trans = float(k[4]); q.k_t_rot = float(k[5]);
        return q;
    }
    Eigen::Matrix3f sum_units(const Interval &iv, const std::array<double, 6> &k)
    {
        Eigen::Matrix3f s = Eigen::Matrix3f::Zero();
        for (int j = 0; j < Interval::NC; ++j) s += float(k[j]) * iv.unit[j];
        return s;
    }
}

int main()
{
    int fails = 0;
    const std::array<double, 6> k_true{90e-6, 8e-6, 12e-6, 200e-6, 1e-6, 0.5e-6};

    // ── 1. bookkeeping ──
    {
        Integrator a(0.3f), b(0.9f);
        a.set_noise(model(k_true)); b.set_noise(model(k_true));
        for (int i = 0; i < 40; ++i) a.add(0.02f, 0.4f, 0.3f, 0.02f);
        for (int i = 0; i < 40; ++i) b.add(0.f, 0.2f, -0.8f, 0.02f);
        const Interval ab = rc::preint::chain(a.result(), b.result());
        const float ea = (a.result().cov - sum_units(a.result(), k_true)).norm() / a.result().cov.norm();
        const float ec = (ab.cov - sum_units(ab, k_true)).norm() / ab.cov.norm();
        const bool ok = ea < 1e-5f and ec < 1e-5f;
        std::printf("1. cov == sum k_j unit_j : add %.1e, chain %.1e  (%s)\n", ea, ec, ok ? "ok" : "FAIL");
        if (not ok) ++fails;
    }

    // ── 1b. heading: a stated density and the model's per-radian term combine as a MAX, never a sum ──
    {
        const float dt = 0.05f, w = 0.8f, dth = w * dt;
        const float mvar = float(k_true[5]) * dt + float(k_true[3]) * dth;
        bool ok = true;
        for (const float sig : {0.001f, 0.2f})   // stated below the model, then above it
        {
            Integrator g(0.f); g.set_noise(model(k_true));
            g.add(0.f, 0.f, w, dt, -1.f, -1.f, sig);
            const float got = g.result().cov(2, 2), want = std::max(mvar, sig * sig * dt);
            const bool o = std::abs(got - want) <= 1e-6f * want;
            std::printf("1b. heading var, stated %.3f: got %.3e want max(model %.3e, stated %.3e)  (%s)\n",
                        sig, got, mvar, sig * sig * dt, o ? "ok" : "FAIL");
            ok = ok and o;
        }
        if (not ok) ++fails;
    }

    // ── 2. closed-loop recovery (clean, then with inconsistent posteriors) ──
    for (const bool adversarial : {false, true})
    {
        std::printf("-- %s\n", adversarial ? "adversarial posteriors" : "consistent posteriors");
        MotionNoiseVC::Params pp;
        pp.memory = 1e9;   // a stationary truth: forget nothing
        pp.k0 = {200e-6, 20e-6, 30e-6, 400e-6, 4e-6, 4e-6};   // deliberately wrong priors (2-4x)
        pp.k_sd = {1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4};        // weak
        MotionNoiseVC L; L.set_params(pp);
        const Eigen::Vector3d rho_true(2e-5, 1e-5, 1e-5);      // scan excess (posterior overconfident)
        const Eigen::Vector3d R_claim(1e-4, 1e-4, 2.5e-5);     // the solver's scan variance, body axes
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> U01(0.f, 1.f);
        std::normal_distribution<double> n01;
        // 60k corrections: the estimator is CONSISTENT (measured: 6k leaves the small per-second and
        // lateral terms prior-dominated, 60k recovers all within 20 %, 300k within ~20 % / 1 % for the
        // dominant ones). A live run is ~1k corrections, so ONE run pins the per-metre / per-radian
        // terms; the small ones need evidence carried across runs.
        for (int s = 0; s < 60000; ++s)
        {
            std::array<double, 6> kh;
            for (int j = 0; j < 6; ++j) kh[j] = L.k(j);
            const float th0 = 6.28f * U01(rng);
            Integrator ia(th0), it(th0);           // applied (k̂) and true (k) — same motion
            ia.set_noise(model(kh)); it.set_noise(model(k_true));
            const float r = U01(rng);
            const int n = 1 + int(U01(rng) * 80.f);
            for (int i = 0; i < n; ++i)
            {
                if (r < 0.3f)      { ia.add(0.f, 0.f, 0.f, 0.05f);  it.add(0.f, 0.f, 0.f, 0.05f); }   // parked
                else if (r < 0.8f) { ia.add(0.01f, 0.45f, 0.1f, 0.05f); it.add(0.01f, 0.45f, 0.1f, 0.05f); } // drive
                else               { ia.add(0.f, 0.f, 0.9f, 0.05f); it.add(0.f, 0.f, 0.9f, 0.05f); }  // pivot
            }
            const float th = th0 + ia.result().delta[2];
            Eigen::Matrix3f T;   // body axes in world: forward, lateral, heading
            T << -std::sin(th), std::cos(th), 0.f,
                  std::cos(th), std::sin(th), 0.f,
                  0.f,          0.f,          1.f;
            const Eigen::Matrix3f P_prev = Eigen::Vector3f(2.5e-5f, 2.5e-5f, 1e-5f).asDiagonal();
            const Eigen::Matrix3f Qa = ia.result().cov, Qt = it.result().cov;
            const Eigen::Matrix3f P_pred = P_prev + Qa;
            const Eigen::Matrix3f Rw = T * R_claim.cast<float>().asDiagonal() * T.transpose();
            const Eigen::Matrix3f K = P_pred * (P_pred + Rw).inverse();
            Eigen::Matrix3f P_post = (Eigen::Matrix3f::Identity() - K) * P_pred;
            // ADVERSARIAL (the live failure, 2026-10-05): the solver's window posterior is not a single
            // Kalman update and, once the motion noise is tight, comes out ABOVE the prediction on some
            // axis. One in four corrections reports a posterior inflated past P_pred on a random body axis.
            if (adversarial and U01(rng) < 0.25f)
            {
                const Eigen::Vector3f t = T.col(int(U01(rng) * 2.999f));
                P_post += 1.5f * t.dot(P_pred * t) * (t * t.transpose());
            }
            // ν ~ N(0, P_prev + Q_true + R_true): what actually happened
            const Eigen::Matrix3d S = (P_prev + Qt).cast<double>()
                                    + (T * (R_claim + rho_true).cast<float>().asDiagonal() * T.transpose()).cast<double>();
            const Eigen::Matrix3d Lc = S.llt().matrixL();
            const Eigen::Vector3d nu = Lc * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
            const Eigen::Vector3f c = (K.cast<double>() * nu).cast<float>();
            std::array<Eigen::Matrix3f, 6> Uu;
            for (int j = 0; j < 6; ++j) Uu[j] = ia.result().unit[j];
            L.observe(c, T, P_pred, P_post, Qa, Uu);
        }
        const char *names[6] = {"k_long", "k_lat", "k_lat_turn", "k_th_turn", "k_t_trans", "k_t_rot"};
        // Tolerances: the per-metre/per-radian terms are well informed here; the per-second ones are
        // small against the scan variance on short parked stretches, so they get a looser bar.
        const double tol[6] = {0.15, 0.35, 0.35, 0.15, 0.6, 0.6};
        for (int j = 0; j < 6; ++j)
        {
            const double e = std::abs(L.k(j) - k_true[j]) / k_true[j];
            const bool ok = e < tol[j];
            std::printf("2. %-10s true %.2e learnt %.2e  (rel err %.2f, bar %.2f)  %s\n",
                        names[j], k_true[j], L.k(j), e, tol[j], ok ? "ok" : "FAIL");
            if (not ok and not adversarial) ++fails;
        }
        for (int b = 0; b < 3; ++b)
        {
            const double e = std::abs(L.rho(b) - rho_true(b)) / rho_true(b);
            const bool ok = e < 0.5;
            std::printf("2. rho[%d]     true %.2e learnt %.2e  (rel err %.2f, bar 0.50)  %s\n",
                        b, rho_true(b), L.rho(b), e, ok ? "ok" : "FAIL");
            if (not ok and not adversarial) ++fails;
        }
        // 2b (adversarial) is INFORMATIONAL: it documents why MotionNoiseLearn is OFF live
        // (motion_noise_vc.h header). It does not count toward PASS.
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails;
}
