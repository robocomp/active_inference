// Offline validation of rc::calib::BatchEstimator against KNOWN parameters.
//
// This is the injected-error protocol done on synthetic data: plant a parameter, generate the
// episodes it would produce, and check the estimator recovers it. It also checks the case that
// matters more than recovery -- that a parameter the data cannot separate is reported as UNINFORMED
// rather than as a confident wrong number.
#include "../src/calibration_estimator.h"
#include <cstdio>
#include <fstream>
#include <locale>
#include <random>
#include <vector>

using namespace rc::calib;

namespace {
std::mt19937 rng(12345);
float noise(float s) { return std::normal_distribution<float>(0.f, s)(rng); }

// Generate one episode as the robot+model would produce it under the true parameters.
Episode make(float d_fwd, float d_th, float dur, const float truth[P_COUNT], float sig_pos,
             float sig_th, float d_lat = 0.f)
{
    Episode e;
    e.d_forward = d_fwd; e.d_theta = d_th; e.duration = dur; e.d_lateral = d_lat;
    // Gyro-only heading (legacy-equivalent rows): every segment's weight is on the gyro, while the
    // curvature of unequal wheels still rides on the distance travelled.
    e.th_gyro = d_th; e.t_gyro = dur; e.th_wheel = 0.f; e.fwd_wheel = d_fwd;
    e.r_forward =  truth[P_K_V]     * d_fwd            + noise(sig_pos);
    e.r_lateral = -truth[P_EPS_YAW] * d_fwd
                 + truth[P_K_LAT]   * d_lat            + noise(sig_pos);
    e.r_theta   =  truth[P_K_OMEGA] * d_th
                 - truth[P_B_OMEGA] * dur          // the gyro reads w + b: a bias is OWED back
                 + truth[P_DK_WHEEL]* d_fwd            + noise(sig_th);
    e.pos_var = sig_pos * sig_pos; e.theta_var = sig_th * sig_th;
    return e;
}

int failures = 0;
float cond_separable = 0.f;   // set by test 2, compared against by test 3
void check(bool cond, const char* what)
{
    std::printf("   %-58s %s\n", what, cond ? "PASS" : "*** FAIL ***");
    if (not cond) ++failures;
}
} // namespace

int main()
{
    const float sig_pos = 0.004f, sig_th = 0.002f;

    // ---- 1. straights only: k_v and eps_yaw recoverable, gyro params NOT excited
    {
        float truth[P_COUNT] = {-0.012f, 0.0093f, 0.f, 0.f, 0.f, 0.f};   // -1.2% scale, 0.53 deg yaw
        BatchEstimator est; est.configure({}, 64);
        std::uniform_real_distribution<float> L(0.5f, 5.0f);
        for (int i = 0; i < 60; ++i) { const float d = L(rng); est.add(make(d, 0.f, d / 0.5f, truth, sig_pos, sig_th)); }
        const auto r = est.solve();
        std::printf("1. STRAIGHT-ONLY window (%d episodes)\n", r.episodes);
        std::printf("   k_v      %+.5f (truth %+.5f) sigma %.5f informed=%d\n", r.value[P_K_V], truth[P_K_V], r.sigma[P_K_V], (int)r.informed[P_K_V]);
        std::printf("   eps_yaw  %+.5f (truth %+.5f) sigma %.5f informed=%d\n", r.value[P_EPS_YAW], truth[P_EPS_YAW], r.sigma[P_EPS_YAW], (int)r.informed[P_EPS_YAW]);
        std::printf("   k_omega  %+.5f (truth  0.00000) sigma %.5f informed=%d\n", r.value[P_K_OMEGA], r.sigma[P_K_OMEGA], (int)r.informed[P_K_OMEGA]);
        check(std::abs(r.value[P_K_V] - truth[P_K_V]) < 0.002f, "k_v recovered");
        check(std::abs(r.value[P_EPS_YAW] - truth[P_EPS_YAW]) < 0.002f, "eps_yaw recovered");
        check(not r.informed[P_K_OMEGA], "k_omega correctly reported UNINFORMED (no rotation)");
    }

    // ---- 2. rotation with VARIED rate: scale and bias separable
    {
        float truth[P_COUNT] = {0.f, 0.f, -0.029f, 3.0e-4f, 0.f, 0.f};   // -2.9% gyro scale + a real bias
        BatchEstimator est; est.configure({}, 96);
        std::uniform_real_distribution<float> R(0.3f, 3.0f), W(0.2f, 1.2f);
        for (int i = 0; i < 90; ++i) { const float th = R(rng), w = W(rng); est.add(make(0.f, th, th / w, truth, sig_pos, sig_th)); }
        const auto r = est.solve();
        cond_separable = r.condition;
        std::printf("\n2. ROTATION, VARIED rate (%d episodes)  condition %.1f\n", r.episodes, r.condition);
        std::printf("   k_omega  %+.5f (truth %+.5f) sigma %.5f informed=%d\n", r.value[P_K_OMEGA], truth[P_K_OMEGA], r.sigma[P_K_OMEGA], (int)r.informed[P_K_OMEGA]);
        std::printf("   b_omega  %+.6f (truth %+.6f) sigma %.6f informed=%d\n", r.value[P_B_OMEGA], truth[P_B_OMEGA], r.sigma[P_B_OMEGA], (int)r.informed[P_B_OMEGA]);
        check(std::abs(r.value[P_K_OMEGA] - truth[P_K_OMEGA]) < 0.004f, "k_omega recovered");
        check(std::abs(r.value[P_B_OMEGA] - truth[P_B_OMEGA]) < 2.0e-4f, "b_omega separated from scale");
    }

    // ---- 3. THE DEGENERATE CASE: every episode at the SAME rate. Then d_theta = w*duration
    //         exactly, the two heading columns are collinear, and NO estimator can separate them.
    //         The right behaviour is to say so, not to split the difference confidently.
    {
        float truth[P_COUNT] = {0.f, 0.f, -0.029f, 0.f, 0.f, 0.f};
        BatchEstimator est; est.configure({}, 96);
        std::uniform_real_distribution<float> R(0.3f, 3.0f);
        const float w_fixed = 0.6f;
        for (int i = 0; i < 90; ++i) { const float th = R(rng); est.add(make(0.f, th, th / w_fixed, truth, sig_pos, sig_th)); }
        const auto r = est.solve();
        std::printf("\n3. ROTATION, CONSTANT rate -- scale and bias are COLLINEAR (%d episodes)\n", r.episodes);
        std::printf("   condition %.1f (large = a direction is unobserved)\n", r.condition);
        std::printf("   k_omega  %+.5f sigma %.5f informed=%d\n", r.value[P_K_OMEGA], r.sigma[P_K_OMEGA], (int)r.informed[P_K_OMEGA]);
        std::printf("   b_omega  %+.6f sigma %.6f informed=%d\n", r.value[P_B_OMEGA], r.sigma[P_B_OMEGA], (int)r.informed[P_B_OMEGA]);
        check(not r.informed[P_B_OMEGA], "b_omega correctly reported UNINFORMED when collinear");
        check(r.condition > cond_separable * 3.f,
              "normalised condition number RANKS collinear worse than separable");
        std::printf("   (separable window scored %.1f, this one %.1f)\n", cond_separable, r.condition);
    }

    // ---- 4. mixed realistic driving: everything at once
    {
        float truth[P_COUNT] = {-0.012f, 0.0093f, -0.029f, 2.0e-4f, 0.035f, 0.006f};
        BatchEstimator est; est.configure({}, 256);
        std::uniform_real_distribution<float> L(0.f, 4.0f), R(-2.0f, 2.0f), W(0.2f, 1.2f),
                                             S(-0.8f, 0.8f);
        for (int i = 0; i < 240; ++i)
        {
            const float d = L(rng), th = R(rng), w = W(rng), lat = S(rng);
            est.add(make(d, th, std::max(std::abs(th) / w, d / 0.5f) + 0.2f, truth, sig_pos, sig_th, lat));
        }
        const auto r = est.solve();
        std::printf("\n4. MIXED driving (%d episodes)  condition %.1f\n", r.episodes, r.condition);
        for (int i = 0; i < P_COUNT; ++i)
            std::printf("   %-8s %+.6f (truth %+.6f) sigma %.6f informed=%d\n",
                        param_name(i).data(), r.value[i], truth[i], r.sigma[i], (int)r.informed[i]);
        check(std::abs(r.value[P_K_V] - truth[P_K_V]) < 0.003f, "k_v recovered under mixed driving");
        check(std::abs(r.value[P_K_OMEGA] - truth[P_K_OMEGA]) < 0.006f, "k_omega recovered under mixed driving");
        check(std::abs(r.value[P_EPS_YAW] - truth[P_EPS_YAW]) < 0.003f, "eps_yaw NOT contaminated by turns");
        check(std::abs(r.value[P_K_LAT] - truth[P_K_LAT]) < 0.006f,
              "lateral scale separated from mount yaw (same component)");
        check(std::abs(r.value[P_DK_WHEEL] - truth[P_DK_WHEEL]) < 0.003f,
              "per-wheel mismatch separated from gyro scale AND bias (same component)");
    }

    // ---- 5. TWO heading factors: wheel and gyro scales are different numbers, and both come back
    // Episodes are mixtures of segments with different gyro weights w_g, as the fused integrator
    // produces them: fast turns lean on the gyro, slow turns and stops on the wheels. The wheels
    // over-report rotation by 7% (scrubbing) while the gyro is 1% short with a bias.
    {
        float truth[P_COUNT] = {0.f, 0.f, 0.010f, 3.0e-4f, 0.f, 0.006f, -0.070f};
        BatchEstimator est; est.configure({}, 512);
        std::uniform_real_distribution<float> U(0.f, 1.f);
        for (int i = 0; i < 400; ++i)
        {
            Episode e;
            const int nseg = 6;
            for (int k = 0; k < nseg; ++k)
            {
                const float rate = (U(rng) < 0.3f) ? 0.f : (U(rng) < 0.5f ? 0.15f : 0.9f) * (U(rng) < 0.5f ? -1.f : 1.f);
                const float v    = (U(rng) < 0.5f) ? 0.f : 0.35f;
                const float dt   = 0.5f + U(rng);
                const float w_g  = rate == 0.f ? 0.6f : (std::abs(rate) < 0.5f ? 0.35f : 0.95f);
                const float th = rate * dt, fwd = v * dt;
                e.d_forward += fwd; e.d_theta += th; e.duration += dt;
                e.th_gyro += w_g * th; e.t_gyro += w_g * dt;
                e.th_wheel += (1.f - w_g) * th; e.fwd_wheel += (1.f - w_g) * fwd;
            }
            e.r_forward = noise(sig_pos); e.r_lateral = noise(sig_pos);
            e.r_theta = truth[P_K_OMEGA] * e.th_gyro - truth[P_B_OMEGA] * e.t_gyro
                      + truth[P_K_OMEGA_W] * e.th_wheel + truth[P_DK_WHEEL] * e.fwd_wheel + noise(sig_th);
            e.pos_var = sig_pos * sig_pos; e.theta_var = sig_th * sig_th;
            est.add(e);
        }
        const auto r = est.solve();
        std::printf("\n5. TWO HEADING FACTORS (%d episodes)  condition %.1f\n", r.episodes, r.condition);
        for (int p : {P_K_OMEGA, P_B_OMEGA, P_DK_WHEEL, P_K_OMEGA_W})
            std::printf("   %-9s %+.6f (truth %+.6f) sigma %.6f informed=%d\n",
                        param_name(p).data(), r.value[p], truth[p], r.sigma[p], (int)r.informed[p]);
        check(std::abs(r.value[P_K_OMEGA_W] - truth[P_K_OMEGA_W]) < 0.01f, "WHEEL rotation scale recovered (-7%)");
        check(std::abs(r.value[P_K_OMEGA] - truth[P_K_OMEGA]) < 0.004f, "GYRO scale recovered separately (+1%)");
        check(std::abs(r.value[P_B_OMEGA] - truth[P_B_OMEGA]) < 1.0e-4f, "gyro bias recovered with its PHYSICAL sign");
        check(std::abs(r.value[P_DK_WHEEL] - truth[P_DK_WHEEL]) < 0.003f, "wheel curvature recovered on the wheel factor");
    }

    // ---- 6. CLOSED LOOP: applying the estimate and re-solving must converge, not ratchet
    // Each round applies the current estimate; the next round's rows carry it in p_applied and their
    // residual is what is LEFT. Before the sign fix the bias column made this a gain-2 iteration.
    {
        float truth[P_COUNT] = {0.f, 0.f, 0.f, 4.0e-4f, 0.f, 0.f, 0.f};
        Eigen::Matrix<float, P_COUNT, 1> applied = Eigen::Matrix<float, P_COUNT, 1>::Zero();
        BatchEstimator est; est.configure({}, 64);
        float worst = 0.f, last = 0.f, sig = 0.f;
        std::uniform_real_distribution<float> R(-1.5f, 1.5f), W(0.2f, 1.2f);
        for (int round = 0; round < 12; ++round)
        {
            for (int i = 0; i < 64; ++i)
            {
                const float th = R(rng), dur = std::abs(th) / W(rng) + 2.f;
                Episode e; e.d_theta = th; e.duration = dur; e.th_gyro = th; e.t_gyro = dur;
                // Remaining heading error after the model subtracted the applied bias.
                e.r_theta = -(truth[P_B_OMEGA] - applied[P_B_OMEGA]) * dur + noise(sig_th);
                e.pos_var = sig_pos * sig_pos; e.theta_var = sig_th * sig_th;
                e.p_applied = applied;
                est.add(e);
            }
            const auto r = est.solve();
            applied[P_B_OMEGA] = r.value[P_B_OMEGA];
            last = r.value[P_B_OMEGA];
            sig  = r.sigma[P_B_OMEGA];
            worst = std::max(worst, std::abs(last - truth[P_B_OMEGA]));
        }
        std::printf("\n6. CLOSED LOOP bias: final %+.6f (truth %+.6f), worst |err| over rounds %.6f\n",
                    last, truth[P_B_OMEGA], worst);
        std::printf("   posterior sigma %.6f\n", sig);
        check(std::abs(last - truth[P_B_OMEGA]) < 3.f * sig, "closed-loop bias converges to the truth (3 sigma)");
        check(worst < 2.0f * truth[P_B_OMEGA], "and never overshoots on the way (no gain-2 ratchet)");
    }

    // ── 7. LEVER ARM (helios offset from the axle midpoint) ───────────────────────────────────────
    // Forward model PER CORRECTION k: r_k = (I - R(dth_k)^T) * lever, body X = lateral, Y = forward.
    // Episodes are SUMS of per-cycle corrections (motion_calibration.h observe()), so the covariates are
    // lever_s = sum sin(dth_k), lever_c = sum (1 - cos dth_k). Live: 10-60 corrections of a few
    // hundredths of a radian each; rarely one correction spanning a large turn. Both are generated.
    // Plan 2026-10-05 Task 1 (r1).
    auto lever_rows = [&](BatchEstimator &est, const float truth[P_COUNT], bool spins, bool straights,
                          const Eigen::Matrix<float, P_COUNT, 1> &applied)
    {
        std::uniform_real_distribution<float> TH(-1.5f, 1.5f), D(0.5f, 2.0f);
        std::uniform_int_distribution<int> M(10, 60);
        for (int i = 0; i < 200; ++i)
        {
            const bool spin = spins and (not straights or i % 2 == 0);
            // ⚠ DEVIATION from the plan's 0.02*TH (+-30 mrad): at 4 mm episode noise, 200 straights
            // carrying +-30 mrad of incidental rotation DO inform the lever (sigma ~16 mm against a
            // 50 mm prior), so "straight-only => uninformed" would be false for a correct estimator.
            // A straight here carries +-3 mrad, which is what "straight-only" is meant to model.
            const float dth = spin ? TH(rng) : 0.002f * TH(rng);
            const float dfw = spin ? 0.02f * D(rng) : D(rng);
            const int m = (i % 10 == 9) ? 1 : M(rng);            // 1 in 10: a single end-of-episode correction
            float ls = 0.f, lc = 0.f;
            for (int k = 0; k < m; ++k) { const float t = dth / m; ls += std::sin(t); lc += 1.f - std::cos(t); }
            const float lx = truth[P_LEVER_X] - applied[P_LEVER_X];
            const float ly = truth[P_LEVER_Y] - applied[P_LEVER_Y];
            Episode e;
            e.d_forward = dfw; e.d_theta = dth; e.duration = 1.f + std::abs(dth);
            e.th_gyro = dth; e.t_gyro = e.duration; e.fwd_wheel = dfw;
            e.lever_s = ls; e.lever_c = lc;
            e.r_lateral = lc * lx - ls * ly - truth[P_EPS_YAW] * dfw + noise(0.004f);
            e.r_forward = ls * lx + lc * ly + truth[P_K_V] * dfw     + noise(0.004f);
            e.r_theta   = noise(0.001f);
            e.pos_var = 0.004f * 0.004f; e.theta_var = 0.001f * 0.001f;
            e.p_applied = applied;
            est.add(e);
        }
    };
    {
        float truth[P_COUNT] = {};
        truth[P_LEVER_X] = 0.03f; truth[P_LEVER_Y] = -0.02f; truth[P_EPS_YAW] = 0.01f;
        const Eigen::Matrix<float, P_COUNT, 1> none = Eigen::Matrix<float, P_COUNT, 1>::Zero();

        // A: mixed driving recovers both lever components AND eps_yaw.
        BatchEstimator a; a.configure(Prior{}, 512); lever_rows(a, truth, true, true, none);
        const auto ra = a.solve();
        std::printf("\n7A. lever x %+.4f (truth %+.4f) +- %.4f | y %+.4f (truth %+.4f) +- %.4f | eps %+.4f\n",
                    ra.value[P_LEVER_X], truth[P_LEVER_X], ra.sigma[P_LEVER_X],
                    ra.value[P_LEVER_Y], truth[P_LEVER_Y], ra.sigma[P_LEVER_Y], ra.value[P_EPS_YAW]);
        check(std::abs(ra.value[P_LEVER_X] - truth[P_LEVER_X]) < 3.f * ra.sigma[P_LEVER_X], "7A lever x recovered (3 sigma)");
        check(std::abs(ra.value[P_LEVER_Y] - truth[P_LEVER_Y]) < 3.f * ra.sigma[P_LEVER_Y], "7A lever y recovered (3 sigma)");
        check(ra.informed[P_LEVER_X] and ra.informed[P_LEVER_Y], "7A lever informed under rotation");
        check(std::abs(ra.value[P_EPS_YAW] - truth[P_EPS_YAW]) < 3.f * ra.sigma[P_EPS_YAW], "7A eps_yaw still recovered beside the lever");

        // B: straight-only driving must leave the lever UNINFORMED at its prior, not confidently wrong.
        BatchEstimator b; b.configure(Prior{}, 512); lever_rows(b, truth, false, true, none);
        const auto rb = b.solve();
        std::printf("7B. straight-only: lever sigma x %.4f y %.4f (prior %.4f)\n",
                    rb.sigma[P_LEVER_X], rb.sigma[P_LEVER_Y], Prior{}.sigma_lever);
        check(not rb.informed[P_LEVER_X] and not rb.informed[P_LEVER_Y], "7B straight-only: lever uninformed");

        // C: closed loop -- the applied lever is undone through p_applied and the total is recovered.
        BatchEstimator c; c.configure(Prior{}, 512);
        Eigen::Matrix<float, P_COUNT, 1> half = none; half[P_LEVER_X] = 0.015f; half[P_LEVER_Y] = -0.01f;
        lever_rows(c, truth, true, true, half);
        const auto rc_ = c.solve();
        check(std::abs(rc_.value[P_LEVER_X] - truth[P_LEVER_X]) < 3.f * rc_.sigma[P_LEVER_X], "7C closed loop: TOTAL lever recovered");

        // D: information() reproduces solve().value exactly.
        const auto info = a.information();
        const Eigen::Matrix<float, P_COUNT, 1> p = info.H.ldlt().solve(info.b);
        check((p - ra.value).cwiseAbs().maxCoeff() < 1e-5f, "7D information() reproduces solve()");
    }
    {
        const char *path = "/tmp/calib_legacy_rows.csv";
        { std::ofstream f(path); f.imbue(std::locale::classic());
          f << "E,1,0,0.5,2,0.01,0,0.001,0.0001,0.0001,0.5,2,0,1,0,0,0,0,0,0,0\n"; }   // 13 + 7 fields
        BatchEstimator l; l.configure(Prior{}, 64);
        check(l.load(path) == 1, "7E a pre-lever (13+7) row still loads: p_applied lever = 0 (exact), "
                                 "lever_s = d_theta, lever_c = 0 (first-order APPROXIMATION)");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
