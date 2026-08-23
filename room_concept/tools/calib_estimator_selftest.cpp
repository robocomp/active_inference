// Offline validation of rc::calib::BatchEstimator against KNOWN parameters.
//
// This is the injected-error protocol done on synthetic data: plant a parameter, generate the
// episodes it would produce, and check the estimator recovers it. It also checks the case that
// matters more than recovery -- that a parameter the data cannot separate is reported as UNINFORMED
// rather than as a confident wrong number.
#include "../src/calibration_estimator.h"
#include <cstdio>
#include <random>
#include <vector>

using namespace rc::calib;

namespace {
std::mt19937 rng(12345);
float noise(float s) { return std::normal_distribution<float>(0.f, s)(rng); }

// Generate one episode as the robot+model would produce it under the true parameters.
Episode make(float d_fwd, float d_th, float dur, const float truth[P_COUNT], float sig_pos, float sig_th)
{
    Episode e;
    e.d_forward = d_fwd; e.d_theta = d_th; e.duration = dur;
    e.r_forward =  truth[P_K_V]     * d_fwd            + noise(sig_pos);
    e.r_lateral = -truth[P_EPS_YAW] * d_fwd            + noise(sig_pos);
    e.r_theta   =  truth[P_K_OMEGA] * d_th
                 + truth[P_B_OMEGA] * dur              + noise(sig_th);
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
        float truth[P_COUNT] = {-0.012f, 0.0093f, 0.f, 0.f};   // -1.2% scale, 0.53 deg yaw
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
        float truth[P_COUNT] = {0.f, 0.f, -0.029f, 3.0e-4f};   // -2.9% gyro scale + a real bias
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
        float truth[P_COUNT] = {0.f, 0.f, -0.029f, 0.f};
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
        float truth[P_COUNT] = {-0.012f, 0.0093f, -0.029f, 2.0e-4f};
        BatchEstimator est; est.configure({}, 128);
        std::uniform_real_distribution<float> L(0.f, 4.0f), R(-2.0f, 2.0f), W(0.2f, 1.2f);
        for (int i = 0; i < 120; ++i)
        {
            const float d = L(rng), th = R(rng), w = W(rng);
            est.add(make(d, th, std::max(std::abs(th) / w, d / 0.5f) + 0.2f, truth, sig_pos, sig_th));
        }
        const auto r = est.solve();
        std::printf("\n4. MIXED driving (%d episodes)  condition %.1f\n", r.episodes, r.condition);
        for (int i = 0; i < P_COUNT; ++i)
            std::printf("   %-8s %+.6f (truth %+.6f) sigma %.6f informed=%d\n",
                        param_name(i).data(), r.value[i], truth[i], r.sigma[i], (int)r.informed[i]);
        check(std::abs(r.value[P_K_V] - truth[P_K_V]) < 0.003f, "k_v recovered under mixed driving");
        check(std::abs(r.value[P_K_OMEGA] - truth[P_K_OMEGA]) < 0.006f, "k_omega recovered under mixed driving");
        check(std::abs(r.value[P_EPS_YAW] - truth[P_EPS_YAW]) < 0.003f, "eps_yaw NOT contaminated by turns");
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
