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

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
