/*  rest_density_learner.h — is the body at rest? A velocity filter whose noise is learnt from its own
 *  innovations, and a Savage-Dickey test of v = 0.
 *
 *  WHY (the history, all measured on the simulated Shadow, 2026-10-01/02)
 *  ---------------------------------------------------------------------
 *  The rest-on-prediction mixture scales each predicted odometry increment by P(moving). Three versions
 *  failed the registered slow-approach test (thesis §ex-rest) or its offline replay:
 *    1. per-cycle mixture with the borrowed PreintZuptDensity* (14-16x the wheels' true rest noise):
 *       everything under ~7 cm/s called rest; 85-96 % of creeping motion withheld.
 *    2. the same with a rest density learnt by EM from raw increment magnitudes weighted by P(rest),
 *       judged over a 0.5 s window: the learnt density RAN AWAY (9.5e-4 -> 5.6e-3) once slow legs
 *       arrived before it had settled -- slow motion weighted as "rest" widened rest, which made more
 *       motion look like rest. Root cause: "moving" was a diffuse uniform, so a widened rest Gaussian
 *       always out-explained a cluster of slow motion.
 *    3. a two-regime IMM that re-mixed the velocity toward "unknown" whenever rest was likely: its
 *       moving prediction never tightened, so nothing under 8 cm/s was ever recognised (selftest).
 *
 *  THE MODEL
 *  ---------
 *  One constant-velocity Kalman filter per CHANNEL of the BODY frame (forward, lateral, rotation), ALWAYS on:
 *      v_t = v_{t-1} + w,   w ~ N(0, q T)          q = acc^2: how fast the base's velocity can change
 *      z_t = v_t T + n,     n ~ N(0, R),  R = d^2 T  d: the odometry's noise density (LEARNT)
 *  The NOISE is learnt from the filter's own innovations e = z - v_pred T, whose variance is
 *  P_pred T^2 + R, so each cycle gives an unbiased sample of R: e^2 - P_pred T^2, weighted by the
 *  information it carries about R, (R/S)^2 (a variance-component / REML weight). A steady creep is
 *  absorbed by the velocity estimate and leaves the innovations at noise size, so slow motion can no
 *  longer inflate the noise -- the runaway of version 2 has no path.
 *  REST is then a statement about the velocity: is v = 0? Against a moving prior v ~ N(0, v0^2) (what
 *  the base can do), the Savage-Dickey ratio gives the Bayes factor for the point hypothesis directly:
 *      BF_rest = N(0; v_post, P_post) / N(0; 0, v0^2),       per channel,
 *      P(moving) = 1 / (1 + BF_rest).
 *  ★ PER CHANNEL, IN THE BODY FRAME. Testing "all three axes zero" against "all free" charged the moving
 *  hypothesis an Occam factor ln(v0^2/P)/2 for every idle axis, so a 0.5 cm/s creep along one axis was
 *  called rest (selftest: 6-11 %). A differential base translates only along its heading, so forward,
 *  lateral and rotation are tested separately, each against its own one-dimensional prior.
 *  Persistence enters through the velocity state (a creep makes v_post settle away from 0 with a
 *  shrinking P_post), not through a window or a regime switch. Every constant is physical; none is a
 *  threshold on the data.
 *
 *  ⚠ A single cycle's e^2 - P T^2 can be negative; the running mean is what is used, and only that is
 *  kept positive, at a numerical epsilon, so that it stays a variance.
 */
#pragma once
#include <algorithm>
#include <cmath>

namespace rc::preint
{
    /// The LEGACY per-cycle two-hypothesis mixture (PreintZuptPredLearnRest = false): rest N(0, d^2 T) vs
    /// a uniform of half-width `span`. Kept so the configuration that FAILED the registered creep test
    /// stays reproducible; returns P(moving).
    [[nodiscard]] inline float mixture_p_moving(float m, float d, float T, float span) noexcept
    {
        const float sigma = d * std::sqrt(std::max(T, 1e-6f));
        if (not (sigma > 0.f) or not (span > 0.f)) return 1.f;
        const float rest   = std::exp(-0.5f * m * m / (sigma * sigma)) / (std::sqrt(2.f * float(M_PI)) * sigma);
        const float moving = 1.f / (2.f * span);
        const float den = rest + moving;
        return den > 0.f ? moving / den : 1.f;
    }

    /// One body-frame channel (forward, lateral or rotation): a velocity Kalman filter with a two-component
    /// process noise, its observation noise learnt from its own innovations, and P(moving) by Savage-Dickey.
    class RestMotionChannel
    {
    public:
        struct Params
        {
            /// Velocity changes are SMOOTH most cycles and occasionally JUMP (start, stop, speed change). A
            /// single random walk cannot be both: smooth enough to average a creep out of the noise, and loose
            /// enough that a step does not leave innovations that inflate the learnt noise (sweep: 0.1 m/s/sqrt(s)
            /// -> 1 cm/s creep only 66 % recognised; 0.003 -> noise inflated 5x by 5-8 cm/s steps). So the
            /// process noise is a two-component mixture, decided per cycle by the innovation.
            float acc      = 0.003f;   ///< per sqrt(s): smooth drift (m/s or rad/s)
            float acc_jump = 0.3f;     ///< per sqrt(s): a speed change, the base's acceleration scale
            float v0       = 0.58f;    ///< sd of the MOVING prior on velocity (uniform on +-1 m/s; rotation +-2 rad/s -> 1.15)
            float p_jump   = 0.02f;    ///< prior share of cycles in which the velocity jumps
            double memory  = 30000.0;  ///< cycles of noise evidence remembered (~25 min at 20 Hz)
        };
        void set_params(const Params &p) noexcept { p_ = p; }
        /// The configured density for this channel: ONE cycle of prior on the learnt noise.
        void set_prior(float d) noexcept { if (d > 0.f and std::isfinite(d)) prior_ = double(d) * d; }
        [[nodiscard]] float density() const noexcept { return float(std::sqrt(noise_d2())); }
        [[nodiscard]] float p_moving() const noexcept { return pm_; }

        /// One cycle: increment z over T s. Returns P(moving) for this channel.
        float step(float z, float T) noexcept
        {
            if (not (T > 0.f) or not std::isfinite(z)) return pm_;
            const double qs = double(p_.acc) * p_.acc, qj = double(p_.acc_jump) * p_.acc_jump;
            const double R  = noise_d2() * T;
            const double e  = z - v_ * T;                                   // innovation
            // which process noise acted this cycle: smooth drift or a jump, by the innovation's likelihood
            const double Ps = P_ + qs * T, Pj = P_ + qj * T;
            const double Ss = Ps * T * T + R, Sj = Pj * T * T + R;
            const double ls = std::log(1.0 - p_.p_jump) - 0.5 * (e * e / Ss + std::log(Ss));
            const double lj = std::log(double(p_.p_jump)) - 0.5 * (e * e / Sj + std::log(Sj));
            const double wj = 1.0 / (1.0 + std::exp(std::clamp(ls - lj, -700.0, 700.0)));
            // Learn the noise from the innovation: E[e^2] = P T^2 + R => a sample of d^2. ★ Against the SMOOTH
            // prediction Ps, weighted by 1 - wj (a jump informs the velocity, not the noise) and by (R/Ss)^2,
            // the information the cycle carries about R (REML weight). Using the mixed P made the subtraction
            // depend on the innovation and biased the noise 2.6x LOW; unweighted, the startup cycles (S >> R)
            // drove it to zero.
            const double decay = std::exp(-1.0 / p_.memory);
            const double w = (1.0 - wj) * (R / Ss) * (R / Ss);
            sw_ = decay * sw_ + w;
            se_ = decay * se_ + w * (e * e - Ps * T * T) / T;
            // Kalman update with the moment-matched prediction
            const double P = (1.0 - wj) * Ps + wj * Pj;
            const double S = P * T * T + R;
            const double K = P * T / S;
            v_ += K * e;
            P_  = (1.0 - K * T) * P;
            // Savage-Dickey: posterior density at v = 0 over the moving prior's density at 0
            const double v0 = double(p_.v0) * p_.v0;
            const double log_bf_rest = (-0.5 * v_ * v_ / P_ - 0.5 * std::log(P_)) + 0.5 * std::log(v0);
            pm_ = float(1.0 / (1.0 + std::exp(std::clamp(log_bf_rest, -700.0, 700.0))));
            return pm_;
        }

    private:
        [[nodiscard]] double noise_d2() const noexcept
        {
            return std::max((se_ + prior_) / (sw_ + 1.0), 1e-14);           // one cycle of prior
        }
        Params p_{};
        double prior_ = 6.4e-3 * 6.4e-3;
        double v_ = 0.0, P_ = 0.34, sw_ = 0.0, se_ = 0.0;
        float pm_ = 0.5f;
    };
}
