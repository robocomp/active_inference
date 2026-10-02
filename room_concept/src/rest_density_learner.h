/*  rest_density_learner.h — the rest mixture's own stationary density, learnt online by EM.
 *
 *  WHY
 *  ---
 *  The rest-on-prediction mixture (RoomConcept::build_motion_prior_selection, PreintZuptOnPrediction)
 *  decides per cycle whether an odometry increment is rest noise N(0, d^2 T) or motion (uniform over
 *  what the base can do). It borrowed d from the preintegrator's PreintZuptDensity*, which describe a
 *  lumped PARKED VELOCITY NOISE for covariance shaping, not the odometry's own noise at rest. Measured
 *  on the truly-still cycles of three sim runs (2026-10-01/02) the wheels' rest noise is 4.5e-4 m/sqrt(s)
 *  and 6.4e-4 rad/sqrt(s), 14x and 16x below the borrowed values -- so the mixture called everything
 *  under ~7 cm/s "rest", and the registered slow-approach test failed in every band (thesis §ex-rest:
 *  85-96 % of creeping motion withheld from the prediction).
 *
 *  THE MODEL
 *  ---------
 *  The mixture's own posterior is the E-step: r = P(rest | m) for each cycle's coupled motion magnitude
 *  m over an interval T. The M-step for a zero-mean Gaussian with variance d^2 T is
 *      d^2 = sum r m^2 / T  /  sum r.
 *  Run online, one cycle at a time, with sufficient statistics.
 *
 *  RETENTION BY EXCITATION, AND A WEAK PRIOR
 *  -----------------------------------------
 *  The statistics decay only by REST-weighted evidence (each cycle multiplies them by exp(-r/kMemory)),
 *  so an hour of driving -- r ~ 0 -- forgets nothing; the memory is ~kMemory rest cycles. The configured
 *  density enters as ONE pseudo-cycle: conservative (wide) at a cold start, outweighed within seconds of
 *  standing still. Never a gate: a robot that has never stood still keeps the prior.
 *
 *  EVIDENCE ACROSS CYCLES (RestWindow)
 *  -----------------------------------
 *  Judged one 50 ms cycle at a time, motion slower than the odometry's noise over that cycle (~0.7 cm/s
 *  with the measured density) is indistinguishable from rest -- and creeping AT that crossover is then
 *  partly classed as rest and widens d (selftest: 2.2x after 5 min at 0.7 cm/s, after which a 1 cm/s
 *  creep passed only 10 % of its motion). Rest noise and motion differ across cycles, not within one:
 *  rest increments point anywhere, so their signed SUM over K cycles grows like sqrt(K) (a random walk,
 *  variance d^2 * K T), while a persisting velocity grows like K. So the mixture is evaluated on the
 *  signed sum over a short window; the crossover speed falls by ~sqrt(K) and the same rest density
 *  describes the window exactly (variance d^2 * T_window). The window length is the time over which a
 *  velocity is assumed to persist -- a model constant, configured, not a threshold on anything.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <deque>

namespace rc::preint
{
    class RestDensityLearner
    {
    public:
        static constexpr double kMemory      = 30000.0;  ///< rest-weighted cycles (~25 min parked at 20 Hz)
        /// ONE pseudo-cycle. The configured density is ~200x the true rest VARIANCE, and a pseudo-cycle at
        /// that variance weighs like ~200 real ones: at 50 it held the estimate at 1.4x the truth after
        /// 10 min parked (selftest). It sets the cold start and nothing more.
        static constexpr double kPriorCycles = 1.0;

        /// The configured density (rad/sqrt(s) or m/sqrt(s)): the prior, not a floor.
        void set_prior(float d) noexcept { if (d > 0.f and std::isfinite(d)) prior_d2_ = double(d) * d; }

        /// Current rest density.
        [[nodiscard]] float density() const noexcept
        {
            return static_cast<float>(std::sqrt((s_rm2_ + kPriorCycles * prior_d2_) / (s_r_ + kPriorCycles)));
        }

        /// One cycle: coupled motion magnitude m over T seconds, with the mixture's rest responsibility r.
        void add(float m, float T, float r) noexcept
        {
            if (not (T > 0.f) or not std::isfinite(m) or not (r >= 0.f) or r > 1.f) return;
            const double decay = std::exp(-double(r) / kMemory);
            s_r_   = decay * s_r_   + r;
            s_rm2_ = decay * s_rm2_ + r * double(m) * m / T;
            n_ += r;
        }
        [[nodiscard]] double rest_cycles() const noexcept { return n_; }

        /// The mixture itself, so the agent and the selftest evaluate exactly the same thing: P(moving).
        /// rest: N(m; 0, d^2 T); moving: uniform of half-width `span` (what the base can do in T).
        [[nodiscard]] static float p_moving(float m, float d, float T, float span) noexcept
        {
            const float sigma = d * std::sqrt(std::max(T, 1e-6f));
            if (not (sigma > 0.f) or not (span > 0.f)) return 1.f;
            const float rest   = std::exp(-0.5f * m * m / (sigma * sigma)) / (std::sqrt(2.f * float(M_PI)) * sigma);
            const float moving = 1.f / (2.f * span);
            const float den = rest + moving;
            return den > 0.f ? moving / den : 1.f;
        }

    private:
        double prior_d2_ = 6.4e-3 * 6.4e-3;
        double s_r_ = 0.0, s_rm2_ = 0.0, n_ = 0.0;
    };

    /// Signed sum of the last `window_s` seconds of increments, and the coupled motion magnitudes the
    /// mixture judges (translation and rotation coupled through the lever arm, so a pivot is not a stop).
    class RestWindow
    {
    public:
        struct Out { float m_tr = 0.f, m_ro = 0.f, T = 0.f; };
        void set_window(float s) noexcept { if (s > 0.f and std::isfinite(s)) window_s_ = s; }
        Out push(float dx, float dy, float dth, float T, float lever) noexcept
        {
            buf_.push_back({dx, dy, dth, T}); sx_ += dx; sy_ += dy; sth_ += dth; st_ += T;
            while (buf_.size() > 1 and st_ - buf_.front().T >= window_s_)
            {
                const auto &f = buf_.front();
                sx_ -= f.dx; sy_ -= f.dy; sth_ -= f.dth; st_ -= f.T; buf_.pop_front();
            }
            const float L = std::max(lever, 1e-3f);
            const float p = std::hypot(float(sx_), float(sy_)), a = std::abs(float(sth_));
            return {p + L * a, a + p / L, float(st_)};
        }
    private:
        struct E { float dx, dy, dth, T; };
        std::deque<E> buf_;
        double sx_ = 0, sy_ = 0, sth_ = 0, st_ = 0;
        float window_s_ = 0.5f;
    };
}
