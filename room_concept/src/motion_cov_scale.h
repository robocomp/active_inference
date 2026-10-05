/*  motion_cov_scale.h — how much of the propagated motion noise is real? A per-axis variance scale, learnt
 *  from the localiser's own corrections.
 *
 *  WHY (2026-10-04)
 *  ----------------
 *  The per-axis surprise log showed the motion covariance too WIDE on every body axis: sum(c^2) over
 *  sum(P_pred - P_post) = 0.58 forward, 0.27 lateral, 0.18 heading on corrections one cycle apart, falling
 *  towards 0.01 over long open-loop stretches. The legacy covariance added constant per-CYCLE sigma floors
 *  (2 cm, 0.01 rad) that summed as a random walk at the localiser's rate. Preintegration replaces those with
 *  densities per sqrt(s); this learns how far those densities (lumped model error + the sensors' stated
 *  noise) are from what the corrections actually show, per axis, instead of asserting it.
 *
 *  THE MODEL
 *  ---------
 *  Between two scored corrections the predictive variance on body axis a is
 *        P_pred,a = P_prev,a + kappa_a * Q_a
 *  (P_prev: the posterior at the previous correction; Q: the motion noise accumulated over the stretch, AS
 *  PROPAGATED, i.e. before this scale is applied). Under an honest model the correction c has
 *        E[c_a^2] = P_pred,a - P_post,a      =>     r = (c_a^2 - P_prev,a + P_post,a) / Q_a
 *  is an unbiased sample of kappa_a. Each sample is weighted by the information it carries about kappa,
 *  (kappa Q / S)^2 with S = P_pred,a: a stretch whose variance is mostly the previous posterior says little
 *  about the motion noise (the same variance-component weight the rest-noise learner uses). One pseudo-
 *  observation of prior at kappa = 1 (the configured densities) and exponential forgetting.
 *
 *  ⚠ KNOWN BIAS, MEASURED NOT MODELLED: the early-exit gate decides which stretches get scored. A stretch
 *  ends in a solve when the prediction looked bad, so long stretches are selected for large errors and
 *  consecutive solves happen when the scan fits badly. tools/surprise_report.py splits the evidence by
 *  open-loop length so the size of that selection can be read off a run.
 *  Only the mean is kept positive (numerical epsilon), as in rest_density_learner.h.
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace rc::preint
{
    class MotionCovScale
    {
    public:
        static constexpr int N = 3;   // forward, lateral, heading (body frame)
        struct Params
        {
            double memory  = 2000.0;   ///< corrections of evidence remembered
            double prior_w = 1.0;      ///< pseudo-observations at kappa = 1
        };
        void set_params(const Params &p) noexcept { p_ = p; }

        /// One scored correction on axis a: c = correction, p_prev = posterior variance at the previous
        /// correction, q = UNSCALED motion noise accumulated since, p_post = posterior variance now.
        void observe(int a, double c, double p_prev, double q, double p_post) noexcept
        {
            if (a < 0 or a >= N or not (q > 0.0) or not std::isfinite(c) or not std::isfinite(p_prev)
                or not std::isfinite(p_post)) return;
            const double k = kappa(a);
            const double S = std::max(p_prev + k * q, 1e-30);
            const double w = std::pow(k * q / S, 2.0);
            const double r = (c * c - p_prev + p_post) / q;
            const double decay = std::exp(-1.0 / p_.memory);
            sw_[a] = decay * sw_[a] + w;
            sr_[a] = decay * sr_[a] + w * r;
            ++n_[a];
        }
        [[nodiscard]] double kappa(int a) const noexcept
        {
            return std::max((sr_[a] + p_.prior_w) / (sw_[a] + p_.prior_w), 1e-6);
        }
        [[nodiscard]] long samples(int a) const noexcept { return n_[a]; }

    private:
        Params p_{};
        std::array<double, N> sw_{}, sr_{};
        std::array<long, N> n_{};
    };
}
