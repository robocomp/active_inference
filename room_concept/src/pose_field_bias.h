/*  pose_field_bias.h — the slow POSE-FIELD bias of the scan-only pose, learnt without ground truth, and the
 *  covariance term it contributes to the PUBLISHED pose.
 *
 *  WHY (Fable window memo §3, 2026-10-05; measured 2026-10-05/06)
 *  -------------------------------------------------------------
 *  The pose posterior is 5-10x overconfident against ground truth, and the error is not noise: it is FROZEN while
 *  the robot is parked and grows with distance driven, saturating over metres — a bias FIELD b(x) over position
 *  (map/scan systematics: a wall a few cm off moves every scan the same way at that place). Every scan sees the
 *  same b, so adding scans shrinks the reported sigma while b stays: the solver's covariance is CONDITIONAL on
 *  the map. A fixed floor cannot fix it — the size that is honest on one tour (40 mm / 0.9 deg) over-covers the
 *  quieter ones 2-3x — so it is learnt per session.
 *
 *  HOW, GT-FREE
 *  ------------
 *  z = x + b(x) + e is the scan-only pose (motion_noise_innov.h). Along an unbroken chain, for two samples a
 *  SPATIAL distance r apart (whatever path joined them),
 *        D = (z_n − z_m) − Σ odom  =  odometry error over the path between + b(x_n) − b(x_m) + e_n − e_m
 *        E|D|² − V_odo(path) = 4 σ_b² (1 − exp(−r/ℓ))          (x+y summed; e is ~1 mm and ignored)
 *  ★ r is the distance between the two PLACES, not the path driven: b is a function of position, so a lap that
 *    returns to where it started has b(x_n) = b(x_m) however far it drove. Binning by path length (the first
 *    version) read those revisits as "8 m apart, no bias" and under-estimated σ_b ~1.6x on a 4x6 m room
 *    (25 vs ~45 mm, replay of 2026-10-05 22-20). The odometry variance still accrues along the PATH.
 *  V_odo is the odometry variance the noise learner states for the same stretch (its per-cycle coefficients,
 *  learnt at ONE-cycle lags where b cancels). The excess over it is the bias field's structure function; its
 *  plateau is 4σ_b² and its rise length ℓ. Samples every `sample_spacing` metres of path; pairs at a few lags;
 *  per-lag means with exponential forgetting; ℓ chosen on a grid by least squares (model selection, not a
 *  gate); σ_b shrunk toward a weak prior σ0 by `prior_pairs` pseudo-pairs so a run with no driving reports the
 *  prior, not zero. σ_b² is projected onto ≥ 0 (a variance).
 *
 *  ★ THE AMPLITUDE IS A FIELD TOO (2026-10-11, Given-mode confidence 0)
 *  --------------------------------------------------------------------
 *  One global σ_b was published everywhere. Measured on the apartment layout (log 22-18-08): a 3.7 m stretch where
 *  the given layout does not explain the scan (median |SDF| 0.43-0.69 m, published pose 0.49-0.72 m off ground
 *  truth) drove σ_b from 3 cm to 24-28 cm; the robot then parked where its error was 9 mm and published
 *  sigma 28 cm for the rest of the session, because σ_b only changes when path accrues. That number was HONEST
 *  about the door region and WRONG about the parked spot: the bias field's amplitude is itself place-dependent,
 *  and a stationary model with one amplitude must summarise both places with one number. The published term is
 *  therefore the amplitude NEAR THE CURRENT PLACE: the same pair statistics, fitted with each pair weighted by
 *  how close BOTH of its ends are to where the robot is (kernel exp(−d/ℓ) at the fitted ℓ — no new constant),
 *  and under a two-component likelihood: a pair is explained by the local amplitude, or it belongs to the field
 *  elsewhere (the global plateau is that component's scale). Responsibilities, not a gate: the door pairs count
 *  for nothing at the parked spot because they cannot be explained by the amplitude there, and the parked-spot
 *  pairs count for nothing at the door for the same reason. Pairs are kept across chain breaks (they are facts
 *  about places), so a parked robot keeps what it learnt on the way in. The global fit is kept for ℓ, for the
 *  outlier scale and for the log.
 *
 *  THE PUBLISHED TERM: Σ_b = β·R, β = σ_b(x)² / (tr R_xy / 2), R = the current scan's CLAIMED covariance. The bias
 *  has the scan's own geometry: a common-mode error of the points that made the fix, so heading follows the
 *  scan's lever (R_θθ / R_xx) instead of needing its own estimate — GT-free heading was not identifiable
 *  (pivots pile odometry heading noise into short path lags). Added to the PUBLISHED covariance only, never to
 *  the solver's state. Selftest: tools/pose_field_bias_selftest.cpp.
 */
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace rc
{
    class PoseFieldBias
    {
    public:
        static constexpr int NL = 5, NG = 6;
        struct Params
        {
            double sample_spacing = 0.25;                              ///< m of path between samples
            std::size_t buffer = 64;                                   ///< samples kept (16 m of path)
            std::array<double, NL + 1> edges{0.25, 0.5, 1.0, 2.0, 4.0, 8.0};   ///< m, spatial-distance bins
            std::array<double, NG> len_grid{0.25, 0.5, 1.0, 2.0, 4.0, 8.0};
            double sigma0 = 0.02;                                      ///< m, the prior bias sigma
            double prior_pairs = 50.0;                                 ///< its weight, in pairs
            double memory_pairs = 20000.0;                             ///< forgetting, in pairs per lag
            std::size_t pair_store = 4096;                             ///< pairs kept for the LOCAL fit (~16 m of path)
            double scan_noise_var = 2.0e-6;                            ///< m², e_n − e_m for the ~1 mm scan noise (the ignored e)
            int local_em_iters = 3;                                    ///< EM passes per local fit, warm-started
        };

        void set_params(const Params &p) { p_ = p; reset_chain(); bins_ = {}; pairs_store_.clear(); pairs_ = 0; fit_valid_ = false; A_loc_ = 4.0 * p_.sigma0 * p_.sigma0; sigma_b_local_ = p_.sigma0; }
        [[nodiscard]] const Params &params() const noexcept { return p_; }

        /// One cycle. z: scan-only pose (world x, y, θ); R: its claimed covariance (world); odom: the odometry
        /// increment (world); odo_var_tr: the odometry variance the noise model states for THIS cycle, x + y
        /// summed; chain_ok: this cycle directly follows the previous one (else the chain restarts here).
        void observe(const Eigen::Vector3f &z, const Eigen::Matrix3f &R, const Eigen::Vector3f &odom,
                     float odo_var_tr, float /*odo_var_th*/, bool chain_ok)
        {
            if (not z.allFinite() or not odom.allFinite() or not std::isfinite(odo_var_tr)) { reset_chain(); return; }
            (void)R;
            // The place the term is asked about is where the robot IS, parked or not: the scan-only pose's own
            // position (world frame, so it is comparable across chains; its error is the bias itself, well inside ℓ).
            xc_ = z.head<2>().cast<double>();
            if (not chain_ok or buf_.empty())
            {
                reset_chain();
                buf_.push_back({z.head<2>().cast<double>(), Eigen::Vector2d::Zero(), 0.0, 0.0});
                fit_local();
                return;
            }
            O_ += odom.head<2>().cast<double>();
            V_ += std::max(0.0, static_cast<double>(odo_var_tr));
            S_ += odom.head<2>().cast<double>().norm();
            if (S_ - buf_.back().S < p_.sample_spacing) { fit_local(); return; }
            const Sample now{z.head<2>().cast<double>(), O_, V_, S_};
            const double forget = std::exp(-1.0 / p_.memory_pairs);
            for (const auto &m : buf_)
            {
                // the distance between the two PLACES, from the odometry chain (exact up to its own drift)
                const double r = (now.O - m.O).norm();
                const auto e = std::upper_bound(p_.edges.begin(), p_.edges.end(), r);
                if (e == p_.edges.begin() or e == p_.edges.end()) continue;   // outside the binned range
                const int b = static_cast<int>(std::distance(p_.edges.begin(), e)) - 1;
                const Eigen::Vector2d D = (now.z - m.z) - (now.O - m.O);
                const double y = D.squaredNorm() - (now.V - m.V);
                auto &bn = bins_[b];
                bn.w = forget * bn.w + 1.0;
                bn.y = forget * bn.y + y;
                bn.L = forget * bn.L + r;
                ++pairs_;
                // the same pair, kept with its two PLACES for the local amplitude (outlives the chain)
                pairs_store_.push_back({y, r, now.z, m.z, pairs_});
            }
            while (pairs_store_.size() > p_.pair_store) pairs_store_.pop_front();
            buf_.push_back(now);
            while (buf_.size() > p_.buffer) buf_.pop_front();
            fit();
            fit_local();
        }

        [[nodiscard]] double sigma_b() const noexcept { return sigma_b_; }               ///< GLOBAL amplitude (log, outlier scale)
        [[nodiscard]] double sigma_b_local() const noexcept { return sigma_b_local_; }   ///< amplitude at the current place: what is published
        [[nodiscard]] double pi_elsewhere() const noexcept { return pi_out_; }           ///< share of kept pairs the local amplitude does not explain
        [[nodiscard]] double length()  const noexcept { return len_; }
        [[nodiscard]] long   pairs()   const noexcept { return pairs_; }

        /// The covariance this bias adds to a pose fixed by a scan of claimed covariance R: β·R, with the
        /// amplitude of the field AT THE CURRENT PLACE.
        [[nodiscard]] Eigen::Matrix3f covariance(const Eigen::Matrix3f &R) const
        {
            const double rxy = 0.5 * (static_cast<double>(R(0, 0)) + static_cast<double>(R(1, 1)));
            if (not (rxy > 0.0) or not R.allFinite()) return Eigen::Matrix3f::Zero();
            return static_cast<float>(sigma_b_local_ * sigma_b_local_ / rxy) * R;
        }

    private:
        struct Sample { Eigen::Vector2d z, O; double V, S; };
        struct Bin { double w = 0.0, y = 0.0, L = 0.0; };
        struct Pair { double y, r; Eigen::Vector2d a, b; long id; };

        void reset_chain() { buf_.clear(); O_.setZero(); V_ = 0.0; S_ = 0.0; }

        void fit()
        {
            const double A0 = 4.0 * p_.sigma0 * p_.sigma0;
            double best_sse = std::numeric_limits<double>::infinity();
            for (const double l : p_.len_grid)
            {
                // A = 4 σ_b²: weighted LS on the per-lag means, plus the prior as a saturated pseudo-bin
                double num = p_.prior_pairs * A0, den = p_.prior_pairs;
                for (const auto &bn : bins_)
                    if (bn.w > 0.0) { const double g = 1.0 - std::exp(-(bn.L / bn.w) / l); num += bn.w * g * (bn.y / bn.w); den += bn.w * g * g; }
                const double A = num / den;
                double sse = p_.prior_pairs * (A0 - A) * (A0 - A);
                for (const auto &bn : bins_)
                    if (bn.w > 0.0) { const double g = 1.0 - std::exp(-(bn.L / bn.w) / l); const double r = bn.y / bn.w - A * g; sse += bn.w * r * r; }
                if (sse < best_sse) { best_sse = sse; len_ = l; sigma_b_ = std::sqrt(std::max(A, 0.0) / 4.0); }
            }
        }

        /// The amplitude at xc_: the same structure-function regression, over the kept pairs, with each pair
        /// weighted by (i) forgetting, (ii) how near BOTH its places are to xc_ (exp(−d/ℓ), ℓ the fitted
        /// correlation length — a pair whose ends are a correlation length away says little about b here), and
        /// (iii) its responsibility under a two-component likelihood. Under the model y/m is ½χ²₂, i.e. y is
        /// exponential with mean m = A·g + e; the other component has the GLOBAL plateau added to its mean
        /// (4(σ_b² + σ0²)·g: the field as it is elsewhere, never narrower than the prior).
        /// ★ EM from THREE starts (last fit, the global plateau, the prior), the one with the higher weighted
        ///   log-likelihood wins — model selection like the ℓ grid, not a gate. One start is not enough: the
        ///   exponential likelihood lets a fit SHRINK into small-y pairs from anywhere, but a small amplitude can
        ///   never grow into big-y pairs (exp(−y/m) underflows), so warm-starting alone would under-report where
        ///   the field really is large. The prior pseudo-pairs stay in the first component and one pseudo-pair in
        ///   the second, so neither mixing weight can reach 0 or 1 and switch the other component off.
        void fit_local()
        {
            const double A0 = 4.0 * p_.sigma0 * p_.sigma0;
            if (pairs_store_.empty() or not (len_ > 0.0)) { A_loc_ = A0; sigma_b_local_ = p_.sigma0; pi_out_ = 0.5; return; }
            // compute cache only: the field cannot change over 5% of its own correlation length, and a parked
            // robot would otherwise refit the same pairs every cycle
            if (fit_valid_ and (xc_ - xc_fit_).norm() < 0.05 * len_ and pairs_ == pairs_fit_) return;
            const double forget = std::exp(-1.0 / p_.memory_pairs);
            const double A_out = 4.0 * (sigma_b_ * sigma_b_ + p_.sigma0 * p_.sigma0);
            // per-pair constants of this fit: g(r), the forgetting x place kernel, y and y⁺ (y < 0 = the odometry
            // variance over-stated; it still pulls A down in the regression, as in the global fit)
            struct Term { double g, k, y, yp; };
            std::vector<Term> terms;
            terms.reserve(pairs_store_.size());
            for (const auto &q : pairs_store_)
                terms.push_back({1.0 - std::exp(-q.r / len_),
                                 std::pow(forget, static_cast<double>(pairs_ - q.id))
                                     * std::exp(-((q.a - xc_).norm() + (q.b - xc_).norm()) / len_),
                                 q.y, std::max(q.y, 0.0)});
            const auto run = [&](double A, double pi, double &ll_out)
            {
                double ll = 0.0;
                for (int it = 0; it < std::max(1, p_.local_em_iters); ++it)
                {
                    double num = p_.prior_pairs * A0, den = p_.prior_pairs, sum_out = 1.0, sum_w = p_.prior_pairs + 1.0;
                    ll = 0.0;
                    for (const auto &q : terms)
                    {
                        const double m_in = A * q.g + p_.scan_noise_var;
                        const double m_out = m_in + A_out * q.g;
                        const double li = (1.0 - pi) / m_in * std::exp(-q.yp / m_in);
                        const double lo = pi / m_out * std::exp(-q.yp / m_out);
                        const double resp = (li + lo > 0.0) ? li / (li + lo) : 1.0;   // P(explained by the amplitude here)
                        const double w = q.k * resp;
                        num += w * q.g * q.y;
                        den += w * q.g * q.g;
                        sum_out += q.k * (1.0 - resp);
                        sum_w += q.k;
                        ll += q.k * std::log(std::max(li + lo, 1e-300));
                    }
                    A = std::max(num / den, 0.0);
                    pi = sum_out / sum_w;
                }
                ll -= p_.prior_pairs * (A - A0) * (A - A0) / (A0 * A0);        // the prior pseudo-pairs, as in fit()
                ll_out = ll;
                return std::pair{A, pi};
            };
            double best_ll = -std::numeric_limits<double>::infinity();
            std::pair<double, double> best{A0, 0.5};
            for (const double A_start : {std::max(A_loc_, 0.0), A_out, A0})
            {
                double ll;
                const auto r = run(A_start, pi_out_, ll);
                if (ll > best_ll) { best_ll = ll; best = r; }
            }
            A_loc_ = best.first;
            pi_out_ = best.second;
            sigma_b_local_ = std::sqrt(A_loc_ / 4.0);
            xc_fit_ = xc_; pairs_fit_ = pairs_; fit_valid_ = true;
        }

        Params p_{};
        std::deque<Sample> buf_;
        std::deque<Pair> pairs_store_;
        Eigen::Vector2d O_ = Eigen::Vector2d::Zero(), xc_ = Eigen::Vector2d::Zero(), xc_fit_ = Eigen::Vector2d::Zero();
        double V_ = 0.0, S_ = 0.0;
        std::array<Bin, NL> bins_{};
        long pairs_ = 0, pairs_fit_ = -1;
        bool fit_valid_ = false;
        double sigma_b_ = 0.02, len_ = 1.0;
        double A_loc_ = 4.0 * 0.02 * 0.02, sigma_b_local_ = 0.02, pi_out_ = 0.5;
    };
}
