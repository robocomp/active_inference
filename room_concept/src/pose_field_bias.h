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
 *  THE PUBLISHED TERM: Σ_b = β·R, β = σ_b² / (tr R_xy / 2), R = the current scan's CLAIMED covariance. The bias
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
        };

        void set_params(const Params &p) { p_ = p; reset_chain(); bins_ = {}; }
        [[nodiscard]] const Params &params() const noexcept { return p_; }

        /// One cycle. z: scan-only pose (world x, y, θ); R: its claimed covariance (world); odom: the odometry
        /// increment (world); odo_var_tr: the odometry variance the noise model states for THIS cycle, x + y
        /// summed; chain_ok: this cycle directly follows the previous one (else the chain restarts here).
        void observe(const Eigen::Vector3f &z, const Eigen::Matrix3f &R, const Eigen::Vector3f &odom,
                     float odo_var_tr, float /*odo_var_th*/, bool chain_ok)
        {
            if (not z.allFinite() or not odom.allFinite() or not std::isfinite(odo_var_tr)) { reset_chain(); return; }
            (void)R;
            if (not chain_ok or buf_.empty())
            {
                reset_chain();
                buf_.push_back({z.head<2>().cast<double>(), Eigen::Vector2d::Zero(), 0.0, 0.0});
                return;
            }
            O_ += odom.head<2>().cast<double>();
            V_ += std::max(0.0, static_cast<double>(odo_var_tr));
            S_ += odom.head<2>().cast<double>().norm();
            if (S_ - buf_.back().S < p_.sample_spacing) return;
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
            }
            buf_.push_back(now);
            while (buf_.size() > p_.buffer) buf_.pop_front();
            fit();
        }

        [[nodiscard]] double sigma_b() const noexcept { return sigma_b_; }
        [[nodiscard]] double length()  const noexcept { return len_; }
        [[nodiscard]] long   pairs()   const noexcept { return pairs_; }

        /// The covariance this bias adds to a pose fixed by a scan of claimed covariance R: β·R.
        [[nodiscard]] Eigen::Matrix3f covariance(const Eigen::Matrix3f &R) const
        {
            const double rxy = 0.5 * (static_cast<double>(R(0, 0)) + static_cast<double>(R(1, 1)));
            if (not (rxy > 0.0) or not R.allFinite()) return Eigen::Matrix3f::Zero();
            return static_cast<float>(sigma_b_ * sigma_b_ / rxy) * R;
        }

    private:
        struct Sample { Eigen::Vector2d z, O; double V, S; };
        struct Bin { double w = 0.0, y = 0.0, L = 0.0; };

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

        Params p_{};
        std::deque<Sample> buf_;
        Eigen::Vector2d O_ = Eigen::Vector2d::Zero();
        double V_ = 0.0, S_ = 0.0;
        std::array<Bin, NL> bins_{};
        long pairs_ = 0;
        double sigma_b_ = 0.02, len_ = 1.0;
    };
}
