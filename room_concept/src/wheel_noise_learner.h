/*  wheel_noise_learner.h — learn the wheels' heading-noise density as a function of motion.
 *
 *  WHY
 *  ---
 *  The heading is the precision-weighted product of a wheel factor and a gyro factor, so the weights
 *  are only as good as the two densities. Measured on Shadow (sim, 2026-10-01, 100 s parked, truth
 *  exactly still): the wheels' STATED density was 0.0041 rad/sqrt(s) against 0.0007 measured -- 6x
 *  pessimistic -- while the gyro's stated 0.0021 matched its measured 0.0019. And the real base
 *  states no wheel variance at all. A single wheel number cannot be right anyway: replaying both
 *  logs, a wheel density good at rest HURT slow driving, because wheel heading error grows with motion
 *  (slip, scrubbing).
 *
 *  THE MODEL
 *  ---------
 *  The wheel-minus-gyro heading difference over a segment cancels the TRUE rotation exactly:
 *      e = dpsi_wheel - dpsi_gyro,   E[e^2 / dt] = sigma_g^2 + d_w^2(v, omega)
 *  with sigma_g^2 the gyro's STATED density (the anchor -- e alone identifies only the sum) and
 *      d_w^2(v, omega) = c0 + c1 |v| + c2 |omega|,   c >= 0.
 *  No ground truth, no map, no localiser: it runs identically on the real robot, provided the gyro
 *  states its variance (no anchor => no sample; the fusion then keeps its stated/model densities).
 *
 *  THE FIT
 *  -------
 *  y = e^2/dt is a scaled chi-square(1): Var(y) = 2 mu^2. Maximum likelihood for that is
 *  iteratively-reweighted least squares with weights 1/mu^2 (the Gamma-GLM identity-link fit). The
 *  constraint c >= 0 is solved EXACTLY by enumerating the 7 non-empty active sets -- three unknowns, so
 *  it is cheaper than any iterative NNLS and has no tolerance to tune.
 *
 *  THE PRIOR, NOT A GATE
 *  ---------------------
 *  Cold start is handled by pseudo-observations, not by a "ready" switch: kPriorSamples rows of a
 *  stationary segment whose wheel density is the stated/model one. They are ordinary rows in the same
 *  likelihood, so real data outweighs them as it arrives (after ~kPriorSamples segments, ~5 s) and no
 *  threshold ever decides when learning "starts". c1 and c2 have no prior mass: an unexcited motion
 *  term stays at zero, which is safe because the fusion ALSO charges the wheels their scale uncertainty
 *  (15.5% of |rotation| until calibrated), so a first fast turn still leans on the gyro.
 *
 *  ⚠ The model is WHITE. Parked, the wheels' 10-s heading spread was 1.75x what white noise predicts
 *  (some low-frequency wheel error), so long-horizon wheel confidence is still somewhat optimistic.
 */
#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace rc::calib
{
    class WheelNoiseLearner
    {
    public:
        static constexpr std::size_t kWindow       = 30000;   ///< segments (~10 min at 50 Hz)
        static constexpr std::size_t kRefitEvery   = 250;     ///< segments between refits (~5 s)
        static constexpr int         kPriorSamples = 250;     ///< prior worth ~5 s of data
        static constexpr int         kIrlsIters    = 6;

        /// The prior's rest density (rad/sqrt(s)): the producer's stated value if it states one, else
        /// the motion model's constant. Called whenever it changes; cheap.
        void set_prior_density(float d) noexcept
        {
            if (not (d > 0.f) or not std::isfinite(d)) return;
            prior_d_ = d;
            if (n_total_ == 0) c_ = {static_cast<double>(d) * d, 0.0, 0.0};   // before any data: the prior
        }

        /// One segment. e = calibrated wheel heading - calibrated gyro heading (rad), dt (s), v (m/s),
        /// omega (rad/s, the WHEELS' rate), dens_g = gyro STATED density (rad/sqrt(s)). Refused unless
        /// the gyro stated its density -- without the anchor the split is not identified.
        void add(float e, float dt, float v, float omega, float dens_g)
        {
            if (not (dt > 0.f) or not (dens_g > 0.f) or not std::isfinite(e)) return;
            Row r{static_cast<double>(e) * e / dt, std::abs(v), std::abs(omega),
                  static_cast<double>(dens_g) * dens_g};
            if (rows_.size() < kWindow) rows_.push_back(r);
            else { rows_[head_] = r; head_ = (head_ + 1) % kWindow; }
            ++n_total_;
            if (++since_fit_ >= kRefitEvery) { since_fit_ = 0; refit(); }
        }

        /// Wheel heading-noise density at this motion (rad/sqrt(s)). Prior-only until data arrives.
        [[nodiscard]] float density(float v, float omega) const noexcept
        {
            const double d2 = c_[0] + c_[1] * std::abs(v) + c_[2] * std::abs(omega);
            return static_cast<float>(std::sqrt(std::max(d2, 0.0)));
        }
        [[nodiscard]] const std::array<double, 3>& coeffs() const noexcept { return c_; }
        [[nodiscard]] long samples() const noexcept { return n_total_; }

    private:
        struct Row { double y, v, w, off; };

        void refit()
        {
            // Data rows plus the prior's pseudo-rows: a stationary segment (x = [1,0,0]) whose y is
            // what the prior density predicts, with the window's mean gyro offset.
            double off_mean = 0.0;
            for (const auto &r : rows_) off_mean += r.off;
            off_mean = rows_.empty() ? 0.0 : off_mean / static_cast<double>(rows_.size());
            const double y_prior = off_mean + static_cast<double>(prior_d_) * prior_d_;

            std::array<double, 3> c = c_;
            for (int it = 0; it < kIrlsIters; ++it)
            {
                Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
                Eigen::Vector3d b = Eigen::Vector3d::Zero();
                double yy = 0.0;
                const auto acc = [&](double y, double off, const Eigen::Vector3d &x, double mult)
                {
                    const double mu = std::max(off + c[0] * x[0] + c[1] * x[1] + c[2] * x[2], 1e-14);
                    const double w = mult / (mu * mu);
                    const double t = y - off;
                    H += w * x * x.transpose();
                    b += w * t * x;
                    yy += w * t * t;
                };
                for (const auto &r : rows_) acc(r.y, r.off, Eigen::Vector3d(1.0, r.v, r.w), 1.0);
                acc(y_prior, off_mean, Eigen::Vector3d(1.0, 0.0, 0.0), static_cast<double>(kPriorSamples));

                // Exact c >= 0: best feasible solution over the 7 non-empty active sets.
                double best = std::numeric_limits<double>::infinity();
                std::array<double, 3> best_c{0.0, 0.0, 0.0};
                for (int mask = 1; mask < 8; ++mask)
                {
                    std::vector<int> idx;
                    for (int i = 0; i < 3; ++i) if (mask >> i & 1) idx.push_back(i);
                    const int k = static_cast<int>(idx.size());
                    Eigen::MatrixXd Hs(k, k); Eigen::VectorXd bs(k);
                    for (int i = 0; i < k; ++i)
                    {
                        bs[i] = b[idx[i]];
                        for (int j = 0; j < k; ++j) Hs(i, j) = H(idx[i], idx[j]);
                    }
                    const Eigen::LDLT<Eigen::MatrixXd> ldlt(Hs);
                    if (ldlt.info() != Eigen::Success) continue;
                    const Eigen::VectorXd s = ldlt.solve(bs);
                    if (not s.allFinite() or (s.array() < 0.0).any()) continue;
                    Eigen::Vector3d full = Eigen::Vector3d::Zero();
                    for (int i = 0; i < k; ++i) full[idx[i]] = s[i];
                    const double cost = yy - 2.0 * b.dot(full) + full.dot(H * full);
                    if (cost < best) { best = cost; best_c = {full[0], full[1], full[2]}; }
                }
                if (not std::isfinite(best)) return;   // nothing feasible: keep the previous fit
                c = best_c;
            }
            c_ = c;
        }

        std::vector<Row> rows_;
        std::size_t head_ = 0, since_fit_ = 0;
        long n_total_ = 0;
        float prior_d_ = 0.0447f;
        std::array<double, 3> c_{0.0447 * 0.0447, 0.0, 0.0};
    };
}
