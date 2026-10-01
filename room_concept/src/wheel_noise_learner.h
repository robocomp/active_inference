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
 *  Cold start is handled by pseudo-observations, not by a "ready" switch: kPriorSamples rows each of a
 *  stationary segment at the stated rest density, a straight segment and a turning segment at the
 *  motion prior density. They are ordinary rows in the same likelihood, so real data outweighs them as
 *  it arrives and no threshold ever decides when learning "starts".
 *  ★ The motion terms MUST carry prior mass. Measured live 2026-10-01: with none, a 37-min stop
 *  evicted every moving row from a single time window, c1 and c2 fell to 0, and the learner declared
 *  the wheels NOISELESS for the first metres of the next drive (offline, the same log gives c1 ~1e-4,
 *  c2 ~1e-3). An unexcited term must revert to "as noisy as the model says", never to "perfect" --
 *  the same rule as the body parameters' drift prior (thesis §9.2).
 *
 *  RETENTION BY EXCITATION, NOT BY TIME
 *  ------------------------------------
 *  Rows are kept in three buffers -- rest, wheel motion dominated by translation, dominated by rotation
 *  -- each its own ring, so hours of standing still cannot evict what driving taught. The split is a
 *  storage policy, not a term of the model: every row enters the same likelihood. Translation vs
 *  rotation compares the wheels' own rim speeds, |v| against |omega|*b/2.
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
        static constexpr std::size_t kWindow       = 10000;   ///< segments PER BUFFER (~3 min of each regime at 50 Hz)
        static constexpr std::size_t kRefitEvery   = 250;     ///< segments between refits (~5 s)
        static constexpr int         kPriorSamples = 250;     ///< rest prior worth ~5 s of data
        /// The motion prior is WEAK on purpose. It asserts the model's pessimistic constant, some 30x the
        /// true moving density, and a pseudo-row counts like a data row: at 250 rows it doubled the speed
        /// term against 7000 real ones in the selftest. 25 rows (~0.5 s of driving) keep a cold start
        /// conservative and wash out within seconds of real motion.
        static constexpr int         kMotionPriorSamples = 25;
        static constexpr double      kHalfTrack    = 0.259;   ///< m; rim speed of rotation is |omega|*b/2
        static constexpr double      kRestSpeed    = 0.01;    ///< m/s rim speed: storage split only, see above
        static constexpr double      kPriorV       = 0.3;     ///< m/s, where the straight pseudo-row sits
        static constexpr double      kPriorW       = 0.5;     ///< rad/s, where the turning pseudo-row sits
        static constexpr int         kIrlsIters    = 6;

        /// The prior's rest density (rad/sqrt(s)): the producer's stated value if it states one, else
        /// the motion model's constant. Called whenever it changes; cheap.
        void set_prior_density(float d) noexcept
        {
            if (not (d > 0.f) or not std::isfinite(d)) return;
            prior_d_ = d;
            if (n_total_ == 0) c_ = prior_coeffs();   // before any data: the prior
        }

        /// One segment. e = calibrated wheel heading - calibrated gyro heading (rad), dt (s), v (m/s),
        /// omega (rad/s, the WHEELS' rate), dens_g = gyro STATED density (rad/sqrt(s)). Refused unless
        /// the gyro stated its density -- without the anchor the split is not identified.
        void add(float e, float dt, float v, float omega, float dens_g)
        {
            if (not (dt > 0.f) or not (dens_g > 0.f) or not std::isfinite(e)) return;
            Row r{static_cast<double>(e) * e / dt, std::abs(v), std::abs(omega),
                  static_cast<double>(dens_g) * dens_g};
            const double rim_v = std::abs(v), rim_w = std::abs(omega) * kHalfTrack;
            const int k = (rim_v < kRestSpeed and rim_w < kRestSpeed) ? 0 : (rim_v >= rim_w ? 1 : 2);
            auto &buf = rows_[k];
            if (buf.size() < kWindow) buf.push_back(r);
            else { buf[head_[k]] = r; head_[k] = (head_[k] + 1) % kWindow; }
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
        /// Density the motion pseudo-rows assert (rad/sqrt(s)): what the wheels are assumed to do while
        /// moving until data says otherwise. Conservative by default -- the motion model's own constant.
        void set_motion_prior_density(float d) noexcept { if (d > 0.f and std::isfinite(d)) motion_d_ = d; }

    private:
        struct Row { double y, v, w, off; };

        void refit()
        {
            // Data rows plus the prior's pseudo-rows: a stationary segment (x = [1,0,0]) whose y is
            // what the prior density predicts, with the window's mean gyro offset.
            double off_mean = 0.0; std::size_t n_rows = 0;
            for (const auto &b : rows_) for (const auto &r : b) { off_mean += r.off; ++n_rows; }
            off_mean = n_rows == 0 ? 0.0 : off_mean / static_cast<double>(n_rows);
            const double y_rest   = off_mean + static_cast<double>(prior_d_) * prior_d_;
            const double y_motion = off_mean + static_cast<double>(motion_d_) * motion_d_;

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
                for (const auto &b : rows_)
                    for (const auto &r : b) acc(r.y, r.off, Eigen::Vector3d(1.0, r.v, r.w), 1.0);
                const double np = static_cast<double>(kPriorSamples);
                const double nm = static_cast<double>(kMotionPriorSamples);
                acc(y_rest,   off_mean, Eigen::Vector3d(1.0, 0.0, 0.0), np);
                acc(y_motion, off_mean, Eigen::Vector3d(1.0, kPriorV, 0.0), nm);
                acc(y_motion, off_mean, Eigen::Vector3d(1.0, 0.0, kPriorW), nm);

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

        /// The coefficients the pseudo-rows alone imply: rest density at rest, motion density at the
        /// reference speed and turn rate.
        [[nodiscard]] std::array<double, 3> prior_coeffs() const noexcept
        {
            const double r2 = static_cast<double>(prior_d_) * prior_d_;
            const double m2 = std::max(static_cast<double>(motion_d_) * motion_d_ - r2, 0.0);
            return {r2, m2 / kPriorV, m2 / kPriorW};
        }

        std::array<std::vector<Row>, 3> rows_;      ///< rest, translation-dominated, rotation-dominated
        std::array<std::size_t, 3> head_{0, 0, 0};
        std::size_t since_fit_ = 0;
        long n_total_ = 0;
        float prior_d_  = 0.0447f;
        float motion_d_ = 0.0447f;
        std::array<double, 3> c_{0.0447 * 0.0447, 0.0, 0.0};
    };
}
