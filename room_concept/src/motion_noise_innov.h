/*  motion_noise_innov.h — learn the odometry noise components from the SCAN-TO-SCAN innovation.
 *
 *  WHY (2026-10-05, Fable window memo, room_concept/docs/fable_window_memo_2026-10-05.md)
 *  -------------------------------------------------------------------------------------
 *  motion_noise_vc.h learnt Q from the CORRECTION c = K·ν with K = I − P_post·P_pred⁻¹. That assumes a
 *  single Kalman update. The localiser is not one: only the newest window slot carries a scan, the past
 *  enters through a boundary prior capped at 500·exp(u_b), so each solve is essentially a fresh scan-only
 *  fix (gain ≈ 1) while the K inferred from the two covariances read ≈ 0.003 on position. The posterior
 *  wobbles ±5 % solve to solve, more than one cycle of odometry noise, so P_post > P_pred on 37 % of
 *  forward solves and the learner collapsed (k_th_turn → 0 in 3 corrections, live 12-07-20).
 *
 *  THE OBSERVATION, every cycle, whether the solver ran or not
 *  ----------------------------------------------------------
 *  z_n is the scan-only pose: the Gauss–Newton step of the newest scan's SDF factor alone, taken from
 *  where the scan was linearised (z = x_lin − H_s⁻¹ b_s). With odom_n the reported increment,
 *        δ_n = z_n − z_{n−1} − odom_n = η_n + ε·odom_n + e_n − e_{n−1} + [b(x_n) − b(x_{n−1})]
 *  η = white odometry noise (the k_j), ε = a constant SCALE error, e = the scan's white error, b = the slow
 *  bias FIELD over position (cancels to < 1 mm over a cycle). Per body axis a (unit vector u_a, world):
 *        E[δ_a²] = Σ_j k_j·u_aᵀU_j u_a  +  s_a·u_aᵀ(R_n + R_{n−1})u_a  +  q_a·(odom_a)²
 *  U_j: this cycle's unit noise components (se2_preintegration.h). R: the scan's CLAIMED covariance
 *  H_s⁻¹, so s_a is the white-to-claimed ratio (measured ~0.01: claims are 10x the sigma). q_a = ε_a²
 *  separates a scale error (grows with the square of the step) from a random walk (with the step), which
 *  otherwise drags k_th_turn to ~1/3 of its long-window value. No gain, no window, no P_post.
 *  This form does not need x̂_{n−1} ≈ z_{n−1} (Fable's (★)), so it also holds on early-exit cycles where
 *  the estimate is dead reckoning.
 *
 *  WHILE MOVING (measured live 2026-10-05 21-09 / 22-20): after the tails, driving cycles still carry ~2x
 *  the model's variance with NO trend in speed — a STEP between rest and motion (vibration, slip, scan
 *  deskew; source not separated). Without a term for it the excess is charged per METRE (k_long 3.7x).
 *  Term: v_a · P(moving) · dt per axis, P(moving) from the rest channel (RestMotionChannel's Savage-Dickey
 *  probability, room_concept zupt_pred_gain_*) — the agent's own belief, not a speed threshold.
 *  ⚠ The coefficient reads 2v if the step is in the SCAN (two scans per innovation) and v if it is in the
 *  ODOMETRY. Nothing here separates them; only the per-metre/per-radian k_j are protected by it.
 *
 *  HEAVY TAILS (measured live 2026-10-05 21-09, 15.6k cycles): 3-5 % of moving cycles exceed 3 sigma (normal:
 *  0.27 %) and the top 2 % carry 20-27 % of Σδ², with NO trend in speed or turn rate — occasional bad scan-only
 *  fixes, not a motion-dependent variance. Squared-residual regression is not robust to them: they were charged
 *  to the odometry coefficients (k_long 5x, k_th_turn 4x). So each axis is a two-component MIXTURE:
 *        δ_a ~ (1 − π_a)·N(0, m_a) + π_a·N(0, κ_a·m_a),   κ_a ≥ 1
 *  learnt by online EM: each observation's outlier responsibility r = π N(δ;0,κm) / [(1−π) N(δ;0,m) + π N(δ;0,κm)]
 *  removes it from the coefficient regression in proportion; π and κ are re-estimated from the same
 *  responsibilities (Beta / inverse-gamma-like pseudo-counts as priors). On clean Gaussian data π → ~0 and the
 *  estimate is the old one, unbiased (selftest part 1); with outliers it recovers the inlier coefficients (part 4).
 *  κ ≥ 1 is what NAMES the outlier component, not a cutoff.
 *
 *  THE ESTIMATOR — same as motion_noise_vc.h: IRLS/REML weights 1/(2m²) (variance of a squared Gaussian),
 *  exponential forgetting, one Gaussian pseudo-observation per parameter, projection onto θ ≥ 0 (all are
 *  variances or variance ratios: a physical constraint, not a gate). Consecutive δ share e_n (MA(1),
 *  correlation ≈ −½ when the scan term dominates): every cycle is fed at weight ma1_weight = 0.5.
 *  Selftest: tools/motion_noise_innov_selftest.cpp.
 */
#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace rc::preint
{
    class MotionNoiseInnov
    {
    public:
        static constexpr int NK = 6, NS = 3, NQ = 3, NM = 3, NP = NK + NS + NQ + NM;
        using Vec = Eigen::Matrix<double, NP, 1>;
        using Mat = Eigen::Matrix<double, NP, NP>;

        struct Params
        {
            double memory = 20000.0;                       ///< cycles of evidence remembered
            std::array<double, NK> k0{};                   ///< prior means (the configured coefficients)
            std::array<double, NK> k_sd{};                 ///< prior sd; <= 0 => 100 % of k0
            std::array<double, NS> s0{1.0, 1.0, 1.0};      ///< prior: the scan's claim is honest
            std::array<double, NS> s_sd{1.0, 1.0, 1.0};
            std::array<double, NQ> q_sd{4e-4, 4e-4, 0.024};   ///< (2 %)², (2 %)², (15.5 %)² — scale priors, mean 0
            std::array<double, NM> mv_sd{1e-4, 1e-4, 1e-4};   ///< m²/s, m²/s, rad²/s while moving — prior mean 0
            double ma1_weight = 0.5;                       ///< consecutive innovations share one scan error
            double pi0 = 0.01, pi_pseudo = 200.0;          ///< outlier share prior: 1 %, worth 200 observations
            double kappa0 = 20.0, kappa_pseudo = 5.0;      ///< outlier variance ratio prior, worth 5 outliers
            bool   mixture = true;                         ///< false = the plain Gaussian (selftest comparison)
        };

        void set_params(const Params &p)
        {
            p_ = p;
            N_.setZero(); b_.setZero(); n_ = 0;
            for (int a = 0; a < 3; ++a) { S_n_[a] = S_r_[a] = S_ry_[a] = 0.0; pi_[a] = p.pi0; kappa_[a] = p.kappa0; }
            for (int j = 0; j < NK; ++j) theta_(j) = p.k0[j];
            for (int a = 0; a < NS; ++a) theta_(NK + a) = p.s0[a];
            for (int a = 0; a < NQ + NM; ++a) theta_(NK + NS + a) = 0.0;
        }

        /// One cycle. delta = z_n − z_{n−1} − odom_n (world x, y, θ; θ wrapped). T: columns = body axes
        /// in world coords (forward, lateral, heading). U: this cycle's unit components (world).
        /// Rsum = R_n + R_{n−1}, the two scans' CLAIMED covariances (world). odom_body = the reported
        /// increment on (forward, lateral, heading). p_move: the belief that the body moved this cycle
        /// (rest channel); dt: the cycle's duration, s.
        void observe(const Eigen::Vector3f &delta, const Eigen::Matrix3f &T, const std::array<Eigen::Matrix3f, NK> &U,
                     const Eigen::Matrix3f &Rsum, const Eigen::Vector3f &odom_body, float p_move, float dt)
        {
            if (not delta.allFinite() or not T.allFinite() or not Rsum.allFinite() or not odom_body.allFinite()
                or not std::isfinite(p_move) or not std::isfinite(dt)) return;
            const Eigen::Matrix3d Td = T.cast<double>(), Rd = Rsum.cast<double>();
            const Eigen::Vector3d dd = delta.cast<double>(), od = odom_body.cast<double>();
            std::array<Vec, 3> xs;  std::array<double, 3> ys{};
            for (int a = 0; a < 3; ++a)
            {
                const Eigen::Vector3d u = Td.col(a);
                Vec x = Vec::Zero();
                for (int j = 0; j < NK; ++j) x(j) = u.dot(U[j].cast<double>() * u);
                x(NK + a)      = u.dot(Rd * u);
                x(NK + NS + a) = od(a) * od(a);
                x(NK + NS + NQ + a) = static_cast<double>(p_move) * static_cast<double>(dt);
                xs[a] = x;  ys[a] = u.dot(dd) * u.dot(dd);
            }
            observe_rows(xs, ys);
        }

        /// The same, from precomputed regression rows (x_a, y_a = δ_a²): what the replay tool feeds from
        /// the innov CSV, so offline re-fits run THIS code, not a re-implementation.
        void observe_rows(const std::array<Vec, 3> &xs, const std::array<double, 3> &ys)
        {
            const double forget = std::exp(-1.0 / p_.memory);
            for (int a = 0; a < 3; ++a)
            {
                const Vec &x = xs[a];
                const double y = ys[a];
                const double m = x.dot(theta_);
                last_x_[a] = x;  last_y_[a] = y;  last_m_[a] = m;  last_r_[a] = 0.0;
                if (not (m > 0.0) or not x.allFinite() or not std::isfinite(y)) continue;
                double r = 0.0;                                   // outlier responsibility
                if (p_.mixture)
                {
                    const double z = y / m, k = kappa_[a];
                    // ratio of the two zero-mean normal densities at δ² = y, in log form (no under/overflow)
                    const double l_out = std::log(pi_[a])        - 0.5 * std::log(k) - 0.5 * z / k;
                    const double l_in  = std::log(1.0 - pi_[a])                     - 0.5 * z;
                    r = 1.0 / (1.0 + std::exp(l_in - l_out));
                    S_n_[a]  = forget * S_n_[a]  + 1.0;
                    S_r_[a]  = forget * S_r_[a]  + r;
                    S_ry_[a] = forget * S_ry_[a] + r * z;
                    pi_[a]    = (S_r_[a] + p_.pi0 * p_.pi_pseudo) / (S_n_[a] + p_.pi_pseudo);
                    kappa_[a] = std::max(1.0, (S_ry_[a] + p_.kappa0 * p_.kappa_pseudo) / (S_r_[a] + p_.kappa_pseudo));
                }
                last_r_[a] = r;
                const double w = (1.0 - r) * p_.ma1_weight / (2.0 * m * m);
                N_ += w * x * x.transpose();
                b_ += w * y * x;
            }
            solve();
        }

        [[nodiscard]] double k(int j) const noexcept { return theta_(j); }
        [[nodiscard]] double s(int a) const noexcept { return theta_(NK + a); }
        [[nodiscard]] double q(int a) const noexcept { return theta_(NK + NS + a); }
        [[nodiscard]] double mv(int a) const noexcept { return theta_(NK + NS + NQ + a); }   ///< variance/s while moving
        /// The ODOMETRY part of the last cycle's model variance on axis a (everything but the scan term): what
        /// the pose-field bias estimator subtracts over a stretch (pose_field_bias.h).
        [[nodiscard]] double odom_var(int a) const noexcept
        { return last_x_[a].dot(theta_) - last_x_[a](NK + a) * theta_(NK + a); }
        /// The part that ACCUMULATES along a path: the per-metre / per-radian / per-second k_j and the scale term,
        /// WITHOUT the while-moving step, whose source (scan vs odometry) the innovation cannot tell — a scan-side
        /// step telescopes over a chain instead of summing (pose_field_bias.h).
        [[nodiscard]] double odom_var_accum(int a) const noexcept
        { return odom_var(a) - last_x_[a](NK + NS + NQ + a) * theta_(NK + NS + NQ + a); }
        [[nodiscard]] long   cycles() const noexcept { return n_; }
        /// The last cycle's regression row per axis, for the replay log.
        [[nodiscard]] const Vec &last_x(int a) const noexcept { return last_x_[a]; }
        [[nodiscard]] double last_y(int a) const noexcept { return last_y_[a]; }
        [[nodiscard]] double last_m(int a) const noexcept { return last_m_[a]; }
        [[nodiscard]] double last_r(int a) const noexcept { return last_r_[a]; }   ///< outlier responsibility
        [[nodiscard]] double pi(int a) const noexcept { return pi_[a]; }          ///< outlier share
        [[nodiscard]] double kappa(int a) const noexcept { return kappa_[a]; }    ///< outlier variance ratio

    private:
        void solve()
        {
            Mat A = N_;  Vec r = b_;
            for (int j = 0; j < NP; ++j)
            {
                double mean = 0.0, sd = 1.0;
                if (j < NK)            { mean = p_.k0[j]; sd = p_.k_sd[j] > 0.0 ? p_.k_sd[j] : p_.k0[j]; }
                else if (j < NK + NS)  { mean = p_.s0[j - NK]; sd = p_.s_sd[j - NK]; }
                else if (j < NK + NS + NQ) { mean = 0.0; sd = p_.q_sd[j - NK - NS]; }
                else                   { mean = 0.0; sd = p_.mv_sd[j - NK - NS - NQ]; }
                if (not (sd > 0.0)) sd = 1e-9;
                A(j, j) += 1.0 / (sd * sd);
                r(j)    += mean / (sd * sd);
            }
            const Vec t = A.ldlt().solve(r);
            if (t.allFinite()) theta_ = t.cwiseMax(0.0);
            const double d = std::exp(-1.0 / p_.memory);
            N_ *= d;  b_ *= d;
            ++n_;
        }

        Params p_{};
        Mat N_ = Mat::Zero();
        Vec b_ = Vec::Zero();
        Vec theta_ = Vec::Zero();
        long n_ = 0;
        std::array<Vec, 3> last_x_{Vec::Zero(), Vec::Zero(), Vec::Zero()};
        std::array<double, 3> last_y_{}, last_m_{}, last_r_{};
        std::array<double, 3> S_n_{}, S_r_{}, S_ry_{}, pi_{0.01, 0.01, 0.01}, kappa_{20.0, 20.0, 20.0};
    };
}
