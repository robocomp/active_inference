/*  motion_noise_vc.h — learn the odometry noise COMPONENTS from the localiser's own corrections.
 *
 *  WHY (2026-10-05)
 *  ----------------
 *  It replaces a single variance scale per body axis (kappa), which could not fit the data: GT showed
 *  the error grows per METRE and per RADIAN and not at all per second, so one kappa had to be ~1e-3
 *  parked, ~0.03 driving and ~1.5 at stride artefacts and wandered between them. Here the motion
 *  noise is a sum of components, each with its own coefficient (se2_preintegration.h, NoiseModel):
 *        Q = k_long·U_long + k_lat·U_lat + k_lat_turn·U_lat_turn + k_th_turn·U_th_turn
 *          + k_t_trans·U_t_trans + k_t_rot·U_t_rot  (+ the producers' stated densities, fixed)
 *  where U_j is that component integrated with coefficient 1 through the same transport as the
 *  covariance (exact: the recursion is linear in Q). Parked stretches carry U_dist = U_turn = 0, so they
 *  inform the per-second terms only; driving stretches inform the per-metre ones; pivots the per-radian.
 *  No stretch-length gate, no regime switch — the regressors separate them.
 *
 *  THE OBSERVATION MODEL, per scored correction, per body axis a (unit vector u_a)
 *  -----------------------------------------------------------------------------
 *  With the gain K = I − P_post·P_pred⁻¹ the correction is c = K·ν, ν = (prediction error + scan
 *  error), so E[c cᵀ] = K (P_prev + Q_true + R_true) Kᵀ. The solver's own claim is that this equals
 *  P_pred − P_post with Q = Q_applied. Writing Q_true = Q_fixed + Σ k_j U_j and R_true = R_claimed + R_ex,
 *        E[c_a²] = o_a + Σ_j k_j · u_aᵀ K U_j Kᵀ u_a + Σ_b ρ_b · u_aᵀ K t_b t_bᵀ Kᵀ u_a,
 *        o_a     = u_aᵀ [ (P_pred − P_post) − K Q_k Kᵀ ] u_a,      Q_k = Σ k̂_j U_j as APPLIED,
 *  with t_b the body axes. ρ_b is the scan side's EXCESS variance per body axis — how overconfident
 *  the posterior is — the modelled replacement for apply_adaptive_covariance's floor. Linear in
 *  (k, ρ): one weighted least-squares problem. When the model is honest (k = k̂, ρ = 0) it reduces
 *  exactly to E[c²] = P_pred − P_post, the identity the old learner used.
 *
 *  THE ESTIMATOR
 *  -------------
 *  IRLS / REML weights w_a = 1 / (2 m_a²), m_a = the model's E[c_a²] at the current estimate (the
 *  variance of a squared Gaussian), exponential forgetting of the normal equations, one Gaussian
 *  pseudo-observation per parameter at its configured value. ★ The three axes of one correction are
 *  correlated through K; they are treated as independent (slight over-weighting of each correction).
 *  ★ Variances cannot be negative: k_j is projected onto [0, ∞) after the solve (a physical
 *  constraint, not a tuning gate). ρ may be negative (an UNDER-confident posterior).
 *  ⛔ OFF LIVE (MotionNoiseLearn = false, 2026-10-05). Correct on consistent posteriors (selftest), but
 *  the live window posterior can sit ABOVE the prediction on an axis once the motion noise is tight;
 *  projecting Λ ⪰ 0 keeps K valid, yet the observed c on such an axis is then far outside the model's
 *  variance and a few corrections drag the coefficients off (live 12-07-20: k_th_turn -> 0 after 3
 *  corrections). A Student-t weight bounds that but biases the clean case ~30 % low. Needs the solver's
 *  own scan information (H_sdf) per correction instead of inferring it from P_post. selftest part 2b.
 *  ⚠ KNOWN, MEASURED NOT MODELLED: P_post is a window marginal, not a single Kalman update, so K is an
 *  approximation; and the early-exit gate selects 1-cycle stretches for scan trouble (biases ρ up).
 */
#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace rc::preint
{
    class MotionNoiseVC
    {
    public:
        static constexpr int NK = 6, NR = 3, NP = NK + NR;
        using Vec = Eigen::Matrix<double, NP, 1>;
        using Mat = Eigen::Matrix<double, NP, NP>;

        struct Params
        {
            double memory = 2000.0;                 ///< corrections of evidence remembered
            std::array<double, NK> k0{};            ///< prior means (the configured coefficients)
            std::array<double, NK> k_sd{};          ///< prior sd; <= 0 => 100 % of k0
            std::array<double, NR> rho_sd{1e-4, 1e-4, 1e-4};   ///< m², m², rad² — scan excess prior sd
        };

        void set_params(const Params &p)
        {
            p_ = p;
            for (int j = 0; j < NK; ++j) theta_(j) = p.k0[j];
            for (int b = 0; b < NR; ++b) theta_(NK + b) = 0.0;
        }

        /// One axis of one correction: x = the NP regressors, y = c_a² − o_a, m = model E[c_a²].
        void observe_axis(const Vec &x, double y, double m)
        {
            if (not x.allFinite() or not std::isfinite(y) or not (m > 0.0)) return;
            const double w = 1.0 / (2.0 * m * m);
            N_ += w * x * x.transpose();
            b_ += w * y * x;
        }
        /// Close one correction: forget, then solve with the prior.
        void end_correction()
        {
            const double d = std::exp(-1.0 / p_.memory);
            Mat A = N_;  Vec r = b_;
            for (int j = 0; j < NP; ++j)
            {
                const double mean = j < NK ? p_.k0[j] : 0.0;
                double sd = j < NK ? (p_.k_sd[j] > 0.0 ? p_.k_sd[j] : p_.k0[j]) : p_.rho_sd[j - NK];
                if (not (sd > 0.0)) sd = 1e-9;
                A(j, j) += 1.0 / (sd * sd);
                r(j)    += mean / (sd * sd);
            }
            const Vec t = A.ldlt().solve(r);
            if (t.allFinite())
            {
                theta_ = t;
                for (int j = 0; j < NK; ++j) theta_(j) = std::max(theta_(j), 0.0);
            }
            N_ *= d;  b_ *= d;
            ++n_;
        }

        [[nodiscard]] double k(int j)   const noexcept { return theta_(j); }
        [[nodiscard]] double rho(int b) const noexcept { return theta_(NK + b); }
        [[nodiscard]] const Vec& theta() const noexcept { return theta_; }
        [[nodiscard]] long corrections() const noexcept { return n_; }

        /// Build the three axis observations of one correction and feed them (then end_correction()).
        /// c: correction (world x, y, θ). T: columns = body axes in world coords (fwd, lat, heading).
        /// P_pred, P_post: the solver's prediction / posterior. Qk: Σ k̂_j U_j as APPLIED over the
        /// stretch. U: the unit components summed over the stretch.
        void observe(const Eigen::Vector3f &c, const Eigen::Matrix3f &T, const Eigen::Matrix3f &P_pred,
                     const Eigen::Matrix3f &P_post, const Eigen::Matrix3f &Qk,
                     const std::array<Eigen::Matrix3f, NK> &U)
        {
            Eigen::Matrix3d Pp = P_pred.cast<double>(), Pq_in = P_post.cast<double>();
            Pp = 0.5 * (Pp + Pp.transpose());  Pq_in = 0.5 * (Pq_in + Pq_in.transpose());
            // ★ The scan's information Λ = P_post⁻¹ − P_pred⁻¹ cannot be negative. The solver's window
            //   posterior is not a single Kalman update and, with tight motion noise, can sit ABOVE the
            //   prediction on some axis; taken literally that is negative information, K leaves [0, 1] and
            //   the model variance goes ≤ 0 (which blew the learner up live, 2026-10-05). Project Λ onto
            //   Λ ⪰ 0 and derive the posterior and the gain from it, so they stay mutually consistent:
            //   an axis the scan did not inform gets K = 0 and carries no information.
            const Eigen::Matrix3d Lp = Pp.inverse();
            Eigen::Matrix3d Lam = Pq_in.inverse() - Lp;
            Lam = 0.5 * (Lam + Lam.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Lam);
            if (es.info() != Eigen::Success) return;
            Lam = es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).asDiagonal() * es.eigenvectors().transpose();
            const Eigen::Matrix3d Pq = (Lp + Lam).inverse();
            const Eigen::Matrix3d K  = Pq * Lam;          // gain in information form (H = I)
            if (not K.allFinite() or not Pq.allFinite()) return;
            const Eigen::Matrix3d Td = T.cast<double>();
            const Eigen::Matrix3d O = (Pp - Pq) - K * Qk.cast<double>() * K.transpose();
            std::array<Eigen::Matrix3d, NP> R;
            for (int j = 0; j < NK; ++j) R[j] = K * U[j].cast<double>() * K.transpose();
            for (int b = 0; b < NR; ++b)
            {
                const Eigen::Vector3d tb = Td.col(b);
                R[NK + b] = K * (tb * tb.transpose()) * K.transpose();
            }
            for (int a = 0; a < 3; ++a)
            {
                const Eigen::Vector3d u = Td.col(a);
                const double ca = u.dot(c.cast<double>());
                Vec x;
                for (int j = 0; j < NP; ++j) x(j) = u.dot(R[j] * u);
                const double o = u.dot(O * u);
                const double m = o + x.dot(theta_);
                // m is a variance of the model's own making; with Λ ⪰ 0 it is ≥ 0, and m = 0 means the
                // scan did not inform this axis at all (x = 0 too) — nothing to learn from it.
                // m = 0: the scan did not inform this axis (x = 0 too) — nothing to learn from it.
                if (m > 0.0) observe_axis(x, ca * ca - o, m);
            }
            end_correction();
        }

    private:
        Params p_{};
        Mat N_ = Mat::Zero();
        Vec b_ = Vec::Zero();
        Vec theta_ = Vec::Zero();
        long n_ = 0;
    };
}
