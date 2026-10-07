/*  drift_monitor.h — the SYSTEMATIC part of the odometry's prediction error, live, GT-free.
 *
 *  The self-calibration measure (tools/surprise_report.py drift() is its offline twin). Per open-loop
 *  stretch between two scored corrections, the correction c (scan solve minus prediction, per body axis)
 *  is regressed on the odometry's own signed motion over the stretch:
 *        E[c_a] = b_a,d · d + b_a,φ · φ + b_a,t · T
 *  d = signed forward travel (m), φ = signed turn (rad), T = duration (s). Calibration errors (wheel scale,
 *  track, gyro scale, mount yaw, gyro bias) grow LINEARLY with a sign; slip and noise grow like a square
 *  root and average out — so b is what the calibration must drive to 0, separated from the noise a raw
 *  "error per metre" mixes in.
 *
 *  Recursive weighted least squares per axis, weight = 1 / (the model's predicted variance of c on that
 *  axis), exponential forgetting (so the curve shows convergence), and a weak Gaussian prior at b = 0 so
 *  an unexcited covariate (no turn yet, say) stays defined at "no evidence" instead of exploding. The
 *  standard error comes from the same normal matrix. Stretches shorter than min_open cycles are not fed:
 *  they end because the scan disagreed (early-exit selection), not because a drift budget ran out.
 */
#pragma once
#include <Eigen/Dense>
#include <array>
#include <cmath>

namespace rc
{
    class DriftMonitor
    {
    public:
        struct Params
        {
            double memory   = 150.0;                       ///< stretches remembered
            Eigen::Vector3d prior_sd{0.05, 0.05, 1e-3};    ///< per m, per rad, per s (in c's units)
            int    min_open = 5;
        };
        void set_params(const Params &p) { p_ = p; }

        /// One scored correction. c: (fwd m, lat m, heading rad); var: the model's predicted variance of
        /// each; d, phi, T: the stretch's signed forward travel, signed turn, duration; open: its cycles.
        void observe(const Eigen::Vector3d &c, const Eigen::Vector3d &var, double d, double phi, double T, int open)
        {
            if (open < p_.min_open or not c.allFinite() or not std::isfinite(d + phi + T)) return;
            const Eigen::Vector3d x(d, phi, T);
            const double decay = std::exp(-1.0 / p_.memory);
            for (int a = 0; a < 3; ++a)
            {
                const double w = 1.0 / std::max(var(a), 1e-12);
                N_[a] = decay * N_[a] + w * x * x.transpose();
                r_[a] = decay * r_[a] + w * c(a) * x;
            }
            ++n_;
        }
        /// b (3 axes × 3 covariates) and their standard errors.
        void solve(Eigen::Matrix3d &b, Eigen::Matrix3d &se) const
        {
            for (int a = 0; a < 3; ++a)
            {
                Eigen::Matrix3d A = N_[a];
                for (int j = 0; j < 3; ++j) A(j, j) += 1.0 / (p_.prior_sd(j) * p_.prior_sd(j));
                const Eigen::Matrix3d C = A.inverse();
                b.row(a)  = (C * r_[a]).transpose();
                se.row(a) = C.diagonal().cwiseSqrt().transpose();
            }
        }
        [[nodiscard]] long stretches() const noexcept { return n_; }

    private:
        Params p_{};
        std::array<Eigen::Matrix3d, 3> N_{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
        std::array<Eigen::Vector3d, 3> r_{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
        long n_ = 0;
    };
}
