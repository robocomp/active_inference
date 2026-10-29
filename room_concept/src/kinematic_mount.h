/*  kinematic_mount.h — the helios (dx, dy, psi) mount from the localiser's innovation stream.
 *  Pure: no Qt, no DSR, no torch.
 *
 *  Split from mount_factors.h (2026-10-29, sensor_calibration Phase 3): the floor and vertical factors and
 *  the shared Info3/Info6/PlaceCells now live in common/sensor_calib/mount_factors.h and run in the
 *  sensor_calibration agent. This factor STAYS here because it reads per-cycle localiser internals
 *  (observe_innovation) and the motion calibrator's posterior. Plan task 4.4 replaces it.
 *  Conventions: see common/sensor_calib/mount_factors.h.
 */
#pragma once
#include "sensor_calib/mount_factors.h"

namespace rc::mountf
{
    /* ─────────────────────────────────────── KinematicMount ───────────────────────────────────────
     *  helios (dx, dy, psi) from the per-cycle innovation of consecutive scan-only poses against the
     *  calibrated odometry (motion_noise_innov.h: delta = z_n - z_{n-1} - odom_n, projected on the body axes
     *  at the MIDPOINT heading th_mid = z_th - odom_th/2, room_concept.cpp observe_innovation).
     *
     *  ROWS (r2.2 item 1). With a planar mount error M = {dx, dy, psi} (rotation about the BODY origin, the
     *  form the 2-D localiser absorbs: z = X_true o M), R(th_mid)^T [R(th_n) - R(th_{n-1})] (dx, dy) =
     *  2 sin(dth/2) J (dx, dy), J = [[0,-1],[1,0]], so per cycle
     *        delta_fwd = +2 sin(dth/2) dx                 (+ u_kv  d_fwd)
     *        delta_lat = -2 sin(dth/2) dy + psi d_fwd     (+ u_lat d_lat)
     *  no (1 - cos) columns, no mount term on delta_th. psi = the helios yaw as the localiser ADDS it to the
     *  heading (theta_est = theta_true + psi); its sign is pinned by the logs (planted +1 deg read +0.81/+0.92,
     *  null +0.03; review table). Body X = lateral, Y = forward. ⚠ The innovation axes come out of `T` in the
     *  order (FORWARD, LATERAL): Cycle names them so the order cannot be mixed up.
     *  The lever regressor is pivots (2 sin(dth/2) with d_fwd = 0), the yaw regressor is straights (d_fwd):
     *  a run with only one kind leaves the other at its prior (selftest B, C).
     *
     *  WEIGHTS. Row variance = the noise learner's model variance for that axis that cycle (`last_m(a)`), times
     *  (1 - outlier responsibility). Every row at weight 1 -- NOT the learner's ma1_weight 0.5, which belongs
     *  to its VARIANCE regression. Consecutive innovations share one scan error (MA(1), lag-1 corr -1/2); for a
     *  MEAN regression on smooth regressors that error TELESCOPES, so the independent-row sigma reported here
     *  is an UPPER BOUND (conservative), never an overclaim.
     *
     *  ODOMETRY READINESS (Task 1b, r2.2 item 4): a COMMON-MODE marginalisation, not a per-row variance. An
     *  error u in the applied odometry parameters (u = theta_true - theta_applied; k_v on delta_fwd with
     *  covariate d_fwd, k_lat on delta_lat with covariate d_lat -- identically 0 on a differential base, kept;
     *  the heading parameters reach translation only at second order and are omitted) is the SAME on every
     *  row, so it biases the mount coherently; as a per-row variance it would be (0.4 mm)^2 against
     *  (4-6 mm)^2 rows: inert. Stored:
     *        N = sum w x x^T,   C = sum w x J_o^T,   D = sum w J_o J_o^T,   b = sum w x y,   e = sum w J_o y
     *  and info() reports the Schur complement over u with prior N(mu, Sigma_o):
     *        H_mount = N - C (D + Sigma_o^-1)^-1 C^T,   b_mount = b - C (D + Sigma_o^-1)^-1 (e + Sigma_o^-1 mu)
     *  Still DECOUPLED: u is integrated out against the calibrator's posterior, never re-estimated, nothing acts.
     *  It reproduces "the extrinsic waits for a stable odometry" BY WEIGHT: Sigma_o wide (fresh robot, reset,
     *  little excitation) -> the lever directions that k_v can mimic stay wide; Sigma_o tight -> costs nothing;
     *  turns in both directions -> C ~ 0, a no-op. mu = (calibrator estimate - applied value): non-zero only
     *  when the calibrator knows a value it is not applying (mask/open loop).
     *  Sigma_o = calibrator posterior + Sigma_drift, the exponentially weighted mean outer product of the
     *  APPLIED theta_o's change per re-solve (a calibration still sliding is distrusted by how much it moves,
     *  even when its reported sigma is small). RE-REFERENCE: rows were formed under the theta_o applied at the
     *  time; when it changes by d, every earlier row's y carries J_o d more than the present reference says,
     *  so b -= C d, e -= D d (exact for the linear model). SINGLE OWNER: eps_yaw / the r1 lever must not act on
     *  the odometry while this runs (room_config: lidar_side_yaw + eps apply bit) and are not in theta_o.
     */
    class KinematicMount
    {
    public:
        static constexpr int K = 2;              ///< odometry nuisance: (k_v, k_lat)
        using V2 = Eigen::Vector2d;
        using M2 = Eigen::Matrix2d;
        using M32 = Eigen::Matrix<double, 3, K>;

        struct Params
        {
            double sigma_xy = 0.05, sigma_yaw = 0.0175;   ///< prior on (dx, dy) m and psi rad
            double memory = 50000.0;                      ///< cycles of evidence remembered (~42 min at 20 Hz)
            double drift_memory = 20.0;                   ///< re-solves remembered by Sigma_drift
            V2     odo_prior_sigma{0.02, 0.05};           ///< Sigma_o until set_odometry (the calibrator's priors)
            bool   drift = true;                          ///< false only for the selftest's control arm
        };
        /// One fed cycle (room_concept observe_innovation, only when `fed`). innov_* = the innovation on the
        /// body axes; dth, d_fwd, d_lat = this cycle's ODOMETRY (rad, m); var_* = the noise learner's model
        /// variance (last_m) per axis; w_* = 1 - its outlier responsibility (last_r).
        struct Cycle
        {
            double innov_fwd = 0.0, innov_lat = 0.0;
            double dth = 0.0, d_fwd = 0.0, d_lat = 0.0;
            double var_fwd = 0.0, var_lat = 0.0;
            double w_fwd = 1.0, w_lat = 1.0;
        };

        KinematicMount() { set_params(Params{}); }
        void set_params(const Params &p)
        {
            p_ = p;
            N_.setZero(); b_.setZero(); C_.setZero(); D_.setZero(); e_.setZero(); n_ = 0;
            Sigma_post_ = p.odo_prior_sigma.cwiseAbs2().asDiagonal();
            mu_.setZero(); theta_prev_.setZero(); odo_init_ = false;
            S_drift_.setZero(); W_drift_ = 0.0; drift_acc_.setZero(); last_solve_ = std::numeric_limits<long>::min();
        }
        [[nodiscard]] const Params &params() const noexcept { return p_; }

        void observe(const Cycle &c)
        {
            const double s = 2.0 * std::sin(0.5 * c.dth);
            if (not std::isfinite(s) or not std::isfinite(c.d_fwd) or not std::isfinite(c.d_lat)) return;
            const double lam = std::exp(-1.0 / p_.memory);
            N_ *= lam; b_ *= lam; C_ *= lam; D_ *= lam; e_ *= lam;
            add_row(Eigen::Vector3d(s, 0.0, 0.0), V2(c.d_fwd, 0.0), c.innov_fwd, c.var_fwd, c.w_fwd);
            add_row(Eigen::Vector3d(0.0, -s, c.d_fwd), V2(0.0, c.d_lat), c.innov_lat, c.var_lat, c.w_lat);
            ++n_;
        }

        /// The odometry calibration as it stands this cycle: its posterior covariance over (k_v, k_lat), the
        /// APPLIED values, mu = estimate - applied, and the calibrator's re-solve counter (Sigma_drift
        /// bookkeeping). Call every cycle, before observe().
        void set_odometry(const M2 &Sigma_post, const V2 &theta_applied, const V2 &mu, long solve_id)
        {
            if (Sigma_post.allFinite()) Sigma_post_ = 0.5 * (Sigma_post + Sigma_post.transpose());
            if (mu.allFinite()) mu_ = mu;
            if (not theta_applied.allFinite()) return;
            if (not odo_init_) { theta_prev_ = theta_applied; odo_init_ = true; last_solve_ = solve_id; return; }
            const V2 d = theta_applied - theta_prev_;
            if (d.squaredNorm() > 0.0)
            {   // re-reference the evidence to the theta_o now in force (exact for the linear model)
                b_ -= C_ * d;
                e_ -= D_ * d;
                theta_prev_ = theta_applied;
                drift_acc_ += d;
            }
            if (solve_id != last_solve_)
            {
                const double lam = std::exp(-1.0 / p_.drift_memory);
                S_drift_ = lam * S_drift_ + drift_acc_ * drift_acc_.transpose();
                W_drift_ = lam * W_drift_ + 1.0;
                drift_acc_.setZero();
                last_solve_ = solve_id;
            }
        }
        /// Sigma_o = posterior + drift: the prior of the shared odometry nuisance.
        [[nodiscard]] M2 sigma_odometry() const
        {
            M2 S = Sigma_post_;
            if (p_.drift and W_drift_ > 0.0) S += S_drift_ / std::max(W_drift_, 1.0);
            return S;
        }

        [[nodiscard]] Info3 info() const
        {
            Info3 o;
            const M2 S = sigma_odometry();
            // (D + S^-1)^-1 = S (I + D S)^-1 and (D + S^-1)^-1 S^-1 = (I + S D)^-1: stable as S -> 0.
            const M2 I = M2::Identity();
            const M2 Minv = S * (I + D_ * S).inverse();
            const V2 nu = Minv * e_ + (I + S * D_).inverse() * mu_;
            o.H_data = N_ - C_ * Minv * C_.transpose();
            o.H_data = 0.5 * (o.H_data + o.H_data.transpose());
            o.b_data = b_ - C_ * nu;
            o.H_prior.diagonal() << 1.0 / (p_.sigma_xy * p_.sigma_xy), 1.0 / (p_.sigma_xy * p_.sigma_xy),
                                    1.0 / (p_.sigma_yaw * p_.sigma_yaw);
            o.n = n_;
            return o;
        }
        [[nodiscard]] Eigen::Vector3d mean() const { return info().mean(); }
        [[nodiscard]] Eigen::Vector3d sigma() const { return info().sigma(); }
        [[nodiscard]] long cycles() const noexcept { return n_; }

    private:
        void add_row(const Eigen::Vector3d &x, const V2 &jo, double y, double var, double w_in)
        {
            if (not (var > 0.0) or not std::isfinite(var) or not std::isfinite(y) or not std::isfinite(w_in)) return;
            const double w = std::clamp(w_in, 0.0, 1.0) / var;
            N_.noalias() += w * x * x.transpose();
            b_.noalias() += (w * y) * x;
            C_.noalias() += w * x * jo.transpose();
            D_.noalias() += w * jo * jo.transpose();
            e_.noalias() += (w * y) * jo;
        }

        Params p_{};
        Eigen::Matrix3d N_ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b_ = Eigen::Vector3d::Zero();
        M32 C_ = M32::Zero();
        M2  D_ = M2::Zero();
        V2  e_ = V2::Zero();
        long n_ = 0;
        M2  Sigma_post_ = M2::Zero(), S_drift_ = M2::Zero();
        V2  mu_ = V2::Zero(), theta_prev_ = V2::Zero(), drift_acc_ = V2::Zero();
        double W_drift_ = 0.0;
        bool odo_init_ = false;
        long last_solve_ = std::numeric_limits<long>::min();
    };
}
