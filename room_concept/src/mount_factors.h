/*  mount_factors.h — the LiDAR mounts against the robot BODY, from three exteroceptive/kinematic factors,
 *  with the odometry treated as already calibrated (decoupled). Pure: no Qt, no DSR, no torch.
 *
 *  Plan: docs/superpowers/plans/2026-10-08-lidar-mounts-kinematic-floor.md (r2.2; the "r2.2 CORRECTIONS"
 *  section supersedes the task text). Review: docs/fable_r2_review_2026-10-08.md.
 *
 *    KinematicMount   helios (dx, dy, psi)        consecutive scan-only poses vs the calibrated odometry
 *    FloorPlaneMount  bpearl (droll, dpitch, dz)  the floor plane seen head-on (absolute, gauge G4)
 *    VerticalMount    helios (droll, dpitch)      walls are vertical, the ceiling is horizontal
 *                     bpearl (dx, dy, dyaw)       RELATIVE to the helios-built wall map
 *
 *  Every accumulator keeps its normal equations in PHYSICAL units (m, rad), the prior SEPARATE (Info*.H_prior),
 *  exponential forgetting, and any per-sweep common mode marginalised by a Schur complement -- never a sigma
 *  floor (CLAUDE.md). Fusion with the cameras: joint_calibration.h solve_mounts().
 *
 *  CONVENTION (plan "Global Constraints"; r1 lidar_mount.h). Every estimate is the CORRECTION M: nominal
 *  points are restored by M (p_true = M(p_nom)), a planted error is injected by inverse(M). For the 6-DoF
 *  blocks M is a small rotation omega = (droll, dpitch, dyaw) ABOUT THE SENSOR ORIGIN s plus a translation
 *  dt, all in the BODY frame:  p_true ~= p_nom + omega x (p_nom - s) + dt.
 */
#pragma once
#include <Eigen/Dense>
#include <map>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rc::mountf
{
    inline constexpr double kDeg = M_PI / 180.0;

    /// Normal equations of a 3-parameter block: data and prior kept apart so the joint can use the data
    /// alone and put ONE prior per physical parameter.
    struct Info3
    {
        Eigen::Matrix3d H_data  = Eigen::Matrix3d::Zero(), H_prior = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b_data  = Eigen::Vector3d::Zero(), b_prior = Eigen::Vector3d::Zero();
        long n = 0;
        [[nodiscard]] Eigen::Vector3d mean() const
        { return (H_data + H_prior).ldlt().solve(b_data + b_prior); }
        [[nodiscard]] Eigen::Vector3d sigma() const
        { return (H_data + H_prior).inverse().diagonal().cwiseMax(0.0).cwiseSqrt(); }
    };
    /// The same for the 6-DoF mount, order (dx, dy, dz, droll, dpitch, dyaw).
    struct Info6
    {
        using M6 = Eigen::Matrix<double, 6, 6>;
        using V6 = Eigen::Matrix<double, 6, 1>;
        M6 H_data = M6::Zero(), H_prior = M6::Zero();
        V6 b_data = V6::Zero(), b_prior = V6::Zero();
        long n = 0;
        [[nodiscard]] V6 mean() const { return (H_data + H_prior).ldlt().solve(b_data + b_prior); }
        [[nodiscard]] V6 sigma() const { return (H_data + H_prior).inverse().diagonal().cwiseMax(0.0).cwiseSqrt(); }
    };

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

    /* ─────────────────────────────────────── FloorPlaneMount ──────────────────────────────────────
     *  bpearl (droll, dpitch, dz) from the floor seen head-on: the floor is z = 0 in the Shadow frame (gauge
     *  G4), absolute. Points p arrive in the SHADOW frame (LidarIngestor polls LIDAR_ROBOT_FRAME = the type-
     *  "robot" node = Shadow, origin on the floor), so the sensor origin t_nom must be in the SAME frame:
     *  bpearl (0, 0.14, 0.7025) = body (0, 0.14, 0.67) + 0.0325 (r2.2 item 5; ROBOT_GEOMETRY.md). With
     *  q = p - t_nom, a small correction (omega, dt) moves a point by omega x q + dt, so a floor point gives
     *        p_z + omega_x q_y - omega_y q_x + dz = 0,     J = [q_y, -q_x, 1]  on (droll, dpitch, dz).
     *  ASSOCIATION WITHOUT A BAND (Review Focus #2): a two-component mixture per point -- floor N(c_s, sigma_f^2)
     *  vs not-floor, UNIFORM over the sweep's OBSERVED z range -- with sigma_f and the floor share learnt per
     *  sweep by EM (sweep-local Gauss-Newton inside, so the residuals are taken at this sweep's best tilt, not
     *  at a stale one); rows are responsibility-weighted.
     *  PER-SWEEP COMMON MODE: the floor is not perfectly flat and the whole sweep shares one timing/pose, so a
     *  per-sweep offset dz_s with prior sigma_flat (5 mm) is marginalised (Schur) before the sweep's 3x3 is
     *  added -- 10^4 points cannot claim 10^4 independent measurements of dz. It is z-only: it does not
     *  absorb roll/pitch (selftest part 2 C pins that).
     *  ⚠ HARDWARE CAVEAT: a wheeled base pitches under acceleration and braking, so on the real robot each
     *  sweep also needs per-sweep (droll_s, dpitch_s) nuisances, or the asymmetry of the driving biases the
     *  mean tilt. Webots' body is rigid, so a simulation pass does NOT test this. Add them before trusting a
     *  real-robot floor tilt.
     *  ⚠ WRITE-BACK: the estimate is a rotation in BODY axes about the sensor origin. The bpearl (and helios)
     *  hang INVERTED (nominal R = Ry(pi)), so the SENSOR-frame increment of a body-frame rotation omega is
     *  Ry(pi)^T omega = (-omega_x, omega_y, -omega_z): roll and yaw FLIP, pitch does not. A future write-back to
     *  shadow.json must COMPOSE the rotation, never add the numbers to its euler angles.
     */
    /// ── PLACE-LEVEL COMMON MODE (2026-10-08, fix A) ───────────────────────────────────────────────────────────
    /// Sweeps taken at one PLACE share that place's own systematic error: the map's wall a few cm off there, the
    /// floor's local slope, furniture at the base of a wall. A parked robot repeats one view thousands of times;
    /// counted as independent, the place error became a confident "mount" reading (bpearl (+48, -32) mm from 2461
    /// parked sweeps in one heading while the 1027 moving sweeps read (+5, -2) mm, run 2026-10-08 18-03).
    /// Model: each place cell c carries a nuisance nu_c ~ N(0, P) entering exactly like the mount (the same
    /// Jacobian), marginalised per cell:  H_eff = H_c - H_c (H_c + P^-1)^-1 H_c,  b_eff = b_c - H_c (H_c + P^-1)^-1 b_c.
    /// One place can then never claim more than P^-1 however many sweeps it repeats; the information that keeps
    /// growing is the information from NEW places. Columns with no place error get P^-1 -> infinity (no nuisance).
    /// Cells forget with the factor's memory. Key = (0.5 m grid, heading octant), chosen by the caller.
    template <int N>
    struct PlaceCells
    {
        using M = Eigen::Matrix<double, N, N>;
        using V = Eigen::Matrix<double, N, 1>;
        struct Cell { M H = M::Zero(); V b = V::Zero(); };
        std::map<std::int64_t, Cell> cells;
        V place_prec = V::Constant(1e12);          ///< 1 / sigma_place^2 per column; 1e12 = no place error there

        void add(std::int64_t key, const M &Hs, const V &bs, double lam)
        {
            for (auto it = cells.begin(); it != cells.end();)
            {
                it->second.H *= lam; it->second.b *= lam;
                // numerical housekeeping, not a model choice: a cell forgotten to ~nothing is dropped
                if (it->second.H.cwiseAbs().maxCoeff() < 1e-12) it = cells.erase(it); else ++it;
            }
            auto &c = cells[key];
            c.H += Hs;  c.b += bs;
        }
        void clear() { cells.clear(); }
        /// The sum over cells of each cell's evidence with its place nuisance marginalised.
        void effective(M &H, V &b) const
        {
            H.setZero(); b.setZero();
            const M Lam = place_prec.asDiagonal();
            for (const auto &kv : cells)
            {
                const M &Hc = kv.second.H;  const V &bc = kv.second.b;
                const Eigen::LDLT<M> l(Hc + Lam);
                if (l.info() != Eigen::Success) continue;
                const M K = l.solve(Hc);                 // (H_c + Lam)^-1 H_c
                H += Hc - Hc * K;
                b += bc - K.transpose() * bc;            // H_c (H_c + Lam)^-1 b_c, symmetric H_c
            }
            H = 0.5 * (H + H.transpose());
        }
        [[nodiscard]] std::size_t size() const noexcept { return cells.size(); }
    };

    /// The place key: 0.5 m grid cell and heading octant of the pose a sweep was taken from.
    [[nodiscard]] inline std::int64_t place_key(float x, float y, float heading)
    {
        const auto kx = static_cast<std::int64_t>(std::floor(x / 0.5f)) + 100000;
        const auto ky = static_cast<std::int64_t>(std::floor(y / 0.5f)) + 100000;
        const float h = std::remainder(heading, 2.f * float(M_PI)) + float(M_PI);   // [0, 2pi)
        const auto kh = static_cast<std::int64_t>(std::floor(h / float(M_PI / 4.0))) & 7;
        return (kx * 1000000 + ky) * 8 + kh;
    }

    class FloorPlaneMount
    {
    public:
        struct Params
        {
            double sigma_roll = 1.0 * kDeg, sigma_pitch = 1.0 * kDeg, sigma_z = 0.02;   ///< prior
            double sigma_flat = 0.005;     ///< m, per-sweep floor offset (flatness + per-sweep timing/pose)
            double memory = 2000.0;        ///< sweeps remembered (~17 min at 2 Hz)
            int    em_iters = 3;
            double sigma_f0 = 0.01, sigma_f_pseudo = 20.0;   ///< weak prior on the floor-plane residual scale
            /// The floor's own error at one place (local slope, unevenness): (droll, dpitch, dz) — see PlaceCells
            Eigen::Vector3d place_sigma = Eigen::Vector3d(0.05 * kDeg, 0.05 * kDeg, 0.005);
        };
        FloorPlaneMount() { set_params(Params{}); }
        void set_params(const Params &p)
        {
            p_ = p; H_.setZero(); b_.setZero(); n_ = 0; share_ = 0.5; sigma_f_ = p.sigma_f0;
            cells_.clear(); cells_.place_prec = p.place_sigma.cwiseAbs2().cwiseInverse();
        }

        /// place: the place key of the pose this sweep was taken from (place_key()); -1 = a place of its own.
        void observe_sweep(const std::vector<Eigen::Vector3f> &p_body, const Eigen::Vector3f &t_nom,
                           std::int64_t place = -1)
        {
            std::vector<Eigen::Vector3d> J;  std::vector<double> y;
            J.reserve(p_body.size()); y.reserve(p_body.size());
            double zmin = std::numeric_limits<double>::infinity(), zmax = -zmin;
            for (const auto &pf : p_body)
            {
                if (not pf.allFinite()) continue;
                const Eigen::Vector3d q = (pf - t_nom).cast<double>();
                J.emplace_back(q.y(), -q.x(), 1.0);
                y.push_back(-static_cast<double>(pf.z()));
                zmin = std::min(zmin, double(pf.z())); zmax = std::max(zmax, double(pf.z()));
            }
            const std::size_t n = y.size();
            const double L = zmax - zmin;
            if (n < 10 or not (L > 1e-3)) return;
            const Info3 g = info();
            const Eigen::Matrix3d Hg = g.H_data + g.H_prior;
            const Eigen::Vector3d bg = g.b_data + g.b_prior;
            Eigen::Vector3d th = Hg.ldlt().solve(bg);
            double sig = sigma_f_, pi = std::clamp(share_, 1e-3, 1.0 - 1e-3), c = 0.0;
            std::vector<double> resp(n, 0.0);
            Eigen::Matrix3d Hs = Eigen::Matrix3d::Zero(); Eigen::Vector3d bs = Eigen::Vector3d::Zero();
            for (int it = 0; it <= p_.em_iters; ++it)
            {
                // E: floor responsibility of every point at the current (sweep-local) tilt and offset
                double sr = 0.0, srr = 0.0;
                for (std::size_t i = 0; i < n; ++i)
                {
                    const double r = J[i].dot(th) + c - y[i];          // corrected z of point i, less the sweep offset
                    const double lf = std::log(pi) - std::log(sig) - 0.5 * (r * r) / (sig * sig) - 0.5 * std::log(2.0 * M_PI);
                    const double lo = std::log(1.0 - pi) - std::log(L);
                    resp[i] = 1.0 / (1.0 + std::exp(lo - lf));
                    sr += resp[i]; srr += resp[i] * r * r;
                }
                // M: scale and share (weak priors), then the sweep's normal equations with dz_s marginalised
                sig = std::sqrt((srr + p_.sigma_f_pseudo * p_.sigma_f0 * p_.sigma_f0) / (sr + p_.sigma_f_pseudo));
                pi  = (sr + 1.0) / (double(n) + 2.0);
                sweep_normal_equations(J, y, resp, sig, Hs, bs, &c, th, Hg, bg);
            }
            const double lam = std::exp(-1.0 / p_.memory);
            cells_.add(place >= 0 ? place : -(++anon_), Hs, bs, lam);
            cells_.effective(H_, b_);
            share_ = pi; sigma_f_ = sig; ++n_;
        }

        [[nodiscard]] Info3 info() const
        {
            Info3 o;
            o.H_data = 0.5 * (H_ + H_.transpose()); o.b_data = b_;
            o.H_prior.diagonal() << 1.0 / (p_.sigma_roll * p_.sigma_roll), 1.0 / (p_.sigma_pitch * p_.sigma_pitch),
                                    1.0 / (p_.sigma_z * p_.sigma_z);
            o.n = n_;
            return o;
        }
        [[nodiscard]] double floor_share() const noexcept { return share_; }
        [[nodiscard]] double sigma_floor() const noexcept { return sigma_f_; }
        [[nodiscard]] long sweeps() const noexcept { return n_; }

    private:
        /// One sweep's 3x3 normal equations on (droll, dpitch, dz) with the per-sweep offset dz_s (prior
        /// sigma_flat) Schur-marginalised. Also returns the sweep-local offset and tilt (for the next E-step).
        void sweep_normal_equations(const std::vector<Eigen::Vector3d> &J, const std::vector<double> &y,
                                    const std::vector<double> &resp, double sig, Eigen::Matrix3d &Hs,
                                    Eigen::Vector3d &bs, double *c, Eigen::Vector3d &th,
                                    const Eigen::Matrix3d &Hg, const Eigen::Vector3d &bg) const
        {
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero(); Eigen::Vector3d a = Eigen::Vector3d::Zero(), beta = a;
            double alpha = 0.0, gamma = 0.0;
            const double iv = 1.0 / (sig * sig);
            for (std::size_t i = 0; i < y.size(); ++i)
            {
                const double w = resp[i] * iv;
                A.noalias() += w * J[i] * J[i].transpose();
                a += w * J[i]; beta += (w * y[i]) * J[i];
                alpha += w; gamma += w * y[i];
            }
            const double den = alpha + 1.0 / (p_.sigma_flat * p_.sigma_flat);
            Hs = A - a * a.transpose() / den;
            bs = beta - a * (gamma / den);
            th = (Hg + Hs).ldlt().solve(bg + bs);                        // sweep-local best tilt
            *c = (gamma - a.dot(th)) / den;                              // its offset dz_s: rows y = J th + dz_s
            // (the E-step's residual is r = J th + dz_s - y, so a point on this sweep's offset floor reads 0)
        }

        Params p_{};
        Eigen::Matrix3d H_ = Eigen::Matrix3d::Zero();
        PlaceCells<3> cells_;
        std::int64_t anon_ = 0;
        Eigen::Vector3d b_ = Eigen::Vector3d::Zero();
        long n_ = 0;
        double share_ = 0.5, sigma_f_ = 0.01;
    };

    /* ─────────────────────────────────────── VerticalMount ────────────────────────────────────────
     *  Walls are vertical, the ceiling is horizontal. One instance per LiDAR, with a ROLE that says which
     *  columns it may inform (r2.2 item 6):
     *    HeliosTilt   helios (droll, dpitch): points at DIFFERENT HEIGHTS on one wall separate a tilt from a
     *                 planar shift (q_z spans ~ -0.55 .. +1.5 m, two-sided), plus the ceiling z = h_c with h_c a
     *                 session nuisance (so helios dz is EXACTLY unobservable here and stays at its prior,
     *                 Review Focus #4). NO helios planar rows: the map was fitted to the helios scan, so
     *                 n^T p - d ~ 0 by construction, and any per-sweep pose prior would turn that into a false
     *                 "dx = 0" claim (10^3 sweeps x (5 cm)^-2) that out-votes the kinematic factor.
     *    BpearlPlanar bpearl (dx, dy, dyaw) RELATIVE TO THE HELIOS-BUILT MAP = bpearl relative to the helios
     *                 (joint_calibration.h maps it with the tilt coupling). Its wall span is one-sided
     *                 (0 .. 0.7 m), so tilt-vs-shift is near collinear: the FLOOR owns the bpearl tilt, the
     *                 verticals give planar only.
     *  ROWS. For a point p (body frame) associated with wall n^T xy = d (n unit), q = p - t_nom:
     *        n^T p_xy - d + n^T (omega x q)_xy + n^T dt_xy + n^T (eta_xy + eta_yaw J p_xy) = 0
     *        (omega x q)_xy = (omega_y q_z - omega_z q_y,  omega_z q_x - omega_x q_z)
     *  eta = the per-sweep planar POSE error of the localiser (identical for every point of the sweep): a
     *  nuisance with a WIDE prior (pose_cov: the pose-field sigma, 0.05 m / 2 deg -- never the solver's own
     *  marginal) MARGINALISED per sweep by Schur, so pose error is never read as mount (Review Focus #1).
     *  Ceiling (helios): p_z + omega_x q_y - omega_y q_x - h_c = 0.
     *  Association: nearest wall SEGMENT (distance to the finite segment, not a band), then a mixture -- that
     *  wall N(0, sigma_w^2) / the ceiling N(0, sigma_w^2) (helios) / not-structure, uniform over the sweep's
     *  observed residual range -- EM with a sweep-local Gauss-Newton inside, responsibility-weighted rows, as
     *  FloorPlaneMount. Soft, at the current estimate: never a hard nearest-model pick (see the rows below).
     *  The sweep must be paired with the localiser pose AT ITS OWN STAMP (the bpearl and helios sweeps are not
     *  simultaneous; the wide nuisance prior absorbs what is left).
     *  ⚠ Same hardware caveat and write-back note as FloorPlaneMount.
     */
    class VerticalMount
    {
    public:
        enum class Role { HeliosTilt, BpearlPlanar };
        /// A wall of the map as a SEGMENT in the BODY frame at this sweep's pose.
        struct Wall { Eigen::Vector2f a, b; };
        using V6 = Eigen::Matrix<double, 6, 1>;
        static constexpr int NG = 7, NN = 3;   ///< global (dx, dy, dz, droll, dpitch, dyaw, h_c), nuisance (x, y, yaw)
        struct Params
        {
            Role role = Role::HeliosTilt;
            V6 prior_sigma = (V6() << 0.05, 0.05, 0.02, 1.0 * kDeg, 1.0 * kDeg, 1.0 * kDeg).finished();
            double sigma_hc = 1.0;           ///< m, the ceiling-height nuisance prior around the hint
            double memory = 2000.0;          ///< sweeps remembered
            int    em_iters = 3;
            double sigma_w0 = 0.02, sigma_w_pseudo = 20.0;   ///< weak prior on the structure residual scale
            /// The place's own error (the map's wall there, furniture at its base): (dx, dy, dz, droll, dpitch, dyaw)
            /// — see PlaceCells. No place error on the ceiling-height column.
            V6 place_sigma = (V6() << 0.03, 0.03, 0.01, 0.05 * kDeg, 0.05 * kDeg, 0.5 * kDeg).finished();
        };
        VerticalMount() { set_params(Params{}); }
        void set_params(const Params &p)
        {
            p_ = p; H_.setZero(); b_.setZero(); n_ = 0; share_ = 0.5; sigma_w_ = p.sigma_w0; hint_ = 0.0;
            cells_.clear();
            cells_.place_prec.head<6>() = p.place_sigma.cwiseAbs2().cwiseInverse();
            cells_.place_prec(6) = 1e12;
        }
        [[nodiscard]] const Params &params() const noexcept { return p_; }

        /// One sweep. p_body: the sweep in the body (Shadow) frame through the NOMINAL mount; t_nom: this
        /// sensor's origin in the same frame; walls: the map in the body frame at the pose of THIS sweep's
        /// stamp; pose_cov: prior covariance of that pose's error (body x, y, yaw); ceiling_hint: the measured
        /// ceiling height (m, same frame), <= 0 or non-finite = no ceiling rows (helios only).
        /// place: the place key of this sweep's pose (place_key()); -1 = a place of its own (the old behaviour).
        void observe_sweep(const std::vector<Eigen::Vector3f> &p_body, const Eigen::Vector3f &t_nom,
                           const std::vector<Wall> &walls, const Eigen::Matrix3f &pose_cov, float ceiling_hint,
                           std::int64_t place = -1)
        {
            if (walls.empty() or not pose_cov.allFinite()) return;
            const bool helios = p_.role == Role::HeliosTilt;
            const bool ceil = helios and std::isfinite(ceiling_hint) and ceiling_hint > 0.f;
            if (ceil) hint_ = ceiling_hint;
            // wall normals / offsets, body frame
            struct L { Eigen::Vector2d a, u, n; double len, d; };
            std::vector<L> lines;
            for (const auto &w : walls)
            {
                const Eigen::Vector2d a = w.a.cast<double>(), b = w.b.cast<double>();
                const double len = (b - a).norm();
                if (not (len > 1e-6) or not a.allFinite() or not b.allFinite()) continue;
                const Eigen::Vector2d u = (b - a) / len, n(-u.y(), u.x());
                lines.push_back({a, u, n, len, n.dot(a)});
            }
            if (lines.empty()) return;
            // rows (built once): a WALL part (nearest segment) and, for the helios, a CEILING part. A point is
            // soft-assigned between wall / ceiling / not-structure by the EM below, at the sweep-local estimate:
            // a HARD nearest-model assignment truncated the points near the wall-ceiling junction and biased the
            // tilt 2 % low under 1 cm noise (selftest part 3 A, found while building this).
            rows_.clear();
            for (const auto &pf : p_body)
            {
                if (not pf.allFinite()) continue;
                const Eigen::Vector3d p = pf.cast<double>(), q = (pf - t_nom).cast<double>();
                const Eigen::Vector2d pxy = p.head<2>();
                double best = std::numeric_limits<double>::infinity(); const L *lw = nullptr;
                for (const auto &l : lines)
                {
                    const double t = std::clamp((pxy - l.a).dot(l.u), 0.0, l.len);
                    const double dist = (pxy - (l.a + t * l.u)).norm();
                    if (dist < best) { best = dist; lw = &l; }
                }
                Row r;
                const Eigen::Vector2d &n = lw->n;
                if (helios) { r.Jw(3) = -n.y() * q.z(); r.Jw(4) = n.x() * q.z(); }
                else { r.Jw(0) = n.x(); r.Jw(1) = n.y(); r.Jw(5) = -n.x() * q.y() + n.y() * q.x(); }
                r.Jn << n.x(), n.y(), -n.x() * pxy.y() + n.y() * pxy.x();
                r.yw = -(n.dot(pxy) - lw->d);
                if (ceil)
                {   // p_z + omega_x q_y - omega_y q_x - h_c = 0
                    r.has_c = true;
                    r.Jc(3) = q.y(); r.Jc(4) = -q.x(); r.Jc(6) = -1.0;
                    r.yc = -p.z();
                }
                rows_.push_back(r);
            }
            if (rows_.size() < 10) return;
            double rmin = std::numeric_limits<double>::infinity(), rmax = -rmin;
            for (const auto &r : rows_) { rmin = std::min(rmin, r.yw); rmax = std::max(rmax, r.yw); }
            const double Lr = rmax - rmin;
            if (not (Lr > 1e-4)) return;
            // global information + priors (for the sweep-local solve)
            Eigen::Matrix<double, NG, NG> Hg = H_;  Eigen::Matrix<double, NG, 1> bg = b_;
            add_priors(Hg, bg);
            const Eigen::Matrix3d P = pose_cov.cast<double>().inverse();
            Eigen::Matrix<double, NG, 1> th = Hg.ldlt().solve(bg);
            Eigen::Vector3d eta = Eigen::Vector3d::Zero();
            double sig = sigma_w_;
            double pw = ceil ? 1.0 / 3.0 : 0.5, pc = ceil ? 1.0 / 3.0 : 0.0, po = ceil ? 1.0 / 3.0 : 0.5;
            Eigen::Matrix<double, NG, NG> Hs = Eigen::Matrix<double, NG, NG>::Zero(); Eigen::Matrix<double, NG, 1> bs = Eigen::Matrix<double, NG, 1>::Zero();
            const double l2pi = 0.5 * std::log(2.0 * M_PI);
            for (int it = 0; it <= p_.em_iters; ++it)
            {
                double sw = 0.0, sc = 0.0, srr = 0.0;
                for (auto &r : rows_)
                {
                    const double ew = r.yw - r.Jw.dot(th) - r.Jn.dot(eta);
                    const double lw_ = std::log(pw) - std::log(sig) - 0.5 * ew * ew / (sig * sig) - l2pi;
                    const double lo  = std::log(po) - std::log(Lr);
                    double ec = 0.0, lc = -std::numeric_limits<double>::infinity();
                    if (r.has_c)
                    {
                        ec = r.yc - r.Jc.dot(th);
                        lc = std::log(pc) - std::log(sig) - 0.5 * ec * ec / (sig * sig) - l2pi;
                    }
                    const double mx = std::max({lw_, lo, lc});
                    const double zw = std::exp(lw_ - mx), zo = std::exp(lo - mx), zc = r.has_c ? std::exp(lc - mx) : 0.0;
                    const double Z = zw + zo + zc;
                    r.rw = zw / Z; r.rc = zc / Z;
                    sw += r.rw; sc += r.rc; srr += r.rw * ew * ew + r.rc * ec * ec;
                }
                sig = std::sqrt((srr + p_.sigma_w_pseudo * p_.sigma_w0 * p_.sigma_w0) / (sw + sc + p_.sigma_w_pseudo));
                const double nr = double(rows_.size());
                pw = (sw + 1.0) / (nr + 3.0);  pc = ceil ? (sc + 1.0) / (nr + 3.0) : 0.0;  po = 1.0 - pw - pc;
                // sweep normal equations over (global | nuisance), nuisance Schur-marginalised with prior P
                Eigen::Matrix<double, NG, NG> A = Eigen::Matrix<double, NG, NG>::Zero();
                Eigen::Matrix<double, NG, NN> Cx = Eigen::Matrix<double, NG, NN>::Zero();
                Eigen::Matrix3d Dn = P;
                Eigen::Matrix<double, NG, 1> ba = Eigen::Matrix<double, NG, 1>::Zero();
                Eigen::Vector3d bn = Eigen::Vector3d::Zero();
                const double iv = 1.0 / (sig * sig);
                for (const auto &r : rows_)
                {
                    const double w = r.rw * iv;
                    A.noalias()  += w * r.Jw * r.Jw.transpose();
                    Cx.noalias() += w * r.Jw * r.Jn.transpose();
                    Dn.noalias() += w * r.Jn * r.Jn.transpose();
                    ba += (w * r.yw) * r.Jw;
                    bn += (w * r.yw) * r.Jn;
                    if (r.has_c)
                    {
                        const double wc = r.rc * iv;
                        A.noalias() += wc * r.Jc * r.Jc.transpose();
                        ba += (wc * r.yc) * r.Jc;
                    }
                }
                const Eigen::Matrix3d Di = Dn.inverse();
                Hs = A - Cx * Di * Cx.transpose();
                Hs = 0.5 * (Hs + Hs.transpose());
                bs = ba - Cx * (Di * bn);
                th  = (Hg + Hs).ldlt().solve(bg + bs);                  // sweep-local best estimate
                eta = Di * (bn - Cx.transpose() * th);                  // and this sweep's pose error
            }
            const double pi = pw + pc;
            const double lam = std::exp(-1.0 / p_.memory);
            cells_.add(place >= 0 ? place : -(++anon_), Hs, bs, lam);
            cells_.effective(H_, b_);
            last_Hs_ = Hs;  last_bs_ = bs;   // this sweep's own (nuisance-marginalised) evidence, for per-sweep logs
            share_ = pi; sigma_w_ = sig; ++n_;
        }

        /// (dx, dy, dz, droll, dpitch, dyaw) with the ceiling height marginalised (its prior included in the
        /// data part); the role's uninformed columns carry exactly their prior.
        [[nodiscard]] Info6 info() const
        {
            Eigen::Matrix<double, NG, NG> H = H_;  Eigen::Matrix<double, NG, 1> b = b_;
            H(6, 6) += 1.0 / (p_.sigma_hc * p_.sigma_hc);
            b(6)    += hint_ / (p_.sigma_hc * p_.sigma_hc);
            Info6 o;
            const double h66 = H(6, 6);
            o.H_data = H.topLeftCorner<6, 6>() - H.topRightCorner<6, 1>() * H.bottomLeftCorner<1, 6>() / h66;
            o.H_data = 0.5 * (o.H_data + o.H_data.transpose());
            o.b_data = b.head<6>() - H.topRightCorner<6, 1>() * (b(6) / h66);
            o.H_prior.diagonal() = p_.prior_sigma.cwiseAbs2().cwiseInverse();
            o.n = n_;
            return o;
        }
        [[nodiscard]] double wall_share() const noexcept { return share_; }
        [[nodiscard]] double sigma_wall() const noexcept { return sigma_w_; }
        /// The LAST sweep's own evidence on (dx, dy, dyaw) with a weak prior (0.1 m, 5 deg): its planar reading and
        /// sigma, for the per-sweep diagnosis log (2026-10-08: bpearl dy -22 mm in a null run, cause unknown).
        [[nodiscard]] bool last_sweep_planar(Eigen::Vector3d &mean, Eigen::Vector3d &sigma) const
        {
            constexpr int ix[3] = {0, 1, 5};
            Eigen::Matrix3d H; Eigen::Vector3d b;
            for (int r = 0; r < 3; ++r) { b(r) = last_bs_(ix[r]); for (int c = 0; c < 3; ++c) H(r, c) = last_Hs_(ix[r], ix[c]); }
            H += Eigen::Vector3d(1.0 / 0.01, 1.0 / 0.01, 1.0 / (0.0873 * 0.0873)).asDiagonal();
            const Eigen::LDLT<Eigen::Matrix3d> l(H);
            if (l.info() != Eigen::Success) return false;
            mean = l.solve(b);  sigma = l.solve(Eigen::Matrix3d::Identity()).diagonal().cwiseSqrt();
            return mean.allFinite();
        }
        [[nodiscard]] long sweeps() const noexcept { return n_; }

    private:
        struct Row
        {
            Eigen::Matrix<double, NG, 1> Jw = Eigen::Matrix<double, NG, 1>::Zero(), Jc = Eigen::Matrix<double, NG, 1>::Zero();
            Eigen::Vector3d Jn = Eigen::Vector3d::Zero();
            double yw = 0.0, yc = 0.0, rw = 0.0, rc = 0.0;
            bool has_c = false;
        };
        void add_priors(Eigen::Matrix<double, NG, NG> &H, Eigen::Matrix<double, NG, 1> &b) const
        {
            for (int k = 0; k < 6; ++k) H(k, k) += 1.0 / (p_.prior_sigma[k] * p_.prior_sigma[k]);
            H(6, 6) += 1.0 / (p_.sigma_hc * p_.sigma_hc);
            b(6)    += hint_ / (p_.sigma_hc * p_.sigma_hc);
        }
        Params p_{};
        Eigen::Matrix<double, NG, NG> H_ = Eigen::Matrix<double, NG, NG>::Zero();
        Eigen::Matrix<double, NG, NG> last_Hs_ = Eigen::Matrix<double, NG, NG>::Zero();
        Eigen::Matrix<double, NG, 1>  last_bs_ = Eigen::Matrix<double, NG, 1>::Zero();
        PlaceCells<NG> cells_;
        std::int64_t anon_ = 0;
        Eigen::Matrix<double, NG, 1>  b_ = Eigen::Matrix<double, NG, 1>::Zero();
        std::vector<Row> rows_;            ///< scratch, reused across sweeps
        long n_ = 0;
        double share_ = 0.5, sigma_w_ = 0.02, hint_ = 0.0;
    };
}
