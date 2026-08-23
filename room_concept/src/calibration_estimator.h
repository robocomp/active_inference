/*  calibration_estimator.h — joint batch estimation of the robot's motion-model parameters.
 *
 *  WHAT IT IS
 *  ----------
 *  A sliding window of EPISODES (one open-loop prediction ramp plus the correction that ended it) is
 *  solved jointly for the parameters of the motion model. The localiser's own correction is the
 *  measurement, so this needs no ground truth and no calibration rig, and behaves identically on
 *  hardware and in simulation.
 *
 *  WHY JOINT, AND NOT ONE FILTER PER PARAMETER
 *  -------------------------------------------
 *  The regressors are not orthogonal in real driving. Estimating them independently is what made the
 *  forward scale oscillate around 1.0 for an entire session: every turn cancelled what the straights
 *  taught, because a turn's pose correction carries a component the translation parameters cannot
 *  explain and an independent Jacobian hands it to them anyway. That was patched by inflating R with
 *  |dtheta| — a symptom fix. A joint solve handles it by construction.
 *
 *  It is also the ONLY way some parameters can be separated at all. eps_yaw and a sensor lever arm
 *  produce the SAME component of the correction and differ only in which covariate they scale with
 *  (distance vs rotation); so do the gyro's scale and its bias (rotation vs elapsed time). No
 *  independent scalar filter can tell those apart.
 *
 *  WHAT "GOOD" MEANS — AND WHY IT IS NOT A THRESHOLD
 *  ------------------------------------------------
 *  Two criteria fall out of the solve itself: the posterior sigma has SHRUNK against the prior
 *  (information was gained), and the information matrix is well-conditioned for that parameter (it is
 *  observable in THIS data, not merely in principle). Reported per parameter, because a run supplies
 *  some covariates and not others: measured live, a turn-heavy run leaves eps_yaw untaught with a
 *  WIDENING sigma while a straight-heavy run does the same to the gyro scale. ★A parameter's VALUE
 *  cannot distinguish "converged" from "never asked"; only its precision can.
 *
 *  THE COLD-START TRAP
 *  -------------------
 *  A window with no motion has near-zero residual, which reads as "my measurements are perfect" and
 *  makes every future manoeuvre look pointless. afford_calib already paid for this: before its fix a
 *  brand-new robot advertised the SMALLEST information gain of its life (0.03 nats) exactly when it
 *  had most to learn, rising to 3.46 once a conjugate prior at a MEASURED density was added.
 *  ★Zero motion is not evidence of a quiet sensor. The prior here plays the same role and is not
 *  optional.
 */
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <string_view>

namespace rc::calib
{
    /// The parameters, in solve order. Extend at the END so stored covariances stay comparable.
    enum Param : int
    {
        P_K_V = 0,      ///< translation odometry scale (fractional, 0 = correct)
        P_EPS_YAW,      ///< body/mount yaw offset (rad)
        P_K_OMEGA,      ///< gyro scale (fractional)
        P_B_OMEGA,      ///< gyro bias (rad/s) — separated from scale ONLY by time-vs-rotation
        P_COUNT
    };

    [[nodiscard]] constexpr std::string_view param_name(int p)
    {
        switch (p)
        {
            case P_K_V:     return "k_v";
            case P_EPS_YAW: return "eps_yaw";
            case P_K_OMEGA: return "k_omega";
            case P_B_OMEGA: return "b_omega";
            default:        return "?";
        }
    }

    /// One ramp-plus-correction. All quantities are accumulated over the whole episode.
    struct Episode
    {
        // Covariates: the motion that would reveal each parameter.
        float d_forward = 0.f;   ///< m, body +Y, as the model believed it
        float d_lateral = 0.f;   ///< m, body +X
        float d_theta   = 0.f;   ///< rad, as the model believed it
        float duration  = 0.f;   ///< s — the covariate that separates a gyro BIAS from a scale

        // The measurement: what the optimizer had to add, in the ROBOT frame.
        float r_forward = 0.f;   ///< m
        float r_lateral = 0.f;   ///< m
        float r_theta   = 0.f;   ///< rad

        // How much this episode should be believed.
        float pos_var   = 1e-4f; ///< m^2, localiser posterior + model-error terms
        float theta_var = 1e-4f; ///< rad^2
    };

    struct Result
    {
        Eigen::Matrix<float, P_COUNT, 1> value = Eigen::Matrix<float, P_COUNT, 1>::Zero();
        Eigen::Matrix<float, P_COUNT, 1> sigma = Eigen::Matrix<float, P_COUNT, 1>::Zero();
        /// True where the posterior actually shrank against the prior — i.e. this window TAUGHT
        /// something about that parameter, as opposed to leaving it where the prior put it.
        Eigen::Matrix<bool, P_COUNT, 1> informed = Eigen::Matrix<bool, P_COUNT, 1>::Zero();
        int   episodes = 0;
        /// Condition number of the CORRELATION-NORMALISED information matrix, D^-1/2 H D^-1/2.
        /// Normalising is not cosmetic: on the raw H the parameters have wildly different units
        /// (a dimensionless scale ~1e-2 against a bias in rad/s ~1e-4), so its eigenvalues span
        /// orders of magnitude whatever the geometry, and the number says nothing about
        /// collinearity. Measured on synthetic data: a SEPARABLE window scored 157145 and a
        /// deliberately COLLINEAR one 118002 -- i.e. the raw condition number ranked them backwards.
        /// Normalised, it measures what it is meant to: how nearly two columns point the same way.
        float condition = 0.f;
        bool  ok = false;
    };

    /// Prior: mean 0 (the model is believed correct until shown otherwise) with a stated sigma.
    /// ★NOT optional — see THE COLD-START TRAP. It is also what keeps the solve well-posed when a
    /// window happens to contain no motion of the kind a given parameter needs.
    struct Prior
    {
        float sigma_k_v     = 0.02f;    ///< 2% — a wheel radius is not wrong by more than this
        float sigma_eps_yaw = 0.0175f;  ///< 1 deg of mount/axis misalignment
        float sigma_k_omega = 0.02f;    ///< 2%
        float sigma_b_omega = 5.0e-4f;  ///< rad/s ~ 0.03 deg/s, a plausible post-calibration residual
    };

    class BatchEstimator
    {
    public:
        void configure(const Prior &p, std::size_t window) { prior_ = p; window_ = std::max<std::size_t>(window, 8); }

        void add(const Episode &e)
        {
            eps_.push_back(e);
            while (eps_.size() > window_) eps_.pop_front();
        }

        [[nodiscard]] std::size_t size() const noexcept { return eps_.size(); }
        void clear() noexcept { eps_.clear(); }

        /// Solve the window. p = (J'WJ + P0^-1)^-1 J'W r, and that same inverse IS the covariance.
        [[nodiscard]] Result solve() const
        {
            Result out;
            out.episodes = static_cast<int>(eps_.size());
            if (eps_.size() < 4) return out;

            Eigen::Matrix<float, P_COUNT, P_COUNT> H = Eigen::Matrix<float, P_COUNT, P_COUNT>::Zero();
            Eigen::Matrix<float, P_COUNT, 1>       b = Eigen::Matrix<float, P_COUNT, 1>::Zero();

            // Prior information. Mean 0, so it contributes to H only.
            Eigen::Matrix<float, P_COUNT, 1> p0;
            p0 << prior_.sigma_k_v, prior_.sigma_eps_yaw, prior_.sigma_k_omega, prior_.sigma_b_omega;
            for (int i = 0; i < P_COUNT; ++i)
                H(i, i) += 1.f / std::max(p0[i] * p0[i], 1e-18f);
            const Eigen::Matrix<float, P_COUNT, P_COUNT> H_prior = H;

            for (const auto &e : eps_)
            {
                const float wp = 1.f / std::max(e.pos_var, 1e-12f);
                const float wt = 1.f / std::max(e.theta_var, 1e-12f);

                // ALONG-track row: only the translation scale moves the robot along its own heading.
                Eigen::Matrix<float, P_COUNT, 1> j_along = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_along[P_K_V] = e.d_forward;
                H += wp * j_along * j_along.transpose();
                b += wp * j_along * e.r_forward;

                // CROSS-track row: a yaw offset rotates forward travel off-axis. The sign follows the
                // integrator's body->world mapping, where +eps moves the step by -d_forward laterally.
                Eigen::Matrix<float, P_COUNT, 1> j_cross = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_cross[P_EPS_YAW] = -e.d_forward;
                H += wp * j_cross * j_cross.transpose();
                b += wp * j_cross * e.r_lateral;

                // HEADING row: scale scales the ROTATION, bias accumulates with TIME. Both land on
                // the same component, and this pair of covariates is the only thing separating them —
                // which is why they must be solved together and not as two filters.
                Eigen::Matrix<float, P_COUNT, 1> j_th = Eigen::Matrix<float, P_COUNT, 1>::Zero();
                j_th[P_K_OMEGA] = e.d_theta;
                j_th[P_B_OMEGA] = e.duration;
                H += wt * j_th * j_th.transpose();
                b += wt * j_th * e.r_theta;
            }

            const Eigen::LDLT<Eigen::Matrix<float, P_COUNT, P_COUNT>> ldlt(H);
            if (ldlt.info() != Eigen::Success) return out;
            out.value = ldlt.solve(b);
            const Eigen::Matrix<float, P_COUNT, P_COUNT> cov = H.inverse();
            if (not cov.allFinite()) return out;

            for (int i = 0; i < P_COUNT; ++i)
            {
                out.sigma[i] = std::sqrt(std::max(cov(i, i), 0.f));
                // "Informed" = this window actually shrank the posterior for THIS parameter. A
                // parameter the driving never excited comes back at its prior sigma and is reported
                // as uninformed rather than as a confident zero.
                out.informed[i] = out.sigma[i] < 0.9f * (1.f / std::sqrt(H_prior(i, i)));
            }
            // Normalise to correlation form before asking about conditioning -- see Result::condition.
            Eigen::Matrix<float, P_COUNT, 1> d;
            for (int i = 0; i < P_COUNT; ++i) d[i] = 1.f / std::sqrt(std::max(H(i, i), 1e-30f));
            const Eigen::Matrix<float, P_COUNT, P_COUNT> Hn = d.asDiagonal() * H * d.asDiagonal();
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, P_COUNT, P_COUNT>> es(Hn);
            const auto ev = es.eigenvalues();
            out.condition = ev.minCoeff() > 0.f ? ev.maxCoeff() / ev.minCoeff()
                                                : std::numeric_limits<float>::infinity();
            out.ok = out.value.allFinite();
            return out;
        }

    private:
        Prior prior_{};
        std::size_t window_ = 64;
        std::deque<Episode> eps_;
    };
}
