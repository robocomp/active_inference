/*  motion_calibration.h — slow, online correction of the motion model's own parameters.
 *
 *  WHY THIS EXISTS
 *  ---------------
 *  The localiser predicts with odometry and corrects with the SDF. Between corrections the
 *  prediction runs open-loop, so any systematic error in the odometry model accumulates until the
 *  early-exit gate trips and the optimizer wipes it. That is the sawtooth: the correction is never
 *  wrong, it is just always late, and it removes the SAME error again every few seconds without ever
 *  learning anything from it. Measured on P3Bot 2026-08-22: the predictor over-rotates ~4% and
 *  over-runs forward ~1.2%, while the CORRECTED output tracks ground truth to <1% — which is why no
 *  amount of looking at the published pose ever revealed it.
 *
 *  A precision (covariance) term cannot fix this. Down-weighting a biased channel widens the
 *  posterior and lengthens the tooth; the mean still walks. What is needed is the LEARNING tier:
 *  slow parameters that change what the model predicts, with precision governing how fast they are
 *  learned. Both fall out of the same free-energy gradient, which is why they live together here.
 *
 *  WHAT IS LEARNED, AND WHY IT IS IDENTIFIABLE
 *  -------------------------------------------
 *  The optimizer hands us a free measurement every time it fires: r = corrected_pose -
 *  predicted_pose, the exact error that accumulated over the ramp. Decomposed in the robot's frame
 *  against the motion that produced it, three parameters load on DIFFERENT components against
 *  DIFFERENT covariates, so they separate instead of blending into one fudge factor:
 *
 *      r_forward  ~=  dk_v     * d_forward     forward odometry scale
 *      r_lateral  ~=  -eps_yaw * d_forward     body/mount yaw offset (rotates travel off-axis)
 *      r_theta    ~=  dk_omega * d_theta       gyro scale
 *
 *  FRAME NOTE, measured not assumed: integrate_odometry_over_window maps body-forward (dy_local)
 *  into the world as (-dy*sin(th), +dy*cos(th)), so the robot's forward axis is th + 90 deg. Verified
 *  live: travel direction minus est_theta = +90 deg on 100% of 725 straight-driving samples across
 *  two runs. Get this backwards and the forward-scale error is read as a yaw offset — it happened.
 *
 *  NO THRESHOLDS
 *  -------------
 *  There is deliberately no "is this episode informative enough" test. The Kalman gain already is
 *  one: K = P*H / (H*P*H + R) goes to zero as H (the motion that would reveal the parameter) goes to
 *  zero, so a parked robot teaches nothing without anyone having to say so. Likewise the learning
 *  rate is not a constant — it is the ratio of parameter precision to innovation precision, and R
 *  comes from the optimizer's own posterior covariance, so a low-confidence correction teaches
 *  little and a crisp one teaches a lot. That IS the precision weighting, doing the job it should.
 *
 *  MARGINALIZATION: these parameters are deliberately NOT part of the sliding-window state and must
 *  never be handed to marginalize_oldest(). A parameter spans the whole window and beyond; freezing
 *  its Jacobian at a stale value is the mechanism that ratcheted loss_boundary 0->559 once already.
 *  See se2_preintegration.h "SCALE-AS-A-STATE".
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace rc::calib
{
    /// One scalar parameter tracked by a random-walk Kalman filter.
    ///
    /// The random walk is the point: a robot's calibration is not a constant to be measured once but
    /// a slowly drifting property (tyre wear, a knocked mount, a re-seated sensor). Q lets the
    /// estimate follow that drift; without it P collapses and the parameter freezes at whatever the
    /// first few minutes happened to say.
    class ScalarParam
    {
    public:
        void init(float value, float p0, float q) noexcept
        { x_ = value; p_ = std::max(p0, 0.f); q_ = std::max(q, 0.f); }

        [[nodiscard]] float value() const noexcept { return x_; }
        [[nodiscard]] float sigma() const noexcept { return std::sqrt(std::max(p_, 0.f)); }
        [[nodiscard]] int   updates() const noexcept { return n_; }

        /// Fold in one RESIDUAL observation: `resid` is the error that REMAINS after the current
        /// estimate was already applied to the model, and `h` is d(resid)/dx.
        ///
        /// The residual form matters. The textbook update subtracts a predicted measurement
        /// (innov = z - h*x), which assumes z was measured with the parameter NOT applied. Here the
        /// parameter is always applied — it is baked into the prediction the optimizer corrected — so
        /// the correction we observe is already the leftover. Using the textbook form would drive x
        /// back toward zero exactly as it started working: the better the calibration got, the
        /// smaller the correction, and the filter would read that success as evidence x should shrink.
        float update(float h, float resid, float r) noexcept
        {
            p_ += q_;                                  // random walk first
            const float s = h * p_ * h + std::max(r, 1e-12f);
            if (not std::isfinite(s) or s <= 0.f or not std::isfinite(resid)) return 0.f;
            const float k = p_ * h / s;                // -> 0 when h -> 0. No threshold needed.
            x_ += k * resid;
            p_ = std::max((1.f - k * h) * p_, 0.f);
            ++n_;
            return resid / std::sqrt(s);
        }

    private:
        float x_ = 0.f, p_ = 0.f, q_ = 0.f;
        int   n_ = 0;
    };

    struct Config
    {
        bool  enabled      = false;
        float yaw_p0       = 1.0e-4f;   // (rad)^2   -> 1 sigma ~ 0.57 deg
        float yaw_q        = 1.0e-9f;   // (rad)^2 per update
        float scale_p0     = 4.0e-4f;   // fractional^2 -> 1 sigma ~ 2%
        float scale_q      = 1.0e-9f;
        float min_obs_var  = 1.0e-6f;   // floor on R so a zero-covariance frame cannot dominate
        // Model error, not sensor error: during a turn the pose correction contains a component the
        // translation parameters cannot explain, and handing it to them anyway is what stops k_v
        // converging. Rather than gate turns out (a threshold), let the observation covariance GROW
        // with the covariate that predicts the model error, so a turning episode is believed less by
        // exactly as much as it deserves. Measured on 29 live episodes: forward scale estimated from
        // rotation-poor episodes is -1.01% (per-episode -1.49/-0.85/-1.09) and from rotation-heavy
        // ones -0.20% with wild scatter (+4.57/+2.21/-1.02) -- the turns were cancelling the straights
        // and k_v oscillated around 1.0 instead of converging. Order-of-magnitude value: it only sets
        // the RELATIVE weight of turning against straight episodes.
        float rot_model_sigma = 0.030f; // m per rad of turning, added in quadrature to the position R
    };

    /// Accumulates one ramp-plus-correction episode and folds it into the parameters.
    class MotionCalibrator
    {
    public:
        void configure(const Config &c)
        {
            cfg_ = c;
            yaw_.init(0.f, c.yaw_p0, c.yaw_q);
            k_v_.init(0.f, c.scale_p0, c.scale_q);      // stored as a DELTA from 1.0
            k_w_.init(0.f, c.scale_p0, c.scale_q);
            configured_ = true;
        }

        [[nodiscard]] bool  configured() const noexcept { return configured_; }
        [[nodiscard]] bool  enabled() const noexcept { return cfg_.enabled and configured_; }
        /// Applied to the body->world rotation of the odometry displacement.
        [[nodiscard]] float yaw_offset() const noexcept { return enabled() ? yaw_.value() : 0.f; }
        /// Multiplies forward (and lateral) wheel displacement.
        [[nodiscard]] float forward_scale() const noexcept { return enabled() ? 1.f + k_v_.value() : 1.f; }
        /// Multiplies the heading increment, whichever channel produced it.
        [[nodiscard]] float omega_scale() const noexcept { return enabled() ? 1.f + k_w_.value() : 1.f; }

        [[nodiscard]] float yaw_sigma() const noexcept { return yaw_.sigma(); }
        [[nodiscard]] float k_v_sigma() const noexcept { return k_v_.sigma(); }
        [[nodiscard]] float k_w_sigma() const noexcept { return k_w_.sigma(); }
        [[nodiscard]] int   episodes() const noexcept { return episodes_; }

        /// Called once per localiser cycle, on BOTH the early-exit and the optimized path.
        ///   d_forward/d_lateral/d_theta : what the motion model predicted this cycle (post-scaling)
        ///   r_forward/r_lateral/r_theta : corrected minus predicted, in the ROBOT frame (0 on early exit)
        ///   pos_var/theta_var           : the optimizer's posterior variance, i.e. how much to believe r
        ///   corrected                   : did the optimizer run this cycle
        void observe(float d_forward, float d_lateral, float d_theta,
                     float r_forward, float r_lateral, float r_theta,
                     float pos_var, float theta_var, bool corrected) noexcept
        {
            if (not enabled()) return;
            const bool finite = std::isfinite(d_forward) and std::isfinite(d_theta)
                            and std::isfinite(r_forward) and std::isfinite(r_lateral)
                            and std::isfinite(r_theta);
            if (not finite) { reset_episode(); prev_corrected_ = corrected; return; }

            acc_fwd_ += d_forward; acc_lat_ += d_lateral; acc_th_ += d_theta;
            if (corrected)
            {
                acc_r_fwd_ += r_forward; acc_r_lat_ += r_lateral; acc_r_th_ += r_theta;
                acc_pos_var_ = std::max(acc_pos_var_, pos_var);
                acc_th_var_  = std::max(acc_th_var_,  theta_var);
            }

            // An episode is one ramp plus the correction that ended it. Flush on the falling edge:
            // that is the moment the whole accumulated error has been observed exactly once.
            if (prev_corrected_ and not corrected) flush();
            prev_corrected_ = corrected;

            // Numerical guard only (not a model threshold): a robot that drives for ever without a
            // single correction would otherwise accumulate unboundedly.
            if (std::abs(acc_fwd_) > 1e4f or std::abs(acc_th_) > 1e4f) reset_episode();
        }

    private:
        void flush() noexcept
        {
            const float rot_model = cfg_.rot_model_sigma * std::abs(acc_th_);
            const float r_pos = std::max(acc_pos_var_, cfg_.min_obs_var) + rot_model * rot_model;
            const float r_th  = std::max(acc_th_var_,  cfg_.min_obs_var);
            // Forward scale and yaw offset are BOTH driven by forward travel but read off orthogonal
            // components of the same correction, which is what makes them separable rather than a
            // single blended gain. The gain vanishes on its own when acc_fwd_ is ~0.
            k_v_.update(acc_fwd_,  acc_r_fwd_, r_pos);
            yaw_.update(-acc_fwd_, acc_r_lat_, r_pos);
            k_w_.update(acc_th_,   acc_r_th_,  r_th);
            ++episodes_;
            reset_episode();
        }
        void reset_episode() noexcept
        {
            acc_fwd_ = acc_lat_ = acc_th_ = 0.f;
            acc_r_fwd_ = acc_r_lat_ = acc_r_th_ = 0.f;
            acc_pos_var_ = acc_th_var_ = 0.f;
        }

        Config cfg_{};
        bool configured_ = false, prev_corrected_ = false;
        ScalarParam yaw_, k_v_, k_w_;
        float acc_fwd_ = 0.f, acc_lat_ = 0.f, acc_th_ = 0.f;
        float acc_r_fwd_ = 0.f, acc_r_lat_ = 0.f, acc_r_th_ = 0.f;
        float acc_pos_var_ = 0.f, acc_th_var_ = 0.f;
        int episodes_ = 0;
    };
}
