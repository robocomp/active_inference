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

#include "calibration_intake.h"

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
        // A correction can only be as good as the FIT that produced it. When the localiser is not
        // tracking, the optimizer fires on nearly every cycle and the "episodes" that reach this
        // filter are 1-2 cycle fragments whose corrections are recovery, not model error. Observed
        // 2026-08-23: two 180 s windows produced 497 such episodes and dragged a healthy k_v from
        // 1.0059 to 0.8907 and yaw to -1.78 deg, wiping 20 minutes of correct estimation.
        // Rather than detect the regime and gate it out, let R carry the localiser's own residual:
        // the SDF error during the episode is exactly the statement "this fit is untrustworthy", and
        // during those windows it ran 0.05-0.26 against a normal 0.027, so the gain collapses on its
        // own. Same rule as rot_model_sigma -- grow the covariance with the covariate that predicts
        // the model error, never a threshold on it.
        float fit_model_gain  = 2.0f;   // multiplies mean |SDF| over the episode, into the position R
    };

    /// Accumulates one ramp-plus-correction episode and folds it into the parameters.
    class MotionCalibrator
    {
    public:
        void configure(const Config &c)
        {
            cfg_ = c;
            rc::calib::Prior pr;
            pr.sigma_k_v     = std::sqrt(std::max(c.scale_p0, 1e-12f));
            pr.sigma_k_omega = std::sqrt(std::max(c.scale_p0, 1e-12f));
            pr.sigma_eps_yaw = std::sqrt(std::max(c.yaw_p0,   1e-12f));
            // The two newest channels keep the estimator's own defaults: they have never been
            // measured on this robot, and inventing a config knob for a prior nobody has data for
            // would dress an assumption up as a setting.

            // A LONG window is what replaces the prior re-centring: it must hold enough driving to
            // contain each parameter's covariate at least sometimes. 512 episodes is roughly the
            // last half hour at the observed ~0.3 Hz, so a stretch of pure turning no longer erases
            // what the straights before it taught.
            rc::calib::IntakeParams ip;
            ip.window = 512;
            intake_.configure(ip, pr);
            configured_ = true;
        }

        [[nodiscard]] bool  configured() const noexcept { return configured_; }
        [[nodiscard]] bool  enabled() const noexcept { return cfg_.enabled and configured_; }
        /// Applied to the body->world rotation of the odometry displacement.
        [[nodiscard]] float yaw_offset() const noexcept
        { return enabled() ? last_.value[rc::calib::P_EPS_YAW] : 0.f; }
        /// Multiplies forward (and lateral) wheel displacement.
        [[nodiscard]] float forward_scale() const noexcept
        { return enabled() ? 1.f + last_.value[rc::calib::P_K_V] : 1.f; }
        /// Multiplies the heading increment, whichever channel produced it.
        [[nodiscard]] float omega_scale() const noexcept
        { return enabled() ? 1.f + last_.value[rc::calib::P_K_OMEGA] : 1.f; }
        /// rad/s, subtracted from the measured rate. Separated from the SCALE only by the
        /// time-vs-rotation covariate pair, which is why it needs the joint solve to exist at all.
        [[nodiscard]] float omega_bias() const noexcept
        { return enabled() ? last_.value[rc::calib::P_B_OMEGA] : 0.f; }
        /// Multiplies LATERAL wheel displacement. Separate from forward_scale(): a mecanum's lateral
        /// channel is the one roller slip corrupts, so the two are physically different errors.
        [[nodiscard]] float lateral_scale() const noexcept
        { return enabled() ? 1.f + last_.value[rc::calib::P_K_LAT] : 1.f; }
        /// rad per metre driven forward: unequal effective wheel radii make a commanded straight
        /// line curve. Added to the heading increment in proportion to DISTANCE, which is what
        /// distinguishes it from a gyro scale (rotation) or a gyro bias (time).
        [[nodiscard]] float wheel_mismatch() const noexcept
        { return enabled() ? last_.value[rc::calib::P_DK_WHEEL] : 0.f; }

        [[nodiscard]] float yaw_sigma() const noexcept { return last_.sigma[rc::calib::P_EPS_YAW]; }
        [[nodiscard]] float k_v_sigma() const noexcept { return last_.sigma[rc::calib::P_K_V]; }
        [[nodiscard]] float k_w_sigma() const noexcept { return last_.sigma[rc::calib::P_K_OMEGA]; }
        [[nodiscard]] const rc::calib::Result& last_solve() const noexcept { return last_; }
        [[nodiscard]] const rc::calib::CalibrationIntake& intake() const noexcept { return intake_; }
        /// Tag episodes that follow as coming from a deliberate manoeuvre. Reporting only -- it
        /// cannot change how an episode is weighted, and must not: see calibration_intake.h.
        void set_source(rc::calib::Source s) noexcept { source_hint_ = s; }
        [[nodiscard]] int   episodes() const noexcept { return episodes_; }

        /// Called once per localiser cycle, on BOTH the early-exit and the optimized path.
        ///   d_forward/d_lateral/d_theta : what the motion model predicted this cycle (post-scaling)
        ///   r_forward/r_lateral/r_theta : corrected minus predicted, in the ROBOT frame (0 on early exit)
        ///   pos_var/theta_var           : the optimizer's posterior variance, i.e. how much to believe r
        ///   corrected                   : did the optimizer run this cycle
        void observe(float d_forward, float d_lateral, float d_theta,
                     float r_forward, float r_lateral, float r_theta,
                     float pos_var, float theta_var, bool corrected,
                     float fit_residual, float dt) noexcept
        {
            if (not enabled()) return;
            const bool finite = std::isfinite(d_forward) and std::isfinite(d_theta)
                            and std::isfinite(r_forward) and std::isfinite(r_lateral)
                            and std::isfinite(r_theta);
            if (not finite) { reset_episode(); prev_corrected_ = corrected; return; }

            acc_fwd_ += d_forward; acc_lat_ += d_lateral; acc_th_ += d_theta;
            // Elapsed time: the covariate that separates a gyro BIAS from a gyro SCALE. Nothing else
            // in the episode carries it, and without it the two are collinear whenever the robot
            // turns at a steady rate -- which is most of the time.
            if (std::isfinite(dt) and dt > 0.f) acc_dur_ += dt;
            // Worst fit seen in the episode, not the mean: one bad frame is enough to make the whole
            // accumulated correction untrustworthy, and averaging would let a long clean ramp hide it.
            if (std::isfinite(fit_residual)) acc_fit_ = std::max(acc_fit_, fit_residual);
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
            const float fit_model = cfg_.fit_model_gain * acc_fit_;
            const float r_pos = std::max(acc_pos_var_, cfg_.min_obs_var)
                              + rot_model * rot_model + fit_model * fit_model;
            const float r_th  = std::max(acc_th_var_,  cfg_.min_obs_var) + fit_model * fit_model;
            // ── ONE ESTIMATOR, NOT THREE ──────────────────────────────────────────────────────────
            // These used to be three independent scalar Kalman filters. Independence cannot separate
            // parameters that land on the SAME component of the correction and differ only in
            // covariate -- gyro scale from gyro bias (rotation vs elapsed time), or a mount yaw from
            // a sensor lever arm (distance vs rotation) -- and it is what made k_v oscillate around
            // 1.0 for a whole session, every turn cancelling what the straights taught. The joint
            // solve handles that by construction. The scalar path is REPLACED, not kept alongside:
            // two estimators that can disagree is the shape of bug this work spent a day chasing.
            rc::calib::Episode e;
            e.d_forward = acc_fwd_;   e.d_lateral = acc_lat_;   e.d_theta = acc_th_;
            e.duration  = acc_dur_;
            e.r_forward = acc_r_fwd_; e.r_lateral = acc_r_lat_; e.r_theta = acc_r_th_;
            e.pos_var = r_pos;        e.theta_var = r_th;
            ++episodes_;

            // OFFER it -- do not assume it will be taken. The intake owns the admission policy and
            // the estimator; this class only BUILDS episodes from whatever the robot happened to do.
            // Source is Passive because this path is ordinary driving; a manoeuvre offers through the
            // same door with a different tag, which changes the reporting and nothing else.
            last_verdict_ = intake_.offer(e, source_hint_, acc_fit_);
            last_ = intake_.estimate();
            reset_episode();
        }
        void reset_episode() noexcept
        {
            acc_fwd_ = acc_lat_ = acc_th_ = 0.f;
            acc_r_fwd_ = acc_r_lat_ = acc_r_th_ = 0.f;
            acc_pos_var_ = acc_th_var_ = acc_fit_ = acc_dur_ = 0.f;
        }

        Config cfg_{};
        bool configured_ = false, prev_corrected_ = false;
        rc::calib::CalibrationIntake intake_;
        rc::calib::Result last_{};
        rc::calib::Verdict last_verdict_ = rc::calib::Verdict::Accepted;
        rc::calib::Source  source_hint_  = rc::calib::Source::Passive;
        float acc_fwd_ = 0.f, acc_lat_ = 0.f, acc_th_ = 0.f;
        float acc_r_fwd_ = 0.f, acc_r_lat_ = 0.f, acc_r_th_ = 0.f;
        float acc_pos_var_ = 0.f, acc_th_var_ = 0.f, acc_fit_ = 0.f, acc_dur_ = 0.f;
        int episodes_ = 0;
    };
}
