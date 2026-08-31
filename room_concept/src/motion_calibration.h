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
#include <filesystem>
#include <system_error>

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
        /// ── OPEN-LOOP MODE: ESTIMATE, BUT DO NOT APPLY ───────────────────────────────────────────
        /// false severs the estimator's output from the odometry while leaving everything else
        /// running: episodes still close, the window still fills, the solve still runs, and the
        /// value and sigma are still logged (via the estimated_* accessors, which ignore this flag).
        ///
        /// ★ THIS IS AN INSTRUMENT, NOT A TUNING KNOB. The estimator's covariate is the POST-scaled
        /// odometry (room_concept.cpp: dy_local = odom.adv * dt * k_v) and an Episode records no
        /// trace of the parameter values that were active when it was written. So once a value is
        /// applied, later episodes report only the error REMAINING after it acted, while the batch
        /// re-solves the window from scratch as though those leftovers were totals. Fixed point:
        /// v = k* S/(2S + P0), i.e. recovery is capped at 50% however much data arrives.
        /// Measured 2026-08-30 on a +3% injected forward-scale error: the unweighted, prior-free
        /// slope of the saved window was 49.8% of the true deviation -- the ceiling, almost exactly.
        /// Setting apply=false breaks that loop: the covariate stays raw and the residual stays the
        /// FULL error, so recovery should jump toward 100%. That prediction is the whole point of
        /// the flag, and it discriminates this account from any rival that does not involve
        /// feedback. Leave it TRUE for normal operation.
        bool  apply        = true;
        /// WHICH parameters may act, as a bitmask over rc::calib::Param (bit 0 = k_v, 1 = eps_yaw,
        /// 2 = k_omega, 3 = b_omega, 4 = k_lat, 5 = dk_wheel). -1 = all, which is normal operation.
        ///
        /// ★ It exists to separate the CHANNELS, not to tune them. Measured 2026-08-31, arm 3: with
        /// everything applied, translation drift was unchanged (39.59 -> 38.98 mm/m, d = 0.04) while
        /// heading drift got WORSE (1.699 -> 2.955 deg/rad, t = -2.45, d = -1.00) and the localiser
        /// lost tracking in 6 windows against 0. The forward scale is the well-conditioned parameter;
        /// the heading three share one component and are separated only by covariate, and they are
        /// the ones whose values move between runs. Applying a well-estimated k_v with a
        /// poorly-estimated heading correction would produce exactly that split, and a mask is how
        /// you ask. 1 = k_v only.
        ///
        /// ⚠ p_applied records what ACTED, so a masked-out parameter is correctly recorded as not
        /// applied and the feedback undo stays exact. Masking changes what the robot DOES, never
        /// what the estimator learns -- every parameter keeps being estimated and logged.
        int   apply_mask   = -1;
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
        /// How much motion closes an episode. NOT sensitivity knobs: below these the Jacobian rows
        /// are ~0 and the episode cannot identify anything, so closing one would add a row that only
        /// dilutes the window. A parked robot must never close an episode.
        float episode_min_trans = 0.25f;  ///< m of accumulated forward travel
        float episode_min_rot   = 0.20f;  ///< rad of accumulated rotation (~11 deg)
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
        // ── HOW FAR AN UNMEASURED EPISODE MAY BE CARRIED ─────────────────────────────────────────
        // An episode that saw no correction is not flushed (see observe): its motion is carried
        // forward so the covariate survives until a real correction arrives to explain it. These cap
        // that carry. They are a LINEARISATION guard, not a sensitivity knob: the episode's Jacobian
        // rows treat the accumulated motion as one small increment, and that stops being true long
        // before the old 1e4 guard they replace. Ten times the identifiability triggers.
        // ★ Hitting one means the localiser has not corrected the pose across several metres, which
        // is itself worth knowing -- dropped_ counts it rather than letting it pass silently.
        float episode_carry_max_trans = 2.5f;   ///< m
        float episode_carry_max_rot   = 2.0f;   ///< rad
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

        /// ── A PARAMETER IS APPLIED ONLY ONCE IT HAS BEEN TAUGHT ──────────────────────────────────
        /// `informed` means the posterior sigma shrank below 0.9x the prior's: this window actually
        /// measured something, as opposed to leaving the parameter where the prior put it.
        ///
        /// ★ WHY THE ACCESSORS GATE ON IT, since 2026-08-26. A solve always returns a NUMBER, and an
        /// unexcited parameter's number is wherever the normal equations happened to leave it — not
        /// zero, just unsupported. Measured live the hour this gate was added: k_v read 0.935209 and
        /// k_omega 0.957712 — a 6.5% and 4.2% correction being applied to every wheel increment —
        /// with sigma still at 0.0200 against a prior of 0.02 and `informed` false for both. The
        /// estimator was saying it had learned nothing while its output steered the odometry.
        /// A value 3.3 prior-sigmas from nominal that its own precision does not support is exactly
        /// the case this project already decided: a value that is honestly absent beats one that is
        /// silently wrong. Untaught therefore returns EXACTLY nominal — 1.0 for a scale, 0.0 for an
        /// offset — and the parameter starts acting the moment, and only the moment, it is measured.
        ///
        /// ★ This does NOT make the estimator quieter about itself. The value and sigma are still
        /// reported and plotted unchanged; what is gated is only whether the number is allowed to
        /// move the robot's odometry. Diagnosis and authority are different questions.
        [[nodiscard]] bool taught(int p) const noexcept
        { return enabled() and last_.informed[p]; }
        /// Taught AND allowed to act. The applied accessors below gate on this; the estimated_*
        /// accessors gate on taught() alone, so the log keeps its meaning in open-loop mode.
        [[nodiscard]] bool acting(int p) const noexcept
        { return taught(p) and cfg_.apply and ((cfg_.apply_mask >> p) & 1); }
        /// Applied to the body->world rotation of the odometry displacement.
        [[nodiscard]] float yaw_offset() const noexcept
        { return acting(rc::calib::P_EPS_YAW) ? last_.value[rc::calib::P_EPS_YAW] : 0.f; }
        /// The ESTIMATE, regardless of whether it is allowed to act. For logging only.
        [[nodiscard]] float estimated_yaw_offset() const noexcept
        { return taught(rc::calib::P_EPS_YAW) ? last_.value[rc::calib::P_EPS_YAW] : 0.f; }
        /// Multiplies forward (and lateral) wheel displacement.
        [[nodiscard]] float forward_scale() const noexcept
        { return acting(rc::calib::P_K_V) ? 1.f + last_.value[rc::calib::P_K_V] : 1.f; }
        /// The ESTIMATE, regardless of whether it is allowed to act. For logging only.
        [[nodiscard]] float estimated_forward_scale() const noexcept
        { return taught(rc::calib::P_K_V) ? 1.f + last_.value[rc::calib::P_K_V] : 1.f; }
        /// Multiplies the heading increment, whichever channel produced it.
        [[nodiscard]] float omega_scale() const noexcept
        { return acting(rc::calib::P_K_OMEGA) ? 1.f + last_.value[rc::calib::P_K_OMEGA] : 1.f; }
        /// The ESTIMATE, regardless of whether it is allowed to act. For logging only.
        [[nodiscard]] float estimated_omega_scale() const noexcept
        { return taught(rc::calib::P_K_OMEGA) ? 1.f + last_.value[rc::calib::P_K_OMEGA] : 1.f; }
        /// rad/s, subtracted from the measured rate. Separated from the SCALE only by the
        /// time-vs-rotation covariate pair, which is why it needs the joint solve to exist at all.
        [[nodiscard]] float omega_bias() const noexcept
        { return acting(rc::calib::P_B_OMEGA) ? last_.value[rc::calib::P_B_OMEGA] : 0.f; }
        /// The ESTIMATE, regardless of whether it is allowed to act. For logging only.
        [[nodiscard]] float estimated_omega_bias() const noexcept
        { return taught(rc::calib::P_B_OMEGA) ? last_.value[rc::calib::P_B_OMEGA] : 0.f; }
        /// Multiplies LATERAL wheel displacement. Separate from forward_scale(): a mecanum's lateral
        /// channel is the one roller slip corrupts, so the two are physically different errors.
        [[nodiscard]] float lateral_scale() const noexcept
        { return taught(rc::calib::P_K_LAT) ? 1.f + last_.value[rc::calib::P_K_LAT] : 1.f; }
        /// rad per metre driven forward: unequal effective wheel radii make a commanded straight
        /// line curve. Added to the heading increment in proportion to DISTANCE, which is what
        /// distinguishes it from a gyro scale (rotation) or a gyro bias (time).
        [[nodiscard]] float wheel_mismatch() const noexcept
        { return taught(rc::calib::P_DK_WHEEL) ? last_.value[rc::calib::P_DK_WHEEL] : 0.f; }

        [[nodiscard]] float yaw_sigma() const noexcept { return last_.sigma[rc::calib::P_EPS_YAW]; }
        [[nodiscard]] float k_v_sigma() const noexcept { return last_.sigma[rc::calib::P_K_V]; }
        [[nodiscard]] float k_w_sigma() const noexcept { return last_.sigma[rc::calib::P_K_OMEGA]; }
        [[nodiscard]] const rc::calib::Result& last_solve() const noexcept { return last_; }
        [[nodiscard]] const rc::calib::CalibrationIntake& intake() const noexcept { return intake_; }
        /// Tag episodes that follow as coming from a deliberate manoeuvre. Reporting only -- it
        /// cannot change how an episode is weighted, and must not: see calibration_intake.h.
        void set_source(rc::calib::Source s) noexcept { source_hint_ = s; }
        [[nodiscard]] int   episodes() const noexcept { return episodes_; }
        /// Spans that reached the close trigger with NO correction in them and were carried forward
        /// instead of emitted as a zero. A large ratio against episodes() is the honest signature of
        /// a localiser that is early-exiting almost everything -- it says the teacher has gone quiet,
        /// which used to be invisible because those spans were emitted as confident zeros.
        [[nodiscard]] int   carried() const noexcept { return carried_; }
        /// Carried spans abandoned at the linearisation cap: the pose went uncorrected too far.
        [[nodiscard]] int   dropped() const noexcept { return dropped_; }

        /// Called once per localiser cycle, on BOTH the early-exit and the optimized path.
        ///   d_forward/d_lateral/d_theta : what the motion model predicted this cycle (post-scaling)
        ///   r_forward/r_lateral/r_theta : corrected minus predicted, in the ROBOT frame (0 on early exit)
        ///   pos_var/theta_var           : the optimizer's posterior variance, i.e. how much to believe r
        ///   corrected                   : did the optimizer run this cycle
        /// A closed pivot, straight into the solve. See BatchEstimator::add_closure for why this is
        /// two rows on the covariates the heading episode already uses, and why two rates are needed
        /// before k_omega and b_omega can be told apart at all.
        void observe_closure(double truth_rad, double turned_rad, double rate_rad_s,
                             double sigma_s) noexcept
        {
            if (not enabled()) return;
            intake_.offer_closure(truth_rad, turned_rad, rate_rad_s, sigma_s);
            last_ = intake_.estimate();     // re-solve now: a closure is worth minutes of robot time
        }
        [[nodiscard]] std::size_t closures() const noexcept { return intake_.closures(); }

        /// Persist / restore what has been MEASURED, and forget it. The parameters are never written:
        /// see BatchEstimator::save for why restoring a fitted value as a prior mean is a ratchet.
        bool save_state(const std::string& path) const { return intake_.save(path); }
        std::size_t load_state(const std::string& path)
        {
            const std::size_t n = intake_.load(path);
            if (n > 0) last_ = intake_.estimate();   // re-solve from the restored evidence
            return n;
        }
        /// Back to the priors, and the file with it — a reset that left the file behind would be
        /// undone by the next restart, which is not what anyone pressing "reset" means.
        void reset_state(const std::string& path) noexcept
        {
            intake_.reset();
            last_ = intake_.estimate();
            std::error_code ec; std::filesystem::remove(path, ec);
            episodes_ = 0;
        }

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
            if (corrected)
            {
                // ── THE FIT THAT PRODUCED A CORRECTION, NOT THE DRIFT THAT PRECEDED IT ───────────
                // Worst fit seen, not the mean: one bad frame is enough to make the whole
                // accumulated correction untrustworthy, and averaging would let a long clean ramp
                // hide it. That rationale is about the fits that ACTUALLY RAN, and this used to be
                // accumulated on every cycle, corrected or not.
                //
                // ★ ON AN EARLY-EXIT CYCLE THE RESIDUAL IS NOT A FIT QUALITY -- IT IS THE
                //   ACCUMULATED DRIFT, which is to say the SIZE OF THE VERY ERROR THE PARAMETERS
                //   EXIST TO EXPLAIN. Taking the max across the whole episode therefore made R grow
                //   with the response rather than with any measure of trustworthiness, so the
                //   episodes carrying the largest true errors were believed least. That is selection
                //   on the outcome, performed by the weights, and it attenuates the fitted slope the
                //   way a regression weighted by |y| would. Measured 2026-08-30 on the live window:
                //   corr(|r_forward|, pos_var) = +0.317, and an unweighted refit of the same 415
                //   episodes moved the slope 16% further from nominal than the weighted one.
                //
                // ⚠ Same term, second inversion. It was added on 2026-08-23 to distrust bad fits;
                //   it had already been found trusting UNMEASURED episodes most (an early-exit
                //   cycle has a small residual by definition, so a span with no correction in it
                //   drew the tightest variance in the window). Both failures come from reading a
                //   number that means "how far off was the prediction" as though it meant "how good
                //   was the fit". They are the same number only on a cycle where a fit happened.
                if (std::isfinite(fit_residual)) acc_fit_ = std::max(acc_fit_, fit_residual);
                acc_r_fwd_ += r_forward; acc_r_lat_ += r_lateral; acc_r_th_ += r_theta;
                acc_pos_var_ = std::max(acc_pos_var_, pos_var);
                acc_th_var_  = std::max(acc_th_var_,  theta_var);
                acc_measured_ = true;   // this episode contains an actual measurement
            }

            // ── WHEN AN EPISODE ENDS ─────────────────────────────────────────────────────────────
            // Originally: one ramp plus the correction that ended it, flushed on the falling edge of
            // "the optimizer ran" — the moment the whole accumulated error had been observed exactly
            // once. That is correct WHEN the optimizer is the only thing that corrects the pose.
            //
            // ★ IT STOPPED BEING TRUE. The optimizer fires on ~0.3-3% of cycles, and the SDF polish
            // now corrects a little on EVERY cycle, so there is no longer a ramp waiting for a single
            // large correction to end it. Measured 2026-08-26: with the calibrator enabled and being
            // fed on every cycle, not one episode had ever been ACCEPTED — every parameter read
            // exactly 0.000 with sigma exactly 0.000, which is "never solved", not "solved and
            // untaught" (that would show the prior's 0.02). The teacher had gone quiet without
            // anything reporting it.
            //
            // ★ SO THE TRIGGER IS THE MOTION, NOT THE CORRECTOR. An episode closes once the robot has
            // done enough to identify something — distance driven or angle turned — and it carries
            // whatever correction accumulated over that span, from the optimizer or the polish or
            // both. The falling edge still closes an episode, so a large correction is never split
            // across two.
            // ★ The two thresholds are IDENTIFIABILITY, not tuning: below them the Jacobian rows are
            // ~0 and the episode teaches nothing whatever its residual says. A parked robot must
            // never close an episode, or the window fills with rows that only dilute.
            const bool moved_enough = std::abs(acc_fwd_) >= cfg_.episode_min_trans
                                   or std::abs(acc_th_)  >= cfg_.episode_min_rot;
            const bool want_close = (prev_corrected_ and not corrected) or moved_enough;

            // ── ★★★ AN EPISODE WITH NO CORRECTION IS NOT AN OBSERVATION OF ZERO ──────────────────
            // Measured 2026-08-30 on a live cold-started run, and this is why the test exists:
            // at 98.4% early exit the optimizer runs on 1.6% of cycles, so 438 of 512 episodes in
            // the window (86%) closed on MOTION having never seen a correction and were emitted as
            // "the correction was exactly zero". Worse, they were the BEST weighted rows in the
            // window -- median pos_var 0.004945 against 0.007436 for real ones, and a larger
            // covariate (0.2512 m against 0.1592) -- because with acc_pos_var_ still 0 the variance
            // is set entirely by fit_model_gain * max|SDF|, and an early-exit cycle is BY DEFINITION
            // one whose SDF residual was small. The term written on 2026-08-23 to distrust bad fits
            // had inverted into one that trusts unmeasured episodes most. Result: 94.2% of the
            // Fisher information on a distance-regressed parameter came from rows in which nothing
            // was measured, every parameter sat at nominal with shrinking sigma, and eps_yaw never
            // became informed at all. See CALIB_UNMEASURED_EPISODES.md.
            //
            // ★ Widening R does NOT fix it and that was checked, not assumed: those rows already
            // carry sigma_pos 0.070 m, about the width of the early-exit gate itself. 438 assertions
            // of "zero" outvote 74 measurements on weight of numbers whatever the variance. The
            // defect is not the precision on the observation -- it is that an observation is
            // asserted when nobody looked.
            //
            // ★ This is NOT a threshold. `corrected` is a boolean fact about whether the optimizer
            // ran; measured-versus-not-measured is the one distinction an estimator may never blur.
            // The motion is CARRIED, not discarded: the covariate survives, and the next real
            // correction simply explains a longer span.
            if (want_close)
            {
                if (acc_measured_) { flush(); waiting_ = false; }
                else if (not waiting_) { waiting_ = true; ++carried_; }
            }
            prev_corrected_ = corrected;

            // Linearisation guard on a carried episode (see episode_carry_max_*). Reaching it means
            // the pose went uncorrected across several metres, so the accumulated motion can no
            // longer be treated as one increment and the episode is dropped rather than distorted.
            if (std::abs(acc_fwd_) > cfg_.episode_carry_max_trans
                or std::abs(acc_th_) > cfg_.episode_carry_max_rot)
            { ++dropped_; reset_episode(); }
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
            // ★ WHAT WAS ACTING WHILE THIS EPISODE WAS ACCUMULATED. Captured here, BEFORE the offer
            // below re-solves and moves last_, so it is the value that was in force over the span --
            // not the one the window is about to produce. acting() and not taught(): a value that is
            // known but withheld (untaught, disabled, or open-loop) never touched the odometry, so
            // nothing needs undoing for it. See Episode::p_applied for why this exists at all.
            for (int i = 0; i < rc::calib::P_COUNT; ++i)
                e.p_applied[i] = acting(i) ? last_.value[i] : 0.f;
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
            acc_measured_ = false;
            waiting_ = false;
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
        /// Has any corrected cycle contributed to the episode being accumulated? Only then is it an
        /// observation at all.
        bool acc_measured_ = false;
        /// Already counted as carried, so a span that keeps exceeding the trigger while waiting for
        /// a correction is one carry, not one per cycle.
        bool waiting_ = false;
        int episodes_ = 0, carried_ = 0, dropped_ = 0;
    };
}
