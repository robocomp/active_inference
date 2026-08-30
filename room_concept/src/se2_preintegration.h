/*
 *  se2_preintegration.h — preintegration of an SE(2) odometry stream into ONE motion factor.
 *
 *  WHY THIS EXISTS
 *  ---------------
 *  The motion factor between two window slots needs a mean AND a covariance for the relative motion
 *  over the interval. Today the mean is integrated from the odometry stream (integrate_odometry_over_
 *  window) but the covariance is ASSERTED by compute_motion_covariance():
 *
 *      cov = Identity();  cov(0,0) = cov(1,1) = position_std²;  cov(2,2) = rotation_std²
 *      position_std = odom_noise_base + odom_noise_trans·|Δp|   (⊕ rotation_position_coupling·|Δθ|)
 *      rotation_std = rotation_noise_base + odom_noise_rot·|Δθ| (⊕ encoder_rot_slip_k·|Δθ|)
 *
 *  Three things are wrong with that shape, independently of how its constants are tuned:
 *
 *   1. IT IS DIAGONAL AND ISOTROPIC IN xy. A heading error at the START of the interval rotates ALL
 *      of the translation that follows, so the true covariance has strong θ↔xy and x↔y terms oriented
 *      along the direction of travel. `rotation_position_coupling` is an attempt to fake that on the
 *      diagonal; it can carry the magnitude but not the direction, and an isotropic inflation is the
 *      wrong shape wherever the robot is actually going somewhere.
 *
 *   2. dt APPEARS NOWHERE, so the constants are rate-dependent. That is not hypothetical: the 08-09
 *      units defect was exactly this — `encoder_rot_slip_k·(|Δθ|/dt)` used as radians, inflating σ by
 *      1/dt (~16x, variance ~260x) at the 62 ms update interval, which drowned the measured prior at
 *      ANY k. A recursion cannot make that mistake, because dt only appears where the physics puts it.
 *
 *   3. IT CANNOT DISTINGUISH RANDOM FROM SYSTEMATIC ERROR. `odom_noise_rot·|Δθ|` and
 *      `encoder_rot_slip_k·|Δθ|` both describe a SCALE error — a fully correlated, sign-stable
 *      fraction of the rotation. Adding it as if it were per-sample noise mis-states how it grows
 *      with the length of the interval (a correlated error grows like T, an independent one like √T),
 *      and — the real cost — there is nowhere for the estimator to LEARN it, so it stays a hand-set
 *      constant. The measured value it is standing in for is meas/post = 1.072 on hardware.
 *
 *  WHAT THIS DOES
 *  --------------
 *  Forster et al.'s preintegration, specialised to SE(2). The interval's high-rate samples are
 *  summarised ONCE into three objects:
 *
 *      Δ    the composed relative motion            (the mean — unchanged from today, see below)
 *      Σ    its covariance, from a linear recursion (replaces the asserted diagonal)
 *      G    ∂Δ/∂s, the SCALE JACOBIANS              (see "THE SCALE TERM")
 *
 *  Σ comes from the standard propagation
 *
 *      Σ_i = A_i Σ_{i-1} A_iᵀ + B_i (Q·dt_i) B_iᵀ
 *
 *  where A_i is how an already-accumulated error transports through step i, B_i is how this step's
 *  noise enters, and Q is a per-sample noise DENSITY — one physical constant per channel, with units,
 *  measurable once from a parked/constant-velocity trace. Q·dt makes Σ a proper Wiener increment: the
 *  random part grows like √T and SHRINKS as the sample rate rises for a fixed T, which is the honest
 *  statement of what a faster odometry stream buys.
 *
 *  THE SCALE TERM (and why G is the interesting output)
 *  ----------------------------------------------------
 *  A scale error is not noise; it is a bias, fully correlated across the interval. Treating it as
 *  per-sample noise under-counts it by √M. So it is handled the way preintegration handles an IMU
 *  bias: propagate its Jacobian G = ∂Δ/∂s alongside Σ, and add ONE rank-structured term
 *
 *      Σ_total = Σ_random + G diag(σ_s²) Gᵀ
 *
 *  This is exact for a constant unknown scale, it generates the correct correlations for free (G is a
 *  direction, not a magnitude), and — the point — the SAME G is what a scale STATE needs. Promoting
 *  the scale from "a covariance term I assert" to "a variable the SDF identifies" is then: register s
 *  in the GN VarIndex, give MotionFactor a third variable slot with Jacobian G, drop the outer product
 *  from Σ. Nothing here has to change. See SCALE-AS-A-STATE below for why that is not switched on yet.
 *
 *  FRAME CONVENTION — deliberately the SAME as the legacy path
 *  ----------------------------------------------------------
 *  Δ is a GLOBAL-frame increment [dx, dy, dθ], accumulated with midpoint θ, bit-for-bit the same
 *  arithmetic integrate_odometry_over_window() already performs. Σ is expressed in that same global
 *  frame. This is what makes the change a drop-in: MotionFactor's residual (x_b − x_a − Δ) and the
 *  torch motion term both consume a full 3×3 precision already, so ONLY the covariance changes and
 *  the two backends stay term-for-term identical with no factor edit. The mean is untouched, so with
 *  the feature off the behaviour is byte-for-byte legacy, and with it on the mean is still legacy —
 *  the only difference is a covariance that was derived instead of asserted.
 *
 *  (A manifold residual Log(ΔT⁻¹·T_a⁻¹T_b) is the natural next step and is what makes the mean exact
 *  for large Δθ — the current form shares the identity-Jacobian approximation already flagged in the
 *  FEJ+Schur note. It is NOT done here because it must land in BOTH backends simultaneously or the
 *  shadow comparison stops measuring objective identity and starts measuring their disagreement.)
 *
 *  SCALE-AS-A-STATE: WHY IT IS NOT ON
 *  ----------------------------------
 *  Two blockers, both real:
 *   • It is not identifiable in simulation. webots-bridge sets pose_data.rot = shadow_velocity[5],
 *     the supervisor's GROUND-TRUTH angular velocity — there is no encoder in the sim loop at all —
 *     and commit 5968a65 (LY 0.237→0.210, plus per-sim-second velocities carrying wall-clock stamps)
 *     already took meas/post from 1.069 to 1.002. Estimating a scale here would be estimating a scale
 *     on ground truth, and any adaptive term must be gated off in sim.
 *   • It changes the marginalization. A scale spans the whole window and beyond, so when the oldest
 *     pose leaves, the Schur complement involves it too, and freezing its Jacobian at a stale value
 *     is precisely the mechanism that ratcheted loss_boundary 0→559. The safe form is to keep it OUT
 *     of the marginalization as a persistent variable with a random-walk prior — which is what most
 *     VIO systems do with a keyframe-persistent bias — not to let marginalize_oldest() see it.
 *  The G outputs below are computed regardless, so that work is an addition, not a rewrite.
 */
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace rc::preint
{
    /// Per-sample noise model of one odometry channel.
    ///
    /// sigma_* are DENSITIES, not per-frame standard deviations: the variance contributed over an
    /// interval of length T is sigma²·T. Units are therefore m/√s and rad/√s. The point of the
    /// density form is that it is a property of the SENSOR and does not have to be re-tuned when the
    /// localizer's update rate changes — which the current `odom_noise_base`-style constants do,
    /// silently, because they are added once per frame regardless of how long the frame was.
    ///
    /// scale_* are FULLY CORRELATED fractional errors (dimensionless): "this channel reads k too
    /// much, consistently, for the whole interval". They enter through the scale Jacobians, not
    /// through Q, because a correlated error grows like T and an independent one like √T.
    /// Defaults reproduce the legacy covariance in the STATIONARY regime, where the robot spends >99%
    /// of its logged frames — sigma = legacy_base / √dt at dt = 50 ms. They deliberately do NOT
    /// reproduce it while moving; see the WHAT CHANGES note in Params::motion_preintegration.
    struct NoiseModel
    {
        float sigma_v_lat  = 0.00447f;  // m/√s   — lateral velocity noise floor  (0.001 m / √0.05 s)
        float sigma_v_long = 0.00447f;  // m/√s   — forward velocity noise floor
        float sigma_omega  = 0.0447f;   // rad/√s — yaw-rate noise floor          (0.010 rad / √0.05 s)
        float scale_v      = 0.08f;     // fraction of |Δp| that is a consistent scale error
        float scale_omega  = 0.155f;    // fraction of |Δθ| that is a consistent scale error

        // ── Zero-velocity update (ZUPT) ────────────────────────────────────────────────────────────
        // The densities above describe a body that is FREE TO DRIFT, so the interval covariance grows
        // as √T whether or not the robot is moving. That is correct for an unaided IMU and wrong here:
        // a parked differential base is not drifting, and its wheels and gyro are actively SAYING so.
        // Measured 2026-08-18 without this term: parked, preint_T reached 95 s median / 294 s p95 and
        // the motion prior's σ grew from 3 cm (moving) to 82 cm — vacuous, so the prior silently
        // stopped constraining anything for three quarters of a tour.
        //
        // "The body is at rest" is an OBSERVATION (z = 0) with its own precision, not a special case,
        // so it belongs in the generative model as a factor. zupt_density_* is the residual velocity
        // noise of a genuinely stationary sensor — encoder quantisation, gyro output noise.
        //
        // ★★★ THESE ARE DENSITIES (m/√s, rad/√s), NOT per-sample standard deviations, and that
        // changed on 2026-08-26 after the per-sample form was caught being rate-dependent. R is
        // formed as density²/dt, so the information the rest hypothesis contributes over an interval
        // is set by ELAPSED TIME and not by how many samples happened to land in it. The old form
        // applied a fixed σ once per sample, so publishing faster silently tightened the parked prior
        // — measured at the live constants, going 48 → 16 ms made it 1.73x tighter for no physical
        // reason. The name changed with the units on purpose: a silent unit change on a field with
        // the same name is exactly how a wrong constant survives a re-measurement.
        //
        // ★★★ MEASURED ON P3BOT, 2026-08-26: 8166 parked samples at 16.0 ms (etc/odom_samples.csv,
        // analysed by tools/odom_whiteness.py) gave a per-sample spread of 0.1183 m/s and
        // 0.2372 rad/s, i.e. densities of 0.01496 m/√s and 0.03001 rad/√s. The per-sample figure is
        // provably 1/√dt — 0.0683 m/s at 48 ms against 0.1183 at 16 ms — which is WHY this has to be
        // a density: no fixed per-sample number can describe a quantity that moves with the sampling
        // interval. The samples are white at that interval (ρ₁ = 0.026), so applying the update once
        // per sample is not double-counting; note the noise is INJECTED in simulation and therefore
        // white by construction, so that finding does NOT transfer to hardware unmeasured.
        // ★ The values before this were 0.0202 / 0.0323, MEASURED ON SHADOW and never re-measured
        // here: 5.9x and 7.3x too tight for this robot. Before Shadow they were 0.002 / 0.005 from a
        // plausible-sounding "encoder quantisation" argument, 10x and 6.5x too tight the other way,
        // which made the motion factor out-weigh the laser 7:1. That is twice this constant has been
        // wrong by an order of magnitude because it was reasoned about instead of measured.
        // RE-MEASURE ON ANY NEW PLATFORM. tools/odom_whiteness.py does it in one parked run.
        //
        // ★NO THRESHOLD, and none is needed. The factor's own variance carries the measured motion:
        // R = density²/dt + (how fast the body is actually moving)². At rest that is tiny and the
        // update bites; in motion the rest hypothesis is inconsistent by exactly the observed speed,
        // R blows up with it, and the update fades out on its own. The crossover is not tuned — it
        // falls out where the two variances meet.
        // ⚠ HONEST LIMIT of the density form: it makes the PARKED case exactly rate-invariant (the
        // regime where this term does its work and where the artefact lived), but the motion term m²
        // does not scale with dt, so a residual dependence survives at speed — 1.47x across a 3x rate
        // change at 0.25 m/s, against 1.57x before. Removing it entirely means lifting the rest
        // hypothesis OUT of the per-sample propagation and applying it as ONE factor per interval,
        // which is how a factor-graph framework expresses "the body was at rest during this stretch"
        // and is also strictly more conservative at speed (it is falsified by the whole interval's
        // motion, not by each segment's). That is the next step, not this one.
        bool  zupt_enabled   = true;
        /// ★ THE REST HYPOTHESIS AS A FACTOR ON THE STATE, instead of as covariance shaping.
        /// The per-sample form (zupt_enabled, above) collapses the velocity VARIANCE and never
        /// touches the mean, because the preintegrator is a pure observer. That leaves the estimator
        /// saying, of a parked robot, "the increment is 3.5 mm of noise and I am very confident" —
        /// confidently wrong rather than correctly still. Measured 2026-08-26: parked with ground
        /// truth motionless for 3963 cycles, the published pose wandered 210 mm in x, 193 mm in y and
        /// 2.07 deg, a textbook random walk of 3.49 mm per cycle (3.49*sqrt(3963) = 220 mm).
        ///
        /// A rest hypothesis that constrains the STATE fixes that, and it is not a new mechanism: it
        /// is a motion factor between the same two slots with a ZERO delta and its own precision, so
        /// the solver weighs "you moved by Delta" against "you did not move" by their stated
        /// precisions and moves the MEAN accordingly.
        ///
        /// Its covariance is the interval-level analogue of R = sigma_rest^2 + m^2, with the noise
        /// integrated over the interval and the inconsistency being the motion the interval itself
        /// measured:
        ///       R_trans = d_v^2 * T + (|dp| + L*|dtheta|)^2
        ///       R_rot   = d_w^2 * T + (|dtheta| + |dp|/L)^2
        /// At rest the second term vanishes and the factor bites; in motion the hypothesis is wrong
        /// by exactly the observed displacement, R grows with it, and the factor fades out on its
        /// own. No threshold, and rate-invariant by construction — it is stated per INTERVAL, so the
        /// number of samples inside cannot change it. That also removes the residual m^2*dt rate
        /// dependence the per-sample form still carries at speed.
        ///
        /// ★ THE TWO FORMS ARE EXCLUSIVE. Running both counts one hypothesis twice — once shaping the
        /// prior's covariance and once as its own factor — which is the double-counting this file
        /// spends its length avoiding. Setting this true therefore DISABLES the per-sample shaping.
        /// OFF by default: it moves the published pose, so it earns its place by A/B against the
        /// 210 mm above, not by argument.
        bool  zupt_as_factor = false;
        float zupt_density_v = 0.01496f;    // m/√s   — measured, P3Bot, 8166 parked samples
        float zupt_density_omega = 0.03001f;// rad/√s — measured, P3Bot, same run
        // Lever arm converting between the two channels, so the rest hypothesis is about the WHOLE
        // BODY: a robot pivoting in place has v_lat = v_long = 0 yet is emphatically not at rest, and
        // its wheels are scrubbing. Without this, a pivot would be handed the parked translation noise.
        // Set it to the wheel anchor offset (±0.259 m on Shadow, see Shadow.proto).
        float zupt_lever_m   = 0.26f;
    };

    /// A summarised interval. `delta` and `cov` share ONE frame: the global (room) frame, with the
    /// heading the integrator was seeded with.
    ///
    /// The interval is self-contained on purpose — it carries the two scale sigmas it was integrated
    /// under, so covariance() and chain() need no side channel telling them which odometry channel
    /// produced it. Intervals from different channels can therefore be chained without a caller
    /// having to remember whose NoiseModel applies.
    struct Interval
    {
        Eigen::Vector3f delta   = Eigen::Vector3f::Zero();   // [dx, dy, dθ], global frame
        Eigen::Matrix3f cov     = Eigen::Matrix3f::Zero();   // RANDOM part only; see covariance()
        Eigen::Vector3f g_omega = Eigen::Vector3f::Zero();   // ∂Δ/∂s_ω
        Eigen::Vector3f g_v     = Eigen::Vector3f::Zero();   // ∂Δ/∂s_v
        float           sigma_scale_omega = 0.f;             // σ of the correlated yaw-scale error
        float           sigma_scale_v     = 0.f;             // σ of the correlated speed-scale error
        float           duration_s = 0.f;
        int             samples    = 0;
        /// Rest-hypothesis densities carried through, so covariance() and the factor below need no
        /// side channel telling them which NoiseModel produced this interval.
        float           zupt_density_v_ = 0.f, zupt_density_omega_ = 0.f, zupt_lever_ = 0.26f;

        /// Covariance of the "this interval had no motion" hypothesis, for the zero-delta factor.
        /// Returns a zero matrix when the interval is degenerate, which the caller reads as "do not
        /// add the factor".
        [[nodiscard]] Eigen::Matrix3f zupt_covariance() const
        {
            if (duration_s <= 0.f or zupt_density_v_ <= 0.f or zupt_density_omega_ <= 0.f)
                return Eigen::Matrix3f::Zero();
            const float L    = std::max(zupt_lever_, 1e-3f);
            const float dp   = delta.head<2>().norm();
            const float dth  = std::abs(delta[2]);
            const float m_tr = dp + L * dth;          // a pivot is NOT rest: the lever says so
            const float m_ro = dth + dp / L;
            const float r_tr = zupt_density_v_ * zupt_density_v_ * duration_s + m_tr * m_tr;
            const float r_ro = zupt_density_omega_ * zupt_density_omega_ * duration_s + m_ro * m_ro;
            Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
            R(0, 0) = r_tr; R(1, 1) = r_tr; R(2, 2) = r_ro;
            return R;
        }

        /// Full covariance: the propagated random part plus the rank-2 correlated-scale term.
        /// Kept separate from `cov` so that promoting the scales to STATES is a subtraction here and
        /// an addition in the solver, with no other edit (see SCALE-AS-A-STATE in the file header).
        ///
        /// ⚠ Call this ONCE, on the interval the motion factor will actually use. Calling it per frame
        /// and then summing would treat one constant scale error as a fresh independent draw every
        /// frame, which is the √M-vs-M under-count this structure exists to avoid.
        [[nodiscard]] Eigen::Matrix3f covariance() const
        {
            return cov
                 + (sigma_scale_omega * sigma_scale_omega) * (g_omega * g_omega.transpose())
                 + (sigma_scale_v     * sigma_scale_v)     * (g_v     * g_v.transpose());
        }
    };

    /// Accumulates samples of an odometry stream into one Interval.
    ///
    /// Seed it with the heading the interval STARTS at (the same `running_theta` the legacy integrator
    /// seeds from the last posterior pose), then feed body-frame velocities and their durations in
    /// order. Sample dt's need not be equal, and the caller is expected to have already clipped them
    /// to the interval — this class does no time bookkeeping of its own on purpose, so it can be
    /// driven from the odometry channel, the command channel, or a test harness identically.
    class Integrator
    {
    public:
        explicit Integrator(float theta_start) : theta_run_(theta_start) {}

        /// One segment: constant body-frame velocity (v_lat along robot X, v_long along robot Y,
        /// omega CCW+) held for dt seconds. Mirrors the legacy midpoint-θ integration exactly.
        void add(float v_lat, float v_long, float omega, float dt)
        {
            add(v_lat, v_long, omega, dt, -1.f, -1.f, -1.f);
        }

        /// Same, but with MEASURED per-sample noise densities for this segment, ADDED IN QUADRATURE
        /// to the NoiseModel's constants (which are model error, not sensor noise -- see below). Units are the model's: m/√s and rad/√s, i.e. the variance
        /// contributed over an interval T is sigma²·T. Pass a negative value for any channel the
        /// producer did not state, and that channel falls back to the model.
        ///
        /// WHY: the sensors state their own per-sample variance (ImuFrame.gyro_var / acc_var, and
        /// the FullPose velCov), and a channel that degrades should be believed less without anyone
        /// editing a config file. But the model constants are a DIFFERENT quantity -- lumped model
        /// error, not sensor noise -- so the two add rather than one replacing the other.
        ///
        /// CONVERSION, and it is the easy thing to get wrong: a producer reports a PER-SAMPLE
        /// variance at ITS OWN rate. The density is that variance times the producer's sample period
        /// (sigma² = var_sample · dt_sample), NOT the variance itself. Handing this a raw per-sample
        /// variance over-states the noise by dt_sample/dt whenever a segment spans several samples,
        /// which at 100 Hz across a 50 ms segment is a factor of five.
        void add(float v_lat, float v_long, float omega, float dt,
                 float sigma_v_lat_meas, float sigma_v_long_meas, float sigma_omega_meas)
        {
            if (dt <= 0.f)
                return;

            const float dtheta    = omega * dt;
            const float theta_mid = theta_run_ + 0.5f * dtheta;
            const float c = std::cos(theta_mid), s = std::sin(theta_mid);

            // Step displacement in the global frame — the legacy arithmetic, unchanged.
            const float dx_local = v_lat  * dt;
            const float dy_local = v_long * dt;
            const float dpx = dx_local * c - dy_local * s;
            const float dpy = dx_local * s + dy_local * c;

            // A: transport of the error already accumulated. An error in the heading accumulated so
            // far rotates THIS step's displacement, which is where every off-diagonal term comes
            // from — and why summing per-frame covariances (the legacy stride_cov_accum_) is wrong.
            Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
            A(0, 2) = -dpy;
            A(1, 2) =  dpx;

            // B: how this step's noise enters. Columns are (n_lat, n_long, n_omega). The yaw column
            // carries the 0.5 of the midpoint rule: a yaw-rate error moves θ_mid by half the step.
            Eigen::Matrix3f B;
            B << c, -s, -0.5f * dpy,
                 s,  c,  0.5f * dpx,
                 0.f, 0.f, 1.f;

            // Q: the covariance of this step's DISPLACEMENT noise (B's inputs are metres and radians,
            // not velocities). Density form: variance = sigma²·dt, i.e. a velocity variance of
            // sigma²/dt carried for dt². Writing it through the velocity makes the ZUPT below a
            // one-line Bayesian update instead of a special case.
            // ADD IN QUADRATURE, NEVER REPLACE. The model constants are NOT the sensors' noise --
            // they are a lumped term covering everything the motion model does not capture: wheel
            // slip, timing, unmodelled dynamics. A sensor's own noise is a SEPARATE, additional
            // contribution. Replacing one with the other was tried on 2026-08-23 and made the prior
            // 135x too confident in heading (the gyro's density is 0.00033 rad/sqrt(s) against the
            // model's 0.0447): the prior then out-argued the SDF, the pose rigidly followed odometry,
            // the prediction error went 0.026 -> 0.080 and the optimizer fired on 68% of cycles
            // instead of 2%. A measured sensor variance can therefore only ever LOOSEN this prior,
            // which is the honest asymmetry -- it tells us when a sensor degrades, and it cannot
            // claim the MODEL is better than we know it to be.
            const auto combine = [](float model, float meas)
            { return meas >= 0.f ? std::sqrt(model * model + meas * meas) : model; };
            const float s_lat  = combine(q_.sigma_v_lat,  sigma_v_lat_meas);
            const float s_long = combine(q_.sigma_v_long, sigma_v_long_meas);
            const float s_om   = combine(q_.sigma_omega,  sigma_omega_meas);
            float var_v_lat  = s_lat  * s_lat  / dt;   // (m/s)²
            float var_v_long = s_long * s_long / dt;
            float var_omega  = s_om   * s_om   / dt;   // (rad/s)²

            if (q_.zupt_enabled and not q_.zupt_as_factor)
            {
                // How inconsistent the measurement is with "the body is at rest", in each channel's
                // own units, coupled through the lever arm so a pivot cannot claim to be parked.
                const float speed = std::hypot(v_lat, v_long);
                const float lever = std::max(q_.zupt_lever_m, 1e-3f);
                const float m_v   = speed + lever * std::abs(omega);          // m/s
                const float m_w   = std::abs(omega) + speed / lever;          // rad/s

                // R = (rest sensor noise)² + (observed motion)². Scalar-Kalman update of the velocity
                // variance: P⁺ = P·R/(P+R). At rest R→density²/dt and P⁺ collapses to it; in motion
                // R→∞ and P⁺→P, recovering the free-drift model exactly.
                // density²/dt, so the rest hypothesis contributes information per unit TIME rather
                // than per SAMPLE. The motion term is a velocity and is NOT divided by dt: it is how
                // wrong the hypothesis is, not how noisy the sensor is.
                const auto zupt = [dt](float P, float density_rest, float motion)
                {
                    const float R = density_rest * density_rest / dt + motion * motion;
                    return (P * R) / (P + R);
                };
                var_v_lat  = zupt(var_v_lat,  q_.zupt_density_v,     m_v);
                var_v_long = zupt(var_v_long, q_.zupt_density_v,     m_v);
                var_omega  = zupt(var_omega,  q_.zupt_density_omega, m_w);
            }

            Eigen::Matrix3f Q = Eigen::Matrix3f::Zero();
            Q(0, 0) = var_v_lat  * dt * dt;
            Q(1, 1) = var_v_long * dt * dt;
            Q(2, 2) = var_omega  * dt * dt;

            iv_.cov = A * iv_.cov * A.transpose() + B * Q * B.transpose();

            // Scale Jacobians, same transport. ∂/∂s_ω of this step: the yaw increment scales, which
            // both adds to Δθ and rotates the step through θ_mid (hence the 0.5·dtheta factor).
            Eigen::Vector3f b_omega;
            b_omega << -0.5f * dpy * dtheta, 0.5f * dpx * dtheta, dtheta;
            iv_.g_omega = A * iv_.g_omega + b_omega;

            // ∂/∂s_v: the displacement scales; the heading does not.
            Eigen::Vector3f b_v;
            b_v << dpx, dpy, 0.f;
            iv_.g_v = A * iv_.g_v + b_v;

            iv_.delta[0] += dpx;
            iv_.delta[1] += dpy;
            iv_.delta[2] += dtheta;
            iv_.duration_s += dt;
            ++iv_.samples;

            theta_run_ += dtheta;
        }

        void set_noise(const NoiseModel& q)
        {
            q_ = q;
            iv_.sigma_scale_omega = q.scale_omega;
            iv_.sigma_scale_v     = q.scale_v;
            iv_.zupt_density_v_     = q.zupt_density_v;
            iv_.zupt_density_omega_ = q.zupt_density_omega;
            iv_.zupt_lever_         = q.zupt_lever_m;
        }
        [[nodiscard]] const NoiseModel&  noise()  const { return q_; }
        [[nodiscard]] const Interval&    result() const { return iv_; }
        [[nodiscard]] float              heading() const { return theta_run_; }

    private:
        NoiseModel      q_{};
        Interval        iv_{};
        float           theta_run_ = 0.f;
    };

    /// Chain two consecutive intervals: `a`, then `b` integrated from the heading `a` ends at.
    ///
    /// This is the operator the strided window needs. The legacy code does `Σ_ab = Σ_a + Σ_b`
    /// ("independent increments", room_concept.cpp), which drops the transport: an error in a's
    /// heading rotates ALL of b's translation. With window_min_turn_rad = 0.15 rad the dropped term
    /// is not small, and it is exactly the term that would tell the solver the accumulated error is
    /// oriented across the path rather than isotropic around it.
    [[nodiscard]] inline Interval chain(const Interval& a, const Interval& b)
    {
        Eigen::Matrix3f A = Eigen::Matrix3f::Identity();
        A(0, 2) = -b.delta[1];
        A(1, 2) =  b.delta[0];

        Interval out;
        // Carry the rest-hypothesis densities across, or a chained interval would silently lose its
        // ability to state the hypothesis at all — zupt_covariance() would return zero and the caller
        // would read that as "do not add the factor" rather than "the densities went missing".
        out.zupt_density_v_     = a.zupt_density_v_     > 0.f ? a.zupt_density_v_     : b.zupt_density_v_;
        out.zupt_density_omega_ = a.zupt_density_omega_ > 0.f ? a.zupt_density_omega_ : b.zupt_density_omega_;
        out.zupt_lever_         = a.zupt_lever_ > 0.f ? a.zupt_lever_ : b.zupt_lever_;
        out.delta      = a.delta + b.delta;          // both already global-frame increments
        out.cov        = A * a.cov * A.transpose() + b.cov;
        out.g_omega    = A * a.g_omega + b.g_omega;
        out.g_v        = A * a.g_v     + b.g_v;
        out.duration_s = a.duration_s + b.duration_s;
        out.samples    = a.samples + b.samples;
        // The scale is ONE unknown constant shared by both halves, so the sigmas do not add; an empty
        // interval (all-zero, sigma 0) must not drag them down either, hence max rather than "take a's".
        out.sigma_scale_omega = std::max(a.sigma_scale_omega, b.sigma_scale_omega);
        out.sigma_scale_v     = std::max(a.sigma_scale_v,     b.sigma_scale_v);
        return out;
    }

} // namespace rc::preint
