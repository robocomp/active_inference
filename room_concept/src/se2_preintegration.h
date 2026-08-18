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
        // so it belongs in the generative model as a factor. zupt_sigma_* is the residual velocity
        // noise of a genuinely stationary sensor — encoder quantisation, gyro output noise.
        //
        // ★NO THRESHOLD, and none is needed. The factor's own variance carries the measured motion:
        // R = zupt_sigma² + (how fast the body is actually moving)². At rest that is tiny and the
        // update bites; in motion the rest hypothesis is inconsistent by exactly the observed speed,
        // R blows up with it, and the update fades out on its own. The crossover is not tuned — it
        // falls out where the two variances meet, at |v| ≈ sigma_v/√dt (~5 cm/s at 125 Hz), i.e. it
        // is set by the sensor's own noise density and sample rate.
        bool  zupt_enabled   = true;
        float zupt_sigma_v   = 0.002f;   // m/s   — residual linear velocity of a parked base
        float zupt_sigma_omega = 0.005f; // rad/s — residual yaw rate of a parked base
        // Lever arm converting between the two channels, so the rest hypothesis is about the WHOLE
        // BODY: a robot pivoting in place has v_lat = v_long = 0 yet is emphatically not at rest, and
        // its wheels are scrubbing. Without this, a pivot would be handed the parked translation noise.
        // Set it to the wheel anchor offset (±0.259 m on Shadow, see ShadowDiff.proto).
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
            float var_v_lat  = q_.sigma_v_lat  * q_.sigma_v_lat  / dt;   // (m/s)²
            float var_v_long = q_.sigma_v_long * q_.sigma_v_long / dt;
            float var_omega  = q_.sigma_omega  * q_.sigma_omega  / dt;   // (rad/s)²

            if (q_.zupt_enabled)
            {
                // How inconsistent the measurement is with "the body is at rest", in each channel's
                // own units, coupled through the lever arm so a pivot cannot claim to be parked.
                const float speed = std::hypot(v_lat, v_long);
                const float lever = std::max(q_.zupt_lever_m, 1e-3f);
                const float m_v   = speed + lever * std::abs(omega);          // m/s
                const float m_w   = std::abs(omega) + speed / lever;          // rad/s

                // R = (rest sensor noise)² + (observed motion)². Scalar-Kalman update of the velocity
                // variance: P⁺ = P·R/(P+R). At rest R→zupt_sigma² and P⁺ collapses to it; in motion
                // R→∞ and P⁺→P, recovering the free-drift model exactly.
                const auto zupt = [](float P, float sigma_rest, float motion)
                {
                    const float R = sigma_rest * sigma_rest + motion * motion;
                    return (P * R) / (P + R);
                };
                var_v_lat  = zupt(var_v_lat,  q_.zupt_sigma_v,     m_v);
                var_v_long = zupt(var_v_long, q_.zupt_sigma_v,     m_v);
                var_omega  = zupt(var_omega,  q_.zupt_sigma_omega, m_w);
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
