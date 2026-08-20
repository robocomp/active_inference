#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE ODOMETRY SCALES, ESTIMATED WHILE THE ROBOT DOES SOMETHING ELSE.
//
// se2_preintegration.h treats `scale_v` and `scale_omega` as ASSERTED CONSTANTS (0.08 and 0.155):
// fractional errors used to build the covariance. They are not measured, they are declared — and its
// own comment names the next step: "the SAME G is what a scale STATE needs. Promoting the scale from
// 'a covariance term I assert' to 'a variable the SDF identifies' is then: register s …".
//
// This is that promotion, done in the least invasive place first: a passive observer that watches the
// motion the robot is ALREADY making and reports a posterior over the scales. It changes no
// behaviour, feeds nothing back, and exists to answer one question continuously — how well does this
// robot currently know its own motion model?
//
// ★★★WHY PASSIVE FIRST, AND WHY IT MAY BE MOST OF THE WORK. The robot drives all day serving
// affordances: every one of those tours turns and translates, and the estimator does not care WHY the
// robot moved. Ordinary exploration is therefore free calibration data, and a deliberate calibration
// manoeuvre is only worth its cost when the tours have not been exciting enough — which is a
// statement this class can make, because it holds the posterior width that says so.
//
// THE MODEL, unchanged from the offline fit in room_concept/tools/motion_calib.cpp:
//
//     e(T) = s · Δ(T) + ε,        ε ~ N(0, σ² T)
//
// A scale error is fully correlated across an interval, so it grows like T; the random walk is
// independent, so it grows like √T. One weighted least squares separates them: the SLOPE is the
// scale, the RESIDUAL is the density. And it is not a heuristic — dF/ds = 0 and dF/dσ = 0 on the
// negative log evidence return exactly these, in closed form, so calibrating the model and maximising
// its evidence are the same computation.
//
// WHAT THIS ADDS TO THE OFFLINE VERSION: recursion, and a scale that is allowed to drift. The offline
// fit sees a whole log at once and assumes one constant scale. A tyre wears, a payload shifts, a
// surface changes — so the scale is a slowly varying quantity, and that belongs in the model as a
// RANDOM-WALK DENSITY (fraction/√s), in exactly the idiom NoiseModel already uses for the sensor
// noise. It is not a forgetting factor chosen to make numbers behave: it says how fast the world's
// scale is believed to change, and it is what makes an unexercised channel widen on its own.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rc::calib
{

/// What the estimator believes about one channel (rotation or translation).
struct ChannelPosterior
{
    double s        = 0.0;   ///< scale, dimensionless: "this channel reads s too much, consistently"
    double s_std    = 0.0;   ///< posterior standard deviation of s — THE "how well do I know it"
    double sigma    = 0.0;   ///< random-walk density, rad/√s or m/√s
    double info     = 0.0;   ///< Fisher information accumulated for s (Σ w·Δ²) — the excitation
    double span     = 0.0;   ///< range of |Δ| seen, so a degenerate slope is visible
    int    windows  = 0;     ///< how many windows have contributed
    double data_precision  = 0.0;   ///< what the MOTION contributes to knowing s  (Σ w·Δ² / σ²)
    double prior_precision = 0.0;   ///< what was assumed before any motion        (1 / prior_std²)
    /// ★HAS THE ROBOT ACTUALLY LEARNED ANYTHING? Not "is s_std small" — that was the first version and
    /// it needed an invented cutoff, which let a channel sitting exactly at its prior call itself
    /// identified. The factor-free statement is that the DATA outweighs the PRIOR: only then is the
    /// number being reported a measurement rather than the assumption it started with.
    [[nodiscard]] bool identifiable() const { return windows >= 3 and data_precision > prior_precision; }
};

/// At namespace scope, not nested: a nested type's default member initialisers are not complete
/// inside its own enclosing class, so it cannot serve as a default argument there. (Same trap as
/// GridPlanner::OccupiedCells, met and fixed earlier the same day.)
struct ScaleEstimatorParams
{
    /// How fast the scale itself is believed to drift, as a density (fraction/√s). Tyres wear,
    /// payloads shift; the scale is not a constant of nature. This is the ONLY number here, it is
    /// a property of the world rather than of the algorithm, and it is what makes a channel the
    /// robot has not exercised widen by itself instead of staying falsely sharp.
    double scale_walk_density = 1.0e-4;
    /// Prior standard deviation of the scale before any evidence. Set from what the asserted
    /// constants in NoiseModel already claim (scale_omega 0.155, scale_v 0.08) so the estimator
    /// starts no more confident than the system was without it.
    double prior_std = 0.15;
    /// ★THE NOISE DENSITY ALSO HAS A PRIOR, AND IT HAD TO. The density is estimated from the windows'
    /// residuals, so before any windows arrive there is nothing to estimate it from — the first
    /// version fell back to 1.0 rad/sqrt(s), an enormous placeholder, and a COLD estimator therefore
    /// advertised the smallest expected gain of its whole life. That is backwards: a robot that knows
    /// nothing about its own motion model is exactly the robot with most to gain from measuring it,
    /// and with the placeholder in place a fresh estimator promised 0.03 nats where the same estimator
    /// after 400 s of turning promised 3.7. It would have lost every arbitration on its first day.
    /// ★MEASURED, NOT CHOSEN: 0.008181 rad/sqrt(s) is the rotation residual density from a live run
    /// (2026-08-20, 209424 frames, 5234 windows, online and batch agreeing to 1e-9). Translation on
    /// the same run is 0.024494 m/sqrt(s); a translation channel should be constructed with that.
    double prior_density = 0.008181;
    /// How many windows the density prior is worth. ONE — the weakest proper prior there is, so it
    /// governs only while there is nothing else and is swamped by the second window onward. Set to 0
    /// to recover the exact unregularised batch estimator (what the offline replay compares against).
    double prior_density_windows = 1.0;
};

class ScaleEstimator
{
public:
    using Params = ScaleEstimatorParams;

    explicit ScaleEstimator(Params p = Params{}) : p_(p) { reset(); }

    void reset()
    {
        // ★DATA AND PRIOR ARE KEPT APART. A first version folded the prior into sxx_ as raw
        // information and then formed the variance as sigma^2/sxx_ — mixing two different scalings.
        // An unexercised channel came out with s_std = 0.0000: infinitely confident about a scale it
        // had never measured, which is precisely the failure this class exists to avoid. The posterior
        // precision of a linear-Gaussian slope is  (1/sigma^2)*Sum w*Delta^2 + 1/prior_var, and the
        // two terms only add once they are in the same units.
        sxx_ = 0.0;                                   // data only: Sum w*Delta^2
        sxy_ = 0.0;                                   // data only: Sum w*Delta*e
        syy_ = 0.0;                                   // data only: Sum w*e^2  (gives Sum w*r^2 exactly)
        s_   = 0.0;
        n_   = 0;
        span_lo_ =  1e300;
        span_hi_ = -1e300;
    }

    /// Time passing without evidence WIDENS the posterior — the random walk on the scale. Information
    /// form: I ← I / (1 + q²·dt·I), which is the scalar Kalman prediction written so it can never
    /// produce a negative variance however long the gap.
    void predict(double dt_s)
    {
        if (dt_s <= 0.0) return;
        // The walk acts on the DATA information: evidence about the scale ages, the prior does not.
        // Scaled by the current density so the decay is in the same units as the information it acts
        // on; with no density yet there is no evidence to age either.
        const double q2 = p_.scale_walk_density * p_.scale_walk_density;
        const double sig2 = std::max(density_squared(), 1e-12);
        sxx_ = sxx_ / (1.0 + q2 * dt_s * sxx_ / sig2);
        sxy_ = s_ * sxx_;                       // keep the mean while the information decays
    }

    /// One window of motion: the odometry increment, the error against the reference, and how long
    /// the window lasted. Weight 1/T because the noise variance is proportional to T — that is the ML
    /// weighting, not a preference.
    ///
    /// ★CALLERS MUST NOT PASS A WINDOW CONTAINING A POSE DISCONTINUITY. The reference is the
    /// localiser's posterior, and a relocalisation moves it further than the robot could have
    /// travelled — measured on one log, 5 steps above 15 cm between consecutive frames, two of them
    /// 0.61 m and 0.39 m. Charging that to the odometry puts a 0.6 m outlier into a fit whose typical
    /// increment is a metre. `window_is_physical` below is the same kinematic test the offline tool
    /// uses; it can only ever reject a physical impossibility, never real motion.
    void add(double delta, double err, double T_s)
    {
        if (not (T_s > 0.0) or not std::isfinite(delta) or not std::isfinite(err)) return;
        const double w = 1.0 / T_s;
        sxx_ += w * delta * delta;
        sxy_ += w * delta * err;
        syy_ += w * err * err;
        // MAP slope: data precision plus prior precision, both in units of 1/sigma^2 once scaled.
        // With no data this is 0 (the prior mean); with plenty it is the weighted least squares.
        s_ = posterior_mean();
        ++n_;
        span_lo_ = std::min(span_lo_, delta);
        span_hi_ = std::max(span_hi_, delta);
    }

    /// The kinematic admissibility of a window, in the caller's units: could the body have covered
    /// this much in this time? Generous by construction — it exists to exclude teleports, not to
    /// judge driving.
    [[nodiscard]] static bool window_is_physical(double reference_travel, double T_s, double max_speed)
    {
        return T_s > 0.0 and std::abs(reference_travel) <= 3.0 * max_speed * T_s;
    }

    [[nodiscard]] ChannelPosterior posterior() const
    {
        ChannelPosterior c;
        c.windows = n_;
        c.info    = sxx_;
        c.s       = s_;
        // dF/dsigma = 0, with the residual sum in closed form: Sum w*r^2 = Syy - s*Sxy for the fitted
        // s. Accumulating residuals against the RUNNING slope instead disagreed with the batch fit in
        // the fourth decimal — small, but it means the online and offline estimators are answering
        // slightly different questions, and the whole point of this class is that they do not.
        const double rss = std::max(syy_ - s_ * sxy_, 0.0);
        // ★ONE DEFINITION OF THE DENSITY, shared with everything that scales by it. Reporting
        // rss/(n-1) here while the gain calculation used density_squared() meant the number on the
        // dashboard and the number in the decision were not the same quantity.
        c.sigma   = (p_.prior_density_windows > 0.0 or n_ > 1) ? std::sqrt(density_squared())
                                                               : 0.0;
        // Posterior precision = data/sigma^2 + prior. Both terms in the same units, so an unexercised
        // channel falls back to the prior width instead of claiming certainty.
        const double prior_prec = 1.0 / (p_.prior_std * p_.prior_std);
        const double sig2 = (c.sigma > 0.0) ? c.sigma * c.sigma : density_squared();
        (void)rss;
        c.data_precision  = (sig2 > 1e-18) ? sxx_ / sig2 : 0.0;
        c.prior_precision = prior_prec;
        c.s_std = std::sqrt(1.0 / (c.data_precision + prior_prec));
        c.span  = (span_hi_ > span_lo_) ? span_hi_ - span_lo_ : 0.0;
        return c;
    }

    /// Expected information gain, in nats, from a manoeuvre that would add `delta` of excitation over
    /// `T_s` seconds — the quantity a calibration affordance must advertise, and the one that makes it
    /// self-extinguishing: as the posterior sharpens this falls to nothing on its own.
    ///
    /// ★AND IT IS THE MARGINAL GAIN THAT MATTERS. Pass the excitation the robot's ordinary tours are
    /// expected to deliver over the same horizon as `passive_delta`; what comes back is what the
    /// DELIBERATE manoeuvre adds beyond the free data. A robot whose day is full of turns should never
    /// be asked to spin on purpose.
    [[nodiscard]] double expected_information_gain(double delta, double T_s,
                                                   double passive_delta = 0.0) const
    {
        if (not (T_s > 0.0)) return 0.0;
        const double w = 1.0 / T_s;
        const double sig2 = std::max(density_squared(), 1e-12);
        const double base = sxx_ / sig2 + 1.0 / (p_.prior_std * p_.prior_std);
        const double i_passive = base + w * passive_delta * passive_delta / sig2;
        const double i_active  = base + w * (passive_delta * passive_delta + delta * delta) / sig2;
        if (i_passive <= 0.0 or i_active <= i_passive) return 0.0;
        return 0.5 * std::log(i_active / i_passive);      // ½ log |Σ_before| / |Σ_after|
    }

private:
    /// The MAP slope: data and prior combined in one scaling. Prior mean is 0 — "no scale error
    /// until something says otherwise" — so the prior only ever pulls the estimate toward honesty.
    [[nodiscard]] double posterior_mean() const
    {
        const double sig2 = std::max(density_squared(), 1e-12);
        const double prec = sxx_ / sig2 + 1.0 / (p_.prior_std * p_.prior_std);
        return (prec > 1e-18) ? (sxy_ / sig2) / prec : 0.0;
    }
    /// The current density estimate, squared; falls back to the residual-free case sensibly.
    /// Conjugate prior on the variance: the measured residual sum, plus n0 windows' worth of the
    /// prior density, over the matching count. n0 = 0 recovers the plain unregularised estimator.
    /// ★This also removes a failure the 1e-12 floor left open: a channel fed windows with no motion
    /// in them accumulates rss ~ 0 and used to conclude its measurements were near-perfect, which
    /// made every future manoeuvre look infinitely informative. Zero motion is not evidence of a
    /// quiet sensor; it is no evidence at all, and the prior is what says so.
    [[nodiscard]] double density_squared() const
    {
        const double n0 = std::max(p_.prior_density_windows, 0.0);
        const double d0 = p_.prior_density * p_.prior_density;
        const double dof = static_cast<double>(n_ > 0 ? n_ - 1 : 0) + n0;
        if (dof <= 0.0) return (n_ < 2) ? std::max(d0, 1e-12) : 1.0;
        const double rss = std::max(syy_ - s_ * sxy_, 0.0);
        return std::max((rss + n0 * d0) / dof, 1e-12);
    }

    Params p_;
    double sxx_ = 0.0, sxy_ = 0.0, syy_ = 0.0, s_ = 0.0;
    double span_lo_ = 0.0, span_hi_ = 0.0;
    int    n_ = 0;
};

}   // namespace rc::calib
