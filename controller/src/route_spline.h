/*
 * route_spline.h — a route as ONE continuous, curvature-continuous curve parameterised by arc length.
 *
 * WHY THIS REPLACES THE WAYPOINT/CARROT/ARRIVAL SCHEME
 * The follower used to be handed the path to the NEXT WAYPOINT, which made every waypoint a goal. Three
 * things followed, all of them measured:
 *   - it decelerated into every waypoint (the nominal seed scales speed by carrot distance inside 1 m,
 *     and the straight-speed override switches off inside 1.5 m). With 1.28 m legs the robot was in the
 *     arrival regime for the entire tour and never reached cruise: 0.30 m/s against a 0.7 m/s limit.
 *   - set_path wiped the MPPI warm start at each waypoint, so continuity was destroyed 30 times a lap.
 *   - extending the path instead made the carrot aim past the waypoint, so the robot cut the corner and
 *     never entered the arrival radius — the leg never completed and it drove back and forth.
 * Patching arrival (proximity, then passage) was patching the wrong thing: cutting the corner is what a
 * smooth route DOES, so any test that asks "did we touch the waypoint" is asking the wrong question.
 *
 * Here the waypoints only DEFINE the route. Once built there are no waypoints, no arrival tests and no
 * legs during execution — there is a curve, an arc-length coordinate, and a carrot a fixed distance
 * ahead of the robot's projection onto it. Progress is a scalar that only increases.
 *
 * WHY C² MATTERS SPECIFICALLY
 * Commanded rotation rate follows path curvature. A C¹ curve (the previous centripetal Catmull-Rom) has
 * DISCONTINUOUS curvature at every knot, so the steering command steps at each one — which is jerk, and
 * it is what the smoothness metric penalises. A uniform cubic B-spline is C² by construction, so
 * curvature is continuous and the steering command has no steps to make. It is also APPROXIMATING
 * rather than interpolating: the curve stays inside the convex hull of its control points, so it cannot
 * overshoot a hand-clicked corner into a wall the way an interpolating spline does.
 *
 * FEASIBILITY IS NOT ASSUMED. Smoothing moves the curve off the planned polyline, and the planned
 * polyline is the only thing known to be footprint-feasible. Every sample is therefore re-tested with
 * the caller's own predicate (the grid planner's exact footprint test) and pulled back toward the
 * polyline until it passes. A smoothed route that the planner would refuse to drive is worse than an
 * unsmoothed one.
 *
 * Pure Eigen/STL — no Qt, no DSR → unit-testable in isolation (self_test()).
 */

#pragma once

#include "route_optimizer.h"

#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

class RouteSpline
{
public:
    // `polyline` must already be footprint-feasible (it comes from the grid planner). `spacing` is the
    // arc-length step of the output samples. `is_free(pos, heading)` is the caller's feasibility test;
    // pass an empty function to skip the feasibility pass (tests only).
    // `smoothing_m` is the SCALE of the smoothing — the spacing of the B-spline's control points, and
    // therefore roughly the radius a corner gets rounded to. It is deliberately independent of
    // `spacing`, which is only the output resolution: tying them together (the first version did) means
    // a 5 cm output step can only round a corner over 5 cm, which is not smoothing, it is resampling.
    // `opt`, when non-null and enabled, VARIATIONALLY OPTIMISES the control polygon before the curve is
    // evaluated — bending prior + one-sided clearance + waypoint fidelity + gauge. See route_optimizer.h.
    // Null (the default) reproduces the previous behaviour exactly: fit, resample, feasibility-check.
    bool build(const std::vector<Eigen::Vector2f> &polyline, float spacing,
               const std::function<bool(const Eigen::Vector2f &, float)> &is_free,
               float smoothing_m = 0.40f,
               const RouteOptimizerConfig *opt = nullptr);

    // The optimiser's report from the last build. Kept so the caller can WRITE IT DOWN: a diagnostic that
    // only exists on stdout is a diagnostic nobody can compare across runs.
    const RouteOptimizerReport &last_optimizer_report() const { return last_opt_; }

    // ── LOCAL ELASTIC BAND ────────────────────────────────────────────────────────────────────────
    // Re-optimise the RETAINED control polygon in place against whatever field `opt` carries — normally
    // the LIVE local ESDF, not the static planner field the route was built with — and re-evaluate the
    // curve. The caller restricts the variable set with opt.freeze_before/freeze_after; with local
    // support that is an exact restriction, not an approximation, so everything outside the window keeps
    // exactly the geometry it had.
    //
    // ★The caller MUST freeze the control points behind the robot. The arc-length parameterisation is
    // measured from s=0, so if the curve moves behind the robot every downstream number that indexes it
    // — progress, waypoint arc lengths, the carrot — silently shifts meaning.
    // Returns a report with ran=false if there is nothing to solve (no retained polygon, no field).
    RouteOptimizerReport deform(const RouteOptimizerConfig &opt);
    const std::vector<Eigen::Vector2f> &control_points() const { return ctrl_; }
    float control_spacing() const { return ctrl_step_; }

    bool  valid() const { return samples_.size() >= 2; }
    float length() const { return length_; }
    float spacing() const { return spacing_; }
    const std::vector<Eigen::Vector2f> &samples() const { return samples_; }

    Eigen::Vector2f position_at(float s) const;
    float heading_at(float s) const;      // atan2(dy,dx) of the tangent
    float curvature_at(float s) const;    // 1/radius, signed

    // SIGNED AVERAGE curvature over a window of `window_m` CENTRED on s: the net heading change across
    // the window divided by its length, which is the mean value theorem applied honestly.
    //
    // ★PREFER THIS TO curvature_at FOR ANYTHING THAT ACTS ON THE ROBOT. curvature_at is a SECOND
    // difference of positions on a piecewise-linear 5 cm resample; this is a FIRST difference of
    // headings. One order less differentiation is one order less noise amplification, and a lone 5 cm
    // spike stops mattering BY CONSTRUCTION (it contributes almost nothing to a net heading change)
    // rather than by being filtered afterwards — while a sustained tight curve accumulates its full
    // turn and still binds. The natural window is the route's own smoothing scale (route_smoothing_m,
    // 0.40 m): the curve was fitted with control points that far apart, so curvature structure finer
    // than that is a property of the fit, not of the route.
    //
    // ★CENTRED, not forward-looking. A forward window is itself a half-window of preview, which would
    // silently add to any explicit lookahead a caller applies and double-count it. A caller that WANTS
    // the forward window [s, s+W] asks for kappa_avg(s + W/2, W) and says so.
    float kappa_avg(float s, float window_m) const;

    // Arc length of the closest point, searched FORWARD from `s_hint` within `window_m`. Monotone by
    // construction: a route that crosses itself (this tour does) would otherwise let the projection jump
    // to a distant branch and teleport progress backwards or forwards.
    float project(const Eigen::Vector2f &p, float s_hint, float window_m = 3.0f) const;

    // How many of the build's samples had to be pulled back to stay feasible — a smoothing that fights
    // the geometry everywhere is a signal, not a detail.
    int corrections() const { return corrections_; }

    // How far the fitted curve strays from the PLANNED POLYLINE it was fitted to. The B-spline is
    // APPROXIMATING, so some deviation is by design (that is what rounds a corner) — but "by design"
    // needs a number, or "the smooth path does not follow the route" cannot be answered. Distance from
    // each curve sample to the nearest point of the polyline, sampled every 4th point.
    float max_deviation_m() const { return max_dev_; }
    float mean_deviation_m() const { return mean_dev_; }

    static bool self_test();

private:
    // Steps 2-4 of build() — B-spline evaluation, arc-length resample, feasibility pullback — as a pure
    // function of ctrl_. Shared by build() and deform() so a deformed curve is produced by exactly the
    // same pipeline as a freshly built one; there is no second code path to drift.
    // `measure_deviation` = recompute max_dev_/mean_dev_ (O(samples x polyline), build-time only).
    bool evaluate_from_ctrl(bool measure_deviation);

    // ── The decision variable, retained ──
    std::vector<Eigen::Vector2f> ctrl_;       // B-spline control polygon
    float ctrl_step_ = 0.40f;                 // its spacing == the bending prior's length scale
    std::vector<Eigen::Vector2f> polyline_;   // what it was fitted to; the feasibility pullback target
    std::function<bool(const Eigen::Vector2f &, float)> is_free_;   // the caller's footprint test

    std::vector<Eigen::Vector2f> samples_;   // uniform in arc length, spacing_ apart
    float spacing_ = 0.1f;
    float length_ = 0.f;
    int   corrections_ = 0;
    RouteOptimizerReport last_opt_;
    float max_dev_ = 0.f, mean_dev_ = 0.f;
};

// ── WHAT THE ROUTE ITSELF MAKES UNAVOIDABLE ──────────────────────────────────────────────────────
// The mission cost J = smooth_lin + smooth_rot + dev_norm answers "how did this run go", not "how good
// is this controller", because every one of its terms is dominated by the ROUTE rather than by the
// tracker: a straight route scores ~0 on smooth_rot for anybody, and a cusped one produces cross-track
// error for anybody. So J cannot be compared across missions, and a gain tuned on one tour does not
// transfer to another.
// These three quantities are the route's own floor, computed from the spline and the IDENTIFIED plant
// with no robot involved:
//   v*(s) = the geometric speed profile (the same curvature and lateral-accel limits route_speed_limit
//           applies), w*(s) = v*(s)*kappa(s), and e*(s) = (v*(s)*T_lag)^2*|kappa(s)|/2, the lateral
//           offset a PERFECT tracker still incurs because the actuator lags by T_lag.
// Dividing a run's measured totals by these gives a J' whose terms are each >= 1, where 1 means "as
// well as the plant permits on this route" — dimensionless, and comparable across routes.
// ⚠On a near-straight stretch tv_w -> 0 and the ratio explodes, so samples below a floor are dropped
// rather than allowed to dominate; `w_span` reports how much of the route actually contributed.
struct RouteIdeal
{
    float tv_v  = 0.f;    // total variation of v*(s) — irreducible speed changes
    float tv_w  = 0.f;    // total variation of w*(s) — irreducible turn-rate changes
    float rms_e = 0.f;    // rms of e*(s) — the lag-induced error floor
    float length_m = 0.f;
    float w_span = 0.f;   // metres of route where |w*| was above the floor and the tv_w ratio is meaningful
    bool  valid = false;
};

// v_cap/a_lat/a_dec/w_max/headroom mirror ControllerSession::route_speed_limit; W is the curvature
// window (plain_W) and T_lag the identified actuator lag.
// ★sharp_q is NOT optional, deliberately. It is the curvature-dependent rotation budget
// (ControllerRuntimeParams::sharp_turn_slowdown), and v* is the DENOMINATOR every J_route term is
// divided by: computing the floor under a speed law the robot is not being driven at would silently
// rescale every mission metric rather than fail. No default, so a new caller has to supply it.
RouteIdeal route_ideal(const RouteSpline &sp, float v_cap, float a_lat, float a_dec,
                       float w_max, float W, float T_lag, float headroom, float sharp_q,
                       float body_radius_m);

}  // namespace rc
