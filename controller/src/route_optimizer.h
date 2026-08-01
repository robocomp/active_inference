/*
 * route_optimizer.h — variational optimisation of a route, over the B-spline CONTROL POLYGON.
 *
 * THE DECISION VARIABLE IS THE CONTROL POLYGON, not the curve samples. The samples are a deterministic
 * linear map of the control points followed by an arc-length resample, and RouteSpline::position_at
 * indexes s/spacing — it ASSUMES uniform arc-length spacing, so perturbing samples directly breaks the
 * parameterisation the whole follower rests on. Optimising the control points also makes the Jacobian
 * exact and free: the B-spline map is linear, so d c(u) / d p_i is just the basis function N_i(u).
 * For a 35 m route at h = 0.40 m that is ~89 control points = 178 free parameters, against 1428 samples.
 *
 * The objective (all terms DIMENSIONLESS, so the weights are pure numbers of order one — written with
 * their natural units the terms would carry m^-1, m^3 and m^2 and could not be compared at all):
 *
 *     S = w_kappa · E_kappa  +  w_clear · E_clear  +  E_anchor  +  w_gauge · E_gauge
 *
 *   E_kappa  = mean (rho·kappa)^2            PRIOR      — bending energy; rho normalises curvature
 *   E_clear  = mean (deficit / d*)^2         PREFERENCE — one-sided: clear routes pay nothing
 *   E_anchor = mean (|c - w_j| / sigma_a)^2  LIKELIHOOD — the authored waypoints are the DESIRE
 *   E_gauge  = mean ((|dp| - h)/h)^2         GAUGE      — not a preference; see below
 *
 * ON E_kappa: for a uniform cubic B-spline the second derivative is the linear interpolation of the
 * second DIFFERENCES of the control points, so the bending energy is a quadratic form in P and its
 * Gauss-Newton block is constant. (We use the P-spline form — identity in place of the tridiagonal
 * (1/6)[1,4,1] Gram of the linear basis. Both are valid priors; the identity is the standard penalised
 * -spline choice and costs one matrix.)
 *
 * ON E_clear: one-sided ascent on the distance field. In a corridor the RIDGE of that field is the
 * medial axis, so "move away from obstacles" and "pass equidistant in narrow zones" are the SAME term —
 * equidistance is a consequence of the model, not a narrow-passage rule bolted on beside it.
 *
 * ON E_anchor: without it the minimiser of the first two terms is a wide circle in the middle of the
 * room — maximal clearance, minimal curvature, visits nothing. It anchors to the AUTHORED waypoints,
 * never to the A* polyline: A* fixes the homotopy class and the initialisation, but its vertices are a
 * grid artefact and carry no intent.
 *
 * ON E_gauge: everything above assumes the control points sit at uniform arc length, which holds at
 * initialisation (they are resampled there) and stops holding after the first step, because nothing in
 * the other three terms constrains their SPACING. Left alone the parameterisation degrades and the
 * curvature term silently stops being the one you believe you are minimising. This removes a
 * reparameterisation freedom the objective is blind to; it carries a small weight and must never be
 * tuned for behaviour.
 *
 * WHAT IT CANNOT DO: it refines geometry WITHIN the homotopy class A* chose. Gradient descent cannot
 * cross an obstacle, so it will not discover that the other side of the table is better. That is the
 * intended division of labour — A* owns discrete topology, this owns continuous geometry — but a route
 * stuck in a bad homotopy class will come out beautifully smooth, well cleared, and wrong.
 *
 * Pure Eigen/STL — no grid, no Qt, no DSR. The distance field arrives as two callables, so this is
 * unit-testable against an analytic corridor with no GridPlanner in sight.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

struct RouteOptimizerConfig
{
    // Distance (metres) to the nearest obstacle at a room-frame point, and its gradient. Supply an
    // EXACT field: an optimiser follows the gradient of whatever it is handed, so a chamfer's
    // direction-dependent error is baked into the SHAPE of the result rather than merely mis-scoring it.
    std::function<float(const Eigen::Vector2f &)> distance;
    std::function<Eigen::Vector2f(const Eigen::Vector2f &)> distance_gradient;

    std::vector<Eigen::Vector2f> anchors;   // authored waypoints, in route order (laps concatenated)
    // Arc length of each anchor along the route. Binding an anchor to a control point by NEAREST
    // POSITION is the same trap as a global carrot search: after a repair splices the polyline, a
    // waypoint can bind to a control point metres away in arc length, and with a quadratic penalty a
    // 2.5 m orphan is worth (2.5/0.30)^2 = 69 — which is what relocated control points 24.8 m and
    // wrecked a live route. Empty ⇒ fall back to the forward-only nearest search.
    std::vector<float> anchor_s;

    float h         = 0.40f;   // control spacing == the prior's length scale
    float d_target  = 0.60f;   // body reach + comfort standoff; deficit below this is what is penalised
    float rho       = 0.49f;   // curvature normaliser, v_max^2 / a_lat_max
    float sigma_a   = 0.30f;   // how far the route may drift from what was clicked
    // Below this the route is not drivable at all, so a solve that lands there has failed rather than
    // traded. Distinct from d_target, which is a PREFERENCE the anchor term is allowed to outvote:
    // a waypoint pulling the route off the medial axis costs clearance BY DESIGN, and a guard that
    // rejected any clearance regression would forbid the likelihood term from ever winning. 0 disables.
    float clearance_floor = 0.f;

    float w_kappa   = 1.0f;
    float w_clear   = 1.0f;
    float w_gauge   = 0.05f;
    // Huber knee for the anchor term, in units of sigma_a. Beyond it the pull grows LINEARLY instead of
    // quadratically, so an anchor that cannot be honoured bends the route rather than tearing it apart.
    float anchor_huber = 2.0f;
    // ★TRIED AND REJECTED 2026-08-01: auto-normalising each term by its value on the initial route.
    // It improved the corridor case (0.175 -> 0.133 m) but BROKE IDEMPOTENCE — re-optimising an optimised
    // route moved it 0.165 m, because the scales are recomputed from whatever route is handed in, so the
    // objective is not a fixed function of the geometry. That is the MPPI's `lambda_used` mistake wearing
    // a different hat, and it is left here as a warning rather than a switch. The imbalance it was meant
    // to fix (measured kappa 0.163 vs clear 0.031, i.e. 5:1) is corrected by a STATED weight instead.
    bool  normalise_terms = false;
    // How hard the WORST sample dominates. The objective's clearance term is u^2 + clear_peak * u^4 with
    // u = deficit/d_target in [0,1]: the quadratic drives the bulk of the route toward clearance, the
    // quartic makes a single tight point outweigh many mildly-tight ones. Pure u^2 (the original) cannot
    // express "do not pass close to ANYTHING" — measured on the real tour, it LOWERED the cost while
    // taking min clearance from 0.454 to 0.396. Pure u^4 over-corrects: its gradient dies as u^3, so on a
    // uniformly-tight route it stalls 0.175 m off the medial axis where u^2 reaches 0.042 m.
    float clear_peak = 4.0f;

    int   iterations = 30;
    // Pin control points [0, freeze_before) in place. Used on a REPAIR: re-optimising the whole route
    // while the robot is driving it moves the curve under the robot and steps the cross-track error.
    // With local support, freezing a prefix is a clean restriction of the variable set — the same
    // mechanism as the always-pinned endpoints, not an approximation.
    std::size_t freeze_before = 0;
    bool  enabled = true;      // off ⇒ RouteSpline fits exactly as before
    bool  verbose = true;
};

struct RouteOptimizerReport
{
    bool  ran = false;
    bool  rejected = false;    // solved, but the result failed the acceptance test and was reverted
    int   iterations = 0;
    float scale_kappa = 1.f, scale_clear = 1.f, scale_anchor = 1.f;   // the normalisers actually applied
    float cost_before = 0.f;
    float cost_after = 0.f;
    float max_move_m = 0.f;       // largest displacement of any control point
    float min_clearance_before = 0.f;
    float min_clearance_after = 0.f;
    // PER-TERM cost at the end. Without this a bad solve is undiagnosable: a big total says nothing about
    // WHICH term won, and a bounded term sitting next to an unbounded one is inert without ever saying so.
    float e_kappa = 0.f, e_clear = 0.f, e_anchor = 0.f, e_gauge = 0.f;
};

// Optimise `ctrl` in place. Endpoints are always pinned (the route must still start at the robot and
// end where the tour ends). Returns a report; `ran` is false if the problem was degenerate (fewer than
// 4 control points, no distance field, or everything frozen), in which case `ctrl` is untouched.
RouteOptimizerReport optimize_route(std::vector<Eigen::Vector2f> &ctrl, const RouteOptimizerConfig &cfg);

bool route_optimizer_self_test();

}  // namespace rc
