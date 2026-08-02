/*
 * route_optimizer.cpp — see route_optimizer.h
 */

#include "route_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace rc
{

namespace
{

// The four uniform cubic B-spline basis weights at t in [0,1). Same polynomial as RouteSpline::bspline,
// factored out so the OPTIMISER and the EVALUATOR cannot drift apart: the Jacobian below is literally
// these numbers, so if the curve is ever re-based the gradient follows automatically.
std::array<float, 4> basis(float t)
{
    const float t2 = t * t, t3 = t2 * t;
    return {(1.f - 3.f * t + 3.f * t2 - t3) / 6.f,
            (4.f - 6.f * t2 + 3.f * t3) / 6.f,
            (1.f + 3.f * t + 3.f * t2 - 3.f * t3) / 6.f,
            t3 / 6.f};
}

// First and second derivatives of the same four basis functions, with respect to the span parameter t.
// The curve is c(t) = sum B_k(t) p_k, so c'(t) and c''(t) are these weights applied to the SAME four
// control points — which is why the exact-curvature Jacobian below needs nothing beyond them.
std::array<float, 4> basis_d1(float t)
{
    const float t2 = t * t;
    return {(-3.f + 6.f * t - 3.f * t2) / 6.f,
            (-12.f * t + 9.f * t2) / 6.f,
            (3.f + 6.f * t - 9.f * t2) / 6.f,
            3.f * t2 / 6.f};
}

std::array<float, 4> basis_d2(float t)
{
    return {(6.f - 6.f * t) / 6.f,
            (-12.f + 18.f * t) / 6.f,
            (6.f - 18.f * t) / 6.f,
            6.f * t / 6.f};
}

// RouteSpline duplicates the first and last control point twice so the curve starts and ends where the
// route does. The optimiser must see the SAME padded polygon, or its idea of the curve is not the curve
// that gets driven. This maps a padded index back to the control point that owns it — the duplicated
// ends therefore accumulate several basis contributions, which is correct and is why this is a mapping
// rather than an offset.
std::size_t pad_to_ctrl(std::size_t j, std::size_t M)
{
    if (j <= 1) return 0;
    if (j >= M + 2) return M - 1;
    return j - 2;
}

// ── CLAMPED-B-SPLINE CURVATURE UPPER BOUND ───────────────────────────────────────────────────────
// K_i = sin(alpha) / (6 l) * ((1 - cos alpha)/8)^(-3/2), with alpha the angle at control point i and
// l = min(|Q_i - Q_{i-1}|, |Q_{i+1} - Q_i|)   [Elbanhawi et al. 2015; used by Choi et al. arXiv:2311.02957]
//
// WHY THIS RATHER THAN |d2 Q| / h^2 OR THE EXACT |c' x c''| / |c'|^3:
//  • it is an UPPER BOUND over the whole span, not a value at a sample — so "no corner tighter than R"
//    becomes something the objective can state, instead of something measured afterwards and hoped for;
//  • l is the MEASURED shorter segment, not an assumed uniform spacing h, so it does not silently stop
//    meaning curvature when the control points bunch up. That assumption is what made w_gauge behave
//    like a behavioural knob and sent a whole afternoon into the exact-curvature dead end;
//  • it costs no quadrature at all.
// Conservative by construction: on a 90-degree corner at h = 0.40 it reports 9.43 1/m where the curve's
// true peak is 7.07 — a bound, and about 33% loose. That is the price of it being a guarantee.
struct CurvBound
{
    float K = 0.f;
    Eigen::Vector2f dK_da{0.f, 0.f}, dK_db{0.f, 0.f};   // a = Q_{i-1} - Q_i,  b = Q_{i+1} - Q_i
};

CurvBound curvature_bound(const Eigen::Vector2f &a, const Eigen::Vector2f &b, float min_seg)
{
    CurvBound out;
    const float A = a.norm(), B = b.norm();
    if (A < 1e-6f or B < 1e-6f) return out;
    const float invAB = 1.f / (A * B);
    const float dot = a.dot(b);
    const float cross = a.x() * b.y() - a.y() * b.x();
    const float c = std::clamp(dot * invAB, -1.f, 1.f);
    const float s = std::abs(cross) * invAB;
    // alpha -> 0 is the polygon doubling back on itself: the bound is genuinely infinite there, so it is
    // clamped rather than allowed to produce an inf that would poison the whole solve. A route that
    // reverses inside one control interval is not a route the optimiser can rescue.
    const float omc = std::max(1e-4f, 1.f - c);
    // L_min: a vanishing segment sends the bound to infinity for a corner that may be perfectly drivable.
    // Choi et al. clip it for exactly this reason and report that the overestimate costs little.
    const float l = std::max(min_seg, std::min(A, B));
    constexpr float C = 22.627417f / 6.f;              // 8^(3/2) / 6
    const float omc32 = omc * std::sqrt(omc);
    out.K = C * s / (l * omc32);

    // Gradients. df/dc and df/ds for f = s * (1-c)^(-3/2), then chain through c(a,b) and s(a,b).
    const float f = s / omc32;
    const float df_dc = 1.5f * s / (omc32 * omc);
    const float df_ds = 1.f / omc32;
    const Eigen::Vector2f ah = a / A, bh = b / B;
    const Eigen::Vector2f dc_da = b * invAB - c * ah / A;
    const Eigen::Vector2f dc_db = a * invAB - c * bh / B;
    const float sgn = cross >= 0.f ? 1.f : -1.f;
    const Eigen::Vector2f dcross_da{sgn * b.y(), -sgn * b.x()};
    const Eigen::Vector2f dcross_db{-sgn * a.y(), sgn * a.x()};
    const Eigen::Vector2f ds_da = dcross_da * invAB - s * ah / A;
    const Eigen::Vector2f ds_db = dcross_db * invAB - s * bh / B;
    // l is a min() and is clipped, so it is piecewise: zero gradient where the clip binds or where this
    // segment is not the shorter one. A kink in the objective, not a discontinuity in it.
    const bool l_live = std::min(A, B) > min_seg;
    const Eigen::Vector2f dl_da = (l_live and A <= B) ? ah : Eigen::Vector2f{0.f, 0.f};
    const Eigen::Vector2f dl_db = (l_live and B < A) ? bh : Eigen::Vector2f{0.f, 0.f};

    out.dK_da = C * ((df_dc * dc_da + df_ds * ds_da) / l - (f / (l * l)) * dl_da);
    out.dK_db = C * ((df_dc * dc_db + df_ds * ds_db) / l - (f / (l * l)) * dl_db);
    return out;
}

struct Problem
{
    // Row ranges per term, so a solve can be split into WHICH term is paying. A total tells you nothing
    // about whether a bounded term is being swamped by an unbounded one.
    int row_bend = 0, row_clear = 0, row_anchor = 0, row_gauge = 0;
    // Residual rows and the sparse-by-construction Jacobian, held dense: at ~600 rows x ~180 columns the
    // dense normal equations are ~180^3/3 flops, i.e. microseconds. Banded storage would be faster and
    // considerably easier to get wrong, and this runs once per route build.
    Eigen::VectorXf r;
    Eigen::MatrixXf J;
};

constexpr int kQuadPerSpan = 3;

// Index of the first free variable column for control point i, or -1 if it is pinned.
int var_of(std::size_t i, std::size_t lo, std::size_t hi)
{
    if (i < lo or i > hi) return -1;
    return static_cast<int>(2 * (i - lo));
}

struct TermScales { float kappa = 1.f, clear = 1.f, anchor = 1.f; };

// The speed<->safety dial, as a log-symmetric split of precision between the two terms. exp(2*ln(3)*(b-0.5))
// spans a factor of 9 in the RATIO across the dial (x3 each way at the ends), which is wide enough to
// change the route's character and narrow enough that neither term ever goes inert.
float bias_gain(float bias)
{
    constexpr float kBiasSpan = 2.1972246f;     // 2 * ln(3)
    return std::exp(kBiasSpan * (std::clamp(bias, 0.f, 1.f) - 0.5f));
}

void assemble(const std::vector<Eigen::Vector2f> &ctrl,
              const RouteOptimizerConfig &cfg,
              const std::vector<std::size_t> &anchor_ctrl,
              std::size_t lo, std::size_t hi,
              const TermScales &sc_t,
              Problem &out,
              float *min_clearance)
{
    const std::size_t M = ctrl.size();
    const int nvar = static_cast<int>(2 * (hi - lo + 1));
    const float bias_g = bias_gain(cfg.safety_bias);

    // Padded control polygon, exactly as RouteSpline builds it.
    std::vector<Eigen::Vector2f> pad;
    pad.reserve(M + 4);
    pad.push_back(ctrl.front()); pad.push_back(ctrl.front());
    pad.insert(pad.end(), ctrl.begin(), ctrl.end());
    pad.push_back(ctrl.back()); pad.push_back(ctrl.back());

    const std::size_t n_spans = pad.size() - 3;
    const std::size_t n_bend = (M >= 3) ? M - 2 : 0;
    const std::size_t n_quad = n_spans * kQuadPerSpan;
    const std::size_t n_anch = anchor_ctrl.size();
    const std::size_t n_gauge = (M >= 2) ? M - 1 : 0;

    // The bound is a SCALAR per interior control point; the second difference is a vector, hence two rows.
    const std::size_t n_bend_rows = cfg.curvature_bound ? n_bend : 2 * n_bend;
    const int rows = static_cast<int>(n_bend_rows + n_quad + 2 * n_anch + n_gauge);
    out.r.setZero(rows);
    out.J.setZero(rows, nvar);

    int row = 0;
    if (min_clearance) *min_clearance = std::numeric_limits<float>::max();
    int mark = 0;

    // ── PRIOR: bending. r = sqrt(w/M)·(rho/h^2)·(p[i-1] - 2p[i] + p[i+1])·alpha(v), v = rho·kappa.
    //    With kappa_peak = 0, alpha = 1 and this is the plain quadratic bending energy, linear in P and
    //    therefore an exact Gauss-Newton block. With kappa_peak > 0 it is the SAME construction the
    //    clearance term uses (see clear_peak): the residual carries sqrt(1 + pk·v^2), so the objective is
    //    v^2 + pk·v^4 and one sharp corner outweighs many gentle ones. The Jacobian below is the exact
    //    derivative of that product, which is no longer diagonal in (x,y) — the extra term is the
    //    curvature of the norm, and dropping it (IRLS) would halve the pull on exactly the worst corner.
    if (n_bend > 0 and cfg.curvature_bound)
    {
        // Same residual shape as below — r = sc·v·sqrt(1 + pk·v²), v = rho·kappa, dimensionless — with
        // kappa taken from the per-span UPPER BOUND instead of the second difference. One scalar row per
        // interior control point; the Jacobian is the bound's own gradient, chained through a = Q_{i-1}-Q_i
        // and b = Q_{i+1}-Q_i.
        const float sc = std::sqrt(sc_t.kappa * (cfg.w_kappa / bias_g) / static_cast<float>(n_bend));
        const float pk = std::max(0.f, cfg.kappa_peak);
        for (std::size_t i = 1; i + 1 < M; ++i)
        {
            const Eigen::Vector2f a = ctrl[i - 1] - ctrl[i], b = ctrl[i + 1] - ctrl[i];
            const CurvBound cb = curvature_bound(a, b, cfg.min_seg_m);
            const float v = cfg.rho * cb.K;
            const float alpha = std::sqrt(1.f + pk * v * v);
            // HUBER, in the same IRLS form the anchor term uses, and for the same reason. The bound is
            // UNBOUNDED where the control polygon doubles back (alpha -> 0 gives K -> infinity), and our
            // tour does exactly that at its authored hairpins. Unsaturated, one impossible corner is worth
            // more than the whole rest of the route: measured, the curvature term reached 2.92 of a total
            // 3.62 and the trust region collapsed on the first step — 0 iterations, nothing optimised.
            // Beyond the knee the residual grows as sqrt(v), so an infeasible corner BENDS the route
            // instead of monopolising it, and a corner that no route can fix stops pretending it can.
            const float knee = 3.f;                     // in units of v, i.e. 3x the full-speed curvature
            const float hw = (v <= knee) ? 1.f : std::sqrt(knee / std::max(v, 1e-6f));
            out.r(row) = sc * hw * v * alpha;
            const float drdv = sc * hw * (1.f + 2.f * pk * v * v) / alpha;
            const Eigen::Vector2f g_a = drdv * cfg.rho * cb.dK_da;
            const Eigen::Vector2f g_b = drdv * cfg.rho * cb.dK_db;
            // dK/dQ_{i-1} = dK/da,  dK/dQ_{i+1} = dK/db,  dK/dQ_i = -(dK/da + dK/db)
            if (const int v0 = var_of(i - 1, lo, hi); v0 >= 0)
            { out.J(row, v0) += g_a.x(); out.J(row, v0 + 1) += g_a.y(); }
            if (const int v1 = var_of(i, lo, hi); v1 >= 0)
            { out.J(row, v1) -= g_a.x() + g_b.x(); out.J(row, v1 + 1) -= g_a.y() + g_b.y(); }
            if (const int v2 = var_of(i + 1, lo, hi); v2 >= 0)
            { out.J(row, v2) += g_b.x(); out.J(row, v2 + 1) += g_b.y(); }
            ++row;
        }
    }
    else if (n_bend > 0)
    {
        const float sc = std::sqrt(sc_t.kappa * (cfg.w_kappa / bias_g) / static_cast<float>(n_bend)) * cfg.rho / (cfg.h * cfg.h);
        const float pk = std::max(0.f, cfg.kappa_peak);
        for (std::size_t i = 1; i + 1 < M; ++i)
        {
            const Eigen::Vector2f d2 = ctrl[i - 1] - 2.f * ctrl[i] + ctrl[i + 1];
            const float n2 = d2.squaredNorm();
            // v is the DIMENSIONLESS curvature rho·kappa: 1 exactly when the turn is at the radius the
            // robot can hold at full speed (rho = v_max^2/a_lat), so the quartic bites past that and not
            // at some radius chosen for it.
            const float v = cfg.rho * std::sqrt(n2) / (cfg.h * cfg.h);
            const float alpha = std::sqrt(1.f + pk * v * v);
            const float beta = (n2 > 1e-12f) ? pk * v * v / (alpha * n2) : 0.f;   // d(alpha)/d(d2) factor
            const std::array<std::pair<std::size_t, float>, 3> nbr =
                {{{i - 1, 1.f}, {i, -2.f}, {i + 1, 1.f}}};
            for (int a = 0; a < 2; ++a)
            {
                out.r(row + a) = sc * alpha * d2[a];
                for (const auto &[j, coeff] : nbr)
                {
                    const int vj = var_of(j, lo, hi);
                    if (vj < 0) continue;
                    for (int b = 0; b < 2; ++b)
                    {
                        const float k_ab = (a == b ? alpha : 0.f) + beta * d2[a] * d2[b];
                        out.J(row + a, vj + b) += sc * k_ab * coeff;
                    }
                }
            }
            row += 2;
        }
    }
    out.row_bend = row - mark; mark = row;

    // ── PREFERENCE: one-sided clearance deficit, sampled along the CURVE (not at the control points —
    //    the curve can cut inside a corner of its own polygon and pass closer than any vertex does).
    {
        // ── p = 4, NOT a mean of squares ─────────────────────────────────────────────────────────
        // A mean over ~270 quadrature samples cannot express "do not pass close to ANYTHING": a sharp
        // corner is one sample, so improving many mildly-tight points while ruining the worst one LOWERS
        // the objective. Measured on the real tour: cost fell while min clearance went 0.454 -> 0.396.
        // The requirement is a MINIMUM; the residual therefore carries the SQUARE of the normalised
        // deficit, so the objective is its fourth power and the worst sample dominates — an L4 norm,
        // which approaches the max while staying smooth and still a sum of squares (Gauss-Newton applies).
        // The elastic band gets this right by accident: it pushes each point individually.
        const float sc = std::sqrt(sc_t.clear * (cfg.w_clear * bias_g) / static_cast<float>(std::max<std::size_t>(1, n_quad)));
        const float dstar = std::max(0.05f, cfg.d_target);
        for (std::size_t s = 0; s < n_spans; ++s)
            for (int q = 0; q < kQuadPerSpan; ++q)
            {
                const float t = (static_cast<float>(q) + 0.5f) / static_cast<float>(kQuadPerSpan);
                const auto B = basis(t);
                Eigen::Vector2f c = Eigen::Vector2f::Zero();
                for (int k = 0; k < 4; ++k) c += B[k] * pad[s + k];

                const float d = cfg.distance ? cfg.distance(c) : 1e6f;
                if (min_clearance) *min_clearance = std::min(*min_clearance, d);
                const float deficit = dstar - d;
                if (deficit <= 0.f) { ++row; continue; }      // clear here: zero residual, zero gradient

                const float u = deficit / dstar;                 // in [0,1]
                // r^2 = u^2 + peak*u^4 — see clear_peak. One residual, both behaviours.
                const float pk = std::max(0.f, cfg.clear_peak);
                const float g_u = std::sqrt(1.f + pk * u * u);
                out.r(row) = sc * u * g_u;
                const Eigen::Vector2f g = cfg.distance_gradient ? cfg.distance_gradient(c)
                                                                : Eigen::Vector2f::Zero();
                // d(deficit)/dc = -grad d, and dc/dp_m = sum of the basis weights that map to m.
                // d/du [ u*sqrt(1+pk*u^2) ] = (1 + 2*pk*u^2) / sqrt(1+pk*u^2);  du/dc = -grad(d)/dstar
                const float dru = -sc * (1.f + 2.f * pk * u * u) / (g_u * dstar);
                for (int k = 0; k < 4; ++k)
                {
                    const int v = var_of(pad_to_ctrl(s + k, M), lo, hi);
                    if (v < 0) continue;
                    out.J(row, v)     += dru * g.x() * B[k];
                    out.J(row, v + 1) += dru * g.y() * B[k];
                }
                ++row;
            }
    }
    out.row_clear = row - mark; mark = row;

    // ── LIKELIHOOD: the authored waypoints. The curve point "at" control point m is span (m+1) at t=0,
    //    i.e. (p[m-1] + 4p[m] + p[m+1]) / 6 through the padding — so this is linear in P as well.
    if (n_anch > 0)
    {
        const float sc = std::sqrt(sc_t.anchor / static_cast<float>(n_anch)) / std::max(1e-3f, cfg.sigma_a);
        const float knee = std::max(0.1f, cfg.anchor_huber);   // in units of sigma_a
        for (std::size_t a = 0; a < n_anch; ++a)
        {
            const std::size_t m = anchor_ctrl[a];
            const std::size_t s = m + 1;                       // padded span whose t=0 sits on ctrl[m]
            const std::array<float, 3> B0 = {1.f / 6.f, 4.f / 6.f, 1.f / 6.f};
            Eigen::Vector2f c = Eigen::Vector2f::Zero();
            for (int k = 0; k < 3; ++k) c += B0[k] * pad[s + k];
            const Eigen::Vector2f e = c - cfg.anchors[a];
            // HUBER. An anchor the route cannot honour — orphaned by a repair, or boxed in — must bend
            // the route, not tear it apart. Quadratic, a 2.5 m orphan at sigma_a = 0.30 is worth 69 and
            // dominates every other term; the observed result was control points relocated 24.8 m.
            // IRLS form: weight w = 1 inside the knee, knee*sigma/|e| outside, so the residual grows as
            // sqrt(|e|) and the pull saturates instead of exploding.
            const float en = e.norm() / std::max(1e-3f, cfg.sigma_a);
            const float hw = (en <= knee) ? 1.f : std::sqrt(knee / en);
            for (int comp = 0; comp < 2; ++comp)
            {
                out.r(row + comp) = sc * hw * e[comp];
                for (int k = 0; k < 3; ++k)
                {
                    const int v = var_of(pad_to_ctrl(s + k, M), lo, hi);
                    if (v >= 0) out.J(row + comp, v + comp) += sc * hw * B0[k];
                }
            }
            row += 2;
        }
    }
    out.row_anchor = row - mark; mark = row;

    // ── GAUGE: keep the control points at spacing h, so s ~ h·u stays true and the curvature term keeps
    //    meaning what it means. Structural, not behavioural.
    if (n_gauge > 0)
    {
        const float sc = std::sqrt(cfg.w_gauge / static_cast<float>(n_gauge));   // gauge is not normalised: it is not a preference
        for (std::size_t i = 0; i + 1 < M; ++i)
        {
            const Eigen::Vector2f dp = ctrl[i + 1] - ctrl[i];
            const float len = dp.norm();
            out.r(row) = sc * (len - cfg.h) / cfg.h;
            if (len > 1e-6f)
            {
                const Eigen::Vector2f u = dp / len;
                const int va = var_of(i,     lo, hi);
                const int vb = var_of(i + 1, lo, hi);
                if (va >= 0) { out.J(row, va) += -sc * u.x() / cfg.h; out.J(row, va + 1) += -sc * u.y() / cfg.h; }
                if (vb >= 0) { out.J(row, vb) +=  sc * u.x() / cfg.h; out.J(row, vb + 1) +=  sc * u.y() / cfg.h; }
            }
            ++row;
        }
    }
    out.row_gauge = row - mark;
}

}  // namespace

RouteOptimizerReport optimize_route(std::vector<Eigen::Vector2f> &ctrl, const RouteOptimizerConfig &cfg)
{
    RouteOptimizerReport rep;
    const std::size_t M = ctrl.size();
    if (M < 4 or not cfg.distance) return rep;

    // Endpoints are ALWAYS pinned: c(0) must stay at the robot and c(L) where the tour ends. freeze_before
    // pins a longer prefix on a repair.
    const std::size_t lo = std::max<std::size_t>(1, cfg.freeze_before);
    // The tail pin. freeze_after == 0 would mean "everything is frozen", which is a degenerate problem
    // rather than an error — it returns with ran == false and ctrl untouched, exactly like too-few points.
    const std::size_t hi = (cfg.freeze_after == 0)
                         ? 0
                         : std::min(M - 2, cfg.freeze_after - 1);
    if (hi < lo or cfg.freeze_after == 0) return rep;

    // Bind each authored waypoint to a control point ONCE, in ORDER with a forward-only hint. A global
    // nearest search would bind a waypoint to the wrong pass on a self-crossing route — the same trap that
    // RouteSpline::project and RouteFollower::repair both guard against.
    std::vector<std::size_t> anchor_ctrl;
    anchor_ctrl.reserve(cfg.anchors.size());
    if (cfg.anchor_s.size() == cfg.anchors.size() and not cfg.anchors.empty())
    {
        // BY ARC LENGTH. Control points sit every `h` along the polyline, so s/h IS the index — monotone
        // by construction, so a self-crossing route or a spliced repair cannot bind a waypoint to a
        // control point that is metres away along the curve. A spatial search can, and did.
        for (const float sj : cfg.anchor_s)
            anchor_ctrl.push_back(static_cast<std::size_t>(
                std::clamp(sj / std::max(1e-3f, cfg.h), 0.f, static_cast<float>(M - 1))));
    }
    else
    {
        // SPATIAL FALLBACK — usable only on a route that does not repeat. On a multi-lap route the same
        // waypoint occupies the same point in space once per lap, control points are h apart with a
        // different phase each lap, and so the nearest one can belong to ANY lap; the forward-only hint
        // then carries every later anchor to the end of the route. That is not a corner case, it is what
        // happened (3 laps: anchor cost 11.87 of 12.09, 15 m of tour deleted). A caller that knows its
        // route must supply anchor_s.
        if (not cfg.anchors.empty())
        {
            std::printf("[route-opt] WARNING: no anchor arc lengths supplied — binding anchors spatially. "
                        "This is UNSAFE on a route that repeats or crosses itself.\n");
            std::fflush(stdout);
        }
        std::size_t hint = 0;
        for (const auto &a : cfg.anchors)
        {
            std::size_t best = hint;
            float best_d2 = std::numeric_limits<float>::max();
            for (std::size_t i = hint; i < M; ++i)
                if (const float d2 = (ctrl[i] - a).squaredNorm(); d2 < best_d2) { best_d2 = d2; best = i; }
            anchor_ctrl.push_back(best);
            hint = best;
        }
    }

    const auto initial = ctrl;
    Problem pr;
    float clearance = 0.f;

    // ── ONE-TIME TERM NORMALISATION ───────────────────────────────────────────────────────────────
    // Measure each term on the INITIAL route with unit weights, then scale so they start comparable.
    // Without it the weights are not preferences: measured on the real tour, kappa 0.163 against clear
    // 0.031, so bending outvoted clearance 5:1 and the solve traded away the thing it was asked to
    // protect. Computed ONCE and held fixed — see the header for why that is not the MPPI's mistake.
    TermScales sc_t;
    if (cfg.normalise_terms)
    {
        Problem p0;
        assemble(ctrl, cfg, anchor_ctrl, lo, hi, TermScales{}, p0, nullptr);
        int o = 0;
        const float e_k = p0.r.segment(o, p0.row_bend).squaredNorm();   o += p0.row_bend;
        const float e_c = p0.r.segment(o, p0.row_clear).squaredNorm();  o += p0.row_clear;
        const float e_a = p0.r.segment(o, p0.row_anchor).squaredNorm();
        // A term that is already zero needs no scaling — leave it at 1 rather than dividing by an epsilon
        // and handing it enormous authority the moment it becomes nonzero.
        if (e_k > 1e-6f) sc_t.kappa  = 1.f / e_k;
        if (e_c > 1e-6f) sc_t.clear  = 1.f / e_c;
        if (e_a > 1e-6f) sc_t.anchor = 1.f / e_a;
    }
    rep.scale_kappa = sc_t.kappa; rep.scale_clear = sc_t.clear; rep.scale_anchor = sc_t.anchor;

    assemble(ctrl, cfg, anchor_ctrl, lo, hi, sc_t, pr, &clearance);
    rep.cost_before = pr.r.squaredNorm();
    rep.min_clearance_before = clearance;

    float mu = 1e-3f;
    float cost = rep.cost_before;
    int it = 0;
    for (; it < cfg.iterations; ++it)
    {
        const Eigen::MatrixXf H = pr.J.transpose() * pr.J;
        const Eigen::VectorXf g = pr.J.transpose() * pr.r;
        if (g.norm() < 1e-7f) break;

        bool accepted = false;
        for (int inner = 0; inner < 8 and not accepted; ++inner)
        {
            Eigen::MatrixXf A = H;
            A.diagonal() += mu * H.diagonal().cwiseMax(1e-9f);
            const Eigen::VectorXf step = A.ldlt().solve(-g);
            if (not step.allFinite()) { mu *= 4.f; continue; }

            auto trial = ctrl;
            for (std::size_t i = lo; i <= hi; ++i)
            {
                const int v = var_of(i, lo, hi);
                trial[i] += Eigen::Vector2f(step(v), step(v + 1));
            }

            Problem tp;
            assemble(trial, cfg, anchor_ctrl, lo, hi, sc_t, tp, nullptr);
            if (const float tc = tp.r.squaredNorm(); tc < cost)
            {
                ctrl = std::move(trial);
                pr = std::move(tp);
                cost = tc;
                mu = std::max(1e-6f, mu * 0.5f);
                accepted = true;
            }
            else
                mu *= 4.f;
        }
        if (not accepted) break;      // the trust region collapsed: this is the local minimum
    }

    assemble(ctrl, cfg, anchor_ctrl, lo, hi, sc_t, pr, &clearance);
    rep.ran = true;
    rep.iterations = it;
    rep.cost_after = cost;
    rep.min_clearance_after = clearance;
    for (std::size_t i = 0; i < M; ++i)
        rep.max_move_m = std::max(rep.max_move_m, (ctrl[i] - initial[i]).norm());
    {
        int o = 0;
        rep.e_kappa  = pr.r.segment(o, pr.row_bend).squaredNorm();   o += pr.row_bend;
        rep.e_clear  = pr.r.segment(o, pr.row_clear).squaredNorm();  o += pr.row_clear;
        rep.e_anchor = pr.r.segment(o, pr.row_anchor).squaredNorm(); o += pr.row_anchor;
        rep.e_gauge  = pr.r.segment(o, pr.row_gauge).squaredNorm();
    }

    // ── UNIFORM VISIBILITY DEFORMATION ────────────────────────────────────────────────────────────
    // Is the optimised route still THE SAME ROUTE? Every other guard here measures a QUANTITY — how far a
    // control point moved, how much clearance was lost — and a route can pass all of them while having
    // been deformed onto the other side of an obstacle, or (measured live, on a 3-lap tour) onto a
    // different pass of a route that repeats. That failure cost 15 m of deleted tour with max_move inside
    // its limit and clearance IMPROVING.
    // The test is Zhou et al.'s UVD (ICRA 2020): two curves belong to the same deformation class if the
    // straight segment joining them AT EQUAL PARAMETER is collision-free for every parameter. It is a
    // topological statement — "nothing was crossed" — which is exactly the property the other guards
    // cannot express, and it costs one sweep of the curve.
    bool uvd_ok = true;
    if (cfg.distance)
    {
        const auto pad_of = [](const std::vector<Eigen::Vector2f> &c)
        {
            std::vector<Eigen::Vector2f> p;
            p.reserve(c.size() + 4);
            p.push_back(c.front()); p.push_back(c.front());
            p.insert(p.end(), c.begin(), c.end());
            p.push_back(c.back()); p.push_back(c.back());
            return p;
        };
        const auto pad_a = pad_of(initial), pad_b = pad_of(ctrl);
        // SPHERE TRACING along each connecting segment, not a fixed number of samples. A fixed count
        // cannot see an obstacle thinner than its own spacing — with five samples over a 1.2 m segment
        // this test walked straight through a bar and reported no violation. Stepping by the distance
        // field's own value is exact instead: nothing can lie within d of a point whose clearance is d,
        // so advancing by d skips only provably empty space. (The field must be a true distance field,
        // which ours is — an exact EDT.)
        const auto segment_hits_obstacle = [&cfg](const Eigen::Vector2f &p0, const Eigen::Vector2f &p1)
        {
            const float len = (p1 - p0).norm();
            if (len < 1e-6f) return cfg.distance(p0) <= 0.f;
            const Eigen::Vector2f dir = (p1 - p0) / len;
            float travelled = 0.f;
            for (int guard = 0; guard < 10000 and travelled <= len; ++guard)
            {
                const float d = cfg.distance(p0 + travelled * dir);
                if (d <= 0.f) return true;
                travelled += std::max(d, 1e-3f);      // the epsilon only guarantees progress
            }
            return false;
        };
        for (std::size_t s = 0; s + 3 < pad_a.size() and uvd_ok; ++s)
            for (int q = 0; q < kQuadPerSpan and uvd_ok; ++q)
            {
                const float t = (static_cast<float>(q) + 0.5f) / static_cast<float>(kQuadPerSpan);
                const auto B = basis(t);
                Eigen::Vector2f ca = Eigen::Vector2f::Zero(), cb = Eigen::Vector2f::Zero();
                for (int k = 0; k < 4; ++k) { ca += B[k] * pad_a[s + k]; cb += B[k] * pad_b[s + k]; }
                // WHERE THE CURVE DID NOT MOVE THERE IS NOTHING TO TEST. UVD is a statement ABOUT THE
                // DEFORMATION — "did the curve sweep across an obstacle getting from a to b" — so at a
                // parameter where a == b it is vacuously satisfied. Without this, the degenerate branch
                // of segment_hits_obstacle (len < 1e-6) instead asks "is this UNCHANGED point inside an
                // obstacle?", which is a different question and belongs to the feasibility pass.
                // ★It never fired for a build-time solve, where the whole route moves and the field is
                // GridPlanner's static EDT on which an A*-derived route is feasible by construction. It
                // fires for the LOCAL BAND, which freezes everything outside a ~10-point window and
                // measures against the LIVE ESDF: any pre-existing route point lying in the live field's
                // zero set then rejected EVERY solve, no matter how good, for a violation in a stretch
                // the solve never touched.
                if ((cb - ca).squaredNorm() < 1e-12f) continue;
                if (segment_hits_obstacle(ca, cb)) uvd_ok = false;
            }
    }
    rep.uvd_violated = not uvd_ok;

    // ── ACCEPTANCE TEST ───────────────────────────────────────────────────────────────────────────
    // An optimiser is trusted to improve its objective, never to be SAFE. These two checks ask whether
    // the result is worth having at all, and revert the whole solve if it is not.
    //   • A control point that moved further than a few times the control spacing has not been refined,
    //     it has been relocated — the route is no longer the route that was planned.
    //   • Clearance getting WORSE means the term meant to protect it was outvoted, which is exactly the
    //     dilution failure that makes a bounded term inert next to an unbounded one.
    // Both were observed live at once (24.8 m of movement, clearance 0.020 -> 0.000 m), which is why
    // this is an explicit guard rather than a comment. It is loud: silently returning the input would
    // look like "the route was already optimal" and hide the failure completely.
    const float move_limit = 6.f * cfg.h;
    const bool ran_away = rep.max_move_m > move_limit;
    // Not "clearance got worse" — that is a legitimate trade (see clearance_floor). This asks whether the
    // solve pushed the route BELOW what the body needs, and did so by making things worse than it found them.
    const bool lost_clearance = cfg.clearance_floor > 0.f
                            and rep.min_clearance_after < cfg.clearance_floor
                            and rep.min_clearance_after < rep.min_clearance_before - 1e-4f;
    if (ran_away or lost_clearance or not uvd_ok)
    {
        std::printf("[route-opt] REJECTED and reverted: %s%s%s%s (max move %.2f m vs limit %.2f, "
                    "clearance %.3f -> %.3f m) [kappa %.3f | clear %.3f | anchor %.3f | gauge %.3f]. "
                    "The un-optimised route is used.\n",
                    ran_away ? "control points ran away" : "",
                    (ran_away and lost_clearance) ? " and " : "",
                    lost_clearance ? "route driven below the feasibility floor" : "",
                    not uvd_ok ? " the route was deformed ACROSS an obstacle (UVD)" : "",
                    rep.max_move_m, move_limit, rep.min_clearance_before, rep.min_clearance_after,
                    rep.e_kappa, rep.e_clear, rep.e_anchor, rep.e_gauge);
        std::fflush(stdout);
        ctrl = initial;
        rep.rejected = true;
        rep.max_move_m = 0.f;
        rep.min_clearance_after = rep.min_clearance_before;
        rep.cost_after = rep.cost_before;
        return rep;
    }

    if (cfg.verbose)
    {
        std::printf("[route-opt] %zu ctrl pts, %d iters: cost %.4f -> %.4f "
                    "[kappa %.3f | clear %.3f | anchor %.3f | gauge %.3f], "
                    "clearance %.3f -> %.3f m, max move %.3f m%s\n",
                    M, rep.iterations, rep.cost_before, rep.cost_after,
                    rep.e_kappa, rep.e_clear, rep.e_anchor, rep.e_gauge,
                    rep.min_clearance_before, rep.min_clearance_after, rep.max_move_m,
                    cfg.freeze_before > 0 ? " (prefix frozen)" : "");
        std::fflush(stdout);
    }
    return rep;
}

bool route_optimizer_self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // An ANALYTIC corridor: walls at y = +-W, running along x. Distance to the nearest wall is W - |y|,
    // and its gradient points at the centreline. No grid, so this tests the optimiser alone.
    constexpr float W = 1.0f;
    const auto dist = [](const Eigen::Vector2f &p) { return std::max(0.f, W - std::abs(p.y())); };
    const auto grad = [](const Eigen::Vector2f &p)
    { return Eigen::Vector2f{0.f, p.y() > 0.f ? -1.f : 1.f}; };

    // A route hugging the upper wall, as a grid planner would produce it.
    const auto make_ctrl = []()
    {
        std::vector<Eigen::Vector2f> c;
        for (int i = 0; i <= 20; ++i) c.push_back({0.4f * static_cast<float>(i), 0.6f});
        return c;
    };

    RouteOptimizerConfig cfg;
    cfg.distance = dist;
    cfg.distance_gradient = grad;
    cfg.h = 0.40f;
    cfg.d_target = W;          // deficit = |y|, so the preference alone is minimised on the centreline
    cfg.rho = 0.49f;
    cfg.w_kappa = 1.0f;
    cfg.w_clear = 1.0f;
    cfg.w_gauge = 0.05f;
    cfg.iterations = 60;
    cfg.verbose = true;

    // (1) THE MEDIAL-AXIS CLAIM, tested rather than asserted. With no anchors pulling it off, the curve
    //     must converge onto the centreline — equidistance as a consequence of the clearance term.
    {
        auto ctrl = make_ctrl();
        const auto rep = optimize_route(ctrl, cfg);
        check(rep.ran, "the optimiser must run on a 21-point polygon");

        // The claim is about the INTERIOR. The endpoints are pinned off-axis at y = 0.6 (which is what an
        // A* route entering a corridor off-centre looks like), so the curve must spend a few control
        // points descending — asserting the medial axis right next to a pinned boundary condition would be
        // asserting that the prior does not exist. Measured over the middle half, away from both ends.
        const std::size_t n = ctrl.size();
        float mid_worst = 0.f, end_worst = 0.f;
        for (std::size_t i = n / 4; i <= 3 * n / 4; ++i) mid_worst = std::max(mid_worst, std::abs(ctrl[i].y()));
        for (std::size_t i = 1; i < n / 4; ++i) end_worst = std::max(end_worst, std::abs(ctrl[i].y()));
        std::printf("  corridor: middle half offset from the medial axis %.4f m, approach %.4f m "
                    "(both started at 0.600)\n", mid_worst, end_worst);
        // 0.07 not 0.05: the clearance term is u^2 + clear_peak*u^4, and the quartic deliberately trades a
        // little pull in the UNIFORMLY-tight case (0.042 -> 0.059 m here) for the ability to see a SINGLE
        // tight point at all. Test (6) is the one that justifies that trade; if it ever fails, this
        // relaxation is unearned and clear_peak should go back to 0.
        check(mid_worst < 0.07f, "the clearance term alone must pull the route's interior onto the medial axis");
        check(end_worst < 0.45f, "the approach from a pinned off-axis end must still descend toward it");
        check(rep.cost_after < rep.cost_before, "the cost must decrease");
    }

    // (2) ANALYTIC JACOBIAN vs FINITE DIFFERENCES. If this passes, every term's derivative is right; if
    //     it fails, the medial-axis test above could still pass for the wrong reason.
    {
        auto ctrl = make_ctrl();
        for (std::size_t i = 0; i < ctrl.size(); ++i)         // perturb off any symmetry
        {
            ctrl[i].y() += 0.05f * std::sin(static_cast<float>(i) * 1.7f);
            // ...and off UNIFORM SPACING. The curvature bound contains min(|a|,|b|), which is
            // non-differentiable where the two adjacent segments are equal — and a uniformly spaced
            // polygon sits exactly on that kink, so a central difference straddles it and disagrees with
            // either one-sided derivative by ~14%. That is the formula's kink, not a wrong Jacobian (the
            // bound's own derivative is checked directly above at 0.3%). Stay off it deliberately.
            ctrl[i].x() += 0.06f * std::sin(static_cast<float>(i) * 2.3f);
        }
        const std::size_t M = ctrl.size(), lo = 1, hi = M - 2;
        const std::vector<std::size_t> ac = {5, 12};

        auto fd_check = [&](bool bound, const char *what)
        {
            RouteOptimizerConfig c2 = cfg;
            c2.curvature_bound = bound;
            c2.anchors = {ctrl[5], ctrl[12]};                     // exercise the anchor rows too
            // kappa_peak ON, so the bending block under test is the NON-LINEAR one. With it at 0 the
            // Delta^2 form is the old linear block and the check would pass without touching the new term.
            c2.kappa_peak = 4.0f;

            Problem p0;
            assemble(ctrl, c2, ac, lo, hi, TermScales{}, p0, nullptr);
            const Eigen::VectorXf ga = 2.f * (p0.J.transpose() * p0.r);   // dS/dx, S = |r|^2

            float worst_rel = 0.f;
            int worst_v = -1;
            float worst_num = 0.f, worst_ana = 0.f;
            // eps must be large enough that the difference of two float32 costs is not dominated by their
            // own rounding: S ~ 1e-1 carries ~1e-8 of absolute noise, so a step giving dS ~ 1e-5 is already
            // only three digits clean. 1e-3 m is well inside the linear regime here and far above that floor.
            const float eps = 1e-3f;
            for (int v = 0; v < static_cast<int>(2 * (hi - lo + 1)); v += 7)   // stride: 50 checks is plenty
            {
                auto up = ctrl, dn = ctrl;
                const std::size_t i = lo + static_cast<std::size_t>(v) / 2;
                const int comp = v % 2;
                up[i][comp] += eps;
                dn[i][comp] -= eps;
                Problem pu, pd;
                assemble(up, c2, ac, lo, hi, TermScales{}, pu, nullptr);
                assemble(dn, c2, ac, lo, hi, TermScales{}, pd, nullptr);
                const float num = (pu.r.squaredNorm() - pd.r.squaredNorm()) / (2.f * eps);
                const float rel = std::abs(num - ga(v)) / std::max(1e-2f, std::abs(num));
                if (rel > worst_rel) { worst_rel = rel; worst_v = v; worst_num = num; worst_ana = ga(v); }
            }
            std::printf("  gradient check (%s): worst relative error %.5f at var %d (fd %.6f vs analytic %.6f)\n",
                        what, worst_rel, worst_v, worst_num, worst_ana);
            check(worst_rel < 2e-2f, "the analytic Jacobian must match finite differences");
        };
        fd_check(false, "second difference, kappa_peak on");
        fd_check(true,  "curvature upper bound");
    }

    // (3) FROZEN PREFIX. On a repair the route ahead may move; the stretch the robot is already on may not.
    {
        auto ctrl = make_ctrl();
        RouteOptimizerConfig c3 = cfg;
        c3.freeze_before = 8;
        const auto before = ctrl;
        const auto rep = optimize_route(ctrl, c3);
        check(rep.ran, "a frozen-prefix optimisation must still run");
        float moved = 0.f;
        for (std::size_t i = 0; i < 8; ++i) moved = std::max(moved, (ctrl[i] - before[i]).norm());
        check(moved < 1e-6f, "control points before freeze_before must not move at all");
        float after = 0.f;
        for (std::size_t i = 10; i + 2 < ctrl.size(); ++i) after = std::max(after, std::abs(before[i].y() - ctrl[i].y()));
        check(after > 0.1f, "control points after the freeze must still be free to move");
        std::printf("  frozen prefix: first 8 moved %.9f m, tail moved up to %.3f m\n", moved, after);
    }

    // (4) IDEMPOTENCE. Re-optimising an optimised route must barely move it. A large second step means a
    //     nonzero stationary step — usually a discontinuous distance gradient.
    {
        auto ctrl = make_ctrl();
        optimize_route(ctrl, cfg);
        const auto once = ctrl;
        const auto rep2 = optimize_route(ctrl, cfg);
        std::printf("  idempotence: second pass moved %.5f m\n", rep2.max_move_m);
        check(rep2.max_move_m < 0.01f, "re-optimising an optimised route must be nearly a no-op");
        (void)once;
    }

    // (5) The anchors must actually bind: a route pulled by waypoints off the centreline cannot sit on it.
    {
        auto ctrl = make_ctrl();
        RouteOptimizerConfig c5 = cfg;
        c5.anchors = {{4.0f, 0.75f}};
        c5.sigma_a = 0.05f;                 // a tight intent precision must win over the clearance term
        optimize_route(ctrl, c5);
        float nearest = std::numeric_limits<float>::max();
        for (const auto &p : ctrl) nearest = std::min(nearest, (p - c5.anchors[0]).norm());
        std::printf("  anchor pull: nearest control point to a 0.75 m off-axis waypoint = %.3f m\n", nearest);
        check(nearest < 0.25f, "a high-precision anchor must hold the route near the authored waypoint");
    }

    // (6) THE ONE TIGHT POINT. A route through mostly-clear space with a single pinch is the case the
    //     island corner and the tour's 5.8 cm passage actually are — and the case a MEAN cannot express,
    //     because one deficient sample among hundreds barely moves an average. This asserts that the
    //     quartic earns its keep: with clear_peak on, the worst clearance must improve materially more
    //     than with a pure mean of squares.
    {
        // A pinch alone is not enough to tell the two apart — with nothing resisting, BOTH open it fully.
        // The mean only fails when improving the worst point COSTS something, so the bending prior is
        // made to resist (rho = 1.5) and mild wall deficit is present along the whole route, which is what
        // a real tour looks like. Verified: with these numbers a pure mean stalls at 0.362 where the
        // quartic reaches 0.388.
        const Eigen::Vector2f obstacle{4.0f, 0.30f};
        constexpr float wall = 0.60f;
        const auto dist1 = [&](const Eigen::Vector2f &p)
        { return std::min(std::max(0.f, wall - std::abs(p.y())), (p - obstacle).norm()); };
        const auto grad1 = [&](const Eigen::Vector2f &p)
        {
            const float dw = std::max(0.f, wall - std::abs(p.y()));
            const Eigen::Vector2f d = p - obstacle; const float dn = d.norm();
            if (dw < dn) return Eigen::Vector2f(0.f, p.y() > 0.f ? -1.f : 1.f);
            return dn > 1e-6f ? Eigen::Vector2f(d / dn) : Eigen::Vector2f(0.f, 1.f);
        };

        auto run = [&](float peak)
        {
            std::vector<Eigen::Vector2f> c;
            for (int i = 0; i <= 20; ++i) c.push_back({0.4f * static_cast<float>(i), 0.f});
            RouteOptimizerConfig k;
            k.distance = dist1; k.distance_gradient = grad1;
            k.h = 0.40f; k.d_target = 0.60f; k.rho = 1.5f;
            k.clear_peak = peak; k.iterations = 150; k.verbose = false;
            optimize_route(c, k);
            float worst = std::numeric_limits<float>::max();
            for (const auto &p : c) worst = std::min(worst, (p - obstacle).norm());
            return worst;
        };
        const float before = 0.30f;              // the initial straight route passes this close
        const float mean_only = run(0.f);        // pure u^2 — the original formulation
        const float with_peak = run(4.0f);       // u^2 + 4*u^4
        std::printf("  one tight point: clearance %.3f -> mean-only %.3f, with peak %.3f\n",
                    before, mean_only, with_peak);
        check(with_peak > mean_only + 0.015f,
              "the quartic must open a single tight point more than a mean of squares does");
        check(with_peak > before, "the worst clearance must actually improve");
    }

    // The tightest radius the CURVE demands — evaluated on the curve, from the exact curvature, because
    // that is the number the robot has to turn through and the whole point of tests (7) and (8) is that
    // it is not the same as the number a control polygon suggests.
    const auto min_curve_radius = [](const std::vector<Eigen::Vector2f> &c)
    {
        std::vector<Eigen::Vector2f> pad;
        pad.push_back(c.front()); pad.push_back(c.front());
        pad.insert(pad.end(), c.begin(), c.end());
        pad.push_back(c.back()); pad.push_back(c.back());
        const std::size_t n_spans = pad.size() - 3;
        float kmax = 0.f;
        for (std::size_t s = 1; s + 2 < n_spans; ++s)     // the extreme spans are straight by construction
            for (int q = 0; q < 32; ++q)
            {
                const float t = static_cast<float>(q) / 32.f;
                const auto B1 = basis_d1(t), B2 = basis_d2(t);
                Eigen::Vector2f g = Eigen::Vector2f::Zero(), a = Eigen::Vector2f::Zero();
                for (int k = 0; k < 4; ++k) { g += B1[k] * pad[s + k]; a += B2[k] * pad[s + k]; }
                const float n = g.norm();
                if (n < 1e-4f) continue;
                kmax = std::max(kmax, std::abs(g.x() * a.y() - g.y() * a.x()) / (n * n * n));
            }
        return kmax > 1e-6f ? 1.f / kmax : 1e9f;
    };

    // (6a) UVD — THE ROUTE MUST NOT BE DEFORMED ACROSS AN OBSTACLE. Every other guard measures a
    //      quantity and can be satisfied by a route that has moved to the wrong side of something. Here
    //      a thin bar sits between the route and an anchor that pulls hard across it; the solve must be
    //      REJECTED with uvd_violated set, not accepted because the numbers looked reasonable.
    {
        // A bar of real THICKNESS at y = 0.5, x in [1,3], with the field zero INSIDE it — exactly how the
        // EDT behaves on occupied cells. A zero-thickness obstacle would be untestable and unrealistic:
        // `d <= 0` would hold only on a set of measure zero, and any tracing scheme can step over it.
        // Written the first time as an infinitely thin line, which is why this test appeared to fail.
        constexpr float kHalfThick = 0.05f;
        const auto bar_dist = [](const Eigen::Vector2f &p)
        {
            const float x = std::clamp(p.x(), 1.f, 3.f);
            return std::max(0.f, (p - Eigen::Vector2f{x, 0.5f}).norm() - kHalfThick);
        };
        const auto bar_grad = [&](const Eigen::Vector2f &p)
        {
            const float x = std::clamp(p.x(), 1.f, 3.f);
            const Eigen::Vector2f d = p - Eigen::Vector2f{x, 0.5f};
            const float n = d.norm();
            return n > 1e-6f ? Eigen::Vector2f(d / n) : Eigen::Vector2f(0.f, 1.f);
        };
        std::vector<Eigen::Vector2f> c;
        for (int i = 0; i <= 20; ++i) c.push_back({0.4f * static_cast<float>(i), 0.f});
        const auto before = c;
        RouteOptimizerConfig k;
        k.distance = bar_dist; k.distance_gradient = bar_grad;
        k.h = 0.40f; k.d_target = 0.30f; k.rho = 0.49f;
        k.sigma_a = 0.03f;                       // a very insistent waypoint, on the far side of the bar
        k.anchors = {{2.0f, 1.2f}};
        k.anchor_s = {2.0f};
        k.iterations = 60; k.verbose = false;
        const auto rep = optimize_route(c, k);
        float moved = 0.f;
        for (std::size_t i = 0; i < c.size(); ++i) moved = std::max(moved, (c[i] - before[i]).norm());
        std::printf("  UVD: rejected=%d uvd_violated=%d, control points moved %.6f m\n",
                    rep.rejected ? 1 : 0, rep.uvd_violated ? 1 : 0, moved);
        check(rep.uvd_violated, "pulling the route across a bar must be detected as a UVD violation");
        check(rep.rejected, "a UVD violation must reject the solve");
        check(moved < 1e-6f, "a rejected solve must leave the control polygon untouched");
    }

    // (6a-bis) UVD MUST NOT FIRE ON A STRETCH THE SOLVE NEVER TOUCHED. This is the LOCAL BAND's case:
    // most of the route is frozen, so a == b there, and the field is the LIVE ESDF, in which a
    // pre-existing route point can legitimately sit inside an obstacle (an object that appeared after
    // the route was built). Before the fix, that unchanged point alone rejected every solve — a route
    // that could never be improved because of a violation nowhere near the window.
    {
        // Obstacle sitting ON the route at x = 1.0, inside the FROZEN prefix. The deformable window is
        // far away at x >= 4, in clear space, so no deformation can possibly cross anything.
        const auto blob = [](const Eigen::Vector2f &p)
        { return std::max(0.f, (p - Eigen::Vector2f{1.0f, 0.f}).norm() - 0.10f); };
        const auto blob_g = [](const Eigen::Vector2f &p)
        {
            const Eigen::Vector2f d = p - Eigen::Vector2f{1.0f, 0.f};
            const float n = d.norm();
            return n > 1e-6f ? Eigen::Vector2f(d / n) : Eigen::Vector2f(0.f, 1.f);
        };
        std::vector<Eigen::Vector2f> c;
        for (int i = 0; i <= 20; ++i) c.push_back({0.4f * static_cast<float>(i), 0.f});
        const auto before = c;
        RouteOptimizerConfig k;
        k.distance = blob; k.distance_gradient = blob_g;
        k.h = 0.40f; k.d_target = 0.30f; k.rho = 0.49f;
        k.anchors = {{5.0f, 0.6f}};              // pulls the WINDOW only, and only in free space
        k.anchor_s = {5.0f};
        k.freeze_before = 10;                    // everything up to x = 4 pinned — the blob is at x = 1
        k.iterations = 20; k.verbose = false;
        const auto rep = optimize_route(c, k);
        float moved_frozen = 0.f, moved_win = 0.f;
        for (std::size_t i = 0; i < c.size(); ++i)
        {
            const float d = (c[i] - before[i]).norm();
            (i < k.freeze_before ? moved_frozen : moved_win) = std::max(i < k.freeze_before ? moved_frozen : moved_win, d);
        }
        std::printf("  UVD frozen-region: uvd_violated=%d rejected=%d, frozen moved %.9f m, window moved %.4f m\n",
                    rep.uvd_violated ? 1 : 0, rep.rejected ? 1 : 0, moved_frozen, moved_win);
        check(not rep.uvd_violated, "an obstacle in a FROZEN stretch must not count as a UVD violation");
        check(moved_frozen == 0.f, "the frozen prefix must not move");
    }

    // (6b) IS THE CURVATURE BOUND ACTUALLY A BOUND? The formula is taken from the literature, so it is
    //      exactly the kind of thing that must be checked against this codebase's own curve before
    //      anything is built on it — an "upper bound" that is not one would silently license routes the
    //      robot cannot drive. Random control polygons, bound vs the curve's true peak curvature.
    {
        // A deterministic pseudo-random polygon generator: no Date/rand, so the test is reproducible.
        std::uint32_t seed = 12345u;
        const auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u;
                                     return static_cast<float>((seed >> 8) & 0xFFFF) / 65535.f; };
        float worst_ratio = 1e9f;      // min over samples of bound/true — must stay >= 1
        float loosest = 0.f;
        int violations = 0, spans = 0;
        for (int trial = 0; trial < 40; ++trial)
        {
            std::vector<Eigen::Vector2f> c;
            for (int i = 0; i < 12; ++i)
                c.push_back({0.40f * i + 0.30f * (rnd() - 0.5f), 0.60f * (rnd() - 0.5f)});
            std::vector<Eigen::Vector2f> pad;
            pad.push_back(c.front()); pad.push_back(c.front());
            pad.insert(pad.end(), c.begin(), c.end());
            pad.push_back(c.back()); pad.push_back(c.back());
            // For each INTERIOR span, the true peak curvature over it vs the bound at the two control
            // points that span sits between. (The extreme spans are straight by construction.)
            for (std::size_t s = 1; s + 2 < pad.size() - 3 + 1; ++s)
            {
                float true_max = 0.f;
                for (int q = 0; q <= 64; ++q)
                {
                    const float t = static_cast<float>(q) / 64.f;
                    const auto B1 = basis_d1(t), B2 = basis_d2(t);
                    Eigen::Vector2f g = Eigen::Vector2f::Zero(), aa = Eigen::Vector2f::Zero();
                    for (int k = 0; k < 4; ++k) { g += B1[k] * pad[s + k]; aa += B2[k] * pad[s + k]; }
                    const float n = g.norm();
                    if (n < 1e-4f) continue;
                    true_max = std::max(true_max, std::abs(g.x() * aa.y() - g.y() * aa.x()) / (n * n * n));
                }
                // The span between pad[s+1] and pad[s+2]; both map to control points via pad_to_ctrl.
                float bound = 0.f;
                for (int k = 1; k <= 2; ++k)
                {
                    const std::size_t m = pad_to_ctrl(s + k, c.size());
                    if (m == 0 or m + 1 >= c.size()) continue;
                    bound = std::max(bound, curvature_bound(c[m - 1] - c[m], c[m + 1] - c[m], 0.05f).K);
                }
                if (true_max < 1e-3f or bound <= 0.f) continue;
                ++spans;
                const float ratio = bound / true_max;
                worst_ratio = std::min(worst_ratio, ratio);
                loosest = std::max(loosest, ratio);
                if (ratio < 1.f) ++violations;
            }
        }
        std::printf("  curvature bound: %d spans, bound/true in [%.3f, %.2f], %d violations of the bound\n",
                    spans, worst_ratio, loosest, violations);

        // The bound's OWN derivative, checked before it is chained into the assembly — a mismatch here is
        // algebra, a mismatch only in the assembly is wiring. Sampled away from the two kinks the formula
        // has by construction (the min() switching, and the L_min clip binding).
        float worst_rel = 0.f;
        int checked = 0;
        for (int trial = 0; trial < 60; ++trial)
        {
            const Eigen::Vector2f a{0.3f + 0.5f * rnd(), -0.4f + 0.8f * rnd()};
            const Eigen::Vector2f b{-0.3f - 0.5f * rnd(), -0.4f + 0.8f * rnd()};
            if (std::abs(a.norm() - b.norm()) < 0.05f) continue;       // too close to the min() kink
            if (std::min(a.norm(), b.norm()) < 0.10f) continue;        // too close to the L_min clip
            const auto cb = curvature_bound(a, b, 0.05f);
            const float eps = 1e-4f;
            for (int comp = 0; comp < 2; ++comp)
            {
                Eigen::Vector2f ap = a, am = a;
                ap[comp] += eps; am[comp] -= eps;
                const float num = (curvature_bound(ap, b, 0.05f).K - curvature_bound(am, b, 0.05f).K) / (2 * eps);
                const float rel = std::abs(num - cb.dK_da[comp]) / std::max(1.f, std::abs(num));
                worst_rel = std::max(worst_rel, rel);
                Eigen::Vector2f bp = b, bm = b;
                bp[comp] += eps; bm[comp] -= eps;
                const float numb = (curvature_bound(a, bp, 0.05f).K - curvature_bound(a, bm, 0.05f).K) / (2 * eps);
                const float relb = std::abs(numb - cb.dK_db[comp]) / std::max(1.f, std::abs(numb));
                worst_rel = std::max(worst_rel, relb);
                ++checked;
            }
        }
        std::printf("  curvature bound gradient: worst relative error %.5f over %d checks\n",
                    worst_rel, checked);
        check(worst_rel < 2e-2f, "the curvature bound's own derivative must match finite differences");
        check(violations == 0, "the clamped-B-spline formula must actually UPPER-bound the curve's curvature");
        check(worst_ratio >= 0.99f, "the bound must not fall below the true peak on any span");
    }

    // (7) THE ONE SHARP CORNER — test (6)'s argument, applied to curvature. A tour is mostly gentle with a
    //     couple of hairpins, and the hairpins are what the robot has to stop and pivot through. A mean of
    //     squares improves the gentle stretches (there are hundreds of them) in preference to the one
    //     corner that actually costs a lap its time, so the tightest radius is what must be measured.
    {
        // An L-bend in open space, so nothing but the two terms is in play. The clearance term is left
        // active (d_target small enough that the walls are irrelevant) — the corner must be opened against
        // the anchor pull that created it, which is what happens on the real tour.
        const auto far_dist = [](const Eigen::Vector2f &) { return 5.f; };
        const auto far_grad = [](const Eigen::Vector2f &) { return Eigen::Vector2f{0.f, 0.f}; };
        auto run = [&](float peak, float w_gauge)
        {
            std::vector<Eigen::Vector2f> c;
            for (int i = 0; i <= 10; ++i) c.push_back({0.4f * static_cast<float>(i), 0.f});
            for (int i = 1; i <= 10; ++i) c.push_back({4.0f, 0.4f * static_cast<float>(i)});
            RouteOptimizerConfig k;
            k.distance = far_dist; k.distance_gradient = far_grad;
            k.h = 0.40f; k.d_target = 0.60f; k.rho = 0.49f;
            k.w_kappa = 1.0f; k.kappa_peak = peak; k.w_gauge = w_gauge;
            k.iterations = 150; k.verbose = false;
            // Anchors along the whole route, including ON the corner: without them a bending term of any
            // shape simply straightens the L, and the test would prove nothing about seeing a PEAK.
            for (std::size_t i = 0; i < c.size(); i += 2) { k.anchors.push_back(c[i]); k.anchor_s.push_back(0.4f * i); }
            const float r_before = min_curve_radius(c);
            optimize_route(c, k);
            float min_spacing = 1e9f;
            for (std::size_t i = 0; i + 1 < c.size(); ++i) min_spacing = std::min(min_spacing, (c[i + 1] - c[i]).norm());
            std::printf("    [L kappa_peak=%.1f w_gauge=%.2f] tightest radius %.3f -> %.3f m, "
                        "min control spacing %.3f m\n",
                        peak, w_gauge, r_before, min_curve_radius(c), min_spacing);
            return min_curve_radius(c);
        };
        const float mean_only = run(0.f, 0.05f);
        const float with_peak = run(4.0f, 0.05f);
        std::printf("  one sharp corner: tightest radius  mean-only %.3f m, with peak %.3f m\n",
                    mean_only, with_peak);
        check(with_peak > mean_only + 0.01f,
              "the quartic must open the tightest corner more than a mean of squares does");

        // (8) E_gauge MUST NOT BE A BEHAVIOURAL KNOB. It is declared gauge-fixing — it removes a
        //     reparameterisation freedom the objective is blind to — so a 10x sweep of it must leave the
        //     GEOMETRY where it was. This is the check that would have caught the mis-specification the
        //     exact-curvature experiment chased (see route_optimizer.h): here in open space the bending
        //     form holds it to a few percent. ★It holds LESS WELL with obstacles present, where the
        //     clearance term is also pushing (measured on a synthetic hairpin: tightest radius 0.118 ->
        //     0.164 m across the same sweep). That residual coupling is a known limit of kappa ~ |d2 p|/h^2,
        //     kept deliberately: the parameterisation-free alternative was measured on the real tour and
        //     was worse on every axis.
        const float g_lo = run(4.0f, 0.05f), g_hi = run(4.0f, 0.50f);
        std::printf("  gauge is structural: tightest radius %.3f -> %.3f m (spread %.1f%%) as w_gauge "
                    "goes 0.05 -> 0.50\n", g_lo, g_hi, 100.f * std::abs(g_hi - g_lo) / std::max(1e-3f, g_lo));
        check(std::abs(g_hi - g_lo) < 0.10f * g_lo,
              "the gauge weight must not move the geometry in open space");
    }

    std::printf("route_optimizer_self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
