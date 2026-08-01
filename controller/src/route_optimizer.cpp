/*
 * route_optimizer.cpp — see route_optimizer.h
 */

#include "route_optimizer.h"

#include <algorithm>
#include <cmath>
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

    const int rows = static_cast<int>(2 * n_bend + n_quad + 2 * n_anch + n_gauge);
    out.r.setZero(rows);
    out.J.setZero(rows, nvar);

    int row = 0;
    if (min_clearance) *min_clearance = std::numeric_limits<float>::max();
    int mark = 0;

    // ── PRIOR: bending. r = sqrt(w/M)·(rho/h^2)·(p[i-1] - 2p[i] + p[i+1]). Linear ⇒ exact Gauss-Newton.
    if (n_bend > 0)
    {
        const float sc = std::sqrt(sc_t.kappa * cfg.w_kappa / static_cast<float>(n_bend)) * cfg.rho / (cfg.h * cfg.h);
        for (std::size_t i = 1; i + 1 < M; ++i)
        {
            const Eigen::Vector2f d2 = ctrl[i - 1] - 2.f * ctrl[i] + ctrl[i + 1];
            for (int c = 0; c < 2; ++c)
            {
                out.r(row + c) = sc * d2[c];
                const int va = var_of(i - 1, lo, hi);
                const int vb = var_of(i,     lo, hi);
                const int vc = var_of(i + 1, lo, hi);
                if (va >= 0) out.J(row + c, va + c) += sc;
                if (vb >= 0) out.J(row + c, vb + c) += -2.f * sc;
                if (vc >= 0) out.J(row + c, vc + c) += sc;
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
        const float sc = std::sqrt(sc_t.clear * cfg.w_clear / static_cast<float>(std::max<std::size_t>(1, n_quad)));
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
    const std::size_t hi = M - 2;
    if (hi < lo) return rep;

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
    if (ran_away or lost_clearance)
    {
        std::printf("[route-opt] REJECTED and reverted: %s%s%s (max move %.2f m vs limit %.2f, "
                    "clearance %.3f -> %.3f m) [kappa %.3f | clear %.3f | anchor %.3f | gauge %.3f]. "
                    "The un-optimised route is used.\n",
                    ran_away ? "control points ran away" : "",
                    (ran_away and lost_clearance) ? " and " : "",
                    lost_clearance ? "route driven below the feasibility floor" : "",
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
            ctrl[i].y() += 0.05f * std::sin(static_cast<float>(i) * 1.7f);
        RouteOptimizerConfig c2 = cfg;
        c2.anchors = {ctrl[5], ctrl[12]};                     // exercise the anchor rows too
        const std::size_t M = ctrl.size(), lo = 1, hi = M - 2;
        const std::vector<std::size_t> ac = {5, 12};

        Problem p0;
        assemble(ctrl, c2, ac, lo, hi, TermScales{}, p0, nullptr);
        const Eigen::VectorXf ga = 2.f * (p0.J.transpose() * p0.r);   // dS/dx, S = |r|^2

        float worst_rel = 0.f;
        int worst_v = -1;
        float worst_num = 0.f, worst_ana = 0.f;
        // eps must be large enough that the difference of two float32 costs is not dominated by their own
        // rounding: S ~ 1e-1 carries ~1e-8 of absolute noise, so a step giving dS ~ 1e-5 is already only
        // three digits clean. 1e-3 m is well inside the linear regime here and far above that floor.
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
        std::printf("  gradient check: worst relative error %.5f at var %d (fd %.6f vs analytic %.6f)\n",
                    worst_rel, worst_v, worst_num, worst_ana);
        check(worst_rel < 2e-2f, "the analytic Jacobian must match finite differences");
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

    std::printf("route_optimizer_self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
