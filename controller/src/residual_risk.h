/*
 * residual_risk.h — how much a residual_concept cell should COST the planner, from residual's own belief.
 *
 * ── WHY (try96 leg 3, 2026-10-19) ────────────────────────────────────────────────────────────────────
 * GridPlanner marked every published residual cell as a HARD wall. residual latches a cell after one near hit
 * (occ_set = 0.40 log-odds), so a thin false "curtain" of a few dozen weakly-evidenced 5 cm cells sealed the
 * robot's region off from the door approach — "no route: 50858 expansions over 15215 free cells" — and the
 * mission failed on a wall nobody had confirmed. A latch is a display/decision bit; it throws away HOW MUCH
 * evidence stands behind the cell, which is exactly the quantity that separates a curtain from a cabinet.
 *
 * ── THE MODEL ───────────────────────────────────────────────────────────────────────────────────────
 * residual publishes, beside the cell list, a dense Beta(α,β) belief over each cell's occupancy RATE θ, as
 * its mean P = grid_occupancy_prob and variance Var = grid_occupancy_var (residual_occupancy_grid.h,
 * BELIEF FIELD). (α,β) are recovered from those two moments. The cost charged for letting the body cover the
 * cell is the SURPRISAL of finding it free:
 *
 *        risk = −log Pr(θ < ½ | α,β) = −log I_½(α,β)            [nats]
 *
 * i.e. the posterior probability that the cell is mostly FREE, which is what driving through it bets on.
 *   · a curtain born from one or two hits (α ≈ 1, β ≈ 0.5–1)   → Pr(θ<½) ≈ 0.3–0.5 → ≈ 1 nat
 *   · a cell at residual's concentration cap (α ≈ 40, β ≈ 1)    → Pr(θ<½) ≈ 2⁻⁴⁰   → ≈ 27 nats
 * so the cost grows EXPONENTIALLY with evidence (linearly in nats per hit) instead of switching at a latch:
 * a confirmed obstacle is never worth crossing, an unconfirmed one is a toll, not a wall.
 * ★The ½ is not a tuned level: θ is a Bernoulli rate, and "occupied more often than not" is its symmetric
 *   decision boundary — the same one residual's own latch approximates in log-odds.
 *
 * ── WHAT STAYS HARD ─────────────────────────────────────────────────────────────────────────────────
 *   · a cell with NO belief behind it (field absent, out of its extent, moments unusable) → +inf: we keep the
 *     old, conservative behaviour exactly where we have nothing better to say;
 *   · a cell a LIVE LiDAR return falls in, inside the body's height band → +inf. That is not a belief, it is a
 *     measurement this cycle of something the body would physically hit (harden_corroborated below).
 * Room walls and object polygons never pass through here and stay hard in GridPlanner.
 *
 * Pure std + Eigen, no DSR: tested in tests/test_grid_planner_curtain.cpp.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>

namespace rc::residual_risk
{

inline constexpr float kHard = std::numeric_limits<float>::infinity();

namespace detail
{
// Continued fraction for the incomplete beta function (Lentz), the standard series (Numerical Recipes §6.4).
// Converges fast for x < (a+1)/(a+b+2); the caller swaps (a,b,x) otherwise.
inline double betacf(double a, double b, double x)
{
    constexpr int    kMaxIt = 300;
    constexpr double kEps = 1e-14, kFpMin = 1e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0, d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= kMaxIt; ++m)
    {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c; if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c; if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < kEps) break;
    }
    return h;
}
// log of the regularised incomplete beta's "front factor" x^a (1−x)^b / (a B(a,b)), in log space so a cell
// with α = 40 does not underflow before it is logged.
inline double log_front(double a, double b, double x)
{
    return std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x) - std::log(a);
}
}  // namespace detail

// log I_x(a,b), stable at both ends: near 0 it is evaluated directly in log space, near 1 as log1p(−I_{1−x}(b,a)).
[[nodiscard]] inline double log_ibeta(double a, double b, double x)
{
    if (x <= 0.0) return -std::numeric_limits<double>::infinity();
    if (x >= 1.0) return 0.0;
    if (x < (a + 1.0) / (a + b + 2.0))
        return detail::log_front(a, b, x) + std::log(detail::betacf(a, b, x));
    const double other = std::exp(detail::log_front(b, a, 1.0 - x)) * detail::betacf(b, a, 1.0 - x);
    return std::log1p(-std::min(other, 1.0 - 1e-300));
}

// Beta(α,β) from residual's published moments. residual builds its Beta as a Jeffreys prior (½,½) plus
// non-negative evidence, so α,β ≥ ½ and α+β ≥ 1 are the MODEL's support, not a clamp chosen here: an EMA'd
// (P,Var) pair can drift a little outside it, and is projected back onto it. Returns false only when the
// moments carry no usable mean.
[[nodiscard]] inline bool beta_from_moments(float P, float var, double &alpha, double &beta)
{
    if (not std::isfinite(P) or not std::isfinite(var) or P < 0.f or P > 1.f) return false;
    const double p = P, v = std::max(0.0, static_cast<double>(var));
    double kappa = v > 0.0 ? p * (1.0 - p) / v - 1.0 : std::numeric_limits<double>::infinity();
    if (not std::isfinite(kappa)) kappa = 1e6;   // Var = 0: a point mass, as concentrated as a double can say
    kappa = std::max(1.0, kappa);
    alpha = std::max(0.5, p * kappa);
    beta  = std::max(0.5, (1.0 - p) * kappa);
    return true;
}

// The cost of covering one residual cell, in nats (see the header). P ≤ 0 is a cell residual itself has
// COLLAPSED (explained by a modelled object, or withheld as speckle): it carries no residual risk.
[[nodiscard]] inline float cell_risk_nats(float P, float var)
{
    if (std::isfinite(P) and P <= 0.f) return 0.f;
    double a = 0.0, b = 0.0;
    if (not beta_from_moments(P, var, a, b)) return kHard;
    const double r = -log_ibeta(a, b, 0.5);
    return std::isfinite(r) ? static_cast<float>(std::max(0.0, r)) : kHard;
}

// residual's dense field, framed by grid_field_meta = [xmin, ymin, cell, w, h] (row-major).
struct BeliefField
{
    std::vector<float> prob, var;
    float xmin = 0.f, ymin = 0.f, cell = 0.f;
    int   width = 0, height = 0;
    [[nodiscard]] bool valid() const
    {
        const auto n = static_cast<std::size_t>(std::max(0, width)) * static_cast<std::size_t>(std::max(0, height));
        return cell > 0.f and n > 0 and prob.size() == n and var.size() == n;
    }
};

// Risk of every cell in `centres`, read off the field at the cell's own position. A cell the field does not
// cover — or no field at all — is HARD: the conservative answer, where we know nothing better.
[[nodiscard]] inline std::vector<float> risks_from_field(std::span<const Eigen::Vector2f> centres, const BeliefField &f)
{
    std::vector<float> out(centres.size(), kHard);
    if (not f.valid()) return out;
    for (std::size_t i = 0; i < centres.size(); ++i)
    {
        const int ix = static_cast<int>(std::floor((centres[i].x() - f.xmin) / f.cell));
        const int iy = static_cast<int>(std::floor((centres[i].y() - f.ymin) / f.cell));
        if (ix < 0 or iy < 0 or ix >= f.width or iy >= f.height) continue;
        const auto k = static_cast<std::size_t>(iy) * static_cast<std::size_t>(f.width) + static_cast<std::size_t>(ix);
        out[i] = cell_risk_nats(f.prob[k], f.var[k]);
    }
    return out;
}

// A residual cell a live return lands in, within the body's height band, is a MEASUREMENT of something the body
// would hit, not a belief about it: it goes back to hard. Cells are keyed on residual's own lattice (the cell of
// size `cell_m` centred on each centre), so "lands in" is cell membership and needs no distance of its own.
// Room-frame returns (xs,ys,zs), the same buffer the approach checks read. Returns how many cells it hardened.
inline int harden_corroborated(std::span<const Eigen::Vector2f> centres, float cell_m, std::vector<float> &risk,
                               std::span<const float> xs, std::span<const float> ys, std::span<const float> zs,
                               float body_z_lo, float body_z_hi)
{
    if (cell_m <= 0.f or risk.size() != centres.size() or centres.empty()) return 0;
    // residual's lattice has an arbitrary origin (its grid starts at the room's bbox), but every centre sits on
    // it, so ONE centre fixes it: corner o = c0 − h. A point p then lies in lattice square floor((p − o)/cell),
    // and a centre c in round((c − h − o)/cell) — the same index for c and for every return inside c's square.
    const float ox = centres.front().x() - 0.5f * cell_m, oy = centres.front().y() - 0.5f * cell_m;
    const auto key = [](std::int64_t ix, std::int64_t iy) { return (ix << 32) ^ (iy & 0xffffffff); };
    const auto point_key = [&](float x, float y)
    { return key(static_cast<std::int64_t>(std::floor((x - ox) / cell_m)),
                 static_cast<std::int64_t>(std::floor((y - oy) / cell_m))); };
    const auto centre_key = [&](const Eigen::Vector2f &c)
    { return key(static_cast<std::int64_t>(std::lround((c.x() - 0.5f * cell_m - ox) / cell_m)),
                 static_cast<std::int64_t>(std::lround((c.y() - 0.5f * cell_m - oy) / cell_m))); };
    std::unordered_set<std::int64_t> hit;
    const std::size_t n = std::min({xs.size(), ys.size(), zs.size()});
    hit.reserve(n / 4 + 1);
    for (std::size_t i = 0; i < n; ++i)
        if (zs[i] >= body_z_lo and zs[i] <= body_z_hi)
            hit.insert(point_key(xs[i], ys[i]));
    int hardened = 0;
    for (std::size_t i = 0; i < centres.size(); ++i)
        if (std::isfinite(risk[i]) and hit.contains(centre_key(centres[i])))
        { risk[i] = kHard; ++hardened; }
    return hardened;
}

}  // namespace rc::residual_risk
