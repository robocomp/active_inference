// cost_to_go.cpp — see cost_to_go.h.
#include "cost_to_go.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace rc::policy
{
namespace
{
constexpr float kInf = std::numeric_limits<float>::infinity();
}

void CostToGo::build(const Grid& g, const std::function<bool(const Eigen::Vector2f&)>& free,
                     const std::function<float(const Eigen::Vector2f&)>& extra_cost_m,
                     std::span<const Eigen::Vector2f> goal_points)
{
    g_ = g;
    const int w = g.w, h = g.h;
    seeded_ = 0;
    field_.assign(static_cast<size_t>(std::max(0, w) * std::max(0, h)), kInf);
    length_.assign(field_.size(), kInf);
    if (w <= 0 or h <= 0 or g.cell <= 0.f) { field_.clear(); length_.clear(); free_.clear(); return; }   // consistent: at() -> +inf
    auto centre = [&](int i, int j) { return Eigen::Vector2f(g.origin.x() + i * g.cell, g.origin.y() + j * g.cell); };

    // ★ Free test and entry cost are evaluated ONCE per cell: the callbacks (distance-field lookups) dominate a
    //   Dijkstra over 40k cells, and the relaxation would otherwise call them ~8x each.
    std::vector<float> enter(field_.size(), kInf);
    free_.assign(field_.size(), 0);   // +inf = not free
    for (int j = 0; j < h; ++j)
        for (int i = 0; i < w; ++i)
        {
            const Eigen::Vector2f c = centre(i, j);
            if (not free(c)) continue;
            if (std::ranges::any_of(blocked_, [&](const Trap& t) { return (c - t.p).squaredNorm() <= t.radius * t.radius; })) continue;
            float e = extra_cost_m ? extra_cost_m(c) : 0.f;
            for (const auto& t : traps_)
                if ((c - t.p).squaredNorm() <= t.radius * t.radius) e += t.cost / (2.f * t.radius);   // density
            // ★ clamp: Dijkstra needs non-negative edge costs (a negative risk would be a bug upstream, not a model term)
            enter[static_cast<size_t>(j) * w + i] = std::max(0.f, e);
            free_[static_cast<size_t>(j) * w + i] = 1;
        }

    using Item = std::pair<float, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<>> pq;
    for (const auto& gp : goal_points)
    {
        const int ci = static_cast<int>(std::lround((gp.x() - g.origin.x()) / g.cell));
        const int cj = static_cast<int>(std::lround((gp.y() - g.origin.y()) / g.cell));
        int bi = -1, bj = -1;
        float bd = kInf;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
            {
                const int i = ci + di, j = cj + dj;
                if (i < 0 or j < 0 or i >= w or j >= h or std::isinf(enter[static_cast<size_t>(j) * w + i])) continue;
                const float d = (centre(i, j) - gp).squaredNorm();
                if (d < bd) { bd = d; bi = i; bj = j; }
            }
        if (bi < 0) continue;
        if (field_[static_cast<size_t>(bj) * w + bi] != 0.f) ++seeded_;
        field_[static_cast<size_t>(bj) * w + bi] = 0.f;
        length_[static_cast<size_t>(bj) * w + bi] = 0.f;
        pq.push({0.f, bj * w + bi});
    }
    constexpr float kDiag = 1.41421356f;
    while (not pq.empty())
    {
        const auto [d, idx] = pq.top(); pq.pop();
        if (d > field_[idx]) continue;
        const int i = idx % w, j = idx / w;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
            {
                if (di == 0 and dj == 0) continue;
                const int ni = i + di, nj = j + dj;
                if (ni < 0 or nj < 0 or ni >= w or nj >= h) continue;
                const int nidx = nj * w + ni;
                const float e = enter[nidx];
                if (std::isinf(e)) continue;
                const bool diag = di != 0 and dj != 0;
                // ★ no corner cutting: a diagonal needs both orthogonal neighbours free
                if (diag and (std::isinf(enter[j * w + ni]) or std::isinf(enter[nj * w + i]))) continue;
                const float len = g.cell * (diag ? kDiag : 1.f);
                const float nd = d + len * (1.f + 0.5f * (enter[idx] + e));
                if (nd < field_[nidx]) { field_[nidx] = nd; length_[nidx] = length_[idx] + len; pq.push({nd, nidx}); }
            }
    }
}

float CostToGo::at(const Eigen::Vector2f& p) const { return sample(field_, p); }
float CostToGo::at_length(const Eigen::Vector2f& p) const { return sample(length_, p); }

// The PRICED field decides which cells count (finite / free / nearest); the values come from `vals` (the price itself, or
// the route length of the same cheapest route) — so both read the SAME cells, and are finite at the same points.
float CostToGo::sample(const std::vector<float>& vals_in, const Eigen::Vector2f& p) const
{
    if (field_.empty() or vals_in.size() != field_.size() or g_.w < 2 or g_.h < 2) return kInf;
    const float fx = (p.x() - g_.origin.x()) / g_.cell, fy = (p.y() - g_.origin.y()) / g_.cell;
    if (fx < 0.f or fy < 0.f or fx > g_.w - 1 + 1e-3f or fy > g_.h - 1 + 1e-3f) return kInf;   // off-grid
    const int i0 = std::min(static_cast<int>(std::floor(fx)), g_.w - 2), j0 = std::min(static_cast<int>(std::floor(fy)), g_.h - 2);
    const float tx = fx - i0, ty = fy - j0;
    const size_t k00 = static_cast<size_t>(j0) * g_.w + i0, k10 = k00 + 1, k01 = k00 + g_.w, k11 = k01 + 1;
    const float v00 = vals_in[k00], v10 = vals_in[k10], v01 = vals_in[k01], v11 = vals_in[k11];
    if (std::isfinite(field_[k00]) and std::isfinite(field_[k10]) and std::isfinite(field_[k01]) and std::isfinite(field_[k11]))
        return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
    const float vals[4] = {v00, v10, v01, v11};
    const bool fin[4] = {std::isfinite(field_[k00]), std::isfinite(field_[k10]), std::isfinite(field_[k01]), std::isfinite(field_[k11])};
    const float wts[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};   // larger weight = nearer
    float best = kInf, bw = -1.f;
    for (int k = 0; k < 4; ++k)
        if (fin[k] and wts[k] > bw) { bw = wts[k]; best = vals[k]; }
    if (std::isfinite(best)) return best;
    // ★ only when the four are NON-FREE (inside the inscribed-radius margin); free-but-inf cells are truly unreachable
    if (free_[static_cast<size_t>(j0) * g_.w + i0] or free_[static_cast<size_t>(j0) * g_.w + i0 + 1] or
        free_[static_cast<size_t>(j0 + 1) * g_.w + i0] or free_[static_cast<size_t>(j0 + 1) * g_.w + i0 + 1]) return kInf;
    // all four non-free: nearest finite cell, searched ring by ring, plus the Euclidean distance to its centre
    const int ci = std::clamp(static_cast<int>(std::lround(fx)), 0, g_.w - 1), cj = std::clamp(static_cast<int>(std::lround(fy)), 0, g_.h - 1);
    const int rmax = std::max(g_.w, g_.h);
    float best_d = kInf, best_v = kInf;
    auto visit = [&](int i, int j)
    {
        if (i < 0 or j < 0 or i >= g_.w or j >= g_.h) return;
        const size_t k = static_cast<size_t>(j) * g_.w + i;
        if (not std::isfinite(field_[k])) return;
        const float v = vals_in[k];
        const float d = (Eigen::Vector2f(g_.origin.x() + i * g_.cell, g_.origin.y() + j * g_.cell) - p).norm();
        if (d < best_d) { best_d = d; best_v = v; }
    };
    for (int r = 1; r <= rmax; ++r)
    {
        // ★ p lies within half a cell of cell (ci,cj), so any cell on Chebyshev ring r is >= (r-0.5)*cell away:
        //   once that exceeds the best found, no later ring can beat it (exact nearest by Euclid, not ring order).
        if ((r - 0.5f) * g_.cell >= best_d) break;
        for (int i = ci - r; i <= ci + r; ++i) { visit(i, cj - r); visit(i, cj + r); }   // top and bottom rows
        for (int j = cj - r + 1; j <= cj + r - 1; ++j) { visit(ci - r, j); visit(ci + r, j); }   // sides, no corners twice
    }
    if (std::isfinite(best_v)) return best_v + best_d;
    return kInf;
}

bool CostToGo::reachable(const Eigen::Vector2f& p) const { return std::isfinite(at(p)); }

void CostToGo::add_trap(const Eigen::Vector2f& p, float radius_m, float cost_m)
{
    // ★ precondition guard, not a model gate: density = cost/(2*radius) is undefined for radius <= 0, and a trap with
    //   no cost carries no evidence.
    if (not (radius_m > 0.f) or not (cost_m > 0.f)) return;
    traps_.push_back({p, radius_m, cost_m});
}

void CostToGo::decay_traps(float factor)
{
    // ★ kDropCost is memory cleanup (a 1 mm-equivalent trap is noise), not a model gate
    constexpr float kDropCost = 1e-3f;
    for (auto& t : traps_) t.cost *= factor;
    std::erase_if(traps_, [](const Trap& t) { return t.cost < kDropCost; });
}
}  // namespace rc::policy
