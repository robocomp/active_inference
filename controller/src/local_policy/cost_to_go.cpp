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
    if (w <= 0 or h <= 0 or g.cell <= 0.f) return;
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
                if (nd < field_[nidx]) { field_[nidx] = nd; pq.push({nd, nidx}); }
            }
    }
}

float CostToGo::at(const Eigen::Vector2f& p) const
{
    if (field_.empty() or g_.w < 2 or g_.h < 2) return kInf;
    const float fx = (p.x() - g_.origin.x()) / g_.cell, fy = (p.y() - g_.origin.y()) / g_.cell;
    if (fx < 0.f or fy < 0.f or fx > g_.w - 1 + 1e-3f or fy > g_.h - 1 + 1e-3f) return kInf;   // off-grid
    const int i0 = std::min(static_cast<int>(std::floor(fx)), g_.w - 2), j0 = std::min(static_cast<int>(std::floor(fy)), g_.h - 2);
    const float tx = fx - i0, ty = fy - j0;
    const float v00 = field_[static_cast<size_t>(j0) * g_.w + i0], v10 = field_[static_cast<size_t>(j0) * g_.w + i0 + 1];
    const float v01 = field_[static_cast<size_t>(j0 + 1) * g_.w + i0], v11 = field_[static_cast<size_t>(j0 + 1) * g_.w + i0 + 1];
    if (std::isfinite(v00) and std::isfinite(v10) and std::isfinite(v01) and std::isfinite(v11))
        return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
    const float vals[4] = {v00, v10, v01, v11};
    const float wts[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};   // larger weight = nearer
    float best = kInf, bw = -1.f;
    for (int k = 0; k < 4; ++k)
        if (std::isfinite(vals[k]) and wts[k] > bw) { bw = wts[k]; best = vals[k]; }
    if (std::isfinite(best)) return best;
    // ★ only when the four are NON-FREE (inside the inscribed-radius margin); free-but-inf cells are truly unreachable
    if (free_[static_cast<size_t>(j0) * g_.w + i0] or free_[static_cast<size_t>(j0) * g_.w + i0 + 1] or
        free_[static_cast<size_t>(j0 + 1) * g_.w + i0] or free_[static_cast<size_t>(j0 + 1) * g_.w + i0 + 1]) return kInf;
    // all four non-free: nearest finite cell, searched ring by ring, plus the Euclidean distance to its centre
    const int ci = std::clamp(static_cast<int>(std::lround(fx)), 0, g_.w - 1), cj = std::clamp(static_cast<int>(std::lround(fy)), 0, g_.h - 1);
    const int rmax = std::max(g_.w, g_.h);
    for (int r = 1; r <= rmax; ++r)
    {
        float bd = kInf, bv = kInf;
        for (int j = std::max(0, cj - r); j <= std::min(g_.h - 1, cj + r); ++j)
            for (int i = std::max(0, ci - r); i <= std::min(g_.w - 1, ci + r); ++i)
            {
                if (std::max(std::abs(i - ci), std::abs(j - cj)) != r) continue;
                const float v = field_[static_cast<size_t>(j) * g_.w + i];
                if (not std::isfinite(v)) continue;
                const float d = (Eigen::Vector2f(g_.origin.x() + i * g_.cell, g_.origin.y() + j * g_.cell) - p).norm();
                if (d < bd) { bd = d; bv = v; }
            }
        if (std::isfinite(bv)) return bv + bd;
    }
    return kInf;
}

bool CostToGo::reachable(const Eigen::Vector2f& p) const { return std::isfinite(at(p)); }

void CostToGo::add_trap(const Eigen::Vector2f& p, float radius_m, float cost_m) { traps_.push_back({p, radius_m, cost_m}); }

void CostToGo::decay_traps(float factor)
{
    // ★ kDropCost is memory cleanup (a 1 mm-equivalent trap is noise), not a model gate
    constexpr float kDropCost = 1e-3f;
    for (auto& t : traps_) t.cost *= factor;
    std::erase_if(traps_, [](const Trap& t) { return t.cost < kDropCost; });
}
}  // namespace rc::policy
