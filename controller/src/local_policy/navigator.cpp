// navigator.cpp — see navigator.h.
#include "navigator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>

namespace rc::policy
{
namespace
{
constexpr float kPiF = 3.14159265f;
}

void Navigator::set_goal(const Goal& goal, const CostToGo::Grid& grid, std::function<bool(const Eigen::Vector2f&)> free,
                         std::function<float(const Eigen::Vector2f&)> extra, std::vector<Eigen::Vector2f> goal_points, float trap_radius_m)
{
    goal_ = goal; grid_ = grid; free_ = std::move(free); extra_ = std::move(extra); goals_ = std::move(goal_points); trap_r_ = trap_radius_m;
    traps_.clear();
    monitor_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    free_field_ = CostToGo{};
    free_field_.build(grid_, free_, extra_, goals_);
    last_build_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    trapped_ = free_field_;                     // no traps yet: identical
}

void Navigator::rebuild_trapped()
{
    const auto t0 = std::chrono::steady_clock::now();
    trapped_ = CostToGo{};
    for (const auto& t : traps_) trapped_.add_trap(t.c, t.r, t.cost);
    trapped_.build(grid_, free_, extra_, goals_);
    last_build_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

float Navigator::max_finite(const CostToGo& f) const
{
    float m = 0.f;
    for (int j = 0; j < grid_.h; ++j)
        for (int i = 0; i < grid_.w; ++i)
        {
            const float v = f.at(grid_.origin + Eigen::Vector2f(i * grid_.cell, j * grid_.cell));
            if (std::isfinite(v)) m = std::max(m, v);
        }
    return m;
}

Eigen::Vector2f Navigator::downhill(const Eigen::Vector2f& p) const
{
    const float h = grid_.cell;
    const float fx1 = trapped_.at(p + Eigen::Vector2f(h, 0.f)), fx0 = trapped_.at(p - Eigen::Vector2f(h, 0.f));
    const float fy1 = trapped_.at(p + Eigen::Vector2f(0.f, h)), fy0 = trapped_.at(p - Eigen::Vector2f(0.f, h));
    if (std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0))
    {
        const Eigen::Vector2f g(fx1 - fx0, fy1 - fy0);
        if (g.norm() > 1e-6f) return -g.normalized();
    }
    // beside a wall a neighbour is +inf: the lowest of 32 probes 0.15 m away
    float best = trapped_.at(p); Eigen::Vector2f dir = Eigen::Vector2f::Zero();
    for (int k = 0; k < 32; ++k)
    {
        const Eigen::Vector2f u(std::cos(2.f * kPiF * k / 32.f), std::sin(2.f * kPiF * k / 32.f));
        const float v = trapped_.at(p + 0.15f * u);
        if (v < best) { best = v; dir = u; }
    }
    return dir;
}

bool Navigator::plan_exists_from(const Eigen::Vector2f& robot) const
{
    const int W = grid_.w, H = grid_.h;
    if (W <= 0 or H <= 0) return false;
    auto centre = [&](int i, int j) { return Eigen::Vector2f(grid_.origin + Eigen::Vector2f(i * grid_.cell, j * grid_.cell)); };
    auto idx = [&](int i, int j) { return j * W + i; };
    std::vector<char> ok(size_t(W * H), 0), seen(size_t(W * H), 0);
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i)
        {
            const Eigen::Vector2f c = centre(i, j);
            if (not free_(c)) continue;
            bool blocked = false;
            for (const auto& t : traps_)
                if ((c - t.c).norm() < t.r and (c - t.c).dot(t.dir) > 0.f) { blocked = true; break; }
            ok[idx(i, j)] = not blocked;
        }
    std::queue<int> q;
    for (const auto& g : goals_)
    {
        const int i = int(std::lround((g.x() - grid_.origin.x()) / grid_.cell)), j = int(std::lround((g.y() - grid_.origin.y()) / grid_.cell));
        if (i >= 0 and j >= 0 and i < W and j < H and ok[idx(i, j)] and not seen[idx(i, j)]) { seen[idx(i, j)] = 1; q.push(idx(i, j)); }
    }
    while (not q.empty())
    {
        const int k = q.front(); q.pop();
        const int i = k % W, j = k / W;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di)
            {
                const int ni = i + di, nj = j + dj;
                if ((di == 0 and dj == 0) or ni < 0 or nj < 0 or ni >= W or nj >= H) continue;
                if (di != 0 and dj != 0 and (not ok[idx(ni, j)] or not ok[idx(i, nj)])) continue;   // no corner cutting
                const int n = idx(ni, nj);
                if (ok[n] and not seen[n]) { seen[n] = 1; q.push(n); }
            }
    }
    for (int j = 0; j < H; ++j)
        for (int i = 0; i < W; ++i)
            if (seen[idx(i, j)] and (centre(i, j) - robot).norm() <= kReachRadiusM) return true;
    return false;
}

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s;
    in.goal = &goal_; in.field = &trapped_; in.progress_field = &free_field_;
    s.d = decide(in, p, &monitor_);
    if (not s.d.prediction_failed) return s;
    const Eigen::Vector2f dir = downhill(in.pose.p);
    const Trap t{in.pose.p + trap_r_ * dir, dir, trap_r_, 2.f * max_finite(free_field_)};
    if (std::ranges::any_of(goals_, [&](const Eigen::Vector2f& g) { return (g - t.c).norm() < t.r; }))
    { s.trap_suppressed_by_goal = true; return s; }
    traps_.push_back(t);
    rebuild_trapped();
    s.trap = t;
    s.unreachable = not plan_exists_from(in.pose.p);
    return s;
}
}  // namespace rc::policy
