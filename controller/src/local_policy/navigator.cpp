// navigator.cpp — see navigator.h.
#include "navigator.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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
    build();
}

void Navigator::build()
{
    const auto t0 = std::chrono::steady_clock::now();
    field_ = CostToGo{};
    for (const auto& t : traps_) field_.block_disc(t.c, t.r);
    field_.build(grid_, free_, extra_, goals_);
    last_build_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

Eigen::Vector2f Navigator::downhill(const Eigen::Vector2f& p) const
{
    const float h = grid_.cell;
    const float fx1 = field_.at(p + Eigen::Vector2f(h, 0.f)), fx0 = field_.at(p - Eigen::Vector2f(h, 0.f));
    const float fy1 = field_.at(p + Eigen::Vector2f(0.f, h)), fy0 = field_.at(p - Eigen::Vector2f(0.f, h));
    if (std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0))
    {
        const Eigen::Vector2f g(fx1 - fx0, fy1 - fy0);
        if (g.norm() > 1e-6f) return -g.normalized();
    }
    // beside a wall a neighbour is +inf: the lowest of 32 probes 0.15 m away
    float best = field_.at(p); Eigen::Vector2f dir = Eigen::Vector2f::Zero();
    for (int k = 0; k < 32; ++k)
    {
        const Eigen::Vector2f u(std::cos(2.f * kPiF * k / 32.f), std::sin(2.f * kPiF * k / 32.f));
        const float v = field_.at(p + 0.15f * u);
        if (v < best) { best = v; dir = u; }
    }
    return dir;
}

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s;
    in.goal = &goal_; in.field = &field_; in.progress_field = nullptr;   // one field for everything
    s.d = decide(in, p, &monitor_);
    if (not s.d.prediction_failed) return s;
    const Eigen::Vector2f dir = downhill(in.pose.p);
    const Trap t{in.pose.p + trap_r_ * dir, dir, trap_r_};
    if (std::ranges::any_of(goals_, [&](const Eigen::Vector2f& g) { return (g - t.c).norm() < t.r; }))
    { s.trap_suppressed_by_goal = true; return s; }
    traps_.push_back(t);
    build();
    monitor_.reset();                    // the field changed: old values are not comparable
    s.trap = t;
    s.unreachable = not field_.reachable(in.pose.p);
    return s;
}
}  // namespace rc::policy
