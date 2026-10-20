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
    plan_ = Plan{};
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

// ★ ROUND 4: THE BRAKE IS A SAFETY TAIL, NOT A MANEUVER. The plan is its HOLD phases only — (v1, w1) for horizon_s,
//   then (two-step) (v2, w2) for horizon_s, entered while still moving; it completes at the end of its holds and the
//   next plan is decided from the current velocity. The brake tail is only ever executed when no new plan passes its
//   risk bar (decide() then returns no motion and the base brakes to rest). The remaining-sweep check still includes
//   each rollout's brake tail: "can it still stop" is part of every motion.
float Navigator::plan_length(const PolicyParams& p) const { return p.horizon_s * (plan_.has2 ? 2.f : 1.f); }

std::pair<float, float> Navigator::command_at(float e, const PolicyParams& p) const
{
    if (e < p.horizon_s) return {plan_.v1, plan_.w1};
    if (plan_.has2 and e < 2.f * p.horizon_s) return {plan_.v2, plan_.w2};
    return {0.f, 0.f};
}

std::vector<Pose2> Navigator::remaining_sweep(const Pose2& start, float v0, float w0, float e, const BaseLimits& b, const PolicyParams& p) const
{
    std::vector<Pose2> sw;
    auto append = [&](std::vector<Pose2> part) { if (not sw.empty() and not part.empty()) part.erase(part.begin()); sw.insert(sw.end(), part.begin(), part.end()); };
    if (e < p.horizon_s)
    {
        append(rollout(start, v0, w0, plan_.v1, plan_.w1, p.horizon_s - e, b, p.dt));
        if (plan_.has2) append(rollout(sw.back(), 0.f, 0.f, plan_.v2, plan_.w2, p.horizon_s, b, p.dt));
    }
    else if (plan_.has2) append(rollout(start, v0, w0, plan_.v2, plan_.w2, std::max(0.f, 2.f * p.horizon_s - e), b, p.dt));
    else sw.push_back(start);
    return sw;
}

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s;
    in.goal = &goal_; in.field = &field_; in.progress_field = nullptr;   // one field for everything
    // ★ denoise ONCE: the plan check and decide() see the same obstacles
    const auto obstacles = in.returns_denoised ? std::vector<Return>(in.returns.begin(), in.returns.end()) : denoise_returns(in.returns);
    in.returns = obstacles; in.returns_denoised = true;

    // ---- continue the committed plan unless (a) its remaining sweep is too risky, (b) it completed, (c) the world
    //      changed / the input is blind / the goal is reached (those go to decide(), which answers them)
    if (plan_.active and in.body != nullptr)
    {
        const float e = float(in.now_s - plan_.t0);
        const bool completed = e >= plan_length(p) - 1e-4f;
        const bool blind_or_done = not std::isfinite(in.pose.p.x()) or in.returns_age_s > p.returns_max_age_s or goal_reached(in);
        if (not completed and not blind_or_done)
        {
            const auto sw = remaining_sweep(policy_start(in), in.v_now, in.w_now, e, in.base, p);
            const float lps = log_p_no_contact_cells(*in.body, sw, in.returns, in.sigma_rel_xy, in.sigma_rel_yaw, in.body_geometry_sigma_m);
            const float P = 1.f - std::exp(lps);
            s.plan_remaining_p = P;
            const float bar = plan_.mode == Decision::Mode::Release ? p.release_risk_max : p.risk_eps;
            if (P <= bar)
            {
                s.d = plan_.d;
                const auto [v, w] = command_at(e, p);
                s.d.v = v; s.d.w = w; s.d.prediction_failed = false;
                return s;
            }
            monitor_.reset();                  // (a): replaced, not judged
        }
        else if (not completed) monitor_.reset();   // (c) blind / arrived: not judged
        plan_.active = false;
    }

    s.replanned = true;
    s.d = decide(in, p, &monitor_);
    if (s.d.mode == Decision::Mode::Drive or s.d.mode == Decision::Mode::Release)
    {
        plan_ = Plan{.active = true, .mode = s.d.mode, .v1 = s.d.v, .w1 = s.d.w, .dur1 = s.d.plan_dur1, .v2 = s.d.plan_v2, .w2 = s.d.plan_w2,
                     .dur2 = s.d.plan_dur2, .has2 = s.d.plan_has_cont, .t0 = in.now_s, .d = s.d};
    }
    if (not s.d.prediction_failed) return s;
    const Eigen::Vector2f dir = downhill(in.pose.p);
    const Trap t{in.pose.p + trap_r_ * dir, dir, trap_r_};
    if (std::ranges::any_of(goals_, [&](const Eigen::Vector2f& g) { return (g - t.c).norm() < t.r; }))
    { s.trap_suppressed_by_goal = true; return s; }
    traps_.push_back(t);
    build();
    monitor_.reset();                    // (c): the field changed — old values are not comparable
    s.trap = t;
    s.unreachable = not field_.reachable(in.pose.p);
    // the Release plan stays committed: it is the response to the failure, and the rebuilt field shapes the next plan
    return s;
}
}  // namespace rc::policy
