// navigator.cpp — see navigator.h.
#include "navigator.h"

#include <algorithm>
#include <cassert>
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
    failed_predictions_ = 0;
    // ★ Task 8 I2: a world queued for the PREVIOUS goal must not overwrite this goal's free/extra (e.g. a doorway's corridor)
    world_pending_ = false; pending_free_ = {}; pending_extra_ = {};
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

void Navigator::rebuild_on_new_world()
{
    build();
    if (monitor_.pending and monitor_.release) monitor_.reset();
    else if (monitor_.pending) monitor_.f0 = field_.at(monitor_.p0.p);
    if (plan_.active) plan_.phase_f0 = field_.at(plan_.phase_p0);
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

// ★ ROUND 4: THE BRAKE IS A SAFETY TAIL, NOT A MANEUVER. The plan is its HOLD phases only — (v1, w1), then (two-step)
//   (v2, w2) entered while still moving; it completes when its last phase ends and the next plan is decided from the
//   current velocity. The brake tail is only ever executed when no new plan passes its risk bar. The remaining-sweep
//   check still includes each rollout's brake tail: "can it still stop" is part of every motion.
// ★ ROUND 5: COMMIT TO AN OUTCOME, NOT A CLOCK. A phase ends at the end of its hold or as soon as its predicted end
//   state is reached; a plan completed early is judged as SUCCEEDED (the monitor is reset, nothing failed).
std::vector<Pose2> Navigator::remaining_sweep(const Pose2& start, float v0, float w0, float e, const BaseLimits& b, const PolicyParams& p) const
{
    std::vector<Pose2> sw;
    auto append = [&](std::vector<Pose2> part) { if (not sw.empty() and not part.empty()) part.erase(part.begin()); sw.insert(sw.end(), part.begin(), part.end()); };
    const bool in1 = plan_.phase == 1;
    const float v = in1 ? plan_.v1 : plan_.v2, w = in1 ? plan_.w1 : plan_.w2;
    append(rollout(start, v0, w0, v, w, std::max(0.f, p.horizon_s - e), b, p.dt));
    if (in1 and plan_.has2) append(rollout(sw.back(), 0.f, 0.f, plan_.v2, plan_.w2, p.horizon_s, b, p.dt));
    return sw;
}

bool Navigator::phase_outcome_reached(const PolicyInput& in) const
{
    if (plan_.mode != Decision::Mode::Drive) return false;               // a Release is judged on what it bought, at its end
    const bool in1 = plan_.phase == 1;
    const float v = in1 ? plan_.v1 : plan_.v2;
    const float turn = in1 ? plan_.d.plan_turn1 : plan_.d.plan_turn2, descent = in1 ? plan_.d.plan_descent1 : plan_.d.plan_descent2;
    auto wrapf = [](float a) { while (a > kPiF) a -= 2.f * kPiF; while (a <= -kPiF) a += 2.f * kPiF; return a; };
    if (std::abs(v) < in.base.v_min_eff)
    {
        // heading-changing phase: the turn it was predicted to make, or the goal's own heading within its tolerance
        const float done = wrapf(in.pose.yaw - plan_.phase_yaw0);
        if (std::abs(turn) > goal_.tol_yaw and done * turn > 0.f and std::abs(done) >= std::abs(turn) - goal_.tol_yaw) return true;
        if (goal_.yaw and std::abs(wrapf(*goal_.yaw - in.pose.yaw)) <= goal_.tol_yaw) return true;
        return false;
    }
    // translating phase: its predicted descent realized within the field's resolution
    const float f = field_.at(in.pose.p);
    if (not (std::isfinite(f) and std::isfinite(plan_.phase_f0)) or not (descent > 0.f)) return false;
    const float res = field_.resolution_m((in.pose.p - plan_.phase_p0).norm());
    return plan_.phase_f0 - f >= descent - res;
}

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s;
    in.goal = &goal_; in.field = &field_; in.progress_field = nullptr;   // one field for everything
    // ★ denoise ONCE: the plan check and decide() see the same obstacles
    const auto obstacles = in.returns_denoised ? std::vector<Return>(in.returns.begin(), in.returns.end()) : denoise_returns(in.returns);
    in.returns = obstacles; in.returns_denoised = true;

    // ---- continue the committed plan unless (a) its remaining sweep is too risky, (b) it completed (its last phase's
    //      hold elapsed or its predicted outcome was reached), (c) the world changed / the input is blind / the goal is
    //      reached (those go to decide(), which answers them)
    if (plan_.active and in.body != nullptr)
    {
        bool completed = false, early = false;
        for (;;)                                                         // advance through phases that have ended
        {
            const float e = float(in.now_s - plan_.phase_t0);
            const bool hold_done = e >= p.horizon_s - 1e-4f;
            const bool outcome = not hold_done and phase_outcome_reached(in);
            if (not hold_done and not outcome) break;
            early = early or outcome;
            if (plan_.phase == 1 and plan_.has2)
            {
                plan_.phase = 2; plan_.phase_t0 = in.now_s; plan_.phase_yaw0 = in.pose.yaw; plan_.phase_p0 = in.pose.p;
                plan_.phase_f0 = field_.at(in.pose.p);
                continue;
            }
            completed = true; break;
        }
        const bool blind_or_done = not std::isfinite(in.pose.p.x()) or in.returns_age_s > p.returns_max_age_s or goal_reached(in);
        if (not completed and not blind_or_done)
        {
            const float e = float(in.now_s - plan_.phase_t0);
            const auto sw = remaining_sweep(policy_start(in), in.v_now, in.w_now, e, in.base, p);
            const float lps = log_p_no_contact_cells(*in.body, sw, in.returns, in.sigma_rel_xy, in.sigma_rel_yaw, in.body_geometry_sigma_m);
            const float P = 1.f - std::exp(lps);
            s.plan_remaining_p = P;
            const float bar = plan_.mode == Decision::Mode::Release ? p.release_risk_max : p.risk_eps;
            if (P <= bar)
            {
                s.d = plan_.d;
                s.d.v = plan_.phase == 1 ? plan_.v1 : plan_.v2; s.d.w = plan_.phase == 1 ? plan_.w1 : plan_.w2;
                s.d.prediction_failed = false;
                return s;
            }
            monitor_.reset();                  // (a): replaced, not judged
        }
        else if (not completed) monitor_.reset();   // (c) blind / arrived: not judged
        else if (early) { monitor_.reset(); s.completed_early = true; }   // reached its outcome early: SUCCEEDED
        else if (monitor_.pending) monitor_.tau = std::min(monitor_.tau, float(in.now_s - monitor_.t0));   // completed: judge it now
        plan_.active = false;
    }

    s.replanned = true;
    s.d = decide(in, p, &monitor_);
    // Task 8: a pending world is applied now — the completed plan was judged inside decide() on the field it was predicted on
    const bool swap = world_pending_;
    if (swap) { free_ = std::move(pending_free_); extra_ = std::move(pending_extra_); world_pending_ = false; s.world_applied = true; }
    if (s.d.mode == Decision::Mode::Drive or s.d.mode == Decision::Mode::Release)
    {
        plan_ = Plan{.active = true, .mode = s.d.mode, .v1 = s.d.v, .w1 = s.d.w, .dur1 = s.d.plan_dur1, .v2 = s.d.plan_v2, .w2 = s.d.plan_w2,
                     .dur2 = s.d.plan_dur2, .has2 = s.d.plan_has_cont, .t0 = in.now_s, .d = s.d,
                     .phase = 1, .phase_t0 = in.now_s, .phase_yaw0 = in.pose.yaw, .phase_f0 = field_.at(in.pose.p), .phase_p0 = in.pose.p};
    }
    if (not s.d.prediction_failed) { if (swap) rebuild_on_new_world(); return s; }
    ++failed_predictions_;
    // ★ round 5: no route evidence where the field cannot point the way (non-finite here, or a slope below what the
    //   field resolves over two cells) — the Release stands, no trap
    const float f_here = field_.at(in.pose.p);
    const float h = grid_.cell;
    const float fx1 = field_.at(in.pose.p + Eigen::Vector2f(h, 0.f)), fx0 = field_.at(in.pose.p - Eigen::Vector2f(h, 0.f));
    const float fy1 = field_.at(in.pose.p + Eigen::Vector2f(0.f, h)), fy0 = field_.at(in.pose.p - Eigen::Vector2f(0.f, h));
    const bool slope_defined = std::isfinite(f_here) and std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0)
                               and std::hypot(fx1 - fx0, fy1 - fy0) > field_.resolution_m(2.f * h);
    if (not slope_defined) { if (swap) rebuild_on_new_world(); s.trap_no_route_evidence = true; return s; }
    const Eigen::Vector2f dir = -Eigen::Vector2f(fx1 - fx0, fy1 - fy0).normalized();
    const Trap t{in.pose.p + 2.f * trap_r_ * dir, dir, trap_r_};
    assert((t.c - in.pose.p).norm() > t.r);              // the disc never contains the robot's current position
    if (std::ranges::any_of(goals_, [&](const Eigen::Vector2f& g) { return (g - t.c).norm() < t.r; }))
    { if (swap) rebuild_on_new_world(); s.trap_suppressed_by_goal = true; return s; }
    traps_.push_back(t);
    build();
    monitor_.reset();                    // (c): the field changed — old values are not comparable
    s.trap = t;
    s.unreachable = not field_.reachable(in.pose.p);
    // the Release plan stays committed: it is the response to the failure, and the rebuilt field shapes the next plan
    return s;
}
}  // namespace rc::policy
