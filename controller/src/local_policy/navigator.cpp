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
    trap_veto_ = {};
    monitor_.reset();
    plan_ = Plan{};
    belief_.reset(grid.cell, trap_radius_m);   // F1: a new goal starts from its route's prior (set every tick)
    claim_ = Claim{};
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
    else if (monitor_.pending) monitor_.f0 = field_.at_length(monitor_.p0.p);
    if (plan_.active) plan_.phase_f0 = field_.at_length(plan_.phase_p0);
}

Eigen::Vector2f Navigator::downhill(const Eigen::Vector2f& p) const
{
    const float h = grid_.cell;
    const float fx1 = field_.at_length(p + Eigen::Vector2f(h, 0.f)), fx0 = field_.at_length(p - Eigen::Vector2f(h, 0.f));
    const float fy1 = field_.at_length(p + Eigen::Vector2f(0.f, h)), fy0 = field_.at_length(p - Eigen::Vector2f(0.f, h));
    if (std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0))
    {
        const Eigen::Vector2f g(fx1 - fx0, fy1 - fy0);
        if (g.norm() > 1e-6f) return -g.normalized();
    }
    // beside a wall a neighbour is +inf: the lowest of 32 probes 0.15 m away
    float best = field_.at_length(p); Eigen::Vector2f dir = Eigen::Vector2f::Zero();
    for (int k = 0; k < 32; ++k)
    {
        const Eigen::Vector2f u(std::cos(2.f * kPiF * k / 32.f), std::sin(2.f * kPiF * k / 32.f));
        const float v = field_.at_length(p + 0.15f * u);
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
    // ★ F2 fix round 1: a Release(stop) (v = w = 0, local_policy.cpp) promises REST, and its outcome is reached the moment
    //   the base is at rest — by the same deadband fact that made the stop a candidate (|v| < v_min_eff and |w| < w_min_eff).
    //   Holding it for the rest of horizon_s would only idle; the next decision is to be made from rest, now.
    if (plan_.mode == Decision::Mode::Release and plan_.v1 == 0.f and plan_.w1 == 0.f)
        return std::abs(in.v_now) < in.base.v_min_eff and std::abs(in.w_now) < in.base.w_min_eff;
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
    const float f = field_.at_length(in.pose.p);
    if (not (std::isfinite(f) and std::isfinite(plan_.phase_f0)) or not (descent > 0.f)) return false;
    const float res = field_.resolution_m((in.pose.p - plan_.phase_p0).norm());
    return plan_.phase_f0 - f >= descent - res;
}

// F1 (fix round 1, C1): the heading the goal asks for at `at` — its yaw if set, a Heading goal without one faces goal.p; none
// otherwise (local_policy.cpp target_yaw, the same rule)
std::optional<float> Navigator::goal_heading(const Pose2& at) const
{
    if (goal_.yaw) return *goal_.yaw;
    if (goal_.kind != Goal::Kind::Heading) return std::nullopt;
    const Eigen::Vector2f d = goal_.p - at.p;
    if (d.squaredNorm() < 1e-12f) return std::nullopt;
    return std::atan2(d.y(), d.x());
}

// F1 (fix round 1, C1): the goal's REMAINING TIME at a pose (reachability_belief.h): route metres at v_max (not for a Heading
// goal — it has no place to reach) + the heading still owed at w_max
float Navigator::remaining_time(const Pose2& at, const BaseLimits& b) const
{
    float T = 0.f;
    if (goal_.kind != Goal::Kind::Heading) T += b.v_max > 0.f ? field_.at_length(at.p) / b.v_max : INFINITY;
    if (const auto h = goal_heading(at); h and b.w_max > 0.f) T += std::abs(std::remainder(*h - at.yaw, 2.f * kPiF)) / b.w_max;
    return T;
}

// F1 (fix round 1, I2): the route's surprisal (nats) from the robot's place — the priced field minus its length is
// residual_risk_weight x the integral of the body's nats of doubt over the route; one independent body placement per body
// length (the circumscribed diameter) — reachability_belief.h PRIOR
float Navigator::route_surprisal(const Eigen::Vector2f& at) const
{
    if (goal_.kind == Goal::Kind::Heading or not (route_risk_weight_ > 0.f) or not (trap_r_ > 0.f)) return 0.f;
    const float priced = field_.at(at), length = field_.at_length(at);
    if (not (std::isfinite(priced) and std::isfinite(length))) return 0.f;
    return std::max(0.f, priced - length) / (route_risk_weight_ * 2.f * trap_r_);
}

// F1 (fix round 1): P(no option really shortens the goal's remaining time | the best measured shortening D, resolution sig).
// Two hypotheses, equal prior: NONE — the true best shortening is 0 and D ~ N(0, sig^2); SOME — it is s, uniform on (0, T] (an
// escape cannot shorten more than the time that remains), so p(D) = (1/T) int_0^T N(D; s, sig) ds = [Phi((T-D)/sig) -
// Phi(-D/sig)] / T. No option eligible (D = -inf): 1. Only the resolution and the remaining time set the scale.
static float phi_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }
float Navigator::no_shortening(float D, float sig, float T)
{
    if (not std::isfinite(D)) return 1.f;
    if (not (sig > 0.f) or not (T > 0.f) or not std::isfinite(T)) return D > 0.f ? 0.f : 1.f;
    const float z = D / sig;
    const float p_none = std::exp(-0.5f * z * z) / (sig * 2.50662827f);
    const float p_some = (phi_cdf((T - D) / sig) - phi_cdf(-z)) / T;
    return p_none / (p_none + p_some);
}

// F1 (fix round 1): where the claimed plan predicted the robot to be `el` seconds after the claim — decide()'s own prediction
// of its holds (local_policy.cpp, round 4): the latency coast at the claim's velocity, phase 1 from that velocity, phase 2
// entered while moving; past the last hold, its brake tail
Pose2 Navigator::predicted_pose(const Claim& c, float el) const
{
    auto at_t = [&](const std::vector<Pose2>& sw, float t) { return sw[std::min(sw.size() - 1, std::size_t(std::max(0L, std::lround(t / c.dt))))]; };
    if (el <= 0.f) return c.at;
    if (el <= c.lat) return at_t(rollout(c.at, c.v0, c.w0, c.v0, c.w0, el, c.base, c.dt), el);
    const Pose2 start = c.lat > 0.f ? at_t(rollout(c.at, c.v0, c.w0, c.v0, c.w0, c.lat, c.base, c.dt), c.lat) : c.at;
    const float t1 = el - c.lat;
    const auto sw1 = rollout(start, c.v0, c.w0, c.v1, c.w1, c.hold, c.base, c.dt);
    if (t1 <= c.hold or not c.has2) return at_t(sw1, t1);
    const int k = std::min(int(sw1.size()) - 1, std::max(1, int(std::lround(c.hold / c.dt)))), j = k - 1;
    const Eigen::Vector2f fwd(std::cos(sw1[j].yaw), std::sin(sw1[j].yaw));
    const float v1e = (sw1[k].p - sw1[j].p).dot(fwd) / c.dt, w1e = (sw1[k].yaw - sw1[j].yaw) / c.dt;
    return at_t(rollout(sw1[k], v1e, w1e, c.v2, c.w2, c.hold, c.base, c.dt), t1 - c.hold);
}

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s = step_core(in, p);
    // ★ F1 — fold this tick into the reachability belief (reachability_belief.h derives every likelihood used here).
    using M = Decision::Mode;
    const Decision& d = s.d;
    const bool fresh = belief_.tick(in.pose, in.v_now, in.w_now, in.base);
    // ★ A TRAP rebuilt the field this tick: the evidence was gathered against the old route, and the trap IS the route-level
    //   answer to it (the field now detours) — the belief restarts on the new route (measured round 0: kept, a dead end's
    //   evidence handed dead_end_release back before its detour)
    if (s.trap) { belief_.reset(grid_.cell, trap_r_); claim_ = Claim{}; }
    belief_.set_route_surprisal(route_surprisal(in.pose.p));
    // ★ A CLAIM is judged whenever its plan ENDS — completed, completed early, or replaced because its remaining sweep turned
    //   risky — against what the plan predicted for the time that actually ELAPSED: D(el) = T0 - T(the plan rolled out to el,
    //   as decide() predicts its holds: the latency coast, phase 1 from the claim's velocity, phase 2 entered while moving).
    //   A plan replaced 0.1 s in had predicted ~nothing yet and says ~nothing (round 0, judged against its full prediction:
    //   jamb_1p6cm_start handed back on 18/20 seeds); one that stopped short at an obstruction says much; an early completion
    //   is judged on what it promised by then (round 1 judged it against the full prediction: a plan 0.05 s old against
    //   0.26 m, +143 nats, try101_corridor handed back on 7/20). An interruption, a Blind tick or a trap (the route changed)
    //   drops it.
    if (claim_.pending and s.replanned and d.mode != M::Blind)
    {
        if (fresh)
        {
            const float el = float(in.now_s - claim_.t0);
            const float D = claim_.T0 - remaining_time(predicted_pose(claim_, el), in.base);
            const float r = claim_.T0 - remaining_time(in.pose, in.base);     // realized shortening of the remaining time
            const float sd = in.pose_drift_sigma_m > 0.f and in.pose_drift_time_s > 0.f
                           ? in.pose_drift_sigma_m * std::sqrt(2.f * (1.f - std::exp(-el / in.pose_drift_time_s))) : 0.f;
            // the ProgressMonitor's resolution, in time: field resolution over the displacement (+) pose drift, at v_max;
            // (+) the relative heading sigma at w_max for a goal that asks for a heading
            const float res_route = goal_.kind == Goal::Kind::Heading ? 0.f
                                  : std::hypot(field_.resolution_m((in.pose.p - claim_.at.p).norm()), sd) / std::max(in.base.v_max, 1e-3f);
            const float res_yaw = goal_heading(in.pose) ? in.sigma_rel_yaw / std::max(in.base.w_max, 1e-3f) : 0.f;
            const float ev = ReachabilityBelief::claim(D, r, std::hypot(res_route, res_yaw));
            belief_.add(claim_.at.p, ev);
        }
        claim_ = Claim{};
    }
    else if (claim_.pending and d.mode == M::Blind) claim_ = Claim{};
    const bool is_stop = d.mode == M::Release and d.v == 0.f and d.w == 0.f;
    if (fresh and s.replanned and not d.sensing_contradiction and not is_stop and (d.mode == M::Drive or d.mode == M::Release or d.mode == M::Stuck))
    {
        belief_.observed(in.now_s, d.mode == M::Stuck ? 0.f : d.plan_dur1 + (d.plan_has_cont ? d.plan_dur2 : 0.f));
        if (d.mode == M::Drive)
        {
            // the CLAIM: its committed plan (judged against its own rollout when it ends, above); a back-up or a realignment
            // (I3) is graded like any plan by what it predicted for the time it ran — never counted as "no escape"
            claim_ = Claim{.pending = true, .at = in.pose, .T0 = remaining_time(in.pose, in.base), .t0 = in.now_s,
                           .v0 = in.v_now, .w0 = in.w_now, .lat = in.latency_s, .v1 = d.v, .w1 = d.w, .v2 = d.plan_v2, .w2 = d.plan_w2,
                           .has2 = d.plan_has_cont, .hold = p.horizon_s, .dt = p.dt, .base = in.base};
        }
        // ★ The COUNT "no escape among n options" (+log(n+1)) is a statement about the OPTIONS, not about the plan chosen; it is
        //   graded by P(no option really shortens T | the best shortening any ELIGIBLE option offers over two steps, along the
        //   sweeps — Decision::best_shortening_s), no_shortening() below. Every decision is graded so (Drive, Release, Stuck),
        //   at the coast's exposure. Measured round 1, the forms that failed: grading the CHOSEN plan's executed descent counted
        //   an overshoot-correcting arc 0.14 s from a Place goal's final turn as "no escape" (2/20 handed back); rest poses
        //   instead of sweeps did the same to a 0.075 rad final turn; one step instead of two counted the jamb start's
        //   turn-then-go as "no option" three times in 0.6 s (jamb_1p6cm_start, 1/20); Phi(-D/sigma) graded a geometric zero
        //   (a turn in place at a blocked face) as half a count.
        {
            const float Db = d.best_shortening_s;
            const float sig = std::hypot(goal_.kind == Goal::Kind::Heading ? 0.f
                                             : field_.resolution_m(std::isfinite(Db) ? std::abs(Db) * in.base.v_max : 0.f) / std::max(in.base.v_max, 1e-3f),
                                         goal_heading(in.pose) ? in.sigma_rel_yaw / std::max(in.base.w_max, 1e-3f) : 0.f);
            const float none = no_shortening(Db, sig, remaining_time(in.pose, in.base));
            const float ev = none * belief_.exposure(in.v_now, in.w_now, in.latency_s, in.base) * ReachabilityBelief::no_escape(d.n_effective);
            belief_.add(in.pose.p, ev);
        }
    }
    const float T = remaining_time(in.pose, in.base);
    s.unreach_L = belief_.log_odds();
    s.unreach_bar = belief_.bar(T);
    s.hand_back = d.mode != M::Arrived and not s.unreachable and s.unreach_L > s.unreach_bar;
    // A hand back is the verdict on THIS attempt: the same goal asked again (the passage re-claims a crossing at once — rtD1b)
    // is a new attempt and starts from the route's prior, which holds from the reset on (C1)
    if (s.hand_back) { belief_.reset(grid_.cell, trap_r_); belief_.set_route_surprisal(route_surprisal(in.pose.p)); claim_ = Claim{}; }
    return s;
}

Navigator::Step Navigator::step_core(PolicyInput in, const PolicyParams& p)
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
                plan_.phase_f0 = field_.at_length(in.pose.p);
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
                     .phase = 1, .phase_t0 = in.now_s, .phase_yaw0 = in.pose.yaw, .phase_f0 = field_.at_length(in.pose.p), .phase_p0 = in.pose.p};
    }
    if (not s.d.prediction_failed) { if (swap) rebuild_on_new_world(); return s; }
    ++failed_predictions_;
    // ★ round 5: no route evidence where the field cannot point the way (non-finite here, or a slope below what the
    //   field resolves over two cells) — the Release stands, no trap
    const float f_here = field_.at_length(in.pose.p);
    const float h = grid_.cell;
    const float fx1 = field_.at_length(in.pose.p + Eigen::Vector2f(h, 0.f)), fx0 = field_.at_length(in.pose.p - Eigen::Vector2f(h, 0.f));
    const float fy1 = field_.at_length(in.pose.p + Eigen::Vector2f(0.f, h)), fy0 = field_.at_length(in.pose.p - Eigen::Vector2f(0.f, h));
    const bool slope_defined = std::isfinite(f_here) and std::isfinite(fx1) and std::isfinite(fx0) and std::isfinite(fy1) and std::isfinite(fy0)
                               and std::hypot(fx1 - fx0, fy1 - fy0) > field_.resolution_m(2.f * h);
    if (not slope_defined) { if (swap) rebuild_on_new_world(); s.trap_no_route_evidence = true; return s; }
    const Eigen::Vector2f dir = -Eigen::Vector2f(fx1 - fx0, fy1 - fy0).normalized();
    const Trap t{in.pose.p + 2.f * trap_r_ * dir, dir, trap_r_};
    assert((t.c - in.pose.p).norm() > t.r);              // the disc never contains the robot's current position
    if (std::ranges::any_of(goals_, [&](const Eigen::Vector2f& g) { return (g - t.c).norm() < t.r; }))
    { if (swap) rebuild_on_new_world(); s.trap_suppressed_by_goal = true; return s; }
    if (trap_veto_ and trap_veto_(t.c, t.r)) { if (swap) rebuild_on_new_world(); s.trap_vetoed = true; return s; }   // Live fix 2
    traps_.push_back(t);
    build();
    monitor_.reset();                    // (c): the field changed — old values are not comparable
    s.trap = t;
    s.unreachable = not field_.reachable(in.pose.p);
    // the Release plan stays committed: it is the response to the failure, and the rebuilt field shapes the next plan
    return s;
}
}  // namespace rc::policy
