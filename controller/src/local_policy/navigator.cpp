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
    belief_.reset(grid.cell, trap_radius_m);   // F1: a new goal knows nothing about its places
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

Navigator::Step Navigator::step(PolicyInput in, const PolicyParams& p)
{
    Step s = step_core(in, p);
    // ★ F1 — fold this tick into the reachability belief (reachability_belief.h derives every likelihood used here).
    //   An OBSERVATION is a decision decide() made this tick on a fresh pose sample (a continued plan searched nothing; Blind /
    //   Arrived / a sensing contradiction / Release(stop) say nothing about the place). A decision that found NO escape (Stuck,
    //   a no-gain release, a repositioning Drive) is folded at once; one that CLAIMED an escape is held as a claim and folded
    //   when its plan ends, at the place and with the exposure it was claimed with.
    using M = Decision::Mode;
    const Decision& d = s.d;
    const bool fresh = belief_.tick(in.pose, in.v_now, in.w_now, in.now_s, in.base, p.dt);
    if (claim_.pending and s.replanned)              // the claimed plan ended this tick: judged, completed early, or dropped
    {
        // ★ WHEN a claim is judged: by the ProgressMonitor when its plan completes (Decision::judged), or — for a Drive — on
        //   the Navigator's EARLY completion, which is a per-PHASE success (navigator.h round 5): a "back up, then forward" plan
        //   completes its forward phase early with a nil NET descent from where the escape was claimed, so the early completion
        //   is graded on the descent like any judged Drive (below). A plan REPLACED because its remaining sweep turned risky, or
        //   interrupted, never had the time to deliver: no evidence (measured: judging those read a plan replaced 0.1 s after
        //   its claim — before the base had moved through the latency — as a failed escape, and handed back jamb_1p6cm_start on
        //   18 of 20 seeds).
        // ★ HOW a Drive claim is graded: phi = realized / predicted descent on the field from the claim's place (both read now,
        //   on the current field), the prediction only if the field resolves it (the monitor's own resolution: the field's over
        //   the displacement and the pose drift over the elapsed time, at slope 1 m/m). Why graded: under U the robot still
        //   inches 3-9 cm into sim2d blocked_doorway's blocked face per attempt, which a binary verdict (any descent above the
        //   resolution = delivered) reads as a full escape. Measured on the final model it is a refinement, not the cure:
        //   binary 20/20 handed back, median 27.0 s / max 56.0 s; graded 20/20, median 25.3 s / max 45.2 s.
        int verdict = d.judged;
        if (verdict == 0 and s.completed_early) verdict = claim_.drive ? 2 : +1;   // 2: judged here, on the descent
        float nats = 0.f;
        if (verdict != 0 and claim_.drive)
        {
            const float realized = field_.at_length(claim_.at) - field_.at_length(in.pose.p);
            const double el = in.now_s - claim_.t0;
            const float sd = in.pose_drift_sigma_m > 0.f and in.pose_drift_time_s > 0.f
                           ? in.pose_drift_sigma_m * std::sqrt(2.f * (1.f - std::exp(-float(el) / in.pose_drift_time_s))) : 0.f;
            const float res = std::hypot(field_.resolution_m((in.pose.p - claim_.at).norm()), sd);
            if (claim_.dF_pred > res and std::isfinite(realized))      // a prediction the field can resolve at all
            {
                const float phi = std::clamp(realized / claim_.dF_pred, 0.f, 1.f);
                nats = claim_.w * (phi * ReachabilityBelief::delivered(claim_.q, claim_.n) + (1.f - phi) * ReachabilityBelief::no_escape(claim_.n));
            }
        }
        else if (verdict > 0) nats = claim_.w * ReachabilityBelief::delivered(claim_.q, claim_.n);
        else if (verdict < 0) nats = claim_.w * ReachabilityBelief::no_escape(claim_.n);
        if (fresh) belief_.add(claim_.at, nats);
        claim_ = Claim{};
    }
    const bool is_stop = d.mode == M::Release and d.v == 0.f and d.w == 0.f;
    if (fresh and s.replanned and not d.sensing_contradiction and not is_stop and (d.mode == M::Drive or d.mode == M::Release or d.mode == M::Stuck))
    {
        belief_.set_prior(d.n_candidates);
        const float w = belief_.exposure(in.v_now, in.w_now, in.latency_s, in.base);
        // ★ A Drive claims an escape only if its PLAN predicts a descent the field resolves (dF at its holds' end as executed,
        //   against CostToGo::resolution_m over at least that separation): a Drive scored on a two-step rate whose executed plan
        //   descends nothing measurable — "back up, then forward" ending behind its start — is a REPOSITIONING decision, and the
        //   ProgressMonitor would call it "too small to judge" (measured: sim2d blocked_doorway seed 3, those retreats of 0.4 m
        //   from the blocked face were 26 of 62 Drive decisions and carried no evidence). It is read as finding no escape.
        const float dF = d.plan_descent1 + d.plan_descent2;
        const bool resolvable = d.mode == M::Drive and std::isfinite(dF) and dF > field_.resolution_m(std::abs(dF));
        const bool escape = resolvable or (d.mode == M::Release and d.log_p_cont > d.log_p_cont_now);
        const float q = 1.f - std::exp(d.log_p_safe);
        if (not escape) belief_.add(in.pose.p, w * ReachabilityBelief::no_escape(d.n_candidates));
        else if (d.armed) claim_ = Claim{.pending = true, .drive = d.mode == M::Drive, .at = in.pose.p, .q = q, .w = w, .n = d.n_candidates,
                                         .dF_pred = d.plan_descent1 + d.plan_descent2, .t0 = in.now_s};
        else belief_.add(in.pose.p, w * ReachabilityBelief::delivered(q, d.n_candidates));
    }
    s.unreach_L = belief_.log_odds();
    s.unreach_bar = belief_.bar(field_.at_length(in.pose.p), in.base.v_max);
    // the hold-or-hand-back question is asked of every tick that is not the goal itself
    s.hand_back = d.mode != M::Arrived and not s.unreachable and s.unreach_L > s.unreach_bar;
    // A hand back is the verdict on THIS attempt: the caller hands the goal back, and if the same goal is asked for again (the
    // passage re-claims a crossing at once — rtD1b) that is a new attempt, which starts from the map's word like any new goal;
    // kept, the evidence would burn every retry on its first tick
    if (s.hand_back) { belief_.reset(grid_.cell, trap_r_); claim_ = Claim{}; }
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
