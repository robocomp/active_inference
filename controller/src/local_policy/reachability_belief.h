// reachability_belief.h — F1 (2026-10-21 doorway-release spec): "Unreachable" is an ACCUMULATED belief about a PLACE, not
// the verdict of one tick. Eigen + std only (no Qt/DSR); the Navigator owns one per goal (navigator.h) and decides what each
// tick observed, so the controller and the sim2d harness run the same code. This class holds the places, the evidence algebra
// and the decision.
//
// ★ WHY: a Stuck is a statement about ONE decision from ONE (often moving) state. rtD1b: 4 of 4 doorway Stucks were followed on
//   the very next tick by Drive with ~70 eligible motions, and each one handed the crossing back. The task layer needs a
//   statement about the PLACE: "from here no motion escapes".
//
// HYPOTHESES (per place): U = "no motion from here escapes" (none both stays within the policy's risk levels and shortens the
//   goal's remaining time); R = "some do". L = log P(U)/P(R) at the robot's place.
//
// THE GOAL'S REMAINING TIME (fix round 1, C1) is the one progress measure: T = route / v_max + |dyaw| / w_max — the field's
//   route metres at v_max, plus, for a goal that asks for a heading (Heading; Place with a yaw), the turn still owed at w_max.
//   A heading goal's or a final turn's progress is time like any descent (round 0 measured descent only, so every turn was
//   "no progress" and a Heading goal — route ~0 — was handed back on its first tick).
//
// EVIDENCE
//   · A CLAIM: every Drive commits to a plan. When that plan ENDS — completed, completed early, or replaced because its
//     remaining sweep turned risky — the realized shortening of T from the claim's place, r, is compared with what the plan
//     predicted for the time that actually elapsed, D(el) (the plan rolled out to el exactly as decide() predicts its holds:
//     latency coast, phase 1 from the claim's velocity, phase 2 entered while moving). sigma = the ProgressMonitor's own
//     resolution (field resolution over the displacement (+) pose drift over el, / v_max; (+) the relative heading sigma /
//     w_max for a goal that asks for a heading):
//       under R  r ~ N(D, sigma^2)          (it delivered what it predicted)
//       under U  r ~ N(min(D, 0), sigma^2)  (it cannot advance past the place — bounded by the obstruction, taken AT the place;
//                                            a predicted retreat is equally possible under both and says nothing)
//       evidence = ((r - D)^2 - (r - mu_U)^2) / (2 sigma^2) — continuous, no clamp. A plan replaced 0.1 s in predicted ~nothing
//       and says ~nothing; a back-up-and-realign plan (D ~ 0) says ~nothing (I3); one that stopped short says much.
//       Dropped: an interruption, a Blind tick, a trap that rebuilt the route, a new goal.
//   · A COUNT — every decision is also a statement about its OPTIONS: "none of my n effective options escapes" has
//       P(0 of n | U) = 1, P(0 of n | R) = integral_0^1 (1 - theta)^n dtheta = 1/(n+1)  =>  +log(n+1),
//     n = Decision::n_effective (distinct end poses at the place resolution). It is GRADED by the probability that no option
//     really shortens T: from the best shortening any ELIGIBLE option offers over the policy's own two steps (a candidate,
//     then a coarse continuation), along the sweeps (Decision::best_shortening_s), D_best, and two hypotheses of equal prior —
//     NONE: D_best ~ N(0, sigma^2); SOME: the true shortening is uniform on (0, T] (no escape shortens more than what remains),
//     p(D_best) = [Phi((T - D_best)/sigma) - Phi(-D_best/sigma)] / T — so P(none | D_best) needs no new scale. No eligible
//     option at all: 1 (a full count). Weighted by the EXPOSURE w = exp(-coast / cell): a moving decision's options all carry
//     the coast; at 0.35 m/s, WheelDecel 0.5 (coast ~0.16 m) w ~ 0.04.
//   · Release(stop), a continued plan, Blind, Arrived, a sensing contradiction, a frozen pose: nothing.
//
// PRIOR: the base rate of what the map cannot hold — UnseenContact::kPriorBlocked = 0.01, "the fraction of commanded motions
//   blocked by something the LiDAR cannot see": L0 = logit(0.01) = -4.6, from the reset on, the same at every place; then the
//   claims and counts speak. (Fix round 2, live rtD1c: a prior from the planner's price, (priced - length) / (weight x body
//   length), summed correlated residual-cell log-odds as independent nats — 22.95 / 81.65 / 5.15 nats — and handed goals
//   back on their first tick. A per-blob existence prior is PARKED.)
//
// PLACE: the cost-to-go field's cells; each keeps the sum of the evidence gathered in it, and
//   L(here) = prior(here) + sum_cells exp(-|cell - here| / cell) x evidence(cell)
//   (M2: the kernel is NOT normalised — evidence gathered in this cell counts once, a neighbour's at e^-1 — and the prior is
//   added once, not per cell). Evidence is folded only while the goal's remaining time is finite (fix round 2). Distance FROM the place, not travelled: a dither returning to a cell recalls it. Cells beyond
//   10 cell lengths (weight e^-10 = 4.5e-5) are skipped in the sum (M1; they are kept, the robot may come back).
//   FROZEN POSE (rtD1b: the pose did not move for 3 ticks at 0.34 m/s): identical to the last sample while the base moves above
//   its deadband — not an observation.
//
// DECISION (the objective's own terms). Waiting for the next observation costs P(U) x dt_obs (wasted if this is a trap);
//   handing back forfeits P(R) x T (the goal's remaining time). Hand back iff L > bar = log(T / dt_obs). dt_obs is the
//   measured interval BETWEEN OBSERVATIONS (I1) — independent of the tick rate; (fix round 2) before a second observation there
//   is no bar (+inf) — no decision on an interval not yet measured. The progress
//   scale of the score cancels: both terms are times.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include "policy_types.h"

namespace rc::policy
{
class ReachabilityBelief
{
public:
    void reset(float cell_m, float body_r_m) { *this = ReachabilityBelief{}; cell_ = std::max(cell_m, 1e-3f); r_ = std::max(body_r_m, 0.f); }

    // Every tick: where the robot is. Returns false for a FROZEN pose sample (nothing read on it is an observation).
    bool tick(const Pose2& pose, float v_now, float w_now, const BaseLimits& base)
    {
        const bool moving = std::abs(v_now) >= base.v_min_eff or std::abs(w_now) >= base.w_min_eff;
        const bool frozen = has_pose_ and moving and pose.p == last_pose_.p and pose.yaw == last_pose_.yaw;
        has_pose_ = true; last_pose_ = pose; here_ = key(pose.p);
        return not frozen;
    }
    // An observation (a decision) was made at now_s. ★ Fix round 2 (live rtD1c): the bar is priced only on a MEASURED interval
    //   between two observations — the first observation's own plan duration made the bar finite on tick 1, and a goal was
    //   handed back before the robot had moved
    void observed(double now_s)
    {
        dt_obs_ = has_obs_ ? float(now_s - last_obs_t_) : 0.f;
        has_obs_ = true; last_obs_t_ = now_s;
    }
    // An interruption (Pause / Halt / a hold): the places are kept, the next interval is a gap, not an observation interval
    void interrupt() { has_obs_ = false; }

    float prior() const { return prior_; }

    // COUNT evidence: no escape among n effective options
    static float no_escape(int n) { return n > 0 ? std::log(float(n) + 1.f) : 0.f; }
    // CLAIM evidence: predicted shortening D, realized r, resolution sigma (same units)
    static float claim(float D, float r, float sigma)
    {
        if (not (sigma > 0.f) or not std::isfinite(D) or not std::isfinite(r)) return 0.f;
        const float mu_u = std::min(D, 0.f);
        return ((r - D) * (r - D) - (r - mu_u) * (r - mu_u)) / (2.f * sigma * sigma);
    }
    float coast_m(float v_now, float w_now, float latency_s, const BaseLimits& b) const
    {
        const float v = std::abs(v_now), w = std::abs(w_now), lat = std::max(latency_s, 0.f);
        const float lin = v * lat + (b.d_lin > 0.f ? v * v / (2.f * b.d_lin) : 0.f);
        const float rot = w * lat + (b.d_rot > 0.f ? w * w / (2.f * b.d_rot) : 0.f);
        return lin + r_ * rot;
    }
    float exposure(float v_now, float w_now, float latency_s, const BaseLimits& b) const { return std::exp(-coast_m(v_now, w_now, latency_s, b) / cell_); }
    // Fold evidence into the place `at`, gathered while the goal's remaining time was T_goal. ★ Fix round 2 (live rtD1c): at
    //   T = inf (the field is infinite at the robot — a statement about the MAP, not about the robot's options; the no-route
    //   clock reports it) nothing is folded: 105 s of releases against an infinite cross field stored ~17.9 nats that were
    //   released against the finite route in one tick
    void add(const Eigen::Vector2f& at, float nats, float T_goal)
    { if (std::isfinite(T_goal) and nats != 0.f and std::isfinite(nats)) cells_[key(at)] += nats; }

    float log_odds() const
    {
        float L = prior_;
        for (const auto& [k, e] : cells_)
        {
            const float d = std::hypot(float(k.first - here_.first), float(k.second - here_.second));
            if (d <= kReachCells) L += std::exp(-d) * e;
        }
        return L;
    }
    float p_unreachable() const { return 1.f / (1.f + std::exp(-log_odds())); }
    // bar = log(T / dt_obs); +inf while no observation interval is known or T is not finite
    float bar(float T_goal) const
    {
        if (not std::isfinite(T_goal) or not (dt_obs_ > 0.f)) return INFINITY;
        return std::log(std::max(T_goal, 0.f) / dt_obs_);
    }
    bool hand_back(float T_goal) const { return log_odds() > bar(T_goal); }
    float dt_obs() const { return dt_obs_; }

    static constexpr float kFreeBase = 1.f - 0.01f;   // 1 - UnseenContact::kPriorBlocked (unseen_contact.h)
    static constexpr float kReachCells = 10.f;        // M1: e^-10 = 4.5e-5 of a cell's evidence — skipped in the sum

private:
    std::pair<std::int32_t, std::int32_t> key(const Eigen::Vector2f& p) const
    { return {std::int32_t(std::floor(p.x() / cell_)), std::int32_t(std::floor(p.y() / cell_))}; }
    std::map<std::pair<std::int32_t, std::int32_t>, float> cells_;
    std::pair<std::int32_t, std::int32_t> here_{0, 0};
    float cell_ = 0.05f, r_ = 0.f, dt_obs_ = 0.f, prior_ = std::log(0.01f / 0.99f);
    bool has_pose_ = false, has_obs_ = false; double last_obs_t_ = 0.0; Pose2 last_pose_{};
};
}  // namespace rc::policy
