// reachability_belief.h — F1 (2026-10-21 doorway-release spec): "Unreachable" is an ACCUMULATED belief about a PLACE, not
// the verdict of one tick. Eigen + std only (no Qt/DSR); the Navigator owns one per goal (navigator.h), so the controller and
// the sim2d harness run the same code.
//
// ★ WHY: a Stuck is a statement about ONE decision from ONE (often moving) state — "from here, at this speed, no release is
//   more likely contact-free than not". rtD1b: 4 of 4 doorway Stucks were followed on the very next tick by Drive with ~70
//   eligible motions, and each one handed the crossing back. The task layer needs a statement about the PLACE: "from here no
//   motion escapes". That is a hypothesis, weighed decision by decision against its rival.
//
// HYPOTHESES (per place): U = "no motion from here ESCAPES" — none is both contact-free (the policy's risk levels) and either
//   makes progress on the field or buys continuability; R = "some fraction theta > 0 of the motions escapes", theta uniform.
//   L = log P(U)/P(R) at the robot's place.
// PRIOR: a place is not unknown — the goal was handed to the policy because the map's cost-to-go is finite here, i.e. the map
//   holds a route that leaves this place. That is the statement a decision makes when it finds an escape, and it is weighed as
//   one delivered at rest, at one decision's resolution: L0 = -log(N+1) (N = the candidates a decision searches; the latest
//   decision's, before any decision 0). Measured why it is needed: with L0 = 0 (P(U) = 1/2 at every place) ONE repositioning
//   decision at rest cleared the bar — jamb_1p6cm_start was handed back on its first decision (3 of 20 seeds) and
//   dead_end_release after two back-ups at the stale wall (7 of 20), before the trap mechanism had judged anything.
//   The Navigator decides what each tick observed (navigator.cpp step()); this class holds the places, the evidence algebra
//   and the decision.
//
// LIKELIHOOD of one DECISION (a tick on which decide() searched its N candidates; a tick that only continues a committed plan
//   searched nothing and is no observation):
//   · NO ESCAPE FOUND — Stuck, a Release that buys no continuability (pass (b)), or a REPOSITIONING Drive (one whose own plan
//       predicts no descent the field resolves: "back up, then forward" ending behind its start — chosen for a two-step rate,
//       but the ProgressMonitor would call it "too small to judge"): 0 of N candidates escape.
//       P(0 of N | U) = 1, P(0 of N | R) = integral_0^1 (1 - theta)^N dtheta = 1/(N+1)  =>  evidence = +log(N+1), FOR U.
//       (The same rule of succession, log((k+1)/(N+2)), that the policy's continuability term uses.) This is the F2 review's
//       carried requirement made exact: a no-gain turn shows that some motion is SAFE, never that one ESCAPES, so it can never
//       count against U — a real trap that dithers stop/turn closes (measured: sim2d blocked_doorway).
//   · AN ESCAPE CLAIMED — a Drive whose plan predicts a resolvable descent, or a Release predicted to BUY continuability — is a
//       PREDICTION. It is folded when its plan ends, at the place it was claimed from and with that decision's exposure:
//         DELIVERED as predicted: under R the escape is real unless it touches, under U only if the contact model is wrong
//           about it — the ratio q/(1-q), q = its P(contact), read no finer than one in N+2 (the factor above, the other way
//           round): delivered(q) = logit(max(q, 1/(N+2))) in [-log(N+1), 0), AGAINST U;
//         FAILED: the escape it claimed did not exist — in hindsight a decision that found none: +log(N+1), FOR U. These are
//           the failures whose traps a measured doorway vetoes (Live fix 2 ruling 2): the veto keeps the field open, the
//           belief keeps the evidence;
//         a Drive is GRADED by the fraction of its predicted descent it delivered, phi = realized / predicted (clamped to
//           [0, 1]): U does not forbid local progress — under U the robot advances to the obstruction and no further — so the
//           claim is read as its predicted descent in steps of the field's resolution, one correlated observation whose
//           evidence is the mean of its steps, phi delivered(q) + (1 - phi) log(N+1). A Release keeps the monitor's binary
//           verdict (its claim is continuability, not descent);
//         a plan that ended before it could deliver (replaced because its remaining sweep turned risky, interrupted, the goal
//           or the field changed, a prediction too small to judge): nothing.
//       The Navigator (navigator.cpp step()) says which decisions and judgements happened; the measurements behind each rule
//       are recorded there.
//   · Release(stop): NO observation — its point is that the next decision is made from rest (F2), which answers the question
//       about the place without the coast. (Never against U: the F2 review's carried requirement.)
//   · A continued plan (searched nothing), Blind, a sensing contradiction, Arrived, a frozen pose sample: no observation; a
//       judgement read on a frozen pose sample is no observation either (its realized descent was read from a stale pose).
//
// EXPOSURE (moving decisions): every candidate of a moving decision first carries the coast (latency + braking), so what it says
//   about the place is said about the coast's end, c metres away; it enters at the place kernel's weight there,
//   w = exp(-c / cell). At 0.35 m/s and WheelDecel 0.5 (coast ~0.16 m) w ~ 0.04: a Stuck while moving barely moves the belief —
//   the coast explains it. At rest c = 0, w = 1. c is the base's own braking: |v| latency + v^2/(2 d_lin) plus, for a turn,
//   the body's farthest point r_body (|w| latency + w^2/(2 d_rot)).
//
// PLACE: the evidence is about the place it was gathered at. Places are the cost-to-go field's cells (two positions in one cell
//   are one node of the field — the finest place the policy distinguishes); each cell keeps the sum of the evidence gathered in
//   it, and the belief at the robot is L = sum_cells exp(-|cell - here| / cell_size) x evidence(cell): one cell away counts e^-1.
//   It is distance FROM the place, not distance travelled: a stop/turn dither returning to the same cell recalls what was
//   gathered there (a path-length discount would erase it every turn and keep the trap open — measured). Heading is not part of
//   the place: a turn in place reaches every heading it sweeps, and a turn that buys nothing is no escape.
//   FROZEN POSE (rtD1b 175.34-175.38: the pose did not move for 3 ticks at 0.34 m/s): a pose sample identical to the last one
//   while the base moves above its deadband is a stale sample — its decision is not an observation of where the robot is.
//
// DECISION (the objective's own terms, no constant). The score values a field-descent RATE at progress_scale_m / horizon_s per
//   nat: a tick that does not descend costs v_max dt / progress_scale_m nats (the rate the field — slope 1 m/m — could have
//   descended at), and the goal is worth route_m / progress_scale_m nats (its remaining route on the field). One more tick:
//     cost  = P(U) x v_max dt / s   (wasted if the place is a trap)
//     value = P(R) x route_m / s    (forfeited by a hand back if it is not)
//   Hand back iff cost > value  <=>  L > bar = log(route_m / (v_max dt)), dt = the measured tick period. s cancels: the goal is
//   worth route_m / v_max seconds of driving, a tick costs dt. Evaluated on every tick (a tick that found an escape lowers L).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include "policy_types.h"

namespace rc::policy
{
class ReachabilityBelief
{
public:
    // cell_m: the field's cell (the place resolution); body_r_m: the body's circumscribed radius (a turn's coast)
    void reset(float cell_m, float body_r_m) { *this = ReachabilityBelief{}; cell_ = std::max(cell_m, 1e-3f); r_ = std::max(body_r_m, 0.f); }

    // Every tick: where the robot is now, and the tick period (dt_fallback on a goal's first tick or after an interruption).
    // Returns false for a FROZEN pose sample (identical to the last one while the base moves above its deadband): nothing read
    // on it is an observation.
    bool tick(const Pose2& pose, float v_now, float w_now, double now_s, const BaseLimits& base, float dt_fallback)
    {
        const bool moving = std::abs(v_now) >= base.v_min_eff or std::abs(w_now) >= base.w_min_eff;
        const bool frozen = has_last_ and moving and pose.p == last_pose_.p and pose.yaw == last_pose_.yaw;
        dt_ = has_last_ ? float(std::max(0.0, now_s - last_t_)) : dt_fallback;
        has_last_ = true; last_t_ = now_s; last_pose_ = pose;
        here_ = key(pose.p);
        return not frozen;
    }
    // An interruption (Pause / Halt / a hold): the places are kept, but the next interval is a gap, not a tick period
    void interrupt() { has_last_ = false; }

    // The evidence of one decision (nats, before the exposure weight): it found no escape among n candidates / an escape with
    // P(contact) q was delivered. n <= 0 (nothing searched) = 0.
    static float no_escape(int n) { return n > 0 ? std::log(float(n) + 1.f) : 0.f; }
    static float delivered(float q, int n)
    {
        if (n <= 0) return 0.f;
        const float qq = std::clamp(q, 1.f / (float(n) + 2.f), 0.5f);
        return std::log(qq / (1.f - qq));
    }
    // The coast (m) of the base's own braking from (v, w) after the latency; a turn's as the body's farthest point; and the
    // place kernel's weight at the coast's end — what a decision made at (v, w) says about the place at rest
    float coast_m(float v_now, float w_now, float latency_s, const BaseLimits& b) const
    {
        const float v = std::abs(v_now), w = std::abs(w_now), lat = std::max(latency_s, 0.f);
        const float lin = v * lat + (b.d_lin > 0.f ? v * v / (2.f * b.d_lin) : 0.f);
        const float rot = w * lat + (b.d_rot > 0.f ? w * w / (2.f * b.d_rot) : 0.f);
        return lin + r_ * rot;
    }
    float exposure(float v_now, float w_now, float latency_s, const BaseLimits& b) const { return std::exp(-coast_m(v_now, w_now, latency_s, b) / cell_); }
    // Fold evidence (already exposure-weighted) into the place `at`
    void add(const Eigen::Vector2f& at, float nats) { if (nats != 0.f) cells_[key(at)] += nats; }
    // The map's word on every place (PRIOR above), at the resolution of a decision that searched n candidates
    void set_prior(int n) { prior_ = -no_escape(n); }
    float prior() const { return prior_; }

    // L at the robot's place: the kernel-weighted evidence of every cell (one cell away counts e^-1)
    float log_odds() const
    {
        float L = prior_;
        for (const auto& [k, e] : cells_)
            L += std::exp(-std::hypot(float(k.first - here_.first), float(k.second - here_.second))) * e;
        return L;
    }
    float p_unreachable() const { return 1.f / (1.f + std::exp(-log_odds())); }
    // The hand-back bar log(route_m / (v_max dt)); +inf when no value can be compared (route not finite, v_max or dt not > 0).
    float bar(float route_m, float v_max) const
    {
        if (not std::isfinite(route_m) or not (v_max > 0.f) or not (dt_ > 0.f)) return INFINITY;
        return std::log(std::max(route_m, 0.f) / (v_max * dt_));
    }
    bool hand_back(float route_m, float v_max) const { return log_odds() > bar(route_m, v_max); }
    float last_dt() const { return dt_; }

private:
    std::pair<std::int32_t, std::int32_t> key(const Eigen::Vector2f& p) const
    { return {std::int32_t(std::floor(p.x() / cell_)), std::int32_t(std::floor(p.y() / cell_))}; }
    std::map<std::pair<std::int32_t, std::int32_t>, float> cells_;
    std::pair<std::int32_t, std::int32_t> here_{0, 0};
    float cell_ = 0.05f, r_ = 0.f, dt_ = 0.f, prior_ = 0.f;
    bool has_last_ = false; double last_t_ = 0.0; Pose2 last_pose_{};
};
}  // namespace rc::policy
