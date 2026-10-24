// door_approach_pose.h — WHERE to stand to use a door, once the consumer has handed a standpoint back (D-A).
// Header-only, Eigen only (no DSR): tests/test_door_approach_pose.cpp.
//
// ★THE LOOP THIS ENDS. The controller does not re-site an object's standpoint any more: when it cannot reach the one it
// was given (an obstacle on the footprint, no route, a pose outside its room) it hands the `approach` affordance back —
// Outcome Unreachable, Infeasible or OutsideRoom. door_concept's pose was a deterministic function of the door (the
// actuation band's midpoint, on the aperture's normal), so the re-armed affordance offered the IDENTICAL pose and the
// controller handed it back again: a hand-back / re-offer loop with no information in it.
// ★THE MODEL. A hand-back is EVIDENCE that the standpoint it names is blocked — not a verdict, and not forever: what blocks
// a standpoint is usually something that moves (a chair, a person, the robot's own map catching up). Each refusal i
// contributes P_i(q) = exp(−age_i/τ) · exp(−|q − p_i|² / 2r²): it decays with age on the memory constant τ, and spreads over
// the robot's FOOTPRINT r (the thing that did not fit). P(refused | q) = 1 − Π(1 − P_i(q)).
// The pose is re-solved over door_concept's OWN feasible set — standoff ∈ [d_min, d_max] (the actuation band) × a lateral
// offset within the support of P(facing the opening) — maximising
//      P(complete | q) · P(inside the room | q) · (1 − P(refused | q))
// where P(complete | q) is the probability that the approach CONTRACT completes there: its predicate is
// door_reach_prob ≥ kApproachCompleteProb (door_pragmatics.h approach_contract), door_reach_prob = P(in band)·P(facing),
// and the robot's own estimate at arrival scatters about q with the pose σ (5×5 Gauss–Hermite). Not raw P(reach): a pose
// with P(reach) 0.5 never satisfies a ≥ 0.90 predicate, and offering it is offering a timeout.
// ★THE SCORE DRIVES THE OFFER'S PROBABILITY (fix round, I-1): o.p = best score now / best score with nothing remembered
// (approach_offer_prob). When hand-backs have used up the band it falls and PragmaticAffordance's own offer/withdraw
// hysteresis withdraws the affordance — continuously, with no counter — and re-offers it as the evidence decays.
// ★TWO σ, TWO ROLES (fix round 2). The pose and the offer's AVAILABILITY are solved at the NOMINAL σ (the pose σ floor), so
// neither depends on the live σ: no availability cliff, and no argmax staircase (1.67 ↔ 1.75 m) churning the epoch as σ
// wanders. The LIVE arrival σ (arrival_sigma: the localiser's pose covariance at the standpoint) only scales the offer's
// VALUE through p_complete_live — a robot that cannot localise well enough to complete the approach is offered it at a
// low value, never withdrawn (a door out of view after a HOLD would otherwise lose it for ever).
// ★REFUSALS LIVE IN THE DOOR'S FRAME, not the room's (M-5). A hand-back is stored as the standpoint's (along, across)
// offsets from the aperture centre, multiplied by the approach side (so a π-re-expression of the aperture — wall_u and
// across_u both flipped, which is what another room's association produces — maps it to the same physical point), and
// keyed by the room the robot was in: a room switch re-expresses the aperture in a new frame and the record still lands
// on the same physical standpoint; a standpoint in another room is not this room's evidence.
// ★ON THE AXIS, NEVER BESIDE IT (2026-10-24, user: "the approach affordance should provide a target for the path planner to a
// vantage position in front of the door"). The feasible set used to include a LATERAL offset within the support of
// P(facing): a hand-back by a bed foot in front of the door re-sited the standpoint sideways, and live the robot followed it
// into the wrong arm of an L-shaped room. The set is now ONE-DIMENSIONAL — the standoff s along the aperture normal, on the
// approach side — searched from the far edge of the actuation band inward, down to s_floor (the body's own reach: a
// standpoint closer than that puts the footprint in the doorway). A hand-back therefore SHORTENS the standoff along the
// axis (the refused footprint is evidence about that stretch of axis) instead of shifting it. Far-to-near order with a
// strictly-better replacement makes the LARGEST standoff win among equally good ones (an open door's plateau).
// ★P(complete) IS A MIXTURE OVER THE DOOR'S STATE, as the contract's measurement is (InteractionState::p_reach =
// P(open)·P(within reach) + P(shut)·P(in band)): a door believed open does not need the robot outside the leaf's arc, so a
// standpoint nearer than the band's near edge can complete it; a door believed shut cannot. p_open = 0 is the old model.
// ⚑ τ (DoorAffordance.ApproachRefusalMemoryS) is a model parameter — how long a blocked standpoint stays blocked — not
// a gate: the evidence fades continuously and nothing switches.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "door_geometry.h"
#include "door_passability.h"   // prob_within, facing_cos, prob_facing_opening
#include "door_room_band.h"     // signed_boundary_distance
#include "door_standpoint.h"    // Pose2, front_pose

namespace rc::door
{

// The approach contract's completion level for door_reach_prob — ONE definition, read by approach_contract and by the
// pose solver that predicts whether the contract can complete.
inline constexpr float kApproachCompleteProb = 0.90f;

struct ApproachMemory
{
    struct Refusal { float a = 0.f, c = 0.f; std::string room; std::int64_t t_ms = 0; };
    std::vector<Refusal> refused;

    // (along, across) of xy in the aperture's frame, times the approach side: invariant under the π re-expression.
    [[nodiscard]] static Eigen::Vector2f door_coords(const Aperture& ap, float side, const Eigen::Vector2f& xy)
    {
        const float sg = side < 0.0f ? -1.0f : 1.0f;
        const Eigen::Vector2f d = xy - ap.centre_xy();
        return {sg * d.dot(ap.wall_u), sg * d.dot(ap.across_u())};
    }
    void refuse(const Aperture& ap, float side, const std::string& room, const Eigen::Vector2f& xy, std::int64_t t_ms)
    {
        const Eigen::Vector2f dc = door_coords(ap, side, xy);
        refused.push_back({dc.x(), dc.y(), room, t_ms});
    }
    // P(a standpoint at q is refused again | the remembered hand-backs), at time now, from the robot's current room.
    [[nodiscard]] float p_refused(const Aperture& ap, float side, const std::string& room, const Eigen::Vector2f& q,
                                  std::int64_t now_ms, float memory_s, float footprint_r) const
    {
        const float tau = std::max(memory_s, 1e-3f), r = std::max(footprint_r, 1e-3f);
        const Eigen::Vector2f dq = door_coords(ap, side, q);
        double p_free = 1.0;
        for (const auto& f : refused)
        {
            if (f.room != room) continue;   // another room's standpoint — not evidence about this one
            const float age = std::max(0.0f, static_cast<float>(now_ms - f.t_ms) * 1e-3f);
            const float k = std::exp(-age / tau) * std::exp(-0.5f * (dq - Eigen::Vector2f(f.a, f.c)).squaredNorm() / (r * r));
            p_free *= 1.0 - std::clamp(static_cast<double>(k), 0.0, 1.0);
        }
        return static_cast<float>(1.0 - p_free);
    }
    // Forget what has decayed to nothing (book-keeping only: an entry at e^-20 changes no decision).
    void prune(std::int64_t now_ms, float memory_s)
    {
        std::erase_if(refused, [&](const Refusal& f)
                      { return static_cast<float>(now_ms - f.t_ms) * 1e-3f > 20.0f * std::max(memory_s, 1e-3f); });
    }
};

struct ApproachProblem
{
    float side = +1.0f;                 // which side of the aperture (approach_side)
    float d_min = 0.0f, d_max = 0.0f;   // the actuation band, from the aperture plane
    float sigma = 0.1f;                 // NOMINAL σ (the pose σ floor): chooses the pose and sets the offer's AVAILABILITY
    float sigma_arrival = 0.1f;         // the LIVE arrival σ (arrival_sigma): only the offer's VALUE (p_complete_live)
    float needed_w = 0.7f;              // the body + margin the opening must show
    float sigma_w = 0.0f;               // Task E: the belief's REPORTED σ of the aperture width (m): the PREDICTED P(complete) is marginalised over it
    float memory_s = 60.0f;             // τ of a refusal's evidence
    float footprint_r = 0.35f;          // the robot's footprint radius
    float complete_at = kApproachCompleteProb;
    float p_open = 0.0f;                // P(the door is open) — the mixture weight of P(complete) (see the header); 0 = shut
    float s_floor = 0.0f;               // nearest standoff searched (the body's reach); ≤ 0 ⇒ the band's near edge (d_min)
    std::string room;                   // the room the robot is in (keys the refusals)
    std::vector<Eigen::Vector2f> polygon;   // that room; empty ⇒ no containment term
};

struct ApproachChoice
{
    Pose2 pose;
    float standoff = 0.0f, lateral = 0.0f;
    float p_reach = 0.0f, p_complete = 0.0f, p_inside = 1.0f, p_refused = 0.0f;
    float p_complete_live = 0.0f;       // P(complete) at the chosen pose under the LIVE arrival σ — scales the offer's value
    float score = 0.0f;                 // P(complete)·P(inside)·(1 − P(refused)) — the offer's probability
};

// door_reach_prob at a robot position r (the agent's own definition: band × facing, under σ).
[[nodiscard]] inline float reach_prob_at(const Aperture& ap, const ApproachProblem& pb, const Eigen::Vector2f& r)
{
    const Eigen::Vector2f c = ap.centre_xy(), n = ap.across_u();
    const Eigen::Vector2f d = r - c;
    const float band = pb.d_max > pb.d_min ? std::max(0.0f, prob_within(d.norm(), pb.d_max, pb.sigma)
                                                          - prob_within(d.norm(), pb.d_min, pb.sigma))
                                           : 0.0f;
    return band * prob_facing_opening(facing_cos(d.x(), d.y(), n.x(), n.y()), ap.w, pb.needed_w, pb.sigma);
}

// P(the contract completes | the robot is sent to q), SMOOTH (fix round 2): the contract completes when the door_reach_prob it
// measures at arrival — band(range; σ_meas)·facing — reaches complete_at. Along the ray through q that is an INTERVAL of
// ranges [r1, r2] (the band is unimodal in range, maximal at the band's midpoint; facing is taken at q's bearing — the
// lateral scatter's effect on it is second order), and the robot's range at arrival ~ N(|q − c|, σ_arr²), so
//      P = Φ((r2 − μ)/σ_arr) − Φ((r1 − μ)/σ_arr)
// in closed form: continuous in q and in both σ — as σ_meas widens the interval shrinks to a point and P goes to 0 with
// it, without the 1 → 0 cliff a coarse quadrature of an indicator produced (review: o.p 1.00 → 0.00 between σ 0.14 and
// 0.15). Empty interval (the band can never read complete_at at this σ_meas) ⇒ 0, honestly.
[[nodiscard]] inline float complete_prob_closed_at_width(const Aperture& ap, const ApproachProblem& pb, const Eigen::Vector2f& q,
                                                         float sigma_meas, float sigma_arr)
{
    const Eigen::Vector2f c = ap.centre_xy(), n = ap.across_u();
    const Eigen::Vector2f d = q - c;
    const float mu = d.norm();
    const float sm = std::max(sigma_meas, 1e-3f), sa = std::max(sigma_arr, 1e-3f);
    const float fac = prob_facing_opening(facing_cos(d.x(), d.y(), n.x(), n.y()), ap.w, pb.needed_w, sm);
    if (not (fac > 0.0f) or pb.d_max <= pb.d_min) return 0.0f;
    const float need = pb.complete_at / fac;              // the band must read at least this
    if (need >= 1.0f) return 0.0f;
    const auto band = [&](float r) { return std::max(0.0f, prob_within(r, pb.d_max, sm) - prob_within(r, pb.d_min, sm)); };
    const float mid = 0.5f * (pb.d_min + pb.d_max);
    if (band(mid) < need) return 0.0f;
    const auto root = [&](float lo, float hi, bool rising)   // band(lo..hi) crosses `need` once
    {
        for (int i = 0; i < 60; ++i)
        {
            const float m = 0.5f * (lo + hi);
            if ((band(m) >= need) == rising) hi = m; else lo = m;
        }
        return 0.5f * (lo + hi);
    };
    const float r1 = root(pb.d_min - 8.0f * sm, mid, true);
    const float r2 = root(mid, pb.d_max + 8.0f * sm, false);
    const auto Phi = [](float z) { return 0.5f * std::erfc(-z / std::numbers::sqrt2_v<float>); };
    return std::clamp(Phi((r2 - mu) / sa) - Phi((r1 - mu) / sa), 0.0f, 1.0f);
}

// ★Task E: …MARGINALISED OVER THE APERTURE WIDTH'S OWN UNCERTAINTY. The contract completes when the width, MEASURED from the
// standpoint, admits the body; what the belief knows of that width is N(ŵ, σ_w²) with σ_w the REPORTED one (the mask's
// disagreement with the belief included, DoorBelief::covariance_reported). So the prediction is the expectation of the
// closed form over w, by 3-point Gauss–Hermite (nodes 0, ±√3 σ_w; weights 2/3, 1/6, 1/6). A width believed 0.75 ± 0.25
// keeps the chance its mask will read wide enough once the robot is there — the offer's VALUE falls with that chance
// instead of the offer vanishing at the first cliff. σ_w = 0 is the single evaluation, bit for bit. (The contract's own
// measurement, door_reach_prob, is untouched: it is read at arrival from the then-current belief.)
[[nodiscard]] inline float complete_prob_band(const Aperture& ap, const ApproachProblem& pb, const Eigen::Vector2f& q,
                                              float sigma_meas, float sigma_arr)
{
    if (not (pb.sigma_w > 0.0f))
        return complete_prob_closed_at_width(ap, pb, q, sigma_meas, sigma_arr);
    constexpr float kNode = 1.7320508f;   // √3
    float p = 0.0f;
    for (const auto& [x, wt] : {std::pair{0.0f, 2.0f / 3.0f}, std::pair{kNode, 1.0f / 6.0f}, std::pair{-kNode, 1.0f / 6.0f}})
    {
        Aperture a = ap;
        a.w = std::max(0.0f, ap.w + x * pb.sigma_w);
        p += wt * complete_prob_closed_at_width(a, pb, q, sigma_meas, sigma_arr);
    }
    return std::clamp(p, 0.0f, 1.0f);
}

// P(complete) under the door-state mixture (header): shut ⇒ the band [d_min, d_max]; open ⇒ anywhere within reach
// [0, d_max] (the arc no longer matters). p_open = 0 is complete_prob_band bit for bit.
[[nodiscard]] inline float complete_prob_closed(const Aperture& ap, const ApproachProblem& pb, const Eigen::Vector2f& q,
                                                float sigma_meas, float sigma_arr)
{
    const float po = std::clamp(pb.p_open, 0.0f, 1.0f);
    const float shut = complete_prob_band(ap, pb, q, sigma_meas, sigma_arr);
    if (not (po > 0.0f)) return shut;
    ApproachProblem within = pb;
    within.d_min = 0.0f;
    return po * complete_prob_band(ap, within, q, sigma_meas, sigma_arr) + (1.0f - po) * shut;
}

// The robot's position σ AT THE STANDPOINT from its localisation covariance (the room↔robot RT edge, (x, y, yaw) in the
// room frame): the translation's mean variance plus the yaw's lever over the standoff it will look at the door from —
// not the far-range mask chain's yaw × CURRENT range (review: that reached the cliff at 3.75 m with a 2° yaw σ, so a far
// door was never approachable — and approach is what brings it close). Floored by the pose σ floor.
[[nodiscard]] inline float arrival_sigma(const Eigen::Matrix3f& P_robot_in_room, float standoff, float floor_m)
{
    const float var_t = 0.5f * (std::max(0.0f, P_robot_in_room(0, 0)) + std::max(0.0f, P_robot_in_room(1, 1)));
    const float var_y = std::max(0.0f, P_robot_in_room(2, 2)) * standoff * standoff;
    const float v = var_t + var_y;
    return std::sqrt(std::max(std::isfinite(v) ? v : 0.0f, floor_m * floor_m));
}

// The feasible set is the AXIS (header): standoffs on a fixed step (the band's width / 6, so the band's own seven samples
// are the old grid's lateral-0 column) from d_max inward to the floor, evaluated FAR TO NEAR with a strictly better score
// replacing the incumbent. Every candidate is front_pose(ap, side, s): on the normal, facing the aperture.
[[nodiscard]] inline ApproachChoice choose_approach_pose(const Aperture& ap, const ApproachProblem& pb,
                                                         const ApproachMemory& mem, std::int64_t now_ms)
{
    constexpr int kNsBand = 7;   // samples across the band (odd: its midpoint is one)
    const float band = std::max(0.0f, pb.d_max - pb.d_min);
    const float step = band > 1e-4f ? band / static_cast<float>(kNsBand - 1) : 0.05f;
    const float s_lo = pb.s_floor > 0.0f ? std::min(pb.s_floor, pb.d_min) : pb.d_min;
    const int n = std::max(1, static_cast<int>(std::floor((pb.d_max - s_lo) / step + 1e-3f)) + 1);

    ApproachChoice best;
    bool have = false;
    for (int k = 0; k < n; ++k)
    {
        const float standoff = pb.d_max - step * static_cast<float>(k);
        const Pose2 q = front_pose(ap, pb.side, standoff);
        const Eigen::Vector2f qxy(q.x, q.y);
        ApproachChoice ch;
        ch.pose = q; ch.standoff = standoff; ch.lateral = 0.0f;
        ch.p_reach    = reach_prob_at(ap, pb, qxy);
        ch.p_complete = complete_prob_closed(ap, pb, qxy, pb.sigma, pb.sigma);
        if (pb.polygon.size() >= 3)
        {   // P(inside the room): the standpoint's signed distance to the polygon under the pose σ
            const float dist = signed_boundary_distance(pb.polygon, qxy);   // > 0 outside
            ch.p_inside = std::isfinite(dist) ? 1.0f - prob_within(0.0f, dist, pb.sigma) : 1.0f;
        }
        ch.p_refused = mem.p_refused(ap, pb.side, pb.room, qxy, now_ms, pb.memory_s, pb.footprint_r);
        ch.score = ch.p_complete * ch.p_inside * (1.0f - ch.p_refused);
        if (not have or ch.score > best.score)
        {
            best = ch;
            have = true;
        }
    }
    // σ_meas stays NOMINAL: door_reach_prob, measured AT the standpoint, uses the range σ floored by the pose floor, and at
    // a 1.5-2 m standoff the chain's yaw lever is below it; what the live σ changes is WHERE the robot ends up relative to
    // the pose it was sent to — the arrival scatter. (Both live made the interval collapse like √(σ* − σ): a steep step.)
    best.p_complete_live = complete_prob_closed(ap, pb, {best.pose.x, best.pose.y}, pb.sigma, pb.sigma_arrival);
    return best;
}

// THE OFFER'S PROBABILITY: the share of the band's serviceability the hand-backs have left — the best score now over the
// best score with nothing remembered. 1 with no hand-back (the old unconditional offer), → 0 as the band is used up, so
// PragmaticAffordance's offer/withdraw hysteresis withdraws an exhausted approach and re-offers it as the evidence decays.
// A band that cannot be served even free (free score 0: the arrival σ too wide for the contract) is honestly 0.
[[nodiscard]] inline float approach_offer_prob(const ApproachChoice& now_best, const ApproachChoice& free_best)
{
    return free_best.score > 0.0f ? std::clamp(now_best.score / free_best.score, 0.0f, 1.0f) : 0.0f;
}

// How far past the aperture plane the published CROSS target sits (DoorPragmaticCfg::cross_target_m, 2026-10-24: "a point
// just at the other side of the door"). cross_target_m > 0: that depth. Otherwise the old through-pose: past the leaf's
// swept radius (2·leaf_half_w — the leaf never reaches past its own width from the plane, whatever side or angle) + half
// the body + the swing clearance, never nearer than CrossStandoffM. Plain floats so it is testable without DSR.
[[nodiscard]] inline float cross_target_depth(float cross_target_m, float cross_standoff_m, float leaf_half_w,
                                              float robot_passage_width_m, float swing_clearance_m)
{
    if (cross_target_m > 0.0f) return cross_target_m;
    return std::max(cross_standoff_m, 2.0f * leaf_half_w + 0.5f * robot_passage_width_m + swing_clearance_m);
}

}  // namespace rc::door
