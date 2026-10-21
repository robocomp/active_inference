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
// ★σ IS THE ARRIVAL σ (the robot's localisation at the door), not the far-range door σ: the contract is measured there.
// ★REFUSALS LIVE IN THE DOOR'S FRAME, not the room's (M-5). A hand-back is stored as the standpoint's (along, across)
// offsets from the aperture centre, multiplied by the approach side (so a π-re-expression of the aperture — wall_u and
// across_u both flipped, which is what another room's association produces — maps it to the same physical point), and
// keyed by the room the robot was in: a room switch re-expresses the aperture in a new frame and the record still lands
// on the same physical standpoint; a standpoint in another room is not this room's evidence.
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
    float sigma = 0.1f;                 // the pose-and-door σ of P(reach), and the arrival scatter
    float needed_w = 0.7f;              // the body + margin the opening must show
    float memory_s = 60.0f;             // τ of a refusal's evidence
    float footprint_r = 0.35f;          // the robot's footprint radius
    float complete_at = kApproachCompleteProb;
    std::string room;                   // the room the robot is in (keys the refusals)
    std::vector<Eigen::Vector2f> polygon;   // that room; empty ⇒ no containment term
};

struct ApproachChoice
{
    Pose2 pose;
    float standoff = 0.0f, lateral = 0.0f;
    float p_reach = 0.0f, p_complete = 0.0f, p_inside = 1.0f, p_refused = 0.0f;
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

// P(the contract completes | the robot is sent to q): its estimate at arrival ~ N(q, σ² I), complete iff
// door_reach_prob there ≥ complete_at. 5-point Gauss–Hermite per axis.
[[nodiscard]] inline float complete_prob_at(const Aperture& ap, const ApproachProblem& pb, const Eigen::Vector2f& q)
{
    static constexpr std::array<double, 5> kX{-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856};
    static constexpr std::array<double, 5> kW{0.019953242059045913, 0.39361932315224116, 0.9453087204829419,
                                             0.39361932315224116, 0.019953242059045913};
    const float s = std::max(pb.sigma, 1e-3f) * std::numbers::sqrt2_v<float>;
    double acc = 0.0;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
        {
            const Eigen::Vector2f r = q + s * Eigen::Vector2f(static_cast<float>(kX[i]), static_cast<float>(kX[j]));
            if (reach_prob_at(ap, pb, r) >= pb.complete_at) acc += kW[i] * kW[j];
        }
    return static_cast<float>(acc / std::numbers::pi);   // ∫ e^{-x²} = √π per axis
}

// The feasible set is sampled on a grid with the old pose (midpoint, lateral 0) at its centre and evaluated centre-out,
// a strictly better score replacing the incumbent — so with no refusal the old pose is returned bit for bit. The lateral
// support is where the opening can show the body at all: |θ| ≤ acos(needed/w), i.e. |lateral| ≤ standoff·tan θ_max.
[[nodiscard]] inline ApproachChoice choose_approach_pose(const Aperture& ap, const ApproachProblem& pb,
                                                         const ApproachMemory& mem, std::int64_t now_ms)
{
    constexpr int kNs = 7, kNl = 9;   // grid resolution of the feasible set (odd: the centre is a sample)
    const float sgn = pb.side < 0.0f ? -1.0f : +1.0f;
    const Eigen::Vector2f u = ap.wall_u, n = ap.across_u(), c = ap.centre_xy();
    const float mid = 0.5f * (pb.d_min + pb.d_max), half_band = 0.5f * std::max(0.0f, pb.d_max - pb.d_min);
    const float cos_max = ap.w > 1e-3f ? std::clamp(pb.needed_w / ap.w, 0.0f, 1.0f) : 1.0f;
    const float tan_max = std::sqrt(std::max(0.0f, 1.0f - cos_max * cos_max)) / std::max(cos_max, 1e-3f);

    ApproachChoice best;
    bool have = false;
    for (int a = 0; a < kNs; ++a)
        for (int b = 0; b < kNl; ++b)
        {
            const int ia = (a % 2 == 0) ? a / 2 : -(a + 1) / 2;   // centre-out: 0, +1, −1, +2, −2, …
            const int ib = (b % 2 == 0) ? b / 2 : -(b + 1) / 2;
            const float standoff = mid + half_band * static_cast<float>(ia) / static_cast<float>(kNs / 2);
            const float lateral  = std::min(standoff * tan_max, 4.0f * standoff) * static_cast<float>(ib) / static_cast<float>(kNl / 2);
            Pose2 q;
            if (lateral == 0.0f)
                q = front_pose(ap, pb.side, standoff);   // the old construction, exactly
            else
            {
                const Eigen::Vector2f p = c + sgn * standoff * n + lateral * u;
                q = {p.x(), p.y(), std::atan2(c.y() - p.y(), c.x() - p.x())};
            }
            const Eigen::Vector2f qxy(q.x, q.y);
            ApproachChoice ch;
            ch.pose = q; ch.standoff = standoff; ch.lateral = lateral;
            ch.p_reach    = reach_prob_at(ap, pb, qxy);
            ch.p_complete = complete_prob_at(ap, pb, qxy);
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

}  // namespace rc::door
