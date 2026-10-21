// door_approach_pose.h — WHERE to stand to use a door, once the consumer has handed a standpoint back (D-A).
// Header-only, Eigen only (no DSR): tests/test_door_approach_pose.cpp.
//
// ★THE LOOP THIS ENDS. The controller does not re-site an object's standpoint any more: when it cannot reach the one it
// was given (an obstacle on the footprint, no route) it hands the `approach` affordance back — Outcome Unreachable or
// Infeasible. door_concept's pose was a deterministic function of the door (the actuation band's midpoint, on the
// aperture's normal), so the re-armed affordance offered the IDENTICAL pose and the controller handed it back again:
// a hand-back / re-offer loop with no information in it.
// ★THE MODEL. A hand-back is EVIDENCE that the standpoint it names is blocked — not a verdict, and not forever: what blocks
// a standpoint is usually something that moves (a chair, a person, the robot's own map catching up). Each refusal i
// contributes P_i(q) = exp(−age_i/τ) · exp(−|q − p_i|² / 2r²): it decays with its age on the memory constant τ, and
// spreads over the robot's FOOTPRINT r (the thing that did not fit). P(refused | q) = 1 − Π(1 − P_i(q)).
// The pose is then re-solved over door_concept's OWN feasible set — standoff ∈ [d_min, d_max] (the actuation band) ×
// a lateral offset along the aperture within its half-width — maximising
//      P(reach | q) · (1 − P(refused | q)),   P(reach | q) = P(in band | range(q), σ) · P(facing the opening | q, σ)
// the same two probabilities the agent already uses for `approach`'s completion and `open`'s precondition. With no
// refusal the argmax is the old pose exactly (the band's midpoint on the normal), so a run without hand-backs is
// unchanged; a changed pose is a CONTENT change, so PragmaticAffordance bumps the epoch by itself.
// ⚑ τ (DoorAffordance.ApproachRefusalMemoryS) is a model parameter — how long a blocked standpoint stays blocked — not
// a gate: the evidence fades continuously and nothing switches.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>

#include "door_geometry.h"
#include "door_passability.h"   // prob_within, facing_cos, prob_facing_opening
#include "door_standpoint.h"    // Pose2, front_pose

namespace rc::door
{

struct ApproachMemory
{
    struct Refusal { Eigen::Vector2f xy; std::int64_t t_ms = 0; };
    std::vector<Refusal> refused;

    void refuse(const Eigen::Vector2f& xy, std::int64_t t_ms) { refused.push_back({xy, t_ms}); }

    // P(a standpoint at q is refused again | the remembered hand-backs), at time now.
    [[nodiscard]] float p_refused(const Eigen::Vector2f& q, std::int64_t now_ms, float memory_s, float footprint_r) const
    {
        const float tau = std::max(memory_s, 1e-3f), r = std::max(footprint_r, 1e-3f);
        double p_free = 1.0;
        for (const auto& f : refused)
        {
            const float age = std::max(0.0f, static_cast<float>(now_ms - f.t_ms) * 1e-3f);
            const float k = std::exp(-age / tau) * std::exp(-0.5f * (q - f.xy).squaredNorm() / (r * r));
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
    float side = +1.0f;          // which side of the aperture (approach_side)
    float d_min = 0.0f, d_max = 0.0f;   // the actuation band, from the aperture plane
    float sigma = 0.1f;          // the pose-and-door σ of P(reach)
    float needed_w = 0.7f;       // the body + margin the opening must show
    float memory_s = 60.0f;      // τ of a refusal's evidence
    float footprint_r = 0.35f;   // the robot's footprint radius
};

struct ApproachChoice
{
    Pose2 pose;
    float standoff = 0.0f, lateral = 0.0f;
    float p_reach = 0.0f, p_refused = 0.0f, score = 0.0f;
};

// The feasible set is sampled on a grid with the old pose (midpoint, lateral 0) at its centre and evaluated centre-out,
// a strictly better score replacing the incumbent — so with no refusal the old pose is returned bit for bit.
[[nodiscard]] inline ApproachChoice choose_approach_pose(const Aperture& ap, const ApproachProblem& pb,
                                                         const ApproachMemory& mem, std::int64_t now_ms)
{
    constexpr int kNs = 7, kNl = 9;   // grid resolution of the feasible set (odd: the centre is a sample)
    const float sgn = pb.side < 0.0f ? -1.0f : +1.0f;
    const Eigen::Vector2f u = ap.wall_u, n = ap.across_u(), c = ap.centre_xy();
    const float mid = 0.5f * (pb.d_min + pb.d_max), half_band = 0.5f * std::max(0.0f, pb.d_max - pb.d_min);
    const float half_lat = 0.5f * std::max(0.0f, ap.w);
    const auto band = [&](float d)
    { return pb.d_max > pb.d_min ? std::max(0.0f, prob_within(d, pb.d_max, pb.sigma) - prob_within(d, pb.d_min, pb.sigma)) : 0.0f; };

    ApproachChoice best;
    bool have = false;
    for (int a = 0; a < kNs; ++a)
        for (int b = 0; b < kNl; ++b)
        {
            // centre-out ordering: 0, +1, −1, +2, −2, …
            const int ia = (a % 2 == 0) ? a / 2 : -(a + 1) / 2;
            const int ib = (b % 2 == 0) ? b / 2 : -(b + 1) / 2;
            const float standoff = mid + half_band * static_cast<float>(ia) / static_cast<float>(kNs / 2);
            const float lateral  = half_lat * static_cast<float>(ib) / static_cast<float>(kNl / 2);
            Pose2 q;
            if (lateral == 0.0f)
                q = front_pose(ap, pb.side, standoff);   // the old construction, exactly
            else
            {
                const Eigen::Vector2f p = c + sgn * standoff * n + lateral * u;
                q = {p.x(), p.y(), std::atan2(c.y() - p.y(), c.x() - p.x())};
            }
            const Eigen::Vector2f d(q.x - c.x(), q.y - c.y());
            const float p_reach = band(d.norm())
                                * prob_facing_opening(facing_cos(d.x(), d.y(), n.x(), n.y()), ap.w, pb.needed_w, pb.sigma);
            const float p_ref = mem.p_refused({q.x, q.y}, now_ms, pb.memory_s, pb.footprint_r);
            const float score = p_reach * (1.0f - p_ref);
            if (not have or score > best.score)
            {
                best = {q, standoff, lateral, p_reach, p_ref, score};
                have = true;
            }
        }
    return best;
}

}  // namespace rc::door
