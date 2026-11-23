/*
 * common/nbv/countable_planner.h — ONE epistemic (next-best-view) planner core for every countable concept agent.
 *
 * chair / table / bottle / hood / refrigerator each carried their own copy of the same twenty lines around
 * rc::nbv::plan_faces: build a Target, call plan_faces, REFUSE when no face is usable, bound the winner's gain by the
 * consumer's adequacy gap (only when a σ* is published), refuse a non-finite result. Those lines live here now.
 *
 * What stays the AGENT's, because it is a statement about that object and not about planning:
 *   - the Target (the object as its belief holds it: pose, extent, vertical span, σ);
 *   - raw_gain_of(face, standoff) — its own D-optimal ΔH given a detection (chair adds its discrete mode term);
 *   - useful(best_raw) — the fraction of the winner's raw gain that is still worth having (see useful_fraction).
 *
 * Header-only, Eigen only (plus viewpoint_score.h / dof_spec.h). Pinned by countable_planner_test.cpp.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include <Eigen/Core>
#include <Eigen/LU>

#include "viewpoint_score.h"
#include "../ai_belief/dof_spec.h"

namespace rc::nbv
{

// Field names IDENTICAL to the per-agent EpistemicProposal it replaces, so common/epistemic_step keeps compiling.
struct CountableProposal
{
    float epistemic_target_x_m     = 0.0f;
    float epistemic_target_y_m     = 0.0f;
    float epistemic_target_yaw_rad = 0.0f;
    float epistemic_gain           = 0.0f;
    bool  valid                    = false;

    bool is_finite() const
    {
        return std::isfinite(epistemic_target_x_m) and std::isfinite(epistemic_target_y_m) and
               std::isfinite(epistemic_target_yaw_rad) and std::isfinite(epistemic_gain);
    }
};

struct CountableInputs
{
    Sensor                            sensor;
    float                             robot_radius_m = 0.0f;
    std::span<const Obstacle>         obstacles;
    std::span<const Eigen::Vector2f>  room_polygon;
};

// `plan` is returned so each agent keeps its own diagnostic log line.
struct CountableResult
{
    CountableProposal proposal;
    Plan              plan;
    bool              refused_no_usable_face = false;
};

// The D-optimal information gain of one look: ½ ln det(I + Σ·ΔI). The existing per-agent formula, verbatim
// (the 1e-9 is the log's domain guard, not a tuning value).
template <int D>
float d_optimal_gain(const Eigen::Matrix<float, D, D>& S, const Eigen::Matrix<float, D, D>& dI)
{
    const Eigen::Matrix<float, D, D> I = Eigen::Matrix<float, D, D>::Identity();
    return 0.5f * std::log(std::max(1e-9f, (I + S * dI).determinant()));
}

// ★BOUND THE GAIN BY THE ADEQUACY GAP — but only if a consumer has actually stated one (chair's guard, verbatim).
// Information beyond σ* is worthless to the consumer, so an adequate object must STOP attracting. The guard matters
// as much as the bound: adequacy_gap_nats() returns 0 for a DOF table with no σ* anywhere — an empty sum — and 0 is
// the value meaning "adequate", so applying it unguarded would zero every gain. No demand ⇒ unbounded (fraction 1).
template <std::size_t N, class VarFn>
float useful_fraction(const std::array<DofSpec, N>& dofs, VarFn&& var, float best_raw)
{
    if (not any_sigma_star(dofs))
        return 1.0f;
    const float gap = adequacy_gap_nats(dofs, var);
    return best_raw > 1e-9f ? std::min(best_raw, gap) / best_raw : 0.0f;
}

// Plan the next view of one countable object. `raw_gain_of(int face, float standoff) -> float` is the agent's;
// `useful(float best_raw) -> float` scales the winner's DETECTION-WEIGHTED gain (useful_fraction, or 1).
template <class RawGainFn, class UsefulFn>
CountableResult plan_countable(const Target& t, const CountableInputs& in, RawGainFn&& raw_gain_of, UsefulFn&& useful)
{
    CountableResult r;
    r.plan = plan_faces(t, in.sensor, in.robot_radius_m, in.obstacles, raw_gain_of, in.room_polygon);
    if (not r.plan.valid)
        return r;
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. plan.best_pos is then the raw argmax, which for a
    // wall-anchored object is a pose behind the wall; the controller would REPAIR it to the floor at the object.
    if (not r.plan.any_usable)
    {
        r.refused_no_usable_face = true;
        return r;
    }
    const int   best = r.plan.best_face;
    // The adequacy bound scales only the FIT part: Σ* is a demand on the geometric DOFs, so an adequate object's
    // geometric value → 0 while the information a look adds about its EXISTENCE (Target::p_exists; 0 when unset)
    // stays — that term is what replaced the 5-nat verification floor (owner ruling 2026-11-23).
    const float gain = r.plan.face_geometric_gains[best] * useful(r.plan.face_raw_gains[best])
                     + r.plan.face_existence_gains[best];
    const CountableProposal p{r.plan.best_pos.x(), r.plan.best_pos.y(), r.plan.best_yaw, gain, true};
    if (p.is_finite())
        r.proposal = p;
    return r;
}

}   // namespace rc::nbv
