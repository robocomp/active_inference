#pragma once

/*
 * dof_spec.h  —  shared per-DOF METADATA for the concept agents' belief state vectors
 *
 * One tiny struct + a projection helper. Each agent declares ONE `<agent>_dof.h` table (kTableDofs,
 * kCabinetDofs, …) giving, per state index: the display name, the physical unit, and the CONSUMER's
 * precision demand σ*. That table is the single source of truth for three consumers that used to carry
 * their own drifting copies:
 *   • the epistemic planner  — σ* drives the adequacy gap and the published ViewpointConstraint;
 *   • the belief dashboard   — names/units/σ* label the BeliefInspector rows;
 *   • the AI2 CSV logger     — column headers.
 *
 * WHY THIS IS NOT IN `<agent>_belief.h`: the belief headers are the pure-Eigen, DSR-free inference cores,
 * and σ* is NOT a property of the belief — it is the downstream consumer's tolerance (see the ★PLACEHOLDER
 * note in every epistemic_planner.cpp: "ultimately PUBLISHED by the consuming grasp/place affordance",
 * cf. ViewpointConstraint::sigma_star in common/affordance_protocol/affordance_protocol.h). Keeping it in a
 * separate seam is what lets the hardcoded values be swapped for a runtime-published Σ* in ONE place.
 */

#include <array>
#include <cstddef>
#include <cmath>

namespace rc
{

// Metadata for ONE belief DOF, indexed by its position in the state vector / covariance.
struct DofSpec
{
    const char* name;        // display name, e.g. "cx", "yaw", "z0"
    const char* unit;        // "m" | "rad" | "" — also tells a reader which scale σ is on
    float       sigma_star;  // consumer's precision demand (same unit). < 0 ⇒ NONE published for this DOF:
                             // it is EXCLUDED from the adequacy gap and its dashboard cells render "–".
                             // Never substitute a number here — an invented target is a tuning knob in
                             // disguise (CLAUDE.md: no thresholds).
};

// Project a DOF table's σ* into the flat array the affordance protocol transports.
// DOFs with no published demand come through as -1, preserving "no demand" end to end.
template <std::size_t M, std::size_t N>
constexpr std::array<float, M> sigma_star_array(const std::array<DofSpec, N>& dofs)
{
    std::array<float, M> out{};
    for (std::size_t j = 0; j < M; ++j)
        out[j] = (j < N) ? dofs[j].sigma_star : -1.0f;
    return out;
}

// Remaining information (nats) to carry a belief down to σ*, summed PER-DOF over the marginal variances
// and clamped per DOF: Σ_j max(0, ½·ln(Σ_jj/σ*_j²)). Per-DOF clamp (not the full ½ln detΣ/detΣ*) on
// purpose — the consumer needs EACH DOF within tolerance, so an over-resolved DOF must NOT compensate for
// an under-resolved one. DOFs with sigma_star < 0 contribute nothing. ≤0 everywhere ⇒ adequate ⇒ no
// epistemic value left ⇒ the affordance goes quiet: a threshold-free "done" set by the CONSUMER's
// precision demand, not a tuned Σ bound.
//
// `var` is a callable j -> Σ_jj so this works for any Eigen matrix type without including Eigen here.
template <std::size_t N, typename VarFn>
float adequacy_gap_nats(const std::array<DofSpec, N>& dofs, VarFn&& var)
{
    float gap = 0.0f;
    for (std::size_t j = 0; j < N; ++j)
    {
        if (dofs[j].sigma_star < 0.0f)
            continue;
        const float s2 = dofs[j].sigma_star * dofs[j].sigma_star;
        const float g  = 0.5f * std::log(static_cast<float>(var(j)) / s2);
        if (g > 0.0f)
            gap += g;
    }
    return gap;
}

}  // namespace rc
