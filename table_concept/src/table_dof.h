#pragma once

/*
 * table_dof.h  —  the table belief's per-DOF metadata (names, units, target precision Σ*).
 *
 * Single source of truth shared by the epistemic planner (adequacy gap + published ViewpointConstraint)
 * and the dashboard's BeliefInspector. See common/ai_belief/dof_spec.h for the rationale.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "table_belief.h"      // TableBelief::N and the state layout this table indexes

namespace rc
{

// [cx, cy, H, w, h, yaw, t] — the order of TableBeliefState::vec() and of Σ.
//
// ★PLACEHOLDER σ* — the posterior std at which the table is "adequately resolved" for its downstream
// consumer. It should be the physical manipulation tolerance (gripper clearance / placement margin /
// approach-cone half-angle) pushed through ∂success/∂θ, and ultimately PUBLISHED by the consuming
// grasp/place affordance. Hardcoded here until that channel exists — REPLACE, don't retune.
inline constexpr std::array<DofSpec, TableBelief::N> kTableDofs = {{
    {"cx",  "m",   0.02f},
    {"cy",  "m",   0.02f},
    {"H",   "m",   0.02f},
    {"w",   "m",   0.02f},
    {"h",   "m",   0.02f},      // 2 cm pos/size
    {"yaw", "rad", 0.05f},      // ~2.9°
    // Depth-tilt calibration nuisance: co-estimated so it stops biasing the geometry, but NO consumer
    // has a tolerance on it, so it carries no demand and is excluded from the adequacy gap.
    {"t",   "rad", -1.0f},
}};

}  // namespace rc
