#pragma once

/*
 * chair_dof.h  —  the chair belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "chair_belief.h"

namespace rc
{

// [cx, cy, yaw] — pose-only 3-DOF (cz is pinned to the floor, size is a fixed template).
//
// NO σ* is published for the chair: its planner scores viewpoints by raw D-optimal gain and has never
// carried a target precision. Every DOF therefore declares -1, which drops the σ*/adequacy columns from
// the dashboard entirely. That is the honest rendering — inventing targets here would silently create
// three tuning knobs (CLAUDE.md: no thresholds). Fill them in when a consumer publishes a real demand.
inline constexpr std::array<DofSpec, ChairBelief::N> kChairDofs = {{
    {"cx",  "m",   -1.0f},
    {"cy",  "m",   -1.0f},
    {"yaw", "rad", -1.0f},
}};

}  // namespace rc
