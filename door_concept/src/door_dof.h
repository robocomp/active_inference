#pragma once

/*
 * door_dof.h  —  the door belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "door_belief.h"

namespace rc
{

// [s, w, h] — wall-frame 3-DOF: along-wall offset of the near edge, panel width, panel height. Yaw,
// lateral off-wall position, and the floor datum are all fixed by the containing wall (not DOFs).
//
// NO σ* is published for the door: its planner scores viewpoints by raw D-optimal gain and has never
// carried a target precision. Every DOF therefore declares -1, which drops the σ*/adequacy columns from
// the dashboard entirely. That is the honest rendering — inventing targets here would silently create
// three tuning knobs (CLAUDE.md: no thresholds). Fill them in when a consumer publishes a real demand.
inline constexpr std::array<DofSpec, DoorBelief::N> kDoorDofs = {{
    {"s", "m", -1.0f},
    {"w", "m", -1.0f},
    {"h", "m", -1.0f},
}};

}  // namespace rc
