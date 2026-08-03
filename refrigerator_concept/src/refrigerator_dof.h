#pragma once

/*
 * refrigerator_dof.h  —  the refrigerator belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "refrigerator_belief.h"

namespace rc
{

// [cx, cy, H, w, h, yaw] — the order of RefrigeratorBeliefState::vec() and of Σ.
//
// ★PLACEHOLDER σ* — see the note in table_dof.h: this is the CONSUMER's precision demand, hardcoded
// until the consuming affordance publishes it.
inline constexpr std::array<DofSpec, RefrigeratorBelief::N> kRefrigeratorDofs = {{
    {"cx",  "m",   0.02f},
    {"cy",  "m",   0.02f},
    {"H",   "m",   0.02f},
    {"w",   "m",   0.02f},
    {"h",   "m",   0.02f},      // 2 cm pos/size
    {"yaw", "rad", 0.05f},      // ~2.9°
}};

}  // namespace rc
