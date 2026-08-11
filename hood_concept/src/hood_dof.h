#pragma once

/*
 * hood_dof.h  —  the hood belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "hood_belief.h"

namespace rc
{

// [cx, cy, H, w, h, yaw] — the order of HoodBeliefState::vec() and of Σ.
//
// ★PLACEHOLDER σ* — see the note in table_dof.h: this is the CONSUMER's precision demand, hardcoded
// until the consuming affordance publishes it.
// ★★CLONED σ* — these are the REFRIGERATOR's demands, and the audit reads them as a pass (6/6).
// A hood has no consumer at all yet: nothing grasps it, nothing plans through it, and the values
// below were inherited by a rename rather than restated from anyone. Per invariant 12 that makes
// them INVENTED, which is the one thing σ* may never be. Either restate a real demand or set them
// to -1 with a `SIGMA-STAR: none — <reason>` line, which the audit accepts as a reviewed absence.
// Left as-is so the scaffold does not silently claim a precision target nobody asked for.
inline constexpr std::array<DofSpec, HoodBelief::N> kHoodDofs = {{
    {"cx",  "m",   0.02f},
    {"cy",  "m",   0.02f},
    {"H",   "m",   0.02f},
    {"w",   "m",   0.02f},
    {"h",   "m",   0.02f},      // 2 cm pos/size
    {"yaw", "rad", 0.05f},      // ~2.9°
}};

}  // namespace rc
