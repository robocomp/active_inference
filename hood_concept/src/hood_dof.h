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
// SIGMA-STAR: none — NO CONSUMER HAS STATED A DEMAND. Nothing grasps a hood, nothing plans a path
// through one, and it hangs above the robot so it is not even an obstacle. The values here were the
// REFRIGERATOR's, inherited by a rename, and the audit read them as a green 6/6 — a FALSE PASS, and
// an invented σ* is the one thing invariant 12 forbids. -1 says "no demand", which any_sigma_star()
// reads correctly (the adequacy gap falls back to logdet and the strip says so) and which the audit
// reports as none(decl) rather than scoring an absence as a pass.
// ★When a consumer appears — a gripper, a planner, a cleaning routine — restate the demand HERE from
// what that consumer actually needs, and change [sigma_star] in the manifest to declared = true.
inline constexpr std::array<DofSpec, HoodBelief::N> kHoodDofs = {{
    {"cx",  "m",   -1.0f},
    {"cy",  "m",   -1.0f},
    {"H",   "m",   -1.0f},
    {"w",   "m",   -1.0f},
    {"h",   "m",   -1.0f},
    {"yaw", "rad", -1.0f},
}};

}  // namespace rc
