#pragma once

/*
 * cabinet_dof.h  —  per-DOF metadata for BOTH cabinet belief models (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "cabinet_belief.h"             // CabinetBelief::N  (free box run)
#include "cabinet_wall_run_belief.h"    // WallRunBelief::N  (wall-anchored run, kitchen model)

namespace rc
{

// ── Free box run: [cx, cy, yaw, L, d, z0, z1] ────────────────────────────────────────────────────
//
// ★PLACEHOLDER σ* — see table_dof.h. Looser on L than on the rest ON PURPOSE: the navigation/landmark
// consumers care about where the run's FACES are, not about its length to the centimetre, and a run is
// routinely seen in pieces — demanding a tight L would keep the affordance permanently hungry.
inline constexpr std::array<DofSpec, CabinetBelief::N> kCabinetDofs = {{
    {"cx",  "m",   0.03f},
    {"cy",  "m",   0.03f},
    {"yaw", "rad", 0.05f},
    {"L",   "m",   0.10f},
    {"d",   "m",   0.03f},
    {"z0",  "m",   0.03f},
    {"z1",  "m",   0.03f},
}};

// ── Wall-anchored run (kitchen model): [t0, t1, d, z0, z1] ───────────────────────────────────────
//
// The wall CHART carries the identity, so yaw and lateral placement are unrepresentable by construction;
// what remains is an interval [t0,t1] along the wall plus the carcass box. d/z0/z1 are the SAME physical
// quantities as in the box model above and reuse its demands. t0/t1 are the free interval endpoints — no
// consumer publishes a tolerance on where a run happens to start or stop, so they declare -1 rather than
// borrow L's (CLAUDE.md: no thresholds).
inline constexpr std::array<DofSpec, WallRunBelief::N> kWallRunDofs = {{
    {"t0", "m", -1.0f},
    {"t1", "m", -1.0f},
    {"d",  "m", kCabinetDofs[4].sigma_star},
    {"z0", "m", kCabinetDofs[5].sigma_star},
    {"z1", "m", kCabinetDofs[6].sigma_star},
}};

}  // namespace rc
