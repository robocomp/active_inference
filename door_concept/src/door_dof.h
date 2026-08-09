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
// SIGMA-STAR: none — a door is, for now, only an object to LOOK AT; nothing plans a path through one.
//
// REVIEWED 2026-08-09 and deliberately left at -1, rather than forgotten. σ* is the CONSUMER's tolerance,
// so it can only be a restatement of a demand someone else publishes — the way chair_concept's is a
// restatement of ring_metaconcept's sigma_slot_m / facing_model_std_deg. Three places were checked and
// none of them states one for a door:
//   · the controller has passability logic ("never makes a passable gap unplannable") but no tolerance on
//     how well an aperture must be KNOWN — nothing routes through a doorway yet;
//   · DOOR_CONCEPT_SPEC.md gives σ 0.60 / 0.06 / 0.08 for s/w/h, but those are PRIORS. Using a prior as σ*
//     is circular: it defines "adequate" as "no better than where we started", so the gap would read ~0
//     from birth and the display would say the belief is finished before it has seen anything;
//   · the only in-repo consumer of `w` is this agent's OWN NBV (the aperture becomes an rc::nbv::WallGap of
//     half_w = 0.5·w), which is the agent consuming its own belief — not an external demand.
//
// CONSEQUENCE, so it is not a surprise later: any_sigma_star() is false, so the adequacy gap is the -1
// sentinel and the inspector + belief strip fall back to ½·ln det Σ. That has no meaningful zero and will
// not collapse when a fixation lands — the honest rendering of "no one has said what good enough means".
//
// WHEN TO FILL IT IN: the moment anything plans a path THROUGH a door. Then the demand is passability —
// the robot is 0.60 m wide (radius 0.30), a leaf is 0.70–0.90 m, so the decision margin is 0.10–0.30 m and
// σ*_w must be small enough not to flip "can I fit" (~0.03 m). `s` follows (where the gap is); `h` still
// carries no demand for a ground robot.
inline constexpr std::array<DofSpec, DoorBelief::N> kDoorDofs = {{
    {"s", "m", -1.0f},
    {"w", "m", -1.0f},
    {"h", "m", -1.0f},
}};

}  // namespace rc
