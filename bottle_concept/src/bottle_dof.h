#pragma once

/*
 * bottle_dof.h  —  the bottle belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the AI2 CSV header + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "bottle_belief.h"

namespace rc
{

// [cx, cy, cz, radius, height] — the order of BottleBeliefState::vec() and of Σ. No yaw (cylinder).
//
// NO σ* is published for the bottle (its planner carries no target precision), so every DOF declares -1
// and the dashboard drops the σ*/adequacy columns. See chair_dof.h for why nothing is invented here.
// SIGMA-STAR: none — the only real consumer is GRASPING, and it has not published a demand yet.
//
// REVIEWED 2026-08-09 and deliberately left at -1 rather than forgotten. σ* is the CONSUMER's tolerance, so
// it can only restate a demand someone else publishes — chair_concept's restates ring_metaconcept's
// sigma_slot_m / facing_model_std_deg. For a bottle the consumer is the arm, and the nearest thing to a
// number is kinova_controller's `arrive_deadband = 0.005 m` ("hold within this radius"; its own comment says
// it should be ≲ the grasp position tolerance). That is the ARM's positioning precision, not a statement
// about how well the bottle must be SEEN, and borrowing it would be wrong in both directions: 5 mm is below
// what vision resolves on a bottle at a metre, so the adequacy gap would never reach zero and the epistemic
// gain would never stop attracting — an affordance that can never say "done".
//
// The honest number has to come from a grasp-success study — the σ at which grasps start failing — not from
// a servo deadband. Grasping from perception is itself documented as unreliable (4/10 on c_insert), so that
// study is the prerequisite, not a formality.
//
// CONSEQUENCE: any_sigma_star() is false, so the adequacy gap is the -1 sentinel and the inspector and belief
// strip fall back to ½·ln det Σ, which has no meaningful zero and will not collapse when a fixation lands.
inline constexpr std::array<DofSpec, BottleBelief::N> kBottleDofs = {{
    {"cx", "m", -1.0f},
    {"cy", "m", -1.0f},
    {"cz", "m", -1.0f},
    {"r",  "m", -1.0f},
    {"h",  "m", -1.0f},
}};

}  // namespace rc
