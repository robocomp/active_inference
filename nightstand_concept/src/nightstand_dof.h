#pragma once

/*
 * nightstand_dof.h  —  the nightstand belief's per-DOF metadata (names, units, target precision Σ*).
 * Single source of truth for the epistemic planner + the dashboard. See common/ai_belief/dof_spec.h.
 */

#include <array>

#include "../../common/ai_belief/dof_spec.h"
#include "nightstand_belief.h"

namespace rc
{

// [cx, cy, H, w, h, yaw] — the order of NightstandBeliefState::vec() and of Σ.
//
// SIGMA-STAR: none — NO CONSUMER HAS STATED A DEMAND (2026-11-14). Scaffolded from refrigerator_concept, whose
// 2 cm / 0.05 rad placeholders would have been a false green by rename (invariant 12). Nothing yet grasps, sits
// on, cleans or plans against a nightstand; -1 says "no demand" (any_sigma_star() falls back to logdet, the audit
// reports none(decl)). ★When a consumer appears, restate its demand HERE and drop the zero bid in
// epistemic_planner.cpp.
inline constexpr std::array<DofSpec, NightstandBelief::N> kNightstandDofs = {{
    {"cx",  "m",   -1.0f},
    {"cy",  "m",   -1.0f},
    {"H",   "m",   -1.0f},
    {"w",   "m",   -1.0f},
    {"h",   "m",   -1.0f},
    {"yaw", "rad", -1.0f},
}};

}  // namespace rc
