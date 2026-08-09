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
// σ* FILLED IN 2026-08-09, on the condition this note itself set: "when a consumer publishes a real demand".
// It does — ring_metaconcept states its own model spread in ring_config.h, so these are a RESTATEMENT of the
// consumer's published numbers, not three new knobs.
//
// WHY IT MATTERED. All three were -1, so any_sigma_star() was false, the adequacy gap came out as the -1
// sentinel, and both the inspector and the belief strip fell back to ½·ln det Σ. That fallback has no
// meaningful zero and does not collapse when a fixation succeeds — which is exactly why the chair alone
// never showed its nats fall on a completed affordance while every other object did.
//
// THE DERIVATION. ring_metaconcept is the real geometric consumer of a chair:
//     sigma_slot_m         = 0.20 m   slack between a seat and the chair on it ("chairs get pushed in and
//                                     out, so this is deliberately loose")
//     facing_model_std_deg = 12°      intrinsic spread of "a chair faces the table"
//                                     (★live estimate from three real chairs: residuals −7°, +16°)
// The demand is that OUR uncertainty must not DOMINATE the consumer's own spread. At σ* = one third of that
// spread our contribution adds ~5% in quadrature (√(1+1/9) = 1.054) — present, not limiting. Hence
//     position  0.20 / 3 ≈ 0.07 m        yaw  12°/3 = 4° ≈ 0.07 rad
// Deliberately looser than the table's 0.02 m / 0.05 rad: a table is a fixed anchor, a chair is furniture
// the rig itself models as mobile. Re-derive if the ring's slack or facing spread is ever re-measured.
//
// SCOPE: display-only today. kChairDofs is read by the dashboard and by one diagnostic printf in
// epistemic_planner.cpp; the chair's planner does NOT bound its gain by the adequacy gap the way the
// refrigerator's does (useful_frac), so this changes what is REPORTED, not what the robot does.
inline constexpr std::array<DofSpec, ChairBelief::N> kChairDofs = {{
    {"cx",  "m",   0.07f},
    {"cy",  "m",   0.07f},
    {"yaw", "rad", 0.07f},   // ≈4°
}};

}  // namespace rc
