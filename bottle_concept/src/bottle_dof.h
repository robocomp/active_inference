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
// ★2026-08-11 — THE NUMBER IS DERIVABLE AFTER ALL; WHAT BLOCKS IT IS SATISFIABILITY, NOT PROVENANCE.
// The note above was right to reject arrive_deadband and wrong that no consumer number exists. The gripper's
// FINGER CLEARANCE is one, and it is a statement about perception rather than servoing: the fingers open to
// +/-0.0475 m (KinovaGen3 / the WaterBottleOpaque proto, authored around exactly this), so a bottle of radius
// r leaves c = 0.0475 - r of lateral slack per side. Past c the finger contacts the bottle instead of closing
// around it — the documented "catches the round surface and tips it". That IS "the perception error at which
// the grasp starts failing", stated in the gripper's own geometry.
//
//     r = 0.0256 (fitted, live)  ->  c = 0.0219 m/side  ->  sigma* = c/3 = 0.0073 m   (c/2 = 0.0110)
//
// MEASURED THE SAME DAY, and this is why it is still -1: over a 3274-row run sigma_cx fell 0.0612 -> 0.0212
// within the first quarter and then PLATEAUED — 0.0211..0.0222 across the final 2400 rows. That is 3x coarser
// than sigma*, flat, with no sign of closing.
//
// Shipping an unsatisfiable sigma* is WORSE than shipping none, and specifically so: it flips
// any_sigma_star() to true, which makes the adequacy gap the live quantity and BOUNDS THE EPISTEMIC GAIN BY
// IT (invariant 9). A gap that cannot reach zero is an object that attracts the robot forever — converting
// bottle from "honestly declares no target" into the exact "never stops asking" defect already open against
// chair and door. Honest silence beats a demand nobody can meet.
//
// ★★AND IN THIS SCENARIO THERE IS NO ARM AT ALL, which settles it more firmly than the measurement does.
// piso.wbt instantiates `DEF shadow Shadow` and Shadow.proto contains no Kinova; the arm lives only in
// shadow_arm.wbt / arm_base.wbt. So the clearance derivation above is CONDITIONAL ON A CONSUMER THAT IS NOT
// PRESENT: nothing in this scene grasps, so nothing needs the bottle's centre to any precision whatever.
// sigma* = none is not a gap awaiting a measurement here — it is the CORRECT answer, and it stays correct
// until an arm is in the world. When one is (the arm scenes), c/3 is the number, already derived.
//
// ★THE UNCOMFORTABLE COROLLARY, worth stating because it is not bottle-specific: if nothing consumes the
// bottle's pose, why does this agent bid for the robot's ATTENTION? Its epistemic planner publishes an
// affordance with a gain that competes against door/table/refrigerator for the robot's time, computed purely
// from its own Sigma — never from a demand. In AI2 terms expected information gain only has value relative
// to a PREFERENCE; uncertainty about something nobody needs is not surprise worth resolving. So in an
// arm-less scenario the bottle's gain should be ~0 and it should be a passive object: tracked when seen,
// never asking to be visited. That is the same "never stops asking" defect logged against chair and door,
// seen from its root rather than its symptom — the fleet allocates attention by who is most UNCERTAIN
// instead of who MATTERS.
//
// CONSEQUENCE: any_sigma_star() is false, so the adequacy gap is the -1 sentinel and the inspector and belief
// strip fall back to ½·ln det Σ, which has no meaningful zero and will not collapse when a fixation lands.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// SET 2026-08-11, deliberately, with the reasoning above intact rather than deleted.
//
// The scenario has no arm, so strictly there is no consumer and -1 was correct. It was also USELESS: with
// no sigma* anywhere the adequacy gap is the -1 sentinel, every display falls back to 1/2*ln det Sigma, and
// that number is a log-VOLUME — negative (it read -20.9), with no zero, so nothing on screen could ever say
// "precise enough". A belief you cannot judge is a belief nobody checks.
//
// So sigma* is set to the GRIPPER-CLEARANCE number derived above, the demand that WILL apply the moment an
// arm is in the world (shadow_arm.wbt / arm_base.wbt), taken at the LOOSER 2-sigma reading:
//
//     c = 0.0475 - r = 0.0219 m/side   ->   sigma*_xy = c/2 = 0.0129 m   (~95% of grasps clear the fingers)
//
// ★c/2 AND NOT c/3, chosen for SATISFIABILITY rather than conservatism. Live sigma_cx has plateaued at
// 0.0212 m: against c/3 = 0.0086 the gap is 1.80 nats and needs a 2.5x improvement, against c/2 it is 0.99
// nats and needs 1.6x. A demand that cannot be met bounds the epistemic gain at a positive floor forever
// (invariant 9) — the object then attracts the robot for all time, which is the "never stops asking" defect
// open against chair and door. The looser figure is the one that can actually reach zero when the robot
// closes to the 0.8 m stand-off (RangeNearM 0.6 gives full precision inside 0.6 m), so the affordance can
// say "done". If it never reaches zero on an approach, THAT is the evidence the plateau is a floor.
//
// cz carries the same demand: the finger box is 0.045 m tall, so its vertical capture window is the same
// order as the lateral clearance, and cz is currently the WORST DOF (sigma 0.0375, 1.23 nats) — the belief
// is far less sure of the bottle's height than its footprint, which is worth seeing.
//
// radius keeps a demand too and currently contributes 0.00 nats (sigma 0.0057, already inside): that is the
// gap doing its job — a DOF that is good enough stops asking, and only the ones that are not show up.
// height stays -1: nothing consumes it (the grasp approaches on cz), so inventing one would be noise.
inline constexpr std::array<DofSpec, BottleBelief::N> kBottleDofs = {{
    {"cx", "m", 0.0129f},
    {"cy", "m", 0.0129f},
    {"cz", "m", 0.0129f},
    {"r",  "m", 0.0129f},
    {"h",  "m", -1.0f},
}};

}  // namespace rc
