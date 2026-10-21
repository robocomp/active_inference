/*
 * door_instance.h
 *
 * Per-door runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h): rc::DoorFitState (the
 * DSR-free part — geometry, belief, leaf angle, existence, bookkeeping; door_fit_state.h) plus the members that are
 * bound to the graph: the epistemic affordance request and the three/four pragmatic affordances, and the
 * InteractionState their header carries. The DoorCore's fit reads and writes the base only.
 */

#pragma once

#include "door_fit_state.h"
#include "../../common/exclusion/exclusion.h"        // (kept: users of this header relied on it for rc::exclusion)
#include "../../common/object_affordance/object_affordance.h"   // rc::ObjectAffordance (SHARED)
#include "../../common/pragmatic_affordance/pragmatic_affordance.h" // rc::pragmatic (SHARED) — approach/open/cross
#include "door_approach_pose.h"  // rc::door::ApproachMemory — hand-backs as decaying evidence (D-A)
#include "door_pragmatics.h"   // rc::door::InteractionState — the three preconditions

namespace rc {

struct DoorInstance : DoorFitState
{
    // Epistemic action request published to DSR (filled by the epistemic planner).
    ObjectAffordance affordance;

    // ── THE THREE PRAGMATIC AFFORDANCES: approach / open / cross ────────────────────────────────────
    // Siblings of `affordance` above under the same door node, and a different KIND of offer: that one
    // asks the controller for a LOOK that shrinks Σ, these three ask it to do something to the door.
    // Each is preconditioned — on the wire only while this agent believes the action is possible at all
    // (rc::pragmatic, see its header for the three rules that keeps honest). `approach` is effectively
    // unconditional for a located door; `open` needs the robot inside the actuation zone with the
    // aperture believed shut; `cross` needs the aperture believed passable for this robot's body.
    rc::pragmatic::PragmaticAffordance aff_approach;
    rc::pragmatic::PragmaticAffordance aff_open;
    rc::pragmatic::PragmaticAffordance aff_cross;
    // `close` (2026-10-18): the door believed open AND the robot clear of the leaf's swing. Asked for by the
    // controller's passage once the robot is through, so it closes the door behind it.
    rc::pragmatic::PragmaticAffordance aff_close;
    bool pragmatic_inited = false;   // init() takes the node id/name, so it waits for the DSR node

    // This cycle's preconditions, recomputed once and then (a) published on the door node as the
    // affordances' completion predicates and (b) used to decide what to offer. Held on the instance so
    // the dashboard and the CSV logs read the SAME numbers the decisions were taken on.
    rc::door::InteractionState interaction{};

    // D-A: the approach standpoints the controller handed back (Unreachable / Infeasible), as DECAYING evidence that
    // they are blocked (door_approach_pose.h), and the pose currently offered (the one a hand-back refers to).
    rc::door::ApproachMemory approach_memory;
    Eigen::Vector2f approach_offered_xy{0.0f, 0.0f};
    bool approach_offered_valid = false;
};

}  // namespace rc
