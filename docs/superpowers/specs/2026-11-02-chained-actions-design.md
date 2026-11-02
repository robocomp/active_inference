# Chained actions: each step ends where the next one wants to begin

Date: 2026-11-02 · Status: DRAFT for the user's review · Scope: controller (mission + target selection), door_concept (close target)

## Why

Live runs warm29–31: the robot repeatedly hits room corners on the way to the next door. Localisation is good (1–2 cm),
the controller runs at ~19 Hz, and the route has ~0.4 m clearance, so the robot is not lost and the map is not wrong.
The pattern is in WHERE each step leaves the robot:

- `close` (door behind) is offered by door_concept as ONE pose: `front_pose(aperture, approach_side, standoff)` —
  straight out along the door normal at the close standoff (~2.1 m), **yaw facing the door just crossed**
  (warm31: `aff_door_24_close target=(0.10,2.15) yaw=-1.58`). The controller drives there and stops facing back.
- The next leg (`open <next door>`) starts from that pose: the robot faces away from its next goal, often with a wall
  or fitting close by, and its first act is a large turn in a tight spot. That is where the corner loops begin.

The user (2026-11-02): "The controller has a target pose. One explanation is that the affordance sets final orientation
towards the wall. We should make the planner anticipate the next move (affordance) so it can concatenate actions."

Each step is chosen as if it were the last. A step's end pose is a free variable that the CURRENT step barely constrains
and the NEXT step depends on heavily.

## Goal

The end state of step k is chosen for step k+1 — so the robot flows from one action into the next without a re-orient
in a tight spot. In Active-Inference terms: score the policy over the two-step sequence (current + next), not one step
at a time; the current step's goal is a REGION of acceptable states, and the next step's expected cost picks the point
in it.

Non-goals: a general multi-step planner; changing which affordances exist; touching the cross servo or the new door
approach law (093fdd9c), which this complements (that law recovers from a bad start; this avoids making one).

## Design

### 1. A step's goal is a region, not a pose (door_concept)

`close` offers the set it actually requires: **robot clear of the leaf's swept arc, on the approach side, within
actuation range** — published as the existing target (a representative point) PLUS the region's description:
`standoff_min` (= `close_standoff(...)`, the clear-of-swing distance with its σ) and a free heading (`yaw` marked
unconstrained). The close needs no heading: the actuator is remote. Other affordances stay as they are unless they also
over-constrain (check `approach`'s yaw: it legitimately needs facing — keep).

Attribute changes go through the type-attributed DSR API (REGISTER_TYPE in cortex + user reinstall if a new attribute is
needed; prefer reusing existing fields if they can carry "heading free").

### 2. The mission passes the next step's goal down (controller)

The mission interpreter already has the whole step list. When it starts step k it also resolves step k+1's target
(e.g. `open room_1` → door_21's vantage V₁ and its normal) and hands both to the target selector. No lookahead past k+1.

### 3. The end pose of step k is chosen for step k+1 (controller)

Within step k's region R_k, pick the end pose x* minimising the expected cost of step k+1 from there:

    x* = argmin_{x ∈ R_k}  C_route(x → V_{k+1})  +  C_turn(heading at x → heading of the route's first segment)

with C_route the planner's cost-to-go (already computed on the grid; it includes clearance) and C_turn the turn the
robot must make at x. Heading at x = the direction of the route to V_{k+1}, so the robot ENDS step k already facing
along its next path. Practically: evaluate the planner's cost-to-go field from V_{k+1} over the cells of R_k (one
distance-field query per candidate; R_k is small), take the minimum, set the heading from the route's first segment.

If step k+1 is unknown (last step, or not resolvable yet) the old behaviour holds: the affordance's representative pose.

### 4. Doors: the close happens on the way

For the common case (close behind, then go to the next door) the cheapest point of R_close is where the robot's route
to the next vantage LEAVES the swept arc — i.e. the close is issued as the robot passes out of the arc, without a stop
or a turn. That is the outcome of §3 with C_route, not a special case; check the tests show it.

## Tests (offline, before any live run)

- A mission `cross R2; close R1; open R3` on a synthetic two-door room: the close end pose lies on the route to R3's
  vantage, heading within a few degrees of that route's first segment; total heading change between the two steps is
  small; no pose within body+margin of a wall.
- Same with the next step unresolvable: old behaviour (representative pose).
- Regression: `test_door_passage` (state machine), `test_door_approach_law`, the close-judgement cases (1aeac589).

## Open questions for the user

1. Is `close` allowed to fire while the robot is still moving (out of the arc, on the way)? It is remote and the arc
   test is already probabilistic — this removes the stop entirely.
2. Should the same apply to `cross`'s end (the post-crossing point)? Today the cross target is a point 1.3 m past the
   door; with §3 it would be chosen toward the next goal — but only if the next step is known at that time.
