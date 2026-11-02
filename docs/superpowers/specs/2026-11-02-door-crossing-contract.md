# Door crossing — the contract between door_concept and the controller

Date: 2026-11-02 · Status: DRAFT for the user's review · Supersedes the hand-off rules scattered across
controller_door_passage.h, door_cross_aif.h, door_approach_law.h, door_pragmatics.h (open/close/cross contracts).

## Why a contract

warm28–33: every crossing failure sat at a HAND-OFF between two controllers that each held their own idea of "done"
(arrived 80° off; a stale "arrived"; P(fit)=0 with nothing to raise it; overshoot into a shut leaf; spinning past the
jambs; a wrong leaf side believed at llr 40–119). The user (2026-11-02): "list all the requirements for the crossing,
as a contract that door and controller comply to proceed".

The principle: **door_concept owns the facts about the door and the actuator; the controller owns the motion.** They
talk ONLY through (a) door-node attributes (beliefs, with their uncertainty) and (b) affordances whose contracts state a
precondition, an effect and a perceptual completion. Neither side infers the other's internal state, and no side keeps
a latched "done" that the other did not publish. Phases (approach/open/cross/close) are names for WORLD STATES the
facts describe, not controller modes.

Status column: ✅ exists · ⚠ exists but violates the requirement · ❌ missing.

## 1. Vocabulary (both sides use exactly these)

| term | definition |
|---|---|
| hole | the physical opening; one per door; two **faces** (door nodes), one per room, related by the passage (ltsm) |
| door frame | the face's own frame: x along the wall (aperture axis), y = normal INTO the face's room; origin = aperture centre |
| s | signed distance of the robot centre along −y… +y through the plane (s<0 on this face's room side before crossing, per face convention below) |
| e | lateral offset of the robot centre from the aperture axis (door-frame x) |
| body | Shadow footprint: half-width 0.334 m, circumscribed radius r_b; the support radius at a heading is the exact extent |
| gap | the clear width of the aperture given the leaf angle φ (≤ aperture width ≈ 0.88–1.0 m) |
| swept arc | the region the leaf sweeps between its current φ and fully open/closed, on its swing side; radius = leaf length ≈ aperture width |
| entered room | the room on the far side; **through** = body wholly past the plane on the far side (s > r_b) |

## 2. What door_concept MUST publish (on each face, every cycle the face is updated)

| # | fact (attribute) | meaning / requirement | status |
|---|---|---|---|
| D1 | existence `exist`/P(exists) | belief the hole exists; absence evidence weighted by P(detect) with exact miss likelihood; **never deleted while a passage claims it** (crossing in progress) | ⚠ absence over-charged (warm30 DOOR_0 died at 5.7 m); no claim-hold |
| D2 | aperture pose (RT room→face) + width_m + pose covariance | in the room frame; covariance honest (grows when unseen) | ✅ (rt_covariance dropped on JSON load — note) |
| D3 | live anchor (door in robot frame, `door_N@robot`) | the opening measured NOW from the robot's sensors; the only geometry the motion near the plane may rely on | ✅ |
| D4 | `door_phi_rad` + σ, `door_open_prob` | leaf angle posterior; P(gap ≥ body passage width) marginalised over φ | ✅ |
| D5 | `door_leaf_into_room_prob`, `door_leaf_hinge_plus_prob` | leaf side belief, seeded from memory (ltsm), refined by every opening; never latched | ✅ (21821bc8, 85aa3f09, 258ebf99) |
| D6 | `door_cross_exec_prob` | P(the run from the robot's CURRENT pose to "through" is free of the leaf and the jambs) — the cross precondition | ✅ (definition must be re-checked against §4) |
| D7 | `door_crossing_progress` | −1…+1 along the normal, signed against the side held at the claim | ✅ |
| D8 | affordances `approach / open / cross / close` with contracts (§3) | offered with P = their precondition; withdrawn only when that P falls, with the reason | ✅ (contract content changes in §3) |
| D9 | actuator acknowledgements | open/close requests logged with the provider id; never assume success: completion is perceptual (D4) | ✅ |

door_concept MUST NOT: steer the robot; publish a pose/heading the robot "must" take unless the affordance genuinely
requires it (open/close are remote: they need no robot heading); latch the leaf side or existence across views.

## 3. The affordance contracts (precondition → effect → completion), door_concept authors, controller executes

| affordance | precondition (P, on the robot's state) | effect | completion (perceptual) | status |
|---|---|---|---|---|
| approach | door exists (D1) | robot moves to where `open`/`cross` become eligible | **the next affordance is eligible** (no own radius) | ✅ c061bd36 (unbuilt) |
| open | P(shut) × P(robot not in the swept arc of the opening leaf) × P(exists) — the actuator is remote: **no facing, no band** unless the sim/real actuator needs a line of sight (state which) | actuator request | `door_open_prob ≥ level`, held | ⚠ today requires P(in band)·P(facing) (orient) |
| cross | `door_open_prob` × D6 (`door_cross_exec_prob`) | robot passes the plane | **through** (s > r_b on the far side) — the exit fact | ✅ (completion = progress ≥ 0.9; make it "through") |
| close | P(open) × P(robot clear of the swept arc of the CLOSING leaf) — from D5, the arc is on the leaf's side only | actuator request; robot does NOT stop (chained, 35e5ad39) | `door_open_prob ≤ level`, held | ⚠ still offers a single standpoint pose (reach) |

The "level" in every precondition/completion is ONE shared number (kApproachCompleteProb / the selector's eligibility
level), not one per affordance.

## 4. What the controller MUST do (motion obligations)

| # | requirement | status |
|---|---|---|
| C1 | **One motion law through the door**: reach a goal pose beyond the plane, re-planned EVERY cycle from the current pose in SE(2) with the body's true footprint over the LIVE local grid (room map ∩ LiDAR); the open gap is free space, the leaf is an obstacle at its belief (D4,D5) inflated by its σ. No separate approach law / cross servo / drive-in modes. | ❌ (today: route + approach law + AIF servo + drive-in) |
| C2 | **No mouth, no crossing**: the body never enters the aperture unless `door_open_prob` (D4) and the LIVE gap (D3) admit it; a shut door makes the goal unreachable ⇒ the only admissible action is `open` | ⚠ (NO MOUTH hold exists; must fall out of C1) |
| C3 | Heading is a consequence, not a check: a feasible SE(2) path through a gap narrower than the body's diagonal faces the door | ❌ (today: facing checks + P(fit) gate) |
| C4 | No latched state across doors or routes: every decision is recomputed from the current pose and the published facts; anything cached is keyed to the door id + route epoch | ⚠ (c5c0ae90 / c061bd36 fixed two instances) |
| C5 | Disturbance = new start state: a push, slip or overshoot is absorbed by the next cycle's plan, without a recovery mode | ❌ (today: escape/MAKING ROOM modes) |
| C6 | Chaining: the controller knows the next step's goal; the post-crossing pose is chosen toward it; `close` is claimed while moving and served by door_concept when its precondition holds | ✅ 35e5ad39 (unproven live) |
| C7 | The crossing is reported by the exit fact (room change) only after **through**; the controller never declares a crossing room_concept/ltsm did not see | ✅ |
| C8 | Rates: the controller runs at its nominal 20 Hz with all of the above (re-planning cost measured) | ✅ 19 Hz now; C1 must be measured |

## 5. Shared invariants (either side breaking one is a bug, not an edge case)

1. A face claimed by an active passage is not deleted; its existence may fall but the deletion waits for the claim to end (D1).
2. The leaf is never assumed: its side comes from D5 with its probability; an unknown side (0.5) is planned as BOTH sides (the arc on both).
3. The body never overlaps the swept arc of a leaf that may move (open/close requests are only sent when the robot is clear, by D5).
4. Completion of open/close/cross is perceptual (D4/through), never a protocol acknowledgement.
5. Every withdrawal of an offered affordance carries its reason (which factor of its precondition fell).
6. Frames: room-frame facts for planning far from the door; the live anchor (D3) for the motion inside r_b + arc of the plane.

## 6. Failure semantics (what each side does, no special modes)

| event | door_concept | controller |
|---|---|---|
| door closes mid-cross | publishes `door_open_prob`↓, gap↓ | the plan through the gap becomes infeasible ⇒ the body backs out along its plan's reverse (C1/C5), then `open` |
| leaf side belief flips mid-approach | D5 changes | the next cycle's plan avoids the new arc |
| door deleted (absence) during a claim | MUST NOT (invariant 1) | — |
| robot pushed / overshoots | facts unchanged | C5 |
| actuator refuses | marks the affordance refused (never re-asks) | mission step fails with the reason |
| next goal unknown | — | old end pose (C6 fallback) |

## 7. Acceptance (offline before live)

- Bench: the existing door harness (test_door_cross_aif: unicycle, 1.0/0.9 m doors, anchor noise 2–5 cm) extended to
  C1, plus the live start states: 80° off at 2 m (warm33), stale arrival 7 m out (warm32), leaf beside (warm24), shut
  door, door closing at s=0, pushed 0.5 m at the vantage. Old chain vs C1: success, contact, time, min clearance.
- TLA+: the existing passage model updated to the four contracts of §3 + invariants §5 (one leg and the ping-pong).
- Live: the 4-leg tour from three_rooms_door0 (sides now ground truth) — pass = 4/4, no manual help, rates nominal.

## Open questions for the user

1. `open`'s precondition: does the actuator (sim and real) need the robot facing / within a band, or is it fully remote?
   If remote, open can be requested from anywhere clear of the arc — another stop removed.
2. C1 replaces the AIF cross servo. Keep the servo's evidence (the live anchor's precision) as the planner's cost near
   the plane, or drop it entirely?
3. The "level" (one shared eligibility number, today kApproachCompleteProb): keep it as THE single decision constant?
