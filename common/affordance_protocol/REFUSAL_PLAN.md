# Affordance refusal — letting a consumer say "I could not get there"

**Status:** PLAN, not implemented. Blocked on a cortex attribute registration (see §5).
**Written:** 2026-08-08, from a live two-producer run (room + refrigerator + 2 doors).

## 1. The problem, as measured

`refrigerator_concept` proposed a standpoint the robot cannot occupy — on the shelf / behind the wall,
the known `room_polygon`-missing case. What followed, in one 3-minute window:

- **12 escapes**, all against `aff_refrigerator_1` at `(-3.81,-2.86)`, the robot achieving
  **2–19 % of every commanded metre** (`commanded 0.69 m over 1.5 s, achieved 0.060 m`).
- The controller's repair slid the goal along the wall — `(-3.81,-2.86)`, `(-3.75,-2.92)`,
  `(-3.75,-2.26)`, `(-3.75,-2.20)`, `(-3.75,-2.08)` — every one still unreachable.
- Meanwhile **three other affordances sat `Offered`** and unserved.
- An earlier window: `goal (-3.77,1.49) is not footprint-feasible at any heading`, held for minutes.

The controller now defends itself (see §6), but **the producer is never told**. It still believes that
viewpoint is its best idea, keeps it top of the NBV ranking, and re-publishes it forever. We stopped the
robot hurting itself; we did not stop the bad proposal.

## 2. Why the existing protocol cannot say it

`decode_protocol_state` spans exactly four states:

| active | pending | state |
|---|---|---|
| false | true  | Offered |
| true  | true  | Executing |
| false | false | Completed |
| true  | false | Invalid |

Neither of the two that a consumer could reach is honest here:

- **Completed** claims an observation that never happened. The producer then resets its neglect clock and
  tightens its belief on evidence it never received — and `affordance-completed-is-not-neutral` already
  records what treating Completed as a neutral state costs.
- **Executing**, held indefinitely, trips the producer's own stall watchdog
  (`RoomSceneGraph::break_execution_stall`) and is a lie of a different kind.

There is no way to say *"your gain may well be real; the approach is not."* That is the missing sentence.

## 3. Design: refuse the OFFER, annotated — do not leave it Offered

The affordance is not the problem — `refrigerator_1` genuinely is worth looking at. **The pose is the
problem.** But the refusal must also *end* the current offer.

An earlier draft of this plan left the affordance `Offered` and expected the producer to notice the flag
and replace the pose. That is a send/wait protocol with no termination guarantee: between the refusal and
the producer's next publish, the affordance is still Offered carrying a pose already known to be
unreachable, so the consumer can select it again, refuse it again, and nothing in the design says the
loop ever stops. The consumer would need its own timer or memory to stay out — which is exactly the
workaround in §6, not a protocol.

So: **the refusal retires the offer**, annotated so nobody mistakes it for success.

Consumer writes, at the moment it gives up:

| attribute | type | meaning |
|---|---|---|
| `epistemic_refused` | `bool` | this offer was refused — the consumer could not reach the standpoint |
| `epistemic_refused_x_m` | `float` | the standpoint it failed at (as PUBLISHED, pre-repair) |
| `epistemic_refused_y_m` | `float` | ditto |

and drives the affordance to the not-offered state, exactly as a completion does. The annotation is what
separates the two: **Completed means observed** — update the belief, reset the neglect clock; **Refused
means not attempted** — change nothing about what is believed, only about where to stand.

Then the producer publishes a new target *later*, when it has a different one, through the path it
already uses. Nothing waits on anything.

**The termination is structural, and already implemented.** `publish_target` declines to re-arm a
non-offered affordance whose proposed target is unchanged. A producer that has not changed its mind
therefore does not re-offer, and one that proposes a genuinely different viewpoint re-arms it as a normal
publish. The same rule that once deadlocked the rotate-in-place recovery (`room_scene_graph.cpp:648`) is
the right rule here: it says "an unchanged proposal is not news", which is precisely the guarantee a
refusal needs.

**Why the pose is still carried:** so the producer can exclude that viewpoint from its next NBV ranking
rather than re-deriving the same best answer and being refused again — and so a human reading the graph
can see *what* was refused. Use a small exclusion radius; the controller uses 0.30 m, roughly a body
width, on the grounds that a standpoint half a body from one that wedged is the same approach through the
same gap.

## 4. Who changes

| component | change |
|---|---|
| cortex `dsr_attr_name.h` | `REGISTER_TYPE(epistemic_refused, bool, false)` + the two floats |
| `common/affordance_protocol` | document the refusal in the protocol header; no state-machine change |
| `common/affordance_manager` | `publish_refusal(graph, node_id, pose)`; clear on publish_target |
| controller | write the refusal where it currently only suppresses locally (§6) |
| room / door ×2 / refrigerator | read it, exclude the pose, re-plan |

★ **Every producer must honour it.** One that ignores the flag keeps insisting exactly as today, so a
partial rollout buys nothing for the agent that skipped it.

## 5. The cortex dependency — read before starting

Per `CLAUDE.md`, new DSR attributes need `REGISTER_TYPE` in
`cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`, which is **root-owned**: the user
reinstalls cortex, we do not `sudo cp` it. Nothing compiles against `epistemic_refused_att` until that
lands, so the order is:

1. register the three attributes in cortex → **user reinstalls** → rebuild the agents;
2. consumer side (controller) — write the refusal;
3. producer side, one agent at a time, verifying each.

Use the type-attributed API (`add_or_modify_attrib_local<epistemic_refused_att>`), never
`runtime_checked_*`.

## 6. What already exists (controller, committed)

The controller-side defence is in and does not need the protocol:

- **3 escapes against one standpoint ⇒ abandon it.** `begin_escape` charges each escape to the affordance
  it happened under; `escape_count_` alone was session-lifetime and could not say *which* target kept
  failing. On the third, the claim is released and the goal dropped.
- **`AffordanceManager::suppress_target(id, rounds)`** — takes it out of contention for ~400 selection
  rounds. Deliberately does **not** yield the way the no-two-in-a-row rule does: taking that one again is
  merely impatient, taking this one again is known to end in the same wedge.
- **Useless-spot memory**, keyed by POSITION, matched within 0.30 m, session-lifetime. Recognised before
  driving, so a re-offered bad standpoint costs nothing.
  ★Not revalidated against the map, on purpose: these spots failed *physically* — the grid reported
  0.31–0.38 m of clearance while the base achieved 2–19 % of commanded travel — so `pose_free()` would
  call them fine and forget them instantly. The map is not the arbiter of a failure the map cannot see.

This is a workaround with a real cost: the memory is per-session and per-consumer. Two robots, or one
after a restart, each re-learn every bad spot the expensive way. The protocol change is what makes the
knowledge belong to the producer, where it can actually change the proposal.

## 7. Verification

- A refused standpoint is never re-proposed by that producer within the session.
- A refusal leaves the affordance **not offered** and annotated — and the producer's belief is unchanged
  across it: no neglect-clock reset, no tightened posterior, nothing that a real observation would do.
- It comes back **only** with a different pose, and it does come back (a refusal must not retire an
  affordance permanently — that is the failure mode on the other side of this rule).
- The robot serves the other affordances while one is refused (it did not, before: 12 escapes with three
  affordances idle).
- Escapes per affordance fall to ~0 in a run where a producer proposes an unreachable pose.
- The controller's useless-spot list stays **empty** once producers honour refusals — it becomes the
  backstop it should have been, and its firing is then a signal that some producer is ignoring the flag.

## 8. Open questions

1. Should a refusal decay? A pose unreachable because of a moved chair becomes reachable later. Leaning
   yes on the producer side (it re-plans anyway), no on the controller's backstop (see §6).
   ★The retire-the-offer design makes this safer than it looks: a producer that later believes the pose
   is fine simply publishes it again, and that is a *new* offer rather than a stale one being retried.
2. Should the refusal say *why* — `not_footprint_feasible` vs `wedged_repeatedly`? The first is a map
   fact the producer could check itself; the second it cannot know. That argues for at least a reason
   code, so producers can treat "your pose is inside geometry" differently from "my body could not do it".
3. Does the fridge's underlying `room_polygon` bug make most of this moot for that agent? Fixing the NBV
   so it stops proposing back-face viewpoints removes the common case — but not the general one, and the
   protocol gap is real regardless.
