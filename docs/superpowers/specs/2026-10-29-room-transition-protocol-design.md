# Room-transition protocol — design

2026-10-29 · agreed with the user in conversation · branch `feat/room-recognition` (worktree `rt-build`)

## Intent and why

When the robot crosses a door, four agents update the shared DSR graph: door_concept, room_concept, ltsm_agent and the
controller. Today each acts on its own clock, with no agreed order, and none of them owns "the crossing is done".
Every failure of the three-room tour on 10-29 sat in a gap between two such updates:

- **A reader read before the writer had finished.** The controller took the door it had just entered through as
  "new" (try30); took a door re-made on the far side as "lost" (try27, try31).
- **Two agents rewrote the same thing.** ltsm's swap deleted the provisional room and room_concept then failed to
  adopt the recalled one (try24 abort, try32 undo). room_concept rebuilt room_3's walls while door_concept's doors
  still hung from the old ones, and ltsm archived both: room_3 was stored with two layouts and the map broke by 90°
  (try32).
- **A decision was taken on half-built data.** A single frame of a half-estimated room swapped room_3 for room_2
  (try30).

The user's ruling: **stop patching the controller; establish a protocol for the agents to coordinate their updates to
the graph.** It must be simple and as robust as a wall thermostat, and there is a deadline. The protocol is the
sequence of events, who owns each step, and the fact each step leaves behind for the next.

## Goals

- The three-room loop completes reliably: room_1 →DOOR_1→ room_2 → back → room_1 →DOOR_0→ room_3 →DOOR_2→
  recognised room_2 → back → room_1.
- One exact sequence for each of two cases (below), each step with **one owner** and ending in **one fact written to
  the graph**; the next step starts only when that fact exists.
- Consumers (controller missions, the tour harness) read **only those facts**, never infer order from timing.

## Non-goals

- New perception (the phantom door on room_1's south wall; DOOR_2 sometimes undetected). Separate work.
- Changing how a room is estimated, how recognition scores evidence, or how the door route is driven. They already
  work; the protocol only orders them.
- Multi-robot, or more than one transition in flight at once.

## Anti-goals (would make this a failure even if it "works")

- Timers, settle delays or `if (x > k)` gates standing in for a missing fact (also the codebase rule: no thresholds
  without a model reason).
- Any agent deleting or rebuilding a node it does not own.
- More controller heuristics guessing what the other agents are doing.

## Constraints

- CLAUDE.md applies: QUEUED (never `DirectConnection`) DSR signals; type-attributed attribute access only; a NEW
  attribute needs a `REGISTER_TYPE` line in cortex AND the user reinstalling cortex — so **prefer existing attributes
  and node types**; `std::from_chars` for parsing; SIGTERM-only stops; builds `-j8` max.
- Each agent owns its nodes (CONCEPT_AGENT_LIFECYCLE.md, `[Owns]`): room_concept owns room nodes, their walls/floor
  and the robot's RT edge; door_concept owns door nodes; ltsm_agent owns passages, `recall_*` nodes, recognition
  nodes and the long-term memory graph; the controller owns no geometry (only its affordance claims).

## Decided: ownership and rules

| Owner | Writes | Is the only one allowed to |
|---|---|---|
| room_concept | room nodes, walls, floor, the robot's RT parent | create / switch / adopt / delete a live room; say where the robot is |
| door_concept | door nodes (incl. a door's face in a new room) | create / restore / delete live doors |
| ltsm_agent | passages, `recall_<room>`, the recognition proposal, memory | archive a room; say what lies behind a door; propose that a room is a remembered one |
| controller | its claims | drive; it reads facts, writes none of these |

Rules for every transition:

1. No agent deletes or rebuilds a node it does not own. (Today's swap deletes room_concept's provisional room from
   ltsm — removed by this design.)
2. No agent acts on a fact before its owner has written it. Waiting for a fact is fine; waiting for time is not.
3. Recognition **proposes**; room_concept **decides**.

## Decided: case 1 — crossing a KNOWN door (do first; fewer steps)

Example: room_2 → room_1 through DOOR_1, where `passage_0` already links both faces.

1. **Before crossing — ltsm.** `recall_room_1` hangs under DOOR_1's live face, carrying room_1's remembered outline
   and the transform room_1 ← door face. *Fact: this door leads to room_1.*
2. **Crossing — controller.** The door route drives the robot past the aperture. Writes nothing else.
3. **Room switch — room_concept.** When the robot is inside, room_concept instantiates room_1 **from the recall**:
   remembered frame and outline; robot pose = door transform ∘ current pose. No provisional room, no recognition.
   *Fact: the robot's RT parent is room_1.*
4. **Doors — door_concept, on fact 3.** Restores room_1's remembered doors, including DOOR_1's room_1 face (the same
   hole, the same passage). No entry mirror is invented.
5. **Memory — ltsm, on fact 3.** Archives room_2; increments `passage_0`'s crossings; posts `recall_room_2` under
   DOOR_1's room_1 face.
6. **Mission — controller.** `cross room_1` completes on fact 3 alone.

## Decided: case 2 — crossing a door for the FIRST time into a room known from elsewhere

Example: room_3 → DOOR_2 → room_2 (room_2 known only through DOOR_1).

1. **Before crossing.** DOOR_2 carries no recall: nobody knows what is behind it.
2. **Crossing — controller.** As in case 1.
3. **Room switch — room_concept.** Creates a **provisional room P** and starts estimating it. *Fact: the robot's RT
   parent is P, and P is marked provisional.*
4. **Doors — door_concept, on fact 3.** Makes DOOR_2's face in P.
5. **Memory — ltsm, on fact 3.** Archives room_3; opens a passage room_3 ↔ P; posts `recall_room_3` under DOOR_2's
   face in P. *From here the way back is known.*
6. **Recognition — ltsm.** Compares P with memory, evidence filtered over time (a117e401). **Changes nothing in the
   live graph.** When convinced, writes **one proposal** on P: "P is room_2, with transform room_2 ← P" (plus the
   pose density it already computes). *Fact: a proposal exists.*
7. **Adoption — room_concept.** Checks the proposal against its own scans.
   - **Accept:** P becomes room_2 **in place**: remembered frame and outline; the robot's pose re-expressed.
     *Fact: room_2 is current and adopted.*
   - **Reject:** writes "rejected" on the proposal. ltsm keeps P as a new room; nothing else changes.
8. **Doors — door_concept, on fact 7 (accept).** Restores room_2's remembered doors and merges P's door faces into
   them by place; DOOR_2's face stays as a new face of room_2.
9. **Memory — ltsm, on fact 7 (accept).** Joins the room_3 ↔ P passage to room_2 (the carry-over); posts
   `recall_room_1` under DOOR_1's room_2 face.
10. **Mission — controller.** The leg completes on fact 3; later legs read only recalls.

## What exists today (starting points, not prescriptions)

- Recall nodes: `ltsm_agent/src/room_recall.cpp` (`step`, `post_late`); they already carry outline + pose prior.
- Swap / undo / carry-over: `ltsm_agent/src/room_swap.cpp` (stage, `on_room_left`, `carry_over`; early carry-over
  1968dbc6). Case 2 replaces the swap's live-graph writes with the proposal.
- Recogniser: `ltsm_agent/src/room_recognizer.cpp` (commit/undo, filtered metric evidence a117e401).
- Promotion / archive: `ltsm_agent/src/room_eviction.cpp`; passage numbering and place matching (ab8a8125).
- room_concept late adoption: `room_concept/src/room_scene_graph_late_recall.cpp`; room creation and walls:
  `room_concept/src/room_scene_graph.cpp`; the no-level guard (b1883649).
- door_concept entry mirror and restore: `door_concept/src/specificworker.cpp` (`[proto] entry mirror`, `[restore]`),
  opaque names (02bd9d4f).
- Controller relations `cross new` / `cross <room>`: `controller/src/mission_door_relation.h`,
  `specificworker_mission.cpp`.

## Left to the builder

- **How each fact is represented.** The user accepts a cortex reinstall (10-29): new attributes are allowed where no
  existing one fits, but ALL of them go into ONE `dsr_attr_name.h` change, handed to the user for a single reinstall
  before the code that uses them is built. E.g. "provisional" and "adopted" as a room attribute, the proposal as a metaconcept
  child of P or attributes on the existing recognition node.
- How each agent notices a fact: polling in its compute loop (the presence monitor is poll-based on purpose) or a
  QUEUED signal.
- The order of implementation inside case 1, then case 2, and the tests that pin each step.
- **My call, to confirm while reading:** once the facts exist, the controller heuristics added on 10-29 that guess
  ordering are removed — the vanished-door re-resolve (ac3581f0, 0d31c13f, 71edf852) and "`cross new` waits for the
  recall of the previous room" (11936d55). The relation grammar itself (45dc5f0f) and explore-while-waiting (13be4e94)
  stay.
- **My call:** the stale-wall defect (room_concept leaving old walls that doors still hang from) falls under rule 1 —
  room_concept must keep the walls it owns consistent with the room it publishes — and is fixed as part of this work,
  since it broke the stored map.

## Settled after the builder check (my calls unless marked)

- **"Inside" (case 1 step 3, case 2 step 3).** room_concept decides with the existing probabilistic aperture test
  (P(inside) from the robot's signed distance past the aperture and its σ — today in ltsm's promotion,
  `room_eviction.cpp` "[promotion] … P(inside)"). It moves to room_concept, the owner of "where is the robot"; ltsm
  reads fact 3 instead of computing it. A model, not a timer.
- **When P is created (case 2).** Unchanged from today (room_concept already builds the provisional room as the robot
  enters); only the switch and its order are new.
- **Rename (user 10-29: option A).** cortex's `DSRGraph::update_node` throws on a name change although the LWW engine
  supports it; the guard is lifted in cortex (user reinstalls), verified by a two-graph sync test first.
- **No new attributes.** "Provisional" = the existing `proto` self-edge; the proposal = a metaconcept node using the
  existing `room_id`, `recall_pose_prior`, `delimiting_polygon_x/y` and `state` attributes.
- **Adoption keeps P's node id.** Doors and the passage hang from P, so P becomes room_2 in place — a rename if cortex
  allows renaming a node; if it does not, the builder reports back before choosing a workaround. The existing late
  adoption (`room_scene_graph_late_recall.cpp`) is reused; its try32 rejection ("no pose explains this view", median
  |SDF| 0.16 m) is investigated as part of the work, since a stored map broken by stale walls is a likely cause.
- **Rejection is per proposal.** ltsm may propose again with new evidence; its filtered evidence is rebased at the
  rejection.
- **Missions.** A case-2 leg is always `cross new` and completes on fact 3. `cross <room>` exists only where a recall
  names the room behind a door, so it never waits on fact 7.
- **Walls (rule 1).** room_concept never deletes a wall that carries another agent's nodes; when the layout changes it
  moves that wall in place (same node, new RT), so the doors on it move with it. Walls nobody hangs from may be
  rebuilt freely.
- **Verification.** Live runs are available here (Webots + `recognition_tour.sh`); each step is pinned by a unit test
  first, then by the live loop.

## Success

The loop above, run end to end by `room_concept/estimation_scripts/scripts/recognition_tour.sh`, completes 6/6 legs with
the report's C-checks passing, and the LTSM map it leaves on disk draws consistently: every door face on a wall of its
room, and every loop of passages closing (the try32 check: two paths room_1 → room_2 agreeing).

---

# Revision 2 — 2026-10-30: the crossing is a fact, the transition is a phase, known rooms are predicted

## Why this revision

Three live tours on 10-30 (recog_proto8–10) failed in three different places, and every fix so far repaired one agent's
private reconstruction of an event that nobody writes down: *the robot went from room A into room B through hole H*.

| Run | What broke | The inference that broke |
|---|---|---|
| proto8 | `cross new` found no door: DOOR_2 was paired with the entry door of the same room | ltsm: "a door under a provisional room is an entry; pair it with the nearest door of the current room" |
| proto9 | room_2 adopted 179° off; DOOR_1's face removed; the robot drove back into room_3 | room_concept: shape recognition AFTER entry, then a free pose search inside K |
| proto10 | a look at a stale door from another run's frame held `explored` < 0.9 | door_concept: "a room is the remembered one if its NAME matches" |

The Python protocol mock (`protocol_mock.py`, 435686b4) adds a fourth: `cross new` and room_concept's case-1/case-2
choice both read the ABSENCE of a recall, and a late write is indistinguishable from no write (281/300 schedules fail
with ltsm lagging). The controller survives today only through the `recall_<prev>` guard (11936d55).

Lucas Bonilla's TFM (2024, ch. 6.3.2–6.5), which closed the loop over four rooms, had three mechanisms this protocol lacks:
an `exit` edge written at the crossing; a transition phase in which nothing else acts (no current room; the Scheduler's
"door crossing exception"); and the known/unknown decision taken BEFORE crossing, from the metric map. This revision adopts
all three. It REPLACES parts of the 10-29 design; it does not add to them.

## R1 — The crossing is ONE fact, written once (`exit` edge; the edge type exists in cortex, no reinstall)

- **room_concept** owns "where the robot is", and already decides the switch from the robot's odometric passage through a
  specific aperture (PassageMemory). At that decision it writes **`exit`: OLD_FACE → NEW_ROOM** (the door node it crossed,
  in the room left; the room entered, provisional or known). This edge IS fact 3 ("the crossing"). The robot's RT
  re-parenting follows; consumers key on the edge, never on the RT parent or on timing.
- **door_concept**, on the `exit` edge, creates NEW_FACE (the hole's face in NEW_ROOM, today's "entry mirror") and writes
  **`exit`: NEW_FACE → OLD_FACE** ("this face is the far side of the hole the robot came through"). It is the door
  owner's statement that the two faces are one hole.
- **ltsm** pairs the two faces of the passage **from that edge only** (no nearest-door search, no "door under a provisional
  room" rule).
- **controller**: `cross new` excludes a door with an outgoing `exit` edge (the way back is known by FACT, not by the
  later recall). The `recall_<prev>` guard (11936d55) is removed.
- The `exit` edges are deleted with their faces (owners: room_concept's OLD_FACE→NEW_ROOM goes when the old room is
  archived; door_concept's with NEW_FACE).

## R2 — Structural changes are SERIALIZED: one transition token freezes the fleet until done

User 10-30: "structural changes kind of freeze the fleet until done to preserve coherence". Lucas's Scheduler allowed one
action at a time; this keeps that discipline only where it matters, without a central agent.

- **The token** is ONE node, `transition` (type `metaconcept`, existing `state` attribute; no new attributes). It exists
  only while a structural change is in progress; at most one exists. Its `state` names the step being executed and,
  through the table below, the ONE agent allowed to act. That agent does its step, then advances `state` (its only write
  on the token). A stuck transition therefore says, in one attribute, who it is waiting for.
- **The freeze.** While `transition` exists:
  - only the agent named by `state` creates, deletes or renames nodes or edges (tracker births, wall creation, look and
    affordance nodes all wait; the agent keeps its estimate internally and publishes after);
  - every agent may still update ATTRIBUTES of nodes it owns (poses, beliefs, the robot's RT) — perception never stops;
  - no door affordance is offered or selected and no mission step resolves a door; the controller only finishes the cross
    already executing (the body clears the doorway) and otherwise holds.
- **Kind `cross`** (opened by room_concept at its switch decision, together with `exit` OLD_FACE → NEW_ROOM, R1):

  | state | actor | step | advances to |
  |---|---|---|---|
  | `cross:entering` | room_concept | verify a predicted K (R3) or create P; re-parent the robot; final NEW_ROOM on the `exit` edge | `cross:entered` |
  | `cross:entered` | door_concept | NEW_FACE + `exit` NEW_FACE → OLD_FACE; restore K's doors from the recall (R4) | `cross:faced` |
  | `cross:faced` | ltsm | archive OLD_ROOM; passage from the `exit` pair; `recall_<OLD_ROOM>` under NEW_FACE | `cross:archived` |
  | `cross:archived` | door_concept | delete OLD_ROOM's doors (and with them their `exit` edges) | `cross:released` |
  | `cross:released` | room_concept | delete OLD_ROOM, its walls and floor; delete the token | — |

- **Kind `adopt`** (case 2 only, R3's fallback; opened by ltsm when it posts a proposal):
  `adopt:proposed` (room_concept judges; accept = rename P→K in place with the accepted transform, or reject) →
  `adopt:judged` (door_concept re-seats/restores on accept; nothing on reject) → `adopt:seated` (ltsm carry-over on accept)
  → deleted by ltsm. A proposal is never opened while another transition exists; it waits.
- **Who may open a token**: room_concept (`cross`), ltsm (`adopt`), each only when none exists. Creation is the opener's
  only structural act before its first step.
- **Failure**: an actor that is absent (presence protocol) leaves the token in place and the fleet frozen — visible and
  safe, never a wrong graph. No timers advance a state; the mission's own budget fails the leg as today.
- `exit` edges (R1) say WHICH hole was crossed; the token says WHERE the transition is. Both are kept.

## R3 — Known rooms are PREDICTED before crossing; recognition after entry becomes the fallback

- **ltsm**, for every live door D of the current room that has no recall: using its map layout (`topology_layout`, the
  rooms placed in one frame through the passages' seams), it takes the point just beyond D (the door's cross target) and
  asks which remembered room K contains it. If one does, it posts `recall_K` under D exactly as for case 1, with
  T(K ← D) composed through the map and its covariance from the layout's chain (it grows with the number of seams
  traversed; recog_try32's loop error ~1 m / 0.8° is the scale to expect). A predicted recall is marked as predicted
  (`recall_late`'s sibling flag, or `state="predicted"` on the recall node — existing attributes).
- Crossing such a door is **case 1**: room_concept instantiates K from the recall. Because a predicted transform carries
  the map's error, room_concept verifies it with the EXISTING adoption test, run as a test of the given pose (Params
  `prior_only`, 7ba0ad18: refine locally, never search K for another basin). Accept: K is current, as in case 1.
  Reject: room_concept falls back to a provisional room P (case 2).
- door_concept, on entry into a predicted K: restores K's remembered doors (merge by place), and NEW_FACE (R1) is K's new
  face of the hole (DOOR_2 seen from room_2 for the first time).
- **Case 2 (shape recognition after entry) remains only where the map predicts nothing** — a door whose far side lies
  outside every remembered room in the map (a genuinely new room), or a map component the current room is not connected
  to (a kidnapped robot). Its proposal/adoption path is unchanged (prior_only applies there too).

## R4 — One long-term memory

- The remembered doors of a room come from ltsm's memory only: the recall carries K's door faces (name, pose in K, width,
  last existence belief), and door_concept restores from the recall, never from `etc/door_identities.csv`.
- `door_identities.csv` keeps only door_concept's naming counter (opaque, monotonic names). A memory wipe (the tour
  harness) is then one wipe, and rows from another run's frame cannot exist.
- door_concept's looks at remembered doors are offered only inside a recalled room (66699077), from the recall's list.

## Ownership additions (the mock's spec gaps)

- `proto` self-edge: room_concept writes it when P is created and removes it when P is adopted as K or when P's estimate
  is first publishable (P stays a new room).
- Proposal node: ltsm creates and deletes it; room_concept owns ONE attribute on it, `state` (+ the accepted transform in
  the same update, Task 20). Attribute-level ownership is written into CONCEPT_AGENT_LIFECYCLE.md.
- A provisional room left before recognition decides is archived as a new room (known limitation: a duplicate in memory).
  R3 makes this rare: a room the map can predict never becomes provisional.
- `cross new` with several qualifying doors: the one door_concept has looked at, else the nearest (unchanged, now stated).

## What is REMOVED by this revision

- ltsm's passage pairing by "door under a provisional room + nearest door" (passage_live.cpp step 2) → R1.
- The controller's `recall_<prev>` guard on `cross new` (11936d55) → R1 + the freeze (R2).
- ltsm's eviction deleting other agents' nodes → R2.
- door_concept's restore and looks from `door_identities.csv` rows → R4.
- The tour's `explore … until=explored>=0.9` step (done, 66699077).

## Verification

1. The protocol mock implements Revision 2 both as the implicit chain of facts (`--protocol v2`) and with the explicit
   token (`--protocol v2t`), compared side by side, with the same adversary (write delays, agent stalls) plus
   map error on predicted transforms, and must pass ≥ 1000 seeds with the faults still detected; it also runs the cases
   R3 routes to the fallback (no prediction; prediction rejected).
2. Each R-item's C++ change is pinned by a unit test in its owner, then the six-leg tour, graded with the ground-truth log
   (gt_pose.tsv, f6f5353d).
