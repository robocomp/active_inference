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
