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

## R2 — Structural changes are SERIALIZED by a chain of facts; the transition is a phase that freezes door actions

User 10-30: "structural changes kind of freeze the fleet until done to preserve coherence". Serialization is achieved by a
CHAIN: each structural step is taken by its one owner only when its predecessor's fact is visible, so no two structural
steps of a transition can overlap, whatever the DDS delays.

- **Chain of kind `cross`** (each step = one owner, one fact):
  1. room_concept: `exit` OLD_FACE → NEW_ROOM (opens the phase), then verifies a predicted K (R3) or creates P, and
     re-parents the robot (fact: the robot's RT parent is NEW_ROOM).
  2. door_concept, on fact 1: NEW_FACE + `exit` NEW_FACE → OLD_FACE; restores K's doors from the recall (R4).
  3. ltsm, on fact 2: archives OLD_ROOM; passage from the `exit` pair; THEN posts `recall_<OLD_ROOM>` under NEW_FACE
     (closes the phase).
  4. door_concept and room_concept, on fact 3: each deletes its own nodes of OLD_ROOM (doors; room, walls, floor).
- **The phase** is open from fact 1 until fact 3. While open: no door affordance is offered or selected, and no mission
  step resolves a door; the controller only finishes the cross already executing. Perception and each agent's attribute
  updates continue. No timer: the mission's own budget is the only bound.
- **Rules found by the protocol mock (10-30, `protocol_mock_v2_report.md`)**:
  - **M1 — the way back is marked by a LASTING fact.** `exit` edges die with OLD_FACE at cleanup; the lasting mark is
    `recall_<OLD_ROOM>` under NEW_FACE (and the passage `match`). `cross new` excludes doors with a recall child or a
    match (as today); because no door resolves while the phase is open and fact 3 closes it, the controller never reads an
    ABSENCE: the `recall_<prev>` guard (11936d55) is removed.
  - **M2 — a leg completes on a lasting fact**: the robot's room (room_concept's RT parent, by room_id), never on an
    `exit` edge.
  - **M3 — ordering inside ltsm**: `recall_<OLD_ROOM>` under NEW_FACE is posted only AFTER OLD_ROOM is archived and the
    passage joined (it is the phase's closing fact). A PREDICTED recall (R3) is never posted under a door of a room while a
    phase is open, and never names OLD_ROOM.
  - **M4 — recall nodes are keyed by identity (R5)**: `recall_<room_id>_<door node id>`, unique while the left room is
    still alive; its `room_id` attribute is what consumers read.
  - **M5 — `proto` lasts while recognition can still answer**: room_concept removes P's `proto` edge only when P is adopted,
    or when ltsm has stated that recognition is over for P (its recognition episode for P ended without a proposal), not at
    "first publishable".
  - **M6 — a predicted recall does not satisfy `cross <room>` until verified**: the controller resolves `cross K` only by a
    recall that is not predicted, or by a predicted one whose crossing room_concept has already verified (the recall is
    re-posted as ordinary after verification). Unverified predictions only shape the crossing (R3).
  - **M8 — adoption is a chain too, and never overlaps a crossing**: ltsm posts a proposal (case 2) only while no
    crossing phase is open; room_concept's verdict, door_concept's re-seat/restore and ltsm's carry-over follow as in case 2
    steps 7–9, each on its predecessor's fact. A crossing decided while a proposal is pending abandons the proposal first
    (room_concept answers it `rejected`, then opens the crossing).
  - **M7 — missions resolve room names through ltsm's identities**: a mission's `room_N` is resolved, when the mission is
    accepted, by ltsm's memory (live room with that display name, else the remembered room with it) to a room_id; from then
    on the mission holds the id.
- **Tried and rejected: an explicit transition token** (one `transition` node whose `state` names the only agent allowed to
  write). In the mock (1000 seeds, ltsm lag 8 / explore 6) it passed 496/1000 vs the chain's 965/1000, and was slower (5.6 vs
  4.1 ticks per transition). Cause: in a replicated graph the token is ADVISORY — agents keep writing until its creation
  reaches them. Making it real needs an acknowledged barrier (every agent confirms "frozen" before the first step), more
  state and more ways to stall. The chain serializes without it.

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

## R5 — One identity rule for rooms AND doors: ltsm issues identities, names are never keys

User 10-30: room naming "faces a similar problem as door naming". Both kinds of long-lived entity had identity split
between a live agent's private counter and ltsm's memory, and a NAME doubling as the identity:

| | today | failures |
|---|---|---|
| rooms | room_concept names `room_<k>`, `room_id = k`, k from the live graph + `etc/room_name_hwm.txt`; ltsm names memory `r<room_id>_<name>`; adoption renames P→K (name AND room_id) | T18 stale prefix cache, T14 controller aliases, proto9 room_4→room_2 burns a number |
| doors | door_concept names `door_<n>` from its counter; remembered rows keyed by ROOM NAME in `etc/door_identities.csv`; a crossed hole gets a new face name (door_13 ↔ door_49) | proto10 stale look from another run, try-era duplicate names, "why door_49?" |

The rule, for both:
- **Identity = (kind, id) issued by ltsm**, the only long-term memory. Rooms carry `room_id`; a hole carries its passage
  (already the hole's identity); a door FACE is identified by (room_id, passage). ltsm publishes the next free `room_id`
  as an attribute of its own agent node; room_concept takes it at step 1 of the crossing chain (R2) for a new room. A provisional room that
  is adopted as K returns its number (ltsm does not advance its counter for it). `etc/room_name_hwm.txt` is removed;
  door_concept's counter only makes DSR node names unique.
- **Names are display only.** `room_<room_id>`; doors keep opaque `door_<n>`. Nothing keys on a name: recalls, remembered
  door rows (R4: they live in ltsm memory, keyed by (room_id, passage)), caches (ltsm's memory prefix), controller
  aliases and missions resolve by id. A mission that names `room_1` is resolved to its room_id once, when it is accepted.
- **Renames are cosmetic and happen only inside a chain (R2: the adoption's).** With R3 a predicted room enters with its own name and id, so
  a room rename happens only in `adopt` (case 2). A door face may be renamed to its hole's remembered face name at
  step 2 of the crossing chain (the 10-30 note "rename the entry mirror to the old face's name"), purely for the human reader; since
  nothing keys on names, it can never break anything, and it can be dropped if it costs anything.
- **One wipe.** Resetting memory = deleting ltsm's memory file; no other agent persists identities.

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
- The controller's `recall_<prev>` guard on `cross new` (11936d55) → R2 (phase + M1).
- ltsm's eviction deleting other agents' nodes → R2.
- door_concept's restore and looks from `door_identities.csv` rows → R4.
- The tour's `explore … until=explored>=0.9` step (done, 66699077).
- `room_concept/etc/room_name_hwm.txt`, and every lookup of a room or door by NAME (prefix caches, aliases, rows) → R5.

## Verification

1. The protocol mock implements Revision 2 (`--protocol v2`, the chain with M1–M7; v2t kept only as the rejected comparison), with the same adversary (write delays, agent stalls) plus
   map error on predicted transforms, and must pass ≥ 1000 seeds with the faults still detected; it also runs the cases
   R3 routes to the fallback (no prediction; prediction rejected).
2. Each R-item's C++ change is pinned by a unit test in its owner, then the six-leg tour, graded with the ground-truth log
   (gt_pose.tsv, f6f5353d).

# Revision 3 — 2026-10-30 evening: the crossing is an ACTION, its fact is written by the agent that performs it

## Why this revision

Revision 2's R1 was never built (no agent writes `exit`), and the day was spent patching the agents that GUESS the
crossing instead. pingpong2 (all of 10-30's fixes in: F1, F2, F3, the door_2 proto-wall fix) passed leg 1 and blocked on
leg 2: door_concept re-judged door_1's leaf from inside the doorway, the entry face inherited it, read "shut", and the
robot was told to open a door it had just walked through. Every failure of the day has the same shape:

| Failure | Who re-decided something the robot had already done |
|---|---|
| door_55 deleted mid-crossing (proto11) | door_concept re-judged the EXISTENCE of the door being crossed |
| room never switched (proto11, pingpong1) | room_concept re-inferred "crossed?" (p_cross) from a room-frame door estimate |
| false room_3 (proto15) | room_concept inferred a crossing through a phantom door |
| robot spinning (proto12) | controller planned on the room-frame door, 0.45 m off |
| door_2 drift (pingpong1) | door_concept re-fitted the hole the robot had just passed through |
| leaf "shut" (pingpong2) | door_concept re-judged the LEAF of the door being crossed |

Four agents each reconstruct, privately, from room-frame estimates that are at their WORST during a crossing (a new frame,
no walls yet, views from inside the doorway), an event one agent MEASURES: the controller drives the body through a hole
it perceives in its own frame (the interaction anchor) and its servo reports `through` once the opening is
`anchor_cross_standoff_m` behind (controller_door_passage.h:197-230, controller_runtime_types.h:204). Every disagreement
between the guesses needed a guard; every guard broke the next case.

Revision 3 changes WHO writes R1's fact and what it carries; everything else follows from it. Agents still coordinate by
guarded actions over shared facts (the DSR pattern); what changes is that the guards read a RECORD OF AN ACTION, written
once by its one performer, instead of thresholded estimates that several agents recompute.

## X1 — The controller writes the crossing fact

- When the anchor servo reports `through` on the door it is executing, the controller writes ONE `exit` fact (edge type
  installed in cortex, dsr_edge_type.h:65-72): the face crossed (OLD_FACE), its room (OLD_ROOM), a sequence number, and the
  room PREDICTED behind the face — the `recall_<k>` the leg resolved its door with when the cross STARTED, or "new". The
  known/unknown decision is therefore taken before crossing (Revision 2 R3's intent, from the recall that exists today) and
  frozen in the fact: no agent reads a recall's ABSENCE at crossing time (CE-4 / the staged-recall gap cannot arise).
- One writer, written once per crossing. Nothing else creates a crossing.

## X2 — Everyone keys on that fact

- **room_concept** switches on the `exit` fact alone, if it names a face of ITS room: the predicted room if that name is
  free, else a new provisional room; it records the fact's seq and hole on the room it enters (`via`). No p_cross, no
  PassageMemory, no survey gate (F1), no first-sight rule (F2).
- **door_concept**, on the `exit` fact once room_concept has entered a room by it: the hole's face in that room
  (NEW_FACE) exists, is passable, and is the twin of OLD_FACE (`exit` NEW_FACE → OLD_FACE as in R1). Both faces are
  BACKED by the crossing: a verdict from poor views (absent, leaf shut) does not overturn a backed face. In Active
  Inference terms the crossing is an observation of passability far more precise than any doorway view, so the posterior
  stays there; no new gate is introduced, the views simply cannot outweigh it. (A door really closed later by a person is
  out of scope here; it needs evidence of comparable precision, e.g. a failed push, and is noted as open.)
- **ltsm** pairs a passage from the twin only (no "door under a provisional room + nearest door"), and remembers it from
  the entered room's `via` (written from the fact) — no nearest-door / derived-face fallback in the eviction.
- **controller** tracks room_concept's room (not ltsm's `current`). A leg completes when room_concept has acted on THIS
  leg's `exit` fact (no "any room change ends the leg"; FIX_COMPLETE_ON_DOOR is built in). `cross new` excludes a backed
  face; the `recall_<prev>` guard (11936d55) goes.
- **door_concept** follows room_concept's room (not `current`).

## X3 — One transition at a time: a leg closes on the archive (Revision 2 R2, reduced to one guard)

The model found that X1+X2 alone are NOT enough in the three-room tour: the controller may open leg 3 (a `new` leg, which
needs no recall) before ltsm has archived leg 2's crossing; ltsm, which handles one switch per archive, then remembers a
passage room_2–room_3 through DOOR_0, which does not exist (`MemoryPassagesSound`). Ping-pong cannot show it, because every
return leg needs a recall that exists only after the archive. The cure is R2's closing fact with nothing else of R2: **a
leg completes only when ltsm's `current` equals room_concept's room** (the left room is archived). No token, no barrier,
no phase flag.

## What X1–X3 remove (each was a guard or a rule added to cover a guess)

| Agent | Removed | Replaced by |
|---|---|---|
| room_concept | PassageMemory crossing detection, p_cross, the survey gate (HEAD:901 / F1), the first-sight rule (F2), the recall read at birth | switch on `exit` (2 guards) |
| ltsm | pairing by provisional-room + nearest door (passage_live.cpp:119-176), eviction's 3-case nearest/derived-face fallback (room_eviction.cpp:700-930), the staging exception (F3) | pair from the twin (1 guard); passage from `via` |
| door_concept | entry mirror's 4-way guess (step_entry_mirrors :3221-3330), adopt/recreate on follow (:3432/:3442), any re-judging of a crossed face | NEW_FACE on `exit` (2 guards); backed faces (1 guard) |
| controller | `recall_<prev>` guard, completion on any `current` change / target_rooms (controller_door_passage.h:799-803, :1169) | `exit` write (1 guard); completion on ack + archive (2 guards) |

In the models: the current code needs ~20 crossing guard conjuncts plus F1–F3 and FIX_HOLD to pass ping-pong; the design
needs 9 plus FIX_HOLD (kept: it protects the face during the approach, before any fact exists).

## Verification (docs/superpowers/specs/tla/RoomTransitionExit.tla, README "Revision 3")

Model of the DESIGN (RoomTransition.tla stays the model of the code). Adversary = proto15 + pingpong2: phantom doors in a
doorway, a stale identities row, wrong recognition proposals, wrong verdicts (absent OR leaf shut) on the faces of the hole
being / just crossed, which may PERSIST.

**Encoding A (user 10-30, the model as committed): the fact is a CHAIN of edges, each dying with its source node** —
link 1 `exit` ROBOT → OLD_FACE (written by the controller, `room_id` = prediction; Lucas Bonilla's edge), link 2 `exit`
OLD_FACE → NEW_ROOM (room_concept), link 3 `exit` NEW_FACE → OLD_FACE (door_concept), link 4 ltsm archives OLD_ROOM only
once link 3 exists (its archive deletes OLD_FACE and with it links 1 and 2). No sequence counters: a fact about leaving
room R cannot be acted on twice. Shared reader/writer: `common/room_transition/exit_fact.h`. (The first model kept the
fact as persistent controller state; in the graph it dies with OLD_FACE, so ltsm could have archived before door_concept
read it — encoding A orders them.)

| Config | FACT_HOLD | CLOSE_ON_ARCHIVE | Result | Distinct states |
|---|---|---|---|---|
| `_pingpong` (6 round trips) | on | on | **no error** (10 invariants, RenamedInPlace, CrossingNoticed, TourCompletes) | 24,933 |
| `_pingpong_nohold` | off | on | **TourCompletes violated** — pingpong2 exactly: room_2's face judged shut after leg 1 | 32,114 |
| `_pingpong_noclose` | on | off | **TourCompletes violated**: a leg opens before ltsm archives; link 4 then waits on a crossing that is no longer the latest | 32,072 |
| `_tour3_noclose` | on | off | **TourCompletes violated** (same) | 36,135 |
| `_tour3` (three rooms, case-2 recognition) | on | on | **no error** | 3,125,038 |

Vacuity (`RoomTransitionExit_vacuity.sh`): tour end, leg 7, a rejected proposal, a wrong verdict, a phantom — reached in
both scenarios; an accepted recognition — reached in tour3 (impossible in ping-pong, where no correct proposal exists).

Not modelled (residual hazards to watch live): the anchor reporting `through` without the body being through (a controller
bug, no adversary for it); a robot that crosses and comes back before the controller writes the fact; a crossing not
driven by a mission (teleop) — it writes no fact, so the room does not switch.

## Order of work

1. controller: write `exit` at `through` (+ unit test on the passage state machine). DONE (route mode): P_post = through →
   link 1; Done on room_concept's switch + ltsm's archive; room = room_concept's RT room; `cross new` skips crossed faces;
   `recall_<prev>` guard, the room-switch guess, the target-rooms filter and the mid-cross shut re-read removed.
2. room_concept: switch on `exit`; PassageMemory crossing detection OFF behind a flag, deleted after the live ping-pong.
3. door_concept: NEW_FACE + twin + backed faces on `exit`; entry-mirror guess OFF.
4. ltsm: pair from the twin; passage from `via`.
5. (folded into 1.)
6. trace_check.py: S9 every room switch has an `exit` fact; S10 no backed face deleted or shut.
7. Live ping-pong ×6 (pingpong_tour.sh); only then the three-room tour.
