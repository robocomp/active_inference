# RoomTransition — a model-checkable spec of the room-transition protocol, and a trace checker

**The pattern.** No barrier forces the four agents into sequence. Each agent is a set of **guarded actions** over
**graph facts**, and every fact has **one writer**. The order comes from the guards: an action is enabled only when the
facts it reads exist. That makes the protocol checkable. **Safety** = an invariant over the facts holds in every
reachable state. **Liveness** = under fair scheduling of the agents, a fact the protocol promises (the room switch, the
end of the tour) eventually holds.

* `RoomTransition.tla` is a model of the **C++ as it is** on `feat/room-recognition` (2026-10-30). It is not a model of
  the design document. Every guard carries a `file:line` citation.
* `trace_check.py` (`room_concept/estimation_scripts/scripts/`) checks the same invariants on a **recorded** graph
  trace (`graph_trace.csv`), plus a bounded liveness check against Webots ground truth.

## Run

```bash
TLA=/home/pbustos/.vscode/extensions/tlaplus.vscode-ide-2026.10.280155/tools/tla2tools.jar
cd docs/superpowers/specs/tla
java -Xmx3g -XX:+UseParallelGC -cp $TLA tlc2.TLC -workers 4 -noGenerateSpecTE -config RoomTransition.cfg RoomTransition.tla
#   other configs: RoomTransition_{bug1,bug1_liveness,bug2,strict,owner,broad}.cfg

python3 room_concept/estimation_scripts/scripts/trace_check.py \
    ~/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/recog_proto11 [--bound 20]
```

Every config finishes in under 10 s; the largest state space is 45k states.

## The model

**World.** Rooms R1, R2, R3 and holes DOOR_1 (R1–R2), DOOR_0 (R1–R3), DOOR_2 (R3–R2), as in `threerooms-piso.wbt`.
**Tour:** `cross new`, `cross room_1`, `cross new`, `cross new` (into the remembered R2: case 2, recognition),
`cross room_3`, `cross room_1`.

| Agent | Action | Guard (from the code) | Writes |
|---|---|---|---|
| room_concept | `RC_Cross` | Robot physically outside the RT room. PassageMemory was evaluating while the robot passed (`HEAD:901` old / `:905` fix). A door of the passed hole hangs from this room (`:1004`). | New room node (`proto` edge, `:1116`), or the recalled room (`:1072-1077`, promoted at once `:1738`). Robot RT re-parented (`:1139`). Name `room_<max(live, hwm)+1>` (`:1054-1068`). |
| room_concept | `RC_Promote` | Start-up survey finished (`:1838`). Not fair. Optionally impossible (`SURVEY_CAN_PROMOTE`). | `proto` edge removed (`:2055`). |
| room_concept | `RC_Judge` | Pending proposal. `admit` (`proposal_adoption.cpp:287-296`): P is room_concept's provisional room (`late_recall.cpp:66`), still proto, name K free. | Rename in place (`:337`) + `accepted`, or `rejected`. |
| ltsm_agent | `LT_Cycle` | One `compute()` (`specificworker.cpp:343-397`), in the code's order, in **one atomic step**: **(1) passages** (`passage_live.cpp`): retire dead passages (`:103`); pair each door of a proto room ≠ `current` with the current room's nearest door (`:119-176`). **(2) Recognition**: verdict → commit/carry-over (`room_swap.cpp:343-383`); withdraw if the robot left P (`:399`); else may propose K (`room_recognizer.cpp:352`). **(3) Promotion**, when room_concept's RT room ≠ `current` (`:546`): archive the old room; memory passage from the live pair, or from the old room's door nearest the robot plus a DERIVED face (`room_eviction.cpp:700-930`); flip `current` (`:975`); **delete the old room's subtree, doors included** (`:990`, `:1288`, `:1543`). **(4) Recalls** (`room_recall.cpp`): `recall_<k>` under every door whose hole memory knows (`:137`), unless room k is live (`:162`). | `passage_N`, memory, `current`, deletions, `recall_*`. |
| door_concept | `DC_Follow` | `current` ≠ the followed room (`:3216`). | Adopt the entry mirror (`:3432`), else recreate the lost crossed door (`:3442`); offers move with the room. |
| door_concept | `DC_Mirror(P)` | P is proto and not followed (`:3282`), not announced by a recall (`:3273`), and has a source door of the crossed hole (`:3300`). | Entry mirror under P (`:3326`). |
| door_concept | `DC_Birth(h)` | Perception, fair: the robot is in the followed room and no door of h exists yet. | Door node. |
| door_concept | `DC_Absent(d)` | Environment, ≤ `MaxAbsent`. A fitted door of the hole the robot has just passed (or **any** door if `ABSENCE_ANYTIME`). **`FIX_HOLD`**: not while `executing` names its `aff_<d>_cross` (`door_existence.cpp:823-848`). | Door deleted (`:1683-1699`). Its recall's RT edge goes with it. |
| controller | `CT_Track` | `current` changed id or name (`controller_world_model.cpp:49`, `track_room`). | Room id, name, `prev_name`. |
| controller | `CT_Start` | `pick_door` over the doors of the `current` room (`mission_door_relation.h:47-67`): `room` → a `recall_<room>` child; `new` → no recall, no match, plus the **`recall_<prev>` guard** (`specificworker_mission.cpp:502`). The door's cross is offered. | `executing` claim. |
| controller | `CT_Pass` | Environment, fair: the robot passes the hole. | Robot's true room. |
| controller | `CT_Done` | The `current` id changed, into a room the door's passage leads to (`controller_door_passage.h:799-803`, `:1169`). | Next leg. |

**Abstractions** (the only places the model departs from the code):

* Geometry and probability become nondeterministic choice.
* The graph is sequentially consistent: an agent cycle is one atomic step. DDS reordering is not modelled here;
  `protocol_mock.py` has that adversary.
* Recognition, when it fires, is correct.
* The harness's closed doors are modelled as "a `new` leg picks only its hole".
* Doors that door_concept restores from `door_identities.csv` are folded into perception births.

## Properties

| Name | Kind | Meaning |
|---|---|---|
| `ExecCrossDoorLiveBeforeSwitch` | safety | While the controller crosses d and `current` is still the room it crossed from, d exists. **Bug 1.** |
| `ExecCrossDoorLive` | safety | The user's literal form, held for as long as the controller holds the claim. |
| `OneOwner` | safety | No agent deletes a node another agent owns (design rule 1). |
| `PassageJoinsTwoRooms` | safety | A live passage matches two faces of one hole, in two different rooms. |
| `MemoryPassagesSound` | safety | A remembered passage joins two remembered rooms, through a hole whose ends they truly are. |
| `LiveNamesUnique`, `NameIsIdentity` | safety | One live room per name; a name (room_id) always denotes the same physical room, live and in memory. |
| `RecallsPointRight` | safety | `recall_<k>` under door d names the room really behind d. |
| `ArrivedRight` | safety | Each completed leg ends with the robot in the leg's destination. |
| `RenamedInPlace` | action | An accepted recognition creates and deletes no room node. |
| `CrossingNoticed` | liveness | The robot physically past a hole ~> room_concept's room is the robot's room. **Bug 2.** |
| `CurrentFollows` | liveness | room_concept's RT room ≠ `current` ~> they agree. |
| `TourCompletes` | liveness | The six-leg tour completes. |

## Results (TLC 2026.10.28, 4 workers, 3 GB)

| Config | FIX_HOLD | FIX_UNPROM. | Result | States |
|---|---|---|---|---|
| `_bug1` | off | on | **`ExecCrossDoorLiveBeforeSwitch` violated** in 5 states: leg 1, the robot passes DOOR_1, door_concept removes the door, and the controller is still `executing` its cross. | 20 |
| `_bug1_liveness` | off | on | **`CrossingNoticed` violated**: with the crossed door gone, room_concept has no aperture to close the crossing on. The door cannot be reborn either, because the robot is no longer in its room. Stuck for ever. | 33,314 |
| `_bug2` | on | off | **`CrossingNoticed` violated**: recog_proto11 leg 2, step for step (below). | 233 |
| `RoomTransition` | on | on | **No error.** All 8 invariants, `RenamedInPlace`, and the 3 liveness properties hold. | 11,663 |
| `_broad` | on | on | **No error**, with any fitted door absent at any time, ≤ 2 verdicts. | 45,131 |
| `_strict` | on | on | **`ExecCrossDoorLive` violated**: ltsm's eviction deletes the crossed door in the same step as it flips `current`, while the controller still holds `executing` (it only sees the switch on its next cycle). | 17 |
| `_owner` | on | on | **`OneOwner` violated** at the first eviction: ltsm deletes room_concept's room, walls and floor and door_concept's doors (`room_eviction.cpp:1288`, `:1543`). | 17 |

**Vacuity checks** (each was violated on purpose, so each is reachable): the tour completes, a recognition is
accepted, a fit rejects, an absence fires, room_4 is born. Larger bounds (`MaxAbsent=3`, `MaxRejects=2`,
`MaxDoorIds=24`): no error, 68k states. Promotion with no "robot inside" gate: no error, 51k states.

**The bug 2 trace** (`_bug2`, 15 states), with the matching recog_proto11 events:

1. room_1 has DOOR_1's face (door_13). `cross new`. The robot passes.
2. `RC_Cross`: room_2 is born, proto (t=…876523).
3. `DC_Mirror`: entry mirror door_55 (…876546).
4. `LT_Cycle`: passage_0 = {door_13, door_55} (…877256); `current` → room_2, room_1 and door_13 deleted, and
   `recall_room_1` posted under door_55 (…877264 / 877276).
5. door_concept follows and adopts door_55. The controller sees the switch: leg 1 done (…877335).
6. passage_0 retires, because door_13 is gone (…878256).
7. `cross room_1` resolves to door_55 (…880578).
8. The robot passes back (…888893). room_2 is still unpromoted, so PassageMemory was not running, and nothing is ever
   written again.

**One model error I found and corrected.** A first version let ltsm flip `current` before door_concept made the entry
mirror. With no live pair, memory then had no passage, and `TourCompletes` failed. The code does not do this:
`evict()` falls back to the old room's door nearest the robot (read through the RETIRED RT edge, which is frozen at the
crossing) and DERIVES the far face (`room_eviction.cpp:700-930`). With that in the model, the race is harmless.

## Code vs design (docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md)

1. **R1 `exit` edge: not written by any agent.** No `exit_edge_type` anywhere in room_concept or door_concept. ltsm
   only asserts in a self-test that a passage carries none (`specificworker.cpp:712`, `:1060`). The crossing fact is
   room_concept's new room node plus the robot RT re-parenting.
2. **Consumers key on ltsm's `current`, not on room_concept's fact.** door_concept follows `current`
   (`step_room_following`). The controller resolves its room and completes legs on `current`
   (`controller_world_model.cpp:49`). This contradicts M2 and "room_concept owns where the robot is". ltsm's promotion
   does follow room_concept's RT (`:546`).
3. **Rule 1 and R2 step 4 are violated by design of 2026-09-14.** ltsm's eviction deletes room_concept's room, walls
   and floor and door_concept's doors: 35 nodes per eviction in every recorded run.
4. **R2: there is no transition phase.** Offers and selections continue throughout. The `recall_<prev>` guard
   (11936d55), which R2/M1 remove, is still in `resolve_door` (`:502`).
5. **M4/R5: identity is still by NAME.** Recalls are `recall_<name>` (`room_recall.cpp:169`). Room names come from
   room_concept's live scan plus `room_name_hwm.txt` (`:1054-1068`), not from ltsm. The controller matches recall names
   to mission names (`pick_door`, plus `expand_recalls` aliases).
6. **M5: `proto` is removed at the survey's promotion** (`:1838` → `:2055`), not when recognition is over. If the survey
   finishes first, `admit` refuses every later proposal ("not provisional"), and the room stays a duplicate.
7. **R1 "pairing from the exit edge only": not implemented.** Pairing is still "door under a proto room ≠ current +
   the current room's nearest door" (`passage_live.cpp:119-176`). Eviction adds its own nearest-door / derived-face
   fallback.
8. **R3 (predicted recalls) and R4 (door restore from the recall, not from `door_identities.csv`): not implemented.**
9. **The crossing HOLD (d57997dc) keys on a consumer's claim (`executing`), not on a protocol fact.** It protects only
   the face that claim names. The far face (the adopted mirror) can be judged absent between the switch and the
   controller's completion. The model shows this is recoverable: rebirth + the recall from memory (`_broad` passes).
10. **Before the uncommitted fix, `step_proto_room` contradicted** the design's "a provisional room left before
    recognition decides is archived as a new room" (bug 2).

## Trace checker results (`trace_check.py`, bound 20 s)

| Run | S1 exec-door | S2 owner | S3–S6 | L1 crossing (gt) | L2 |
|---|---|---|---|---|---|
| recog_proto11 | **2**: door_55 by door_concept, BEFORE-SWITCH (**bug 1**, t=1793362894464); door_13 by ltsm, AFTER-SWITCH | 35 (ltsm eviction) | OK | **1**: DOOR_1 crossed back at t=1793362888893, no room fact for the remaining 254 s of ground truth (**bug 2**) | OK |
| recog_proto12 | 1 (door_13, AFTER-SWITCH) | 35 | OK | OK (1 crossing, fact at −0.1 s) | OK |
| recog_proto13 | 1 (door_13, AFTER-SWITCH); door_57 at shutdown, not counted | 35 | OK | OK | OK |
| recog_proto14 | 1 (door_13, AFTER-SWITCH) | 35 | OK | OK | OK |

Notes on how the checker reads the trace:

* door_concept's teardown deletes `aff_<door>_cross` a millisecond **before** the door, which silently drops the
  `executing` edge (the recorder logs no edge_delete for edges into a deleted node). The checker therefore attributes
  an `executing` edge lost with the affordance to the door removal that follows from the same agent within 1 s.
* A removal by an agent whose own agent node was just deleted is a graceful shutdown and is reported but not counted.
