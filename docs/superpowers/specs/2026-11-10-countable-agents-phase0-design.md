# Countable concept agents — Phase 0 (shared groundwork before cloning)

Date: 2026-11-10. Status: DRAFT, awaiting the owner's approval. Nothing is built.

## 1. Intent

The owner wants new concept agents for the bathroom and bedroom of `threerooms-piso.wbt`:

| room | objects |
|---|---|
| bedroom | bed ×1, nightstand ×2 |
| bathroom | toilet, bathtub, vanity (detected as `sink`), mirror |

All of them are COUNTABLE objects. Planned seeds: chair for the nightstand and toilet; table or refrigerator for
the bed, bathtub and vanity; hood for the mirror. Before any clone is made, Phase 0 fixes what every clone would
otherwise inherit and multiply.

**Why now:** every per-agent copy drifts. Room-following exists in chair only (commit `d2a1177c`); the other five
countable agents latch their first room. There are 9 hand-copied `epistemic_planner.cpp` files that differ by 220–280
lines on 140–250-line files. Six more clones would make 15. The owner's standing rule is
[simple structure, not patches]: fix structure once in `common/` rather than patching N copies.

## 2. Goals

### G1. The working room follows the robot, in every countable agent
- Each countable agent works ONLY in the robot's current room (owner, 11-10: "only the robot's current room is enough").
- Every cycle it resolves the room with `rc::room::follow_current_room(held, rc::room::current_room(*G))` and deletes
  (graph node, affordance and approach, fitter state) any of its objects whose `parent` is another room:
  `rc::room::left_behind`. LTOM offers the object back when the robot returns. That is the existing recall path,
  unchanged here.
- Chair already does this (`chair_concept/src/specificworker.cpp:388,414`, `drop_left_behind` at `:1057`). The two
  rules are already shared in `common/room_resolve/room_resolve_rule.h`; only the per-cycle wiring and the drop are
  per-agent.
- Apply it to: **table, bottle, hood, refrigerator**. Each latches its room once (`specificworker.cpp` around
  lines 256–278).
- **Cabinet** is a MASS object (CONCEPT_AGENT_LIFECYCLE.md §6.1). It also latches its room and gets the same
  fix, because the hazard is the frame, not the object kind. Its planner is out of scope (G3).
- The drop logic moves into one shared helper. Each agent supplies only "how to forget instance X" (its fitter,
  affordance and approach teardown), which differs per agent.

### G2. An object is filed only in the room that contains it
- An object is created and kept only if its centre lies inside the CURRENT room's published outline
  (`delimiting_polygon_x/y` on the room node). Anything seen through a doorway in another room is ignored. A concept
  agent cannot write long-term memory (only `ltom_agent` does), so "note it for the other room" is not an option
  (owner, 11-10).
- Chair already has this rule as a containment prior: births outside the polygon are suppressed, and an
  instance outside draws `Existence.OutOfRoomGain` per frame (`chair_fitter.h:88`, `chair_existence.cpp:102`).
  **But it is INACTIVE whenever the room has no polygon yet** (`chair_concept/src/specificworker.cpp` around line
  955 logs "containment prior INACTIVE").
- **`chair_39` investigation (first task of Phase 0):** a chair filed under room_2, although all four chairs in the
  world are in room_1 (y < 9.38). Main hypothesis: it was born in room_2 while room_2 had no published polygon
  (a new proto-room right after a door crossing), so nothing stopped a room_1 chair seen through the open door.
  The room session reports that room outlines are held at known doorways, so with a polygon present a
  containment test would NOT put it in room_2. Confirm or refute this from logs or a reproduction BEFORE
  designing the fix; the fix depends on the cause.
- The rule moves to shared code with G1, so every countable agent and every clone gets the same containment
  prior. Agents that have none today are listed in the conformance audit (G4).

### G3. One epistemic planner for countable objects
- Owner, 11-10: countable objects should have a common epistemic planner, and the copies' differences are most
  likely accumulated patches, not deliberate per-object design.
- Scope: **chair, table, bottle, hood, refrigerator**, plus every new agent. **Out of scope** for now: cabinet
  (mass), door, human and room keep their own planners.
- What is shared: candidate generation around the object, scoring with the existing `common/nbv/viewpoint_score.h`
  (detection-weighted expected information gain), reachability against the room polygon, obstacle rejection,
  robot-radius floor, and the proposal type.
- What each agent supplies: its candidate faces or view directions, and its belief's information term (each belief
  already has `predicted_information`).
- `common/epistemic_step` already shares the lifecycle AROUND the planner call. The planner itself is what G3 shares.
- **Wall-mounted objects (hood, mirror)** are handled by the same shared planner, not by a separate type (owner,
  11-10). They are seen only from the room side, usually from further away, and more face-on than a chair or table.
  Per the project's no-threshold rule this comes from the MODEL, not from a per-type minimum distance or an
  angle cutoff:
  - **room side only:** viewpoints behind the wall fall outside the room polygon and fail the existing reachability test;
  - **further away:** a high, large object fills the frame badly up close. The detector envelope (P(detect) from
    projected fill) and the camera's vertical field of view push the best stand-off back;
  - **face-on:** a grazing view foreshortens a flat face, so it yields little information and a low P(detect).
  If the model does not produce this behaviour, report why before adding any constant.

### G4. Conformance audit refreshed
In CONCEPT_AGENT_LIFECYCLE.md §6:
- chair's REMOVE cell says "❌ hand-rolled", but `chair_existence.h` now uses the shared `rc::exist` policy. Verify
  in code, then fix the cell.
- Add the missing **hood** row.
- Add the room-following and containment state per agent once G1 and G2 land.

## 3. Non-goals

- No new agents. The nightstand is Phase 2.
- No retina or vocabulary changes. Adding `toilet` and `bathtub` to `accepted_labels` is Phase 1.
- No change to how LTOM stores or recalls objects, or to `recall_match`.
- No change to the room-resolution rules themselves (`resolve_current_room` etc.); they are owned by the room/LTSM work.
- No planner unification for cabinet, door, human or room.

## 4. Anti-goals: failure even if it "works"

- **Any agent's rate drops.** Rates come before features: room_concept, controller and retina must stay at their
  nominal rates, and the agents touched here must keep theirs. Measure before and after.
- **Behaviour changes silently.** Every place where a unified planner proposes different viewpoints than the old
  copy, or where filing changes, is listed and explained. Merely absorbing a difference is a failure.
- **A new threshold, gate or clamp** introduced to make wall objects or containment behave.
- **Two sources of truth:** an agent keeps a private copy of a rule that also exists in `common/`.
- **Other sessions disturbed:** see the constraints in §5.

## 5. Constraints

- Work in a **separate git worktree**. Other sessions' tours write into the main checkout, and main carries other
  sessions' uncommitted edits.
- Follow CLAUDE.md:
  - DSR signals Queued only;
  - typed attribute access only;
  - `std::from_chars` for parsing;
  - `cbuild` / `make -j8`, after checking `pgrep -a cc1plus` for another build;
  - `and`/`or`/`not` instead of `&&`, `||`, `!`;
  - stop agents with SIGTERM, never `kill -9`.
- Webots is shared. Before any live tour, ask the room-concept session (`room-concept-fd`) and tell it when Webots is
  free. Do not reload the world without that coordination: a reload kills `shadow_bridge_dds`, which must then be
  restarted from the user's terminal.
- Commit or push only when the owner asks.

## 6. Validation

1. **Unit:** the shared helpers get tests next to `common/room_resolve/room_resolve_test.cpp` (pure functions, no graph).
2. **Planner A/B (offline where possible):** for each of the five agents, the same belief states in → old and new
   proposals out. Report target, yaw and gain differences, with every difference explained.
3. **Live tour in the three-room world** (coordinated as in §5):
   - each agent drops its objects on leaving a room and gets them back from LTOM on return;
   - a look through an open door creates nothing from the other room (the `chair_39` case);
   - per-agent rates are no worse than the baseline.

## 7. Settled after the builder check (the designer's calls, not the owner's, unless marked)

- **Sequencing:** the `chair_39` investigation gates only the G2 FIX. G1, G3 and G4 do not depend on it and may proceed
  in parallel.
- **Containment is the soft prior, not a hard test.** The shared rule is chair's existing containment prior:
  out-of-room births suppressed, out-of-room instances drawing `Existence.OutOfRoomGain` per frame. A hard
  centre-in-polygon test would be a new threshold (CLAUDE.md). Objects whose centre sits near the outline (wall-mounted,
  tall furniture) are the reason it must be soft. Folding the published layout's corner sigma into the prior is
  allowed if it falls out of the model; it is not required for Phase 0.
- **Planner baseline is each agent's OWN current planner,** not chair's. Every difference between old and new
  proposals gets classified:
  - deliberate per-object design → becomes that agent's hook or parameter;
  - accumulated patch → dropped, and the drop is listed.
  Regressions are allowed only with the owner's explicit approval, case by case.
- **Build and rollout:**
  - changing every touched agent's CMake to pull the new `common/` module is expected;
  - agents may migrate one at a time, each merge leaving the others untouched;
  - builds go through `cbuild` at -j8, one agent at a time if another build is running.
- **Cabinet:** it already resurrects from LTOM recall with a persisted uid and parents its nodes to `room_node_id_`
  (`cabinet_concept/src/specificworker_lifecycle.cpp:658`, recall at `:1123`), so drop-on-leave plus recall-on-return
  applies to it unchanged.
- **Definition of done:**
  - code is ready to merge after the unit tests and the offline planner A/B pass;
  - Phase 0 is DONE only after the live tour in §6 passes;
  - if Webots is busy, merging may go ahead (when the owner asks), but Phase 0 is not reported as done.
- **Hood's audit row** is filled by reading the hood agent's code, like every other row (§6 of the lifecycle doc
  says rows are "derived by reading each agent"), not copied from a neighbour.
- **Decided by the owner (11-11): WAIT** while the current room has NO published outline yet (right after a door
  crossing, a new proto-room). See §8.

## 8. No outline yet: decided (owner, 11-11)

**Decision: WAIT.** While the current room has no published outline (e.g. a new proto-room right after a door
crossing), an agent keeps tracking evidence but creates no new room objects until the outline exists. This is the
same pattern as the existing "no room known yet: wait" (`chair_concept/src/specificworker.cpp:389`). It is not a
threshold: the agent has no room fact yet. If the `chair_39` investigation shows another cause, the fix follows that
cause, but the wait rule stands.

**Known future exception (NOT built in Phase 0).** While a room's layout is being estimated, some tall objects
(cabinets, shelves) should be detected and handled in the ROBOT's frame, before the room is stable, so the estimator can
exclude their volume instead of fitting them as wall. That is a different role: an estimation aid in the robot or
odometry frame, handed into the room frame once the outline exists, and recognised by the same tracker rather than born
twice. It will be added later, per object type. Ties to the earlier note that tall furniture needs object models to be
told apart from walls.

**Constraint this puts on Phase 0:** the shared code takes "the frame this agent works in" as an INPUT (today always
the current room). It must not hard-wire "room or nothing", so the future exception does not require reworking Phase 0.

## 9. Left to the builder

- Names and file layout of the shared helpers in `common/`.
- The exact interface between the shared planner and each belief, provided each agent supplies only its candidates
  and its information term.
- The order of G1–G4, except that the `chair_39` investigation comes first.
- How the planner A/B is harnessed.
