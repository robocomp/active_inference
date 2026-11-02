# Plan: `imagine` — subjunctive "would X if Y" answers in chat_agent (v1)

## Context

The person wants to ask the robot questions about an imagined state of the world:
- "would I fit through door_1 if it were half open?"
- "would the mission succeed if you went through door_2 instead?"
- "would you see the fridge from there?"
- "do you fit between the wall and the chair?" (added 2026-10-31: fit generalised from doors to any gap)

Answering means **intervening on a copy of the world facts** and evaluating the robot's **own generative models**.
The user chose "existing models first": no physics engine in v1, PyBullet later for contact questions. Imagined
state is never written to the live graph. These four questions are the acceptance cases.

Each answer gives:
- a probability or verdict;
- the limiting factor;
- the assumption it made (for example the angle taken as "half open");
- an honest "no data / no model" when that applies.

After this plan, work returns to perception cards and the object-kind hierarchy and collectives
(`docs/superpowers/specs/2026-10-31-collective-objects-design.md`).

Work happens in the worktree `active_inference.worktrees/vlm-missions` (branch feat/vlm-missions). All CLAUDE.md
rules apply:
- C++23;
- and/or/not;
- from_chars;
- type-attributed DSR attributes;
- no thresholds where a model term works;
- `cbuild -j8`;
- SIGTERM only.

## Approach

**One tool, `imagine`, with flat arguments.**
- Arguments:
  - a required `question ∈ {fit, see, succeed}`;
  - optional strings `door`, `opening` (`shut|ajar|half|wide|as_is` or degrees), `via`, `target`, `from`, `to`.
- Its description is keyed on SUBJUNCTIVE: would / if / instead / what if.
- Reasons:
  - There are already about 25 tools, and three of them sit close to these questions (`route_to`, `outcome_stats`,
    `where_is`).
  - One routing decision plus an enum choice suits gemma4:26b better than three new tools or nested intervention
    lists.
- `imagine` is read-only, so it goes in `allowed_in_wake` (`chat_agent/src/mission_plan.h:458`).

**`ImaginedWorld`: a pure value type with no DSR include.**
- New file `chat_agent/src/imagined_world.h` (`rc::chat::imagine`). It cannot write to the graph by construction.
- Fields:
  - the room polygon;
  - the robot pose and its σ;
  - the body half-width;
  - an `rc::nbv::Sensor`;
  - doors (centre, normal, width, σ, phi, open_prob);
  - objects (subtype, pose, w, d, h, z span, σ);
  - a copy of `Topology`;
  - today's EM records.
- **Built lazily**, only when `imagine` is called, by `WorldDigest::snapshot(...)` on the main thread.
- **Body width** comes from the mesh footprint (`common/footprint/footprint.h` `RobotFootprint::from_obj`, as in
  `controller/src/controller_robot_body.cpp:84-127`). It is never hand-transcribed. If the mesh does not resolve,
  the answer says "no body model".
- **Sensor:** use `rc::nbv::sensor_from_graph` (`common/nbv/graph_obstacles.h:136`) if it compiles in chat_agent.
  Otherwise build it from the zed intrinsics plus the room→zed height.
- **Interventions** are pure functions that return a new world: `with_door_phi`, `with_robot_at(Standpoint)`,
  `with_forced_passage`.
- **Opening words** map to angles, and the answer always states the angle it assumed: shut 0, ajar π/12,
  half π/4, wide π/2 (the physical travel is `Openable.PhiMaxRad = π/2`), or explicit degrees.
- **Hinge and swing are not on the door node,** so the evaluator marginalises over 2 × 2 hypotheses at p = 0.25
  each. The mixture code stays, ready for a published hinge belief later.
- **"There" in v1:** the LLM must pass `from` as a name.
  - a door → the doorway centre;
  - an object → its circumscribed radius plus the body radius, on the line toward the robot;
  - an empty `from` or the current room → the robot's pose;
  - anything else → `ASK: which place do you mean by 'there'?`.
  - The referent-resolution project replaces this rule later.

**Q1, fit: the controller's own model.**
- P_fit = Σ_k p_k · `anchor_p_fit(gap_k.width, body_half, 0, σ_lat)`, with gap_k from `anchor_clear_gap`. This
  subtracts the leaf thickness.
- σ_lat comes from the door and robot RT covariances, projected onto the wall direction. If none is available,
  say "certain-geometry limit".
- Limiting factor:
  - the slack in cm;
  - the closed-form angle needed for P = 0.5, φ* = acos(1 − (2·body_half + 2·t)/w).
- Always state that it is an upper bound, because the leaf's depth across the doorway is not modelled.
- **Reuse:** move `anchor_std_normal_cdf`, `anchor_p_fit`, `AnchorGap`, `anchor_clear_gap` and
  `anchor_p_fit_leaves` **verbatim** from `controller/src/controller_anchor_servo.h` into a new
  `common/interaction_anchor/anchor_fit.h`. `controller_anchor_servo.h` then includes it, with no behaviour change.
- Report only this number, never door_concept's `passable_prob`, which uses a different 0.70 m model. Two numbers
  would confuse the person.

**Q1b, fit between two referents (added 2026-10-31, user).** This generalises Q1 from "through a door" to "through
the gap between A and B".
- **Arguments:** `imagine(question=fit, between=[A, B])`. Each of A and B is an object instance, a door jamb, or
  `wall`. `wall` means the room-polygon segment nearest the other referent.
- **Resolving each side:**
  - An object becomes its footprint: an oriented box from its pose and size, with its pose covariance. Only the
    part whose z-span overlaps the robot body's height counts; a table top above the body is ignored, a chair seat
    is not.
  - A wall becomes its polygon segment, with the room's pose σ.
- **Gap:** the minimum distance between the two footprints (box–segment or box–box). For convex footprints this is
  the narrowest point of the passage along its whole length.
- **σ_gap:** the referents' pose covariances and the robot's localisation, projected onto the direction of the
  minimum.
- **Model:** the same as Q1. P_fit = `anchor_p_fit(gap, body_half, 0, σ_gap)` from
  `common/interaction_anchor/anchor_fit.h` (Task 1's move), with the body width from the mesh footprint.
- **Limiting factor:**
  - the slack in cm;
  - which side is closer, so the answer can say what to move;
  - for an intervention, the displacement needed for P = 0.5.
- **Interventions:** `with_object_moved(w, name, dx, dy)` answers "would I fit if you moved the chair 20 cm?".
- **Caveat, always stated:** this is a prediction from beliefs. While driving, the controller's local policy
  decides from live LiDAR and may meet unmodelled things (legs, cables, clutter).
- **Follow-up (v2):** a controller dry-run query of its own body-fit check at that spot. This needs a small
  request/response channel through the graph.

**Q3, see: `common/nbv/viewpoint_score.h` (Eigen only).**
- ⚠ Superseded by the perception-cards work. Q3 must call `rc::card::evaluate` with the target's perception card
  (common/perception_card), using the same world snapshot as chat_agent's `can_see` (`perception_world.h`). It must
  not keep its own `[Imagine.Detect.*]` table. See the perception-cards spec §4 and §7b. The text below is kept for
  the obstacle and door handling only.
- Yaw = the bearing from the standpoint to the target, i.e. the best case (the robot can turn).
- Then `predicted_fill_axes` + `predicted_fill_sigma` + `visible_fraction` → `expected_p_detect`.
- Obstacles:
  - `wall_obstacles(polygon, t, gaps)`, with door gaps at the current or imagined φ, so Q1's intervention changes
    Q3's answer;
  - other objects as boxes.
- Limiting factor:
  - the smallest of the visibility term, too few pixels (fill_min) and overflowing the frame (fill_max);
  - the occluder is named by leave-one-out: the obstacle whose removal raises visibility the most.
- **Detector envelope:** an `[Imagine.Detect.<subtype>]` config table in chat_agent, with a comment citing the
  owning agent's `DetectMinFill/MaxFill`. The perception-card work later replaces this table.
- A target in another live room is converted through the RT chain, with that room's walls added. A target that is
  only remembered gets "no data".

**Q2, succeed.**
- `chat_agent/src/topology.h`:
  - refactor the Dijkstra into `route_cost(...)`, keeping `route()` as a wrapper;
  - add `route_via(t, from, to, passage)`, the cheaper orientation of route(from, a) + p + route(b, to);
  - add `resolve_passage(t, name)`.
- The destination comes from the current mission IR or plan card, else from `to`, else ASK.
- A `via` that cannot reach the destination gets an explicit "no remembered route through door_2 reaches room_2".
- **Per leg:** closed `cross` step records whose target is the leg's door, both directions pooled, with counts
  shown per direction.
  - **Success:** `satisfied`, or `preempted` with `room_after` ≠ `room`.
  - **Failure:** `door_wont_open`, `cross_unreachable`, `approach_unreachable`, `unreachable`.
  - **Excluded:** `stopped_by_user`, `superseded`, `lost`, and `preempted` without a room change.
- **Ruling (mine), to avoid double counting:**
  - LTSM `Beta(pass_alpha, pass_beta)` is the prior only when today's EM has n = 0 for that door. Otherwise, a flat
    Beta(1, 1) plus today's counts.
  - Reason: LTSM folds the same crossings at eviction.
  - Cost if wrong: answers are less informed early in the day. Revisit once `target_ltsm` is filled (see the
    follow-up below).
- **Mission P** = Π E[p_leg], with the sd from ΠE[X²]. It is always compared with the normal route: "≈0.81 via
  door_1 vs 0.70 via door_2".
- **Duration:** the per-leg mean of t1 − t0, with its n.
- **Wording:** no n cutoff; always say "s of n". With nothing to go on, say "I have no data on door_X".

## Tasks (TDD, one commit each)

1. **Move `common/interaction_anchor/anchor_fit.h` verbatim** and include it from `controller_anchor_servo.h`.
   - Before editing controller/, message the room session ("Guardes actions, ever shared facts", worktree
     rt-build). It may be editing controller files, so the move must be merge-safe.
   - Build the controller with `cbuild -j8` in our worktree, never in rt-build. Its anchor-servo tests must stay
     green.
2. **topology.h:** `route_cost` / `route_via` / `resolve_passage`, plus cases in `tests/test_topology.cpp`:
   forced passage, unreachable via, parallel passages.
3. **`imagined_world.h`:** types, interventions, opening angles, standpoint resolver. New
   `tests/test_imagine.cpp` (plain main + assert, registered in `tests/run_tests.sh` and CMake like test_topology).
4. **Q1, `answer_imagine_fit`.** Tests:
   - a 0.9 m door at π/4 gives a gap of about 0.26 m → P ≈ 0;
   - at π/2 → P high;
   - φ* round-trips;
   - a larger σ moves P toward 0.5.
4b. **Q1b, `answer_imagine_fit_between`.** Tests:
   - a chair 0.55 m from a wall, with a 0.62 m body → P ≈ low, and the answer names the chair as the side to move;
   - a 0.80 m gap → P high;
   - a table top above the body's z-band is ignored, so a gap under a high table counts as open;
   - `with_object_moved` 0.2 m → P rises, and the needed displacement round-trips;
   - box–box and box–segment minimum distances checked against hand-computed cases.
5. **Q3, `answer_imagine_see`.** Tests:
   - a box between standpoint and fridge is named as the occluder;
   - standing too close → "too close";
   - a target behind a wall, through a door gap, at φ = 0 vs π/2;
   - also the first unit test of `viewpoint_score.h` itself.
6. **Q2, `answer_imagine_succeed`.** Fixture: synthetic Records with the Oct 24 counts (door_1 15/3, door_2 7/1).
   Tests:
   - per-leg and mission P;
   - sd;
   - n = 0 wording;
   - exclusions;
   - the LTSM prior used only when n = 0.
7. **`world_digest.{h,cpp}`:** `DoorFact`, the object room pose, a lazy `snapshot()`. Extend `tests/test_objects.cpp`.
8. **`chat_session.cpp`:** `tool_def("imagine", …)` near :541, the dispatch near :1003, the config table. Add
   `imagine` to `allowed_in_wake` and update `test_mission_plan`.
9. **`tests/check_llm_tools.py`:** add the tool to `tools()`.
   - Positive cases: the four questions (incl. "do you fit between the wall and the chair?"); an unresolved "there" (ASK); a Spanish case ("¿cabría por door_1 si
     estuviera medio abierta?").
   - Negative cases:
     - "can you see the fridge" → not imagine;
     - "go through door_2" → plan_mission or cross_door;
     - "how often did door_1 fail" → outcome_stats.

## Verification

- **Unit tests:** `bash tests/run_tests.sh` in chat_agent, all green, including the new test_imagine and
  test_topology cases.
- **Build:** chat_agent and the controller with `cbuild -j8`, then the controller anchor-servo tests unchanged.
- **LLM eval on the Orin:**
  - `OLLAMA=http://192.168.1.34:11434 python3 tests/check_llm_tools.py gemma4:26b 3`.
  - The new cases pass, and the old 315 cases stay at 307/315 or better. That confirms the extra tool doesn't hurt
    the others.
- **Live check:** in Webots, with the person present, and after coordinating with the room session:
  - ask the three questions in the chat;
  - check that the answers state their assumptions;
  - check that the graph is unchanged: no new nodes or attributes from chat_agent.

## Follow-ups (not in v1)

- **episodic_agent:** fill `target_ltsm` on cross steps, and fold "preempted with a room change" as a success. That
  unlocks per-door history older than today, and lets the prior rule above be revisited.
- **door_concept:** publish hinge and swing beliefs on the door node (a new cortex attribute).
- **PyBullet** for contact and physics questions.
