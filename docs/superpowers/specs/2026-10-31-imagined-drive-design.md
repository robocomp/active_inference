# Imagined drive: the controller rolls out its own policy on a frozen copy of its world — design

Date: 2026-10-31. Status: APPROVED DECISIONS: hosted **in the controller**; mission pre-check automatic when idle; no dry runs
during a mission. The user answered both §10 questions on 2026-10-31.

This is the middle row of the imagination table: kinematic questions (time and motion), between static
geometry (perception cards, `imagine` fit/see) and contact physics (PyBullet, later).

## 1. Intent

Some questions can only be answered by driving:
- "would you get through between the chair and the wall while turning?";
- "could you reach the fridge without hitting the chair?";
- "would the mission succeed through door_2, and how long would it take?";
- "is that stand-off reachable from here?".

The robot should answer them by **running its own controller faster than real time, on a frozen copy of the
controller's own belief about the world**, with optional interventions. It should not use a second planner or a
second map. Your notes say this twice: "the bench must run the agent's code", and a harness that re-implements what
it tests ends up testing the re-implementation.

What it is for:
- `imagine` drive questions;
- pre-checking a mission before its card is shown;
- checking that a `where_to_stand` spot is reachable;
- trying gemma's proposed fix in imagination before offering it;
- in the long run, learning the transition model B by comparing imagined and real drives.

## 2. What exists

- **The live policy.** `rc::policy::Navigator` (`controller/src/local_policy/navigator.h`) is pure Eigen/std,
  with no DSR or Qt.
  - `set_goal(Goal, CostToGo::Grid, free_fn, extra_fn, goal_points, trap_radius)` and
    `step(PolicyInput, PolicyParams) → Step`.
  - `Step` carries the `Decision` mode (Drive/Release/Arrived/Stuck/Blind), plus `unreachable`, `hand_back` and the
    trap information.
- **The live field build.** `controller/src/controller_policy_drive.cpp:~291-318`. The controller builds `free_fn`
  and `extra_fn` as lambdas over its live state:
  - the GridPlanner distance (`dist_fn`), the inscribed radius;
  - door jambs and the leaf evidence (`policy_leaf_evidence_`);
  - the unseen-space density.

  It then builds `rc::policy::field_grid(...)` and calls `policy_nav_.set_goal(...)`. This is the code the dry run
  must reuse.
- **The simulator.** `controller/tests/sim2d/sim2d.h`, used by `tests/test_policy_scenarios.cpp`:
  - `World` (Box/Segment obstacles with `in_map`);
  - `scan`/`scan_mount` (1440 beams with noise, Shadow's two LiDAR mounts);
  - `ContactTester` (the true `RobotFootprint::shadow()` hull, sampled every 5 mm);
  - `Base` (diff-drive limits, 60 ms latency, deadband, slip and jitter);
  - `PoseError` (OU believed-pose error);
  - `integrate`.

  It has no CMake target today; each test is a one-line g++ build.
- **The graph request/response pattern.** chat_agent's `chat_intention` node, with the controller writing
  `state`/`bt_state` on the `has_intention` edge (`controller/src/chat_inbox.h`). Each object has one writer, and
  readers poll.

## 3. Architecture

### 3.1 The world snapshot: the same predicates, frozen

The core move is a refactor. **Extract the `free_fn`/`extra_fn` lambdas into functions of a copyable value,
`PolicyWorldSnapshot`.**
- It holds:
  - the GridPlanner distance raster (or the ESDF) as a dense copy;
  - the inscribed and circumscribed radii;
  - the jamb list;
  - the leaf-evidence state;
  - the unseen-density raster;
  - the room polygon;
  - the tracked obstacles (boxes) with their σ;
  - the current robot pose and its σ.
- The live controller then builds its lambdas FROM a snapshot of itself: `snapshot.free(p)`, `snapshot.extra(p)`.
  The live path and the imagined path evaluate the same code by construction. This is the same guarantee as the
  perception cards' moved terms.
- **Taking a snapshot** is a main-thread copy. Its cost is bounded by the raster size and must be measured; the
  budget is under 5 ms. It is taken only when a dry run is requested.

### 3.2 The imagined world for the simulator

`sim2d::World` needs geometry for ray-casting and contact. It is built **from the snapshot**:
- the occupied raster cells merged into boxes (row-run merge);
- the polygon edges as Segments;
- the tracked obstacles as Boxes.

**In imagination the belief IS the world.** There is no hidden "truth". The one exception is deliberate stale-map
draws (`in_map = false`), described in §3.4.

**Interventions are edits to the copy:**
- `move_object(name, dx, dy)`;
- `set_door(name, phi)`, which also updates the jamb and leaf state the predicates read;
- `add_box(...)`;
- `remove_object(name)`.

### 3.3 The runner

A new library target, `controller/src/imagination/`, pure, with no DSR and no Qt.
- `Outcome rollout(const PolicyWorldSnapshot&, const sim2d::World&, Pose2 start, rc::policy::Goal, const RolloutParams&, std::uint64_t seed)`
- It uses `sim2d::Base`, `scan_mount` for helios and bpearl, `ContactTester`, `PoseError` and `Navigator`.
- `step()` runs at the live control period until one of: Arrived, Stuck, Blind, `unreachable`, `hand_back`, a
  timeout, or the step budget.
- `Outcome` carries:
  - the termination reason, the time, the path (decimated);
  - min clearance (the footprint hull to the nearest obstacle), contact count, stuck windows;
  - where it stopped, with the policy's own `why`.
- `sim2d.h` moves from tests/ into this target unchanged, and the tests include it from there. The two policy test
  suites then share one simulator with the dry run.

### 3.4 Uncertainty: many seeds

`Summary rollout_n(..., int n, std::uint64_t seed)` varies, per seed:
- the start-pose error from the live pose σ (`PoseError`);
- wheel slip and jitter (`Base`), using the controller's own identified values;
- LiDAR noise;
- **door states drawn from their beliefs** (P(open), φ). This is the same mixture idea as can_see's R10;
- optionally, stale-map obstacles: tracked obstacles moved within their σ.

The `Summary` gives:
- P(reach) with a Wilson interval;
- the median time and its 90% range;
- the modal stop reason, and where it happens (a cluster of stop poses);
- min clearance quantiles.

The default is n = 16, under a time budget.

### 3.5 Hosting in the controller (the user's choice)

- **The worker.** A dedicated `std::thread` with low priority (nice +10, or SCHED_IDLE), owned by
  ControllerSession. It is started on demand and joined in the destructor, which runs on SIGTERM.
- **No graph access on the worker.** It sees only the snapshot and the copied world, so CLAUDE.md's thread rules
  are respected by construction.
- **Budget.** Each request carries `budget_ms`, default 2000. The worker checks it between steps and returns a
  partial Summary with `truncated = true`.
- **The live loop always wins.**
  - The worker never takes a lock that compute() needs.
  - If the robot is executing a mission, a dry run is NOT run: it is refused as busy (the user's decision,
    2026-10-31, §10).
  - If CPU pressure is measured (compute() overrunning its period), the dry run is paused.
- **One request at a time.** A new request supersedes the pending one.

### 3.6 Request and response through the graph

- **Request.** chat_agent owns a node `imagine_request` (type intention, the chat_inbox pattern) with:
  - `current_intention`: a JSON `{id, start: "here" | pose, goal: {kind: point|object|door|room, …}, interventions: […], n, seed, budget_ms}`;
  - `timestamp_agent`.
- **Response.** The controller writes the result on the edge `Shadow --has_intention--> imagine_request`:
  - `state = "running" | "done" | "failed|<reason>"`;
  - `bt_state` = the Summary as compact JSON.
  - The controller is the only writer of the edge, and chat is the only writer of the node.
- **New attributes.** None. It reuses the `current_intention`, `state` and `bt_state` attributes the chat inbox
  already uses. This must be checked against the installed cortex header.
- **The goal resolution.** "chair_1" resolves to its approach points, a door to its crossing anchor, a room to its
  centroid. It reuses the controller's own GoalSpec conversion on the main thread at snapshot time, i.e. the
  goal_adapter, which is DSR-bound and therefore cannot run on the worker.

## 4. Consumers

- **`imagine(question = drive | succeed)`** in chat_agent writes the request, polls the edge, and words the
  Summary. For example: "Probably yes (14 of 16 imagined runs got there, about 25 s). The two that failed got stuck
  at the chair near door_2." Its `succeed` answer combines this, "can I physically drive it?", with the EM per-door
  history, "did the door open in the past?".
- **The mission pre-check (AUTOMATIC, the user's decision on 2026-10-31).**
  - After `plan_mission` produces a mission and before the card is shown, each motion leg is rolled out, when the
    robot is idle.
  - The card gets a line per risky leg. It is advisory only: the person still confirms.
  - While a mission runs, the card says "not pre-checked: the robot is busy".
- **`where_to_stand`:** reachability of the top stands by one rollout each, with a cheap n.
- **Failure handling:** a proposed fix is rolled out before gemma offers it.

## 5. Learning B (later phase)

- When a real leg finishes, the controller replays it as a rollout from the recorded start, with the snapshot taken
  at the leg's start.
- EM records the imagined time and stop reason next to the real ones.
- The systematic gaps (slip, latency, the noise models, stale-map rates) are the calibration signal: B is learned
  from experience, as `card_vs_probe` does for A.

## 6. Phases

1. **Library.** Move sim2d into `controller/src/imagination/`. Add `rollout`/`rollout_n` and a CMake target, with
   tests on hand-built worlds: reach, stuck behind a gap narrower than the body, time within 10% of the live
   scenario test, determinism per seed.
2. **Snapshot refactor.** `PolicyWorldSnapshot` with the extracted predicates. The live controller builds its
   lambdas from it, and the policy scenario tests plus a replay prove behaviour is unchanged.
3. **Hosting.** The worker thread, budget and priority. A test measures that compute()'s period is unaffected under
   a running dry run.
4. **Graph channel.** The request/response channel, plus the chat side `imagine(drive)`.
5. **Consumers:** the mission pre-check and where_to_stand reachability.
6. **Calibration** against EM.

## 7. Goals, non-goals, anti-goals

**Goals:**
- Dry runs that use the controller's own policy, predicates and field on a frozen copy of its belief.
- P(reach), time and stop reasons, with uncertainty.
- No effect on live control.

**Non-goals:**
- Contact dynamics: pushing, opening doors by force (PyBullet, later).
- Moving people.
- Manipulation.
- Replanning the global route. v1 rolls out the local policy toward the goal the controller would itself pursue.

**Anti-goals:**
- A second field builder, or a second copy of the predicates.
- Rollouts that read the graph from the worker.
- A dry run that delays or perturbs a real drive.
- Presenting one deterministic run as the answer.

## 8. Constraints

- CLAUDE.md rules:
  - C++23 with and/or/not;
  - from_chars;
  - type-attributed DSR attributes;
  - never DirectConnection;
  - graph access only on the main thread, with worker threads on copies;
  - SIGTERM shutdown joins the worker;
  - `cbuild -j8`.
- **Coordination.** The room session works on controller files on `feat/room-recognition` (worktree rt-build) and
  runs live tours with that binary. This work needs its own branch from main, a message to that session before the
  phase-2 refactor of `controller_policy_drive.cpp`, and merge coordination. That file is the overlap.
- Use the base limits and slip values the controller actually identified (the `BaseLimits` in local_policy, and
  `tracker_sim`'s identified plant), not new numbers.

## 9. Decided vs my calls vs left to the builder

**Decided with the user:**
- The kinematic rollout is the important middle layer.
- It uses the controller's own policy.
- It is hosted in the controller.

**My calls:**
- The snapshot refactor (one predicate implementation for live and imagined).
- The belief-is-the-world rule, with stale-map draws as an explicit option.
- The graph channel reuses the chat_inbox pattern and attributes.
- The defaults: n = 16, a 2 s budget, low-priority worker, one request at a time.
- The mission pre-check is advisory.

**Left to the builder:**
- The raster-to-box merge, path decimation, the stop-pose clustering, and the exact JSON fields.

## 9b. Builder-check rulings (my calls; please check)

The live predicates capture more than a raster. The rulings below keep "the same code by construction" honest.

1. **A frozen planner.**
   - The snapshot holds a value copy of the whole `GridPlanner`, after forcing `build_distance_field()` on the main
     thread. That copy includes occ_/live_only_/risk_/dist_ and the per-heading footprint `offsets_`.
   - `body_risk_at`, `distance_at` and `cell_size` then run the identical code on the copy.
   - The worker NEVER touches the live planner, because the `mutable dist_` is not thread-safe.
   - The copy size and time are measured in phase 2. If they exceed 5 ms, the budget is revised; nothing is
     precomputed into a different representation.
2. **Every captured input is frozen,** including the ones §3.1 missed:
   - `DoorLeafEvidence`, a value;
   - the per-tick `DoorwayRisk` jambs and `policy_held_jambs_`;
   - the unseen density;
   - the inputs behind `also_free` (the Task 3c standpoint) and `trap_veto` (the held doorway).
   The live lambdas are rebuilt FROM the snapshot by one shared function, `make_policy_predicates(const PolicyWorldSnapshot&)`.
   The live controller uses it every tick and imagination uses it once. Dropping any of these inputs is forbidden.
3. **The extraction boundary.**
   - The `PolicyInput` assembly in `policy_tick` (the ~400–600 region: odometry, sweeps, latency, sigmas) is
     factored into a pure `assemble_policy_input(...)` used by both the live tick and the rollout.
   - The rollout feeds it from `sim2d`'s Base, scans and PoseError instead of from DSR.
   - The live tick's DSR reads stay in the session, and only the assembly moves.
4. **The Navigator.**
   - For `start = here` toward the goal the controller is CURRENTLY pursuing, the rollout copies the live Navigator
     by value: plan, monitor, traps, latch, sign safety, UnseenContact and tick gate. Its "imagined continuation"
     then matches the live state.
   - For any other goal it starts a fresh Navigator plus `set_goal`.
   - The Navigator must be copyable. If any member blocks that, phase 1 makes it so.
5. **Logging.** The Navigator gains an optional log sink. Imagination passes a quiet sink, so a dry run never
   prints into the live log.
6. **Belief versus surprise.** A pure "belief is the world" rollout cannot meet anything the policy does not
   expect, so it is optimistic. v1 includes, per seed:
   - tracked obstacles jittered within their σ;
   - unseen-density cells that become `in_map = false` obstacles with probability equal to their density.
   These are the policy's own uncertainty terms, sampled. The remaining optimism (truly unmodelled things) is
   stated in every answer.
7. **Door interventions are out of v1.**
   - `set_door` would need a DoorwayRisk/DoorLeafEvidence builder from a door state, and no DSR-free one exists. v1
     interventions are objects only: move, add, remove.
   - Doors stay at their current belief.
   - Varying the door state per seed, and `set_door`, come in a later phase together with that builder.
8. **Cost and n.**
   - Phase 1 measures the cost per step: scan, two mounts, ContactTester, step.
   - The target is ≥ 10× real time.
   - `rollout_n` runs as many seeds as fit `budget_ms` (minimum 4, maximum 16) and reports the n it actually used.
   - The field is built once per snapshot, not per seed. Door variation, deferred by ruling 7, would add one field
     per distinct door state.
9. **Goals in v1:**
   - `GoalSpec` points;
   - an object approach, as the controller's own approach standpoints.
   - Doorway goals come with ruling 7.
   - If the start pose lies outside free space in the snapshot (field = +inf), the Outcome is `start_infeasible`
     with a reason. It is never a NaN or a hang.
10. **Graph channel.**
    - A second chat-owned node `imagine_request`, with its own `Shadow --has_intention--> imagine_request` edge.
    - The controller runs a second `ChatInbox`-style mirror keyed by that node name, so the mission inbox's
      dedup/stamp logic is untouched.
    - The Summary JSON is kept compact (under 2 kB). The attribute is a string; check how its length is handled.
11. **Threads.**
    - A raw `std::thread` (no Qt) owned by ControllerSession, joined in the destructor.
    - Phase 1 audits sim2d.h and the Navigator for statics, `getenv` and global RNG state. Every RNG is seeded per
      rollout.
12. **Phase-1 acceptance.**
    - For 3 reference scenarios from `test_policy_scenarios.cpp` (doorway, corridor, clutter), on the same world and
      the same seed, the library runner and the scenario harness give IDENTICAL trajectories, because they run the
      same code.
    - Determinism per seed is required, including across the worker thread.

## 10. Open questions for the user

1. ~~Should the mission pre-check run automatically?~~ **Answered by the user on 2026-10-31: AUTOMATIC, with
   skip-if-busy.**
   - Every planned mission is pre-checked before its card is shown, when the robot is idle. That adds up to the
     ~2 s budget.
   - If a mission is running, the card shows "not pre-checked: the robot is busy" instead, consistent with
     answer 2.
2. ~~Should a dry run be allowed while the robot is executing a mission?~~ **Answered by the user on 2026-10-31:
   NO, not while a mission is running.**
   - A request arriving during a mission is refused, with `state = "failed|busy: a mission is running"`, and the
     consumer says so in its answer.
   - This replaces the "lower priority during a mission" sentence in §3.5.
