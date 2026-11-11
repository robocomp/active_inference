# Countable Agents Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before any new bathroom/bedroom agent is cloned, give every countable concept agent the same shared
room-following, the same room-outline containment (with WAIT while no outline exists), and one shared epistemic
planner.

**Architecture:**
- **Room code:** pure rules go in `common/room_resolve/`, unit-tested standalone. A thin graph-side helper does the
  DSR work. Each agent keeps only "how to forget my instance X".
- **Planner:** a header-only `common/nbv/countable_planner.h` wraps the existing `rc::nbv::plan_faces`. Each agent
  supplies its target, its per-face raw gain and its adequacy bound.
- **Rollout:** one agent per task, each verified alone.

**Tech Stack:** C++23, Eigen, DSR (cortex), header-only `ai_common::core` (new `common/` headers need NO CMake
change), standalone tests run by `common/run_tests.sh`, and agent tests in `<agent>/tests/CMakeLists.txt`.

**Spec:** `docs/superpowers/specs/2026-11-10-countable-agents-phase0-design.md` (commit 12a60515). Read §2, §7 and §8 before any task.

## Global Constraints

- Work in a separate git worktree off `main` (`superpowers:using-git-worktrees`). Never in the main checkout.
- Code rules (CLAUDE.md):
  - DSR signals Queued, never `Qt::DirectConnection`;
  - type-attributed graph access only (`get_attrib_by_name<foo_att>`), never `runtime_checked_*`;
  - `std::from_chars` for any parsing;
  - `and`/`or`/`not`, not `&&`, `||`, `!`, in new or touched code;
  - C++23 containers and algorithms.
- Build with `cbuild` (= `cmake -B build` + `make -C build -j8`, memory-capped). Run `pgrep -a cc1plus` first; if
  another build is running, wait. One agent build at a time.
- Stop agents with SIGTERM or Ctrl-C, NEVER `kill -9`.
- **No new thresholds, gates, clamps or σ-floors** (spec §4, CLAUDE.md). Physical dimensions (an object's
  half-depth, the robot radius) are not thresholds.
- **Rates come before features:**
  - every touched agent's `Fps` must be no worse than its Task 1 baseline;
  - room_concept, controller and retina must stay at their nominal rates (≈20, ≈19 and ≈40 Hz).
- **Webots is shared:**
  - before any live run, message `room-concept-fd` and wait for "free";
  - when done, message it that Webots is free;
  - never reload the world without that coordination (a reload kills `shadow_bridge_dds`, which the user restarts).
- Commit locally per task in the worktree. Push or merge to `main` ONLY when the owner asks.
- Every behaviour difference gets listed and explained in the task's commit message and in
  `docs/superpowers/plans/2026-11-11-countable-agents-phase0-ledger.md` (created in Task 1). Absorbing one silently
  is a failure.

## Review Focus

1. **Wall-mounted objects at birth.** A hood or mirror centre sits a few cm inside the outline, by its own half-depth.
   It must be ADMITTED, not suppressed as "outside". Expected: born normally. Pinned in Task 3
   (`birth_wall_object_admitted`).
2. **Room node recreated with a new id** (room_concept recreates the room after relocalisation). Every object then
   has a `parent` that names a dead room. Expected: treated as left behind (dropped, then recalled by LTOM), never
   fitted in a dead frame and never crashing on a missing node. Pinned in Task 3 (`left_behind_dead_parent`).
3. **Degenerate outline** (fewer than 3 vertices, or x/y lengths that differ). Expected: treated as "no outline yet",
   so WAIT, not admit. Pinned in Task 3 (`birth_defers_on_degenerate_outline`).
4. **A deferred candidate must survive the wait.** The tracker's tentative track must keep its evidence while births
   are deferred, so that it is born, or suppressed, the cycle the outline arrives, not lost. Pinned in Task 4
   (`chair_tracker_defer_keeps_candidate`).
5. **Proto-room promoted to a real room.** If promotion changes the room id, the objects born in the proto are
   dropped and recalled. Expected: they come back with the same uid. Checked live in Task 17.

---

### Task 1: Worktree, baseline rates, ledger

**Files:**
- Create: `docs/superpowers/plans/2026-11-11-countable-agents-phase0-ledger.md`

- [ ] **Step 1:** Create the worktree `../active_inference.worktrees/phase0` on branch `feat/countable-phase0` from `main`.
- [ ] **Step 2:** Build the six agents (chair, table, bottle, hood, refrigerator, cabinet) unmodified in the worktree,
  one at a time with `cbuild`. Expected: all build.
- [ ] **Step 3:** Coordinate Webots (Global Constraints). Run a parked plus one-door-crossing session in the
  three-room world with the six agents up. Record each agent's `Fps` (its "Epoch time = …ms. Fps = N" line), plus
  room_concept, controller and retina rates, as the BASELINE table in the ledger. Release Webots.
- [ ] **Step 4:** Commit the ledger.

### Task 2: chair_39 investigation (first; gates only the G2 fix)

**Files:**
- Modify: the ledger (findings section)

- [ ] **Step 1:** Locate chair_39's birth.
  - Search the tour logs (`room_concept/estimation_scripts/crossing/*/logs/chair_concept.log`) and the LTOM store for
    `chair_39`.
  - Record: birth time, the room it was parented to, its birth position, whether a "containment prior INACTIVE" or
    "no room polygon yet" line preceded it, and the door-crossing time.
- [ ] **Step 2:** Classify the cause in the ledger as one of:
  - (a) born while the current room had no outline (the spec's hypothesis);
  - (b) the outline contained the position;
  - (c) other, described.
  If the logs cannot settle it, reproduce live: park in room_2 facing the open DOOR_0 toward room_1's chairs, right
  after a crossing. Coordinate Webots first.
- [ ] **Step 3:** If the cause is (b) or (c), STOP and report to the owner before Task 3. The spec's G2 fix assumes (a).
- [ ] **Step 4:** Commit the ledger.

### Task 3: Shared room rules (pure) + graph helper

**Files:**
- Create: `common/room_resolve/room_containment.h` (pure; Eigen only)
- Create: `common/room_resolve/room_containment_test.cpp`
- Create: `common/room_resolve/room_follow.h` (graph side; DSR)
- Modify: `common/room_resolve/room_resolve_rule.h` only if `left_behind` needs the dead-parent case (see the test)

**Interfaces:**
- Produces (pure, `room_containment.h`, namespace `rc::room`):
  - `enum class BirthVerdict { Admit, Suppress, Defer };`
  - `bool outline_usable(std::span<const Eigen::Vector2f> outline)` — true iff size ≥ 3.
  - `BirthVerdict birth_verdict(std::span<const Eigen::Vector2f> outline, const Eigen::Vector2f& centre, float tolerance_m)`:
    - `Defer` if not `outline_usable`;
    - else `Admit` if `rc::nbv::point_in_polygon(centre, outline)` or
      `rc::nbv::distance_to_boundary(centre, outline) <= tolerance_m`;
    - else `Suppress`.
    - `tolerance_m` is the agent's PHYSICAL footprint half-depth: chair passes its existing
      `cfg_.exist_room_margin_m` unchanged (today's `point_in_room` margin, `specificworker.cpp:809`), so chair's
      Admit/Suppress decisions stay identical wherever an outline exists; a wall-mounted object passes its half-depth.
  - `float out_of_room_logodds(bool inside, float gain)` — returns `inside ? 0.f : -gain` (the existence-channel term chair uses today).
- Produces (graph, `room_follow.h`, namespace `rc::room`):
  - `std::vector<Eigen::Vector2f> read_outline(DSR::DSRGraph& G, std::uint64_t room_id)` — reads
    `delimiting_polygon_x/y` (typed access); returns `{}` when the node or the attrs are absent, or their lengths differ.
  - `template <class Forget> int drop_left_behind(DSR::DSRGraph& G, std::string_view name_prefix, std::uint64_t working_frame, Forget&& forget)`:
    - for every `object` node whose name starts with `name_prefix` and for which `left_behind(parent, working_frame)` holds,
      call `forget(node_id)` (the agent tears down its own affordance, approach and fitter state), then `G.delete_node(id)`;
    - log `"<prefix>: [room] '<name>' left behind in another room, dropped"`;
    - return the count.
    - `working_frame` is an INPUT (spec §8 constraint): the helper never resolves the room itself.

- [ ] **Step 1: Write the failing tests** in `room_containment_test.cpp`, in the style of `room_resolve_test.cpp`:
  - `birth_admits_inside`: unit square outline, centre (0.5,0.5), tol 0 → `Admit`
  - `birth_suppresses_outside`: centre (2,0.5), tol 0.1 → `Suppress`
  - `birth_wall_object_admitted`: centre (0.5, 1.03) just outside the top edge, tol 0.05 → `Admit`
  - `birth_defers_without_outline`: empty outline → `Defer`
  - `birth_defers_on_degenerate_outline`: 2-vertex outline → `Defer`
  - `left_behind_dead_parent`: `left_behind(std::optional<uint64_t>{77}, 5)` → true (77 = a room that no longer
    exists; the rule does not consult the graph)
  - `out_of_room_term`: `out_of_room_logodds(true, 1.5f) == 0.f`, `out_of_room_logodds(false, 1.5f) == -1.5f`
- [ ] **Step 2:** Run `common/run_tests.sh`. Expected: BUILD FAILED for room_containment_test (header missing).
- [ ] **Step 3:** Implement `room_containment.h` and `room_follow.h` as specified.
- [ ] **Step 4:** Run `common/run_tests.sh`. Expected: "common: all test binaries passed".
- [ ] **Step 5:** Commit `feat(common): shared room containment + left-behind drop helper`.

### Task 4: chair on the shared room code (WAIT added)

**Files:**
- Modify: `chair_concept/src/specificworker.cpp` (line 388 room follow; lines 410–416 per-node loop; `drop_left_behind`
  at 1057 becomes the `forget` lambda; birth suppression at 806–815; existence room term in `chair_existence.cpp:102`)
- Modify: `chair_concept/src/specificworker.h` (remove `drop_left_behind` declaration)
- Test: `chair_concept/tests/chair_tracker_defer_test.cpp` (+ CMake entry)

**Interfaces:**
- Consumes: Task 3 (`birth_verdict`, `drop_left_behind`, `read_outline`, `out_of_room_logodds`)

- [ ] **Step 1:** Write the failing test `chair_tracker_defer_keeps_candidate`:
  - drive `rc::InstanceTracker` (as chair configures it) with the same detection for `birth_frames + 2` frames while
    births are deferred;
  - assert no birth is emitted;
  - then allow births and assert the birth is emitted on the NEXT frame, at the tracked position.
  If the tracker has no "defer" notion, the test pins the mechanism Step 3 adds to the agent (tentative tracks kept,
  birth withheld).
- [ ] **Step 2:** Build and run it. Expected: FAIL.
- [ ] **Step 3:** Implement:
  - the per-cycle loop calls `rc::room::drop_left_behind(*G, "chair", room_node_id_, forget)`;
  - the birth site calls `birth_verdict(fitter_->room_polygon(), c, cfg_.exist_room_margin_m)`: `Defer` → withhold
    and log `SUPPRESS … "defer: no outline"` once per episode; `Suppress` → as today; `Admit` → as today;
  - the existence term uses `out_of_room_logodds`;
  - `refresh_room_geometry` uses `read_outline`.
- [ ] **Step 4:** Run chair's tests and `cbuild` chair. Expected: PASS, build OK.
- [ ] **Step 5:** Ledger: behaviour difference = "births deferred while no outline (was: admitted)". Commit.

### Tasks 5–9: table, bottle, hood, refrigerator, cabinet on the shared room code

One task per agent, in this order: **table (5), bottle (6), hood (7), refrigerator (8), cabinet (9)**. Same steps for each.

**Files (per agent `<a>`):**
- Modify: `<a>_concept/src/specificworker.cpp`:
  - the latch at the first `room_node_id_ = *room` (table :278, bottle :257, hood :278, refrigerator :270, cabinet :255);
  - the re-resolve at the second (table :462, bottle :402, hood :432, refrigerator :410, cabinet :376);
  - the per-node loop in `compute()`.
- Modify: the birth site, just before `fitter_->note_birth` (table `specificworker_lifecycle.cpp:305`, bottle
  `specificworker.cpp:805`, hood `specificworker_lifecycle.cpp:429`, refrigerator `specificworker_lifecycle.cpp:428`,
  cabinet: its `create_instance_from_detection` path).
- Modify: the agent's existence update, where it has an `rc::exist` channel (not bottle: its REMOVE is tracker death only).

**Interfaces:** Consumes Task 3.

- [ ] **Step 1:** Replace the latch with `room_node_id_ = rc::room::follow_current_room(room_node_id_, rc::room::current_room(*G)); if (room_node_id_ == 0) return;`
  at the top of `compute()`, as chair does at :388.
- [ ] **Step 2:** Add `rc::room::drop_left_behind(*G, "<a>", room_node_id_, forget)`. The `forget` lambda removes that
  instance's affordance and approach and calls the fitter's forget; name the existing per-instance teardown the agent
  uses on tracker death.
- [ ] **Step 3:** Birth site: `birth_verdict(outline, c, tolerance)`.
  - `tolerance` is the object's physical half-depth: hood = its belief's depth/2; refrigerator, table, cabinet = their
    existing room margin if they have one, else half their footprint's short side.
  - Bottle has no outline in its fitter: read it with `read_outline(*G, room_node_id_)` once per cycle.
- [ ] **Step 4:** Existence: add `out_of_room_logodds(inside, cfg.OutOfRoomGain)`. Add the config key
  `Existence.OutOfRoomGain = 1.5`, chair's value, to `<a>_concept/etc/config.toml`. Skip for bottle and record why in the ledger.
- [ ] **Step 5:** Run the agent's tests (if any) and `cbuild`. Expected: all pass, build OK.
- [ ] **Step 6:** Ledger: list the behaviour differences:
  - room now follows the robot (was latched);
  - out-of-room births suppressed (was: none);
  - births deferred without an outline.
  Commit.

### Task 10: Shared countable planner core

**Files:**
- Create: `common/nbv/countable_planner.h`
- Create: `common/nbv/countable_planner_test.cpp`

**Interfaces:**
- Consumes: `rc::nbv::{Target, Sensor, Obstacle, Plan, plan_faces}` (`common/nbv/viewpoint_score.h`),
  `rc::any_sigma_star`, `rc::adequacy_gap_nats` (`common/ai_belief/dof_spec.h`).
- Produces (namespace `rc::nbv`):
  - `struct CountableProposal { float epistemic_target_x_m = 0, epistemic_target_y_m = 0, epistemic_target_yaw_rad = 0, epistemic_gain = 0; bool valid = false; bool is_finite() const; };`
    The field names are IDENTICAL to today's per-agent `EpistemicProposal`, so `common/epistemic_step` keeps compiling.
  - `struct CountableInputs { Sensor sensor; float robot_radius_m; std::span<const Obstacle> obstacles; std::span<const Eigen::Vector2f> room_polygon; };`
  - `struct CountableResult { CountableProposal proposal; Plan plan; bool refused_no_usable_face = false; };`
    `plan` is returned so agents keep their diagnostic logs.
  - `template <int D> float d_optimal_gain(const Eigen::Matrix<float, D, D>& S, const Eigen::Matrix<float, D, D>& dI)`
    — `0.5f * std::log(std::max(1e-9f, (I + S * dI).determinant()))`, the existing formula verbatim.
  - `template <std::size_t N, class VarFn> float useful_fraction(const std::array<DofSpec, N>& dofs, VarFn&& var, float best_raw)`
    — 1 when `not any_sigma_star(dofs)`; otherwise `best_raw > 1e-9f ? std::min(best_raw, adequacy_gap_nats(dofs, var)) / best_raw : 0`.
    This is chair's existing guard, verbatim.
  - `template <class RawGainFn, class UsefulFn> CountableResult plan_countable(const Target& t, const CountableInputs& in, RawGainFn&& raw_gain_of, UsefulFn&& useful)`:
    - calls `plan_faces(t, in.sensor, in.robot_radius_m, in.obstacles, raw_gain_of, in.room_polygon)`;
    - invalid plan → empty proposal;
    - `not plan.any_usable` → empty proposal with `refused_no_usable_face = true`;
    - else `gain = plan.face_gains[best] * useful(plan.face_raw_gains[best])`, proposal = (`best_pos`, `best_yaw`, `gain`, valid);
    - a non-finite result → empty proposal.
    - `raw_gain_of(int face, float standoff) -> float` is the agent's.

- [ ] **Step 1:** Write the failing tests, each with a synthetic `Target` (1×1 m box at the origin), a square room
  polygon and a constant raw gain:
  - `refuses_when_every_face_outside_room`: polygon excludes all viewpoints → `refused_no_usable_face`, `not valid`
  - `wall_object_viewed_from_room_side`:
    - the box touches the polygon's top edge, so 3 faces are unreachable;
    - the valid proposal's position is inside the polygon, on the open side.
  - `useful_fraction_unbounded_without_sigma_star`: empty-demand dofs → 1
  - `d_optimal_matches_formula`: a 2×2 case checked by hand
  - `non_finite_gain_refused`: a raw gain returning NaN → `not valid`
- [ ] **Step 2:** Run `common/run_tests.sh`. Expected: FAIL (header missing).
- [ ] **Step 3:** Implement `countable_planner.h`.
- [ ] **Step 4:** Run `common/run_tests.sh`. Expected: all pass.
- [ ] **Step 5:** Commit `feat(common/nbv): shared countable planner core`.

### Tasks 11–15: chair, table, bottle, hood, refrigerator planners on the shared core

One task per agent, in this order: **chair (11), table (12), refrigerator (13), hood (14), bottle (15)**. Bottle goes
last: its signature differs (`camera_xy`, no `lat_rate`).

**Files (per agent `<a>`):**
- Modify: `<a>_concept/src/epistemic_planner.{h,cpp}`:
  - `EpistemicProposal` becomes `using EpistemicProposal = rc::nbv::CountableProposal;`;
  - `compute()` keeps its public signature and builds the `Target`, the `raw_gain_of` lambda (which keeps the agent's
    own terms, e.g. chair's discrete mode-entropy × view-novelty) and `useful`, then calls `plan_countable`.
- Create: `<a>_concept/tests/planner_ab_test.cpp` and `<a>_concept/tests/legacy_epistemic_planner.{h,cpp}` (a
  verbatim copy of the pre-task planner in namespace `rc::legacy`), plus the CMake entries.

**Interfaces:** Consumes Task 10.

- [ ] **Step 1:** Copy the current planner to `tests/legacy_epistemic_planner.*` (namespace `rc::legacy`). Write
  `planner_ab_test.cpp`:
  - build ≥ 20 belief states through the agent's own belief API: fresh prior; after 1, 5 and 30 updates from varied
    bearings; mode-ambiguous where the agent has modes; near a wall of the room polygon; one face blocked by an obstacle;
  - for each state, call `legacy::compute` and the new `compute` with identical inputs;
  - PASS when |Δx|, |Δy| < 1e-4 m, |Δyaw| < 1e-4 rad, |Δgain| ≤ 1e-5·|gain| and validity is equal, OR the state's
    index is in the `expected_differences` table with a reason string;
  - an unlisted difference FAILS and prints both proposals.
- [ ] **Step 2:** Build and run the A/B test against the UNCHANGED planner. Expected: PASS (identity), which checks the harness.
- [ ] **Step 3:** Move `compute()` onto `plan_countable`.
- [ ] **Step 4:** Run the A/B test. For each difference, classify it:
  - deliberate per-object behaviour → restore it as that agent's hook or parameter until the test passes;
  - an accumulated patch → add it to `expected_differences` with the reason.
  Expected: PASS.
- [ ] **Step 5:** Hood (Task 14) only: assert in the A/B test that every valid proposal lies inside the room polygon,
  on the room side of the wall the hood is on.
  - Report in the ledger whether the new planner stands further back and more face-on than for a chair-sized target.
  - If the model does not produce that, record why and STOP for the owner (spec G3): no constant may be added.
- [ ] **Step 6:** `cbuild` the agent. Ledger: the differences and their classification. Commit.

### Task 16: Conformance audit refresh

**Files:**
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` §6 table (lines 217–232)

- [ ] **Step 1:** Fill chair's REMOVE cell by reading `chair_concept/src/chair_existence.{h,cpp}`. Expected: `✅ rc::exist`.
- [ ] **Step 2:** Add a hood row, every cell derived by reading `hood_concept/src/`.
- [ ] **Step 3:** Add two columns, "room follows robot" and "outline containment", for every countable agent, from the
  state after Tasks 4–9.
  - Bottle's containment cell: `⚠️ birth only (no existence channel)`.
- [ ] **Step 4:** Commit `docs: lifecycle audit after Phase 0`.

### Task 17: Live validation (definition of done)

**Files:**
- Modify: the ledger (results)

- [ ] **Step 1:** Coordinate Webots. Run the six agents from the worktree in the three-room world through
  room_1 → room_2 → room_1, crossing DOOR_0.
- [ ] **Step 2:** Check and record, per agent:
  - (a) on leaving room_1, its objects are dropped; on return, they come back with the SAME uids
    (`ltom_agent` recall log), including any born in a proto-room (Review Focus 5);
  - (b) parked in room_2 facing the open DOOR_0, NO object of room_1 is born (the chair_39 case; Task 2's reproduction pose);
  - (c) right after the crossing, births are deferred until room_2's outline appears, then proceed;
  - (d) `Fps` no worse than the Task 1 baseline, with room_concept, controller and retina at nominal rate.
- [ ] **Step 3:** Release Webots and tell `room-concept-fd`.
- [ ] **Step 4:** Commit the ledger. Phase 0 is DONE only if (a)–(d) hold. Report to the owner. Merge only when asked.
