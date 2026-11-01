# Warm Start from a Stored LTSM Memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The fleet starts in room_1 already knowing room_1 and room_2 (pingpong41's memory), so a tour can go room_1 → room_2 → R3 directly.

**Architecture:** ltsm posts a *start-up recall* for room_1 (the same `recall_<room>` node it posts at a crossing, hung from the robot, no door). room_concept in the new `MapMode = "ltsm"` waits for it, adopts its outline as the given layout in the stored frame and localises by its global search. door_concept, seeing that recall announce its first room, restores room_1's doors from its own remembered table through its existing recall path. A fixture (ltsm generation + door table + room-name counter) is installed by a new tour.

**Tech Stack:** C++23, DSR/cortex (no new attributes), Qt, bash tour scripts. Build with `cbuild`-equivalent at `-j8` (see CLAUDE.md "Build").

**Spec:** `docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md` (read it first; §4 and §5 were revised while planning).

## Global Constraints

- All work in the worktree `/home/pbustos/robocomp/components/active_inference.worktrees/rt-build`, branch `feat/room-recognition`.
- No new DSR attribute (no cortex reinstall). Use: `object_subtype` = `"room_recall"`, `room_id`, `delimiting_polygon_x/y`, `parent`, `level`, `recall_late` (existing).
- Type-attributed DSR access only (`get_attrib_by_name<x_att>`, `add_or_modify_attrib_local<x_att>`), never `runtime_checked_*`.
- No `Qt::DirectConnection` on DSR signals; always check `get_transformation_matrix`'s optional.
- Number parsing from files: `std::from_chars` only; data files written through `imbue(std::locale::classic())`.
- `and` / `or` / `not`, C++23 containers/algorithms.
- Builds at `-j8` inside the memory-capped scope; check `pgrep -a cc1plus` first.
- Stop agents with SIGTERM only.
- `MapMode` values `given` / `estimate` / `last` behave exactly as before.

## Review Focus

1. **ltsm posts before room_concept is up, or room_concept restarts:** the start-up recall must still be there when room_concept looks (ltsm keeps it while room_1 is the current room) — Task 2's live check reads it after room_concept is up.
2. **The global search cannot decide** (symmetric-looking view at the start pose): room_concept holds the pose undecided, never publishes a guess — Task 4 Step 5 (no recall ⇒ no room, no pose) + Task 6 acceptance checks the first reloc search's result in room_concept.log.
3. **A stale `recall_room_1` / `room_1` from a SIGKILLed previous run:** room_concept's stale sweep runs before waiting; ltsm's existing `remove_all()` at start-up clears old recalls (specificworker.cpp:299-300) — Task 4 keeps the sweep, Task 2 posts only after `remove_all`.
4. **door_concept sees the room before the recall** (ordering): its entry decision for room_1 is taken once — Task 3 makes the start-up announcement latch before `enter_room_memory` and Task 6 checks `[restore] 'room_1' recalled` in door_concept.log.
5. **The fixture is modified by the run:** Task 5 records sha256s; Task 6 compares after the run.

---

### Task 1: The start-up recall, one definition for three agents

**Files:**
- Create: `common/room_transition/startup_recall.h`
- Test: `common/room_transition/tests/test_startup_recall.cpp`

**Interfaces:**
- Produces:
  - `namespace rc::room_transition { inline constexpr std::string_view kRecallSubtype = "room_recall"; }`
  - `[[nodiscard]] inline bool is_startup_recall(std::string_view subtype, std::string_view parent_type, bool late)` — true iff `subtype == "room_recall"`, the node hangs from a node of type `"robot"`, and it is not a late recall (`late == false`).
  - `[[nodiscard]] inline std::optional<std::vector<Eigen::Vector2f>> recall_outline(std::span<const float> xs, std::span<const float> ys)` — the polygon; `nullopt` when sizes differ or fewer than 3 vertices or any value is not finite.

- [ ] **Step 1: Write the failing test** `test_startup_recall.cpp` (plain asserts + `std::print`, like the other tests in that folder):
  - `is_startup_recall("room_recall", "robot", false) == true`
  - `is_startup_recall("room_recall", "robot", true) == false` (late recall)
  - `is_startup_recall("room_recall", "object", false) == false` (crossing recall, hangs from a door)
  - `is_startup_recall("passage", "robot", false) == false`
  - `recall_outline({0,4,4}, {0,0,3})` has 3 vertices, second `(4,0)`; `recall_outline({0,4}, {0,0})` → nullopt; `recall_outline({0,4,4}, {0,0})` → nullopt; a NaN → nullopt.
- [ ] **Step 2: Run it, expect a compile failure** (`startup_recall.h` missing): `g++ -std=c++23 -O1 -I/usr/include/eigen3 common/room_transition/tests/test_startup_recall.cpp -o /tmp/tsr && /tmp/tsr`
- [ ] **Step 3: Implement** the three items in `startup_recall.h` (header-only, Eigen + std).
- [ ] **Step 4: Run it, expect all checks PASS.**
- [ ] **Step 5: Commit** `common: one definition of the start-up recall (warm start)`.

### Task 2: ltsm posts the start-up recall (`[WarmStart] Room`)

**Files:**
- Modify: `ltsm_agent/src/room_recall.h`, `ltsm_agent/src/room_recall.cpp` (new method), `ltsm_agent/src/specificworker.h/.cpp` (config + call), `ltsm_agent/etc/config.toml` (commented key, OFF)

**Interfaces:**
- Consumes: Task 1 `kRecallSubtype`.
- Produces:
  - `std::uint64_t RoomRecall::post_startup(const std::string& live_room)` — finds the memory room `r<k>_<live_room>` (same lookup as `RoomEviction::remembered`), creates in the LIVE graph `recall_<live_room>`: type `metaconcept`, `object_subtype "room_recall"`, `room_id` k, `delimiting_polygon_x/y` read exactly as `RoomRecall::step` reads them for a crossing recall, `parent` = the robot node (type `"robot"`), `level` = robot's + 1, NO RT edge, NO `recall_late`. Returns the node id, 0 if the memory room, its outline or the robot node is missing (logs which). Idempotent: an existing `recall_<live_room>` is left as is and its id returned.
  - `void RoomRecall::step_startup()` — removes the start-up recall once a `current` edge points at a room OTHER than `live_room` (the robot has left it; everyone has latched it by then).
  - Config `[WarmStart] Room` (string; absent or empty = OFF).

- [ ] **Step 1:** Add the key to `ltsm_agent/etc/config.toml` commented out with a one-line explanation (`# Room = "room_1"   # warm start: post a start-up recall of this remembered room (docs/superpowers/specs/2026-11-01-ltsm-warm-start-design.md)`); read it in `SpecificWorker` with the existing `rc::cfg::Reader` (default empty).
- [ ] **Step 2:** Implement `post_startup` / `step_startup`. In `compute()`, after `recall_` exists and after its start-up `remove_all()`, call `post_startup(room)` every cycle until it returns non-zero (then log once `[warm] start-up recall recall_<room> posted (room_id k, N vertices)`), and `step_startup()` every cycle after.
- [ ] **Step 3:** Build ltsm (`-j8`, no errors).
- [ ] **Step 4: Live check (no tour yet):** set `Room = "room_1"` in a scratch copy of the config, start ltsm alone on a memory that has `r1_room_1` (copy `crossing/ltsm_archive/ltsm_history_before_angle90_20261101_104949_active_inference/ltsm_20261101_104058_346*.json` into ltsm's `etc/ltsm_history/`), and check the log line and that `recall_room_1` exists in the live graph (graph viewer or `graph_recorder`). Expected: the log line with `room_id 1` and 32 ± 1 vertices. Restore the config; SIGTERM ltsm.
- [ ] **Step 5: Commit** `ltsm: [WarmStart] Room posts a start-up recall of a remembered room`.

### Task 3: door_concept restores the first room's doors when a start-up recall announced it

**Files:**
- Modify: `door_concept/src/door_identity_table.h:415-423` (`plan_room_entry`), `door_concept/src/specificworker.cpp` (`latch_recall_names` ~3548, `enter_room_memory` ~3742), `door_concept/src/specificworker.h` (one member)
- Test: `door_concept/tests/test_door_identity_table.cpp`

**Interfaces:**
- Consumes: Task 1 `is_startup_recall`.
- Produces: `RoomEntryPlan plan_room_entry(bool first_resolution, bool recall_announced, bool recall_late, std::string_view left_room, std::string_view entered_room, bool startup_announced = false)` — `recalled = recall_announced and (not first_resolution or startup_announced)`; `drop_left` unchanged.

- [ ] **Step 1: Failing test** in `test_door_identity_table.cpp`: `plan_room_entry(true, true, false, "", "room_1", true).recalled == true`; `plan_room_entry(true, true, false, "", "room_1", false).recalled == false` (today's rule kept); `plan_room_entry(false, true, false, "room_2", "room_1").recalled == true`; `plan_room_entry(true, false, false, "", "room_1", true).recalled == false` (no announcement, nothing).
- [ ] **Step 2: Run** (build line in the file's header comment); expect a compile failure on the 6th argument.
- [ ] **Step 3: Implement** the parameter; in `latch_recall_names` also record `recall_startup_names_` (a `std::unordered_set<std::string>`) for nodes where `is_startup_recall(subtype, parent's type, recall_late)`; pass `recall_startup_names_.contains(room->name())` to `plan_room_entry` in `enter_room_memory`. Make sure `latch_recall_names()` runs before the first `enter_room_memory` of the run in the cycle order (read `compute()`; move the call earlier only if it is not already before).
- [ ] **Step 4: Run the test, PASS;** build door_concept (`-j8`).
- [ ] **Step 5: Commit** `door_concept: a start-up recall announces the first room too (its doors restored from the remembered table)`.

### Task 4: room_concept `MapMode = "ltsm"`

**Files:**
- Modify: `room_concept/src/room_config.cpp:212-239` (parse), `room_concept/src/localizer_params.h:624` (enum), `room_concept/src/room_concept.h` + `room_concept_reloc_bootstrap.cpp:96` (wait flag), `room_concept/src/specificworker_startup.cpp:388-410` (startup branch), `room_concept/src/specificworker.cpp` (`compute()` poll + new `initialize_room_model_from_recall`), `room_concept/etc/config.toml:1162-1256` (document the value; keep `given` as the default)

**Interfaces:**
- Consumes: Task 1 `is_startup_recall`, `recall_outline`.
- Produces:
  - `Params::MapMode { Given, Estimate, Ltsm }`; `"ltsm"` parses to `Ltsm`; `estimating()` unchanged (Ltsm is NOT estimating).
  - `void RoomConcept::set_awaiting_layout(bool)` / `bool awaiting_layout() const` (an `std::atomic<bool>`, release/acquire); `bootstrap_initialization_from_lidar()` returns `false` immediately while it is set.
  - `bool SpecificWorker::initialize_room_model_from_recall()` — scans `metaconcept` nodes for a start-up recall (Task 1), takes its outline, calls `calib_->set_room_polygon(poly, Eigen::Vector2f::Zero())` and `room_concept_.configure_room_from_polygon(...)` WITHOUT recentring (the stored frame is kept exactly), sets `room_initialized_from_svg_polygon_ = true`, then `set_awaiting_layout(false)`; logs `[room] MapMode=ltsm: layout of '<name>' (room_id k, N vertices) from ltsm's start-up recall — localising`. Returns true once done.

- [ ] **Step 1:** Parse `"ltsm"` (and keep the warning text listing all four values). In the startup branch: `Ltsm` ⇒ `set_awaiting_layout(true)`, no SVG, status `loaded("layout", "", "MapMode=ltsm: waiting for ltsm's start-up recall")`; do NOT call `set_seed_pose_file` (status line "ignored in ltsm mode"). The stale-room sweep (`cleanup_room_graph_nodes`) stays as it is.
- [ ] **Step 2:** In `compute()`, while `awaiting_layout()`, call `initialize_room_model_from_recall()`; every 10 s while still waiting log `[room] MapMode=ltsm: still waiting for ltsm's start-up recall (N s)`. Never fall back to SVG or estimate.
- [ ] **Step 3:** The room node: the existing given-mode creation (`dsr_create_room_and_reparent`) names it `room_1` / `room_id 1`; if the recall's name is not `room_1`, log a warning `[room] MapMode=ltsm: start-up recall is '<name>', only room_1 is supported for now` and do not adopt (spec non-goal).
- [ ] **Step 4:** Build room_concept (`-j8`); run `reloc_selftest` (expect it identical to `tools/oracle/reloc_selftest.txt`, as on 2b074601).
- [ ] **Step 5:** Live smoke check, `MapMode=ltsm` with NO ltsm running: room_concept logs the waiting line every 10 s, creates no room node, publishes no pose (Review Focus 2/3). SIGTERM.
- [ ] **Step 6: Commit** `room_concept: MapMode=ltsm waits for ltsm's start-up recall and adopts its outline in the stored frame`.

### Task 5: The two-room fixture

**Files:**
- Create: `room_concept/estimation_scripts/scripts/make_warm_fixture.sh`
- Creates data: `/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/fixtures/two_rooms_pingpong41/` (not committed; data)

**Interfaces:**
- Produces: a fixture directory with `ltsm_history/ltsm_<stamp>.json` + `ltsm_<stamp>_layout.json`, `door_identities.csv`, `room_name_hwm.txt`, `SHA256SUMS`, `README` (source run, date).

- [ ] **Step 1:** `make_warm_fixture.sh <fixture_name> <ltsm_generation.json> <door_identities.csv> <hwm>` copies the generation and its `_layout.json` sidecar, the door table and writes the counter file in the format `room_scene_graph.cpp:1611` reads (check it), then `sha256sum * > SHA256SUMS`.
- [ ] **Step 2:** Run it for pingpong41: generation `crossing/ltsm_archive/ltsm_history_before_angle90_20261101_104949_active_inference/ltsm_20261101_104058_346.json`, door table `crossing/door_identities_archive/door_identities_before_angle90_20261101_104949.csv`, hwm `2`.
- [ ] **Step 3: Verify** the fixture: the generation holds memory rooms `r1_room_1` and `r2_room_2` and a `passage_*` whose two faces are in them (python/jq on the JSON); the door table has `room_1,door_1,-0.538…,4.613…` with a pose. Expected: both true.
- [ ] **Step 4: Commit** the script `scripts: make_warm_fixture.sh (warm-start fixture from a run's ltsm generation + door table)`.

### Task 6: The warm-start tour and live acceptance

**Files:**
- Create: `room_concept/estimation_scripts/scripts/warm_start_tour.sh` (from `pingpong_tour.sh`: same locking, ground-truth recorder, `run_leg`, grading, `N_LEGS`)

**Interfaces:**
- Consumes: Tasks 2–5.

- [ ] **Step 1:** The script, `warm_start_tour.sh <run_name> <fixture_dir>`:
  - archive (never delete) whatever is in ltsm's `history_dir`, `ltsm_memory.json`, door_concept's `door_identities.csv` and room_concept's `room_name_hwm.txt` exactly as `pingpong_tour.sh` does, then COPY the fixture's files in (never link; the fixture is read-only);
  - set `RoomShape.MapMode = "ltsm"` in `$WT/room_concept/etc/config.toml` and `[WarmStart] Room = "room_1"` in `$WT/ltsm_agent/etc/config.toml`, restoring both in an `EXIT` trap (same pattern as the angle sweep);
  - close DOOR_0, DOOR_1, DOOR_2; teleport to the usual start pose (`-5.33 3.66 0.58`, via `run_crossing.sh` as now); delete `etc/last_robot_pose.txt`;
  - wait (≤ 120 s) for room_concept's `MapMode=ltsm: layout of 'room_1'` line AND door_concept's `[restore] 'room_1' recalled`; abort with a message otherwise;
  - legs (graded on ground truth like `pingpong_tour.sh`): 1 `cross room_2` (want room_2: y > 9.3796), 2 `explore room_2` then `cross new` (want R3: x > -1.705), 3 `cross room_2` (want room_2), 4 `cross room_1` (want room_1); `N_LEGS` honoured;
  - after the run: `sha256sum -c` the fixture's `SHA256SUMS`; print OK/CHANGED.
- [ ] **Step 2: Acceptance run** `warm_start_tour.sh warm1 crossing/fixtures/two_rooms_pingpong41`. Pass when:
  - start-up: room_1's localised pose within 0.10 m / 3° of ground truth (from `gt_error.csv` after the first reloc search; the room frame ↔ world offset as in the pingpong runs), `door_1` restored, `recall_room_2` posted under it (ltsm log), no room estimate started;
  - leg 1 OK; the new room after leg 2 is named `room_3` (room_concept log);
  - the fixture checksums unchanged.
- [ ] **Step 3:** Record the result in the crossing memory note and commit the script `scripts: warm_start_tour.sh (two-room fixture → room_3)`.
