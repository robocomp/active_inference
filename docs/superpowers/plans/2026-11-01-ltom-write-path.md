# ltom_agent write path (rollout step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new agent `ltom_agent` that watches the live graph and keeps one persisted repertory entry per `object_uid`
for every chair, table, bottle, cabinet and refrigerator instance it ever sees (lives, deaths, pose keyframes),
with a CSV dump to check it against a tour. No recall, no permanence, no memory graph yet.

**Architecture:** A pure C++ core (`Repertory`: sightings in, entries out) is the model and is unit-tested
without DSR. A live-graph reader turns object nodes into `Sighting`s. The repertory persists itself as one JSON
file (atomic write, `.prev` fallback) plus a CSV view. The agent is a robocompdsl `options dsr` skeleton
joining domain 0 only, with the presence protocol copied from episodic_agent.

**Tech Stack:** C++23, cortex DSR (type-attributed attributes), Qt6 Core (QJsonDocument), robocompdsl, cbuild.

**Spec:** `docs/superpowers/specs/2026-11-01-long-term-object-memory-design.md` (§4, §6.1, §8 step 2, §12).
Code facts this plan relies on are restated in the task that needs them, with file:line references.

## Global Constraints

- CLAUDE.md throughout: and/or/not; C++23; `std::from_chars` for any number read from text and
  `imbue(std::locale::classic())` on every file written; type-attributed DSR only, never `runtime_checked_*`;
  no `Qt::DirectConnection` (connect no DSR signals at all: poll); SIGTERM only; `cbuild` with `-j8`
  (check `pgrep -a cc1plus` first); every `std::optional` from inner_eigen checked.
- `[Agent] id = 28` (27 is self_calibration; the recipe's "next free 27+" is stale). Reserve DDS domain **5** for
  ltom's future memory graph in the registry; step 2 joins domain 0 only.
- ltom never writes an object node, never decides identity, and no concept agent requires it (spec anti-goals).
- A node disappearing is never evidence of absence; deletions while the owner is gone are "owner exit", not
  death; ltom's own downtime is a gap (spec §12).
- Scope: `object` nodes whose name starts `chair_`, `table_`, `bottle_`, `cabinet_`, `refrigerator_` and that carry
  `object_uid`. Doors excluded. Objects whose room is a proto-room (room node with a `proto` self-edge) are skipped.
- Live runs only after coordinating with the room session ("Guardes actions, ever shared facts"); never build in
  rt-build.
- Work in worktree `active_inference.worktrees/ltom`, branch `feat/ltom-write-path`, from main.

## Rulings in this plan (my calls, against the spec)

- **No memory graph in step 2.** The repertory file is the store of record; the domain-5 memory graph becomes a
  projection of it, added when its first reader exists (language_agent, step 4). Why: step 2 has no reader, and one
  source of truth is simpler than a graph plus a JSON blob attribute plus a copy of ltsm's MemoryStore.
  Cost if wrong: one task later to project entries into a graph; the file format does not change.
- **No permanence in step 2.** Its update needs the visibility test (§5, perception-card evaluator); counting only
  re-sightings would bias it upward. It arrives with recall (step 3). Cost if wrong: none stored today to migrate.
- **Room re-key on ltsm merges deferred** (§12): entries store `room_key` = the live room node's `room_id`
  attribute (else the trailing digits of its name, ltsm's own `room_index()` rule). Re-keying needs an ltsm
  mapping that does not exist yet; logged as the dependency.
- **Keyframe test is a significance test, flagged:** a new pose keyframe is written when the squared Mahalanobis
  distance of the sighting from the entry's last keyframe in the same room, under the SUM of both covariances over
  (x, y, yaw), exceeds χ²₃(0.99) = 11.34. Writing a record is a binary act, so some decision rule is unavoidable;
  this one scales with the estimate's own uncertainty rather than a metric cutoff. A room change always keyframes.

## Review Focus

1. Graceful exit of a concept agent deletes its instances *before* its own agent node: those deletions must end
   as `owner_exit`, not `death` (Task 2 test `test_owner_exit_after_graceful_cleanup_order`).
2. ltom restarted while instances are alive: the open lives must continue (gap closed and reopened), not split
   into a second life (Task 2 `test_gap_then_same_uid_reopens`, Task 3 `test_load_closes_open_lives_as_gap`).
3. A torn or empty repertory file at start: load the `.prev`, log loudly, never start empty silently
   (Task 3 `test_corrupt_main_falls_back_to_prev`).
4. es_ES locale: the CSV must use decimal points (Task 3 runs under `setlocale(LC_ALL,"es_ES.UTF-8")`).
5. A bottle re-parented under a table: its room is found by walking parents to the `room` node, and its pose is
   room-relative (Task 4 `test_bottle_on_table_resolves_room`).

---

### Task 1: Agent skeleton, config, registry

**Files:**
- Create: `ltom_agent/ltom_agent.cdsl` (`Component ltom_agent { Communications { }; language Cpp11; options dsr; };`,
  the same shape as `episodic_agent/episodic_agent.cdsl`), the generated tree (`robocompdsl ltom_agent.cdsl .`),
  `ltom_agent/etc/config.toml`, `ltom_agent/src/specificworker_presence.cpp`, `ltom_agent/README.md`
- Modify: `CONCEPT_AGENT_RECIPE.md` §1 registry table (add 28 ltom_agent; domain 5 reserved; next free 29)

**Interfaces:**
- Produces: a built `ltom_agent/bin/ltom_agent` that joins domain 0, publishes its agent node
  `"ltom_agent 28"`, runs presence with `required_agent_names = []`, and exits cleanly on SIGTERM.

- [ ] **Step 1:** Generate the skeleton; set `[Agent] id = 28`, `name = "ltom_agent"`, one graph table
  `[Agent.dsr] configFile = "" domain = 0` with tree/graph/2d all false (every `[Agent.*]` table must define all
  three, see `ltsm_agent/etc/config.toml`). Add an `[LTOM]` table now with the keys later tasks read:
  `period_ms = 500`, `owner_grace_ms = 3000` (comment: = presence `heartbeat_timeout_ms`), `save_period_s = 30`,
  `repertory_path = "etc/ltom_repertory.json"`, `csv_path = "etc/ltom_repertory.csv"`,
  `classes = ["chair", "table", "bottle", "cabinet", "refrigerator"]`.
- [ ] **Step 2:** Copy episodic_agent's presence file shape (debounced Degraded with a ~3000 ms grace timer,
  `cleanup_owned_nodes()` with `rc::owned::Spec{.agent = "ltom_agent"}` and no instance prefix: ltom owns only its
  agent node in step 2). Compute body empty for now.
- [ ] **Step 3:** Build: `cd ltom_agent && cbuild -j8`. Expected: `Built target ltom_agent`.
- [ ] **Step 4:** Smoke test without a fleet (robot_concept alone is enough; if the room session holds Webots, run
  on a private domain by setting `domain = 64` in a copy `etc/config_smoke.toml`, not committed):
  start, wait 5 s, SIGTERM. Expected in the log: the agent node created, then `[Graph] node 'ltom_agent 28'
  deleted`, exit code 0 (or 143).
- [ ] **Step 5:** Registry row in CONCEPT_AGENT_RECIPE.md §1 (id 28, domain 5 reserved for ltom memory); fix
  "next free" to 29. Commit: `ltom_agent: skeleton (id 28, domain 0 only, presence, no compute)`.

### Task 2: Repertory core (pure C++)

**Files:**
- Create: `ltom_agent/src/repertory.h`, `ltom_agent/src/repertory.cpp`
- Test: `ltom_agent/tests/test_repertory.cpp`, `ltom_agent/tests/CMakeLists.txt` (plain main + CHECK macro as in
  `common/instance_identity/instance_identity_test.cpp`; one executable per test file; Qt6::Core linked for Task 3)

**Interfaces (namespace `ltom`):**
- Produces:
  ```cpp
  struct Pose2   { double x, y, yaw; double vx, vy, vyaw; };          // room frame, metres/rad; variances
  struct Sighting{ std::uint64_t uid; std::string cls, name; std::int64_t room_key;
                   Pose2 pose; float w, d, h; std::int64_t born_ms; };
  enum class End { alive, death, owner_exit, gap };
  struct Life    { std::int64_t born_ms, first_seen_ms, last_seen_ms, end_ms = -1; End end = End::alive; };
  struct Keyframe{ std::int64_t t_ms, room_key; Pose2 pose; };
  struct Entry   { std::uint64_t uid; std::string cls, name; float w, d, h; std::int64_t n_sightings = 0;
                   std::vector<Life> lives; std::vector<Keyframe> keyframes; };
  struct Changes { std::vector<std::uint64_t> born, reopened, died, owner_exited; };
  inline std::uint32_t owner_of(std::uint64_t uid) { return static_cast<std::uint32_t>(uid >> 40); }
  class Repertory {
  public:
      explicit Repertory(std::int64_t owner_grace_ms);
      Changes observe(std::span<const Sighting> alive, const std::set<std::uint32_t>& owners_present, std::int64_t now_ms);
      void close_open_lives_as_gap();                // at load: every alive life -> End::gap, end_ms = last_seen_ms
      const std::map<std::uint64_t, Entry>& entries() const;
      void insert_loaded(Entry e);                   // Task 3's loader
      bool dirty() const; void clear_dirty();
  };
  constexpr double kKeyframeChi2 = 11.34;            // χ²₃(0.99), see the plan's ruling
  ```

Rules `observe` implements (each one a test):
- unknown uid → new Entry, one Life (born_ms from the sighting), one keyframe; in `born`.
- known uid, last life alive → n_sightings++, last_seen, sizes/name updated; keyframe per the χ² rule or a room change.
- known uid, last life `gap` → the same life is reopened (end = alive, end_ms = -1); in `reopened`.
- known uid, last life `death`/`owner_exit` → a NEW life; in `born`.
- uid alive last call, absent now: owner absent now → `owner_exit` at once. Owner present → pending; resolved on a
  later call: owner seen absent at any call within `owner_grace_ms` of the vanish → `owner_exit`, else `death`
  once `owner_grace_ms` has passed. The life's `end_ms` is the vanish time either way. A uid that reappears while
  pending simply continues its life (a flap, no end recorded).

- [ ] **Step 1: Write the failing tests** in `test_repertory.cpp`:
  `test_new_uid_creates_entry_and_keyframe`, `test_resighting_counts_without_keyframe_when_still`
  (same pose, σ=0.03 m → 1 keyframe after 10 sightings), `test_moved_object_keyframes` (move 0.5 m with σ=0.03 →
  2 keyframes), `test_room_change_keyframes`, `test_death_after_grace` (vanish at t, owner present through
  t+3000 → `died` at the call after t+3000, end_ms == t), `test_owner_exit_after_graceful_cleanup_order`
  (instance vanishes at t with owner present, owner absent at t+500 → `owner_exited`, never `died`),
  `test_owner_absent_at_vanish_is_owner_exit`, `test_flap_within_grace_is_not_an_end`,
  `test_gap_then_same_uid_reopens`, `test_death_then_same_uid_is_new_life`, `test_owner_of`
  (`owner_of((20ull<<40)|7) == 20`).
- [ ] **Step 2:** Run: `cmake -S ltom_agent/tests -B ltom_agent/tests/build && make -C ltom_agent/tests/build -j8 && ltom_agent/tests/build/test_repertory`.
  Expected: build fails (no repertory.h).
- [ ] **Step 3:** Implement `repertory.{h,cpp}`. Yaw residual wrapped to (−π, π] before the χ² term.
- [ ] **Step 4:** Run the Step 2 command. Expected: `test_repertory: ok`.
- [ ] **Step 5:** Commit: `ltom_agent: repertory core (lives, deaths vs owner exit, gaps, χ² keyframes)`.

### Task 3: Persistence (JSON store of record + CSV view)

**Files:**
- Create: `ltom_agent/src/repertory_io.h`, `ltom_agent/src/repertory_io.cpp`
- Test: `ltom_agent/tests/test_repertory_io.cpp` (added to the tests CMake)

**Interfaces:**
- Consumes: Task 2's `Entry`, `Repertory::insert_loaded`, `close_open_lives_as_gap`.
- Produces:
  ```cpp
  enum class LoadResult { loaded, loaded_prev, missing, unreadable };
  LoadResult load_repertory(const std::filesystem::path& file, Repertory& into);   // then caller closes gaps
  bool save_repertory(const std::filesystem::path& file, const Repertory& r);      // tmp + rename, keeps file.prev
  bool write_csv(const std::filesystem::path& file, const Repertory& r);
  ```
- JSON: `{"version":1,"entries":[{uid (as a decimal STRING: QJson doubles lose uint64 precision), cls, name, w, d,
  h, n_sightings, lives:[{born_ms, first_seen_ms, last_seen_ms, end_ms, end:"alive|death|owner_exit|gap"}],
  keyframes:[{t_ms, room_key, x, y, yaw, vx, vy, vyaw}]}]}`; uid parsed back with `std::from_chars`.
- CSV header (one row per entry, its LAST life and LAST keyframe):
  `uid,owner,class,name,lives,state,born_ms,first_seen_ms,last_seen_ms,end_ms,room_key,x,y,yaw,sx,sy,w,d,h,n_sightings,n_keyframes`
  (`sx`,`sy` = standard deviations), written with `imbue(std::locale::classic())`.
- Save: write `file.tmp`, flush, check the stream, rename the old `file` to `file.prev`, rename tmp to `file`.
  Load: `file` if it parses with `"version":1`; else `file.prev` (log `⚠ … falling back to .prev`); else
  `unreadable` (log loudly; the agent then starts empty but never overwrites the unreadable file: it saves to
  `file` only after moving the bad one to `file.bad.<epoch_ms>`).

- [ ] **Step 1: Write the failing tests** (main sets `std::setlocale(LC_ALL, "es_ES.UTF-8")`, falls back to `""`):
  `test_round_trip_preserves_entries` (two entries, a uid > 2^53, three lives with every End value),
  `test_load_closes_open_lives_as_gap` (alive life saved → after load + `close_open_lives_as_gap` it is `gap`
  with end_ms == last_seen_ms), `test_corrupt_main_falls_back_to_prev`, `test_missing_is_missing`,
  `test_csv_has_points_not_commas` (x = 1.25 → the row contains `1.25`, no `1,25`),
  `test_csv_row_count_and_header`.
- [ ] **Step 2:** Run `make -C ltom_agent/tests/build -j8 && ltom_agent/tests/build/test_repertory_io`.
  Expected: build fails.
- [ ] **Step 3:** Implement `repertory_io.{h,cpp}` with QJsonDocument/QJsonObject (no new dependency).
- [ ] **Step 4:** Run both test binaries. Expected: `test_repertory: ok`, `test_repertory_io: ok`.
- [ ] **Step 5:** Commit: `ltom_agent: repertory persistence (JSON + .prev fallback) and CSV view`.

### Task 4: Live-graph reader

**Files:**
- Create: `ltom_agent/src/graph_sightings.h`, `ltom_agent/src/graph_sightings.cpp`
- Test: `ltom_agent/tests/test_graph_sightings.cpp` (DSR-linked, like `episodic_agent/tests/CMakeLists.txt`:
  dsr_core dsr_api fastcdr fastdds Qt6; a private DDS domain **63**; no `-march=native`)

**Interfaces:**
- Consumes: Task 2's `Sighting`, `Pose2`.
- Produces:
  ```cpp
  struct GraphView { std::vector<ltom::Sighting> alive; std::set<std::uint32_t> owners_present; };
  GraphView read_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& ie, std::span<const std::string> classes);
  ```
- Facts it relies on (from the code): instances are type `object`, named `<class>_<N>`, carrying `object_uid_att`
  (uint64), `width_m_att`/`depth_m_att`/`height_m_att` (metres), `timestamp_creation_att` (epoch ms); the parent
  is reached through `parent_att`; the RT edge parent→instance carries `rt_covariance_att` (36 floats, 6×6
  row-major over x,y,z,rx,ry,rz: variances at 0, 7, 35). Agent nodes are type `agent`, named
  `"<agent_name> <id>"` (`common/agent_state_publisher/agent_state_publisher.h:8`); `owners_present` = the ids
  parsed with `std::from_chars` from those names.
- Room: walk `parent_att` up to the first node of type `room`; skip the instance if that room has a `proto`
  self-edge or if no room is found. `room_key` = `room_id_att` on the room, else trailing digits of its name
  (`ltsm_agent/src/room_eviction.cpp:128-170` rule). Pose: `ie.get_transformation_matrix(room.name(),
  instance.name())` (ts 0, main thread; skip the instance on nullopt); yaw from the rotation. Variances from the
  instance's own parent edge (an approximation for a bottle on a table: the parent's own uncertainty is not
  added; noted in a comment).
- An instance without `object_uid` is skipped (pre-step-1 binary).

- [ ] **Step 1: Write the failing tests:** build a graph on domain 63 (root → room `room_1` with `room_id` 1 →
  `chair_3` with `object_uid` (20<<40)|3, RT at (1, 2, yaw 0.5), cov diag 0.01; `table_1` under the room;
  `bottle_2` under `table_1`; a `door_1` object; a proto room `room_9` with `chair_9`; agent node
  `"chair_concept 20"`). Tests: `test_reads_chair_pose_and_sizes`, `test_bottle_on_table_resolves_room`
  (room_key 1, pose = table pose ∘ bottle offset), `test_door_and_proto_skipped`,
  `test_missing_uid_skipped`, `test_owners_present_parsed` (== {20}).
- [ ] **Step 2:** Run the test binary. Expected: build fails.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run all three test binaries. Expected: all `ok`.
- [ ] **Step 5:** Commit: `ltom_agent: live-graph reader (object instances -> sightings, owners present)`.

### Task 5: Wire the agent

**Files:**
- Modify: `ltom_agent/src/specificworker.{h,cpp}`, `ltom_agent/src/CMakeLists.txt` (append the three sources;
  clean reconfigure: remove `build/CMakeCache.txt`), `ltom_agent/README.md`
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` conformance audit: one row for ltom_agent (observer: CREATE/UPDATE/REMOVE
  of instances = n/a, owns only its agent node). ⚠ the main checkout has uncommitted edits to this file; keep
  the row a pure addition so the merge applies cleanly.

**Interfaces:** consumes Tasks 2–4.

- [ ] **Step 1:** `initialize()`: read `[LTOM]`; `load_repertory`; then `close_open_lives_as_gap()`; log the
  result and entry count. Own one `InnerEigenAPI` on the main thread.
- [ ] **Step 2:** `compute()` every `period_ms` while Operating: `read_graph` → `observe(now)`; one log line per
  change (`[ltom] born chair_3 uid=… room 1`, `died`, `owner_exit`, `reopened`); when dirty and
  `save_period_s` has passed: `save_repertory` + `write_csv`. Destructor (graceful exit): save + CSV.
- [ ] **Step 3:** Build `cbuild -j8`. Expected: `Built target ltom_agent`. Run all test binaries again: all `ok`.
- [ ] **Step 4:** Commit: `ltom_agent: observe-consolidate-persist loop, save on period and exit`.

### Task 6: Live check against a tour (controller-run, not dispatched)

- [ ] **Step 1:** Ask the room session for a slot. Start ltom_agent with a short fleet (robot, room, retina,
  chair, table; Webots world with at least one chair and one table) or alongside a room-session tour if it agrees.
- [ ] **Step 2:** Expected: every `BIRTH '<class>_N'` line in the concept agents' logs has one CSV row with the
  same name and uid `(agent_id<<40)|N`; positions within 2σ of the agent's own last logged pose.
- [ ] **Step 3:** SIGTERM chair_concept → its rows show `owner_exit`, not `death`. SIGTERM ltom, restart it while the
  table lives → the table row keeps `lives = 1` and state `alive` after a few cycles (gap reopened).
- [ ] **Step 4:** SIGTERM everything; append the result to the ledger; tell the room session it is clear.
