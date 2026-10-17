# Episodic Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-observer episodic agent that records every mission, step and affordance claim (human and self) as
commitment records with pre/post facts, outcome and side-effect events, kept on a domain-4 DSR graph (today + all-time
statistics) and an append-only archive, queryable live by chat_agent.

**Architecture:** All logic lives in pure, header-only modules under `common/episodic/` (schema, journal, builder,
consolidation, archive, query), each unit-tested without DSR. The controller publishes an append-only **journal**
(missions, step attempts, claims, drive mode) as one string attribute on its own agent node; the new `episodic_agent`
polls it plus the live graph, builds records, writes them to the archive and to its domain-4 graph. chat_agent joins
domain 4 read-only and answers through `em_query.h`.

**Tech Stack:** C++23, Qt6 Core (`QJsonDocument` — locale-independent), RoboComp/DSR (cortex), robocompdsl, g++ test
scripts in the style of `chat_agent/tests/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-10-17-episodic-memory-design.md` (read it first; this plan argues from it).

## Global Constraints

- Graph signals: never `Qt::DirectConnection`. EM **polls** (4 Hz) and connects no signals.
- Attributes: type-attributed access only (`get_attrib_by_name<x_att>`), never `runtime_checked_*`.
- New cortex attributes are added by the user (reinstall); Task 7 is a hard gate — stop there until it lands.
- Numbers from files: `std::from_chars` or `QJsonDocument` only; every written file `imbue(std::locale::classic())`.
  Tests that parse files call `std::setlocale(LC_ALL, "es_ES.UTF-8")` first (fall back to `""`).
- `[Agent] id = 26` for episodic_agent (next free; record it in `CONCEPT_AGENT_RECIPE.md` §1 table, line ~496).
- DDS domains: live 0, ltsm memory 2, **EM 4**. EM is never a required peer of anyone, and requires nobody but
  tolerates everything missing.
- Presence: copy bottle_concept (debounced Degraded, ~3000 ms grace). Stop with SIGTERM, never `kill -9`.
- Builds: `cbuild` / `make -j8`; NO `-march=native`, NO `-DEIGEN_MAX_ALIGN_BYTES=0`.
- Style: C++23 containers/algorithms; `and`/`or`/`not`; comment density like neighbouring code.
- Absent ≠ zero: a fact or number that cannot be read is omitted (`std::optional`), never defaulted.
- EM never invents an outcome: outcomes come from the journal; EM only writes `lost` (owner vanished / journal gap /
  controller restarted) or `interrupted_by_em_shutdown`.
- Timestamps: wall-clock ms since epoch (`system_clock`); a record's **day** is the local date of its `t0`
  (`std::chrono::current_zone()`).

## Review Focus

1. **Controller restart mid-mission** — journal serials restart; EM must close every open record from the old
   controller epoch as `lost`, not wait for ever or mis-attach new entries. (Test: Task 5 `test_controller_epoch_change`.)
2. **Journal overflow while EM is down or slow** — entries fall off the bounded ring; EM must detect the serial gap and
   close affected open records `lost`, never silently skip. (Task 5 `test_journal_gap`.)
3. **EM restart mid-day** — reseeding from the archive must produce no duplicate and no lost record, and record ids
   must keep increasing. (Task 2 `test_reload_last_line_wins`, Task 9 live check.)
4. **A mission that spans midnight** — it belongs to the day of its `t0`; rollover must not fold an open record.
   (Task 3 `test_rollover_skips_open`.)
5. **es_ES locale** — archive written and re-read under `LC_ALL=es_ES.UTF-8` round-trips decimals exactly.
   (Task 2 `test_locale_roundtrip`.)

---

## File Structure

| File | Responsibility |
|---|---|
| `common/episodic/em_schema.h` | `Record`, `Fact`, `Event`, `Stats`, enums, JSON (de)serialisation, fact/event whitelists |
| `common/episodic/em_journal.h` | controller-side bounded journal of `JournalEntry` with serial + epoch; JSON wire |
| `common/episodic/em_builder.h` | EM-side pure state machine: journal entries + facts + structural events → closed `Record`s |
| `common/episodic/em_consolidate.h` | fold records into `Stats` keyed by (action kind, identity) |
| `common/episodic/em_archive.h` | append-only JSONL day files, reload (last line per id wins) |
| `common/episodic/em_query.h` | pure queries over records/stats + `load_from_graph(DSRGraph&)` adapter |
| `common/episodic/tests/*.cpp`, `run_tests.sh` | unit tests (plain `CHECK` macro, as `mission_interpreter_test.cpp`) |
| `common/mission/mission_interpreter.h` | + structured per-attempt/iteration transitions |
| `common/affordance_manager/affordance_manager.{h,cpp}` | + claim observer callback (start / preempted / outcome) |
| `controller/src/…` | feed and publish the journal |
| `episodic_agent/` (new, robocompdsl) | the observer agent |
| `chat_agent/…` | domain-4 graph, EM tools, digest line |
| `tools/em_report.py` | acceptance report over an archive |

---

### Task 1: Record schema and JSON

**Files:**
- Create: `common/episodic/em_schema.h`, `common/episodic/tests/em_schema_test.cpp`, `common/episodic/tests/run_tests.sh`

**Interfaces:**
- Produces (namespace `rc::em`):
  ```cpp
  enum class Level { Mission, Step, Claim };            // "mission" | "step" | "claim"  (reserved: "arm")
  enum class Author { Human, Self };                    // "human" | "self"
  struct Fact  { std::string node, type, ltsm_id;       // ltsm_id empty = none
                 std::map<std::string, double, std::less<>> values; };   // only attrs actually read
  struct Event { std::int64_t dt_ms; std::string kind, node; };  // kind: born|died|current_moved|offered|withdrawn
  struct Record {
      std::string id, parent;                     // id "<YYYYMMDD>-<seq>", parent "" for a mission
      Level level; Author author;
      std::string kind, target, target_ltsm;      // kind = op / affordance kind / "mission"; target = live node name
      std::string action_json;                    // IR / command text / claim contract, verbatim JSON
      std::string source;                         // mission only: chat|remote|panel|self
      std::uint64_t t0 = 0; std::optional<std::uint64_t> t1;   // t1 empty = still open
      std::string room;                           // current room at t0
      std::vector<Fact> pre, post;
      std::string outcome;                        // "" while open
      int attempt = 0, iteration = 0;             // step only
      std::optional<double> promised, delivered;  // claim only
      std::string alternatives_json;              // claim only: selector candidates, verbatim JSON
      std::vector<Event> events;
  };
  QJsonObject to_json(const Record&);  std::optional<Record> record_from_json(const QJsonObject&);
  inline constexpr std::array kFactAttrs{"door_open_prob","door_reach_prob","door_crossing_progress",
                                         "obj_exist_prob","passage_passable"};   // + pose x,y,yaw + rt_covariance diag
  inline constexpr std::array kEventNodeTypes{"room","door","object","person","table","chair","bottle",
                                              "refrigerator","box","affordance"};
  std::string day_of(std::uint64_t t_ms);    // "YYYYMMDD", local time
  ```
  JSON keys are exactly the field names above.

- [ ] **Step 1: Write failing tests** in `em_schema_test.cpp`:
  `test_roundtrip` (a fully populated Claim record → `to_json` → `record_from_json` equals original);
  `test_absent_stays_absent` (`promised`/`delivered`/`t1` empty → keys absent in JSON → still empty after parse; a
  `Fact` with no `door_open_prob` has no such key — never `0`);
  `test_unknown_level_rejected` (`"level":"fly"` → `std::nullopt`);
  `test_day_of` (a known epoch ms → expected local date string).
- [ ] **Step 2:** Write `run_tests.sh` (copy the compile line style of `chat_agent/tests/run_tests.sh`, Qt6Core via
  pkg-config, `-std=c++23 -Wall -Wextra`), run it — expected: compile failure (header missing).
- [ ] **Step 3:** Implement `em_schema.h` (header-only, Qt Core only, no DSR).
- [ ] **Step 4:** Run `common/episodic/tests/run_tests.sh` — expected: all checks pass, exit 0.
- [ ] **Step 5:** Commit `common/episodic: record schema + JSON, with tests`.

### Task 2: Append-only archive

**Files:**
- Create: `common/episodic/em_archive.h`, `common/episodic/tests/em_archive_test.cpp`; Modify: `run_tests.sh`

**Interfaces:**
- Consumes: `Record`, `to_json`, `record_from_json`, `day_of` (Task 1).
- Produces:
  ```cpp
  class Archive {
  public:
      explicit Archive(std::filesystem::path dir);            // creates dir; files "<dir>/<YYYYMMDD>.jsonl"
      void append(const Record&);                             // one compact JSON line, flushed; file = day_of(r.t0)
      std::vector<Record> load_day(std::string_view day) const;   // last line per id wins, ordered by t0
      std::vector<std::string> days() const;                  // sorted
      int next_seq(std::string_view day) const;               // 1 + max seq among ids of that day
  };
  ```
  Opening records are appended when they open and appended again when they close; readers keep the last line per id.
  A truncated last line (crash mid-write) is skipped with a warning, never fatal.

- [ ] **Step 1: Write failing tests:** `test_reload_last_line_wins` (append open then closed version of one id → load
  returns one record, closed); `test_next_seq` (ids `-1`,`-7` → 8); `test_truncated_tail_skipped` (append garbage
  half-line → load returns the good records); `test_locale_roundtrip` (`setlocale(LC_ALL,"es_ES.UTF-8")`, write
  `promised = 0.260417`, reload → exactly `0.260417`).
- [ ] **Step 2:** Run tests — expected FAIL (missing header).
- [ ] **Step 3:** Implement with `std::ofstream` + `imbue(std::locale::classic())`, lines via
  `QJsonDocument::toJson(QJsonDocument::Compact)`.
- [ ] **Step 4:** Run tests — expected PASS.
- [ ] **Step 5:** Commit `common/episodic: append-only archive, with tests`.

### Task 3: Consolidation

**Files:**
- Create: `common/episodic/em_consolidate.h`, `common/episodic/tests/em_consolidate_test.cpp`; Modify: `run_tests.sh`

**Interfaces:**
- Consumes: `Record` (Task 1).
- Produces (add `Stats` to `em_schema.h` with JSON like `Record`):
  ```cpp
  struct Stats {
      std::string key_kind, key_identity;              // identity = target_ltsm, "" = per-kind only
      std::map<std::string, int, std::less<>> outcomes;   // outcome word → count
      int n = 0; double dur_mean_s = 0, dur_m2 = 0;      // Welford over closed records' t1-t0
      int n_gain = 0; double promised_sum = 0, delivered_sum = 0;   // only records with both present
      std::uint64_t first_t0 = 0, last_t0 = 0;
  };
  QJsonObject to_json(const Stats&); std::optional<Stats> stats_from_json(const QJsonObject&);
  std::pair<std::string,std::string> stats_key(const Record&);   // (kind, target_ltsm)
  void fold(Stats&, const Record&);                               // precondition: r.t1 has value
  // Folds every CLOSED record of days < today into `stats`; returns ids folded. Open records are never folded.
  std::vector<std::string> rollover(std::map<std::pair<std::string,std::string>, Stats>& stats,
                                    std::span<const Record> records, std::string_view today);
  ```
- [ ] **Step 1: Write failing tests:** `test_fold_counts_and_duration` (3 records, outcomes satisfied×2/timeout,
  durations 2,4,6 s → counts, mean 4, variance 4); `test_gain_only_when_both` (one record lacking `delivered` → `n_gain`
  unchanged); `test_key_without_ltsm` (empty `target_ltsm` → key `(kind,"")`); `test_rollover_skips_open` (yesterday's
  open record not folded, not returned; yesterday's closed one folded; today's closed one untouched).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `common/episodic: consolidation into all-time stats, with tests`.

### Task 4: Controller journal (pure)

**Files:**
- Create: `common/episodic/em_journal.h`, `common/episodic/tests/em_journal_test.cpp`; Modify: `run_tests.sh`

**Interfaces:**
- Produces:
  ```cpp
  struct JournalEntry {
      std::uint64_t serial = 0, t_ms = 0;
      enum class Kind { MissionStart, MissionEnd, StepStart, StepEnd, ClaimStart, ClaimEnd, DriveMode } kind;
      std::string mission;        // controller mission serial as text ("m12"); "" outside a human mission
      std::string step_id;        // StepStart/End: IR step id ("2.1"); single commands use "1"
      std::string op;             // step op / affordance kind / drive mode name
      std::string source;         // MissionStart: chat|remote|panel
      std::string text;           // MissionStart: IR JSON or command text · StepStart: resolved grounding JSON
                                  // ClaimStart: {"affordance","producer","target","target_type","contract"} JSON
      std::string outcome;        // *End: outcome word (claims: an aff_outcome word or "preempted")
      int attempt = 0, iteration = 0;
      std::optional<double> promised;     // ClaimStart: quoted epistemic_gain
      std::string candidates;             // ClaimStart: selector candidates JSON array
  };
  class Journal {
  public:
      explicit Journal(std::uint64_t epoch_ms, std::size_t capacity = 256);
      std::uint64_t push(JournalEntry e);        // assigns serial = last+1, returns it
      std::string to_wire() const;               // {"epoch":…, "entries":[…]} compact JSON
      bool dirty() const; void clear_dirty();    // publish only on change
  };
  struct WireView { std::uint64_t epoch; std::vector<JournalEntry> entries; };
  std::optional<WireView> journal_from_wire(std::string_view);
  ```
- [ ] **Step 1: Write failing tests:** `test_serials_monotonic`; `test_capacity_drops_oldest` (capacity 3, push 5 →
  wire has serials 3,4,5); `test_wire_roundtrip` (every field, `promised` absent stays absent); `test_dirty_flag`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement (std::deque). **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `common/episodic: controller journal + wire codec, with tests`.

### Task 5: Record builder (pure EM state machine)

**Files:**
- Create: `common/episodic/em_builder.h`, `common/episodic/tests/em_builder_test.cpp`; Modify: `run_tests.sh`

**Interfaces:**
- Consumes: Tasks 1, 4.
- Produces:
  ```cpp
  struct FactSource {                           // implemented by the agent over DSR; faked in tests
      virtual ~FactSource() = default;
      virtual std::optional<Fact> fact(std::string_view node) const = 0;   // nullopt = node absent
      virtual std::string current_room() const = 0;
  };
  class Builder {
  public:
      Builder(const FactSource&, std::function<std::string(std::uint64_t t0)> next_id);
      // One poll. Returns records that OPENED or CLOSED this call (to append to the archive / graph).
      std::vector<Record> on_journal(const WireView&, std::uint64_t now_ms);
      std::vector<Record> on_events(std::span<const Event>, std::uint64_t now_ms);  // attach to every open record
      std::vector<Record> shutdown(std::uint64_t now_ms);   // close all open: "interrupted_by_em_shutdown"
      std::vector<Record> owner_lost(std::uint64_t now_ms); // controller node gone: close all open "lost"
      std::vector<Record> open() const;
  };
  ```
  Rules (from spec §5–§6):
  - MissionStart opens a Human mission (`source` from the entry); StepStart opens a step whose parent is the innermost
    open step whose `step_id` is a prefix (`"2"` for `"2.1"`), else the mission. Each (step_id, attempt, iteration) is
    its own record.
  - ClaimStart opens a claim; parent = innermost open step, else the open self mission. `target` from the entry's
    `text.target`; `pre` = `fact(target)`; `promised`, `alternatives_json` copied.
  - *End closes the matching record: `post` from `fact(...)`, `outcome` from the entry. `delivered` =
    ½·ln(det Σ_pre / det Σ_post) over the `rt_covariance` xy diagonal when both present, else absent.
  - Self mission: while drive mode is `AffordancesOnly` and no human mission is open, a Self mission is open; it
    closes with outcome `room_changed` (and a new one opens) when `current_room()` changes, `superseded` when a human
    mission starts or drive mode changes.
  - Epoch change (wire `epoch` ≠ last seen): close every open record from the old epoch with `lost`.
  - Gap (first new serial > last_seen+1): close every open record whose closing entry may have been dropped
    (all open records) with `lost`, then process the entries.
  - Entries with serial ≤ last_seen are ignored (idempotent polling).
- [ ] **Step 1: Write failing tests** with a `FakeFactSource` (map node→Fact, settable room) and scripted wires:
  `test_mission_step_claim_nesting` (mission → step "1" → claim → ends → 3 closed records, parents correct, claim pre/post
  from facts); `test_attempts_are_records` (step "2" attempts 1 and 2 → two step records); `test_composite_nesting`
  (repeat step "2" with children "2.1" → parent of "2.1" is "2"); `test_preempted_claim` (ClaimEnd outcome
  `preempted` copied verbatim); `test_self_mission_split_on_room` (drive mode AffordancesOnly, claim, room flips → self
  mission closed `room_changed`, new one open); `test_controller_epoch_change`; `test_journal_gap`;
  `test_idempotent_repoll` (same wire twice → second call returns nothing); `test_delivered_absent_without_cov`;
`test_shutdown_closes_all`; `test_owner_lost_closes_all_lost`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `common/episodic: record builder (journal → commitment records), with tests`.

### Task 6: Query library (pure part)

**Files:**
- Create: `common/episodic/em_query.h`, `common/episodic/tests/em_query_test.cpp`; Modify: `run_tests.sh`

**Interfaces:**
- Consumes: `Record`, `Stats`.
- Produces:
  ```cpp
  struct Store { std::vector<Record> today; std::vector<Stats> stats; };
  std::vector<Record> episodes(const Store&, std::optional<std::uint64_t> since, std::optional<std::uint64_t> until,
                               std::optional<Author>, std::string_view kind = {}, std::string_view target = {});
  std::vector<Record> episode(const Store&, std::string_view id);   // the record + all descendants, pre-order
  std::vector<Stats>  stats(const Store&, std::string_view kind, std::string_view identity = {});  // "" = all
  std::optional<Record> last(const Store&, std::string_view target);   // newest t0 touching target (target or a Fact)
  std::string today_line(const Store&);   // e.g. "3 missions (2 succeeded); last: cross door_27 → room_2"
  ```
  The DSR adapter `Store load_store(DSR::DSRGraph&)` is added in Task 9 (needs the attributes).
- [ ] **Step 1: Write failing tests:** filters by author/kind/target/time; `episode` returns descendants in order;
  `last` matches a node that appears only in a Fact; `today_line` on an empty store returns `"no missions today"`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit `common/episodic: query library, with tests`.

### Task 7: Cortex attributes — USER GATE

**Files:**
- Create: `docs/superpowers/plans/2026-10-17-episodic-memory-cortex-attrs.md` (the lines to add)

- [ ] **Step 1:** Write the three lines for `core/include/dsr/core/types/type_checking/dsr_attr_name.h`:
  ```cpp
  REGISTER_TYPE(em_journal, std::string, false)  // controller agent node: episodic journal wire (em_journal.h)
  REGISTER_TYPE(em_record,  std::string, false)  // domain-4 episode node: one Record as JSON (em_schema.h)
  REGISTER_TYPE(em_stats,   std::string, false)  // domain-4 stats node: one Stats as JSON (em_consolidate.h)
  ```
- [ ] **Step 2:** Commit the note, then **STOP and ask the user to add them and reinstall cortex.** Do not `sudo cp`.
- [ ] **Step 3:** After the user confirms: `grep -n "em_journal\|em_record\|em_stats" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h`
  — expected: 3 lines.

### Task 8: Controller publishes the journal

**Files:**
- Modify: `common/mission/mission_interpreter.h` (+ `mission_interpreter_test.cpp`)
- Modify: `common/affordance_manager/affordance_manager.{h,cpp}` (+ `tests/affordance_protocol_test.cpp`)
- Modify: `controller/src/specificworker.{h,cpp}`, `controller/src/specificworker_mission.cpp`

**Interfaces:**
- Consumes: `Journal`, `JournalEntry` (Task 4); `em_journal_att` (Task 7).
- Produces:
  - `Interpreter::take_transitions() -> std::vector<StepTransition>` with
    `struct StepTransition { std::string id, op; int attempt, iteration; StepState state; std::string outcome; std::uint64_t t; };`
    one entry when an attempt/iteration starts (state Running) and one when it ends.
  - `AffordanceManager::set_claim_observer(std::function<void(const ClaimEvent&)>)` with
    `struct ClaimEvent { enum class Kind { Start, End } kind; std::string affordance, producer, target, target_type, outcome; float gain; int epoch; };`
    Start fires in `publish_executing` only when the claimed (affordance, epoch) changes; for each stale claim it
    deletes it fires End with outcome `"preempted"`; `mark_reached` fires End with `to_string(outcome)`.
  - Controller: one `rc::em::Journal journal_` (epoch = process start ms). On every accepted command in
    `run_mission_command` / `start_mission` (sources chat/remote/panel): MissionStart; single commands
    (`cross`, `move`, `turn`, `uncross`, `stop`) also push StepStart `"1"` and, at their terminal, StepEnd + MissionEnd.
    Interpreter transitions → StepStart/StepEnd; MissionHost pushes the resolved grounding (door name for a Passage
    ref, pose for an anchor) into the StepStart `text`. Claim observer → ClaimStart (with
    `affordance_manager.last_candidates()` serialised as `candidates`) / ClaimEnd. `set_mode` → DriveMode entry.
    Once per control cycle, if `journal_.dirty()`, write `em_journal` on the controller's own agent node, then clear.
- [ ] **Step 1: Failing tests:** in `mission_interpreter_test.cpp` `test_transitions_per_attempt` (on_fail retry 1,
  first attempt fails → transitions: Running a1, Failed a1, Running a2, …) and `test_transitions_per_iteration`
  (repeat times 2 → iterations 0 and 1); in `affordance_protocol_test.cpp` `claim_observer_start_preempt_outcome`
  (claim A, claim B → events Start A, End A "preempted", Start B; `mark_reached(Satisfied)` → End B "satisfied";
  republishing the same claim fires nothing).
- [ ] **Step 2:** Run `chat_agent/tests/run_tests.sh` and the affordance_manager tests build — expected FAIL.
- [ ] **Step 3:** Implement the interpreter and AffordanceManager parts. Run both — expected PASS.
- [ ] **Step 4:** Wire the controller. `cbuild` in `controller/` — expected clean build.
- [ ] **Step 5: Live check (sim):** start the fleet, run `cross door_N` via `remote_cmd.txt`, then let it wander; dump
  the controller agent node's `em_journal` (a small script or the DSR viewer) — expected: MissionStart(remote) →
  StepStart "1" → ClaimStart(s) → ClaimEnd → StepEnd → MissionEnd, then DriveMode AffordancesOnly + claims with
  `preempted`/outcome words. Save the dump to the scratchpad, not the repo.
- [ ] **Step 6:** Commit `controller: publish the episodic journal (missions, step attempts, claims, drive mode)`.

### Task 9: episodic_agent

**Files:**
- Create: `episodic_agent/episodic_agent.cdsl` (as `ltsm_agent/ltsm_agent.cdsl`, name changed), generate with
  `robocompdsl episodic_agent.cdsl .`; `episodic_agent/em_root.json` (root only, like `ltsm_root.json`);
  `episodic_agent/etc/config.toml`; `episodic_agent/src/{specificworker.{h,cpp}, dsr_fact_source.{h,cpp}, em_graph.{h,cpp}}`
- Modify: `common/episodic/em_query.h` (+ `load_store`), `CONCEPT_AGENT_RECIPE.md` (id 26)

**Interfaces:**
- Consumes: everything in `common/episodic/`; `em_journal_att`, `em_record_att`, `em_stats_att`.
- Produces: on domain 4 — under root, one node per today record (type `metaconcept`, name = record id, attr
  `em_record`), `has` edge parent→child (mission under root); one node per Stats (type `metaconcept`, name
  `stats_<kind>_<identity or "any">`, attr `em_stats`). `rc::em::Store load_store(DSR::DSRGraph&)` reads them back.
- Config: `[Agent] id = 26, name = "episodic_agent"`; `[Agent.dsr] domain 0` (no viewer), `[Agent.em] domain 4,
  configFile "em_root.json"`, `[Agent.ltsm] domain 2, configFile ""` (joined read-only for identities). Keep `dsr`
  alphabetically first (see the comment block in `ltsm_agent/etc/config.toml`). `[Presence]`: no required agents;
  `controller`, `ltsm_agent` optional. `[EM] archive_dir = "etc/episodes"`, `poll_ms = 250`.

- [ ] **Step 1:** Generate the component, adapt CMake flags from `ltsm_agent/src/CMakeLists.txt` (the `--no-as-needed`
  and Eigen-ABI block verbatim), add sources. `cbuild` — expected clean build of the empty agent.
- [ ] **Step 2:** `DsrFactSource : FactSource` over the live graph: whitelist attrs from `kFactAttrs`, pose = RT
  edge from the node's parent room (`rt_translation`/`rt_rotation_euler_xyz`), `rt_covariance` xy diagonal when
  present; `ltsm_id` = for a door, the `passage_N` in domain 2 whose `match` edge points to the door's memory face
  (see `ltsm_agent/src/passage_live.h`), else "". `current_room()` = the room the `current` edge points to.
- [ ] **Step 3:** Structural diff each poll on domain 0: node ids by type in `kEventNodeTypes` (born/died, affordance
  ⇒ offered/withdrawn), the `current` edge target (current_moved) → `Builder::on_events`.
- [ ] **Step 4:** Poll loop (timer, `poll_ms`): read `em_journal` from the `controller` agent node →
  `journal_from_wire` → `Builder::on_journal`; every returned record → `Archive::append` and `EmGraph::upsert`.
  If the `controller` agent node is absent for a poll after having been seen → `Builder::owner_lost`. At startup: `Archive::load_day(today)` → `EmGraph` nodes; load stats from the latest archive-derived fold
  (recompute: `rollover` over all past days on startup); new ids from `next_seq`. Day change on the first record of a
  new day → `rollover` → upsert stats nodes, delete folded record nodes. Destructor → `Builder::shutdown` → append.
- [ ] **Step 5:** `load_store(DSR::DSRGraph&)` in `em_query.h` (guarded `#ifdef` so the pure tests don't need DSR,
  or a separate `em_query_dsr.h` — implementer's call).
- [ ] **Step 6: Live check (sim):** fleet + episodic_agent; `cross door_N` round trip + 5 min wandering; SIGTERM the
  agent; restart it; repeat one crossing. Expected: `etc/episodes/<today>.jsonl` has nested records; after restart no
  duplicate ids, ids continue; domain-4 viewer shows today's records; no `open` record after the clean stop except
  those closed `interrupted_by_em_shutdown`.
- [ ] **Step 7:** Commit `episodic_agent: observer that records commitments to domain 4 + archive`.

### Task 10: chat_agent reads EM

**Files:**
- Modify: `chat_agent/etc/config.toml` (single-instance block → `[Agent.dsr]` domain 0 + `[Agent.em]` domain 4,
  `configFile ""`), `chat_agent/src/specificworker.{h,cpp}`, `chat_agent/src/chat_session.{h,cpp}` (tools),
  `chat_agent/src/world_digest.{h,cpp}` (one line), tool schema where `find_objects`/`where_is` are declared.

**Interfaces:**
- Consumes: `em_query.h` incl. `load_store`.
- Produces: LLM tools `episodes(since?, until?, author?, kind?, target?)`, `episode(id)`, `outcome_stats(kind,
  target?)`, `last_episode(target)`, handled in `ChatSession::answer_query`; text answers. Digest line
  `today: <today_line>`.
- [ ] **Step 1: Failing test** in `chat_agent/tests/` (pure, no DSR): formatting of each tool's answer from a fixed
  `Store`, and `"episodic memory is not available"` when the store is empty and no EM graph is joined.
- [ ] **Step 2:** Run `chat_agent/tests/run_tests.sh` — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS.
- [ ] **Step 5:** `cbuild` chat_agent; live: ask "what did you do today?" and "how often was door_N shut?" after
  Task 9's run — answers match the archive. With episodic_agent stopped, chat answers that EM is unavailable and
  everything else works.
- [ ] **Step 6:** Commit `chat_agent: episodic memory tools + today line`.

### Task 11: Acceptance report

**Files:**
- Create: `tools/em_report.py`

**Interfaces:** `python3 -I tools/em_report.py etc/episodes/<day>.jsonl [--controller-log <file>]`
- [ ] **Step 1:** Implement checks from spec §9: every claim/step has an existing parent; no record without `t1`
  except `interrupted_by_em_shutdown`/`lost` closures; ids unique (last line per id); counts per level/author/outcome;
  number of `lost` and why. Exit non-zero on a structural failure.
- [ ] **Step 2:** Run on the Task 9 archive and on a full `room_concept/estimation_scripts/scripts/round_trips.sh`
  day + wandering; also run the fleet with and without episodic_agent and compare round-trip success/time
  (spec §9: no measurable change).
- [ ] **Step 3:** Commit `tools: episodic memory acceptance report`; record the result (numbers) in the spec's §9 as a
  dated note.
