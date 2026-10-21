# door_concept DSR-free Core with Record/Replay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move door_concept's per-cycle decision logic into a deterministic, DSR-free `DoorCore` driven by a recorded
`DoorCycleInput`, so real live runs can be replayed offline and door defects fixed against many recordings at once.

**Architecture:** First instrument TODAY's agent (decision log + full input snapshot + recorder) and record baseline runs.
Then extract the logic slice by slice into `door_concept/src/core/` behind small injected interfaces; `SpecificWorker`
becomes the adapter. The acceptance test is replay equivalence: baseline recordings made with the OLD code replayed
through `DoorCore` reproduce the OLD decision logs.

**Tech Stack:** C++23, Eigen, standalone g++ test/tool programs (door_concept's existing style), `cbuild -j8`.

**Spec:** `docs/superpowers/specs/2026-10-21-door-core-replay-design.md` (approved 2026-10-21) — read §3 and §6.

## Global Constraints
- CLAUDE.md: no `Qt::DirectConnection`; type-attributed DSR access (adapter only); `std::from_chars`; data files written
  with `imbue(std::locale::classic())`; `cbuild -j8` after `pgrep -a cc1plus`; stop agents with SIGTERM only.
- `door_concept/src/core/` contains NO `#include` of DSR, Qt, Ice, media_transport or any graph header (enforced by a test
  that greps the directory).
- The core reads no clock and no unseeded randomness: time comes in `DoorCycleInput`.
- Behaviour-preserving: no decision change in Tasks 3–7; equality checked in Task 8 (bitwise, or relative 1e-6 where float
  evaluation order unavoidably changes — stated per field).
- Recording budget ~2 GB per tour; at the cap recording STOPS with a log line.
- Live tours are run by the controller session (Claude) with `three_rooms_tour.sh`; recordings are copied into the run
  folder on `collect`.

## Review Focus
1. **A foreign graph value read mid-cycle after another agent changed it** (old code reads lazily, snapshot reads at cycle
   start) — Task 8 must report any such mismatch as a named field, not hide it in a tolerance.
2. **Events between cycles** (node-deleted slots, actuator acks) delivered in a different order than the old code saw them
   — Task 3 latches them with timestamps; Task 8 replays them in that order.
3. **Recorder at the size cap mid-room-switch** — the file must stay readable (record boundaries, final flush on SIGTERM).
4. **Locale**: the recorder/replay under `LANG=es_ES.UTF-8` (Qt sets it) — round-trip test runs with `setlocale(LC_ALL,"")`.
5. **Identity table load before graph sync** — the load event and its content are in the recording (Task 3), replayed
   exactly (Task 9 name-drift case).

---

### Task 1: Decision log in today's door_concept
**Files:** Create `door_concept/src/decision_log.h`; modify `door_concept/src/specificworker.cpp` (emit at the existing
decision points: tracker BIRTH/SUPPRESS/MERGE/REMOVE, existence L/p per door per cycle, φ estimate per door, identity
writes, room switch carry/restore/look decisions, affordance OFFERED/WITHDRAWN/EXECUTING/COMPLETED, anchor HUNG/REMOVED),
`door_concept/etc/config.toml` (`DecisionLog = false`, path).
**Interfaces — Produces:** `struct DoorDecision { uint64_t cycle; int64_t t_ms; uint16_t kind; uint64_t door_id; std::string name; double a, b, c, d; std::string note; };`
binary writer + reader (`DoorDecisionLog::open/append/read_all`), versioned header "DDEC" v1.
- [ ] Failing test `tests/test_decision_log.cpp`: write N decisions under es_ES locale, read back identical.
- [ ] Implement; wire the emit points; default OFF. Build (`cbuild -j8`), all door tests pass. Commit.

### Task 2: The input/output types and their serialization
**Files:** Create `door_concept/src/core/door_cycle_io.h/.cpp` (types only + versioned binary (de)serialization),
`tests/test_door_cycle_io.cpp`.
**Interfaces — Produces:** `DoorCycleInput` (fields per spec §3.1, incl. `std::vector<DoorEvent> events` and an optional
`IdentityLoad` event), `DoorCycleOutput`, `write(std::ostream&, const DoorCycleInput&)`, `read(std::istream&) ->
std::optional<DoorCycleInput>`, same for output; header "DCIN" v1.
- [ ] Enumerate every graph/media read in compute() and in slots called between cycles (file:line list in the report) —
  this list defines the fields. Failing round-trip test (es_ES locale, random-filled inputs, partial-file read).
- [ ] Implement. Test passes; `core/` include-grep test passes. Commit.

### Task 3: Snapshot + event latching + recorder in today's agent
**Files:** Create `door_concept/src/input_snapshot.{h,cpp}` (adapter: graph/media → `DoorCycleInput` at the start of
compute(); latches between-cycle events), `door_concept/src/input_recorder.h`; modify `specificworker.cpp`,
`config.toml` (`RecordInputs = false`, path, `RecordMaxMB = 2000`); `run_crossing.sh collect` copies `etc/door_inputs_*.bin`
and `etc/door_decisions_*.bin` into the run folder.
- [ ] Failing test: a synthetic sequence of snapshots + events written and re-read in order; cap reached → stops cleanly.
- [ ] Implement (the old logic still reads the graph; the snapshot is recorded alongside). Build, tests, commit.
- [ ] LIVE: run `three_rooms_tour.sh tourR1` and `tourR2` with `RecordInputs = true` and `DecisionLog = true` (WT config);
  record the run ids, file sizes, and whether door_2 was re-born and removed (the tourP10 case) in the ledger.

### Task 4: DoorCore skeleton + tracker slice
**Files:** Create `door_concept/src/core/door_core.{h,cpp}` (`class DoorCore { public: explicit DoorCore(const DoorCoreConfig&);
DoorCycleOutput step(const DoorCycleInput&); }`), interfaces `core/sensing.h` (`MaskSource`, `CameraModel`, `LeafSweep`
views over the input); move birth/association/merge (`specificworker.cpp:1197-1790`) and the parts of `DoorFitter` it
needs, reading only from `DoorCycleInput`. Adapter calls the core for this slice and applies its outputs.
- [ ] Failing test `tests/test_door_core_tracker.cpp`: synthetic input with a mask packet at a door → BIRTH decision; a
  second packet associates; core include-grep passes.
- [ ] Implement; door tests + new test pass; build; LIVE smoke one leg (decision log shows births as before). Commit.

### Task 5: Existence slice
Move `update_existence_beliefs` (`specificworker.cpp:2144-2848`) into the core (silhouette, contour/depth evidence from
the input's derived evidence, P(detect), removal). Test: synthetic existence sequence → same L trajectory as a recorded
`door_existence_log.csv` fragment (state the source). Build, tests, live smoke, commit.

### Task 6: Identity, room switch, carry, restore, look slice
Move identity match/upsert (`:1911-2043`), room following/release/carry/restore/look (`:3747-4415`, `door_look.h` use) into
the core; identity file I/O stays in the adapter (load → `IdentityLoad` event; core emits table writes). Tests: the
existing identity/look tests run against the core; a room-switch sequence test (release → recall → restore). Build,
tests, live smoke, commit.

### Task 7: Affordances + anchor slice; adapter only
Move pragmatic/epistemic affordance decisions (`:3224…`, `door_pragmatics.h`) and the anchor step
(`specificworker_anchor.cpp:475-1062`) into the core; the adapter only translates outputs to graph writes / actuator
calls. `SpecificWorker` holds no decision logic (state the remaining line count). Tests: existing anchor tests through the
core. Build, tests, live smoke, commit.

### Task 8: Replay tool and equivalence
**Files:** Create `door_concept/tools/door_replay.cpp` (read `door_inputs_*.bin`, run `DoorCore`, write a decision log;
`--diff <decisions.bin>`, `--until <cycle>`, `--trace <door>`), `tests/test_replay_equivalence.cpp` (runs the tool on the
tourR1/R2 recordings when present, SKIP otherwise).
- [ ] Replay tourR1/R2 → decision streams equal to the recorded OLD decision logs (bitwise; or per-field 1e-6 where
  justified). Every mismatch is a named field with its cause; fix the core until equal (or document an irreducible
  mid-cycle foreign-read difference, Review Focus 1). Commit.

### Task 9: Regression cases
Tests on recorded data: (a) the tourP10 false removal (door re-born then removed with silhouette free≫occ) — committed as
EXPECTED FAILURE until its fix task; (b) a restored door after recall gets its remembered pose and affordances; (c) the
main door keeps its name across a recall (no door_4). Use the tourR1/R2 recordings (record more tours if a case is not
present — state which run). Commit.

### Task 10: Live check after the extraction
Run one three-room tour with recording ON; compare its decision stream behaviour with tourR1/R2 (same failure classes, no
new ones); record in the ledger. Then the door defects are fixed one by one against the recordings (separate plan).
