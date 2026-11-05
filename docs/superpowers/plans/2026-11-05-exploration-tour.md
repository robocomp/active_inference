# Exploration tour from scratch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An autonomous controller drive that, from an empty memory, explores each room until it is promoted (or a max time), then crosses the nearest found-but-uncrossed door, else returns through the door the room was first entered by — plus the tour script and the grading for the paper.

**Architecture:** A PURE decision core (`controller/src/exploration_drive.h`: inputs → decision, plus a small DFS memory of each room's first entry door) unit-tested on its own; `SpecificWorker` builds the inputs each cycle from the graph (`door_facts()`, `rc::room::is_proto`, the `ltsm_topology` JSON) and acts through the existing `start_door_passage(door, Step::Cross, "explore", true)`. Enabled by a `remote_cmd.txt` verb. The harness is `explore_spike.sh` grown into `explore_tour.sh`.

**Tech Stack:** C++23, Qt JSON (decode, as language_agent), DSR; bash + Python 3 for the harness/grader.

**Spec:** `docs/superpowers/specs/2026-11-05-exploration-tour-design.md` (decisions 1–8 and the "Calls made" section are binding). Spike result: crossing-born rooms promote after 1a8b4814 (explore_spike3).

## Global Constraints

- Rooms' planners and door_concept's look valuation unchanged; no scripted list of doors/rooms anywhere.
- Crossing only through the existing door passage (`start_door_passage`), never by letting the open contest pick `cross`.
- "Promoted" = the room node has no `proto` self-edge (`rc::room::is_proto(G, room_id) == false`).
- "Found" = a node `aff_<door>_approach` exists (door_concept's own offer). "Crossed" = live `DoorFacts::crossed` OR the `ltsm_topology` holds a passage with `crossings >= 1` whose `live_door`, `a.face` or `b.face` equals the door's name (survives restarts).
- Max-time fallback: config `[ExploreTour] MaxRoomS = 300` (flagged in the config comment as the one deliberate time limit, decision 5); every leave logs its cause.
- No thresholds beyond that; `and/or/not`; type-attributed DSR attrs; check every `get_transformation_matrix` optional; never `Qt::DirectConnection`.
- Builds: `cbuild` (-j8); check `pgrep -a cc1plus`. Commits on `feat/room-recognition` (rt-build); do not push. Warn active-inference-cc / language before Webots.

## Review Focus

1. A room with NO door found and promoted, in the START room (no entry door) → `Done`, not a crash or a passage to "" — Task 1 test `start_room_nothing_left_is_done`.
2. The door the robot ENTERED by must never be a "new" candidate (it is crossed; also excluded explicitly) — Task 1 test `entry_door_never_a_candidate`.
3. A return into a room already promoted must choose again at once (decision 8), using that room's FIRST entry door for a further return (DFS), not the door just used — Task 1 test `dfs_return_uses_first_entry_door`.
4. A passage running or just failed: the drive must not start a second one while `passage_active`; a failed crossing must not loop on the same door for ever — Task 1 test `failed_door_is_not_retried_immediately` (the drive remembers doors whose passage FAILED in this visit and skips them; logged).
5. `ltsm_topology` absent or malformed JSON (ltsm not up yet) → treat as "no remembered crossings", never block — Task 2 test `topology_missing_is_empty`.

---

### Task 1: The pure decision core

**Files:**
- Create: `controller/src/exploration_drive.h`
- Test: `controller/tests/test_exploration_drive.cpp` (header comment with its g++ line, `-Isrc`, no DSR)

**Interfaces:**
- Produces:
  ```cpp
  namespace rc::explore {
  struct DoorCand { std::string door; bool found = false; bool crossed = false; double dist_m = INFINITY; };
  enum class Cause { None, Promoted, MaxTime, Return, Done };
  struct Decision { Cause cause = Cause::None; std::optional<std::string> door; bool is_return = false; };
  [[nodiscard]] const char* to_string(Cause);
  class Drive {
  public:
      explicit Drive(double max_room_s);
      // one cycle; `now_s` monotonic seconds
      Decision step(const std::string& room, bool promoted, const std::vector<DoorCand>& doors,
                    bool passage_active, double now_s);
      void note_entered(const std::string& room, const std::string& via_door, double now_s); // after a crossing
      void note_failed(const std::string& door);                                            // passage failed
      [[nodiscard]] std::optional<std::string> entry_door(const std::string& room) const;   // FIRST entry (DFS parent)
  };
  }
  ```
  Rules (spec §B): `passage_active` → None. Room first seen by `step` starts its clock (start room: entered at its first step, no entry door). Not promoted and `now - entered < max_room_s` → None. Otherwise: candidates = found ∧ ¬crossed ∧ ≠ entry door ∧ ¬failed-this-visit; nearest by `dist_m` → {Promoted|MaxTime, door}. None and an entry door exists → {Return, entry door, is_return}. Else → Done. `note_entered` keeps the FIRST via_door per room only — never for the START room (the room of the first `step`, the DFS root: the ring R1→R2→R3→R1 re-enters it through DOOR_0 and must not give it an entry door) — and resets that room's clock and failed set.

- [ ] **Step 1: Write the failing tests** — `waits_while_unpromoted` (promoted=false, 10 s < 300 → None); `max_time_leaves` (unpromoted, 301 s, one candidate → MaxTime + door); `promoted_picks_nearest_uncrossed_found` (doors a{found,!crossed,3 m}, b{found,!crossed,1 m}, c{!found,0.5 m}, d{found,crossed,0.2 m} → Promoted, "b"); `entry_door_never_a_candidate` (note_entered("room_2","door_7"); door_7 found & !crossed → not chosen, Return via door_7 when nothing else); `start_room_nothing_left_is_done`; `dfs_return_uses_first_entry_door` (enter room_2 via d1, room_3 via d2; return to room_2 via d3 (its mirror) → note_entered("room_2","d3") keeps d1; room_2 promoted, no candidates → Return "d1"); `failed_door_is_not_retried_immediately`; `passage_active_holds`; `start_room_never_gets_entry_door` (first step in room_1; later note_entered("room_1","door_0") → entry_door("room_1") is nullopt, and room_1 promoted with no candidates → Done).
- [ ] **Step 2: Build+run** (`g++ -std=c++23 -O1 -Isrc tests/test_exploration_drive.cpp -o /tmp/ted && /tmp/ted`), expect compile failure.
- [ ] **Step 3: Implement** `exploration_drive.h` (header-only).
- [ ] **Step 4: Run** → all pass.
- [ ] **Step 5: Commit** `controller: exploration drive decision core (leave on promotion/max time, nearest uncrossed door, DFS return)`.

### Task 2: Remembered crossings from `ltsm_topology`

**Files:**
- Create: `common/ltsm_topology/topology_decode.h` — MOVE `rc::chat::TopoRoomInfo`, `TopoPassageInfo`, `Topology`, `decode_topology` from `language_agent/src/topology.h` into namespace `rc::topo` (language_agent's header then `#include`s it and keeps `using` aliases so nothing else there changes).
- Modify: `language_agent/src/topology.h`
- Test: `controller/tests/test_topology_crossed.cpp`

**Interfaces:**
- Produces: `[[nodiscard]] std::set<std::string> rc::topo::crossed_doors(std::string_view json);` — the names (`live_door`, `a.face`, `b.face`) of every passage with `crossings >= 1`; empty on absent/malformed JSON.
- [ ] **Step 1: Failing tests** — `topology_missing_is_empty` (""/"{" → empty); `crossed_faces_and_live_door` (one passage crossings 2 faces door_3/door_9 live_door door_9, one with crossings 0 → {door_3, door_9} only).
- [ ] **Step 2: Run, expect FAIL.**  **Step 3: Implement** (move + `crossed_doors`). **Step 4:** run it AND language_agent's existing topology tests (find them with `grep -rln decode_topology language_agent/tests`) → pass; `cbuild` language_agent.
- [ ] **Step 5: Commit** `common: ltsm topology decode shared; crossed_doors for the exploration drive`.

### Task 3: Wire the drive into the controller

**Files:**
- Modify: `controller/src/specificworker.h` (member `std::optional<rc::explore::Drive> explore_drive_;`, `std::ofstream explore_csv_;`), `controller/src/specificworker.cpp` (remote verbs; per-cycle tick next to where the door passage's `take_finished()` is handled), `controller/etc/config.toml` (`[ExploreTour] MaxRoomS = 300`)
- Test: the two pure tests above stay green; integration is live (Task 5)

**Interfaces:**
- Consumes: Task 1 `Drive`, Task 2 `crossed_doors`, existing `door_facts()`, `rc::room::is_proto`, `start_door_passage`.
- Remote verbs: `explore_tour on` (creates the drive, sets `DriveMode::AffordancesOnly`, opens `etc/exploration_drive.csv` through `imbue(std::locale::classic())`), `explore_tour off`.
- Per cycle while on: build `DoorCand`s from `door_facts()` (found = node `aff_<name>_approach` exists; crossed = `f.crossed or crossed_doors(ltsm_topology json).contains(name)`); `passage_active = session_.door_passage().active()`; call `step`. On a door decision → `start_door_passage(door, Step::Cross, "explore", true)` and remember it as in-flight; when the passage finishes in a new room → `note_entered(new_room, entry_mirror)` where the entry mirror is the door of the new room `match`-linked to the crossed door through a live `passage_*` node (fallback: the nearest door of the new room, logged); a failed passage → `note_failed(door)`. On `Done` → log `[explore-tour] DONE` and switch the drive off.
- CSV, one row per non-None decision: `t_ms,room,promoted,cause,door,is_return,candidates` with candidates `name:found:crossed:dist;...`. Stdout line `[explore-tour] <room>: <cause> -> <door|->` for each.
- [ ] **Step 1:** implement; `cbuild` controller (-j8) → builds clean.
- [ ] **Step 2:** re-run Task 1/2 tests + `test_door_passage` (expect only its pre-existing l.1009 failure) + `test_motion_verdict`.
- [ ] **Step 3: Commit** `controller: explore_tour remote verb drives the exploration (spec 2026-11-05)`.

### Task 4: Tour script + grader

**Files:**
- Create: `room_concept/estimation_scripts/scripts/explore_tour.sh` (from `explore_spike.sh`: same lock, archive-then-clear memory, MapMode=estimate restore-on-exit, fleet start; then `run` + `explore_tour on`; wait for `[explore-tour] DONE` or `RUN_LIMIT_S` (default 3600); copy `controller/etc/exploration_drive.csv`; collect; run the grader)
- Create: `room_concept/estimation_scripts/scripts/grade_explore_tour.py`
- Commit `explore_spike.sh` too (it is the spike's record)

**Interfaces:**
- Grader input: run dir. Output `explore_report.txt` + `explore_rooms.csv`: per room — entered (s), promoted (s, from `proto_room.csv` `promoted` events), leave cause (from `exploration_drive.csv`), doors found (first sighting of `aff_<door>_approach` in `door_concept.log` OFFERED lines), crossings and returns; true room per crossing from `true_room.py`'s logic on `gt_pose.tsv`; IoU of each promoted outline vs `wbt_rooms.py`'s polygon for the true room (rasterised on a 5 cm grid; frames: the promoted outline is in the room's published frame — map it to world through the gt-vs-estimate alignment `gt_error.csv` already uses; if that alignment is not available for a room, report IoU as `n/a` and say why).
- [ ] **Step 1:** grader on `crossing/explore_spike3` (a real run) → prints room_1/room_2 rows with promotion 53 s / 630 s; commit both scripts.

### Task 5: Live validation

- [ ] Warn the peers. `bash scripts/explore_tour.sh explore_tour1` (and `explore_tour2`). Pass: all three rooms promoted, DOOR_0/1/2 each crossed at least once, ends with `[explore-tour] DONE`, every leave cause logged, rates nominal (room 20, controller ≥ 17, retina 40), no manual help. Report the grader tables and any max-time leaves. Free Webots; update memory `exploration-tour-from-scratch-11-05`.
