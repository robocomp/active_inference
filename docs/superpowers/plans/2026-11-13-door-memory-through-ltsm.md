# Door memory through LTSM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** door_concept gets remembered doors back from LTSM (notes posted on recall) instead of its private
`door_identities.csv`, and keeps only refuted places ("phantoms") in a private per-agent file keyed by LTSM room_id.

**Architecture:** door_concept publishes the belief that today lives only in the CSV as typed attributes on its own door
nodes; LTSM archives them for free (`RoomEviction::twin` copies all attrs). A new stateless LTSM step posts one
`remembered_door` metaconcept per remembered door face of the current room; door_concept turns each note into the
`IdentityRow` its unchanged restore core already consumes, marks it adopted, and LTSM deletes adopted notes. Refuted
places move to `common/phantom_memory`. Everything new on the door side runs behind `DoorConcept.RestoreFromLtsm`.

**Tech Stack:** C++23, cortex DSR (type-attributed attrs), Eigen, existing door_concept/ltsm_agent test harnesses (CTest).

**Spec:** `docs/superpowers/specs/2026-11-13-door-memory-through-ltsm-design.md` (APPROVED 2026-11-13, §8 decisions).

## Global Constraints

- No thresholds/gates/clamps; use model terms (CLAUDE.md). The only clamp allowed is the existing numerical one in `leafmem::logit` (±13.8 nat).
- Type-attributed attribute access only (`<foo_att>`), never `runtime_checked_*`.
- New attributes: `REGISTER_TYPE` in cortex `core/include/dsr/core/types/type_checking/dsr_attr_name.h`; **the owner reinstalls cortex** — never `sudo cp`.
- DSR signals: none new; if one is needed, Queued only.
- Files: write with `imbue(std::locale::classic())`; read numbers with `std::from_chars` only. Tests that parse files call `std::setlocale(LC_ALL, "")`.
- `and`/`or`/`not`, C++23 ranges where natural.
- Build with `make -C build -j8` after `pgrep -a cc1plus` is empty (or `cbuild`). Clean reconfigure after adding sources.
- Stop agents with SIGTERM only. Webots is shared: claim it, send FREE to peers after.
- Notes are type `metaconcept`, never `door`/`object` (door_concept's owned-node sweep deletes those by name).
- Commit only when the owner says so; the "Commit" steps below are prepared commits, run on the owner's go.
- Work in a new worktree `active_inference.worktrees/door-memory`, branch `feat/door-memory-ltsm` from `main` (f0a4c051).
  rt-build carries the chat session's uncommitted restored-door fix in the same files; do not mix them.

## Review Focus

1. **A recalled room whose LTSM notes arrive AFTER door_concept's outline is ready.** Expect: the doors appear the cycle the notes land, not never (restore is driven by notes each cycle, not by a one-shot `pending_restore_room_`). Test in Task 4.
2. **A note for a face whose hole a live door already holds** (the door just crossed, the entry mirror). Expect: skipped by place (`plan_restore` PlaceLive), note still marked adopted so it doesn't repost forever. Test in Task 4.
3. **Leaf log-odds above the float-probability range** (CSV holds hinge_llr up to 95). Expect: restored as 13.8 nat (the `logit` clamp), MAP side preserved. Test in Task 1.
4. **Recognition swap: the provisional room's phantoms.** Expect: dropped, as `drop_room` does today (their frame dies with the provisional room). Test in Task 5.
5. **A room LTSM forgot** (`room_forget`). Expect: its phantom rows are dropped on the next save, and no note is posted for it. Test in Task 5.

---

### Task 0: cortex attribute types (owner reinstall)

**Files:**
- Modify: `/home/pbustos/robocomp/cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h` (next to `door_leaf_hinge_plus_prob`, ~line 885) — confirm the cortex source path with `locate dsr_attr_name.h | grep -v /usr/local`.

- [ ] **Step 1: Check none exist already**

Run: `grep -n "door_exist_\|door_crossed\|door_seen_ms\|door_sighted_ms\|door_place_sd\|door_height_sd\|door_phi_sd\|door_view_ledger\|door_visit\|door_adopted" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h`
Expected: no output.

- [ ] **Step 2: Add, with a one-line comment each (who writes, what it means):**

```
REGISTER_TYPE(door_exist_evidence, float, false)          // door_concept: PlaceBelief L_evidence (log-odds), the existence memory
REGISTER_TYPE(door_exist_physical, float, false)          // its physical part (actuation, crossing)
REGISTER_TYPE(door_crossed,        bool,  false)          // the place was crossed: proof for ever
REGISTER_TYPE(door_seen_ms,        uint64_t, false)       // wall ms, last detection (age of door_exist_evidence)
REGISTER_TYPE(door_sighted_ms,     uint64_t, false)       // wall ms, last SIGHTING (hazard exposure)
REGISTER_TYPE(door_place_sd,       float, false)          // aperture centre sigma along its wall (m)
REGISTER_TYPE(door_height_sd,      float, false)          // fit posterior sd of height (m)
REGISTER_TYPE(door_phi_sd,         float, false)          // leaf-angle sigma (rad), pairs with door_phi_rad
REGISTER_TYPE(door_view_ledger,    std::reference_wrapper<const std::string>, false)  // ViewLedger::serialise()
REGISTER_TYPE(door_visit,          int,   false)          // place-table visit counter
REGISTER_TYPE(door_adopted,        bool,  false)          // on a remembered_door note: door_concept has restored it; ltsm deletes it
```

Leaf side reuses `door_leaf_into_room_prob` / `door_leaf_hinge_plus_prob` / `door_phi_rad`; geometry reuses `width_m`/`height_m`/`depth_m` + the RT edge. No new type for the note (`metaconcept`).

- [ ] **Step 3: Owner reinstalls cortex.** Verify: the grep of Step 1 now prints 11 lines.

---

### Task 1: the note codec (pure, DSR-free)

**Files:**
- Create: `door_concept/src/door_memory_note.h`
- Test: `door_concept/tests/test_door_memory_note.cpp` (+ `add_door_test(...)` line in `door_concept/tests/CMakeLists.txt`, same pattern as `test_door_leaf_memory`)

**Interfaces:**
- Consumes: `IdentityRow`, `RestorePose`, `LeafMemory` (`door_identity_table.h`); `rc::door::leafmem::{p_into_room, p_hinge_plus, swing_from_into_room, hinge_from_plus, logit}` (`door_leaf_memory.h`).
- Produces:

```cpp
namespace rc::door::memnote {
struct NoteAttrs {                       // exactly what a door node / note carries
    Eigen::Vector2f xy; float yaw;       // the RT edge room -> node (room frame)
    float width, height, depth;          // width_m / height_m / depth_m
    std::optional<float> exist_evidence, exist_physical, place_sd, height_sd, phi, phi_sd;
    std::optional<float> leaf_into_room_prob, leaf_hinge_plus_prob;
    std::optional<bool> crossed;
    std::optional<std::uint64_t> seen_ms, sighted_ms;
    std::optional<std::string> ledger;
    std::optional<int> visit;
};
NoteAttrs note_of(const IdentityRow& r, std::optional<float> height_sd);
IdentityRow row_of(const NoteAttrs& a, std::string room, std::string name);
std::string face_live_name(std::string_view note_name);   // "remembered_door_5" -> "door_5"
}
```

`note_of` takes a row that has `pose` (a door node always has geometry). Leaf: `LeafMemory.hinge_xy` = xy ± (width/2)·(cos yaw, sin yaw) on the MAP hinge side; `open_dir` = ± the normal (−sin yaw, cos yaw) on the MAP swing side, signs per the conventions in `door_leaf_memory.h` header (+x = Far; swing +1 = into the room). Decode with `swing_from_into_room` / `hinge_from_plus` so the sign rules live in one place.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(memnote, row_round_trip_every_field)      // row -> note -> row: xy, yaw, w/h/d, exist_logodds, exist_physical,
                                               // crossed, seen_ms, sighted_ms, place_sd, ledger, visit equal;
                                               // leaf hinge_xy/open_dir within 1e-5 m, phi/phi_sigma equal
TEST(memnote, leaf_llr_beyond_float_range)     // hinge_llr 95, swing_llr 70 -> restored |llr| == logit(1-1e-6) (13.8), MAP sides kept
TEST(memnote, absent_fields_stay_absent)       // a note without ledger/visit/seen_ms -> row with nullopt there (row_of of PlaceTable then
                                               // uses its documented fallbacks)
TEST(memnote, removed_never_encoded)           // note_of ignores removed_ms; row_of never sets it
TEST(memnote, face_live_name_strips_prefix)    // "remembered_door_5" -> "door_5"
```

- [ ] **Step 2:** `make -C door_concept/build -j8 test_door_memory_note && ctest --test-dir door_concept/build -R memory_note` → FAIL (header missing).
- [ ] **Step 3:** Implement `door_memory_note.h`.
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5: Commit** `door_concept: a door's memory as node attributes (codec)`.

---

### Task 2: door_concept publishes the belief on its door nodes

**Files:**
- Modify: `door_concept/src/door_scene_graph.h/.cpp` — new `void write_memory_state(DSR::Node&, const memnote::NoteAttrs&)` next to `write_interaction_state` (~line 547).
- Modify: `door_concept/src/specificworker.cpp` — where a live door's identity row is upserted (`grep -n "upsert_row" specificworker.cpp`), build `memnote::note_of(row, height_sd)` and call `write_memory_state` on that door's node in the same `update_node` the interaction state uses (no new timer, no extra `update_node`).
- Test: `door_concept/tests/test_door_memory_note.cpp` (add case).

**Interfaces:** Consumes Task 1 `note_of`. Produces: on every live door node, the 10 Task-0 attrs (all but `door_adopted`) + `door_phi_sd`.

- [ ] **Step 1: Failing test** `TEST(memnote, written_attrs_read_back)`: on a standalone `DSR::DSRGraph` (copy the graph fixture of `test_door_card` / `door_card_test.cpp`), `write_memory_state` then read each `<..._att>` back equal to the NoteAttrs. Optional fields absent ⇒ attribute not written (not a 0).
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** run → PASS; whole door suite `ctest --test-dir door_concept/build` → no new failures (baseline: 4 pre-existing fails, list them before starting).
- [ ] **Step 5:** Live smoke (Webots claimed): one bedroom-start explore_tour; at the end `ltsm_agent` memory file holds `door_exist_evidence` on every `r*_door_*` face (`grep -c door_exist_evidence <ltsm memory json>` ≥ doors seen). FREE Webots.
- [ ] **Step 6: Commit** `door_concept: publish the existence memory on the door node (ltsm archives it)`.

---

### Task 3: LTSM posts and retires `remembered_door` notes

**Files:**
- Create: `ltsm_agent/src/remembered_doors.h/.cpp` (class `RememberedDoors`, modelled on `RoomRecall`).
- Modify: `ltsm_agent/src/specificworker.cpp/.h` — construct it next to `recall_`, call `step()` in the same cycle as `recall_->step()`, `remove_all()` in the start-up sweep (~line 98) and on graceful exit; config flag `Recall.remembered_doors` (default true; it only adds metaconcept nodes) via the existing `flag(...)` helper.
- Modify: `ltsm_agent/src/CMakeLists.txt` (source), `ltsm_agent/CMakeLists.txt` tests list.
- Test: `ltsm_agent/tools/remembered_doors_test.cpp` (fixture style of `room_swap_test.cpp`: a live and a memory `DSRGraph`).

**Interfaces:**

```cpp
namespace ltsm {
class RememberedDoors {
public:
    RememberedDoors(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory);
    int step();        // returns notes live after the cycle
    int remove_all();  // every metaconcept with object_subtype "remembered_door"
};
}
```

`step()`:
1. `room = rc::room::current_room(*live_)`; none ⇒ retire all, return.
2. Memory twin = memory room named `std::format("r{}_{}", room_id, room.name())`; absent or `is_staged_room` ⇒ retire all (a pending recognition is not yet the robot's room).
3. For each memory door face under the twin (memory `object` nodes named `r<k>_door_*` whose parent is the twin): if a note `remembered_<face live name>` exists and carries `door_adopted == true` ⇒ delete it and remember the face id in `adopted_` (never repost while this room stays current); else create/refresh it: type `metaconcept`, `object_subtype = "remembered_door"`, `parent` = live room id, `level` = room level + 1, `room_id`, every attribute of the face except `parent`/`level`/RT (same copy rule as `RoomEviction::twin`), RT edge room → note = face's RT in memory (`RoomEviction::rt_attributes`).
4. Retire every remembered_door note whose parent is not the current room. Clear `adopted_` when the current room changes.

- [ ] **Step 1: Failing tests**

```cpp
TEST(remembered_doors, posts_one_note_per_face_of_current_room)   // 3 faces -> 3 notes, RT == face RT, door_exist_evidence copied
TEST(remembered_doors, staged_twin_posts_nothing)
TEST(remembered_doors, adopted_note_deleted_not_reposted)          // set door_adopted, step, step -> 0 notes
TEST(remembered_doors, room_change_retires_notes)
TEST(remembered_doors, notes_are_metaconcept_never_object)
```

- [ ] **Step 2:** `make -C ltsm_agent/build -j8 remembered_doors_test && ./ltsm_agent/build/remembered_doors_test` → FAIL.
- [ ] **Step 3:** implement. **Step 4:** → PASS; all ltsm tools tests that passed on f0a4c051 still pass.
- [ ] **Step 5: Commit** `ltsm: a recalled room's doors are posted as remembered_door notes`.

---

### Task 4: door_concept restores from notes (behind the flag)

**Files:**
- Modify: `door_concept/src/specificworker.cpp` (`enter_room_memory` ~3979, `step_pending_restore` ~4030, `restore_remembered_doors` ~4044), `specificworker.h`, config loader (`cfg_.restore_from_ltsm`, key `DoorConcept.RestoreFromLtsm`, default `false`; add to `door_concept/etc/config.toml` commented with the spec path).
- Create: `door_concept/src/door_note_reader.h` — `std::vector<NoteInput> read_notes(const DSR::DSRGraph&, std::uint64_t room_id)` returning `{note id, memnote::NoteAttrs, name}` (reads RT room→note, the attrs; skips notes with `door_adopted`).
- Test: `door_concept/tests/test_door_restore_from_notes.cpp`.

**Behaviour with the flag on:**
- Every cycle (after the outline check of `step_pending_restore`, which stays), unadopted notes under `room_node_id_` become rows via `memnote::row_of(attrs, room->name(), face_live_name(note name))` and go through the existing `plan_restore` → `restored_belief` → `restore_leaf_phi` → `create_door_copy` → `seed_restored_instance` path. Refactor: split `restore_remembered_doors` into `rows_to_restore()` (CSV rows or note rows) + the existing loop taking `std::span<const IdentityRow>`; no logic change in the loop.
- Each note handled (restored OR skipped PlaceLive) gets `door_adopted = true`. A note skipped as `Removed` cannot happen (notes never carry removed_ms) — phantoms veto through the place table (Task 5).
- With the flag on, `enter_room_memory`'s FRESH strip and `drop_room` still run on the CSV (harmless; deleted in Task 6).

- [ ] **Step 1: Failing tests** (pure part: rows from notes into `plan_restore`)

```cpp
TEST(restore_notes, note_rows_equal_csv_rows)        // same door via CSV row and via note -> identical restored_belief (L, physical, ledger) and aged phi
TEST(restore_notes, place_live_note_is_skipped)      // live door within same_hole -> skipped, reported for adoption
TEST(restore_notes, late_notes_restore_when_they_land)  // outline ready, 0 notes -> nothing; then 2 notes -> 2 restored
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** → PASS; door suite no new failures.
- [ ] **Step 5: Live A/B** (Webots claimed; spec §7): 3 explore_tours with the flag off, 3 with it on, one per start room, worktree builds. Per tour: on each re-entry of a remembered room, every remembered door exists within one cycle of the outline (door_concept.log `[restore]` lines vs ltsm.log note posts), 0 duplicate births on restored holes, grade with `grade_explore_tour.py` (run inside `crossing/`). Pass = flag-on results no worse than flag-off and than clean62–76 (13/15). Disk check ≥ 15 GB first. FREE Webots.
- [ ] **Step 6: Commit** `door_concept: remembered doors come back from ltsm notes (DoorConcept.RestoreFromLtsm)`.

---

### Task 5: private phantom memory

**Files:**
- Create: `common/phantom_memory/phantom_memory.h` (header-only, Eigen + `<charconv>`; no DSR).
- Modify: `door_concept/src/specificworker.cpp` — load at start; on room entry inject the rows of the room's `room_id` into the place table as removed rows (`PlaceRow.removed_wall_ms`, `height_sd`); on save write every removed place row of a room with known room_id; on recognition swap drop the provisional room's rows (as `drop_room`).
- Modify: `common/ltsm_topology/topology_decode.h` + `ltsm_agent/src/topology_publish.cpp` ONLY if `TopoRoomInfo.name` does not carry the `r<k>_` prefix: add `std::uint64_t room_id` to `TopoRoomInfo` (check first by printing a live `ltsm_topology` JSON).
- Test: `common/phantom_memory/tests/test_phantom_memory.cpp` (+ door_concept CMake adds it).

**Interfaces:**

```cpp
namespace rc::phantom {
struct Row { std::uint64_t room_id; float x, y, yaw, width, height, height_sd; std::int64_t removed_ms; };
struct Memory {
    std::vector<Row> rows;
    static std::optional<Memory> load(const std::filesystem::path&);   // missing file -> empty Memory; bad row -> skipped
    bool save(const std::filesystem::path&) const;                       // imbue(classic), atomic rename
    std::vector<Row> of_room(std::uint64_t room_id) const;
    int forget_except(std::span<const std::uint64_t> remembered_room_ids);  // returns rows dropped
};
}
```

File: `door_concept/etc/door_phantom_memory.csv`, header `room_id,x,y,yaw,width,height,height_sd,removed_ms`.

- [ ] **Step 1: Failing tests**

```cpp
TEST(phantom, round_trip_under_comma_locale)   // setlocale(LC_ALL,"es_ES.UTF-8" or ""); save -> load equal to 1e-6
TEST(phantom, forgotten_room_rows_dropped)     // ids {1,3} kept, row of 2 dropped
TEST(phantom, missing_file_is_empty)
TEST(phantom, glass_partition_not_reborn)      // door_concept core: a removed row injected for room_id k at the GlassBay place
                                               // (-4.8, 0.93, height 3.0) + a weak detection there -> no birth (reuse the fixture of
                                               // test_door_remembered_height.cpp)
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement; one-off converter `door_concept/tools/phantoms_from_identities.py` (run `python3 -I`): removed rows of `door_identities.csv` → phantom CSV, mapping room name → room_id through the live room nodes' `room_id` (`--map room_1=1,room_2=2,...` argument; rows it cannot map are dropped and listed).
- [ ] **Step 4:** → PASS. Live: rerun one bath-start tour with the flag on; the GlassBay partition is born/removed at most once (grep door_concept.log) — cf. door_7 ×18 in clean60.
- [ ] **Step 5: Commit** `common/phantom_memory: refuted places are a private per-agent file keyed by ltsm room_id`.

---

### Task 6: flip the default and delete the second memory (after Task 4/5 live pass, owner's go)

**Files:** `door_concept/etc/config.toml` (`RestoreFromLtsm = true`), `specificworker.cpp` (remove the CSV restore branch, the FRESH strip and `drop_room` on identities; keep `#next_door`/`#hazard` and name allocation), `door_identity_table.h` (drop the restorable columns from write; reader still accepts old rows), `MODEL_HISTORY.md` §5 (persistence = private phantom file, owner ruling 2026-11-13), `CONCEPT_AGENT_LIFECYCLE.md` door row (HISTORY: "restores from ltsm remembered_door notes").

- [ ] **Step 1:** Remove; door + ltsm suites → no new failures.
- [ ] **Step 2:** One live tour per start room → pass.
- [ ] **Step 3: Commit** `door_concept: ltsm is the only memory of the world's doors`.
