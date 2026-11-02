# Instance identity (persisted name counter + object_uid) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An internal instance name (`chair_7`) and its `object_uid` are never issued twice by an agent, across restarts, in chair, table, bottle, cabinet and refrigerator.

**Architecture:** One header-only module `common/instance_identity/instance_identity.h` (no DSR, no Qt) owns a per-agent counter persisted in `etc/instance_identity.csv` of the agent's working directory. Each agent's existing `create_instance_from_detection` keeps scanning the live graph for its highest `<prefix>_<N>` and asks the module for the next number, which is `max(persisted counter, live max) + 1`, written to disk **before** it is returned. `object_uid = (agent_id << 40) | N` is published on the new node once cortex has the attribute.

**Tech Stack:** C++23, header-only, `<charconv>`/`<filesystem>`/`<fstream>`; DSR type-attributed API in the agents; standalone test under `common/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-11-01-long-term-object-memory-design.md` §7 (identity without waiting for the memory), §8 step 1, §12 (lost or corrupt counter file). Bug it fixes: `docs/superpowers/specs/2026-11-01-instance-names-design.md` §11 and `chair_concept/src/chair_scene_graph.cpp:38-49` ("REMAINING GAP").

## Global Constraints

- CLAUDE.md: `and`/`or`/`not`; C++23; numbers read with `std::from_chars` only (es_ES locale); every file written through `imbue(std::locale::classic())`; type-attributed DSR (`add_or_modify_attrib_local<object_uid_att>`), never `runtime_checked_*`; never `Qt::DirectConnection`; SIGTERM only; build with `cbuild -j8` (check `pgrep -a cc1plus` first; another build running ⇒ wait).
- A birth never waits for or fails because of this module: a failed write is logged loudly and the in-memory counter is used.
- Never start from zero when the file is missing or corrupt: `max(live graph, file)` (spec §12; ltom_agent and caches do not exist yet).
- uid = `(static_cast<std::uint64_t>(agent_id) << 40) | N` — agent id from `G_->get_agent_id()`, N the same number as the name.
- New cortex attribute: `REGISTER_TYPE(object_uid, std::uint64_t, false)` in `/usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h` — **the user adds and reinstalls cortex**; never `sudo cp`.
- Work on branch `feat/instance-identity` in a worktree under `active_inference.worktrees/`; never build in rt-build; no live fleet runs without coordinating with the room session.

## Review Focus

1. Agent killed between writing the counter and inserting the node ⇒ that number is skipped, never reused (gaps are fine). Test: `issue` persists before returning (Task 1 `test_persist_before_return`).
2. File written by a run whose agent id differs (config copied between agents) ⇒ still `max(...)`, logged as a mismatch, no reset. Test in Task 1 (`test_agent_mismatch_keeps_max`).
3. File with a decimal comma or garbage (`"7,0"`, `"abc"`, empty) ⇒ treated as corrupt, `max(live)`+1, loud log. Test in Task 1 under `setlocale(LC_ALL,"es_ES.UTF-8")` (`test_corrupt_file`).
4. Two births in one cycle ⇒ two consecutive numbers, both persisted. Test in Task 1 (`test_two_issues_consecutive`).
5. Tours that reset state (warm_start_tour.sh copies door_identities from fixtures) and expect `chair_1`: after this change numbers continue across runs. Task 2 documents "delete `etc/instance_identity.csv` to restart numbering" in each agent's config comment and in CONCEPT_AGENT_LIFECYCLE.md; no script is changed (the room session owns them).

---

### Task 1: `common/instance_identity` module

**Files:**
- Create: `common/instance_identity/instance_identity.h`
- Test: `common/instance_identity/instance_identity_test.cpp` (picked up automatically by `common/run_tests.sh`)

**Interfaces:**
- Produces (namespace `rc::identity`):
  - `enum class LoadStatus { ok, missing, corrupt, agent_mismatch };`
  - `class InstanceCounter` with
    - `static InstanceCounter open(std::filesystem::path file, std::uint32_t agent_id);` — reads the file; never throws.
    - `LoadStatus status() const;`
    - `int issue(int live_max);` — `n = max(counter, live_max) + 1`; writes the file (temp in same dir + `rename`, `imbue(classic)`), then returns n. On write failure: one `std::print(stderr, ...)` line naming the file and error, keeps n in memory, returns n.
    - `int last() const;` — highest number issued or loaded.
  - `std::uint64_t make_uid(std::uint32_t agent_id, int n);` — `(agent_id << 40) | n`.
  - `int max_suffix(std::span<const std::string> names, std::string_view prefix);` — highest N over names `"<prefix><N>"` (prefix includes the `_`), parsed with `from_chars`, ignoring names that do not parse; 0 if none.
- File format (two lines, `key value`, ASCII): `agent_id <uint>` and `last <int>`. Anything else ⇒ `corrupt`.
- `open` logs one line per non-ok status: `[identity] <file>: missing — numbering continues from the live graph` / `corrupt (<reason>) — …` / `written by agent <a>, I am <b> — keeping the larger counter`.

- [ ] **Step 1: Write the failing tests** in `instance_identity_test.cpp`: plain `main` + `assert`-style checks printing `instance_identity_test: ok` at the end; first line of `main` is `std::setlocale(LC_ALL, "es_ES.UTF-8") or std::setlocale(LC_ALL, "");` (CLAUDE.md locale rule); each test uses a fresh temp dir.

```cpp
// test_missing_file_starts_from_live:   open(tmp/"id.csv", 20).status()==missing; issue(3)==4; issue(0)==5
// test_reload_continues:                c.issue(0)==1, c.issue(0)==2; open(same,20).last()==2; .issue(0)==3
// test_persist_before_return:           n=c.issue(5); file content parses to last==n (read with from_chars) before any other call
// test_live_max_wins:                   file last=2; open().issue(9)==10
// test_file_wins_over_live:             file last=9; open().issue(2)==10       // the restart bug itself
// test_corrupt_file:                    file "last 7,0" / "abc" / "" -> status()==corrupt; issue(4)==5
// test_agent_mismatch_keeps_max:        file "agent_id 7\nlast 12"; open(.,20).status()==agent_mismatch; issue(3)==13
// test_two_issues_consecutive:          issue(0)==1; issue(0)==2; reopened last()==2
// test_make_uid:                        make_uid(20,7)==(std::uint64_t(20)<<40 | 7); make_uid(20,7)!=make_uid(21,7)
// test_max_suffix:                      {"chair_3","chair_12","table_40","chair_x","chair_"} prefix "chair_" -> 12; {} -> 0
// test_unwritable_dir_still_issues:     dir chmod 0500; issue(0)==1 and issue(0)==2 (no throw)
```

- [ ] **Step 2: Run to verify they fail**
Run: `bash common/run_tests.sh 2>&1 | grep -A3 instance_identity`
Expected: `BUILD FAILED: instance_identity/instance_identity_test.cpp` (header missing).

- [ ] **Step 3: Implement `common/instance_identity/instance_identity.h`** per the Interfaces block. Header comment (match the style of `common/view_atlas/atlas_io.h`): what it fixes (cite `chair_scene_graph.cpp` 08-11 teleport), the persist-before-return rule, the file format, "delete the file to restart numbering".

- [ ] **Step 4: Run to verify they pass**
Run: `bash common/run_tests.sh`
Expected: `instance_identity_test: ok` and `common: all test binaries passed`.

- [ ] **Step 5: Commit**
```bash
git add common/instance_identity/
git commit -m "common/instance_identity: per-agent persisted instance counter + object_uid (never re-issue a name across restarts)"
```

---

### Task 2: Wire the counter into the five agents (names only)

One dispatch, one review: the same edit in five files.

**Files (each `create_instance_from_detection`, the block that computes `max_n` and `name_high_water_`):**
- Modify: `chair_concept/src/chair_scene_graph.{h,cpp}` (cpp ~:38-57)
- Modify: `table_concept/src/table_scene_graph.{h,cpp}` (cpp ~:62-67)
- Modify: `bottle_concept/src/bottle_scene_graph.{h,cpp}` (cpp ~:114-119)
- Modify: `cabinet_concept/src/cabinet_scene_graph.{h,cpp}` (cpp ~:54-59; nodes are type `box`)
- Modify: `refrigerator_concept/src/refrigerator_scene_graph.{h,cpp}` (cpp ~:55-62)
- Modify: each agent's `etc/config.toml` comment block near `[Owns]` — one line: `# instance names: etc/instance_identity.csv (common/instance_identity); delete it to restart numbering`
- Modify: `CONCEPT_AGENT_LIFECYCLE.md` §1 CREATE — one bullet: names and uids come from `common/instance_identity`; never from a live-graph scan alone.

**Interfaces:**
- Consumes: `rc::identity::InstanceCounter`, `max_suffix` (Task 1).
- Produces: member `std::optional<rc::identity::InstanceCounter> identity_;` replacing `int name_high_water_` in each `*SceneGraph`; opened lazily on first birth with `InstanceCounter::open("etc/instance_identity.csv", static_cast<std::uint32_t>(G_->get_agent_id()))` (lazy because the graph id is known only after construction). Member `int last_issued_n_` is not needed: `identity_->last()`.

- [ ] **Step 1: In each of the five files** replace the high-water block with: collect the agent's live names (the existing `get_nodes_by_type(...)` + prefix filter, unchanged), `const int n = identity_->issue(rc::identity::max_suffix(names, "<prefix>_"));`, `name = "<prefix>_" + std::to_string(n)`. Remove `name_high_water_` and its "REMAINING GAP" comment; keep the 08-11 explanation, now ending "fixed by common/instance_identity (persisted)". Remove the `try { std::stoi } catch (...)` (max_suffix replaces it).

- [ ] **Step 2: Build each agent in the worktree**, one at a time:
Run: `cd <agent> && bash -ic 'cbuild -j8'` for chair_concept, table_concept, bottle_concept, cabinet_concept, refrigerator_concept
Expected: each ends `[100%] Built target <agent>`. Any agent with offline tests (`tests/run_tests.sh` or a `*_test` target) runs them; expected pass.

- [ ] **Step 3: Check nothing still uses the old counter**
Run: `git grep -n "name_high_water_" -- chair_concept table_concept bottle_concept cabinet_concept refrigerator_concept`
Expected: no output. (The restart rule itself is covered by Task 1's tests; the live restart is Task 4.)

- [ ] **Step 4: Commit**
```bash
git add chair_concept table_concept bottle_concept cabinet_concept refrigerator_concept CONCEPT_AGENT_LIFECYCLE.md
git commit -m "concept agents: instance names from common/instance_identity (persisted counter) — chair_N never re-issued after a restart"
```

---

### Task 3: Publish `object_uid` on each new instance

**Blocked until the user has added** `REGISTER_TYPE(object_uid, std::uint64_t, false)` to cortex's `dsr_attr_name.h` (comment: `// agent-scoped persistent instance id: (agent_id << 40) | N, common/instance_identity — 2026-11-01`) **and reinstalled cortex.** The controller asks the user before dispatching this task; until then the plan is complete at Task 2.

**Files:** the same five `*_scene_graph.cpp` (where the node's other attributes are set, before `insert_node`).

**Interfaces:**
- Consumes: `rc::identity::make_uid` (Task 1), `n` from Task 2.
- Produces: attribute `object_uid` (`std::uint64_t`) on every instance node born after this change. Read by future ltom_agent and the view atlas (VIEW_ATLAS.md §5).

- [ ] **Step 1:** in each file add `G_->add_or_modify_attrib_local<object_uid_att>(node, rc::identity::make_uid(static_cast<std::uint32_t>(G_->get_agent_id()), n));` next to `object_subtype_att`.
- [ ] **Step 2:** build the five agents (`cbuild -j8`, one at a time). Expected: `[100%] Built target <agent>` each; a compile error naming `object_uid_att` means cortex was not reinstalled — stop and tell the user.
- [ ] **Step 3: Commit**
```bash
git add chair_concept table_concept bottle_concept cabinet_concept refrigerator_concept
git commit -m "concept agents: publish object_uid on each new instance"
```

---

### Task 4: Live check (with the user and the room session)

Not dispatched to a subagent; the controller runs it after coordinating with the room session ("Guardes actions, ever shared facts").

- [ ] **Step 1:** In Webots with chair_concept (worktree binary) and its required peers: let a chair be born (`chair_N`), stop chair_concept with SIGTERM (its nodes are cleaned), start it again, let a chair be born. Expected: the new chair's number is `N+1` or more, never `N`; `etc/instance_identity.csv` shows `last` ≥ N+1. With Task 3 in: the node's `object_uid` = `(20<<40)|N'` (chair_concept id 20).
- [ ] **Step 2:** Record the result in the plan's ledger; report to the user.
