# ltom recall for every object agent (step 4a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** table, bottle, cabinet, refrigerator and hood resurrect a remembered instance from ltom's recall nodes
exactly as chair_concept does since step 3: same name, same uid, when the newborn is the MAP match.

**Architecture:** chair's `recall_candidates.h` moves to `common/recall_match/` with the class as a parameter, plus
one pure helper `choose_recall` that runs the MAP decision and takes the winner. Each agent gains an optional
`rc::Resurrect` argument on its existing `create_instance_from_detection` and a ~20-line block in its birth loop.
ltom offers recalls for all six classes. Birth stays tracker-only: resurrection only picks the identity.

**Tech Stack:** C++23, DSR/cortex, Eigen, CMake (`cbuild -j8`).

**Spec:** `docs/superpowers/specs/2026-11-01-long-term-object-memory-design.md` §8 step 4 ("Other agents"). Step 3
reference: `docs/superpowers/plans/2026-11-01-ltom-recall-chair.md` and the merged chair code.

**Worktree:** `active_inference.worktrees/ltom-recall-all`, branch `feat/ltom-recall-all` off main.

## Global Constraints
- CLAUDE.md: and/or/not; C++23; `std::from_chars`; type-attributed DSR only, never `runtime_checked_*`; Queued
  connections only; SIGTERM only; optional `inner_eigen` results always checked.
- Build with `cbuild -j8` (check `pgrep -a cc1plus` first). Never build in the rt-build worktree.
- No new cortex attributes: `object_uid`, `object_subtype` and `rt_covariance` already exist.
- No thresholds: the decision is `rc::recall::decide` (MAP, ties to "new"); the only per-class knobs are the two
  priors below.
- Per-agent `[Tracker]` priors (my calls; GUESS until step-4b learned permanence replaces them):

  | agent | RecallPiMove | RecallLogPriorNew | why |
  |---|---|---|---|
  | table | 0.5 | 0.0 | moved sometimes, like chairs |
  | bottle | 0.9 | 0.0 | moved most of the time |
  | cabinet | 0.05 | 0.0 | fixed furniture |
  | refrigerator | 0.05 | 0.0 | fixed furniture |
  | hood | 0.05 | 0.0 | fixed furniture |

- Measurement covariance at decision time = `DetectionNoiseM² · I`, the agent's own tracker R (bottle 0.05,
  the others 0.35), as in chair.

## Review Focus
1. **Two births in one cycle near the same remembered object.** Only one may take it; the other gets a fresh
   number. Test: `choose_recall` twice on the same list (Task 1).
2. **refrigerator/hood overwrite `object_subtype` with round/square after birth.** The recall node's subtype is
   ltom's CLASS (the name prefix), so the filter must match on the class string "refrigerator"/"hood", never on
   the instance's published subtype. Test in Task 1 with cls "refrigerator".
3. **Bottle born on a table.** The recall pose is room frame; the bottle's birth centroid `c` is room frame
   before `decide_support_surface` re-parents it. The decision must use the room-frame `c`, and the resurrected
   bottle may still be parented under the table (Task 6).
4. **Cabinet's kitchen-model nodes** (`cabinet_peninsula`, `cabinet_w*_base|_up`) carry no uid, so ltom never
   records them and never offers them. They must stay untouched (Task 5).
5. **A room change in the middle of a birth cycle.** `recall_room(latched, current)` returns none, and the
   newborn takes a fresh number. This is already tested for chair, and the test moves with the header (Task 1).

---

### Task 1: Generalise the recall consumer into common/

**Files:**
- Create: `common/recall_match/recall_candidates.h`, the body of `chair_concept/src/recall_candidates.h`,
  generalised.
- Delete: `chair_concept/src/recall_candidates.h`.
- Modify: `common/recall_match/recall_match_test.cpp`, adding the `choose_recall` tests (pure, no DSR).
- Modify: `chair_concept/tests/recall_candidates_test.cpp` and `chair_concept/tests/CMakeLists.txt`, pointing at
  the common header and adding the class cases.
- Modify: `chair_concept/src/chair_scene_graph.h`, `chair_concept/src/specificworker.cpp`, switching to the
  common types. No behaviour change.

**Interfaces (produces):**
```cpp
namespace rc {
struct Resurrect { int n; std::uint64_t uid; };          // replaces ChairSceneGraph::Resurrect (keep `using Resurrect = rc::Resurrect;` there)
struct RecallCand { recall::Candidate cand; int n = 0; };
inline constexpr std::uint64_t kUidNumberMask;
std::optional<std::uint64_t> recall_room(std::uint64_t latched, std::optional<std::uint64_t> current);
std::vector<RecallCand> recall_candidates(DSR::DSRGraph& G, std::uint64_t room_id, std::string_view cls,
                                          std::uint32_t my_agent_id, int max_n, const std::set<std::string>& live_names);
RecallCand take_candidate(std::vector<RecallCand>&, std::size_t);
std::vector<recall::Candidate> as_candidates(const std::vector<RecallCand>&);
float polygon_area(const std::vector<Eigen::Vector2f>&);
struct RecallChoice { Resurrect who; float p; };
// MAP over {cands, new} for one newborn at room-frame xy; on a win, removes the candidate from `cands` (take once).
std::optional<RecallChoice> choose_recall(const Eigen::Vector2f& xy, float detection_noise_m,
                                          std::vector<RecallCand>& cands, const recall::Params& p);
}
```
- `recall_candidates` filters `object_subtype == cls`, and live names on `std::string(cls) + "_" + N`. Its
  warning text names the class (`"[recall] {cls}_{n} is above the identity counter"`), with no "chair_concept"
  string.
- `choose_recall` is the decide-and-take part of chair's birth loop (`specificworker.cpp`, the RESURRECTION
  block), with R = `detection_noise_m² · I`. The polygon-area fallback warning stays in each agent, because it
  needs the agent's own `warned_` flag.

- [ ] **Step 1: Failing tests**
  - In `recall_match_test.cpp`, add these tests:
    - `test_choose_recall_takes_once`: two candidates at (1,1) and (4,4), variances 0.02, `Params{0.5, 0.0}`,
      area 20. The first `choose_recall((1,1), 0.35, cands, p)` returns `who.uid == uid_a` with `p > 0.5`, and
      `cands.size() == 1`. A second call at (1,1) does not return `uid_a` again.
    - `test_choose_recall_empty`: an empty list gives `nullopt`.
  - In `recall_candidates_test.cpp`, add these tests:
    - `test_class_is_a_parameter`: a "table" recall and a "chair" recall under the room. With cls "table", only
      the table is read; with cls "chair", only the chair.
    - `test_live_name_uses_class_prefix`: live `{"table_5"}` hides the table uid with N=5 but does not hide chair
      N=5.
    - `test_refrigerator_class_not_subtype`: a recall with `object_subtype` "refrigerator" is read with cls
      "refrigerator".
  - Change the existing chair cases to pass `"chair"`.
- [ ] **Step 2: Run them and see them fail.**
  - Run `bash common/run_tests.sh`. Expected: a compile failure, `choose_recall` not declared.
  - Build and run the chair test: `cbuild -j8` in `chair_concept/tests`, then
    `LANG=es_ES.UTF-8 ./recall_candidates_test`. Expected: a compile failure.
- [ ] **Step 3: Implement the common header.**
  - Make chair include `../../common/recall_match/recall_candidates.h`.
  - Replace chair's inline decide/take block with `choose_recall`. Keep its `RESURRECT` logging.
- [ ] **Step 4: Run the tests again.**
  - Expected: both suites PASS.
  - `cbuild -j8` in `chair_concept/` builds clean.
- [ ] **Step 5: Commit.** Message: `recall_match: class-generic recall consumer in common/, chair uses it`.

### Task 2: ltom offers recalls for all six classes

**Files:**
- Modify: `ltom_agent/etc/config.toml` (the `[LTOM]` block).
- Modify: `ltom_agent/src/specificworker.cpp:164-167` (the defaults).

- [ ] **Step 1:** Set `classes` and `recall_classes` to
  `["chair", "table", "bottle", "cabinet", "refrigerator", "hood"]`, in both the config and the code defaults.
  hood is new to `classes`.
- [ ] **Step 2:** Run `bash ltom_agent/tests/run_tests.sh` (or the ltom test target). Expected: PASS.
  `cbuild -j8` in `ltom_agent/` builds clean.
- [ ] **Step 3: Commit.** Message: `ltom: observe hood, offer recalls for all six object classes`.

### Task 3: table_concept resurrects

**Files:**
- `table_concept/src/table_scene_graph.{h,cpp}`: `create_instance_from_detection(c, room, std::optional<rc::Resurrect> r = std::nullopt)`
  and `int identity_last()`. Both follow chair's `chair_scene_graph.cpp`, including the two ⚠ consistency logs.
- `table_concept/src/specificworker_lifecycle.cpp:262` (the birth loop) and `specificworker.h` (`bool warned_recall_area_ = false;`).
- `table_concept/src/table_config.{h,cpp}`: `recall_pi_move` / `recall_log_prior_new` from `Tracker.RecallPiMove` / `Tracker.RecallLogPriorNew`.
- `table_concept/etc/config.toml` `[Tracker]`: the two keys with the values from Global Constraints, commented as chair's are.

**Interfaces:** consumes Task 1's `recall_room`, `recall_candidates(G, room, "table", ...)` and `choose_recall`.

- [ ] **Step 1:** Port the birth-loop pattern from chair.
  - Gather the candidates once per cycle, only if `res.births` is non-empty. Use `rc::recall_room(room_node_id_, rc::room::current_room(*G))`.
  - For each birth, call `choose_recall(Vector2f(c.x(), c.y()), cfg_.tracker_detection_noise_m, cands, rp)`.
  - Set the room area from table's `room_polygon_` member through `rc::polygon_area`. If the polygon is empty,
    fall back to the default area and warn once.
  - Log `RESURRECT` through the agent's existing tracker/phantom event logs, as chair does.
- [ ] **Step 2:** `cbuild -j8` in `table_concept/`. Expected: clean.
- [ ] **Step 3:** Run the existing `table_concept/tests` targets (`ctest` in `tests/build`). Expected: all PASS.
- [ ] **Step 4: Commit.** Message: `table_concept: resurrect a remembered table from ltom recall`.

### Task 4: refrigerator_concept and hood_concept resurrect

The two agents have the same structure, so they are one dispatch. Their birth calls are at
`specificworker_lifecycle.cpp:385` and `:386`.

**Files:** the same set as Task 3, in each of the two agents. The room area comes from `fitter_->room_polygon()`,
when `fitter_->has_room_polygon()`.

- [ ] **Step 1:** Port as in Task 3, using cls `"refrigerator"` and `"hood"`.
  - The `BIRTH_REFUSED` plausibility gate runs BEFORE the identity choice. Call `choose_recall` only for a birth
    that passed the gate, so a refused birth never consumes a candidate.
- [ ] **Step 2:** `cbuild -j8` in both agents, then their `tests/` targets. Expected: clean, PASS.
- [ ] **Step 3: Commit.** Message: `refrigerator, hood: resurrect from ltom recall`.

### Task 5: cabinet_concept resurrects (two birth paths)

**Files:** the same set as Task 3, in cabinet_concept. The birth calls are at `specificworker_lifecycle.cpp:1116`
(detections) and `:1297` (residual-arm seed, `c3`).

- [ ] **Step 1:** Port to both paths, sharing one candidate list per cycle if both run in the same cycle.
  Otherwise gather the list in each path. The decision uses the 2-D `(c.x, c.y)` and `(c3.x, c3.y)`. The room
  area comes from `fitter_->room_polygon()`. Leave `run_kitchen_model()` untouched (Review Focus 4).
- [ ] **Step 2:** `cbuild -j8` and `tests/` targets. Expected: clean, PASS.
- [ ] **Step 3: Commit.** Message: `cabinet_concept: resurrect from ltom recall on both birth paths`.

### Task 6: bottle_concept resurrects, plus docs

**Files:**
- The same set as Task 3, in bottle_concept. The birth call is at `specificworker.cpp:791`, and the room area
  comes from the `room_polygon_` member.
- `CONCEPT_AGENT_LIFECYCLE.md`: in the conformance audit, add "resurrects from ltom recall" to the rows of all
  six agents.
- Spec §8 step 4: one status line saying 4a is done, with the commit range.

- [ ] **Step 1:** Port as in Task 3 with cls `"bottle"`.
  - Decide on the room-frame centroid `c`, BEFORE `decide_support_surface` and the parent transform
    (Review Focus 3).
  - The parent choice is unchanged. A resurrected bottle keeps its number and uid whichever surface it lands on.
- [ ] **Step 2:** `cbuild -j8` and `tests/` targets. Expected: clean, PASS.
- [ ] **Step 3: Commit.** Message: `bottle_concept: resurrect from ltom recall; lifecycle audit + spec status`.

## Verification after merge (live, coordinated with the room session)

- Rebuild all six object agents and ltom in main.
- Run a tour that leaves a room and comes back.
- For each class present in the world, check that the returning instance logs `RESURRECT <class>_N uid=... p=...`
  with the same uid, and that `ltom_recall_*` nodes disappear once the instances are alive.
- Do not merge or push without the user's go.
