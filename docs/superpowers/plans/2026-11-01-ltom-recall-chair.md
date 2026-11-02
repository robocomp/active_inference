# ltom recall + chair resurrection (rollout step 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the robot is in a room, ltom publishes a recall node for each remembered (not alive) chair of that room;
chair_concept, at a birth, scores the newborn against them and, when "same chair" is the most probable hypothesis,
takes back its old internal name `chair_N` and `object_uid` instead of issuing new ones.

**Architecture:** Three pieces with one interface each. (1) `common/recall_match/recall_match.h`: pure Eigen maths,
built on `common/view_atlas/view_atlas.h` (`location_log_lik`, `reid_posterior`). (2) ltom: a pure selector (which
entries to recall) and a graph publisher (insert / move / delete `ltom_recall_<uid>` nodes). (3) chair_concept: reads
recall nodes of subtype `chair`, calls the helper, and passes an optional `{N, uid}` into its birth. ltsm's room
eviction learns to skip ltom's recall nodes.

**Tech Stack:** C++23, Eigen, cortex DSR (type-attributed), Qt6 Core, cbuild.

**Spec:** `docs/superpowers/specs/2026-11-01-long-term-object-memory-design.md` (§6.2, §8 step 3, §12 matching helper,
duplicates, visibility).

## Global Constraints

- CLAUDE.md throughout: and/or/not; C++23; from_chars; type-attributed DSR only; no `Qt::DirectConnection` (no DSR
  signals at all: poll); SIGTERM only; `cbuild -j8` (check `pgrep -a cc1plus`); every inner_eigen optional checked.
- **Identity stays the concept agent's decision** (spec U4): ltom publishes candidates; only chair_concept decides.
- **Birth stays tracker-only** (spec §12): resurrection only chooses which identity the newborn takes; it never
  creates, delays or suppresses a birth.
- **Recall node:** type `metaconcept`, name `ltom_recall_<uid decimal>`, RT edge from the live room node at the
  remembered pose (x, y, yaw of the entry's last keyframe in that room), `rt_covariance` from the keyframe variances;
  attributes `object_uid` and `object_subtype` (= the entry's class) only. No width/depth/height (viewers draw
  metaconcepts with sizes as bodies). Never type `object`, never a name starting `recall_` (ltsm/door_concept/
  language_agent match that prefix).
- **Which entries are recalled:** class in `[LTOM] recall_classes` (= `["chair"]` in this step), last life not
  `alive`, last keyframe's `room_key` == the current room's key, uid not present on the graph. Removed when any of
  these stops holding (robot left the room, the uid is alive again).
- Current room: `rc::room::current_room(G)` (common/room_resolve/room_resolve.h); its key by Task-4-of-step-2's rule
  (`room_id_att`, else trailing digits) — reuse `graph_sightings`' function, do not copy it.
- Live runs coordinated with the room session ("Guardes actions, ever shared facts"); never build in rt-build.
- Worktree `active_inference.worktrees/ltom-recall`, branch `feat/ltom-recall`, from main.

## Rulings in this plan (my calls)

- **No cortex change.** The recall node carries no permanence attribute and no internal-name attribute: the name is
  `chair_<N>` with N = `uid & (2^40−1)`, and the class is `object_subtype`. Cost if wrong: one REGISTER_TYPE later.
- **Permanence = class prior, owned by the consumer, for now.** chair_concept's config gets
  `RecallPiMove = 0.5` (a guess, flagged: chairs get moved), the mixture weight of "moved" in
  `location_log_lik`. Learned permanence (spec §5, visibility-weighted) is step 4, and it will then ride on the recall
  node. Cost if wrong: resurrections too eager/shy until step 4; tunable in config.
- **The decision is the MAP hypothesis, not a cut-off:** the newborn takes candidate k when `reid_posterior` gives k
  the largest mass of {candidates…, new}. The prior of "new" is a config term (`RecallLogPriorNew = 0.0`, i.e. equal
  to one candidate); the "new" location likelihood is uniform over the room: `−log(room_area)`.
- **Birth-time evidence is position only.** At birth chair_concept knows the room-frame centroid and nothing else (no
  yaw, no fitted size, no atlas yet). Measurement covariance = `DetectionNoiseM²·I` (its tracker's R). Spec accepts
  duplicates in v1; a wrong resurrection is visible in the log (`RESURRECT`) and in ltom's lives count.
- **ltsm skip lives on this branch:** one condition in `ltsm_agent/src/room_eviction.cpp` (~285-296) so a room's RT
  children named `ltom_recall_*` are not archived. It is the room session's file: tell it before editing; keep the
  change to that condition and a comment.
- **Live check uses a restart, not a room change:** stop chair_concept (its chairs end `owner_exit`), ltom recalls
  them, restart chair_concept, the same chair is reborn as the same `chair_N`. Leaving and re-entering a room needs
  the controller and is left to a later tour.

## Review Focus

1. A recall node must disappear the poll the real chair is alive again (else chair_concept would see its own chair
   as a candidate for the next birth) — Task 2 `test_alive_uid_not_recalled`, Task 3 `test_removed_when_alive`.
2. Two newborns in one cycle must not both take the same candidate — Task 4 `test_candidate_taken_once`.
3. A candidate whose `chair_N` is live (name clash) is never offered — Task 4 `test_live_name_excluded`.
4. ltom restarted mid-run: its stale `ltom_recall_*` nodes from the previous run are swept at start (Task 3
   `test_stale_sweep`).
5. Room unknown (`current_room` nullopt): publish nothing and remove all recall nodes (Task 3
   `test_no_room_no_recall`).

---

### Task 1: Matching helper `common/recall_match`

**Files:** Create `common/recall_match/recall_match.h`, `common/recall_match/recall_match_test.cpp` (picked up by
`common/run_tests.sh`).

**Interfaces (namespace `rc::recall`):**
```cpp
struct Candidate { std::uint64_t uid; Eigen::Vector2f mu; Eigen::Matrix2f cov; };
struct Params    { float pi_move = 0.5f; float log_prior_new = 0.f; float room_area_m2 = 20.f; };
struct Decision  { std::optional<std::size_t> index; std::vector<float> posterior; };  // posterior: candidates…, new
Decision decide(const Eigen::Vector2f& x, const Eigen::Matrix2f& meas_cov,
                std::span<const Candidate> cands, const Params& p);
```
Each candidate's location term is `view_atlas::location_log_lik(x, mu, cov + meas_cov, pi_move, room_area)`;
`log_prior` 0; the "new" term `log_prior_new − log(room_area)`; `index` = argmax if it is not "new".

- [ ] Tests first: `test_close_candidate_wins` (x 0.1 m from mu, σ 0.1 → index 0), `test_far_candidate_loses`
  (x 3 m away in a 20 m² room → new), `test_nearest_of_two`, `test_no_candidates_is_new`,
  `test_pi_move_one_never_resurrects` (pi_move 1 → posterior of each candidate == new's when log_prior_new 0, ties
  go to new), `test_posterior_sums_to_one`.
- [ ] Run `bash common/run_tests.sh` → FAIL (missing header), implement, → `common: all test binaries passed`
  (the 3 pre-existing Qt-less mission test build failures are known and unrelated; say so if still there).
- [ ] Commit `common/recall_match: MAP re-identification of a newborn against recalled candidates`.

### Task 2: ltom recall selection (pure)

**Files:** Create `ltom_agent/src/recall_select.{h,cpp}`; test `ltom_agent/tests/test_recall_select.cpp` (tests CMake).

**Interfaces (namespace `ltom`):**
```cpp
struct Recall { std::uint64_t uid; std::string cls; Keyframe kf; };
std::vector<Recall> select_recalls(const Repertory& r, std::int64_t room_key,
                                   const std::set<std::uint64_t>& present, std::span<const std::string> classes);
```
- [ ] Tests: `test_dead_chair_in_room_recalled` (owner_exit chair, last kf room 1 → recalled with that kf),
  `test_other_room_not_recalled`, `test_alive_uid_not_recalled` (life alive, or uid in `present`),
  `test_class_filter` (a dead table not recalled with classes {"chair"}), `test_last_keyframe_of_that_room`
  (entry seen in room 1 then room 2, dead: recalled in room 2 only).
- [ ] RED → implement → GREEN; commit `ltom_agent: select which remembered objects to recall in a room`.

### Task 3: ltom recall publisher (graph)

**Files:** Create `ltom_agent/src/recall_publisher.{h,cpp}`; modify `ltom_agent/src/specificworker.{h,cpp}`,
`ltom_agent/etc/config.toml` (`[LTOM] recall_classes = ["chair"]`; `[Owns]` nodes add `"ltom_recall_*"`),
`ltom_agent/src/specificworker_presence.cpp` (stale sweep `rc::owned::Spec{.name_prefix = "ltom_recall_",
.node_types = {"metaconcept"}}` at initialize and first Operating), README. Test
`ltom_agent/tests/test_recall_publisher.cpp` (DSR, domain 63).

**Interfaces:**
```cpp
class RecallPublisher {
public:
    explicit RecallPublisher(std::shared_ptr<DSR::DSRGraph> G);
    // Make the graph hold exactly `want` under room `room_id`: insert missing, move changed, delete the rest.
    void sync(std::optional<std::uint64_t> room_id, std::span<const Recall> want);
};
```
Insert = `Node::create<metaconcept_node_type>("ltom_recall_" + uid)` + `object_uid_att`, `object_subtype_att` +
`insert_node` + `rt_api->insert_or_assign_edge_RT(room, node, {x, y, 0}, {0, 0, yaw})` + `rc::rtcov::publish` with
the keyframe variances. The agent calls `sync` each poll after `observe`, with `select_recalls(..., present_from_
read_graph, classes)` and the current room's id/key; `nullopt` room → `sync(nullopt, {})`.
- [ ] Tests: `test_inserts_under_room`, `test_moves_when_keyframe_changes`, `test_removed_when_alive` (want empty →
  node gone), `test_no_room_no_recall`, `test_stale_sweep` (a pre-existing `ltom_recall_9` node removed by the Spec
  sweep), `test_never_touches_other_nodes` (a `recall_room_1` metaconcept and a `chair_3` object survive).
- [ ] RED → implement → GREEN; agent builds; commit `ltom_agent: publish recall nodes for the current room`.

### Task 4: chair_concept resurrection

**Files:** Modify `chair_concept/src/specificworker.cpp` (birth loop ~780-856), `chair_concept/src/chair_scene_graph.
{h,cpp}` (`create_instance_from_detection` gains `std::optional<Resurrect> r` with `struct Resurrect { int n;
std::uint64_t uid; };` — when set, skip `identity_->issue` and use n/uid), `chair_concept/etc/config.toml`
(`RecallPiMove = 0.5`, `RecallLogPriorNew = 0.0`, with comments), CONCEPT_AGENT_LIFECYCLE.md (chair row: resurrection
from ltom recall). Test: a pure function extracted for testing,
`chair_concept/src/recall_candidates.h`: `std::vector<RecallCand> recall_candidates(DSR::DSRGraph&, std::uint64_t
room_id, std::uint32_t my_agent_id, const std::set<std::string>& live_names)` (reads `metaconcept` nodes named
`ltom_recall_*` whose RT parent is the room, `object_subtype == "chair"`, `(uid >> 40) == my_agent_id` (inline; chair_concept does not include ltom headers), and whose
`chair_<N>` is not live), tested in `chair_concept/tests/` (DSR, domain 62 is episodic's — use 65).
- In the birth loop: build candidates once per cycle; for each birth call `rc::recall::decide`; a taken candidate is
  removed from the list for the rest of the cycle; log `RESURRECT chair_N uid=… p=…` via `log_tracker_event` and
  `log_phantom_event`.
- Room area: from the room polygon chair_concept already holds (find it via `fitter_->point_in_room`'s data); if
  none, the helper's default.
- [ ] Tests: `test_reads_own_chair_recalls`, `test_live_name_excluded`, `test_other_agent_uid_excluded`,
  `test_candidate_taken_once` (two births at the same spot in one cycle: one resurrects, the other is new — test the
  loop's take-once step as a small pure function).
- [ ] RED → implement → GREEN; chair builds; commit `chair_concept: resurrect a remembered chair from ltom recall`.

### Task 5: ltsm skips ltom recall nodes

**Files:** Modify `ltsm_agent/src/room_eviction.cpp` (~285-296, the RT-children archive filter).
- [ ] Controller first tells the room session (file is theirs). One condition: skip a child whose name starts with
  `ltom_recall_`, with a comment pointing at this plan. Build ltsm_agent (in this worktree). Commit
  `ltsm_agent: room eviction never archives ltom recall nodes`.

### Task 6: Live check (controller-run)

- [ ] Slot from the room session. Fleet: room_concept (main), retina, ltom (this branch), chair_concept (this branch,
  run from the main chair dir so its counter continues), table optional.
- [ ] Expected: chairs born; SIGTERM chair_concept → `owner_exit`; ltom log + pydsr show `ltom_recall_<uid>` under
  room_1 at the chair poses; restart chair_concept → `RESURRECT chair_N` with the SAME N and uid; ltom shows `lives 2`
  for that uid and the recall node is gone.
- [ ] SIGTERM all; tell the room session it is clear.
