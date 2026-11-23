# Changed Object (moved chair: decide, ask, learn) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a remembered chair turns up elsewhere in its room, the robot links the live chair to the remembered one, asks the owner (or later reports), records the answer in LTOM, and learns from it.

**Architecture:** A pure posterior in `common/recall_match/identity_link.h`. The chair agent runs it after birth for every live chair against the room's untaken recalls. Each recall's old spot carries a "ghost" existence belief, fed with the chair's own absence evidence. The chair agent publishes the link on the live node (`same_as_uid`, `same_as_prob`, `left_spot_prob`). LTOM reads the link from its sightings. It records the continuation (the owner's choice B: the live chair keeps its name, and memory joins the two), opens a question node on the memory graph, applies the owner's answer (a new claim kind), and keeps dated evidence for two beliefs. language_agent reads the questions and beliefs, chooses ask or report, phrases from graph facts, and sends the answer back as a claim.

**Tech Stack:** C++23, DSR/cortex (working graph domain 0, LTOM memory graph domain 5), Qt6, Eigen. Standalone g++ tests: `common/run_tests.sh`, `ltom_agent/tests` (CMake), `language_agent/tests/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-11-15-changed-object-design.md` (approved 2026-11-15). Owner decision after the spec: **B**. The live chair keeps its fresh number (for example chair_57). It carries "I believe this is chair_50, p", and LTOM joins the histories. A "no" cuts the link, and nothing in the map changes.

## Global Constraints

- No thresholds where a model term works (CLAUDE.md). Decisions are MAP comparisons, as in `rc::recall::decide`. Any unavoidable cutoff is flagged in a comment.
- Agents talk only through the graph and LTOM; no file crosses agents.
- DSR attributes are type-attributed only (`<foo_att>`), never `runtime_checked_*`. New attributes go through cortex `REGISTER_TYPE`. **The owner reinstalls cortex; never `sudo cp`.**
- DSR signals are never `Qt::DirectConnection`. Graph work runs on the main thread, as the existing ltom and chair paths do.
- Parse numbers with `std::from_chars`. Write data files through `imbue(std::locale::classic())`.
- Use `and` / `or` / `not`. Use C++23 containers and algorithms.
- Build at `-j8`, or `-j4` if `pgrep cc1plus` shows another build. Stop agents with SIGTERM, never `kill -9`.
- Commit only on the plan's commit steps, in the worktree branch `feat/changed-object`. **Merging to main and pushing need the owner's word.**
- Live runs need Webots, CLAIMED and FREED with the peer sessions (room, objects).
- Question wording (spec, exact):
  - Ask: `"The <colour> chair in room <N> is <new place> now; it was <old place>. Did you move it?"`
  - Away: `"By the way, while you were away the <colour> chair in room <N> moved <new place>. I've assumed it's the same chair. Did you move it?"`
  - Report: `"The <colour> chair has moved again, <new place>. I assume you moved it."`

## Review Focus

1. **Two recalls at the same old spot** (chair_50 magenta, believed; chair_44 colourless, exist_prob 0.06). The link and the question must name chair_50, never chair_44. Test: `link_prefers_coloured_believed_recall` (Task 1) and `question_names_link_winner` (Task 4).
2. **Old spot never seen.** The link may form, but no question may be opened while `left_spot_prob ≤ 0.5`. Test: `no_question_before_old_spot_seen_empty` (Task 4).
3. **A "no" while the live chair persists.** LTOM stops offering chair_50 in that room, and the chair agent then clears `same_as_*`. The live chair must not re-link to chair_50 in the same run. Test: `gone_from_room_not_recalled` (Task 3), plus the chair-side check in Task 5 step 6.
4. **Restart in the middle of a question.**
   - A restarted language_agent re-reads the open question and asks it again; it never answers it itself.
   - A restarted LTOM reloads the links, questions and beliefs from its file.
   - Tests: `repertory_io_roundtrips_links_and_questions` (Task 3) and `human_io_roundtrips_beliefs` (Task 4).
5. **An answer arrives for a question that has already closed** (stale stamp, or the live chair died). LTOM rejects it with a reason, and nothing is learned. Test: `answer_to_closed_question_rejected` (Task 4).

---

### Task 1: The identity-link posterior (pure)

**Files:**
- Create: `common/recall_match/identity_link.h`
- Create: `common/recall_match/identity_link_test.cpp`
- Modify: `common/run_tests.sh`, to build and run `identity_link_test` next to `recall_match_test`

**Interfaces:**
- Produces:
```cpp
namespace rc::recall {
struct LinkCand {
    std::uint64_t uid;
    float log_prior = 0.f;               // log obj_exist_prob, as in Candidate
    std::string colour_name;             // "" = never measured
    float colour_prob = 0.f;
    float still_there_llr = 0.f;         // ghost: L_now - L0 (log P(evidence|still there) - log P(evidence|not))
};
struct LinkParams { float pi_move = 0.10f; float pi_gone = 0.05f; float lambda_new = 0.05f; };
struct Link { std::optional<std::size_t> index; std::vector<float> posterior; };   // candidates..., new
// colour_now: the live chair's 12-term posterior over appearance::kPalette, nullopt = no colour yet.
[[nodiscard]] float colour_log_ratio(const std::optional<std::array<float, 12>>& colour_now,
                                     std::string_view name_then, float prob_then);
[[nodiscard]] Link link_posterior(std::span<const LinkCand> cands,
                                  const std::optional<std::array<float, 12>>& colour_now, const LinkParams& p);
}
```
- The model, in the header comment. A live chair L was born NEW (position alone lost at birth). Compare "L is recall k, moved" with "L is a new arrival".
  - Same_k: k moved (π_move), and its old spot is empty by construction (evidence likelihood 1).
  - New: an arrival (λ_new), and k either stays (π_stay = 1 − π_move − π_gone, weighed by the absence evidence exp(still_there_llr)) or is gone (π_gone).
  - Position is uniform over the room under both, so it cancels.
  - Colour: likelihood ratio 12·Σ q_now[i]·q_then[i]. q_then is `prob_then` on its term and `(1−prob_then)/11` on the others. It is 0 (no evidence) when either side is unmeasured.
  - `logodds_k = log π_move + colour_lr_k + log_prior_k − log λ_new − log(π_stay·exp(still_there_llr_k) + π_gone)`.
  - Posterior: softmax over {logodds_k..., 0 (new)}.
  - `index` = argmax over candidates, **only if strictly larger than new** (same rule as `decide`).
  - Several candidates share the mass, which is how the room's count enters: one chair gives a sure "which"; many similar chairs give an unsure one.

- [ ] **Step 1: Write the failing tests** in `identity_link_test.cpp` (style of `recall_match_test.cpp`, print and return non-zero on failure):
  - `moved_magenta_old_spot_empty`: one cand, magenta 0.74 then; now magenta 0.74 (others equal); still_there_llr = −3 → index 0, posterior[0] > 0.95.
  - `moved_old_spot_unseen_still_links`: same, still_there_llr = 0 → index 0, posterior[0] lower than in the previous case (strictly).
  - `colour_mismatch_is_new`: then magenta 0.74, now green 0.8 → no index.
  - `colourless_now_is_no_evidence`: colour_now = nullopt → `colour_log_ratio` == 0 exactly.
  - `link_prefers_coloured_believed_recall`: cands chair_50 (magenta 0.74, log_prior log 0.9) and chair_44 ("" colour, log_prior log 0.06), both still_there_llr = −3, now magenta → index 0, posterior[1] < 0.05.
  - `two_identical_candidates_split`: two magenta cands with equal everything → |posterior[0] − posterior[1]| < 1e-6.
  - `no_candidates`: empty → no index, posterior == {1}.
- [ ] **Step 2: Run it and watch it fail.** `bash common/run_tests.sh 2>&1 | tail -20`. Expected: compile error, `identity_link.h` not found.
- [ ] **Step 3: Implement `identity_link.h`** per the Interfaces block. Compute in log space and subtract the max before `exp`. `#include "../appearance_belief/colour_terms.h"` for `kPalette` order and size; `static_assert(kPalette.size() == 12)`.
- [ ] **Step 4: Run it and watch it pass.** `bash common/run_tests.sh 2>&1 | tail -20`. Expected: all identity_link checks PASS, and every other common test still passes.
- [ ] **Step 5: Commit.** `git add common/recall_match/identity_link.h common/recall_match/identity_link_test.cpp common/run_tests.sh && git commit -m "recall: identity-link posterior for a live object against untaken recalls (moved vs new arrival)"`

### Task 2: Cortex attributes (owner reinstall gate)

**Files:**
- Modify: `/home/pbustos/robocomp/components/cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`, after `human_claim_answer_stamp` (line 1361). Work on a new cortex branch `feat/changed-object-attrs` cut from `development`. Do not touch its current branch `rename-node`; `git stash` nothing of the owner's. If the cortex tree is dirty, stop and ask.

**Interfaces:**
- Produces (exact lines):
```cpp
REGISTER_TYPE(same_as_uid,          std::uint64_t,                                     false)  // recall uid this live object is believed to be (0 = none)
REGISTER_TYPE(same_as_prob,         float,                                             false)  // P(same) from identity_link; 1 = the owner confirmed
REGISTER_TYPE(left_spot_prob,       float,                                             false)  // P(the recalled object left its remembered spot)
REGISTER_TYPE(question_kind,        std::string,                                       false)  // "moved"
REGISTER_TYPE(question_live_uid,    std::uint64_t,                                     false)
REGISTER_TYPE(question_memory_uid,  std::uint64_t,                                     false)
REGISTER_TYPE(question_old_xy,      std::reference_wrapper<const std::vector<float>>,  false)  // room frame, m
REGISTER_TYPE(question_new_xy,      std::reference_wrapper<const std::vector<float>>,  false)
REGISTER_TYPE(question_room,        std::string,                                       false)  // room node name
REGISTER_TYPE(question_again,       bool,                                              false)  // the memory already holds a move
REGISTER_TYPE(question_stamp,       std::uint64_t,                                     false)  // ms when opened; the answer claim echoes it in human_claim_value
REGISTER_TYPE(belief_yes,           int,                                               false)
REGISTER_TYPE(belief_no,            int,                                               false)
REGISTER_TYPE(belief_evidence,      std::string,                                       false)  // one line per record: "YYYY-MM-DD HH:MM | yes|no | <words> | <object>"
```
- No new claim attributes: new claim kinds reuse `human_claim_kind` / `_value` / `_uid` / `_words` / `_stamp`.

- [ ] **Step 1:** Add the lines, then commit in cortex: `git commit -m "attrs: identity link, LTOM questions and beliefs (changed-object)"`.
- [ ] **Step 2: STOP and ask the owner to reinstall cortex.** Then verify: `grep -c "question_memory_uid\|belief_evidence\|same_as_uid" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h`. Expected: `3`. Tasks 1, 3 and 4 do not need the attributes, so the tests in Tasks 3–4 may run before the reinstall. Tasks 5 and 6, and the graph steps of Tasks 3–4, wait for it.

### Task 3: LTOM: recall colour, the link record, and the "gone from room" record

**Files:**
- Modify:
  - `ltom_agent/src/repertory.h`, `repertory.cpp`: links and moves on `Entry`; `Sighting` gains the link.
  - `ltom_agent/src/graph_sightings.cpp`: read `same_as_uid` / `same_as_prob` / `left_spot_prob` from live nodes.
  - `ltom_agent/src/recall_select.h`, `.cpp`: skip gone-from-room and continued entries.
  - `ltom_agent/src/recall_publisher.cpp`: publish `color_name` / `color_prob`, plus `same_as_uid` / `same_as_prob` for a link.
  - `ltom_agent/src/repertory_io.cpp`: save and load the new fields; an older file loads with them empty.
  - `common/recall_match/recall_candidates.h`: read the colour and the link off recall nodes.
- Test:
  - `ltom_agent/tests/test_recall_select.cpp`
  - a new `ltom_agent/tests/test_repertory_links.cpp`, registered in `ltom_agent/tests/CMakeLists.txt` like `test_human`
  - `common/recall_match/recall_match_test.cpp`, for the `RecallCand` fields

**Interfaces:**
- Consumes: Task 1 (`LinkCand` field names).
- Produces:
```cpp
// repertory.h
struct LinkObs { std::uint64_t memory_uid = 0; float same_prob = 0.f, left_spot_prob = 0.f; };
// Sighting gains: std::optional<LinkObs> link;
struct Move { std::int64_t t_ms = 0; std::uint64_t live_uid = 0; Keyframe from, to; bool confirmed = false; };
struct GoneFromRoom { std::int64_t t_ms = 0, room_key = -1; std::string words; };
// Entry gains:
//   std::optional<std::uint64_t> continued_by;   // live uid this memory continues as (owner's B)
//   std::optional<std::uint64_t> continues;      // on the live uid's entry: the memory uid it continues
//   std::vector<Move> moves;  std::vector<GoneFromRoom> gone;
// Repertory gains:
void record_link(std::uint64_t memory_uid, std::uint64_t live_uid, std::int64_t now_ms);   // MAP link seen: sets both sides + a Move (from = memory's last keyframe, to = live's)
void confirm_link(std::uint64_t memory_uid);                                               // owner's yes
void cut_link(std::uint64_t memory_uid, std::int64_t now_ms, std::string words);           // owner's no: clears both sides, adds GoneFromRoom for the memory's last room
// RecallCand (recall_match.h) gains: std::string colour_name; float colour_prob = 0.f; std::uint64_t same_as_uid = 0; float same_as_prob = 0.f;
```
- Rules:
  - `Repertory::observe` calls `record_link` when a sighting's link has `same_prob > 0.5` (MAP of the binary {same, not}) and no link is held for that memory uid yet. When a held, unconfirmed link's sighting drops to `same_prob ≤ 0.5`, it cuts with `words = ""` and adds no GoneFromRoom.
  - `select_recalls` skips an entry whose `gone` has its last-keyframe room. An entry with `continued_by = X` is offered **only while X is present**, carrying `same_as_uid = X` (the chair agent then knows the candidate is taken by X); otherwise it is skipped.
  - A recall with `same_as_uid ≠ 0` is never a birth candidate: `recall_candidates` keeps it in the list but marks it. The birth path in Task 5 filters it out; the link path keeps it.

- [ ] **Step 1: Write the failing tests.**
  - `record_link_keeps_old_keyframe`: after `record_link(50, 57, t)`, `entries().at(50).moves.back().from` == 50's last keyframe; `continued_by == 57`; `entries().at(57).continues == 50`.
  - `gone_from_room_not_recalled`: after `cut_link(50, t, "no")`, `select_recalls(room 2)` does not contain 50; for room 1 it would (when its last keyframe is there).
  - `continued_offered_only_while_live_present`: 50 continued_by 57; present {57} → recalled with same_as 57; present {} → not recalled.
  - `link_dropped_below_map_cuts_without_gone`.
  - `repertory_io_roundtrips_links_and_questions` (links, moves, gone in this task; Task 4 extends it).
  - `older_file_loads_without_links`.
  - In `recall_match_test.cpp`: none needed; `recall_candidates` is DSR-bound and is checked in the Task 6 live run.
- [ ] **Step 2: Run and watch them fail.** `cmake -B ltom_agent/tests/build -S ltom_agent/tests && make -C ltom_agent/tests/build -j8 2>&1 | tail -5`. Expected: compile errors on `record_link`, `continued_by`.
- [ ] **Step 3: Implement** per the Interfaces. In `repertory_io.cpp`, use the field names `continued_by`, `continues`, `moves`, `gone`; a missing field means empty.
- [ ] **Step 4: Run all ltom tests and watch them pass.** `make -C ltom_agent/tests/build -j8 && for t in ltom_agent/tests/build/test_*; do $t || echo FAIL $t; done`. Expected: no `FAIL` line.
- [ ] **Step 5: Graph steps (after the reinstall).**
  - `graph_sightings.cpp`: fill `Sighting::link` when the node has `same_as_uid ≠ 0`.
  - `recall_publisher.cpp`: write `color_name_att` / `color_prob_att` from the entry, and `same_as_uid_att` / `same_as_prob_att` when offered as continued (prob = 1 if the Move is confirmed, else the last sighting's).
  - `recall_candidates.h`: read those four into `RecallCand`.
  - Build ltom_agent and chair_concept: `cmake -B build && make -C build -j8` in each.
- [ ] **Step 6: Commit.** `git commit -am "ltom: identity links (continued_by/continues, moves kept), gone-from-room records, recall colour; recall nodes carry the link"`

### Task 4: LTOM: questions, the answer claim, and beliefs

**Files:**
- Modify:
  - `ltom_agent/src/human.h`, `human.cpp`: `ClaimKind::moved`, `ClaimKind::forget`; beliefs on `Human`.
  - `ltom_agent/src/repertory.h`, `.cpp`: open questions.
  - `ltom_agent/src/repertory_io.cpp`: save questions and beliefs.
  - `ltom_agent/src/human_graph.cpp`: belief nodes.
  - `ltom_agent/src/memory_graph.cpp`: question nodes.
  - `ltom_agent/src/specificworker.cpp`: wiring.
- Test: `ltom_agent/tests/test_human.cpp`, `test_repertory_links.cpp`, `test_human_graph.cpp`, `test_memory_graph.cpp`

**Interfaces:**
- Consumes: Task 3 (`record_link`, `confirm_link`, `cut_link`, `LinkObs`, `Move`).
- Produces:
```cpp
// repertory.h
struct Question { std::uint64_t stamp = 0; std::uint64_t memory_uid = 0, live_uid = 0; std::int64_t room_key = -1;
                  Keyframe old_kf, new_kf; bool again = false; };
// Repertory gains: const std::map<std::uint64_t, Question>& questions() const;   // keyed by stamp
//   Opened inside observe(): when a held, unconfirmed link's latest sighting has left_spot_prob > 0.5 (MAP) and no
//   question is open for that memory uid. stamp = now_ms. again = the memory entry (or the one it continues) already has a confirmed Move.
//   Closed when: answered, the link is cut, or the live uid's life ends.
// human.h
enum class ClaimKind { name, owns, disowns, moved, forget };
// Claim for moved:  value = "yes" | "no" | "unknown" ; object_uid = memory uid ; words ; stamp. The question stamp travels as
//   value "yes:<question_stamp>" etc. — parse with std::from_chars.
// Claim for forget: value = belief key ("moves_things" | "gets_moved:<uid>").
struct BeliefRecord { Said said; bool yes = false; std::uint64_t object_uid = 0; };
struct Belief { std::string key; std::vector<BeliefRecord> records; std::vector<Said> forgotten; };   // forgotten: dated "forget that" notes
// Human gains: std::vector<Belief> beliefs;
// HumanBook::apply(const Claim&, Repertory&, now) — Repertory becomes non-const for moved claims (confirm/cut + close).
```
- Answer rules (each answers through the existing `Answer`, `answer_after_save`):
  - `moved yes`: `confirm_link`; add `BeliefRecord{yes}` to `moves_things` and to `gets_moved:<memory_uid>`; close the question → `"recorded"`.
  - `moved no`: `cut_link(words)`; add `BeliefRecord{no}` to `moves_things` only (the chair was not moved, so `gets_moved` is not about it); close → `"recorded"`.
  - `moved unknown`: close the question; no record; link kept → `"recorded"`.
  - Stale stamp, or no such question → `"rejected: that question is no longer open"`.
  - `forget <key>`: remove all records of that key and add a dated `forgotten` note → `"recorded"`. Records whose `said.t_ms` predates a forgotten note are never re-added from a replayed claim (same stamp ⇒ same answer already covers replays; the note is for the record).
- Graph (memory graph, after the reinstall):
  - one `metaconcept` node `ltom_question_<stamp>` under root per open question, with the Task 2 `question_*` attrs; deleted on close;
  - one `metaconcept` node `ltom_belief_<key>` per belief, with `belief_yes`, `belief_no`, `belief_evidence`;
  - both written only on change.

- [ ] **Step 1: Write the failing tests.**
  - `no_question_before_old_spot_seen_empty`: a link with left_spot_prob 0.3 → `questions()` empty; next sighting 0.8 → one question with old_kf == 50's last keyframe.
  - `question_names_link_winner`: entries 50 (magenta) and 44 (colourless) both in room 2, sighting link → 50 → question.memory_uid == 50.
  - `moved_yes_confirms_and_learns`: beliefs `moves_things` yes = 1, `gets_moved:50` yes = 1, the record's words are the claim's, question closed, link confirmed.
  - `moved_no_cuts_and_learns_no`: link cut, gone has room 2, `moves_things` no = 1, no `gets_moved` record.
  - `moved_unknown_closes_without_learning`.
  - `answer_to_closed_question_rejected`: rejected text starts with `"rejected: "`, beliefs unchanged.
  - `second_move_question_is_again`: after a confirmed Move, a new link from 57 → a new live chair opens a question with again == true.
  - `forget_drops_records_keeps_note`.
  - `human_io_roundtrips_beliefs`; extend `repertory_io_roundtrips_links_and_questions` with an open question.
  - In `test_memory_graph.cpp` / `test_human_graph.cpp`, in their existing in-process graph style: `question_node_published_and_removed`, `belief_node_lines`.
- [ ] **Step 2: Run and watch them fail.** As in Task 3 step 2. Expected: compile errors on `ClaimKind::moved`, `questions()`.
- [ ] **Step 3: Implement.** `parse_claim_kind` / `claim_kind_name` learn `"moved"` and `"forget"`. `belief_evidence` lines use `rc::provenance`-style local time `YYYY-MM-DD HH:MM`, words verbatim, and the object's display name.
- [ ] **Step 4: Run all ltom tests and watch them pass.** Same command. Expected: no `FAIL`.
- [ ] **Step 5: Build ltom_agent** (`make -C build -j8`). Expected: links clean.
- [ ] **Step 6: Commit.** `git commit -am "ltom: questions on a moved object, the owner's moved/forget answers, dated beliefs (moves_things, gets_moved)"`

### Task 5: Chair agent: ghost absence at old spots, and the published link

**Files:**
- Modify:
  - `chair_concept/src/specificworker.cpp`: around the birth block, lines ~804–880, where `recall_cands` is gathered.
  - `chair_concept/src/chair_fitter.h`, `.cpp`: ghost evaluation.
  - `chair_concept/src/chair_scene_graph.h`, `.cpp`: write the link attrs.
  - `chair_concept/src/chair_config.h`, `chair_concept/etc/config.toml`: `[Tracker]` gets `LinkPiMove = 0.10`, `LinkPiGone = 0.05`, `LinkLambdaNew = 0.05`, each with a one-line comment that it is a guess that the learned `gets_moved` belief will replace later.
- Test: `chair_concept/tests/`, if a pure ghost helper is split out. Otherwise this is covered by the Task 6 live run plus step 6's replay check.

**Interfaces:**
- Consumes: Task 1 (`link_posterior`, `colour_log_ratio`), Task 3 (`RecallCand` colour and `same_as_*`).
- Produces: on every live chair node with a candidate, `same_as_uid_att`, `same_as_prob_att` and `left_spot_prob_att`. When no candidate remains, all three are removed (or `same_as_uid` = 0). The node's name is never changed.
- Rules:
  - **Birth path:** filter out recall candidates with `same_as_uid ≠ 0` before `choose_recall` (they are taken).
  - **Ghosts:** one `rc::ExistenceBelief` per untaken recall uid in the current room, created at `L0 = logit(exist_prob)` (0 if unstated). Each cycle, evaluate P(detect) for a chair-sized box at the recall pose, by the same route a live instance uses:
    - a transient `ChairInstance` with `model` state at (mu, yaw 0) and the prior seat size;
    - `compute_projected_roi`, `zed_detectability` and `los_occlusion`;
    - `p_vis = pd · (1 − occ)`, the absence ratio `−g` (`exist_vacate` gain as in `chair_existence.cpp:186–203`);
    - `integrate(p_vis, ratio)` and `note_viewpoint`.

    A ghost is dropped when its recall leaves the list (taken at birth, or LTOM stopped offering it). `still_there_llr = L − L0`, and `left_spot_prob = 1 − p_exists()`.
  - **Link:** each cycle, for each live chair born this run that did not resurrect, plus each live chair whose uid appears as some candidate's `same_as_uid`, run `link_posterior` over the untaken candidates (plus the one naming it). Feed it `colour_now` from `name_colour(inst.appearance).probs` (nullopt when `name` is empty).
    - A candidate whose `same_as_uid` is this chair and whose `same_as_prob` is 1 is confirmed: publish 1.
    - Otherwise publish the MAP candidate's uid, its posterior, and its ghost's `left_spot_prob`. With no MAP candidate, remove the attrs.
  - Each candidate is linked to at most one live chair: the one with the highest posterior.

- [ ] **Step 1: Implement the config keys and the birth filter.** Build: `make -C chair_concept/build -j8`. Expected: clean.
- [ ] **Step 2: Implement the ghosts**, logging one line per change of `left_spot_prob` across 0.5: `[link] ghost chair_50 left_spot p=…`.
- [ ] **Step 3: Implement the link and its publication.** Log `[link] chair_57 -> chair_50 p=… left=…` on change.
- [ ] **Step 4: Run the chair unit tests and the common tests.** `bash common/run_tests.sh 2>&1 | tail -5`, then the chair_concept tests target if present. Expected: all pass.
- [ ] **Step 5: Commit.** `git commit -am "chair_concept: ghost absence at remembered spots; identity link (same_as_uid/prob, left_spot_prob) on live chairs"`
- [ ] **Step 6: Gone-path check.** In the Task 6 run C log, after the "no", `[link]` for that live chair shows no candidate, and it never re-links to chair_50 in that run.

### Task 6: language_agent: ask, re-raise, report, answer, and "what have you learned"

**Files:**
- Create:
  - `language_agent/src/question_policy.h` (pure): mode, wording, raise state.
  - `language_agent/src/place_words.h` (pure): "by door_25", "in the corner", "near the middle of the room".
  - `language_agent/tests/test_question_policy.cpp`, `test_place_words.cpp`, both added to `run_tests.sh`'s first loop.
- Modify:
  - `language_agent/src/ltom_query_dsr.h`: `load_questions`, `load_beliefs`.
  - `language_agent/src/object_facts.h`: `ObjectFact` gains `same_as_uid`, `same_as_prob`; `mark_owned` treats a live object as owned when its `same_as_uid` is owned and `same_as_prob > 0.5`.
  - `language_agent/src/world_digest.cpp`: read `same_as_*`.
  - `language_agent/src/chat_session.h`, `.cpp`: a `track_questions()` beside `track_object_changes()` in the tick (line ~323); the answer path in `on_user_text`; new tools.
  - `language_agent/src/skills.h` (the tool schema): tools `what_i_learned` and `forget_belief`.
  - `language_agent/etc/config.toml`: `[Questions] CostAsk = 0.4`, `CostWrong = 1.0`, with a comment that one yes tips to report: P = 2/3 → 0.33 < 0.4; none → 0.5 > 0.4.
- Test: the files above, plus `language_agent/tests/test_human_claim.cpp`.

**Interfaces:**
- Consumes: Task 4's graph nodes and claim kinds (`moved` with value `"yes:<stamp>"` / `"no:<stamp>"` / `"unknown:<stamp>"`, and `forget`).
- Produces:
```cpp
namespace rc::chat {
struct OpenQuestion { std::uint64_t stamp, memory_uid, live_uid; std::string room, colour, memory_name;
                      Eigen::Vector2f old_xy, new_xy; bool again; };
struct BeliefFacts { std::string key; int yes = 0, no = 0; std::vector<std::string> evidence; };
enum class QMode { ask, report };
[[nodiscard]] QMode choose_mode(int yes, int no, float cost_ask, float cost_wrong);   // p = (1+yes)/(2+yes+no); report iff (1-p)*cost_wrong < cost_ask
enum class Reply { yes, no, unknown, other };
[[nodiscard]] Reply classify_reply(std::string_view text);     // fleet::is_yes, plus no/"I didn't"/"no lo moví", "don't know"/"no sé", else other
[[nodiscard]] std::string question_text(const OpenQuestion&, QMode, bool owner_was_away, std::string_view old_place, std::string_view new_place);
struct RaiseState { int raised = 0; };                          // per stamp: raised on open (1), once more after the next non-answer (2), then silent
[[nodiscard]] std::string place_words(const Eigen::Vector2f& xy, std::span<const Eigen::Vector2f> room_polygon,
                                      std::span<const std::pair<std::string, Eigen::Vector2f>> doors);
}
```
- Behaviour:
  - **On a new open question:**
    - mode = `choose_mode(moves_things.yes, moves_things.no, …)`;
    - The first raise uses the ask or report form. `owner_was_away = true` is used only for the second raise, the one after the owner's next message, because a question that waited for that message was by definition asked while they were away. No time window is involved.
    - Say the text into the chat at once, push it to the history as an assistant turn, and set `raised = 1`.
  - **On a user message while a question is raised:**
    - `classify_reply`: yes, no or unknown → post the `moved` claim immediately (no read-back card: the question was the read-back), then say `claim_recorded_words`-style text.
    - `other`: handle the message as usual. After the reply, if `raised == 1`, say the question again (the away form) and set `raised = 2`.
    - Only one raised question is live at a time: the oldest open one.
  - **Report mode:** said once. A later "no" / "I didn't" still classifies against it and posts `moved no`. No answer → nothing is posted.
  - **`what_i_learned`** returns the belief evidence lines, which the LLM phrases. **`forget_belief(key)`** goes through the existing claim card (read-back + yes), kind `forget`.
  - **If the owner is in the middle of a mission:** while a planner or mission card is pending (`planner_` or `pending_mission_`), the raise waits until the card is resolved.

- [ ] **Step 1: Write the failing tests.**
  - `test_question_policy.cpp`:
    - `choose_mode(0,0,.4,1) == ask`; `choose_mode(1,0,.4,1) == report`; `choose_mode(1,1,.4,1) == ask`.
    - `classify_reply("yes, I moved it") == yes`; `("no, I didn't") == no`; `("no sé") == unknown`; `("go to the kitchen") == other`.
    - `question_text` ask form == `"The magenta chair in room 2 is by the window now; it was by door_25. Did you move it?"` for those place strings.
    - The away form starts with `"By the way, while you were away the magenta chair in room 2 moved "`.
    - The report form with again == `"The magenta chair has moved again, in the corner. I assume you moved it."`
  - `test_place_words.cpp`: a point 0.4 m from a door → `"by door_25"`; a point 0.3 m from a corner with no door near → `"in the corner"`; the room centre → `"near the middle of the room"`. The nearest landmark is chosen by distance; there is no cutoff.
  - `test_human_claim.cpp`: a `moved` claim's value round-trips `"no:1763200000000"`.
- [ ] **Step 2: Run and watch them fail.** `bash language_agent/tests/run_tests.sh 2>&1 | tail -10`. Expected: compile error, `question_policy.h` not found.
- [ ] **Step 3: Implement** `question_policy.h`, `place_words.h`, the `human_claim.h` change, and the `object_facts.h` change.
- [ ] **Step 4: Run and watch them pass.** Same command. Expected: all pass, including `test_mission_plan` and the rest of the existing suite.
- [ ] **Step 5: Wire it into ChatSession** (`track_questions`, the answer path, the tools, the config). Build: `make -C language_agent/build -j8`. Expected: clean.
- [ ] **Step 6: Commit.** `git commit -am "language_agent: asks or reports a moved object from LTOM's question, re-raises once, answers back as a moved claim; what_i_learned / forget_belief"`

### Task 7: Live runs (Webots)

**Files:** none in code. The run folder goes under `room_concept/estimation_scripts/crossing/` as `moved_chair_<n>`, with a short `REPORT.md` there.

- [ ] **Step 0:** CLAIM Webots with room and objects; wait for OK. Use the fixture that remembers chair_50 in room 2 (the world already has the chair moved). Fleet as in `chat_run6.sh`, with ltom_agent, chair_concept and language_agent from the worktree build.
- [ ] **Run A (owner in the chat):** write something to the chat, then send the robot to room 2. Expect:
  - `[link] chair_N -> chair_50`;
  - the ghost going left_spot > 0.5 once the old spot is in view;
  - then the ask-form question, with both places right.

  Answer "yes, I moved it". Check on the memory graph: `ltom_belief_moves_things` yes = 1 with the words, `gets_moved:<50>` yes = 1, the question node gone.
- [ ] **Run B (owner away):** reset LTOM to the pre-run file (keep a copy). Drive to room 2 without writing. Expect the link and the decision with the task not paused. Then write "hello". Expect the reply, then the away-form question.
- [ ] **Run C ("no"):** reset as in B. Answer "no". Expect `gone` in room 2 for 50, the link attrs removed from the live chair, no re-link (Task 5 step 6), and `moves_things` no = 1.
- [ ] **Visit 2:** after run A's state, move the chair in Webots (ask the owner to do it) and restart the fleet. Expect the report form with "again" and "I assume". Answer "no, I didn't" and expect `moves_things` no = 1.
- [ ] **Inspect:** ask "What have you learned about me?" and expect the dated evidence. Ask "Forget that", confirm the card, and expect the belief gone and a forgotten note in the file.
- [ ] **Pass criterion (spec):** every question names the chair that moved, and nothing is asked about a chair that did not move. Grep the logs for every question text and check each against the `[link]` lines.
- [ ] Stop the fleet with SIGTERM and FREE Webots. Commit the REPORT.md only if the owner asks.

## Rulings in this plan (for the owner to check)

- **Fading:** a question closes when it is answered, when its link is cut, or when the live chair dies. The spec's "its worth fades as evidence settles identity" is deferred. The answer also teaches who moved the chair, and evidence never settles that, so fading on identity alone would throw away the learning. Cost if wrong: an old question stays open (silent after two raises) until the chair dies.
- **The link is published by the chair agent and recorded by LTOM**, not decided by LTOM. This keeps the spec's split.
- **The `gets_moved` belief** is recorded and published, but the ask/report choice reads only `moves_things`. The spec gives the owner's belief as the reason for the visit-2 report.
- **Ghost P(detect)** reuses the live chair's projection through a transient instance rather than a new projector, which keeps a single definition of "would have seen it".
