# The human at the chat, part 1 (name and things) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The robot remembers the one human at the chat — their name and the objects they own — across restarts, and the chat uses it ("go to my chair"), never storing a link without the person's yes.

**Architecture:** LTOM owns a `HumanBook` (pure C++, persisted in the repertory file) and reflects it on its memory graph as a `person` node `human_1` with `owns` edges to object memory nodes. language_agent proposes a claim as a card, and after the person's yes posts it as attributes on its own `chat_intention` node; LTOM records it, answers on `human_1`, and the chat clears the claim on seeing the answer.

**Tech Stack:** C++23, Qt6 Core (JSON), DSR/cortex, Eigen; tests are standalone executables (ltom_agent/tests CMake, language_agent/tests/run_tests.sh); tool-choice eval `language_agent/tests/check_llm_tools.py` against gemma4:26b on the Orin.

**Spec:** `docs/superpowers/specs/2026-11-11-human-model-design.md` (approved 2026-11-11).

**Branch / worktree:** `active_inference.worktrees/object-colour` (feat/object-colour). Commit only when the owner asks; the "Commit" steps below are checkpoints to propose, not to run unasked.

## Global Constraints

- ONE human: id `"human_1"`, role `"developer"`; on disk a LIST `"humans"` so more can come later; no code path chooses between humans.
- Ownership keyed on `object_uid` (uint64; JSON string, as the repertory already writes uids); `chair_N` is display only.
- Never store an ownership without the person's yes in the chat (anti-goal); the model may propose, only code stores.
- Retractions and name changes are kept as history, never deleted.
- Type-attributed DSR access only (`*_att`), no `runtime_checked_*`; DSR signals Queued only (none needed here).
- Numbers parsed with `std::from_chars`; data files written through `std::locale::classic()`.
- `and`/`or`/`not`; C++23 containers and algorithms; no thresholds (the claim answer wait is an operational timeout, flagged).
- Builds via the capped scope: `systemd-run --user --scope -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=8G make -C build -j8`, empty `CMAKE_BUILD_TYPE`.
- Stop agents with SIGTERM, never `kill -9`. Claim/free Webots with room-concept-fd around live runs.

## Review Focus

1. **The yes binds only to the pending claim.** A yes with a mission card AND a claim pending must not store the claim twice or start the mission instead — the newest card wins; test in Task 7.
2. **A claim for an object LTOM has no entry for** (just born, not yet observed) is rejected with a reason, and the chat says so; test in Task 1 + Task 7.
3. **"My chair" with two owned chairs** asks which; with one owned chair that is only remembered, it goes to the remembered one; test in Task 6.
4. **A repertory file without `"humans"`** (today's file) loads with no human and saves back WITH an empty-or-filled section without losing entries; test in Task 2.
5. **The human node never appears as a perceived object** ("what do you see?"); test in Task 5.

---

### Task 0: Cortex registrations (owner reinstalls cortex)

**Files:**
- Modify: `~/robocomp/components/cortex/core/include/dsr/core/types/type_checking/dsr_edge_type.h` (source tree, not /usr/local)
- Modify: `~/robocomp/components/cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`

**Produces:** edge type `owns` (`owns_edge_type`); attributes `human_claim_kind_att` (`std::reference_wrapper<const std::string>`), `human_claim_value_att` (string), `human_claim_uid_att` (`std::uint64_t`), `human_claim_words_att` (string), `human_claim_stamp_att` (`std::uint64_t`), `human_claim_answer_att` (string), `human_claim_answer_stamp_att` (`std::uint64_t`).

- [ ] **Step 1:** Add `REGISTER_EDGE_TYPE(owns)` with a comment block in the style of `exit` (person → object memory node; written by ltom_agent only; meaning: the person said, and confirmed, that they own it).
- [ ] **Step 2:** Add the seven `REGISTER_TYPE` lines under one comment block naming the protocol (writer: language_agent on `chat_intention` for `human_claim_*`; ltom_agent on `human_1` for `human_claim_answer*`).
- [ ] **Step 3:** Ask the owner to reinstall cortex; verify with `grep -c "human_claim_" /usr/local/include/dsr/core/types/type_checking/dsr_attr_name.h` → `7` and `grep -c "REGISTER_EDGE_TYPE(owns)"` → `1`. Nothing below that touches DSR compiles before this.

### Task 1: LTOM `HumanBook` (pure)

**Files:**
- Create: `ltom_agent/src/human.h`, `ltom_agent/src/human.cpp`
- Test: `ltom_agent/tests/test_human.cpp` (+ `add_executable(test_human test_human.cpp ../src/human.cpp ../src/repertory.cpp)` in `ltom_agent/tests/CMakeLists.txt`)

**Interfaces — Produces:**
```cpp
namespace ltom {
struct Said { std::int64_t t_ms = 0; std::string words; };
struct Ownership { std::uint64_t object_uid = 0; std::string cls, name_when_said; Said said; std::optional<Said> retracted; };
struct NameRecord { std::string value; Said said; };
struct Human { std::string id = "human_1", role = "developer"; std::vector<NameRecord> names;   // last = current
               std::vector<Ownership> things; };
enum class ClaimKind { name, owns, disowns };
struct Claim { ClaimKind kind; std::string value; std::uint64_t object_uid = 0; std::string words; std::uint64_t stamp = 0; };
struct Answer { std::uint64_t stamp = 0; std::string text; };   // "recorded" | "rejected: <reason>"
std::optional<ClaimKind> parse_claim_kind(std::string_view);     // "name" | "owns" | "disowns"
class HumanBook {
public:
    const Human& human() const;                                  // the one human (created empty)
    std::optional<std::string> name() const;                     // current name
    std::vector<const Ownership*> owned() const;                 // not retracted, in claim order
    Answer apply(const Claim& c, const Repertory& r, std::int64_t now_ms);   // records, never throws
    void set(Human h);                                           // the loader's door
    bool dirty() const; void clear_dirty();
};
}
```
Rules `apply` implements: `name` with empty value → `rejected: no name`; a new name appends a `NameRecord` (history). `owns` for a uid with no repertory entry → `rejected: I have no memory of that object`; already owned and not retracted → `recorded` without a duplicate; otherwise appends an `Ownership` (cls and name_when_said from the entry). `disowns` sets `retracted` on the live link (`rejected: it was not yours` when none). A claim whose `stamp` equals the last applied stamp is ignored and returns the same answer (idempotent: LTOM sees the claim every cycle until the chat removes it).

- [ ] **Step 1: Write the failing tests** — `test_name_is_kept_with_history` (two names → `name()=="Paul"`, `human().names.size()==2`), `test_owns_needs_a_memory_entry` (unknown uid → answer starts with `"rejected:"`, `owned().empty()`), `test_owns_records_once` (same claim twice, different stamps → one `Ownership`, both answers `"recorded"`), `test_disown_keeps_history` (owns then disowns → `owned().empty()`, `human().things.size()==1`, `things[0].retracted.has_value()`), `test_reclaim_after_disown_is_a_new_entry` (→ `things.size()==2`, `owned().size()==1`), `test_same_stamp_is_applied_once` (`apply` twice with the same stamp → `dirty()` false after `clear_dirty()` between them), `test_two_owned_chairs` (two uids → `owned().size()==2`).
- [ ] **Step 2:** Build and run `test_human`; expect compile failure, then (with an empty `HumanBook` stub) assertion failures.
- [ ] **Step 3:** Implement `human.h/.cpp` as specified.
- [ ] **Step 4:** Run all ltom tests (`make -C <tests build> -j8` then each `test_*`); expect all pass.
- [ ] **Step 5: Commit checkpoint** `ltom: HumanBook — the human's name and owned things (pure)`.

### Task 2: Persist the human in the repertory file

**Files:**
- Modify: `ltom_agent/src/repertory_io.h`, `ltom_agent/src/repertory_io.cpp`
- Test: `ltom_agent/tests/test_repertory_io.cpp` (link `../src/human.cpp`)

**Interfaces:**
- Consumes: `HumanBook`, `Human` (Task 1).
- Produces: `LoadResult load_repertory(const std::filesystem::path&, Repertory&, HumanBook*)` and `bool save_repertory(const std::filesystem::path&, const Repertory&, const HumanBook*)` — the pointer is optional (`nullptr` = old behaviour); the two-argument forms stay and forward with `nullptr`.

JSON shape (spec §On disk): `"humans": [{ "id", "role", "names": [{value,t_ms,words}], "things": [{object_uid (string), cls, name_when_said, t_ms, words, retracted?: {t_ms, words}}] }]`. Note: names is a LIST (history, spec call 7), not the spec example's single object.

- [ ] **Step 1: Failing tests** — `test_human_round_trip` (name history of 2, two things one retracted → identical after save+load, uid `(20ull<<40)|123456789ull` exact), `test_file_without_humans_loads` (today's file shape → `LoadResult::loaded`, `book.name()` empty, entries intact), `test_save_keeps_entries_with_humans` (entry count unchanged).
- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement; reuse the file's existing uid-as-string and atomic `.prev` handling.
- [ ] **Step 4:** All ltom tests pass.
- [ ] **Step 5: Commit checkpoint** `ltom: the human is saved with the repertory`.

### Task 3: The human on LTOM's memory graph + reading claims

**Files:**
- Create: `ltom_agent/src/human_graph.h`, `ltom_agent/src/human_graph.cpp`
- Test: `ltom_agent/tests/test_human_graph.cpp` (fixture like `test_memory_graph.cpp`, domain 63, `ltom_root.json`)

**Interfaces:**
- Consumes: `HumanBook`, `Claim`, `Answer`, `parse_claim_kind` (Task 1); Task 0 attributes.
- Produces:
```cpp
namespace ltom {
std::optional<Claim> read_claim(DSR::DSRGraph& working);          // chat_intention's human_claim_* attrs; nullopt if absent/incomplete
class HumanGraph { public:
    explicit HumanGraph(std::shared_ptr<DSR::DSRGraph> memory);
    // person node "human_1" under root (person_name = current name if any, person_role), `owns` edges to the memory
    // nodes of owned (non-retracted) uids (found by object_uid_att), stale `owns` edges removed; the last answer as
    // human_claim_answer + human_claim_answer_stamp. Repeatable; writes only on change.
    void sync(const HumanBook& book, const std::optional<Answer>& last);
};
}
```
- [ ] **Step 1: Failing tests** — `test_person_node_with_name` (`type()=="person"`, `person_name_att=="Pablo"`, `person_role_att=="developer"`), `test_owns_edge_to_the_memory_node` (memory node with `object_uid` X present → edge `human_1 → it` of type `owns`), `test_retracted_link_not_drawn`, `test_answer_published` (stamp and text), `test_read_claim` (attrs on a `chat_intention` node in a working-graph fixture → `Claim{owns, "", X, "that's my chair", 77}`; missing `human_claim_stamp` → nullopt).
- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. The memory nodes are `MemoryGraph`'s (ltom_agent/src/memory_graph.cpp); find them by `object_uid_att`, do not modify them.
- [ ] **Step 4:** All ltom tests pass.
- [ ] **Step 5: Commit checkpoint** `ltom: the human and its things on the memory graph; claims read from the chat`.

### Task 4: Wire it into ltom_agent

**Files:** Modify `ltom_agent/src/specificworker.h`, `ltom_agent/src/specificworker.cpp` (load at :190, save at :214, compute loop near :253), `ltom_agent/src/CMakeLists.txt` (add `human.cpp`, `human_graph.cpp`).

**Interfaces:** Consumes Tasks 1–3. Each compute: `if (auto c = read_claim(*G_working)) last_answer_ = book_.apply(*c, *repertory_, now)`; `human_graph_->sync(book_, last_answer_)`; save when `repertory_->dirty() or book_.dirty()` (both cleared after a successful save). Log `[ltom] human claim <kind> -> <answer>` once per stamp.

- [ ] **Step 1:** Build ltom_agent (capped scope); expect rc 0.
- [ ] **Step 2:** Offline check: start nothing; run `load_check`-style harness (`scratchpad/load_check.cpp` pattern) on the main repertory file → loads, no human.
- [ ] **Step 3: Commit checkpoint** `ltom: keeps the human at the chat`.

### Task 5: language_agent reads the human

**Files:** Modify `language_agent/src/ltom_query_dsr.h`, `language_agent/src/object_facts.h`; Test `language_agent/tests/test_objects.cpp`.

**Interfaces — Produces:**
```cpp
namespace rc::chat {
struct HumanFacts { std::optional<std::string> name; std::string role; std::set<std::uint64_t> owned_uids;
                    std::optional<std::uint64_t> answer_stamp; std::string answer; };
HumanFacts load_human(DSR::DSRGraph& memory);                  // ltom_query_dsr.h; empty when no human_1
std::string owner_words(std::uint64_t uid, const HumanFacts&); // object_facts.h: ", Pablo's" | ", yours" (no name) | ""
}
```
`ObjectFact` and `RememberedObject` gain `std::uint64_t uid = 0` (filled from `object_uid_att`); the object line and `remembered_line` append `owner_words` after the colour (spec call 6). Object listings take only `object`-typed memory nodes and skip any node named `human_*` (spec call 1).

- [ ] **Step 1: Failing tests** — `owner_words` with a name → `", Pablo's"`; without → `", yours"`; not owned → `""`; an object line for an owned chair contains `"magenta (p 0.69), Pablo's"`; `answer_find_objects` on a fact list that includes a `person` named `human_1` does not list it.
- [ ] **Step 2:** Run `language_agent/tests/run_tests.sh`; expect the new checks to fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `run_tests.sh` passes.
- [ ] **Step 5: Commit checkpoint** `language_agent: reads the human and marks their things`.

### Task 6: Resolving "that" and "my <class>" (pure)

**Files:** Create `language_agent/src/human_refs.h`; Test `language_agent/tests/test_human_refs.cpp` (add to `run_tests.sh` pure loop).

**Interfaces — Produces:**
```cpp
namespace rc::chat {
struct RefCandidate { std::string name; std::uint64_t uid = 0; double weight = 0; };
/// "that": objects in view weighted by centredness (cos of bearing, only |bearing| < camera half-FOV from the
/// digest's camera model), plus objects named in the last `recent_names` turns, plus the last approach target.
/// Sorted by weight, highest first; the caller asks when more than one remains.
std::vector<RefCandidate> that_candidates(const std::vector<ObjectFact>& live, std::span<const std::string> recent_names,
                                          const std::optional<std::string>& last_target, double half_fov_deg);
/// "my chair": owned objects of the class, live first then remembered; empty = none owned.
std::vector<RefCandidate> my_candidates(std::string_view cls, const HumanFacts&, const std::vector<ObjectFact>& live,
                                        const std::vector<RememberedObject>& remembered);
}
```
The weights combine as a sum of the three evidence sources' normalised scores; no cutoff decides — the CALLER always reads back and asks.

- [ ] **Step 1: Failing tests** — centred chair outranks one at 40°; a chair named last turn outranks an off-centre one; a chair behind the robot (outside FOV) is not a candidate unless named; `my_candidates("chair")` with two owned chairs returns both; with one owned only remembered returns it; with none returns empty.
- [ ] **Step 2–4:** fail → implement → `run_tests.sh` passes.
- [ ] **Step 5: Commit checkpoint** `language_agent: what "that" and "my chair" refer to`.

### Task 7: The claim card, the yes, and the answer

**Files:** Modify `language_agent/src/chat_session.h/.cpp`, `language_agent/src/intention_node.h/.cpp`; tool text mirrored in `language_agent/tests/check_llm_tools.py`.

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces: `IntentionNode::post_claim(const HumanClaim&)` / `clear_claim()` (attrs on `chat_intention`); in ChatSession a `struct ClaimCard { HumanClaim claim; std::string readback; std::uint64_t serial, turn, posted_stamp = 0; std::int64_t posted_ms = 0; }` held in `std::optional<ClaimCard> pending_claim_`.
- New tool `human_claim` (`kind`: name | owns | disowns; `object`: a name, "that", or "my <class>"; `name`: for kind name). It never stores: it resolves (Task 6), and shows a card — "chair_34, the magenta one ahead, is yours?" — or asks which when more than one candidate (the person's next message is a normal turn). A `name` claim is read back too ("Your name is Pablo?").
- Yes fast path (next to 1c): a plain yes (`fleet::is_yes`) with `pending_claim_` set and still valid posts the claim. If a mission card is also pending, the NEWER card (higher serial) takes the yes (Review Focus 1). Any other message drops the claim card.
- Answer: each cycle, `load_human(*ltom_G_)`; when `answer_stamp == posted_stamp` tell the person the answer once, `clear_claim()`, drop the card. No answer within 5 s (`kClaimAnswerWaitMs`, flagged as an operational timeout) → "LTOM is not answering; nothing was stored", `clear_claim()`.
- `do_approach`, `do_where_is`, `do_find_objects` and the remembered-goto fast path accept "my <class>" / "mine": resolved with `my_candidates` (Task 6); two candidates → ask which; none → "I don't know which <class> is yours yet".
- STATE gains "You are talking to <name> (<role>)." when a name is known; the interface rules say ownership is changed only through `human_claim`.

- [ ] **Step 1: Failing tests (pure parts)** — readback text for an owned-claim card contains the object's name and colour; `newer_card_takes_the_yes(mission_serial, claim_serial)` picks the higher; wait expiry `claim_answer_overdue(posted_ms, now_ms)` true at 5001 ms, false at 4999.
- [ ] **Step 2: Eval cases (RED)** in `check_llm_tools.py`: "that's my chair" → `human_claim{kind: owns, object: that}`; "my name is Pablo" → `human_claim{kind: name, name: Pablo}`; "that is not my chair" → `disowns`; "go to my chair" → `approach_object` with object containing "my chair" (NOT `human_claim`); "where is my chair?" → `where_is` with "my chair"; "call it the reading chair" → NOT `human_claim`. Run with `OLLAMA=http://192.168.1.35:11434 ONLY="my chair|my name|not my|reading chair" python3 tests/check_llm_tools.py gemma4:26b 2`; expect FAIL (tool missing).
- [ ] **Step 3:** Implement; `run_tests.sh` passes; rerun the eval → all new cases pass, and the full eval shows no new failures against the 2026-11-11 baseline (42/44).
- [ ] **Step 4: Commit checkpoint** `language_agent: the person names themself and claims their things, with a read-back`.

### Task 8: Live check (Webots, owner present)

- [ ] **Step 1:** Claim Webots with room-concept-fd; `bash chat_run4.sh chat_e2e11` (scratchpad launcher; ltom runs from the main `ltom_agent` dir).
- [ ] **Step 2:** In the chat: "my name is Pablo" → yes; "that's my chair" while chair_34 is centred → read-back names chair_34 → yes → "recorded"; "where is my chair?" → chair_34; stop and restart ltom_agent (SIGTERM) → "what is my name?" → Pablo, and the chair line still says "Pablo's".
- [ ] **Step 3:** Check `ltom_agent/etc/ltom_repertory.json` has the `"humans"` section with both statements and their words; LTOM's rate unchanged (`[ltom]` cycle log before/after).
- [ ] **Step 4:** Stop the fleet (stop file + SIGTERM), free Webots, report.
