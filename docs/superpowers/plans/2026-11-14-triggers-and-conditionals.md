# Triggers and Conditionals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Missions gain `watch <pred> (continue|stop) [each=NAME]` + reaction body, `if <pred>` + then / `else`
bodies, and the predicate `seen(<class>|<node>)`; the controller answers it; the chat plans, shows and reports them.

**Architecture:** Everything control-flow lives in the shared mission interpreter (`common/mission/`). The
controller only adds one set-valued Host query (which object nodes match `seen(X)` now); the interpreter turns
"present now and not at arming" into firings. Firings travel to the chat as two optional fields on each step of the
existing progress snapshot (the watch's count and the list of objects it fired on).

**Tech Stack:** C++23 header-only `common/mission` (Qt6Core for JSON), controller (RoboComp/DSR, rt-build layout),
language_agent (Qt6, ollama gemma4:12b).

**Spec:** `docs/superpowers/specs/2026-11-14-triggers-and-conditionals-design.md` (read it first; its "Settled after
the builder check" section is binding).

## Global Constraints
- Work in a new worktree `active_inference.worktrees/mission-triggers`, branch `feat/mission-triggers` from `main`
  db8d39e3. **No commits** — the owner commits; each task ends with its tests green, not a commit.
- The controller change is written here (main's `controller/` equals rt-build's at 1444c335) and **reviewed by the
  room session before any merge**.
- CLAUDE.md: `and`/`or`/`not`; `std::from_chars` only; no thresholds beyond the person's stated boundary (the
  planner writes 0.5 when none is said); `make -j8`; never `kill -9`.
- Unknown op / predicate still rejects the whole mission (§7.1). New `Host` methods have defaults so an old Host
  compiles.
- Interpreter invariant kept: at most one leaf started and not finished.

## Review Focus
1. An object already in the graph when a watch arms never fires it ("stop when you see a chair" in a room with a known chair).
2. A class watch fires once per object: a chair that vanishes and comes back (recall/resurrection) does not fire again.
3. Firings that arrive while a moving reaction runs are queued and each runs once, in arrival order.
4. A `stop` watch firing mid-`cross` cancels that leaf exactly once and the mission ends `completed`, not failed.
5. Two firings between two chat snapshots give two report wakes, each about its own object (the `fired` list, not the step state, is compared).

---

### Task 1: IR — `watch`, `if`, `seen` and their validation (JSON form)

**Files:**
- Modify: `common/mission/mission_ir.h` (ops table ~249, preds ~299, `Validator` ~375)
- Modify: `common/mission/mission_json.h` (nothing new expected: `steps` already carries bodies — verify round trip)
- Test: `common/mission/mission_ir_test.cpp`

**Interfaces:**
- Produces ops: `watch` {claims kNone, instant false, has_body true, fields: `when` Pred required positional,
  `then` String required positional (`"continue"` | `"stop"`), `each` String}; `if` {claims kNone, has_body true,
  fields: `cond` Pred required positional}. An `if`'s `steps` = one `seq` (then) or two (then, else).
- Produces pred: `seen` {probabilistic false, fields: `what` Ref required kNode} — `what` names a class
  (`chair`) or a node (`chair_34`); the IR does not tell them apart.
- `$NAME` from `each` is visible only inside that watch's body (Validator `vars_` push before, pop after).

- [ ] **Step 1: Write the failing tests** — `test_watch_if_validate()`:
  - valid: `watch` (when=seen(chair), then=continue, each=c, body [report "the colour of $c"]) followed by `cross door_22` → `validate()` ok.
  - `watch` as the LAST step of its list → `bad_value`, text contains "needs steps after it".
  - `then="maybe"` → `bad_value` on `then`.
  - `approach $c` AFTER the watch (outside its body) → error (unbound var), inside the body → ok.
  - `if` with 1 seq child ok, 2 ok, 0 → "if needs a body", 3 → `bad_value` ("then and at most one else"), a non-seq child → `bad_value`.
  - `seen` with a cmp (`seen(chair)>=0.5`) → error (a fact takes no boundary).
  - JSON round trip of both shapes: `decode_json(encode_json(m)) == m`.
- [ ] **Step 2: Run** `cd language_agent && bash tests/run_tests.sh` — expected: mission_ir_test FAILs (unknown op `watch`).
- [ ] **Step 3: Implement** the two OpSpec entries + `check_watch`/`check_if` (field values, child shape), the
  `seen` PredSpec, and in `Validator::steps(list)` the "watch needs a following sibling" rule and the `each` scope.
- [ ] **Step 4: Run** the same command — expected: mission_ir_test 0 failures, all other suites unchanged.

### Task 2: Text form — `watch … continue each=c`, `if …` / `else`

**Files:**
- Modify: `common/mission/mission_text.h` (parser `levels` ~503–580, `print_steps` ~465)
- Test: `common/mission/mission_text_test.cpp`

**Interfaces:**
- Consumes: Task 1 ops.
- Text forms (canonical print = these exactly):
  ```
    watch seen(chair) continue each=c
      report "the colour of $c"
    cross door_22 expect=room_3
  ```
  ```
    if door_open(door_1)<0.5
      open door_1
    else
      cross door_1 expect=room_2 retry=2
  ```
  Lines under `if` go into its first `seq`; an `else` line at the `if`'s indent opens the second; the printer never
  prints the two `seq` steps themselves. `else` anywhere else → syntax error "else without if".
  The positional order is `when` then `then`, so `each=` is a keyword after them (the spec's example order is
  superseded by this).

- [ ] **Step 1: Write the failing tests** — `test_text_watch_if()`: both snippets parse to the IR of Task 1 and
  `print_text(parse_text(x)) == x`; `else` with no `if` before it at that indent → error mentioning "else without if";
  `if` with an `else` and nothing under it → "else needs a body".
- [ ] **Step 2: Run** run_tests.sh — expected: mission_text_test FAILs.
- [ ] **Step 3: Implement** in the parser (wrap the `if` body lines in seq children; ids follow the existing
  hierarchical scheme) and in `print_steps`.
- [ ] **Step 4: Run** — expected: mission_text_test 0 failures.

### Task 3: Interpreter — `if`, `seen` in `until`, `Host::matches`

**Files:**
- Modify: `common/mission/mission_interpreter.h` (Host ~60, `holds` ~400, `run_composite` ~469)
- Test: `common/mission/mission_interpreter_test.cpp` (FakeHost gains `std::map<std::string, std::vector<std::string>> present;` keyed by `what`, and overrides `matches`)

**Interfaces:**
- Produces: `virtual std::vector<std::string> Host::matches(const Pred& p) { return {}; }` — the node names that
  satisfy a set predicate NOW (for `seen`: object nodes of that class, or that node, present in the graph).
- `if`: evaluated once on entry with `host_.eval(cond)`; nullopt → step fails `cannot_evaluate:<pred name>`; true →
  run child 0; false → child 1 if present (child 0 marked Skipped), else the `if` is satisfied (outcome `"else-none"`).
- `seen` in an `until` (wait/explore/repeat): the step's snapshot = `matches()` when the step starts; holds when
  `matches()` contains a name not in the snapshot.

- [ ] **Step 1: Write the failing tests:** `if_true_runs_then`, `if_false_runs_else`, `if_false_no_else_satisfied`,
  `if_unknown_fails_cannot_evaluate` (eval returns nullopt → outcome `cannot_evaluate:door_open`, on_fail=skip
  skips it), `explore_until_seen_new_object` (present["bottle"] = {"bottle_3"} at start → still running; then
  {"bottle_3","bottle_7"} → explore satisfied "until"), `until_seen_never_holds_budget_ends` (matches empty →
  explore ends at budget with outcome "budget_exhausted").
- [ ] **Step 2: Run** run_tests.sh — expected: interpreter test FAILs.
- [ ] **Step 3: Implement** (the `if` case in `run_composite`; a per-Run `std::set<std::string> seen_at_start`
  filled when a leaf or composite with a `seen` until starts).
- [ ] **Step 4: Run** — expected: all interpreter tests pass (the 138 existing + the new).

### Task 4: Interpreter — `watch` (arming, firing, continue / stop, queue, re-arm)

**Files:**
- Modify: `common/mission/mission_interpreter.h`
- Test: `common/mission/mission_interpreter_test.cpp`

**Interfaces:**
- Consumes: `Host::matches` (Task 3).
- Produces: `StepView` gains `std::vector<std::string> fired{};` (for a watch: every node it fired on, in firing
  order; its size is the firing count), filled in `steps()`.
- Semantics (spec "Behaviour" + "Settled"): a watch arms when its enclosing sequence reaches it (snapshot =
  `matches(when)`), its own Run goes Running, and the sequence moves on at once. Each tick, before advancing the
  current sibling, every armed watch of the enclosing composites (innermost first) computes `new = matches(when) −
  snapshot − fired`; each new name is appended to that watch's queue and to `fired`. A queued firing runs the
  watch's body once with `vars_[each] = name` (reset the body's Runs per firing):
  - `continue`, body claims nothing (`claims_of(body) == kNone`): the body runs in the same tick; the sibling is untouched.
  - `continue`, body claims the base: the running leaf of the scope is cancelled (`host_.cancel`) and reset to
    Pending without spending an attempt; the body runs; then the scope resumes (that leaf restarts with its whole budget).
  - `stop`: the remaining siblings in scope are cancelled with outcome `watch:<pred name>`, the body runs, the
    enclosing composite is satisfied when the body ends (normal flow continues above it).
  - The watch's Run is satisfied when its scope ends; a watch inside a `repeat` body re-arms (fresh snapshot, empty
    `fired`) each iteration.

- [ ] **Step 1: Write the failing tests:**
  - `watch_ignores_known_objects` (present["chair"]={"chair_34"} before arming → a full `cross` completes, fired empty).
  - `watch_continue_report_every_new` (chair_51 then chair_52 appear while `cross` runs → fired ==
    {"chair_51","chair_52"}, cross never cancelled, mission completed).
  - `watch_class_fires_once_per_object` (chair_51 appears, vanishes, reappears → fired == {"chair_51"}).
  - `watch_continue_moving_body_pauses_and_resumes` (body `approach $c` → FakeHost sees cancel(cross),
    start(approach chair_51), then start(cross) again; attempt of cross stays 1).
  - `watch_queue_in_order` (two chairs appear in one tick while a moving body runs → bodies run chair_51 then
    chair_52, each once).
  - `watch_stop_cancels_scope_mission_completes` (bottle appears mid-`explore` → host cancel(explore) exactly once,
    explore state Cancelled outcome `watch:seen`, body report satisfied, mission status `completed`).
  - `watch_in_repeat_rearms` (repeat 2 × [watch seen(chair) continue; wait s=1s]: a chair present in iteration 1's
    snapshot does not fire in iteration 2 unless it newly appears after that re-arm).
  - `watch_scope_ends_with_siblings` (after the last sibling, the watch Run is Satisfied and later appearances do nothing).
- [ ] **Step 2: Run** run_tests.sh — expected: new tests FAIL.
- [ ] **Step 3: Implement** in the interpreter (per-composite list of armed watches with snapshot, fired set,
  queue, and a "body running" state; `steps()` reports `fired`).
- [ ] **Step 4: Run** — expected: all interpreter tests pass.

### Task 5: Progress snapshot carries firings

**Files:**
- Modify: `common/mission/mission_progress.h` (`encode_progress` ~95, `decode_progress` ~120)
- Test: `common/mission/mission_progress_test.cpp`

**Interfaces:**
- Consumes: `StepView::fired` (Task 4).
- JSON: each step object gains `"fired"` (array of strings), written only when non-empty; decode defaults to empty.

- [ ] **Step 1: Failing tests:** round trip of a snapshot with a watch fired on {"chair_51","chair_52"}; a
  snapshot WITHOUT the key (today's format, a literal string in the test) decodes with `fired` empty.
- [ ] **Step 2: Run** run_tests.sh — expected FAIL. **Step 3: Implement.** **Step 4: Run** — expected PASS.

### Task 6: Controller — `MissionHost::matches` and `seen` (room reviews)

**Files:**
- Modify: `controller/src/specificworker.h:179` (MissionHost declaration)
- Modify: `controller/src/specificworker_mission.cpp:510` (next to `eval`)

**Interfaces:**
- Consumes: `Host::matches` signature (Task 3).
- `matches(p)` for `p.name == "seen"`: the names of nodes of type `"object"` whose `subtype` attribute equals
  `p.refs.at("what").name` (a class), or whose name equals it (a node); every other predicate → `{}`.
  `eval(seen)` → nullopt (seen is only meaningful as a difference, which the interpreter computes).
  Read with the type-attributed accessors (CLAUDE.md: never `runtime_checked_*`).

- [ ] **Step 1: Build** the controller in the worktree: `cd controller && cmake -B build && make -C build -j8` — expected: builds (before the change: proves the tree builds with Tasks 1–5).
- [ ] **Step 2: Implement** `matches`.
- [ ] **Step 3: Build** again — expected: clean build, `bin/controller` updated.
- [ ] **Step 4: Ask the room session for review** (SendMessage to `room`): the diff of `controller/` and
  `common/mission/` in the worktree, the spec path, and the test names; wait for its verdict; apply its fixes.

### Task 7: language_agent — checker, skill, firing wakes

**Files:**
- Modify: `language_agent/src/mission_plan.h` (`check_planned`, `new_reports`/`report_wake` area ~556, `progress_lines`)
- Modify: `language_agent/src/chat_session.cpp:2817` (queue the watch report wakes)
- Modify: `language_agent/etc/skills/missions.md` (steps list + 3 examples)
- Test: `language_agent/tests/test_mission_plan.cpp`, `language_agent/tests/test_skills.cpp` (examples parse)

**Interfaces:**
- Consumes: Progress steps' `fired` (Task 5), text form (Task 2).
- Produces: `struct WatchReport { std::string topic, bound; };`
  `std::vector<WatchReport> new_watch_reports(const Progress& before, const Progress& now, const Mission& ir)` —
  for each watch step, one entry per name in `now.fired` beyond `before.fired.size()` (in order), with `$NAME` in
  each `report` text of its body replaced by that node.
- `check_planned` rejects `seen(X)` unless X is a node in the STATE or one of `chair, table, bottle, cabinet,
  refrigerator, hood, nightstand, bed, toilet, person`, with the reason "I cannot recognise a X".
- `progress_lines` prints one line per firing: `"watch seen(chair): chair_51"`.
- Skill: the two step lines (watch, if) + examples: "tell me the colour of any chair on the way" (continue),
  "explore and stop when you see a bottle" (stop, at the top level), "if door 1 is shut open it, otherwise go through".

- [ ] **Step 1: Failing tests:** `new_watch_reports` (fired {} → {chair_51, chair_52} gives 2 entries,
  topics "the colour of chair_51", "the colour of chair_52"); `check_planned` refuses `seen(cup)` with "I cannot recognise a cup" and accepts `seen(bed)`;
  `progress_lines` firing line; test_skills parses the new examples.
- [ ] **Step 2: Run** run_tests.sh — expected FAIL. **Step 3: Implement.** **Step 4: Run** — expected: all suites pass; rebuild `language_agent` (`make -C build -j8`).

### Task 8: Planner eval, then one live run

**Files:**
- Modify: `language_agent/tests/check_llm_missions.py` (cases d-cond-1, d-cond-2-as-mission, d-perc-1, "any chair on the way" from `docs/utterance_testset.yaml`)

- [ ] **Step 1: Run** `python3 tests/check_llm_missions.py gemma4:12b 3 --form text` — expected: the new cases
  produce a watch/if that `check_planned` accepts in ≥ 2 of 3 runs each; old cases no worse than before. Record the
  numbers in the ledger.
- [ ] **Step 2: Live run** (claim Webots with the peers first; fleet as chat_e2e20 but `controller` from the
  worktree): say "ve a la habitación 3 y dime de qué color es cada silla que veas por el camino" → card shows a
  `watch seen(chair) continue each=…` + `cross`; yes → each newly seen chair produces one chat line with its colour;
  then "explora y para en cuanto veas una botella" with a bottle in the room → the explore ends early and the mission
  completes. Save the run folder name and the chat log path.
- [ ] **Step 3: Report** to the owner: test counts, eval numbers, live outcome, room's review verdict; the owner
  decides on commit and merge (and room merges into rt-build).
