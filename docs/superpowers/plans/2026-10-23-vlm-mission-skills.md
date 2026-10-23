# VLM missions and skills — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** chat_agent's model turns multi-step requests into missions. The controller's interpreter runs them, with a
`report` step, `bind`/`$NAME` for rooms not yet entered, fixed progress lines, failure wake-ups, and
pause/resume/kill.

**Architecture:**
- **Language and interpreter** (`common/mission/`, header-only, pure): `report`, `bind`, pause/unpause.
- **Controller:** maps them to the robot.
- **chat_agent:**
  - a `plan_mission` tool runs a separate *planner* request, whose system prompt is the mission skills;
  - parse, validate and name-check, with up to 2 fix rounds;
  - a card the person confirms with "yes";
  - the mission JSON is written to `chat_intention`, as `go_around_object` already does;
  - progress, report and failure handling, driven from `mission_watch`.
- **Skills:** markdown files in `chat_agent/etc/skills/`, shared by the agent and the eval.

**Tech Stack:** C++23, Qt6 Core, DSR (cortex), Ollama `/api/chat`, Python 3 for the LLM eval.

**Spec:** `docs/superpowers/specs/2026-10-23-vlm-mission-skills-design.md`. It extends
`chat_agent/docs/mission_language_2026-10-13.md`. Read both.

## Global Constraints

- **CLAUDE.md applies:**
  - `and`/`or`/`not`;
  - C++23 containers and algorithms;
  - `std::from_chars` only, never `strtof`/`stod`/`atof`;
  - type-attributed DSR access only, never `runtime_checked_*`;
  - never `Qt::DirectConnection`.
- **Builds:** `cbuild -j8`, never more than `-j8`. Check `pgrep -a cc1plus` before a controller build.
- **Interpreter purity:** `common/mission/*.h` stays pure: no graph, no Qt in `mission_ir.h`/`mission_interpreter.h`.
- **Person in the loop:** no mission and no fix reaches `chat_intention` without the person's "yes" to its card.
- **No model in the loop for `stop`, pause, continue or kill.** These are fixed fast paths.
- **Nothing is claimed as done that the interpreter did not report.** Progress lines come from `controller_mission`,
  never from the model.
- **Fix rounds:** at most 2 automatic rounds, after which the model explains.
- **`bind`:** only on `cross`. `$NAME` is accepted only where a `Node` ref is accepted, and only after a step that
  binds it.
- **Pause:** freezes the clock and budgets. Resume restarts the interrupted step from its beginning, without
  consuming a `retry`.
- **`resume` keeps its meaning:** the controller word `resume` (end a `wait`) is unchanged. Un-pausing is the new word
  `unpause`.

## Review Focus

1. **A "yes" that arrives after the card went stale.** The person says something else, then "yes". Expected: the
   card is discarded as soon as the next non-yes message arrives, and a later "yes" does not send it (Task 8 test).
2. **Pause while no mission runs, or twice in a row.** Expected: a harmless "nothing to pause" reply, with no state
   change (Tasks 3 and 4 tests).
3. **`$next` used before or without its `bind`, or a `bind` on a non-`cross` op.** Expected: a validation error that
   names the variable, which goes back to the planner as a FIX (Task 1 test).
4. **A `report` that fires while the chat model is mid-answer.** Expected: the report is queued and asked after the
   current answer ends, never dropped (Task 9 test on the queue).
5. **A mission started from `remote_cmd.txt` or the panel**, not from chat. Expected:
   - progress lines are shown;
   - failure and report wake-ups happen only for missions this chat sent (`mission_source == "chat"`).

   Task 9 test.

---

### Task 1: Language: `report` op and `bind`/`$NAME`

**Files:**
- Modify: `common/mission/mission_ir.h`:
  - the op table (~238);
  - `Validator::step` (~368-382), plus the `ref()` check, which lives next to `field()`.
- Test: `common/mission/mission_ir_test.cpp`, `common/mission/mission_text_test.cpp`

**Interfaces:**
- Produces:
  - op `report` with one required positional `String` field, `text`; `.claims = kNone`, `.instant = true`;
  - `validate()` accepts `bind=NAME` on `cross` only, and accepts `Ref::Kind::Var` named `NAME` wherever a field's
    `ref_kinds` include `kNode` (op fields and predicate refs), only if an earlier step (pre-order) binds `NAME`;
  - new error code `"unbound"`, its text naming the variable.

- [ ] **Step 1: Write the failing tests.** In `mission_text_test.cpp`, using the DSL:
  - parse and validate succeed for:
    ```
    mission "t"
      cross door_1 bind=next
      explore $next budget_s=10s
      report "objects here"
    ```
  - `explore $next budget_s=10s` alone gives `code == "unbound"` and text containing `next`;
  - `explore room_1 budget_s=10s bind=x` gives `code == "unsupported"`, `name == "bind"`;
  - `report` with no text gives `code == "missing_field"`;
  - `print_text(parse_text(s)) == s` round-trips for the first mission.

  In `mission_ir_test.cpp`: the JSON round trip (`decode_json(encode_json(m)) == m`) of the same mission.
- [ ] **Step 2: Run them to see them fail.** Run `cd chat_agent/tests && bash run_tests.sh 2>&1 | grep -E "FAIL|mission_"`.
  Expected: FAIL lines from `mission_text_test` / `mission_ir_test`.
- [ ] **Step 3: Implement.** Add the `report` entry to `ops()`. In `Validator`:
  - keep a `std::set<std::string, std::less<>> vars_`;
  - in `step()`, replace the blanket `bind` refusal with: `bind` on an op other than `cross` → `unsupported("bind", s)`;
    otherwise insert into `vars_` after the step's own fields are checked;
  - in the ref check, a `Var` passes iff `kinds & kNode` and `vars_.contains(name)`; else `err("unbound", s, field,
    "$" + name + " is not bound by an earlier step")`;
  - update the `Var` and `bind` comments ("RESERVED … refused by v1") to say what v1 now allows.
- [ ] **Step 4: Run the tests to see them pass.** Run `bash run_tests.sh`. Expected: all `mission_*` checks pass.
- [ ] **Step 5: Commit.** Run `git commit -m "mission: report op; bind on cross + \$NAME room refs"`.

### Task 2: Interpreter: `report` no-op and variable binding

**Files:**
- Modify: `common/mission/mission_interpreter.h`:
  - `Host`;
  - `run_leaf`;
  - `resolve`;
  - predicate evaluation (`holds`).
- Test: `common/mission/mission_interpreter_test.cpp` (and update its `FakeHost`).

**Interfaces:**
- Consumes: Task 1's `report` and `bind`.
- Produces:
  - a new pure-virtual `virtual std::optional<std::string> Host::current_room() = 0;`, meaning the room the robot is
    physically in now, including a proto-room; `nullopt` = unknown;
  - a `report` step is satisfied at once, with no host call;
  - after a `cross` with `bind=NAME` succeeds (including its `expect` check), `$NAME` is bound to
    `host_.current_room()`; `nullopt` fails the step with outcome `"no_room"`;
  - `resolve()` replaces each `Var` with `Ref{.kind = Node, .name = bound}`, and an unbound one fails the step as
    `"unbound"`. Predicate refs are substituted the same way before `host_.eval`.

- [ ] **Step 1: Write the failing tests.** `FakeHost` gets `std::optional<std::string> room = "room_1";` and
  `current_room()` returns it.
  - **Test A:** the mission `cross door_1 bind=next` / `explore $next budget_s=5s` / `report "x"`.
    1. Tick until step 1 started.
    2. Set `host.room = "room_2"` and `leaf["1"] = Succeeded`.
    3. Tick, then assert `host.started.back().ref("room")->name == "room_2"` and `->kind == Ref::Kind::Node`.
    4. Advance the clock 5 s and tick, then assert the step-3 state is `Satisfied` and no `"start 3:report"` is in
       `host.log`.
    5. Assert `state() == Completed`.
  - **Test B:** same, with `host.room = std::nullopt` at success. Assert `status() == "failed|1|no_room"`.
- [ ] **Step 2: Run the test to see it fail.** Run `bash run_tests.sh`. Expected: compile error (`current_room`) or
  FAIL.
- [ ] **Step 3: Implement as specified in Interfaces.** Keep `std::map<std::string, std::string, std::less<>> vars_`,
  cleared in `start()`.
- [ ] **Step 4: Run the tests to see them pass.** Expected: all `mission_interpreter_test` checks pass. Then build the
  controller with `cd controller && cbuild -j8`. Expected: a compile error in `MissionHost` (abstract
  `current_room`). Leave it for Task 4: do not commit a broken controller. Instead, add a one-line
  `current_room()` to `MissionHost` now, returning `w_.world_model_.graph_state().room_name` (or `nullopt` if
  empty), so the tree builds.
- [ ] **Step 5: Commit.** Run `git commit -m "mission interpreter: report no-op; bind \$NAME to the room entered"`.

### Task 3: Interpreter: pause and unpause

**Files:**
- Modify: `common/mission/mission_interpreter.h`
- Test: `common/mission/mission_interpreter_test.cpp`

**Interfaces:**
- Produces:
  - `bool Interpreter::pause(std::uint64_t now_ms)`: true if a running mission was paused. It cancels the running
    leaf via `host_.cancel` (if started), sets it not-started, and freezes the mission.
  - `bool Interpreter::unpause(std::uint64_t now_ms)`: true if a paused mission resumed. The interrupted leaf's
    `began = now_ms` (full budget again), and `attempt` is unchanged.
  - `[[nodiscard]] bool paused() const`.
  - While paused, `tick()` does nothing. `status()` returns `"paused|<step id>"`, and `progress()` reports the
    paused step.
  - `cancel()` works while paused, giving `aborted|<outcome>`, and clears the pause. `start()` clears it too.
  - Events: `"step <label>: paused"` and `"step <label>: restarted"`.

- [ ] **Step 1: Write the failing tests.**
  - **Test A:** `explore room_1 budget_s=10s`.
    1. Tick at t = 0; the leaf starts.
    2. `pause(3000)` returns true; assert `"cancel 1:explore"` is in the log and `status() == "paused|1"`.
    3. Tick at t = 60000; assert the step state is still `Running` (not failed or satisfied).
    4. `unpause(60000)` returns true; the next tick starts `1:explore` again (a second `"start 1:explore"`).
    5. At t = 69000 it still runs; at t = 70000 the budget ends it and the state is `Completed`.
    6. The attempt counter is still 1 (`steps()[0].attempt == 1`).
  - **Test B:** `pause()` with no mission returns false, and a second `pause()` returns false.
  - **Test C:** pause, then `cancel("cancelled", t)` gives `status() == "aborted|cancelled"` and `paused() == false`.
- [ ] **Step 2: Run the tests to see them fail.** Run `bash run_tests.sh`. Expected: compile error or FAIL.
- [ ] **Step 3: Implement.** Use `running_leaf(root_)` to find the leaf. Store the paused leaf's id for `status()`.
- [ ] **Step 4: Run the tests to see them pass.** Run `bash run_tests.sh`. Expected: pass.
- [ ] **Step 5: Commit.** Run `git commit -m "mission interpreter: pause/unpause with step-resolution restart"`.

### Task 4: Controller: pause, unpause and the room for `bind`

**Files:**
- Modify:
  - `controller/src/specificworker.cpp`: `dispatch_mission_command` (~2010), next to `cancel`/`resume`;
    `gui.mission.on_pause` (~533); the chat-inbox branch (~2278).
  - `controller/src/specificworker_mission.cpp`: `MissionHost::start` explore branch (~95); the publish of
    `controller_mission`.

**Interfaces:**
- Consumes: Task 3's `pause`/`unpause`/`paused`.
- Produces:
  - the controller command words `pause` and `unpause`, giving `MissionResult{.accepted, .kind = "pause"|"unpause",
    .reason}`. "Nothing to pause" and "not paused" are not accepted, with a reason.
  - `pause` also sets `paused_ = true` and calls `stop_robot()`; `unpause` clears `paused_`.
  - The panel's Pause button calls the same two functions, so the panel and chat cannot disagree.
  - The chat inbox treats `pause`/`unpause` like `cancel`/`resume` (about the mission, not a new one).
  - `controller_mission.state` shows `paused|<id>` (it already publishes `status()`).

- [ ] **Step 1: Wire it up as specified.** Explore branch: accept `room->name` when it equals `here` or equals the
  robot's physical room from `current_room()`. If those are the same thing in practice, this is a no-op. Task 10
  checks it live.
- [ ] **Step 2: Build.** Run `cd controller && cbuild -j8`. Expected: builds.
- [ ] **Step 3: Smoke test against a running controller in Webots.** Use `remote_cmd.txt`:
  1. Write a mission with `explore <room> budget_s=20s`.
  2. Write `pause`. The robot holds, and the log shows `step 1:explore …: paused`.
  3. Wait 30 s, then write `unpause`. The log shows `restarted`, then `completed` about 20 s later.
  4. Write `pause` with no mission running. The log says nothing to pause.
- [ ] **Step 4: Commit.** Run `git commit -m "controller: pause/unpause words + panel → interpreter freeze"`.

### Task 5: chat_agent: mission helpers (pure)

**Files:**
- Create: `chat_agent/src/mission_plan.h`
- Test: `chat_agent/tests/test_mission_plan.cpp`, wired into `tests/run_tests.sh` like `test_approach` (Qt6Core
  flags).

**Interfaces:**
- Produces:
  - `std::expected<rc::mission::Mission, std::string> parse_planned(std::string_view text)`: if the text starts with
    `{`, use `decode_json`, else `parse_text`; then `validate`. The error is a one-line message for the planner.
  - `std::optional<std::string> unknown_name(const rc::mission::Mission&, const std::function<bool(std::string_view)>& exists)`:
    the first `Node` or `Passage` ref (op args and predicate refs, recursively) whose name does not exist. `Var`,
    `Anchor` and `Pose` refs are skipped.
  - `std::vector<std::string> progress_lines(const rc::mission::Progress& before, const rc::mission::Progress& after, const rc::mission::Mission& m)`:
    the fixed lines for every step-state change, oldest first.
    - Formats: `→ i/N <label>`, `✓ <label> (<s> s)`, `⟳ retry a/n <label>`, `✗ <label>: <outcome>`,
      `⏸ paused at i/N <label>`, `▶ resumed: i/N <label> (restarted)`, `■ mission cancelled`.
    - A `say` step shows its text verbatim. N counts top-level steps.
  - `std::vector<std::string> new_reports(const Progress& before, const Progress& after, const Mission& m)`: the
    `text` of each `report` step that became Running or Satisfied.

- [ ] **Step 1: Write the failing tests.**
  - `parse_planned` accepts the Task 1 mission in text and in JSON, and returns an error containing `unbound` for
    `explore $x budget_s=5s`.
  - `unknown_name` with `exists = {door_1}` returns `"room_9"` for `explore room_9 …`, and `nullopt` for
    `cross door_1 bind=n` / `explore $n budget_s=5s`.
  - `progress_lines`: two snapshots where step 1 goes Pending → Running gives `{"→ 1/3 1:cross door_1"}`.
    Running → Satisfied with `t1 - t0 = 42000` gives a line starting `"✓ 1:cross door_1 (42 s)"`. A status change to
    `paused|2` gives a line starting `"⏸ paused at 2/3"`.
  - `new_reports` returns `{"objects here"}` once, and an empty vector on the next identical pair.
- [ ] **Step 2: Run them to see them fail.** Run `bash run_tests.sh`. Expected: compile error.
- [ ] **Step 3: Implement as specified.**
- [ ] **Step 4: Run them to see them pass.** Run `bash run_tests.sh`. Expected: `test_mission_plan: N checks passed`.
- [ ] **Step 5: Commit.** Run `git commit -m "chat_agent: mission_plan.h — parse/validate, name check, progress lines, reports"`.

### Task 6: chat_agent: skills

**Files:**
- Create:
  - `chat_agent/src/skills.h`;
  - `chat_agent/etc/skills/missions.md`, `doors.md`, `exploring.md`.
- Modify:
  - `chat_agent/src/chat_session.cpp`: the `tools_json()` `use_skill` entry, its dispatch, and the system-prompt
    list;
  - `chat_agent/etc/config.toml`: `[Skills] dir = "etc/skills"`.
- Test: `chat_agent/tests/test_skills.cpp`

**Interfaces:**
- Produces:
  - `struct Skill { std::string name, description, body; };`
  - `std::vector<Skill> load_skills(const std::filesystem::path& dir)`:
    - one per `*.md`, with `name` taken from the file stem;
    - the front matter is `---\ndescription: <one line>\n---`;
    - a file without a description is skipped, with a warning.
  - `std::string skills_index(const std::vector<Skill>&)`: one line per skill, `- <name>: <description>`.
  - The tool `use_skill(name)` returns the body, or `REJECTED: no skill '<name>'`.
- **Content of `missions.md`, written from spec §5:**
  - the step list and keywords, with units;
  - `bind`/`$NAME`;
  - "never invent names: use STATE names";
  - output ONLY the mission text, or `ask: <question>`;
  - five examples, the spec §5.1 mission among them.
- **`doors.md` and `exploring.md`:** the tips from spec §5.4.

- [ ] **Step 1: Write the failing test.** Load a temp dir with two files, one without a description. Assert:
  - `size() == 1`;
  - the name, description and body are right;
  - `skills_index` is `"- a: does A\n"`.
- [ ] **Step 2: Run it to see it fail.** Run `bash run_tests.sh`. Expected: compile error.
- [ ] **Step 3: Implement the loader, the tool, the index in the system prompt, and the three skill files.**
- [ ] **Step 4: Check.** Run `bash run_tests.sh`. Expected: pass. Then `cbuild -j8`. Expected: builds.
- [ ] **Step 5: Commit.** Run `git commit -m "chat_agent: skills — loader, use_skill tool, missions/doors/exploring"`.

### Task 6b: LTSM: topology summary on the live graph, plus chat tools (person's request, 2026-10-23)

Long-term memory lives on ltsm's own DSR domain 2, which chat_agent cannot read (10-13 design §1.5, §3.2(i)). This
task publishes the 10-13 design's `ltsm_topology` summary and gives the chat model two read-only tools.

**Files:**
- Modify: cortex `core/include/dsr/core/types/type_checking/dsr_attr_name.h`: `REGISTER_TYPE(ltsm_topology,
  std::string, false)`. **The person reinstalls cortex.** Do not `sudo cp`. The task blocks here until they confirm.
- Create:
  - `ltsm_agent/src/topology_digest.h`: pure encode;
  - `ltsm_agent/tests/test_topology_digest.cpp`, wired into ltsm's existing test build, or as a standalone `g++` line
    like chat_agent's tests;
  - `chat_agent/src/topology.h`: pure decode and route;
  - `chat_agent/tests/test_topology.cpp`.
- Modify:
  - `ltsm_agent/src/specificworker.cpp`: build, publish and clean up the node;
  - `chat_agent/src/chat_session.cpp`: two tools, plus one STATE line;
  - `chat_agent/tests/check_llm_tools.py`: add the tools, plus 3 cases.

**Interfaces:**
- **Produces in ltsm** (`topology_digest.h`):
  ```cpp
  struct TopoRoom { std::string name; bool live = false; };
  struct TopoPassage { std::string id, room_a, face_a, room_b, face_b; int crossings = 0;
                       double pass_alpha = 1, pass_beta = 1, open_alpha = 1, open_beta = 1;
                       std::optional<std::string> live_door; };
  std::string encode_topology(const std::vector<TopoRoom>&, const std::vector<TopoPassage>&);
  ```
  - The JSON shape follows 10-13 §3.2(i): `{"rooms":[{"name","live"}],"passages":[{"id","a":{"room","face"},
    "b":{"room","face"},"crossings","p_pass":{"alpha","beta"},"p_open":{"alpha","beta"},"live_door"?}]}`.
  - Room names are the LIVE names (`room_1`, not the memory prefix `r<idx>_room_1`).
  - Numbers are written locale-proof (`std::format`).
- **ltsm publishes** one ltsm-owned node `ltsm_topology`:
  - type `metaconcept`, never `door` (door_concept's sweep reaps doors);
  - no RT edge, under root;
  - attribute `ltsm_topology_att`;
  - rebuilt only when memory changes (after each `MemoryStore::save` that wrote something, and at startup after
    `load`);
  - deleted in `cleanup_owned_nodes`; a previous run's node is replaced on first publish.
- **chat side** (`topology.h`):
  - `std::expected<Topology, std::string> decode_topology(std::string_view json)`;
  - `std::vector<std::string> route(const Topology&, std::string_view from_room, std::string_view to_room)`:
    passage ids in order, by Dijkstra with edge cost `(alpha + beta) / alpha` (that is, 1 / E[p_pass]); empty if
    unreachable.
- **Tools:**
  - `known_rooms()`: the rooms, and for each passage the two rooms, its live door if any, crossings and E[p_pass];
  - `route_to(room)`: the passages in order, with the live door of the first leg. It feeds `plan_mission`: the
    mission skill learns `cross passage:<id>@<room>` for later legs.
  - Both are read-only.
- **STATE gets one line:** `remembered rooms: room_2 (behind door_1, crossed 3x), room_3 (…)`.

- [ ] **Step 1: Write the failing tests.**
  - `encode_topology` on two rooms and one passage gives JSON that `decode_topology` reads back field-for-field.
  - `route` over a chain room_1–room_2–room_3 gives `{p0, p1}`, and over a disconnected room gives `{}`.
  - With two parallel passages whose `alpha` / `beta` are 9/1 and 1/9, it picks the 9/1 one.
- [ ] **Step 2: Run them to see them fail.** Run each test build. Expected: compile error.
- [ ] **Step 3: Cortex attribute.** Add the `REGISTER_TYPE` line, then STOP and ask the person to reinstall cortex.
  Continue when they confirm.
- [ ] **Step 4: Implement encode, publish, decode, route, the two tools and the STATE line.**
- [ ] **Step 5: Check.**
  1. Run the unit tests. Expected: pass.
  2. Build ltsm and chat_agent with `cbuild -j8`.
  3. Live: start the fleet, let the robot cross door_1 and back once, then confirm the `ltsm_topology` node shows
     both rooms and the passage.
  4. Run `check_llm_tools.py qwen3-vl:8b-instruct 3`. Expected: no regression below 268/279, and the 3 new cases
     pass.
- [ ] **Step 6: Commit.** Run `git commit -m "ltsm_topology summary node + chat known_rooms/route_to tools"`.

### Task 7: Mission eval (decides text versus JSON)

**Files:**
- Create:
  - `chat_agent/tests/mission_check.cpp`: reads stdin, prints `OK` and then one `op arg…` line per step in pre-order,
    or `ERROR <message>`. It uses `parse_planned` (Task 5).
  - `chat_agent/tests/check_llm_missions.py`

**Interfaces:**
- Consumes:
  - `etc/skills/missions.md`, `doors.md` and `exploring.md`, concatenated, as the planner system prompt;
  - `check_llm_tools.py`'s `STATE` / `state()` (import the file as that script's `exec` pattern does);
  - the `OLLAMA` env var, as in `check_llm_tools.py`.
- **Produces:** `python3 tests/check_llm_missions.py <model> [repeats=3] [--form text|json]`. It prints
  PASS/FAIL per case and `N/M passed`, plus median latency.
- **Request:** `stream: false`, no tools, `temperature 0`, `think: false`, `num_predict` from config. `--form json`
  adds a `format` JSON schema mirroring `mission_json.h`, with depth ≤ 2.
- **Grading (spec §7):** a case passes if `mission_check` says `OK` and the case's checklist holds over the op lines.
  Checklist examples for the test case:
  - a `cross door_1` with `bind`;
  - an `explore $<that name>`;
  - a `report` after it;
  - a final cross back, or `go_to anchor:…`.

- [ ] **Step 1: Write about 20 cases.** Include:
  - the spec §5.1 test case;
  - paraphrases of it in English and Spanish;
  - "explore room_2 and come back";
  - "cross door_1 twice";
  - "go to room_2, wait 30 s, come back";
  - a request naming a non-existent door, which must yield `ask:` or a mission that `mission_check` / the name check
    rejects (graded as PASS only for `ask:`);
  - single-step requests, which should still yield a valid one-step mission.
- [ ] **Step 2: Run on the 8b and on qwen3.8, both forms.** On the Orin: `OLLAMA=http://192.168.1.34:11434 python3
  tests/check_llm_missions.py qwen3.8 3 --form text`, and likewise for `--form json` and for
  `qwen3-vl:8b-instruct`. Record the four totals and latencies in the commit message.
- [ ] **Step 3: Decide.** If text ≥ JSON − 2 cases, keep text (spec §6 default). Otherwise the skill teaches JSON and
  Task 8 sends `format`. Write the decision and numbers at the top of `missions.md` as a comment line. If no
  model/form reaches 90%, stop and report to the person before Task 8.
- [ ] **Step 4: Commit.** Run `git commit -m "chat_agent: mission eval — <form> chosen (<numbers>)"`.

### Task 8: chat_agent: `plan_mission`, the card and "yes"

**Files:**
- Modify:
  - `chat_agent/src/chat_session.cpp`, `.h`: tool entry, dispatch, planner request, pending card, fast path 1c;
  - `chat_agent/etc/config.toml`: `[LLM] planner_model`, defaulting to `model`;
  - the config struct holding `cfg_.llm`;
  - `chat_agent/etc/system_prompt.md`: one rule, "more than one step, a condition or a return → plan_mission";
  - `chat_agent/tests/check_llm_tools.py`: add `plan_mission` and `use_skill` to `tools()`, and cases for routing
    multi-step → `plan_mission` and single-step → direct tool.
- Test: `chat_agent/tests/test_mission_plan.cpp`, extended for the pending-card rule as a pure function.

**Interfaces:**
- Consumes: Tasks 5, 6 and 7 (the form).
- Produces:
  - tool `plan_mission(request: string)`;
  - **planner request:** run after the current turn's `on_done`, via the same `OllamaClient`, never concurrently.
    System = the concatenated mission skills; messages = the request plus the STATE; no tools; `temperature 0`; plus
    `format` if Task 7 chose JSON.
  - **Outcome handling:**
    - `ask: <q>` → tool result `ASK: <q>`;
    - a parse, validate or `unknown_name` error → re-ask the planner with the error appended, at most 2 times, then
      `REJECTED: <error>`;
    - OK → store `pending_mission_` (IR, text, serial), show the card (text form plus the model's one-line summary),
      tool result `OK: card shown; wait for the person's yes`.
  - **Fast path 1c**, after 1a: if `pending_mission_` is set and `fleet::is_yes(text)`, send
    `encode_json(*pending_mission_)` with `send_command`, clear the pending card, and reply `"Starting: <title>"`.
    Any other user message clears `pending_mission_` before normal handling.
  - A pure `bool card_still_valid(std::uint64_t card_turn, std::uint64_t user_turn)` (true iff
    `user_turn == card_turn + 1`) for the test.

- [ ] **Step 1: Write the failing test.** `card_still_valid(5, 6)` is true; `(5, 7)` is false.
- [ ] **Step 2: Implement the tool, the planner request, the fix loop, the card and fast path 1c.**
- [ ] **Step 3: Check.** Run `bash run_tests.sh` and `cbuild -j8`. Then `python3 tests/check_llm_tools.py
  qwen3-vl:8b-instruct 3`. Expected: no regression below 268/279, and the new routing cases pass.
- [ ] **Step 4: Commit.** Run `git commit -m "chat_agent: plan_mission — planner request, fix loop, card, yes fast path"`.

### Task 9: chat_agent: progress, report, failure, pause/continue/cancel

**Files:**
- Modify:
  - `chat_agent/src/chat_session.cpp`: `tick_status()` (~1476), `on_user_text` fast paths, a wake queue;
  - `chat_agent/src/command_gate.h`: phrase matchers.
- Test: `chat_agent/tests/test_gate.cpp` (phrases), `chat_agent/tests/test_mission_plan.cpp` (wake queue as a pure
  struct).

**Interfaces:**
- Consumes: Task 5's `progress_lines`/`new_reports`, Task 4's command words.
- Produces:
  - **In `tick_status()`:** keep the previous `Progress`, and append each `progress_lines(prev, now, mission)` entry
    with `window_->append_robot(...)`.
  - **Wake-ups, only when `mission_source == "chat"`:**
    - each `new_reports` topic, plus a snapshot of `find_objects` for the robot's current room taken now, is queued
      as a wake;
    - a status change to `failed|…` queues a failure wake (failed step, outcome, undone steps);
    - `completed` queues a summary wake.
  - **Wake queue:** `struct WakeQueue { void push(std::string note); std::optional<std::string> pop_if_idle(bool busy); };`.
    A note goes to the model as a user-role message prefixed `(robot event, not the person) `, and only when the model
    is idle; it is never dropped.
  - **Phrase matchers** in `command_gate.h`, with the same anchored style as `is_stop`:
    - `is_pause`: `pause` / `pausa` / `espera`;
    - `is_continue`: `continue` / `resume` / `sigue` / `continúa`;
    - `is_cancel_mission`: `cancel the mission` / `abort` / `cancela la misión`.
  - **Fast paths:**
    - `is_pause` → `send_command("pause")`;
    - `is_continue` → `send_command("unpause")` if the progress status starts with `paused|`, else
      `send_command("resume")`;
    - `is_cancel_mission` → `send_command("cancel")`.

- [ ] **Step 1: Write the failing tests.**
  - The matchers: true for each phrase above, and false for `"pause the chair agent"` (program words, as in
    `is_stop`).
  - `WakeQueue`: `push("a")`, then `pop_if_idle(true)` is `nullopt` and `pop_if_idle(false)` is `"a"`.
- [ ] **Step 2: Implement as specified.**
- [ ] **Step 3: Check.** Run `bash run_tests.sh` and `cbuild -j8`.
- [ ] **Step 4: Commit.** Run `git commit -m "chat_agent: mission progress lines, report/failure/summary wakes, pause/continue/cancel"`.

### Task 10: Live validation in Webots (person present)

**Files:**
- Create: `docs/superpowers/plans/2026-10-23-vlm-mission-skills-live.md`, a results log.

- [ ] **Step 1: Start the fleet.** Use the three-room world and the usual fleet (`run_crossing.sh … NORUN=1`
  pattern), plus chat_agent, using the model chosen in Task 7.
- [ ] **Step 2: The person types the spec §5.1 test.** Check, from `chat_agent`'s ndjson log and `controller.log`:
  - the card is correct;
  - progress lines equal the controller events;
  - the report lists room_2's objects while the robot is in room_2;
  - the robot ends in room_1.

  If `explore $next` is refused as `wrong_room` (proto-room not current), fix the explore check from Task 4 and
  rerun.
- [ ] **Step 3: Failure run.** Hold door_1 shut (`python3 scripts/door.py close`). Expect:
  - `✗ … door …`;
  - a fix card proposed by the model;
  - after "no", a final report.
- [ ] **Step 4: Pause run.** "pause" during explore; wait twice the budget; "continue"; the mission completes. Then
  "cancel the mission" while paused gives `■ mission cancelled`.
- [ ] **Step 5: Stop run.** "stop" mid-mission halts the robot.
- [ ] **Step 6: Record and commit.** Record each result (pass/fail with log excerpts) in the results file, and
  commit.
