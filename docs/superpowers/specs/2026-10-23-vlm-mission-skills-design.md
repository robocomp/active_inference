# VLM missions and skills — design (2026-10-23)

Extends `chat_agent/docs/mission_language_2026-10-13.md` (the "10-13 design"). Where this document is silent, the
10-13 design stands. Where they differ, this one wins.

## 1. Intent

People must be able to give the robot **multi-step requests in plain language** ("cross door_1, tell me what is in
the next room, then come back") and see them carried out reliably. The chat model must be a **small model on the
robot's Jetson AGX Orin** (32 GB). The current choice is `qwen3-vl:8b-instruct`, at 268/279 on
`tests/check_llm_tools.py` and about 1.4 s per decision on the Orin.

The model plans and explains. The controller's mission interpreter (`common/mission/`) executes. Small models are
not reliable enough to drive a robot step by step: at about 96% per decision, a 10-step chain succeeds about 66% of
the time. They also invent completions ("agent stopped successfully" with no call, measured 10-22). A written,
checked mission runs the same way every time and can be shown to the person before it runs.

## 2. Goals

1. **Free composition.** The model writes any mission the language can express. It is not limited to pre-written
   routines.
2. **Person in the loop.** Every mission, and every fix after a failure, is shown to the person and runs only on
   their yes. A later relaxation, per routine and from its track record (`outcome_stats`), is anticipated but not
   built now.
3. **Report from inside a mission.** A step can wake the model to tell the person something about where the robot is
   now, for example the objects of room_2 while it is in room_2.
4. **Visible progress.** Every step transition appears in the chat as a fixed line that the model did not write.
5. **Failure handling with a proposal.** When a mission fails, the model is woken, proposes a fix as a new mission
   and, when the run ends, reports to the person.
6. **Skills: know-how on demand.** Long instructions (how to write missions, topic tips) are loaded only when the
   model needs them. Each skill stays a small text that can be tuned per model.

## 3. Non-goals

- The model does not steer, poll or sequence primitives. The interpreter does.
- No change to quick single actions. `move_relative`, `approach_object`, `cross_door`, `stop` and the rest stay as
  direct tools.
- No confidence-based auto-approval yet (goal 2's relaxation).
- No new controller behaviours beyond what §5.3 lists.

## 4. Anti-goals: failure even if it "works"

- The chat says something happened that the interpreter did not report.
- A mission runs, or a fix runs, without the person's yes.
- `stop` (fast path or tool) fails to win over a running mission.
- The model invents a door, room or object name, and it reaches the person's confirmation.

## 5. Design

### 5.1 The flow, on the test case

The test case is: *"What doors do you see?"* then *"Approach door_1, cross to the next room, tell me what objects are
there, and come back to room_1."*

1. **"What doors do you see?"** The model answers from the STATE, as today. No mission.
2. **The multi-step request.** The model calls the new tool `plan_mission`. With the mission-writing skill loaded
   (§5.4), it produces a mission, in the form decided in §6. For the test case:
   ```
   mission "visit the room behind door_1 and come back"
   anchor start
     cross door_1 retry=2 bind=next
     explore $next budget_s=60s
     report "the objects in this room"
     cross door_1 expect=room_1 retry=2
     go_to anchor:start
   ```
   "Approach door_1" folds into `cross`, which drives to the door first. On a first visit the room behind door_1 does
   not exist when the mission is written. It is born, as a proto-room, only when the robot crosses. So the mission
   names it through `bind` (§5.3). The way back is written by the door's own name, as `round_trips_mission.sh` does
   live.
3. **Checking.** chat_agent parses and validates the mission with the existing checker (`parse_text` / `decode_json`
   + `validate`) and checks that every name exists in the current graph. On an error, the error text goes back to
   the model to fix, at most twice, as failed `draw` scripts are today. After that, the model explains the problem
   to the person.
4. **Confirmation.** The person sees the mission's text form and a short plain-language summary, and answers yes or
   no.
5. **Run.** On yes, chat_agent writes the mission JSON into `chat_intention`. The controller already accepts a `{`
   payload (`controller/src/specificworker.cpp` ~2257).
6. **Progress.** chat_agent already reads `controller_mission` (`mission_watch.h`). Each step transition becomes one
   fixed chat line, for example:
   - `→ 1/5 crossing door_1`
   - `✓ crossed into room_2 (42 s)`
   - `⟳ retry 1/2`
   - `✗ cross failed: door shut`
7. **Report.** When the `report` step runs, chat_agent wakes the model with the report's topic and a snapshot of the
   current room's objects, taken at that instant (§5.3). The model's answer appears in the chat.
8. **End.** On `completed`, the model writes a short closing summary. On `failed|<step>|<outcome>`, see §5.5.

### 5.2 New chat tool: `plan_mission`

- Offered on every turn, like the other tools.
- Its one-line description says to use it for anything with more than one step, a condition, or a return.
- Calling it shows nothing to the person. It returns one of three results to the model:
  - `OK: card shown`;
  - `FIX: <checker error>`;
  - `ASK: <question>`, when the request is unclear.
- `plan_mission` is the only path by which a mission reaches the controller from chat. The person's "yes" to a card
  is handled by the session (fast path), not by the model, like `agent_control`'s confirmation today.

### 5.3 Mission language additions

- **`report "<topic>"`.** An instant step: it completes immediately, and the mission goes on. When chat_agent sees it
  reach `Running` or `Satisfied`, it snapshots the objects of the robot's current room at that moment, then wakes the
  model with the topic and that snapshot. The robot keeps moving while the model writes. The snapshot guarantees
  that the report is about room_2, even if the robot has left it by the time the answer arrives.
  - It is added to the op registry (`mission_ir.h`) like `say`.
  - The controller treats it as a no-op.
- **`bind=NAME` on `cross` is enabled (person's decision, 2026-10-23).** v1 reserves `bind` and `Ref::Kind::Var` and
  refuses them (`mission_ir.h` ~69, 116, 374). This project enables them for one case only:
  - When a `cross … bind=NAME` step succeeds, the interpreter binds `$NAME` to the room the robot is now in,
    including a proto-room.
  - Later steps may use `$NAME` wherever a room reference is accepted.
  - The checker accepts a `$NAME` only if an earlier step in the same mission binds it.
  - The chat-side name check skips `$NAME` references.
  - Binding from other ops stays refused.
  - This is controller and language work, and an exception to the non-goal in §3.
- **Room during a first visit (builder's call, marked).** While the new room is still proto, the controller may
  still count the robot as being in room_1 (`ltsm_agent/src/passage_live.h`). `explore` and passage references
  check the current room (`specificworker_mission.cpp`). The live test (§7) settles whether `explore $next` is
  accepted in time. If it is not, the builder makes `explore` accept the bound proto-room, within this project.
  The `report` snapshot uses the room the robot is physically in (the bound room when one exists), not `current`.
- **Nothing else is new in the language.** `say` stays. A `say` text is shown verbatim as a progress line; the
  model is not woken.

### 5.4 Skills

- A skill is a text file under `chat_agent/etc/skills/<name>.md`, with a one-line description in its front matter.
- The system prompt lists each skill's name and description, a few tokens each.
- A tool `use_skill(name)` returns the file's body as the tool result. It lands at the end of the conversation, so
  the cached prefix survives.
- The first skills:
  - **`missions`**: the language's steps, references, conditions and the units rule, plus about five worked
    examples. The test case is one of them. This replaces the 10-13 design's "planner prompt" (§4.3).
  - **`doors`**: shut doors, passages and how to name the way back, `retry`.
  - **`exploring`**: `explore` needs a budget; `explored()` measures the walls, not the objects.
- Inside `plan_mission`, the `missions` skill is loaded automatically, so the model cannot skip it.
- Topic skills are optional reading, loaded by the model when it judges that they apply.
- Existing long tool how-tos (`draw`, `show`) may move into skills later. That is out of scope for the first slice.

### 5.5 Failure

1. On `failed|<step>|<outcome>`, chat_agent wakes the model with:
   - the failed step and its outcome;
   - the steps still undone;
   - the current STATE.
2. The model either proposes a fix through `plan_mission`, which goes through the normal checking and card path (for
   example: "door_1 stayed shut; I can try door_0 instead, then come back. Confirm?"), or explains why there is
   none.
3. If the person declines, the model gives the final report.
4. No automatic retry beyond the step's own `retry=N`.
5. **A fix is a new, self-contained mission (my call).** Anchors do not carry over, because `validate` accepts only
   anchors defined in the same mission. A fix aims to bring the robot back to the original room (`in_room(room_1)`,
   or a `cross` back). It returns to the exact start pose only if the controller already publishes that anchor's
   pose where chat_agent can read it. No new controller output is added for this.

### 5.6 Stop

The existing `is_stop` fast path and the `stop` tool must abort a running mission at once. The controller's existing
Stop handling is assumed to cover chat-started missions. The builder verifies this and adds a test.

## 6. Open decision: how the model writes the mission

The 10-13 design chose a second request, with no tools and a JSON-schema `format`. That gives grammar-constrained,
always-parseable output, at the cost of a second request and an extra prompt. The alternative is the model writing
the people-facing text form directly as the `plan_mission` argument, with the strict parser's error fed back for a
fix.

- **Decision: measure both, then choose.** Build the eval first (§7) and run both forms on the 8b and on whatever
  else is under test.
- **Default if the eval is a tie:** the text form. It needs one request and is what the card shows.

## 7. How we know it works

1. **Offline eval, `tests/check_llm_missions.py`.** Same pattern as `check_llm_tools.py`: request plus STATE
   produces a mission, graded on the op sequence and references, over 3 repeats. It covers about 20 requests,
   including the test case and paraphrases. Target: at least 90% before live use.
   - **Grading (my call):** several missions can be correct. A mission passes if it validates and meets the
     request's checklist. For the test case, the checklist is:
     - crosses door_1;
     - explores the room it entered;
     - has a `report` while there;
     - ends back in room_1.

     Any valid form passes, for example `cross door_1` or the passage form for the way back. The exact op sequence is
     not graded.
   - It can also run against the Orin, via `OLLAMA=http://192.168.1.34:11434`.
   - Models to run: `qwen3-vl:8b-instruct`, and `qwen3.8`. qwen3.8 is a 27B dense model that tied the 8b at 270/279
     on the tool eval (2026-10-23, Orin), at 2.3 s against 1.5 s median latency. Run any later candidate as well.
     Missions may be written by a different model than the one that holds the conversation, if the eval says so.
2. **Live test case in Webots** (three-room world): the §5.1 sequence end to end. Pass criteria:
   - the card is correct;
   - the progress lines match the controller log;
   - the report names room_2's objects while the robot is in room_2;
   - the robot ends in room_1;
   - one run with door_1 held shut exercises §5.5.
3. **Stop test:** "stop" mid-mission halts the robot within the controller's normal stop latency.

## 8. Left to the builder

- Wire format details of `plan_mission`'s arguments, the card's layout, and the exact wording of the progress lines.
- How the session tracks "a card is waiting for yes".
- Few-shot wording in the skills.
- Whether `use_skill` results are trimmed from the history after the turn, as images are today.
