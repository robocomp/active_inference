# language_agent — execution contracts: aspect → execution pattern (2026-11-14)

Step 5 of the command-taxonomy task (`command_taxonomy.md`). For every action tool / mission op: its aspect, the
ROS-style pattern it should follow, what it provides today, and the mismatches. Evidence is cited from code
(`main` db8d39e3) and from live runs (run folders under `room_concept/estimation_scripts/crossing/`). Nothing is
changed by this document.

## The four patterns

| Aspect | Pattern | Must provide |
|---|---|---|
| Achievement / semelfactive | service-like call | success/failure result, timeout |
| Accomplishment | action-like (goal / feedback / result / cancel) | goal predicate on the belief graph, progress feedback, partial-failure report, cancellation |
| Activity | continuous behaviour | explicit stop condition or default bound, "stop" handling, status while running |
| State (maintain) | monitored invariant | condition to watch, violation report, release command |

## How things run today (two paths)

1. **Missions** (`plan_mission` → card → yes → controller interpreter, `common/mission/mission_interpreter.h`): an
   action server in all but name. Goal = the IR; feedback = the `controller_mission` node's progress snapshot (step
   states, per-leaf WAIT notes, read by `mission_watch`); result = `completed | failed|<step>|<outcome> |
   aborted|<outcome>`; cancel = `stop`, `cancel mission`, `pause`/`continue`, and supersession by a new mission. The
   chat is woken on completion, failure and skipped steps (`chat_session.cpp:2787–2831`), so the model closes the loop.
   Budgets are the termination guarantee — but only `explore` (required), `wait`, `repeat` carry one.
2. **Single actions from the chat** (`cross_door`, `approach_object`, `go_around_object`, `move_relative`, the grammar
   fast path): the tool writes a command word to the intention node and returns at once ("sent to the controller").
   Feedback = controller `intention_state` lines shown to the person as status lines. **No wake to the model** on
   their completion or failure; no budget on the chat side.

## Per item

| Item | Aspect (VP) | Pattern it should follow | Provides today | Mismatch |
|---|---|---|---|---|
| `stop`, `cancel_passage`, pause/continue/cancel | achievement | service | immediate effect; code fast path, never through the LLM | — |
| `say`, `report`, `anchor` (IR) | achievement | service | instant leaves, result in the progress snapshot | — |
| `look` | achievement | service | image returned in the next message | — |
| `show` / `draw` / `table` (live=false) | achievement | service | canvas result; `draw` returns the script error for a retry | — |
| `show` live=true | activity | continuous | redraws until `hide_canvas` or another show | no status line saying a live view is running (minor) |
| `sim_door` | achievement | service | "accepted" from the simulator | result is the REQUEST accepted, not the door's state; no check that P(open) followed (M3) |
| `agent_control` | achievement | service | result + confirmation for controller/retina | — |
| `human_claim` | achievement (informative) | service + confirmation | claim card; stored on yes | — |
| `move_relative` / IR `move` | accomplishment (distance) / semelfactive (turn) | action | controller refuses if blocked; status lines | single-action path: no result to the model (M2) |
| `cross_door` / IR `cross` | accomplishment | action | goal = in the expected room (exit fact), progress = door-route phases, failure outcomes (`door_wont_open`, `approach_not_offered`, …), cancel = stop/cancel_passage, retry=N | **no budget and no stall detection** (M1); single-action path: no result to the model (M2) |
| `approach_object` / IR `approach` | accomplishment | action | goal = the object's approach affordance holds ("goal holds"), on_fail=skip | no budget (M1); no result to the model on the single path (M2) |
| `go_around_object` | accomplishment | action | loop of approaches, skips unreachable sides | no budget (M1); partial result (which sides skipped) reaches only the status lines (M2) |
| IR `go_to` | accomplishment | action | goal pose, `no_route` failure, on_fail=skip | no budget (M1) |
| IR `open` / `close` | accomplishment | action | door_concept affordance claimed; "close not offered within 5.0 s … door left open" | the 5 s close wait is a fixed timeout, not a belief; acceptable but should be the step's budget |
| IR `explore` | activity | continuous | `budget_s` REQUIRED by the validator; optional `until`; budget end without until = done, with until unmet = `budget_exhausted` | — (the model case) |
| IR `repeat` | iterative / bounded activity | continuous | times, or until + budget_s (the validator requires the budget with until: `mission_ir.h:216`) | — |
| IR `wait s=` / `until=` / `at=` | state (hold still) bounded | monitored invariant | note "until 18:00 (in 42 min)", cancel | **claims the base**; used as a TRIGGER it freezes the robot; not persisted (M3); `until=` alone has no bound (M4) |
| `plan_mission` (the card) | composite | action (goal = the mission) | card, then mission path | — |
| Constraints ("don't…") | state (invariant) | monitored invariant | **nothing** | missing pattern entirely (taxonomy G2 / M2 store) |
| Standing orders, reminders | commissive (state + firings) | monitored invariant + scheduled services | **nothing** | missing (taxonomy G3 / M1 scheduler) |

## Mismatches and fixes

**M1 — Accomplishments with no bound and no stall report.** `cross`, `approach`, `go_to`, `go_around` can stay
"Running/WAITING" for ever. Evidence: tape_e2e19 (11-14), `cross door_24` sat 10+ min with `[why] still STOPPED
WAITING … servo not started … heading +174 deg ✗` and never failed; chat_e2e20 the same at door_25 (50+ s, rescued by
the person). The interpreter shows the wait note but the action has no termination guarantee.
*Fix:* every accomplishment gets a budget like `explore` does, defaulted from the episodic memory's own outcome stats
for that step kind (`outcome_stats`: cross mean 20 s, sd 2 s over 12 runs) rather than a constant — the step fails
with `stalled|<the wait note>` when the elapsed time makes success improbable under that duration model, and the chat
is woken with the note. The person can override on the card. (A fixed multiple of the mean would be a threshold; the
model-level version is P(still succeeding | elapsed) from the duration distribution, which is the same object the
deadline proposal needs.)

**M2 — Single actions are fire-and-forget for the model.** `cross_door`, `approach_object`, `go_around_object`,
`move_relative` return "sent" and their result reaches only the person's status lines. The model cannot say "I
arrived", report a partial failure (skipped sides), chain the next step, or notice a stall; it answered "I am
currently crossing door_25" from stale state (chat_e2e20).
*Fix:* run every single action as a ONE-STEP mission without a card — the same goal/feedback/result/cancel path and
the same wake on completion/failure. One generic executor (the interpreter) then owns progress, budgets,
cancellation and the report for every action; the fast-path grammar and the chat tools only build the one-step IR.

**M3 — `wait` mixes "hold still" with "trigger later", and nothing persists.** `wait at=17:00` claims the base, so a
reminder or a deferred mission freezes the robot until then; the waiting dies with a controller restart; `sim_door`
reports the request, not the outcome.
*Fix:* keep `wait` as the in-mission HOLD; move deferral, recurrence and event triggers into the scheduler (taxonomy
M1), held by language_agent, persisted, never claiming the base before it fires. For `sim_door`, return after the
door's P(open) moves (or say it did not), which is an achievement with a success check.

**M4 — An unbounded hold through `wait until=` alone.** `wait until=pred` validates with no `s` and no budget; if the
predicate never holds it waits for ever. `repeat until=` already requires `budget_s` (`mission_ir.h:216`) and
`explore` requires it too, so `wait until` is the one exception.
*Fix:* the same rule as `explore`: an atelic or open-ended step must carry a bound; the planner supplies the skill
default and the card shows it (the clarification policy for an atelic command with no bound).

**M5 — No invariant pattern at all** (constraints, "stay in this room", watches). *Fix:* the constraint store
(taxonomy M2) for route/plan/cross checks, with a violation report (the refusal names the constraint) and `release`;
watches go through the scheduler's `on:` trigger.

## Top 3 mismatches
1. **M1** accomplishments (cross / approach / go_to / go_around) have no budget and no stall failure — seen live,
   10+ min stall in tape_e2e19.
2. **M2** single actions from the chat are fire-and-forget for the model — results only on status lines; no wake, no
   partial-failure report, no chaining. Fix by running them as one-step missions through the one executor.
3. **M3** `wait` is used as a trigger but claims the base and is not persisted — reminders and deferred missions
   freeze the robot and die with a restart. Fix by splitting hold (mission `wait`) from trigger (scheduler).
