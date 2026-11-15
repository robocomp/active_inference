# Tense in commands — design (2026-11-14)

Owner's question: "What about time? How can we use tense in the commands?" Owner's ruling: write this spec, then build
**1** (a mission that starts at a clock time) and **3** (the robot's own past as something a command can refer to).
2, 4 and 5 are specified here only so the format leaves room for them.

## What exists

| Tense / mood | Example | Served by |
|---|---|---|
| present | "go to room 3" | the live graph → a mission card |
| past | "where is my chair?", "go to the chair you saw" | LTOM, episodic memory (`episodes`, `last_episode`), `remembered_goto` |
| duration | "wait 30 s", "repeat until …" | `wait s=`, `repeat times= / until= budget_s=` |
| conditional | "would the chair fit there?" | `imagine` |

"In 10 minutes" already works: `wait s=600s` as the first step. What is missing is a CLOCK time ("at 18:00"), the
robot's own whereabouts ("where you were 10 minutes ago", "if you have already been in room 3"), deadlines, "while"
and periodic missions.

## The model view

Each tense refers to a different belief:

- **past** = memory. Its uncertainty grows with the time since it was observed (the same ageing that lowers a restored
  door's P(exists)). A past pose is a pose in a remembered room's frame; ltsm recalls the room in the same frame, so
  the pose stays meaningful after leaving and re-entering.
- **present** = the live map.
- **future** = a preference over states AT a time. A start time is a precondition on the clock; a deadline is a
  preference over arrival time (item 2), never a hard cutoff.

## 1 — start at a clock time (BUILD)

**IR.** `wait` gains an optional field `at`: an absolute local date-time with its UTC offset,
`"2026-11-14T18:00:00+01:00"`. `wait` needs at least one of `s`, `until`, `at`. With `at` the step ends once the wall
clock reaches it (`satisfied`), or earlier if `until` holds. A time already past ends the step at once. The IR stays
grounded: `at` is absolute, so the controller never has to guess a date, a time zone or "next occurrence".
validate() rejects an `at` that is not a full date-time with offset (`bad_value`).

**The wall clock.** The interpreter's `now` is the caller's steady clock (durations, budgets). `Host` gains
`wall_ms()`, the epoch milliseconds now. Its default reads `std::chrono::system_clock`, so the controller needs no
change; tests override it.

**Text form (what the planner writes).** `wait at=18:00` (or `18:00:30`). The DSL parser accepts a bare clock literal
`HH:MM[:SS]` for a string field; the printer quotes the ISO form. language_agent GROUNDS a bare clock time before the
card is made (like `passages_to_rooms`): the next occurrence in local time, today if it is still ahead, else
tomorrow. A full ISO value is kept as written.

**People see** the card line `wait at="2026-11-14T18:00:00+01:00"` and, while it runs, the wait note
`waiting until 18:00 (in 42 min)` through the existing "… waits: note" line.

**Behaviour while waiting.** `wait` claims the base: the robot holds still. A new mission supersedes it (as now).
"At 18:00 go to room 3" = `wait at=18:00` then the legs. The planner skill gains the step and one example.

**Not in 1.** Recurring times ("every day at 9"), a wait that survives a controller restart, a wait under `repeat`
measured from the clock (it is measured per iteration, like `s`).

## 3 — the robot's own past (BUILD)

Two uses: refer to a past PLACE ("go back to where you were 10 minutes ago"), and condition on a past VISIT ("if you
have already been in room 3 today, skip it").

**Whereabouts (language_agent, pure `whereabouts.h`).** Every digest refresh (~500 ms) gives the current room and the
robot pose in its frame. Whereabouts keeps a time-stamped trace: a sample is stored when the room changes, and
otherwise at most one per `period_ms` (5 s). In memory only, at most 24 h; a language_agent restart starts it empty
(the episodic memory keeps the long record; see "Later"). It answers:

- `at(t)`: the last sample at or before `t` (room, x, y, yaw, the sample's time). Nothing before the first sample.
- `visits()`: per room, the intervals the robot was in it (merged across samples), the latest last.

**In the STATE** (every turn, for the chat model and the planner), one line:
`where you have been (now 00:31): room_1 now since 00:28; room_2 00:22–00:27; room_3 00:15–00:21 (not yet: none)`
— each room once, the latest visit, oldest visits dropped past a few. With it the planner can write "skip room_3" and
the chat can answer "have you been in the bathroom?".

**A tool `where_was(ago_min | at)`** for the chat model. It answers with the sample:
`at 00:21 (10 min ago) the robot was in room_3 at pose:room_3(1.20, -0.40, 90deg)`, plus the `ways from here` legs to
that room when it is not the current one. The model then calls plan_mission with a request that names that pose; the
planner already writes `cross` legs + `go_to pose:ROOM(x, y)` for remembered places (missions.md, remembered
objects). If nothing is known that far back the tool says so ("no record before 00:12, when the chat started").

**Uncertainty.** A past pose is where the robot believed it was then; the reply says how long ago, and the go_to
arrival tolerance is the planner's as for any pose.

**Not in 3.** A run-time predicate `visited(room) within s` evaluated by the controller during a mission; poses in
proto-rooms that were later renamed (the sample keeps the name it had); persistence across restarts.

## 2 — deadlines (LATER)

`before="…ISO…"` on a mission or a step, as a PREFERENCE over arrival time: the plan's expected value falls with
P(arrive after the deadline). It needs a duration model per leg (from the episodic memory's outcome stats: cross,
approach, explore durations), so the robot can say "I won't make it by 12" before starting. No hard cutoff.

## 4 — while (LATER)

"While you go to room 3, look for bottles": a step that runs alongside the body and claims no resource the body
claims (the IR's Resources already allow it, §7.4 `parallel`). Close to the shelved watch tool (10-31).

## 5 — periodic (LATER)

"Check the bathroom every hour": a mission that re-schedules itself; needs persistence of the schedule (the chat or
the controller restarting must not drop it) and a way to list and cancel standing missions.

## Tests

- IR: `wait at` validates (ISO with offset), rejects a bare `18:00` and a malformed value; JSON and text round trip.
- Text: `wait at=18:00` parses (clock literal) and prints quoted; `wait at=18:00 until=…` keeps both.
- Interpreter: a fake Host wall clock — the step runs before `at`, is satisfied at `at`, a past `at` ends at once,
  `until` still ends it early; the note says "until HH:MM (in N min)".
- Grounding: `18:00` at 17:00 → today 18:00; at 19:00 → tomorrow 18:00; a full ISO value unchanged; nested steps.
- Whereabouts: sampling (room change always stored, one per period otherwise), `at(t)` before/after/between samples,
  visits merge, the 24 h cap, the STATE line, the where_was answers (same room / other room / too far back).
