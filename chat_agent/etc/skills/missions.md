---
description: how to write a mission (multi-step, conditional or return-trip task) in the robot's mission language
note: form TEXT chosen (eval 2026-10-23, tests/check_llm_missions.py, 24 cases x3, Orin; JSON arms graded under OLD rules, not re-run: qwen3.8 63/72 both forms, 8b json 48/72). Final regraded text arm (round 2): gemma4:26b 66/72 (0.93 s median), qwen3.8 66/72 (3.59 s), 8b 31/72 (round 1 grading). Models fence their output in ``` blocks: the planner must strip code fences before parsing.
---
Output ONLY the mission text, nothing else. If the request is unclear or names no concrete goal or target, output one line: `ask: <question>` (just the question, no angle brackets). If the request does not say WHAT to do (it asks for no step: cross, open, go_to, approach, explore, wait, move, say, report, stop), output the `ask:` line; never invent a task for the person. You never choose the goal: when the request leaves the choice of what to do to you, the answer is `ask:`.
Use only names from the STATE (doors, rooms, objects). Never invent a name.

Form: the line `mission "<what it does>"`, then the steps indented by 2 spaces.
A step is: op, positional values, then key=value keywords. Numbers carry their unit: budget_s=60s, turn_deg=90deg, standoff_m=0.8m.

Steps:
- cross DOOR: approach the door, open it, go through; optionally add expect=ROOM
- open DOOR
- go_to REF: REF is anchor:NAME, pose:ROOM(x, y) or pose:ROOM(x, y, yaw); never a door (doors: cross / open); optionally add standoff_m=0.8m
- approach OBJECT: optionally add side=right
- explore ROOM budget_s=60s: budget_s is required; optionally add until=explored(ROOM)>=0.9
- wait s=5s: optionally add until=PRED
- move distance_m=1m turn_deg=90deg: turns first, then drives; either keyword alone is fine (move turn_deg=90deg). There is no op called turn
- say "text": shown to the person
- report "topic": the model describes the room the robot is in now
- anchor NAME: a step; remembers this spot; return with go_to anchor:NAME
- repeat 3 / seq: their body is the steps indented below; repeat until=PRED also needs budget_s=
- stop
Any step: retry=N, on_fail=skip|abort.
Predicates: explored(ROOM)>=p, in_room(ROOM), door_open(DOOR)<p.
bind=NAME works on cross only (never on open); it names the room you arrive in; use it later as $NAME (the room may not exist yet).
Rooms the robot remembers but cannot see: the STATE line `ways from here` gives each room's legs in order; write them as cross steps exactly as given, in that order, each with retry=2; after them that room may be named (explore room_7).
Way back: cross the same door again with expect=ROOM_YOU_CAME_FROM, one cross per door, in reverse order. To end on the very spot you started from, put `anchor start` first and go_to anchor:start AFTER crossing back (go_to only drives inside the room where its anchor was taken).

```
mission "go through door_1"
  cross door_1 retry=2
```
```
mission "look at the room behind door_1 and come back"
  anchor start
  cross door_1 retry=2 bind=next
  explore $next budget_s=60s
  report "the objects in this room"
  cross door_1 expect=room_1 retry=2
  go_to anchor:start
```
```
mission "explore until the room is known"
  explore room_1 until=explored(room_1)>=0.9 budget_s=300s
  say "done exploring"
```
```
mission "turn around and go to the table"
  move turn_deg=180deg
  approach table_1
```
```
mission "visit the rooms behind door_1 and door_0, report each, return"
  anchor home
  cross door_1 retry=2 bind=a
  explore $a budget_s=60s
  report "what is in the first room"
  cross door_1 expect=room_1 retry=2
  cross door_0 retry=2 bind=b
  explore $b budget_s=60s
  report "what is in the second room"
  cross door_0 expect=room_1 retry=2
  go_to anchor:home
```
A remembered room two rooms away, with the robot in room_5 and the STATE line `ways from here (each leg in order): room_7 = cross door_4 expect=room_6, cross passage:passage_3@room_6 expect=room_7`:
```
mission "go to room_7, report, come back"
  cross door_4 expect=room_6 retry=2
  cross passage:passage_3@room_6 expect=room_7 retry=2
  report "what is in room_7"
  cross passage:passage_3@room_7 expect=room_6 retry=2
  cross door_4 expect=room_5 retry=2
```
