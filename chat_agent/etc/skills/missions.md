---
description: how to write a mission (multi-step, conditional or return-trip task) in the robot's mission language
note: form TEXT chosen (eval 2026-10-23, tests/check_llm_missions.py, 24 cases x3, Orin; JSON arms graded under OLD rules, not re-run: qwen3.8 63/72 both forms, 8b json 48/72). Final regraded text arm (round 2): gemma4:26b 66/72 (0.93 s median), qwen3.8 66/72 (3.59 s), 8b 31/72 (round 1 grading). Models fence their output in ``` blocks: the planner must strip code fences before parsing.
---
Output ONLY the mission text, nothing else. If the request is unclear, output one line: `ask: <question>`.
Use only names from the STATE (doors, rooms, objects). Never invent a name.

Form: the line `mission "<what it does>"`, then the steps indented by 2 spaces.
A step is: op, positional values, then key=value keywords. Numbers carry their unit: budget_s=60s, turn_deg=90deg, standoff_m=0.8m.

Steps:
- cross DOOR [expect=ROOM]: approach the door, open it, go through
- open DOOR
- go_to REF [standoff_m=0.8m]: REF is anchor:NAME or pose:ROOM(x, y[, yaw]); never a door (doors: cross / open)
- approach OBJECT [side=right]
- explore ROOM budget_s=60s [until=explored(ROOM)>=0.9]: budget_s is required
- wait s=5s [until=PRED]
- move distance_m=1m turn_deg=90deg
- say "text": shown to the person
- report "topic": the model describes the room the robot is in now
- anchor NAME: a step; remembers this spot; return with go_to anchor:NAME
- repeat 3 / seq: their body is the steps indented below; repeat until=PRED also needs budget_s=
- stop
Any step: retry=N, on_fail=skip|abort.
Predicates: explored(ROOM)>=p, in_room(ROOM), door_open(DOOR)<p.
bind=NAME works on cross only; it names the room you arrive in; use it later as $NAME (the room may not exist yet).
Rooms the robot remembers but cannot see: call route_to(room); it lists the passages in order. Write the first leg as cross door_N (the live door it names), every later leg as cross passage:ID@ROOM expect=NEXT_ROOM retry=2, ROOM being the room that leg starts in.
Way back: cross the same door with expect=ROOM_YOU_CAME_FROM, or go_to anchor:NAME.

```
mission "go to door_1 and cross"
  cross door_1 expect=room_2 retry=2
```
```
mission "look at room_2 and come back"
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
mission "visit two rooms, report each, return"
  anchor home
  cross door_1 retry=2 bind=a
  report "what is in the first room"
  cross door_2 retry=2 bind=b on_fail=skip
  report "what is in the second room"
  go_to anchor:home
```
```
mission "go to room_3, two rooms away"
  cross door_1 expect=room_2 retry=2
  cross passage:passage_1@room_2 expect=room_3 retry=2
```
