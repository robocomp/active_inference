# c2_x1 (2026-11-21): DOOR_0 shut, robot held 45+ min at the vantage — door_concept says SHUT and OPEN at once

For the door session (from the room session, AuRo C2 series; owner asked this be sent to door). Build main 4ff4684b, binaries:
door_concept 2026-11-18 08:07, controller 2026-11-20 22:57. Explore tour from scratch (`explore_tour.sh`, START room_1), all doors
shut at start (`door.py close`). Logs: `crossing/c2_x1/logs/{door_concept,controller}.log`.

## What happened
room_1 promoted, the exploration drive chose `door_2` (= wbt DOOR_0, room_1 <-> bathroom). The door stayed SHUT all run (owner
confirmed by eye). The controller parked at the vantage (-3.70, 5.44 world) from ~6 min in until the 60 min run limit:
`[why] still STOPPED WAITING for 44m37s — door passage step cross ... holding at the vantage — no mouth seen through the door: P(open) 0.99`.

## The contradiction (the bug)
door_concept publishes two beliefs about the same leaf that disagree, and the controller reads the wrong one:
- `aff_door_2_open` precondition, all run: `P(shut)=0.98..1.00` (door_concept.log 1415, 1579, 2165: `P(shut)=0.99`, `1.00`,
  `0.98`; offer P 0.97-0.99). Correct.
- the door node's `door_open_prob`, which the controller's passage reads (controller_door_passage.h:506/550, door_reads_open /
  door_reads_shut): ~1.00. controller.log: `[door-route] door_2: at the vantage, NO MOUTH seen through the door (P(open) 1.00 ...)`.
Because `door_open_prob` > 1/2, the passage's guard "door reads SHUT while crossing -> back to `open`" (controller_door_passage.h
~1089-1105, added for warm37) never fires, so nobody claims the standing `aff_door_2_open` offer and the actuator is never asked.

## How it got there (controller.log 706-745, door_concept.log 923-936)
1. door_concept.log 923: `[door_2] SWING FLIPPED — ... pooled log-odds of along -0.60 over 1 viewpoints ... phi 34 deg` — the shut
   leaf read as swung 34 deg, from ONE viewpoint.
2. controller 706-735: at the vantage, door reads shut, claims `aff_door_2_open` twice (orienting to its standpoint).
3. controller 736: `the door is open — the route continues through it`; 740: door_concept briefly offered `aff_door_2_cross`
   with `P(passable)=0.58, clear span 0.49 m at phi=1.03 rad` (a 59 deg leaf on a shut door), then withdrew it (741).
4. From then on: `door_open_prob` stays ~1 while the open precondition's P(shut) is ~1. Deadlock: the passage waits in `cross` for a
   mouth; the open offer stands unclaimed.

## Asks (door's call)
- Why do `door_open_prob` and the open affordance's P(shut) diverge? They should be one belief (one leaf phi posterior).
- One-viewpoint SWING FLIP at phi 34 deg on a shut door: same family as the clean71 "weak DOOR_0" failure.
Nothing was changed by the room session. The tour is counted as a failed C2 tour; it is rerun (c2_x1b) at the end of the series.

## Owner's suggestion (2026-11-21)
In a deadlock like this, ask the VLM for a second opinion on the leaf state to unblock it. The room session's reading: treat the VLM
answer as one more observation of the leaf with its own measured reliability, asked as an epistemic action when the passage is
stalled or door_concept's two beliefs disagree, rather than as an override or a timeout. Not built; door's design call.
