# Item 14: removed objects, 3 runs, 2026-11-22 21:31-22:08

## Setup
- **World**: the owner's 19:26 world. Main fd0830ce, agents chair, table and nightstand (countable), plus ltom_agent. LTOM starts empty each run.
- **Tour**: warm three-room tour, legs R2 R3 R2 R1.
- **Removal**: after leg 2, with the robot in room_3, three objects are moved out of the house to (30–32, 30):
  - TABLE_ROUND (-6.68, 1.87), room_1;
  - CHAIR_2 (-6.12, 1.90), room_1;
  - NIGHTSTAND_R (-4.00, 13.22), room_2.
- **Removal verified**: during r2, a pose read from Webots after removal put all three outside the house. The readback in removed.txt is taken too early and shows home.
- **Return**: the robot returns to room_2 (leg 3) and room_1 (leg 4).
- **Runs**: all 3 completed 4/4 legs. Report: `c14_report.csv` (`scripts/c14_report.py`).

## Results
**Testable instances.** An object counts only if it was learned before removal: its nearest belief is within 1 m of home, and LTOM holds a uid for it.
- Nightstand: 3 of 3 runs.
- Chair: 2 of 3. Not learned in r3.
- Table: 1 of 3. Not learned in r1 or r3; the robot spends only its start in room_1 before the removal.
- 6 instance-tests in total.

| run | object | home in view after return (s) | belief re-born at home | LTOM recall | memory at end |
|---|---|---|---|---|---|
| r1 | CHAIR_2 | 53.7 | no | offered on entry, held until the run ended | kept, P(exists) 0.88 |
| r1 | NIGHTSTAND_R | 7.1 | no | offered on entry, held until the robot left the room | kept (countable publishes no exist_prob) |
| r2 | TABLE_ROUND | 33.3 | no | offered, held to the end | kept |
| r2 | CHAIR_2 | 36.1 | no | offered, held to the end | kept, P(exists) 0.98 |
| r2 | NIGHTSTAND_R | 24.2 | no | offered, held until leaving | kept |
| r3 | NIGHTSTAND_R | 10.7 | no | offered, held until leaving | kept |

- **No removed object was ever found missing.**
  - Time to removal is never, in all 6 instance-tests.
  - The spot was in view for 7–54 s after return, and the memory survived every time.
  - The recall is withdrawn only when the robot leaves the room, or when the run ends.
- **No phantom.** No belief was ever re-born at a removed object's home. Memory never asserted presence in the graph.
- **No false removals.**
  - Every memory of a non-removed object (7–9 per run) is still in the end repertory.
  - The two chair "[existence] REMOVE" events (r1 −198 s, r3 −244 s) happened before the removal, during the first room_1/room_2 looks. Not examined.

## Why (same cause as item 13, see c3b5_summary.md)
- Leaving a room deletes its beliefs. On return, a recall only renames a NEW detection, by LIFECYCLE §1 "birth is tracker-only".
- A removed object is therefore never re-born, so the agent's absence channel (which removes a live belief) never gets an instance to act on.
- Nothing tests the memory against the empty spot.
- Fix proposed for later, not built: the recall node carries its own look offer and absence channel (owner ruling 2026-11-22).
