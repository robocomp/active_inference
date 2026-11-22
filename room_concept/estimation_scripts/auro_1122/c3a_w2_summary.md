# C3a object tours in the new world, 3 runs, 2026-11-22 22:14-22:53

## Setup
- **World**: the owner's 19:26 world:
  - CLOSET_0 replaces WoodenTable(1);
  - chair(4) and chair(5) added;
  - the bottle moved to (0.44, -0.58).
- **Builds**: the same as c3a_3 and c3a_cab1. Cabinet, nightstand, bed and toilet come from the cabinet-exit worktree (e5310a2a); the rest from main.
- **Start**: LTOM empty each run.
- **CHAIR_3**: at (-7.43, 8.51) in all 3 runs, left there by the moved-chair series. Graded with `C3A_POSE_OVERRIDE="chair(3)=-7.4323,8.5130"`. Moved back to the owner's (-7.97, 9.50) at 22:55.
- **Runs**: all 4/4 legs, trace checks S1-S5 OK.

## Results
| run | held / seen | instances per held | phantoms (cabinet / of which near the closet / bottle / other) | precision excl. cabinet + bottle | median planar error (m) |
|---|---|---|---|---|---|
| c3a_w2_1 | 13 of 19 | 1.23 | 13 (10 / 3 / 3 / 0) | 11/11 = 1.00 | 0.17 |
| c3a_w2_2 | 14 of 19 | 1.14 | 10 (7 / 3 / 3 / 0) | 12/12 = 1.00 | 0.15 |
| c3a_w2_3 | 15 of 19 | 1.13 | 16 (14 / 6 / 2 / 0) | 12/13 = 0.92 | 0.25 |
| old world: c3a_2 / c3a_3 / c3a_cab1 | 11/15, 14/19, 13/19 | | 0, 7, 5 (all cabinet) | 1.00, 0.93, 1.00 | 0.14, 0.20, 0.18 |

### Per class across the 3 runs (held of 3)
- Held 3 of 3: chair (6 of 6 in two runs, 5 of 6 in one), nightstand 2/2, bed, toilet.
- Refrigerator: 2 of 3.
- Hood: 1 of 3; never born in 2.
- Tables: held 0, 1 and 1 of 4. 2–3 were never born per run. The old world shows the same table and hood weakness.

### Phantoms
- **Cabinets**: 7–14 per run, up from 5–7 in the old world.
  - 3–6 per run lie within 1.3 m of CLOSET_0, a Cabinet proto the grader does not list as a cabinet (it lists only the Worktop*).
  - So part of the rise is the closet itself, possibly correct detections graded as phantoms.
- **Bottle**: 2–3 per run, always at about (-6.33, 5.37) on the kitchen side. The real bottle is at (0.44, -0.58), and none was ever born there.
  - The bottle phantom is new with this world. Unexamined.

## Data notes
- **Harvest guard misfire**: the new guard took agents.txt's mtime as the run start, but each agent launch appends to it.
  - Refused wrongly: the hood log in all three runs, the refrigerator log in w2_2, and the table log in w2_1 (older than the last append).
  - The hood and fridge logs of w2_3 were recovered by re-harvesting after the fix. The others were overwritten by later runs and are lost. They hold NEES rows only; the C3a grade comes from LTOM and is complete.
  - Fixed: the run start is now agents.txt's creation time.
