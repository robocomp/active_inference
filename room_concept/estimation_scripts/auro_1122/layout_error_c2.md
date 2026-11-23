# Layout error for C2, "Fig. threerooms" (2026-11-23)

Script: `auro_1122/scripts/review2/layout_err.py` (scratch only, run with `python3 -I`; no run, no Webots, no process started). Output: `layout_err.json`.
Nothing existing was edited.

## Which tours
The paper's Fig. threerooms (`submission/cortex-auro.tex:1400-1413`, script `figures/src/plot_layouts_3starts.py`, `RUNS = clean75, clean68, clean76`) draws
the three tours of the frozen series, one per start room: **clean75 = start in the bath, clean68 = start in the bedroom, clean76 = start in the big room**.
Data are the offline re-solves in `Papers/.../Autonomous Robots /data/fig9/clean{75,68,76}_layout_dump.json` (made with `ltsm_agent/tools/layout_harness.cpp:75-110`
from the last archived memory of each tour; `before_pose` = `r.world_raw`, the layout chained through the passages, `after_pose` = `r.world`, the MAP).
The live `ltsm_history/*_layout.json` only holds the MAP (`x,y,yaw`), not the chained layout, and its energies differ (clean75: F 17.2 live vs 3.85 re-solved), so the dumps are the only source of the "before" layout.
Run folders: `crossing/clean75`, `clean68`, `clean76` (all dated 2026-11-22 10:56 by mtime, a copy; the runs were made 2026-11-13).

## Room frame convention (read before using the numbers)
The room frame is NOT a geometric property of the room. Room 1 sits at pose (0,0,0) in the layout and its outline is not centred on the origin
(clean75 room_1 outline vertices start at (3.12,-2.44)); the frame is the one the room concept started in (the robot's pose at the start, or at the crossing for later rooms).
So "true room pose" is defined here as follows, with no use of any layout:
1. Each room's own-frame outline is recovered from the dump: `outline_local = R(yaw)^T (outline - (x,y))`. For before and after the local outline is the same (max difference 8e-5 m), so the solve moves poses only.
2. The true room polygon is built from the COMMITTED world (`git -C webots-shadow show HEAD:worlds/piso/threerooms-piso.wbt`, copied to scratchpad with `meshes/` symlinked; `scripts/wbt_rooms.py rooms()`; areas 60.88 / 25.10 / 31.77 m2 for big room / bath / bedroom).
3. The true pose of each room frame T_i (room frame to world) is the rigid transform that maximises IoU of the local outline with the true room polygon (Nelder-Mead, 5 starts, seeded from the dump's `truth_transform` composed with the MAP pose). The fit IoUs are 0.955-0.994, column 3.
4. The layout (before, or after) is then mapped to the world by ONE rigid 2-D alignment A (rotation = circular mean of the per-room yaw differences, translation = mean of the per-room residuals), and the error of room i is the difference between A applied to its layout pose and T_i. Translation in metres, rotation in degrees (signed).

The per-visit est-to-GT fit with the gauge (`scripts/run_frames.py`) was NOT used: it gives the room frame only to the extent the localiser is a rigid copy of the world per visit, and the room belief is outline-based anyway.

## Results
| tour (start room) | room | fit IoU of outline to true room | before: translation (m) | before: rotation (deg) | after (MAP): translation (m) | after: rotation (deg) |
|---|---|---|---|---|---|---|
| clean75 (bath) | bath (r1_room_1) | 0.987 | 0.101 | -0.12 | 0.030 | -0.08 |
| clean75 (bath) | bedroom (r2_room_2) | 0.993 | 0.117 | -0.12 | 0.011 | -0.40 |
| clean75 (bath) | big room (r3_room_3) | 0.980 | 0.069 | +0.24 | 0.020 | +0.48 |
| clean68 (bedroom) | bedroom (r1_room_1) | 0.984 | 0.017 | +0.39 | 0.030 | -0.29 |
| clean68 (bedroom) | bath (r2_room_2) | 0.994 | 0.067 | -0.72 | 0.013 | -0.25 |
| clean68 (bedroom) | big room (r3_room_3) | 0.974 | 0.084 | +0.33 | 0.042 | +0.54 |
| clean76 (big room) | big room (r1_room_1) | 0.975 | 0.099 | -0.25 | 0.020 | +0.28 |
| clean76 (big room) | bath (r2_room_2) | 0.993 | 0.046 | +1.79 | 0.032 | +0.19 |
| clean76 (big room) | bedroom (r3_room_3) | 0.955 | 0.123 | -1.54 | 0.025 | -0.47 |

**Medians over the 9 room-tour pairs (3 tours x 3 rooms)**

| | translation (m) median [max] | |rotation| (deg) median [max] |
|---|---|---|
| before the solve (chained through passages) | 0.084 [0.123] | 0.33 [1.79] |
| after the solve (MAP) | 0.025 [0.042] | 0.29 [0.54] |

Per tour medians (translation m, |rotation| deg): clean75: before 0.101 m / 0.12 deg, after 0.020 m / 0.40 deg; clean68: before 0.067 m / 0.39 deg, after 0.030 m / 0.29 deg; clean76: before 0.099 m / 1.54 deg, after 0.025 m / 0.28 deg; 

## Caveats
- The truth of a room frame comes from fitting the LEARNT outline to the true polygon (step 3), so errors below roughly the outline's own accuracy (a few cm; fit IoU 0.955-0.994, lowest for the bedroom of clean76) are not resolved. Read the "after" translations (1-4 cm) as "at the limit of what this method can tell", not as a measured 1 cm.
- The world file is the committed HEAD version (last commit 2026-11-21, adds DEF names on the chairs); the runs were made on 2026-11-13. Walls are what the polygons use; I did not diff the wall geometry between the two dates.
- These are the offline re-solves of the paper's figure, not the layout the agent held live. The live MAP of clean75 had F=17.2 (`ltsm_history/ltsm_20261113_192519_249_layout.json`) against 3.85 in the re-solve; the live poses were not graded here.
- "Before" is the chained layout of the solver's `world_raw`; the one rigid alignment has 3 free parameters fitted to 9 numbers (three room poses), so the residuals are not independent samples and are smaller than an alignment to an external reference would leave.
