# Room-layout estimation in furnished rooms (commit 22d763b9, 2026-10-11)

This folder is
`/home/pbustos/robocomp/components/active_inference/room_concept/thesis_results/2026-10-11_furnished_22d763b/`.

| File | What it contains |
|---|---|
| `results_table.tex` | `\input`-ready table, label `tab:estimation-synthetic-furnished`. It is a NEW table; it does not replace `tab:estimation-synthetic`. |
| `stats.txt` | Summary statistics over the 10 runs. |
| `layouts_panel_furniture.png` | One panel per room with the furniture drawn in grey. Green = ground truth, black = promoted, red = refined, orange = path, blue dot = start. |
| `layouts_panel.png` | The same panel without furniture, aligned by the trajectory fit. |
| `results_shape.csv` | The raw per-run numbers. |
| `rooms.png` | The ten ground-truth rooms. |

The campaign itself, with per-run logs, is in
`room_concept/estimation_scripts/synthetic/campaign_v4_furn_fixed/`.

## 1. Build and set-up
- **Commit:** `22d763b9` on branch `wip/box-estimator-rework`. room_concept behaves as in `bdb6095`. The only code change is the residual fix described in section 4.
- **Sensor chain:** `shadow_bridge_dds` (sensors sent straight from the Webots controller to DDS).
- **Rooms:** the same ten rooms as the empty L/T campaign (`synthetic_v2_LT` in the `2026-10-11_bdb6095` package): six Manhattan rooms with random notches and bays, two L-shaped rooms (1 and 6) and two T-shaped rooms (3 and 8). Same start poses.
- **Furniture:** 47 pieces over the ten rooms.
  - 19 tall cabinets, 2.0 m high, against the walls.
  - 15 worktops, 0.9 m high, against the walls.
  - 13 tables, 0.75 m high, standing on legs away from the walls.
  - The tall cabinets hide the wall behind them from the LiDAR. The worktops hide its lower part. The tables are seen mostly as four legs and a top.
- **Protocol:** one run per room, first attempt only, every run counted. A run that does not promote within 10 minutes is a failure. None was repeated.

## 2. Headline numbers (this campaign, 10 runs, first attempt)
All IoU values are shape IoU: the ground-truth outline is moved by the best rigid fit onto the estimated outline, so the number measures shape only.

- **Success:** 10/10 (Wilson 95% CI 0.72–1.00).
- **Refined IoU** (after one minute of refinement): median **.994** [95% CI .991, .997], worst **.989** (rooms 4 and 9).
- **Promoted IoU** (at the moment the room is accepted): median .985 [.976, .992], worst .954 (room 2).
- **Refinement helps every time:** refined is better than promoted in 10/10 runs, median gain +.008, Wilcoxon p = .002.
- **Pose error:** median over runs of each run's median error, 1.1 cm (range 0.4–1.7 cm).
- **Time to promotion:** median 27 s, range 0–80 s. Room 5 took 80 s and a 9.6 m path; it took 23 s in the pre-fix campaign. Four rooms were promoted from the first 360° scan, without moving.

**Vertex count against ground truth, per room:**

| Room | Type | GT vertices | Promoted | Refined | Refined IoU |
|---|---|---|---|---|---|
| 0 | M | 8 | 8 | 8 | .997 |
| 1 | L | 6 | 14 | 12 | .998 |
| 2 | M | 16 | 14 | 16 | .994 |
| 3 | T | 8 | 12 | 12 | .995 |
| 4 | M | 24 | 22 | 26 | .989 |
| 5 | M | 28 | 28 | 28 | .993 |
| 6 | L | 6 | 6 | 6 | .994 |
| 7 | M | 16 | 18 | 18 | .993 |
| 8 | T | 8 | 8 | 8 | .999 |
| 9 | M | 24 | 24 | 24 | .989 |

- The refined count is exact in 6 of 10 rooms.
- The extra vertices are small steps in the wall, not misplaced walls. In room 1 the extra edges are 0.4 cm, 2.3 cm, 3 cm and 14 cm long; in room 3 they are 3 cm, 12 cm and 14 cm; in room 7, 2.4 cm; in room 4, 0.2 cm. They sit where a piece of furniture meets the wall. They cost almost nothing in IoU (room 1 is .998 with 12 vertices against 6).
- In the empty rooms the two rooms that now carry these steps (L room 1 and T room 3) had the exact count. The steps are therefore an effect of the furniture.

## 3. Comparison: empty rooms, furnished before the fix, furnished after the fix
Same ten rooms, same starts. Shape IoU, refined unless marked.

| | Empty (campaign v2 L/T) | Furnished, before the fix (campaign v3) | Furnished, after the fix (this) |
|---|---|---|---|
| Build | `bdb6095` | `bdb6095` behaviour, old residual | `22d763b9` |
| Sensor chain | old (Webots2Robocomp + lidar3d_dds) | `shadow_bridge_dds` | `shadow_bridge_dds` |
| Success, first attempt | 10/10 [.72, 1.00] | **9/10** [.60, .98] | **10/10** [.72, 1.00] |
| Refined IoU, median [95% CI] | .994 [.991, .996] | .994 [.990, .996] (9 runs) | .994 [.991, .997] |
| Refined IoU, worst | .987 | .990 (9 runs) | .989 |
| Promoted IoU, median | .983 | .987 | .985 |
| Refined better than promoted | 9/10, p = .004 | 8/9, p = .020 | 10/10, p = .002 |
| Refined vertex count exact | 7/10 | 6/9 | 6/10 |
| Pose error, median | 1.1 cm | 1.1 cm | 1.1 cm |

**Reading the table:**
- Furniture does not lower the accuracy of the recovered outline. The median refined IoU is .994 in all three campaigns, and the worst room stays at .987–.990.
- The cost of furniture shows in the vertex count (small steps where furniture meets a wall), not in IoU.
- The fix changes the success rate, not the accuracy: it removes the one failure of the pre-fix campaign.
- The pre-fix furnished numbers are over 9 runs, because the failed run is excluded from the IoU statistics. That makes them slightly optimistic.
- The empty campaign ran on the old sensor chain, so the empty-vs-furnished comparison also changes the chain. The two furnished campaigns share the chain and differ only in the fix.

## 4. Failure analysis: the stall before the fix
The same failure happened twice before the fix: apartment v4, start 9, first attempt (reported in the `2026-10-11_bdb6095` note), and furnished room 8, first attempt (campaign v3). Both runs timed out without promoting. Both logs are kept: `estimation_scripts/apartment/campaign_v4/failed_run9_try1/` and `estimation_scripts/synthetic/campaign_v3_furn/failed_run8_try1/`.

- **Symptom:**
  - The robot's first target turned out to be unreachable, so the robot parked.
  - While it was parked, the region the controller could reach around it shrank until nothing was left. In apartment start 9 the route search went from 4,526 expanded cells to 2. In furnished room 8 it went from 8,492 to 1.
  - The controller then logged "start was NOT footprint-feasible": it judged that the robot's own footprint was blocked where it stood. It refused every route, and the robot never moved again until the 10-minute timeout.
- **Cause:**
  - The obstacle map came from residual_concept, the agent that marks what the room's walls do not explain. It used two radii around the robot that did not agree.
  - It ignored LiDAR returns within 0.45 m of the robot (they are hits on the robot's own body). It also refused to clear any map cell within 0.55 m of the robot.
  - Cells in the ring between 0.45 m and 0.55 m could therefore be marked as occupied by a stray return, but never cleared while the robot stood still.
  - The ring filled up around the parked robot and walled it in. The controller read that wall as an obstacle overlapping the robot.
  - The stall needs the robot to be parked for a while, which is why it was rare and looked unreproducible.
- **Fix (commit `22d763b9`, residual_concept):** one body disc instead of two. The radius inside which returns are ignored is now at least the radius inside which cells may not be cleared (0.55 m). Nothing can be marked that cannot later be cleared.
- **Verification:** this campaign. 10/10 on first attempt, and "NOT footprint-feasible" appears in none of the ten controller logs. The single re-run of room 8 on the fixed build, made earlier, also promoted (shape IoU .999).
- **Recommendation:** report first-attempt success before and after the fix: 9/10 → 10/10 in the furnished rooms, and 9/10 first attempt in apartment v4 (10/10 counting the repeat). Present the stall as a failure found by the experiments and fixed.

## 5. Failures and anomalies in this campaign
- **No failures.** All ten runs promoted on the first attempt.
- **Transient "no route" holds (not the stall):**
  - The controller logged "HOLDING — no route" once in room 1 and six times in room 6.
  - These are a different situation from the stall. The route search was large (10,000–30,000 expanded cells) and the start was footprint-feasible. The target was unreachable at that moment, but the robot was free.
  - Both rooms promoted (room 6 from the first scan) and finished normally. The same holds appear in the pre-fix campaign.
- **Extra vertices:** the small steps where furniture meets a wall, described in section 2. In the furniture panel, the refined outlines (red) follow the walls behind the furniture. No furniture is pulled into a refined outline.
  - The promoted outline (black) is less clean. In room 1 it wraps around a tall cabinet against the left wall, so the cabinet is cut out of the room. In room 3 one long wall is drawn slightly outside its true position (not near furniture).
  - Refinement corrects both.
- **Frame offset:** as in the earlier campaigns, the map's start-anchored frame is rotated by +0.1° to +0.9° from the true start heading in every run. This is still not diagnosed and does not affect shape IoU.

## 6. What this supersedes
- Nothing previously sent to the thesis covered furniture. These furnished synthetic numbers are new.
- The empty-room numbers (`synthetic_v2_LT`) and the apartment numbers (`apartment_v4`) in the `2026-10-11_bdb6095` package stay as they are.
- The pre-fix furnished campaign (`campaign_v3_furn`) was never sent. If it is quoted, quote it only as the "before the fix" column above, from its `stats.txt` and `results_shape.csv`. Its `results.csv` and `results_anchor.csv` were regraded after a re-run of room 8 and no longer describe the first attempt.
