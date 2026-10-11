# Room-layout estimation: update of 2026-10-11 (commit bdb6095)

This folder is
`/home/pbustos/robocomp/components/active_inference/room_concept/thesis_results/2026-10-11_bdb6095/`.

| Folder | What it contains |
|---|---|
| `apartment_v4/` | 10 starts in the simulated apartment on the new build. Same starts as campaign v3. Replaces v3. |
| `synthetic_v2_LT/` | 10 synthetic empty rooms on the new build. Rooms 1 and 6 are now L-shaped and rooms 3 and 8 T-shaped; rooms 0, 2, 4, 5, 7 and 9 are unchanged. Replaces the first synthetic campaign. |
| `regraded_previous/` | The numbers already in the thesis (apartment v3, synthetic v1, repeatability), re-scored with the new IoU metric. |

Each campaign folder holds:
- `stats.txt`
- `results_table.tex`, `\input`-ready, using the same labels as before (`tab:estimation-campaign`, `tab:estimation-synthetic`) so it replaces the old table in place;
- `layouts_panel.png`: green = ground truth, black = promoted, red = refined, orange = path;
- `results_shape.csv`, the raw per-run numbers.

`synthetic_v2_LT/rooms.png` shows the ten ground-truth rooms.

The 100-layout test has not been run.

## 1. Build
- **Commit:** `bdb6095` on branch `wip/box-estimator-rework`, room_concept.
- **Builds the same as f92a016:** the controller, residual and the rest of the fleet.
- **Sensor chain:**
  - Synthetic v2 ran on the old chain (Webots2Robocomp + lidar3d_dds), the same as v3 and synthetic v1.
  - **Apartment v4 ran on `shadow_bridge_dds`**, the new direct-to-DDS bridge, because that was the chain running at the time. This is the one environmental difference from v3.

## 2. What changed in the code since f92a016

**Design change (one):** the layout refiner now proposes more candidate rectangles. In `src/layout_refiner.h`, `rectangle_proposals()` adds, for each connected patch of contested cells, its bounding box. These sit beside the largest rectangles it already proposed.
- **Contested cells** are free cells outside the outline, or never-swept cells inside it.
- **Why:**
  - A bay that the robot only sees at an angle through its mouth is swept as a triangle, the visibility wedge.
  - The largest rectangles inside a triangle form a staircase. Each step buys back a few free cells, so each step was adopted. In synthetic room 2 the outline grew from 14 to 38 vertices; the truth is 16.
  - The single rectangle covering the whole bay was never proposed.
- **The selection criterion is unchanged:** two-part MDL, i.e. the weighted negative log-likelihood plus κ per vertex. The scorer still chooses between the bounding box and the steps.

**Instrumentation (no behaviour change):** `tmp/refine_input.bin` records the refiner's exact inputs at each evaluation, so a refinement can be replayed offline.

**Bug fixes:** none in the estimator.

**Evaluation change:** the scoring of IoU. This is not in the robot code, but it moves every number.
- **Old metric:** the old graders aligned ground truth to the estimate with an SVD fit over the whole trajectory. That fit inherits the pose error and lands up to 0.9° off, which is about 11 cm across a 7 m room. In synthetic v2 it made room 9 read .962.
- **New metric:** IoU is now taken after the best rigid fit of the ground-truth outline to the estimated outline. It measures shape only.
- **Frame error:** reported separately, from an alignment anchored exactly at the start pose (the map's own frame).
- **Precision and recall:** still use the trajectory alignment.
- **Script:** `synthetic_rooms/scripts/regrade_anchor.py`.

## 3. Superseded numbers
All IoU numbers below are refined (after 1 minute of refinement), shape metric unless marked SVD.

| | In the thesis (SVD) | Same runs, shape metric | New build, shape metric |
|---|---|---|---|
| Apartment: median [95% CI] | .974 [.971, .978] | .978 [.973, .981] | **.977 [.975, .980]** |
| Apartment: worst | .968 | .970 | **.973** |
| Synthetic: median [95% CI] | .988 [.985, .989] | .989 [.987, .990] | **.994 [.991, .996]** |
| Synthetic: worst | .979 | .984 | **.987** |

**Recommendation:** use the new-build tables. Quote shape IoU throughout and say how it is defined: the best rigid alignment of the ground-truth outline to the estimate.

**Apartment v4:**
- Success 10/10 (Wilson 95% CI 0.72–1.00).
- IoU at promotion: median .966, worst .898.
- Refined IoU is better than promoted in 10/10 runs, median gain +.012, Wilcoxon p = .002.
- Time to promotion: median 41 s, range 25–184 s.
- Pose error: median 1.7 cm.

**Synthetic v2:**
- Success 10/10.
- IoU at promotion: median .983.
- Refined IoU is better than promoted in 9/10 runs and unchanged in 1, median gain +.012, Wilcoxon p = .004.
- Refined vertex count equals the ground truth in 7 of 10 rooms; the other three have 2 extra.
- All four L and T rooms are recovered with the exact vertex count, at IoU .994–.997.
- Room 2 now has 16 vertices, matching ground truth (it was 38).
- Room 0's bay, which the explorer never visits (below its minimum-gain threshold), is now added by the refiner: IoU .985 → .995. **Correct the earlier wording.** It is still true that the explorer does not go there, but the final layout no longer misses the bay. It now has 10 vertices against 8 in ground truth.

**Repeatability (shape metric):**
- Refined IoU: within-start sd .0036, between-start sd .0034, ICC .46.
- One-way ANOVA p = .046; Kruskal–Wallis p = .12.
- Under the old metric this was p = .09. The evidence that the start position matters is borderline and depends on the test.
- The effect is small either way: sd about .003.
- Path to promotion still depends on the start (p = .015).

## 4. Results to exclude, or measured through a defect
- **SVD-aligned IoUs:** treat every IoU computed with the SVD trajectory alignment as superseded, as in section 3. They are not wrong in kind, but their spread includes up to about 0.9° of alignment noise.
- **Apartment v4, start 9, first attempt:** the run timed out with no promotion and is excluded. The repeat (in the table) promoted in 31 s.
  - **What happened:** the controller judged the robot's starting pose infeasible for its footprint ("start was NOT footprint-feasible"). It refused every route, so the robot never moved. The explorer, meanwhile, rated the target reachable.
  - **Not reproducible:** same landing pose, same binaries. It did not happen in any of the 28 other apartment runs (v3, repeatability, v4).
  - **For honesty:** 10/10 on first attempt is not true here. It is 9/10 first attempt, and 10/10 counting the one repeat. Kept in `estimation_campaign/campaign_v4/failed_run9_try1/`.
- **Open observation, not an exclusion:** in the start-anchored frame, almost every run in all four campaigns has the map rotated the same way, by +0.1° to +0.8° (one run −0.03°), relative to the true start heading. That includes both chains and both worlds. It is systematic, so it is probably a convention or mount offset (gt_theta convention or LiDAR yaw), not map error. It has not been diagnosed. It does not affect shape IoU.
- **Pose error is higher in both new campaigns:**
  - Apartment: 1.7 cm vs 1.0 cm in v3. Confounded with the chain change.
  - Synthetic: 1.1 cm vs 0.4 cm, same chain; room 9 went from 4 to 19 mm.
  - The metric is the median over the whole run, including localisation against the refined layout after promotion.
  - Not explained yet. If the thesis quotes pose error, quote the v3 and v1 values as plain SLAM before promotion, and note this.
