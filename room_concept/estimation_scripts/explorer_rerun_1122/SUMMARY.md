# Explorer rerun on real floor plans (AuRo item 2), 2026-11-22

## Setup
- Dataset: MatterportLayout annotated room layouts, the 594 used before.
  - File: `room_concept/datasets/matterport_layout/corpus594.txt`, one polygon per line, metres.
  - Source: the MatterportLayout annotations (Zou et al., "Manhattan Room Layout Reconstruction from a Single 360° Image", IJCV 2021) of Matterport3D (Chang et al., 3DV 2017).
  - Exporter and split files: `datasets/matterport_layout/`.
  - All 594 plans were run, not a subset; each takes about 1 s.
- Build: main 640298ef plus an uncommitted change to `tools/wall_slam_selftest.cpp`. The change adds `WS_NR_LAYOUTS=<file>` to the `WS_NEWROOM` branch.
  - The start pose matches test 8: the deepest interior point of a 0.25 m scan, heading +x.
  - Binary: `bin/wall_slam_selftest`, built 2026-11-22 11:00. The system libdsr_api.so is dated 2026-11-21 22:17.
- Current explorer: `tools/newroom_bench.h`, run_newroom. This is room_concept's current pipeline:
  - plain SLAM → free-space raster → manhattan_layout;
  - the coverage explorer `rc::layout::plan_coverage`, which is what publishes afford_room live;
  - description-length promotion, then the LayoutRefiner.
- Budgets: 900 frames, the earlier cap, and 8000 frames, the bench default. They give identical results.
  - 590 of 594 plans end on their own at 900 frames, and 594 of 594 at 8000.
  - Median 365 frames, max 1270.
  - Per-plan difference between the budgets: 593 ties and 1 loss; the mean differs by −0.00002.
- Earlier explorer, both arms rerun on the same binary today so that every plan can be paired: `run_explore`, test 8, 900-frame cap.
  - EIG arm: default settings.
  - Heuristic arm: `WS_NO_INFOGAIN=1`.
  - This rerun reproduces the 09-14 null: EIG − heuristic = −0.0009, 95% CI [−0.0026, +0.0008], 244/74/264 wins/ties/losses.
- Files in `room_concept/estimation_scripts/explorer_rerun_1122/`:
  - `per_plan.csv`: per plan, the current explorer's IoU, seen fraction, frames and stop reason at both budgets, plus the earlier arms' IoU and coverage;
  - `pair.py`;
  - `newroom_matterport594_f{900,8000}.csv`: the bench's full per-plan CSVs;
  - `old_eig/run.log`, `old_heur/run.log`.

## Results (IoU against the true polygon)
| arm | n | IoU mean | IoU median | min | hard failures | seen / cover mean |
|---|---|---|---|---|---|---|
| current coverage explorer (newroom), 900 | 594 | 0.962 | 0.970 | 0.261 | 0 | 0.979 |
| earlier run_explore, EIG | 594 | 0.982 | 0.987 | — | 9 | 0.999 |
| earlier run_explore, heuristic | 594 | 0.983 | 0.988 | — | 6 | 0.999 |

Paired differences per plan. Plans where either arm failed are dropped; failures count as IoU 0.000.
- current − EIG: −0.0199, 95% CI [−0.0235, −0.0164], median −0.017. Wins/ties/losses 67/3/515, n = 585.
- current − heuristic: −0.0208, 95% CI [−0.0238, −0.0178], median −0.018. Wins/ties/losses 72/11/505, n = 588.

## How to read it (important)
- **The comparison crosses harnesses, not just explorers.** run_explore is the old wall-SLAM estimator plus its own explorer. The newroom harness is the current free-space and manhattan_layout pipeline plus plan_coverage. So the 0.02 gap mixes estimator and explorer. It is not a pure explorer effect.
- **The current explorer stops on its own.** The earlier one never did: 594 of 594 plans ran to the cap, which is why its A/B was saturated.
  - The current explorer ends at a median of 365 frames, using about 40% of the budget, with 0 hard failures.
  - The price is a tail of early stops: 84 of 594 plans end with seen < 0.95, and their median IoU is 0.960 against 0.971 for the rest.
  - The worst plan, room 92 (10 corners), was promoted with 20% of the room seen, IoU 0.261. The earlier arms scored 0.901 and 0.797 there.
  - Other bad plans: room 492 at 0.749 (seen 0.58) and room 305 at 0.799 (seen 0.63).
  - This is a real weakness: the room is declared done before it has been covered.
- **No frontier baseline exists in the newroom harness**, so only the current explorer was run there.
- The earlier run's baseline was the in-house heuristic in `tools/wall_slam_selftest.cpp:1570-1680`, the `cfg.info_gain=false` path. It is not a published method and has no citation.
  - Targets: unknown wall cells from `collect_unknowns`.
  - Viewpoint score: the visible unknown mass divided by (1 + 0.1·distance).
  - Every fourth replan goes to the nearest frontier instead, as a coverage turn.
  - The nearest-frontier part resembles Yamauchi (CIRA 1997), but it is not an implementation of Yamauchi's method.
