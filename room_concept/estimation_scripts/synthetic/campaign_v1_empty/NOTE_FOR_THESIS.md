# Synthetic-room generalisation campaign (2026-10-11, code f92a016, unchanged)

The same algorithm and configuration as campaign v3 were run on 10 procedurally generated **empty** rectilinear rooms (no furniture).
Everything lives in `/home/pbustos/claude_scratch_2026-10-10/synthetic_rooms/`.

## Setup
- **World:** `synthetic_rooms.wbt`, also at `webots-shadow/worlds/piso/synthetic_rooms.wbt`. It holds 10 rooms placed 30 m apart in one Webots world, with walls 2.5 m high and ceilings.
- **Room generation:** `scripts/make_world.py`, seeds 1000–1009.
  - Each room is a rectilinear polygon of 8–28 vertices and 24.6–87.6 m², built by cutting notches or bays into a rectangle.
  - Ground truth is in `rooms.json`; `rooms.png` shows all 10.
- **Runs:** one run per room. The robot is teleported to a random free start (`starts.csv`), and room_concept starts fresh each time.
- **Grading:** same as v3 (rigid SVD alignment of the trajectory, then polygon boolean IoU, precision and recall). The script is `scripts/grade_synthetic.py`.

| room | GT verts | area m² |
|---|---|---|
| 0 | 8 | 36.7 |
| 1 | 12 | 26.9 |
| 2 | 16 | 36.4 |
| 3 | 20 | 47.5 |
| 4 | 24 | 70.7 |
| 5 | 28 | 87.6 |
| 6 | 12 | 24.6 |
| 7 | 16 | 57.4 |
| 8 | 20 | 64.3 |
| 9 | 24 | 49.1 |

## Headline numbers (n = 10)
- **Success:** 10/10 (Wilson 95% CI 0.72–1.00).
- **IoU at promotion:** median .985 [.979, .987], worst .952.
- **IoU after refinement:** median .988, 95% bootstrap CI [.985, .989], worst .979. Precision .992, recall .997.
- **Refinement gain:** median +.003; IoU improved in 9/10 runs and got worse in 1. Wilcoxon p = .006.
- **Pose:** plain-SLAM median position error 4 mm.
- **Promotion from the first scan:** 6 of 10 rooms promoted at t < 0.1 s with 0 m driven, because the whole room was visible from the start.
  - The other 4 promoted after 27–38 s and 1.6–5.4 m.
- **Stability:** leave-one-out moves the refined-IoU median by at most .001.

## Defects worth noting
- **Room 0 (refined .979, the worst):** a 0.9 × 0.5 m bay was never entered or seen.
  - It is smaller than the robot-footprint gain threshold (0.347 m²) as seen from any reachable standpoint, so the explorer correctly judged it "not worth a trip".
  - This is a designed resolution limit, not a failure.
  - (This cause is inferred from geometry, not instrumented.)
- **Room 2:** the promoted outline is too coarse (14 vertices against 16 in GT, IoU .952). The refiner then fixes the area (IoU .985) but builds a staircase of 38 vertices.
  - The MDL vertex charge did not prevent this; the cause has not been diagnosed yet.
  - This is open.

## Comparison with the apartment (campaign v3)
| | apartment (furnished, 60.4 m², 32 v) | synthetic (empty) |
|---|---|---|
| Refined IoU median | .974 | .988 |
| Refined IoU worst | .968 | .979 |
| Pose error median | 1.0 cm | 0.4 cm |
| Time to promotion median | 43 s | 0.06 s (6/10 instant) |

The roughly .014 gap is consistent with furniture being the main error source in the apartment: worktops and cabinets lie about 30 cm off the walls, and the outline flip-flops between the two.

## Files
- `results.csv`, `results_table.tex` (`\input`-ready), `stats.txt`.
- `layouts_panel.png`: green = GT, black = promoted, red = refined.
- `run_*/`: per-run logs.
- `scripts/`.
