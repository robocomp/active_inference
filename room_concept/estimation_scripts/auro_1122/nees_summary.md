# Per-agent NEES calibration from existing logs (AuRo item 1), 2026-11-22

Script: `room_concept/estimation_scripts/scripts/nees_agents.py <run>...` (uncommitted). It writes `<run>/nees_rows.csv`, one row per logged update.
No new runs were made.

- **NEES/DOF** = eᵀP⁻¹e / DOF, computed with the agent's own published σ. Only diagonal σ are logged, so cross terms are absent.
- **Calibrated reference**: median NEES/DOF is 0.455, 0.693 or 0.789 for 1, 2 or 3 DOF, and P(NEES/DOF ≤ 1) is 0.68, 0.63 or 0.61.
- **Builds**: c3a runs are on 6f8a850b (`<run>/agents.txt`); the c3b and c2 binaries are as listed in the C2 report.

> **⚠ REGRADED 2026-11-22 (later the same day).**
> - The first object numbers were a grading artefact. room_concept keeps an internal frame (robot pose, `est_x`) and a published frame (DSR nodes, objects), related by a logged gauge: `proto_room.csv` gauge_x/y/theta = T_pub_int, `room_scene_graph.cpp:1363`. The gauge changes at every crossing and re-anchor.
> - The first pass mapped objects through the fixture layout's room poses and ignored the gauge. Every room_2/3 object and every return-visit object therefore carried the gauge as error, up to 0.58 m and 0.09 rad. Fable found this; I verified it.
> - Objects are now taken world → internal (per-visit fit) → published (gauge at the row's time). `c3a_objects.py` uses the same path.
> - `NEES_FIXTURE_FRAMES=1` and `C3A_FIXTURE_FRAMES=1` restore the old path. The old grades are kept as `<run>/c3a_*_fixtureframes.csv`.
> - The gauge is constant inside all 63 visits of the 12 runs, so the room-pose rows (per-visit fit) are unaffected.
> - The old "one fit per run, 18.4" row is withdrawn: it measured the gauge jumps between visits, not pose error.

## Results
| agent | state (DOF) | runs | units | n rows | median NEES/DOF | frac ≤ 1 | median ‖e‖ | median σ (‖(σx,σy)‖) |
|---|---|---|---|---|---|---|---|---|
| room | robot pose x,y,θ (3), per-visit frame fit | 12 | 63 visits | 93 967 | 0.39 | 0.62 | | |
| room | robot position x,y (2), per-visit fit | 12 | 63 visits | 93 967 | 0.08 | 0.82 | 0.013 m | 0.051 m |
| room | robot heading θ (1) | 12 | 63 visits | 93 967 | 0.75 | 0.54 | | |
| door | width w, height h (2) — uninformative, prior mean = truth | 12 | 105 node×run | 44 568 | 0.08 | 0.95 | | ~0.02 m each |
| chair | centre cx,cy (2), first visit | 4 | 15 | 2 593 | 0.47 | 0.65 | 0.140 m | 0.137 m |
| chair | centre, return visit | 3 | 9 | 4 009 | 1.03 | 0.50 | 0.204 m | 0.127 m |
| table | centre, first visit | 4 | 4 | 1 118 | 2.33 | 0.38 | 0.063 m | 0.053 m |
| table | centre, return visit | 3 | 6 | 3 097 | 4.39 | 0.16 | 0.186 m | 0.097 m |
| refrigerator | centre, first visit | 4 | 4 | 1 918 | 0.79 | 0.68 | 0.122 m | 0.356 m |
| refrigerator | centre, return visit | 3 | 3 | 1 620 | 0.00 | 0.76 | 0.027 m | 0.400 m |
| hood | centre, first visit | 1 | 1 | 1 188 | 29.8 | 0.00 | 0.128 m | 0.026 m |
| hood | centre, return visit | 1 | 1 | 191 | 1.80 | 0.15 | 0.150 m | 0.103 m |
| nightstand (countable) | centre, first visit | 3 | 6 | 7 593 | 5.70 | 0.24 | 0.074 m | 0.038 m |
| nightstand (countable) | centre, return visit | 2 | 2 | 1 756 | 1.67 | 0.38 | 0.062 m | 0.043 m |
| bed (countable) | centre, first visit | 3 | 3 | 4 512 | 1.30 | 0.39 | 0.055 m | 0.042 m |
| bed (countable) | centre, return visit | 2 | 2 | 2 637 | 0.38 | 0.89 | 0.063 m | 0.424 m |
| toilet (countable) | centre, first visit | 2 | 3 | 4 831 | 44.4 | 0.02 | 0.167 m | 0.021 m |

Final belief per instance (last row), NEES/DOF median with frac ≤ 1:

| class | instances | median NEES/DOF | frac ≤ 1 |
|---|---|---|---|
| chair | 16 | 0.56 | 0.56 |
| refrigerator | 4 | 1.13 | 0.50 |
| bed | 4 | 0.91 | 0.50 |
| hood | 1 | 0.85 | 1.00 |
| nightstand | 6 | 1.20 | 0.33 |
| table | 7 | 5.6 | 0.14 |
| toilet | 3 | 27 | 0.00 |

Runs used:
- Room and door rows: c3a_1, c3a_2, c3a_3, c3a_cab1, c3b_on1, c3b_off1, c3b_on_hidden1, c3b_off_hidden1, c3b_on_ltom1, c3b_off_ltom1, c2_w2, c2_w3.
- Object rows: c3a_1, c3a_2, c3a_3, c3a_cab1 only (static tours).

## How to read it (after the regrade)
- **Chair, refrigerator and bed are close to calibrated.** Their final beliefs sit around NEES/DOF 0.6–1.1.
- **Table and toilet are overconfident; the nightstand is close.** At the final belief the table is about 8× off, the toilet about 40× (17 cm error against 2 cm σ, 3 instances), and the nightstand 1.7×. The countables' errors are now 5–17 cm against σ of 2–4 cm.
- **Hood looks overconfident during the first visit** (30× over 1 188 rows), but its final belief is calibrated. It is one instance, and its Webots origin is unverified.
- **The remaining overconfidence fits one mechanism (Fable's diagnosis, not yet tested).** A persistent error (visible surface, shape prior, pose chain) is treated as independent from frame to frame, so σ shrinks with the number of frames and not with the number of viewpoints.
- **Return visits are no longer catastrophic.** Chair 1.03 and table 4.4 come with 0.19–0.20 m errors, up from about 0.06–0.14 m on the first visit. The cause is unexamined.
- **Most of the refrigerator's old 0.3 m error was the reference point** (fixed 22:20, see the caveats). Graded against the footprint centre, the median error is 0.12 m on the first visit and 0.03 m on return. σ stays at 0.36–0.40, so the fridge is now UNDER-confident (return-visit NEES/DOF about 0).
- **The room-pose and door readings are unchanged.**

## Regraded C3a table (c3a_objects.py, gauge frames), old → new
| run | held / seen | phantoms | median planar error (m) | precision (matched / instances) | precision excl. cabinets | recall (seen objects with a match) |
|---|---|---|---|---|---|---|
| c3a_1 | 8 → 10 of 15 | 12 → 8 | 0.28 → 0.22 | 0.44 → 0.59 | 0.71 → 0.71 | 0.67 → 0.80 |
| c3a_2 | 9 → 11 of 15 | 2 → 0 | 0.29 → 0.14 | 0.83 → 1.00 | 0.83 → 1.00 | 0.67 → 0.80 |
| c3a_3 | 10 → 14 of 19 | 8 → 7 | 0.32 → 0.23 (0.20 with the fridge reference fix) | 0.56 → 0.68 | 0.71 → 0.93 | 0.53 → 0.74 |
| c3a_cab1 | 12 → 13 of 19 | 5 → 5 | 0.34 → 0.19 (0.18 with the fridge reference fix) | 0.68 → 0.74 | 0.90 → 1.00 | 0.63 → 0.68 |

- Planar error is the per-object median from the report.
- After the regrade every remaining phantom is a cabinet, except one chair and one nightstand in c3a_1.
- The former nightstand/toilet "phantoms" in c3a_2 were real objects graded through the wrong frame.
- The reports are in `<run>/c3a_report.txt`; the old ones are in `<run>/c3a_*_fixtureframes.csv`.

## Caveats
- Ground-truth reference points:
  - Chair, nightstand and bed: the Webots origin is the footprint centre.
  - Toilet: the origin is 0.06 m from the footprint-box centre.
  - Checked 22:20 from the Collada meshes (`meshes/*.dae`: metres, Z up, identity node matrix) and the PROTO-less Solids:
    - **Fridge**: the origin is at one edge of the footprint. fridge.dae spans x ∈ [0, 0.539], y ∈ [−0.34, 0.34], so the centre lies 0.27 m along the fridge's local +x.
    - **Table1 (table1.dae)**: centre 0.004 m from the origin, ignored.
    - **RoundTable and KitchenHood** (cylinders, boxes, a symmetric face set): the origin is the centre.
  - `c3a_objects.py` now grades the fridge against its footprint centre (`REF_OFFSET`); `C3A_ORIGIN_REF=1` restores the origin.
  - Old runs are graded against the world they ran in, the committed `threerooms-piso.wbt` (`--wbt`, or `WBT_PATH` for `nees_agents.py`). The live file has changed since.
  - The C3a grade itself changes only in the fridge errors (0.44/0.36/0.27/0.33 → 0.33/0.19/0.02/0.17 m) and in the median planar error of c3a_3 (0.23 → 0.20) and c3a_cab1 (0.19 → 0.18).
  - The pre-fix files are kept as `<run>/*_originref.*`.
  - The chair return-visit row above was stale from an earlier pass. It is corrected here, and the rows are unchanged by this fix.
- Consecutive rows are strongly autocorrelated. The independent units are the instances and visits.
- c3a_3's countable logs were byte-identical copies of c3a_2's. Cause: in c3a_3 the bed, nightstand and toilet agents ran from the cabinet-exit worktree (agents.txt), but the logs were copied from main's etc/. The real logs were recovered from `active_inference.worktrees/cabinet-exit/<class>_concept/etc/ai2_log.csv` (timestamps inside c3a_3's window). The stale copies are kept as `*.stale_copy_of_c3a_2`. `concept_agents.sh harvest <out_dir>` now copies from the checkout each agent ran from and refuses logs older than the run. The C3a grade comes from ltom and is unaffected.
- Object yaw and size are not scored.
- Only diagonal σ are logged, so NEES ignores cross terms.
- Builds: c3a on 6f8a850b (`agents.txt`).
