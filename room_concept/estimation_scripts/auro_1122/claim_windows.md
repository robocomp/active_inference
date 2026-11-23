# Promised vs delivered for object offers: what the claim window measures

Scripts (scratch, all run): `auro_1122/scripts/review5/cw.py` (per-claim parser + visibility), `an.py` (end-class), `st.py` (stats);
base `auro_1122/scripts/review/t3s1.py`. Rows: `.../review5/A.json`. Runs: `crossing/{s1_rs035_3,4,5, s1_rs100_2,3, lam_0p2}`.
Time resolution: claim times come from the 1 Hz controller line count in gt_pose.tsv (`aff_arbitration.line_times`), so any window shorter than ~1 s reads as 0.0 s.
Visibility = 4 m, +-55 deg, wall LOS (`blocked()` from c3b_report.py), robot pose from gt_pose.tsv, GT object = c3a_instances `match`, else nearest same-class GT <=1 m of the belief.

## 1. The 35 step-1 (0.35) object claims: 18 have no ai2 row in [claim, next [aff-select]]
Reproduced: 35 claims, 18 with nrows==0, 17 with rows. Of the 18:

| how the window ended | n | detail |
|---|---|---|
| refused at once, "occupancy belief ... under the body ... standpoint handed back" (`STOPPED TARGET REFUSED`), robot never moved | 9 | all 0.0 s (<1 s) |
| completed within the cycle, no controller line states the outcome | 4 | chair_10 (4327), chair_44 (4030), nightstand_11, bed_7; 0.0 s; reason unknown |
| preempted by a door mission (next claim `aff_door_N_open`, required name) | 2 | chair_9 (25.5 s), table_6 (26.4 s); en route |
| "no route ... reporting Unreachable" | 2 | table_6 (4.8 s), refrigerator_9 (9.4 s) |
| object node deleted while Executing (agent log) | 1 | table_12 (15.9 s) |

* Reached the offer's viewpoint (`[affordance] aff_X reached -> SERVO lock-on`): **0 of 18** (9 of the 17 updated ones).
* Object in view at some time in the window: 2 of 17 evaluable (table_6 4.8 s, table_12 1.2 s); the other 15 have 0.0 s; 1 has no GT (table_10 `wrong_class`). 13 of 18 windows are <1 s, so 0 s in view is by construction.
* Contrast, 17 updated: dwell completed 8, node deleted 3, refused after driving 3 (refrigerator_7 24.6 s, chair_39 39.1 s, chair_42 16.7 s), door-mission 2, unreachable 1; in view >0 s in 16 of 16 evaluable.
* So "no update" is mostly "no visit": the promise is not tested. It is not a failure of the look.

## 2. How windows end (affordance_manager.cpp, controller_session.cpp)
* An Executing claim is kept before any scoring (`affordance_manager.cpp:1035`, keep-current branch; resumed Executing at :1075). The switch margin (`neg_efe`, :967-969) and "not twice in a row" (`just_done`, :991; yield :1162 ff.) apply only to Offered candidates. A claim therefore never loses a score contest. It ends by: `mark_reached(outcome)` (dwell done, Refused, Unreachable, OutsideRoom, :1340-1390), `complete_claim` by name, the producer deleting the node, or "out of this contest" release (:1020s; required name / room scope; this is the door-mission takeover).
* Dwell is bounded by mask hits (`dwell over: 'X' mask seen n/5`, controller_session.cpp:1131, :1157; config `AffordanceDwellMaskHits`/`AffordanceDwellMaxMs`). 
* After a refusal the state shows `JustCompleted` in the next line (:991-998), then room offer or another object. This is why 12 of 35 claims (0.35 runs) are refusals.
* The `t3s1` window ends at the next `[aff-select] ->` line. `NONE ELIGIBLE` lines do not bound it. Here own end (dwell over / refusal) and next selection differ by <1 s in every case, so this changes nothing.

## 3. Does the measure count visits correctly?
* Chair/table/etc. ai2 rows are written per matched mask frame (`chair_fitter.cpp:608, 633`; median dt 0.12-0.29 s per node, gaps = no mask). So no rows = no mask matched: for chairs, a correct "not observed". 4 windows have rows but all `gated=1` (chair_6 3:749, chair_25, chair_39, chair_42 4453): observed, yet no belief update.
* Zero-length windows are not weak-detection completions. They are refusals/instant completions with no visit (section 1). Delivered = 0 there is true but is an execution result, not information.
* Sigma_before = last row before the claim; lag median is large (up to 39.5 s). If the filter predicts between rows, sigma_before is under-stated; unknown, not measurable from the CSV.
* Consecutive claims of the same object are not split visits. Same-node successive claims (0.35): chair_9, chair_31, chair_45 (each a completed visit, then a new claim 23-46 s later after other offers) and chair_44 (refused, 27 s later). Each is a separate visit. lam_0p2: nightstand_17 (refused, re-claimed 0.0 s later, again), chair_246.

## 4. Better window: per-visit (arrival -> end of dwell), reported next to the all-claims number
Reason: the promise (expected gain of looking from the viewpoint) is only tested when the robot is at the viewpoint. Claim->next-claim mixes in navigation, refusals and mission takeovers, which score 0 and say nothing about the gain. Merging same-object claims was rejected (they are separated by other offers, merged windows would add room-driving time). Window = `reached ->` line to the dwell/refusal end; Sigma_before = last row before arrival. Attrition is reported, not hidden: 0.35: 9 of 35 claims reached (8 completed after dwell); 1.0: 1 of 5; lam_0p2: 5 of 17.

Promised and delivered in nats; median (IQR); delivered = position information + existence entropy where logged.

| set | window | class | n | promised | delivered | Spearman |
|---|---|---|---|---|---|---|
| 0.35 | all claims | ALL | 35 | 3.19 (1.48-3.32) | 0.00 (0.00-0.71) | rho 0.06, p 0.73 |
| | | chair | 22 | 3.19 (1.82-3.21) | 0.00 (0.00-2.33) | 0.10, p 0.65 |
| | | refrigerator | 5 | 4.72 (3.82-5.40) | 0.00 (-1.86-2.44) | 0.40, p 0.50 |
| | | table | 6 | 6.11 (2.68-8.78) | 0.00 (0.00-0.00) | -0.13, p 0.80 |
| | | bed, nightstand | 1, 1 | 0.01, 0.10 | 0.00, 0.00 | n too small |
| | arrival->end | ALL | 9 | 3.19 (3.19-3.26) | 1.54 (0.43-2.38) | -0.48, p 0.19 |
| | | chair | 7 | 3.19 (2.19-3.19) | 1.94 (1.40-3.52) | -0.04, p 0.94 |
| | | refrigerator | 2 | 5.06 | -0.97 | n=2 |
| 1.0 | all claims | chair | 5 | 3.19 (2.20-3.19) | 0.54 (0.00-1.91) | -0.41, p 0.49 |
| | arrival->end | chair | 1 | 2.20 | 3.55 | n=1 |
| lam_0p2 | all claims | ALL | 16* | 5.00 (2.58-5.00) | 0.00 (-0.04-0.10) | 0.18, p 0.51 |
| | | chair | 6 | 2.31 (1.42-3.19) | 0.00 (0.00-2.25) | 0.80, p 0.06 |
| | | nightstand | 5 | 5.00 | 0.00 (-0.51-0.00) | n.s. (promised constant) |
| | | refrigerator / table / hood | 2 / 2 / 1 | 7.67 / 3.66 / 5.00 | 2.40 / -0.08 / 0.41 | n too small |
| | arrival->end | ALL | 5 | 3.19 (3.19-5.00) | -0.04 (-0.05-0.00) | -0.50, p 0.39 |
| | | chair | 2 | 3.19 | 2.19 | n=2 |
| | | table | 2 | 3.66 | -0.03 | n=2 |

*17 claims, 1 (bottle) has no ai2 rows. lam_0p2 promised 5.00 is the floor price (nightstand all 5.00).

Totals, 0.35 runs (nats): promised 120.5 over 35 claims; completed-after-dwell claims promised 25.3 and delivered 23.5; refused 12 claims promised 37.7; door-mission preempted 4 promised 15.3 delivered 2.8; node deleted 4 promised 24.1 delivered 0.6; unreachable 3 promised 13.5.

## 5. Reading
* The per-visit window shows delivery when the robot arrives (chair median 1.94 vs promised 3.19), but rank agreement is absent (rho -0.04, n=7) and n is too small to call anything else: the promised gains of reached chair claims are nearly all 3.19 (no spread).
* Refrigerator and table deliver ~0 or negative at arrival (sigma grows, n=2 each at arrival in 0.35 and lam_0p2). Cause unknown from these logs.
* What the all-claims number mostly measures is the share of claims that reach a viewpoint: 9/35 (0.35), 1/5 (1.0), 5/17 (lam_0p2).
* Caveats: n small; 1 Hz times; GT for phantom/wrong-class instances is nearest-<=1 m or missing; 4 "completed <1 s" cases have unknown outcome.
