# Engine1 live tour (shared estimation engine, first live run)

Status 2026-11-24. Analysis only, no agent code changed. n = 1 tour, one pass per room, one instance per physical object (two nightstands). Nothing here is a statistic about the build.

## The run
- Run dir: `room_concept/estimation_scripts/crossing/objects_engine1`. Fixture `three_rooms_door0`, 2 legs (room_2 in 31 s, room_3 in 26 s, both OK). Agents started 06:16:17 CET.
- Build: branch `feat/shared-engine`, head `bada10a3` (`agents.txt`), countable_concept from the worktree.
- Comparison: `objects_step0`, same fixture and legs (56 s and 45 s), old binary main `81ab85d1`, started 01:19 CET.
- World: live `threerooms-piso.wbt`, mtime 2026-11-22 19:29:47, before both runs. It is the world they ran in.
- Names: the graph's `room_3` (the bathroom) is where the toilet is. The scorer's polygon names are different (toilet in its `room_2`, bed and nightstands in its `room_3`). I use the scorer's names only inside the NEES table.
- The runs differ in how long the robot spent in the rooms (the engine1 legs are faster), so counts of rows are not comparable one to one.

## 1. Agent rates
Source: each agent's `etc/timing_log.csv` (one row per compute tick). The run dir did not harvest it; I read it from the checkouts (`active_inference.worktrees/shared-engine/<class>_concept/etc/` for engine1, `active_inference/<class>_concept/etc/` for step0) and cut it to each run's start time. The rows span 265 to 314 s, so they cover the whole tour. Nominal is 10 Hz (`Period.Compute = 100`).

| class | step0 rate (Hz) | engine1 rate (Hz) | gaps > 150 ms (step0 / engine1) | max gap (ms) (step0 / engine1) |
|---|---|---|---|---|
| bed | 9.93 | 9.85 | 13 / 23 | 525 / 700 |
| nightstand | 10.00 | 10.00 | 1 / 1 | 158 / 152 |
| toilet | 10.00 | 10.00 | 0 / 2 | 132 / 159 |

Nightstand and toilet are at nominal. Bed is 1.5% below nominal and 0.08 Hz below step0. The loss is not spread over the run. It comes from a few bursts.

Where the time goes (median / p95 / max ms of the tick, when the agent holds at least one instance):

| class | n_inst | step0 tick | engine1 tick | step0 publish | engine1 publish | engine1 infer | engine1 existence |
|---|---|---|---|---|---|---|---|
| bed | 1 | 11.1 / 53.3 / 444.5 | 31.0 / 74.4 / 699.1 | 0.8 / 1.2 / 24.2 | 23.1 / 39.8 / 77.9 | 0.0 / 34.0 / 657.6 | 1.0 / 5.0 / 21.3 |
| nightstand | 2 | 16.2 / 37.7 / 72.0 | 15.1 / 36.2 / 89.1 | 1.1 / 1.6 / 8.5 | 5.9 / 12.3 / 29.6 | 0.0 / 9.1 / 64.2 | 2.8 / 9.2 / 30.4 |
| toilet | 1 | 12.5 / 32.5 / 45.8 | 20.1 / 41.8 / 72.4 | 0.5 / 1.7 / 14.3 | 3.6 / 7.2 / 20.5 | 6.8 / 22.3 / 26.8 | 4.3 / 8.1 / 24.4 |

- `publish` holds the epistemic offer (the planner call and the node write). It rose from about 1 ms to a per-tick cost, because the offer is now computed every tick.
- Per instance and tick: bed 25.5 ms mean, nightstand 3.3 ms, toilet 4.3 ms. The expected 20 ms is true for the bed only. The nightstand and the toilet are 5 to 6 times cheaper.
- Bed, whole agent: mean tick 40.1 ms against 19.7 ms in step0; ticks over the 100 ms period 35 against 13; p99 134 ms against 111 ms.
- The bed stalls:
  - Step0 had one 8-tick burst of 260 to 525 ms at 138 s, all inside `infer` (the first big view).
  - Engine1 has the same kind of burst at 113 to 118 s (8 ticks, 330 to 700 ms, `infer` 290 to 660 ms), plus 15 smaller ticks of 150 to 190 ms.
  - Six of those are `publish` spikes of 66 to 78 ms with `infer` = 0. That looks like the planning cycle at its full cost, about 70 ms (I did not confirm the cause in code).
- The stall in `infer` is not new. It is larger by 30% in its worst tick (700 against 525 ms). I cannot say from one run whether that is the build or the view.
- Conclusion: rates are at nominal for nightstand and toilet. For the bed they are 9.85 Hz. The offer cost (about 25 ms per tick on the bed) takes a quarter of the 100 ms budget, and its spikes of 70 ms plus a normal tick are what push ticks past 100 ms. This is the point to watch if more beds are born.

The other agents' rates are not in these files and I did not measure them.

## 2. First-visit NEES/DOF and planar error
Tool: `tools/step0/nees_persistent.py` (default `--gt mesh`, which already carries the footprint centre, so the toilet is scored against its footprint centre and `NEES_CENTRE` changes nothing here). Per-row output: `nees_persistent_rows.csv` in a scratch copy of each run (the run dirs were not written). Every row is a first visit. NEES uses the agent's own diagonal sigma (`std_cx`, `std_cy`). Calibrated 2-DOF reference: median 0.693, P(<= 1) = 0.63.

| node | GT object | rows | median NEES/DOF | frac <= 1 | median err (m) | max err (m) | median sigma (m) | last-row NEES/DOF |
|---|---|---|---|---|---|---|---|---|
| step0 bed_19 | bed(1) | 1141 | 0.45 | 0.77 | 0.048 | 0.125 | 0.059 | 0.13 |
| **engine1 bed_2** | bed(1) | 874 | **0.90** | 0.55 | **0.138** | 0.333 | **0.270** | 0.50 |
| step0 nightstand_35 | right | 433 | 0.07 | 1.00 | 0.020 | 0.079 | 0.069 | 0.57 |
| engine1 nightstand_1 | right | 800 | 0.18 | 0.96 | 0.023 | 0.035 | 0.055 | 0.10 |
| step0 nightstand_36 | left | 795 | 0.61 | 0.92 | 0.037 | 0.139 | 0.082 | 0.33 |
| engine1 nightstand_2 | left | 639 | 0.59 | 0.96 | 0.018 | 0.038 | 0.045 | 0.04 |
| step0 toilet_11 | Toilet | 72 | 3.67 | 0.00 | 0.130 | 0.168 | 0.072 | 2.16 |
| **engine1 toilet_1** | Toilet | 129 | **86.8** | 0.00 | **0.192** | 0.211 | 0.089 | 91.75 |

Frame fits (rms over the room visit): room_1 0.029 m, room_2 0.016 m, room_3 0.011 m in engine1; 0.031, 0.037, 0.010 in step0. Same size, so the frame does not explain the differences.

Reading:
- Nightstands: at or better than step0. Errors are 2 cm, sigmas 4 to 6 cm, NEES/DOF 0.18 and 0.59. Their maximum errors (3.5 and 3.8 cm) are about half of step0's. This is the best result of the run.
- Bed: the error went up from 4.8 cm to 13.8 cm (median), up to 33 cm at the start, and the median sigma went up 4.6 times (0.059 to 0.270 m). NEES/DOF is 0.90, which looks calibrated, but it is calibrated because sigma is large, not because the belief is accurate. 93% of rows are inside the 95% limit (2.99 per DOF) of the 2-DOF test; 55% are at or below 1.
- Toilet: worse than step0 on both counts. The error is 19 cm (13 cm in step0) and `std_cx` is 1.4 cm (about 3 times tighter than step0's 4.4 cm), so the NEES/DOF is 87. It does not recover: first row 0.21 m, last row 0.19 m. The belief gets 128 updates (ungated rows) in 13 s, against 9 in 2.5 s in step0, and sigma shrinks around a position that stays 19 cm off. This is a miscalibrated, over-confident belief with more evidence behind it. One instance, one view, so I cannot say if it is bias in the toilet model, the truth point or the engine. The truth point (footprint centre) is the same one as in step0.

### Sigma per DOF (`std_*` in `<class>_ai2_log.csv`, medians over all rows; floor-near values in brackets)
| node | cx | cy | H | w | h | yaw |
|---|---|---|---|---|---|---|
| step0 bed_19 | 0.054 | 0.022 | 0.019 | 0.107 | 0.044 | 0.157 |
| engine1 bed_2 | 0.045 | 0.267 | 0.046 | 0.176 | 0.200 | 0.186 |
| engine1 nightstand_1 | 0.051 | 0.019 | 0.021 | 0.025 | 0.024 | 0.055 |
| engine1 nightstand_2 | 0.044 | 0.010 | 0.015 | 0.019 | 0.019 | 0.031 |
| engine1 toilet_1 | 0.014 | 0.088 | 0.020 | 0.044 | 0.028 | 0.157 |

Bed w, the DOF the shape latent is meant to hold near its floor (about 0.04 m):
- It does not. Median sigma_w is 0.176 m (step0 0.107); 27% of rows are at or below 0.06 m, 33% at or below 0.1 m, and 43% are at 0.2 m, which is the cap.
- It sits near the floor only at the first frame (0.043) and in the one ungated stretch near cycle 451 (0.049, 8332 points). The 91 ungated rows of the run (of 874) have median sigma_w 0.048 and sigma_cy 0.022, so during an update the sigmas are at their floors.
- On the 783 gated rows (no update) sigma grows with no new data to caps of 0.3 (cy) and 0.2 (w, h): 42% of rows have `std_cy` at 0.3. Step0's bed on its 1057 gated rows did not do this (0% at 0.3).
- So the shape latent does not hold sigma_w low across gated stretches. The belief widens while not looked at, then falls again on the next update. Whether that is the intended persistent-latent behaviour (sigma grows between views) or a defect is a question for the owner; the logs cannot tell. The wide sigma helps the bed's NEES.
- The estimated bed size at the last row is w 1.64 and h 2.08 (step0 1.57 and 2.15); the mesh gives mattress 1.57 x 1.97 (`tools/gt_mesh/mesh_extents.py`; the step-0 doc has the box). Dimension errors were not scored here.

## 3. Promise against delivered
`promise_delivery.py` on `objects_engine1`. The step0 run has no offers or delivered files (old binary), so there is no comparison.

Offer and delivery logs:
| class | offers (rows) | nodes | offer rows with promise <= 0 | median promised (nats) | delivered rows | delivered, sum (nats) | delivered rows < 0 |
|---|---|---|---|---|---|---|---|
| bed | 1270 | bed_2 873, bed_1 397 | 22 | 6.62 | 228 | 30.0 | 136 |
| nightstand | 1800 | 899 + 901 | 2 | 0.74 | 334 | 32.8 | 110 |
| toilet | 131 | 131 | 0 | 0.39 | 128 | 13.3 | 14 |

Claim windows (the controller's `[aff-select] -> 'aff_<class>_<id>'` lines):
| class | n offers | claim windows | with a reach | median promised | median delivered | Spearman |
|---|---|---|---|---|---|---|
| bed | 1270 | 0 | 0 | none | none | none |
| nightstand | 1800 | 4 | 0 | 1.20 | 0.00 | too few (n = 4; script says rho -0.82, p 0.18) |
| toilet | 131 | 0 | 0 | none | none | none |

- Only the nightstand was ever claimed (4 selections). No window has a reach (the "arrival to end" window has n = 0), so the "delivery after a reach" test cannot be done on this run.
- Of the four, three have a window of 0 s (reselected or refused in the same second: see section 4). The one real window is `nightstand_2`, 7.0 s: promised 0.709 nats, delivered 1.958 nats (2.8 times the promise). One window is not a comparison.
- The rho of -0.82 is from 4 points where 3 deliveries are zero. I do not use it.
- Over the whole run: the sum delivered is 30.0 (bed), 32.8 (nightstand) and 13.3 (toilet) nats, with 136 of 228 bed delivery rows negative (an update that raised the entropy). Negative deliveries are 60%, 33% and 11% of rows. I do not know whether a negative value is expected when sigma was inflated while gated; it may be the same effect as in section 2.
- The bed promise is large (median 6.6, up to 11 nats) while the controller shows `gain=5` for `aff_nightstand_*` and `aff_bed_*` in both runs. The controller's number is not the logged promise. I did not follow where the 5 comes from (a verification floor is the likely source: it is also 5 in step0).

## 4. Offers and refusals
Search: `STOPPED TARGET REFUSED` in `logs/controller.log`.

| run | refusals in total | on a countable target | on other targets |
|---|---|---|---|
| step0 | 3 | 1 (`aff_nightstand_36` at (-1.52, 2.69)) | 2 (`aff_door_21`, `aff_door_22`) |
| engine1 | 2 | 2 (`aff_nightstand_1` at (0.95, 2.60), again at (0.99, 2.61)) | 0 |

- Cause, all of them: "occupancy belief: a wall, object or unbelieved residual cell lies under the body in the turn it ends with at the published standpoint ... reported unreachable" (the same text as step0 Step 3). The log does not separate wall from object from residual.
- Both refused engine1 claims are on the same node, 0.04 m apart in standpoint, 5 s apart (lines 1284 and 1335). The nightstand's own offers put the standpoint at (0.857, 2.63), 0.09 m from the refused one. So the controller refused a standpoint that the agent's p_stand had not blocked, or a standpoint that moved by 0.1 m between the offer and the claim. I cannot tell which; p_stand is not logged.
- `aff_nightstand_2` was claimed once and did not get refused (7 s window, 1.96 nats delivered).
- Bed and toilet were never selected (no `aff_bed*` or `aff_toilet*` string in the controller log). Their affordances were created with gain 0, so there was nothing to refuse.

Published offers with gain 0 (`nbv_gain_pub` in `<class>_ai2_log.csv`, i.e. `epistemic_gain` after the suppression):
| node | rows | published gain = 0 | share |
|---|---|---|---|
| bed_1 | 256 | 18 | 7% |
| bed_2 | 874 | 852 | 97% |
| nightstand_1 | 804 | 279 | 35% |
| nightstand_2 | 639 | 302 | 47% |
| toilet_1 | 131 | 131 | 100% |

- A zero here can come from three causes: p_stand (the standpoint is not feasible), the epistemic cooldown (`EpistemicCooldownCycles = 200`), and the verification floor. The logs carry no p_stand column, so I cannot say how many of these are "blocked by p_stand". The number that is certain: 0 offers had `promised_nats` exactly 0 (23 bed, 2 nightstand rows had a promise at or below 0.01, the minimum being -1.78).
- The `nbv_gain_raw` column is 0 in every row of every class, so it cannot be used to separate the causes. Its sibling `promised_nats` in `offers.csv` is the raw gain. The log hook is written to record the raw gain; the column being zero looks like a logging fault, not a zero belief. I did not check the code path further.
- Nightstand: five `[NBV] no usable face - every viewpoint is inside an obstacle or outside the room. REFUSING` lines at the birth of nightstand_2 (1795497491.07 to .50). Step0 has the same five.

## 5. Errors and anomalies
| item | engine1 | step0 |
|---|---|---|
| births (bed / nightstand / toilet) | 2 / 2 / 1 | 1 / 2 / 1 |
| deaths in the run (before the agents stop) | bed_1 (dropped) | none |
| NaN lines | 0 | 0 |
| beam warnings | 0 | 0 |
| agent self-test FAIL lines (door-mode lock, wall no-cross, front-resolver), all three classes | 3 per class | 3 per class |
| `CONDEMNED BUT UNEXECUTABLE` | bed_2 at 06:19:34 (L = -4, p = 0.018) | none |
| presence | clean until the harness stopped room_concept | same |

- Self-test FAILs are at startup in both runs and are not new.
- **Phantom bed, not a twin.** `bed_1` was born at 1795497448.357 (3 s after the agent was ready, before leg 1 started at ...457) at room_1 position (2.12, -0.67), from 5876 points and a residual mass of 110. There is no bed in room_1 in the world (one bed, `bed(1)`, in the bedroom). It was a confident existence belief (p up to 0.98), published an offer of about 9 nats and an affordance (`aff_bed_1`). The agent dropped it at 1795497488.361 (`left behind in another room, dropped`) when the robot left room_1. Step0 had no such birth. This is a false positive of the build or of the starting view; one run says nothing about the rate. It was a first row in the bed's log (256 rows, all gated), and it is not in the NEES table because it matches no ground-truth bed.
- **Twins: none.** Every GT object has one instance:
  - `bed_2`, born 1795497490.061, room_3 (-0.34, 2.27), matches bed(1).
  - `nightstand_1` (born 489.457, room (0.90, 3.76)) matches the right nightstand; `nightstand_2` (born 491.157, room (-1.55, 3.79)) matches the left one. They are 2.45 m apart, and the nearest-object match gave each a distinct GT object in the NEES table.
  - `toilet_1` (born 636.256, room (-2.16, 1.30)) is the only toilet.
  - `bed_1` and `bed_2` never live together: bed_1 was dropped 1.7 s before bed_2 was born, and in a different room frame. So bed_1 could not have explained away bed_2's returns through the neighbour density; they are not twins.
  - No second node of the same class was within the object's footprint in the graph at any time that I could see in these logs. I did not scan `graph_trace.csv` node by node for a short-lived twin; the births list is the evidence.
- **Bed existence collapse:** bed_2's existence log odds sit at L = 4 (p 0.98) while observed, reach -4.0 (p 0.02) in 544 to 569 s (after the robot left the bedroom, with the LiDAR carve seeing 130 to 490 free), and return to p 0.86 at 581. It was marked `CONDEMNED BUT UNEXECUTABLE` at 574 (no resolving view for 300 cycles). In step0 the same dip appeared once (L = -4 at 758) with no CONDEMN line. It is a not-yet-removed doubt, not a death. Present in both builds.
- Bed ai2 log: 256 rows for bed_1 span only 1.45 s of the log even though the agent kept it alive 40 s, so the per-cycle log under-counts bed_1. Its timing numbers in section 1 do include it.

## What this run does and does not show
- Shows: rates are at nominal for nightstand and toilet and 1.5% low for the bed; nightstands are as good as before or better; no twin; no NaN.
- Shows: three things worse than step0 on this run: bed error 3 times larger with sigma 4.6 times larger; toilet 19 cm off with a tight sigma (NEES/DOF 87); one phantom bed in the first room.
- Cannot show: whether promise predicts delivery. Only one claim window had a nonzero length (7 s, 0.71 promised, 1.96 delivered), none had a reach, and bed and toilet were never claimed.
- Cannot show: how many zero-gain offers p_stand blocked. The log has no p_stand column.
- n = 1 tour. Each class has 1 or 2 instances. Do not read a rate from these numbers.

Files: `tools/step0/engine1_timing.py` (rates and stage costs, used for section 1); `tools/step0/nees_persistent.py` (section 2); `room_concept/estimation_scripts/scripts/promise_delivery.py` (section 3).
