# Miss-counter counterfactual on the seven c3a tours, and were the robot's eyes on the 9 false existence removals?

Log-only, 2026-11-23. Nothing run or edited. Scripts (scratch only): `SCR=auro_1122/scripts/review3/`
`m1.py` (per-life tick table, run once per world group; `RNG=4` for the 4 m sensitivity), `m1c.py`/`m1c4.py` (replay + tally), `m1d.py` (view analysis of the 9). Worlds: c3a_1..3 + c3a_cab1 graded against the
committed world copy (`WBT_PATH`), c3a_w2_1..3 against the live file with `C3A_POSE_OVERRIDE="chair(3)=-7.4323,8.5130"`.

## 1. The rule, exactly (and the fact that it is NOT armed in these runs)

`common/instance_tracker/instance_tracker.h:220-234`: each cycle, for every track that got NO detection assigned (`trk_used[t]` false): if `expected_visible` is false the counter is HELD
(`next_miss = miss_count_`); else `miss+1`; at `miss >= death_frames` the track id goes to `out.deaths`. A supported cycle resets the counter to 0. So it is **consecutive** unsupported-while-expected-visible
cycles, not k-of-n. `expected_visible` is the agent's own flag: chair `inst.roi_valid` (`chair_concept/src/specificworker.cpp:538`, set to false on a peripheral confirm `:967`); bottle `update_expected_visible` = centre projects inside the image
(`bottle_concept/src/bottle_fitter.cpp:737-761`); residual range < 6 m (`residual_concept/src/specificworker.cpp:1125-1127`).
`Tracker.DeathFrames = 90` is a **bottle-only** configured value (`bottle_concept/etc/config.toml:229`; and bottle disables it while its existence channel is on, `bottle_concept/src/specificworker.cpp:632`).
chair, table, nightstand/bed/toilet/hood (countable), refrigerator set `death_frames = INT_MAX` (`chair_concept/src/specificworker.cpp:518`, `table_concept/src/specificworker_lifecycle.cpp:91`,
`countable_concept/src/specificworker_lifecycle.cpp:158`, `refrigerator_concept/src/specificworker_lifecycle.cpp:106`). So the counterfactual is a replay of a rule that the agents in these runs do not run.

## 2. What can and cannot be reconstructed

| piece | source | status |
|---|---|---|
| frame = one compute cycle | `cycle` column; chair rows: median dt between consecutive cycles = 100.0 ms (987 pairs, c3a_2) | 10 Hz measured; 90 frames = 9 s, 30 frames = 3 s |
| supported frame | row with `npts > 0` (chair rows exist only when a mask was associated; countable/table rows also exist with `npts = 0`) | reconstructed |
| row time | chair `t_ms`; others `pkt_ts` (frozen when no packet arrives; then `anchor_ts + 0.1 s * dcycle`) | approximated for unsupported rows |
| `expected_visible` | NOT logged for any class | **approximated three ways**: A none (every unsupported frame counts, upper bound); B1 GT frustum (Webots yaw 0 faces +y, +-55 deg, <= 6 m, no wall test); B2 = B1 plus wall line of sight (`blocked()` of `c3b_report.py`). B1 is the closest to the agents' frustum-only flags. Oracle (GT pose, GT object place) and blind to furniture occlusion. Range is my assumption: the only configured range in the repo is residual's 6 m; 4 m sensitivity run too. |
| life end | `[affordance] removed node for '<node>'` line; else the leg leaving the instance's room (+3 s, the rule of `c3a_objects.death_kind`); else last supported row | approximated for 33 lives (last row) |
| existence removals | `[existence] REMOVE/removing` lines | exact; the replay of those 11 lives stops at the removal time |

Scope: 126 lives of chair, table, nightstand, bed, toilet, hood, refrigerator (rows exist). **Excluded**: bottle (no ai2 log: its 2 phantom removals are not derivable here), cabinet (own run model),
and `c3a_cab1` nightstand/bed/toilet: their `*_ai2_log.csv` are byte-identical (md5) to `c3a_2`'s, i.e. stale copies (the cab1 run used worktree binaries). That leaves **122 lives**:
112 real unmoved objects (verdict `match` in `c3a_instances.csv`), 2 phantom, 4 wrong class, 4 unmatched hoods (no instance row; reported as `?`, they never decide a verdict).
Nothing was moved or removed in these tours (Task 4 of review_logs_summary.md), so "real" = still present.

## 3. Counts (lives on which the counter fires; `m1c.py` / `m1c4.py` output)

Existence channel (actual): 11 of the 122 lives removed = **9 real, 1 wrong class, 1 phantom** (+ 2 bottle phantoms outside the table = the 13 of Task 4).
Real removed: 9/112 = 8.0 %. Phantoms removed: 1/2.

| rule, range 6 m | DF = 90 (9 s) | DF = 30 (3 s) |
|---|---|---|
| A no visibility gate | 100 lives (93 real, 2 phantom, 1 wrong class, 4 ?) | 109 (101 real, 2, 2, 4) |
| B1 frustum gate | **6** (4 real, 2 phantom) | **31** (27 real, 2 phantom, 2 ?) |
| B2 frustum + wall LOS | **5** (3 real, 2 phantom) | **22** (20 real, 2 phantom) |

Sensitivity, range 4 m: B1 DF 90: 5 (3 real, 2 phantom); B1 DF 30: 21 (19 real, 2 phantom); B2 DF 90: 5 (3, 2); B2 DF 30: 18 (16, 2).

Which real lives the 90-frame counter would delete (B2): `c3a_2 table_17` (also an existence removal), `c3a_2 nightstand_6`, `c3a_w2_3 chair_223`. B1 adds `c3a_1 chair_96` (also existence).
The 2 phantoms (`c3a_1 chair_98`, `c3a_1 nightstand_5`) are deleted by the counter at 90 frames (after 95.7 s and 152.6 s of life); the existence channel removed chair_98 only, `nightstand_5` was never removed.
Overlap of the counter's 90-frame set with the 9 false existence removals: 1 of 9 (B2: table_17) or 2 of 9 (B1: chair_96, table_17).

## 4. Verdict

At **DF = 90 with a visibility gate** the counter would have deleted fewer real unmoved objects (3-4 of 112 = 2.7-3.6 %) than the existence channel did (9 of 112 = 8.0 %), and would have deleted both phantoms (2/2 vs 1/2).
Fisher exact on real removals 9/112 vs 3/112: p = 0.135; vs 4/112: p = 0.25 (n too small to call it better). At **DF = 30** it is clearly worse on real objects: 20-27 of 112 deleted (p = 0.045 and 0.0017 vs the existence 9).
Without the visibility gate (A) it deletes 93 of 112 real objects: the gate is what carries the rule, and here it is an oracle I supplied. Verdict: **parity-to-slightly-better at 90 frames, worse at 30, and not shown to be better**:
n = 2 phantoms, 3-9 events per arm, and the 90-frame result depends on a 9 s window whose "visible" flag the agents do not log. It also depends on `expected_visible` being right: the agents' own flag (`roi_valid`) is a frustum
test with no wall or furniture term (B1 is the faithful stand-in, B2 is more generous to the counter).

## 5. The 9 false existence removals: was the robot looking at the object's place?

Window = the last 30 s of the life before the removal (some lives are shorter). "Absence ticks" = 0.1 s ticks without an associated detection (`npts > 0` row). In view = object GT place inside +-55 deg, <= 6 m, GT pose interpolated
from `gt_pose.tsv` (rows > 2 s apart skipped, as `c3a_objects`). Logged P(detect) = the agent's `ex_pdetect` / `ex_central` on rows with `npts = 0` (countable/table only: chair_ai2_log has no `ex_*` columns and
`chair_existence_log.csv` is not preserved per run: **not derivable for the 5 chair removals**).

| run | node | absence s | in frustum s (share) | in view with wall LOS s (share) | last support before removal | logged p_detect on absence rows (median [min,max]), central |
|---|---|---|---|---|---|---|
| c3a_1 | chair_94 | 29.9 | 7.2 (24 %) | 0.0 (0 %) | 29.9 s | not derivable |
| c3a_1 | chair_96 | 21.6 | 11.0 (51 %) | 0.5 (2 %) | 21.1 s | not derivable |
| c3a_1 | nightstand_3 | 26.8 | 7.9 (29 %) | 7.9 (29 %) | 0.1 s | 0.07 [0.04, 0.10], central 0.16; L -4.00 -> -3.60 (2 rows with L decreasing) |
| c3a_2 | table_17 | 26.1 | 13.7 (52 %) | 13.7 (52 %) | 1.0 s | 0.18 [0.00, 0.65], central 0.45; L 0.00 -> -4.00 (34 rows with L decreasing, 174 of 186 rows npts 0) |
| c3a_3 | chair_106 | 30.0 | 7.0 (23 %) | 0.0 (0 %) | 33.4 s | not derivable |
| c3a_w2_1 | nightstand_5 | 29.4 | 7.3 (25 %) | 7.3 (25 %) | 28.5 s | 0.00 [0.00, 0.38], central 0.00; L -0.69 -> -4.00 (28 rows decreasing) |
| c3a_w2_2 | chair_214 | 7.4 | 7.4 (100 %) | 7.4 (100 %) | 2.1 s | not derivable |
| c3a_w2_2 | chair_216 | 29.0 | 0.7 (2 %) | 0.7 (2 %) | 10.5 s | not derivable |
| c3a_w2_2 | nightstand_7 | 29.0 | 7.9 (27 %) | 7.9 (27 %) | 0.4 s | 0.33 [0.33, 0.34], central 0.68; L -4.00 -> -3.99 (1 row decreasing) |

Totals: 229.2 s of absence over the nine; 70.1 s (31 %) with the object's place in the frustum; **45.4 s (20 %) in view with wall line of sight**. Eight of the nine had the place out of view (wall LOS included) for at least 48 % of their absence ticks (shares out of view: 100, 98, 71, 48, 100, 75, 0, 98, 73 %); only chair_214 was in view for the whole of its short absence (7.4 s after 11.5 s of support in a 19 s life). Where the agent logged it, P(detect) was low (median 0.00-0.33) while the
robot's view of the place was poor (central <= 0.68), consistent with absence weighted by P(detect), not a missing term. Two cases need a note: nightstand_3 and nightstand_7 were removed with L already at -4.00 at the start
of the window (the L at the window start is the clamp), i.e. the evidence that condemned them lies earlier than 30 s (not derivable from these rows); chair_94/_96/_106 had last support 21-33 s before the removal, with the place
out of wall line of sight at most 2 % of the time: the removal there is an absence accrued mostly while NOT looking, which the log of a chair cannot refute or support (no `ex_*` columns).
Caveats: GT place vs belief (all within 0.3 m); yaw is the GT pose of a 1.5 s log, interpolated; furniture occlusion not modelled.
