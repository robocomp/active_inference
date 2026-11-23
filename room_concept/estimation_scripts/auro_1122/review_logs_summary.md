# Log-only review analyses, 2026-11-23

Runs: `c3a_2 c3a_3 c3b_on_hidden1 c3b_off_hidden1 c2_w2 c2_w3` (the six "tab:arbitration" tours), under
`room_concept/estimation_scripts/crossing/<run>/`. Nothing was run, started or stopped; no existing file was edited.
Analysis code (new, scratch only): `auro_1122/scripts/review/`
(`t1.py t1c.py t2.py t2b.py t2c.py t3.py t4.py t4b.py t5.py`, outputs `t*_out.txt`).
Ground-truth world for the old runs: `git show HEAD:worlds/piso/threerooms-piso.wbt` of webots-shadow (d7ec96d, 2026-11-21) saved in the scratch `world/`
with a `meshes` symlink; all six runs date from 2026-11-21 (13:37 to 23:25), before the owner's 19:26 world of 11-22.
The `c3a_w2_*` runs (11-22) were graded against the working-tree world, with `C3A_POSE_OVERRIDE` for chair(3).

**clean62..clean76 exist**, at `crossing/clean62 ... crossing/clean76` (also clean1..clean61). They have only door/room/ltsm/residual/retina/controller logs
(no object-agent logs, no `legs.tsv`, no `gt_error` object products), so they enter only the room-identity and controller counts of Task 2 (separate table there).

## Task 1. Room scale and the 5-nat floor

**Code (what is logged).** `affordance_manager.cpp:931-945` sets `object_competing` when any affordance whose `target->parent_node_type != "room"` is Offered or Executing
(a door offer would also count, if doors are not parented to "room"; not checked). `:962-968`: only room-parented, non-`Orient` targets have their GAIN multiplied by
`select_room_gain_scale_` (0.35, `controller/etc/config.toml:555` AffordanceRoomGainScale; effective csv confirms 0.35); `score = gain' - 0.2*dist (+0.5 incumbent)` (`affordance_manager.h:398-399`).
`:994-995` logs `target->epistemic_gain` and `neg_efe(*target)`, so **the logged `gain=` is PRE-scale and the logged `score=` is POST-scale.**
No-scale score of the room offer is therefore `s + 0.65*g` (the distance and incumbent terms stay).
The 5-nat floor is the verification floor `epistemic_step.h:136-137` (`prop.epistemic_gain = max(gain, verify_gain_floor)`), `ExistenceVerifyGain = 5.0` in
hood/countable (bed, nightstand, toilet)/cabinet/table/refrigerator configs (e.g. `hood_config.h:337`). `chair_concept` and `bottle_concept` do not use it (chair offers are priced by the planner: 3.185 nats typical).

**Method.** Re-parsed all `[aff-select] ->` lines of `logs/controller.log` at full precision with the regex of `scripts/aff_arbitration.py`; contested = at least 2 concepts among the Offered/Executing candidates plus the winner.
(`t1.py`, `t1_out.txt`).

**Counts (6 runs).**
- 212 selections; 133 contested; object wins in contested = 78 (= all object wins); **object beat room 77; room beat object 6** (also 15 door wins and 40 room wins in contested). These match the paper's 78 / 77 / 6.
- Recomputing the argmax of the logged scores reproduces the logged winner in 132 of 133 contested rows (exception: c2_w2 controller.log line 4125, logged winner aff_chair_114 at score 0.39 while aff_nightstand_8 4.18 was eligible; a rule outside the plain argmax acted, not investigated).
- **Winners priced exactly at the floor (gain == 5.0): 41 of 78 object wins** (nightstand 18, table 9, bed 7, hood 4, toilet 2, refrigerator 1). 40 of those 41 beat the room offer.
- **Room scale removed** (room score + 0.65*g, same candidate set, same refused-skip rule): winners change in **43 of 133 contested choices**: 36 object-to-room, 7 door-to-room (no other kind of change). Object-beat-room goes 77 to **41**; room-beat-object goes 6 to **44**; object winners 78 to 42.
  This is a static counterfactual of each selection: it does not replay what the robot would then have done (later states, incumbent margins and the visits would differ).
- **Floor removed.** A raw pre-floor gain IS logged, but only in each concept's `<cls>_ai2_log.csv` (`nbv_gain_raw` beside `nbv_gain_pub`; `epistemic_step.h:132-146`, written by `record(prop, raw_gain)`), and only on cycles in which the concept processed a packet (the object was observed). The controller log has no raw value.
  Of the 41 floor-priced winners the most recent logged raw is within 5 s of the selection for 4, and exists at any earlier time for 25 (lag median 10.7 s, p90 29.7 s over all candidate lookups); **16 have no row before the selection (not derivable).**
  Among the 25: raw gain median 0.0 (20 are 0.0), max 3.37, none >= 5. The raw value is stale by design (the floor applies while an object is not observable), so treat it as an indicative bound, not a measurement of the gain at that instant.
  Counterfactual (floor candidates re-priced at their last logged raw; rows where every floor-priced candidate has one: 27 of the 44 contested rows holding a floor-priced candidate; the other 17 are not derivable): the winner changes in 20 of the 27 (11 object-to-room, 6 object-to-door, 3 object-to-object).
- Caveat: with the 0.35 scale and the 5-nat floor both removed together is not computed (not derivable for 16 floor winners and it would stack two static counterfactuals).

## Task 2. Decision-level conformance counts (6 runs)

Source patterns: births `[tracker] BIRTH '<node>'` in `<agent>_concept.log` (excludes `BIRTH SUPPRESSED` and `[birth-fusion]` lines); removals as below; room identity from `ltsm_agent.log`; controller from `aff_arbitration.csv` and `controller.log`.
(`t2.py`, `t2b.py`, `t2c.py`.)

| quantity | count | log text / source |
|---|---|---|
| births (tracker count/evidence rule) | **94** (chair 48, nightstand 13, table 11, bed 9, toilet 6, refrigerator 4, hood 3, bottle 0) | `[tracker] BIRTH '`; 41 of them took a remembered uid (`[tracker] RESURRECT`, printed with the BIRTH) |
| removal by existence / absence channel | **7** (chair 6, table 1) | `[existence] REMOVE id=` (chair; `chair_existence.h`), `[existence] removing table id=` (`table_existence.cpp:553`; countable twin `countable_existence.cpp:542`) |
| removal by tracker miss-count death | **0** | `[tracker] DEATH id=` (bottle only, `bottle_concept/src/specificworker.cpp:791`) never appears; no `deaths=N>0` in any `[tracker] instances=` status line |
| removal "shape argued against" | 0 | `[<cls>-class] removing id=` (`countable_concept/src/specificworker_lifecycle.cpp:624`) |
| room-exit cleanup | **64** affordance nodes `[affordance] removed node for '<node>' (state=idle)` inside a leg window `[t0-1 s, t1+1 s]` of `legs.tsv` | the 3 further `removed node` lines outside a leg coincide to the millisecond with an existence removal (its own affordance teardown) |
| room-identity decisions | **28 promotions, 28 by passage lookup, 0 by the recogniser** | `[promotion] PROMOTING` (`ltsm_agent/src/specificworker.cpp:686`), each preceded by `[passage] passage_N match door_X <-> door_Y` and `[recog] room_N not eligible: the crossed door ... is a passage in memory` (`room_recognizer.cpp:299`); `[recog] SWAP requested` (`room_recognizer.cpp:828`) = 0 |
| controller selections | **212**: 133 contested (>= 2 concepts eligible) vs 79 single-concept; by `n_eligible`: 76 with exactly one eligible entrant, 136 with >= 2 | `aff_arbitration.csv` |
| of which pragmatic door steps winning | 68 | `winner_kind in door_open/close/cross/approach` |
| mission steps (external orders, not EFE choices) | **118**: explore 34, open 28, cross 28, close 28 | `[mission] step N:<verb> ...: started` |

Per run (births / resurrected / existence removals / exit cleanups in leg / promotions = passage lookups / recogniser swaps / contested vs single selections):
c3a_2 20/8/1/13/4/0 and 27/11; c3a_3 20/5/1/14/4/0 and 35/11; c3b_on_hidden1 7/6/0/4/4/0 and 16/11; c3b_off_hidden1 8/6/2/0/4/0 and 5/13; c2_w2 19/8/1/16/6/0 and 27/17; c2_w3 20/8/2/17/6/0 and 23/16.
Caveats: cleanup of an affordance node is not the death of the belief (the instance is archived at the owner exit; c3a_instances `last_end=owner_exit`); the c3b_off arm publishes no affordance nodes (EpistemicOfferEnabled=false), hence 0 cleanups. The in-leg window is a proxy for "at a room exit" (the leg also contains the approach).
"Contested" here counts concepts (door and room are concepts), so a room-vs-door choice counts as contested.

**clean62..clean76 (15 runs, no object agents, same patterns):** 43 promotions, 43 `[passage] ... match` lookups (all promotions again by passage), **76 `[recog] SWAP requested`, 38 `[recog] swap to room_id ... executed`, 41 `episode begins`**, only 2 `not eligible: ... passage in memory`;
336 aff-selects (78 contested, 258 single-concept; 192 pragmatic door wins). So the recogniser path is exercised in the clean series and not in the six object tours. (`c_t2b_out.txt`)

## Task 3. Promised versus delivered information, claimed object offers

**Method.** Claim = every `[aff-select] -> 'aff_<cls>_<N>'` with an object parent (78 claims, controller line to seconds via the `gt_pose.tsv` 1 Hz controller line counter, 1 s resolution). Visit window = claim to the next `[aff-select]` line of any winner (3 claims end at the end of the run).
Promised = logged `gain=` at the claim (pre-scale, includes the floor). Delivered = 0.5*ln(det S_before / det S_after), S = diag(std_cx^2, std_cy^2), from the last `<cls>_ai2_log.csv` row at or before the claim and the last row at or before the window end (these logs write a row per processed packet, so no row in the window means no update, delivered = 0.0 exactly).
Planar error: belief (cx,cy) to world with `run_frames.make_to_world`; GT = `gt_object` of `c3a_instances.csv` where the run has it (c3a_2, c3a_3, c3b_on_hidden1), else nearest same-class GT object within 1 m of the end-of-window belief (the moved chair(3) uses its post-move pose after the move time in `chair_move.txt`). (`t3.py`, `t3_out.txt`, `t3_rows.json`.)

**Not derivable for 32 of the 78 claims:** `c2_w2` and `c2_w3` have no `<cls>_ai2_log.csv` (only the combined `ai2_log.csv`, which is the room agent's), so there is no per-object sigma. 46 claims remain (c3a_2, c3a_3, c3b_on_hidden1, c3b_off_hidden1); of these 32 have at least one belief update in the window and 14 have none.

Median [Q1, Q3] per class (n = claims with data; error change = error after minus before in m, negative = better; n for error in brackets):

| class | n | promised (nats) | delivered (nats) | change in planar error (m) |
|---|---|---|---|---|
| all | 46 | 5.00 [3.19, 5.00] | 0.00 [0.00, 2.35] | 0.000 [-0.007, 0.000] (45) |
| chair | 16 | 3.19 [2.84, 3.19] | 1.36 [0.00, 4.80] | 0.000 [-0.045, 0.000] (15) |
| nightstand | 10 | 5.00 [5.00, 5.00] | 0.13 [0.00, 0.45] | -0.0001 [-0.019, 0.000] (10) |
| table | 7 | 5.00 [5.00, 5.89] | 0.56 [0.08, 2.61] | 0.000 [-0.004, 0.016] (7) |
| bed | 4 | 5.00 [5.00, 5.00] | 0.00 [0.00, 0.00] | 0.000 (4) |
| hood | 4 | 5.00 [5.00, 5.00] | -0.12 [-0.17, -0.08] | 0.000 (4) |
| refrigerator | 3 | 5.50 [5.25, 5.84] | 1.94 [0.97, 3.99] | 0.000 [-0.004, 0.013] (3) |
| toilet | 2 | 5.00 | 0.00 | 0.000 (2) |

Spearman, promised vs delivered: all rho = -0.064 (p 0.67, n 46); chair +0.45 (p 0.080, n 16); table -0.08 (p 0.87, n 7); other classes have too few distinct promised values (the floor makes them constant). Promised vs error reduction: all -0.17 (p 0.26, n 45), chair +0.20 (p 0.49, n 15), table +0.17 (p 0.72, n 7).
Floor-priced claims (promised exactly 5.0, n 25): delivered median 0.00 [0.00, 0.48]; claims below the floor (n 17): delivered 0.16 [0.00, 4.73]. Claims with a belief update only (n 32): promised 4.09 [3.19, 5.00], delivered 0.51 [0.00, 2.76], rho +0.09 (p 0.64). Window median 11.1 s. 7 claims delivered a negative value (sigma grew; the hood class median is -0.12), 19 exactly 0.0, 1 claim without a matched GT object.
Caveats: the window end (next selection) is my definition, 1 s time resolution; log rows come from the packet time (`pkt_ts`) for non-chair agents; the floor-priced "promise" of 5 nats is a verification priority, not a planner estimate, so comparing it to delivered information measures the floor, not the planner. The error changes are tiny because most beliefs were already near the object.

## Task 4. Removals by the existence / absence channel in the c3a tours

No `removed.txt` or `chair_move.txt` exists in any c3a_* folder, and no mover script was recorded, so no object was taken away or moved in c3a_1, c3a_2, c3a_3, c3a_cab1, c3a_w2_1..3 (c3a_w2 summary: CHAIR_3 left at (-7.43, 8.51) all three runs, the override used in grading).
Times are seconds since the first `gt_pose.tsv` row; world position = last logged belief row of the node mapped with `make_to_world` (lag in s between that row and the removal); "gt" = nearest same-class object of the run's world. (`t4.py`, `t4b.py`, `t4_out.txt`.)

| run | t (s) | agent | node | cause | world pos (x, y), row lag | gt | note |
|---|---|---|---|---|---|---|---|
| c3a_1 | 52.0 | chair | chair_94 | existence | (-5.91, 1.99), 29.9 s | chair(2) 0.23 m | real object, removed while present |
| c3a_1 | 114.3 | chair | chair_96 | existence | (-5.85, 2.00), 21.1 s | chair(2) 0.29 m | real, removed during leg 1 (`deaths_at_exit`) |
| c3a_1 | 703.4 | chair | chair_98 | existence | (0.85, 9.52), 82.8 s | none (7.4 m) | phantom per `c3a_instances.csv` |
| c3a_1 | 331.9 | nightstand | nightstand_3 | existence | (-4.04, 13.24), 0.1 s | nightstand_bed1_right 0.04 m | real, removed while present |
| c3a_1 | 321.3 | table | table_14 | existence | (-4.02, 12.99), 46.8 s | wrong class (nightstand 0.23 m) | `wrong_class` |
| c3a_2 | 593.3 | table | table_17 | existence | (-1.91, 4.43), 0.9 s | Table1 0.08 m | real, removed while present |
| c3a_3 | 60.6 | chair | chair_106 | existence | (-5.91, 2.01), 33.4 s | chair(2) 0.24 m | real, removed while present |
| c3a_cab1 | none | | | | | | no removal |
| c3a_w2_1 | 196.0 | nightstand | nightstand_5 | existence | (-4.10, 13.24), 28.5 s | nightstand_bed1_right 0.10 m | real |
| c3a_w2_2 | 70.0 | chair | chair_214 | existence | (-1.29, 2.40), 2.0 s | chair(1) 0.17 m | real |
| c3a_w2_2 | 628.8 | chair | chair_216 | existence | (-7.27, 8.52), 10.5 s | chair(3) 0.16 m | real, during leg 4 |
| c3a_w2_2 | 261.6 | nightstand | nightstand_7 | existence | (-4.11, 13.24), 0.4 s | nightstand_bed1_right 0.11 m | real |
| c3a_w2_2 | 135.9 | bottle | bottle_29 | existence | not derivable (no bottle ai2 log); birth pos room (-2.30, 1.07) = world about (-6.54, 5.70) | GT bottle (0.44, -0.58) is 9.4 m away | removal of the first of about 26-28 births logged at the same room-frame spot in that run (a birth cluster); the bottle itself was never seen |
| c3a_w2_3 | 58.1 | bottle | bottle_55 | existence | same as above (world about (-6.55, 5.71)) | 9.4 m | same |

Totals: **13 existence removals** in the 7 c3a runs (c3a_1 5, c3a_2 1, c3a_3 1, c3a_cab1 0, c3a_w2_1 1, c3a_w2_2 4, c3a_w2_3 1; 0 tracker-miss deaths, 0 shape removals).
By GT: 9 removals of a node sitting on a real, unmoved object (chair 5, nightstand 3, table 1), 1 wrong-class table, 1 phantom chair, 2 bottle phantoms. So no object that is really gone was ever removed, because none was removed from the world; 9 of 13 removals delete a belief about an object that was still there.
`c3a_instances.csv` (c3a_1..3) agrees on every node it covers: `deaths_present=1` for chair_94, nightstand_3 (c3a_1), table_14, table_17, chair_106; chair_96 is `deaths_at_exit=1`, chair_98 `phantom`.
The belief rows are 21-33 s old for chair_94, chair_96 and chair_106, i.e. the robot had not been logging that chair for that long before the removal. Whether the removed node was a duplicate of another live instance of the same object (chair_94 and chair_96 both match chair(2)) was not checked row by row.

**Bottle's removal path.** The CONCEPT_AGENT_LIFECYCLE.md audit row (line 231: "tracker death only", line 424: "no existence channel") is STALE for the 11-22 builds. `bottle_concept` now has an existence channel: `BottleModel.ExistenceEnabled` default true (`bottle_config.cpp:131-132`, `bottle_config.h:95-104`), the channel is the only removal path for absence and `Tracker.DeathFrames` is disabled while it is on (`bottle_concept/src/specificworker.cpp:468-473`, removal at `:493-500`; the config line `Tracker.DeathFrames = 90` is still printed in the log but unused). Evidence: `[existence] REMOVE id=... (log-odds -4.00 < -1.99; p_detect 0.19 ...)` in c3a_w2_2 and c3a_w2_3 bottle logs, zero `[tracker] DEATH` lines in any c3a run. Without the channel (ExistenceEnabled=false) a bottle dies by the miss counter `[tracker] DEATH id=` after `Tracker.DeathFrames=90` unsupported frames (`:791`), and a second path `[tracker] RETIRE-DIVERGED` on clutter (`:591-601`). In c3a_1, c3a_2, c3a_3, c3a_cab1 the bottle agent logged no birth at all, so there is nothing to remove.

## Task 5. c3b_off_hidden1 (Fig. 13), moved chair after the return

`scripts/c3b_report.py crossing/c3b_off_hidden1` (with `WBT_PATH` set to the committed world): `first_seen_s` is **empty: the chair at its new spot (-6.35, 8.60) was never in view** after the return (view rule of the script: 4 m, +-55 deg, no wall in between), `corrected=0`.
Return to room_2 = leg 3 `t1_ms` 1795266157725; leaving = leg 4 `t0_ms` 1795266250537; **return to leave = 92.8 s**. Therefore "seconds between the first sighting and leaving" does not exist (no sighting).
Geometry between the return and the leave (`t5.py`): minimum distance of the robot to the new spot 3.31 m, never within 2 m; 7 samples within 4 m, 1 of them inside the +-55 deg cone but wall-blocked, 0 in view. The robot first came within 2 m of the new spot at +115.7 s after the return (22.9 s after the leave order), 14 samples to +184.7 s, all 14 wall-blocked (the robot was on its way to room_1), 16.2 s in total, equal in kind to the script's `near_2m_s=15.6` (the script counts from the return to the end of the run, i.e. after leaving). At the leave order the robot was 4.38 m from the spot.
Contrast, same script on `c3b_on_hidden1`: first seen 7.4 s after the return, corrected at 25.8 s by its own offer (aff_chair_80), 18.4 s from first sighting to correction.
Caveat: the "chair moved" time is 1795266033932 (during leg 2 to leg 3 gap) from `c3b_timeline.py`; `chair_move.txt` gives the old pose (-7.43, 8.51) and the read-back new pose (-6.35, 8.60), a 1.08 m move.
