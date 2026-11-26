# Part 1 grading: shared estimation engine (AuRo tests a–d)

> **Status: GRADED COLUMNS PENDING.** The graded numbers come from the combined rerun batch, which the room session runs from
> one build commit (§6). They are not in this file yet: every "graded" column below says *pending*. What this file does
> contain is the structure and the method, and the preliminary numbers that exist now. Those come from single live runs, one
> scripted visit per class, and several of them were made on earlier commits than the build commit. They are not the graded
> result and must not be quoted as such.

- Branch feat/shared-engine. Build commit for the batch: **91598c88** (code = fix wave 17f13198), tree `2d0c2d5107011e10da553ad8ac569535b7b33c48`.
- Spec: `docs/superpowers/specs/2026-11-23-shared-engine-design.md` (Tests, Graded quantities). Plan revision 2, Task G1x.
- Count (test c): `2026-11-23-patch-constants-baseline.md`, section "After part 1".
- Closed prior-term list: `2026-11-26-closed-prior-list.md`.

## 1. What is graded

| test | statistic (as registered; amended for AuRo on 11-24) | pass |
|---|---|---|
| (a) first-visit calibration | 6-DOF NEES/DOF with the full θ covariance, e = (graded quantity − mesh truth), averaged over **n ≥ 10 independent first visits per class** | mean in [0.5, 2] |
| (b) promise tracks delivery | scripted A/B/A visit per object; promise = the engine's gain logged at the claim; delivered = the NET θ-entropy change over the visit; Spearman over the visits | ρ > 0, with spread in the promises |
| (c) patch-constant count | the baseline's build-derived scope rule, on the same five targets | the registered rows removed (spec table) |
| (d) held-out world | one world not used for tuning | **no world exists and no one is assigned (Fable review risk 10): not gradable in part 1** |

- Graded quantities and ground truth per class are in the spec table "Graded quantities". The bed's h is graded against the
  frame including the headboard (2.09–2.12 m, AuRo A3). The chair's yaw is in the countable convention (+y = backrest). The
  table uses the c2v minimum over group images.
- Part 3 (online noise and envelope learning) is not built, so grading is with part 3 off, as the owner ruled.

## 2. Method

- **Node to object.** `aba_runlog.match_node_to_truth`: posterior Mahalanobis distance of the class node to the truth, χ²₂
  bound 9.21. A class with no node inside the bound is **not matched**; it is reported and it fails.
- **Scorers.**
  - `room_concept/estimation_scripts/scripts/sbs_compare.py` gives the final error per DOF, e/σ, and NEES/DOF over the
    dwells.
  - `aba_report.py` gives the arrival promise against the dwell net, and the claim promise against the visit net.
  - `parked_report.py` gives the parked checks.
  - Truth comes from the mesh extents (`aba_plan_<DEF>.json`, written by the P1 scripts).
- **Preliminary statistics in §3** (single runs):
  - **first-visit NEES/DOF** is the mean of the diagonal NEES/DOF over the A1 dwell, the scripted visit's first look. It was
    computed with sbs_compare's `errors()`; the one-off script is quoted in §7. Its ~600 rows are ONE draw (spec G1:
    rows inside one view are not independent), so each cell is n = 1 visit. It is diagonal: the ai2 log holds only σ per
    DOF, not the θ block.
  - **planar error** is √(e_cx² + e_cy²) at the end of the A1 dwell and at the end of the A2 dwell, in the visit's published
    room frame.
  - **Spearman** is pooled over the matched objects' segments. The claim form (claim promise against visit net) is the
    amended test (b). The arrival form (arrival promise against dwell net) is shown for context.
- **Graded statistics in §3** (batch): as registered in §1. The full-covariance NEES needs the θ block, which the ai2 log does
  not hold. The batch therefore turns on `Model.ReplayDump`, and the replay tool (Task R1, bit-identical with full P) recovers
  it (§6).

## 3. Per-class table

Engine commits of the preliminary runs:
- nightstand, bed, toilet: **f3f60a4f** (before F1 r9, K1–K3 and S1);
- refrigerator, hood: **93f3b35b**;
- chair: **8acac8a5**;
- table: **b336d4d0**.

None of these is the build commit.

| class | run (crossing/) | node match d² (bound 9.21) | first-visit NEES/DOF, A1 dwell, diag (n = 1 visit) | NEES/DOF over all dwells | planar error end A1 / end A2 | registered A/B/A checks | **graded (a) mean NEES/DOF, n ≥ 10** | **graded (b) ρ** | count: patch total (baseline → now) |
|---|---|---|---|---|---|---|---|---|---|
| nightstand | p1r8_nightstand | 1.34 (nightstand_25) | 2.54 | 2.31 | 3.1 / 2.6 cm | 5/6 FAIL (net A2 > net B, a wrong registered inequality) | *pending* | *pending* | 131 → 134 |
| bed | p1r8_bed | 3.81 (bed_11) | 3.09 | 3.49 | 4.7 / 4.3 cm | 6/6 PASS | *pending* | *pending* | 131 → 134 |
| toilet | p1r8_toilet | **none** (toilet_5, d² 126.9) | 29.10 | 40.53 | 23.7 / 24.2 cm | NO VERDICT, FAIL | *pending* | *pending* | 131 → 134 |
| refrigerator | k1_fridge | 3.16 (refrigerator_1) | 2.05 | 2.23 | 5.0 / 2.7 cm | 4/6 FAIL (B: promise 0.97, delivery 0: no detections at B; net A2 > net B) | *pending* | *pending* | 129 → 134 |
| hood | k1_hood | **none** (hood_3, d² 25.2) | no dwell updates | — | 14.2 / 14.2 cm (birth estimate, never updated) | NO VERDICT, FAIL | *pending* | *pending* | 128 → 134 |
| chair | k2b_chair | 3.27 (chair_2) | 1.92 | 3.44 | 3.9 / 4.3 cm | 6/6 PASS | *pending* | *pending* | 110 → 134 |
| table | k2c_table | **none** (table_2, d² 262.6) | 193.8 | 193.8 | 50.5 / 50.5 cm | NO VERDICT, FAIL | **cut**: graded as "old engine" (legacy binary) | *pending* (legacy) | 120 → 121 (legacy) |

Reading the preliminary column:
- **Overconfident.** Every matched class is over 1 at first visit: chair 1.92, fridge 2.05, nightstand 2.54, bed 3.09.
  - Inside [0.5, 2] would be chair only, and with n = 1 that says nothing: the registered statistic needs n ≥ 10.
  - On the batch build, the bed's replay after F1 r9 gave NEES 3.49 → 1.82 and d² 3.81 → 0.04 (replay of p1r8_bed, not a
    live run).
- **Replays are context only.** The bed's F1 r9 numbers and the toilet's K2a-L numbers (§4) are replays of recorded dumps,
  not live runs on the build commit.

### Test (b), preliminary (single runs, pooled)

| pairs | n | Spearman ρ | promise spread |
|---|---|---|---|
| claim promise vs visit net (the amended test) | 10 (nightstand 3, bed 3, refrigerator 2, chair 2: no claim promise at A1 for the K1/K2 runs) | **+0.47** | 0.122 … 1.341 nats |
| arrival promise vs dwell net (context) | 12 (4 objects × A1, B, A2) | −0.06 | 0.003 … 0.971 nats |

- The claim-form ρ is positive on these 10 pairs, which come from four objects and three engine commits.
- The arrival form shows no correlation. Most of the arrival promises are near zero (a parked look at ρ ≈ 1), while the
  dwell nets are not.
- Neither is the graded number.

### Test (c)

**FAILS as registered.** On countable_concept, where six of the seven classes now run, the registered rows break down as:
- removed 0, moved 3, kept 48;
- new rows +48, of which patch +6. Two of those are flagged thresholds, and there is one unflagged threshold:
  `persistent_engine.h:286`.

Patch totals went up for every class, as the last column of the table shows. See "After part 1" in the baseline file. The
closed prior-term list check finds six violating terms in every graded class, four of them not argued in the ledger.

## 4. Cut, failing, ungraded

- **Table: CUT from part 1** (owner ruling 11-26, progress.md:237). The K2c live run failed:
  - no match (d² 263);
  - depth +0.77 m (29.6σ), NEES 194;
  - births square, under the c2v broad prior;
  - the short side was never seen;
  - the depth was pinned by `AI2DepthUnobsPrecision`.

  The diagnosis is recorded in `task-K2c-live-diagnosis.md`. The later remedies were a swap-mixture size prior, unpinning the
  depth, and a top-only solid. The batch runs the **legacy** table (`LEGACY_AGENTS="table_concept"`), and its test (a) is
  reported as "old engine", as the spec allows.
- **Toilet: FAILING.** It ran in P1 as one box; no node matched (d² 127, centre −24 cm).
  - K2a added the bowl and tank as parts, and K2a-L added the proportion latent with honest common modes. Replay at the final
    model: d² 63.6–75.2, h −10σ (progress.md:212).
  - The cause is recorded: hard beam refutation at the part boundaries, plus the mode being chosen at birth.
  - The stand-off is the cited 0.03 ± 0.02 m. This world puts the toilet 17–18 cm off its wall, about 7σ; that is reported,
    not fitted.
  - No live run exists on the build commit. The toilet is expected to fail (a) and to be unmatched in (b).
  - **No spread was widened** (owner rule).
- **Hood: UNGRADED in (b), expected to fail in (a).**
  - In k1_hood the hood was born at the start (hood_3, H 2.40 = the chimney top). The node was dropped when the robot left
    room_1, and it was **never re-born**: there were 0 `hood` detections through the whole A/B/A visit (A at 1.23 m; the top
    is likely out of the ZED's view).
  - The one estimate is H +0.47 m (28.7σ), w −0.56 m, and d² 25.2 (no match).
  - This is a detection gap (an owner item, together with the unregistered `hood_roi_*`), not an engine defect.
  - In the batch tours, which start in the kitchen, the hood may get first visits from the boot view. Those rows are graded if
    a node is matched.
  - The hood's ε_px (0.066) is borrowed from the fridge, not measured.
- **Refrigerator (b) at B: the promise ignores P(detect | standpoint).** In k1_fridge B the promise was 0.97 nats, but YOLO
  gave 0 detections from the 45° side view, so the delivery was 0. This is the promise/detection-envelope gap (part 3
  territory). A B-type check fails where the detector is blind.
- **Chair:** its ε_px (0.066) is also borrowed from the fridge, and its yaw is graded in the countable convention.

## 5. Registered checks that already fail (reported as failing, no bound widened)

1. **The "net A2 ≤ net B" inequality (P1 predictions).** It fails for the nightstand and the refrigerator. P1's results doc
   shows the registered model itself predicts the opposite order under the logged ρ: the registered prediction was wrong.
   Recorded, and not re-registered.
2. **The parked yaw level checks.** The yaw σ is below the per-DOF model's lower bound for the nightstand and the bed. This
   is a registered omission: there are no room-axis or wall terms in the test model.
3. **Toilet, hood and table: no node matched.** They FAIL every check that needs a node.
4. **Test (c).** It fails (§3).
5. **Test (d).** It cannot be run: there is no held-out world.

## 6. Batch request (for the room session; the controller sends it)

```
BATCH REQUEST — shared-engine part 1 grading (G1x, 2026-11-26)

BUILD
  feat/shared-engine @ 91598c88 (tree 2d0c2d5107011e10da553ad8ac569535b7b33c48; code = fix wave 17f13198, 91598c88 adds docs only). Any later docs-only commit is the same code.
  Pin a worktree at that commit (as k1-live / k2b-live) and build there with -j8 (check `pgrep -a cc1plus` first):
    countable_concept, controller (K3 moved its planner kernels into common/standpoint — the controller MUST be built
    from this commit too), table_concept (legacy, for the cut table); the rest of the fleet as usual.
  Cortex: installed from development >= 54c6508 (size_std_m, Task S1).
CLASSES
  nightstand, bed, toilet, refrigerator, hood, chair on the countable binary (the default links <cls>/countable/<cls>);
  table on its LEGACY binary: LEGACY_AGENTS="table_concept" on every run below (else concept_agents.sh launches K2c's link).
  CONCEPT_AGENTS="nightstand_concept bed_concept toilet_concept refrigerator_concept hood_concept chair_concept table_concept"
  CONCEPT_WT=<the pinned worktree>. Part 3 is not built (nothing to switch off).
PART A — test (a), first visits: 12 tours, the graded start set (main bd1dfa70:
  room_concept/estimation_scripts/auro_1122/graded_starts.tsv / .md, S01..S12, all start in room_1, every route visits
  room_2 and room_3 and ends in room_1). Per start SNN:
    set Model.ReplayDump = true in every countable config of CONCEPT_WT for the run and restore it after (exactly as
    parked_test.sh step 1 does) — the ai2 logs hold only per-DOF sigma; the full theta covariance for the registered
    6-DOF NEES comes from the replay dumps (Task R1 replay tool, bit-identical, full P). Disk: ~0.6 MB per bed update —
    expect GBs per tour.
    START="<x y yaw of SNN>" START_ROOM=room_1 LEGS="<legs of SNN>" \
    ON_FLEET_UP="bash $CONCEPT_WT/room_concept/estimation_scripts/scripts/concept_agents.sh start <out>" \
      main's room_concept/estimation_scripts/scripts/warm_start_tour.sh g1x_tour_SNN crossing/fixtures/three_rooms_door0
    then concept_agents.sh stop; concept_agents.sh harvest <out>; move the run's replay dumps into <out>/replay/.
  Output: room_concept/estimation_scripts/crossing/g1x_tour_S01 … g1x_tour_S12 (with replay/).
  Scoring: NOT yet scriptable as registered — no scorer computes 6-DOF full-covariance first-visit NEES over tours for all
  seven classes (tools/step0/nees_persistent.py covers nightstand/bed/toilet, cx/cy only, diagonal). The runs are recorded
  now; the scorer is written after (it reads ai2 logs + replay dumps; it must not be tuned on these runs' results).
PART B — test (b), scripted A/B/A: 6 objects x 2 repeats = 12 runs, from CONCEPT_WT:
    LEGACY_AGENTS="table_concept" DWELL_S=60 \
      $CONCEPT_WT/room_concept/estimation_scripts/scripts/parked_test.sh g1x_aba_<obj>_<k> <DEF>
    <DEF> in NIGHTSTAND_R, BED_1, TOILET_1, FRIDGE_0, HOOD_0, CHAIR_5; k = 1, 2 (<obj> = nightstand, bed, toilet, fridge,
    hood, chair). parked_test.sh turns ReplayDump on, runs aba_visit.sh, parked_report.py and aba_report.py itself.
  Output: crossing/g1x_aba_<obj>_<k>/{aba_report.txt, parked_report.txt, replay/, *_ai2_log.csv, *_gain_query.csv, ...}.
  Scores (CPU, after the runs, from CONCEPT_WT's scripts):
    python3 -I sbs_compare.py crossing/g1x_aba_*  > crossing/g1x_aba_scores.txt
    for r in crossing/g1x_aba_*; do python3 -I aba_report.py $r; done > crossing/g1x_aba_reports.txt
PART C — C3 non-regression (spec "C3 non-regression"): the C3 moved/hidden-object tours on this build, reported against the
  v1 numbers (8a133295), with the room/auro session's own C3 scripts (not specified here).
RESULTS
  Paths go to the auro session: docs/superpowers/results/2026-11-26-part1-grading.md (this file; its pending columns are
  filled from the runs above), the 24 run folders, crossing/g1x_aba_scores.txt, crossing/g1x_aba_reports.txt.
```

## 7. Reproducing the preliminary numbers

All CPU, from the worktree's `room_concept/estimation_scripts/scripts/`, with
`C=/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing`:
- `python3 sbs_compare.py $C/p1r8_nightstand $C/p1r8_bed $C/p1r8_toilet $C/k1_fridge $C/k1_hood $C/k2b_chair $C/k2c_table`
  gives the match, the final e/σ, the all-dwell NEES and the events.
- `python3 aba_report.py $C/<run>` gives the promise and net rows, from which ρ is computed.
- The first-visit (A1-dwell) NEES and the planar errors came from a one-off script, not committed. Per run it uses:
  - `aba_runlog.segments`, `plan_of`, `ai2_rows`, `match_node_to_truth`, `updates` and `last_before`;
  - `sbs_compare.errors`.

  It averages Σ_k (e_k/σ_k)²/6 over the updates with `t_dwell0_ms ≤ t ≤ t_dwell1_ms` of segment A1, and takes the planar
  error at the last update before A1's and A2's `t_dwell1_ms`.
- Spearman uses average ranks for ties. The claim pairs are (claim promise, visit net) for each segment that has a claim
  promise.
