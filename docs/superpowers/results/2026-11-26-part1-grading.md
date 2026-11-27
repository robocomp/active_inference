# Part 1 grading: shared estimation engine (AuRo tests a–d)

> **Status: GRADED (Parts A, B and C of the batch, 2026-11-26 21:29 – 11-27 01:30, room session, g1x-live @ 91598c88).** The graded columns of §3 and the section "Graded results" below are the
> result; the other columns are the earlier single runs, kept for context.
>
> **Registry.** Batch 1 is graded against `REGISTERED_TESTS.md` in the AuRo paper folder, the registry the owner approved
> (registered 2026-11-24, before the batch; graded there by the auro session on 2026-11-27, section "Grading of graded batch
> 1"). Where this file differs from it, the registry wins. The two differences:
> - **(a)** The registry's pass band is the two-sided 95 % χ² interval for 6n DOF (n = 10: [0.675, 1.388]), narrower than the
>   [0.5, 2] used below; the verdicts are the same. A class with n < 10 is "not graded (n = k)", not failed. The registry also
>   registers a **last-visit** statistic with the same rule; it is computed in "Test (a), last visit" below.
> - **(b)** The registry's test is the median ratio of promise to delivery in [0.5, 2] plus a pooled slope (delivery on
>   promise) whose 95 % interval contains 1, applied only to classes that pass (a). No class passes (a), so **(b) is not
>   graded**. The Spearman ρ below (from the spec's table, §1) is reported, not graded.

- Branch feat/shared-engine. Build commit for the batch: **91598c88** (code = fix wave 17f13198), tree `2d0c2d5107011e10da553ad8ac569535b7b33c48`.
- Spec: `docs/superpowers/specs/2026-11-23-shared-engine-design.md` (Tests, Graded quantities). Plan revision 2, Task G1x.
- Count (test c): `2026-11-23-patch-constants-baseline.md`, section "After part 1".
- Closed prior-term list: `2026-11-26-closed-prior-list.md`.

## 1. What is graded

| test | statistic (as registered; amended for AuRo on 11-24) | pass |
|---|---|---|
| (a) first-visit calibration | 6-DOF NEES/DOF with the full θ covariance, e = (graded quantity − mesh truth), averaged over **n ≥ 10 independent first visits per class** | mean in [0.5, 2] |
| (b) promise tracks delivery | scripted A/B/A visit per object; promise = the engine's gain logged at the claim; delivered = the NET θ-entropy change over the visit; Spearman over the visits (spec form; **the registry's form is the median ratio + pooled slope, only for classes passing (a): see Registry above**) | ρ > 0, with spread in the promises |
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
| nightstand | p1r8_nightstand | 1.34 (nightstand_25) | 2.54 | 2.31 | 3.1 / 2.6 cm | 5/6 FAIL (net A2 > net B, a wrong registered inequality) | **4.51** [2.45, 6.98], n 17/24 (6 unmatched) FAIL | pooled **+0.36** | 131 → 134 |
| bed | p1r8_bed | 3.81 (bed_11) | 3.09 | 3.49 | 4.7 / 4.3 cm | 6/6 PASS | **3.06** [2.17, 4.07], n 11/12 FAIL | pooled **+0.36** | 131 → 134 |
| toilet | p1r8_toilet | **none** (toilet_5, d² 126.9) | 29.10 | 40.53 | 23.7 / 24.2 cm | NO VERDICT, FAIL | **4.09** [2.69, 6.39], n 9/12 (3 unmatched) FAIL | unmatched 2/2 (not in ρ) | 131 → 134 |
| refrigerator | k1_fridge | 3.16 (refrigerator_1) | 2.05 | 2.23 | 5.0 / 2.7 cm | 4/6 FAIL (B: promise 0.97, delivery 0: no detections at B; net A2 > net B) | **3.27** [1.75, 5.87], n 4/12 (8 unmatched) FAIL | pooled **+0.36** | 129 → 134 |
| hood | k1_hood | **none** (hood_3, d² 25.2) | no dwell updates | — | 14.2 / 14.2 cm (birth estimate, never updated) | NO VERDICT, FAIL | **2.20**, n 1/12 (11 unmatched) FAIL | unmatched 2/2 (not in ρ) | 128 → 134 |
| chair | k2b_chair | 3.27 (chair_2) | 1.92 | 3.44 | 3.9 / 4.3 cm | 6/6 PASS | **6751** [213, 16049], n 30/72 (41 unmatched) FAIL | pooled **+0.36** | 110 → 134 |
| table | k2c_table | **none** (table_2, d² 262.6) | 193.8 | 193.8 | 50.5 / 50.5 cm | NO VERDICT, FAIL | **cut**; old engine, diagonal: **2.33** [0.88, 3.78], n 2/24 (22 unmatched) FAIL | not run (cut) | 120 → 121 (legacy) |

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

## Graded results (batch)

Runs: `crossing/g1x_tour_S01..S12` (Part A, 12/12 tours, all legs) and `crossing/g1x_aba_<obj>_<k>` (Part B, 12/12 runs).
Every run's `resolve.txt` = 6 countable + legacy table from g1x-live. Scores: `crossing/g1x_tour_nees.txt`
(first_visit_nees.py, f2d456a3, written and reviewed before any batch result was seen), `crossing/g1x_aba_scores.txt`,
`crossing/g1x_aba_reports.txt`.

**Scoring note.** The first scoring pass used g1x-live's replay binary, built from 91598c88, which predates the opt-in
`--theta-cov` output. Every engine visit then read "no full P", misreported as "the dump has 0 updates". The score above
uses the shared-engine worktree's replay binary: between 91598c88 and the scorer commit the only non-script change is
`countable_concept/tools/replay/replay.cpp` (the flag), and every graded draw replays bit-identically (dθ vs ai2 ≤ 5e-6,
dP_live = 0). No definition changed between the passes.

### Test (a): FAILS for every class

| class | visits | matched | graded | mean NEES/DOF | 95 % CI | per-DOF mean e²/P_kk (cx cy H w h yaw) | frame fit² vs P_cx, P_cy (report only) |
|---|---|---|---|---|---|---|---|
| nightstand | 24 | 18 | 17 | 4.51 | 2.45–6.98 | see g1x_tour_nees.txt | 5.9e-4 vs 2.5e-3, 1.1e-3 |
| bed | 12 | 12 | 11 | 3.06 | 2.17–4.07 | (h inside the interval contributes 0) | 5.4e-4 vs 1.7e-3, 6.2e-3 |
| toilet | 12 | 9 | 9 | 4.09 | 2.69–6.39 | | 2.5e-3 vs 6.4e-4, 1.7e-3 |
| refrigerator | 12 | 4 | 4 | 3.27 | 1.75–5.87 | | 1.3e-3 vs 5.6e-4, 6.7e-4 |
| hood | 12 | 1 | 1 | 2.20 | — | | 2.1e-3 vs 5.7e-2, 5.1e-2 |
| chair | 72 | 31 | 30 | 6751 | 213–16049 | 1.44 4.09 36.0 10.4 19.0 40384 | 1.2e-3 vs 1.9e-2, 2.1e-2 |
| table (old engine, diagonal) | 24 | 2 | 2 | 2.33 | 0.88–3.78 | | 1.3e-3 vs 1.4e-3, 1.7e-3 |

- **Overconfident by ~3–4.5×** where the class is born reliably (bed, nightstand, toilet, fridge). Same direction and size as
  the single runs. This is what removing the range-growing camera common mode (T5 ruling, I3) risked; restoring it is the
  open owner decision.
- **Chair: the yaw carries the mean.** 5 of 30 visits are above 1000 (6 above 100) (yaw e²/P ~ 4e4 on average), consistent with a confident
  heading in the wrong mode (≈180°); the remaining visits are mostly 3–40, still overconfident. Not yet diagnosed.
- **Births are the second failure.** Fridge born in 4/12 tours, hood 1/12 matched, table 2/24, chair 41/72 unmatched (incl.
  the two sunken chairs, graded as in the world).
- **Frame-fit error** (the room-frame transform's own error, not in P) is small against P for bed, nightstand and chair;
  comparable to P for the toilet and the fridge. It does not change any verdict.

### Test (a), last visit (registered in REGISTERED_TESTS.md 2026-11-24, A1; computed 2026-11-27): FAILS for every class

Same runs, same scorer (`first_visit_nees.py` at f2d456a3, unchanged), same matching, the same bit-identity gate. The only
difference is the visit window: the robot's LAST stay in the object's room, not its first
(`scripts/last_visit_nees.py`, which imports the scorer and substitutes that one function). The draw is the node's last
update up to the end of that stay. For an object whose room the route enters once, the last visit is the first visit.
Output: `crossing/g1x_tour_last_nees.txt`. The replay binary is the pre-F1 BASE build, and it reproduces the committed first-visit
table line for line. The registry's pass band is the two-sided 95 % χ² interval for 6n DOF, given per class.

| class | visits | matched | graded n | mean NEES/DOF (last) | 95 % CI | registry band for n | verdict (registry) | first visit, for comparison |
|---|---|---|---|---|---|---|---|---|
| nightstand | 24 | 18 | 17 | 4.46 | 2.40–6.92 | [0.745, 1.293] | fail (over-confident) | 4.51 |
| bed | 12 | 12 | 11 | 2.62 | 1.82–3.59 | [0.688, 1.369] | fail (over-confident) | 3.06 |
| toilet | 12 | 9 | 9 | 3.92 | 2.43–6.31 | — | not graded (n = 9) | 4.09 |
| refrigerator | 12 | 3 | 3 | 1.96 | 1.04–3.01 | — | not graded (n = 3) | 3.27 (n = 4) |
| hood | 12 | 2 | 2 | 1.63 | 1.19–2.07 | — | not graded (n = 2) | 2.20 (n = 1) |
| chair | 72 | 36 | 35 | 3442 | 106–9546 | [0.818, 1.200] | fail (yaw: e²/P_yaw 2.1e4) | 6751 (n = 30) |
| table (old engine, diagonal) | 24 | 2 | 2 | 2.00 | 0.61–3.40 | — | not graded (n = 2) | 2.33 |

- At the last visit the over-confidence is the same size as at the first: nightstand 4.5, bed 2.6, toilet 3.9. More updates
  do not bring the estimate back to calibration. The chair is again carried by its yaw mode error.
- Fridge and hood: the matched instances change between the first and the last stay (a node born or re-matched later),
  so these n are not the first-visit instances.

### Test (b): NOT GRADED under REGISTERED_TESTS.md (no class passes (a)); spec-form Spearman reported, not graded

Pairs (claim promise, visit net) over the matched objects' segments: **n = 17, Spearman ρ = +0.36**, promises 0.107–1.343
nats (bed 6, chair 4, refrigerator 4, nightstand 3). Not in ρ: toilet 2/2 and hood 2/2 unmatched (NO VERDICT), and
nightstand_1 (NO VERDICT: arrival at B 0.307 m from the standpoint, tolerance 0.25 m). The largest discordant pair is the
fridge's 1.343-nat promise that delivered 0.136 (the P(detect) gap of §4).

Registered A/B/A checks: bed 6/6, 6/6 PASS; chair 6/6, 6/6 PASS; refrigerator 4/6, 4/6 FAIL; nightstand NO VERDICT, 5/6
FAIL; toilet NO VERDICT ×2; hood NO VERDICT ×2.

### Part C: C3 non-regression (room session, 23:43–01:30; numbers as reported by room, its own scripts at 91598c88)

Runs: `crossing/g1x_c3a_{1,2,3}` (c3a_grade.txt each), `crossing/g1x_c3b_{on,off}_r{1,2,3}` (g1x_c3b_report.txt/.csv).

| | this build | v1 (8a133295) |
|---|---|---|
| C3a held (of 15; cabinet not run) | 8 / 10 / 12 | — |
| C3a phantoms | 3 / 4 / 4 | — |
| C3a median planar error | 0.09 / 0.11 / 0.09 m | 0.17 / 0.15 / 0.25 m (c3a_w2) |
| C3b world-frame corrected, ON | 3/3 | 1/4 valid |
| C3b world-frame corrected, OFF | 3/3 | 2/5 |

- Per class (C3a): bed, toilet, fridge held 3/3; chair 4, 5, 6 of 6; nightstand 1, 2, 2 of 2 (v1 2/2 each); **hood 0/3, never
  born** (v1 1/3); table 0, 0, 1 of 4 (v1 0, 1, 1); bottle never born.
- C3b: all six corrections are a new node born on sight, none by the chair's own offer, so the v1 mechanism is unchanged (no
  belief, no offer until seen). ON arms still claim one room_2 chair offer 3×, as in v1.
- Regressions against v1: the nightstand in one run (1 of 2) and the hood (0/3 vs 1/3). Planar error and C3b improve.

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
  nightstand, bed, toilet, refrigerator, hood, chair on the countable binary; table on its LEGACY binary. Since the fix wave
  (17f13198) the launcher defaults fridge/hood/chair to LEGACY, so EVERY run below sets
  COUNTABLE_AGENTS="refrigerator_concept hood_concept chair_concept" LEGACY_AGENTS="table_concept", and CONCEPT_WT explicitly
  (its default is the phase0 worktree, which has no countable links: the opt-ins would silently fall back to legacy).
  Check before each run: `concept_agents.sh resolve` = 6 countable (fridge/hood/chair from <cls>/countable/) + 1 legacy table.
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
  Scoring: room_concept/estimation_scripts/scripts/first_visit_nees.py (G1a, f2d456a3; written and reviewed before any batch
  result was seen): python3 -I first_visit_nees.py crossing/g1x_tour_S* > crossing/g1x_tour_nees.txt, with the replay tool
  built from the batch commit (make -C countable_concept/tests/build replay).
PART B — test (b), scripted A/B/A: 6 objects x 2 repeats = 12 runs, from CONCEPT_WT:
    COUNTABLE_AGENTS="refrigerator_concept hood_concept chair_concept" LEGACY_AGENTS="table_concept" DWELL_S=60 \
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
