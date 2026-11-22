# LiDAR measures — live12 (spec §7c.8 rev 3.3.12: κ ≡ 1 on the beam path)

Branch `feat/cost-removal` at d4b33761 (Task 22). Same tour as live11: fixture `three_rooms_door0`, 2 legs, flag on.
Both runs **2/2**, trace S1–S8 OK. Toilet not born (bathroom has no outline, as in live11). Scorer output:
`room_concept/estimation_scripts/crossing/objects_live12*/` (scored with `lidar_measures_score.py`).

- `objects_live12` — flag on, κ ≡ 1
- `objects_live12shift` — flag on, κ ≡ 1, nightstand priors footprint 0.51 → 0.57 m, height 0.62 → 0.72 m

## What κ ≡ 1 changed: the LiDAR now informs every instance

| instance | live11 Itr_L median (updated rows) | live12 Itr_L median |
|---|---|---|
| bed | 9.4 (κ = 0 on 39/80 staged rows) | **354** |
| nightstand B (flush) | 365 | 7.3e4 |
| nightstand A (flush) | 244 | 5.8e4 |

No staged row had Σq = 0 (bed Σq median 94.7, nightstands 34.6 / 26.7).

## Pass (Task 23), against live11

| item | bound | live12 (on) | verdict |
|---|---|---|---|
| no DOF over-confident | 0 % below rms/3 | **bed w 81.9 % below** (shift 100 %); shift nightstand A cx/cy/h > 3σ on 100 / 67.5 / 100 % of rows (never settled, so the settled check skipped it). Was reported PASS: w was not scored | **FAIL** (corrected 11-22) |
| fit rate | ≥ 8 Hz | 37.7 / 47.8 / 71.1 Hz | PASS |
| gn_conv = 0 share with rays | below live10's (bed 5 %, nightstands 12 % / 3 %) | bed **14.6 %**, nightstand B 0 %, nightstand A **24.2 %** (some at the 25-step cap); all but one on camera+LiDAR rows (see Correction) | **FAIL** (bed, nightstand A) |
| final centre error | ≤ live10 + 1 cm | bed 0.050 (live10 0.084), nightstand B 0.029 (0.017, +1.2 cm), nightstand A 0.053 (0.034, +1.9 cm) | **FAIL** (both nightstands) |

## Calibration (settled rows inside [rms/3, 3·rms])

- **bed:** cx, cy, H **100 %** (live11: 36.9 / 100 / 44.2 %); h 0 % (too wide: std 0.06 vs rms 0.010). Median NEES 1.90
  (live11 0.89; expected ≈ 2.4).
- **nightstand B:** cx, h PASS; cy and H too wide (std 0.013–0.087 vs rms 0.002 / 0.009). NEES 0.40.
- **nightstand A:** never within 2.5 m with a fresh mask: not scored.

## Final centre / height error (m)

| object | live11 (on) | **live12 (on)** | live12shift |
|---|---|---|---|
| bed | 0.037, 0.014 | **0.050, 0.012** | 0.015, 0.012 |
| nightstand B | 0.022, 0.013 | **0.029, 0.010** | 0.032, 0.095 |
| nightstand A | 0.013, 0.007 | **0.053, 0.014** | **0.526**, 0.023 |

## Shifted priors (live12shift)

| | w err | h err | H err | shifted prior would read |
|---|---|---|---|---|
| nightstand B | −0.049 | +0.053 | +0.095 | +0.06 / +0.06 / +0.10 |
| nightstand A | +0.047 | **+0.259** | +0.023 | |
| bed (not shifted) | **−0.203** | +0.043 | +0.012 | |

The size priors are still not corrected from one look: nightstand B's H stays at the shifted prior (+0.095). Nightstand A
~~runs away in depth: born at 3.0 m range already with h 0.69 m~~ (corrected 11-22: the birth seed had h 0.96 and w 1.04,
centred near the truth; h 0.69 and cx −1.055 are the OUTPUT of the first update, which already carries 0.47 m of the final
0.53 m error and lands on the bed's head corner). It then grows to h 0.77 (about 0.1 m, in converged LiDAR-only cycles);
the final centre is 0.53 m from the GT nightstand. Calibration of the shift run: nightstand B all four DOF PASS (NEES 2.41); bed cy too wide, others PASS.

## Reading

κ ≡ 1 did what rev 3.3.11 predicted on information (the bed now measures; bed calibration fixed; no over-confidence
anywhere). It also exposed solver trouble: with the full ray weight the GN hits its 25-step cap on far, one-look
instances (bed 14.6 %, nightstand A 24–44 %), and one shifted-prior instance ran 26 cm deep. A failure is reported,
not tuned. Candidates to examine (not tested): the first updates at long range, where the beam now carries 1e4–1e5 of
information against a wide prior and LM must cover a large step; the shifted footprint prior letting the depth slide
along a flush wall.

## Correction (11-22, after an advisor review of the logs; re-scored with w and an all-rows check)

The sections above were written before these were checked; each item below supersedes them where they conflict.

1. **Non-convergence follows the camera+LiDAR updates, not range or one look.** gn_conv = 0 on staged rows, split by
   `gated` (0 = the camera updated too): live12 bed 6/17 camera+LiDAR vs 0/24 LiDAR-only; nightstands 7/54 vs 1/31;
   shift bed 3/19 vs 0/57; shift nightstands 21/82 vs 1/72 (nightstand A alone 21/22). The bed fails at 0.93 m; nightstand
   B has more information than A and never fails. "Far, one-look" in the Reading is the wrong covariate.
2. **Bed w is a κ ≡ 1 regression in BOTH runs**, not a shift effect: final w 1.492 (live12, −0.26 vs 1.75) and 1.547
   (shift, −0.20); live11 1.730 / 1.707. Its std is 0.02–0.03 m: over-confident (table above).
3. **Over-confidence was not established.** The scorer did not score w and skipped instances with no settled rows. With w
   and the all-rows |err| > 3σ share added (`lidar_measures_score.py`): bed w 79 % / 100 % of rows; shift nightstand A cx
   100 %, cy 67.5 %, h 100 %. live11 for comparison: bed w 0 %, nightstand_40 w 6 %.
4. **The shift comparison is confounded:** shift nightstand A's birth mask had 221 points against 355 in live12, so the bad
   first basin cannot be attributed to the shifted prior from one run.
5. **Itr_L medians mix regimes:** 6e4–5e5 on parked camera+LiDAR rows, 30–350 on LiDAR-only rows.

Next (owner 11-22, step 1): `Model.GnTrace` (31d53767, default off) logs every GN step and, on unconverged updates, a
per-channel finite-difference check of the solver's gradient and curvature against the energy; a live run with it on
decides between a G/W mismatch, a kink, or overstated curvature before any model change.

## Open

1. GN non-convergence on camera+LiDAR updates (live12 bed 14.6 %, nightstand A 24 %; shift A 44 %).
2. Nightstand A's first update lands on the bed's corner in live12shift (centre 0.53 m; over-confident).
6. Bed w −0.20 to −0.26 m, over-confident, in both κ ≡ 1 runs.
3. Wrong size priors still survive one look (H +0.095 on nightstand B).
4. h / cy / H still too wide on some instances (safe side).
5. Toilet never born (bathroom outline, room side).
