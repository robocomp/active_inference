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
| no DOF over-confident | 0 % below rms/3 | 0 % on every DOF of every instance, both runs | **PASS** |
| fit rate | ≥ 8 Hz | 37.7 / 47.8 / 71.1 Hz | PASS |
| gn_conv = 0 share with rays | below live10's (bed 5 %, nightstands 12 % / 3 %) | bed **14.6 %**, nightstand B 0 %, nightstand A **24.2 %** (some at the 25-step cap) | **FAIL** (bed, nightstand A) |
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
**runs away in depth**: born at 3.0 m range already with h 0.69 m (prior 0.57, truth 0.51), it grows to h 0.77 while
the first updates hit the 25-step GN cap with very large Itr_L (≈ 6e4); the final centre is 0.53 m from the GT
nightstand. Calibration of the shift run: nightstand B all four DOF PASS (NEES 2.41); bed cy too wide, others PASS.

## Reading

κ ≡ 1 did what rev 3.3.11 predicted on information (the bed now measures; bed calibration fixed; no over-confidence
anywhere). It also exposed solver trouble: with the full ray weight the GN hits its 25-step cap on far, one-look
instances (bed 14.6 %, nightstand A 24–44 %), and one shifted-prior instance ran 26 cm deep. A failure is reported,
not tuned. Candidates to examine (not tested): the first updates at long range, where the beam now carries 1e4–1e5 of
information against a wide prior and LM must cover a large step; the shifted footprint prior letting the depth slide
along a flush wall.

## Open

1. GN non-convergence on far one-look instances (live12 bed 14.6 %, nightstand A 24 %; shift A 44 %).
2. Nightstand A depth runaway in live12shift (h +0.26 m, centre 0.53 m).
3. Wrong size priors still survive one look (H +0.095 on nightstand B).
4. h / cy / H still too wide on some instances (safe side).
5. Toilet never born (bathroom outline, room side).
