# LiDAR measures — live11 (plan rev 5, spec §7c.8 rev 3.3.10, smooth beam model)

Branch `feat/cost-removal` at dd5779cc (Tasks 15–20); agent binary built 11-21 20:52, no source newer. Three warm tours on
fixture `three_rooms_door0`, 2 legs each (room_1 → room_2 → room_3), all **2/2 OK**, trace S1–S7 OK in all three, S8 OK in
off and shift. Scorer: `countable_concept/tools/lidar_measures_score.py` (room→world fitted per run). Score dumps:
`room_concept/estimation_scripts/crossing/objects_live11*/` + the printouts quoted here.

- `objects_live11` — flag on (`Model.LidarMeasuresOnGated = true`), beam rays staged by geometry, floor + room walls in p_f
- `objects_live11off` — flag off
- `objects_live11shift` — flag on, nightstand class priors wrong: footprint 0.51 → 0.57 m, height 0.62 → 0.72 m

**Toilet: not scored in any run.** Its births were HELD all run ("room … has no outline yet (or is still a proto)"), and the
scorer's bathroom frame fit had no usable samples. Room-side, flag-independent (live10off/shift had the same).

## Pass "Live11"

(The bed rows below are not beam evidence: κ = 0 on 39/80 staged rows, see the paragraph after the staging table.)

| item | bound | live11 (on) | verdict |
|---|---|---|---|
| fit rate | ≥ 8 Hz | bed 31.2, nightstand_1 29.2, nightstand_0 79.8 Hz | PASS |
| gn_conv = 0 share on updates with rays | below live10's 5 / 12 / 3 / 20 % | bed 2.4 % (1/41), nightstands 0 % (0/78, 0/24) | PASS |
| final centre error | ≤ live10 + 1 cm | bed 0.037 (live10 0.084); nightstand_1 0.022 (0.017); nightstand_0 0.013 (0.034) | PASS |
| no DOF over-confident (two-sided §8) | 0 % below rms/3 | 0 % below on every DOF of every instance | PASS |

## Final centre error and height error (m)

| object | live10 (on) | **live11 (on)** | live11off | live11shift |
|---|---|---|---|---|
| bed | 0.084, 0.044 | **0.037, 0.014** | 0.025, 0.058 (0.85 m jump between looks) | 0.171, 0.010 (1 look) |
| nightstand B (nightstand_1, flush) | 0.017, 0.012 | **0.022, 0.013** | 0.017, 0.024 | 0.039, 0.123 |
| nightstand A (nightstand_0, flush) | 0.034, 0.062 | **0.013, 0.007** (never within 2.5 m) | 0.131, 0.040 | 0.156, 0.086 |

**Flush nightstands (the owner's watch item, rev 3.3.10 (3)).** Both nightstands stand flush against the bedroom wall.
The fixtures predicted a stated-limit-4 width blow-up at the live ε_px 0.094 (+14.4 cm w with a flush neighbour, +22.8 cm
per edge at b = 5 cm). **Live it does not appear:** final centre errors 2.2 and 1.3 cm, |H − top| 1.3 and 0.7 cm, no step
toward the wall between looks (≤ 1 cm). The wall is a known room surface in p_f, which is what the flush-neighbour fixture's
30 cm unknown surface is not; an unknown flush neighbour (another object, not a wall) has not been seen live.

## Calibration (settled rows inside [rms/3, 3·rms], per DOF; w and cx separately)

- **live11 (on):** nightstand_1 cx, cy, H PASS (100 / 100 / 97.8 %); h too wide (std 0.031–0.035 vs rms 0.006). Bed cy PASS;
  cx 36.9 %, H 44.2 %, h 0 % — all on the WIDE side (std above 3·rms; 0 % below rms/3). Median NEES 0.89 / 0.80 (expected
  ≈ 2.4): conservative. h over-wide is the open item from live10, unchanged.
- **live11off:** every DOF fails; bed std up to 2.0 m on cy, NEES 0.04, 7.8 % below rms/3 on cy (over-confident rows).
- **live11shift:** nightstand_0 all four DOF PASS, NEES 4.95; bed cy/H PASS, cx and h FAIL (wide on h; cx rms 0.17 m with
  std ≤ 0.18 — the bed's single look).

## Shifted priors

| | w err | h err | H err | (shifted prior would read) |
|---|---|---|---|---|
| nightstand_0 | +0.064 | +0.037 | +0.086 | +0.06 / +0.06 / +0.10 |
| nightstand_1 | +0.043 | +0.016 | +0.123 | |
| bed (priors not shifted) | −0.043 | −0.008 | +0.010 | |

The nightstands keep the shifted prior's size from one look each (shift run: every instance 1 look). The beam does not
pull w or H back to truth from a single look; live10shift had the same tendency (0.096 / 0.052 centre). FAIL against the
scorer's 3 cm / 5 cm size tolerances.

## Per-instance beam staging (live11, on)

| instance | staged rows | staged rays med / max | κ med | rays without camera info (sum / med) | no-membership cycles (longest run) | no-sweep rows | gap rays (sum / max / rows) |
|---|---|---|---|---|---|---|---|
| bed_21 | 80 | 245 / 455 | 0.000 | 4000 / 55 | 0 (0) | 1 | 443 / 40 / 38 |
| nightstand_40 (B) | 78 | 111 / 129 | 0.061 | 333 / 0 | 0 (0) | 0 | 810 / 19 / 77 |
| nightstand_41 (A) | 26 | 126 / 127 | 0.006 | 183 / 0 | 0 (0) | 0 | 14 / 4 / 7 |

No cycle lacked membership (Task 15's open question on GATED cycles: none occurred on these instances).

**The beam did not measure the bed (read this before citing Pass Live11).** Recomputed from `bed_ai2_log.csv` and
`nightstand_ai2_log.csv` (rows with `beam_staged` > 0; updated rows = Itr_L > 0):

| instance | staged rows | κ = 0 exactly | κ med / max | updated rows | Itr_L med (range) | σ at med Itr_L |
|---|---|---|---|---|---|---|
| bed_21 | 80 | **39** | 5e-7 / 0.032 | 41 | **9.4** (0.10 – 459) | ≈ 0.33 m |
| nightstand_40 | 78 | 0 | 0.061 / 0.183 | 78 | 365 (27 – 11985) | ≈ 0.052 m |
| nightstand_41 | 26 | 2 | 0.006 / 0.017 | 24 | 244 (9 – 1069) | ≈ 0.064 m |

bed_21 staged rays on 80 rows, but κ is exactly 0 on 39 of them, so the beam made no measurement on those cycles. On the
rest κ is at most 0.032 and the median Itr_L is 9.4, i.e. σ ≈ 0.33 m: the beam barely informs the bed. **The bed's
improvement over live10 (0.084 → 0.037 m) is therefore NOT evidence for the beam model.** The nightstands did get real
information (Itr_L 245 – 365), though nightstand_41's κ is small (median 0.006) and nightstand_40's is 0.061.

Likely mechanism: κ = coverage · ca with ca = (1 − R)^power, R the resultant length of the ray bearings. A one-sided view
(the bed seen from one side) has R near 1, so ca → 0 and κ → 0. That is flag-off's endpoint-factor weighting applied to a
model in which misses already carry the information. It is a spec-level question, not a code defect.

The "Pass Live11" table above is a pass on the stated bounds; for the bed it is a pass of the camera path with the beam
inert on about half its rows, and only the nightstand rows (and the flush-neighbour result) speak for the beam.

## Inputs carried to the owner

- ε_px (owner ruling 11-20, max over runs): bed 0.085, nightstand 0.094, toilet 0.001; b*(ε_px) per Task 15.
- Task 20's printed fixture numbers: flush neighbour w +14.4 cm with rays (camera-only 0); 5 cm background +22.8 cm per
  edge at ε_px 0.094 (+6.3 cm at 2 %, predicted +5.6); gap under furniture PASS; calibration NEES/2 1.04 (20 cm) / 0.89 (1 m).

## Open

1. h over-wide on bed and nightstands (std 5–10× rms), as in live10.
2. Wrong size priors survive one look (shift run).
3. Toilet never born: bathroom has no outline (room side).
4. Flush against an unknown (non-wall) neighbour: fixture +14 cm, not yet seen live.
5. κ collapses on one-sided views (bed): owner 11-22 — revisit κ in the spec.
