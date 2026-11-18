# LiDAR measures — live10 (plan rev 4, spec §7c.7 rev 3.2.1)

Branch `feat/cost-removal` at 081b739d (Tasks 11–14 + final fix wave). Three warm tours on fixture
`three_rooms_door0`, 2 legs each (room_1 → room_2 → room_3), all 2/2 OK, trace S1–S7 OK, S8 3 violations each
(room side, open). Scorer: `countable_concept/tools/lidar_measures_score.py` (fitted room→world per run).

- `objects_live10` — flag on (`Model.LidarMeasuresOnGated = true`)
- `objects_live10off` — flag off
- `objects_live10shift` — flag on, nightstand class priors wrong: footprint 0.51 → 0.57 m, height 0.62 → 0.72 m

**Toilet:** it was born only in `objects_live10`. In live10off and live10shift no toilet instance was created, and the
`toilet_ai2_log.csv` in those two run folders is a stale copy of live10's (same md5). Toilet rows of those runs
are not data. Cause not investigated (birth-side; flag-independent code).

## Final centre error and height error (m)

| object | live9 (on) | **live10 (on)** | live10off | live10shift |
|---|---|---|---|---|
| bed | 0.080, \|H−top\| 0.160 | **0.084, 0.044** | 0.101, 0.008 (0.61 m jump between looks) | 0.092, 0.046 |
| nightstand B (nightstand_1) | **0.374** (b_chain jump) | **0.017, 0.012** | 0.057, 0.082 | 0.096, 0.052 |
| nightstand A (nightstand_0) | 0.030, 0.061 | **0.034, 0.062** | 0.094 (never within 2.5 m) | 0.052, 0.027 |
| toilet | 0.362 | **0.215, 0.024** | — (not born) | — (not born) |

## Calibration (settled rows inside [rms/3, 3·rms], per DOF)

- **live10 (on):** bed cx/cy/H 100 %, h too wide (std 0.036–0.082 vs rms 0.009); nightstand_14 cx 97.5 %, cy 100 %,
  H and h too wide (std ≈ 0.08 vs rms 0.010); nightstand_15 all four PASS. Median NEES 4.57 / 0.59 / 2.72
  (expected ≈ 2.4). No DOF is over-confident (0 % below rms/3 everywhere).
- **live10off:** bed fails all four DOF — cx rms 0.41 m, σ up to 7.7 m on cy, NEES 0.04; nightstand_16 cx 10 %,
  cy 4 %. The flag-off belief loses calibration after the 0.61 m bed jump.
- **live10shift:** bed cx, H PASS; cy/h partly too wide (σ hits 0.30/0.20). Nightstands: cx and H PASS, cy/h too wide.

## Rev 4 checks (Task 10 again)

| check | live10 | live10shift |
|---|---|---|
| \|e\| ≤ 3σ of Σ_pose (updated rows) | worst 0.83 / 0.49 / 0.52 — PASS | worst 1.54 / 0.43 / 0.50 — PASS |
| parked Σ_pose vs proto_room σ (radial + ψ, 1 %) | 100 % within 1 % (rel 0.0000) — PASS | PASS |
| fit rate ≥ 8 Hz | 11.7 / 9.8 / 8.4 / 23.9 — PASS | PASS |
| gn_conv = 0 share, with rays | 5 % / 12 % / 3 % / 20 % | 8 % / 7 % / 0 % |
| gn_conv = 0 share, camera-only | **bed 61 % (median 12 steps)** / 4 % / 2 % | **bed 91 %** / 4 % / 5 % |

The Σ_pose match is exact (rel 0.0000), which is what a correct slot extraction of the same quantity gives (room
publishes the σ it logs); a wrong slot would not reproduce it.

## Verdict

1. **The live9 failure is fixed.** nightstand B 0.374 → 0.017 m, with |e| ≤ 0.83 σ of Σ_pose on every updated row.
   The bed's top-height error 0.160 → 0.044 m. No phantom instances.
2. **The flag now pays on this run pair.** live10off lost the bed (0.61 m step, σ to 7.7 m, every DOF miscalibrated);
   flag on stayed calibrated. (live9 had shown no position gain; the D-rescale bug masked it.)
3. **Wrong priors:** with +6 cm footprint / +10 cm height priors, the nightstands end 0.05–0.10 m off and H is pulled
   +2.7 / +5.2 cm toward the wrong prior — the prior still wins partly, as in live9shift.
4. **Open, not regressions:**
   - over-wide σ on h (all runs) and on nightstand H (on): safe-side, but NEES 0.59 says information is left unused;
   - bed camera-only updates rarely converge (61 % / 91 % run 12 steps) — candidate: the smooth-ray follow-up does
     not apply (no rays); needs a look at the bed's camera energy (large flat top, yaw near-symmetric);
   - toilet births only in 1 of 3 runs;
   - parked follow-ups from the ledger: smooth ray energy (T14-5/T14-7 deferrals), update_size_likelihood still
     reads the D-era chain cov, membership σ uses full Σ_pose (conservative).
