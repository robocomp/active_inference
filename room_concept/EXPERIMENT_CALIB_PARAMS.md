# Testing the calibrated parameters, one at a time

Plan, pre-registered. Written 2026-08-30. Companion to `EXPERIMENT_CALIB_LOCALIZATION.md` (which
asks the downstream question: does any of this improve localization). Analysis:
`tools/calib_localization_ab.py`; live readout: the localization plot on the canvas
(`src/localization_drift.h`).

## 1. What is actually being estimated

Six parameters, solved jointly (`src/calibration_estimator.h:56-71`). They are identifiable
because each lands on a **different component of the same correction** against a **different
covariate** — not because they were assumed independent:

| # | parameter | correction component | covariate | unit |
|---|---|---|---|---|
| 0 | `k_v` | forward | distance driven | fractional |
| 1 | `eps_yaw` | **lateral** | distance driven | rad |
| 2 | `k_omega` | heading | **rotation** | fractional |
| 3 | `b_omega` | heading | **elapsed time** | rad/s |
| 4 | `k_lat` | lateral | **lateral** distance | fractional |
| 5 | `dk_wheel` | heading | **distance** | rad/m |

★ Three of them (`k_omega`, `b_omega`, `dk_wheel`) land on the *same* component — heading — and
are separated **only** by their covariate: rotation, time, distance. That is the sharpest
experimental constraint in this whole plan, and it dictates the routes in §4. A route that turns
at one constant rate for its whole length cannot separate a gyro scale from a gyro bias, and the
solver will hand you a confident split anyway.

## 2. What can be injected today — the honest coverage

Injection lives in `webots-bridge/etc/config`, `SensorNoise.*`; the corruption is applied to the
published velocities (`src/specificworker.cpp:799-825`) and the gyro (`:2140-2143`), while the
supervisor pose passes through untouched, so **ground truth survives** (`:828-838`).

| parameter | knob that injects it | status |
|---|---|---|
| `k_v` | `SensorNoise.WheelScaleV` | ✅ testable now |
| `b_omega` | `SensorNoise.GyroBias` | ✅ testable now |
| `k_omega` | — a gyro **scale** | ❌ **no knob exists** |
| `eps_yaw` | — a **rotation** of the velocity vector | ❌ no knob exists |
| `dk_wheel` | — a heading drift per metre | ❌ no knob exists |
| `k_lat` | lateral-only scale | ❌ no knob, and not excitable on a differential base at all |

**So 2 of 6 are injection-testable today.** `WheelScaleV` scales the x and y velocity components
*equally*, which is an isotropic scale: it moves `k_v` and cannot produce `eps_yaw` (that needs a
rotation). On a mecanum base the same knob would move `k_v` and `k_lat` together and confound
them; on Shadow, `d_lateral` is identically zero, so it is clean.

⚠ The earlier draft of the companion document said `GyroScale = 0.03`. **That knob does not exist
on this bridge** — it was carried over from p3bot-bridge. `k_omega`, the parameter with the best
prior result (recovered to 0.07% on P3Bot), is the one this rig currently cannot test.

### The four knobs to add, if full coverage is wanted

All four are a few lines in `webots-bridge/src/specificworker.cpp`, in the block that already
applies the others. They are listed here so the decision is explicit rather than implied:

| new key | applied as | unlocks |
|---|---|---|
| `SensorNoise.GyroScale` | `gz *= (1 + s)` beside the existing bias at `:2143` | `k_omega` |
| `SensorNoise.OdomYawOffset` | rotate `(velocity_local.x, .y)` by a constant angle at `:819` | `eps_yaw` |
| `SensorNoise.WheelDriftPerM` | `rot_velocity += d * forward_speed` at `:823` | `dk_wheel` |
| `SensorNoise.WheelScaleLat` | scale the lateral component only | `k_lat` (P3Bot only) |

## 3. The free tests — negative controls, no injection needed

These cost nothing and catch the failure mode that an injection test cannot: a parameter that
moves when it has no business moving.

- **`k_lat` on Shadow must never leave its prior.** A differential base cannot strafe, so its
  covariate is identically zero and the code says it "correctly stays at its prior for ever"
  (`calibration_estimator.h:62-64`). If it moves, or reports `informed`, something is leaking
  between parameters. This is a standing assertion for **every** run below, not a separate arm.
- **Every arm is a cross-talk control for the other five.** In each injected arm, exactly one
  parameter should move. The others must stay within 2 of their own sigmas.
- **`informed` must agree with the route.** A parameter whose covariate the route never exercised
  must report `informed = false` and keep its prior sigma. On P3Bot, `eps_yaw` correctly widened
  its sigma during a turns-only run — "not asked", not "no signal". A parameter that reports a
  confident number after a route that could not identify it is a bug, not a result.

## 4. Per-parameter protocol

Common to all arms — see §5 for why each of these is not optional:

- delete `etc/motion_calib_state.csv` before the arm
- `SensorNoise` sigmas set to **zero** (bias recovery is the test; noise only widens the posterior)
- one driver only; same route within a block; ABA ordering (OFF / ON / OFF)
- ≥ 8 usable 60 s windows of moving driving per arm

### A. `k_v` — forward odometry scale ✅ runnable now

- **inject**: `SensorNoise.WheelScaleV = 0.03`
- **route**: **straight-heavy**. Covariate is distance; episodes close at 0.25 m of travel
  (`episode_min_trans`), so 25 m is ~100 episodes. Target a few hundred metres.
- **predicted**: reported velocity is `true × 1.03`, so `k_v → 1/1.03 = 0.97087 ×` its native
  value (native measured ≈ 0.998 on Shadow, §6).
- **pass**: within 20% of the injected magnitude; `informed = true`; `sigma` below 0.9× prior;
  no other parameter moves more than 2 of its own sigmas.
- **precedent**: P3Bot recovered 83% of the same injection, stable to ±0.001 across seven windows.

### B. `b_omega` — gyro bias ✅ runnable now

- **inject**: `SensorNoise.GyroBias = 0.005` rad/s (≈ 0.29 °/s, 10× the 5e-4 prior sigma)
- **route**: ★ **vary the turn RATE, not just the amount.** `b_omega` is separated from `k_omega`
  *only* by time-versus-rotation, so the route must contain both slow sustained turns (much time,
  little rotation) and brisk ones (little time, much rotation). A route that turns at one rate
  makes the two collinear, and the solver will still return a confident split.
- **predicted**: magnitude 0.005 rad/s. Read the *sign convention* off the first run and then hold
  it fixed — the magnitude is the pre-registered prediction, the sign is a convention to record,
  not a result to interpret after the fact.
- **pass**: as A, plus `k_omega` must stay put — that is the whole point of this arm.

### C. `k_omega` — gyro scale ⛔ blocked

Needs `SensorNoise.GyroScale`. Once it exists: inject 0.03, turn-heavy route with **mixed rates**
(same reason as B), predict `k_omega → 1/1.03 ×` native. This is the parameter with the strongest
prior evidence, so it is the highest-value knob to add.

### D. `eps_yaw` — body/mount yaw offset ⛔ blocked

Needs `SensorNoise.OdomYawOffset`. Straight-heavy route (covariate is distance, component is
lateral). Note the native value is worth having on its own: P3Bot converged to −0.536° ± 0.13 from
cold, reproducibly; Shadow's pilot sat at −0.0385° with σ 0.51°, i.e. **entirely uninformed** —
so on this robot `eps_yaw` has never actually been measured.

### E. `dk_wheel` — per-wheel mismatch ⛔ blocked

Needs `SensorNoise.WheelDriftPerM`. Straight-heavy. ⚠ There is already a real signal here: a
closure test recorded in `webots-bridge/etc/config:127-136` found Shadow's wheels over-reporting
rotation by ~16% intrinsically. `dk_wheel` is the parameter that should absorb the distance-driven
part of that, and nobody has checked whether it does.

### F. `k_lat` — lateral scale ⛔ not applicable on Shadow

Structurally unexcitable on a differential base. Run it on P3Bot, and only with a lateral-only
knob — `WheelScaleV` would move `k_v` at the same time and confound the two. Until then it is the
negative control of §3.

## 5. Method rules — each one has already produced a wrong answer here

- **★ Delete `etc/motion_calib_state.csv` between arms.** It persists 512 *episodes* of evidence
  across restarts. An OFF arm inheriting a warm window is not OFF; an ON arm inheriting a converged
  window is not learning. The pilot on 2026-08-30 ran with 72 stale episodes still loaded.
- **★ There is no parameter state to perturb.** This is a *batch* estimator over a window plus a
  prior whose **mean is nominal and not settable** (`Prior` carries sigmas only,
  `calibration_estimator.h:145-154`), and restoring a fitted value as a prior mean is deliberately
  refused as a ratchet (`motion_calibration.h:236`). So "start the parameters wrong" is not
  available: the error must be put in the **world**, which is what §2 does.
- **★ A parameter is only APPLIED once `informed`** — sigma below 0.9× prior. During recovery
  there is a window where the estimator has learnt the correction but is not yet using it, so the
  parameter trace and the pose will legitimately disagree. Do not read that gap as a bug.
- **★ Never compare two sessions.** `opt/m` showed a 5× spread between two runs with identical
  configuration. Arms are back-to-back inside one session or they are not comparable.
- **★ Separate parked from moving.** A parked robot predicts nothing and always looks accurate.
- **★ Exclude burst windows.** The regime where the localiser is not tracking once dragged a
  healthy `k_v` from 1.0059 to 0.8907 in two 180 s windows.

## 6. Baseline first — the native values

Before any injection, one clean converged run per robot, state file deleted, mixed route, long
enough that the parameters that *can* be identified report `informed`. This is the reference every
injected arm is measured against, and it is also the only way to tell "no error exists" from "not
observable". Native values so far:

| | P3Bot (2026-08-23) | Shadow (2026-08-30 pilot, NOT converged) |
|---|---|---|
| `k_v` | 1.00004 ± 0.003 | 0.998058, σ 0.0139 |
| `k_omega` | 0.9974 ± 0.002 | 0.999578, σ 0.0161 |
| `eps_yaw` | −0.536° ± 0.13 | −0.0385°, σ 0.51° — **uninformed** |

⚠ The Shadow column is a 35 m pilot with noise injection on and a turn-heavy route; its sigmas had
only fallen from 0.020 to 0.0139. It is a starting point, not a baseline.

## 7. Measurement

Per arm, three things, and all three must agree before a result is claimed:

1. **The parameter trajectory** against the predicted target, versus distance driven — from the
   `calib_*` columns of `tmp/sdf_localizer/gt_error.csv`. Time-to-converge and the residual.
2. **Cross-talk**: every other parameter's excursion in units of its own sigma.
3. **Localization**: `pred drift mm/m` from `tools/calib_localization_ab.py` — must fall from the
   OFF arm to the ON arm. On the 96.1% of cycles that early-exit the published pose *is* the raw
   prediction, so a real model improvement has nowhere to hide.

A parameter that converges to its target while `pred drift` does not improve is not a success —
it means the parameter is not the one carrying the error, and that contradiction is the finding.

## 8. Order of execution

1. Baseline, clean, per robot (§6).
2. Arm A — `k_v` via `WheelScaleV`. Runnable today; validates the whole pipeline end to end.
3. Arm B — `b_omega` via `GyroBias`, with the mixed-rate route.
4. Decide whether to add the four knobs of §2. `GyroScale` first: it unlocks the parameter with
   the strongest prior evidence, and it is the smallest of the four.
5. Arms C–E once their knobs exist.

If Arm A shows no separation, stop: nothing downstream is worth interpreting, and the finding is
about the pipeline rather than the calibrator.
