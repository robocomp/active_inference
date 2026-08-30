# Does applying the self-calibrated motion parameters improve LOCALIZATION?

Protocol, pre-registered. Written 2026-08-29. Analysis: `tools/calib_localization_ab.py`.

## The question, stated so it can fail

`MotionCalibrator` learns three motion-model parameters online (`k_v` forward odometry scale,
`k_omega` gyro scale, `eps_yaw` body/mount yaw offset) and the localiser predicts with them. Two
different claims are easy to confuse, and only the second is open:

1. **The estimator recovers a real parameter.** SETTLED. Injected `GyroScale = 0.03`, 73 m /
   24 episodes: predicted `k_omega` 0.96833, measured 0.96901 — **0.07% apart**. Wheels, 109 m /
   39 episodes: 83% of an injected error recovered. (`thesis-online-motion-calibration-experiment`.)
2. **Applying the learnt parameters makes the ROBOT BETTER LOCALISED.** OPEN. This document.

The 2026-08-22 A/B (262 m ON vs 105 m OFF) answered a third, adjacent question — it graded the
**predictor**: err/m 0.0263 vs 0.0303, +14.9%, t = 2.67, significant, with the pre-registered
reading *"it lowers the error floor, it does not flatten the sawtooth"*. It never looked at the
published pose against ground truth, because at the time the optimiser was correcting every
1–6 s and would have hidden any predictor difference anyway.

## Why it is worth running NOW and was not before

`early_exit_pct` went **0 → 100** on 2026-08-29 (the ceiling-band fix `8164fc3` plus the
covariance fix `f2336d0`). The optimiser no longer fires at all. The published pose is now the
prediction plus the per-cycle SDF polish, so the coupling that used to hide prediction quality
behind the optimiser's corrections is gone. Whatever the calibration is worth reaches the pose
far more directly than it did in August.

## Endpoint and pre-registration — fix these BEFORE the first run

| | |
|---|---|
| primary | RPE translation, mm per metre travelled, `ds = 1.0 m`, moving rows only |
| secondary | RPE rotation, deg per radian turned, `dth = 0.5 rad`; aligned ATE RMS (mm) |
| unit of analysis | one 60 s window; arms compared window-to-window |
| exclusions | parked rows (GT speed < 0.02 m/s); burst windows (>50% of cycles with `iters > 0`) |
| predicted direction | calibration ON ≤ OFF on both RPE channels |
| expected size | **small** on the healthy robot — see below |

**Why RPE and not "distance from the true position".** GT is the Webots **world** frame; the
estimate is the **room** frame, whose orientation room_concept picks from its own fit and which
differs between runs. A constant offset between them is expected and carries no information.
`|est − gt|` therefore measures the frame choice, not the localiser. RPE compares *increments*
and is invariant to the frame entirely — and increments are exactly what `k_v`, `k_omega` and
`eps_yaw` act on.

**The effect is expected to be small, and that is the whole difficulty.** On the uninjected robot
`k_v` = 1.00004 and `k_omega` = 0.9974 — there is almost nothing to correct. Only
`eps_yaw` = −0.536° is a sizeable term. A small effect measured across two separate sessions will
return noise: `opt/m` showed a **5× spread between two runs with identical configuration**. So:

> **Never compare two sessions. Arms must be back-to-back inside one session, on the same route.**

## Arms, in this order

Run the injected arm FIRST. It is the one with the power to detect an effect, and if the pipeline
cannot see a 3% injected error there is no point interpreting the healthy robot at all.

### Where the injection actually lives (VERIFIED 2026-08-30, not remembered)

`/home/pbustos/robocomp/components/webots-bridge/etc/config` — the `SensorNoise.*` keys. That
component builds `Webots2Robocomp`, which is the bridge process actually running. ⚠ There are four
config files in that directory (`config`, `config.new`, `config.toml`, `config.toml.new`); the
running command line is `bin/Webots2Robocomp etc/config`, so **`etc/config` is the live one**.

★★★ **THE GROUND TRUTH SURVIVES THE INJECTION.** The startup banner says *"this bridge is NO LONGER
publishing ground truth"*, which reads like it kills this whole experiment. It does not: at
`src/specificworker.cpp:799-825` the corruption is applied to `velocity_local.x/y` and
`rot_velocity` only, and at `:2140-2143` to the gyro `gz`. The published pose (`pose_data.x/y/z`,
`pose_data.rz`, `:828-838`) comes straight from the supervisor's `shadow_position` / `orientation`
and is never touched. So `robot_gt_*` stays clean while the odometry and IMU streams are
corrupted — exactly the separation this experiment needs. The banner means "the ODOMETRY you are
being handed is no longer truth".

★★★ **THERE IS NO `GyroScale` ON THIS BRIDGE.** An earlier draft of this document said
`GyroScale = 0.03`; that was carried over from p3bot-bridge and is wrong here. What exists:

| key | injects | which parameter it tests |
|---|---|---|
| `SensorNoise.WheelScaleV` | forward/lateral velocity scale | **`k_v`** — the clean, high-power channel |
| `SensorNoise.GyroBias` | additive rad/s on the gyro | **`b_omega`** (the BIAS state), *not* `k_omega` |
| `SensorNoise.WheelScaleW` | wheel yaw-rate scale | nearly inert: the gyro carries ~99% of heading, and the wheels already over-report rotation ~16% intrinsically (closure test, `etc/config:127-136`) |

**No gyro SCALE knob exists**, so `k_omega` cannot be injection-tested on this bridge without
adding one. Use `WheelScaleV` for the headline arm.

⚠ **Noise is currently ON** (`SensorNoise.Enabled = true`, `WheelSigmaV = 0.006`,
`WheelSigmaW = 0.010`). Zero the sigmas for the injected arms: this tests recovery of a *bias*,
and noise only widens the posterior — it is also part of why the pilot's sigmas fell so slowly.

### Block 1 — injected error (high power)

| arm | `WheelScaleV` | sigmas | `MotionCalibEnabled` | starting calibrator state |
|---|---|---|---|---|
| A1 | 0.03 | 0 | **false** | delete `etc/motion_calib_state.csv` |
| B1 | 0.03 | 0 | **true** | delete `etc/motion_calib_state.csv` |
| A1′ | 0.03 | 0 | **false** | delete `etc/motion_calib_state.csv` |

`k_v`'s regressor is distance, so this block wants a **straight-heavy** route.

### Block 2 — healthy robot (honest, low power)

| arm | `WheelScaleV` | sigmas | `MotionCalibEnabled` | starting calibrator state |
|---|---|---|---|---|
| A2 | 0 | as configured | **false** | delete `etc/motion_calib_state.csv` |
| B2 | 0 | as configured | **true** | delete `etc/motion_calib_state.csv` |
| A2′ | 0 | as configured | **false** | delete `etc/motion_calib_state.csv` |

`MotionCalibEnabled` is `room_concept/etc/config.toml:894`.

**Each arm ≥ 8 usable 60 s windows of MOVING driving** (~8–10 min), same route, mixed straight and
turning — `k_v` loads on distance, `k_omega` on rotation, `eps_yaw` on distance. A turns-only route
cannot identify `k_v` and a straight-only route cannot identify `k_omega`; a run that exercises one
will report the other as "not asked", which is correct behaviour and a useless experiment.

## Traps, each of which has already produced a false finding here

- **★ The calibrator's state file persists across restarts.** `etc/motion_calib_state.csv` is
  *evidence, not parameters* — it holds up to 512 episodes and is reloaded at startup (75 rows as
  of this writing). An OFF arm inheriting a warm window is not OFF, and an ON arm inheriting a
  converged window is not learning. **Delete it between arms, or declare in the log that you did
  not and why.** This is the single easiest way to get a confident wrong answer here.
- **★ `gt_theta` and `est_theta` do not share a heading convention.** Measured on
  `gt_error_2026-08-29_21-54-45.csv` as mean(course − θ): GT **+0.01°** (R = 1.000, forward = +x,
  the supervisor's), EST **+89.77°** (R = 0.904, forward = +y, the body frame). The CSV does not
  say so. Applying it as a declared +90° from the frame docs turned a nonsensical RPE of
  **1361 mm/m into 49 mm/m** — the entire figure had been the convention. The script applies the
  documented constant and refuses to run if the file disagrees by more than 5°. **It must never
  be fitted**: a free constant angular offset is the same shape as `eps_yaw`, so fitting it would
  absorb the effect under test and guarantee a null.
- **Pooling parked with moving cycles.** A parked robot predicts nothing and therefore always
  looks accurate. Excluded by GT speed, from GT alone.
- **Burst windows.** The regime where the localiser is not tracking and the optimiser fires on
  nearly every cycle smears any average it lands in — and it is also the regime that once dragged
  a healthy `k_v` from 1.0059 to 0.8907. Excluded and counted separately.
- **Reading `t` on single-digit window counts.** The script prints `n` as the *window* count for
  this reason. Extend the run; do not read harder.

## Before you start, confirm the instrument is live

```
python3 -c "import csv,numpy as np; d={}
[d.setdefault(k,[]).append(float(v)) for r in csv.DictReader(open('tmp/sdf_localizer/gt_error.csv')) for k,v in r.items() if k]
print({k:len(np.unique(d[k])) for k in ('gt_x','gt_y','gt_theta')})"
```

All three must show **more than one unique value**. As of 2026-08-29 22:20 the live run showed
exactly **1** — GT frozen at (−5.135, 4.709, 2.376) across 5192 rows — so either the robot was
parked or `robot_gt_*` was not updating. GT exists **only in simulation**; without it this whole
protocol is unrunnable.

## Analysis

```
python3 tools/calib_localization_ab.py \
    tmp/sdf_localizer/<A1>.csv "OFF injected" \
    tmp/sdf_localizer/<B1>.csv "ON  injected" \
    tmp/sdf_localizer/<A1prime>.csv "OFF injected (repeat)"
```

Report the ABA triple, not the pair: A1 and A1′ bracketing B1 is what shows the difference is the
toggle rather than the session drifting. The script prints the calibrator's parameter trajectory
per arm — **a difference between arms is only attributable to calibration if a parameter actually
moved**, and the ON arm starting from a deleted state file is what makes that checkable.

## What counts as an answer

- **Yes**: injected block shows ON below OFF on RPE translation *and* rotation, outside the A/A′
  spread, with `k_omega` visibly moving toward 1/1.03 during the ON arm.
- **No / not detectable**: injected block shows no separation. Then the healthy block is not worth
  interpreting, and the finding is about the pipeline, not the calibrator.
- **Yes, but not on this robot**: injected block separates, healthy block does not. That is the
  most likely outcome given `k_v` = 1.00004, and it is a **legitimate, publishable result** — the
  calibration works and this robot has nothing much to calibrate. State it that way rather than
  hunting for a favourable window.
