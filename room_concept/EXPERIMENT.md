# Motion self-calibration — the experiment

**This is the single source. It replaces `EXPERIMENT_CALIB_PARAMS.md`,
`EXPERIMENT_CALIB_LOCALIZATION.md`, `EXPERIMENT_RECORD_SHADOW_BASELINE.md` and
`CALIB_UNMEASURED_EPISODES.md`, all deleted.** Analysis: `tools/calib_localization_ab.py`.
Live readout: the localization plot on the canvas (`src/localization_drift.h`).

The plan in §4 is **frozen**. §9 states the only thing that may reopen it.

---

## 1. What is being tested

Not "can a robot be calibrated" — **lifelong opportunistic self-calibration**. A mount creeps over
a robot's life; a factory calibration stage is exactly the artificial step that widens the sim2real
gap. So there is no calibration stage and no scripted manoeuvre (`CalibPivotEnabled = false`): the
estimator harvests whatever the robot's ordinary driving happens to make identifiable.

Two claims, routinely confused, and only the second is open:

1. **The estimator recovers a real parameter.** Settled on P3Bot, 2026-08-22/23: injected gyro
   error recovered to **0.07%**, injected wheel error to **83%**.
2. **Applying what it learns makes the robot better localised.** Open. Arm 3.

## 2. The inverse model: parameter → motion

Six parameters, solved jointly (`src/calibration_estimator.h:56-71`), identifiable because each
lands on a **different component** of the same correction against a **different covariate**:

| parameter | component | covariate | identified by |
|---|---|---|---|
| `k_v` | forward | distance | straight driving |
| `eps_yaw` | **lateral** | distance | straight driving |
| `k_omega` | heading | **rotation** | turning |
| `b_omega` | heading | **elapsed time** | turning at **varied rates** |
| `k_lat` | lateral | **lateral** distance | strafing — impossible on a differential base |
| `dk_wheel` | heading | distance | straights at **varied speeds** |

★ **Three land on the same component and are separated ONLY by covariate.** A pivot at one rate
makes rotation proportional to time, and no estimator can split `k_omega` from `b_omega` — the
self-test's normalised condition number goes 14.5 → 216.4 on exactly that degeneracy
(`calib_pivot.h:45-51`). **Varying speed is the only handle on the time axis**, because a parked
robot never closes an episode.

★★★ This is why the routes are ordinary navigation and not manoeuvres. The controller's own
curvature law — `v = sqrt(a_lat/kappa)` plus the sharp-turn slowdown — makes tight turns slow and
open runs fast, so **the cluttered apartment supplies the rate diversity a fixed-rate pivot cannot**.
A clear arena would give long fast straights and gentle turns: less diversity, not more.

`k_lat` is the standing **negative control**: on a differential base its covariate is identically
zero, so it must never leave its prior. If it moves, parameters are leaking into each other.

## 3. Setup

**Robot.** Shadow, **differential** (`ShadowDiff.proto`), Webots `piso.wbt` — a furnished
apartment. Base capability from the base component's own config
(`SVD48VBase/etc/config_diferential.toml`, the file the real robot runs): `maxLinSpeed` 900 mm/s,
`maxRotSpeed` 2 rad/s, `wheelRadius` 100 mm, `axesLength` 518 mm, `baseType` Differential.

**Bridge.** `webots-bridge` → `Webots2Robocomp`, config `etc/config` (⚠ four config files live in
that directory; `etc/config` is the one on the command line). Acting as Webots supervisor.

**Sensors.** `helios` 3-D LiDAR; `ricoh` 360 panorama (driving camera for the RGB corner channel);
`zed` pinhole (calibration channel only); IMU gyro; wheel odometry.

**Localiser.** `room_concept`, SDF fit against the room model, Gauss-Newton backend, 20 Hz.
Measured early-exit rate **99.1%**, and `|published − predicted|` is exactly zero on 99.8% of those
cycles — so the published pose **is** the raw prediction, and model error reaches it directly.

**Calibrator.** Joint batch estimator over a 512-episode window. Priors: sigma 0.02 on the scales,
0.0175 rad on `eps_yaw`, 5e-4 rad/s on `b_omega`. Prior *means* are nominal and **not settable** —
`Prior` carries sigmas only, and restoring a fitted value as a prior mean is deliberately refused as
a ratchet. So "start the parameters wrong" is not available: an error must be put in the **world**.

**Ground truth.** `robot_gt_{x,y,angle}`, from the Webots supervisor via `robot_concept`.
Independent of the bridge's velocity path. **Simulation only.**

**Injection.** `SensorNoise.*` in `webots-bridge/etc/config`. Ground truth **survives** it: the
corruption is applied to `velocity_local.x/y` and `rot_velocity` (`specificworker.cpp:799-825`) and
the gyro `gz` (`:2140-2143`), while `pose_data` comes from the supervisor untouched (`:828-838`).
The startup banner "this bridge is NO LONGER publishing ground truth" means the *odometry*, not the
pose. Available knobs: `WheelScaleV` → `k_v`; `GyroBias` → `b_omega`; `WheelScaleW` → nearly inert
(the gyro carries ~99% of heading). **There is no gyro *scale* knob**, so `k_omega` cannot be
injection-tested without writing one.

**Routes.** `controller/etc/missions.toml`. `calib straight` (the y ≈ −1.5 corridor, out and back)
for the distance-regressed parameters; `calib turns` (kitchen alcove) for rotation and time. Every
waypoint is copied from a mission already driven — a straight line between two valid waypoints is
not necessarily clear of furniture.

## 4. The plan — THREE ARMS, FROZEN

| arm | config | question |
|---|---|---|
| **1 Baseline** | cold, calib **ON**, `calib straight` | what are Shadow's native parameters? |
| **2 Injection** | cold, calib **ON**, `WheelScaleV = 0.03`, sigmas zeroed | does it recover a **known** error? |
| **3 Localization A/B** | cold each, calib **OFF** then **ON**, back to back | does applying it **improve the pose**? |

Arm 2's pre-registered prediction: **`k_v` → 0.991556 / 1.03 = 0.9627.**
Arm 3's endpoint: **`pred drift` mm/m** from `tools/calib_localization_ab.py`.

Pass for arm 2 requires all three: the parameter reaches its target; no other parameter moves more
than 2 of its own sigmas; and `k_lat` stays at its prior. A parameter that converges beautifully
while the localisation metric does not move means that parameter was not carrying the error — and
that contradiction would be the finding, not a failure.

★ **Why the endpoint is a RELATIVE pose error.** Ground truth is the world frame; the estimate is
the ROOM frame, whose orientation room_concept picks from its own fit and which differs between
runs. `|est − gt|` is dominated by that arbitrary choice and changes when nothing about the
localiser changed. Comparing motion over a fixed span drops the frame entirely — and increments are
exactly what these parameters act on. Normalised by motion, never by time: a parked robot predicts
nothing and would otherwise score perfectly for standing still.

## 5. Standing method rules

Each of these has already produced a false finding here.

- **Delete the evidence with the agent STOPPED, before every arm.** Four files:
  `motion_calib_state.csv`, `camera_calib_<robot>_{ricoh,zed}.txt`, `image_edge_mount.csv`. They
  persist across restarts and the agent rewrites them every window, so deleting under a live process
  just restores the warm one. An arm inheriting a warm window is not the arm it claims to be.
- **Never compare two sessions.** `opt/m` showed a **5× spread** between two runs with identical
  configuration. Arms are back to back inside one session or they are not comparable.
- **One driver.** The controller and the xbox pad must not both drive; the route stops being
  reproducible. (The pad is safe when untouched — it returns before publishing after 5 all-zero
  cycles — but any use of it during an arm invalidates that arm.)
- **Separate parked from moving.** A parked robot predicts nothing and always looks accurate.
- **Exclude burst windows.** The not-tracking regime once dragged a healthy `k_v` from 1.0059 to
  **0.8907** in two 180 s windows.
- **`calib_eps` is not a sample size.** Many episodes close with almost no forward travel, so
  `H ≈ 0` for the distance-regressed parameters.
- **`opt/m` is not comparable across sessions; steepness (mm/m, mm/rad) is.**

## 6. Results

### Arm 1 — baseline. DONE, 2026-08-30.

47.2 min, **282.8 m**, 0.67 rad/m, early exit 99.1%. Episodes: 292 emitted, 203 carried, 17 dropped.
Cold start verified.

| parameter | value | sigma | prior | reduction |
|---|---|---|---|---|
| `k_v` | **0.991556** (−0.84%) | 0.00414 | 0.020 | 4.8× |
| `k_omega` | **0.994682** (−0.53%) | 0.00715 | 0.020 | 2.8× |
| `eps_yaw` | **−0.122°** | 0.223° | 0.573° | 2.6× |
| `b_omega` | −6e-6 rad/s | — | 5e-4 | — |

`informed` = **15 — all four**, against 5 (two) before the Appendix A fix. Condition number **1.12**,
so they are genuinely separated. `k_lat` stayed at its prior throughout ✅.

**Shadow has a real −0.84% forward scale error, where P3Bot had none** (`k_v` = 1.00004 ± 0.003).

Localisation over the same run, 498.5 m / 25 windows: **RPE translation 31.8 mm/m**, RPE rotation
1.52 deg/rad, aligned ATE 36.8 mm. (`k_v` accounts for ~8 mm/m of that, heading for ~17 mm/m.)
⚠ Reference figure only — the calibrator was converging throughout, so it is not a matched control
for arm 3.

### Cross-robot comparison

| | P3Bot (2026-08-23) | Shadow (arm 1) |
|---|---|---|
| `k_v` | 1.00004 ± 0.003 | **0.991556 ± 0.00414** |
| `k_omega` | 0.9974 ± 0.002 | **0.994682 ± 0.00715** |
| `eps_yaw` | −0.536° ± 0.13 | **−0.122° ± 0.223** (not converged) |

⚠ Not controlled: different robot, drive type, route and localiser regime.

### Arms 2 and 3 — not yet run.

## 7. Caveats on the arm 1 numbers

1. **The route is turn-heavy** — 0.67 rad/m, so the corridor's U-turns dominate its 6.6 m straight
   legs. `k_v` and `eps_yaw` are identified from the tail of the longer straights, not the bulk.
2. **`eps_yaw` is not converged** — still moving when the run ended. It neither confirms nor
   contradicts P3Bot's −0.536°.
3. **The window was warm across one restart** mid-session (~61 episodes carried over).
4. **17 spans (≈6%) hit the 2.5 m linearisation cap.** Not distorting anything yet, but the cap sits
   close to the operating point.

## 8. Parked — deliberately NOT in the critical path

The SDF-polish A/B (Appendix B); the camera/mount block; injections for `k_omega`, `eps_yaw` and
`dk_wheel` (their bridge knobs do not exist and would have to be written); the pose-free three-device
extrinsics design (`DESIGN_THREE_DEVICE_EXTRINSICS.md`, kept separate because it is a design, not
part of this experiment); and the window retention policy.

## 9. What may reopen the plan

**Only a finding that makes the current arms uninterpretable**, as Appendix A did. Not a finding
that is merely interesting. Anything else goes on the parked list and is reported after the run.

This rule exists because we broke it: the plan mutated three times on 2026-08-30 and therefore
nearly never ran.

---

## Appendix A — the empty-episode defect (fixed, `96d48bc`)

Necessary because it dates the validity of every calibration number.

At 98.4% early exit the optimiser ran on 1.6% of cycles, but episodes closed on **motion**. A span
that saw no correction was emitted as *"the correction was exactly zero"*:

| | zero-correction | real correction |
|---|---|---|
| episodes in window | **438 (86%)** | 74 |
| median `pos_var` | **0.004945** | 0.007436 |
| median `\|d_forward\|` | **0.2512 m** | 0.1592 m |

**94.2% of the Fisher information** on a distance-regressed parameter came from episodes in which
nothing was measured.

★ **The safeguard had inverted.** With no correction, `acc_pos_var_` is 0 and the variance is set
entirely by `fit_model_gain * max|SDF|` — and an early-exit cycle is *by definition* one whose SDF
residual was small. The term added on 2026-08-23 to distrust bad fits had become the one that
trusted unmeasured episodes most.

★ **Cause: a config key that did not survive a merge.** `eb9fbab` (08-26) introduced
`SdfPolishOnEarlyExit` **and** moved the episode trigger to motion as one coherent design — the
polish corrects every cycle, so every episode carries a correction. But the key existed **only in
`config_p3bot_webots.toml`**; `config_shadow_webots.toml` never had it, and `b6c40b4` (08-29 16:43)
merged the configs and kept it in neither, so it fell back to `sdf_polish_enabled = false`.
So the invalid window is **per robot**: Shadow from 08-26 continuously; P3Bot only from `b6c40b4`.
The ceiling/covariance fixes decided how *much* of the window went empty, not that it did.

★ **Immune:** everything before 08-26. Until `eb9fbab` an episode closed only on the falling edge of
"the optimizer ran", which cannot produce an episode with no correction in it. That covers the whole
P3Bot record of 08-22/23 — the 262 m A/B, `k_v` 1.00004, `k_omega` 0.9974, `eps_yaw` −0.536°. And
`eps_yaw` is doubly safe: the straight-episode least squares giving −0.532° does not use the episode
window at all.

★ **Fix:** an episode emits a row only if at least one corrected cycle contributed; otherwise its
motion is **carried forward**, not discarded. Not a threshold — `corrected` is a boolean fact about
whether the optimiser ran. It restores the right *pairing*: the correction IS the drift accumulated
since the last correction, so it belongs with the motion accumulated since the last correction.
Widening `R` was tested and rejected — those rows already carried `sigma_pos` 0.070 m, about the
gate width, and 438 assertions of zero outvote 74 measurements whatever the variance.
`carried` / `dropped` counters now ship in `gt_error.csv`, because invisibility was the failure mode.

★ **The lesson worth keeping:** the same starvation was previously LOUD (every parameter exactly
0.000 with sigma 0.000 — noticed and fixed on 08-26) and became SILENT (plausible values, shrinking
sigmas, apparent convergence). *It got quieter as it got worse.*

## Appendix B — the SDF polish (parked)

`SdfPolishOnEarlyExit` takes one Gauss-Newton step on early-exit cycles using the optimiser's own
`SdfFactor` — same query, same weights, same IRLS Huber, same Jacobian. Without it the gate's
verdict is implemented as **do nothing**, so the pose is dead reckoning on ~99% of cycles.

It is **off by accident**, not decision (Appendix A). Still fully wired: `room_config.cpp:270`,
gated blocks at `room_concept.cpp:3231` and `:3396`.

★ **Not a pose-only switch.** It carries the SHRINK half of the per-cycle covariance recursion; the
GROWTH half is gated on it because running growth alone once lost the track (`sdf_mse` 0.026 →
0.4611, early exit to 21%, `cov_tt` 7.1 rad²). *A one-sided recursion is not a recursion.*

★ **Its published validation does not cover the shipped code**: growth exponent 0.50 → 0.167 was
measured on an earlier *scalar* form; the Gauss-Newton form "has NOT yet been measured".

★ **And its motivating figure was largely a noise artefact**: the 210 mm parked wander came from a
flat noise density injecting 0.118 m/s into a stationary robot; making `sigma(v)` speed-dependent
took parked drift from **3.49 to 0.122 mm/cycle** independently of the polish.

If run: `MotionCalibEnabled = false` in **both** arms, or the polish's effect on the pose and its
flooding of the calibrator with dense corrections are unattributable. OFF-arm reference already
measured at 31.8 mm/m (§6). Repeat the parked-wander test alongside.
