# Experiment record — Shadow motion self-calibration baseline

2026-08-30. Written for transcription into the thesis. Companion to
`thesis-online-motion-calibration-experiment` (the P3Bot record, 2026-08-22/23), which this does
**not** supersede: it is the same method on a different robot, after a defect whose invalid window
is per robot and is dated precisely in §2 — Shadow from 2026-08-26, P3Bot only from `b6c40b4` on
2026-08-29. Work before 2026-08-26 is structurally immune (§3a).

Everything below is measured, with the sample size. Nothing is projected.

---

## 1. Setup

**Robot.** Shadow, **differential** drive (`ShadowDiff.proto`), in Webots `piso.wbt` — a furnished
apartment, not a clear arena. Base capability from the base component's own config
(`SVD48VBase/etc/config_diferential.toml`, the same file the real robot runs):

| | |
|---|---|
| `baseType` | Differential |
| `maxLinSpeed` | 900 mm/s |
| `maxRotSpeed` | 2 rad/s |
| `wheelRadius` | 100 mm |
| `axesLength` | 518 mm |

**Bridge.** `webots-bridge` (binary `Webots2Robocomp`, config `etc/config`), acting as a Webots
supervisor. Synthetic sensor error **on**, with **no scale or bias injected** — this is the
uninjected baseline arm:

```
SensorNoise.Enabled          = true
SensorNoise.WheelSigmaVFloor = 0.0005 m/sqrt(s)      SensorNoise.WheelSigmaWFloor = 0.0010 rad/sqrt(s)
SensorNoise.WheelSigmaV      = 0.006  m/sqrt(s)      SensorNoise.WheelSigmaW      = 0.010  rad/sqrt(s)
SensorNoise.WheelScaleV      = 0.0                   SensorNoise.WheelScaleW      = 0.0
SensorNoise.GyroSigma        = 0.002  rad/sqrt(s)    SensorNoise.GyroBias         = 0.0
```

The wheel sigmas are **speed-dependent**, `sigma(v) = sqrt(floor^2 + (k_slip*|v|)^2)`: an encoder at
rest emits no counts, so a flat density injects motion into a stationary robot.

**Sensors.** `helios` 3-D LiDAR; `ricoh` 360 panorama (the driving camera for the RGB corner
channel); `zed` pinhole (calibration channel only); IMU gyro; wheel odometry.

**Localiser.** `room_concept`, SDF fit against the room model, Gauss-Newton backend
(`OptimizerType = "GN"`), 20 Hz. The optimiser is SKIPPED when the predicted-pose residual passes
the early-exit gate. **Measured early-exit rate: 99.1%**, so the published pose is the raw
prediction on essentially every cycle (`|published - predicted|` is exactly zero on 99.8% of
early-exit cycles).

**Calibrator.** `rc::calib::BatchEstimator`, joint solve over a 512-episode window, six parameters
identified by *component x covariate*: `k_v` (forward|distance), `eps_yaw` (lateral|distance),
`k_omega` (heading|rotation), `b_omega` (heading|time), `k_lat` (lateral|strafe — structurally
unexcitable on a differential base, a standing negative control), `dk_wheel` (heading|distance).
Priors: sigma 0.02 on the scales, 0.0175 rad on `eps_yaw`, 5e-4 rad/s on `b_omega`.
`MotionCalibEnabled = true`; `CalibPivotEnabled = false` for Shadow, so **no calibration manoeuvre
ran** — this is ordinary navigation throughout.

**Ground truth.** `robot_gt_{x,y,angle}` published on the robot node by `robot_concept`, taken from
the Webots supervisor. Independent of the bridge's velocity path. Simulation only.

**Route.** Mission `calib straight` (`controller/etc/missions.toml`) — the y ≈ −1.5 corridor driven
out and back, 6 loops per pass, every waypoint copied from a mission already driven. Driven by
`bin/controller`; the xbox pad was up but publishes nothing when idle (it returns before publishing
after 5 all-zero cycles), so there was one writer on the base.

**Initial state.** Cold. All four evidence files deleted with the agent stopped, and verified
absent before restart: `motion_calib_state.csv`, `camera_calib_Shadow_{ricoh,zed}.txt`,
`image_edge_mount.csv`. Evidence persists across restarts, so this matters: an arm that inherits a
warm window is not the arm it claims to be.

---

## 2. ⚠ The defect this baseline had to be taken after

Recorded here because the baseline is meaningless without it, and because the failure mode is the
transferable part.

**★ CORRECTED 2026-08-30 (this supersedes the dating in the first version of this record).** The
proximate cause is not the ceiling/covariance fixes. It is a config key that did not survive a merge.

`eb9fbab` (2026-08-26) introduced `SdfPolishOnEarlyExit` — one Gauss-Newton step on early-exit
cycles — *and* moved the episode trigger to motion, as one coherent design: the polish corrects on
every cycle, so every episode carries a correction. The commit says so explicitly ("the polish now
counts as a corrector").

Verified in git: `SdfPolishOnEarlyExit` appeared **only in `config_p3bot_webots.toml`** (2
occurrences). `config_shadow_webots.toml` never contained it. `b6c40b4` (2026-08-29 16:43, "one file
for both robots") merged the two configs and the key survived in neither, so it now falls back to
its declared default, `room_concept.h:490` — `sdf_polish_enabled = false`.

Therefore the invalid window is **per robot**:

| | polish | episodes carried a correction? |
|---|---|---|
| **P3Bot**, 2026-08-26 → `b6c40b4` | ON | yes — mechanism NOT active |
| **P3Bot**, after `b6c40b4` | OFF | no — mechanism active |
| **Shadow**, any time after 2026-08-26 | **never enabled** | no — mechanism active throughout |

So on Shadow the empty-row regime began with the 08-26 trigger change and has been continuous since;
on P3Bot it began at `b6c40b4`. Work predating 2026-08-26 is structurally immune (see §3a).

⚠ The polish is still OFF as of this record. That is a live configuration question, not something
this fix settles: the carry-forward change makes an absent corrector *safe*, it does not restore the
corrector.

At 98.4% early exit the optimiser ran on 1.6% of cycles, but episodes closed on **motion**
(0.25 m / 0.20 rad). A span that saw no correction was emitted as *"the correction was exactly
zero"*:

| | zero-correction | real correction |
|---|---|---|
| episodes in window | **438 (86%)** | 74 |
| median `pos_var` | **0.004945** | 0.007436 |
| median `\|d_forward\|` | **0.2512 m** | 0.1592 m |

Fisher information on a distance-regressed parameter: **94.2% from episodes in which nothing was
measured.**

★ The safeguard had inverted. With no correction, `acc_pos_var_` is 0 and the variance is set
entirely by `fit_model_gain * max|SDF|` — and an early-exit cycle is *by definition* one whose SDF
residual was small. The term added on 2026-08-23 to distrust bad fits had become the one that
trusted unmeasured episodes most.

★ Cause: the episode trigger was moved from the optimiser's falling edge to motion on 2026-08-26,
on the premise that "the SDF polish now corrects a little on EVERY cycle" — true when written, false
once the key was lost (above). The ceiling-band and covariance fixes (`8164fc3`, `f2336d0`) took
early exit from 0 to ~98% and so decided how MUCH of the window went empty; they did not create the
mechanism.
**Fixing the localiser starved the calibrator**, and the trigger change turned the starvation from
loud (every parameter exactly 0.000, sigma 0.000 — noticed and fixed on 2026-08-26) into silent
(plausible values, shrinking sigmas, apparent convergence). *The failure got quieter as it got
worse.*

★ Fix (`96d48bc`): an episode emits a row only if at least one corrected cycle contributed; its
motion is otherwise **carried forward**, not discarded. This restores the physically correct
pairing — the optimiser's correction IS the drift accumulated since the last correction, so it
belongs with the motion accumulated since the last correction. Widening `R` was tested and rejected:
those rows already carried `sigma_pos` 0.070 m, about the early-exit gate width, and 438 assertions
of zero outvote 74 measurements whatever the variance.

### 3a. What is NOT affected

Work before 2026-08-26 is structurally immune, and the reason is not a judgement call: until
`eb9fbab` an episode closed **only on the falling edge of "the optimizer ran"**, which by
construction cannot produce an episode without a correction in it. The empty-row failure requires
the motion trigger.

That covers the P3Bot record of 2026-08-22/23 in full — the 262 m ON vs 105 m OFF A/B, the injected
error tests, `k_v` = 1.00004 +/- 0.003, `k_omega` = 0.9974 +/- 0.002 and `eps_yaw` = −0.536 +/- 0.13
deg. `eps_yaw` is doubly safe: the straight-episode least squares that independently gave −0.532 deg
is a different method that does not use the episode window at all.

★ And the flattening signature does not apply to P3Bot readings near 1.0. P3Bot's true forward scale
IS ~1.0000, established independently by the injected-error test (an injected 3% was recovered to
83%, which is what proves the forward channel observable and `k_v` = 1.00004 a genuine null rather
than a blind spot). A P3Bot `k_v` of 0.9994 is a small honest deviation, not a flattened 0.84%.

---

## 3. Result

**47.2 min, 282.8 m driven, 0.67 rad/m, early exit 99.1%. Episodes: 292 emitted, 203 carried,
17 dropped at the 2.5 m linearisation cap.**

| parameter | value | sigma | prior sigma | reduction |
|---|---|---|---|---|
| `k_v` | **0.991556** (−0.84%) | 0.00414 | 0.020 | 4.8x |
| `k_omega` | **0.994682** (−0.53%) | 0.00715 | 0.020 | 2.8x |
| `eps_yaw` | **−0.122°** | 0.223° | 0.573° | 2.6x |
| `b_omega` | −6e-6 rad/s | — | 5e-4 rad/s | — |

`informed` = **15 — all four parameters**, against 5 (two) before the fix. Normalised condition
number **1.12**, so they are genuinely separated rather than a ridge being split arbitrarily.
`k_lat` correctly remained at its prior throughout (the negative control: a differential base cannot
strafe, so its covariate is identically zero).

Convergence: `k_v` stepped early and held at 0.9913–0.9916 across the final third; `eps_yaw` was
still moving at the end (0 → −0.036 → −0.119 → −0.122) and should not yet be quoted as converged.

**Shadow has a real forward odometry scale error of −0.84%, where P3Bot had none** (`k_v` =
1.00004 ± 0.003). The defective estimator reported −0.04% — it flattened a genuine 0.84% error into
nothing, which is the harm the empty rows were doing, quantified.

### Comparison across robots

| | P3Bot (2026-08-23) | Shadow (this record) |
|---|---|---|
| `k_v` | 1.00004 ± 0.003 | **0.991556 ± 0.00414** |
| `k_omega` | 0.9974 ± 0.002 | **0.994682 ± 0.00715** |
| `eps_yaw` | −0.536° ± 0.13 | **−0.122° ± 0.223** (not converged) |

⚠ Not a controlled comparison: different robot, drive type, route and localiser regime. The P3Bot
figures were taken when the optimiser fired every 1–6 s and the window was full of real corrections.

---

## 4. Caveats, stated before anyone reads a number into these

1. **The route is turn-heavy.** 0.67 rad/m on moving rows, so the corridor mission's U-turns
   dominate its 6.6 m straight legs. `k_v` and `eps_yaw` are identified from the tail of the longer
   straights, not the bulk of the run.
2. **`eps_yaw` is not converged.** It was still moving at the end of the run.
3. **The window was warm across one restart** mid-session (~61 episodes carried over). The cold
   start above applies to the beginning of the sequence, not to every reading.
4. **Episode counts overstate distance evidence.** Median `|d_forward|` per emitted episode is
   small; many close on the falling edge with almost no forward travel and therefore `H ≈ 0` for the
   distance-regressed parameters. `calib_eps` is not a sample size.
5. **`opt/m` is not comparable across sessions** (5x spread on identical configuration, 2026-08-22).
   Steepness in mm/m and mm/rad is the cross-session-stable figure.
6. **17 spans (≈6%) hit the 2.5 m linearisation cap.** Not distorting anything yet, but the cap sits
   close to the operating point and would bite if the early-exit rate rose further.

---

## 5. The SDF-polish A/B — pre-registered, OFF arm already measured

### Why this exists

The early-exit gate's verdict is "good enough not to need a full solve", implemented as **do
nothing**. So on every cycle it passes, the published pose is dead reckoning with no absolute
reference. `SdfPolishOnEarlyExit` (`eb9fbab`, 2026-08-26) takes ONE Gauss-Newton step on those
cycles using the optimiser's own `SdfFactor` — same query, same observation weights from
`room_obs_weights.h`, same IRLS Huber, same Jacobian — against the newest slot only. Sharing the
weights is deliberate: a polish weighting its points differently from the solver is a second
estimator that can disagree with the first. `delta = -(H_sdf + Lambda_prior)^-1 b`; at the
prediction the prior's residual is zero so it contributes to `H` alone, a Levenberg term in metres
and radians rather than a bare number, and the step is bounded by the motion the prior calls
plausible. Nothing in it is tuned.

★ **This is the strongest case the polish will ever have.** At a 1.6% optimiser firing rate the pose
runs open-loop for ~3 s (≈1.2 m at 0.4 m/s) between absolute references. If it helps at all, it
must show here.

⚠ It is currently **OFF**, and not by decision: the key existed only in `config_p3bot_webots.toml`
and did not survive the `b6c40b4` merge (§2). It is still fully wired — `room_config.cpp:270` reads
`RoomConcept.SdfPolishOnEarlyExit`, gated blocks at `room_concept.cpp:3231` and `:3396`.

### ★ It also carries half of a covariance recursion — do not treat this as a pose-only switch

`current_covariance` is assigned only on the optimised path, so on every other cycle it carried a
stale value or its `Identity*0.1` initialiser, and two consumers read that placeholder as a
measurement: the polish's own regulariser, and **the calibrator's episode weight** — recorded at
`pos_var` = exactly 1.000000 m², a sigma of one metre. That made 2994 episodes weigh 1 against 6
optimised ones weighing 1269, so reaching `informed` would have needed ~9408 episodes, about 2.4 km.

The per-cycle covariance GROWTH (the ordinary Kalman prediction step) is therefore gated on the
polish, because the SHRINK half lives inside the polish block. Running growth alone once lost the
track outright: `sdf_mse` 0.026 → 0.4611, early exit down to 21%, `cov_tt` peaking at 7.1 rad².
*A one-sided recursion is not a recursion: growth and shrink ship together or not at all.*

So turning the polish on also changes the published covariance and everything downstream of it.

### Design — and the confound it is built to avoid

★★★ **`MotionCalibEnabled = false` in BOTH arms.** Turning the polish on changes two things at
once: it constrains the pose, AND it makes `corrected` true on essentially every cycle, which floods
the calibrator with dense polish corrections instead of sparse optimiser ones. A localisation
improvement would then be unattributable between the two. With calibration off, the polish is the
only difference.

| arm | `SdfPolishOnEarlyExit` | `MotionCalibEnabled` | route |
|---|---|---|---|
| P-OFF | false | **false** | `calib straight` |
| P-ON | true | **false** | `calib straight` |
| P-OFF′ | false | **false** | `calib straight` |

Endpoint: `pred drift mm/m` from `tools/calib_localization_ab.py`. Secondary: RPE rotation, aligned
ATE, and the early-exit percentage itself (the polish should not change what the gate decides, and
if it does, that is a finding).

### The OFF arm, already measured

Taken during the baseline run of §3 — 498.5 m over 25 windows, polish off:

| metric | value |
|---|---|
| **RPE translation** | **31.8 mm/m** |
| RPE rotation | 1.52 deg/rad |
| aligned ATE | 36.8 mm |

Decomposition, roughly: `k_v` = 0.992 accounts for ~8 mm/m, and heading for ~17 mm/m
(1.52 deg/rad x 0.67 rad/m ≈ 1° per metre ≈ 17 mm of cross-track). The remainder is the open-loop
drift between corrections — which is what the polish exists to remove.

⚠ This arm ran with `MotionCalibEnabled = true` and the calibrator converging, so it is a *reference
figure*, not a matched control. The pre-registered P-OFF arm above must be re-run with calibration
off before P-ON is compared against it.

### Also repeat the original validation

The polish's published effect — growth exponent 0.50 → 0.167, span/√N 4.53 → 1.03, "the wander
creeps instead of diffusing" — was measured on an **earlier SCALAR form** (step `L/|g|²` on the mean
residual), which shivered because it is the step that would zero a linear residual and overshoots a
curved one. **The Gauss-Newton form that actually shipped has never been measured.** So repeat the
original test alongside the A/B: robot stationary, ground truth motionless, measure `est` drift per
cycle over a few thousand cycles.

★ Note the motivating wander has already been attacked from the other side: the flat noise density
was injecting 0.118 m/s into a demonstrably stationary robot, and making `sigma(v)` speed-dependent
took parked drift from **3.49 to 0.122 mm/cycle**. So the 210 mm figure that motivated the polish
was largely a noise-model artefact. Whether the polish still earns its place after that fix is an
open question this A/B answers, and a null result would be a legitimate one.

---

## 6. The next arm, pre-registered

Injected-error validation, **which has not been re-run since the early-exit regime changed** — and
which is precisely the check that would have caught the defect in §2.

Set `SensorNoise.WheelScaleV = 0.03` in `webots-bridge/etc/config` (ground truth survives the
injection: the corruption is applied to the published velocities and the gyro, while `pose_data`
comes from the supervisor untouched). Zero the sigmas. Delete the four evidence files. Same route.

**Prediction: `k_v` converges to 0.991556 / 1.03 = 0.9627.**

Pass requires all three: the parameter reaches that target; no other parameter moves more than 2 of
its own sigmas; and `pred drift mm/m` (`tools/calib_localization_ab.py`) falls between the OFF and
ON arms. A parameter that converges beautifully while the localisation metric does not move means
that parameter was not carrying the error — and that contradiction would be the finding.
