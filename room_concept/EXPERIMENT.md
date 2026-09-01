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

### Arm 2 — injection. DONE, 2026-08-31. PASSES.

Run PAIRED, back to back in one session, `WheelScaleV = 0.03` on the injected leg and the sensor
sigmas zeroed on both (this arm tests recovery of a BIAS, so the noise is removed rather than
averaged over):

| leg | `k_v` |
|---|---|
| uninjected | 0.976630 |
| injected (`WheelScaleV = 0.03`) | 0.946424 |
| **difference** | **+0.030206** |

The injected truth is `1 − 1/1.03 = 0.029126`, so the estimator recovered **104.2%** of it. ✅

★ **Use the PAIRED form.** It needs no absolute baseline, which matters because the estimator
(3.4% deviation) and the direct odometry/GT ratio (~1.1%) still DISAGREE by ~3x on the absolute
forward scale on this turn-heavy route. That disagreement is unresolved (§10) and it does not touch
the paired difference, because whatever it is, it is present in both legs and subtracts out.

⚠ The pre-registered target in §4 (`k_v → 0.9627`) was written against `0.991556`, a pre-fix value
that the two estimator defects biased low. It is void. The paired difference above is the test that
was actually available once those defects were fixed, and it is the stronger one.

### Arm 3 — localization A/B. DONE, 2026-08-31. NULL on accuracy, 3-4x on EFFORT.

Three-way, matched inside one session (203 / 215 / 208 m, 10-11 windows each, **burst windows = 0
in all three**):

| arm | optimiser firing | odom correction (mm/m) | RPE trans (mm/m) | RPE rot (deg/rad) |
|---|---|---|---|---|
| calibration **OFF** | **4.92%** | 48.99 | 37.56 | 1.649 |
| **ON**, all params (`mask = -1`) | **1.74%** | 38.79 | 37.23 | 1.763 |
| **ON**, `k_v` only (`mask = 1`) | **1.11%** | 43.51 | 37.86 | 1.824 |

**Accuracy: null.** No pairwise |t| above 1.37 on either RPE channel — nothing significant.
**Effort: 3-4x.** Optimiser firing falls 4.92% → 1.74% → 1.11%.

★★★ **THE RESULT, and the framing it needs.** The SDF localiser was already removing Shadow's
odometry scale error either way, so RPE measures the CORRECTOR, not the calibration. What
calibration changes is how hard the corrector has to work: a better-calibrated prediction lands
inside the early-exit band more often, so the expensive Gauss-Newton solve is needed on a third to a
quarter as many cycles. **Quote the claim with its dose: at a ~2-3% initial scale error,
calibration buys EFFORT, not accuracy.** This was the concern raised at the very start of the design
and then forgotten for a day — a corrector doing its job absorbs model error, so grading calibration
on corrected output grades the corrector.

⚠ **RETRACTED — "applying heading corrections harms the localiser."** The 2026-08-30 night figure
(rotation RPE 1.699 → 2.955 deg/rad, t = −2.45) is REFUTED by the matched re-run above: 1.649 →
1.763, t = −0.68. The old ON leg carried **6 burst windows against 0**; it was simply a worse run,
not a treatment effect. One bad run dressed as a treatment effect. `MotionCalibApplyMask = 1` is
therefore no longer justified by that finding — it stands only as the lowest-effort arm.

### Arm 4 — DOSE RESPONSE at a larger initial error. Pre-registration, written 2026-08-31 BEFORE the run.

★ Kept verbatim. Both endpoints below answered the OPPOSITE of what is predicted here, and the
reason turned out to be the instruments rather than the robot — see the results section that
follows. A pre-registration that was wrong is worth more on the page than one quietly amended.

Arm 3's result is conditional on the dose. Arm 4 asks the next question directly: **does the effort
saving grow with the initial error, and does accuracy eventually separate?**

**Dose.** `WheelScaleV = 0.10` (reported forward velocity = true × 1.10, true `k_v` = 1/1.10 =
0.9091), ~4x Shadow's native −0.84% and ~3.4x arm 2's. Sensor noise stays at the BASELINE values —
this arm is compared against arm 3, whose runs had the full noise model live, so the only difference
from the live file is that one line. Config prepared: `webots-bridge/etc/config.toml.arm4-inj10`.

**Legs**, back to back in one session, cold each, evidence deleted with the agent stopped:

| leg | bridge | room_concept |
|---|---|---|
| 4-OFF | `config.toml.arm4-inj10` | `MotionCalibApply = false` |
| 4-ON | `config.toml.arm4-inj10` | `MotionCalibApply = true`, `MotionCalibApplyMask = 1` |

`mask = 1` because the injected error is exactly a forward-scale error: applying only `k_v` puts the
dose and the response on the same channel and avoids re-opening the (badly conditioned, and now
un-incriminated) heading parameters. A third leg at `mask = -1` is optional and answers a different
question — what normal operation does.

**Pre-registered endpoints, in this order.**

1. **PRIMARY — optimiser firing %.** Predicted: 4-OFF ≫ 4.92% (arm 3's OFF), 4-ON ≈ arm 3's ON
   (~1.1%) once converged, because a converged calibrator returns the prediction to native quality
   whatever the dose. The contrast should therefore be much wider than arm 3's 4.4x.
2. **SECONDARY — RPE translation (mm/m) and rotation (deg/rad)**, same instrument as arm 3
   (`tools/calib_localization_ab.py`, ds = 1.0 m, moving rows only). Predicted: this is where the
   null may BREAK. There is a dose at which the uncorrected prediction leaves the optimiser's
   convergence basin and the corrector can no longer absorb the error; arm 4 either finds it or
   raises the floor on where it is.
3. **THIRD, AND NEW — burst-window rate.** ★★★ **Arm 3's exclusion rule would delete the very effect
   a large dose is meant to produce.** "Exclude burst windows" was right when bursts were a nuisance
   regime unrelated to the treatment (arm 3 had 0 in every arm). At a large dose, losing tracking IS
   the treatment effect, and excluding it would report a null while the robot was failing. So: report
   burst windows as a COUNT per leg alongside the excluded-window analysis, never silently drop them.
   If 4-OFF bursts and 4-ON does not, that is the headline and RPE is a footnote.
4. **CONTROL — paired parameter recovery.** `k_v`(uninjected, arm 2 leg) − `k_v`(4-ON) should be
   **+0.0909**. Expect a few percent of shrinkage toward the prior: `MotionCalibScaleP0 = 4e-4` is a
   2% 1σ, so 9.1% is 4.5σ out, but arm 1 measured the data at ~23x the prior's precision, so the pull
   is ~4%. `MotionCalibScaleP0` is deliberately NOT widened — changing it would break comparability
   with arm 3 for the sake of a correction smaller than the effect.

⚠ **BLOCKER — `MapMode` must be `"given"`, and it is not right now.** The live `etc/config.toml`
carries an UNCOMMITTED `MapMode = "estimate"` (the wall-SLAM layout experiment, committed default is
`"given"`). Measured on the running agent 2026-08-31 over 155 s: `sdf_mse` median **0.55** against
`StableSdfMseMax = 0.076`, so the stability gate **cannot pass** — the optimiser fires on **100%** of
cycles at 46-94 ms median and the solver alone eats **81% of one core**. Arm 4's primary endpoint is
optimiser firing %; under `"estimate"` that endpoint is pinned at 100 and would measure the layout
estimator rather than the calibration. `tools/arm4_setup.sh` sets `given` on both legs. Arm 4 and the
wall-SLAM run cannot share the simulator.

**Verify the injection FROM BEHAVIOUR, not from the banner.** The odometry/GT distance ratio should
read ~1.10 (vs ~0.99 uninjected) and resolves within ~10 m of driving. A component reads its config
once at start-up: four runs on 2026-08-30 were driven believing an injection was live when the flat
`etc/config` it had been written into was not the file the bridge reads. `ps` shows
`Webots2Robocomp etc/config.toml` — **`etc/config.toml` is the live file**; the flat `etc/config`
is dead and still carries a stale `SensorNoise.WheelScaleV = 0.03`.

### Arm 4 — RESULTS, 2026-09-01. The endpoints INVERT, and the pre-registered ones were the wrong ones.

Two legs, one session, bridge never restarted between them: 199.4 m OFF, 205.8 m ON, 10 windows
each, matched on route composition (0.698 vs 0.753 rad/m of turning). Injection verified from
behaviour, not from a banner: odometry/GT ratio 1.1003 over the OFF leg against an injected 1.10.

| endpoint | 4-OFF | 4-ON | t | z | d |
|---|---|---|---|---|---|
| **correction load, mm/m** | **87.17** | **58.77** | **3.57** | **3.02** | **1.59** |
| correction per solve, mm | 30.00 | 18.46 | 3.36 | 2.87 | 1.50 |
| RPE translation, mm/m | 64.46 | 48.65 | 3.00 | 2.27 | 1.34 |
| aligned ATE, mm | 67.8 | 53.8 | 2.99 | 2.42 | 1.34 |
| RPE rotation, deg/rad | 1.982 | 1.758 | 1.46 | 1.36 | 0.65 |
| optimiser firing, % | 5.80 | 6.49 | **−0.60** | −0.83 | −0.27 |
| burst windows | 0/16 | 0/12 | — | — | — |

**Accuracy moved and effort did not — the exact inverse of arm 3.** Both pre-registered endpoints
answered, and both answered the opposite of the prediction written before the run.

**Control passed.** Recovery 92.8% of the injected 0.0909; `k_v` settled at 0.91566 against a
pre-registered ~0.913, the small shortfall being the predicted pull toward a prior centred at 1.0.

**The effort null is NOT the convergence transient**, which was the obvious alternative and is why
the per-window series was pre-registered. The ON leg's RPE falls 72.6 → 53.2 → 48.2 → 37.3 → 40.0 →
36.0 mm/m as `k_v` converges, reaching arm 3's NATIVE accuracy — calibration restores a 10%-broken
robot to as-new. Over those same windows its firing is 9.66, 6.82, 6.65, 7.14, 8.07, 10.16, 9.29,
8.48, 8.08. Flat. Firing does not fall as the model becomes correct.

**Third endpoint returned empty.** No burst windows in either leg. The localiser never lost tracking
at 10%, so there is no robustness finding; the corrector is more robust than predicted.

#### ★★★ WHY BOTH PRE-REGISTERED ENDPOINTS WERE THE WRONG INSTRUMENT

Neither RPE nor firing % can carry this experiment, and the reason is structural rather than bad luck.

- **RPE is measured DOWNSTREAM of the corrector.** It reports what the localiser managed to achieve
  after removing the model error, so it grades the corrector. That was already written down after
  arm 3 and then not acted on.
- **Firing % is an INDICATOR OVER A THRESHOLD.** It asks whether `iters > 0`, so it discards how hard
  each solve was and it saturates between 0 and 100. Measured directly: iterations per metre 23.61
  vs 22.69 (1.04x), iterations PER SOLVE 13.90 vs 13.27 (1.05x), achieved `sdf_mse` 0.0405 vs 0.0418.
  The optimiser removed 52% more error at the same price and reached the same residual.

**The reason the price is flat is a number nobody had looked at: the mean correction is 2.22 mm.**
At ~0.5 m/s and 10 Hz a 10% scale error produces ~5 mm of drift per cycle. Gauss-Newton converging
from 2.2 mm instead of 1.5 mm is the same 13 iterations. The experiment never left the regime where
the solve is trivially easy, so no cost endpoint could have moved. ★ A cost endpoint measured in a
regime with no cost is not a null result about cost.

#### The endpoint that works: CORRECTION LOAD

`|est − pred|` is what the optimiser had to remove from the motion model's guess. It is the
optimiser's **INPUT**, so the corrector cannot absorb it, and it is continuous rather than an
indicator, so it cannot saturate. It is now computed by the instrument
(`tools/calib_localization_ab.py`), not by hand.

Validated before adoption: on cycles that did not solve it is 0.06–0.19 mm (i.e. zero), and on
cycles that did it is 13.7–23.7 mm. Computed two ways that must agree on the effect and do — mean of
per-window ratios 87.17/58.77 = **1.48x**, ratio of whole-leg sums 48.38/31.82 = **1.52x**. The
levels differ because they are different estimators; the effect does not.

**And the dose-response the effort channel never showed is present here.** Reading arm 3's own table,
where this column sat unused:

| | correction OFF | correction ON (`mask = 1`) | REMOVED by calibration |
|---|---|---|---|
| arm 3, native ~2–3% | 48.99 | 43.51 | **5.5 mm/m** |
| arm 4, injected 10% | 48.38 | 31.82 | **16.6 mm/m** |

**3.0x more correction removed for ~3.4x more error** — near-proportional. The ~48 mm/m present in
BOTH uncalibrated legs is the irreducible part `k_v` cannot touch (heading, scene, noise), which is
also why the TOTAL looked insensitive to the dose and why only the removed part carries the signal.

⚠ The arm 3 row is a single figure from a run whose raw CSVs no longer exist. It cannot be windowed,
re-cut, or given a t-statistic, and it is a different session. It is a strong hint about where to
look, NOT a measurement — which is an argument for pre-registering correction load in arm 5, not for
rewriting arm 3 around it.

### Arm 5 — DOES THE COST APPEAR WHEN THE CORRECTOR IS SCARCE? Pre-registered 2026-09-01, not run.

**The question.** Arm 4 established that an uncalibrated model raises the correction LOAD by ~1.5x
while the PRICE of removing it stays flat, because at full corrector rate the per-cycle error is
~2 mm and the solve never leaves its easy regime. The cost of a wrong model should be paid when
error accumulates BETWEEN corrections. So: starve the corrector and the same model error accumulates
over ~17x longer before anything removes it.

This is the sim2real question stated as an experiment. A real robot has a slower, noisier, more often
unavailable localiser, and that is precisely the regime in which an uncalibrated model should stop
being free.

**Design: 2×2, ONE session, four legs of ~200 m on the same route.** The injection is held CONSTANT
at `WheelScaleV = 0.10` in all four legs — it is not a variable here. The two manipulated variables:

| leg | calibration | corrector | config |
|---|---|---|---|
| 5-A | OFF | abundant | `MotionCalibApply=false`, `StableSdfMseMax=0.076` |
| 5-B | ON | abundant | `MotionCalibApply=true`,  `StableSdfMseMax=0.076` |
| 5-C | OFF | **starved** | `MotionCalibApply=false`, `StableSdfMseMax=0.16` |
| 5-D | ON | **starved** | `MotionCalibApply=true`,  `StableSdfMseMax=0.16` |

`mask = 1` throughout, matching arm 4 and putting the dose and the response on one channel.

**Why 0.16, and it is not a guess.** Measured over both arm-4 legs (n = 32204 cycles), the share of
cycles whose `sdf_mse` exceeds the gate: 4.93% at 0.076, 1.25% at 0.10, 0.72% at 0.13, **0.29% at
0.16**, 0.13% at 0.20, and 0.00% at 0.25. 0.16 is ~17x fewer solves while the corrector still
EXISTS; 0.25 would switch it off entirely, which is a different experiment and an uncontrolled one.

⚠ **That 0.29% is an UNDERESTIMATE and must not be reported as the manipulation.** The residual
distribution is endogenous: starving the corrector lets drift accumulate, which raises `sdf_mse`,
which pushes cycles back over the gate. The loop is self-limiting. The knob sets a target; the
REALISED firing rate is a measurement and has to be reported as one.

**Pre-registered endpoints, in order.**

1. **PRIMARY — the INTERACTION on correction load (mm/m):** (5-C − 5-D) > (5-A − 5-B). The claim is
   that calibration removes MORE load when the corrector is scarce. Arm 4 measured (5-A − 5-B) at
   28.4 mm/m; the prediction is that the starved pair separates by more.
2. **SECONDARY — iterations PER SOLVE.** This is where the price should finally move: after a long
   blind stretch the solve starts from a large offset instead of 2 mm. Predicted 5-C > 5-D, and both
   above arm 4's flat 13.3–13.9. ★ If this stays flat too, the solve is not offset-limited at ANY
   reachable dose and the "effortful inference" reading of firing does not survive.
3. **THIRD — RPE translation**, the interaction again. Predicted: the accuracy penalty for being
   uncalibrated is far larger under starvation.
4. **FOURTH — burst-window rate**, counted, never excluded, per the arm 4 rule. Starvation is the
   condition most likely to finally produce bursts, and if it does in 5-C but not 5-D that is the
   headline and everything above is a footnote.
5. **MANIPULATION CHECK, reported before any endpoint:** realised firing % per leg, and the
   odometry/GT ratio confirming the injection acted (≈1.10 on the OFF legs, falling toward 1.0 on
   the ON legs as `k_v` converges).

**What would falsify the thesis's reading.** If 5-C and 5-D are indistinguishable on every endpoint,
then corrector abundance is not what protects the robot from a bad model, and the claim that
calibration buys anything at all under a working localiser does not survive contact with a scarce
one. That outcome is reportable and must be reported.

**What is NOT being claimed.** Arm 5 holds the dose fixed, so it says nothing further about
dose-response. The 5.5 → 16.6 mm/m scaling above remains a cross-session hint until a within-session
dose ladder measures it.

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
