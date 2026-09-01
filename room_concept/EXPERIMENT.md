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

### Arm 3R — arm 3 REPEATED with an instrument that measures work. 2026-09-01. IT DOES NOT REPRODUCE.

Arm 3's conditions exactly: no injection, the native robot, `mask = 1`, calibration off then on,
back to back in one session, bridge never restarted between legs. 219.9 m and 216.0 m, 13 and 11
windows. Injection absence verified from behaviour: odometry/GT 1.0001 and 0.9789.

| endpoint | 3R-OFF | 3R-ON | t | z | d |
|---|---|---|---|---|---|
| **correction per solve, mm** | **22.04** | **17.33** | **2.41** | **2.17** | **0.94** |
| correction load, mm/m | 50.11 | 36.96 | 1.09 | 0.96 | 0.42 |
| RPE translation, mm/m | 41.33 | 34.98 | 1.85 | 1.48 | 0.72 |
| aligned ATE, mm | 45.2 | 43.0 | 0.53 | 0.32 | 0.22 |
| **optimiser firing, %** | **5.41** | **5.08** | **0.20** | −0.20 | **0.08** |

★★★ **ARM 3'S HEADLINE RESULT DOES NOT REPRODUCE.** Arm 3 recorded 4.92% -> 1.11%, a 4.4x drop, and
that number is what the effort claim in the thesis rests on. Repeated under its own conditions,
firing goes 5.41% -> 5.08%: t = 0.20, d = 0.08, nothing. What DOES move is the correction per solve,
down 21% with the only significant statistic in the table.

#### The two native-error runs are COMPATIBLE, not contradictory — and what may be quoted beside what

**RPE: same code, verifiably, so arm 3 and arm 3R may be quoted side by side.** Every functional
change to `calib_localization_ab.py` since `7f4f572` (2026-08-30 11:58, before arm 3 was driven) is
additive — new dict keys, blocks that only append to them, print formatting, new entries in the
pairwise loop. Nothing touches `rpe()`, `load()`, `windows()`, `aligned_ate()`, the `trans`/`rot`/`ate`
accumulation, or `DS_TRANS`/`DTH_ROT`/`WINDOW_S`/`PARKED_MPS`/`BURST_FRAC`, and the burst/parked
exclusions still precede the RPE computation.

★ **The CORRECTION column is the opposite case and is STRONGER than "unknown definition": the script
had no correction column until 2026-09-01** — the old header was `arm, wins, dist_m, RPE mm/m,
RPE d/rad, ATE mm`. Arm 3's 48.99 therefore did NOT come from this tool and is definitely not the
same computation. **Never quote it beside arm 3R's or arm 4's correction figures.**

**The contrasts differ by 1.85 SE — compatible.** Levels agree: 37.56 vs 41.33 OFF (10%), 37.86 vs
34.98 ON (8%). Contrasts differ: −0.30 vs +6.35 mm/m. But arm 3R's per-window scatter is OFF sd 10.70
(n=13), ON sd 5.73 (n=11), pooled **8.79**, so the SE of a single run's contrast is **3.60 mm/m** and
the two runs sit **1.85 SE** apart — routine for two runs sampling one effect.

★★★ **So the statement is not "two matched runs disagree". It is that the treatment contrast at the
native error is small relative to run-to-run variation, so a single 200 m arm cannot resolve its
sign.** One run put it slightly negative, the other moderately positive, and neither had the
precision to distinguish those. Same lesson as the firing channel, one level less severe: there the
instrument could not resolve what it was quoted for at all; here the instrument is sound and the arm
is too small for the question.

★ **Sizing the confirmatory arm**: at pooled SD 8.79 and a 6.35 mm/m contrast, ~32 windows per leg
reaches 80% power — roughly **500 m per leg**, not 200.

⚠ **DO NOT CALL THE Pose-ERROR RESULT HERE "UNCHANGED" — IT IS UNDERPOWERED, NOT NULL.** RPE
translation is 41.33 → 34.98 mm/m at **d = 0.72**, and the power to detect d = 0.72 at n = 13/11
windows is about **39%**. A non-significant result is therefore the MODAL outcome even if the effect
is entirely real. Arm 4's d = 1.34 at n = 10/10 had ~81% power, which is why it separated. ★ The
claim to make is that the EFFECT SIZE grows with the dose — 0.72 at the native error, 1.34 at four
times it — because effect size does not depend on how many windows an arm happened to yield.
Treating a 39%-power non-detection as a measurement of zero is the mirror image of the error this
section convicts the experiment of: discounting an effect for missing significance, having earlier
believed one for being large. ⚠ This applies ONLY to arm 3R. Arm 3's own registered null is a
different measurement in a different session and stands as recorded.

⚠ **A CROSS-SESSION CLAIM MADE EARLIER TODAY IS WITHDRAWN.** It was reported that the correction load
"reproduces across sessions within 8% (48.99 vs 45.27) while firing drifts 60%". The 45.27 was
computed over ALL rows including parked ones; on the moving-only basis the instrument uses, that leg
is 24.99 mm/m. Arm 3's raw CSVs are gone, so its 48.99 cannot be re-derived on a matching basis and
NO cross-session comparison of correction load is established in either direction. The within-session
arm 3R comparison above is unaffected — both its legs use one definition.

#### The four conditions on one basis — what responds and what does not

| condition | model error | firing % | iters/solve | **corr/solve, mm** |
|---|---|---|---|---|
| native, calibrated | ~0% | 5.60 | 13.64 | **17.30** |
| injected 10%, calibrated | ~0.8% | 8.17 | 13.27 | **18.11** |
| native, uncalibrated | ~2.9% | 5.57 | 13.99 | **20.24** |
| injected 10%, uncalibrated | ~10% | 7.80 | 13.90 | **25.59** |

★★★ **Correction per solve is MONOTONE in the model error** — 17.30, 18.11, 20.24, 25.59 against
0%, 0.8%, 2.9%, 10%. Across a range of model error spanning more than an order of magnitude:

- **`iters/solve` is FLAT within 1.05x** (13.27–13.99). The per-solve work is a constant of the
  solver, not a function of how wrong the motion model is.
- **Firing sorts by SESSION, not by treatment** — ~5.6% in both arm 3R legs, ~8.0% in both arm 4
  legs. Between-session drift (1.47x) exceeds every within-session treatment effect measured, and it
  is the documented 5x `opt/m` spread showing up again.

⚠ **THAT DOES NOT EXPLAIN ARM 3, and an earlier draft of this section claimed it did.** Session
sorting explains why the absolute LEVELS differ between arms. It cannot explain a 4.4x difference
between two legs PAIRED INSIDE ONE SESSION, because within-session pairing evidently controls firing
well in both arms measured here. **Arm 3's within-session 4.4x is unexplained, and its raw CSVs are
gone, so it cannot be explained.** What is established is narrower and sufficient: the endpoint
cannot support the interpretation placed on it, and the result does not reproduce. ★ An unexplained
non-reproducing result reported AS unexplained is stronger than one handed a cause it cannot carry —
which is the same error as believing a 4.4x from an instrument documented to spread fivefold.

**Conclusion, stated for the thesis.** Calibration reliably reduces the ERROR THE OPTIMISER MUST
REMOVE and reliably does not reduce how often or how hard it solves. The effort result came from a
binary over a threshold and does not reproduce; whether it is a property of calibration is settled
(it is not), why the original run produced it is not.

★★★ **BURST-WINDOW RATE IS NULL EVERYWHERE**: 0/16 and 0/12 in arm 4, 0/13 and 0/13 in arm 3R. Zero
burst windows in all four legs across a model error from ~0% to 10%. The localiser never lost
tracking under any condition produced, which quantifies the corrector's slack: a 10% odometry scale
error, 4x the robot's native error, is absorbed without one lost window. The registered third
endpoint returns empty and that is a result, not a gap. ★ The claim that survives is about the correction load, it has a
monotone dose-response, and it is the one to carry.

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

## 10. Optimal excitation — the analytic design, and what it says about active calibration

Derived 2026-09-01, tool `tools/excitation.py` (`74f42d5`). This section replaces the question "what
experiment identifies each parameter?" with an answer that is mostly algebra: the model is LINEAR in
the parameters, so the Fisher information has a closed form and the optimal motion falls out of it.

### 10.1 The information rate, in one line

Per episode the residual is `r = J p + noise`, `J` built from the covariates (§2). For a motion held
at constant `(v, w)` for duration `T`, every Jacobian entry is a rate times `T`, so information per
EPISODE goes as `(rate * T)^2 / sigma^2`, and since episodes arrive at `1/T`:

| parameter | Jacobian entry | information per SECOND | maximise |
|---|---|---|---|
| `k_v` | `v*T` (along) | `v^2 T / sigma_p^2` | **v and T** |
| `eps_yaw` | `-v*T` (cross) | `v^2 T / sigma_p^2` | **v and T** — identical to `k_v` |
| `k_omega` | `w*T` (head) | `w^2 T / sigma_th^2` | **w and T** |
| `b_omega` | **`T`** (head) | **`T / sigma_th^2`** | **T only** |
| `dk_wheel` | `v*T` (head) | `v^2 T / sigma_th^2` | **v and T** |
| `k_lat` | `l*T` (cross) | zero on a differential base | — |

Everything below follows from that table without any data:

- **`k_v` and `eps_yaw` have IDENTICAL optimal designs** — same covariate, different component. A
  numerical search "discovering" this is reproducing algebra.
- ★★★ **`b_omega`'s Jacobian contains no rate at all.** No manoeuvre can help it; the only lever is
  episode duration. This is why its column sits at 1.00 for every candidate the tool scores, single
  or combined. ★ A score column stuck at 1.00 is the theory being right, not the library being short.
- **`T` is a common factor in EVERY row.** Episode duration is the universal lever, and it is the one
  a rate-mixture search cannot see if it holds `T` fixed. Mine did, which is why its designs lost to
  ordinary driving until `T` was freed.
- Two degeneracies share the heading component and have DIFFERENT cures: a single-rate pivot makes
  every `(w, 1, v)` parallel so `k_omega`/`b_omega` go rank 1 (cured by rate diversity, cond
  inf -> 12.4), while `dk_wheel` needs FORWARD DISTANCE, which no pivot at any rate supplies (cured
  only by adding straights, cond inf -> 51.6 -> 32.5 with arcs). Turning separates the gyro pair;
  DRIVING reveals the wheel mismatch.

**What is NOT derivable:** `sigma^2` as a function of the motion. The formulas take it as given, but
`pos_var` is the localiser's posterior plus the model-error terms, and its behaviour is a property of
the localiser and the route. Measured here it FALLS with duration (log-log slope −0.20), so
information per second scales as `T^1.20` — superlinear, and that exponent is empirical.

★ **And one thing the algebra gets WRONG by construction:** information per second grows with `T`
without bound, so the formulas say make episodes infinitely long. That must break, because the
Jacobian linearises the whole accumulated motion as one increment. Nothing in the inverse model knows
this; it is why `episode_carry_max_trans = 2.5 m` and `episode_carry_max_rot = 2.0 rad` exist as
separate guards. **The design question the algebra cannot answer is how long an episode may get
before linearisation breaks, and that is the one thing worth driving for.**

### 10.2 What a dedicated manoeuvre can buy — the ceiling, from the platform

A manoeuvre can only raise the rate and the episode length, both capped by the base
(`MaxAdvSpeed = 0.7 m/s`, `MaxRotSpeed = 1.0 rad/s`). So its value is bounded by a ratio of squares
against what purposeful driving already does. Measured over a 220 m apartment window:

| channel | ordinary driving | platform max | rate headroom | measured gain |
|---|---|---|---|---|
| forward (`k_v`, `eps_yaw`, `dk_wheel`) | v p90 = 0.68 m/s | 0.70 | **1.05x** | 1.15–1.26x |
| heading (`k_omega`) | w p90 = 0.51 rad/s | 1.00 | **3.81x** | **8.74x** |

Shrink against the prior over a matched 600 s budget:

| design | `k_v` | `eps_yaw` | `k_omega` | `dk_wheel` | `b_omega` |
|---|---|---|---|---|---|
| ordinary driving (observed) | 7.07 | 6.20 | 4.18 | 8.11 | 1.06 |
| rate-mixture search, `T` fixed at 0.55 s | 2.65 | 2.37 | 1.85 | 1.97 | 1.00 |
| analytic: max v, straight, `T` = 0.55 s | 2.65 | 2.37 | 1.00 | 2.60 | 1.00 |
| analytic: max v, straight, `T` = 5 s | 11.19 | 9.80 | 1.00 | 11.69 | 1.00 |
| analytic: + sustained max turn, `T` = 5 s | 7.94 | 6.96 | **12.36** | 8.68 | 1.00 |

★ At fixed `T` the analytic design and the numerical search agree exactly (2.65 vs 2.65). Freeing `T`
is the whole difference. Ordinary driving was never beating a good design — it was beating a search
that had the wrong free variable.

### 10.3 ★★★ WHEN TO CALIBRATE, NOT WHETHER — and why wait-and-watch wins in a cluttered room

> A parameter is excited for free exactly to the extent that its covariate is ALIGNED with what the
> task already wants to do.

Forward speed **is** what navigation wants: going fast in straight lines is the job, so `k_v`,
`eps_yaw` and `dk_wheel` are excited as a by-product, at 0.68 of a 0.70 m/s ceiling, during exactly
the long corridor runs that dominate the information. A dedicated manoeuvre buys 15–26%. **No
calibration affordance is justified for the forward channel, and that follows from the platform
limits rather than from this dataset.**

Rotation looks like the opposite: turning is a COST the planner minimises, so the heading channel
ought to be systematically starved by purposeful behaviour. **In an apartment it is not**, and this
is what closes the argument:

| how often the apartment supplies rotation, measured over 185 m / 464 s of motion |
|---|
| a turn > 0.5 rad **every 3.6 m** (every 9.1 s of motion) |
| a turn > 1.0 rad **every 6.4 m** (every 16.0 s) |
| a turn > 1.5 rad every 18.5 m |
| **46.2 rad of turning per 100 m driven, for free** |

★★★ **And it is already enough.** After ONE ordinary window `k_omega` is `informed` at 3.70x shrink:
sigma 0.0054 on a 0.02 prior, i.e. **the rotation scale is known to ±0.54%**. A dedicated pivot would
sustain 1.0 rad/s against the 0.184 rad/s the apartment averages — 5.4x the rate — and would tighten
±0.54% to perhaps ±0.2%. There is no evidence that difference changes anything, and the finding that
heading corrections improve the pose at all was RETRACTED (§6, arm 3R).

★★★ **CONCLUSION: NOTHING HERE JUSTIFIES INTERRUPTING A TASK TO CALIBRATE.** Translation because
purposeful navigation already runs at 0.68 of a 0.70 m/s ceiling during the long runs that dominate
the information. Rotation because a cluttered apartment forces a hairpin every few metres, and
geometry the robot cannot avoid is excitation it does not have to pay for. **Wait-and-watch is not the
fallback here; it is the optimum.**

⚠ Note the shape of that claim, which is NARROWER than a prohibition — see the idleness rule below. An
earlier draft of this section read "no calibration affordance is justified", and the two statements
drifted apart the moment the idleness exception was added. The generalisation, and the part that
transfers off this robot:

> An affordance is worth its cost only where the task's own motion does NOT already span the
> parameter's covariate. Compute that span before building the affordance — it is a ratio of squared
> rates against the platform limits, and it needs no experiment.

⚠ The scope is real and should be stated wherever this is quoted: a LARGE open environment would
invert the rotation half. Long gentle arcs and few hairpins would starve `k_omega` exactly as the
rate-headroom figure (3.81x) predicts. The apartment is not a limitation of this result — it is a
condition of it.

★★★ **NEED AND COST ARE THE SAME VARIABLE, which is why the rule bites.** The rule above states the
BENEFIT side. The COST side is not independent of it: a planner avoids a motion BECAUSE that motion
costs it something, so the degree to which a route lacks rotation is precisely the degree to which
inserting rotation deviates from that route. Cluttered — excitation free AND a manoeuvre would be
cheap, and it is unnecessary. Open — the parameter genuinely starved AND the manoeuvre must override
a route the planner considered optimal. **An affordance must overcome its own justification.** That
is why the answer comes out against affordances in general and not merely in this apartment.

★★★ **THE ONE EXCEPTION, AND IT IS THE PRACTICAL CONCLUSION: the anti-correlation holds only while
the robot is EXECUTING A TASK.** It vanishes when the robot is idle — a manoeuvre performed during
idle time has near-zero opportunity cost and undiminished benefit. So the design rule is not "build
no affordance", it is:

> **A calibration affordance should compete with IDLENESS, never with a task.**

★ Which is what `afford_calib` already does (passive estimation live, manoeuvre off) — arrived at
empirically before this derivation existed. See [[base-self-calibration-affordance]].

`b_omega` sits outside the argument: no rate appears in its Jacobian, so wait-and-watch is trivially
optimal there in the sense that nothing else works either. Separately it is PRIOR-SWAMPED — its
`5e-4 rad/s` asserts 4e6 of information against 9.6e5 from 220 m of driving, a **19.4% data share**
where every other live parameter gets 93–98%. Halving its sigma would need ~49 HOURS. The question
there is whether 0.03 deg/s is a justified prior, not which route to drive.

⚠ **The awkward part.** The one parameter a manoeuvre could materially improve is `k_omega`, and it
is the one with the least evidence that applying it helps at all — the heading finding was RETRACTED
on 2026-08-31 (§6, arm 3R). Real in information terms, unproven in outcome terms.

### 10.4 Hairpins already do most of this, and the episode machinery is damaging them

The apartment's hairpins ARE the rotation excitation: the top 10% of episodes carry **82.4%** of
`k_omega`'s information, with median `d_theta` = **1.37 rad** over **3.73 s**. Purposeful driving is
not starving the heading channel as badly as the rate headroom alone suggests — it is producing good
turns and then losing them twice:

- **16.7% of episodes close exactly at `episode_min_rot = 0.20 rad`.** Information goes as
  `(d_theta)^2`, so a 3.14 rad hairpin taken as ONE episode is worth 1580 while the same hairpin
  chopped into sixteen 0.20 rad pieces is worth 103 — **15.4x less**.
- **No episode exceeds 1.984 rad**, sitting hard against `episode_carry_max_rot = 2.0`. A 180 deg
  hairpin is 3.14 rad, so the largest turns are clipped or dropped at the linearisation guard —
  and by `(d_theta)^2` those are the highest-information episodes in the window.

★★★ So a large part of the `k_omega` gap is NOT a missing manoeuvre. It is the episode machinery
destroying turns the robot is already performing. That is a config question before it is an
affordance question, and the two are confounded in every number in §10.2 — the 8.74x is measured
against driving whose hairpins were being chopped.

### 10.5 ⚠ ROTATION EXPERIMENTS STILL NEEDED — this section is not self-sufficient

The translation conclusion stands on the algebra plus the platform limits. **The rotation conclusion
does not, and must not be quoted as settled.** Three things are unmeasured:

1. **How much of the 8.74x is chopping rather than starvation.** Re-run the same route with
   `episode_min_rot` raised so a hairpin survives as one episode, and re-measure `k_omega`'s shrink.
   If most of the gap closes, the affordance is not needed and a constant was the whole story.
2. **Where linearisation actually breaks.** The algebra says `T` without bound; the carry caps are a
   GUESS at where that stops being true. The endpoint is per-parameter sigma per metre against
   episode length, watching for the point where longer episodes stop paying — the first measurement
   of a limit currently asserted by two constants.
3. **Whether a sustained-rotation manoeuvre delivers its predicted 12.36x in practice.** Predicted
   from a model whose `sigma_th^2` is assumed to hold at rates the robot rarely sustains; a fast
   sustained pivot may degrade the localiser's own fit and inflate `sigma_th^2`, which would eat the
   gain. The prediction is untested and it is the one the affordance's case rests on.

⚠ Do not build a rotation affordance before (1). The cheapest experiment is a constant, not a
manoeuvre, and it may remove the reason for the manoeuvre entirely.

★ Note what these three experiments are now FOR. They are no longer deciding whether to build an
affordance — §10.3 answers that with geometry and the platform limits. They are asking whether the
episode machinery is throwing away rotation the robot already performs, which is a defect question,
and where linearisation actually breaks, which is two constants currently asserted without
measurement. Both are worth driving for; neither is a manoeuvre.

## 11. The self-model — what these parameters are, in the architecture's own terms

★ **This section is INTERPRETATION, not result.** It is the reading under which §6 and §10 cohere,
and it should be labelled as such wherever it is quoted. Every number in it is measured; the frame
around them is not.

### 11.1 Three factors, three timescales, one free energy

The agent's generative model factorises over three things, all minimising the same quantity:

| factor | what it is | timescale | in this system |
|---|---|---|---|
| `q(s)` | **states** — where I am now | per cycle | the SDF localiser's pose |
| `q(θ_world)` | **the world's structure** | slow | the room shape; walls as landmarks |
| `q(θ_body)` | **my own body** — how action becomes motion | slow | `k_v`, `eps_yaw`, `k_omega`, `b_omega`, `dk_wheel` |

`θ_body` parameterises `p(s_{t+1} | s_t, a_t)`. **The transition model IS the self-model**, so an
agent carrying a wrong one is systematically surprised by the consequences of its own actions. That
is what makes calibration a first-class inference problem here rather than a maintenance chore: the
same free energy is minimised over where-I-am, what-the-room-is, and what-I-am.

### 11.2 The correction load is the free energy that `θ_body` OWNS

§10 justified the endpoint structurally — it is the optimiser's input, so the corrector cannot absorb
it. The better reason is what it MEANS: `|est − pred|` is the prediction error the BODY-model failed
to explain, isolated from what the world-model then explains. RPE measures the residual after both
factors have done their work; the correction load isolates the component attributable to `θ_body`.

That is why it, and nothing else, tracked injected body error monotonically — 17.30, 18.11, 20.24,
25.59 mm per solve at 0%, 0.8%, 2.9%, 10% (§6). An endpoint that measures the right factor responds
to a manipulation of that factor. The others were measuring the sum, or the leftovers.

### 11.3 ★★★ FREE ENERGY IS NOT COMPUTE — and assuming otherwise is what produced the retraction

The withdrawn effort claim (§6) rested on reading optimiser firing as *the rate at which the agent
must do effortful inference because its model failed to predict*. The intuition is right and the
identification is wrong, and the wrongness is measurable: **iterations per solve are FLAT at
13.27–13.99 across a body-model error spanning 0% to 10%.**

The agent pays for a bad self-model in **revision magnitude** — the information-theoretic quantity,
which moved 1.5x — while the COMPUTATION performing that revision is constant. Surprise and FLOPs
are different currencies, and a Gauss-Newton step converging from 2.2 mm instead of 1.5 mm costs the
same thirteen iterations.

★ This matters beyond this experiment. Any argument of the form "a better model saves the agent
work" needs to say which work. Here the model error is paid in how far beliefs must be revised, not
in the cost of revising them, and an endpoint chosen on the other reading measured nothing for a day.

### 11.4 ★★★ ACTING IS THE EXPERIMENT

§10 showed that a dedicated calibration manoeuvre buys 15–26% on the forward channel and that a
cluttered room supplies the rotation for free. The reason is structural rather than lucky:

> **A purposeful agent instruments its own body as a by-product of acting, because every action
> passes through the body. The epistemic value for `θ_body` is already contained in the pragmatic
> action.**

This is the active-inference statement of the wait-and-watch result. Epistemic action for the SELF is
rarely needed, because there is no such thing as an action that does not exercise the body.

### 11.5 The asymmetry — the world needs looking at, the body does not

The same argument run over `θ_world` gives the opposite answer, and the contrast is the point:

| | `θ_world` | `θ_body` |
|---|---|---|
| sampled | only where the agent happens to look | by **every** action |
| epistemic action | routinely necessary (NBV, exploration — and this system builds them) | rarely necessary |
| limit | occlusion, reach | only the covariates the action excites |

**The world is sampled where you look; the body is present in everything you do.** That asymmetry is
why this architecture correctly carries next-best-view affordances for objects and rooms while
needing none for the base's kinematics.

★ Its limit is exact, not vague: the body is instrumented only along the covariates a given action
excites. `k_lat` is never excited on a differential base; rotation is barely excited in open space.
And §10.3's need/cost coupling closes it — deviating to excite a starved covariate is most expensive
precisely where it is most starved, so the exception does not readily become an argument for
manoeuvres.

### 11.6 ⚠ THE SELF-MODEL IS LEARNED THROUGH THE WORLD-MODEL, so map error is charged to the body

The calibrator learns from the localiser's corrections, and those corrections are produced by fitting
against the map. **`θ_body` is therefore estimated through `θ_world`, and an error in the world-model
is attributed to the body.** This is a structural hazard of the factorisation, not an implementation
detail, and it has already produced defects:

- `fit_model_gain` exists precisely for it — an episode is weighted by the FIT that produced its
  correction, so a poor fit teaches the body-model less.
- `c71aab9` was this hazard biting: weighting by the drift BEFORE the fit rather than the fit itself
  meant the largest map failures taught the self-model hardest. Exactly backwards.
- The `MapMode = "estimate"` regime (§10 blocker) is the extreme case — with the layout itself
  unconverged, every correction carries world-model error into the body-model's evidence.

★ The general rule: **a factor estimated through another factor inherits that factor's errors, so its
evidence must be weighted by the other's confidence.** That is what precision-weighting is for, and
it is the one place in this design where getting the weight wrong is worse than having no estimate —
because an unweighted body-model update does not merely learn SLOWLY, **it learns the world's
mistakes as facts about itself.** The failure is a bias, not a variance.

★★★★ **AND THE CORRECTION LOAD CANNOT SEE IT — a scope limit on §11.2's endpoint, not a defect in it.**
Follow the bias one step further. Once the body-model has absorbed the map's distortion, its
predictions match what the distorted map will say the motion was, so `|est − pred|` SHRINKS. The two
factors settle into a state that is internally consistent and externally wrong, and the endpoint this
work makes central **reports that as an improvement**. It is measured entirely inside the agent, so a
bias shared between the body-model and the world-model is invisible to it: the failure mode is not
divergence but **silent self-consistency**, and the instrument that best measures the body-model in
normal operation is precisely the one that cannot see it.

> The correction load isolates the body-model's contribution to prediction error, but only RELATIVE
> to the world-model that produced the correction. It therefore measures the self-model's fit to the
> agent's own world-model, not to the world, and must be read alongside a measure referenced OUTSIDE
> the agent.

★ Which is the real justification for carrying ground truth (`robot_gt_*`) rather than "it is
available": the two measures fail in different directions, so neither alone is sufficient. In one
line — **a factor estimated through another can be biased into agreement with it, and internal error
measures go quiet exactly when that happens.**

### 11.7 A prior can place part of the body beyond experience

`b_omega`'s prior of 5e-4 rad/s asserts 4.0e6 of information against 9.6e5 supplied by 220 m of
driving — a 19.4% data share where every other live parameter reaches 93–98% (§10.2). The agent has
declared a belief about its own body precise enough that ordinary experience cannot revise it, and
its `informed` flag can therefore never fire.

★ This is a legitimate modelling choice and it is currently an INHERITED one. A prior that strong is
a claim that the gyro's bias is known to 0.03 deg/s before the robot has moved; if that is true it
should be cited, and if it is not the parameter is unlearnable for no reason.

### 11.8 What is measured and what is frame

**Measured**: the correction load's monotone dose-response; iterations per solve flat within 1.05x;
the rate-headroom ceiling and the 46 rad/100 m the environment supplies; `b_omega`'s 19.4% data
share; the identifiability structure of §10.1, which is algebra.

**Frame**: the three-factor reading, the identification of the correction load with `θ_body`'s free
energy, and the epistemic-value asymmetry between body and world. These are the interpretation under
which the measurements cohere. They are not tested by them, and a reader should be able to reject the
frame and keep every number.

## 12. Arm 6 — does episode LENGTH buy information, and where does linearisation break?

Pre-registered 2026-09-01. NOT RUN. Driver `tools/arm6_setup.sh`.

**Why this and not a manoeuvre.** §10 established that `T` is a common factor in every row of the
information table — episode duration is the universal lever — and that the estimator sits at its
FLOOR: `episode_min_trans` = 0.25 m at 0.43 m/s is 0.58 s against an observed 0.55 s median, so
episodes close the instant they are allowed to. §11 established that no affordance is justified here.
So the question that remains is not which motion to perform but **how much of the motion the robot
already performs is being thrown away**, and that is a constant, not a manoeuvre.

The rotation channel is where it bites hardest, because information goes as the SQUARE of the
accumulated covariate. Measured over the arm 3R window: **16.7% of episodes close exactly at
`episode_min_rot` = 0.20 rad**, and a 3.14 rad hairpin taken as ONE episode is worth 1580 against 103
for the same hairpin chopped into sixteen — **15.4x**. Meanwhile the surviving long turns (median
1.37 rad over 3.73 s) already carry **82.4%** of `k_omega`'s information.

**Design.** Two legs, same route, ~220 m each to match arm 3R, back to back in one session, cold each,
UNINJECTED (the native robot). `MotionCalibEpisodeMinRot` is the ONLY thing that varies:

| leg | `MotionCalibEpisodeMinRot` |
|---|---|
| 6-base | **0.20 rad** (the current value) |
| 6-long | **1.00 rad** (~57°, five times the trigger) |

Everything else is pinned to arm 3R's conditions — `MotionCalibEpisodeMinTrans` = 0.25,
`MotionCalibApply` = true, `mask` = 1, `StableSdfMseMax` = 0.076, `MapMode` = "given" — so a
difference cannot be attributed to anything but the trigger.

★ 1.00 rad is chosen against two bounds, not guessed: it must stay well below
`episode_carry_max_rot` = 2.0, or episodes would be DROPPED at the carry cap rather than closed; and
it must sit below a typical hairpin so that turns still close on rotation rather than being carried.
The observed `d_theta` p90 is 1.066 rad and the maximum is 1.984 — hard against the carry cap.

★★ Raising the ROTATION trigger alone is deliberate. Episodes close on EITHER threshold, so during an
arc the translation trigger still fires first; only tight and pure turns are affected, which is
exactly the hairpin case this arm is about. The translation lever is a separate question (§12.4).

### 12.1 Pre-registered endpoints, in order

1. **PRIMARY — `k_omega` posterior sigma at MATCHED ROTATION.** Not per metre and not per run: this
   arm is about information per radian TURNED, so the legs must be compared at equal cumulative
   rotation or the comparison is against the route. Predicted: 6-long lower. The information law says
   information per unit rotation is LINEAR in the trigger, so the ceiling is 5x more information,
   i.e. sigma down by up to sqrt(5) = 2.2x. Anything approaching that closes the rotation gap and
   removes the last argument for a rotation affordance.
2. **SECONDARY — the heading block's lambda_min and condition number**, from
   `tools/excitation.py` on each leg's episode log. Predicted: lambda_min up in 6-long,
   condition number roughly unchanged (the trigger scales the block, it does not re-orient it).
3. **DIAGNOSTIC — episode bookkeeping**: emitted, carried and DROPPED counts, median `d_theta` and
   duration. 6-long must show a larger median `d_theta`; if it does not, the trigger is not binding
   and the arm has not manipulated what it claims to.

### 12.2 ★★★ THE DISCRIMINATOR: sigma must improve while the VALUE must not move

This is the pre-registered test for the linearisation limit, and it is the reason this arm is worth
driving rather than reasoning about.

Both legs observe **the same robot on the same route**, so `k_omega`'s TRUE value is identical. A
longer episode should therefore buy PRECISION and nothing else:

- **sigma falls, value agrees within its sigma** ⇒ the trigger was throwing information away and
  raising it is free. The rotation gap was a constant all along.
- **sigma falls but the VALUE SHIFTS** ⇒ ★ **the signature of linearisation breaking.** The Jacobian
  treats the whole accumulated motion as one small increment; when that stops being true the estimate
  acquires a BIAS, and a bias is exactly what a shift in the value on an unchanged robot means. This
  is the first measurement of a limit currently asserted by two constants and never tested.
- **neither moves** ⇒ the trigger was not the binding constraint; look at `theta_var` instead, since
  the weight may be growing with the episode as fast as the covariate does.

⚠ **Report the value comparison BEFORE the sigma comparison.** A precision improvement bought with a
bias is worse than no improvement, and reporting sigma first invites reading the arm as a success.

### 12.3 What would make this arm uninterpretable

- **Unmatched rotation content.** The endpoint is per radian turned; if the legs differ much in total
  rotation the primary comparison is against the route. Drive the same circuit.
- **A change in dropped-episode count.** If 6-long drops many more episodes at the carry cap, it is
  not measuring longer episodes but fewer of them, and the two effects are confounded. `dropped_`
  is logged; report it.
- **Window lag.** Fewer, longer episodes mean the 512-episode window spans more distance, so the
  estimator adapts more slowly. On a static parameter this only affects the transient, but it is the
  reason not to raise the trigger further than this arm tests.

### 12.4 Parked, and deliberately

The translation trigger. `MotionCalibEpisodeMinTrans` is now config-exposed alongside the rotation
one and the same argument applies to `k_v`, `eps_yaw` and `dk_wheel` — but those three are already
`informed` at 5.5-7.3x shrink from ordinary driving, so the headroom that matters is in the channel
that is not. Raise it only if arm 6 shows the mechanism is real.

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
