# Sim → real calibration, minimum effort

**Purpose.** Get the fleet's motion model, its published uncertainty, and the speed limiter that
consumes it onto numbers measured from the *real* robot, with as little robot time as possible.

**The enabling fact, and the reason this is cheap.** Room-robot localisation already works on the real
robot to within some error. That makes the localiser's own posterior a usable *reference*: it is
anchored to a fixed room polygon, so it is an independent witness of true motion in a way dead
reckoning can never be. No motion capture, no external tracker, no ground-truth rig. Everything below
is a robot standing still, a robot turning, a robot driving a leg, and one ordinary tour.

**Total robot time: about 20 minutes**, in four phases that do not need repeating if taken in order.

---

## 0. Before you touch the robot

**Nothing in the simulator's tuning transfers.** Every constant in the fleet was fitted against a
bridge that published the supervisor's *ground truth*: measured over 24434 parked frames, that
odometry has σ_v = 3.9e-7 m·s^−½, five to six orders below what the motion model is configured for.
So the motion prior has never done real work in simulation, and anything tuned against it was tuned
against a robot that does not exist. Treat every number below as unmeasured until you have measured it.

**Confirm these before starting** — each has bitten before:

| check | why |
|---|---|
| `LearnMotionModel = false` | A/B'd and rejected: it collapsed early exit 91.6% → 68.5%. Its residual is a post-optimisation window-pair quantity carrying the optimiser's own correction, so it double-counts. Leave off. |
| `SensorNoise.Enabled` (webots-bridge) | Simulator-only. It does not run on hardware, but make sure you are not reading a sim log by mistake. |
| `PoseClampVMax` / `PoseClampWMax` | Must be the **controller's** limits, not room_concept's beliefs. They are currently 0.7 / 0.8. If the real base is slower, the clamp fires constantly and inflates the published covariance. |
| locale | These machines run `LANG=es_ES.UTF-8`. Any tool that parses agent CSVs must use `std::from_chars` and any harness must `setlocale(LC_ALL,"")`, or it silently reads `0.26` as `0`. |

**Tools that already work on hardware, unchanged:**

```
make -C build motion_calib && ../bin/motion_calib <debug_log.csv>   # calibrator + scorecard
make -C build gn_selftest  && ../bin/gn_selftest                    # solver Jacobians, no robot
```

`motion_calib --selftest` validates the estimator against synthetic truth in seconds and needs no
robot at all. Run it once so you know the tool is sound before trusting its numbers.

---

## Phase 1 — parked, motors enabled (5 min)

**Measures:** `PreintOdomSigmaVLat`, `PreintOdomSigmaVLong`, `PreintOdomSigmaOmega` — the odometry noise
densities. **This is the only measurement that needs no localisation at all.**

Park the robot with the motors **enabled** (a disabled base hides driver noise). Let room_concept run
and log for ~5 minutes. Then:

```
../bin/motion_calib tmp/sdf_localizer/<log>.csv
```

Read the **`INDEPENDENT CHECK`** block, not the regression table:

```
── INDEPENDENT CHECK: the odometry stream's OWN noise, parked ──
  N parked samples at DT s spacing
  sigma_v     = ... m/sqrt(s)   (direct, localiser not involved)
  sigma_omega = ... rad/sqrt(s)
```

Those two numbers go straight into `PreintOdomSigmaVLat/VLong` and `PreintOdomSigmaOmega`.

> **⚠ The naive estimator is optimistic if the encoder velocity is low-pass filtered**, which most are.
> The robust version is an **Allan / T-sweep**: integrate the reported velocity over disjoint windows of
> length T (0.1, 0.2, 0.5, 1, 2, 10 s) and plot the variance of that integral against T.
> Variance ∝ T ⇒ genuine white noise and the slope is σ². Variance ∝ T² ⇒ a *bias*, not noise.
> That discrimination is not a nicety: it is exactly the σ√T vs scale·T split the motion model encodes,
> so one parked log calibrates the structure of both. If the two estimators disagree, trust the sweep.

**Sanity bound.** A wheeled base with encoders should land near 1e-3 m·s^−½. If you get 1e-7 you are
reading a simulator log; if you get 1e-1 something is wrong with the base driver.

---

## Phase 2 — one closing pivot (2 min)

**Measures:** the gyro's accuracy, the wheel channel's rotation error, and **whether that error is
geometry or scrubbing** — which decides whether you fix a constant or accept a physical effect.

Pivot in place through **exactly N full turns** (4 is convenient) and stop on the starting heading.

> **The closure test is the tool for rotation** because a pivot returning to its start has a true
> rotation of *exactly* 2πN — no appeal to any estimator. It only qualifies if:
> - **net travel < 15 cm** (otherwise it is an arc and the test is invalid), and
> - the accumulated turn lands within a few percent of an integer.
>
> Anything else and you are dividing by a number you do not know.

Then, from the room_concept log, integrate three channels over the pivot and divide by 2πN:

| channel | source | reads |
|---|---|---|
| gyro | sum of `meas_dth` | should be **≈ 1.00** |
| posterior | unwrapped `pred_theta + innov_theta` | tracks the gyro |
| wheel | `odom_rot_norm` integrated over `odom_ingress_ts` intervals | the one under test |

**Reference numbers from simulation**, for shape not for value — a 4-turn pivot gave
gyro 1.0114, posterior 1.0120, wheel 1.2302 against a test resolving to ~1.2%.

**Interpreting the wheel figure:**

- **Constant across motion types ⇒ geometry.** ω = r·(ω_R − ω_L)/(2b), so a ratio of `k` means `r/b`
  is off by `k`. Check `Base.WheelRadius` and `Base.HalfTrack` against the real base's drawing.
  There is precedent: `LY` was once 0.237 instead of 0.210, worth 9% of over-rotation.
- **Larger on a pivot than on a tour ⇒ scrubbing**, and it is physical, not a bug. In simulation the
  same base read 1.078 on a mixed tour and 1.16 on a pure pivot — scrubbing scales with how rotational
  the motion is. Do **not** "correct" it in the kinematics; it is real, and the gyro is the answer to it.

**This is also the check that the gyro heading path is earning its place.** `[ImuInject]` prints
`coverage=` and `dtheta wheel/gyro=` every 5 s; the ratio is suppressed below 0.5 rad of accumulated
turn, so read it *while turning*. Coverage should be ~99%; a ratio pinned at 1.0000 means the two
channels are the same source twice, not two sensors agreeing.

---

## Phase 3 — one straight leg (2 min)

**Measures:** `PreintOdomScaleV` — the translational scale error.

Drive a straight leg between two points you have **tape-measured**, and compare against the odometry's
net displacement.

> **⚠ Use NET displacement over the whole leg, never the sum of per-frame displacement magnitudes.**
> Path length ≥ net displacement, so estimator jitter biases the denominator by an amount that depends
> on frame rate. That mistake once produced three mutually incompatible answers (0.975 / 0.997 / 0.963)
> for the same quantity, one of them arithmetically impossible for a pure scale.

`PreintOdomScaleOmega` comes from Phase 2's wheel ratio **minus** whatever you attribute to scrubbing.
If the wheel error is pure scrubbing and the pipeline takes heading from the gyro, this can stay small.

---

## Phase 4 — one ordinary tour (10 min)

**Measures:** everything that is a *property of the estimator in its environment* rather than of a
sensor — and it is the run that sets the speed limiter.

Drive a normal, varied tour: both rotation directions, a range of speeds, ≥ 10 minutes.

### 4a. Confirm the motion model

```
../bin/motion_calib tmp/sdf_localizer/<tour_log>.csv
```

- Read the **trend down the window ladder**, not one row. σ̂ rising with window length means the
  reference is contaminated by the motion prior; take the long-T end.
- The scale needs **long windows** to converge. Against an injected truth of +0.10 the estimate ran
  0.038 → 0.065 → 0.071 → 0.087 → 0.091 → 0.0945 at 8 → 192 frames, and only the last was within 1σ.
- **⚠ Watch for the `REFERENCE-LIMITED` warning.** It fires when the regression's σ is more than 3×
  the direct parked measurement, and it means the tool is reporting the *localiser's* jitter rather than
  an encoder property. In simulation it was reference-limited by 6000×. On a real robot with real
  encoders it may not be — that is the whole difference, and the warning tells you which world you are in.

### 4b. Set the speed limiter's knees — the part with the most leverage

The limiter (`ControllerMotionCommander::apply_uncertainty_speed_limit`) is the only place localisation
quality bounds speed. It reads σ from the room→robot RT edge and ramps:

```
σ_xy below PoseXYStdSlow  → scale 1.0
σ_xy above PoseXYStdStop  → MinAdvSpeedScale
between                   → linear, then × PoseUncertaintyCoupling
```

**The rule, and it has failed twice in both directions:** the knee belongs at roughly **p90 of σ while
MOVING**. Set it below what the localiser can deliver and the ramp never releases, degenerating into a
constant tax — that was `PoseXYStdSlow = 0.03` against a best-observed σ of 0.0429, throttling 89% of a
lap. Set it above the tail and the limiter never acts.

Get the distribution from the tour's own `<stamp>_profile.csv`, which records what the limiter actually
saw and applied:

| column | meaning |
|---|---|
| `pose_xy_std_m`, `pose_theta_std_rad` | the σ it read (−1 = no covariance reached it, limiter inert) |
| `unc_adv_scale`, `unc_rot_scale` | the multipliers it applied; 1.0 = no throttling |

Then set `PoseXYStdSlow` ≈ p90(σ_xy while moving), `PoseXYStdStop` ≈ slow + 0.04, and the same for θ.

> **⚠ Do not use room_concept's own `cov_xx` for this.** The controller reads
> `sqrt(max(cov_xx, cov_yy))` — the larger axis — and the kinematic clamp adds its un-applied residual
> downstream of room_concept's log write. Both differences are real and I got this wrong once by
> comparing against `cov_xx` alone. The log now carries `pub_cov_xx` / `pub_cov_tt` (post-clamp) and
> `clamp_hit` so the two can be compared directly; **join them to the frame BEFORE the one they appear
> on**, they are recorded one frame late by construction.

**Verify with `PoseUncertaintyCoupling`, which exists for exactly this.** Drive the same route at 1.0
and at 0.0 and compare `mean_speed_mps`. In simulation that pair read 125.0 s vs 83.1 s with tracking
*better* unthrottled — i.e. the limiter was costing a third of the tour and buying nothing. A run at 0.0
is a **measurement, not a setting**; restore it afterwards.

### 4c. The SDF residual floor — expect this to be worse than simulation

`SigmaSdf` (0.15) and `PredictionTrustFactor` (0.7) set the early-exit gate at `mean|SDF| < 10.5 cm`.
In simulation the floor sat at 8.7 cm, and its decomposition matters:

- ~30 mm is **sampling noise**, and it obeys 1/√N exactly (verified: 1.85 mm at N=405 vs 2.60 mm at
  N=200, ratio 1.41 against a predicted 1.42). Shrink it by raising `MaxLidarPoints` if you need to.
- ~57 mm is **systematic model mismatch** — furniture, doorways, wall thickness the polygon omits.
  No amount of averaging removes it, and **a real room will have more of it than a simulated one.**

So if early exit collapses on the real robot, look at the *floor* before the gate: measure
`early_exit_metric` while parked and see how much headroom is left under the threshold. Widening the
threshold buys speed with pose error and is the wrong lever.

`STABLE_SDF_MSE_MAX = 0.06` gates room creation. If the real floor exceeds it, the room never
stabilises — check this first if the agent seems stuck before anything else.

---

## What to re-derive, in one table

| parameter | file | from |
|---|---|---|
| `PreintOdomSigmaVLat/VLong/Omega` | room_concept config | Phase 1 direct parked estimate |
| `PreintOdomScaleV` | room_concept config | Phase 3 tape-measured leg |
| `PreintOdomScaleOmega` | room_concept config | Phase 2 pivot, minus scrubbing |
| `Base.WheelRadius`, `Base.HalfTrack` | base driver | Phase 2, only if the error is geometry |
| `PoseXYStdSlow/Stop`, `PoseThetaStdSlow/Stop` | controller config | Phase 4b, p90 of moving σ |
| `PoseClampVMax/WMax` | room_concept config | the controller's real limits, not beliefs |
| `SigmaSdf`, `STABLE_SDF_MSE_MAX` | room_concept config | Phase 4c, the real residual floor |

Everything else can start at its simulation value.

---

## The traps, all of which cost real time here

1. **Match the motion population before comparing anything.** A 74%-parked run and a 79%-moving run
   give different aggregates for identical behaviour. I read a population difference as a regression
   twice in one day. Bin by |ω| and compare bins.
2. **A posterior-referenced calibration can only measure a channel NOISIER than its reference.**
   Always compute the direct parked estimate as a guard; `motion_calib` does this and warns.
3. **Regress the error on the REFERENCE, not on the measurement.** Otherwise errors-in-variables
   reports pure noise as a positive scale (+0.0029 measured with a true scale of exactly 0) *and*
   attenuates real slopes (0.0654 where the truth was 0.0700).
4. **Never re-integrate a sparse stepwise channel from logged velocity columns.** Use the localiser's
   own `meas_dth` / `cmd_dth`, which clip segments and use midpoint θ. Continuous odometry survives
   naive integration; the command channel does not (34% error where the truth was 11%).
5. **Timestamps are doubles.** An epoch-ms stamp in float32 quantises to 131 s steps, and every
   interval silently becomes zero.
6. **A statistic that cannot say "I don't know" will eventually be believed when it shouldn't be.**
   The wheel/gyro ratio printed `-9.6824` for a robot standing still, because its guard admitted a
   1 milliradian denominator.
7. **Check which quantity the consumer actually reads** before tuning against a different one.

---

## What is already known to be true and needs no re-measuring

- **The GN/LM solver is correct.** `gn_selftest` checks every factor's analytic Jacobian against
  central differences and recovers a known pose; it needs no robot. Run it after any rebuild.
- **The preintegration algebra is correct**, validated against a 40 000-trial Monte Carlo at 0.6%
  relative Frobenius error, with exact interval chaining and rate invariance.
- **The calibrator itself is correct**, validated against injected truth at 99% / 102% / 84% / 93%
  recovery on σ_v, σ_ω, s_v, s_ω.

What is *not* known is any of the values, on this robot, in this building. That is what the four phases
above are for.
