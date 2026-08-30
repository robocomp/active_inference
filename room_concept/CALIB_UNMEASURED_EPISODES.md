# The motion calibrator is learning from episodes in which nothing was measured

Finding, 2026-08-30. Measured on a live Shadow run, cold start, `calib straight`.
Nothing has been changed yet — this is the argument for a change, written first on purpose.
Related: `EXPERIMENT_CALIB_PARAMS.md`, `src/motion_calibration.h`, `src/calibration_estimator.h`.

## 1. The headline

**94.2% of the Fisher information in the calibration window comes from episodes that contain no
measurement at all.**

Not 94% of the episodes — 94% of the *information*. The estimator is overwhelmingly being taught
by rows asserting "the correction was exactly zero" that were produced on cycles where nothing
solved for a correction.

## 2. The measurement

Live window, 512 episodes, and `gt_error.csv` from the same run:

```
iters==0   5369 cycles (98.4%)   nonzero correction on  0.2% of them
iters>0      89 cycles ( 1.6%)   nonzero correction on 96.6% of them

episodes in window                     512
zero-correction episodes               438   (86%)
median |r_forward| / |r_lateral| / |r_theta| over the window:  0.00000 / 0.00000 / 0.00000
```

And the part that turns a nuisance into a defect — the zero rows are **better weighted** than the
real ones on every axis:

| | zero-correction | real correction |
|---|---|---|
| median `pos_var` | **0.004945** | 0.007436 |
| implied `sigma_pos` | **0.070 m** | 0.086 m |
| median `\|d_forward\|` (the covariate) | **0.2512 m** | 0.1592 m |
| count | **438** | 74 |

Tighter variance, **larger** covariate, six times as many. Fisher information on a
distance-regressed parameter, `sum d_forward^2 / pos_var`:

```
zero-correction episodes : 6.740e+03   (94.2%)
real-correction episodes : 4.124e+02   ( 5.8%)
```

Visible in the parameters right now, cold-started 10 minutes earlier: `k_v` 0.999640,
`k_omega` 0.999641, `eps_yaw` **exactly 0.00000**, every sigma shrinking, `informed` = 5
(`k_v` and `k_omega` only — `eps_yaw` never becomes informed because the lateral correction is the
rarest of the three, so its rows are the emptiest).

## 3. The mechanism

`MotionCalibrator::observe` (`src/motion_calibration.h`):

```cpp
if (corrected) {                    // ← only when the optimizer actually ran
    acc_r_fwd_ += r_forward; acc_r_lat_ += r_lateral; acc_r_th_ += r_theta;
    acc_pos_var_ = std::max(acc_pos_var_, pos_var);
    acc_th_var_  = std::max(acc_th_var_,  theta_var);
}
const bool moved_enough = std::abs(acc_fwd_) >= cfg_.episode_min_trans
                       or std::abs(acc_th_)  >= cfg_.episode_min_rot;
if ((prev_corrected_ and not corrected) or moved_enough) flush();
```

The episode closes on **motion**. The correction accumulates only when the optimiser **ran**. At
98.4% early exit those two almost never coincide, so an episode closes having travelled 0.25 m with
`acc_r_* == 0` and `acc_pos_var_ == 0`, and `flush()` emits it as a row saying *"over this span the
correction was zero"*. **"Not measured" is recorded as "measured zero".**

## 4. ★★★ The safeguard inverted

`flush()` sets the observation variance:

```cpp
const float fit_model = cfg_.fit_model_gain * acc_fit_;          // 2.0 x max|SDF| in the episode
const float r_pos = std::max(acc_pos_var_, cfg_.min_obs_var) + rot_model^2 + fit_model^2;
```

`fit_model_gain` was added on 2026-08-23 for a good reason: *a correction is only as good as the
FIT that produced it*, so a window where the localiser is not tracking inflates its own variance and
collapses its own gain. It was the cure for the burst that dragged `k_v` from 1.0059 to 0.8907.

But with `acc_pos_var_ == 0` that term is the ONLY thing setting the variance — and **an early-exit
cycle is by definition one whose SDF residual was small**, because that is what the gate tests. So a
never-measured episode carries a small `acc_fit_` and therefore the *tightest* variance in the
window. The term written to distrust bad fits now makes unmeasured episodes the **most trusted
evidence the estimator has**. That is the whole defect in one sentence.

## 5. How we got here — and it is our own change

The episode trigger was moved from the optimiser's falling edge to motion on 2026-08-26, and the
comment states the premise:

> *"the SDF polish now corrects a little on EVERY cycle, so there is no longer a ramp waiting for a
> single large correction to end it"*

**That premise is false in this build.** Measured today: of 5369 early-exit cycles, **0.2%** carry
any correction at all. There is no per-cycle polish on the published pose.

It may have been true when written. What made it false is ours: the ceiling-band fix `8164fc3` and
the covariance fix `f2336d0` took `early_exit_pct` from 0 to ~98 on 2026-08-29. The trigger change
kept episodes *flowing*; nothing kept them *carrying anything*. **Fixing the localiser starved the
calibrator**, and the 08-26 change converted that starvation from "no episodes" (loud — every
parameter reads exactly 0.000 with sigma 0.000, which is the signature that was noticed and fixed)
into "empty episodes" (silent — plausible values, shrinking sigmas, apparent convergence).

★ The failure got quieter as it got worse. That is the part worth remembering.

## 6. Why widening the variance is NOT the fix

The obvious Active-Inference-shaped answer is to grow `R` for an unmeasured episode rather than gate
it out. Measured, that does not work: the zero rows already carry `sigma_pos` 0.070 m, which is
about the size of the early-exit gate itself (`0.06 + 0.2|dtheta|` m). They are not being believed
too precisely. **438 assertions of "zero" at 7 cm beat 74 measurements at 8.6 cm** on weight of
numbers alone, whatever the variance.

The problem is not the precision attached to the observation. It is that **an observation is being
asserted at all.** No correction was computed; there is no measurement; a row saying `r = 0` with
any finite variance claims something nobody looked at.

## 7. The proposed cure

**An episode emits a row only if at least one corrected cycle contributed to it.** Otherwise its
accumulated motion is NOT discarded — it carries forward into the next episode, so the covariate is
preserved and the next real correction explains a longer span.

This is not a threshold. `corrected` is a boolean fact about whether the optimiser ran, not a tuned
cutoff, and the distinction it draws — measured versus not measured — is the one distinction an
estimator is never allowed to blur.

It also restores what the falling-edge trigger got right, without reintroducing what it got wrong:

| | falling edge (pre 08-26) | motion (current) | proposed |
|---|---|---|---|
| episode ends when | a correction ends | enough motion | enough motion |
| emits a row when | a correction ended | always | a correction was seen |
| at 98% early exit | almost never emits | emits constantly, empty | emits when there is something to say |

A cap on carried-forward motion is still needed so a long uncorrected stretch does not produce one
enormous episode whose linearisation is invalid. The existing `1e4` guard is far too loose to serve;
size it against the span over which the motion model is linear, and call it a numerical guard rather
than a tuning knob.

## 8. How to verify the cure

Pre-register these, because "the parameters look more sensible" is not a result:

1. **Zero-correction rows in the window fall to 0%.** Directly checkable in
   `etc/motion_calib_state.csv`.
2. **`eps_yaw` becomes `informed` on a distance-biased route.** It has been pinned at exactly
   0.00000 through every run today; its regressor is forward travel, so the corridor route is
   precisely where it should move. If it still does not, the cure is not sufficient and the
   inverse-model table has a second error in it.
3. **Episode rate drops by roughly the correction rate** (~1.6% of cycles carry corrections, so
   expect far fewer episodes per metre). Slower, correct learning is the intended trade.
4. **The injected-error arm still recovers.** `SensorNoise.WheelScaleV = 0.03` must still drive
   `k_v` to 1/1.03 x native. A cure that cannot recover a known error has removed too much.

★ Note that (4) is the one that would have caught this defect. The estimator was validated against
injected errors on 2026-08-22/23, when the optimiser fired every 1–6 s and the window was full of
real corrections. It has not been re-validated since the early-exit regime changed — and the
validation is exactly what the regime change invalidated.

## 9. What it means for the experiment right now

**Phase 1 cannot start until this is settled.** Every baseline taken in this regime measures the
zero rows, not the robot: 94% of the information is fictitious, and a baseline is what every
injected arm would be graded against. The ten `calib straight` repeats are on hold.

The route bias is a real but secondary issue — `calib straight` is running at 0.64 rad/m on moving
rows, so the shuttle's U-turns dominate its 6.6 m corridor and it is not the distance-biased route
it was designed to be. Worth fixing, but not while the evidence is 86% empty.
