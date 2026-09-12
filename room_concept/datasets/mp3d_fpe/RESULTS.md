# MP3D-FPE: what it can and cannot measure for this system

401 rooms, 18645 real 360° scans, 49 scene-versions (33 houses), sliced from real Matterport depth
and real camera trajectories. Registration verified independently: modal offset **7.5 mm** median
across 12 random rooms. Corpus is in the `WS_LAYOUTS` format; `tools/mp3d_replay` drives the real
estimator from the recorded scans.

## ❌ It cannot grade our layouts — and the reason is a property of the benchmark

MP3D-FPE annotates **rooms inside open-plan flats**, and a 360° depth panorama sees straight through
every doorway. Measured over all 401 rooms, the fraction of returns falling outside the polygon being
scored:

| target | returns outside |
|---|---|
| the annotated ROOM | median **57%** (p25 45%, p75 69%, best room 12%) |
| the whole FLOOR (union of that scene's rooms) | **50%** |

**Not one room of 401 is below 10% leakage; three are below 20%.** Per-room the sensor sees more than
the label covers; per-floor it still does, because corridors, stairwells and closets are not annotated
at all. There is no granularity at which the annotation and the observation describe the same region.

Our method models a *bounded* room and has no mechanism to attribute a return to one, so it fits a
polygon around whatever it is shown: direct per-room replay gives IoU ≈ 0.14. **That number measures
the task mismatch, not the estimator, and must not be reported as a result.**

## ✅ What it does measure: the real-sensor split our calibration claim rests on

The claim is that error splits into a part that averages down with more observation and a part that
does not. Both halves were previously measured only against a *simulated* 2 cm sensor. Fitting each
annotated wall segment from the returns that land on it (middle 70% of the segment, 12 cm band, 588
segments over 120 rooms):

| | median | p90 |
|---|---|---|
| scatter about the fitted wall — **averages down** | **0.0149 m** | 0.0382 |
| fitted wall vs annotated wall — **does not** | **0.0181 m** | 0.0572 |
| tilt, fitted vs annotated | 0.80° | |

**1. The harness's simulated sensor is honest.** Real Matterport depth scatters at 1.5 cm against the
2.0 cm simulated. The 594-room calibration, and the 0.019 m floor measured against that simulated
sensor, rest on a noise model that is slightly conservative rather than optimistic.

**2. The systematic term belongs to the MAP, not the sensor.** A careful Matterport annotation sits
1.8 cm from the observed wall; our traced SVG of the apartment sits **9.5 cm** from it (measured on
the robot, see `TODO_PER_WALL_OFFSET.md`). Same sensor class, same estimator, systematic component
different by 5×. That is the empirical case for modelling the offset as an inferred state with a
prior rather than a constant: a constant tuned in one environment is wrong by 5× in the next.

⚠ The 1.8 cm includes any residual registration error from the slicer (modal offset 7.5 mm), so it is
an **upper bound** on annotation error; the true figure is likely nearer the sensor scatter, which
would make the two indistinguishable.

## Reusable

`tools/mp3d_replay` works and drives the production estimator through the real code path (one frame
bug found and fixed in it: odometry is a MAP-frame delta, `pred = est + odom` componentwise — a
body-frame composition drops IoU to 0.14 on its own and looks exactly like an estimator failure).
If a bounded-environment dataset with real depth turns up, the runner is ready.
