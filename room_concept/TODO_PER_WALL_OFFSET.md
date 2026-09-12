# TODO (after ICRA 2026-09-15): model the per-wall layout offset as a state

## Why

A stationary robot's pose covariance collapses to 0.0152 m while its innovation *grows* to 0.085 m.
Measured over a 200 m tour (126k candidates, `tmp/corner_probe.csv`, analysis in
`tmp/TOUR_CALIBRATION.md`), with inter-frame motion recovered by a Kabsch fit on the predicted
corners:

| motion | frames | sigma_pred | sigma_det | \|nu\| median | NIS |
|---|---|---|---|---|---|
| PARKED <0.25°, <5 mm | 16946 | 0.0152 | 0.0432 | 0.0851 | 3.045 |
| turning >1° | 1896 | 0.0475 | 0.0435 | 0.0657 | 1.500 |

85% of the tour is spent parked, which is where the channel is worst. `sigma_det` is flat across every
band, so this is **not** the corner channel — the NIS instrument localised a defect in another module.

**The split is measured.** Per corner across 25+ consecutive stationary frames (20 stretches, 230
corner-stretch pairs): the part of the innovation that REPEATS is **0.0952 m**, the part that
AVERAGES is **0.0113 m** — ratio 8.1x, bias exceeds scatter in 97% of cases. The estimator handles
the noise correctly: averaging 1.1 cm of real scatter earns exactly the 0.0152 m it claims. The other
9.5 cm is a systematic offset between the traced layout and the real wall, identical in every frame
from a fixed viewpoint, and represented nowhere in the model.

## What to build

Give each wall landmark an offset along its normal, `b_w ~ N(0, sigma_b²)`, estimated jointly with
pose and walls. From a fixed viewpoint pose and offsets are partly unidentifiable — that is correct,
and it is exactly why the covariance must stop shrinking there. Motion restores identifiability, so
`P_pose` shrinks when and only when the robot has earned it.

- **sigma_b ≈ 0.095 m**, measured above. Not a guess.
- **Cheaper first step**: common-mode marginalisation instead of new states — `R_w = sigma²I +
  sigma_b²·11ᵀ` applied by Woodbury, so information from N repeated looks saturates at `1/sigma_b²`
  instead of growing as `N/sigma²`. This codebase already solved this exact problem once, for
  correlated mask points (see CLAUDE.md's modelling philosophy).
- **Rejected**: inflating the covariance at rest. It is a threshold, it contradicts the project's own
  "no constant process noise" rule, and it leaves `|nu|` untouched — symptom, not cause.

## The constraint that comes with it

⚠ **`CornerDetector::Params::base_sigma` must drop back to sensor level in the same change.** It is
currently 0.06 m precisely because the map's misfit is modelled nowhere else. Introduce `b_w` and the
same physical quantity is counted twice — which is the exact bug `f28a21d` was fixing. The constant is
a stand-in for a state; once the state exists, the stand-in has to go.

Note also why no constant can be right: the misfit is 9.5 cm parked but 0.06 m calibrates the whole
run, because in motion the per-wall offsets partially CANCEL in the pose solution. Which value applies
depends on the trajectory. States get that cancellation from geometry instead of from a tuned average.

## Acceptance criteria (pre-registered, before building)

1. **NIS ≈ 1 in BOTH the parked and the turning bands**, not on average — an aggregate can look right
   while both halves are wrong in opposite directions.
2. `sigma_pred` at rest stops falling and settles near its moving value (~0.045 m).
3. **The residual after subtracting `b_w` falls to ~0.011 m**, the true sensor-level scatter. This is
   the one that tests the mechanism rather than the symptom, and the hardest to hit by accident.

Instrument already exists: `tmp/corner_probe.csv` + the motion-banded analysis.

✅ Ruled out: duplicate frames. New scans confirmed arriving while parked.
❌ Ruled out: hairpin turns. Turning is where the channel behaves *best* (corr(|dθ|, NIS) = −0.219).
