# Pre-registration — real-robot tour, Shadow, real apartment

**Written and committed BEFORE any data from this run is examined.** The paper argues that a declared
uncertainty is only worth something if it is checked; choosing after the fact which numbers to report
would be the one unforced error available to us here. This document fixes the analysis so that it
cannot drift.

Date: 2026-09-14. Platform: Shadow, real hardware. Place: the real apartment (`Scenario.apartamento`).
Reference layout: MRPT-SLAM-derived, offline, precision to be stated by its author and quoted as an
UPPER bound on our offset, exactly as the Matterport 1.8 cm is.

## What will be reported, whatever the values

1. **NIS/dof at the corner association gate**, over the whole tour, and binned by per-frame |Δθ| into
   the SAME bands as Fig. 4: `<0.25`, `0.25–0.5`, `0.5–1`, `1–2`, `2–4` deg/frame. Report **n per
   band**. A band with n < 100 is reported and labelled underpowered; it is not dropped.
2. **The repeat/scatter decomposition of the corner innovation at stationary frames.** For each corner
   observed over a stationary stretch: the per-corner MEAN innovation is the part that *repeats*, the
   spread about it is the part that *scatters*. Report both, their ratio, and the fraction of corners
   whose bias exceeds its scatter. Stationary is defined BEFORE the run as |v| < 0.02 m/s and
   |ω| < 0.02 rad/s sustained for ≥ 2 s.
3. **The map offset against the MRPT layout**: median and p90 of the fitted-wall-to-reference-wall
   distance, with the count of walls contributing.

All three are map-relative or internal. **None requires ground-truth pose**, which this run does not
have — that is why these three and no others.

## What will NOT be claimed

- No realised-corner-error calibration, and no IoU. Both need truth corners in a common frame, which
  needs a pose truth we do not have. This run is real-sensor validation of the **precision
  decomposition**, not of the calibration table.
- No absolute localisation accuracy.
- The offset will be quoted as an upper bound: it contains the MRPT reference's own error.

## Decision rules, fixed in advance

- **Phase 1 GO/NO-GO.** Parked spread is compared against the configured `PreintZuptDensityV/Omega`
  (σ_v 0.0202 m/s, σ_ω 0.0323 rad/s, measured 2026-08-18 over 83 791 zero-command rows). Within 2x:
  the constants transfer, proceed to the tour unchanged. Beyond 2x: re-derive Phases 1–3 first, and
  report in the paper that the tour ran on re-derived constants. Running the tour on constants that
  fail this check would make the NIS a measurement of our own un-transferred tuning.
- **Validity of the tour for the motion-state claim**: ≥ 50 m travelled, ≥ 60 s of cumulative
  stationary time, and ≥ 2 rotation bands with n ≥ 100. If not met, we report only the aggregate and
  say the band contrast is not supported by this run.
- **Webots vs real disagreement**: both are reported side by side. Neither replaces the other. This
  follows how the null exploration ablation is already handled.
- **Hard abort**: if clean data is not in hand by the time set by the author, the paper is submitted
  as it stands. Both prior reviews rate it submission-quality; this experiment is additive.

## Run conditions fixed in advance

- `given` mode, MRPT layout as the map, so the comparison with the Webots tour is like-for-like.
- Tour contains deliberate stationary stretches AND rotation at several rates; not one constant-speed
  loop.
- Files kept before the next run truncates them: `tmp/corner_probe.csv`, `tmp/layout_trace.csv`, the
  reference layout in `WS_LAYOUTS` format, the debug log for `motion_calib`.
- Bench checks first: `LearnMotionModel = false`; `PoseClampVMax/WMax` set to the controller's real
  limits; `gn_selftest` and `motion_calib --selftest` both pass.

## Anonymity

Described in the paper as "a differential-drive robot with a 3D LiDAR, in an apartment-scale indoor
space". No robot name, laboratory, institution or building.
