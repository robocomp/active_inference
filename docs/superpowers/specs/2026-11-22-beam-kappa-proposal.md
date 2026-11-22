# Proposal: the sweep weight κ in the beam energy (spec §7c.8) — owner 11-22 chose A, adopted as rev 3.3.11

Owner 11-22: "Revisit κ in the spec". Draft only; no code until the owner picks an option.

## What the spec says today

§7c.8 "Sweep weight": the ray-set energy is κ·Σ_i E_i with κ = coverage·ca, both "computed as today on the returns whose
endpoint is inside the box, from the unfloored label q_i":

- coverage = min(1, Σq / n0), n0 = `LidarCoverageN0` = 60;
- ca = (1 − R)^p, R the q-weighted bearing resultant about the centre, p = `LidarCoverageAngPower` = 1.

Both come from the pre-beam model (`lidar_select.h` `select_into_weighted`), where they set a per-sweep precision
`base·coverage·ca` for endpoint-in-box SDF factors. Their stated purpose (config.toml comment): stop one-sided sweeps
from over-committing the near-square w ↔ h mode from a single view.

## What live11 shows (from the ai2 logs, staged rows only)

| instance | beam-staged rays med | endpoint-in-box returns med | Σq med (of n0 60) | ca med | κ med / max | Itr_L med |
|---|---|---|---|---|---|---|
| bed_21 | 245 | 2 | 0.71 | 0.151 | 5e-7 / 0.033 | 0.28 |
| nightstand_40 | 111 | 22 | 11.5 | 0.424 | 0.061 / 0.183 | 365 |
| nightstand_41 | 126 | 10 | 3.7 | 0.105 | 0.006 / 0.017 | 192 |

The energy sums over ~100–250 rays staged by ray geometry, but its weight is computed from a DIFFERENT, much smaller set
(the 2–22 returns whose endpoint lies in the box). On the bed the second set is nearly empty, so κ ≈ 0 and the beam does
not measure, although 245 rays crossed the model. This is a mismatch inherited from the old model, not a property of
the beam.

## Why κ double-counts in the beam model

- **Ray count.** Each ray already contributes its own term E_i, weighted by its own attribution m_i. Twice the rays is
  twice the information, which is right for independent rays. What makes rays NOT independent is the shared errors
  (robot pose e, the per-sweep range offset), and the persistent update already carries them as latents (rev 4,
  persistent shared errors). Tempering by n/n0 on top of that treats the correlation twice.
- **Angular spread.** A one-sided view constrains the near face and the two silhouette edges, and says almost nothing
  about the far face. In the beam model that shows up in the likelihood itself: the far-face direction is flat in
  J's curvature, so P stays wide there. ca shrinks EVERY direction, including the ones the view does constrain (cx,
  w across the line of sight). The over-commitment it was added against came from endpoint-in-box SDF factors,
  which had no miss/see-through terms; the beam model has them.
- This also gives a candidate explanation for two open live11 items: h over-wide (std 5–10× rms) and shifted size
  priors surviving one look — both are what a LiDAR channel tempered toward zero looks like. Candidate only.

## Options

**A (recommended): κ ≡ 1 on the beam path.** Remove the sweep weight from the beam energy, gradient and curvature.
"The LiDAR channel measured" becomes: the staged set is non-empty and Σ m_i over it > 0 (the camera attributes
something to this object). coverage and ca stay computed for the instruments and the flag-off path. Risk: if the
shared-error latents under-model the ray correlation, P comes out over-confident — the two-sided calibration check
catches exactly that.

**B: recompute κ on the staged set.** Same formula, but Σm and the bearing resultant over the beam-staged rays. Keeps
the old tempering, fixes the mismatch. Still shrinks the directions a one-sided view does constrain.

**C: derive the correlation.** Replace κ with an effective-sample factor from a measured along-face range-error
correlation. Most principled for genuinely correlated rays, needs a measurement first; only if A fails calibration.

## How A would be tested (Pass, before any number is seen)

- Unit: calibration_nees (silhouette fixture, rev 3.3.10 (2)) with κ ≡ 1: NEES/2 ∈ [0.5, 2] as today; a one-sided
  fixture (box seen from one face only): the far-face direction's σ stays ≥ its prior σ/2 and NEES ∈ [0.5, 2].
- Unit: creep (16 cases) and silhouette bounds unchanged (ray spacing, rev 3.3.9/3.3.10).
- Live (live12 on + shift, the live11 tour): no DOF over-confident (0 % below rms/3) — the guard; h std / rms reported;
  shift run's final w, h, H errors reported against the shifted prior's +0.06 / +0.06 / +0.10; bed Itr_L reported.
  A failure is reported, not tuned.
