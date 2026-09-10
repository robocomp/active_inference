# Ego-motion mask corruption — producer done, consumer (table) next

Handoff for the table_concept session. Producer side (retina) is implemented and builds green
(UNCOMMITTED). This explains what's now on the `masks` node and what the consumer must do with it.

## Idea (one paragraph)

You can't tell a noisy mask from genuine unmodelled state change without strong world-priors that
kill adaptability. So instead **downweight/gate masks captured while the camera is moving** — saccadic
suppression. This is a prior on the *sensor channel* (efference-gated heteroscedastic `R`), **not** on
the world, so it costs no adaptability: the moment the camera is still, the penalty vanishes and the
mask is trusted fully. Stronger for **rotation** (no `1/Z` depth attenuation; range-amplified;
edge-compounded) and for masks at the **FoV edge / truncated** (a *bias*, not noise). Rule of thumb:
**symmetric/random error → downweight; biased/systematic error → gate.**

Math: image interaction matrix → metric corruption ≈ `Z·‖ṡ‖·Δt`. `Δt` splits into exposure blur +
timing jitter (→ variance) and a timing **offset** (→ bias). Lives in `common/motion_corruption/`.

## Division of labour (already decided)

- **Producer (retina)** — reports the *physics of capture*: the corruption decomposition. DONE.
- **Consumer (table_concept)** — decides what it does to the belief. **THIS IS THE WORK.**

## What the producer now publishes on the `masks` DSR node

Per-mask arrays (1-to-1 with `mask_label_ids`, gated by `MaskMotion.enabled`, default on):

| attr | meaning | consumer use |
|---|---|---|
| `mask_motion_dotd` | metric position-corruption speed `Z·‖ṡ‖` (m/s) | diagnostic / per-DOF input |
| `mask_motion_bias` | systematic displacement from a known timing offset (m) | **gate** when `> b_max` |
| `mask_motion_var` | variance to ADD to `R` (m²): blur + jitter, ×peripheral | fold into innovation cov |
| `mask_trunc_frac` | fraction of silhouette pixels on the image border | truncated face → unobserved |
| `mask_centroid_radius` | normalized centroid radius from principal point | periphery handling |

Per-frame (global) on the node: `mask_cam_twist` = `[vx,vy,vz,wx,wy,wz]` (optical frame),
`mask_frame_dt_s`. (`bias` is 0 until `MaskMotion.timing_offset_s` is set to the measured RT-vs-mask lag.)

All are dynamic attrs read the same way as the existing `mask_*` (e.g. `mask_centroids_xyz`).

## Consumer TODO (table_concept)

1. **`common/mask_ingestor`** (the single producer↔consumer agreement point): parse the new attrs into
   `MaskSlice` (e.g. `motion_var`, `motion_bias`, `trunc_frac`, `centroid_radius`).
2. **Fitter**: add `motion_var` to the measurement/innovation cov `S` (alongside the YOLO-score cov and
   chain cov already there). This is the downweight.
3. **Gate**: when `motion_bias > b_max` (a *per-concept* tolerance — table is cm-scale, so looser than
   bottle), drop the geometric UPDATE for that frame, but **keep tracker association running on `S`**
   (don't lose the instance just because the robot moved).
4. **Truncation**: `mask_trunc_frac > 0` → treat the clipped boundary as an **unobserved face**
   (zero info), reusing the face-aware RFE / per-face precision machinery — do NOT inflate variance for
   it (it's a bias, not spread).
5. (Later) **per-DOF split**: centroid-flow → position precision; flow-gradient across the mask →
   extent/`info_w/info_h`. v1 can use the single scalar `motion_var` for all DOFs.

## Verify first

Twist is finite-differenced from the zed pose, so with a static robot `mask_motion_dotd` should be
≈ 0 and spike when the base/camera pans. Confirm that before trusting the weights.

Files: `common/motion_corruption/motion_corruption.h`, `retina/src/graph_publisher.cpp`
(`upload_masks` + contract comment), `retina/src/retina_params.{h,cpp}` (`MaskMotion.*`).
