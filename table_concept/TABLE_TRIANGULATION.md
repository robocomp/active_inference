# Table landmark: anisotropic-covariance triangulation

## Problem (measured, 2026-07-13)

A single-view mask centroid is a **biased** estimate of the table centre — you see the near face + top,
so the centroid sits toward the camera along the viewing ray. Consequence, from `table_landmark_*.csv`:
the table's room-frame estimate is **coupled to the robot pose** — `corr(room_y, robot_y)=+0.84` on
*fresh* frames, and the estimate shifts ~0.44 m by viewing side. The physical table does **not** move;
its estimate is dragged because the robot's motion doesn't cancel when composing `robot_pose ∘ centroid`.

This is a systematic, robot-correlated bias — worse than IID noise, because a landmark factor would try to
explain it by moving the **robot** pose. So the table can't be a trustworthy landmark until the per-view
bias is handled by fusing across viewpoints.

## Model: declare the ray untrustworthy, let fusion discard the bias

Measurement model for a single view: `z = c + δ·d + noise`, where `c` = true centre, `d` = horizontal
unit viewing ray (camera→centroid), `δ` ≈ half the unseen depth (unknowable from one view).

Give each detection an **anisotropic** covariance, tight ⊥-ray, loose along-ray:

```
R   = σ_perp²·(I − d dᵀ) + σ_along²·(d dᵀ)          σ_along ≫ σ_perp
R⁻¹ = (1/σ_perp²)·(I − d dᵀ) + (1/σ_along²)·(d dᵀ)
```

**Why it removes the bias:** `δ·d` lies along `d`, which `R⁻¹` weights by `1/σ_along² ≈ 0`. The biased
component is in the near-null-space → it contributes nothing; only the **⊥-ray component of `z`, which is
unbiased**, carries weight. Fuse ⊥-components across azimuths → the true centre, bias never corrected
because never trusted. In the `σ_along→∞` limit this **is** bearing-only triangulation (each view = a line
constraint ⊥ its ray; rays from different sides intersect at `c`).

Verified in `scratchpad/raycov_test.cpp`: single view cond ≈ 132 (elongated along ray); two ⊥ views fuse to
cond 1.0 (tight); 0.25 m/view bias → 0.000 m fused error.

## Setting the two sigmas (see `common/object_anchor/ray_anisotropic_cov.h`)

```
d       = normalize(centroid_xy − cam_xy)                 # cam = ZED optical centre, NOT robot origin
L_ray   = a·|cos φ| + b·|sin φ|                            # box half-shadow on d; (a,b)=half-extents, φ=∠(d,box x)
σ_perp² = σ0²                                              # lateral floor, a few cm
σ_along = max(σ_along_min, (c_bias·L_ray + α·range²) / obliquity_cos)
R       = σ_perp²(I − d dᵀ) + σ_along² d dᵀ
```
`c_bias`~0.7 (near-face fraction), `α`~0.01/m (ZED depth noise), `obliquity_cos`∈(0,1] (grazing → bigger
σ_along). All are in `RayCovParams`.

## Frames (critical)

room_concept's object-anchor factor computes the residual **in the ROBOT/body frame**
(`r = z_o − R(−θ)(p_o − t)`), so **R must be built in the robot/body frame**. That's natural: `d` comes
from the body-mounted camera. `box_yaw` for `L_ray` must be in the body frame too:
`box_yaw_body = belief_yaw_room + yaw(body←room)`.

## Two integration points (both use the same `R`)

**A. Publish the per-view R_o so room's factor is correct (implemented 2026-07-13).**
`TableFitter::compute_object_observation` now fills `inst.obs_robot_cov` (2×2, body frame) via the helper;
`object_anchor::write_observation` publishes it as `obj_obs_robot_cov = [Rxx, Ryy, Rxy]` (position-only,
3 floats = symmetric 2×2). room_concept's `object_anchor_source` reads the 3-float form and folds the full
2×2 into the innovation `S`. This alone stops the along-ray bias from corrupting the robot pose (the factor
just won't pull along the untrusted axis). Does NOT change table_concept's own fit/belief/control.

**B. Fix the MAP pose so it stops sliding with the robot (TODO — belongs with the belief work).**
The table's published room pose (`cx,cy` in `ai2_belief`) is still folded from each view's centroid
*isotropically* → it stays viewpoint-coupled. Change the belief's centroid measurement update to use the
same anisotropic `R` (rotate to the belief's frame): its information contribution is `R⁻¹` (Jacobian wrt
`cx,cy` is identity). Then multi-view fusion converges `cx,cy` on the true centre, `corr(room,robot)→0`,
and `obj_obs_robot` should become the fused belief mean rather than the raw per-frame centroid. This is the
full triangulation; A is the defensive subset.

## Notes
- Viewpoint **diversity** does the work, not frame count — same-side views share the bias. Weight by the
  anisotropic R and the geometry takes care of it (a redundant same-side view adds little new ⊥ info).
- Freshness-as-precision (room side) already inflates R_o by staleness; this is orthogonal (per-view
  accuracy vs staleness). They compose.
- Sparse YOLO detections are fine here — fusion is incremental; each rare good view tightens `Λ`.
