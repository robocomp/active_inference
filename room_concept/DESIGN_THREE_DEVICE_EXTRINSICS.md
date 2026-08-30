# Pose-free extrinsic calibration from three devices agreeing about one corner

Design note, 2026-08-30. Not built. Extends the camera block in `src/camera_calibration.h`
(`rc::camcal`) from one camera-against-LiDAR pair to the LiDAR plus both cameras.

## 1. The idea, and the one distinction it rests on

Three devices — `helios` (LiDAR), `ricoh` (panorama), `zed` (pinhole) — all mounted on `body`,
frequently see the *same physical room corner* at the same instant. Ask: **what device extrinsics
make those three observations agree best?**

★★★ **The model corner supplies CORRESPONDENCE, never a MEASUREMENT.** This is the whole reason
the scheme is pose-free, and it is easy to get wrong. The matched model vertex is what licenses the
claim "these three observations are of the same physical thing". It must not enter the cost. The
moment you score a device against the model vertex's *position* you need the robot pose to bring the
map into the body frame, the localiser's error contaminates the mount estimate, and the method
becomes an ordinary hand-eye calibration with all the coupling that implies.

The existing single-pair block already works this way, and says so:

> *"the mount, estimated from RGB corners against LiDAR corners with the pose FROZEN — in fact with
> the pose absent from the residual altogether"* — `camera_calibration.h:11`

Everything below preserves that property.

## 2. State, and where the gauge lives

Per camera, the four parameters already defined in `rc::camcal::Param`:

| | |
|---|---|
| `P_PITCH` | boresight pitch (rad) |
| `P_HEIGHT` | mount height (m) |
| `P_YAW` | boresight yaw (rad), **relative to the LiDAR** |
| `P_DT` | image/LiDAR time offset × odometry velocity scale |

Joint state = 8 parameters (ricoh ×4, zed ×4), all expressed **relative to the LiDAR**.

★ **The gauge is fixed by that choice, not by the data.** A rotation common to all three devices
cancels in every residual here and is therefore unobservable — no amount of three-way agreement
recovers it. Parameterising relative to the LiDAR means that common mode is simply *not in the
state vector*; it stays inside the localiser's own gauge, exactly as the current header warns:

> *"A yaw common to both mounts is invisible here and stays inside the localiser's own gauge. That
> is the right target (the goal is the two terms agreeing about one room) but the number must not be
> read as a physical bolt angle."*

So: **no output of this method is a physical bolt angle.** It is a set of relative corrections that
make three devices tell one story. That is the useful quantity; it is not the same quantity.

## 3. The residuals

For each model vertex `k` co-observed at time `t`, with `L` the LiDAR corner in the body frame:

```
r_ricoh = uv_ricoh_meas − project_ricoh(L ; θ_ricoh)     (pixels, ricoh model)
r_zed   = uv_zed_meas   − project_zed  (L ; θ_zed)       (pixels, zed model)
r_cross = uv_zed_meas   − project_zed  ( X_ricoh(θ_ricoh) ; θ_zed )
```

The first two are what exists today, one per camera. The third is new and is the only one that does
**not route through the LiDAR corner**.

★ **Why the cross term is worth the trouble.** Today both cameras are calibrated *against the
LiDAR*, so the LiDAR corner's own positional error is **common mode**: it enters both estimates
identically and cannot be detected from either. `r_cross` is LiDAR-free, so a joint solve can
down-weight a poor LiDAR corner instead of inheriting it into both cameras.

⚠ **And why it is the hard part.** `r_cross` needs the corner's 3-D position from the cameras
alone. Two routes, both with a catch:
- **The corner's known height.** This is what `place_triple_points_in_room` already does — intersect
  the ray with the wall-wall vertical edge at a known height. But that height is the very thing
  `P_HEIGHT` is estimating, so the term becomes partly circular and its Jacobian must include it.
- **The stereo baseline.** The two cameras sit at different heights and offsets on `body`
  (read the actual values from `shadow.json`, do not assume them), so a co-observed corner is
  weakly triangulable with no height prior at all. Baseline is small relative to typical corner
  range, so this is a weak constraint — but weak and *independent* is worth more here than strong
  and common-mode.

Recommend building `r_ricoh` + `r_zed` jointly FIRST (section 6, mode B), and treating `r_cross` as
a second stage once the first is validated.

## 4. The payoff: the pitch/height ridge

This is the concrete reason to do it at all.

`cam_pitch` and `cam_height` are today's standing blocker. From `camera_calibration.h`'s own
`param_why`: they hold **−0.95 to −0.98 correlation in every window**, and driving deliberately
close to walls did **not** fix it, because a LiDAR corner's uncertainty projects as `fy/d` — so the
near-range corners that would separate them carry ~0% of the fit weight (`sigma_v` 18–161 px under
2 m against 1.1 px at 4–6 m). The combination is determined; the split may not be.

★★★ **Two cameras break that ridge in a way one cannot.** The degeneracy direction is set by the
camera's own `fy/d` weighting. The zed is a pinhole with a real focal length at one height; the
ricoh is a panorama whose "focal length" is pixels-per-radian, at a different height. **Their ridges
are not parallel.** A corner seen by both plus the LiDAR constrains the pitch/height combination
along two different directions at once, and the intersection of two near-degenerate constraints can
be well conditioned even though neither is.

**This is a mechanism, not a hope, and it is directly falsifiable:**

> Does the joint zed+ricoh solve reduce the pitch/height correlation below what either camera
> achieves alone, on the same data?

Pre-register that as the acceptance test. If the correlation stays at −0.95 in the joint solve, the
ridges *are* effectively parallel, the argument is wrong, and the honest outcome is that pitch and
height remain a combination — which is worth knowing and costs one run to establish.

## 5. The closure test, which needs no ground truth

The three pairwise relations form a cycle:

```
(zed ← lidar)  ∘  (lidar ← ricoh)  ==  (zed ← ricoh)
```

Estimate the third **independently** and the mismatch is a residual that needs no ground truth, no
pose, and no external instrument. It is the cheapest possible validation of the whole scheme.

⚠ **But it is a test only if you do NOT fold `r_cross` into the joint solve.** With the cross term
in the cost, closure is enforced by construction and the check becomes vacuous — a common way to
lose an instrument by improving an estimator. So the two modes are mutually exclusive per run:

| mode | what you get | what you lose |
|---|---|---|
| **A — joint, with `r_cross`** | best estimates; LiDAR error no longer common-mode | closure is enforced, so no free check |
| **B — three independent pairwise solves** | a closure residual as an honest validation | weaker estimates; LiDAR error stays common-mode |

**Run B first** (it validates the geometry and the associations), then A.

## 6. Observability — which views identify what

Same inverse model as the single-camera block, now needing **co-visibility**:

| parameter | identified by |
|---|---|
| `cam_yaw` | a spread of **bearings** — best determined of the four |
| `cam_pitch` + `cam_height` | a spread of **ranges**; see §4 — judge as a pair |
| `cam_dt` | **ego-motion**; unobservable parked |

★ A three-way constraint exists only where all three devices genuinely see the corner, so the route
must produce **co-visible** corners, not merely many corners. The zed's narrow FOV is the binding
constraint: the ricoh sees everything, the LiDAR sees everything not occluded, and the zed sees a
sector. So the mission wants the robot pointing *at* corners at a spread of ranges, which is a
different route from "tour the perimeter".

★★★ **This design was not safely buildable before 2026-08-30.** Co-visibility is exactly what the
`pi_vis` defect corrupted: `TriplePoint::pi_vis` was pooled with the floor segment weighted by `w`,
and `w` collapses under occlusion, so hidden corners reported themselves visible — **45.3% of
emitted crossings had their vertex behind a wall and 48.5% of those passed the 0.5 gate**. A
three-way scheme fed on that would have been quietly built on corners one device could not see.
Use `rc::visible_and_matched` (`image_edge_types.h`) as the admission rule.

## 7. Traps

- **`cam_dt` × `k_v` is a cross-block confound.** `P_DT` is the time offset *multiplied by* the
  odometry velocity scale; the residual only ever sees the product. Never interpret a `cam_dt` move
  without checking the motion block's `k_v` from the same run.
- **Per-corner detection offsets stay out**, deliberately. They are per-corner, not global, and they
  are why the pose factor was reverted on 2026-08-28 (a ~1.7 px per-corner bias converting into
  heading error through the `corr(x,θ)=0.98` ridge). Putting them in a global vector would be wrong;
  they need corners as landmarks with their own state, which is a larger design.
- **Association must be the same model vertex in all three devices.** A vertex mis-associated in one
  device creates a confident, geometrically consistent, entirely wrong constraint. The LiDAR side
  has `assoc_prob`; require it as well as `visible_and_matched` on the camera side.
- **The two cameras run through different code paths today.** `ricoh` is the driving camera and
  accumulates into `mp_pool_` via `mount_pair_update()`; `zed` is a calib channel with its own
  `ch.calib` inside `calib_channels_`. A joint solve wants them symmetric — normalise this before
  adding parameters, not after.
- **Persistence is keyed per (robot, camera)** — `etc/camera_calib_<robot>_<camera>.txt`, holding
  H and b as sufficient statistics, never fitted values. A joint solve has a joint H that does not
  factor into those files. Either keep per-camera evidence and joint-solve at load, or change the
  layout deliberately; do not let two representations of the same evidence coexist.
- **Evidence, not parameters.** Whatever the layout, keep the existing discipline: saving fitted
  values and restoring them as prior means is a ratchet in which each session inherits the last
  one's answer as though it were data.

## 8. Staging

1. **Normalise the two camera paths** so ricoh and zed produce the same `PairObs` stream through the
   same code. No new maths. (Prerequisite; also fixes a real asymmetry.)
2. **Mode B**: solve the two pairs independently, plus an independent zed↔ricoh pair, and log the
   closure residual. Validates geometry and association with no new parameters.
3. **Measure the §4 test**: pitch/height correlation, joint versus each camera alone.
4. **Mode A**: fold `r_cross` in, with the height circularity handled explicitly in its Jacobian.
   Only worth doing if step 3 says the ridges are not parallel.

Step 3 is the decision point. If the correlation does not improve, stop at mode B — a closure test
that needs no ground truth is a good outcome on its own, and it is the part that generalises to the
real robot where there is no supervisor to appeal to.
