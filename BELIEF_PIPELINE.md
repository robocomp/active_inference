# Object belief pipeline — what changed (2026-06-27..29)

Short orientation for anyone working on the mask-fitting concept agents (`bottle_concept`,
`table_concept`, `chair_concept`). These changes harden how each agent's **object belief
(pose mean + covariance)** is formed, tracked, and published on the DSR RT edge. All UNCOMMITTED-ish
(some committed), all build green. `retina` produces the masks; the agents fit + publish.

## The belief = mean + covariance, formed in stages
1. **Perception** — `retina` writes the `masks` node (per-instance YOLO mask slices + 3-D support
   points). It dual-publishes the support points in BOTH room frame (`mask_support_points`, legacy) and
   camera frame (`mask_support_points_cam`).
2. **Association / lifecycle** — shared `common/instance_tracker/instance_tracker.h` assigns detections
   to instances, births new ones, retires gone ones.
3. **Fit** — each agent's SDF free-energy fit estimates the object pose/shape from the support points.
4. **Publish** — the fitted pose + a 6×6 covariance go on the room→object RT edge (`rt_covariance`).

## 1. YOLO score → covariance (all 3 agents)
The detection confidence is per-observation reliability → it weights the observation precision via
`w = clamp01((conf−floor)/(ref−floor))^power`, widening the belief for weak masks. Applied at the
front of the fit (NOT a post-hoc Σ scale). table/chair already had it; bottle was added. Config
`*.MaskConf{Weight,Floor,Ref,Power}`. Verified: σ scales ~`1/√w`.

## 2. Instance tracker (shared) — robustness rules
- **Association gates on the INNOVATION covariance `S = P + R²I`** (`Tracker.DetectionNoiseM`), NOT P
  alone — gating on P alone (overconfident σ~mm fit) rejects every real detection → churn.
- **Merge operator** (worker-level): two instances on the same object are collapsed (geometry-specific
  overlap: circle for bottle, oriented-rectangle for table/chair).
- **Negative-information DEATH** (`TrackView.expected_visible`): an instance is retired only when it
  SHOULD be visible (projects inside the camera frustum) yet isn't — out-of-FoV absence is NOT death, so
  objects PERSIST when you look away.
- Optional NLL assignment cost (`Tracker.NllCost`, off) — compete by likelihood under clutter. Matching
  is greedy (fine at these instance counts; Hungarian is the upgrade if dense clutter appears).

## 3. CV motion model — BOTTLE ONLY (movable object)
A bottle is movable: its position TRACKS via a constant-velocity Kalman filter
(`common/motion_filter/cv_filter.h`) on cx,cy, instead of the maturity-locked static stabiliser (which
froze and spawned a new instance when the bottle moved). Key points:
- Measurement = the FRESH-frame observation centroid (de-projected onto the cylinder axis), pinned to
  the capture timestamp — NOT the queue-dragged SDF fit.
- Robustness: innovation gate + velocity clamp + a position-variance cap (keeps the gate tight across
  dropouts → no swap/jump) + zero-velocity on detection gap (HOLD, don't coast).
- Tuned static-when-resting: a resting bottle is "as static as a table"; real motion comes from a GRASP.
- Config `[Dynamics]` (Model/CvAccelStd/CvMeasStd/CvGate/CvMaxSpeed/CvMaxPosStd/CvLostFrames/…).
- table/chair are STATIC (furniture) → no CV; they keep the harden info-filter.
- ★NEXT (not done): grasp mode = known control input — re-parent the bottle under the gripper frame so
  its pose comes from the kinematic chain (occlusion-robust). `BottleInstance::grasped` is the seam.

## 4. Part-B chain covariance (all 3 agents)
The published RT covariance now includes the **localization/chain term `J·Σ_chain·Jᵀ`** — the uncertainty
the room-frame object pose inherits from the robot not knowing exactly where it is. Computed with
`DSR::InnerGaussianAPI`: transform the fitted centre room→"zed" then back with ZERO input cov →
`transform_point` returns exactly `J·Σ_chain·Jᵀ` (`Σ_chain` = per-edge `rt_covariance` adjoint-composed
along the tree), pinned to the mask capture stamp. `room_concept` publishes the `robot↔room` localization
cov, so this is live. Config `*.RtCovAddChain` (default true).
- **Scope = covariance only.** bottle additionally consumes camera-frame masks (transforms the MEAN in
  the consumer); table/chair keep ROOM-frame masks (synchronous retina deproject) — the camera-frame
  mean port caused robot-rotation timing jitter and gives furniture nothing, so it was deliberately NOT
  ported there.

## Key facts / gotchas
- Mask `timestamp_ms` (zed) and the robot `rt_covariance` stamp (lidar) are BOTH epoch-ms via
  robot_concept's `to_epoch_ms` → same clock. Residual rotation jitter = `room_concept` RT publishes
  ~60 ms (lidar-paced) + DSR clamps at the leading edge (no extrapolation). See
  [DSR RT must be timestamped] memory.
- `common/` headers (mask_ingestor, instance_tracker, motion_filter, belief_stabilizer, sample_queue)
  are shared; a change there hits all agents — keep new options default-OFF / backward-compatible.
- Diagnostics: each agent's Fisher CSV logs the per-cycle belief; bottle's adds `vx,vy,motion_var` and
  `chain_xx,chain_yy,rtcov_xx,rtcov_yy`.
