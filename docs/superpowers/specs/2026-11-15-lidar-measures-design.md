# Camera attributes, LiDAR measures: the LiDAR updates the countable belief on gated mask cycles

Date: 2026-11-15. Status: APPROVED by the owner 11-15 ("Approved, after lidar": the cabinet exit-release fix comes after this). Nothing built.
Branch: `feat/cost-removal` (worktree `active_inference.worktrees/cost-removal`) — this is the prerequisite the
cost-based-removal spec's G0b found (§2), so it lands on the same branch, before G0b resumes.

## 1. Intent (owner)

The LiDAR returns must update the countable belief's pose and size, not only test existence, so that the belief's
covariance Σ says what all the sensors know and its height follows the LiDAR's top. Owner, 11-15: "camera attributes,
lidar measures" — the camera decides WHICH returns belong to the object; the LiDAR decides WHERE its surfaces are.
Rejected by the owner: a LiDAR-only update with attribution by an explaining-away mixture ("we take the risk of
attributing wrong lidar points to the night stand").

Why now: the existence check of the removal work (spec 2026-11-15-cost-based-removal, G0b) marginalises the box state
over Σ. On objects_live6 Σ is not honest: it ages while the camera may not update (nightstand_6 depth σ 0.67 m, bed yaw
σ 0.70 rad, true errors 6–14 cm) and H keeps the mask's bias (+1.4..+3.2 cm). With that Σ no existence likelihood can
pass the real objects and fail the phantoms (`docs/superpowers/results/2026-11-15-carve-replay.md`, Task 4c report).

## 2. What the code does today (verified 11-15 on feat/cost-removal)

- **The fit has a LiDAR factor already**: `CountableLidarRangeChannel::feed` (`countable_lidar_range_channel.cpp`)
  selects Helios returns inside a box anchored on the fresh MASK-CLOUD CENTROID, sized from the class's BIRTH footprint,
  z-band [−m, birth_H + m]; final membership is the factor's own sphere-trace against the belief SDF, with a robust
  kernel (`LidarRobustCM`), ray-count and angular-coverage down-weights; precision `LidarPrecision` 2500 (≈ 2 cm per ray).
  bpearl is an extra ray set (`LidarBpearlPrecision`, 0 for the three classes).
- **It only runs inside an admissible camera frame.** `CountableFitter` (`countable_fitter.cpp:847`):
  `gated = trunc_frac > AI2TruncGateFrac (0.10) or confirm_only(inst)` (moving and off-centre); a gated cycle calls
  `ai2_belief.predict()` (mean held, Σ grows by Q); `feed()` is reached only in the not-gated branch (line ~978). A cycle
  with no fresh mask ages Σ (`inflate_for_age`, `AI2AgeNominalDtS` 0.1).
- **Measured (objects_live6, nightstand_6):** 852 fit-log rows, 712 gated, so the LiDAR moved geometry on 140; and 557
  rows had no detection.
- **Lifecycle contract** (`CONCEPT_AGENT_LIFECYCLE.md` §2): "a frame must earn the right to move geometry —
  resolvable, centred, still"; otherwise predict-only. Written for camera frames; per-frame quality goes into the
  common-mode covariance Σc, not into gates.
- **The mask is available as pixels**: `MaskIngestor::MaskSlice` carries a raw 2-D foreground pixel range
  (`mask_pixels`, col,row) and the slice's capture timestamp; `CountableProjection::room_T_zed_matrix(ts)` gives the
  camera extrinsic pinned to that stamp and the ZED `CameraAPI` its intrinsics (the silhouette existence channel already
  projects through them).

## 3. Goals

- **G1. On every cycle with a fresh mask assigned to the instance, the LiDAR returns update the belief**, whether or
  not that camera frame is admissible:
  - admissible frame → as today (camera points + LiDAR rays in one update) except that the rays carry the G2 weights;
  - gated frame → an update whose frame carries the LiDAR rays only; the camera points stay out (the camera gate keeps
    its meaning for camera points).
- **G2. Attribution by the mask itself.** Each selected return is projected into the ZED image with the extrinsic at the
  mask's capture timestamp. Its precision is multiplied by its membership weight m ∈ [0, 1]: m = Φ(s / σ_px), s the
  signed pixel distance to the mask boundary (positive inside), σ_px the projection's own pixel uncertainty —
  σ_px² = (f / z)² (σ_range² + σ_chain²) + q², with f the focal length, z the return's camera depth, σ_range the LiDAR
  range σ (`ExistenceSensorSigmaM`), σ_chain the pose-chain σ at the object (`compute_chain_cov`, already per frame) and
  q the mask pixel stride (the ingestor's grid). No new constant. A return that projects outside the image, or behind the
  camera, has m = 0.
- **G3. Correlated rays do not collapse Σ.** The LiDAR-only update goes through the same engine (`rc::ai::update`) with
  the frame's common-mode Σc (pose-chain + ego-motion, as for camera frames), so N rays from one sweep saturate.
- **G4. A cycle with no assigned mask is unchanged** (predict / age): nothing attributes the returns.
- **G5. Opt-in**: `Model.LidarMeasuresOnGated` (bool, default false) and the membership weights behind the same key;
  true in the nightstand, bed and toilet configs. Off ⇒ byte-identical to today.
- **G6. The contract says so**: `CONCEPT_AGENT_LIFECYCLE.md` §2 gains one paragraph — a LiDAR ray set attributed by an
  assigned mask is a frame of its own; its "resolvable" is the existing ray-count/coverage precision, its "centred" is
  attribution by the mask, its "still" is the common-mode Σc. The countable audit row's UPDATE column is updated.

## 4. Non-goals

- No update from LiDAR on a cycle without an assigned mask (owner: camera attributes).
- No change to the camera gate, the camera points, birth, association, or the existence channels.
- No change to the selection box, the sphere-trace, the robust kernel or the precision values.
- Other agents (table, chair, bottle, hood, refrigerator, cabinet) untouched; they may adopt it later.
- The removal work (G0b Task 4c) is not finished here; it resumes on this change's live logs.

## 5. Anti-goals

- The box drifting onto the wall, the bed beside a nightstand, or any neighbour (the risk the owner named).
- Σ shrinking below the true error (overconfidence from N correlated beams).
- A real object's pose or size getting worse than on objects_live6.
- The fit rate dropping below ~10 Hz per class.

## 6. Approaches (owner picked B, 11-15)

- A. Unfreeze the LiDAR on gated cycles with today's selection (mask centroid + birth box + sphere-trace). Small, but
  inside the birth box sit the wall and the bed edge, and on a LiDAR-only cycle no camera point outvotes them.
- **B. A plus attribution by the mask's pixels (picked).** The literal "camera attributes, LiDAR measures"; closes the
  wrong-attribution hole at the cost of one projection per selected return.
- (Rejected by the owner: LiDAR-only updates with an explaining-away mixture.)

## 7. Constraints

CLAUDE.md: no thresholds (membership is a continuous weight from the projection's own σ); and/or/not; `std::from_chars`;
-j8 capped builds; SIGTERM only; Queued signals; type-attributed attributes, no new graph attribute; `cv::Mat` deep-copied
at any thread boundary (the mask raster is built and read on the compute thread). Lifecycle §2 as amended (G6).
Commits local on `feat/cost-removal`; Webots shared (claim from chat and room, FREE after).

## 7b. Decided (my calls after the builder check, owner may change)

- **Anchor on a gated cycle:** the selection box stays anchored on the assigned slice's MASK-CLOUD centroid (the
  deprojected camera points are built before the gate); on a gated frame they are staged for the anchor and the
  sky-free checks but NOT fed to the engine. Never the belief mean (circular — the drift the owner named). The box,
  sphere-trace and robust kernel are otherwise unchanged.
- **Which mask:** the instance's assigned ZED slice of this cycle (its pixel range in `MasksPacket::mask_pixels`). Ricoh
  slices never reach the fit (bearing-only, unchanged). The pixel set the silhouette channel builds
  (`countable_projection.cpp` ~269) is the model; membership needs a signed distance, so a distance transform over the
  slice's bounding region (+ the projection's 3σ_px margin) is built once per instance per cycle on the compute thread.
- **Time:** the sweep is in the room frame at its own capture time; the world is static, so projecting a room-frame
  return with the camera extrinsic at the mask stamp is exact up to localisation error at the two stamps, which σ_chain
  (the pose-chain covariance at the object, already per frame) carries. No separate timing term.
- **q:** the slice's measured pixel spacing (1 px for a dense mask); f and the image size from the same CameraAPI path
  the silhouette channel uses, including its intrinsic-size vs delivered-frame handling.
- **The LiDAR-only frame:** the same factor set as a camera frame MINUS the terms that are camera data (the mask-point
  data term, the moment/second-moment terms, `z_top_obs` and the top-no-float anchor, the per-point azimuth terms). The
  structural priors (wall flush / no-cross, room-axis alignment, footprint/height priors, clearance) stay, as they do in
  every update. All six DOFs are free; what the rays observe is decided by the engine's Jacobians, not by a list.
- **m and the down-weights:** m multiplies each ray's precision BEFORE the ray-count and coverage down-weights, and the
  ray count is Σ m (effective rays), so a mostly-outside set gets little coverage credit.
- **Divergence net:** the existing snapshot-and-reject around the update applies to LiDAR-only updates too.

## 8. Pass

- **Unit:** membership m = 1 well inside, 0.5 on the boundary, → 0 outside, symmetric in s; σ_px grows with 1/z; a
  return behind the camera or outside the image has m = 0; with `LidarMeasuresOnGated = false` a recorded fit trajectory
  is unchanged.
- **Live objects_live8** (same tour as objects_live6, the room session's validated fleet set):
  - per real object (bed, both nightstands, toilet): final centre error ≤ objects_live6's (5.5 cm, 2–6 cm, 23 cm);
    |H − true top| smaller than live6 for bed and nightstands;
  - honest Σ: over each object's run, the logged std_cx, std_cy, std_H stay within 3× the true error's rms and never
    exceed 0.15 m once the object has been seen from within 2.5 m;
  - no drift: no real object's centre moves more than 10 cm toward a wall or a neighbour between consecutive looks;
  - ~10 Hz per class.
- **Then** the removal spec's G0b replay (Task 4c rev 2) runs on objects_live8: real objects ≥ 0, phantoms < 0.

## 9. Left to the builder

- How the mask raster / boundary-distance map is built from the slice's pixels (distance transform on the mask's
  bounding region is the obvious way) and where it is cached for the cycle.
- How the LiDAR-only frame is assembled for the engine (an empty camera point set plus `frame.lidar`) and which
  existing debug columns record it (one new ai2_log column `lidar_only` 0/1 and the summed membership weight).
- Whether the bpearl ray set gets the same weights (it is off for these classes).
