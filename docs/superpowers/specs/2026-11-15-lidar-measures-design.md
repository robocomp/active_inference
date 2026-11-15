# Camera attributes, LiDAR measures: the LiDAR updates the countable belief on gated mask cycles

Date: 2026-11-15. Status: APPROVED by the owner 11-15 ("Approved, after lidar": the cabinet exit-release fix comes after this); §7b and §8
amended 11-15 after Fable's plan review (owner: C1 = A, C2 = two-sided); §7c added 11-16 (owner: "apply active
inference rigorously" — yes to the mixture with the camera prior, yes to LiDAR-first). Tasks 1–2 built; DRAFT §7c awaits review.
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
  σ_px² = (f / z)² (σ_range² + σ_chain² + mot_pos_var) + q² (mot_pos_var added in §7b, 11-15), with f the focal length, z the return's camera depth, σ_range the LiDAR
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
  (the pose-chain covariance at the object, already per frame) carries; the mask's own ego-motion displacement is
  `mot_pos_var` (see "Ego-motion in σ_px" below).
- **q:** the slice's measured pixel spacing (1 px for a dense mask); f and the image size from the same CameraAPI path
  the silhouette channel uses, including its intrinsic-size vs delivered-frame handling.
- **The LiDAR-only frame (amended 11-15 after Fable's plan review, owner ruling C1 = A):** it carries ONLY the
  factors that are functions of its own data — the mask-weighted LiDAR rays — plus the ONE-SIDED constraints that are
  zero unless violated (wall no-cross, short-height). The Gaussian class anchors (footprint, height prior, wall
  flush / parallel, room-axis alignment) are NOT added: the predicted belief already holds them from every earlier
  camera frame, and re-adding them on a frame without camera data counts the prior as a measurement (700 times per
  run on the nightstand: mean relaxed to the class prior, σ_H ≈ 2 cm with no height datum). As Σm → 0 the update
  tends continuously to a predict, with no switch. The camera-data terms (mask points, moments, `z_top_obs` /
  top-no-float, per-point azimuth) are out as before. All six DOFs are free; what the rays observe is decided by the
  engine's Jacobians. Consequences: the depth-unobserved boost (`two_sided_along`, footprint + wall-flush only) does
  not enter, because neither term is in the frame; the volatility estimate ω is NOT updated on a LiDAR-only frame
  (it stays inferred from camera-measured cycles, as today — my call: a weak ray update's δθ ≈ 0 would otherwise
  read as "never moves").
- **Common mode of the LiDAR-only frame (my call, Fable I2):** Σc = the config floors + the pose-chain covariance
  (`chain_cov_xx/yy/yaw`) + the ego-motion terms. The camera deprojection terms (range lateral/size, obliquity yaw
  cap, moment variance) model the camera and stay out; at 2.5 m the size term alone would cap the per-frame H
  information at 25 /m² and the LiDAR could not move H. The admissible frame's fill is unchanged.
- **Ego-motion in σ_px (my call, Fable I3; replaces "no separate timing term"):** the gated frames are the moving
  ones, and the fitter already models the mask's displacement on them as a shared error (`mot_pos_var`, from
  `MotionCmPosGain`). σ_px² = (f / z)² (σ_range² + σ_chain² + mot_pos_var) + q². No new constant.
- **Pixel frames (Fable I6):** mask pixels are in the DELIVERED image on the slice's grid (stride q from the bbox
  corner); the projection is in the intrinsic frame. col, row and f are scaled to the delivered size taken from the
  staged RGB frame, as the contour path does. Delivered size unknown ⇒ no membership ⇒ predict on a gated cycle,
  unweighted rays on an admissible one.
- **m and the down-weights:** m multiplies each ray's precision BEFORE the ray-count and coverage down-weights, and the
  ray count is Σ m (effective rays), so a mostly-outside set gets little coverage credit.
- **Divergence net:** the existing snapshot-and-reject around the update applies to LiDAR-only updates too.

## 7c. The two corrected assumptions (11-16, owner: "can't we apply active inference rigorously?")

Task 2's review measured the owner's named risk in a unit test: a nightstand 5 cm off a wall, wall returns at
m = 0.001, the same sweep 100 times ⇒ the back face pulled from −5.0 cm to −1.3 cm (h 0.51 → 0.55 m); box rays alone
−4.9 cm. Live it is worse: σ_px ≈ 12 px at 2 m, so a wall return 5 cm beside the mask has m ≈ 0.1. The cause is not
a missing term but two false statements in the generative model. This section corrects them; it supersedes G3,
the §7b bullet "m and the down-weights" and the Woodbury-only reading of C2.

### 7c.1 The LiDAR's shared error persists across sweeps (supersedes G3)

**What the model said:** each frame's shared error (pose chain, extrinsic, mask boundary, which surface points the
beams sample) is drawn afresh every frame — that is what the per-frame Woodbury cap assumes. **What is true:** the
world is static; while the viewpoint does not change, sweep t samples the same surface points with the same pose
and the same mask boundary as sweep t−1. 100 identical sweeps are about one observation. Treating them as 100 is
what collapses Σ (C2) and what lets m = 0.001 accumulate into a 3.7 cm pull.

**The corrected model.** The LiDAR channel measures z = θ + b_L + ε: θ the box state, ε the per-ray range noise
(independent per sweep), b_L ∈ ℝ⁶ the channel's SHARED error, a latent with prior N(0, Σc_L) (Σc_L = the LiDAR-only
common mode of §7b: config floors + pose chain + ego-motion) and Gauss–Markov dynamics between sweeps:

  b_L,t = ρ_t · b_L,t−1 + w_t,  w_t ~ N(0, (1 − ρ_t²) Σc_L)   (stationary at Σc_L for any ρ)

b_L is carried in the belief as a real latent (state augmented to 12: θ and b_L, with their cross-covariance).
Each sweep: predict b_L with ρ_t; linearise the rays at the current θ into (Id_L, bd_L) exactly as
`accumulate_lidar_rays` does now; update the augmented state with the measurement Jacobian [I I]. Consequences, all
by construction, none by a rule:
- ρ = 0 reproduces today's per-frame Woodbury exactly (b redrawn every frame and marginalised). That is the
  regression test.
- ρ → 1 (nothing changed): a repeated sweep re-observes the same θ + b_L; the posterior of θ cannot go below what
  one look plus Σc_L allows. 100 identical sweeps ≈ 1. The accumulated pull of a wrongly attributed return is
  bounded by one sweep's worth.
- The robot moves: ρ falls, b_L decorrelates, a new viewpoint counts as new information. Moving around an object is
  what earns confidence — which is the active-inference reading of "look from elsewhere".

**Where ρ comes from (physical, no tuned constant).** The shared error decorrelates when the beams sample different
surface points and the mask boundary falls on different pixels of the object, i.e. when the camera/LiDAR viewpoint
relative to the object changes by about one beam footprint. With δ = max(`azimuth_step_deg`, the largest ring gap)
from the Helios `SensorModel` (media descriptor), z the range to the object and Δ the viewpoint change since the last
sweep (sensor displacement perpendicular to the line of sight, plus z·|Δbearing|):

  ρ_t = exp(−½ (Δ / (δ·z))²)

The pose-chain part of b_L decorrelates with the localiser's own corrections, not with the beam footprint; the
localiser does not publish a cross-time covariance, so this spec treats it as decorrelating with the viewpoint too
(conservative while parked, possibly optimistic while turning in place). OPEN for review: whether a better source
exists.

**Scope:** the LiDAR channel only (owner, 11-16). Camera points keep the per-frame Woodbury (ρ_cam = 0); the same
correction for the camera path is later work. On an admissible frame the rays therefore leave the camera's
saturated (Id, bd) and enter through the b_L channel after it; on a LiDAR-only frame only the b_L channel runs.

### 7c.2 The camera's m is a prior on ownership, not a precision multiplier (supersedes "m and the down-weights")

**What the model said:** a return's precision is scaled by m — a tempered likelihood, an approximation. **What is
true:** the camera says p(z_i = object) = m_i; the return then comes either from the box or from the rest of the
scene, and the scene is not unknown — the room model knows its walls and floor. Owner 11-16: the mixture is
accepted with the camera's m as its prior (this is not the camera-free explaining-away mixture rejected in §1).

  p(ρ_i) = m_i · p(ρ_i | box) + (1 − m_i) · p(ρ_i | rest)
  p(ρ | rest) = (1 − w_u) · N(ρ; t_room, σ_r² + σ_wall²) + w_u · U(0, ρ_max)   (the room predicts a hit at t_room)
             = w_u · U(0, ρ_max)                                       (the room predicts none)

t_room = the ray's first hit on the room's walls (the polygon edges, extruded) and floor (z = 0); σ_wall = the wall's
`sigma_m`; σ_r and ρ_max from the Helios `SensorModel`; w_u = 0.2 and the form are the fleet's existing beam model
(`lidar_ray_cost`, the door's and rc::exist's) — reused, not new. p(ρ | box) is today's first-hit Gaussian with the
Cauchy kernel unchanged. The E-step responsibility

  r_i = m_i p(ρ_i | box) / (m_i p(ρ_i | box) + (1 − m_i) p(ρ_i | rest))

replaces m_i as the ray weight in `LidarRays::weights` (Task 1's mechanism, unchanged), recomputed at each GN
iterate; coverage counts Σ r_i. Numbers (σ 2 cm, ρ_max 10 m): a return on the box face with the wall 0.5 m behind
it, m = 0.9 ⇒ r ≈ 1; a wall return beside the box, 5 cm behind the back face, m = 0.1 ⇒ r ≈ 0.009 (the wall explains
it); the same wall return at m = 0.5 ⇒ r ≈ 0.07. Known limit: a NEIGHBOUR (bed beside the nightstand) is not in the room
model, so it is explained only by the uniform floor; there the mask prior and 7c.1 carry the case. Using other
agents' published boxes as further "rest" components is later work.

### 7c.3 Pass additions
- Unit: ρ = 0 reproduces the per-frame Woodbury result (rel 1e-5); ρ = 1 and 100 identical sweeps give Σ within 5 %
  of one sweep's posterior; a viewpoint step of 3 δz gives the full second-sweep information (within 5 %).
- Unit: `wall_behind_does_not_pull` asserts the BACK-FACE gap against a box-rays-only baseline: the weighted case
  changes it by < 0.5 cm over 100 identical sweeps with wall returns at m = 0.1 (the live value, not 0.001).
- Unit: responsibilities — the three worked numbers above within 20 %.
- Live: unchanged (§8, two-sided Σ and ≤ 10 cm drift).

## 8. Pass

- **Unit:** a LiDAR-only frame whose memberships all vanish leaves the belief at the prediction (state, Σ and ω);
  repeated LiDAR-only sweeps settle Σ at a pinned, analytically checked value; a wall behind the object does not
  pull it when the mask covers only the object; membership m = 1 well inside, 0.5 on the boundary, → 0 outside, symmetric in s; σ_px grows with 1/z; a
  return behind the camera or outside the image has m = 0; with `LidarMeasuresOnGated = false` a recorded fit trajectory
  is unchanged.
- **Live objects_live8** (same tour as objects_live6, the room session's validated fleet set):
  - per real object (bed, both nightstands, toilet): final centre error ≤ objects_live6's (5.5 cm, 2–6 cm, 23 cm);
    |H − true top| smaller than live6 for bed and nightstands;
  - honest Σ, TWO-SIDED (owner ruling C2, 11-15): over each object's run, the logged std_cx, std_cy, std_H stay
    within a factor 3 of the true error's rms in BOTH directions (rms/3 ≤ std ≤ 3·rms) and never exceed 0.15 m once
    the object has been seen from within 2.5 m. A std below rms/3 (overconfidence from correlation ACROSS sweeps,
    which the per-frame Woodbury cap does not cover) is a FAIL reported to the owner as a model question, not tuned;
  - no drift: no real object's centre moves more than 10 cm toward a wall or a neighbour between consecutive looks;
  - ~10 Hz per class.
- **Then** the removal spec's G0b replay (Task 4c rev 2) runs on objects_live8: real objects ≥ 0, phantoms < 0.

## 9. Left to the builder

- How the mask raster / boundary-distance map is built from the slice's pixels (distance transform on the mask's
  bounding region is the obvious way) and where it is cached for the cycle.
- How the LiDAR-only frame is assembled for the engine (an empty camera point set plus `frame.lidar`) and which
  existing debug columns record it (one new ai2_log column `lidar_only` 0/1 and the summed membership weight).
- Whether the bpearl ray set gets the same weights (it is off for these classes).
