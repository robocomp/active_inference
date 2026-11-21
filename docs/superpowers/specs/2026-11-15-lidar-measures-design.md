# Camera attributes, LiDAR measures: the LiDAR updates the countable belief on gated mask cycles

Date: 2026-11-15. Status: APPROVED by the owner 11-15 ("Approved, after lidar": the cabinet exit-release fix comes after this); §7b and §8
amended 11-15 after Fable's plan review (owner: C1 = A, C2 = two-sided); §7c added 11-16 (owner: "apply active
inference rigorously" — yes to the mixture with the camera prior, yes to LiDAR-first); §7c rev 3 11-16 (owner: class
prior counted once + Σ bounded by it (B); camera persistence now with the object-size scale). Tasks 1–2 built and
merged to main 8f4cc821; §7c rev 3.1 folds Fable's rev 3 review (READY WITH CHANGES). §7c.7 rev 3.2 11-18 (owner: "Go"):
persistent errors never rescale; b_chain becomes the robot pose error with a fixed lever in B. Rev 3.2.1 folds Fable's
rev 3.2 review (NOT READY; all 3 Critical, 7 Important, 7 Minor written in as R4-*; C2 = keep F, no parallax claim).
§7c.8 rev 3.3 11-18 (owner: "Go" on the smooth-ray follow-up): a miss is evidence — the persistent ray likelihood is a
smooth beam model. Rev 3.3.1 11-18 folds Fable's rev 3.3 review (NOT READY; 4 C, 8 I, 9 M); owner C3 = A (returns
staged by ray geometry, mask ownership as a mixture; reverses 7c.3's "first-hit physics stays out").
Rev 3.3.2 folds the rev 3.3.1 re-review (READY WITH CHANGES; 2 C, 5 I, 9 M). Rev 3.3.3 folds the rev 3.3.2 re-review
(READY WITH CHANGES; 1 C, 3 I, 6 M) and the owner's rulings of 11-18; APPROVED by the owner 11-20 ("Yes"). Rev 3.3.6 (11-20, owner: "Spec fix, Fable checks"): the quadrature amendment at the end of §7c.8 (rev 3.3.4 NOT READY, R8 folded in 3.3.5; 3.3.5 READY WITH CHANGES, R9 folded), APPROVED by the owner 11-20 ("I approve").
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

## 7c. Corrected assumptions (rev 3, 11-16; owner: "can't we apply active inference rigorously?")

Owner rulings 11-16: the class prior is counted ONCE (at birth) and the belief's Σ is bounded by it, on every countable
frame, behind `Model.LidarMeasuresOnGated` (rev 3 decision B, replacing rev 2's OU transition after Fable's rev 2
review C4); the camera gets the same persistent shared error as the LiDAR now (rev 1 C4 = A), its decorrelation scale
taken from the object's own angular size (no new constant); the camera's m is the ownership prior of a mixture (not
the camera-free explaining-away mixture of §1). Reviews: rev 1 `fable-7c-review.md`, rev 2 `fable-7c-rev2-review.md`
(cited R1-…, R2-…), both in `.superpowers/sdd/2026-11-15-lidar-measures/`. This section supersedes G3, the §7b bullet
"m and the down-weights", and §7c revs 1–2.

### 7c.0 What was measured (corrected, R2-C1)

- **The real fault: Σ grows without bound along directions no LiDAR ray sees.** On a LiDAR-only frame (C1 = A, no
  anchors) seen from the front, the depth/back-face direction g = (Δcy, Δh) = (δ/2, δ) has no data, and Q is added on
  every predict: σ_h 6.0 → 20.9 cm over 2000 sweeps (Task 2 fixture, R2-C1). Any existence or removal score that
  marginalises Σ then reads an honest-looking but unfounded uncertainty.
- **The back-face pull reported in rev 1/rev 2 was a fixture artefact.** The Task 2 `wall_behind` fixture put 120 wall
  returns on a grid with no visibility test; 20 of them lie on rays the TRUE box occludes. With those 20 removed the
  pull is exactly zero at weights 0.001, 0.1 and 1.0 over 2000 sweeps, on today's code (R2-C1 table). Reason: a ray that
  does not cross the belief box has no term (`lidar_ray_factor.h:190`).
- **Consequence for attribution.** With physical returns, a foreign return carries a term only when the belief differs
  from the truth: (i) the belief over-extends (deeper or taller than the object), so real wall returns cross it — those
  are CORRECT refutations; (ii) a neighbour's face lies within σ of the belief's face; (iii) the mask is
  over-segmented onto the wall. 7c.3 is specified and tested on those three cases only.
- **Every fixture builds its returns by ray-casting a TRUE scene** (box ∪ walls ∪ floor ∪ neighbour; first hit only).
  The Task 2 tests `wall_behind_does_not_pull` and the dead vanishing control are rewritten under this rule in the
  same change.

### 7c.1 The class prior is counted once (decision B; supersedes rev 2's OU transition)

**What the model said.** The class anchors — footprint (2), height (2a), room-axis alignment, wall flush/parallel (1)
— are added to Id on every camera frame while `predict` adds Q: a filter told that a fresh measurement of the class
mean arrives every frame. Its fixed point along an unseen direction is 1–3 cm (R2-M1: 0.97 cm at the floor, 2.8 cm at
2 m), several times tighter than the class's own spread, from no data. On a LiDAR-only frame the same direction has
no prior at all (7c.0).

**What is true.** The class prior p₀(θ) is what is known about an object of this class before any measurement. A
recursive Bayesian filter multiplies it in once. The relational parts (stands against a wall with π_flush, aligned
with the room axis) are prior knowledge too, and enter on the same footing.

**The model, under the flag.**
- **Birth.** The belief is born as today (`birth_on_priors`: data-seeded mean, Σ = `prior_cov_diag` = the class P,
  R2-M3). That Σ₀ IS the class's size/pose prior, counted once (stated limit: it is centred on the data-seeded mean,
  not on the class mean). The FIRST ADMISSIBLE CAMERA update of the instance (not a LiDAR-only or gated one; a
  `prior_added` flag PER relational anchor: wall flush/parallel and room axis each enter once, at the first admissible camera update on which their context (wall, room axis) is staged — ruling T6-1) adds the RELATIONAL anchors once — wall flush/parallel (1) and the room axis —
  in a θ-only, unsaturated channel; the footprint (2) and height (2a) anchors are not added (Σ₀ already carries the
  class, R3-I2). The flush responsibility is evaluated at that pose only (stated: an object born mid-room that later
  settles on a wall never gets the term). `sanitize_covariance` re-seeds Σ to P and clears `prior_added` (the prior
  re-enters once); the divergence restore copies it; an LTOM/LTSM recall restores a belief with the flag set.
- **Amendment 11-17 (live8 analysis fault 2; owner "Go"): the class prior is a factor AT THE CLASS MEAN, counted once.**
  Centring Σ₀ on the data-seeded birth state (the stated limit above) turned a bad first mask into the prior's mean:
  live8's nightstand_1 was born with h 2.14 from a mask over the bed side and never came back (live6's per-frame
  anchors pulled the same birth back in 4 frames). Under the flag, the belief's FIRST update has prior mean x̄_θ =
  (seed cx, seed cy, class H, class w, class depth, nearest room-axis yaw if staged else seed yaw) with Σ₀ = the class
  P; the data seed is only the GN starting point θ₀ of that update (x̃₀ = (seed, b̄)). Position has no class mean
  (prior centred on the seed, σ = prior_pos_std). From the second update on, the recursion is unchanged. Nothing is
  added per frame; the relational anchors keep their per-anchor once-flags.
- **Every later frame** (camera, LiDAR-only, admissible or gated): no anchor. The one-sided constraints (2b)
  short-height and (2c) wall no-cross stay on every frame — they are likelihood terms (zero unless violated), not
  priors. Camera data terms (z_top_obs / top-no-float, moments, per-point azimuth) stay.
- **Predict.** Today's: the mean is held (static object), Σ ← Σ + Q (Q = exp(ω) learnt, R2-I4 unchanged: ω learns
  from the post-minus-previous-posterior step of camera-measured cycles), followed by the engine's existing
  `clamp_to_prior` congruence with the class P: Σ never says LESS is known than before the object was seen. This is
  the one bound in the design; it is the engine's documented stationary-prior saturation (recursive_laplace.h
  predict, already used by table and chair), it acts on Σ only and never on data or the mean. The aged predict
  (`inflate_for_age`) uses the same bound.
- **`depth_unobs_precision` stays for now (R2-I5, R3-I1) — the one class-mean term still counted per frame.** All
  its shares (w-axis, depth, and the flush share in `accumulate_wall`) enter in the CAMERA channel, so a parked camera
  counts them once through b_cam. It counters a likelihood bias (one-face points drag the extent outward) that the
  prior counted once does not hold. Consequence, stated: on one-face camera frames depth sits at the class depth to
  1–3 cm (σ_h 0.97 cm near, 2.8 cm at 2 m), rises toward 6 cm on LiDAR-only frames and collapses at the next one-face
  camera frame; the two-sided Σ check on depth is expected to fail as on live6, and the 1σ-shifted live run will show
  depth following the shifted mean. It is removed only when a camera one-face replay on recorded live6 frames
  (parked and approaching) passes without it — a separate later change.

**Consequences.**
- A measured size is kept (except depth on one-face camera frames, above): nothing pulls the mean back to the class after birth (R2-C4: rev 2 forgot a measured
  depth in 30–80 s; gone). A parked look does not leak into the bias (R2-I6: that came from the reversion; gone).
- Σ along an unseen direction rises to the class P and stays there (nightstand depth ±6 cm) wherever the boost is not
  applied: the honest statement, and what the existence/removal marginals need.
- Camera frames change with the flag on (anchors leave after birth). The flag-off live run (objects_live8off) is the
  A/B; the golden 40-frame trajectory pins flag-off bit-exact.
- No mean reversion means no bound on a mean walk under a CONSTANT false force; 7c.0 shows no such force exists with
  physical returns. The mean is protected by 7c.2 (a parked look is one look) and 7c.3 (ownership).
- The room axis no longer holds yaw after birth: yaw is held by its data. The flag-off A/B notes yaw at |Δ| > 22.5°
  (R2-I3: today's term has a stable minimum at 45°) so a difference there is not read as a regression.

**The size summary (R2-I10, R3-C2): unchanged in this change.** `size_lik_` already excludes the anchors and the
boost (`size_prior_off_`) and forgets by +Q as decision B's predict does. Two known gaps, later work: it counts a
parked camera once per frame (the belief counts it once), and LiDAR-only frames do not reach it. Deriving it from the
belief by Gaussian division was rejected (R3-C2: wrong order, wrong in time under Q, mean scored against itself, the
boost read as data).

### 7c.2 Shared errors persist across frames (supersedes G3; R1-C1, C4, I1–I5, I10; R2-C2, C3, I8)

**What the model said.** Each frame's shared error is drawn afresh (the per-frame Woodbury `saturate`). **What is
true.** The world is static; while the viewpoint does not change, frame t sees the same surface, through the same
localisation error, with the same deprojection and mask bias as frame t−1. 100 parked frames are about one look.

**Latents (R2-I8).** Three, carried in the belief beside θ (covariance P 20×20 in double):
- b_chain ∈ ℝ² on (cx, cy): the localisation (pose-chain) error at the object. SHARED by both channels — it is the
  same robot pose. Marginal = `inst.chain_cov_xx`, `inst.chain_cov_yy` (the localisation part only; `frame.chain_cov_*`
  are lumps that also hold range and ego-motion, R3-I3). No localisation yaw variance is published, so yaw has no
  chain latent (stated). A zero marginal (chain cov disabled, transform missing) is a zero-variance latent: handled
  by the inverse-free update below.
- b_cam ∈ ℝ⁶: the camera points' sensor-specific shared error. Marginal = today's camera common mode WITHOUT the chain
  and WITHOUT the ego-motion terms: config floors + range lateral/size + obliquity yaw (the moment variance is inert in
  this model — no footprint-moment factor — and carries motion terms, so it is left out; Task 9 review).
- b_L ∈ ℝ⁶: the LiDAR rays' sensor-specific shared error. Marginal = the config floors (§7b, without chain and
  ego-motion).
- The ego-motion terms (`mot_pos_var`, `mot_size_var`, `mot_yaw_var`) are fresh every frame by construction: they
  stay in the per-frame Woodbury of their channel (ρ = 0), outside every b.
- Camera points measure θ + E b_chain + b_cam; rays measure θ + E b_chain + b_L (E embeds (cx, cy) into ℝ⁶).
  b_chain, b_cam, b_L are a priori independent of each other and of θ (state x ∈ ℝ²⁰).
- The config floors are in b_cam AND b_L as independent errors (R3-I4, my call): whatever part of a floor is model
  mismatch common to both sensors is then counted twice when parked (position information 2/0.03² instead of
  1/0.03², σ low by √2 at the floor). Stated; the two-sided live check sees it.

**[SUPERSEDED by 7c.7 (rev 3.2): no D rescale; b_chain is the robot pose error e ∈ ℝ³ with the lever in B.]**
**Dynamics.** Per latent and per DOF: b_t = ρ_t D_t b_{t−1} + w_t, D_t = σ_t / σ_{t−1} (one source per latent, so the
rescale is physical: a deprojection bias scales with its σ), w_t ~ N(0, (1 − ρ_t²) σ_t²). Marginal exactly σ_t²,
lag-1 correlation exactly ρ_t; when σ_{t−1} = 0 the source is new: ρ = 0 for that transition (variance σ_t², ruling T4-1c). ρ = 0 for all three is today's Woodbury: the regression test. The θ–b and b–b cross
blocks are propagated with the same F = blockdiag(I, ρD) (the b_cam–b_L block included, R2-C3).

**ρ (R2-C2; decision: camera scale from the object's angular size).** Viewpoint change is measured from ODOMETRY, not
from the estimated pose (pose jitter is not motion): the robot node's typed `robot_current_advance_speed`,
`robot_current_side_speed`, `robot_current_angular_speed` (+ `robot_current_speed_timestamp`), integrated between the
channel's updates of THIS instance (aged and predict-only cycles add up). Parked ⇒ Δ = 0 ⇒ ρ = 1 exactly.
- From the integrated sensor motion relative to the object centre: Δφ = change of the sensor's bearing ABOUT the
  object (aspect), Δz = change of its range.
- **Camera:** ρ_cam = exp(−|Δφ| / φ_obj − |Δz| / z), φ_obj = atan(w_⊥ / 2z), w_⊥ the belief's extent perpendicular to
  the line of sight: a new face starts to show after the camera has turned about the object by its angular half-width,
  and the range-proportional deprojection terms decorrelate with a relative range change. From the state; no constant.
  Tangential e-fold z·φ_obj ≈ w_⊥/2 = 25 cm for the nightstand at any range; the LiDAR's is 3.9 / 7.8 / 11.8 cm at
  1 / 2 / 3 m, so the camera decorrelates 2–6× more slowly (R3-M6).
- A stale speed attribute gives Δ = 0 and ρ = 1 (fails closed); a rotation in place gives Δφ ≈ 0 about the object and
  ρ = 1 although the object crosses the image (conservative) (R3-M5).
- **LiDAR:** lateral DOFs and yaw: ρ = exp(−z|Δφ| / (g·z)) = exp(−|Δφ| / g), g = `max_ring_gap_rad()` of the Helios
  `SensorModel` (the beam pattern shifts by a footprint); H: ρ = exp(−|Δz| |tan ε| / (g z)) (ring ε moves over the top
  with range). SensorModel absent ⇒ ρ_L = 1, logged per instance (R2-M6).
- **Chain:** ρ_chain = exp(−½ cᵀ Σ_chain⁻¹ c), c = (pose increment from the localiser) ⊖ (odometry increment) over the
  same stamps = the localiser's own correction. Parked without correction ⇒ 1; a relocalisation ⇒ ≈ 0. Scale = the
  chain covariance itself. Stated limit (R3-I5): under white pose jitter j it still decays, 1/e after 2σ²/j² frames
  (89 frames at 3 mm, σ_chain 2 cm), and N corrections do not equal one summed correction. ρ_chain and c are logged
  while parked; the named fallback, if live8 shows the jitter case, is ρ_chain = 1 except at a relocalisation event.
- The exponential (OU-in-path) form composes for any update rate (R1-I1). Stated limit: a path that returns to the
  same viewpoint counts as new.

**Update (R2-C3, R3-C1), in order, one predict and one GN per cycle:**
1. Predict x̄, P̄ (θ by 7c.1's predict on the θθ block, the clamp applied as a congruence on θ's rows/columns of P; b by
   the dynamics above).
2. Per GN iterate k, the biases RESTART at their prediction: x̃_k = (θ_k, b̄). Trace/linearise the camera points at
   θ_k + E b̄_chain + b̄_cam and the rays at θ_k + E b̄_chain + b̄_L, exactly as `accumulate_*` do now, giving (Id_ch,
   bd_ch). Marginalise each channel's per-frame FRESH covariance Σf (ego-motion) in double, inverse-free:
   I_ch = (𝟙 + Id Σf)⁻¹ Id, g_ch = (𝟙 + Id Σf)⁻¹ bd — the identity at Σf = 0 (parked), unlike `saturate`'s inverse.
3. Joint step, inverse-free (the form `predict_size_likelihood` already uses), 20×20 in double:
   W = Σ_ch H_chᵀ I_ch H_ch + H_θᵀ I_θ H_θ,  G = Σ_ch H_chᵀ g_ch + H_θᵀ g_θ,
   x_{k+1} = x̃_k + (𝟙 + P̄W)⁻¹ [P̄ G − (x̃_k − x̄)],  keep θ_{k+1}; the b of the LAST iterate is the posterior b.
   H_cam = [I E I 0], H_L = [I E 0 I], H_θ = [I 0 0 0]. The θ-only channel (I_θ, g_θ) carries the one-sided
   constraints (2b), (2c) and the birth relational anchors, unsaturated: the biases cannot absorb a wall crossing.
   No P̄⁻¹, so zero-variance latents are valid.
4. P = (𝟙 + P̄W)⁻¹ P̄ at the last iterate, symmetrised. Published Σ = the θθ block.
Eliminating b from step 3 at ρ = 0 gives exactly today's `recursive_laplace.h` line 230 at every iterate (R3-C1,
checked: 1-D case 0.0878 at every iterate). The regression holds on SINGLE-CHANNEL frames (camera-only, LiDAR-only);
a flag-on admissible frame with both channels differs from today by design (rays get b_L, points b_cam; today one
Σc covers both) (R3-I4). Tolerance measured in float32 (float input, double solve).
The divergence snapshot copies x and P. `sanitize_covariance` re-seeds the b blocks at their marginals with zero
cross terms; `canonicalize` / `apply_mode_rotation` swap the w, h components of b_cam and b_L and their blocks;
b_chain is an additive room-frame error and is untouched (R3-M1). LTSM restore stores no b: the first look after a
restart is new (stated). Lives in a new `common/ai_belief/persistent_update.h`, generic in the latent layout;
`ai::update` and the 8 other agents unchanged. The clamp is the hard `if (Σ_ii > P_i)` congruence of the engine
(R3-M2); the short-height term (2b) enters unsaturated while violated, 2500 /m² per frame at a 10 cm deficit — zero
when not violated, harmless on these classes (R3-M4).

**Consequences.** Parked: each channel counts as one look, and the chain error once for both. H at 2 m parked: the
LiDAR's share is set by the two channels' sensor-specific σ (R2-C2: ~95 % with ρ_cam = ρ_L = 1), so H follows the LiDAR
top. Moving around an object earns information — the active-inference reading of "look from elsewhere". Σ along a
direction one channel alone sees cannot go below P σ²/(P + σ²) of that channel's persistent errors; the MEAN of a
direction no datum holds is held only by the absence of force (7c.0), not by 7c.2 (R1-C1).

**Log per cycle:** ρ_cam, ρ_L, ρ_chain, Δφ, Δz, b̂ and √diag of each latent, Σm, Σr, ray count, trace of each
channel's I_ch, and the measured Δ while parked (so pose jitter vs odometry becomes a number, R2-C2).

### 7c.3 The camera's m is a prior on ownership (supersedes "m and the down-weights"; R1-C2, C3, I6, I8, I9, I11; R2-I7)

Per selected return i (range r_i; ρ is reserved for correlation):

  p(r_i) = m_i · N(r_i; t_i, σ_r² + J_i S_L J_iᵀ) + (1 − m_i) · p(r_i | rest)
  p(r_i | rest) = (1 − w_u) · N(r_i; t_rest,i, σ_r² + σ_rest,i²) + w_u / L_i   (some rest surface predicted inside the box)
               = 1 / L_i                                                      (none predicted)

- **Box component (R1-C3):** the PDA innovation variance; J_i the range Jacobian the factor already computes; S_L the
  predicted covariance of θ + E b_chain + b_L. No Cauchy factor on the mixture path; every other caller of
  `accumulate_lidar_rays` unchanged.
- **Rest (R1-C2, and the neighbour in scope — my call under "go with your recommendations", owner may cut it):** t_rest,i
  = the first hit, inside the selection box, of the ray on (a) the room polygon's vertical faces and z = 0 (σ_rest =
  `wall_explain_sigma_m`, shared with the camera-point wall component — 7c.3 is its LiDAR counterpart), and (b) every
  OTHER object box published in the graph (countable, cabinet, table, chair, refrigerator, … `box`-type nodes with
  their pose and size attributes, read through the graph as agents must; σ_rest = that box's published σ along the ray,
  or the common floor 3 cm when it publishes none). The neighbour R2-I7 named (bed beside the nightstand, r > m within
  8–10 cm) is then explained by its own box, not by the uniform floor. An object no agent publishes falls back to the
  uniform 1/L_i (stated limit). The ray-cast is a small `common/` function on the compute thread. Details (R3-I7):
  the instance's own node is excluded, the same agent's OTHER instances included; size from `width_m_att`,
  `depth_m_att` and the height attribute the node carries, pose through its RT edge at a real timestamp (optional
  checked); a box node missing any of them is skipped; σ floor = `common_mode_pos_std`; boxes are static, so the set
  is staged at ≤ 1 Hz (node copies and RT walks cost rate). A wrong or duplicate foreign box on top of this object
  costs at most a factor m: a true return at e = 0 under a coincident foreign box keeps r = 0.90 at m = 0.9, 0.49 at
  m = 0.5.
- **Unexplained:** uniform over the ray's chord through the selection box, 1/L_i. Limit: a surface within σ of the
  predicted face cannot be told apart by range (any mixture).
- **m = π_mask · Φ(s_i / σ_px)**, π_mask = the slice's published detector `confidence` (an upper bound on pixel
  ownership, R2-M4). σ_px has no segmentation-boundary term yet (stated).
- **First-hit physics stays out (R1-I9, R2-M8):** a return beyond the box's predicted hit gets r ≈ 0 and no term. With
  physical data those are the only wall returns that carry a term at all, and they are correct refutations of an
  over-extended belief; giving them r ≈ 0 is a choice, so an over-deep box is corrected by camera frames only. The
  full model (range = first hit of box ∪ rest) is later work.
- **σ_r² = 1/`LidarPrecision` + σ_sensor²** (σ_sensor from the Helios `SensorModel`, 0 when absent; R2-M5). w_u = 0.2
  (the fleet beam model's form). L_i from the selection box.
- **Interface (R1-I6, R2-M7):** `LidarRays::weights` keeps the prior m. New per-ray θ-independent inputs: the rest
  hits (t_rest,i, σ_rest,i) or none, L_i. The factor computes r_i per GN iterate from its own trace. The coverage
  scale uses Σ r at the predicted state, computed in `CountableBelief::update` before the solve and frozen for it.
  Empty inputs ⇒ today's behaviour bit-exact for table/chair/bottle/door. The ray-count down-weight counts Σ r.
- **Information:** complete-data r·JJᵀ/σ², slightly optimistic where r is ambiguous (R1-M2), as the point mixture.
- **The responsibility table** the tests pin, computed from THIS formula by `countable_concept/tools/responsibility_table.py`
  (σ_r² = 1/2500 + 0.02², σ_rest 0.05, L 0.6 m, w_u 0.2; rev 2's numbers mixed three formulas and are withdrawn). Assumptions
  the script states: "room hit in chord" puts the rest surface 30 cm behind the return; "wall" puts the rest surface ON the
  return; the belief face is e in front of the return; σ_box is along the ray. Read row E with care: a mask that claims wall
  pixels at π = 0.9, s = +3σ_px keeps m ≈ 0.90, and then the range alone (a wall 2 cm behind the face) does NOT refute it
  (r 0.94 at σ_box 0) — ownership of an over-segmented mask is the camera's job, not the range's.

```
Constants: sigma_r^2 = 1/2500 + 0.02^2 = 0.00080 m^2 (sigma_r = 2.8 cm), sigma_rest = 5 cm, L = 0.6 m, w_u = 0.2

A. TRUE return on the object (m = 0.9); belief face offset by e, box sigma along the ray 5 cm
| rest prediction                                | e = 10 cm | e = 14 cm | e = 23 cm |
|---|---|---|---|
| room hit in chord (wall 30 cm behind the return) |     0.976 |     0.906 |     0.058 |
| none predicted                                 |     0.892 |     0.658 |     0.012 |

B/C. FOREIGN returns; the belief face is e in front of the return; s_box in cm
| case (m = 0.1 unless stated)                   | s_box 0 | s_box 2 | s_box 4 | s_box 6 |
|---|---|---|---|---|
| wall 2 cm behind the face (rest = the wall)    |   0.172 |   0.155 |   0.124 |   0.098 |
| wall 5 cm behind the face (rest = the wall)    |   0.053 |   0.071 |   0.084 |   0.079 |
| neighbour 2 cm off, UNPUBLISHED (rest = none)  |   0.423 |   0.394 |   0.333 |   0.277 |
| neighbour 5 cm off, UNPUBLISHED (rest = none)  |   0.165 |   0.213 |   0.244 |   0.232 |

D. COINCIDENT foreign box (return at e = 0 on this object and on the foreign box; s_box = s_rest = 5 cm)
| m = 0.9 | m = 0.5 |
|---|---|
|   0.914 |   0.541 |

E. OVER-SEGMENTED wall return: pi = 0.9 at s = +3 sigma_px gives m = 0.8988; the return is on the wall (rest = the wall)
| belief face e in front of the wall                          | s_box 0 | s_box 2 | s_box 4 | s_box 6 |
|---|---|---|---|---|
| e = 2 cm                                       |   0.943 |   0.936 |   0.919 |   0.897 |
| e = 5 cm                                       |   0.817 |   0.860 |   0.879 |   0.872 |
```

### 7c.4 What stays from §7b

The LiDAR-only frame's content (C1 = A), the selection anchor on the mask-cloud centroid, membership G2 (σ_px with
mot_pos_var; the mask lag is in σ_px only, not in any b), the divergence net, ω from camera cycles only.

### 7c.5 Pass additions (replace revs 1–2; R2-I9, R3-I6, I8). Every fixture ray-casts a true scene, first hit only.
- 7c.1: born, first admissible camera update adds the relational anchors once (`prior_added`), then 5000 predict-only
  cycles: mean unchanged, Σ = the class P (2 %); a LiDAR-only or gated first frame does not add them; flag off: golden
  trajectory bit-exact.
- 7c.1: Task 2's fixture with physical returns, 2000 LiDAR-only sweeps: σ_h ≤ prior_footprint_std (today 20.9 cm),
  mean back face unchanged (1 mm).
- 7c.2: ρ = 0 reproduces today's update at every GN iterate on camera-only and on LiDAR-only frames (measured float
  tolerance); ρ = 1, Q = 0, zero fresh covariance, a LINEAR fixture: n identical sweeps = ONE sweep with n·Id; N
  predict-only cycles (θ with q_scale = N) then one update = one predict over the summed path then one update, exact
  for b_cam and b_L (the chain excluded, R3-I5); the shared chain: parked camera + LiDAR give data information on
  position = 1/(Σ_chain + (1/Σ_cam + 1/Σ_L)⁻¹) (pinned equality); zero-variance chain latent runs; sanitize /
  mode-rotation handle b correctly; ρ from odometry = 1 when parked.
- 7c.2 alone (weights = m, no mixture, a foreign surface within σ, belief ≠ truth): the analytic pull β·G/(G + 1/P)
  (R1-C1) — a number.
- 7c.3: the responsibility table (10 %); `recovers_from_14cm`: parked, the residual is the analytic Σb/(P + Σb)·14 cm
  (3.7 cm at belief σ 5 cm with the 3 cm floor), pinned; moving around the object (ρ < 1), the error falls below 3 cm;
  `over_deep_belief_is_not_held` (belief 10 cm deeper than the truth against a wall: r ≈ 0, depth held by Σ only —
  documents the choice); `neighbour_box_explains_its_returns` (a published neighbour box 3 cm beside the face, m = 0.1,
  2000 sweeps: pull within the analytic bound from the table); `coincident_foreign_box` (r = 0.90 / 0.49 at m = 0.9 /
  0.5); `unpublished_neighbour` (same without the box: the stated limit, pinned).
- Live: §8 unchanged (two-sided Σ, ≤ 10 cm drift, H closer to the true top than live6, ~10 Hz); a depth failure of the
  two-sided check on one-face views is the expected boost outcome (7c.1), reported, not tuned. One nightstand run with
  its class means shifted by 1σ is added to live8, so a belief that merely sits at the class mean cannot pass.

### 7c.6 First-crossing rays and fixed-precision one-sided constraints (11-16, owner: yes; creep investigation)

Measured (`.superpowers/sdd/2026-11-15-lidar-measures/creep-investigation.md`): the shared ray factor's sphere-trace
counts a ray passing within `surf_eps` (2 mm) of a box edge as a HIT without crossing the surface; the Jacobian there
explodes or mixes face normals and the residual is large, and such rays switch on and off with sub-mm changes of
unobserved DOFs (cx, w). On a 16-case set (cx offsets 0–2 mm, with/without wall returns, 4000 sweeps) the persistent
path fails 10/16 (w to 1.27 m, back face 13 cm through the wall); the flag-off path fails on the same scene. The
one-sided terms (2c), (2b) have λ ∝ violation with half their energy's Hessian, which overshoots when unsaturated.
On the persistent path only (flag-off bit-identical; the flag-off defect is a later, owner-visible decision):
- **(A) A ray hits the box only where it crosses it:** the first sign change of the box SDF along the ray, refined to
  the surface; a near miss is a miss (no term). The range variance gains the surface-incidence term σ_s²/(n̂·u)²,
  σ_s = `model_sigma_m`. Behind a `LidarRays` field (default: today's loop) so every other caller is unchanged.
- **(B) (2c) and (2b) are one-sided Gaussian factors with FIXED precision:** zero on the legal side, ½λ r² when
  violated, exact Hessian λJJᵀ. (2c): λ = 1/σ_wall² with σ_wall = `f.wall.sigma_m`; (2b): its existing gain as the
  fixed precision. A probit (−log Φ) wall term was rejected (pushed the box 3.6 cm off the wall in one update).
- Stated limit: cx and w stay unobserved by front rays (a miss carries no information); a beam model in which a miss
  is evidence is later work.

### 7c.7 Persistent errors never rescale; the localisation error is the robot's pose error (rev 3.2, 11-18)

Supersedes 7c.2's b_chain definition, its Dynamics paragraph (the D map) and the stated "no chain latent on yaw".
Owner 11-18: "Go" (option 2 of the root-cause report, `.superpowers/sdd/2026-11-15-lidar-measures/bchain-jump-report.md`).

**What was measured (live9, nightstand_9).** On the second persistent update b_chain,x went 0.0015 → 0.445 m (8σ of its
~5 cm prior) and cx went 0.89 → 0.54 m, opposite signs, so the result is not the MAP of the intended model. The final
centre error was 0.374 m. The linear test `bchain_jump_test::linear_scale_coupling` (branch `wip/bchain-jump`,
f5b5496f) reproduces the live split within 1 cm and 3 cm from the logged inputs with no GN iteration at all.

**Why (the D map makes our own σ an observation).** 7c.2's b_t = ρ D b_{t−1} + w, D = σ_t/σ_{t−1}, at ρ = 1 is the
deterministic map b_t = D b_{t−1}. A frame pins the measured sum u = θ + b, so the predicted prior pins θ + b/D and
the next frame pins θ + b. The difference (1 − 1/D) b is then MEASURED: with consistent data it forces b → 0, and an
innovation Δu is explained as b ≈ Ic·k·Δu/(Ic·k² + 1/σ_b²), k = 1 − 1/D (no-prior limit Δu·D/(D − 1)), a gain that
peaks near D − 1 = 1/(σ_b√Ic) ≈ 0.06, i.e. at the live value (R4-M1). A change of OUR
uncertainty is not an event in the world, so it must carry no information. In live9 D_chain ≈ 1.05 came from
`compute_chain_cov` evaluating the lever at `inst.model`'s centre (the birth seed on one cycle, the posterior on the
next); the parked jitter of the mask range (1.65017 → 1.65029 m) does the same to the range share of b_cam.
Supplying Δu = 9.5 cm: the first update's GN did not converge in its 4 iterates (cx 0.835 → 0.831 → 0.913 → 0.853 in
the fixture) and P was taken at the last iterate, claiming mm precision at a point the data does not support.

**(a) Rule: a latent's scale lives in its measurement matrix B, never in its transition.** Every latent block evolves
as b_t = R F_t b_{t−1} + w_t, R = diag(ρ_t), F_t the PHYSICAL transport of that error (identity unless stated),
Var w_t = S Σ_t S, S = diag(√(1 − ρ_t²)), Σ_t the block's current marginal (diagonal for b_cam, b_L; the full 3×3
Σ_pose,t for e). D is gone; there is no created-error term Q in this spec (R4-I1). Predict, per block: x_b ← R F x_b,
P's rows and columns of the block by R F (cross blocks with θ and with the other latents included), P_bb += S Σ_t S.
PSD by construction. A block is a NEW SOURCE when it has never been predicted or its previous Σ was zero: ρ = 0 for
that transition (T4-1c). `predict_latents` becomes this block form (R4-I4).

**(b) b_chain becomes e = (δx, δy, δψ) ∈ ℝ³, the robot's pose error in the room frame** (e = estimated − true pose).
- **Σ_pose,t (R4-C3)** = the robot's pose covariance in the ROOM frame on (x, y, ψ). The published edge is robot→room
  (the robot is the RT root): its `rt_covariance` slots {0, 1, 5} at the capture stamp hold Σ_edge for
  T(robot←room) — the room origin's uncertainty in the robot frame, NOT the robot's in the room. Σ_pose,t =
  J Σ_edge Jᵀ, J = `se2_inverse_jacobian` (common/place_memory/place_map.h) at the stored edge pose (inversion is an
  involution). Unit test: round trip against `room_scene_graph.cpp`'s forward map (`rt_block_of`), and Σ_pose,t equals
  the `proto_room.csv` pose σ of the same stamp (1 %). No other edge of the camera / Helios chain carries a covariance
  (verified at d8215e99: robot_concept writes the mounts with `insert_or_assign_edge_RT_static`, no covariance); a
  mount error is constant in the robot frame and belongs to b_cam / b_L. 7c.2's "no localisation yaw variance is
  published" was wrong.
- **Measurement (R4-C1).** B_chain,t (6×3): row cx [1, 0, −ℓ_y], row cy [0, 1, ℓ_x], row yaw [0, 0, 1], rows H, w,
  h = 0. The lever ℓ ∈ ℝ² is STATE OF THE INSTANCE, never a function of the belief: set when e is a new source to
  (seed centre − robot position at the capture stamp), then moved only by the odometry, ℓ ← ℓ − Δr, with the same Δr
  that enters F. When ρ_chain < 1 it relaxes with the error: ℓ ← ρ_chain(ℓ − Δr) + (1 − ρ_chain)(c − r_t), c the
  belief's centre, r_t the robot position at the capture stamp (a re-drawn error may take a new lever; a persistent
  one may not). B_chain is constant over the GN iterates of a cycle. Linearising at the belief's centre instead was
  rejected: re-linearising the bilinear term θ_c + δψ·J(θ_c − r) at a moving estimate creates information along an
  unobservable direction (the EKF-SLAM inconsistency; first-estimate Jacobians are the standard cure) — with the
  live inputs it gives e_ψ = −9.3σ and cx 0.89 → 0.59 m, the live failure again. Stated limit: ℓ differs from the
  true lever by the seed's error plus odometry drift; the model error is σ_ψ·|Δℓ| (7 mm at 0.024 rad and 30 cm),
  constant while parked. Both channels use the same B_chain (one robot pose per cycle; the camera–Helios stamp
  difference is stated, not modelled). H_cam = [I B_chain,t I 0], H_L = [I B_chain,t 0 I]; the θ-only channel is
  unchanged. K = 3 + 6 + 6 = 15.
- **Transport.** F_t = [[1, 0, −Δr_y], [0, 1, Δr_x], [0, 0, 1]], Δr the robot's room-frame displacement integrated
  from the odometry 7c.2 already walks (a heading error turns into a position error as the robot drives).
- **Consequence (R4-C2).** At ρ_chain = 1, B_chain,t F_t = B_chain,t−1 exactly: the error e predicts at the object
  does not change, parked or driving. An object alone does not measure the robot's pose error (physically true: a
  heading error rotates the whole dead-reckoned path about where it arose, so the object's measured room position
  does not move), and nothing is learned from our own lever. e's job is to be counted ONCE for both channels, with
  the right correlation between cx, cy and yaw, and to be forgotten at the rate the localiser corrects (ρ_chain).
  Equivalently, at ρ_chain = 1 the model is a constant 3-D error B₀e₀ on (cx, cy, yaw) in the object's coordinates
  (R4-M3). Rejected alternative (F = I, parallax kept): it reads a dead-reckoned stretch as a measurement of δψ.
- **ρ_chain** as 7c.2, with the Mahalanobis scale Σ_pose,t (3×3) and the correction c ∈ ℝ³ (R4-I4): (pose
  increment) ⊖ (odometry increment) on (x, y, ψ), the ψ part from `robot_current_angular_speed` over the same
  stamps (`ChainTracker` / `odometry_step_room` gain ψ); an interval that cannot be integrated is absorbed into the
  reference on all three (fail closed, as today).
- **Stated limits.** The marginal of e follows Σ_pose,t only through ρ_chain: on a stretch with no correction at all
  (ρ_chain = 1 while Σ_pose,t grows) it lags, logged as √diag Σ_e against √diag Σ_pose,t (R4-I1). e is carried per
  instance: two instances seen in one cycle hold independent copies of one physical error (per-object marginals
  right, cross-object correlation not represented, R4-M4). e is as honest as Σ_pose,t: a collapsed published pose
  covariance while parked makes ρ_chain → 0 on mm jitter and shrinks the mixing target (R4-M5; logged).
- **Bookkeeping (R4-M7).** The belief stores the last Σ_pose,t (3×3) and ℓ; `sanitize_covariance` re-seeds e at that
  Σ_pose with zero cross terms; `canonicalize` / mode rotation leave e and ℓ untouched; `sig_prev_` remains only as
  the new-source flag of b_cam, b_L. LTSM restore stores neither e nor ℓ (a pose error is not an object property).
  `compute_chain_cov` stays only for the flag-off path.

**(c) b_cam and b_L keep their 6-vector in θ's coordinates, with F = I and no rescale.** ρ_cam already contains
exp(−|Δz|/z): the same relative range change that changes σ_cam decorrelates b_cam, so its marginal follows
σ_cam,t² with a lag of about one e-fold of relative range (range share of σ 1.79× wide after 3 → 1 m, 0.71× after
1 → 3 m, inside §8's factor 3), and a parked range jitter reaches nothing (it enters only through S Σ_t S = 0 at
ρ = 1). Range-in-B for b_cam (a unit latent times range) was rejected for these reasons (R4-M2): σ_cam is a
quadrature lump of range-free floors and range terms, so one latent has no single scaling; ρ_cam already declares
the error re-drawn after one e-fold of relative range, so nothing is claimed across ranges; and a range taken from
the belief would need (b)'s fixed-lever treatment. b_L's σ is constant (D was 1 already).

**(d) The update converges before P is taken (R4-I6).** A solver tolerance is not a modelling threshold.
- Damping: a proximal (Levenberg) term on θ only, in the inverse-free form
  (𝟙 + P̄(W + λM)) δ = P̄G − (x̃_k − x̄), M = blockdiag(diag(W_θθ), 0), x_{k+1} = x̃_k + δ.
  W + λM is PSD, so the solve exists for any PSD P̄; a zero-variance latent keeps δ = 0; λ = 0 is today's step.
- Acceptance by the joint objective, without P̄⁻¹: J = Σ_ch E_ch(θ + B b) + E_θ(θ) + ½ yᵀ(x − x̄),
  y = G − (W + λM)δ (then x − x̄ = P̄y, so the last term is the prior energy). J falls: accept, λ ← λ/4; else reject,
  λ ← 4λ (first λ = 10⁻³), re-solve without re-tracing. λ and its factors are solver constants.
- Stop when the accepted θ step (after `apply_constraints`, yaw as the wrapped difference) is below 1 mm on cx, cy,
  H, w, h and 1 mrad on yaw, or after `gn_iters_max` = 12 accepted or rejected steps. The plan measures the cycle
  cost on the nightstand fixture and reports Hz before live (§8 asks ~10 Hz per class).
- P at the returned θ, plus P_θθ += δ_last δ_lastᵀ always (below the tolerance it adds < 1 mm²; an unconverged frame
  carries its own remaining step as uncertainty). A frame that ends unconverged is logged (`gn_conv` = 0); no gate.
- Caveat for the plan: with the b̄ restart the fixed point is the joint stationary point only to first order in
  b − b̄. If J stalls above the tolerance on the fixture, the consistent variant is x̃_k = (θ_k, b_k) linearised at
  θ_k + B b_k (same solve); the per-iterate ρ = 0 equality then becomes a fixed-point equality.

**Log per cycle (additions):** e (3) and √diag Σ_e, ℓ, √diag Σ_pose,t (also reported while parked), the correction
c (3), `gn_it`, `gn_conv`, the final λ.

**Pass (additions to 7c.5):**
- `bchain_jump_test::linear_scale_coupling` (live inputs: belief centre (0.92, 3.75) on update 1, (0.89, 3.90) on
  update 2, robot (0.83, 2.24), Σ_pose (3.4 cm, 4.35 cm, 0.024 rad)): the result equals the run whose belief centre
  does not move (1e-9), cx = 0.9385 (1 mm), |e| ≤ 0.1σ per component. `lever_arm_sensitivity` (ray-cast): e and θ
  equal the control whose seed centre is the truth, within σ_ψ·|seed − truth|.
- Parked range jitter: 1000 identical frames (ρ = 1), the mask range jittered by ±10⁻⁴ relative, against the same run
  without jitter: θ, b and P equal to float tolerance.
- Transport invariance: a ray-cast nightstand, the robot pose dead-reckoned with a true heading error of 0.02 rad, a
  90° arc at 1.5 m, ρ_chain = 1: B_chain,t F_t = B_chain,t−1 to 1e-9 at every update; √diag of e's posterior does
  not fall below its value after the first update (0.1 %); NEES of e within [0.1, 10]; θ's error stays within its
  own 2σ.
- Σ_pose: the round trip and the `proto_room.csv` equality of (b).
- Convergence: the fixture's first update (seed at the birth on priors, class mean 6 cm off) returns `gn_conv` = 1
  within `gn_iters_max`; J is non-increasing over accepted steps; θ equals the same solver run for 200 steps (1 mm,
  1 mrad) and √diag P equals it within 5 % per component. Second update, parked: θ moves less than the tolerance.
- The ρ = 0 regression of 7c.5 runs with Σ_pose = 0 (a zero-variance e; b_cam, b_L as today) and pins the persistent
  path against the flag-off path without chain covariance, on fixtures where no step is rejected (stated in the
  test). The "chain once for both" equality becomes the 3×3 form on (cx, cy, yaw): data information =
  (B Σ_pose Bᵀ + (Σ_cam⁻¹ + Σ_L⁻¹)⁻¹)⁻¹, B the 3 non-zero rows of B_chain. Every other 7c.5 check still passes.
- Live: rerun on, off, shift; no instance's |e| exceeds 3σ of Σ_pose; √diag Σ_pose,t while parked reported; §8 as
  before.

**Amendment 11-18 (rulings T14-1…T14-7, T13-1; final review I2).** Where this conflicts with (d) or the Pass list above,
this governs.
- **Convergence (T14-4, supersedes T14-1's definition).** `gn_conv` = 1 when an ACCEPTED θ step is below the tolerance
  (1 mm on cx, cy, H, w, h; 1 mrad on yaw), OR when a damped proposal is rejected below it and the UNDAMPED (λ = 0)
  proposal from the same θ is below it too (one extra solve per linearisation; a growing λ cannot fake it). Otherwise
  the update stops at `gn_iters_max` with `gn_conv` = 0, logged, no gate.
- **δ_last (T14-1, kept by T14-4).** P_θθ += δ_last δ_lastᵀ with δ_last = the last ACCEPTED θ step, zero if none was
  accepted. Not the remaining (rejected) step: on a ray edge that is the cm-scale undamped kink step, and adding it on
  every parked frame would be process noise by another name.
- **Why the ray fixtures do not converge in the GN sense.** The first-crossing ray energy (§7c.6) is piecewise: a ray
  that stops crossing the model drops its term, ≈ 9 nats per ray edge (Task 6b). The minimum sits on a ray edge, where
  every step into the edge is rejected and the undamped step still points ~1 cm into it, so `gn_conv` = 0 is the true
  report there (T14-7). Read live `gn_conv` = 0 with `gn_it` = 12 while parked as "θ on a ray edge", not "diverged".
- **Pass items deferred to the smooth-ray follow-up** (a smooth ray likelihood removes the kink; until then these are
  printed, not checked):
  - "the fixture's first update returns `gn_conv` = 1": CHECKed on the camera-only fixture (smooth energy, converges in
    4); on the camera + ray fixture it reports `gn_conv` = 0, while θ equals the 200-step run (1 mm, 1 mrad) and √diag P
    equals it within 5 % — both CHECKed (T14-7).
  - `lever_arm_sensitivity`, "θ equals the truth-seeded control within σ_ψ·|seed − truth|" (T13-1: the FULL control,
    seed and lever): e is CHECKed per component (|e| ≤ 0.1σ, measured ~1 mm) and yaw within σ_ψ; θ's position is
    PRINTED, measured cx +4.41 mm against a bound of 3.87 mm — both runs stop ~0.2 mm from a ray edge, so the
    difference is the resolution of the piecewise ray energy, not the update (T14-5).
- **"Second update, parked: θ moves less than the tolerance" is replaced (T14-2, T14-5).** Repeated frames carry
  independent per-frame noise, so θ legitimately moves away from the class-prior pull after the first frame. Camera
  only: over 10 identical parked frames the θ step sizes are non-increasing (1e-9 slack) and the step at frame 10 is
  below the tolerance. With rays: the step is exactly 0 from some frame ≤ 10 on, and the total θ drift after frame 1
  is < 1 cm per position component.
- **ρ = 0 regression (T14-3).** Damping changes the iterates by design; the CONVERGED results are compared (1e-6).
- **Yaw near ±π (final review I1).** Inside the loop each iterate's yaw is kept on x̄'s branch (the multiple of 2π
  nearest x̄'s yaw, after `apply_constraints`), so the prior pull and the acceptance's prior energy see one unwrapped
  yaw difference; θ is wrapped once after the loop. Pass: `wrap_invariance_near_pi` — the same π-symmetric fixture
  seeded at −5 mrad and at π − 5 mrad gives the same θ (mod π), b and P, to the float floor of a float state at yaw ≈ π.

### 7c.8 A miss is evidence: the persistent ray likelihood is a smooth beam model (rev 3.3.3, 11-18, owner: "Go"; C3 = A)

Supersedes 7c.6 (A)'s "a near miss is a miss (no term)" and its stated limit ("cx and w stay unobserved by front rays";
"a beam model in which a miss is evidence is later work"), and the Amendment 11-18 deferrals of 7c.7 (T14-5, T14-7).
**Owner ruling 11-18 (C3 = A): this also reverses 7c.3's parked "First-hit physics stays out"** — on the persistent
path a return beyond the model's predicted hit now carries a term (it refutes the model), returns are staged by ray
geometry, and the camera's ownership m enters as a mixture, not as a weight. Rev 3.3.1 folded Fable's rev 3.3 review
(`.superpowers/sdd/2026-11-15-lidar-measures/fable-7c8-review.md`, NOT READY; findings cited as R5-*); rev 3.3.2 folds
the re-review (`fable-7c8-rereview.md`, READY WITH CHANGES; cited as R6-*); rev 3.3.3 folds the second re-review
(`fable-7c8-rereview2.md`, READY WITH CHANGES; cited as R7-*) and the rulings of 11-18. A line marked "owner 11-18" is
the owner's ruling; a line marked "controller's call" is the controller's, and the owner may change it.
Persistent path only: the flag-off path and every other caller of `lidar_ray_factor.h` and `lidar_select.h` (hood,
cabinet, table, door, bottle) stay bit-identical; the new code is new functions and new types, the old ones are not
edited.

**What was measured.** live10 on (`docs/superpowers/results/2026-11-18-lidar-measures-live10.md`): on updates with rays
5–20 % stop at `gn_iters_max` with `gn_conv` = 0, and the ray fixtures cannot converge in the GN sense at all: the
first-crossing energy drops ½pc²·log(1 + e²/(c²A)) ≈ 9.4 nats (p = 2500 m⁻², c = 0.05 m, A ≈ 1.25) the moment a ray
stops crossing the box, so the minimum sits on a ray edge and every step into it is rejected (T14-7). Two Pass items
are printed, not checked, because of it (T14-5, T14-7). The same jump hides the other half of the information: a ray
the model misses carries no term, so a box can shrink away from a ray that DID stop on the object at no cost, and the
silhouette edges — the only thing a front-facing sweep says about cx and w — are not used.

**(a0) The data: rays are staged by ray geometry (R5-C3; owner: A), each with a camera label that is never certain.**
Today `select_into_weighted` stages a return only when its ENDPOINT lies in the select box and m > 0, so a ray that
flew past the object is absent or has weight ≈ 0, and only on-object rays would act: every silhouette edge would be
pulled outward by about 1–2 σ_s with nothing to balance it. New staging function and ray-set type (in
`common/lidar_select` and `common/ai_belief`; `LidarRays` and the old functions untouched):
- **Staged:** every return of the sweep whose ray, from the sensor through the return and on to ρ_max, passes through
  the select box (the same box as today: mask-centroid anchored, birth-sized, θ-independent), whatever its range,
  PROVIDED the camera has information on it (next bullet). A return short of the box is staged too (it costs nothing
  unless the camera claims it, see (a)). A return with ρ_i > ρ_max is staged as ρ_i = ρ_max ("no return in range";
  R6-M4).
- **No camera information ⇒ not staged (R6-C1; owner 11-18).** A return whose endpoint projects outside the image or behind the
  camera has no label. Under the mixture m = 0 would say "certainly foreign", the strongest possible statement, and a
  return on the object's real surface outside the image would then push the face back by 4–5 cm. So the ray is left
  out: the same missing-input rule as the cycle without membership below, applied per ray. Stated limit: the part of
  an object outside the image gets no LiDAR term (at close range the LiDAR covers only what the camera sees).
  Interface (R7-M6): `make_membership` returns a float and reads "no projection" as 0; it stays as it is for the
  flag-off path (§8's Unit line "a return behind the camera or outside the image has m = 0" is about that function).
  The persistent label is a new function that returns an optional: no value ⇒ not staged.
- **The label is never certain (R6-C1; owner 11-18: one measured per-class constant, accepted).** m_i = ε_px + (1 − 2ε_px)·π_mask·Φ(s_i/σ_px), so m_i ∈ [ε_px, 1 − ε_px]. ε_px is
  the probability that a pixel's mask label is wrong for the surface behind it, either way: the camera's label read
  through a symmetric binary channel. The re-review offered an off-mask floor or a stated limit; the floor was
  chosen and the owner accepted it, because the alternative is known to be wrong in the direction that matters — with m → 0 off the mask,
  returns on the object's real surface in the band a mask erodes (the legs and rim the select band spans on purpose)
  refute the model, and the LiDAR silhouette settles on the mask boundary, enforcing the erosion it was meant to
  correct. With m = ε_px such a return SUPPORTS the model once ε_px > 0.5 % (it is neutral where
  m/(1 − m)·St(0)/p_pass + G(0) = 1). The same bound removes the division by zero at `sl.confidence` = 1.
  **ε_px is a new constant, flagged:** a measured property of the detector, not a tuned gain. `Model.MaskPixelErrorRate`,
  per class; the plan measures it from the Webots ground-truth masks on live10's frames (the larger of: the fraction
  of object pixels more than 3σ_px outside the mask, the fraction of mask pixels more than 3σ_px outside the object)
  and the value and its measurement are reported to the owner before live, together with the flush depth b*(ε_px) it
  implies (see "Near background" in (a): a larger ε_px buys robustness to erosion and costs a deeper band in which a
  flush neighbour is rewarded; the owner sees both numbers). Measurement notes (R7-M3): the rate the mixture needs
  is over LiDAR RETURNS (the label at the projected endpoint against the object the return actually hit), which
  includes the parallax between the two sensors; use returns if the Webots ground truth allows, pixels otherwise,
  and say which. The plan first confirms that live10 recorded ground-truth masks; if not, the measurement is a new
  run. If the measured value is below 0.5 % the erosion argument above no longer holds: Camera coverage (ii) in Pass
  decides, and the value is not raised to pass it. Load check 0 < ε_px < ½ (a config validation). It is the one place the model needs it: 1 − m_i is the probability that the camera's attribution is
  wrong, and it is what a miss of a claimed return costs.
- **Sweep weight.** The two sweep down-weights (ray-count coverage on Σq, angular coverage ca) are computed as today
  on the returns whose endpoint is inside the box, from the UNFLOORED label q_i = π_mask·Φ(s_i/σ_px) (today's value;
  0 for a return without a projection), and handed over as one number κ = coverage·ca ∈ [0, 1] in the new ray-set
  type. The floored m_i is used in the likelihood only (R7-I1): with the floor in Σ, κ would be positive whenever
  anything stands in the cylinder, and a sweep the camera attributes to nothing would count as a measurement. They do NOT enter a staged precision (R5-C4, see (a)); `precision` stays the flag-off path's.
- **"The LiDAR channel measured" (R6-I4)** ⇔ κ > 0 and the staged set is not empty: today's rule (precision > 0 was
  Σm > 0) on the new field. `frame_has_lidar`, `rays_usable`, the b_L prediction, `record_channels` and the
  `lidar_only` cycle read that on the persistent path. A sweep with κ = 0 is not a measurement: §8's "a LiDAR-only
  frame whose memberships all vanish leaves the belief at the prediction" holds through this line. The instruments
  (`probe`: resid_m, meanz, topz, floorz; `dbg_lidar_rays`; the coverage inputs) keep reading the endpoint-in-box
  subset, so they mean what they meant; the staged-ray count is a new ai2_log column. The bpearl extra set is not
  staged on this path (it is off for these classes, §9).
- **A cycle without membership (owner 11-18: the rays are skipped)** (no RGB frame staged, camera not bound, no focal length, empty slice): the ray-set
  is not staged on the persistent path, the camera update of that cycle runs as usual, and the cycle is logged. The
  reason (R6 §4): the camera is what separates this object from its neighbours; without it the rays would either lock
  the box onto whatever stands within 10 cm (a flat label) or erode it (label 0). Nothing is asserted about the world
  and P is not touched by rays. (Today such a cycle runs unweighted.) Before building, the plan counts in live10's
  logs how often it happens and the longest run of consecutive such cycles per instance: a systematic cause (a pose
  lookup failing while the robot turns) would remove the LiDAR exactly while moving.

**(a) The beam model: marginalise the surface offset instead of linearising it.** 7c.6 already says the model surface
is uncertain by σ_s = `model_sigma_m`, and turned that into a range variance σ_s²/(n̂·u)² — a linearisation that is
exact on a face and fails at an edge, where the same offset decides whether the ray hits at all. Here the offset is
integrated, independently per ray and per sweep (R5-M4: the part of the model error shared by all rays is b_L's, whose
position and size floors cap what repeated sweeps can accumulate). For ray i (origin o, direction u, observed range
ρ_i, ownership m_i), s_i(t) = SDF(o + t·u; θ) on t ∈ [0, ρ_max], ρ_max = `max_range_m` (6 m). The model dilated by δ has
SDF − δ.

- **Convex parts (R6-I1).** The beam path asks the model for its convex parts: the box alone, or the box and the
  headboard panel for the bed (two new hooks on the model, the part SDF and its ANALYTIC Jacobian; the compound stays
  one primitive for everything else). The model SDF is the minimum over parts, s_i(t) = min_p s_ip(t), each s_ip convex
  in t. No finite-difference Jacobian remains on this path, so `fd_eps` and `jac_slope_clamp` do not act here.
- **Closest approach.** d_ip(θ) = min_t s_ip(t) per part, d_i = min_p d_ip: positive for a miss, negative for a
  crossing (how deep the ray goes). There is no ρ_i + 5c window any more (R5-I2): with the window, a model face 5c
  behind a return put a 5-nat ridge in the energy that pushed the model away from the return.
- **First crossing of a level.** t_ip(δ) = inf{t ≥ 0 : s_ip(t) ≤ δ} for δ > d_ip (0 when the sensor is inside the dilated
  part: no special case, the formula then refutes the model by itself), continuous and decreasing in δ;
  t_i(δ) = min_p t_ip(δ) over the parts with d_ip < δ. e_i(δ) = t_i(δ) − ρ_i.
- **Range densities, at the SENSOR's precision (R5-C4).** St(e) = the Student-t whose −log is ½p₀c²·log(1 + e²/c²),
  normalised: ν = p₀c² − 1, scale² = c²/ν, with p₀ = `Model.LidarPrecision` (the base value, 2500; with σ_sensor from the
  SensorModel, 1/p₀ = 1/`LidarPrecision` + σ_sensor²) and c = `robust_c_m`. ν = 5.25, peak 17.5 m⁻¹ at the configured
  values. G(e) = its CDF. ν is fixed at load, so the builder may tabulate G; the gradient then uses the table's own
  derivative, or the table error is below 1e-6 so that St = G′ holds to the gradient check (R6-M9). The staged,
  per-sweep precision `base·coverage·ca` is not used: as a density parameter it changes the peak with the ray count
  and gives ν ≤ 0 below coverage·ca = 0.16. Load-time check: p₀c² > 1, else the agent refuses to start with the
  message (a config validation: the density does not exist otherwise).
- **Foreign density.** p_pass = 1/ρ_max: a return that is not this object's comes from somewhere the model does not
  predict, anywhere along the ray. Under ray staging the ray is not selected on its range, so the uniform over
  [0, ρ_max] is the right one (R5-I3; with endpoint staging it would have been 7c.3's 1/L_i). 7c.3's rest surfaces
  (room polygon, neighbour boxes) are not used here: a neighbour's return is a foreign ray, and it costs nothing
  unless the model blocks it. What that withdraws (R6-M7): a low-m return within about one c of the model's surface
  rewards the model for covering it (a wall 2 cm behind a face, m = 0.05: E ≈ −1.3 nats per ray) — 7c.3's "a surface
  within σ of the predicted face cannot be told apart by range", still a stated limit.
- **Near background behind a silhouette (R7-C1; owner 11-18: accepted as a stated limit).** The same limit acts
  sideways. The model is paid for covering a return whenever m/(1 − m)·St(e)/p_pass + G(e) > 1. For a surface a depth b
  behind the face plane (e = −b on a ray just outside the true edge) the neutral label m* is: b = 2 cm, below 1 %;
  b = 5 cm, 7.5 %; b = 10 cm, 60 %; b ≥ 20 cm, none (never paid). The label falls to 7.5 % only about 1.5 σ_px outside
  the mask boundary, so with a background 5 cm behind the edge settles outside the truth: predicted +1.9 / +3.7 /
  +5.3 / +6.9 cm per edge at σ_px = 1 / 2 / 3 / 4 cm at the object (π_mask = 0.9, ε_px = 2 %; the re-review's 1-D
  integration of this likelihood, not a measurement), +0.1–0.2 cm at 10 cm, 0 from 20 cm. Live σ_px at the object is
  at least 3 cm. The floor removes the bound the mask gave: any surface within the flush depth b* of a face plane,
  ε_px/(1 − ε_px)·St(b*)/p_pass + G(−b*) = 1 (b* ≈ 3 cm at ε_px = 2 %, ≈ 0 at 0.5 %), is rewarded at ANY distance from the
  mask, so along a flush neighbour the rays set no limit on w; the camera points and the size prior do. This is the
  price of the floor, not a reason to drop it (a mask eroded by 3σ_px with a far background leaves the edge within
  2–5 mm). Pass checks the 20 cm and 1 m backgrounds, prints the 5 cm case against this prediction and prints a
  flush-neighbour fixture; b* goes to the owner with ε_px. The model-level answer, if the printed numbers are not
  acceptable, is 7c.3's rest surfaces (a published neighbour explains its own returns); not in this revision.
- **The likelihood: geometry (hit with probability π_i over δ) × ownership (member with probability m_i).** A member
  return is on the model's surface: density St(e), and impossible if the ray misses. A foreign return is uniform, and
  must lie in front of the model's surface if the ray crosses it (first of two returns, R5-I2): density p_pass·G(e)
  on a crossing, p_pass on a miss.

    p(ρ_i | θ) = (1 − π_i)(1 − m_i) p_pass + π_i · E_{δ | δ > d_i}[ m_i St(e_i(δ)) + (1 − m_i) p_pass G(e_i(δ)) ]
    π_i = Φ(−d_i/σ_s)            (computed with erfc)

  This corrects the first review's C3-A formula, which left the miss branch at p_pass for any m (a claimed return
  30 cm in front of the model then cost 2.3 nats on a crossing and 0 on a miss, so the model was paid to dodge the
  rays the camera attributes to it); the re-review confirms the (1 − m_i). The first-of-two density also puts a factor
  (1 − ρ_i/ρ_max) on the member term; it does not depend on θ, acts as a slightly smaller m, and is dropped (R6-M2).
- **Energy.** E_i = −log[ p(ρ_i | θ) / ((1 − m_i) p_pass) ], zero for a ray far from the model (π_i → 0), so the sum
  stays comparable across cycles. The ray-set's energy is κ·Σ_i E_i: the sweep down-weights temper the energy, the
  gradient and the curvature alike (inherited informativeness weights, not new; one difference from 7c.6, whose
  staged precision p₀κ left the σ_s part of the range variance untempered: here κ tempers that part too, R7-I3).
  Persistent J (7c.7 d) uses it in
  place of `lidar_rays_first_crossing_energy`.
- **Quadrature (R5-I5, R6-I1), in double.** With one part, nodes at conditional quantiles in the upper tail:
  δ_k = −σ_s Φ⁻¹(π_i u_k), u_k and weights ω_k the K Gauss–Legendre nodes on (0, 1); E_δ[f] = Σ_k ω_k f(δ_k). No
  cancellation in either tail; as d_i → −∞ the nodes are fixed. Φ⁻¹ accurate to 1e-15 (AS241 or equal). K is chosen by
  the accuracy test in Pass, not by the fit rate; K = 8 is the starting point.
  With several parts the δ integral is split into panels at every level where the first-hit part changes, so the
  integrand is smooth inside each panel: at each part's own minimum d_ip > d_i (the level at which part p starts to be
  hit; t_i may jump there), and at each level where the first-hit part switches (t_ip = t_iq; one root-find in δ per
  switch, found from the sign of t_ip − t_iq at the panel ends and nodes). With upper-tail probabilities
  b_0 = 0 < b_1 < … < b_n = π_i at those levels, panel j has nodes δ_jk = −σ_s Φ⁻¹(b_{j−1} + (b_j − b_{j−1}) u_k) and weight
  (b_j − b_{j−1})/π_i · ω_k, so the weights sum to 1 and E_δ is the same conditional mean as with one part (R7-I2; the
  likelihood multiplies it by π_i once). Rev 3.3's π_k ∏(1 − π_j) and rev 3.3.1's records are withdrawn: one offset moves the whole model,
  so the crossings are nested events, and a record that appears together with a local maximum (a ring at elevation
  ≈ 0 running over the mattress to the headboard) left a near-step inside one panel. What is true of continuity: the
  panel limits move continuously with θ and a panel's weight goes to 0 when its two limits meet, so the sum is
  continuous in θ; it is C¹ except where two parts tie at the minimum (d_ip = d_iq) and at the kinks listed below. A
  double switch between two neighbouring nodes is not found; the accuracy test bounds what that costs.

Numbers at the configured values, approximate (the unconvolved peak; the δ integral makes them about 0.11 nats (measured 0.107 at K = 8, Task 16; rev 3.3.3 said 0.07)
smaller in magnitude face-on, so a fixture is compared with the computed value, not with these; R6-M6): a claimed
return on the surface has E ≈ −6.85 nats at m = 0.9 and −4.66 at m = 0.5, and 0 if the model misses it. A foreign
return (m = 0.05) 40 cm behind a model that fully blocks its ray costs ≈ +11 nats, growing as 3.1·log(1 + e²/c²). A
return in front of the model costs at most −log(1 − m) and nothing if the camera does not claim it.

**The depth basin is finite (R6-I2).** With the model BEHIND claimed returns the member term is masked by the foreign
one once m·St(e) < (1 − m)·p_pass: beyond e ≈ 14 cm at m = 0.9, 9 cm at m = 0.5. At 30 cm and m = 0.9, E ≈ −0.012 nats
and the gradient is 0.24 nats/m per ray, against about 20 nats/m per ray today. That is what the mixture says ("foreign,
in front of me" costs the same as a miss), and it is a behaviour change: a seed more than ~10–14 cm behind the returns
gets almost no ray pull and no ray curvature, and is brought in by the camera points. Pass checks that it converges.
Accepted (controller's call, owner may change).

Continuity and its limits (R5-C1, M5). E is continuous in θ everywhere. It is C¹ except: (i) at d_i = 0 on an edge or
corner, where the slope of d_i changes by a factor in [0.71, 1] (outside, d is the distance q to the edge; inside a
chord cutting the corner with legs a, b, d = −q·√(a² + b²)/(a + b)) — both sides have the same sign, so no minimum is
held on the kink; (ii) for a ray parallel to a face (non-unique minimiser), a kink in yaw; (iii) where two parts tie.
The silhouette-distance definition of d (support function, linear through tangency) would remove (i); not taken here,
noted for the plan if `gn_conv` suffers.

Stated limits. (1) **Gaps under furniture (R6-I3).** A solid, floor-anchored box is refuted by rays that pass through a
real gap (under furniture on legs) and return from behind, and ray staging stages every such ray wherever it lands:
e = −1…−3 m costs 19–26 nats per ray. Owner 11-18: the legged fixture with its bound now, τ later. NO transmission
term now. A solid that transmits with
probability τ (foreign density p_pass·[τ + (1 − τ)·G(e)], refutation bounded at −log τ) is the model-level answer, but τ
would be a per-class constant with no measurement behind it — a tuned constant by another name. Instead the limit
gets a number before live (the legged fixture in Pass, with a bound), and a per-cycle log (the count of rays with
π_i > 0.99 and e < −5c). If the fixture fails its bound the result goes to the owner with τ, or a model whose box
stands off the floor, as the proposals; the fixture is not tuned. (2) The beam width is not modelled (see (d)).
(3) The part of an object outside the image gets no LiDAR term ((a0); owner 11-18). (4) A background or neighbour
less than ~10 cm behind a silhouette moves the edge outward ("Near background" in (a); owner 11-18).

**(b) The step.**
- **d and ∇d (R5-C1, R6-M1).** For d_ip > 0 (a miss of a box-shaped part): the closed-form distance from the ray to the
  part and its analytic gradient. Two tangents are not used there: outside a corner s(t) = √(q² + x²), and a
  two-tangent estimate is ≈ q²/ε for q below the tangent offset ε, with half the right gradient. For d_ip < 0 the
  part's SDF along the ray is a maximum of linear functions, so its minimum is on a crease where two faces tie, and
  the Jacobian at that point is not the derivative of d: take the slopes s₋ = −∂s/∂t at t* − ε and s₊ = +∂s/∂t at t* + ε
  and the Jacobians J₋, J₊ there; d = the intersection of the two tangent lines (exact for a V, no 1e-4 m staircase);
  ∇d = (s₊ J₋ + s₋ J₊)/(s₋ + s₊). A minimum at t = 0 or t = ρ_max has one side only: d = s there, ∇d its Jacobian
  (R6-M4). ∇d_i is that of the part with the smallest d_ip.
- **Gradient:** the exact gradient of the discretised κ·Σ E_i, through d_i (π_i and the node levels), the panel
  limits, and the crossings: ∂t/∂θ = −(∂s/∂θ)/(∂s/∂t) and ∂t/∂δ = 1/(∂s/∂t) at each node's crossing, as today's implicit
  differentiation.
- **Curvature, PSD, two parts per ray, each × κ.**
  - Range part, total variance (R5-C2): r_i · J̄ J̄ᵀ / V_i, with t̄, J̄ and Var(t) the mean of t_i(δ_k), the mean of
    ∂t_i(δ_k)/∂θ and the variance of t_i(δ_k) under the quadrature weights (the prior over δ given a hit),
    V_i = (1 + ē²/c²)/p₀ + Var(t), ē = t̄ − ρ_i, and r_i = the posterior probability that the return is a member hit
    (its term's share of p(ρ_i | θ); complete-data information, as 7c.3). On a face Var(t) = σ_s²/(n̂·u)² and this is
    7c.6's a aᵀ/((n̂·u)²·R_range + σ_s²) exactly, bounded by 1/σ_s² at grazing incidence. Rev 3.3 averaged the per-node
    information instead, which drops the σ_s term from W: 1.2× too much face-on, 7× at n̂·u = 0.2, unbounded beyond.
    Not bounded by 1/σ_s² off a face (R6-M3): for a claimed ray the model misses by 1.5–3σ_s outside a corner, J̄·∇d
    grows as (d/σ_s)·√(π/2) while r_i stays ≈ 1, so the range part reaches about 2/σ_s² along ∇d. Transient (P is
    taken at the final θ); the largest per-ray curvature is printed on the silhouette fixtures.
  - Edge part, expected information of what this ray could answer (R5-I4, R6-C2): (1 − o_i)·I_d,i · ∇d_i ∇d_iᵀ,
    I_d,i = (φ(d_i/σ_s)/σ_s)² · [ m_i/π_i + (1 − m_i)(1 − t̄_i/ρ_max)/(1 − π_i) ]
    o_i = (1 − m_i)·p_pass·Ḡ_i / ( m_i·S̄_i + (1 − m_i)·p_pass ),   S̄_i = E_δ[St(e)], Ḡ_i = E_δ[G(e)]
    I_d is the Fisher information of the hit/miss outcome on d (a member return can only come from a hit; a foreign
    return beyond the surface only from a miss; exact when the peak-to-pass ratio is large, 105 here). o_i is the
    posterior probability that the return is foreign and in front of the surface: such a return never reached the
    surface and says nothing about hit or miss. Without the factor, twenty rays stopped by a person in front of one
    silhouette claimed σ_d ≈ 5 mm on an edge nothing observed, and "occlusion HOLDS" was broken. o_i uses only whether
    the return reached the surface, not whether the ray was a hit or a miss, so it does not bring back rev 3.3's
    realised score (most information when the fit is worst). Evaluated with the Mills ratio in the tails; bounded by
    0.64/σ_s². Known side (R6-C2): for an unoccluded ray with far background the outcome is fully observed and its
    information is φ²/(σ_s²·π(1 − π)); the bracket gives about half of that at m = 0.9, the under-confident side. The
    Calibration item decides between the two forms on a number; the bracket is kept until that number says otherwise
    (controller's call, owner may change).
  - Bed, a ray whose first hit switches part (R7-M1): t̄, J̄ and Var(t) are taken over all nodes, so Var(t) is of
    order m² when the mattress edge and the headboard are metres apart and that ray's range part vanishes; the edge
    part uses d_i = min_p d_ip only. Under-confident side, PSD. If the bed's H or `gn_conv` suffers, the plan moves to
    per-panel range parts, Σ_j r_ij·J̄_j J̄_jᵀ/V_ij with r_ij the panel's posterior share.
  - The cross term between a ray's range score and its edge score is dropped (R5-M8).
- 7c.7(d)'s damped step and acceptance on J are unchanged. P is `joint_cov` at the final θ with this curvature in W.
  Within a sweep both parts enter I_L and are saturated by b_L and e like every ray term.

**(c) Cost (R5-I7), stated before building.** Today a crossing ray costs ≈ 37 SDF evaluations per build (march ~10,
bisection ~13, slope 2, one finite-difference Jacobian 12). Here a ray near the model costs ≈ 220 SDF-equivalents per
build (closest approach and tangents ~40, K = 8 crossings × ~10, 8 Jacobians) and ≈ 110 per energy evaluation, per
part (the bed has two); ray staging also brings in the rays that pass the object (estimate 2–3× the count), most of
which are far misses. Estimate at 256 near rays and ~30 ns per SDF: 1.7 ms per build, 0.8 ms per trial energy, ≈ 10 ms
for a 4-step update and ≈ 30 ms at `gn_iters_max`; twice that for the bed. live10's slowest class runs at 8.4 Hz
(119 ms); 8 Hz is 125 ms, so the headroom is 6 ms and the estimate does not fit without the savings below. They are
part of the task, not optional:
- one monotone march per part collects all K levels (they are ordered along t for a convex part);
- cos/sin of the yaw computed once per build, not per SDF sample; the analytic part Jacobians of (a);
- rays with d_i > 8σ_s stop after the closest approach: π_i < 7e-16 there and, with m_i ≤ 1 − ε_px, the term is below
  the rounding of the double sum (R6-M5). Flagged as a numerical floor at machine precision, not a model gate.
Constants of the section. Solver or numerical, none a model threshold: K (set by the accuracy test), ε of the
one-sided tangents and the 1e-4 m bracket (as 7c.6's bisection), the 8σ_s floor. Data-side and inherited: the select
box, κ. **New model constant, flagged: ε_px** ((a0); measured, reported to the owner). Config validations: p₀c² > 1,
0 < ε_px < ½.

**(d) Not in scope.**
- The LiDAR beam width (R5-I8). The footprint ρ·θ_div (≈ 6 mm at 2 m, 12 mm at 4 m for 3 mrad) is the size of σ_s,
  grows with range, and makes mixed-range returns exactly at silhouettes. It belongs in the edge scale as
  σ_s² + (ρ·θ_div/2)², θ_div from the sensor model — a covariance that grows with range, not a gate — when the sensor
  model carries it. Webots has no beam width, so live11 cannot show its absence; with independent per-ray offsets a few
  tens of edge rays claim mm-level σ on w and cx, bounded only by b_L's floors (3 cm position, 2 cm size). The
  real-robot validation must re-check σ_w and σ_cx against the surveyed truth before this is trusted there.
- The transmission term τ and a box that stands off the floor (stated limit 1).
- The flag-off path's sphere-trace defect (7c.6) stays an owner decision. The bed camera-only non-convergence (live10:
  61 %, 91 % of updates run 12 steps) has no rays and is a separate investigation.
- For the plan, not the model (R6-M8): `lidar_ray_factor.h`'s header note "a ray that misses the model carries no
  term at all" names the persistent exception; consumers of this path are told that the published Σ on w and cx
  shrinks.

**Pass (rev 3.3.3).** Every fixture ray-casts a true scene, first hit only; "live staging" means the rays go through
the new staging function with a membership built from a rendered mask (σ_px and ε_px as live), not labels set by
hand. "The tolerance" is 7c.7's 1 mm / 1 mrad. The bounds chosen in this list without a measurement behind them —
10 frames, 1 cm for the gap fixture, σ_s/2 = 5 mm for the live-staging and camera-coverage fixtures — stand
(controller's call, owner may change).
- Continuity: along a 2 cm sweep of cx across a ray edge in 0.05 mm steps, the second difference
  |E(x + h) − 2E(x) + E(x − h)| of the total ray energy is < 0.01 nats at every step (a jump shows whatever the slope;
  today ≈ 9.4). Same on the bed with the headboard, across a level where the first-hit part switches and across the
  appearance of the headboard's own minimum, with rings at elevations −1°…+1° relative to the mattress top.
- Gradient: analytic against central differences (h = 0.1 mm) to 1 % or 1e-3 nats/m at every step, on the box and on
  the headboard fixture; at a step where some ray's d_i changes sign inside the stencil, against the one-sided
  difference on each side (the stated kink); the same one-sided rule on the headboard fixture at a step where two
  parts tie (d_ip = d_iq) or a panel appears or disappears inside the stencil (R7-M2). One fixture with rays cutting a corner at n̂·u far from 45° checks ∇d's
  weights (R5-C1); one with rays passing 0.05–1 mm outside a corner checks the closed-form d (R6-M1).
- Quadrature: |E_K − E_64| < 0.01 nats per ray over n̂·u ∈ [0.1, 1], e ∈ [−0.3, 0.3] m, d ∈ [−4σ_s, 4σ_s], at m = 0.9 and
  m = 0.05, on the box and on the headboard fixture (elevations as above); on a face, E_K decreasing toward its
  minimum from each side (the minimum sits at e > 0: 7 µm at m = 0.9, 1.3 mm at m = 0.05). The smallest K that passes
  is the K used. The several-part code run on a model with one part (no headboard) equals the one-part code to 1e-12
  per ray, energy and gradient (R7-I2).
- Reduction: rays deep inside a face (π > 1 − 1e-6), m = 1 − 1e-6 and κ = 1 both set by hand (7c.6 puts κ in the staged
  precision, 7c.8 on the whole term; they agree only at κ = 1, R7-I3), at n̂·u = 1 and at n̂·u = 0.25: the
  curvature equals 7c.6's a aᵀ/((n̂·u)²·R_range + σ_s²) within 1 %; θ equals 7c.7's update within 0.1σ per component and
  √diag P within 3 % (the difference is the convolution against its linearisation, printed).
- Density and inputs: with Σq = 5 rays (κ ≈ 0.08) the update is finite and the hit/miss energy difference per ray is
  the same as at κ = 1, scaled by κ. A config with p₀c² ≤ 1, or ε_px outside (0, ½), is refused at load.
  π_mask = 1.0f gives a finite energy. A sweep with κ = 0 leaves state, Σ and ω at the prediction and does not
  restart the channel's odometry path (R6-I4).
- Ownership: rays with d_i > 8σ_s contribute exactly 0. An occluder 30 cm in front with m = 0.05: θ moves less than
  the tolerance AND every component of √diag P is within 1 % of the same update with those rays removed from the
  staged set and κ unchanged (R6-C2, R7-I3). An occluder covering ONE silhouette of an otherwise visible box: σ_w and
  σ_cx are not below, by more than 1 %, the run in which the occluded rays are not staged (1 − o_i is small, not 0). An over-segmented mask on a wall 2 cm behind the face (7c.3 row E), at m = 0.9 and at m = 0.05: printed, with
  the depth each produces (R6-M7).
- Depth basin (R6-I2): camera + ray fixture, seed 30 cm behind the surface, m = 0.9: the depth is within its own σ
  of truth in ≤ 10 frames, and the summed ray gradient along the view axis never points away from the returns. The
  rays-only run of the same seed is PRINTED (expected: almost no motion).
- Silhouette, both directions, hand labels (0.9 on the object, ε_px off it; 10 frames; R7-M4): a front-only sweep of a
  box whose seed is 3 cm too narrow, and one 3 cm too wide, each with the background 20 cm behind and 1 m behind: w
  moves to within σ_w of truth and σ_w falls below the prior's. Today w stays at the seed. With hand labels there is
  no σ_px band, so the 5 cm background is run too and printed. The largest per-ray curvature along ∇d is printed (R6-M3).
- Silhouette, live staging (R5-C3; backgrounds per owner 11-18, R7-C1): the same seeds and a correctly sized seed,
  through the new staging and a rendered mask. CHECK on the 20 cm and the 1 m background: the correctly sized seed
  keeps |Δw| and |Δcx| below σ_s/2 = 5 mm over 10 frames; the ±3 cm seeds end within σ_w of truth. PRINT on the 5 cm
  background: Δw, Δcx against the predicted shift of "Near background" at the fixture's own σ_px, with one bound so
  that a builder defect still shows: the shift per edge is below 2σ_px at the object.
- Flush neighbour, live staging, PRINTED (owner 11-18, R7-C1): a surface 2 cm behind the face plane extending 30 cm
  beside the box, 10 frames: Δw with rays against the camera-only run. The number goes to the owner with ε_px and b*;
  above live11's 1 cm, with 7c.3's rest surfaces as the proposal.
- Camera coverage, live staging (R6-C1): (i) a box half outside the rendered image, correctly sized seed: |Δw|, |Δcx|,
  |Δcy| below σ_s/2 over 10 frames, and no ray outside the image is staged; (ii) a mask eroded by 3 cm on one side:
  the bias of w and of cx is below the camera-only run's (the LiDAR must not make erosion worse), both printed.
- Gap under furniture, live staging (R6-I3): the box raised on a 15 cm gap, background 1 m behind, sensor at the
  Helios height: against the same run without the gap rays staged, cy, h and H each move by less than 1 cm (live11's
  centre tolerance); the count of gap rays and their energy are printed. A failure is reported to the owner
  (stated limit 1), not tuned.
- Calibration: the silhouette fixture through live staging and a rendered mask (m enters both curvature parts), over
  200 noise draws (range noise and a per-ray offset of σ_s): the mean NEES of (cx, w) divided by 2 lies in [0.5, 2];
  cx and w also reported each alone. CHECK on the 20 cm and the 1 m background, each reported separately (the number
  that decides between the bracket and φ²/(σ_s²π(1 − π)) in (b)); the 5 cm background is printed, not checked (R7-C1). The curvature is an approximation, so this is measured.
- Parked: 1000 identical sweeps: σ_w and σ_cx do not fall below b_L's floors and w moves less than 1 mm after
  frame 10.
- The two deferred items come back as CHECKs: the camera + ray fixture's first update returns `gn_conv` = 1; in
  `lever_arm_sensitivity` θ's position equals the truth-seeded control within σ_ψ·|seed − truth| (T13-1 as written).
  Parked: with rays the θ step sizes over 10 identical frames are non-increasing and below the tolerance at frame 10
  (the camera-only rule of T14-5 now holds for rays too).
- Earlier items this section changes: the ρ = 0 regression of 7c.5 / 7c.7 pins the persistent path against the
  flag-off path on CAMERA-ONLY frames; on frames with rays the two differ by design. 7c.5's
  `over_deep_belief_is_not_held` documented the choice this section reverses: it is re-run, its result printed, and
  its expectation rewritten by the plan from that number.
- Creep: the 16-case set of 7c.6 stays 0/16; golden (`lidar_golden_test`) bit-exact; the door and hood tests that
  include `lidar_ray_factor.h` pass unchanged.
- Cost: the ray work of one update on the ray fixture at 256 near rays, first update and parked, box and bed,
  measured and reported in ms; median ≤ 5 ms (the 6 ms headroom of the 8.4 Hz class). The new staging (a
  ray–cylinder test over the whole sweep and one label call per staged ray, 2–3× today's count) sits in the LiDAR
  feed, outside the update: its ms are printed beside it and count against the same headroom (R7-M5).
- Live11 (on, off, shift as live10, ai2_log cleared before each run): fit rate ≥ 8 Hz; `gn_conv` = 0 share on updates
  with rays below live10's (5 / 12 / 3 / 20 %); final centre errors not worse than live10's by more than 1 cm; no DOF
  over-confident (two-sided §8 rule), w and cx reported separately; reported per instance: the staged ray count, the
  rays left out for lack of camera information, the cycles without membership and their longest run, and the
  gap-ray count of stated limit 1.

**Amendment 11-20 (rev 3.3.6; owner: "Spec fix, Fable checks"): the likelihood is integrated in t on a composite rule.**
Rev 3.3.6 folds Fable's rev 3.3.5 review (`fable-7c8-amend-3.3.5-review.md`, READY WITH CHANGES, R9-*).
Rev 3.3.4 (integrate the member term in v = G(e), the foreign term by parts) was reviewed NOT READY
(`fable-7c8-amend-3.3.4-review.md`, R8-*): its change of variable and identities are right, but in v the crossing
bump becomes a spike whenever the surface sits in St's tail (21 nats face-on at e = −30 cm), the by-parts form is
ill-conditioned where a ray refutes the model, and the mass at t = 0 was dropped. This revision folds all of R8. It
SUPERSEDES: (a) "Quadrature" (the δ-quantile nodes are no longer the rule for the likelihood; they stay for the
curvature's moments, below), the rev 3.3.3 sentence "the weights sum to 1 and E_δ is the same conditional mean", (b)
"d and ∇d"'s t* ± ε tangents (its one-sided rule at t = 0 and ρ_max stays), (b)'s "Gradient" bullet (replaced by the
one below), and (c)'s cost count, estimate and constants line (R9-M5).
What was measured (Task 17, 32d879f5, `task-17-report.md`): with δ-quantile Gauss–Legendre nodes no K ≤ 16 meets
|E_K − E_64| < 0.01 nats per ray (box 1.94, headboard fixture 4.95 at K = 8; n̂·u = 0.1 never converges): in δ the
member density St(t(δ) − ρ) is a peak of width ≈ c·(n̂·u) and G a step of the same width, against nodes spread
over σ_s.
- **The map (R8-C3, M1, M2).** For a convex part p the first crossing of a level δ ∈ (d_p, s_p(0)) lies on the
  descending branch t ∈ [0, t*_p], t*_p the SMALLEST minimiser of s_p along the ray, where it is inverted by one SDF
  evaluation: δ = s_p(o + t·u), dδ = ∂_t s_p dt. Every δ ≥ s(0) = min_p s_p(o), the minimum over ALL parts including those the 8σ_s floor drops (R9-M2), maps to t = 0 (the sensor inside the
  dilated model): an atom of mass F(s(0)), F(δ) = P(δ' > δ) the upper tail of N(0, σ_s²). Part p is first hit at t
  iff t ≤ t*_p and s_p(t) = M(t), the running minimum over all parts and all earlier t; a panel behind a nearer
  part q's minimum starts where s_p(t) = d_q (a level crossing), and the first-hit part switches where s_p = s_q.
- **The likelihood, unconditional, in t (R8-C3, I1).** With φ_σ the N(0, σ_s²) density,
      A_i = St(−ρ_i)·F(s(0)) + Σ_panels ∫ St(t − ρ_i)·φ_σ(s_p(t))·|∂_t s_p| dt
      B_i = G(−ρ_i)·F(s(0))  + Σ_panels ∫ G(t − ρ_i)·φ_σ(s_p(t))·|∂_t s_p| dt
      p(ρ_i | θ) = (1 − π_i)(1 − m_i)·p_pass + m_i·A_i + (1 − m_i)·p_pass·B_i,     S̄ = A/π, Ḡ = B/π.
  This is the same quantity as "The likelihood" (the push-forward of δ through the running minimum); only its
  quadrature changes. No division by π: r_i = m·A/p, o_i = (1 − m)p_pass·B/(m·A + (1 − m)p_pass·π). The foreign term
  is integrated directly, never by parts (R8-C2). Every term is positive.
- **The composite rule (R8-C1, I3).** On each panel, Gauss–Legendre with n nodes on each subinterval between
  consecutive breakpoints, integrating the two integrands above directly (one SDF evaluation per node, no crossing
  solve). Breakpoints, clipped to the panel: the quantile levels of BOTH densities — t = ρ + G⁻¹(q_j) (θ-independent)
  and the crossings t_p(δ_j) of the levels δ_j = −σ_s·z_j — with z_j = Φ⁻¹(q_j) on the same L levels for both
  families; t = 0, t = ρ_max and t*_p; and the points where the integrand is not smooth: inside the part (δ < 0) the
  points where the nearest face changes (∂_t s jumps), for δ_c > −8σ_s, and outside the face/edge/vertex region
  boundaries (∂²s/∂t² jumps). All are in closed form in Task 17's box geometry. Starting point: L = 11 levels at
  z_j = −z_max … +z_max at spacing 0.6 (z_max = 3, L = 11) and n = 3: Fable's script on an exact box SDF, one part,
  worst 3.3e-3 / 2.6e-3 / 1.7e-3 nats on face-entry / edge-miss / corner-chord rays (R9-I2; without the geometric
  breakpoints 0.036 and 0.16, so they are required). z_max (with L at the same spacing) and n are numerical constants
  set by the Quadrature item, searched in this order: n ∈ {3, 4}, then z_max and L together (raising n alone at fixed
  z_max plateaus). The part-level 8σ_s
  floor (Task 17) stays: a part with d_p > 8σ_s opens no panel. Why it is exact to rounding (R8-M3): a dropped part
  matters only if it lies in front of the first-hit part, where on a refuted ray St and G are increasing toward
  e = 0, so the dropped mass is at most F(8σ_s)/π of the term kept; on a ray not refuted
  p ≥ (1 − m)p_pass·max(1 − π, π/2) ≥ (1 − m)p_pass/3 (R9-M1). Either way the relative change is below ~1e-15.
- **The curvature's moments (R8-M5)** (t̄, J̄, Var t of the range part, (b)) stay on the δ-quantile nodes, the right
  nodes for the crossing measure, with their own K_m; the node variance of δ is scaled by the truncated normal's
  closed form: Var(t) is taken as Var_nodes(t)·Var_exact(δ | δ > d)/Var_nodes(δ), with
  Var_exact(δ | δ > d) = σ_s²·(1 + αλ − λ²), α = d/σ_s, λ = φ(α)/(1 − Φ(α)) (the Mills ratio, in log space), which is
  exact on a face at any K_m (R9-M3) (Task 17's 2.6 % at
  n̂·u = 0.25, K = 8, is this). K_m starts at 8 and is set by the Reduction item. I_d and o_i use A, B.
- **Gradient (R8-I2).** The exact gradient of the discretised energy: through s_p(t; θ) and ∂_t s_p(t; θ) at every
  node (∂²s/∂t∂θ, analytic on a box part); through π_i and the atom, −φ_σ(s(0))·∂_θ s(0)·[m·St(−ρ) + (1 − m)p_pass·G(−ρ)];
  and through every θ-dependent breakpoint, which moves its subinterval's nodes (needs ∂_t of the integrand at each
  node: St′(e) or St(e), φ_σ′(δ)·∂_t s, and ∂²s/∂t², zero on a face piece, not outside an edge or corner): the level
  crossings t_p(δ_j) (dt = −∂_θ s_p/∂_t s_p, which diverges next to a tangent minimum: the motion term is formed as
  the finite product St·φ_σ·∂_θ s_p, never as ∞·0, R9-I1), t*_p (tangent case −(∂²s/∂t∂θ)/(∂²s/∂t²); crease case the derivative of
  the intersection of the two face lines; for a ray parallel to a face the one-sided rule of stated kink (ii)),
  plateau ends (s_p(t) = d_q: dt = (∇d_q − ∂_θ s_p)/∂_t s_p), switch roots (s_p = s_q:
  dt = −(∂_θ s_p − ∂_θ s_q)/(∂_t s_p − ∂_t s_q)), and the face-change and region-boundary
  points. The integrand at t*_p is NOT zero on a crease (St(t* − ρ)·φ_σ(d_p)·s₋ for a crossing ray): that limit's
  motion counts. Roots at machine precision (Task 17: 1e-4 m was not enough). Stated kink (iv) of the discretised
  energy (R9-I1): a breakpoint that appears, is clipped to a panel limit, or changes order with another is a small
  kink of the gradient (about 1e-3 nats/m per ray); the Gradient item applies its one-sided rule there.
  *Erratum 11-21 (Task 17, review-confirmed; the energy is unchanged, only its derivative is stated correctly):*
  (a) a zero-width subinterval (two breakpoints that coincide, e.g. a return exactly on a face) adds no energy but
  keeps its motion term h(b)·(∂_θb₂ − ∂_θb₁); dropping it gave ∂E/∂cx −44.4 against −1.85e-2 (finite difference) on
  a face-on box ray. (b) The kink (iv) size depends on the fixture: 1e-6 … 1e-4 nats/m on box face ties (n̂·u 1 …
  0.01), up to ≈ 0.6 nats/m reported on a grazing bed ray (unverified by the reviewer); on ray-cast fixtures at the
  truth it is systematic, not measure-zero (the z = 0 crossing and ρ + G⁻¹(½) coincide on every face ray). Tied
  breakpoints are ordered per component, so the analytic value is a one-sided derivative; the one-sided rule covers it.
- **The crease (Task 17).** For d_ip < 0 the minimum is on the crease of the two faces of the part that meet there,
  found exactly from the box geometry; the t* ± ε tangents are withdrawn (near a grazing face they picked the wrong
  face: one bed ray gave ∂d/∂cy 0.638 against 0.034). ∇d = (s₊J₋ + s₋J₊)/(s₋ + s₊) on that pair.
- **Pass changes (R8-I1, I4, I5, M6).**
  - Quadrature: the reference is an INDEPENDENT integration in t (adaptive, or composite with ≥ 10³ nodes and
    breakpoints at the kinks), itself checked by doubling to 1e-4 nats; the grid is rev 3.3.3's with n̂·u extended to
    0.017, the bed fixture included, and the builder adds edge-miss and corner-chord profiles (Fable's script covered
    a face entry only). Bound |E_rule − E_ref| < 0.01 nats per ray; the first (n, z_max) in the search order above that
    meets it is used; if none with n ≤ 4 and z_max ≤ 5 does, the table goes to the owner (not tuned).
  - Mass: per panel, the rule applied to φ_σ(s)|∂_t s| alone returns F(δ_lo) − F(δ_hi), and Σ panels + atom = π (the
    cheapest detector of a missed panel), with an ABSOLUTE bound: |Σ panels + atom − π| < 0.01 and the same per panel
    (a relative bound fails correct code at d = 6σ_s, where π ≈ 1e-9; R9-I3). The several-part code on a one-part model equals
    the one-part code to 1e-12 per ray (as before).
  - Continuity (box and bed, same 0.01 nats, no new constant; R9-C1 replaces R8-I5's centred-halving rule, which a
    jump escaped once h/2 fell below its distance from the sweep point): with D(x, h) = E(x + h) − 2E(x) + E(x − h),
    a step passes if |D(x, h)| < 0.01 nats; otherwise it is replaced by the two half-size stencils that cover
    [x − h, x + h] (centres x ± h/2, half-width h/2), each tested the same way, recursively; it fails if any stencil
    still has |D| ≥ 0.01 at half-width 1e-9 m. A smooth E falls 4× per level, a kink 2×, a jump stays at |J| in the
    sub-stencil that holds it.
  - Gradient: on the box as rev 3.3.3, and a miss at h = 0.1 mm is re-checked at h = 1e-7 m (R9-M4); on the bed at
    h = 1e-7 m (1 %), with the one-sided rule at stated kinks. It
    cannot see a jump in E, so Continuity stays a CHECK beside it.
  - Sensor inside a part (the atom): sensor 20 cm inside, ρ = 1 m, m = 0.9 ⇒ E within 0.01 nats of the independent
    reference (Fable: 11.9 nats); sensor 2σ_s outside the surface likewise.
  - Face minima (measured, K = 64 δ-rule, converged face-on): e = +10 µm at m = 0.9, +1.70 mm at m = 0.05 (rev 3.3.3's
    7 µm / 1.3 mm were estimates).
- **Cost and constants (R8-M4, R9-M5).** Per ray near the model: the breakpoints (closed form), up to ~2L + a few
  subintervals × n nodes (up to about 84 nodes per panel at L = 11, n = 3), one SDF evaluation and one Jacobian each,
  plus K_m crossings and Jacobians for the moments. (c)'s estimate no longer stands; only the Cost item's measurement
  against the same 5 ms median counts. Numerical constants of the section, none a model
  threshold: z_max and L (spacing 0.6), n, K_m, the root tolerance (machine precision), the 8σ_s floor (ray and part),
  and the test-side h = 1e-7 m, the 1e-9 m halving limit and the n ≤ 4 / z_max ≤ 5 reporting limit. ε of the tangents and the
  1e-4 m bracket are withdrawn.

**Amendment 11-21 (rev 3.3.7; owner rulings after the Task 19 review): known room surfaces, the λ rule, the cost
rule, joint damping.** The Task 19 review found no implementation defect behind its 20 Pass misses; they are model and
plan matters, settled here.

- **Known room surfaces in the foreign density (owner: "Floor as known surface").** The review traced the width bias to
  the floor: on the ring meeting the front face 4 mm above it, floor returns lie 1.5 cm behind the face plane, inside
  b* ≈ 3 cm, so the model is paid to cover them (w optimum +6–8 mm; an independent integration without the floor gives
  0.0 mm). This is the "rest surfaces" answer the foreign-density paragraph named and left out. Per staged ray, t_r is
  its first crossing, in (0, ρ_max], with the room's known surfaces: the floor plane z = 0 of the room frame and the
  walls of the room polygon the belief already reads for (2c) (walls by the same mechanism is the controller's call —
  the over-deep fixture's 7.6–9 cm through the wall is the same bias; owner may strike it). The foreign density
  becomes, per ray,

      p_f(ρ) = p_pass·𝟙[ρ < t_r] + (1 − p_pass·t_r)·St(ρ − t_r)          (no crossing before ρ_max: p_f = p_pass)

  i.e. unmodelled clutter stays uniform at today's 1/ρ_max in front of the room surface, and the mass that went beyond
  it now lands on it, smeared by the same Student-t. No new parameter, and it is a density on [0, ∞) by construction.
  p_f,i = p_f(ρ_i) does not depend on θ (the room surfaces are fixed during the update), so it replaces p_pass in every
  formula of (a), (b) and the rev 3.3.6 block as a per-ray constant — p, r, o, the energy normaliser (1 − m)p_f,i,
  the atom term, the gradient — with no new derivative. The flush depth b* is then measured against p_f: near a room
  surface it shrinks to the scale where St(b*)/St(0) ≈ ε_px/(1 − ε_px). Pass: the silhouette fixture's w optimum with
  the floor present within 1 mm of truth (the review's per-ring probe as a test); creep 0/16 with Δw ≤ 2 mm; the
  over-deep fixture's measured face no deeper than the wall's own σ behind it; mass and reduction items unchanged
  (fixtures without room surfaces reduce to rev 3.3.6 exactly, bit-exact).
- **The λ rule and the cap (owner: "Nielsen λ rule + cap 25").** (d)'s ×4 / ÷4 rule is replaced by Nielsen's:
  gain ratio ϱ = (J_k − J_{k+1}) / (J_k − J_model(δ)), with J_model the quadratic model of the solve; accept when
  ϱ > 0: λ ← λ·max(⅓, 1 − (2ϱ − 1)³), ν ← 2; reject: λ ← λ·ν, ν ← 2ν (first λ = 10⁻³, ν = 2). `gn_iters_max` = 25.
  Solver constants, not model thresholds. Pass: the first-update fixture with rays returns `gn_conv` = 1 within 25 and
  θ equals the 200-step run (1 mm / 1 mrad); seeds settle within the same tolerance.
- **The cost rule (owner: "Bound = agent rate").** The 5 ms median of the Cost item cannot hold at this per-ray cost
  (2.4–3.2 ms per evaluation per 256 rays; live ≈ 1000 rays). It is withdrawn: the unit test PRINTS the ms (first
  update and parked, box and bed) and fails nothing; the binding rule is live11's fit rate ≥ 8 Hz per class.
- **Damping on the joint state (owner: "Ratify, write into spec").** (d)'s Levenberg term acts on the whole joint
  state x = (θ, b): M = diag(W) over θ and the latents (a zero-variance latent still keeps δ = 0), with the consistent
  linearisation θ_k + B·b_k of (d)'s caveat. On the beam energy the θ-only term rejected every step; the review found
  the joint form sound (same fixed point). It applies to every flag-on update, camera-only included.
- **Pass protocol.** Every Pass bound of this section is a CHECK that fails ctest when missed. "Printed" is reserved
  for items this spec itself marks printed (and the Cost ms above). A miss goes to the owner as a failing test, never
  as a silenced print.

**Amendment 11-21b (rev 3.3.8; owner rulings after Task 19 fix rounds 1–2).** Rounds 1–2 fixed the code defects
(square-box gradient, p_f subsets, undamped stop); what remained is settled here.

- **Convergence in posterior σ (owner: "Tolerance in posterior σ").** (d)'s stop tolerance (1 mm / 1 mrad) becomes
  0.1·σ_k per component, σ_k = √P_kk of the posterior at the current linearisation (P = (𝟙 + P̄W)⁻¹P̄, one solve):
  the update stops when the undamped (λ = 0) step is below 0.1σ_k on every θ component (reading B of round 2: applied
  if J does not rise). "θ equals the 200-step run" becomes "within 0.1σ per component". Rationale: the first update's
  residual lay along a direction with 0.03 nats over 4 mm against σ_cx ≈ 4 cm — information the data does not hold.
- **Bounds against ray sampling (owner: "Bound in ray spacing").** Helios rays are ≈ 1 cm apart at the fixtures'
  ranges, comparable to σ_s, and identical sweeps lock one sampling phase (a parked robot does so live too). The creep
  bound on Δw/Δh and the silhouette w-optimum bound become ½ the ray spacing at the object along that axis
  (s = range × the SensorModel's azimuth step for w, ring spacing for h), printed beside the measured value.
- **Known surfaces with their own σ (owner: "Walls with their σ").** Each known surface carries its published σ:
  the walls the room polygon's (the layout σ room_concept publishes), the floor `Model.FloorSigmaM` (no default when
  the flag is on; the configs carry 0.005 m, the controller's call from the review's 5 mm floor-error scale). Along a
  ray the surface σ projects as σ_r,i = σ_surface/|n̂·u| at t_r, and the smear St(ρ − t_r) takes scale² + σ_r,i² at the
  same ν (moment match of the scale; ν stays p₀c² − 1). Still θ-independent per ray. Pass adds the flush-wall bchain
  fixture: one basin (the lever-only control within its bound).
- **Miss mass with p_f (spec gap S1, controller's definition).** In I_d the foreign return's probability of lying
  beyond t̄, (1 − t̄/ρ_max), becomes the p_f tail Q_f(t̄) = 1 − p_pass·min(t̄, t_r) − (1 − p_pass·t_r)·G_r(t̄ − t_r),
  G_r the CDF of the smeared St; with no room crossing it is 1 − t̄/ρ_max exactly.
- **Stated limits (owner: "Accept both").** (1) Reduction yaw: at n̂·u = 0.25 the beam's θ differs from 7c.7's by
  0.28σ in yaw (the convolved density's normaliser depends on n̂·u; 7c.6's factor has no such term); bound 0.3σ.
  (2) Floor contact: a floor-standing box's face returns on the ring just above the floor have π ≈ 0.54 (the box's
  own blurred bottom edge), so that ring's w optimum is −4.4 mm; bound 5 mm on that ring.
- **Tie rule (round-2 re-review N1).** At an exact (g, t) tie among envelope pairs, approach() keeps the pair whose
  multipliers bracket the vertex (β_L < 0 < β_R), never one chosen by ulp-level differences; a lattice-ray fuzz
  against a square yaw-0 box is a CHECK (0 gradient mismatches outside the subdifferential).

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
