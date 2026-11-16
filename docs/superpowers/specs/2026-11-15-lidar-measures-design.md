# Camera attributes, LiDAR measures: the LiDAR updates the countable belief on gated mask cycles

Date: 2026-11-15. Status: APPROVED by the owner 11-15 ("Approved, after lidar": the cabinet exit-release fix comes after this); §7b and §8
amended 11-15 after Fable's plan review (owner: C1 = A, C2 = two-sided); §7c added 11-16 (owner: "apply active
inference rigorously" — yes to the mixture with the camera prior, yes to LiDAR-first); §7c rev 3 11-16 (owner: class
prior counted once + Σ bounded by it (B); camera persistence now with the object-size scale). Tasks 1–2 built and
merged to main 8f4cc821; §7c rev 3.1 folds Fable's rev 3 review (READY WITH CHANGES).
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
  `prior_added` flag in the belief) adds the RELATIONAL anchors once — wall flush/parallel (1) and the room axis —
  in a θ-only, unsaturated channel; the footprint (2) and height (2a) anchors are not added (Σ₀ already carries the
  class, R3-I2). The flush responsibility is evaluated at that pose only (stated: an object born mid-room that later
  settles on a wall never gets the term). `sanitize_covariance` re-seeds Σ to P and clears `prior_added` (the prior
  re-enters once); the divergence restore copies it; an LTOM/LTSM recall restores a belief with the flag set.
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
  and WITHOUT the ego-motion terms: config floors + range lateral/size + obliquity yaw + moment variance.
- b_L ∈ ℝ⁶: the LiDAR rays' sensor-specific shared error. Marginal = the config floors (§7b, without chain and
  ego-motion).
- The ego-motion terms (`mot_pos_var`, `mot_size_var`, `mot_yaw_var`) are fresh every frame by construction: they
  stay in the per-frame Woodbury of their channel (ρ = 0), outside every b.
- Camera points measure θ + E b_chain + b_cam; rays measure θ + E b_chain + b_L (E embeds (cx, cy) into ℝ⁶).
  b_chain, b_cam, b_L are a priori independent of each other and of θ (state x ∈ ℝ²⁰).
- The config floors are in b_cam AND b_L as independent errors (R3-I4, my call): whatever part of a floor is model
  mismatch common to both sensors is then counted twice when parked (position information 2/0.03² instead of
  1/0.03², σ low by √2 at the floor). Stated; the two-sided live check sees it.

**Dynamics.** Per latent and per DOF: b_t = ρ_t D_t b_{t−1} + w_t, D_t = σ_t / σ_{t−1} (one source per latent, so the
rescale is physical: a deprojection bias scales with its σ), w_t ~ N(0, (1 − ρ_t²) σ_t²). Marginal exactly σ_t²,
lag-1 correlation exactly ρ_t; D = 1 when σ_{t−1} = 0. ρ = 0 for all three is today's Woodbury: the regression test. The θ–b and b–b cross
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
