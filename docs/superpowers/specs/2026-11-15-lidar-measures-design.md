# Camera attributes, LiDAR measures: the LiDAR updates the countable belief on gated mask cycles

Date: 2026-11-15. Status: APPROVED by the owner 11-15 ("Approved, after lidar": the cabinet exit-release fix comes after this); §7b and §8
amended 11-15 after Fable's plan review (owner: C1 = A, C2 = two-sided); §7c added 11-16 (owner: "apply active
inference rigorously" — yes to the mixture with the camera prior, yes to LiDAR-first); §7c rev 2 11-16 (owner: OU
stationary class prior yes; camera persistence now, C4 = A). Tasks 1–2 built; §7c rev 2 awaits Fable's review.
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

## 7c. Three corrected assumptions (rev 2, 11-16; owner: "can't we apply active inference rigorously?")

Owner rulings 11-16: (1) the class prior becomes a stationary (Ornstein–Uhlenbeck) prior on every countable frame,
behind `Model.LidarMeasuresOnGated`; (2) the camera gets the same persistent shared error as the LiDAR now (Fable
C4 = A); (3) the camera's m is the ownership prior of a mixture (not the camera-free explaining-away mixture of §1).
This section supersedes G3, the §7b bullet "m and the down-weights", and rev 1 of §7c (2e5c9213), whose review is
`.superpowers/sdd/2026-11-15-lidar-measures/fable-7c-review.md` (cited below as C1–C4, I1–I11, M1–M6).

### 7c.0 The mechanism (measured 11-16, Task 2 fixture, per-ray diagnostic)

The back-face pull in `wall_behind` (back face −5.0 → −1.3 cm, h 0.51 → 0.546 over 100 sweeps; box rays alone
−4.9 cm) is not one strong ray. From a front view the direction g = (Δcy, Δh) = (δ/2, δ) — back face moves, front
face fixed — has no data. With the class anchors removed from the LiDAR-only frame (C1 = A) and Q added on every
predict, σ along g grows without bound (σ_h 6.0 → 7.5 cm in 100 frames), and the sum of the tiny wall-ray forces
(weights 0.001, ≈ 1e-5 per frame each) walks the mean along g as far as Σ lets it. Fable's hand trace (I7) found no
single strong ray because there is none. Two faults in the generative model make it possible: the class prior is
used as a per-frame measurement instead of a prior (7c.1), and every frame's shared error is assumed new (7c.2).
The attribution of an individual return (7c.3) is a third, separate fault.

### 7c.1 The class prior is a stationary prior, not a per-frame measurement

**What the model said.** The class anchors — footprint (w, h ~ N(prior_footprint_m / depth_mean, prior_footprint_std²)),
height (H ~ N(prior_height_m, prior_height_std²)), the depth-unobserved boost (`depth_unobs_precision`), the room-axis
alignment and the wall flush/parallel term — are added to Id on EVERY camera frame, while `predict` adds Q. That is a
filter told "a fresh measurement of the class mean arrives every frame". Its fixed point along a direction no datum
sees is σ = (Q/λ)^¼ (nightstand depth: (0.005² · 0.06²)^¼ = 1.7 cm), three times tighter than the class's own 6 cm, and
the mean is held at the class mean by 700 fake measurements per run. On a LiDAR-only frame the anchors are rightly
left out (C1 = A), so the same direction then has NO prior and diffuses: 7c.0.

**What is true.** The class prior is p₀(θ): where an object of this class is BEFORE any measurement, and where an
unobserved object's belief returns to as its evidence ages. In a state-space model that is a property of the
TRANSITION, not of the likelihood: the static object's state follows a mean-reverting (Ornstein–Uhlenbeck) process
whose stationary distribution is p₀. Per predict step with process variance Q_i (the existing `process_noise_diag`,
including the inferred volatility exp(ω) and the `q_scale` of an aged step) and class variance P_i:

  a_i = exp(−Q_i / (2 P_i))
  θ_i ← μ_i + a_i (θ_i − μ_i),   Σ ← A Σ A + diag(P_i (1 − a_i²)),   A = diag(a)

Stationary marginal exactly N(μ_i, P_i), for any Q and any step size (the exponential form composes; no clamp, no
max(0,·)). For Q ≪ P the injected variance P(1 − a²) ≈ Q, so a measured direction behaves as today. No new constant:
μ, P from the class config, Q from the existing process noise. Time constants (nightstand, Q = 0.005² per frame,
10 Hz): w, h 288 frames ≈ 29 s; H 800 frames ≈ 80 s; yaw toward the room axis (3°, Q = 0.01²) 55 frames ≈ 5.5 s.

**Which prior terms move into the transition.**
- Size (H, w, h): the class Gaussians above. μ = (prior_height_m, prior_footprint_m, depth_mean()), P = std².
- Yaw: the room-axis term is a 4-fold periodic prior; locally it is N(nearest room-axis yaw, room_axis_std²). μ_yaw =
  the room-axis multiple of 90° nearest the current yaw (recomputed at each predict; it never decides front/back —
  the same property as today's ¼·sin 4Δ). No room axis staged ⇒ P_yaw = prior_yaw_std² about the current yaw, i.e.
  free (a = exp(−Q/(2P)) ≈ 1 − 1e-4).
- Position along the wall normal (the flush term): the furniture prior "stands against a wall with probability
  π_flush" is a two-component prior on the back-face gap d to the staged wall: flush N(0, σ_wall²) with weight
  π_flush, free otherwise. Its OU reversion acts on d with weight = the current flush posterior w_flush (already
  computed, `flush_posterior`): μ_d = 0, P_d = σ_wall² / w_flush (w_flush → 0 ⇒ P → ∞ ⇒ a → 1, no pull; continuous).
  The parallel part acts on yaw exactly like the room axis, toward the wall's direction.
- Position along the wall and in a free room: the class has no prior mean there (prior_pos_std = 0.30 m is birth
  slack). μ = the current mean, P = prior_pos_std² ⇒ Σ bounded at 30 cm instead of a free walk; the mean does not move.
- `depth_unobs_precision` (20 000 /m² on the nightstand) is NOT carried over. Its job — "the footprint prior must
  prevail absent two-face evidence" — is what the stationary prior does by construction: an unseen extent stays at
  N(depth_mean, 6 cm²) for ever. OPEN for review: the boost was also fighting the one-face points dragging the extent
  outward (a likelihood bias: the size common mode caps Σ, not the mean). With 7c.2's persistent b_cam that drag is
  bounded by one look's worth; whether that suffices is checked by `one_face_keeps_depth` (§8) with the boost gone.
- The one-sided constraints (wall no-cross 2c, short-height 2b) are not priors; they stay in the likelihood of every
  frame. The camera-data terms (z_top_obs / top-no-float, moments, per-point azimuth) are data and stay.

**Consequence for the LiDAR-only frame.** The C1 = A frame keeps its exact content; it is now consistent because the
prior is in the predict that every frame runs. Along g (7c.0) the restoring force of the stationary prior bounds the
walk: a constant per-frame gradient f (1/m) along a direction with OU coefficient a and steady variance Σ_ss moves the
mean by Σ_ss·f per frame and reverts it by (1 − a)·Δ, so it settles at Δ = Σ_ss·f/(1 − a) ≈ 2 P² f / Q when nothing
else sees that direction (Σ_ss → P). Today's LiDAR-only walk is the a = 1 case: no fixed point. 7c.5 pins the number.

**Consequence for camera frames.** They change with the flag on: the anchors leave Id and enter the predict. An
unseen extent is honestly ±6 cm, not ±1.7 cm; existence and removal scores that marginalise Σ see the honest value.
The flag-off live run (objects_live8off) is the A/B; the golden 40-frame trajectory pins flag-off bit-exact.

**Existing engine mechanism.** `clamp_to_prior` (recursive_laplace.h, table/chair) bounds Σ by a congruence but does
not move the mean and uses `prior_cov_diag` (the birth prior, not the class prior). The OU predict replaces it for the
countable belief; table/chair are untouched (§4).

### 7c.2 Every channel's shared error persists across frames (supersedes G3; Fable C1, C4, I1–I5, I10)

**What the model said.** Each frame's shared error (pose chain, extrinsic, mask boundary, which surface points the
beams sample, the deprojection bias) is drawn afresh: the per-frame Woodbury `saturate`. **What is true.** The world is
static; while the viewpoint does not change, frame t sees the same surface, through the same pose, with the same
mask bias as frame t−1. 100 parked frames are about one look.

**Model.** Two latents carried in the belief beside θ: b_cam (camera points' shared error) and b_L (LiDAR rays'),
each ∈ ℝ⁶, prior N(0, Σc_ch,t) (Σc_cam = today's camera common mode; Σc_L = §7b's LiDAR-only common mode: floors +
pose chain + ego-motion). Camera points measure θ + b_cam, rays measure θ + b_L. Dynamics per channel and per DOF
(Fable I3: Σc changes every frame):

  b_t = ρ_t D_t b_{t−1} + w_t,  D_t = Σc_t^{½} Σc_{t−1}^{−½} (diagonal),  w_t ~ N(0, (1 − ρ_t²) Σc_t)

Marginal exactly Σc_t, lag-1 correlation exactly ρ_t. ρ = 0 is today's Woodbury: the regression test.

**ρ (Fable I1, I2).** The Markov kernel in path length, which composes for any sweep rate: ρ_t = exp(−Δ_t / ℓ), per
DOF, Δ_t the viewpoint change of the channel's sensor relative to the object since the channel's last update of THIS
instance (path lengths add across predict-only and aged cycles).
- LiDAR: lateral (cx, cy, w, h) and yaw from the tangential path, ℓ = ring_gap·z; H from the radial path,
  ℓ = ring_gap·z / |tan ε| (ε the elevation of the ring hitting the object top). ring_gap = `max_ring_gap_rad()` of the
  Helios `SensorModel`.
- Camera: ℓ = q·z / f (the mask's pixel footprint at the object), all DOFs from the tangential + radial path.
- Stated limits: the footprint is the FASTEST-decorrelating source; model mismatch and neighbour returns decorrelate
  with aspect (tens of degrees), so ρ is optimistic while moving and exact while parked (the case that matters, C4).
  A path that returns to the same viewpoint counts as new.
- Pose-chain part: ρ_chain = 1 within a localisation session (the localiser publishes no cross-time covariance and
  the odometry increment is not read by this agent); a relocalisation event resets it to 0. Combined per DOF:
  ρ_i = (ρ_f Σf_i + ρ_chain Σchain_i)/(Σf_i + Σchain_i), exact for the lag-1 correlation of a sum of two independent
  processes.
- SensorModel field absent ⇒ ρ = 1 for that channel (conservative: no new information until the viewpoint is known);
  logged once.

**Update (Fable I4, I5).** One joint GN per cycle on x = [θ; b_cam; b_L], one predict (θ by 7c.1, each b by its
Gauss–Markov step, cross-blocks multiplied by A·ρ). Per iterate: points linearised at θ_k + b̄_cam, rays traced at
θ_k + b̄_L (b̄ = the PREDICTED bias, held over the iterates); each channel's (Id, bd) enters through `saturate` with
S_ch = the predicted covariance of θ + b_ch (Pθθ + Pθb + Pθbᵀ + Pbb), then x ← x̄ + P Hᵀ beff, P ← P − P Hᵀ Ieff H P.
Only 6×6 inverses and the existing stable operator; S in double. At ρ = 0 it is the matrix-inversion-lemma form of
today's update. A frame without camera points updates only through the rays (the LiDAR-only frame); a frame without
rays only through the points.

**Consequences, by construction.**
- Σ: a repeated look cannot add more than Σc_ch⁻¹ per channel; the posterior of θ in a direction one channel alone sees
  stays ≥ P Σc/(P + Σc). Fable C1 is right that this bounds Σ, not the mean: the mean of a direction the data does not
  hold is held by 7c.1's stationary prior, and a wrongly attributed return is refused by 7c.3.
- Parked: camera and LiDAR are each one look; the LiDAR's share of H is no longer ~11 % (C4) but set by the two
  channels' Σc. Moving around an object earns information on both — the active-inference reading of "look from
  elsewhere".
- Life of b (I10): `sanitize_covariance` resets b and its blocks; `canonicalize` / `apply_mode_rotation` permute b and
  Pθb with θ; the divergence snapshot copies them; the published Σ is the θ marginal; LTSM restore does not store b, so
  the first look after a restart counts as new (stated). ω is updated from camera-measured cycles only (§7b).
- Log per cycle: ρ_cam, ρ_L, b̂_cam, b̂_L, √diag Pbb per channel, Σm, Σr, ray count, trace of each channel's Ieff.

### 7c.3 The camera's m is a prior on ownership (supersedes "m and the down-weights"; Fable C2, C3, I6, I8, I9, I11)

The camera says p(return i is the object) = m_i; the return then comes from the box or from the rest of the scene,
and the room model knows its walls and floor. Per selected return (range r_i, M3: r for range, ρ only for correlation):

  p(r_i) = m_i · N(r_i; t_i, σ_r² + J_i S_L J_iᵀ) + (1 − m_i) · p(r_i | rest)
  p(r_i | rest) = (1 − w_u) · N(r_i; t_room,i, σ_r² + σ_room²) + w_u / L_i     (the room predicts a hit inside the box)
               = 1 / L_i                                                     (it predicts none; M1)

- Box component (C3): the PDA innovation variance, σ_r² + J_i S_L J_iᵀ, J_i the range Jacobian the factor already
  computes, S_L the predicted covariance of θ + b_L. No Cauchy factor on the mixture path (the mixture IS the outlier
  model); every other caller of `accumulate_lidar_rays` unchanged. Capture: a true return m = 0.9 with the belief
  14 cm off keeps r = 0.99 at 5 cm σ along the ray (Fable C3). Price (stated): an uncertain box refuses the wall less
  firmly (wall 5 cm behind, box σ 4 cm, m = 0.1 ⇒ r = 0.045).
- Unexplained component (C2): uniform over the ray's chord through the SELECTION box, 1/L_i (≈ 1.67 /m), because the
  returns were selected inside it. Limit (stated, inherent to any range mixture): a foreign surface within σ of the
  predicted face cannot be told from the face (neighbour at e = 2 cm, m = 0.1 ⇒ r = 0.45; e = 5 cm ⇒ 0.055).
  Neighbour boxes published by the other countable agents as further "rest" components: later work, owner to accept
  (named, not built).
- m (I8): m_i = π_mask · Φ(s_i / σ_px), π_mask the slice's published detector `confidence` (`MaskIngestor` slice).
  Limit: confidence is not a calibrated pixel precision; inside an over-segmented mask m stays high (Fable: 0.98 for a
  wall return 3σ_px inside), and there 7c.1/7c.2 bound the damage. Retina-side mask precision: later work.
- First-hit physics (I9): kept out. A return beyond the box's predicted hit gets r ≈ 0 and no term; an over-deep box is
  corrected by camera frames and by the stationary prior, not by LiDAR-only frames. The full model (range = first hit
  of box ∪ room) is later work.
- Sources, one each (I11): σ_r = the Helios `SensorModel` range σ (absent ⇒ `ExistenceSensorSigmaM`), consumer
  constants in quadrature as `sensor_model.h` says; σ_room = `wall_explain_sigma_m` (the camera-point wall component's
  σ, shared: 7c.3 is its LiDAR counterpart); w_u = 0.2 (the fleet beam model's form, `lidar_ray_cost`); t_room,i = first
  hit of the ray on the room polygon's vertical faces (extruded to the wall height the room publishes, else unbounded)
  and on z = 0, a small new `common/` function on the compute thread; L_i from the selection box.
- Interface (I6): `LidarRays::weights` keeps the prior m (Task 1). New per-ray θ-independent inputs: t_room,i (or
  none), L_i; the factor computes r_i per GN iterate from the current trace. Coverage uses Σ r at the predicted state,
  computed once at the first accumulation and frozen for the update. Empty inputs ⇒ today's behaviour bit-exact for
  table/chair/bottle/door. The ray-count down-weight is kept but counts Σ r (one face is still one look; 7c.2 covers
  across frames, the count within one).
- The complete-data information r·JJᵀ/σ² is slightly optimistic where r is ambiguous (M2); the same practice as the
  camera-point wall mixture; noted.

### 7c.4 What stays from rev 1 / §7b

The LiDAR-only frame's content (C1 = A), the selection anchor on the mask-cloud centroid, membership G2 (σ_px with
mot_pos_var; the mask-lag term is in σ_px only, not in Σc_L — I3), the divergence net, ω from camera cycles only.

### 7c.5 Pass additions (replace rev 1's 7c.3; Fable I7)
- Unit, 7c.1: an unobserved belief (no data, 5000 predicts) is N(μ_class, P_class) within 2 % in mean and σ; an
  observed direction behaves as the flag-off predict within 1 % over 40 frames; a constant per-frame force along g
  settles at the analytic OU offset (pinned).
- Unit, 7c.2: ρ = 0 reproduces today's per-frame Woodbury (tolerance set after measuring it in float32 with S in
  double); ρ = 1 and n identical sweeps equal ONE sweep with n·Id and the same predicts (analytic); N small steps and one
  step of the same path length give the same Σ within 5 %; the chain-ρ combination formula; reset/permute of b under
  sanitize and mode rotation.
- Unit, 7c.2 alone (weights = m, no mixture): the analytic pull β·G/(G + 1/P) of C1 — a number, not a smallness pass.
- Unit, 7c.3: responsibilities at the worked numbers above (C2, C3) within 10 %; `recovers_from_14cm`: a true box
  offset 14 cm, m = 0.9, LiDAR-only sweeps bring the error below 3 cm; `neighbour_beside_does_not_pull`: an
  unmodelled surface 5 cm beside the face at m = 0.1, 2000 sweeps, pull below a pinned bound.
- Unit, whole: `wall_behind_does_not_pull` started with the back face 2 cm from the wall, m = 0.1 (the live value,
  f ≈ 600 px, σ_px ≈ 12 px at 2 m), 2000 sweeps, back-face gap within 0.5 cm of the box-rays-only baseline.
- Unit: `one_face_keeps_depth` with the depth boost gone (7c.1 OPEN item).
- Live: §8 unchanged (two-sided Σ, ≤ 10 cm drift, H closer to the true top than live6, ~10 Hz).

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
