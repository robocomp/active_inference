# Shared estimation engine: part 1 (design)

Branch `feat/shared-engine` from main 752e93a8. Baseline count of constants: `docs/superpowers/results/2026-11-23-patch-constants-baseline.md` (5823d634).
Stop date 2026-12-21. Part 1 is due to land 7–10 Dec. If it is going to slip past 10 Dec, tell the owner early; the fallback is to submit the current paper.

## Why

The AuRo paper measured over-confident object beliefs at first visit (NEES/DOF):

| class | NEES/DOF |
|---|---|
| toilet | 44 |
| hood | 30 |
| nightstand | 5.7 |
| table | 2.3 |

It also measured offers that promise more than they deliver:

| object | promised (nats) | delivered (nats) |
|---|---|---|
| chairs | 3.19 | about 1.9 |
| refrigerator | 6.0 | 0.2 |

The diagnosis is that each frame is treated as independent. The error a frame carries is shared by every frame taken from the same standpoint: mask under-segmentation, the pose chain, and the shape prior's misfit. So σ shrinks like 1/√frames, not with the number of standpoints.

Each class file then patched the symptoms. The baseline counts 91–104 patch constants per class.

The aim is that once the sensor is modelled, a class file holds only class knowledge. The sensor model covers:
- free space and visibility;
- mask error;
- shared error that persists within a view and decays as the view changes.

## Owner's decisions (11-23)

- **Engine (approach A).** The countable belief's persistent update becomes the shared engine:
  - `common/ai_belief/persistent_update.h`, the beam model `beam_ray.h` / `beam_room.h`, and the viewpoint decorrelation now in `countable_concept/src/countable_viewpoint.*`.
  - Chair, table, refrigerator and hood keep their own shape models and run on this engine.
  - Bottle, door, cabinet, residual and ring stay on `recursive_laplace.h`, unchanged.
- **Masks:** YOLO-seg as it is. SAM is out of scope.
- **Order:** part 1, then part 3 (online noise and envelope learning), then grading. Part 2 (part-level shapes for chair, table and fridge in the countable vocabulary) is decided after grading.
- **Grading:** part 1 is graded with part 3 off.
- **Bed width:** handled in part 1. The graded quantity is named per class and graded against the matching mesh (see "Graded quantities").
- **Occupancy check:** each offered standpoint is checked against the occupancy belief before the offer is published.
- **Parked test first:** a parked test with one standpoint against two runs before any tour reruns.

## Why part 1 needs the engine move (and why it is not part 2)

The owner chose approach A in the objects session on 11-23. The paper session asked that the design restate it for approval with the design.

- **Why part 1 needs it.** The per-view mechanism already exists and is tested, but only in the countable belief:
  - the shared-error latents carried between updates (`persistent_update.h`);
  - the continuous viewpoint ρ (`countable_viewpoint.*`);
  - the beam model that gives free space (`beam_ray.h`);
  - fixes A and B.

  `recursive_laplace::update` is stateless per frame, so it cannot hold a latent between frames. Adding one there (approach B) means building the same latents, ρ, free space and mask model a second time. The per-class count would then be split across two engines.
- **It brings no parts.** Each class keeps its own state θ, its own SDF primitives and its own template. Only the update step, the latents and the offer computation change. Part-level shapes in the countable vocabulary (approach faces, fridge door) stay in part 2, which the owner decides after grading.
- **C3 non-regression.** Each class gets a config switch, old engine against new, until grading. The C3 tours (moved and hidden objects) are rerun on the new engine in the combined batch. A C3 regression is reported against the v1 numbers, not tuned away.
- **The date.** The ports take about 6 of the 14 working days: fridge and hood are mechanical (same θ as the countable belief), the chair takes 1–2 days, and the table, with its quotient chart, 2–3. They are the main risk to 7–10 Dec.
  - If the table port slips, the table stays on the old engine. Its test (a) is then reported as "old engine", and the slip is told to the owner at once.

## Goals

1. **G1. Shared error per view, not per frame.** Within one standpoint σ must stop shrinking. A new standpoint adds information in proportion to how far the view has decorrelated: ρ from `countable_viewpoint`, a continuous function of the camera path with no gate. The marginal variance of each channel's shared-error latent is the class's existing common-mode Σc (`common_mode_inv_diag` / `common_mode_inv`), so no new constant comes in.
2. **G2. Free space and unobserved faces from the beam model.** A ray that passes through where a face would be is evidence. A face no ray or mask point has seen is held by the class shape prior, not by an extra precision. This replaces the depth-unobserved precisions, the coverage gains and the wall-flush/no-cross family wherever they stand in for a hidden face. Room walls are already in the foreign density (`beam_room.h`).
3. **G3. Mask error model for YOLO-seg.** Two parts:
   - The boundary displacement shared within a view joins the camera latent. Its σ_mask is a sensor property, measured once from Webots mask dumps against projected ground truth. The `feat/mask-dump` branch has the dump. The value is per detector and, if the data shows it, per class. Part 3 later learns it online.
   - Leakage from a neighbour's mask is handled by a per-point foreign-membership mixture whose foreign density includes the other objects' beliefs, not only walls and floor. This was candidate 2 in the 11-22 analysis.
4. **G4. Shape misfit as a persistent latent.** A single box (or the chair template) cannot say which layer of a real object it means: the bed is 1.57 m at the mattress and 1.73–1.77 m at the frame and blanket. Each class manifest gets a per-DOF shape-spread σ (class knowledge), carried as a latent that never decorrelates with the view (ρ = 1). Its value comes from the class's part spread: (Δ)²/12 for a uniform spread over the layer range Δ.
5. **G5. Offers under the same cap.** The epistemic gain of a candidate standpoint is the expected entropy reduction under the engine's own joint covariance update (`joint_cov`), with the latents predicted for that standpoint's ρ. So a return to a seen view promises little and a new view promises a lot. The chair and table self_test face-ranking checks must keep passing. Each offer logs promised and delivered nats per visit (AuRo test b).
6. **G6. Occupancy check on offered standpoints.** Before publishing, the standpoint's robot footprint is scored against the occupancy belief, and the offer's value is multiplied by P(stand and turn): the probability that every cell the body covers through the turn in place it ends with is free. It comes from ONE shared standpoint oracle (`common/standpoint/standpoint_oracle.h`, K3), the same code the controller decides with. (Amended by K3, 2026-11-26; see the amendment at the end.)
   - Step 0 reads the AuRo claim logs to confirm why half the no-update claims are refused: the auro report says the view lies under the robot's own body.
   - If the cause is geometric rather than occupancy (a standpoint inside the robot's current footprint), the check is that geometry, stated in the spec before coding.

## Non-goals

- Part-level shapes (part 2), learning noise online (part 3), SAM masks.
- Existence and plausibility patches (FeRef / FeScale, plausibility-to-existence gains). They are counted and reported, not removed in part 1.
- The room's 0.35 scale. It is counted under test (c) and is expected to fail; that is accepted.
- Agents outside the acceptance tests.

## Anti-goals (a failure even if the code works)

- A new constant per class to make a test pass. No bound may be fitted (memory feedback-no-fitted-bounds); a failing test is reported as failing.
- A view-change threshold. ρ stays continuous.
- Losing the ability to rebuild arXiv v1. v1 was built from main 8a133295; keep the old paths selectable by config until grading is done.
- A drop in agent rates below nominal (rates before features).

## Registered prediction for test (c): rows each item should remove

Row names are from the baseline (5823d634 plus addendum 1), where each is itemised per class. A row is removed only when the item that replaces it is in, and its unit test passes. Whatever stays is reported as staying.

| item | baseline rows it should remove (where present in a class) |
|---|---|
| G1 shared error per view | `AI2CommonModeYawStd/PosStd/SizeStd` become the latent's marginal Σc: kept as sensor rows, no longer patch. `AI2RangeNoiseYawPerM`, `AI2RangeNoiseSizePerM`, `MotionCmYawGain/PosGain/SizeGain`, `AI2PeriphRef`, `AI2AngLeverM`, `AI2MotionRefMps`, the `Fixation*` gates, `FixationPeriphFloor`, `FixationCentrePrecision`, `AI2MotionConfirmOnly`, `AI2StillLinMps/AngRadps/Dotd`, `AI2MovingUpdateCenterRadius`, `kFlipClamp`, `kFrontClamp`, `ShapeEvidenceClamp`, `AI2ViewBudget` / `front_view_budget_`, `AI2ModeSatBackPts`, `ShapeEvidenceEmaAlpha` |
| G2 free space and unobserved faces | `AI2DepthUnobsPrecision`, `AI2DepthObsBandM`, `FreeSpacePrecision`, `CoveragePrecision`, `CoverageRobustCM`, `LidarCoverageAngPower`, `LidarCoverageN0`, `AI2Coverage*Gain`, `AI2CoverageMin`, `FootprintMomentCompletenessGain`, `FootprintMomentMinCompleteness`, `WallNoCrossPrecision/MarginM`, `AI2WallExplainFrac/SigmaM`. Expected to become sensor or class rows rather than patch: `AI2WallPrecision`, `AI2WallParallelPrecision`, `AI2WallReachM` (whether they remain at all, given `WallFlushPrior`, is the referee's call). |
| G3 mask error | `AI2TopNoFloatPrecision`, `AI2TopNoFloatMarginM`, `AI2TopOversegSigmaPerM`, `AI2ClutterFrac`, `AI2ClutterScaleM`, `AI2ClutterStructureGain`, `AI2SigmaBaseM` (its mask-edge part goes to the measured σ_mask, a sensor row), `SdfThresholdForStorage`; `LidarPrecision` used as a weight becomes the sensor's σ |
| G4 shape misfit | `ObliquityMomentGain`, `kObliquityYawGain` and its cos clamp, `FootprintMomentPrecision`, `FootprintMomentRangePerM`, `FootprintMomentMotionGain`, `PlausHeightPriorGain` (as a fit term). The footprint and height prior stds stay class rows if the referee keeps them as class. |
| G5 offers under the cap | `EpistemicCooldownCycles` if the referee counts it; offer-side gains that compensate unsaturated predicted information |
| G6 occupancy check | none (it is new); it must add no constant beyond the robot footprint, which is a robot property |

Not expected to move in part 1: existence and plausibility (`FeRef`, `FeScale`, `PlausToExistenceGain`, `SingletonInhibition`, `FeBaseline*`, `FeSurpriseSmooth`, `ExistenceVerify*`), process noise, birth seeds, and the room's 0.35.

## Graded quantities (named per class)

| class | graded quantity | ground truth |
|---|---|---|
| bed | (AuRo A3, 11-25) per DOF, the extent of the mesh surfaces the likelihood fits: w = outer width of frame ∪ blanket 1.766; h (depth) = frame incl. headboard 2.09–2.12; H = blanket top 0.72; centre of that box | Bed.proto visual meshes (the prior's "mattress layer" comment is recorded as inconsistent; prior unchanged) |
| nightstand | body box | proto body mesh |
| toilet | footprint of bowl + tank | proto mesh |
| chair | seat centre and yaw (yaw in the countable convention: +y local = the backrest; the legacy chair's yaw = this − π; see the nees_agents.py header, K2b) | proto seat mesh |
| table | top centre, extent, yaw | proto top mesh |
| refrigerator | body footprint centre | proto body mesh (the 11-22 regrade already uses this) |
| hood | canopy box | proto mesh; the Webots origin must be verified first |

The scorers (`countable_concept/tools/lidar_measures_score.py`, `room_concept/estimation_scripts/scripts/nees_agents.py`) read the ground truth from mesh extents, not from boundingObject. The collision-box ground truth is kept as a second column so old numbers can still be compared.

## Engine shape

- A new header `common/ai_belief/persistent_engine.h`, templated on the model.
- **State:**
  - θ ∈ ℝᴺ, the class's own state:
    - fridge and hood: N = 6 (cx, cy, H, w, h, yaw);
    - chair: its N;
    - table: N = 7 in the quotient chart.
  - Latents:
    - chain, 3: the robot pose;
    - camera shared error, N;
    - LiDAR shared error, N, when the class uses rays;
    - fresh ego-motion shares for camera and LiDAR, N each (rev 3.3.13);
    - shape misfit, N, with ρ = 1.
  - So K = 3 + 6N with LiDAR (shape latent per channel), 3 + 3N without.
- **Hooks kept from today:** `sdf_prim`, `sdf_jacobian`, `responsibilities`, `common_mode_inv(_diag)`, `accumulate_extra` (θ-only channel), `apply_constraints`, `canonicalize`.
- **New hooks:**
  - `beam_parts(θ)`: the convex parts the beam model intersects. Chair seat/back/legs, table top/legs and fridge/hood box are all boxes.
  - `shape_spread()`: from the manifest.
  - `selector(channel)`: which θ components a latent touches.
- **Solver:** the damped GN/LM with the Nielsen λ rule from the countable belief, with the stop rule and GnTrace diagnostics. Fixes A and B come along.
- **Viewpoint ρ:** moved to `common/ai_belief/viewpoint.h`. Every agent computes ρ the same way from odometry and the object's angular half-width.
- **Port order:** fridge, then hood, then chair, then table. Fridge and hood share θ with the countable belief. The table's quotient chart is the hardest.

## Tests (before any tour)

1. **Engine unit tests (synthetic).**
   - N frames from one standpoint give σ within roundoff of one frame with all those points; it plateaus at Σc.
   - Two standpoints give the ρ-predicted drop.
   - A return to a seen standpoint gives a small promised gain.
   - At ρ = 0 everywhere, the engine reproduces today's per-frame saturated step (the existing persistent_update property).
2. **Port tests per class.** On recorded frames the new engine converges to the old mean within the old σ, and the self_test face-ranking checks pass.
3. **Parked test (live, minutes).**
   - Park in front of one static object (a chair in room_1) for 60 s at standpoint A.
   - Move to standpoint B (about 1 m, 45° round the object), park 60 s.
   - Pass when σ over the second half at A is flat within the latent prediction, and B's drop agrees with ρ. Both are stated before the run, not fitted.
   - Then the same at A again: a small promise and a small delivery.
4. **Grading run (on room's combined build at the end).** AuRo tests (a)–(d):
   - (a) first-visit NEES/DOF in [0.5, 2] per class;
   - (b) Spearman of promised against delivered information > 0, with spread in the promises;
   - (c) the patch-constant count redone on the same scope as the baseline;
   - (d) one held-out world. Someone must build it, and it must not be tuned on.

## Schedule

| dates | work |
|---|---|
| 11-24 … 11-26 | Step 0 (claim-refusal cause, mask σ from dumps). Engine extraction with unit tests. Countable moved onto the extracted engine with no behaviour change (ctest 27/27, one bedroom tour). |
| 11-27 … 12-02 | Fridge and hood ports. Shape latent and graded-quantity scorers. Offers under the cap with promise/delivery log. Occupancy check. |
| 12-03 … 12-06 | Chair and table ports, with the self_test ranking. Mask foreign-density from neighbour beliefs. |
| 12-07 … 12-10 | Parked test, then grading tours (room runs them). Count redone. Report. |

## Answers to the builder check (designer's calls unless marked owner)

1. **Countables already carry G1.** On main, bed, nightstand and toilet run the persistent path (`LidarMeasuresOnGated = true`). The AuRo NEES figures (v1, 8a133295) predate it.
   - On the persistent path (live15birth, n = 1), both nightstands are calibrated: 0 % of rows beyond 3σ on every DOF.
   - Bed width is over-confident, a G4 problem. The toilet has never been measured on it.
   - Step 0 therefore reruns `nees_agents.py` on live14fixA and live15birth, and one tour that includes the bathroom measures the toilet on the persistent path. For the countables, part 1 then means G3, G4 and G5 only.
   - The "Why" section's numbers stand for chair, table, fridge and hood, not for the countables' current path.
2. **Latent variances from sensor properties, not gains.**
   - Camera σ = σ_px · z / f, projected onto the face normal (1/cos of the obliquity, from geometry).
     - σ_px is the measured mask boundary error in pixels (G3); z and f come from the view.
     - This replaces `AI2RangeNoise*`, `kObliquityYawGain` and `AI2PeriphRef` with one sensor row plus geometry.
     - Under test (c) they are reported as moved, from patch to sensor, not as vanished.
   - Fresh ego-motion variance = (v · τ)², with v the odometry speed and τ the measured capture-to-pose latency of that sensor. It replaces `MotionCm*`.
   - Constant floors that remain are reported as they are.
3. **Shape latent (owner, 11-23).**
   - θ is the graded layer.
   - The shape misfit is a zero-mean latent with ρ = 1, and the camera and the LiDAR each have their own, since they see different layers. σ = Δ/√12 per DOF.
   - Δ comes from typical real objects of the class, written in the class manifest as class knowledge, not from the Webots meshes, so test (d) stays honest.
   - The mean still leans toward the layer the sensors see, and the NEES passes through an honest wider σ. Part 2's parts are what would remove that lean.
4. **Beam model (owner, 11-23).**
   - Table, fridge and hood move from `lidar_ray_factor` to the beam model.
   - The chair gets the Helios beam channel, with seat, back and legs as boxes: about +1 day. ZED depth is not used as rays.
   - `MaskPixelErrorRate` is the measured value (item 5). `FloorSigmaM` stays as a sensor or patch row according to the referee.
5. **Mask error.** One measurement from the mask dumps gives both the boundary σ_px and the pixel error rate ε_px. They are one sensor model with two numbers per detector, and per class only where the data differs. `MaskPixelErrorRate` becomes that measured value.
   - Neighbours (the 11-22 candidate 2, in `docs/superpowers/results/2026-11-22-fixA-live14.md`) enter as a fixed foreign density. It is built from the other instances' published pose, size and std in the same room, any class, as of the previous cycle.
   - Each point's membership compares the two densities. That is an ordinary mixture, so a point goes to whichever object explains it better, and there is no joint solve and no loop inside one update.
6. **Occupancy for G6.** Use residual_concept's published `grid` node, which the countable birth-surprise probe already reads. Step 0 confirms it covers the standpoints and finds why the AuRo claims were refused.
7. **Offers per standpoint.** Each face maps to its standpoint, and ρ for a candidate is predicted along the straight path from the last update pose. "Delivered" is the drop in the θ-marginal entropy of the published Σ, the same quantity `claim_windows` uses. The self_test ranking compares faces by this gain.

## Left to the builder

- Numerical details: where ρ is clamped by geometry (degenerate z), the quadrature settings already set in `beam_ray.h`, and how the latent blocks are laid out in memory.
- How each class's quotient or template chart maps onto the selector.
- The logging format for promise/delivery, as long as `claim_windows`-style per-visit totals can be computed.

## State naming and graded mapping (added 11-24 for AuRo A1)

All countable classes share θ = (cx, cy, H, w, h, yaw):
- **cx, cy:** the footprint centre in the room frame.
- **H:** the height, the top above the base. The base is the floor for floor-standing classes; for hanging classes (hood) the box hangs below its attachment.
- **w:** the width along the object's local x, parallel to its back face or wall.
- **h:** the DEPTH along the local y, away from the wall. It is the footprint's second extent; it is not a height.
- **yaw:** R(yaw)·ŷ is the direction from the front into the back.

The countable publisher writes `width_m = w`, `depth_m = h`, `height_m = H`. So the graded box (w, d, h) maps as width = θ.w, depth = θ.h, height = θ.H.

For box-part classes (chair, table) the parts are fixed functions of θ, so the graded quantity is either θ itself (marginal = the θθ block) or g(θ) (marginal = J Σθθ Jᵀ, with J = ∂g/∂θ from parts_geometry.h).


**AuRo A3 (11-25):** for every class and graded DOF, the reference is the extent of the mesh surfaces the likelihood fits (from the manifest and the measurement model), never chosen from the estimates. Base-world-measured priors: bed, nightstand, toilet = yes; chair = partly (template heights and leg size from the Webots SimpleChair); refrigerator, hood, table = no (nominal).

## Amendment (F1, 2026-11-25): no-cross is replaced by the stand-off prior on the persistent path

On the persistent path a class that declares a wall stand-off (`[prior.attachment] standoff_m / standoff_std_m`, cited or
marked "uncited convention") gets **no separate no-cross term**. Spec 7c.8 rev 3.3.9's "no-cross contact" (penetration ≤ ½σ_wall,
the one-sided hinge λ = 1/σ_wall² re-entered on every update) is **superseded on the persistent path**; the legacy path keeps it.

Why:
- The hinge re-entered as a measurement on every update pinned a parked toilet's back to the millimetre (engine1 diagnosis
  item 2): a fixed relation counted as new evidence each frame.
- The wall's position error is now the wall latent δ (one ρ = 1 latent per physical wall, σ_wall² from the room), and the wall
  priors enter ONCE, at the instance's first association, on (θ, δ). The stand-off prior N(gap − δ; μ ≥ 0, σ) already carries
  the class knowledge "the back stands at or off the wall".
- The exact one-sided form of no-cross on top (−log Φ(gap/σ_wall), tried in the first F1.4 commit) counted σ_wall twice — as δ's
  marginal and as its own scale — and pushed a flush box about 2 cm into the room on its one entry (φ/Φ(0) = 0.80); the creep
  set's wall-on cases failed on it.
- Every class on the countable binary (bed, nightstand, toilet) declares a stand-off, so the persistent no-cross is deleted. A
  future class without a stand-off gets the old flush form once (mean 0) and no no-cross; if it needs one, it declares a stand-off.
- For T14: `WallNoCrossPrecision` / `WallNoCrossMarginM` do not vanish — they remain the legacy path's terms; on the persistent
  path the wall's containment knowledge moved into the stand-off prior and the wall latent.

## Amendment (CW1, 2026-11-27): containment on the persistent path is the prior's support, in the mode only

The F1 amendment above assumed that the face nearest the wall is the back, so the stand-off would also keep the object inside the
room. It does not: a body held side-on, or one whose far extent is held only by the class size prior, can put its corners past a
wall while the back-face centre is legal. Live: chairfig_1127 chair_291 (91598c88) ended with two corners 0.22 m past the west
wall. Its replay on that engine shows the cause was data-driven drift on LiDAR-only updates with nothing constraining the corners.
The once-entered stand-off had entered back-on, at update 0.

- **(i) Containment, every object (fix round 1 form).** E = Σ_k ½ min(0, d_k − δ̄_j)²/(σ_s² + v_j).
  - p_k are the footprint's four corners and d_k is the signed distance to the room polygon the frame carries; when the frame
    has none, the staged wall's half-plane is used. j is the wall nearest corner k.
  - σ_s = Model.ModelSigmaM is the box surface's std.
  - (δ̄_j, v_j) is wall j's error: for the staged wall, the wall latent's carried mean and variance; for every other edge, mean 0
    and the staged σ². ⚠ That σ² is a stand-in: room_concept publishes no per-edge covariance.
  - It is the profile likelihood, jointly over the real corner and δ_j, of the hard support "the real object is inside the true
    walls". Each wall's error is therefore counted once, as its own marginal.
  - It is exactly 0 while every corner is inside. The corners are separate smooth terms.
- **A prior on θ ONLY, entered on every measured update.** Its selector is 0: the wall latent is never moved by it (review I1;
  the round-0 form, evaluated at θ + B_w b, ratcheted δ̂).
  - It is in the GN energy, so the mode respects it.
  - It is in the move statistic's factor: a support is part of the prior of both "stayed" and "moved", so it enters as a factor
    common to both (review I2).
  - It is NOT in the posterior covariance. A support restriction is idempotent (1[c]·1[c] = 1[c]), while a Gaussian factor
    re-entered n times is n·λJJᵀ, the hinge that pinned the parked toilet. P therefore stays the data's covariance, which is
    conservative.
  - Entering it once, like the attachment priors, would not work: the violation develops later, and a one-sided term is zero at
    an entry where the corners are inside.
- **(ii) Attachment** (stand-off and parallel, `[prior.attachment]`, flush-weighted) is unchanged. It is still on the back face,
  entered once per physical wall.

## Amendment (K3, 2026-11-26): G6 as built — one shared standpoint oracle

G6 now reads (item 6 above): the offer's value is multiplied by **P(stand and turn)** from the shared oracle
`common/standpoint/standpoint_oracle.h`. What that means exactly:

- **The quantity.** P = exp(−Σ risk(c)) over the union of the eight heading footprints of the robot body at the standpoint (each
  cell once), and P = 0 if any covered cell is hard. risk(c) = −log Pr(θ_c < ½ | residual's Beta) for a residual cell with a belief
  behind it (`residual_risk.h cell_risk_nats`, now in `common/standpoint/`), 0 for a collapsed cell and for unlatched ground.
  Hard = outside the room (and every passage), inside an object polygon, a residual cell with no belief behind it, or (controller
  only, and not at the Planning stage) a cell a live LiDAR return lands in. It is continuous in the belief and has no threshold
  of its own. The hard 0 is of two kinds: BELIEF-BACKED (a live return = a measurement of contact; an object polygon, treated as
  certain; the room's wall) and NO BELIEF (a residual cell with no usable Beta: the field is absent, does not cover it, or its
  moments are unusable) — the latter is the absence of a belief mapped to hard by residual_risk.h's inherited conservative
  convention, not a statement of the belief. Consequence: if a consumer cannot read residual's field (`read_residual_field`
  fails), every latched cell becomes hard and every standpoint with one under its turn gets P = 0.
- **A threshold moved, not added.** The rasteriser keeps GridPlanner's room-mask guard: a room polygon marking ≥ 95 % of the
  lattice outside is discarded (degraded mode: obstacles only). It is a pre-existing threshold, flagged ⚠ in the header, kept
  because K3 must not change the controller's decisions; a model-level replacement needs a belief over the polygon's validity.
- **One model, not a mirror.** The oracle holds the rasterisation `GridPlanner::set_world` did and the fit kernel
  `GridPlanner::fit_nats_*` did; the controller now calls them. Its decisions are unchanged (golden recorded at aea54896:
  `controller/tests/test_standpoint_oracle_parity.cpp`): it hands an object standpoint back iff P = 0 (`standpoint_fit.h`), and
  `pose_free` / `can_turn_here` are P > 0 at one / all headings. The binary decisions are the controller's, as consumers of P.
- **No constant beyond the robot footprint.** The footprint is the robot's. The lattice cell and the margin are the CALLER's: the
  controller passes `Planner.CellSize` and `Controller.FootprintSafetyMarginM`; a countable, which cannot read another agent's
  config, passes residual's own `grid_cell_size` and margin 0. Measured on synthetic cluttered worlds
  (`common/standpoint/tests/p_stand_parity_test.cpp`): the agent side stands where the controller would hand back at 9.5 % of
  standpoints, never the reverse; with the controller's margin on the agent's lattice it is 4.6 % (the lattice). Publishing the
  margin (and the planner cell) as a robot property would remove it; that is an owner decision.
- **Applied once.** The countable planner ranks faces by p·gain, carries p beside the proposal's gain, and the epistemic step
  multiplies the PUBLISHED gain by p once, after the verification floor (so a floor never re-offers a blocked standpoint).
- **This supersedes Task 8's ruling** (p_stand ∈ {0,1} by an object-side mirror of the controller's test). That ruling rejected a
  product over `grid_occupancy_prob` (P(occupied ∧ ¬explained) with a Beta-mean floor: 0.33 at best, ~1e-27 over unexplored
  ground). The oracle does not use that product: it prices only latched residual cells, by the controller's own surprisal, so
  free and unexplored floor cost 0. With no priced cell under the turn, P is 0/1 exactly as the mirror was; under priced cells it
  is exp(−nats), where the mirror read 1. Decision differences from the mirror are confined to the lattice boundary (3.6 % of
  standpoints, each within one cell of a flip).
