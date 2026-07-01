# TABLE.md — Mathematical Lifecycle of the `table_concept` Agent

The math and data flow of one table belief, from **creation** → **update** → **removal**, as
actually implemented (AI2 belief, `UseAI2 = true`). This supersedes the per-frame mechanics in
`TABLE_FIT_AI2.md` (design rationale) and is the reference for porting the lineage to other agents
(`bottle`, `chair`). Companion: `../CONCEPT_AGENT_RECIPE.md`.

Conventions: room-frame coordinates, `+Z` up, `z = 0` the robot-base datum (≈ floor; see
`scripts/analyze_table_fit.py` for the world↔room z offset). Vectors bold, matrices upper-case.

---

## 0. State and architecture

The belief is a recursive **variational-Laplace Gaussian filter** over a 6-DOF state

```
θ = [cx, cy, H, w, h, yaw]ᵀ ∈ ℝ⁶ ,   posterior  p(θ) ≈ 𝒩(θ̂, Σ),  Σ ∈ ℝ⁶ˣ⁶ (full).
```

- `cx, cy` — table-centre position (room frame).
- `H` — tabletop height (top surface z).
- `w, h` — footprint extents (canonical `w ≥ h`).
- `yaw` — heading about `+Z`.

The legs are **derived, not free DOFs**: 4 cylinders of radius `r = LEG_RADIUS` at the corners
`(±(w/2−r), ±(h/2−r))`, spanning `z ∈ [0, H−t]` with `t = TOP_THICKNESS`. They stay in the SDF so
leg observations still constrain `w, h, yaw`.

Module split (all in `src/`):

| unit | role |
|---|---|
| `table_belief.{h,cpp}` | the belief: generative model + recursive inference (pure Eigen, no torch/DSR, `self_test()`able) |
| `table_fitter.{h,cpp}` | per-cycle glue: mask ingest, tracker association, chain-cov, drives the belief |
| `table_instance.h` | one table's mutable state (belief + bookkeeping) |
| `table_scene_graph.{h,cpp}` | all DSR writes: node, RT edge + covariance, mesh, diagnostics |
| `specificworker.{cpp}` | orchestration: per-cycle merge → tracker → per-node fit → publish |
| `common/instance_tracker/instance_tracker.h` | object-agnostic birth / associate / death (shared) |

---

## 1. Generative model (compound SDF + soft mixture)

### 1.1 Signed-distance primitives

A point `p` is mapped to the table's local frame by un-rotating about the centre:

```
[lx, ly]ᵀ = R(−yaw) · [px − cx, py − cy]ᵀ
```

**Top slab** (box, half-thickness `t/2`, centre height `top_cz = H − t/2`):

```
sdf_top(p) = box_sdf( |lx| − w/2 ,  |ly| − h/2 ,  |pz − top_cz| − t/2 )
box_sdf(dx,dy,dz) = ‖max(d, 0)‖₂  +  min( max(dx,dy,dz), 0 )            (exact box SDF)
```

**Leg k** (vertical cylinder, half-height `hh = ½ max(0, H − t)`, centre at `leg_center_local`):

```
sdf_leg_k(p)  = cyl_sdf( lx − cₖx , ly − cₖy , pz − hh ,  r , hh )
cyl_sdf(dx,dy,dz,r,hh):  d_rad = √(dx²+dy²) − r ,  d_vert = |dz| − hh
               = ‖max([d_rad, d_vert], 0)‖₂ + min( max(d_rad,d_vert), 0 )
```

(`table_belief.cpp:20–68`.) The diagnostic union is `sdf_compound = min(sdf_top, sdf_leg_0..3)`.

### 1.2 Soft responsibilities (the data-association EM E-step)

Each point is explained by a 6-component mixture `{top, leg₀..₃, clutter}` with per-point
measurement variance `R` (m²). Surface priors `π_surf = (1−ε)/5`, clutter prior `ε = clutter_frac`:

```
u₀     = π_surf · top_z · exp( −sdf_top² / 2R )
u_{1+k}= π_surf · leg_z · exp( −sdf_leg_k² / 2R ),   k = 0..3
u₅     = ε      ·         exp( −clutter_scale² / 2R )          (uniform-clutter floor)
rₚᵣᵢₘ  = uₚᵣᵢₘ / Σ u        (responsibilities, Σ = 1)
```

**Height-based attribution** (`table_belief.cpp:90–106`) — the term that fixed the under-size /
h-collapse. The leg lateral surface physically exists only for `z ∈ [0, H−t]`, the slab only for
`z ∈ [H−t, H]`; they meet at the join plane `z_join = H − t`. A tabletop corner point (z ≈ H,
laterally near an inset corner leg) is otherwise mis-attributed to the leg, and GN shrinks `w, h` to
slide the legs under it. A smooth vertical-compatibility gate (band = slab half-thickness, physical,
**not a threshold**) splits them:

```
leg_z = 1 / (1 + exp((pz − z_join)/band))      → 1 below the join, → 0 at tabletop height
top_z = 1 − leg_z                              → 1 at tabletop, → 0 below
```

EM holds the responsibilities fixed within a Gauss-Newton iteration, so this needs no Jacobian change.

### 1.3 Jacobian

`∂sdf_prim/∂θ` is a central finite difference per DOF (`fd_eps = 1e-3`), `table_belief.cpp:122–135`.
The SDF is piecewise-smooth; FD with the clutter escape valve keeps the seams from spiking the
information.

---

## 2. CREATION (birth)

A table comes into existence in three stages: **tracker birth** (decide a new instance is warranted)
→ **DSR node creation** → **first-frame belief seeding**.

### 2.1 Tracker birth (`common/instance_tracker/instance_tracker.h:164–202`)

Per cycle the worker builds a `DetectionView` per mask slice and a `TrackView` per instance and calls
`tracker_.update()`. A detection spawns a candidate only if it is **unassigned** and **separated**
from every existing track:

```
far_from_tracks(xy):  ‖xy − tₖ.xy‖₂ ≥ birth_min_sep_m   ∀ tracks k
```

Candidates persist across frames (matched within `birth_match_m`), incrementing a `streak`. Promotion
to a birth requires

```
streak ≥ birth_frames        (and no duplicate promotion within birth_min_sep_m this cycle)
```

For static furniture `birth_min_sep_m` is wide and `birth_frames` is several, so a transient/partial
view does not spawn duplicates. (Caveat: a brief partial view that *does* clear the gate can fragment
one table into two — see §4.3.)

### 2.2 DSR node creation (`table_scene_graph.cpp:26–67`)

`create_instance_from_detection()`:
- creates a node `DSR::Node::create<table_node_type>("table_<N>")` (sequential `N`),
- sets `width_m / depth_m / height_m` from the birth-size config, `level = 3`, `parent = room`,
- inserts it, then writes the **room→table RT edge**:
  `insert_or_assign_edge_RT(room, table, {cx, cy, z=H/2}, {0,0,0})`.

⚠ **Ownership rule** (`specificworker.cpp:893–896`): instance creation is owned by the `compute()`
loop (`process_table_node → ensure_instance`), *not* a DSR node-create slot — else the model is born
at the default `(0,0)` RT-read before its detection seed lands, freezes there, and the tracker
re-births forever.

### 2.3 First-frame belief seeding (`table_fitter.cpp:355–386`)

On the first fitted frame (`ai2_initialized == false`), the belief is cold-started from the data, not
the prior, so the points don't all look like clutter:

```
cₓ, c_y  ← centroid of all observed points (candidate ∪ residual)
H        ← 95th-percentile of observed z  (the visible top face)
w, h, yaw ← model/prior defaults
```

The prior covariance is broad (`init_prior_cov`, `table_belief.cpp:149–159`):

```
Σ₀ = diag( 0.30², 0.30², σ_s², σ_s², σ_s², 0.60² ),   σ_s = prior_size_std
```

so only the empty-cloud degeneracy is broken; data dominates immediately.

---

## 3. UPDATE (one perception cycle)

Pipeline per node per cycle (`run_inference_ai2`, `table_fitter.cpp:348–467`):

```
observe() → compute_chain_cov() → [gate?] → belief.predict()/update() → write-back → DSR publish
```

### 3.1 observe — evidence assembly (`table_fitter.cpp:241–302`)

- The tracker's gated assignment picks the mask slice (`assigned_mask_idx`); `−1` ⇒ no fresh mask ⇒
  **freeze-on-stale** (predict only, the information-filter axiom).
- Slice scalars are stashed on the instance: `confidence`, `timestamp_ms`, `motion_var`,
  `motion_dotd`, `trunc_frac`, **`range`** (mean camera→mask depth Z, from the voxelizer).
- Support points are split by the current SDF: `|sdf(p)| < sdf_threshold` → `candidate_pts`
  (near-surface inliers); else `residual_pts`. Both feed the belief; their ratio is the
  `explanation_ratio` diagnostic.

### 3.2 Common-mode covariance Σc (the saturation cap)

The per-frame **common-mode** is the error *shared* by all points of one mask (localization + mask
boundary + deprojection) — it does **not** average out over points. Its inverse (diagonal) is

```
Σc⁻¹ = diag[ 1/(p² + χₓₓ),  1/(p² + χ_yy),  1/s², 1/s², 1/s²,  1/(y² + χ_yaw) ]
```

with `p = common_mode_pos_std`, `s = common_mode_size_std`, `y = common_mode_yaw_std`
(`table_belief.cpp:192–199`). Two physical covariates enter the diagonal:

**(a) Pose-chain (localization) cov** `χₓₓ, χ_yy` — `compute_chain_cov`, `table_fitter.cpp:37–60`.
The fit is in the room frame but its position is conditional on the robot pose
(camera→robot→room). Transform the table centre room→`zed` and back with **zero** input covariance;
`DSR::InnerGaussianAPI::transform_point` returns exactly the chain contribution

```
χ = J · Σ_chain · Jᵀ ,    Σ_chain = adjoint-composed rt_covariance of each RT edge,
```

pinned to the mask capture stamp (`room_concept` publishes the robot↔room term).

**(b) Static range** — `table_fitter.cpp:397–423`. Even at zero camera motion, deprojection noise
grows with distance *and* a far mask subtends a tiny angle, so orientation becomes unobservable.
This is a **continuous covariance, no gate** (per the workspace no-threshold rule):

```
range_lat_var = (lat_rate · range)²        range_yaw_var = (yaw_rate · range)²
χₓₓ += range_lat_var,  χ_yy += range_lat_var,  χ_yaw = range_yaw_var
```

(`lat_rate = AI2RangeNoiseLatPerM`, `yaw_rate = AI2RangeNoiseYawPerM`.) The yaw term is the binding
one: it lowers the per-frame information *cap* so a 7 m view's yaw gain against a converged table
drops ~50× — it can confirm existence but not rotate the table. (The motion×distance term is already
in `motion_var` via the voxelizer interaction matrix `L(xₙ,yₙ,Z)`; this is the missing *static* part.)

### 3.3 Predict (`table_belief.cpp:161–171`)

Rigid + static ⇒ small isotropic process noise, mean unchanged:

```
Σ ← Σ + Q,   Q = diag(q_m, q_m, q_m, q_m, q_m, q_y),   q_m = process_std_m², q_y = process_std_yaw²
μ_prior ← θ̂
```

### 3.4 Update — MAP mean + calibrated covariance (`table_belief.cpp:175–254`)

Let `P₀ = Σ_pred⁻¹` (transition-prior precision). Per Gauss-Newton iteration, accumulate the
**data-only** information from every point × primitive, weighted by responsibility:

```
wᵢ = rᵢ[prim] / R ,   J = ∂sdf_prim/∂θ ,   d = sdf_prim(p)
Id += wᵢ · J Jᵀ        bd += −wᵢ · J d                  (table_belief.cpp:204–213)
```

**MEAN (MAP)** uses the *full* data information so the point estimate converges fast and unbiased:

```
θ ← θ + (P₀ + Id)⁻¹ ( P₀ (μ_prior − θ) + bd ),   repeat gn_iters times, apply constraints + w≥h fold
```

**COVARIANCE (calibration)** caps the frame's information at the common-mode via exact Woodbury
marginalization — this is what stops 10⁴ correlated points collapsing σ to sub-mm:

```
I_eff = Id − Id (Id + Σc⁻¹)⁻¹ Id     →  Σc⁻¹  as Id → ∞
Σ     = (P₀ + I_eff)⁻¹                                       (table_belief.cpp:216–254)
```

The split is deliberate: the **mean** sees all data (MLE/MAP), the **covariance** accounts for the
unmodeled within-frame correlation. Applying the saturation to the GN mean step instead throttles the
weakly-observed DOFs and breaks recovery (regression-tested in `self_test()`).

Constraints (`apply_constraints`, `table_belief.cpp:139–145`): `w,h ≥ 0.10`, `H ≥ t+0.05`, `yaw`
wrapped to `(−π, π]`; plus the **canonical w ≥ h fold** (if `w < h`, swap and `yaw += π/2`) so the
180°/90° rectangle symmetry doesn't drive a yaw flip.

**Truncation gate** (`table_fitter.cpp:405`, `419`): a mask clipped by the image border has a chopped
silhouette → biased fit. If `trunc_frac > AI2TruncGateFrac` the geometric update is skipped (predict
only); the instance is kept. This is the one explicit gate, justified because a truncated silhouette
is *structurally* biased, not just noisy.

### 3.5 Write-back + DSR publish

- The belief mean is copied into the legacy `TableState` (legs derived: `leg_length = H − t`,
  `inset = r`), so all viewer/RT/mesh code is unchanged (`table_fitter.cpp:444–449`).
- **Geometry gate** (`table_scene_graph.cpp:90–98`): geometry/mesh attrs are rewritten only if any
  DOF moved > `kPosEps = 3 mm` / `kYawEps ≈ 0.3°` — kills viewer jitter. `free_energy` is written
  every cycle; `model_generation` increments on each geometry republish.
- **RT pose** is written with a ~5 cm dead-band (`table_scene_graph.cpp:299–304`) to suppress churn.
- Diagnostic point clouds exported: `rfe_pts`, `table_voxel_bank_pts`, `residual_pts`.

**RT covariance under AI2** (`write_rt_covariance`, `table_scene_graph.cpp:163–207`). When `UseAI2`,
the published 6×6 edge cov is mapped from the belief's full Σ (not the legacy `fisher_info_raw`
accumulator, which AI2 never populates — that path published `big = 1e3` garbage on every DOF):

```
cov_se3[0,0] = scale·Σ[cx]  + chain_cov_xx      cov_se3[3,3] = big   (roll, unobservable)
cov_se3[1,1] = scale·Σ[cy]  + chain_cov_yy      cov_se3[4,4] = big   (pitch, unobservable)
cov_se3[2,2] = scale·¼·Σ[H]   (z = H/2)         cov_se3[5,5] = scale·Σ[yaw]
```

The chain (localization) term is re-added even though Σ already folds the per-frame common-mode floor:
across-frame accumulation can tighten `cx,cy` below that floor, so adding `chain_cov` keeps the
published cov conservative — the safe direction for the controller's uncertainty governor. (A fully
honest fix is the deferred *across-frame* common-mode; until then, conservative is correct.)
Off-diagonals remain 0 (the SE3 block is diagonal; cross-terms unmodelled).

---

## 3½. EPISTEMICS — next-best-view selection (where to look next)

The belief is passive; *moving the robot to resolve it* is the epistemic loop. Each cycle the agent
proposes a viewpoint that most reduces the table's pose/shape uncertainty — the active-inference
**epistemic value** (expected ambiguity reduction). The proposal is published as an `aff_table_*`
affordance node; the **controller** does the final next-spot selection across all affordances by
minimizing expected free energy `G = cost − epistemic_value − pragmatic_value` (with commitment
hysteresis to avoid thrash). So the table agent *scores viewpoints*; the controller *picks the spot*.

### Current implementation (`EpistemicPlanner::compute`, legacy)
The four vertical faces are candidate look targets. For each face `i`:

```
I_pred(i) = model.observation_information( sample_face_surface(state, i) )   // predicted Fisher diag
gain(i)   = P(detect | v_i) · expected_info_gain( I_pred(i), posterior_info )  // P(detect)=1 for now
```

`expected_info_gain` is the entropy reduction from adding `I_pred(i)` to the current posterior
information (`ΔH ∝ face_area / σ²`). The best face's viewpoint is placed just outside it:

```
v*      = face_centre + outward_normal · standoff      (standoff keeps the face inside a ~70° FoV)
heading = atan2(cy − v*_y , cx − v*_x)                 (face the table centre)
```

and emitted as `EpistemicProposal{x, y, yaw, gain}`. A low-but-finite gain is **not** withdrawn — the
node persists carrying its true gain so the controller's EFE selection simply ranks it low (the
belief→knowledge governor expressed as a small gain, not a deleted node).

### ⚠ AI2 status — the NBV is not yet AI2-native
`compute` reads `queue.face_coverage(model)` (the **sample queue**) and `posterior_info` (the legacy
**diagonal Fisher** `inst.stab.fisher_info_raw`) and the torch model's `observation_information` —
**all three are empty/zero under AI2**, so the gains degenerate and the proposal carries no real
information. (Call site `specificworker.cpp:822`.) This is the *one* remaining legacy dependency
besides the now-fixed RT-cov.

### AI2-native Σ-based D-optimal NBV (✅ implemented, built green + unit-tested; pending live validation)

Replace the diagonal proxy with **expected entropy reduction on the belief's full Σ**. The four
vertical faces stay the candidate look-targets; the geometry (face centres, normals, stand-off,
heading) is unchanged. Only the **score** changes — from a Fisher-diagonal coverage proxy to a true
information gain:

```
for each candidate face i:
   standoffᵢ = (existing FoV-fit stand-off for face i)
   Rᵢ        = σ_base² + (lat_rate · standoffᵢ)²            // range-aware: a far face yields less info
   pts       = N synthetic points on vertical face i of the top slab (posed by cx,cy,yaw,H,w,h)
   ΔI(i)     = Σₚ (1/Rᵢ) · Jₚ Jₚᵀ ,   Jₚ = ∂sdf_top/∂θ|ₚ    // belief's own 6-vector SDF Jacobian
   gain(i)   = ½ · ln det( I₆ + Σ · ΔI(i) )                 // D-optimal: expected log-det/entropy drop
pick argmax gain;  emit EpistemicProposal{ v* = faceᵢ.centre + nᵢ·standoffᵢ , heading→centre , gain }
```

Properties:
- **D-optimal on the full Σ** (not the diagonal): `½ ln det(I + Σ·ΔI)` is the expected reduction in the
  belief's differential entropy from observing face `i`. It automatically targets the viewpoint that
  most shrinks the **dominant uncertainty eigen-direction** of Σ — typically an unobserved far extent
  (`w`/`h`) or `yaw`. (E-optimal "attack the single worst DOF" is the variant
  `½ ln(1 + λ₁·e₁ᵀΔI(i)e₁)` with `(λ₁,e₁)` the top eigenpair; D-optimal is preferred — it accounts for
  all directions and Σ's cross-terms.)
- **Range-aware** by construction: `Rᵢ` reuses the §3.2 static-range model, so the planner prefers
  faces it can approach and discounts range-degraded ones — consistent with the belief's own weighting.
- **No sample queue, no Fisher diagonal** — only Σ (`belief.covariance()`) and the SDF Jacobian the
  belief already computes, via a new `TableBelief::predicted_information(pts, R) → Matrix6f`
  (`I += (1/R)·J Jᵀ`, prim = top). A low-but-finite gain is still emitted (not withdrawn) so the node
  persists and the controller ranks it.

Implementation outline (full steps + signatures in the migration plan): `(1)`
`TableBelief::predicted_information`; `(2)` a belief-state face sampler `sample_face_surface_ai2`;
`(3)` an `EpistemicPlanner::compute(belief, lat_rate, σ_base)` overload doing the score above; `(4)`
worker wiring under `UseAI2`; `(5)` optional `P(detect|v)` visibility factor; `(6)` unit test —
inflate `Σ` on one extent, assert the planner picks the face perpendicular to it. Then `(7)` lift the
scorer into the shared engine so chair/bottle inherit NBV. The placement, proposal struct, and the
controller-side EFE pick (`G = cost − epistemic − pragmatic`) are all unchanged.

---

## 4. REMOVAL (death and merge)

Removal has two distinct mechanisms. **Merge is implemented; death is intentionally OFF** for tables.

### 4.1 Merge — physical exclusion (`specificworker.cpp:442–476`)

Two physical tables can't share space, so each cycle **before** associate/birth the worker collapses
instances whose footprints overlap:

```
ratio = footprint_overlap_ratio(stateₐ, state_b)         (oriented-rect Sutherland–Hodgman clip,
                                                          area of intersection / area of smaller)
if ratio ≥ tracker_merge_overlap:  keep arg max matched_frames, delete + forget the other
```

This is the only active removal path for static tables; it reconciles a table accidentally fitted
twice (two tracks locked onto one object). The overlap metric is geometry-specific (oriented
rectangle for table/chair; circle-lens for bottle).

### 4.2 Death — disabled by design (`instance_tracker.h:149–162`)

The tracker supports negative-information death: a miss accrues only when a track **should** be
visible (centre projects inside the `zed` frustum) yet isn't detected; out-of-FoV the miss timer is
**held**, so an object persists when the robot looks away. Retirement fires at `miss ≥ death_frames`.

For tables, `DeathEnabled = false` (`Tracker.DeathEnabled`): a table is rigid persistent furniture, a
long occlusion is not absence. So `res.deaths` is ignored (`specificworker.cpp:548–551`) and tables
are never retired by timeout. To enable removal-on-genuine-absence: set `DeathEnabled = true`, feed
`TrackView.expected_visible` per cycle, and on a death `delete_node` + `forget_node` + `affordance.remove()`.

**The negative-information gate is the whole removal contract** — get it right or an instance flickers
(created, then deleted within seconds). *Any* removal timer must advance ONLY when the instance
**should** be seen — its model projects into the camera FoV (`inst.roi_valid` / `TrackView.
expected_visible`) yet no mask associated. Out-of-FoV (robot looked away) the timer is **held**, so a
piece of furniture glimpsed once and left behind persists; a true phantom sits at a detected location →
projects in-frame → its timer still advances and it is removed. This applies to death **and** to
`chair_concept`'s stillbirth prune (`tracker_prune_enabled` — the chair worker's `unassigned_streak`),
which without the gate deleted a real chair the robot turned away from within `prune_patience` cycles
(the "chair appears and disappears in seconds" flicker, fixed 2026-07-01 by gating the prune increment
on `roi_valid` + setting `TrackView.expected_visible = roi_valid`). Table has no prune (merge-only
removal), so it never flickers; the same gate is the template if a table prune/death is ever added.

### 4.3 Open issue — fragmentation vs merge

A table born from a *brief partial view* is under-sized/mis-placed; a later view that fails the
association Mahalanobis gate **and** clears `birth_min_sep_m` spawns a second instance, and because
both boxes are under-sized their footprints don't overlap enough to trip the merge. Transient (settles
with dwell). Principled fix (no threshold): gate birth on whether the detection falls within an
existing instance's footprint + pose covariance (OBB/Mahalanobis), not a fixed centroid distance.

---

## 5. Lifecycle summary

```
        ┌─────────────────────────── per perception cycle ──────────────────────────┐
        │  merge_overlapping_instances()        (§4.1 physical exclusion)            │
        │  tracker.update(tracks, dets)         (§2.1 associate / birth / [death])   │
        │     ├─ births  → create_instance_from_detection   (§2.2 DSR node + RT edge)│
        │     └─ deaths  → ignored (DeathEnabled=false)      (§4.2)                   │
        │  for each table node:                                                      │
        │     observe()            (§3.1  tracker slice → candidate/residual split)  │
        │     compute_chain_cov()  (§3.2a J·Σ_chain·Jᵀ localization)                 │
        │     run_inference_ai2:                                                     │
        │        first frame → seed (§2.3 centroid + 95-pct H)                       │
        │        predict (§3.3)                                                       │
        │        if trunc_frac>τ: predict-only   else  update (§3.4 MAP + Woodbury)   │
        │     write-back + DSR publish (§3.5; ⚠ RT-cov gap)                          │
        └────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Config reference (AI2 path, `etc/config.toml [TableModel]` + `[Tracker]`)

| key | symbol | role |
|---|---|---|
| `UseAI2` | — | select the AI2 belief (true) vs legacy Fisher path |
| `AI2SigmaBaseM` | √R₀ | base on-surface obs noise std (m) |
| `AI2ClutterFrac` / `AI2ClutterScaleM` | ε / cs | clutter mixture prior / distance scale |
| `AI2PriorSizeStd` | σ_s | broad size-prior std (breaks empty-cloud degeneracy) |
| `AI2ProcessStdM` / `AI2ProcessStdYaw` | √q_m / √q_y | predict process noise (rigid+static ⇒ small) |
| `AI2CommonModePosStd/SizeStd/YawStd` | p / s / y | within-frame common-mode (Woodbury saturation cap) |
| `AI2RangeNoiseLatPerM` | lat_rate | static range → R + position common-mode (m per m) |
| `AI2RangeNoiseYawPerM` | yaw_rate | static range → yaw common-mode (rad per m) — anti far-rotation |
| `AI2TruncGateFrac` | τ | predict-only when silhouette truncation exceeds this |
| `AI2GnIters` | — | Gauss-Newton iterations per frame |
| `Tracker.GateMahalanobis` | χ²₂ | association gate on `S = P + R²I` |
| `Tracker.DetectionNoiseM` | R | innovation-cov inflation (centroid-vs-fit offset) |
| `Tracker.BirthFrames` / `BirthMinSepM` | — | birth persistence / anti-duplicate separation |
| `Tracker.MergeOverlap` | — | footprint-overlap fraction to collapse duplicates |
| `Tracker.DeathEnabled` / `DeathFrames` | — | timeout retirement (OFF for tables) |

---

*Generated 2026-06-30 from the live AI2 implementation. Keep in sync with `table_belief.{h,cpp}`,
`table_fitter.cpp`, `table_scene_graph.cpp`, `common/instance_tracker/instance_tracker.h`.*
