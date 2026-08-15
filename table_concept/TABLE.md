# TABLE.md — the `table_concept` belief, current mathematics

The authoritative math of the `table_concept` agent as **actually implemented today** (post-2026-07
rework). One table is a recursive **variational-Laplace Gaussian belief** over a 6-DOF compound-box
model, driven by a per-point SDF mixture plus several sensor-independent channels, and exposed to the
controller as an **object-relative viewpoint affordance**.

This file is the single source of truth; it supersedes the retired design notes (`TABLE_FIT_AI.md`,
`TABLE_FIT_AI2.md`, `TABLE_CONCEPT*.md`, `TABLE_FIT_EXPERIMENT.md`, `ARCHITECTURE.md`). For the shared
concept-agent *structure* (module split, DSR I/O, presence, media plane, tracker) see
[`../CONCEPT_AGENT_RECIPE.md`](../CONCEPT_AGENT_RECIPE.md) — not repeated here.

Conventions: room-frame coordinates, `+Z` up, `z = 0` the robot-base datum (≈ floor). Vectors bold,
matrices upper-case. Live code anchors: `src/table_belief.{h,cpp}`, `src/table_fitter.cpp`,
`src/epistemic_planner.cpp`, `src/table_config.h`; shared engine
`common/ai_belief/recursive_laplace.h`.

---

## 0. State and philosophy

```
θ = [cx, cy, H, w, h, yaw]ᵀ ∈ ℝ⁶ ,   posterior  q(θ) = 𝒩(θ̂, Σ),  Σ ∈ ℝ⁶ˣ⁶ (FULL).
```

- `cx, cy` — table-centre position (room frame).
- `H` — tabletop height (top-surface z).
- `w, h` — footprint extents (canonical `w ≥ h`).
- `yaw` — heading about `+Z`.

**No thresholds.** Everything the old code did with gates/ratchets/hysteresis (viewpoint gate,
maturity stiffening, size ratchet, CUSUM, position lock, yaw barrier, GNC anneal) emerges from the
math: a *full* Σ (unobserved directions stay loose because the measurement information is rank-deficient
there), a mixture likelihood (soft responsibilities over {top, legs, clutter}), process noise `Q` in the
predict, and heteroscedastic per-channel precision `R`/common-mode `Σc` (sensor-model priors, not σ
floors). The single genuine remaining switch is the truncation gate (§7) and the grow-only guard (§5),
both flagged and justified. The `leg_inset` DOF was removed: legs sit rigidly at the outer edge.

---

## 1. Generative model — compound SDF + soft mixture

### 1.1 Signed-distance primitives (`table_belief.cpp:22–87`)

Un-rotate a point into the table's local frame: `[lx,ly]ᵀ = R(−yaw)·[px−cx, py−cy]ᵀ`.

**Top slab** — oriented box, half-thickness `t/2`, centre height `top_cz = H − t/2`:

```
sdf_top(p) = box_sdf( |lx|−w/2 , |ly|−h/2 , |pz−top_cz|−t/2 )
box_sdf(dx,dy,dz) = ‖max(d,0)‖₂ + min( max(dx,dy,dz), 0 )              (exact box SDF)
```

**Leg k** — vertical cylinder, half-height `hh = ½·max(0, H−t)`, corners `(±(w/2−r), ±(h/2−r))`,
`r = leg_radius`:

```
sdf_leg_k(p) = cyl_sdf( lx−cₖx , ly−cₖy , pz−hh , r , hh )
cyl_sdf: d_rad=√(dx²+dy²)−r,  d_vert=|dz|−hh
       = ‖max([d_rad,d_vert],0)‖₂ + min( max(d_rad,d_vert), 0 )
```

Legs are **derived**, not free DOFs, but stay in the SDF so leg observations inform `w,h,yaw`. The
5-primitive union `sdf_compound = min(sdf_top, sdf_leg_0..3)` is a diagnostic only.

### 1.2 Soft responsibilities — the EM E-step (`table_belief.cpp:89–136`)

Each point is explained by a **6-component mixture** `{top, leg₀..₃, clutter}` at per-point variance
`R` (m²). Surface priors `π_surf = (1−ε)/5`, clutter prior `ε = clutter_frac`:

```
u₀      = π_surf · top_z · exp(−sdf_top²/2R)
u_{1+k} = π_surf · leg_z · exp(−sdf_leg_k²/2R),   k=0..3
u₅      = ε      ·         exp(−clutter_scale²/2R)          (uniform-clutter Gaussian floor)
rₖ = uₖ / Σⱼ uⱼ            (responsibilities, Σ = 1)
```

There is no hard `min()`/winner-take-all: the soft responsibility *is* the principled "which part of
the table caused this point". An undersized box's far top-edge points fall outside the slab, raising the
corner-leg / clutter responsibility and pulling the corners *outward* (growing `w,h`) rather than pinning
them.

**Height-based attribution** (the fix for the under-size / h-collapse). The leg lateral surface exists
only for `z ∈ [0, H−t]`, the slab only for `z ∈ [H−t, H]`; they meet at the join plane `z_join = H−t`. A
tabletop corner point (z≈H, laterally near an inset corner leg) is otherwise mis-attributed to the leg,
and GN shrinks `w,h` to slide the legs under it. A smooth vertical-compatibility gate (band = slab
half-thickness — physical, not tuned) splits them:

```
leg_z = 1 / (1 + exp((pz−z_join)/band))     → 1 below the join, → 0 at tabletop height
top_z = 1 − leg_z
```

EM holds responsibilities fixed within a GN iteration, so this needs no Jacobian change.

### 1.3 Jacobian

`∂sdf_prim/∂θ` is a central finite difference per DOF (`fd_eps = 1e-3`, `table_belief.cpp:140–153`).
The SDF is piecewise-smooth; FD with the clutter escape valve keeps the seams from spiking the
information.

---

## 2. Belief — recursive Laplace + common-mode saturation

The Bayesian bookkeeping is the shared engine `rc::ai::{predict,update,inflate_for_age,
predicted_information}` (`common/ai_belief/recursive_laplace.h`); `TableBelief` supplies only the model
hooks (SDF, responsibilities, Jacobian, `Q`/prior/common-mode diagonals, the canonical fold). Static
object ⇒ transition `F = I`.

### 2.1 Predict (`table_belief.cpp:177–182`)

Rigid + static ⇒ small isotropic process noise, mean held:

```
Σ ← Σ + Q,   Q = diag(q_m,q_m,q_m,q_m,q_m,q_y),   q_m = process_std_m², q_y = process_std_yaw²
μ_prior ← θ̂
```

### 2.2 Update — MAP mean, calibrated covariance

Let `P₀ = Σ_pred⁻¹`. Per Gauss-Newton iteration, accumulate **data-only** information from every point ×
primitive, responsibility-weighted:

```
wᵢ = rᵢ[prim]/R,   J = ∂sdf_prim/∂θ,   d = sdf_prim(p)
Id += wᵢ·J Jᵀ        bd += −wᵢ·J d
```

plus the extra factors of §4–§5 folded into `(Id, bd)` by `accumulate_extra`.

**MEAN (MAP)** uses the *full* data information so the estimate converges fast and unbiased:

```
θ ← θ + (P₀ + Id)⁻¹ ( P₀(μ_prior−θ) + bd ),   × gn_iters, then constraints + canonical fold
```

**COVARIANCE (calibration)** caps the frame's information at the per-frame **common-mode** `Σc` via
exact Woodbury marginalization — this is what stops N≈10⁴ correlated mask points collapsing σ to sub-mm:

```
I_eff = Id − Id (Id + Σc⁻¹)⁻¹ Id      →  Σc⁻¹  as Id → ∞
Σ     = (P₀ + I_eff)⁻¹
```

The split is deliberate: the **mean** sees all data (MLE/MAP); the **covariance** accounts for the
unmodeled within-frame correlation. Applying the saturation to the mean step instead throttles the
weakly-observed DOFs and breaks recovery (regression-tested in `self_test()`).

### 2.3 Common-mode Σc — the shared, non-averaging error (`table_belief.cpp:184–199`)

`Σc` is the error shared by *all* points of one mask (localization + mask boundary + deprojection),
which does not average out. Its inverse diagonal:

```
Σc⁻¹ = diag[ 1/(p²+χₓₓ), 1/(p²+χ_yy), 1/(s²+χ_sz), 1/(s²+χ_sz), 1/(s²+χ_sz), 1/(y²+χ_yaw) ]
```

with `p = common_mode_pos_std`, `s = common_mode_size_std`, `y = common_mode_yaw_std`, and the
range/pose covariates `χ` from §6.

### 2.4 Constraints & canonical fold (`table_belief.cpp:161–167, 472–497`)

Per-GN bounds (physical floors that keep the SDF well-posed, not evidence gates): `w,h ≥ 0.10 m`,
`H ≥ t+0.05 m`, `yaw` wrapped to `(−π, π]`.

The box SDF is invariant under the exact symmetry group `{e, r_π, swap∘r_{+π/2}, swap∘r_{−π/2}}` — four
representations of the *same* table. The old `sign(w−h)` fold was a **noise** process when `w≈h`,
converting extent jitter into 90° yaw snaps. `canonicalize()` instead picks the representative with the
smallest **prior Mahalanobis** to the predicted mean under `Σ_pred` — the MAP over the covering sheets,
so the chart follows continuity and the identity representative wins in the common case (no fold on
noise). The genuine near-square cross-class flip is owned by `resolve_orientation` (§5.2), not here.

### 2.5 Freshness as precision — age-inflation (`table_belief.cpp:198–199`, `table_config.h:45–49`)

A dead/stale mask stream must not freeze the belief. When no fresh mask associates, the fitter calls
`inflate_for_age(dt, dt_nominal)`:

```
Σ ← F Σ Fᵀ + Q·(dt/dt_nominal),   mean held.
```

so Σ keeps growing on the agent's own clock (measurement-age → covariance), and a dead ZED/mask feed
reads downstream as a widening Σ rather than a stale-but-confident pose. `AI2AgeNominalDtS ≤ 0` disables
it (historic information-filter freeze).

---

## 3. Free energy — clutter-inclusive marginal (`table_belief.cpp:499–520`)

The free energy is the mean per-point **negative-log marginal likelihood of the FULL mixture, clutter
included**:

```
F = mean_p [ −log Σₖ πₖ N(dₖ; 0, R) ] = mean_p [ −log( Σⱼ uⱼ(p) ) ]      (u₅ = clutter term)
```

A point far from every surface falls to the clutter floor `u₅ = ε·exp(−cs²/2R)`, so `−log(sum)` is
large: **F rises with misfit**. The `(2πR)^{−3/2}` normaliser is dropped (it cancels in every comparison
and in Δ over frames).

> Superseded bug (do NOT reintroduce): the old readout summed only the surface prims weighted by
> responsibility, so a misfit point (routed to clutter, `r_surface≈0`) contributed ≈0 and a badly-fit
> model read `F≈0` — blind to exactly the errors that matter. That broke mode comparison and made the
> orientation logic need leak/hysteresis band-aids. The clutter-inclusive F removes all of them.

---

## 4. Extent & orientation

The per-point mixture is structurally **degenerate for yaw/extent on a flat top**: interior top points
have `∂sdf/∂yaw ≈ ∂sdf/∂w ≈ 0`, and the outboard rim/leg points that *do* carry the signal fall outside
the box → routed to clutter (zero gradient). Two mechanisms restore that DOF.

### 4.1 Footprint second-moment factor (`table_belief.cpp:201–235, 386–458`)

A **global statistic** breaks the trap: the top-band cloud's 2D centroid + inertia tensor is a
sufficient statistic of the filled-rectangle model. Select points in the tabletop band
`z ∈ [H−t−3σ, H+3σ]`, form the 2×2 second-moment (inertia) tensor, closed-form eigendecompose:

```
λ₁ ≥ λ₂         → full extents of the equivalent uniform rectangle  ext = √(12·λ)
major eigenvector angle  φ = ½·atan2(2·cxy, cxx−cyy) ∈ (−π/2, π/2]   → yaw
```

This is fused as a **separate 3-DOF (w,h,yaw) Kalman channel with its OWN covariance, deliberately
OUTSIDE the per-point common-mode**. Rationale: the shared radial error largely *cancels* in a global
direction/scale estimate, so the moment stays informative at range where the per-point channel is
frozen. Standard information update on the full 6×6 Σ with H = `[e₃ᵀ e₄ᵀ e₅ᵀ]`, innovation
`y = (mw−w, mh−h, myaw−yaw)`, and measurement variances

```
r_w = r_h = 1/Pm + moment_extra_var           (1/Pm = static floor; extra_var = ego-motion + range term)
r_yaw     = r_base / max(aniso², 1e-3),   aniso = (ext_major−ext_minor)/(ext_major+ext_minor)
```

Key properties:
- **yaw variance ∝ 1/anisotropy²** — a near-square footprint carries no orientation info → K→0 →
  `resolve_orientation` owns that flip. Never applied to `cx,cy` (a partial view's centroid is biased;
  the per-point channel fixes position).
- The (major,minor)→(w,h) label is chosen **nearest the current (w,h)** so the moment reinforces the
  present mode and never itself triggers a 90° flip.
- **GROW-ONLY guard** (the file's one hard switch, flagged): a mask is only ever a *lower bound* on the
  table extent (occlusion, foreshortening, FoV clip, YOLO under-segmentation all shorten it; nothing
  lengthens it). So if the moment would *shrink* `w` or `h`, that row's variance is set to `kDrop = 1e12`
  (K→0: no mean move, no Σ reduction). Legitimate shrink is the vacate channel's job (§5.3). Extent grows
  to the largest footprint seen across the orbit and holds.
- `moment_extra_var` (ego-motion + gentle range term) makes the moment **accumulate** over frames rather
  than snap to each noisy footprint. `FootprintMomentPrecision = 0` disables (baseline unchanged). Also
  seeds `(w,h,yaw)` at birth via the same `footprint_moment()`.

### 4.2 Orientation-mode resolution (`table_belief.cpp:522–577`)

The 6×6 Σ carries only the **within-mode** yaw width (~1°). For a near-square footprint the two classes
`[(w,h,ψ)]` and `[(h,w,ψ)]` (a w↔h swap ≡ 90° rotation) have near-equal data energy — an ambiguity a
unimodal Gaussian cannot hold, so the per-frame MAP used to *snap* 90°. `resolve_orientation` is a **pure
sequential-Bayesian** comparison on the *true* (clutter-inclusive) F:

```
flip_evidence += clamp(evidence_weight,0,1)·(E_swap − E_now),   clamped to [−6,6]
if flip_evidence < 0:  swap w↔h in state AND rows/cols 3↔4 of Σ;  flip_evidence = −flip_evidence
```

Accumulation gives the belief **memory**: one partial/biased frame cannot flip it. The boundary is
zero accumulated evidence (the MAP over the discrete mode) — **no tuned threshold, no leak, no
discriminability band-aid** (all removed; a near-square footprint gives `E_swap≈E_now` → the accumulator
rests near 0, the honest ambiguity). `evidence_weight ∈ [0,1]` is the continuous ego-motion reliability
(a smeared/moving frame barely votes — the observed reshapes/flips all arrived on motion frames).

The **reported** yaw variance folds in the discrete-mode entropy so a still-ambiguous table reports an
honest ~45°:

```
p = mode_posterior = σ(−flip_evidence)                       (p of the alternative/swapped mode)
yaw_marginal_var = Σ(5,5) + p(1−p)(π/2)²
covariance_reported = Σ with (5,5) ← yaw_marginal_var
```

`covariance_reported` is what the NBV planner and the RT-edge upload consume, so an unresolved mode
drives an orbit that resolves it and never advertises false yaw confidence.

---

## 5. Multi-sensor channels (all fold into the SAME GN normal equations)

All extra factors accumulate into `(Id, bd)` inside `accumulate_extra`, so the engine's common-mode
Woodbury saturation de-correlates their sweep-shared error exactly as it does the depth points.

### 5.1 LiDAR first-hit range channel (`accumulate_extra`, `common/ai_belief/lidar_ray_factor.h`)

A **YOLO-independent** channel: lidar3D returns landing on the table sphere-trace *this belief's own
SDF* (`n_prims=5`), the residual measured **along the ray**. Its error mechanism is uncorrelated with
YOLO segmentation, so it attacks the **mask-erosion under-size** the mask cannot self-correct (a mask
boundary inside the true rim yields a uniformly shrunk top cloud; LiDAR returns on the true rim push the
face back out). Angular-coverage weighting `×(1−R)^p` (R = mean-resultant length of return bearings)
down-weights one-sided sweeps that would over-commit the near-square mode. `LidarPrecision = 0` ⇒
dormant (no DDS participant).

### 5.2 Coverage / traction (grow-only) (`table_belief.cpp:254–286`)

On-plane mask points the mixture ceded to **clutter** still exert a robust, **grow-only** pull on the top
slab, so a model under-covering a large mask is pulled out to explain it instead of parking with the
excess dumped to clutter for free (the escape-valve). For each point *outside* the slab (`sdf_top > 0`)
and near the tabletop plane:

```
w = coverage_precision · top_z · clut · 1/(1 + e²/cc²),   e = sdf_top(p), clut = clutter responsibility
Id += w·J Jᵀ,   bd += −w·J·e,   J = ∂sdf_top/∂θ
```

One-sided (never shrinks), self-bounded (`sdf→0` when covered), on-plane-gated (off-plane
legs/floor/contamination ignored). `CoveragePrecision = 0` ⇒ OFF.

### 5.3 Free-space / VACATE (shrink-only) (`table_belief.cpp:288–369`)

The occlusion-aware counter-force that **bounds** coverage. A LiDAR beam that traverses the solid
top-slab z-band `[H−t, H]` and continues **beyond** the far face proves that crossing is empty. Ray-vs-box
slab test gives `[t_near, t_far]`; the passed-through soft weight is `p_through = Φ((1−t_far)/(σ_surf/len))`
(a beam that *returns* inside → far face beyond endpoint → `p_through≈0` → real tabletop untouched). The
box-centre point `p_free` (inside the footprint) drives the **2D footprint SDF** (not the thin-thickness
3D SDF, whose `∂/∂w=∂/∂h=0`) toward 0 → retreats the nearest face = shrink:

```
e = sdf_footprint(p_free) < 0,   w = free_space_precision · p_through
Id += w·J Jᵀ,   bd += −w·J·e      (J = FD of the 2D footprint SDF)
```

Occupy-where-masked (coverage) + vacate-where-through (this) settle the extent where camera and LiDAR
**agree**, so coverage can no longer inflate onto clutter. `FreeSpacePrecision = 0` ⇒ OFF.

---

## 6. Range / motion / freshness as precision (no gates)

Even at zero camera motion, deprojection noise grows with distance *and* a far mask subtends a tiny
angle, so orientation becomes unobservable. This is expressed as **continuous covariance growth**, never
a range gate (`table_fitter.cpp` `compute_chain_cov` + range terms; `table_config.h:55–62`):

```
χₓₓ, χ_yy += (lat_rate·range)²         → position common-mode grows with range
χ_yaw      = (yaw_rate·range)²          → the binding term: a 7 m view's yaw gain vs a converged
                                          table drops ~50× — confirms existence, can't rotate it
χ_sz       = (size_rate·range)²         → geometry freezes afar; only existence/occupancy survives
```

**Pose-chain (localization) cov, Part B** (`compute_chain_cov`): the fit is in the room frame but its
position is conditional on the robot pose (camera→robot→room). Transforming the centre room→`zed`→room
with **zero** input covariance returns exactly the chain contribution `χ = J·Σ_chain·Jᵀ`
(`Σ_chain` = adjoint-composed RT-edge covariances, pinned to the mask capture stamp; `room_concept`
publishes the robot↔room term). This is added into `Σc` (§2.3) and, on republish, into the RT-edge
covariance (`rt_cov_add_chain`) so across-frame accumulation cannot advertise `cx,cy` tighter than the
localization actually supports — the conservative, safe direction for the controller's uncertainty
governor.

**Ego-motion downweight**: the interaction-matrix mask-motion variance `L(xₙ,yₙ,Z)` (retina
producer) enters per-point `R`; the moment channel adds `(motion_gain·motion_dotd)²` to
`moment_extra_var`, and the orientation accumulator is scaled by `evidence_weight` — so a
going-away/rotation frame cannot reshape or flip the established fit.

**Freshness**: age-inflation (§2.5) turns a silent stream into growing Σ.

---

## 7. Truncation gate — the one justified switch

`table_fitter.cpp`: a mask clipped by the image border has a chopped silhouette → a **structurally
biased** extent, not measurement noise. If `trunc_frac > AI2TruncGateFrac` the geometric update is
skipped (predict only); the instance is kept. Justified because the bias is not something a covariance
can down-weight away; the conjugate extent direction of Σ simply stays loose until a non-truncated view
arrives. (A step-bound divergence net `max_step_m` likewise rejects a corrupted single-frame outlier
step and widens Σ — a safety net, not an evidence gate.)

---

## 8. Existence & removal — log-odds, not a miss counter

Every instance carries a scalar existence log-odds `L = log P(exists)/P(¬exists)`
(`common/existence_belief/existence_belief.h`), updated each cycle by the log-likelihood-ratio of the
frame's evidence under {exists vs not}. Two interchangeable channels:

- **LiDAR occupancy carve**: a beam RETURNING from inside the object volume ⇒ occupancy evidence; a beam
  passing THROUGH to beyond ⇒ free-space (absence); a beam that stops short (occluded) or misses ⇒ NO
  evidence (`n_reached` gate). The cycle's beams share a registration error ⇒ the summed ΔL is
  **common-mode saturated** (`tanh`) to one confident observation's worth — the same discipline as the
  metric belief's Woodbury cap.
- **Mask silhouette (existence)**: the same log-odds math on projected-silhouette evidence
  (predicted-detectable pixels that ARE / ARE NOT lit).

**Absence degradation (no gate)**: "predicted-visible but absent" is weak removal evidence when the
sensor likely could not resolve the object anyway. The FREE/absence half of both channels is scaled by
`c = (range_ref/range)^power · visible_fraction` (capped at 1) — P(detect | present) falling with range
and occlusion — while OCCUPANCY stays fully informative (a distant detection still confirms). LiDAR
absence is OFF by default (`existence_lidar_absence`): horizontal beams passing *under* a thin real
tabletop read false "free" against the solid-box model, so removal is the camera silhouette's job.

Removal is a Bayesian decision `L < log(p/(1−p))` held `existence_remove_frames` consecutive cycles
(debounce), `|L|` clamped so evidence stays finite and recoverable. **OFF by default** — a mis-removal
deletes furniture; the live removal path is the merge operator (physical exclusion: collapse two
instances whose oriented footprints overlap ≥ `MergeOverlap` of the smaller). Timeout death is disabled
for tables (rigid persistent furniture); if ever enabled, the negative-information gate is the whole
contract — a removal timer must advance ONLY when the model projects into the FoV yet no mask associated,
and be **held** out-of-FoV, or an instance flickers.

---

## 9. FE-surprise attention (active-perception trigger)

Surprise = the smoothed positive gap of free energy over an **asymmetric-EMA baseline**
(`table_fitter.cpp:465–477`):

```
baseline += a·(F − baseline),   a = adapt_down (fast, if F < baseline)  |  adapt_up (slow, else)
surprise += smooth·( max(0, F − baseline) − surprise )
```

The baseline consolidates a better fit quickly (down fast) but a sustained rise — the table *moved* —
stays surprising long enough to attend (up slow). `surprise` is the attention signal that reopens active
perception on an established table without any hard "re-detect" trigger.

---

## 10. Epistemic NBV & the object-relative viewpoint affordance

### 10.1 Σ-based D-optimal next-best-view (`epistemic_planner.cpp`)

Only the four **vertical** faces (±X, ±Y in the table frame) are candidates — a floor-navigating robot
cannot see the top from above. Each face is scored by the expected posterior-entropy reduction on the
belief's full **reported** covariance (with the discrete-mode yaw entropy folded in, so a mode-ambiguous
table scores a mode-discriminating view highly):

```
for each face i:
  standoffᵢ = clamp( half_span/tan(FoV/2) + margin, min_standoff, d_obs )   (frame the face in ~70° FoV)
  Rᵢ        = σ_base² + (lat_rate·standoffᵢ)²                               (range-aware R, reuses §6)
  ΔI(i)     = Σₚ (1/Rᵢ)·Jₚ Jₚᵀ ,  pts = synthetic points on vertical face i, Jₚ = ∂sdf_top/∂θ
  face_gain(i) = max(0, ½·ln det( I₆ + Σ·ΔI(i) ))                            (RAW D-optimal, nats — the RANKING)
best           = argmax_i face_gain(i)                                       (most-informative face, by RAW gain)
epistemic_gain = p_observable · min( face_gain(best), adequacy_gap )         (SCALAR value — adequacy-BOUNDED)
emit  ranked face_gain[·]  +  hint pose v* = face_best.centre + n·standoff, heading → centre  +  epistemic_gain
```

D-optimal on the full Σ automatically targets the viewpoint that most shrinks the **dominant
uncertainty eigen-direction** (an unseen far extent, or yaw when near-square) and accounts for Σ's
cross-terms.

**Adequacy gap — threshold-free "done"** (`epistemic_planner.cpp:91–108`). Remaining information (nats)
to carry the belief down to the **consumer's** target precision `Σ*`, clamped **per DOF**:

```
adequacy_gap = Σⱼ max(0, ½·ln( Σⱼⱼ / Σ*ⱼⱼ ))
```

Per-DOF clamp (not the full `½ ln detΣ/detΣ*`) on purpose — the consumer needs *each* of `w,h,yaw` within
tolerance, so an over-resolved DOF must not mask an under-resolved one.

**The bound lives on the SCALAR value, not on the per-face ranking.** Only `epistemic_gain` (the affordance's
scalar worth) is clamped at the gap: information beyond `Σ*` is worthless to the consumer, so an
already-adequate table stops being attractive and the affordance goes quiet as `epistemic_gain → 0` — a "done"
set by the consumer's precision demand, not a tuned Σ bound. The **per-face** gains are published **raw
(unbounded)**: because each face's single-view info typically *exceeds* the remaining gap, clamping every face
at the gap makes all four tie (and pins `best` to the first face), erasing the ranking the controller needs to
choose a feasible face. So `argmax` and the published `face_gain[·]` use the raw D-optimal gain; the adequacy
bound applies only to the scalar. (`Σ*` is currently a placeholder `[0.02,0.02,0.02,0.02,0.02,0.05]`; it should
be PUBLISHED by the consuming grasp/place affordance — see §10.2.)

### 10.2 The object-relative viewpoint affordance (producer↔controller contract)

The epistemic output is an **object-relative viewpoint constraint**, not a baked world pose. The
producer (this agent) declares **what view it needs in the object frame**; the controller owns
**feasibility** (collision-free reachability, occlusion, final pose selection across all affordances via
`G = cost − epistemic − pragmatic`). Shared wire format:
`common/affordance_protocol/affordance_protocol.h` (`aff_*` attributes on the affordance node; type-level
default + per-node overrides).

The declared view is, per candidate face:
- a **ranked set of candidate faces**, each with its **raw** D-optimal gain `ΔH` in **nats** (the ranking
  the controller uses to pick a feasible face; opposite faces tie by the box's 2-fold face symmetry, so
  feasibility breaks the ±). The affordance's *scalar* `epistemic_gain` is the winning face's gain
  **adequacy-bounded** (§10.1) — ranking is raw, worth is bounded;
- a **stand-off band** (min stand-off ≤ d ≤ FoV-fit distance) so the face frames in the camera;
- a **framing-fill** target (`table_roi_fill` → advance) and **ROI-centring** error
  (`table_roi_offset` → base yaw/side), plus a **validity gate** (`table_roi_valid`);
- a **per-DOF precision demand Σ\*** — the manipulation tolerance (gripper clearance / placement margin /
  approach-cone half-angle pushed through `∂success/∂θ`) that defines when the view is adequate (§10.1);
- an **observation-stillness precondition** (`.still(v, ω)`): hold the base below these speeds for the
  look to count — a moving/rotating capture smears the deprojected mask by ≈ ω·lag·range and biases the
  fit. For a table, ω is kept tight (it fills the frame).

The controller resolves this into a collision-free pose, servos the lock-on (Policy::Servo) until the
completion predicate (fresh mask alive + confidence floor, held stable, within timeout) holds, then
consumes the affordance. A low-but-finite gain is **not** withdrawn — the node persists carrying its true
gain so the controller simply ranks it low (the belief→knowledge governor as a small gain, not a deleted
node).

---

## 11. Verification

`TableBelief::self_test()` (`table_belief.cpp:582–921`) is a torch-free Eigen unit test exercising every
factor: SDF correctness, Jacobian (coarse-vs-fine FD), full 6-DOF fit recovery from a perturbed init,
clutter responsibility, Σ finite+SPD, D-optimal NBV (w-uncertain → +x face wins), LiDAR mask-erosion
correction, coverage grow-only + on-plane gate, free-space vacate + occupancy floor, footprint-moment
yaw/extent recovery at range, and the moment grow-only guard (partial view holds, larger mask grows).

---

## 12. Config reference (`etc/config.toml [TableModel]` + `[Tracker]`; defaults in `table_config.h`)

| key | symbol | role |
|---|---|---|
| `AI2SigmaBaseM` | √R₀ | base on-surface obs noise std (m) |
| `AI2ClutterFrac` / `AI2ClutterScaleM` | ε / cs | clutter mixture prior / distance scale |
| `AI2PriorSizeStd` | σ_s | broad size prior (breaks empty-cloud degeneracy) |
| `AI2ProcessStdM` / `AI2ProcessStdYaw` | √q_m / √q_y | predict process noise (rigid+static ⇒ small) |
| `AI2CommonModePosStd/SizeStd/YawStd` | p / s / y | within-frame common-mode (Woodbury saturation cap) |
| `AI2RangeNoiseLatPerM/YawPerM/SizePerM` | lat/yaw/size_rate | static range → covariance growth (§6) |
| `AI2AgeNominalDtS` | — | mask-stream nominal period for age-inflation (§2.5); ≤0 = freeze |
| `AI2TruncGateFrac` | τ | predict-only when silhouette truncation exceeds this (§7) |
| `AI2GnIters` | — | Gauss-Newton iterations per frame |
| `MaxStepM` | — | step-bound divergence net (reject outlier frame, widen Σ) |
| `LidarPrecision` / `LidarRobustCM` / `LidarCoverageAngPower` | — | YOLO-independent LiDAR range channel (§5.1) |
| `CoveragePrecision` / `CoverageRobustCM` | — | grow-only clutter-reclaim traction (§5.2) |
| `FreeSpacePrecision` | — | occlusion-aware shrink / vacate (§5.3) |
| `FootprintMomentPrecision` / `FootprintMomentRangePerM` | Pm | 2D inertia-tensor (w,h,yaw) factor (§4.1) |
| `FeBaselineAdaptDown/Up` / `FeSurpriseSmooth` | — | FE-surprise attention baseline (§9) |
| `FootprintMomentMotionGain` / `OrientationMotionRef` | — | ego-motion downweight of moment + mode vote (§6) |
| `ExistenceRemoval*` / `ExistenceLidarAbsence` | — | log-odds existence & removal (§8; OFF by default) |
| `RtCovScale` / `RtCovAddChain` | — | RT-edge covariance upload (Σ→SE3) + chain Part-B |
| `Tracker.GateMahalanobis` / `DetectionNoiseM` | χ²₂ / R | association gate on `S = P + R²I` |
| `Tracker.BirthFrames` / `BirthMinSepM` / `MergeOverlap` | — | birth persistence / anti-dup / merge (§8) |
| `Tracker.DeathEnabled` / `DeathFrames` | — | timeout retirement (OFF for tables) |

## The static table rotates: two independent bugs (2026-07-21)

Symptom as reported: *"it rotates a static table when seen again from far away or rotating."* It was two
unrelated faults with the same visible signature. Both were diagnosed from `etc/ai2_log.csv`; the method is
as reusable as the findings, so it is recorded here for the other concept agents.

### Bug 1 — float32 cancellation in the shared Woodbury saturation (FIXED, affects EVERY concept)

`common/ai_belief/recursive_laplace.h` evaluated the common-mode saturation the way it is derived:

    Ieff = Id - Id*(Id + Scinv).inverse()*Id;

`Id` (raw data information, thousands of points at 1/R each) reaches 1e6-1e7, while `Ieff` saturates at
`Sc^-1` ~ 1e3. In float32 that subtraction's error EXCEEDS the result, so `Ieff` comes out INDEFINITE
(measured min eigenvalue -4.03 at real magnitudes; relative error 6.7e-3). `Sigma = (P0+Ieff).inverse()`
inherits it, Sigma acquires zero/negative variances — "yaw is known exactly" — and the next update teleports
the state. Near-square footprints land 45/90 deg away and stay.

Fixed by the algebraically identical Woodbury rearrangement (`saturate<N>()`), which approaches the same cap
from below without ever forming the large difference: relative error 2.7e-8, PD preserved. NOT a threshold,
NOT a model change — the mathematics is unchanged.

Live, matched viewpoints, guard off: indefinite-Sigma frames 102 -> 0 (range 3.84 m) and 33 -> 0 (1.79 m);
yaw wander 16.6 -> 3.4 deg. No regression where the bug does not fire (two matched pairs, identical to 3 dp).
`self_test` PASS for table, bottle AND chair.

Signature to look for in any concept: a `std_*` column that is EXACTLY 0.0. That means Sigma(i,i) <= 0, i.e.
the covariance has lost positive-definiteness. It is never legitimate.

### Bug 2 — the moment channel becomes CONFIDENT about a foreshortened axis (OPEN)

Independent of Bug 1: it reproduces with a perfectly well-conditioned Sigma (zero indefinite frames).

The fit is correct up to ~2.4 m and wrong past ~2.9 m, with a sharp transition and no graceful degradation:

| range  | npts | belief yaw | belief extent | moment_aniso | moment_r_yaw | verdict |
|--------|------|-----------|---------------|--------------|--------------|---------|
| 1.79 m | 7408 | +0.3 deg  | 0.95 x 0.96   | 0.038        | 105          | correct |
| 2.10 m | 5393 | +0.4 deg  | 0.96 x 0.95   | 0.029        | 212          | correct |
| 2.42 m | 3592 | -1.2 deg  | 0.97 x 0.95   | -            | -            | correct |
| 2.89 m | 2334 | -45.2 deg | 1.30 x 0.72   | 0.184        | 6.2          | WRONG   |
| 3.84 m |  831 | -45.0 deg | 1.27 x 0.67   | 0.160        | 15.4         | WRONG   |

Mechanism: grazing-amplified image-space CENSORING (mask erosion), not depth noise.

The proximal chain is: spurious footprint anisotropy -> `r_yaw = r_base / aniso^2` collapses 20-30x -> the
moment channel confidently rotates the box onto the apparent axis. What produces the anisotropy is DATA LOSS.

An earlier attribution in this file blamed along-ray depth noise stretching the cloud. That is REFUTED by the
logged raw footprint statistic (`mom_major`/`mom_minor`, added 2026-07-21):

| pose    | mom_major | mom_minor | truth side |
|---------|-----------|-----------|------------|
| 1.94 m  | 1.105     | 1.020     | 0.95       |
| 3.59 m  | 1.165     | **0.724** | 0.95       |

Three independent checks kill the noise story:
- The MINOR axis SHRINKS (1.020 -> 0.724). Additive noise can only ever GROW an eigenvalue. Decisive and
  assumption-free: a single row settles it.
- `mom_major` = 1.165 EXCEEDS the noise ceiling. At 3.59 m, sigma_d = 0.006+0.004r^2 = 0.057 m adds at most
  sigma_d^2 to lambda, capping mom_major at sqrt(12*(0.95^2/12 + sigma_d^2)) = 0.971.
- The belief FAITHFULLY TRACKS what it sees: the far belief tensor sits 0.0242 from the RAW OBSERVED strip but
  0.0727 from truth. The estimator is not being fooled; it is correctly fitting a strip.

The strip: at obliquity_cos ~0.08-0.10 an image-space mask boundary erosion of eps ~2 px maps onto the plane as
eps*(z/f)/obliquity_cos - a ~10x GRAZING AMPLIFICATION - so ~8-10 cm is eaten off each along-ray boundary while
the transverse silhouette extent survives. With the camera bearing near the square's diagonal, that transverse
span is the corner-to-corner 1.34 m, giving mom_major ~1.17-1.30 and mom_phi ~-45 deg. So the fit IS latching
the diagonal, but -45 deg is simply transverse-to-bearing, not a degenerate yaw direction or a chart artifact.
A face-on far approach should therefore NOT reproduce it (untested prediction).

This single mechanism also explains what the noise story could not: why mom_phi is pinned at -41..-45 deg at
EVERY range including the correct ones (erosion DIRECTION is set by the bearing; only its magnitude, ~z/obliquity,
grows), and why close-range aniso is small but nonzero (~5 cm erosion -> aniso 0.03-0.06, matching 0.029-0.038).

CRUCIALLY it makes the far bug and the FootprintResidual close-range regression ONE defect. The close regression
back-solves to a ~13% per-side shrink = the self_test's measured 0.880 mask erosion. FootprintResidual removes the
moment channel and with it the grow-only guard that was silently compensating erosion, without ever MODELLING the
erosion. Model it once and both ends are fixed together.

Neither existing protection engages:
- `1/aniso^2` correctly encodes "near-square => distrust yaw", but has no defence against "a far view makes a
  square LOOK elongated".
- The completeness backoff is inert (completeness 0.97-1.16, i.e. >=1, so `cs = gain*(1/c - 1) ~ 0`). It
  compares observed area to BELIEVED area, so once the belief has been dragged to 1.30 x 0.72 the areas agree.
  It structurally cannot see foreshortening.

`FootprintResidual` (a1'+a2') attacks exactly this and its header comment predicts it. Measured, alone, on the
fixed engine, error = Frobenius distance of the footprint 2nd-moment tensor from close-range truth
(symmetry-invariant, so no fold can flatter it):

| pose            | control          | FootprintResidual |
|-----------------|------------------|-------------------|
| close 1.94 m    | ERR 0.0015, E 2.03 | ERR 0.0256, E 2.47 |
| far   3.59 m    | ERR 0.0727, E 2.47 | ERR 0.0081, E 1.69 |

9x better far, 17x WORSE close — so it stays OFF. The regression is explained: `FootprintResidual` REPLACES
the moment channel, and the moment channel carries the grow-only guard ("a mask is only ever a LOWER BOUND").
That guard is what compensates YOLO mask erosion, which `self_test` measures at mask-only 0.880 vs truth
1.000. Remove the channel, lose the compensation, and the box shrinks ~12%.

SHIPPABLE FIX (agreed with a Fable second opinion, 2026-07-21; NOT DONE): FootprintResidual PLUS a shared
per-frame IMAGE-SPACE erosion nuisance eps_px, then delete the moment channel and its `kDrop` grow-only ratchet.
Grow-only is the wrong compensation: it is a hard threshold (self-flagged in table_belief.cpp), and it is a
RATCHET that helped CAUSE the far bug - it let w climb to 1.30 and could never recant. Model the bias instead of
patching it with an asymmetric update rule. eps_px is a DETECTOR property in pixels (range-independent in image
space), calibrated once from the self_test 0.880 extent ratio, with sigma ~1-2 px; its on-plane column carries
the 1/obliquity amplification, so Schur-marginalising it removes exactly the along-ray extent + aliased yaw
information the fake strip boundary injects, while transverse information from real edges survives. At close
range the same eps absorbs the uniform ~12% shrink, curing the regression. Absolute scale stays anchored by the
erosion-free LiDAR range/vacate channels. This is `s_erode` from PRECISION_AS_INFORMATION.md, which was dropped
when a2' narrowed to depth-tilt - the missing insight being that erosion lives in PIXELS and the grazing
geometry is what amplifies it into the far-range failure.
`AnisotropicR` alone is not the answer (0.0718 vs 0.0727 control = no effect): per-point noise reshaping
cannot fix a MEAN BIAS; only marginalising the per-frame SHARED nuisance can.

### Method notes (reusable)

- Measure orientation with the footprint 2nd-moment tensor `R diag(w^2/12, h^2/12) R^T`, not raw yaw. It is
  invariant under all four box symmetries, so a w<->h relabel cannot masquerade as a 90 deg rotation. Two
  wrong conclusions in this investigation came from a symmetry-blind yaw metric.
- Always report FUSED row count. A belief that is merely AGEING (mean held, no fresh mask) looks perfectly
  stable and will fake a passing result.
- Free energy adjudicates: a fit that is wrong-but-confident shows HIGHER energy. That is how the far-range
  basin (E 2.47) was distinguished from a merely uninformed fit.
- Verify a viewpoint actually triggers the fault before A/B-ing a fix there. Several viewpoints do not, and
  one clean run at a non-triggering pose proves nothing.

---

*Keep in sync with `src/table_belief.{h,cpp}`, `src/table_fitter.cpp`, `src/epistemic_planner.cpp`,
`src/table_config.h`, and the shared `common/ai_belief/recursive_laplace.h` engine. Shared agent
structure: `../CONCEPT_AGENT_RECIPE.md`.*
