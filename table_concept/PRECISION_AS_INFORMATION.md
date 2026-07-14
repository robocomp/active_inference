# Precision = Information — a reformulation of table_concept's yaw & existence channels

> Theoretical design doc (2026-07-13). Diagnosis from live ai2_log.csv (yaw rotation on re-acquisition +
> spurious removal of a distant table); reformulation authored by Fable after full source ground-truthing
> (TABLE.md, recursive_laplace.h, table_belief.{h,cpp}, table_fitter.cpp, table_existence.cpp,
> existence_belief.h, table_projection.cpp). This replaces symptom-patching with derived precisions.

## Verdict: one disease, four faults

Both failures are "update precision ≠ Fisher information of the observation about the updated quantity."
But the mechanism is NOT "the Woodbury fails to cap yaw" — the Woodbury operator `Ieff = (Id⁻¹+Σc)⁻¹ ⪯ Σc⁻¹`
caps yaw information ~4 orders of magnitude per frame, as designed. The real faults are upstream:

- **A — misspecified per-point noise (bias, not variance).** Every point carries a scalar isotropic residual
  variance R. The true deprojection noise is violently ANISOTROPIC: pixel noise → transverse `(z/f)σ_px`,
  depth noise → along-ray `σ_d(z)`. At grazing incidence (camera at tabletop height, ray·ẑ≈0.06) the along-ray
  error lies almost IN the tabletop plane, so the observed strip's principal axis is rotated toward the ray →
  the isotropic-R likelihood reports a razor-sharp but BIASED yaw. Correct observed yaw info:
  `I_yaw = Σᵢ (∂dᵢ/∂yaw)²/(nᵢᵀΣᵢnᵢ + σ_model²)`, which collapses ∝cos²θ geometry — COMPUTED, not tuned.
  The six yaw gains are hand-drawn contour lines of this one pushforward.
- **B — mean exempt from saturation under a weak prior.** When P₀→0, the mean step = full ML step regardless
  of Σc (intentional, for recovery). Re-acquisition is exactly the weak-prior regime (aged/predict-only'd), so
  the first grazing frames hand the mean the biased ML yaw of Fault A at ~full gain.
- **C — frames treated as independent, but the common mode is persistent.** The Woodbury redraws the common
  mode i.i.d. per frame; the real shared error (pose, mask-boundary, foreshortening) has correlation time = the
  viewpoint DWELL. k frames from one standpoint deliver k·Σc⁻¹ → the per-frame cap is laundered into an
  unbounded total. (This is the SAME fault the existence 15-frame debounce compensates for.)
- **D (yaw) — discrete symmetry as branch relabeling.** 87°≈90° = a `swap∘r_{π/2}` representative + 3° within
  branch. C2v symmetry is arbitrated by THREE conflicting deciders (canonicalize fold / resolve_orientation
  clamp / moment label). A scalar-yaw chart cannot represent the quotient; every bookkeeping failure prints as
  a 90/180° "rotation". (The CSV `dyaw_points` attribution can't even distinguish a fold from a real rotation,
  because it's logged AFTER canonicalize.)

Note: TABLE.md §2.2 is STALE — it says the mean uses full data info; the code was changed to the
Schur-consistent form (`beff = bd − IdMinv·bd`). Fix the doc.

## Reformulation

> ⚠ CORRECTION (2026-07-13, after implementing + measuring): (a1) below as a **scalar per-point R** does NOT
> work for a TOP-PLANE observation (a tabletop mask). Measured: aniso grazing yaw-info goes UP ~2× vs isotropic,
> because yaw on a top plane lives ONLY on the lateral boundary (weighted by ly²) and the side edges ⊥ the ray
> keep tiny transverse R → retain full yaw info. The top-slab 3-D factor barely engages yaw (∂sdf_top/∂yaw=ly,
> nonzero only at the rim); the yaw is really driven by the moment channel, and the along-ray in-plane smear
> passes straight THROUGH the z-only top-slab factor. The corrected fix for a top-plane observation (supersedes
> a1) is **(a1′)+(a2′) landing TOGETHER**:
> - **(a1′) 2-D footprint residual** — route tabletop points to the exact 2-D rectangle SDF `r=sdf_fp(q;cx,cy,w,h,yaw)`
>   (the lambda already coded in the vacate branch) with variance `n_fpᵀΣ_2D n_fp + σ_model²`, `Σ_2D` = the IN-PLANE
>   2×2 block of the deprojection cov (`trans²(I−r̂∥r̂∥ᵀ)+σ_d²r̂∥r̂∥ᵀ`, r̂∥ = horizontal ray). Near/far edges
>   (n_fp∥r̂∥)→σ_d² downweighted; side edges (⊥)→trans² retained. This models yaw AND the bias where they live, and
>   **REPLACES the footprint-moment channel** (it IS the moment done with correct per-point covariances — keeping
>   both double-counts).
> - **(a2′) shared per-frame DEPTH-AFFINE nuisance b=(δz_bias, δz_scale)** marginalized on the footprint residual
>   (Schur form of a2). THIS is the yaw lever: under the grazing ray-FAN a common depth bias/scale maps to an
>   in-plane SHEAR that aliases into yaw; marginalizing (b,s) removes exactly the aliased yaw info (large at
>   grazing where ∂φ/∂lx is large, ~0 top-down). Per-point INDEPENDENT σ_d re-averages away over N≈3000 and does
>   NOT fix the bias — the SHARED affine is the fix. (Mask-boundary erosion s is an EXTENT nuisance → helps w,h,
>   not yaw.) So Stage 1 does NOT stand alone; corrected-1 + 2 land together on the footprint residual.
> - Honest ceiling: for a foreshortened/sparse/occluded top sliver, yaw is GENUINELY near-unobservable → the
>   right outcome is a large honest σ_yaw (the weighted factor reports it by construction) + the quotient chart
>   holding the ambiguity + NBV driving to a crisp-side-edge view. Don't extract yaw the geometry lacks.
> Discriminating self_test: true-yaw=0, grazing cam, per-point ray noise + a COMMON per-frame depth SCALE error;
> assert (A) I_yaw(topdown)/I_yaw(grazing) ≥ ~10 with 1+2; (B) 200 frames drift <3° WITH the shared-affine
> marginalization but >15° WITHOUT it (proves the shared affine, not per-point variance, is the lever); (C)
> elongated table still converges; (D) near-square σ_yaw ≥25°, no 90° jumps (quotient chart).

**(a) Pose/yaw precision = correctly-marginalized observed information.**
- (a1) [SUPERSEDED for top-plane by (a1′) above] Anisotropic per-point R projected on the SDF normal: `R_i = nᵢᵀΣᵢnᵢ + σ_model²`,
  `Σᵢ = (zσ_px/f)²(I−r̂r̂ᵀ) + σ_d²(z)r̂r̂ᵀ`. Scalar-per-point-per-prim → engine shape unchanged. Kills the 6
  yaw gains (they become redundant, not "removed and hoped for").
- (a2) Common mode as a physical nuisance b=(δx,δy,δψ,s_erode) with its own Jacobian Gᵢ=∂dᵢ/∂b, marginalized
  by Schur complement (small generalization of the current Woodbury). Σ_b from the ALREADY-computed pose-chain
  cov + mount, NOT tuned. The pushforward C automatically contains the range/obliquity dependence → χ_yaw,
  χ_lat, χ_size stop being authored per-meter rates.
- (a3) Cross-frame correlation: drive b's random-walk process noise by EGO-MOTION (odometry increment).
  Stationary robot → shared b → joint info saturates over the dwell; robot moves → b decorrelates → fresh info.
  One physical constant (decorrelation length ξ_ref) replaces the motion gains + debounce. AInf-native
  "commit-and-gather": new information requires a new viewpoint.
- Moment factor: propagate per-point Σᵢ through the inertia tensor (delta method, closed form) → a grazing
  sliver reports its honestly-enormous principal-axis variance. Grow-only kDrop → compare against the
  PREDICTED moment of the VISIBLE part of the model footprint (censored prediction), so partial views are
  explained, not vetoed; legitimate shrink returns to the mask channel.

**(b) Existence = detection-model log-odds.**
The decisive channel: **carve the ZED DEPTH image exactly like the LiDAR carve.** Per predicted silhouette
sample with predicted range r̂ vs measured depth d:
- `d≈r̂` → OCCUPANCY (geometry confirms the table EVEN WHEN YOLO fails to label it — fixes Failure 2 outright).
- `d>r̂+kσ` → passed through → genuine absence.
- `d<r̂−kσ` → OCCLUDED by real geometry → HOLD (the depth sensor IS the occlusion oracle; no wall catalogue).
- invalid → no evidence.
YOLO absence stays only as a weak object-level Bernoulli with `pd(g)=v·q(A_px)` (v=visible fraction from the
depth carve; q=measured YOLO recall vs projected area, or a Beta(α,β) learned online → "absence of evidence is
evidence of absence only to the degree you know your detector"). Clock ΔL by ego-motion (a3). Debounce → ≤3
numerical guard.

## The box's discrete symmetry — estimate on the quotient

State θ′ = [cx, cy, H, s, a₁, a₂] where the footprint 2nd-moment `M = R(ψ)diag(w²/12,h²/12)R(ψ)ᵀ` maps to
`s=(w²+h²)/24, a₁=((w²−h²)/24)cos2ψ, a₂=…sin2ψ`. ALL FOUR representatives map to the same point → no fold, no
flip, no accumulator, no mode-entropy patch — the bug class is unrepresentable. Moment factor becomes LINEAR.
Near-square ambiguity is automatic: `var(ψ)≈Σ_a/(4|a|²)` diverges smoothly as a→0 (the hand-built
`p(1−p)(π/2)²` inflation falls out as the small-|a| limit). SDF hooks use the chart `(w,h,ψ)(M)`. Caveat: M
over-quotients the exact square (corner phase, a measure-zero 4ψ quantity observable only from legs/corners,
irrelevant to every current C2-symmetric consumer) — don't let it block the move.

## Knob ledger: ~15 tuned → ~6 physical

ELIMINATED: kObliquityYawGain; ai2_range_noise_{yaw,lat,size}_per_m; obliquity_moment_gain;
footprint_moment_completeness_gain(+min); footprint_moment_range_per_m; footprint_moment_motion_gain;
footprint_moment_precision; grow-only kDrop; kYawScale2 fold + resolve_orientation clamp +
orientation_motion_ref + mode-entropy; existence_absence_range_{ref,power}; observed-guard;
existence_lidar_absence; existence_remove_frames(→≤3); existence_detection_prob(const→q(A_px)).
REMAINS (legitimate): σ_px, σ_d(z) (datasheet); Σ_chain (already produced); ξ_ref (one constant); q(A_px) +
clutter rate (measured/learned); mixture ε/clutter_scale/Q (generative priors); p* (task utility); the
truncation gate + step-bound net (the two already-justified switches).

## Staged implementation (each: flag-gated, self_test, strictly deletes a tunable)

1. **Anisotropic per-point R** (kills yaw gains 1–4). Add ray dir/range to TableFrame; R_i from FD normal +
   Σᵢ. Self-test: grazing view yaw info drops ≥20× vs top-down; 200 grazing frames move yaw <3° with ALL
   obliquity/range yaw knobs = 0; top-down convergence unchanged.
2. **Nuisance-Jacobian common mode** (kills authored χ; sets up a3). b=(δx,δy,δψ,s_erode), 6+4 Schur. Σ_b from
   chain cov. Ego-motion-scaled Σ_b⁻¹. Self-test: rigid 5° cloud bias over 50 stationary frames → σ_yaw never
   drops below bias, mean <1°; "move robot" → tightens+centers.
3. **Quotient chart** (kills the flip/fold bug class). θ′ with (s,a₁,a₂); delete canonicalize symmetry logic /
   resolve_orientation / flip_evidence / mode terms. Self-test: 500 alternating edge-on near-square views → 0
   readout jumps >20°, σ_yaw ≥25°; elongated table converges as before.
4. **Censored moment factor** (kills moment gains 5–9). Predicted-visible footprint moment + delta-method cov.
   Self-test: trimmed/sliver hold, larger grows, genuinely smaller now SHRINKS from mask alone.
5. **Depth-carve existence + honest detection model** (kills removal knobs 11–15). Depth image into
   compute_silhouette_existence; classify occ/through/occluded/invalid; YOLO = weak Bernoulli pd=v·q(A_px);
   ego-clocked ΔL. Self-test: table@4m before 6m wall, zero YOLO masks, 500 frames → L stays >0; deleted table
   → L crosses boundary; occluder@2m → HOLD; stationary → L bounded not railing.

Order: 1–2 remove the CAUSE of rogue mean steps before 3 removes the machinery papering over their symmetry
appearance; 5 is independent/parallel. Every stage deletes code+config, adds no tunable.
