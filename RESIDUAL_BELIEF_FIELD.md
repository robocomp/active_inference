# Residual concept as a continuous belief field — planning over belief, not geometry

Status: DESIGN (2026-07-09). Direction note for `residual_concept` (the null-hypothesis
safety layer). Companion to `EXISTENCE_BELIEF_PLAN.md` (shared removal log-odds) and the
AI2 belief docs (`table_concept/TABLE.md`). No code yet — captures the argument and
a verifiable, non-regressing migration path so the decision survives context resets.

## 0. The one idea

The `residual_concept` occupancy grid is *already* a probabilistic field — a **factorised
(per-cell-independent) Bernoulli posterior** P(occ | x), stored as log-odds. The gap the
project actually wants closed is at the **read-out**: `occupied_components()` collapses that
posterior into a hard occupied/free set → connected components → convex hulls, and hands the
planner a *geometric model*. The posterior — above all its **uncertainty** — is discarded
before the planner sees it.

"Plan over belief, not geometry" is therefore **two separable moves**:

1. **Stop collapsing.** Let the planner consume the continuous P(occ | x) and score a path by
   its **collision probability** (chance-constrained / risk-aware planning). No representation
   change — the grid's Bernoulli field is a legitimate posterior.
2. **Upgrade the field** to one carrying a *calibrated variance*, so the planner also gets a
   principled **epistemic** term (go look where the field is uncertain).

Do (1) first (cheap, de-risks the planner). Do (2) only if the epistemic term earns its cost,
and do it **behind the same field interface** so the planner never changes and safety never
regresses.

## 1. Why this is the active-inference-aligned move

With a field posterior that has a variance channel, Expected Free Energy over a candidate
trajectory decomposes natively:

```
G(path) =  pragmatic (reach the goal / preferred outcome)
         + risk       ( ∫ P(occ | x) over the swept robot footprint )      ← collision probability
         − epistemic  ( expected reduction in field variance by going there ) ← "unexplored" pull
```

The current grid gives **risk** (via P_occ) but a *poor* epistemic term: `log-odds = 0`
conflates "never observed" with "balanced hit/miss evidence." Active inference wants those
distinguished — the unexplored is exactly where epistemic value lives. A field that carries a
real posterior variance separates them cleanly, and the null concept becomes a proper
generative model P(measurement | latent occupancy field) that the planner minimises EFE over.

## 2. The techniques, judged for THIS role (safety null-hypothesis layer)

Safety constraints that any representation here must respect:

- **Completeness.** A single real return must raise collision probability *somewhere the
  planner looks*, over a bounded support. (The grid gives this trivially: one hit → cell P up.)
- **Unknown ≠ free.** Occluded / never-observed space must not read as safe.
- **Association-free.** No per-object identity — the failure mode that sank the old box tracker
  (fitting persistent parametric objects to identity-less residual points → overconfident σ,
  teleporting instances).
- **Forgetting.** Dynamic obstacles must clear in seconds (bounded memory).

### Hilbert maps / GP occupancy maps — RECOMMENDED
Kernelised logistic regression in a random-Fourier-feature space (Ramos & Ott). Continuous,
differentiable **P(occ | x) with a posterior variance**, updatable online, association-free.

- It is the **continuous generalisation of the grid we already have** — the log-odds grid is
  its piecewise-constant, kernel-less special case. Same hit/miss evidence stream; add a
  spatial kernel and the field gains smoothness + variance.
- The kernel **fills gaps for free** — the grazed-tabletop LiDAR ring closes without ad-hoc
  inflation.
- Connects to the pose-covariance thread (§3): the lever-arm localisation uncertainty enters
  as **heteroscedastic input noise** on the training points — the natural place for it.
- The `self_test` completeness property is provable but needs care on the length scale (below).

### Mixture of Gaussians over occupied points — AVOID for this role
A GMM over returns is a **density of "where stuff is,"** with latent component assignments —
i.e. the object-identity / data-association problem that broke the box tracker. It has no
natural representation of *free* space or the *occluded/unknown* distinction. It is an
object-level generative model, not a safety occupancy posterior. (It may be right for a
*specialist* concept; wrong for the null layer.)

### Gaussian splatting / NeRF-style — AVOID for this role
Optimises anisotropic Gaussians + opacity to **render** observations. It is a geometry +
appearance representation, not a calibrated occupancy posterior with unknown-space semantics,
and proving the completeness/safety guarantee over an opacity field is hard. Overkill for a
null-hypothesis safety layer. (Potentially interesting later for dense reconstruction, not
for collision-risk planning.)

## 3. Unifying framing — the residual is a residual *in field space*

Model the **total** occupancy field as

```
P_total(occ | x) = combine( Σ_k  P_specialist_k(occ | x) ,  P_null(occ | x) )
```

where each specialist (table/chair/bottle/floor/walls) predicts an occupancy field over its
support, and **P_null absorbs whatever the object models don't explain**. The residual then
lives in field space, not as a subtraction on extracted geometry — cleaner, and it keeps the
safety layer association-free. This is the field-space version of the "explained-cell" mask
already used at read-out (`CellExplained`), promoted from a boolean gate to a predicted field.

## 4. The pose-covariance thread folds in here

Separately established (2026-07-09): the grid currently integrates LiDAR points at their
**point-estimate** room coordinates and ignores the robot→room localisation covariance. The
lever arm — a yaw uncertainty σ_ψ displaces a return at horizontal range r by ≈ r·σ_ψ (≈9 cm
at r=5 m, σ_ψ=1°, larger during rotation) — means far hits are marked too crisply and free
space is cleared too confidently.

Good news: `room_concept` **already publishes** this covariance —
`RoomSceneGraph::write_robot_room_rt` Jacobian-transforms the localiser's SE2 3×3 into the
robot→room direction and writes it into the RT edge's covariance block via the timestamped
`insert_or_assign_edge_RT` overload. `residual_concept` already instantiates
`InnerGaussianAPI` (`specificworker.cpp`). So consuming it is **consumer-side only**.

- **In the grid (interim):** transform each hit through the chain *with covariance*
  (`InnerGaussianAPI.transform_point`) → room-frame Σ; **splat** the hit's occupied evidence
  over a ~k·σ kernel (crisp when well-localised, blurred + larger when uncertain — the
  safer-clearance behaviour) and **down-weight miss-clearing** as σ grows (uncertainty must
  never erase an obstacle). Model-level covariance, not a threshold — same pattern as the
  ego-motion mask downweight and the distant-view common-mode covariance.
- **In a Hilbert map (target):** the same Σ is just heteroscedastic input noise on the
  training points. The interim splat *is* a discrete approximation of it — so the covariance
  work is not throwaway; it is the grid-shaped version of the field-shaped goal.

## 5. Caveats to weigh before committing to §0 move (2)

- **Kernel length-scale is safety-critical.** It sets the smallest resolvable obstacle and how
  far one hit's evidence spreads. Too smooth → a chair leg vanishes; too sharp → lose the
  gap-filling. Tie it to the smallest obstacle we must not miss.
- **Forgetting / dynamics.** The grid forgets via the log-odds clamp; a Hilbert map needs an
  explicit decay / sliding-window mechanism.
- **Completeness proof is harder** on a smooth field — keep a `self_test` asserting a single
  return raises P above the planner's risk threshold over a bounded support.
- **The planner is the real work.** The field is the enabler; the payoff (and the effort) is
  the chance-constrained EFE trajectory evaluation that consumes it.

## 6. Migration path (each step verifiable, none regresses safety)

1. **Expose the continuous field.** Publish P(occ | x) (not just the thresholded occupied set)
   from the grid; keep the current hard read-out in parallel.
2. **Plan over belief.** Planner scores paths by collision probability integrated over the
   swept footprint instead of polygon intersection. → "planning over belief" achieved on the
   representation we already have.
3. **Add pose covariance** (§4 interim splat) — improves the same field, still a grid.
4. **Swap grid → Hilbert map behind the field interface** — planner unchanged; gains a
   smoothed field + variance channel; wire the epistemic EFE term.
5. **Field-space residual** (§3) — promote `CellExplained` from a boolean gate to specialist
   predicted fields.

Steps 1–3 stand on their own and are worth doing regardless of whether we ever take step 4.
