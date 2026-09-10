# Hierarchical precision in room_concept — top-down modulation of the loss factors

> Design note. Captures the 2026-07-23 discussion (session `b8b89d18`) on turning
> room_concept's hand-wired precision gates into genuine hierarchical Active Inference:
> a higher level predicts the **precision** of a lower-level factor, and the lower-level
> residual updates that prediction. The sliding-window **boundary prior** is chosen as
> the first factor to convert, because it is the one term whose gate already closes the
> top-down loop. Nothing here is live yet; Stage 1 lands behind a config flag, default OFF.
>
> See [[precision-as-information-rewrite]], [[frame-reparenting-precision]],
> [[room-fast-rotation-track-lag]] (the FEJ fix this depends on), [[no-threshold-patches]],
> [[ai2-concept-fit-principles]].

## 0. Where we are: room_concept is already a precision-weighted factor graph

The per-tick free energy the optimizer minimizes over the pose `s = [robot_pos, robot_theta]`
is a sum of factors, each of the form ½·rᵀ·Λ·r — **one precision Λ per evidence source**:

| Factor (`compute_rfe_loss`) | Residual r | Precision Λ | Where |
|---|---|---|---|
| `loss_obs` | LiDAR point ↔ wall SDF (≈0) | `1/σ_obs²` · per-point wᵢ (heteroscedastic) | `room_concept.cpp:3641`, `room_model.cpp:84` |
| `loss_boundary` | `x_front − μ` (oldest surviving pose vs frozen FEJ point) | Schur `Λ_b` × `boundary_weight` | `room_concept.cpp:3611-3639` |
| `loss_motion` | `s − μ_pred` (dead-reckoning) | fused odometry `Λ` | `room_model.cpp:120` |
| `loss_corner` | corner landmark | anisotropic `Λ_det` | corner block |
| `loss_object` | object anchor | landmark precision | RFE loss #5 |

The **ratio** of these precisions arbitrates the estimate — no thresholds, no gates. That is
the codebase's governing philosophy (`CLAUDE.md` modeling section). The remaining hand-tuned
σ's are exactly the knobs [[precision-as-information-rewrite]] wants to *derive* rather than tune.

room_concept already contains several **primitive, single-step** versions of "a slower/higher
variable modulates a faster factor's precision" — but implemented as heuristics, not inference:

- `velocity_adaptive_weights` — motion state scales the x/y/θ motion-prior precision (`:2624`).
- ω-loosened rotation noise mid-turn — angular velocity down-weights θ precision ([[room-fast-rotation-track-lag]]).
- `boundary_weight = min(1, σ_sdf²/sdf_mse_prev)` — the **previous frame's fit quality** gates the
  marginalization prior's gain (`room_concept.cpp:2633-2638`).
- per-point range/incidence weights — geometry modulates each observation's precision.
- freshness-as-precision — stream age grows covariance ([[freshness-as-precision]]).

Each is hand-wired top-down precision control. Hierarchical AIF **replaces the heuristic with
inference**: the modulating quantity becomes a state in a higher-level generative model, and the
gate becomes a *predicted precision* with its own free-energy cost.

## 1. Why the boundary prior is the right first factor

`loss_boundary` is the FEJ+Schur **sliding-window marginalization prior** (`room_concept.cpp:3611`):
a Gaussian on the oldest pose still in the window, `½·(x_front − μ)ᵀ·Λ_b·(x_front − μ)`, where
`μ` is the frozen First-Estimates linearization point and `Λ_b` the Schur-complement precision
produced by `marginalize_oldest()` (`:3962`) when a frame drops off the back of the window. It is
what lets a bounded window behave as if it still remembers all marginalized-out constraints.

Its gate `boundary_weight` (`:2633`) is **the only existing term that already closes a top-down
loop**: a lower-level belief ("was my recent pose trustworthy?", proxied by `sdf_mse`) modulates a
lower factor's gain. So it is the cleanest place to demonstrate a *real* hyperprior rather than a
relabeled knob.

**Precondition — already satisfied:** the hyperprior update below assumes a **fixed linearization
point** (`μ, Λ_b` frozen). That is exactly what FEJ guarantees. The FEJ+Schur fix already landed
([[room-fast-rotation-track-lag]]) is therefore the enabling condition — without it, re-linearizing
`μ` each step would double-count information into the precision update.

## 2. Stage 1 — recast `boundary_weight` as an inferred precision scale (Gamma hyperprior)

Currently the effective boundary precision is `π·Λ_b` with `π = boundary_weight ≥ 0`, a deterministic
function of last frame's fit. Make `π` a **random variable we infer**; `boundary_weight` becomes its
posterior mean `⟨π⟩`.

Conjugate hyperprior (the boundary factor is a 3-D Gaussian in `x_front` with precision `πΛ_b`):

```
π ~ Gamma(a₀, b₀)
p(x_front | π) = N(μ, (πΛ_b)⁻¹)
```

Mean-field `q(x)q(π)`. The closed-form `q(π) = Gamma(aₙ, bₙ)` update is

```
aₙ = a₀ + d/2                    (d = 3)
bₙ = b₀ + ½·⟨r_b⟩
⟨π⟩ = (a₀ + d/2) / (b₀ + ½·⟨r_b⟩)          ← the generalized boundary_weight
⟨r_b⟩ = (x̂ − μ)ᵀ Λ_b (x̂ − μ)  +  tr(Λ_b Σ_x)
```

What changes vs the current clamp:

- **Evidence is the boundary factor's OWN residual `r_b`**, not the borrowed `sdf_mse`. More correct:
  `sdf_mse` was a proxy for "is my pose good"; `r_b` directly measures "does the marginalization prior
  disagree with everything else after fitting." When the SDF pulls `x_front` far from `μ`, `r_b` ↑ →
  `⟨π⟩` ↓ → the stale prior is automatically distrusted.
- **The `min(1,·)` clamp disappears.** `⟨π⟩` is a smooth `1/(1+residual)` shape, self-limiting.
  `b₀` plays the role `σ_sdf²` played (a prior scale); `a₀ + d/2` is the natural ceiling — both now in
  interpretable **pseudo-count** units ("how many prior observations of the marginalization being reliable").
- The **`tr(Λ_b Σ_x)` term** discounts the residual by how uncertain the pose posterior still is.
- The **`KL[q(π)‖p(π)]` cost is what makes it a hyperprior and not a relabeled knob**: down-weighting
  the prior now *costs free energy*; the residual must be large enough to pay for it. The current clamp
  has no such cost.

This is a two-block variational coordinate descent: the Adam loop is the `q(x)` update; the box is the
`q(π)` update. Alternate them (or run a π-step every few Adam steps).

## 3. Stage 2 — make it hierarchical (top-down `g(v)`)

### Where the higher level lives — decision: **Option A** (in-process slow hyper-state)

"Higher" means **slower timescale + broader context**, not necessarily a separate process. Three
placements were considered:

- **A — same process, second timescale (CHOSEN).** Add one more state `v` to room_concept itself — a
  scalar `map_trust` — updated on a *slow* loop (small `lr_v`) from the accumulated boundary
  prediction error, feeding `g(v)` into the fast per-frame π-update. The fast Adam loop optimizes the
  pose `q(x)`; the medium loop infers the boundary precision `q(π)` per frame; the slow loop infers the
  context `v`. No new agent, no DSR traffic, single-writer trivial. This is the truest "one-more-layer
  of the same recursive-Laplace belief," and the lowest-risk thing to validate.
- **B — the `EpistemicPlanner` owns `v`.** Promote to this once the trust belief must drive *action*
  ("go re-observe a corner *because* map-trust dropped") — the epistemic loop and the precision loop
  become one loop.
- **C — a separate DSR agent** (`localization_context` / `map_trust`) writing a predicted-precision
  attribute back down a non-RT edge, exactly the `ring_metaconcept` down-prior pattern. Only worth it
  when the *same* context must modulate several agents' precisions at once (room + table + corner);
  heaviest, and reintroduces the leave-one-out double-counting hazard.

Start at A; the code path is identical to Stage 1 except `u₀` is replaced by `g(v) = u₀ + k·v`.

### The math

Move to **log-precision** so it composes with predictive coding:

```
u = log π ,   u ~ N( g(v), σ_u² )
g(v) = u₀ + k·v            (linear top-down map; k = hier_prec_g_gain)
```

where `v` is the in-process `map_trust` hyper-state and `g` the top-down mapping. The stationarity
condition of the π-update becomes

```
∂F/∂u = ½·eᵘ·r_b  −  d/2  +  (u − g(v))/σ_u²  =  0
```

Three forces balance: the **bottom-up boundary residual**, the dimensionality term, and the **top-down
prediction `g(v)`**. `σ_u²` controls how much the higher level *lets* `π` deviate from what it predicts
(small = higher level clamps hard; large = residual dominates, ≈ what we have today with a flat top).

Close the loop: `v` minimizes **its own** free energy, `F_v = (u − g(v))²/(2σ_u²) + v²/(2σ_v²)`
(a unit-scale prior `v ~ N(0, σ_v²)` is what makes it a belief, not a free integrator):

```
∂F_v/∂v = −k·(u − g(v))/σ_u²  +  v/σ_v²
v ← v − lr_v · ∂F_v/∂v          (slow: lr_v ≪ lr_u)
```

A persistent need to suppress the boundary prior across many frames drives `v` down → `g(v)` drops
globally → the per-frame gate becomes a slow, principled **map-trust belief**. That is the same
machinery that would subsume symmetry-flip handling ([[room-symmetry-flip-investigation]]). When trust
recovers, `v` climbs back toward 0 (`g(v) → u₀`, full prior). Promote `v` to the `EpistemicPlanner`
(Option B) only once it must drive action.

## 4. Stage 3 — vectorize, then generalize to the other factors

Make `π` a **vector** of per-channel precision scales, each with its own log-hyperprior:

- `π_xy` on translation, `π_θ` on rotation; drive `g(v)` for `π_θ` from a "turning" regime state. This
  **unifies the ω-loosening and `velocity_adaptive_weights` heuristics** into one hyperprior mechanism —
  mid-turn the higher level predicts low `π_θ`, but the residual can still override if the data insists.
- The same scheme drops onto `Λ_obs`, `Λ_det`, motion `Λ`: each becomes `π_k Λ_k` with a Gamma/log-normal
  hyperprior updated by that factor's residual, and `v` becomes the shared context predicting the whole
  precision vector. **That is the [[precision-as-information-rewrite]] endgame as one recipe: every
  remaining σ becomes `⟨π_k⟩`, inferred per-factor and predicted top-down.**

## 5. Minimal first step in code (Stage 1, scalar, flag-gated OFF)

Prototype just the boundary case, no higher level yet — behavior identical to today unless the flag is on:

1. Persist a scalar state `u_b` (log-precision) across frames instead of recomputing `boundary_weight`
   from `sdf_mse` each tick.
2. After each pose optimization, take one gradient step on `u_b`:
   ```
   u_b += lr_u · ( −½·exp(u_b)·r_b  +  d/2  −  (u_b − u₀)/σ_u² )
   ```
   with `u₀, σ_u²` a fixed hyperprior for now, `d = 3`, and `r_b = (x̂−μ)ᵀΛ_b(x̂−μ) + tr(Λ_b Σ_x)`
   read straight off the boundary factor already computed in `compute_rfe_loss` (`:3636`).
3. Feed `boundary_weight = exp(u_b)` into the next frame's `compute_rfe_loss` call (`:2653`).

That alone gives the smooth, unclamped, cost-bearing gate. Swapping the fixed `u₀` for `g(v)` from the
epistemic level is then the single change (Stage 2) that turns it into genuine hierarchical attention.
The `sdf_mse → r_b` substitution is what makes the whole thing self-consistent rather than borrowing
evidence from a different factor.

### Config surface (Stage 1 + Option A)

```toml
# room_concept config.toml — default OFF, behavior identical to today
RoomConcept.HierPrecBoundaryEnabled = false   # OFF: keep min(1, σ_sdf²/sdf_mse_prev)
RoomConcept.HierPrecU0              = 0.0      # log-precision prior mean (exp(0)=1 → full prior)
RoomConcept.HierPrecSigmaU2         = 1.0      # how far the residual may pull u from g(v)
RoomConcept.HierPrecLrU             = 0.1      # π-step size (fast, per-frame)
RoomConcept.HierPrecGGain           = 1.0      # k in g(v) = u0 + k·v (top-down map)
RoomConcept.HierPrecLrV             = 0.02     # v-step size (slow hyper-state, Option A)
RoomConcept.HierPrecSigmaV2         = 1.0      # prior scale on the map_trust state v
```

## 6. Downstream use — FE-native relocalization on map-trust collapse

The `map_trust` belief earns its keep by *driving an action*: when `exp(u_b_)` collapses below a floor
for a few frames, the map no longer explains the robot, so we run the **existing** 3-stage hierarchical
`grid_search_initial_pose` (the same action `RecoveryManager` takes) — but triggered by the inferred
belief instead of a raw `sdf_mse` threshold. This is the AI-native version of "abrupt rotation → FE up →
relocalize": relocalization is an action the higher level selects when its map-trust drops. The discrete
fire still needs a cutoff, but it reads a *continuous inferred belief*, satisfying the no-raw-threshold rule.

- **Trigger** (`room_concept.cpp`, block "8b" beside RecoveryManager): `exp(u_b_) < hier_prec_reloc_floor`
  for `hier_prec_reloc_consecutive` frames (debounced, with `hier_prec_reloc_cooldown_frames`). Action:
  `grid_search_initial_pose` + `window_mgr_.clear()` + reseed hyper-state (`u_b_init_=false; map_trust_v_=0`).
  Coexists with `RecoveryManager` (kept as an independent raw-sdf backstop). Gated by `hier_prec_reloc_enabled`.
- **Rotation early-exit gap** (`nudge_map_trust_early_exit`): abrupt-rotation frames often early-exit
  (`rot_boost` widens the trust threshold), so they never update `u_b_` and a degrading rotation could
  evade the trigger. Fix: on early-exit with `|Δθ| > hier_prec_ee_dtheta_min`, apply a **fast-only** `u_b_`
  step using a surrogate residual `r_ee = (mean_sdf_pred/σ_sdf)²` (no boundary factor exists on early-exit;
  the slow `v` stays driven by real boundary evidence on the optimized path).
- **Observability:** `etc/hier_prec.csv` now carries `src ∈ {opt, ee, reloc}` and a `reloc_fired` column;
  a `[reloc]` qWarning fires on relocalization.
- **Config** (all `load_optional`, default OFF): `HierPrecRelocEnabled`, `HierPrecRelocFloor` (0.10),
  `HierPrecRelocConsecutive` (3), `HierPrecRelocCooldownFrames` (30), `HierPrecEeDthetaMin` (0.15 rad).
- **STATUS 2026-07-27: BUILT (compiles+links clean), flag OFF, unrun.** Requires `HierPrecBoundaryEnabled`
  (already ON in beta) since the belief must be live.

### Acceptance / A-B for Stage 1

- Flag OFF ⇒ byte-identical behavior to current `boundary_weight` path (regression guard).
- Flag ON ⇒ `boundary_weight` traces smooth (no `min(1,·)` chatter at the σ_sdf² boundary — the
  chatter that seeded the turn-failure ratchet, [[room-fast-rotation-track-lag]]).
- Turn/relocalization loss tail no worse than the FEJ baseline; ideally better, since `r_b` reacts to
  the prior's *own* disagreement rather than the lagged `sdf_mse` proxy.
- ⚠ Per [[ask-before-changing-running-params]]: do not enable on the live localizer without asking.
```
