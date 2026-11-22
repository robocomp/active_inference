# GN trace — live13trace (Model.GnTrace = 1, binary 31d53767, κ ≡ 1)

Same tour as live12: fixture `three_rooms_door0`, 2 legs, **2/2**, S1–S8 OK. The world was reloaded first (owner 11-22: new
DEF names, a closet cabinet, chairs 4 and 5; bedroom geometry unchanged). Logs are in
`room_concept/estimation_scripts/crossing/objects_live13trace/` (`*_ai2_log.csv`, `*_gn_steps.csv`, `*_gn_fd.csv`).
Summary from `countable_concept/tools/gn_trace_report.py`.

## Unconverged updates (gn_conv = 0)

| instance | camera+LiDAR (gated 0) | LiDAR-only (gated 1) | accepted steps (median of 25) | final λ (median) | ρ of accepted steps (median) |
|---|---|---|---|---|---|
| bed_26 | 4/4 | 14/91 | 7 | 1.2e10 | 2.06 |
| nightstand_50 | 14/53 | 10/37 | 5.5 | 4.8e15 | 1.71 |
| nightstand_51 | 21/41 | 0/51 | 8 | 2.4e9 | 0.33 |

**Mode:** 51 of 66 unconverged updates are rejection streaks (≤ 10 steps accepted); only 1 is a crawl. λ reaches 1e40–1e87.
The undamped step that blocks the 0.1σ stop is 0.2–0.86σ (median): the solver never reaches the stationary point of its
own model.

## The finite-difference check: G is not ∇J

Median ratio, analytic over central difference, taken across θ components. "Raw" uses −bd, the gradient of the energy the
channel accumulates. "Marg" uses −g after the fresh (ego-motion) marginalisation, which is what the step uses. "All" is the
solver's joint −G_θ against data_energy along θ.

| instance, gated | cam raw | cam marg | L raw | L marg | all |
|---|---|---|---|---|---|
| bed, 0 | 0.98 / 0.54 | **0.15 / 0.008** | 1.00 | 0.99 | **0.13 / 0.001** |
| bed, 1 | — | — | 1.00 | **0.004 / 0.002** | **0.004 / 0.002** |
| nightstand_50, 0 | 1.00 | **0.04** | 1.00 | **0.36** | **0.07** |
| nightstand_50, 1 | — | — | 1.00 | **0.002** | **0.002** |
| nightstand_51, 0 | **0.11** | 0.11 | 1.00 | 1.00 | **0.19** |

(a / b = at θ₀ / at the returned θ where they differ.) FD self-consistency (h 1e-4 vs 1e-3) is 1.000 everywhere it
matters.

1. **The fresh marginalisation scales G and W, but J does not contain it.** The marginalised gradient is 3–500× smaller
   than the energy's. The shrink tracks the ego-motion variance: correlation of log(G/FD) with log(motion_var) is −0.65
   (bed) and −0.83 (nightstand). LiDAR-only rows are mostly moving rows (motion_var median 1e-4 to 5e-4, against 1e-9 to
   1e-7 for camera+LiDAR rows). It also bites the camera at tiny motion, because I_cam is 1e6–1e7, so I·Σf ≫ 1 even at
   Σf ≈ 1e-8. The solver solves for the minimum of a flattened (marginalised) model and accepts against the unflattened J.
   Steps aimed at the model's minimum raise J, so they are rejected and λ explodes. The header comment on
   `persistent_data_energy` ("Σf is the ego-motion share, zero while parked, and enters the step only") is wrong in
   practice: Σf is never zero, and the camera's information makes even a tiny Σf decisive.
2. **A second, camera-only mismatch on nightstand_51:** its RAW camera gradient is 0.11 of the energy's, with no
   marginalisation involved. The per-point gradient (slope-clamped finite-difference SDF Jacobian, `jac_slope_clamp` 5,
   plus the top/wall/footprint blocks) does not match the mixture energy there. Cause not traced.
3. **The LiDAR channel is consistent** (raw G/FD = 1.000). The GN curvature W is 1.0–1.9× the FD curvature, and the FD
   curvature is negative on 9–46 % of components (non-convex beam energy). This is secondary.

## Candidate fixes (not applied; owner to choose)

- **A (recommended): make the fresh share a per-frame latent inside the joint state.** Give it a zero-mean prior Σf each
  frame (ρ = 0, a new source), as the persistent latents already handle a new source. J then contains its prior, so G = ∇J
  by construction and the separate `marginalise_fresh` goes away. The cost is a larger state.
- **B: put the same marginalisation into J.** Profile the fresh offset u inside each channel's energy:
  min_u E_ch(θ + u) + ½uᵀΣf⁻¹u, with one GN step for u*. This is equivalent to A at the optimum and keeps the state size, but
  it is more code in the energy.
- In either case, trace the nightstand_51 camera mismatch (2) separately before trusting the camera channel: test the slope
  clamp and the blocks one at a time with the same FD harness.
