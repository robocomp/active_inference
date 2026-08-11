# Perception + Association plan (cluttered scenes) — Fable-reviewed

Context: the AI2 belief fits a CLEAN single-object mask well (chair ~4–7% clutter, physical dims). The
remaining failures are UPSTREAM and dominate in clutter (a table surrounded by chairs):
- **Mask contamination** — one chair's mask merges neighbours (point count spikes 2.7k→16–28k), worse
  while moving. This is **bias, not noise**: R-inflation (ego-motion `motion_var`) can't fix a biased
  likelihood, it only slows the drift. View-geometry-dependent, not only motion.
- **Association teleporting** — 3 identical chairs ~1 m apart; a merged mask's centroid lands *midway*
  between chairs → falls in the wrong track's Mahalanobis gate → greedy argmax flips → wrong mask fed to
  GN → the instance teleports.

Guiding principle (workspace rule): **no decision thresholds** — encode effects as continuous
covariance/precision/posteriors. Fixation is the right *policy* but must **emerge from the EFE planner**
(moving-camera views in clutter honestly carry ≈0 geometric info), not be a hardwired "commit mode".
Inference weighs evidence honestly; policy seeks vantages where evidence will be good.

## §2 — Per-frame mask-validity latent `v` (the clean form of "existence-only update")
Augment each instance with `e` (Bernoulli existence; the tracker miss-counter half-has it) and per-frame
`v ∈ {clean, contaminated}`. Under `v=contaminated` the likelihood is θ-independent (uniform clutter) so
the geometry posterior = prior *by construction*; under `v=clean` it's the compound-SDF mixture. Scale
the frame's information (the Hessian added to Λ, not just R) by `w = P(v=clean|frame)`. As `w→0`,
geometry update →0 *exactly* while `e` still updates. `w` is a marginal-likelihood posterior (same
MAP-boundary pattern as `resolve_orientation`, no `if`). Feed `P(v)` from: **point-count vs predicted**
(Poisson/neg-binom around the belief+pose-predicted support size — the 6–10× spikes give overwhelming
Bayes factors, no tuned cut), the mixture's own `clutter_fraction()` (already computed), extent /
2-vs-1-component multimodality, with `motion_var`/`trunc_frac`/range as *priors* on `v`. Corollaries:
**deletes the `ai2_trunc_gate_frac` hard gate** (truncation → a term in `P(v)`); **absence annotation
falls out** (degraded frame → low `P(detect|degraded)` → log-odds contribution ≈0).

## §3 — Association (fix the statistic, then the assignment)
1. **Associate by model evidence, not centroid.** Score det↔track with the track's `mean_energy(pts,·)`
   (shape+position+yaw at once). A merged mask scores badly against *every* single-chair model → correct.
   Stop using the merged-mask centroid (a statistic of no object). ← START HERE.
2. **Soft assignment + a "merged/clutter" hypothesis.** Effective frame weight = `P(v=clean)·P(assign)`;
   near-ties → both small → no update, no flip ("when in doubt, don't move" as a posterior). Guard JPDA
   coalescence via point-level responsibility exclusivity.
3. **Endgame — explaining-away.** Fit each instance with the neighbours' models *frozen as extra fixed
   primitives*; a merged mask self-splits (each SDF claims its points, clutter takes the table).
   Per-point responsibility replaces per-mask ownership — the analogue of the table top-vs-legs fix.
   Cheap approximation: sequential (neighbours frozen).
4. **Birth requires `P(v=clean)`.** Else an unassigned merged-mask centroid spawns a PHANTOM chair midway
   between two real ones. Birth/death → log-odds accumulators on `e`.

## §1 — Fixation as emergent EFE policy
Score NBV viewpoints with the R the robot *will actually have* (motion_var on approach, ~0 on dwell) AND
**predicted contamination** (project neighbours' silhouettes from the candidate pose; overlap with the
target → low `P(clean|pose)`). Objective = **`P(clean|pose)·ΔH`** → NBV seeks *separations* (circle to
angularly-isolate the target). `.still()` executes it; dwell terminates on info-gain saturation, not a
timer.

## Composition & risks
Additive to existing machinery (`motion_var→R`, range→common-mode, `chain_cov`, Σ-NBV, `.still()`). The
one REPLACEMENT: hard gates (`trunc_frac`, greedy argmax, frame-count birth/death) → posteriors/weights.
**Top risk — self-sealing beliefs**: `P(v)` conditioned on the belief's own fit could label good masks
"contaminated" forever → keep θ-independent terms (count/motion/truncation) in `P(v)`, and let the
existence channel keep the instance alive so NBV goes and *earns* a clean view. Other risks: evidence
starvation while moving (continuous `w` vs a binary mode mitigates), JPDA coalescence (→ §3.3),
phantom births during transition (→ §3.4).

## Implementation order
**§3.1 + birth-validity** (small, immediate — kills the teleport) → **§2** (`w`-weighted frame + `e`) →
**§1** (NBV objective) → **§3.3** (structural endgame).

*Authored 2026-07-02 from a Fable design review. See CLAUDE.md (no-threshold rule) + table_concept/TABLE.md.*
