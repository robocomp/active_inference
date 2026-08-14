# cabinet_concept — wall-keyed identity re-key (implementation plan)

**Status:** proposed, awaiting approval. Countertop-as-evidence step already landed (uncommitted).
**Source:** Fable strategic review, 2026-07-24. Goal: stop the split→birth→associate→merge churn by
removing the discrete-structure guessing, not by adding more gates.

## The problem this fixes (one sentence)
Run identity is currently *rediscovered every frame* by three independent hard estimators — the
mask splitter, the tracker, the merger — kept consistent only by χ²-style gates; a gate defended by
an unconverged covariance always eventually misfires, which is the single root cause behind every
patch so far (yaw-gate, z-gate, birth-z, `ai2_initialized` merge gate, 0.8 m near-parallel fusion).

## The idea (one sentence)
A wall-anchored kitchen run is uniquely identified by **(wall_id, tier)** using the *trusted, nominal*
room polygon (the Step-0 circularity check passed), turning an open-ended model-selection problem into
a finite lookup: identity by construction, association by per-point routing, no merge test at all.

---

## Design

### 1. Identity = (wall_id, tier)
- `wall_id` = canonical, collinear-merged room-wall segment id (the fitter already has
  `rebuild_wall_ids()` / `nearest_wall()` producing canonical ids — reuse).
- `tier` ∈ {Base, Wall} — already a discrete latent decided by `resolve_tier()`.
- An instance's key is `(wall_id, tier)`. Consequences that fall out **for free**:
  - Two fragments of the same wall run **are the same instance** → the union-interval fusion becomes
    the *definition of the update*, not a gated `collinear_merge` operation. **Delete `collinear_merge`
    and `merge_overlapping_instances`.**
  - Perpendicular arms have **different wall_ids** → they can never fuse. **Delete the yaw parallel
    gate + the `ai2_initialized` merge gate.**
  - A second instance on an occupied `(wall_id, tier)` cell **cannot be born** → the phantom duplicate
    (d=1.56, 0 pts) is impossible. **Delete the birth churn.**

### 2. Association = per-point soft routing (delete the splitter)
- For each cabinet/counter mask point, compute responsibility over the finite set of `(wall_id, tier)`
  cells from **lateral distance to the wall line** and **z vs the tier band** — both strongly observed.
  Same soft-responsibility (box-vs-clutter) form the belief already uses; NOT a hard argmin.
- A U-shaped mask decomposes into per-wall point sets automatically → **delete
  `split_lshaped_cabinet_masks` + `cabinet_lshape_split.h` + its 3 thresholds
  (`min_arm_pts`, `bin_m`, `arm_halfwidth_m`).**
- The tracker degenerates to a lookup for wall-anchored runs → **delete `run_position_cov`,
  `tracker_z_gate_m`, the birth-z centroid seed.** The generic `rc::InstanceTracker` survives ONLY for
  the free-standing/island mode (§4).

### 3. Reparametrize to the wall chart (gauge-fixing)
Represent a wall-anchored run in the wall's own frame:

    θ_wall = [t0, t1, δ_lat, δ_yaw, d, z0, z1]      (t along the wall from corner A)

- **Flush** = tight prior on `δ_lat` (keep the {flush, free} mixture as the anchor-mode latent so an
  island degrades honestly). WT1-not-reaching-wall becomes a strong prior, not a weak penalty.
- **Yaw** = wall direction + `δ_yaw`, tight prior → retire `accumulate_axis_alignment`,
  `wall_parallel_precision`, `room_axis_capture_rad`, and the seed's `aniso>0.10` yaw-commit gate.
- **End caps** = `t0, t1` live in `[0, wall_length]` → the grow-hinge / retract-hinge become **intrinsic
  one-sided priors at the segment ends**. Through-wall growth (open issue C) is eliminated *by
  construction*, replacing the extrinsic `flush_weight`-scaled segment terms in `accumulate_extent`.
- Retire `nearest_wall(back_centre())`'s **per-frame argmin** (a hidden hard switch with positive
  feedback — the source of the oblique-drift + impossible-d) → wall_id is a *persistent* latent.

### 4. Discrete decisions via one ΔlogZ engine
Generalize `resolve_tier()` (sequential bounded log-evidence, MAP-at-zero) into the single decision
engine for every remaining discrete latent:
- anchor mode {back-flush, end-abut (peninsula), free-standing},
- island birth (open-ended, generic tracker) — a thin-evidence sliver's small per-frame ΔlogZ
  **defers commitment automatically**, deriving the birth debounce instead of counting frames.
- Also make `accumulate_extent`'s precision ∝ effective weight sum (remove the `sw.size()<8` hard
  return) so a sliver contributes weak-but-nonzero extent info rather than toggling.

---

## What gets deleted vs kept
**Delete:** `split_lshaped_cabinet_masks`, `cabinet_lshape_split.h`, `collinear_merge`,
`merge_overlapping_instances`, the `ai2_initialized` merge gate, `run_position_cov`,
`tracker_z_gate_m`, birth-z centroid seed, `nearest_wall` per-frame argmin, `room_axis_*`, seed
`aniso` gate. (Several are patches *I added this week* — this removes them, net-negative LOC.)
**Keep (legitimate per CLAUDE.md):** `max_step_m` divergence net, `kMinExtent`/NaN sanitizing,
the whole per-run generative model (`cabinet_belief`), `resolve_tier` machinery, the countertop
ingestion just landed. **Re-examine:** `ai2_trunc_gate_frac` (a truncated mask is a censored span the
extent factor already tolerates).

## Staging (each stage builds + runs green before the next)
1. **Wall chart + intrinsic anchoring** — add `θ_wall` parametrization and the intrinsic flush/yaw/
   end-cap priors to `cabinet_belief`; keep the existing pipeline feeding it. Validate WT1 reaches the
   wall and no run grows through a wall. (Lowest risk; pure model.)
2. **Persistent wall_id latent** — replace `nearest_wall` per-frame argmin with a per-instance
   ΔlogZ-accumulated wall_id. Validate oblique-drift + impossible-d gone.
3. **Re-key + per-point routing** — introduce `(wall_id, tier)` identity + soft routing; delete the
   splitter and the merge stage. Validate the U → 3 base runs + upper, zero merge/birth churn.
4. **ΔlogZ engine unification + island mode** — generalize `resolve_tier`; route free-standing runs
   to the surviving generic tracker. Validate the apartamento island case.

## Risks
- `rebuild_wall_ids()` canonicalization must be stable frame-to-frame (it feeds identity now). Verify.
- Free-standing/island runs have no wall_id → must fall through to the generic tracker cleanly (the one
  place open-ended birth still lives). Don't let §3 strand them.
- The room polygon must remain nominal-and-trusted; if room_concept ever starts fitting it to returns,
  the whole key is compromised (revisit the Step-0 circularity check before shipping).

## Related memory
`[[cabinet-lshape-split-merge-fix]]`, `[[cabinet-concept-run-model]]`, `[[ai2-concept-fit-principles]]`,
`[[no-threshold-patches]]`.
