# Existence / Explanation belief — shared removal + traction for all model agents

Status: DESIGN (2026-07-06). Generalises residual_concept's occupancy-carving log-odds
into a shared module used by table / chair / bottle / residual. Companion to
`table_concept/TABLE.md` (the metric belief) and `chair_concept/PERCEPTION_ASSOCIATION_PLAN.md`.

## 0. The one idea

Every model instance carries, alongside its metric belief θ (pose/extent), a **scalar
existence log-odds** `L = log P(exists)/P(¬exists)`, updated each frame by the
**log-likelihood-ratio of the frame's evidence under {exists vs not}**. Removal is a
Bayesian decision on `L`, not a miss-counter.

The key realisation is that **removal and fit-traction are the same coin**:

- **predicted-but-ABSENT** stimulus (model says there should be mask pixels / LiDAR
  returns in a visible, in-range, un-occluded volume, and there aren't) → **negative**
  evidence → eventually **remove**.
- **observed-but-UNEXPLAINED** stimulus (mask/returns present that the model doesn't
  account for — dumped to the clutter component) → the model should feel **traction**
  to grow/move (or a sibling should be born).

Both are `expected − observed` mismatches. Handling only the first gives removal;
handling both, with a shared free-space term, *also* fixes the "no-traction" failure in
`table_1.png` (below). One module, two payoffs.

## 1. What already exists (the seed — residual_concept)

`residual_clusterer.h::carve_box` + `specificworker.cpp::remove_by_occupancy_evidence`
already implement the LiDAR/occupancy half, correctly and AI-faithfully:

- **Per-beam classification** against a box volume: `e_occ` (beam returns from *inside*
  → occupancy evidence) vs `e_free` (beam passes *through* to beyond → the volume is
  empty → absence evidence). Soft, weighted by the surface-localisation σ.
- **Log-odds increment** from *physical sensor rates*, not gates:
  `ΔL = e_occ·log(p_det/p_clut) + e_free·log((1−p_det)/(1−p_clut))`.
- **Common-mode saturation**: `ΔL ← llr_occ·tanh(ΔL/llr_occ)` — one cycle's correlated
  beams count as ≈ one confident observation. The *same* "N correlated points can't
  collapse σ" discipline as the metric belief's Woodbury cap, reused for log-odds.
- **Negative-information gate**: `n_reached == 0` (out of range / fully occluded /
  behind) → **HOLD** (no update, no removal). Absence only counts where detection was
  *possible*.
- **Hollow-object guard**: while the object is actively `observed` this cycle, free
  evidence *through its interior* is suppressed (a solid box is a lossy abstraction of a
  table that is empty underneath). Occupancy always counts.
- **Decision**: integrate into `inst.occupancy_logodds`, clamp to ±`L_max`, remove when
  `L < L_remove = log(p_remove/(1−p_remove))`.

This is exactly the target mechanism — it is just (a) residual-only, (b) LiDAR-only, and
(c) box-shaped. The plan is to lift it to shared, multi-modal, any-SDF.

## 2. Shared module: `common/existence_belief/`

```cpp
namespace rc::exist {

struct SensorModel {          // physical rates, interpretable — not gates
  float detection_prob = 0.85f;   // P(detect stimulus | exists & the location is observable)
  float clutter_prob   = 0.05f;   // P(detect stimulus | ¬exists) — spurious rate
  float surface_sigma_m= 0.03f;   // localisation blur for the soft occ/free split
};

struct Evidence {  // one modality, one cycle, already common-mode saturated
  float e_occ = 0.f, e_free = 0.f;   // soft occupancy / free-space masses
  int   n_reached = 0;               // observable extent seen this cycle; 0 ⇒ HOLD
  float log_odds_delta = 0.f;
};

// A model plugs in ONLY these — the same hooks the metric belief already exposes.
struct Predictor {
  // occupied test for carve/ray-cast (SDF interior); the module ray-casts the sweep.
  std::function<float(const Eigen::Vector3f&)> sdf;         // <0 inside
  // z-extent of the solid for carve (so beams under a tabletop aren't "through").
  float z_min, z_max;
  // project the model silhouette into a camera → the pixels it SHOULD light up,
  // given FoV + range; returns the predicted-detectable footprint (or a rasteriser).
  std::function<ProjectedFootprint(const Camera&)> project_silhouette;
};

class ExistenceBelief {
public:
  // LiDAR / occupancy channel — generalises carve_box to any SDF (ray-march instead of
  // box-slab intersection). observed ⇒ suppress interior free-evidence (hollow guard).
  Evidence lidar_evidence(const Predictor&, const LidarSweep&, bool observed, const SensorModel&) const;
  // Mask channel — project the silhouette; predicted-but-empty in-FoV/un-occluded pixels
  // → e_free; explained pixels → e_occ. Occluders = other instances + room walls.
  Evidence mask_evidence(const Predictor&, const MaskFrame&, const Occluders&, const SensorModel&) const;

  void integrate(const Evidence& e);        // L ← clamp(L + e.log_odds_delta, ±L_max); HOLD if n_reached==0
  float logodds() const { return L_; }
  float p_exists() const { return 1.f/(1.f+std::exp(-L_)); }
  bool  should_remove(float p_remove) const;  // L_ < log(p_remove/(1−p_remove))
private:
  float L_ = 0.f;   // prior: 0 (born ⇒ neutral) or a small + after birth confirmation
};
}
```

Each agent's worker: build a `Predictor` from the instance's SDF + projected ROI (both
already computed — `compute_projected_roi`, the `sdf_prim` hooks), call
`lidar_evidence` / `mask_evidence` per cycle, `integrate`, and `should_remove`. This
**replaces** the scattered per-agent heuristics (chair's `roi_valid`/`expected_visible`
FoV-prune, table's merge-only, bottle's death, residual's bespoke carve) with one
consistent, tunable, unit-tested contract.

## 3. The traction complement (why this fixes `table_1.png`)

`table_1.png`: the green YOLO mask sits **abundantly outside** the orange fitted box, yet
the fit **does not react** — no traction toward the unexplained points. Cause: points
farther than `clutter_scale_m` from every surface get responsibility → the **clutter
component**, whose gradient is ≈0. So a model that explains a *subset* of the mask parks
happily and **dumps the rest to clutter for free** (the "escape-valve" degeneracy — see
the chair coverage/extent notes). The M-step has no term that says *explain the whole
observation*.

The fix is a **coverage / recall term** — unexplained observed stimulus exerts a force to
grow/move the model — but coverage *alone* is unstable (it grew chairs to 5 m: positive
feedback, cover more → grab more). The missing counter-force is exactly the **free-space
carve** above: coverage pushes the surface OUT to cover observed points; free-space
evidence pushes it IN where beams pass through empty volume. Balanced, they are the
complete objective:

> occupy where stimulus is observed **and** vacate where the volume is demonstrably empty.

So the same module that removes dead instances (free-space dominates → `L` falls) also
supplies the bound that makes traction safe (free-space caps the coverage growth). That
is the unification: **build the free-space/occupancy evidence once, get removal AND
bounded traction.** For `table_1.png` specifically the immediate remedy is the coverage
term; it is only *shippable* once the free-space counter-force exists — hence one plan.

## 4. The one honest threshold

`L_remove` (and the coverage↔free-space balance) is a genuine decision boundary — flag it
per the no-threshold rule. It is defensible: it is the Bayesian MAP/​cost-weighted removal
boundary on a properly-accumulated log-odds (with common-mode saturation so dwell can't
manufacture confidence), not a frame counter or a magic distance. Hysteresis emerges from
the accumulation itself.

## 5. Active-inference upgrade (later)

A low-|L| (ambiguous existence) instance is a prime **epistemic target**: instead of
silently removing, emit a "go look to confirm/deny" affordance through the epistemic
planner (the commit-and-gather / adequacy-gap line). "I'm no longer sure this table
exists → verify it" is the active-perception form of removal, and the natural home for
this once the passive accumulator is in.

## 6. Steps

1. Lift `carve_box` + the log-odds integration/decision out of residual_concept into
   `common/existence_belief/` (box → general SDF ray-march; keep the tanh saturation,
   the `n_reached==0` HOLD, the hollow guard). residual_concept switches to consume it
   (no behaviour change — regression gate).
2. Add the **mask channel** (`mask_evidence`): project silhouette, occluders = other
   instances + room walls, predicted-but-empty in-FoV pixels → `e_free`.
3. Wire `ExistenceBelief` into table (first), then chair, then bottle: per-instance `L`,
   fed each cycle, remove on boundary; retire the bespoke prune/death paths.
4. Add the **coverage/traction term** to the metric M-step, bounded by the free-space
   evidence — validate on `table_1.png` (model grows to cover the green mask) and on the
   chair runaway (bounded, no 5 m seats).
5. (Later) epistemic-verify affordance for low-|L| instances.
