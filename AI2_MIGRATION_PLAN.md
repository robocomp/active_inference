# AI2 Migration Plan — bring `bottle` and `chair` onto the AI2 belief

Goal: replace each concept agent's legacy belief core (`torch <obj>_model` + per-DOF **diagonal**
Fisher filter + `belief_stabilizer` + `sample_queue`/RFE + `prior_store`) with the AI2 recursive
variational-Laplace belief proven in `table_concept` (full math: `table_concept/TABLE.md`). The
shared scaffolding — `instance_tracker`, `mask_ingestor`, worker merge, `<obj>_scene_graph`,
affordance/epistemic/dashboard, the producer's `mask_range` — stays; only the belief core changes.

**Locked decisions (2026-06-30):**
1. **Extract a shared engine** (Phase 1) — one templated inference core in `common/ai_belief/`, each
   agent authors only its model. No 3-way divergence.
2. **Bottle dynamics = unified** (3b) — augment the belief state with velocity for the movable
   position DOFs; `predict` is a constant-velocity transition. One filter, no side `cv_filter`.
3. Each phase is A/B-gated behind `<Obj>Model.UseAI2` so the legacy path stays runnable until the
   AI2 path is validated live.

Workspace rule in force: **no thresholds unless unavoidable; encode effects as continuous
covariance in the generative model** (`CLAUDE.md` → Modeling philosophy).

---

## Phase 1 — Extract the engine to `common/ai_belief/` ✅ DONE for table (behavior-preserving)

✅ **`common/ai_belief/recursive_laplace.h`** (header-only) now holds the object-independent math as free
templates `rc::ai::predict<N>`, `rc::ai::update<N>` (GN-MAP on full data info + Woodbury common-mode Σ),
`rc::ai::predicted_information<N>`. `TableBelief` is re-expressed as a thin **model**: it supplies only
the hooks (`sdf_prim`, `sdf_jacobian`, `responsibilities`, `apply_constraints`, `canonicalize` = the w≥h
fold, and the `transition`/`process_noise_diag`/`prior_cov_diag`/`common_mode_inv_diag` diagonals) and
delegates `update`/`predict`/`predicted_information` to the engine. Golden `self_test()` PASSES
**byte-identical** (recovered values, Σ, NBV gains unchanged); full agent builds green. Chair/bottle now
author only a model class with the same hook signatures and reuse the engine verbatim. The `transition()`
hook is `I` for table (static) and will carry the constant-velocity block for the movable bottle.

### 1.1 `common/ai_belief/recursive_laplace_belief.h` (header-only, templated on a `Model`)

The engine owns: `Σ`, `predict()`, `update(Frame)`, `state()`, `covariance()`. It is generic over
the **transition** so it serves both static furniture and movable objects:

```
predict():   θ ← F·θ ;   Σ ← F Σ Fᵀ + Q          (static: F = I)
update(Frame):
   P₀ = Σ_pred⁻¹
   Σc⁻¹ = diag( Model::common_mode_diag(Frame) )⁻¹      // per-frame shared error (chain + range + base)
   for gn_iters:                                         // MEAN — full data info (unbiased MAP)
      (Id, bd) = accumulate over points × prims of  wᵢ JJᵀ , −wᵢ J d    // w = resp/R
      θ ← θ + (P₀ + Id)⁻¹ (P₀(μ_prior − θ) + bd) ; Model::apply_constraints(θ)
   I_eff = Id − Id (Id + Σc⁻¹)⁻¹ Id                      // COVARIANCE — Woodbury saturation
   Σ = (P₀ + I_eff)⁻¹
```

### 1.2 The `Model` concept (the only creative surface per agent)

```cpp
struct Model {
  static constexpr int N;                              // DOF count (incl. velocity DOFs if movable)
  using State; State(Vec<N>); Vec<N> State::vec();     // <-> vector
  int   n_prims() const;                               // SDF primitive count
  float sdf_prim (const Vec3&, const State&, int prim) const;
  Vec<N> jacobian(const Vec3&, const State&, int prim) const;     // central FD
  std::vector<float> responsibilities(const Vec3&, const State&, float R) const;  // per-prim + clutter, Σ=1
  void  apply_constraints(State&) const;               // bounds, canonical fold, anchored DOFs
  Vec<N> process_noise() const;                        // Q diagonal
  Mat<N,N> transition(float dt) const;                 // F — identity for static; CV block for movable
  Vec<N> prior_cov_diag() const;
  Vec<N> common_mode_diag(const Frame&) const;         // Σc diagonal: base + chain + range, per DOF
};
```

`Frame` carries `points`, per-point `R`, and the per-DOF common-mode additions (chain cov, range
cov) the fitter computes — the engine never touches DSR.

### 1.3 Validation gate
- `TableBelief = RecursiveLaplaceBelief<TableModel>`; the existing `self_test()` must still **PASS**
  byte-for-byte (mean recovery + calibrated Σ). This is the golden gate for the extraction.
- Live `table_concept` two-table run shows no change vs the current build.

---

## Phase 2 — `chair_belief` (static, 8-DOF) — ⏳ belief core DONE + verified; wiring next

✅ **`chair_concept/src/chair_belief.{h,cpp}`** authored on the shared engine
(`ChairBelief = rc::ai` hooks, N=8, `transition=I`). Compound SDF `min(seat, backrest, 4 legs)`, the
3-band **part-attribution gate** (legs below the seat, seat band, backrest above — continuous, no
threshold), `canonicalize` = no-op (the backrest fixes the front → no symmetry fold). `self_test()`
**PASS**: from a realistic seed (centroid + coarse yaw, as the fitter provides) all 8 DOFs recover
(yaw 0.388 vs 0.40, sizes ~1 cm), Σ calibrated/SPD. Added to `src/CMakeLists.txt`; chair agent builds green.

✅ **Fit path WIRED (2026-06-30, builds green, UseAI2=true in config.toml):**
1. ✅ `chair_instance.h`: `ai2_belief`, `ai2_initialized`, `last_motion_var/dotd/trunc_frac/range`, `chain_cov_yaw`.
2. ✅ `chair_fitter.{h,cpp}`: `observe()` stashes slice motion/range; `run_inference_ai2` (first-frame
   centroid + z-percentile seed for cz/seat_h; R = σ²+motion_var+range_lat; chain+range → chain_cov_xx/yy/yaw;
   trunc gate; predict/update; belief→ChairState) + `log_ai2_csv`.
3. ✅ `chair_config.{h,cpp}` + `etc/config.toml`: `ChairModel.UseAI2` + `AI2*` params (mirror table).
4. ✅ `chair_scene_graph.cpp`: RT-cov from belief Σ under `UseAI2`.
5. ✅ `specificworker.cpp`: TrackView cov from belief Σ (Mahalanobis association).

✅ **Epistemic Σ-NBV + analyzer DONE (builds green):**
- `epistemic_planner.{h,cpp}`: `compute(ChairBelief, lat_rate, σ_base)` overload (D-optimal
  `½ ln det(I₈+Σ·ΔI)`, chair seat-side faces, range-aware R, normalized dom-unc log). Wired in
  `step_epistemic` under `UseAI2` (guard `ai2_initialized`).
- `scripts/analyze_chair_fit.py`: per-instance size/consistency, gates, range-vs-yaw (no fold — chair
  yaw is fully determined), tracker health. EDIT the `GT` dims to the real chair.

⏳ **Only remaining: LIVE TEST.** Run chair (UseAI2=true). Confirm the fit converges from the seed (the
belief is a local filter — a near-square seat needs a coarse yaw seed; the tracker birth provides it).
Watch the self_test-flagged wrong-yaw basin; if real chairs land yaw-flipped, add a backrest-direction
yaw seed in `run_inference_ai2`. Chair is otherwise feature-complete on the AI2 line.

### original spec ↓

State `θ = [cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h]` (N=8), `cz` anchored by the support
step (port from legacy chair → not freely fit, or fit then snap).

### 2.1 `ChairModel` (author)
- Compound SDF: `min(seat_box, back_box, 4×leg_cyl)` in the local frame, posed by `(cx,cy,cz,yaw)`.
- **Part-attribution gate** (the chair analogue of table's height gate, continuous, no threshold):
  smooth z-band membership separates **seat** (z≈seat_h), **backrest** (z>seat_h, one edge), **legs**
  (z<seat_h) so a seat-edge point isn't stolen by a leg, nor a backrest point by the seat. Backrest
  also needs a y-band (it's on one side) — a smooth `tanh` membership on local-y.
- `common_mode_diag`: base size/pos/yaw stds + chain (`cx,cy`) + range (`cx,cy` and `yaw`,`seat_*`).
- `transition = I` (static); `process_noise` small (rigid furniture).

### 2.2 Wiring (`run_inference_chair_ai2`, mirror table)
First-frame seed: centroid `(cx,cy)`, `seat_h` ← a z-percentile of the seat cluster; R = σ² +
motion_var + range_lat_var; chain + range fed via `Frame`; truncation gate; predict/update;
write-back to legacy `ChairState`. Behind `ChairModel.UseAI2`.

### 2.3 Validation
- `ChairBelief::self_test()` (synthetic chair, recover GT, Σ SPD/calibrated).
- Live: static chair + two-chair tracker run; adapt `analyze_table_fit.py` → `analyze_fit.py
  --object chair` (per-instance size/consistency + tracker health).

---

## Phase 3 — `bottle_belief` (movable, unified CV — the real design work)

State augments position with velocity: `θ = [cx, cy, vx, vy, cz, radius, height]` (N=7; `cz` anchored,
no yaw — a cylinder is rotationally symmetric). The SDF depends only on the **non-velocity** DOFs, so
the velocity rows of every `jacobian` are zero — velocity is observed only through the transition
coupling (a textbook CV Kalman embedded in the Laplace filter).

### 3.1 `BottleModel` (author)
- SDF: single vertical cylinder → `n_prims = 1`; responsibilities degenerate to `{surface, clutter}`.
- **`transition(dt)`** = constant-velocity block on `(cx,cy)`:
  `cx ← cx + vx·dt`, `cy ← cy + vy·dt`, `vx,vy` persist; identity on `cz,radius,height`.
- **`process_noise`**: CV accel noise on `vx,vy` (→ position drift), small on shape. This *replaces*
  `cv_filter.h` and the static hardening — the belief is now the single source of motion.
- `common_mode_diag`: base + chain `(cx,cy)` + range `(cx,cy,radius,height)`. No yaw DOF.
- Resting-vs-moving falls out of the filter (low velocity posterior when static); the grasp stays a
  future **known input** (re-parent under the gripper frame).

### 3.2 Wiring (`run_inference_bottle_ai2`)
- Measurement = the **fresh-frame deprojected centroid** at the **capture timestamp**
  (`room_T_zed_matrix(ts)`), per the movable contract — never the queue-dragged fit, never latest pose.
- Support pre-step: `cz` anchored to the table surface; re-parent under the support table node.
- The InstanceTracker must gate on the belief's **position** Σ (which now inflates during motion via
  the CV process noise) so a moved bottle still associates instead of spawning a duplicate.
- Behind `BottleModel.UseAI2`.

### 3.3 Validation
- `BottleBelief::self_test()`: a moving synthetic bottle (constant velocity) is tracked, Σ inflates
  under motion and tightens at rest.
- Live: the existing bottle grasp-success harness (`bottle_static_ab.sh` / grasp runs) — **no
  regression** in confirmed-lift rate; "move bottle slowly" must follow one instance (the CV win the
  legacy `cv_filter` already gave us, now native to the belief).

---

## Phase 4 — Retire legacy (after both validated live)

Legacy removal is **gated on porting the belief's remaining legacy consumers** — anything that still
reads `inst.stab.fisher_info_raw` / `sample_queue` / the torch model. There are exactly two:

1. ✅ **RT covariance — DONE for table** (`table_scene_graph.cpp:163–207`): under `UseAI2` the published
   6×6 edge cov is now mapped from the belief Σ (`vx=Σ[cx]+chain_xx`, `vy=Σ[cy]+chain_yy`,
   `vz=¼Σ[H]`, `vyaw=Σ[yaw]`, roll/pitch=`big`), not the dead `fisher_info_raw`. Generalize the same
   map into the shared `<obj>_scene_graph` write during Phase 1.
2. ✅ **Epistemic NBV — DONE for table** (built green, unit-tested; pending live validation). New
   `EpistemicPlanner::compute(belief, lat_rate, σ_base)` scores each face by the **Σ-based D-optimal**
   gain `½ ln det(I + Σ·ΔI(i))`, `ΔI(i)=Σₚ (1/Rᵢ) Jₚ Jₚᵀ` via `TableBelief::predicted_information`
   (nearest-primitive Jacobian), with range-aware `Rᵢ = σ_base² + (lat_rate·standoffᵢ)²`. Wired in
   `step_epistemic` under `UseAI2` (guarded on `ai2_initialized`). Unit test (belief `self_test` (f)):
   w-uncertain Σ → +x face wins, h-uncertain → +y face wins. **Generalize into the shared engine in
   Phase 1** so chair/bottle inherit it.

Then, once both consumers are AI2-native and all three agents validated live:

3. **Delete** `belief_stabilizer` use, `sample_queue`/`sample_queue_geometry.h`, `prior_store`, and the
   torch `<obj>_model` fit path from table/chair/bottle (keep `<obj>_model` only as a thin State/mesh
   holder if the scene-graph still needs it). Large code reduction; removes the diagonal Fisher and all
   the gates/ratchets it forced.
4. Update `CONCEPT_AGENT_RECIPE.md` to drop the legacy lineage once no agent uses it.

> ⚠ **Ordering matters:** do **not** delete the legacy code before steps 1–2 — the scene-graph and
> epistemic planner still depend on it under `UseAI2=false`, and the epistemic NBV has no AI2 source
> until step 2 lands. RT-cov (step 1) is the template; epistemic NBV (step 2) is the blocker.

---

## Ordering, risk, gates

| phase | risk | gate before proceeding |
|---|---|---|
| 1 engine extract | medium (refactor) | `table self_test` PASS + live table unchanged |
| 2 chair | low (static, mirrors table) | `chair self_test` + live static/two-chair |
| 3 bottle | high (CV unification) | `bottle self_test` (moving) + grasp harness no-regression |
| 4 retire + RT-cov | low-medium | all three live-green; controller reads honest cov |

Commit per phase (never without approval). Keep `UseAI2` A/B until a phase is validated, then flip the
default and, in Phase 4, remove the legacy branch.

---

*Authored 2026-06-30. Reference: `table_concept/TABLE.md`, `CONCEPT_AGENT_RECIPE.md` (AI2 lineage).*
