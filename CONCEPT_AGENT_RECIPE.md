# Concept-Agent Recipe — generating a new `<obj>_concept`

Concise, technical generation contract. Companion to `CONCEPT_AGENT_PATTERN.md` (rationale/history).
This file is the spec a coding agent follows to emit a new, operational concept agent from an
**object spec** (Step 0) + the closest reference skeleton. `<obj>` = lowercase class, `<Obj>` = PascalCase.

Everything except the **belief model** (the SDF + its hooks) and a handful of spec-driven wiring is
mechanical copy + token-rename. The object spec is the only human input.

---

## The belief core — shared recursive-Laplace AI2 (the ONLY lineage)

`table_concept`, `chair_concept`, and `bottle_concept` are now **all** the same pattern; the old
torch/Fisher/stabiliser/sample-queue lineage is fully removed. A new agent clones the closest of the three
and authors one `<obj>_belief`. (Full math in `table_concept/TABLE.md`.)

Each agent carries a single `<obj>_belief.{h,cpp}` — **pure Eigen, no torch, no DSR, `self_test()`able in
isolation** — holding a compound-SDF generative model + **soft per-point mixture responsibilities** (EM,
replaces `min()`/height-split) + a **recursive variational-Laplace Gaussian filter** with a **full N×N
covariance** (predict `Σ+=Q` → Gauss-Newton MAP on the full data info → posterior `Σ=(P₀+I_eff)⁻¹`). The
three first-principles terms that make posteriors honest:

- **Within-frame common-mode marginalization (Woodbury):** each frame's information saturates at the
  shared-error covariance `Σc`, so N≈10⁴ correlated points can't collapse σ. No σ-floor / novelty gate /
  maturity ratchet. Applied to the **covariance only**; the MAP mean uses full data info.
- **Range-dependent covariance:** position + (binding) yaw common-mode grow with the producer's
  `mask_range` Z, so a distant view can't resolve pose/orientation. Continuous, no gate.
- **Pose-chain cov** `J·Σ_chain·Jᵀ` folded into the common-mode (localization uncertainty).

The inference math is **shared and header-only**: `common/ai_belief/recursive_laplace.h` holds
`rc::ai::predict / update / predicted_information` (the GN-MAP + Woodbury-common-mode engine). An
`<obj>_belief` provides only the MODEL hooks and delegates `update`/`predict`/`predicted_information` to
the engine:

```cpp
static constexpr int N = <dofs>;   using State = <Obj>BeliefState;   using Frame = <Obj>Frame;
float update(const Frame& f) { return ai::update<N>(*this, state_, Sigma_, prior_mean_, f); }   // → free energy
void  predict()              { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
Eigen::Matrix<float,N,N> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }              // Fisher info for the NBV
// ── model hooks the engine calls ──
float sdf_prim(const Vec3&, const State&, int prim) const;                    // one primitive's signed dist
Eigen::Matrix<float,N,1> sdf_jacobian(const Vec3&, const State&, int prim) const;   // finite-diff (CLAMP the slope!)
std::array<float,K> responsibilities(const Vec3&, const State&, float R) const;     // soft mixture [prims…, clutter]
void apply_constraints(State&) const;      // clamp sizes ≥ min, wrap angles
void canonicalize(State&) const;           // symmetry fold (table w↔h/yaw); no-op for chair/bottle
Eigen::Matrix<float,N,N> transition() const;          // I for static furniture (all three today)
Eigen::Matrix<float,N,1> process_noise_diag() const;  // Q
Eigen::Matrix<float,N,1> prior_cov_diag() const;      // Σ₀
Eigen::Matrix<float,N,1> common_mode_inv_diag(const Frame&) const;   // Σc⁻¹ incl. range + chain_cov_xx/yy
// OPTIONAL extra GN factor, detected by the engine via a C++23 `requires` (see below):
void accumulate_extra(const State&, const Frame&, Eigen::Matrix<float,N,N>& Id, Eigen::Matrix<float,N,1>& bd) const;
```

`transition()` is `I` for static furniture (**all three** today: table, chair, and — importantly — bottle,
whose movable-object CV tracking was removed in the AI2 port; a moved bottle is re-acquired via the
instance-tracker gate widening on `predict_stale()`, which inflates the position block of Σ only). A
CV-coupled AI2 transition (velocity DOFs) is a future extension, not the current pattern.

**`accumulate_extra` (optional, C++23 `requires`-detected).** The engine folds an extra per-frame GN factor
`(Id, bd)` into the normal equations iff the model defines it — no virtuals, zero cost when absent. Use it
for a model-specific structural term that isn't a plain point-SDF likelihood:
- **chair:** a **seat-layer height anchor** + a **footprint-extent anchor** (fixes the `seat_h` gauge
  runaway and `seat_d` collapse — see the chair notes).
- **bottle:** the **occluding-contour silhouette** tangent term `dist(axis, ray)=radius` — the ONLY term
  that pins the depth-degenerate radius of a symmetric cylinder from a one-sided depth cloud.
- **table:** none required (the box+legs SDF + extent likelihood suffice).

`<obj>_model.{h,cpp}` is now **only the state holder (`<Obj>State`) + the compound SDF** (+ any silhouette
store). All inference lives in `<obj>_belief`; the fitted posterior is written back into `<Obj>State` so the
downstream publish/viewer/RT code is unchanged. See `table_concept/src/table_belief.{h,cpp}` (box+legs, N=6),
`chair_concept/src/chair_belief.{h,cpp}` (seat+back+legs, N=8, `accumulate_extra` anchors — the most evolved
reference), `bottle_concept/src/bottle_belief.{h,cpp}` (single cylinder, N=5, silhouette `accumulate_extra`).

Everything else is **shared** and unchanged by a new object: `instance_tracker`, `mask_ingestor`, the worker
merge operator, `<obj>_scene_graph` (RT-cov maps the belief Σ), affordance / epistemic (Σ-based NBV) /
dashboard. So the §1 contract below holds; only the model hooks are authored.

---

## 0. Object spec (the prompt syntax)

Fill this from the human prompt + reference images. It is the complete input to generation.

```yaml
object:
  name:            chair          # → <obj>=chair, <Obj>=Chair, namespace rc
  dsr_node_type:   chair          # DSR node type the agent owns (get_nodes_by_type)
  node_prefix:     chair_         # instance node name prefix → "chair_1", …
  detection_labels: [chair]       # YOLO / open-vocab labels passed to MaskIngestor::select_nearest

belief:                           # the ONLY genuinely-authored code (→ <obj>_belief.{h,cpp} + <obj>_model SDF)
  dofs:                           # fitted state, in State::vec() order; tag each length|angle
    - {name: cx,     kind: length}   # room-frame centre X
    - {name: cy,     kind: length}
    - {name: cz,     kind: length}   # often anchored (see support), not freely fitted
    - {name: yaw,    kind: angle}
    - {name: seat_w, kind: length}
    - {name: seat_d, kind: length}
    - {name: seat_h, kind: length}   # seat-top height above floor
    - {name: back_h, kind: length}
  sdf:                            # per-primitive signed distances in object-LOCAL frame; the belief's
                                  # responsibilities() soft-mix them (NO hard min()). Pose (cx,cy,cz,yaw)
                                  # is applied by the model, NOT baked per-primitive.
    - {prim: box, name: seat, centre: [0,0,seat_h], half: [seat_w/2, seat_d/2, T/2]}
    - {prim: box, name: back, centre: [0,-seat_d/2, seat_h+back_h/2], half: [seat_w/2, T/2, back_h/2]}
    - {prim: box, name: legs, pattern: 4corners(seat_w,seat_d), z: [0, seat_h], half: [R,R,seat_h/2]}
  constants: {T: 0.04, R: 0.02}   # fixed (non-fitted) thicknesses
  canonicalize:    none           # symmetry fold: table = w↔h/yaw±90°; chair/bottle = none
  accumulate_extra: seat-anchor   # optional structural GN factor: seat-anchor | silhouette | none

priors:                           # broad priors (break the empty-cloud degeneracy only) → belief params
  size: 0.45  sigma_pos: 0.30  sigma_size: 0.15

support:                          # does it rest on a surface (z anchor)? (like bottle/chair on floor/table)
  rests_on_surface: true          # true → cz anchored to the surface, removed from the free fit
  anchored_dof:    cz             # which DOF the surface determines

dynamics:                         # static furniture is the current pattern for ALL three
  model: static                   # static → transition() = I. (constant_velocity is a documented FUTURE
                                  #   extension: CV velocity DOFs in the AI2 transition — NOT how bottle works today.)

epistemic:                        # next-best-view target for the affordance (object-specific)
  target: hidden-face             # hidden-face (far side from camera) | low-coverage-face | none
  degenerate_dof: seat_d          # the DOF a better view most resolves → drives ΔH (index into dofs)

capabilities:                     # opt-in; each is a copyable unit
  dashboard:     true             # Custom_widget + TimeSeriesPlot (docked): FE, dims, posterior σ
  affordance:    true             # EpistemicPlanner + <Obj>Affordance
  evaluator:     false            # Webots GT/sweep → adds `requires Webots2Robocomp` to .cdsl (bottle has it)
  ai2_csv:       true             # gated <Obj>Model.AI2CsvPath (per-cycle belief state + Σ diag)
  epistemic_csv: true             # gated Epistemic.CsvPath (only if affordance:true)
```

---

## 1. Fixed contract (copy verbatim; rename only)

### Module set (identical across agents; `+` = opt-in capability)
```
<obj>_config.{h,cpp}  <obj>_instance.h  <obj>_model.{h,cpp}  <obj>_belief.{h,cpp}  <obj>_fitter.{h,cpp}
<obj>_scene_graph.{h,cpp}  specificworker.{h,cpp}  specificworker_presence.cpp
+epistemic_planner.{h,cpp}  +<obj>_affordance.{h,cpp}  +<obj>_evaluator.{h,cpp}
shared (do NOT copy, #include + add to CMake):
  common/ai_belief/recursive_laplace.h                (header-only; rc::ai predict/MAP/Woodbury engine —
    the belief delegates update/predict/predicted_information to it. This replaced belief_stabilizer,
    the per-DOF Fisher filter, sample_queue, and prior_store — none of those exist in any agent anymore.)
  common/mask_ingestor/mask_ingestor.{h,cpp}          (call select_nearest(centroid, "<obj>"))
  common/instance_tracker/instance_tracker.h          (header-only; THE only instance-lifecycle path —
    run every cycle, not gated. Multi-instance birth/associate/death: gated 1-to-1 mask↔instance
    association (cov from the belief Σ), data-driven birth of unexplained masks, death of unsupported
    instances. Worker builds a TrackView per instance + DetectionView per slice, calls `update()`, then
    deaths→`delete_node`+`forget_node`, births→`<Obj>SceneGraph::create_instance_from_detection`, assoc→
    `inst.assigned_mask_idx` (read in `observe()`). There is NO prior-TOML scaffold fallback — instances
    exist only after a detection births one (a fresh clone from a crashed run is swept at startup).
    ★GOTCHA: the Mahalanobis gate uses the INNOVATION cov `S = P + R²I` (`Tracker.DetectionNoiseM`,
    R≈0.05 m), NOT P alone. The posterior P is honest-but-tight and the mask centroid wanders cm-scale vs
    the fit centre, so gating on P alone gives `assigned=0` every frame → every instance starves and
    dies+reborn each `DeathFrames` (the classic "objects flicker in/out" churn). R must be ≥ that offset.
    ★MERGE operator (worker-level, all agents): physical objects can't share space, so each cycle BEFORE
    associate/birth, collapse two instances whose footprints overlap ≥ `Tracker.MergeOverlap` (frac of the
    smaller), keeping the more-observed (`matched_frames`) and `delete_node`+`forget_node`+
    `affordance.remove()` on the other. The overlap metric is GEOMETRY-SPECIFIC: oriented-rectangle
    (Sutherland–Hodgman clip) for table/chair seat footprints; circle-lens for bottle radii.
    Matching is greedy-lowest-cost (exact 1-to-1 at these tiny counts). Cost is raw squared-Mahalanobis
    `m²` by default, or — with `Tracker.NllCost` — the Gaussian NLL `½(m²+ln|S|)` so tracks with different
    cov sizes compete by likelihood (a tight mature track out-bids a wide newborn for a contested
    detection). GATE stays `m²≤gate_mahalanobis` either way; NLL only changes outcomes under contention.
    ★NEGATIVE-INFORMATION DEATH (`TrackView.expected_visible`): a track accrues an unsupported "miss"
    ONLY when it SHOULD be visible — its centre projects inside the camera frustum yet wasn't detected.
    Out-of-FoV the miss timer is HELD → the object PERSISTS when the robot looks away; retired only when
    in-view-and-absent. Set per cycle by projecting the centre into the "zed" frustum. Death is OFF by
    default (`Tracker.DeathEnabled=false`) for persistent furniture; removal is then MERGE-only.)
  common/dashboard/timeseries_plot.{h,cpp} + custom_widget.h   (if dashboard; pass the title to the
      Custom_widget ctor; list both headers in CMake for AUTOMOC)
```
No `prior_store`, no `sample_queue_geometry.h`, no `belief_stabilizer`, no torch in CMake. (bottle was the
last to carry those; they were removed when it moved to the shared engine.)

### Fitter — PURE belief (no DSR writes). Public API exactly:
```cpp
bool                ensure_instance(const DSR::Node&, std::uint64_t room);   // true on first create; inits affordance
<Obj>Observation    observe(<Obj>Instance&, const DSR::Node&);               // mask → candidate/residual split
float               run_inference(<Obj>Instance&, const <Obj>Observation&);  // [support pre-step] → belief.update → [z-anchor] → FE
std::unordered_map<std::uint64_t,<Obj>Instance>& instances();
void                forget_node(std::uint64_t id);
bool                should_log(const <Obj>Instance&) const;                  // NOT should_log_<obj>
void                note_birth(std::uint64_t id, const Eigen::Vector2f& xy); // tracker seeds the birth centroid
void                set_chain_cov_source(DSR::InnerGaussianAPI*, std::string source_frame, bool enabled);  // Part B
```
Fitter holds `<Obj>SceneGraph*` for **reads only**. Object-specific belief steps (support surface, z anchor)
live INSIDE `run_inference`, never the worker. The fit is `inst.<obj>_belief.update(frame)`; the posterior
is copied back into `inst.model` so downstream publish/RT is unchanged.

### Worker — orchestration. `compute()`:
```cpp
mask_ingestor_->refresh();
run_instance_tracker();                     // THE birth/associate/death path — every cycle, not gated
for (const auto& node : G->get_nodes_by_type("<dsr_node_type>"))   // (bottle: "cylinder" named "bottle*")
    process_<obj>_node(node);
fps_counter_.print("[Compute]", 3000);      // heartbeat, once per cycle (see below)
```
`process_<obj>_node`:
```cpp
fitter_->ensure_instance(node, room_node_id_);
auto& inst = fitter_->instances().at(node.id());
++inst.processed_cycles;
auto obs = fitter_->observe(inst, node);
if (not obs.has_fresh_data and inst.matched_frames < 5) return;    // startup stale-skip
float fe = fitter_->run_inference(inst, obs);
if (auto n = G->get_node(node.id()); n) scene_graph_->step_write_model(inst, n.value(), fe);  // ALL DSR writes here
step_epistemic(inst);                       // if affordance
publish_<obj>_diagnostics(inst, fe);        // if dashboard
evaluator_->log_eval(inst, fe);             // if evaluator
inst.prev_free_energy = fe;
```
DSR slots forward to the affordance: `del_node_slot`→`on_node_deleted`, `modify_node_attrs_slot`→
`on_node_modified`. No `Qt::DirectConnection`; poll-only is fine.

**`[Compute]` heartbeat (every agent).** Hold an `FPSCounter fps_counter_;` (`#include <fps/fps.h>`) and
call `fps_counter_.print("[Compute]", 3000);` as the LAST line of `compute()` — once per cycle. Prints
`Period = …ms. Fps = …` every 3 s on `std::cout` (NOT `qInfo`, which RoboComp filters).

### Startup: birth is tracker-only
There is **no prior-scaffold**. `initialize()` sweeps any leftover `<obj>*` nodes from a crashed run
(`remove_owned_<obj>_nodes()`), resolves the room, builds the collaborators, and starts. Instances then
appear purely from the tracker as detections arrive.

### Convergence gate — per agent, but always committed-vs-committed
- **table/chair** compare a committed **state delta** vs `prev_conv_state`
  (`Σ|sᵢ−prevᵢ| < state_eps → ++frames_converged`). Inherently committed-vs-committed → SAFE.
- **bottle** uses an FE-stability test (`|fe − prev_free_energy| < fe_eps`). Both sides are the **committed**
  (post-`update`) free energy — a convergence test must compare the SAME final quantity across frames.

### YOLO score → observation reliability (`MaskConf*`)
The detector's per-mask confidence is a per-observation RELIABILITY. It maps to `w = clamp01((conf−floor)/
(ref−floor))^power` and scales the per-observation precision fed to the belief (the per-point `R` and, where
present, the silhouette precision in `accumulate_extra`), so a weak mask WIDENS Σ while the SDF geometry
still sets Σ's SHAPE. It is detection confidence, NOT localisation confidence. Prod `Ref≈0.5` (typical
conf→w≈1; bites only on weak conf). Verified by a static A/B: σ scales as `1/sqrt(w)` on well-observed DOFs.
This is separate from the RT-edge chain-propagation (common-mode, added to the POSTERIOR, not the per-obs
weight). Config namespace `<Obj>Model.MaskConf*`.

### Published RT covariance + the localization/chain term (all agents)
`<Obj>SceneGraph` writes a 6×6 `rt_covariance` on the room→obj RT edge from the belief Σ (row-major
[x,y,z,rx,ry,rz]; unobservable roll/pitch = big). ★ADD the **chain covariance `J·Σ_chain·Jᵀ`** — the
uncertainty the room-frame pose inherits from robot localization (the fit cov is conditional on the robot
pose). Enabled uniformly via `fitter_->set_chain_cov_source(gaussian_api_.get(), "zed", cfg_.rt_cov_add_chain)`:
the fitter computes it with `DSR::InnerGaussianAPI` (transform the fitted centre room→"zed" then back with a
ZERO input cov; `transform_point` returns exactly `J·Σ_chain·Jᵀ`, Σ_chain = each RT edge's `rt_covariance`
adjoint-composed; `room_concept` publishes the robot↔room term), pinned to the capture stamp. Its xy block
adds to the published translation cov. The scene-graph write self-gates (geometry republish OR trace change
>5%) to suppress churn once settled. Config `*.RtCovAddChain`. COV ONLY — masks stay room-frame.

### Dashboard (`publish_<obj>_diagnostics`, if `dashboard`) — stacked `TimeSeriesPlot` panels, per node:
FE → dimensions → **posterior σ(mm)** per DOF (from `sqrt(diag Σ)` — the honest calibrated uncertainty; this
is what the old counter-evidence/CUSUM panel became once the stabiliser was removed). The gated AI2 CSV
(`<Obj>Model.AI2CsvPath`) logs per cycle: `cycle,node,pts,R,energy,frames_converged[,extra]` — the belief
state + Σ diag. (`TimeSeriesPlot` strokes lines only — never set a brush before `drawPath`, or peaks fill.)

---

## 2. Authored from the spec (the small creative surface)

| Spec field | Emits | Notes |
|---|---|---|
| `belief.dofs` | `<Obj>BeliefState` (+ `vec()/from_vec()`) and the geometry `<Obj>State` (`to_array`) | the fitted vector; N = count |
| `belief.sdf` + `constants` | `<Obj>Belief::sdf_prim()` per primitive + `<Obj>Model` compound SDF | **the one creative function** |
| `belief.sdf` (K prims) | `responsibilities()` = soft mixture `[prim₀…prim_{K-1}, clutter]` | replaces hard `min()` |
| `belief.canonicalize` | `canonicalize()` — symmetry fold or no-op | table only |
| `belief.accumulate_extra` | the optional structural GN factor (seat-anchor / silhouette / none) | C++23 `requires`-detected |
| `priors` | `<Obj>BeliefParams` (prior_pos_std, prior_size_std, clutter, common-mode, gn_iters) | broad; `<Obj>Model.AI2*` config |
| `detection_labels` | the label arg in `observe()`'s `mask_ingestor_->select_nearest(centroid, "<obj>")` | shared MaskIngestor |
| `support.*` | keep/port the cz anchor inside `run_inference` (or drop if `rests_on_surface:false`) | |
| `dynamics.model` | `static` → `transition()` = I (the current pattern for all three). `constant_velocity` = future | |
| `epistemic.target` + `degenerate_dof` | `EpistemicPlanner::compute` viewpoint + ΔH DOF index | hidden-face/face-coverage |
| `affordance` capability | a `default_contract_for("aff_<obj>")` branch in `common/affordance_protocol.h` | servo completion clauses (`<obj>_detection_alive/_confidence`) **+ `.still(v, ω)`** so the controller dwells for a clean, motion-free look |
| `dsr_node_type` + `belief.dofs` | `<Obj>SceneGraph` geometry attrs + node scaffold + `persist_<obj>_belief` + `create_instance_from_detection` | DSR I/O |

> **`sdf_jacobian` clamps the slope.** It finite-differences a non-smooth SDF — **clamp the per-DOF slope**
> (`±gmax`) before use, or one point on a primitive seam spikes the information 1000×. (Hard-won; see the
> belief `.cpp`s.)

---

## 3. Generation procedure (deterministic)

1. **Pick the closer skeleton** (all AI2): `table` (box top + inset legs, symmetry fold), `chair`
   (seat+back+legs, floor-anchored, `accumulate_extra` seat/extent anchors — the most evolved), `bottle`
   (single cylinder, silhouette `accumulate_extra`, single primitive). Primitive-box/legged → table/chair;
   single round body → bottle.
2. **Copy + token-rename** the module set (`table`→`<obj>`, `Table`→`<Obj>`); keep the
   `common/ai_belief/recursive_laplace.h` `#include`.
3. **Author `<Obj>Belief`** from `belief.sdf`: `sdf_prim`, `responsibilities`, the Q/prior/common-mode
   diagonals, `canonicalize`, and (optional) `accumulate_extra`. Author `<Obj>Model` = state holder +
   compound SDF. This is the only real code.
4. **Fill priors** (`<Obj>BeliefParams` + `<Obj>Model.AI2*` config), **detection label**, **support**
   (cz anchor or drop), **epistemic target** (`EpistemicPlanner::compute`), **scene-graph attrs**.
5. **Set `N`** from `belief.dofs`.
6. **Toggle capabilities** per `capabilities` (add/remove files + CMake entries + worker wiring).
7. **`.cdsl`:** `options dsr` (+ `requires Webots2Robocomp` iff `evaluator:true`).
8. **CMake:** list every `src/*.{cpp,h}` (headers too, for AUTOMOC), including `<obj>_belief.{cpp,h}`.
   New sources via INCLUDEd list → **nuke `build/CMakeCache.txt` and reconfigure** (incremental cmake misses
   appended sources). NO torch (`find_package(Torch)` is gone).
9. **Build:** `cmake -B build && make -C build -j`. No `-march=native` / no `-DEIGEN_MAX_ALIGN_BYTES`
   (must match the prebuilt libdsr Eigen alignment or it SIGBUS/stack-smashes).
10. **Verify the belief in isolation:** it's pure Eigen — compile `<obj>_belief.cpp` + a `main` calling
    `<Obj>Belief::self_test()` against Eigen only and confirm it prints `PASS` (recovers a synthetic pose).
    The agent also runs `self_test()` once at startup in `initialize()`.

---

## 4. Operational checklist

- [ ] Builds green; binary links (no torch).
- [ ] `<Obj>Belief::self_test()` prints PASS (isolated Eigen unit test + at startup).
- [ ] Launches, presence reaches Operating (copy bottle's presence protocol verbatim; unique `[Agent] id`).
- [ ] On a fresh mask the tracker births a `<obj>_*` node; `ensure_instance` fires once; FE is finite and
      the fit moves toward the object.
- [ ] Dashboard panels populate (FE / dims / posterior σ); the gated AI2 CSV writes when its path is set.
- [ ] RT edge carries a calibrated 6×6 `rt_covariance` (+ chain term when `RtCovAddChain`).
- [ ] (affordance) `aff_<obj>_*` node appears with a sane target + ΔH; controller honours the contract's
      `.still(v, ω)` (dwells before completing a look).
- Day-one ≠ tuned: gains/deadbands/`epistemic.obs_distance` refine over a few live iterations.

---

## 5. Worked example — `chair_concept` (the most evolved reference)

- `<Obj>Model` SDF = per-primitive box distances `{seat, back, leg×4}` in local frame, posed by
  `(cx,cy,cz,yaw)`; `<Obj>Belief::responsibilities` soft-mixes them `[seat, back, leg0..3, clutter]`.
- `N=8`, DOFs `[cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h]`.
- `support.rests_on_surface=true` → `cz` pinned to the floor (`ai2_floor_z`), floor uncertainty → common-mode z.
- `canonicalize` = no-op (a chair has a front); `resolve_orientation` does the 180° yaw disambiguation
  (backrest breaks symmetry).
- `accumulate_extra` = **seat-layer height anchor + footprint-extent anchor** (fixes `seat_h` gauge runaway
  and `seat_d` collapse); density-aware clutter (`ai2_clutter_structure_gain`) closes the coplanar escape valve.
- `epistemic.target=hidden-face`, `degenerate_dof=seat_d`. Tracker uses the belief mixture NLL
  (`association_nll`) for association under same-class clutter.
- Authored code ≈ `sdf_prim` + `responsibilities` + the two anchors in `accumulate_extra` + `ChairBeliefState`;
  everything else is rename + spec-fill.

---

## 6. Meta-concept variant — a concept-of-concepts over other agents' beliefs

A **meta-concept** (naming suffix `_metaconcept`) is a higher-level latent whose state *generates* the poses
of already-inferred objects and pushes **empirical priors back down** to the per-object agents. It is the SAME
construction recipe (§1–§4) for ~80% of the agent — start from a `chair_concept` copy for presence,
DSR-safety, the multi-instance tracker, the recursive-Laplace engine, and the dashboard/affordance
capabilities — with **three organs replaced**. Capture those three here; everything else in §1–§4 still holds
verbatim.

This variant is empirical Bayes made explicit: the arrangement latent is one more layer of the same
recursive-Laplace belief. The "strong priors" pushed to the constituents ARE this layer's top-down
predictions. Do NOT implement it as a classifier that flips a flag and writes edges — that reintroduces a
threshold and drops the covariance the down-prior needs.

**Name the agent after the ARRANGEMENT SCHEMA, not the concrete concept.** The schema is the level-2 analogue
of the SDF: at level 1 the fixed chassis + one authored `sdf_prim` yields any object; at level 2 the fixed
meta-concept chassis + one authored **slot function** yields any *configuration*. `ring` (N slots radially
around an anchor, members facing inward) is the first and most common schema — so the first agent is
**`ring_metaconcept`**, which births *typed instances* (`dining_set`, `coffee_cluster`, `meeting_set`) that
differ only by which member node-types populate anchor+ring and by scale/count, all decided by the same model
evidence (below). The multi-instance tracker (§1) gives that for free. Other schemas are each just a different
slot function on this identical chassis, authored later when a second concrete case exists to validate the
abstraction (author `ring` ONLY now — don't build an any-schema engine speculatively):
- **row / linear** — books on a shelf, shelves of a bookcase, a run of cabinets.
- **facing-pair** — sofa ↔ TV, bed + flanking nightstands (bilateral), two facing armchairs.
- **grid / workspace** — desk + monitor + keyboard + mouse (not radial).

### Delta 1 — the front end is a GRAPH-READER, not a mask/media consumer

There is **no sensor and no media plane**. Replace the entire `mask_ingestor` + `make_*_subscriber_from_graph`
half of the recipe with a graph-reader (a responsibility of `<obj>_scene_graph`, whose data direction inverts:
it now READS peers). Each `compute()` polls the constituent nodes and their beliefs — e.g. every `table` and
`chair` node with its room→obj RT pose **and its published `rt_covariance`**. Those peer posteriors (mean +
Σ) are this agent's "measurements". So:

- `detection_labels` / `select_nearest` / `MaskConf*` / `SeedDeproject` — **gone** (no masks).
- The measurement noise is not `R²I`; it is each peer's **own published Σ** (a chair the network is unsure of
  contributes weakly — honest by construction, no gate).
- ★**SAFETY (CLAUDE.md):** reacting to constituent nodes appearing/moving is exactly when you'd reach for
  `update_node_signal` / `update_edge_signal` — which is the `DirectConnection` heap-corruption crash. This
  agent's primary input is the graph, so it is the MOST exposed to that trap. **Poll the graph on the main
  thread each `compute()`** (same discipline as poll-based presence); connect NONE of the update signals.

### Delta 2 — the belief is an ARRANGEMENT latent over peer poses (still recursive-Laplace)

Reuse `common/ai_belief/recursive_laplace.h` unchanged — predict `Σ+=Q` → GN-MAP → `Σ=(P₀+I_eff)⁻¹`. Only the
MODEL hooks change *meaning*: the "measurement" is a peer **pose+Σ**, not a 3D mask point; the "SDF residual"
becomes a **slot residual**.

```yaml
metaconcept:
  agent:           ring_metaconcept                  # named for the SCHEMA (radial arrangement)
  schema:          ring
  first_instance:  dining_set                        # DSR node type of the first concrete config
  members:         {anchor: table, ring: chair}      # node types this instance groups (per-config)
  latent:                                            # <Set>BeliefState, in vec() order
    - {name: cx,     kind: length}                   # set centre (≈ table centre)
    - {name: cy,     kind: length}
    - {name: yaw,    kind: angle}                    # set orientation
    - {name: radius, kind: length}                   # chair ring radius
  count:           model-selection                   # N chairs is DISCRETE → evidence over hypotheses,
                                                     #   NOT a continuous DOF (see detection below)
  slot_fn:         g(latent, k, N) = centre + radius·(cos θk, sin θk), θk = yaw + 2πk/N,
                     chair_yaw = θk + π              # chairs face inward — the strong yaw prior
```

Hooks (analogues of §0's belief hooks):
- `slot_pose(latent, k, N)` replaces `sdf_prim` — the predicted pose of ring-slot `k` (the one creative fn).
- `responsibilities(member_pose, latent)` = **soft slot assignment** `[slot₀…slot_{N-1}, not-a-member]` — a
  chair is softly bound to the nearest slot; the `not-a-member` column is the clutter escape valve (a chair
  that isn't part of this set). NO hard "assign to nearest within R".
- measurement Jacobian of `g` w.r.t. the latent (finite-diff, clamp the slope as in §2).
- `transition() = I` (a dining set is static furniture-of-furniture); `process_noise/prior/common-mode`
  diagonals as usual — the common-mode grows the set pose Σ when few members are seen.

### Delta 3 — the DOWN-PRIOR back channel (node + non-RT membership edges)

This is the first agent that writes **empirical priors back to peers**, and the reason A (node) beats B
(edges-only): the node OWNS the arrangement latent (pose, radius, N, existence probability); the edges are the
message channel down.

- **A + B, node is the point:** birth a `dining_set` node (multi-instance tracker, §1 — several sets possible)
  holding the latent belief; connect it to each constituent by a **non-RT `member_of` edge**.
- **Edge payload = the top-down message:** each `member_of` edge carries the **predicted slot pose + its
  covariance**. The constituent agent reads it and folds it in as an empirical-prior GN factor through the
  `accumulate_extra` hook it already has (chair: alongside its seat/extent anchors) — exactly the existing
  mechanism, zero engine change on the consumer side.
- ★**Single-writer:** the relational agent OWNS the `member_of` edge + payload; constituent agents only READ
  it. Same discipline as the RT chain-cov (one writer per edge).
- ★★**Leave-one-out (the one real hazard):** the message sent DOWN to chair *i* must be the arrangement
  posterior computed from **the other members**, never including chair *i*'s own contribution — else you feed
  a belief its own output and manufacture confidence (the overconfidence failure mode Woodbury fixed
  within-frame, now between agents). In message-passing terms: never send a node the message it sent up.
  Concretely: recompute (or down-date) the slot prediction excluding member *i* before writing its edge.
- The predicted-pose Σ must **grow** when the arrangement is poorly determined (few members, wide spread) so a
  well-observed chair isn't tyrannised by a shaky set hypothesis — honest covariance, no clamp.

### Detection without a threshold — model evidence, not a distance gate

Do NOT gate on "≥3 chairs within R of a table". The set **exists to the degree** its arrangement explains the
joint configuration better than the independent-objects null — a continuous log-Bayes-factor (compare the
belief's converged free energy under "generated by the arrangement" vs "independent"). That log-evidence is
the node's existence probability; the discrete member count N is chosen the same way (evidence over
`N ∈ {3,4,6,…}` hypotheses). Birth is the tracker's usual **persistent-evidence** process, so a fleeting
coincidence doesn't spawn a room; death is the negative-information path (§1) when the evidence collapses.

### Payoff (why the down-prior earns its keep)

The prior bites on exactly the cases §5 struggles with: it **resolves chair yaw** (chairs face inward), it
**fills in** a peripheral (Ricoh-360 bearing-only) or table-occluded chair from its slot, and it **seeds
birth** at an empty slot — which also hands the `EpistemicPlanner` a natural next-best-view target ("go
confirm the 4th slot").

### Capabilities / checklist delta

- Reuse: presence (verbatim), tracker, recursive-Laplace engine, dashboard (FE / latent dims / posterior σ),
  affordance (the empty-slot NBV).
- Drop: `mask_ingestor`, media subscribers, `MaskConf*`, `support`/z-anchor, `sdf`/`responsibilities`-over-points.
- Add to the §4 checklist: `member_of` edges appear with predicted-pose+Σ; the leave-one-out down-date is
  verified (a member's own obs is excluded from its own down-prior); the constituent agent's `accumulate_extra`
  actually consumes the edge; existence probability tracks the log-evidence (rises as members corroborate,
  decays when they scatter) rather than a hard count.
