# Concept-Agent Recipe — generating a new `<obj>_concept`

Concise, technical generation contract. Companion to `CONCEPT_AGENT_PATTERN.md` (rationale/history).
This file is the spec a coding agent follows to emit a new, operational concept agent from an
**object spec** (Step 0) + the closest reference skeleton (bottle or table). `<obj>` = lowercase
class, `<Obj>` = PascalCase.

Everything except the **model** (the SDF) and a handful of spec-driven hooks is mechanical
copy + token-rename. The object spec is the only human input.

---

## 0. Object spec (the prompt syntax)

Fill this from the human prompt + reference images. It is the complete input to generation.

```yaml
object:
  name:            chair          # → <obj>=chair, <Obj>=Chair, namespace rc
  dsr_node_type:   chair          # DSR node type the agent owns (get_nodes_by_type)
  node_prefix:     chair_         # instance node name prefix → "chair_1", …
  detection_labels: [chair]       # YOLO / open-vocab labels passed to MaskIngestor::select_nearest

model:                            # the ONLY genuinely-authored code (→ <obj>_model.{h,cpp})
  dofs:                           # fitted state, in to_array() order; tag each length|angle
    - {name: cx,     kind: length}   # room-frame centre X
    - {name: cy,     kind: length}
    - {name: cz,     kind: length}   # often anchored (see support), not freely fitted
    - {name: yaw,    kind: angle}
    - {name: seat_w, kind: length}
    - {name: seat_d, kind: length}
    - {name: seat_h, kind: length}   # seat-top height above floor
    - {name: back_h, kind: length}
  sdf:                            # union = min() over signed primitives, in object-LOCAL frame
                                  # (pose cx,cy,cz,yaw is applied by the model, NOT baked per-primitive)
    - {prim: box, name: seat, centre: [0,0,seat_h], half: [seat_w/2, seat_d/2, T/2]}
    - {prim: box, name: back, centre: [0,-seat_d/2, seat_h+back_h/2], half: [seat_w/2, T/2, back_h/2]}
    - {prim: box, name: legs, pattern: 4corners(seat_w,seat_d), z: [0, seat_h], half: [R,R,seat_h/2]}
  constants: {T: 0.04, R: 0.02}   # fixed (non-fitted) thicknesses

priors:                           # → etc/object_priors.toml  [[<obj>s]]  + PriorStore <Obj>Prior fields
  seat_w: 0.45  seat_d: 0.45  seat_h: 0.45  back_h: 0.45
  sigma_pose: 0.10  sigma_size: 0.10

support:                          # does it rest on a surface (z anchor + re-parent), like bottle?
  rests_on_surface: true          # true → port bottle's update_support_surface + cz anchor
  anchored_dof:    cz             # which DOF the surface determines (removed from the free fit)

epistemic:                        # next-best-view target for the affordance (object-specific)
  target: hidden-face             # hidden-face (far side from camera) | low-coverage-face | none
  degenerate_dof: seat_d          # the DOF a better view most resolves → drives ΔH (index into dofs)

capabilities:                     # opt-in; each is a copyable unit
  dashboard:     true             # Custom_widget + TimeSeriesPlot (docked)
  affordance:    true             # EpistemicPlanner + <Obj>Affordance
  evaluator:     false            # Webots GT/sweep → adds `requires Webots2Robocomp` to .cdsl
  fisher_csv:    true             # gated BottleModel/TableModel.FisherCsvPath
  epistemic_csv: true             # gated Epistemic.CsvPath (only if affordance:true)
```

---

## 1. Fixed contract (copy verbatim; rename only)

### Module set (identical across agents; `+` = opt-in capability)
```
<obj>_config.{h,cpp}  <obj>_instance.h  <obj>_model.{h,cpp}  <obj>_fitter.{h,cpp}
<obj>_scene_graph.{h,cpp}  prior_store.{h,cpp}  sample_queue_geometry.h
specificworker.{h,cpp}  specificworker_presence.cpp
+epistemic_planner.{h,cpp}  +<obj>_affordance.{h,cpp}  +<obj>_evaluator.{h,cpp}
shared (do NOT copy, #include + add to CMake):
  common/belief_stabilizer/belief_stabilizer.h        (header-only)
  common/instance_tracker/instance_tracker.h          (header-only; multi-instance birth/associate/death,
    behind `Tracker.Enabled` — gated 1-to-1 mask↔instance association (cov from the stabiliser posterior),
    data-driven birth of unexplained masks, death of unsupported instances. Worker builds TrackView per
    instance + DetectionView per slice, calls `update()`, then deaths→`delete_node`+`forget_node`,
    births→`<Obj>SceneGraph::create_instance_from_detection`, assoc→`inst.assigned_mask_idx` (read in
    `observe()` instead of greedy nearest). OFF → legacy prior-scaffold + greedy nearest.
    ★GOTCHA: the Mahalanobis gate uses the INNOVATION cov `S = P + R²I` (`Tracker.DetectionNoiseM`,
    R≈0.05 m), NOT P alone. The posterior P is overconfident (σ~mm) and the mask centroid wanders
    cm-scale vs the fit centre (worse with seed de-projection), so gating on P alone gives
    `assigned=0` every frame → every instance starves and dies+reborn each `DeathFrames` (the classic
    "objects flicker in/out" churn). R must be ≥ that centroid-vs-fit offset.
    ★MERGE operator (worker-level, all multi-instance agents): physical objects can't share space, so
    each cycle BEFORE associate/birth, collapse two instances whose footprints overlap ≥
    `Tracker.MergeOverlap` (frac of the smaller), keeping the more-observed (`matched_frames`) and
    `delete_node`+`forget_node`+`affordance.remove()` on the other. The overlap metric is
    GEOMETRY-SPECIFIC: oriented-rectangle (Sutherland–Hodgman clip) for table/chair seat footprints;
    circle-lens for bottle radii. Without it two tracks can lock onto one object and never reconcile.
    Matching is greedy-lowest-cost (exact 1-to-1 at these tiny instance counts; no Hungarian). Cost is
    raw squared-Mahalanobis `m²` by default, or — with `Tracker.NllCost` — the Gaussian NLL
    `½(m²+ln|S|)` so tracks with DIFFERENT cov sizes compete by likelihood, not raw distance (a tight
    mature track out-bids a wide newborn for a contested detection). The GATE stays `m²≤gate_mahalanobis`
    either way; NLL only changes outcomes under contention (same-class clutter). Off by default.)
  common/mask_ingestor/mask_ingestor.{h,cpp}          (call select_nearest(centroid, "<obj>"))
  common/sample_queue/sample_queue.h                  (header-only; SampleQueue<<Obj>Model>)
    → author sample_queue_geometry.h: specialise SampleQueueGeometry<<Obj>Model> with the 4 geometry
      primitives (bin_index/edge_score/to_local_frame/face_coverage); add `using State=<Obj>State;` to the model
  common/dashboard/timeseries_plot.{h,cpp} + custom_widget.h   (if dashboard; pass the title to the
      Custom_widget ctor — `new Custom_widget("<Obj> Model — …")`; list both headers in CMake for AUTOMOC)
```

### Fitter — PURE belief (no DSR writes). Public API exactly:
```cpp
bool                ensure_instance(const DSR::Node&, std::uint64_t room);   // true on first create; inits affordance
<Obj>Observation    observe(<Obj>Instance&, const DSR::Node&);               // mask → candidate/residual split
float               run_inference(<Obj>Instance&, const <Obj>Observation&);  // [support pre-step] → fit → [z-anchor] → FE
std::unordered_map<std::uint64_t,<Obj>Instance>& instances();
void                forget_node(std::uint64_t id);
bool                should_log(const <Obj>Instance&) const;                  // NOT should_log_<obj>
```
Fitter holds `<Obj>SceneGraph*` for **reads only**. Object-specific belief steps (support surface,
z anchor) live INSIDE `run_inference`, never the worker.

### Worker — orchestration. `compute()` → for each node → `process_<obj>_node`:
```cpp
fitter_->ensure_instance(node, room_node_id_);
auto& inst = fitter_->instances().at(node.id());
++inst.processed_cycles;
auto obs = fitter_->observe(inst, node);
if (not obs.has_fresh_data and inst.matched_frames < 5) return;     // startup stale-skip
float fe = fitter_->run_inference(inst, obs);
if (auto n = G->get_node(node.id()); n) scene_graph_->step_write_model(inst, n.value(), fe);  // ALL DSR writes here
step_epistemic(inst);              // if affordance
publish_<obj>_diagnostics(inst, fe);   // if dashboard
evaluator_->log_eval(inst, fe);    // if evaluator
inst.prev_free_energy = fe;
```
DSR slots forward to the affordance: `del_node_slot`→`on_node_deleted`, `modify_node_attrs_slot`→`on_node_modified`.
No `Qt::DirectConnection`; poll-only is fine (bottle connects no update signals).

**`[Compute]` heartbeat (every agent).** Hold an `FPSCounter fps_counter_;` member (`#include <fps/fps.h>`)
and call `fps_counter_.print("[Compute]", 3000);` as the LAST line of `compute()` — once per cycle, not
per node. It self-accumulates and prints `Period = …ms. Fps = … [Compute] cpu = …% mem = …MB` every 3 s
on `std::cout` (NOT `qInfo`, which RoboComp filters). This is the standard liveness/throughput signal; the
loop period should track `Period.Compute`. (`get_frequency()` is also available if you regulate on rate.)

### Stabiliser (shared) — `rc::BeliefStabilizer<N>` over `StabilizerState<N> stab` on the instance:
- `N` = number of DOFs; `StabilizerLayout<N>` flags the periodic (yaw) DOF and the `is_position` DOFs
  (cx,cy) — position gets its own CUSUM barrier so the centre locks harder than the extents.
- Per fresh frame: `weight_observation(stab, obs_info, mask_conf)` → `compute_acceptance(stab, raw, prev)`
  (applies the maturity-stiffened Kalman gain + CUSUM/SPRT gate to the accepted state — table) → `accumulate(stab)`.
- **Fold-only agents** (bottle keeps its convergence gate, doesn't apply the gain) STILL call
  `compute_acceptance` **diagnostically** (`ce_gate=true`, gains ignored, innovation = this fresh fit vs the
  previous one held in `prev_diag_state`) so `counter_evidence`/`last_ce_gate` are live for the dashboard/CSV.
- `posterior_std_milli(stab, j)` gives σ for the dashboard/CSV. Quality-rescale (ratchet+EMA) is config
  (`Fisher*` keys) — re-opens the fit only on a genuinely better view, else a DOF wanders.
- **YOLO score → covariance (`MaskConfWeight/Floor/Ref/Power`).** The detection score is per-observation
  RELIABILITY → it weights the OBSERVATION via `w = clamp01((conf−floor)/(ref−floor))^power`, so a weak
  mask widens the belief and a confident one tightens it (floor keeps a low-but-real detection
  contributing; the score WIDENS but the SDF geometry still sets Σ's SHAPE — it is detection confidence,
  NOT localisation confidence). Where `w` must be applied depends on **how the lineage publishes its cov**
  — get this right per lineage or it silently never reaches the RT edge:
  - **table/chair lineage** (publishes the Fisher-accumulator cov: `inst.stab.fisher_info_raw` →
    `<Obj>SceneGraph::write_rt_covariance`). The published cov IS fed by the stabiliser obs-info, so
    weighting `weight_observation(stab, obs_info, mask_conf)` ALONE reaches it. This lineage already wires
    it (config namespace `WarmStart.MaskConf*`, on by default) — nothing to add.
  - **bottle lineage** (fold-only; publishes a Laplace `model.pose_covariance(fit_pts, queue.weights())`).
    The Laplace cov does NOT see the stabiliser, so the score must ALSO multiply the per-frame queue
    capture precision + residual fit weight by `w` in `run_inference` (helper `mask_confidence_weight()`),
    in addition to the `weight_observation` call. Config namespace `BottleModel.MaskConf*`.
  - Prod `Ref≈0.5` (typical conf→w≈1; bites only on weak conf). Verified by a static A/B (force a bite
    with `Ref≈0.95`, toggle `MaskConfWeight`): σ scales as `1/sqrt(w)` on the well-observed DOFs. This is
    NOT a post-hoc add of robot-room cov — that is the separate RT-edge chain-propagation step
    (common-mode, added to the POSTERIOR, not the per-obs weight).
- **Convergence gate — handled differently per lineage; don't "re-fix" the safe one:**
  - **table/chair lineage** compares a committed **state delta** vs `prev_conv_state`
    (`Σ|sᵢ−prevᵢ| < state_eps → ++frames_converged`). This is inherently committed-vs-committed → SAFE,
    leave it.
  - **bottle lineage** (fold-only) uses an FE-stability test (`|fe − prev_free_energy| < fe_eps`). ★GOTCHA:
    it MUST compare the **committed** (post-stabiliser) FE both sides. The stabiliser holds the committed
    state away from the per-frame raw gradient optimum, so comparing the raw pre-stabiliser FE against
    `prev_free_energy` (which stores the committed FE) never agrees within `fe_eps` even when the belief is
    rock-stable → `frames_converged` pins at 0, the posterior never latches, useless as a cov source. Run
    the bookkeeping AFTER the stabiliser block. (General rule: a convergence test compares the SAME final
    quantity across frames.)

### Dashboard (`publish_<obj>_diagnostics`, if `dashboard`) — stacked `TimeSeriesPlot` panels, per node:
FE → dimensions → posterior σ(mm) → **counter-evidence (CUSUM)** as the last panel (`ce_<deg-DOF>` per
size DOF, from `stab.counter_evidence[j]`): ≈0 = locked, spike-then-decay = rejected glitch, ramp-then-reset
= a real re-fit. The fisher CSV (gated) logs per DOF: `state/obs/acc/raw/std/gain/ce/ceg` — identical schema
across agents. (`TimeSeriesPlot` strokes lines only — never set a brush before `drawPath`, or peaks fill.)

---

## 2. Authored from the spec (the small creative surface)

| Spec field | Emits | Notes |
|---|---|---|
| `model.dofs` | `<Obj>State` fields + `to_array/from_array` (DOF order) | the fitted vector |
| `model.sdf` + `constants` | `<Obj>Model::sdf_point()` = `min()` over primitives in local frame | **the one creative function** |
| `model.dofs` | `<Obj>ModelParams` (priors, σ_obs, robust loss, optim) | mostly defaults |
| `priors` | `etc/object_priors.toml [[<obj>s]]` + `<Obj>Prior` struct fields | metric seeds |
| `detection_labels` | the label arg in `observe()`'s `mask_ingestor_->select_nearest(centroid, "<obj>")` | perception hook (shared `MaskIngestor`, no per-object method) |
| `support.*` | port/keep `update_support_surface` + cz anchor (or drop if `rests_on_surface:false`) | |
| `epistemic.target` + `degenerate_dof` | `EpistemicPlanner::compute` viewpoint + ΔH DOF index | hidden-face/face-coverage |
| `affordance` capability | a `default_contract_for("aff_<obj>")` branch in `common/affordance_protocol.h` | servo completion clauses (`<obj>_detection_alive/_confidence`) **+ `.still(v, ω)`** observation-stillness so the controller dwells for a clean, motion-free look (no blur / pose smear); the executor ANDs base-speed ≤ thresholds into completion |
| `dsr_node_type` + `model.dofs` | `<Obj>SceneGraph` geometry attrs + node scaffold + `persist_<obj>_belief` | DSR I/O |

### Model contract (what `<Obj>Model` must expose — mostly mechanical given `sdf_point`)
```cpp
const <Obj>State& state();  void set_state(); void set_prior();         // + State::to_array/from_array
float sdf_point(const Vec3&) const;  std::vector<float> compute_sdf(...);
float compute_free_energy(pts,w);  FreeEnergyDecomposition compute_free_energy_decomposition(...);
std::array<float,N> observation_information(pts,w) const;   // Fisher diag = Σ (∂sdf/∂θ_j)²/σ² (finite-diff; CLAMP the slope!)
void gradient_step(...);   const <Obj>ModelParams& params();
```
> `observation_information` finite-differences a non-smooth `min()` SDF — **clamp the per-DOF slope**
> (`±gmax`) before squaring, or one point on a primitive seam spikes the Fisher 1000×. (Hard-won; see
> `table_model.cpp` / the stabiliser.)

---

## 3. Generation procedure (deterministic)

1. **Pick the closer skeleton:** primitive-box/legged → clone `table`; single round body → clone `bottle`.
2. **Copy + token-rename** the module set (`table`→`<obj>`, `Table`→`<Obj>`), keep `common/belief_stabilizer` `#include`.
3. **Author `<Obj>Model`** from `model.sdf` (the only real code) + `<Obj>State` from `model.dofs`.
4. **Fill priors** (`object_priors.toml` + `<Obj>Prior`), **detection label**, **support** (port or drop),
   **epistemic target** (`EpistemicPlanner::compute`), **scene-graph attrs**.
5. **Set `N` + layout** for `BeliefStabilizer<N>` from `model.dofs`.
6. **Toggle capabilities** per `capabilities` (add/remove files + CMake entries + worker wiring).
7. **`.cdsl`:** `options dsr` (+ `requires Webots2Robocomp` iff `evaluator:true`).
8. **CMake:** list every `src/*.{cpp,h}` (headers too, for AUTOMOC). New sources via INCLUDEd list →
   **nuke `build/CMakeCache.txt` and reconfigure** (incremental cmake misses appended sources).
9. **Build:** `cmake -B build && make -C build -j`. No `-march=native` / no `-DEIGEN_MAX_ALIGN_BYTES`
   (must match the prebuilt libdsr Eigen alignment or it SIGBUS/stack-smashes).

---

## 4. Operational checklist

- [ ] Builds green; binary links.
- [ ] Launches, presence reaches Operating (copy bottle's presence protocol verbatim; unique `[Agent] id`).
- [ ] Creates a `<obj>_*` node from the prior/scaffold; `ensure_instance` fires once.
- [ ] On a fresh mask, FE is finite and the fit moves toward the object (cold-start centroid snap).
- [ ] Dashboard panels populate (incl. the counter-evidence panel: ≈0 settled, ramp-then-reset on a re-fit);
      gated CSVs write with the shared `state/obs/acc/raw/std/gain/ce/ceg` schema when their path is set.
- [ ] (affordance) `aff_<obj>_*` node appears with a sane target + ΔH; controller reads it AND honours the
      contract's `.still(v, ω)` (it dwells before completing a look).
- Day-one ≠ tuned: gains/deadbands/`epistemic.obs_distance` refine over a few live, human-assisted iterations.

---

## 5. Worked example — `chair_concept` from the spec in §0

- `<Obj>Model` SDF = `min(seat_box, back_box, leg_box×4)` in local frame, posed by `(cx,cy,cz,yaw)`.
- `N=8`, layout `[len,len,len,ang,len,len,len,len]`.
- `support.rests_on_surface=true` → port bottle's `update_support_surface`, anchor `cz`.
- `epistemic.target=hidden-face`, `degenerate_dof=seat_d` (depth from a front view).
- Clone `table` (legged, box-ish), capabilities: dashboard+affordance+fisher_csv+epistemic_csv, evaluator off.
- Authored code ≈ the `sdf_point` function + `ChairState`; everything else is rename + spec-fill.
