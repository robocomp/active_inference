# Concept-Agent Recipe — generating a new `<obj>_concept`

Concise, technical generation contract. Companion to `CONCEPT_AGENT_PATTERN.md` (rationale/history).
This file is the spec a coding agent follows to emit a new, operational concept agent from an
**object spec** (Step 0) + the closest reference skeleton. `<obj>` = lowercase class, `<Obj>` = PascalCase.

Everything except the **belief model** (the SDF + its hooks) and a handful of spec-driven wiring is
mechanical copy + token-rename. The object spec is the only human input.

> ★ **Read [`CONCEPT_AGENT_LIFECYCLE.md`](CONCEPT_AGENT_LIFECYCLE.md) first.** This file specifies the
> *structure* a new agent must have (files, classes, DSR wiring); the lifecycle contract specifies the
> *behaviour* it must obey — CREATE · UPDATE · REMOVE · HISTORY · OWNERSHIP, which shared module owns each
> stage, and the invariants that are not negotiable. A generated agent is not correct until it has a
> conforming row in that file's §6 audit table. The contract exists because the same defect class was found
> and fixed three separate times in three agents that all had the right *structure*.

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
- **bottle:** the **occluding-contour silhouette** tangent term `dist(axis, ray)=radius` — the ONLY term
  that pins the depth-degenerate radius of a symmetric cylinder from a one-sided depth cloud.
- **chair:** none. Chair is now **pose-only** (see below): a fixed standard-chair TEMPLATE, so there is no
  size DOF to anchor. (Historically it carried seat-height + footprint-extent anchors when it fitted size;
  those were removed with the size DOFs — a per-axis-size chair would reinstate them.)
- **table:** none required (the box+legs SDF + extent likelihood suffice).

`<obj>_model.{h,cpp}` is now **only the state holder (`<Obj>State`) + the compound SDF** (+ any silhouette
store). All inference lives in `<obj>_belief`; the fitted posterior is written back into `<Obj>State` so the
downstream publish/viewer/RT code is unchanged. See `table_concept/src/table_belief.{h,cpp}` (box+legs, N=7,
w↔h/yaw symmetry fold + orientation-mode entropy — **the most evolved reference**),
`bottle_concept/src/bottle_belief.{h,cpp}` (single cylinder, N=5, silhouette `accumulate_extra`),
`chair_concept/src/chair_belief.{h,cpp}` (seat+back+legs but **pose-only N=3** — `[cx,cy,yaw]`, fixed
template, `cz` pinned to floor; the simplest legged model — see §5).

Everything else is **shared** and unchanged by a new object: `instance_tracker`, `mask_ingestor`, the worker
merge operator, `<obj>_scene_graph` (RT-cov maps the belief Σ), affordance / epistemic (Σ-based NBV) /
dashboard. So the §1 contract below holds; only the model hooks are authored.

### Belief invariants (2026-07 refresh) — a copy MUST keep these

- **Free energy is the true `−log` mixture MARGINAL likelihood** (the clutter component **included**), so F
  rises with misfit. A surface-only / responsibility-weighted energy reads ≈0 on a bad fit and *silently*
  breaks mode resolution and any reject-worse-fit logic. ★The shared engine's `update()` RETURN is that
  surface-only energy — do **not** publish/log/converge on it. Compute F with the model's `mixture_nll(pts,
  s, R)` (clutter-inclusive) and use THAT everywhere the agent reasons about fit quality: the published FE,
  the convergence gate, orientation-mode comparison, and any divergence/"explains-nothing" signal. (2026-07:
  chair published the surface-only FE; bottle additionally *converged and RETIRED nodes* on its `energy==0`
  all-clutter quirk — a band-aid for exactly this blindness. Both now use `mixture_nll`; bottle's divergence
  sentinel became a direct `clutter_fraction > k` test instead of the energy==0 proxy.)
- **The discrete orientation mode must reach the REPORTED covariance.** The N×N Σ carries only the
  *within-mode* yaw width. A sequential-Bayesian mode accumulator (`resolve_orientation`) that only edits the
  mean/state, without folding its entropy into the yaw variance the scene-graph and NBV planner consume,
  advertises a falsely-confident heading. Expose `covariance_reported()` (Σ with `p(1−p)`-style mode entropy
  added to the yaw block) and have the RT-cov upload + `EpistemicPlanner` read it, not raw `covariance()`.
  ★CLAMP the mode accumulator (±~6 nats) so a long confident run can still RECANT a physical rotation, and
  weight each frame's vote by ego-motion reliability `1/(1+(dotd/ref)²)` (reshapes/flips arrive on motion
  frames). Table (2-mode, w↔h) and chair (4-mode, backrest) both do this.
- **FE-surprise is the attention trigger** (once F is honest): an asymmetric-EMA baseline (down fast /
  up slow) + the smoothed positive gap `max(0, F−baseline)`; a sustained rise = the object moved, and
  reopens active perception without a hard re-detect. Update it only on an accepted-measurement frame. Table,
  chair, and bottle all carry it.
- **Obliquity yaw cap** (for any agent with a yaw DOF): grow the SHARED yaw variance as the yaw-carrying
  surface grazes edge-on (`gain·(1/|cos|−1)`), so a grazing view confirms the object but can't rotate a
  converged one. ★Key on the RIGHT surface: table = incidence vs the horizontal top; chair = the VERTICAL
  backrest (`n̂=(sinψ,−cosψ)`), a different geometry — port the *form*, re-tune the gain per surface.
- **Ego-motion → common-mode ("be still to UPDATE, else CONFIRM" — the VOR/fixation term).** A moving frame's
  mask is smeared/displaced by ego-motion (≈ effective-lag · speed), a per-mask SHARED error. Route it into the
  per-frame **common-mode** (`frame.chain_cov_*`), NOT per-point `R` — per-point `R` lets N points average the
  shared error away, while the Woodbury common-mode caps the frame's authority to move the **mean**, so a moving
  frame confirms the object but can't reshape/reposition/rotate a converged one; geometric updates concentrate
  at stillness. Add `(gain·|motion_dotd|)²` (motion_dotd = Z·‖ṡ‖ m/s, from the voxelizer) to the common-mode of
  each fitted geometric channel — position always; **size** (w,h / radius,height) — the anti-RESHAPE lever, the
  worst offender on rotation; **yaw** (agents with a yaw DOF) — the anti-ROTATE lever. Continuous (0 at
  stillness), no gate; one gain per channel (0 disables). Reference: `table_fitter.cpp` (`motion_cm_*_gain`),
  cabinet. Complement (action side): the affordance's `.still(v,ω)` dwell creates fixation windows.
  ★**Discrete confirm-only companion (`confirm_only()`).** The continuous common-mode above SOFTENS a moving
  frame; pair it with a HARD predict-only branch for frames where the robot is clearly moving — when
  ego-speed exceeds still thresholds, take `belief.predict()` (Σ carries one-step Q, **mean unchanged**)
  instead of `belief.update(frame)`, so a strongly-moving frame cannot touch geometry at all (it still
  CONFIRMS existence). Ego-speed = `max(|motion_dotd|, ego_lin + ang_lever·ego_ang)` where `ego_lin/ego_ang`
  come from **`room←zed` transform diffs** (producer-independent, works even when `motion_dotd` is stale).
  Config: `AI2MotionConfirmOnly` (master), `AI2StillLinMps`/`AI2StillAngRadps`/`AI2StillDotd` (thresholds).
  Reference: `chair_fitter.cpp` / `refrigerator_fitter.cpp` (`confirm_only()`, gate `gated = … or
  confirm_only(inst)`). ★2026-07 audit: **bottle_concept has only the continuous half** — port this gate.
- **No thresholds/gates** unless strictly necessary and flagged — encode the effect as a covariance/precision
  and let it fall out of inference (see `CLAUDE.md` modeling philosophy).
- **A mask is a LOWER BOUND on extent** (occlusion / foreshortening only shorten it) → extent evidence is
  **grow-only**; shrink is the occlusion-aware free-space (vacate) channel's job (through-beam = empty →
  shrink; short-return = occluded → hold).
- **Common-mode saturation**: the per-frame information caps at a shared-error covariance, so N correlated
  points can't collapse σ.

#### Box-fit robustness (any floor-anchored cuboid — fridge/cabinet/table-top; 2026-07 fridge refresh)
A single-box, partially-viewed object has three degenerate extent directions the mask cannot pin. Each is a
covariance keyed on the right observability covariate, NOT a clamp:
- **Unobserved-depth prior.** An extent DOF (depth) is identifiable ONLY when points fall on BOTH of its
  opposing faces — a front-only view is a thin slab that still *spuriously* drags the extent to its clamp
  (the size common-mode caps σ, not the mean). Grow that DOF's prior precision by `k·(1 − two_sided)` where
  `two_sided = 2·min(n₊,n₋)/(n₊+n₋)` counts points on the ± faces using a **fixed** forward/back margin δ (NOT
  keyed to the collapsing half-extent — that circularity re-collapses it). So a single face holds the extent
  at its footprint prior and it relaxes to data-driven the moment the far face is seen. Ref: `refrigerator_belief.cpp`
  `accumulate_extra` (`AI2DepthUnobsPrecision`, `AI2DepthObsBandM`).
- **Floor-anchored top is a FREE upper boundary.** Extending the top ABOVE the cloud costs nothing (empty
  surface is unpenalised) so `H` ratchets up on the mask's over-segmentation tail (junk points above the
  object) — observed `H`→2.37 m for a 1.9 m mask. Fix = (a) a firm TWO-SIDED anchor pinning `H` to the
  **observed robust top** (`z-p97`, which sits below the junk tail) + (b) grow per-point `R` with height ABOVE
  that top so the junk tail fades and can't ratchet. Data still sets the LOWER bound (front points force
  `H ≥ real top`). Ref: `refrigerator` `AI2TopNoFloatPrecision` / `AI2TopOversegSigmaPerM`.
- **Height-only birth for tall objects.** A partial front view is a vertical face → its 2-D footprint is a
  thin line → aspect/size are MEANINGLESS at birth and reject a real object. Gate BIRTH on the mask's
  z-extent alone (tall vs short); let the footprint settle later from the fit + footprint prior.
- **Per-cycle plausibility singleton.** Judge shape-plausibility (aspect·size·height) EVERY cycle from the
  CURRENT fitted state — NOT only on an accepted mask-fit. Otherwise a mis-detection that diverged to a wrong
  shape then coasts out-of-FoV FREEZES its birth-time positive evidence and stays immortal. Fold the per-cycle
  score into a bounded log-odds accumulator that drives existence; a stronger instance inhibits a weaker
  duplicate (soft singleton). Ref: `refrigerator` `specificworker_lifecycle.cpp` (singleton loop) +
  `singleton_existence_deltas`.

### Multi-sensor policy — one sensor births, peripheral sensors only cue attention

The primary dense/foveal sensor (**ZED**) is the ONLY one that **births and fits**. A peripheral 360 sensor
(**ricoh**) is used ONLY as a bearing+range **ATTENTION cue**: an unassigned detection (no known instance
along its bearing + rough range) raises a "seek a foveal view here" target and **never** touches
pose/extent/birth — its centroid/extent are biased by the oblique partial view. The robust range comes from
the **MEDIAN of the mask's 3D points**; association is **tight-in-bearing, generous-in-range**.

### Data flow (one `compute()` cycle)

```
masks (media plane) ──► MaskIngestor ──► InstanceTracker ──► assigned mask slices per <obj>
                                                                     │
lidar (media plane) ──► <Obj>LidarIngestor ──► <Obj>LidarRangeChannel.stage()
                                                                     ▼
   for each <obj> node:  Fitter.observe_slice() ──► Fitter.run_inference()  (once per assigned slice)
                              (ZED points → belief update: per-point SDF mixture
                               + footprint-moment factor + LiDAR range factor)
                                                                     ▼
                         Belief (recursive Laplace) ──► model ──► SceneGraph (RT + cov + geometry)
                                                                     ▼
                         EpistemicPlanner ──► affordance request (next-best-view)

   ricoh 360 slices ──► process_ricoh_bearings()  [ATTENTION ONLY — never fits]
```

---

## 0. Object spec (the prompt syntax)

Fill this from the human prompt + reference images. It is the complete input to generation.

> The `chair` example below is written in its **size-fitting** form (8 DOF, seat/extent `accumulate_extra`)
> to exercise every spec field. The **shipped** `chair_concept` is the pose-only N=3 reduction (§5) — a
> template with only `[cx,cy,yaw]` fitted. Both are valid; pick per the object (size-fitting like table, or
> pose-only-on-a-template like chair).

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

### ★ Agent-ID registry (authoritative — a collision = CRDT actor clash → SIGSEGV)
Every `[Agent] id` in `etc/config.toml` MUST be unique across the shared graph. A clone keeps its template's
id, so **re-number it as the FIRST edit** and record the new id HERE (the per-file id-map comments drift; this
table is the source of truth). Taken:

| id | agent | id | agent | id | agent |
|----|-------|----|-------|----|-------|
| 5 | room_concept | 10 | bottle_concept | 20 | chair_concept |
| 6 | robot_concept | 11 | kinova_controller | 21 | cabinet_concept |
| 7 | table_concept | 12 | self_calibration | 22 | refrigerator_concept |
| 8 | controller | 13 | human_concept | 23 | ring_metaconcept |
| 9 | voxelizer | 14 | residual_concept | 15 | door_concept |
| 16 | scene_graph_viewer | 24 | kitchen_metaconcept | | |

Next free: **17–19, 25+** (15 = door_concept, 2026-07-26; 24 = kitchen_metaconcept, 2026-08-09 — the
second meta-concept schema, RECTILINEAR. ★16 was already taken by scene_graph_viewer and was MISSING
from this table — the exact omission that caused both collisions below; added now.) (2026-07: cabinet=21 and refrigerator=21 collided because cabinet was never
recorded — refrigerator moved to 22. Same cause again 2026-07-26: the ring_metaconcept scaffold shipped
on 21, colliding with cabinet — moved to 23. A clone keeps its template's id; renumber it FIRST.) A one-line self-check: `grep -rE '^\s*id\s*=' */etc/config.toml`.

### ★ Node type = `object`, class in `object_subtype` (graph schema convention)
Every concept agent creates its instance node as the GENERIC DSR type **`object`** (`Node::create<object_node_type>`)
and writes its **class** into the **`object_subtype`** string attribute (`add_or_modify_attrib_local<object_subtype_att>`,
`"table"`, `"chair"`, `"bottle"`, `"cabinet"`, `"refrigerator"`). `object_subtype` is the CLASS and nothing else —
finer distinctions live in their own channel (e.g. table round/square is carried by the shape-selected `mesh_path`,
`round_table.obj` vs `table.obj`, NOT in `object_subtype`).
Rationale: `type()` is the FIRST-level discriminator after the root hierarchy; everything below branches on the
subtype string — so a cross-cutting consumer (controller, room, voxelizer, residual) does ONE
`get_nodes_by_type("object")` and reasons over all furniture, reading `object_subtype` for the specifics. Do NOT
mint a per-class cortex node type (`table_node_type`, `cylinder_node_type`, …). Consequences a copy MUST honour:
its own compute-loop query, stale-sweep, and affordance parent-backstop all key on `type()=="object"` **filtered by
the `<obj>_` NAME prefix** (many `object`s share the graph), NOT on a class-specific type. Reference: refrigerator.

### ★ Display mesh (required): publish `mesh_path` + `mesh_texture_path` per instance
Every concept agent MUST publish, on each instance node, a `mesh_path_att` (a **bare** OBJ/DAE filename resolved
under `voxelizer/meshes/`, e.g. `"table.obj"`) and a `mesh_texture_path_att` base-colour image, so the voxelizer
renders the real solid+textured mesh (it scales the asset to the fitted box; asset must be **pre-normalised and
pre-oriented** — orientation baked in, see the `mesh_path` contract in `graph_object_box.h`). A real asset file must
exist in `voxelizer/meshes/`. Empty `mesh_path` ⇒ the viewer falls back to the fitted box. table/cabinet/refrigerator
ship assets (`table.obj`/`round_table.obj`, `cabinet.obj`, `fridge.obj`); **chair and bottle currently publish only a
procedural `mesh_vertices` and need an OBJ asset added**.

### Module set (identical across agents; `+` = opt-in capability)
```
<obj>_config.{h,cpp}  <obj>_instance.h  <obj>_model.{h,cpp}  <obj>_belief.{h,cpp}  <obj>_fitter.{h,cpp}
<obj>_scene_graph.{h,cpp}  specificworker.{h,cpp}  specificworker_presence.cpp
+epistemic_planner.{h,cpp}  +<obj>_affordance.{h,cpp}  +<obj>_evaluator.{h,cpp}
sensor/memory collaborators (extracted from the fitter this session — behavior-preserving, so the fitter
stays a THIN orchestrator; each is opt-in per the object's sensing/memory needs):
+<obj>_projection.{h,cpp}     (camera projection: room_T_cam + in-image ROI (controller lock-on) +
    pixel-silhouette existence evidence; owns the CameraAPI)                              — +projection
+<obj>_lidar_range_channel.{h,cpp}  (YOLO-independent LiDAR first-hit range factor: stages the sweep and
    feeds range residuals to a frame)                                                     — +lidar
+<obj>_voxel_bank.h           (header-only free functions: <obj>-owned historical point memory —
    ownership gate + FNV voxel keys)                                                      — +voxel_bank
+<obj>_existence.{h,cpp}      (support-mass log-odds EXISTENCE belief → node REMOVAL, `common/existence_belief.h`:
    OCCUPANCY confirms (holds L up) · LiDAR through-beam free-space REMOVES · out-of-FoV / occlusion HOLDs.
    Debounced by a removal streak; NO age-immunity (evidence-based only). This is the "does it still exist?"
    channel — SEPARATE from and complementary to the tracker's negative-info death (which is OFF by default
    for persistent furniture, removal=MERGE+existence). `update_and_remove(fitter, lidar, …)` each cycle →
    `delete_node`+`forget_node`+`affordance.remove()`. Ref: table/refrigerator have the file; chair inlines
    it in `specificworker.cpp`; cabinet adapts it (kitchen presence log-odds). ★2026-07 audit: bottle has
    only tracker-death, no support-mass existence belief.)                                — +existence
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
<Obj>Observation    observe_slice(<Obj>Instance&, int slice_index);         // ONE assigned mask slice → observation
<Obj>Observation    observe(<Obj>Instance&, const DSR::Node&);               // stale fallback: whole-node mask → cand/residual split
float               run_inference(<Obj>Instance&, const <Obj>Observation&);  // [support pre-step] → belief.update → [z-anchor] → FE
std::unordered_map<std::uint64_t,<Obj>Instance>& instances();
void                forget_node(std::uint64_t id);
bool                should_log(const <Obj>Instance&) const;                  // NOT should_log_<obj>
void                note_birth(std::uint64_t id, const Eigen::Vector2f& xy); // tracker seeds the birth centroid
void                set_chain_cov_source(DSR::InnerGaussianAPI*, std::string source_frame);  // Part B (calling it enables the chain term)
```
**Multi-sensor fusion (the primary path).** A cycle now fuses *several* observations per instance. For each
mask slice the tracker assigned to the instance, the worker calls `observe_slice(inst, slice_index)` to build
an observation from that ONE slice and feeds it to a `run_inference`. Running the assigned slices one after
another is a **sequential Bayesian update** — the joint likelihood factored across sensors — with each sensor
keeping its own `R` (and its own within-frame common-mode marginalization). `observe(inst, node)` remains only
as the stale single-slice fallback. Fitter holds `<Obj>SceneGraph*` for **reads only**. Object-specific belief
steps (support surface, z anchor) live INSIDE `run_inference`, never the worker. The fit is
`inst.<obj>_belief.update(frame)`; the posterior is copied back into `inst.model` so downstream publish/RT is
unchanged.

### Worker — orchestration. `compute()`:
```cpp
mask_ingestor_->refresh();
run_instance_tracker();                     // THE birth/associate/death path — every cycle, not gated
for (const auto& node : G->get_nodes_by_type("<dsr_node_type>"))   // (bottle: "cylinder" named "bottle*")
    process_<obj>_node(node);
// per-cycle compute-rate heartbeat on std::cout (see below) — table/cabinet keep an EvidenceMonitor Hz EMA
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

**Compute-rate heartbeat (REQUIRED, every agent).** Every agent MUST log a per-cycle perf line on `std::cout`
(NOT `qInfo`, which RoboComp filters) so a stalled or CPU-pegged agent is visible from the terminal. Use the
shared `FPSCounter` (`#include <fps/fps.h>`, robocomp_core, header-only — no CMake/link change): a private
member `FPSCounter fps_counter_;` and, as the LAST line of `compute()`,
`fps_counter_.print("[<obj>_concept Compute]");`. Its `print()` emits **`Period = …ms. Fps = … <text> cpu = …%
mem = …MB`** — FPS **and** CPU% **and** RSS, once per ~1 s. All five concept agents carry this
(2026-07-25). The EvidenceMonitor `ev_g_.compute_hz` EMA (dashboard/monitor Hz readout) is a COMPLEMENT, not a
substitute — it is not on the log and carries no CPU; keep both where a dashboard exists.

### Primary-input stream gate (readiness + staleness) — every agent that consumes a live stream
The required-peer set proves the *producers exist*, not that the *stream is flowing*. Sitting in Operating
with a dead primary input looks healthy while nothing converges and — worse — re-integrates stale/frozen
frames as fresh evidence (CLAUDE.md: stop on a stalled consumed stream). Gate the FSM on the agent's
**primary input**, keyed per agent, reusing the SAME `presenceReady`/`presenceLost` signals as an extra
axis on top of presence (the coordinator has no notion of stream age — this is worker-owned):
- **Admission (Waiting→Operating).** AND a stream-**liveness** probe into `request_presence_ready` *and*
  re-poll it in an `on_waiting_loop` hook (the guarded event-driven fire declines silently when the stream
  isn't up yet at the instant peers arrive; the loop is what re-admits). ★Admit on **actual freshness**
  (a frame within the timeout), NOT on "the producer node exists" — a graph node (e.g. `masks`) PERSISTS
  after its producer dies, so admitting on existence re-admits straight into an instant re-stall (a
  tick-rate flap). room → `lidar_stream_ready()` works as bare existence only because its LiDAR ingestor is
  a **free-running thread** whose age advances off the FSM path. ★A **graph-node-polled** input (table
  reads the `masks` node only inside `compute()`) has NO such thread → you MUST pump the ingest
  (`ingestor->refresh()`) inside `on_waiting_loop` too, or liveness never updates while Waiting and the
  agent can neither avoid the flap nor detect the producer returning.
- **Staleness demotion (Operating→Degraded→Waiting).** In `on_operating_loop`, before `compute()`:
  `if (not stall_reported_ and stream_stalled(&age)) { stall_reported_=true; degraded_from_stream_=true;
  emit presenceLost(); return; }`. `stream_stalled` = ingestor `ms_since_last_frame() >
  <Stream>StallTimeoutMs`, with a **cold-start grace** measured from `operating_since_ms_` when no frame
  has ever arrived (`ms_since_last_frame()` returns −1). Reset `operating_since_ms_` + `stall_reported_` in
  `on_operating_enter`.
- **Degraded is recoverable, not terminal.** A stall routes through Degraded (its only exit is →Waiting) but
  peers are intact, so set a `degraded_from_stream_` flag: `on_degraded_enter` logs "stall, peers intact"
  and the SHARED required-loss grace timer finds all peers present and declines to shut down. The gate then
  re-admits when the producer returns. Contrast a *real* peer loss (flag false) → shutdown after grace.
- **Ingestor bookkeeping (shared readers).** Stamp a wall-clock ms on the single fresh-frame success path
  only; expose `ms_since_last_frame()` (−1 = never) + `stream_ready()`. Stamp on the wall clock, NEVER the
  source stamp — a producer republishing an old frame must read as *not* live. **Key on an id that advances
  on every producer heartbeat even with zero detections** (table's `mask_frame_id` is a monotonic *publish*
  counter → an empty scene never false-trips; a per-detection counter would). ★**Handle producer restart:**
  that publish counter is per-PROCESS and RESETS when the producer restarts, so a "new frame = id >
  last-seen" guard silently rejects the whole restarted stream (starving every consumer) until the fresh
  counter climbs past the stale value. Treat a **backward** id jump as a restart and adopt the new stream.
  Additions to a shared ingestor (`common/mask_ingestor`) must be backward-compatible so non-gating
  consumers ignore them.
- **Config.** One liveness key, `<Stream>StallTimeoutMs` (default 3000 ms) that MUST exceed the producer's
  HOLD/hysteresis window. This is a **lifecycle/liveness** bound, NOT a belief-precision knob (see §Belief
  invariants / no-thresholds) — orthogonal to Σ-aging, which handles *belief* staleness on the different,
  belief axis (both can be on at once; the stall gate trips before Σ-aging is even exercised). 0 disables.

References: `table_concept` (masks; `common/mask_ingestor` + `specificworker_presence.cpp`),
`room_concept` (lidar; `lidar_ingestor` + `specificworker_presence.cpp`). The **controller** deliberately
does NOT use this FSM-routing model — it uses a *local hold + stop-robot* response to a stalled stream
(it must halt the arm, not bounce to Waiting); do not collapse the two.

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
pose). Enabled uniformly via `fitter_->set_chain_cov_source(gaussian_api_.get(), "zed")` (calling it turns the chain term on):
the fitter computes it with `DSR::InnerGaussianAPI` (transform the fitted centre room→"zed" then back with a
ZERO input cov; `transform_point` returns exactly `J·Σ_chain·Jᵀ`, Σ_chain = each RT edge's `rt_covariance`
adjoint-composed; `room_concept` publishes the robot↔room term), pinned to the capture stamp. Its xy block
adds to the published translation cov. The scene-graph write self-gates (geometry republish OR trace change
>5%) to suppress churn once settled. The chain term is unconditionally on (the `set_chain_cov_source` call
enables it — there is no longer an `RtCovAddChain` toggle). COV ONLY — masks stay room-frame.

### Dashboard (`publish_<obj>_diagnostics`, if `dashboard`) — stacked `TimeSeriesPlot` panels, per node:
FE → dimensions → **posterior σ(mm)** per DOF (from `sqrt(diag Σ)` — the honest calibrated uncertainty; this
is what the old counter-evidence/CUSUM panel became once the stabiliser was removed). The gated AI2 CSV
(`<Obj>Model.AI2CsvPath`) logs per cycle: `cycle,node,pts,R,energy,frames_converged[,extra]` — the belief
state + Σ diag. (`TimeSeriesPlot` strokes lines only — never set a brush before `drawPath`, or peaks fill.)

### Comment & formatting conventions (two-tier) — *scannable at a glance, deep on demand*
- **File header** (every `.h`/`.cpp`): one paragraph — what this unit is, its single responsibility, and its
  collaborators. Reference-agent files also note their role in the pattern.
- **Two-tier doc comments.** Every public type/method opens with **one crisp summary line** (what it does /
  returns). Any non-obvious *why* follows below it, or is deferred to a design doc by name. Don't bury the
  summary inside a dense paragraph.
- **Section dividers**: `// ─── Title ─────────` inside a `.cpp` to group related methods.
- **Style**: C++23; write `and`/`or`/`not` (not `&&`/`||`/`!`); ~110-column soft limit; align member
  declarations. Prefer the shared `common/` factory/util over a hand-rolled copy.
- **Flag every threshold.** If a magic cutoff is truly unavoidable, say so in the comment and name the
  physical quantity it stands for.

### DSR attribute access — typed `<foo_att>` template forms ONLY (never `runtime_checked_*`)

Every graph attribute read/write uses the **type-attributed** template overloads —
`add_or_modify_attrib_local<foo_att>(node, v)`, `get_attrib_by_name<foo_att>(node)` — which are
compile-time-checked (`valid_type<name,Ta>()`) and self-documenting. **Do NOT** use the
`runtime_checked_*` forms (name-as-string, validated at runtime → a typo/type-mismatch is a runtime
throw). **`table_concept` is the exemplar: 100% typed, zero `runtime_checked_*`** — a copy MUST match it.

To add a NEW attribute: `REGISTER_TYPE(foo, <c++ type>, false)` in cortex's
`core/include/dsr/core/types/type_checking/dsr_attr_name.h` (generates the `foo_att` alias; large
read-mostly blobs → `std::reference_wrapper<const std::vector<T>>`, but you still SET by passing a plain
`std::vector<T>` by value — `valid_type` unwraps it), **the user reinstalls cortex** (root-owned header),
then rebuild the agent so its TUs see `foo_att`. Until the alias exists the typed call won't compile —
that's expected, not a reason to fall back to `runtime_checked_*`.

Two **legitimate** `runtime_checked_*` exceptions (flag them in a comment):
1. **Genuinely dynamic attribute name** — the attr name is a runtime value, not a compile-time constant
   (e.g. the per-node media-descriptor helper in `robot_concept/sensor_media_publisher.h`, which writes an
   `attr_name` param).
2. **A deliberately DSR-att-header-free header** — a lightweight header that takes `DSRGraph&` from its
   includer and includes no `dsr/` headers (so no `foo_att` alias is in scope); coupling it to the heavy
   att header just to type one write isn't worth it. (Same `sensor_media_publisher.h`.)

Anything else carrying `runtime_checked_*` is **opportunistic-migration debt**: register the attribute in
cortex, then switch to `<foo_att>`. Don't add new `runtime_checked_*` sites.

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

1. **Pick the closer skeleton** (all AI2): `table` (box top + inset legs, symmetry fold, size-fitting N=7 —
   the most evolved), `chair` (seat+back+legs, floor-anchored, **pose-only N=3 on a fixed template**, 4-way
   backrest disambiguation), `bottle` (single cylinder, silhouette `accumulate_extra`, single primitive).
   Legged + fit size → `table`; legged + rigid known dims → `chair`; single round body → `bottle`.
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
- [ ] **Unique `[Agent] id`** — re-numbered from the template and added to the §1 Agent-ID registry (a
      collision SIGSEGVs both agents). Verify: `grep -rE '^\s*id\s*=' */etc/config.toml`.
- [ ] Launches, presence reaches Operating (copy bottle's presence protocol verbatim; Degraded DEBOUNCE grace
      timer ~3000 ms, NOT immediate cleanup+exit).
- [ ] Primary-input stream gate wired: admission probe AND-ed in + re-polled in `on_waiting_loop`; stall
      demotion in `on_operating_loop`; recoverable-Degraded branch. Kill the producer → demote to Waiting &
      stay alive; restart → re-admit. `<Stream>StallTimeoutMs` > producer HOLD; empty scene never trips it.
- [ ] **No-update-while-moving**: continuous ego-motion→common-mode gains set AND the discrete `confirm_only()`
      predict-only gate wired (`AI2MotionConfirmOnly` + `AI2Still*`). Moving the robot confirms but does not
      reshape/rotate a converged object.
- [ ] **Removal**: either `+<obj>_existence` support-mass belief (occupancy-confirms / free-space-removes /
      out-of-FoV-holds, no age-immunity) or the tracker MERGE+negative-info death — a genuine physical
      disappearance retires the node; a look-away does NOT. Graceful SIGTERM deletes owned nodes;
      startup stale-sweep reaps leaked ones.
- [ ] All graph-attribute access is typed `<foo_att>` (zero `runtime_checked_*`).
- [ ] Node created as type **`object`** with `object_subtype="<obj>"`; the agent's own query / stale-sweep /
      affordance-backstop key on `type()=="object"` + the `<obj>_` name prefix (NOT a per-class type).
- [ ] Publishes `mesh_path` (bare OBJ/DAE filename in `voxelizer/meshes/`) + `mesh_texture_path` per instance;
      the asset exists and is pre-normalised/oriented; the voxelizer draws it scaled to the fitted box.
- [ ] Per-cycle **`FPSCounter fps_counter_.print("[<obj>_concept Compute]")`** on the last line of `compute()`
      → `std::cout` shows `Period/Fps/cpu%/mem` once/s. (`ev_g_.compute_hz` dashboard EMA is a complement, not a
      substitute.)
- [ ] On a fresh mask the tracker births a `<obj>_*` node; `ensure_instance` fires once; FE is finite and
      the fit moves toward the object.
- [ ] Dashboard panels populate (FE / dims / posterior σ); the gated AI2 CSV writes when its path is set.
- [ ] RT edge carries a calibrated 6×6 `rt_covariance` (+ the always-on `J·Σ_chain·Jᵀ` chain term).
- [ ] (affordance) `aff_<obj>_*` node appears with a sane target + ΔH; controller honours the contract's
      `.still(v, ω)` (dwells before completing a look).
- Day-one ≠ tuned: gains/deadbands/`epistemic.obs_distance` refine over a few live iterations.

---

## 5. Worked example — `chair_concept` (a POSE-ONLY legged model)

Chair is the **simplest legged agent**: a chair is rigid standard furniture, so the belief estimates only
its POSE against a fixed standard-chair TEMPLATE — it does **not** fit size. This deletes the whole
degenerate size space (`seat_w/seat_d/seat_h/back_h`) that a flat-clutter escape valve kept
collapsing/inflating one DOF at a time. (A global size scale could return later as ONE well-conditioned DOF;
per-axis size is a further extension that would reinstate the seat/extent `accumulate_extra` anchors.)

- `<Obj>Model` SDF = per-primitive box distances `{seat, back, leg×4}` in local frame, posed by
  `(cx,cy,yaw)` with the **template** dims and `cz` pinned to the floor; `<Obj>Belief::responsibilities`
  soft-mixes them `[seat, back, leg0..3, clutter]` with a z-band part attribution.
- `N=3`, DOFs `[cx,cy,yaw]`. Size = fixed template (`tpl_seat_w/d/h`, `tpl_back_h`); `cz = ai2_floor_z`
  (pinned, floor uncertainty → common-mode z). This is bottle-level conditioned: position is the point
  centroid, yaw is fixed by the asymmetric backrest.
- `canonicalize` = no-op (a chair has a front). `resolve_orientation` is a **4-way** sequential-Bayesian test
  over `{0,π/2,π,3π/2}` on the clutter-inclusive NLL (the seat+legs are ~symmetric, so only real backrest
  evidence moves it); its mode entropy folds into `covariance_reported()`'s yaw block, and it is clamped +
  ego-motion-weighted (see the belief invariants above).
- `accumulate_extra` = **none** (no size DOF to anchor). The obliquity yaw cap keys on the **vertical
  backrest** normal `n̂=(sinψ,−cosψ)`, not a horizontal top.
- `epistemic.target=hidden-face`, `degenerate_dof=yaw` (the backrest-revealing face). Tracker uses the belief
  mixture NLL (`association_nll`) for association under same-class clutter.
- Authored code ≈ `sdf_prim` + `responsibilities` + `resolve_orientation`/`covariance_reported` +
  `ChairBeliefState`; everything else is rename + spec-fill.

The **spec YAML in §0 shows the full 8-DOF size-fitting form** as an illustration of the prompt syntax; the
shipped chair is the pose-only reduction above. Start a new legged agent from whichever matches the object:
size-fitting (like table) or pose-only-on-a-template (like chair).

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
  `accumulate_extra` hook (the C++23 `requires`-detected extra GN factor — chair currently defines none, so a
  meta-consumer chair would ADD this one; bottle already has silhouette) — exactly the existing mechanism,
  zero engine change on the consumer side.
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
