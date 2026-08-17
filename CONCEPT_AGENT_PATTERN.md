# Concept-Agent Pattern (bottle / table / chair …)

Canonical structure for an `active_inference` "concept" agent: a CORTEX/DSR agent that
maintains a probabilistic (free-energy / SDF) belief over one object class, fed by YOLO
masks, written back to the shared DSR graph. `bottle_concept` and `table_concept` are the
two reference implementations; `chair_concept` (and any future object) is generated from
this pattern. Replace `<obj>` with the lowercase class (`bottle`, `table`, `chair`) and
`<Obj>` with the PascalCase form.

> **Generating a new agent?** Use the companion **`CONCEPT_AGENT_RECIPE.md`** — the concise,
> followable generation contract (object-spec schema + deterministic steps). This doc is the
> rationale, taxonomy, and history behind it.

---

## 1. Layer contract (confirmed canonical — table-style)

The agent is a **pure belief engine + thin orchestration + pluggable capabilities**. Each
layer has one responsibility; the Fitter never touches DSR.

| Layer | Owns | Must NOT |
|---|---|---|
| `SpecificWorker` | the per-node orchestration loop + capability wiring + Qt/lifecycle | hold belief math or raw DSR attr writes |
| `<Obj>Fitter` | the instance map + the belief: `ensure_instance` / `observe` / `run_inference→FE` | write DSR, hold a SceneGraph/Evaluator pointer |
| `<Obj>SceneGraph` | all DSR node/RT/attr I/O: `scaffold_missing_<obj>_nodes`, `persist_<obj>_belief`, covariance/diagnostics | run belief math |
| `<Obj>Model` | the generative SDF + free energy: `<Obj>State`, `<Obj>ModelParams`, `FreeEnergyDecomposition`, `<Obj>Model` | know about DSR |
| capabilities (opt-in) | validation / active-perception / dashboards | be required for the core fit to run |

**Canonical compute loop (in `SpecificWorker`):**
```
compute():
    mask_ingestor_->refresh()
    scene_graph_->scaffold_missing_<obj>_nodes(...)
    for each <obj> DSR node:
        process_<obj>_node(node)

process_<obj>_node(node):
    fitter_->ensure_instance(node, room)
    obs = fitter_->observe(inst, node)
    fe  = fitter_->run_inference(inst, obs)         # pure belief, no DSR
    scene_graph_->persist_<obj>_belief(inst, ...)   # all DSR writes here
    # optional capability hooks:
    [ evaluator_->... ]                             # validation
    [ step_epistemic(inst, node) ]                  # active perception
    [ dashboard feed ]                              # Qt
```

**Object-specific belief steps live INSIDE `run_inference`, not the worker.** Steps that are part of
the belief but unique to the object (they read DSR but never write the belief back) belong in the
fitter. The reference example is bottle: `BottleFitter::run_inference` calls a private
`update_support_surface(inst)` at its head (decide the resting surface room-vs-table + set the
table-top z anchor, reading via `scene_graph_`) and applies the z anchor (`cz = table_top + h/2`) at
its tail. The worker stays the thin orchestrator. (table has no analog — it *is* the surface.)

> **Done (Phase 2, 2026-06).** `BottleFitter` was migrated to this contract: `process_bottle_node`
> was removed, the `BottleEvaluator*` dropped from the fitter, and write-back/eval moved to
> `SpecificWorker::process_bottle_node`. bottle and table are now shape-identical.

---

## 2. File taxonomy — the part that matters for `chair`

The line-count "divergence" between bottle and table is **not drift**; it is each file's
relationship to the object. There are three classes of file:

### (A) Object-specific — COPY + rename + retype for each new agent
These are inherently bound to the object's geometry/schema. There is no shared copy.

| File | Type(s) | Object binding |
|---|---|---|
| `<obj>_config.{h,cpp}` | `<Obj>Config` + `load_<obj>_config()` | the agent's tunables |
| `<obj>_instance.h` | `<Obj>Instance` | per-object runtime state |
| `<obj>_model.{h,cpp}` | `<Obj>State`, `<Obj>ModelParams`, `<Obj>Model` | **the SDF — the only genuinely new code** |
| `<obj>_fitter.{h,cpp}` | `<Obj>Fitter` | typed on `<Obj>Model`/`<Obj>Instance` |
| `<obj>_scene_graph.{h,cpp}` | `<Obj>SceneGraph` | object DSR node type + geometry attrs |
| `prior_store.{h,cpp}` | `PriorStore` + `<Obj>Prior` | prior schema (`[[bottles]]`/`[[tables]]`/`[[chairs]]`) |

`<obj>_affordance.{h,cpp}` (`<Obj>Affordance`) and `<obj>` epistemic planner are also class (A): the
affordance node mechanics are object-agnostic but its parent + the planner's *target* are object-specific
(table = lowest-coverage face viewpoint; bottle = hidden-face far-side viewpoint), so each agent copies
+ adapts them.

### (B) Object-coupled-but-generic — share only via templating

| File | State |
|---|---|
| **`common/mask_ingestor/mask_ingestor.{h,cpp}`** | ✅ **HOISTED (2026-06).** The only coupling was `select_for_<obj>(<Obj>Instance)` → genericised to `select_nearest(query_centroid, label)` (caller builds the centroid). Now a single shared file used by both. |
| **`common/sample_queue/sample_queue.h`** | ✅ **HOISTED (2026-06).** Header-only `template<class Model> SampleQueue`; the 4 geometry primitives go to a per-object `SampleQueueGeometry<Model>` policy (`<obj>_concept/src/sample_queue_geometry.h`, bodies moved verbatim). Verified: the bulk methods were byte-identical between agents except the per-frame admission strategy, now gated by `SampleQueueParams::diversity_admission` (bottle=false / table=true) → **behaviour-preserving for both**. Requires `using State=<Obj>State;` on the model. ⚠ pending a **live golden-trace A/B** to confirm byte-identical FE/state. |

### (C) Object-agnostic — shared or copyable
| Unit | State |
|---|---|
| **`common/belief_stabilizer/belief_stabilizer.h`** | ⚠**SUPERSEDED — do NOT use in a new agent.** The seven object-concept agents moved to `common/ai_belief/recursive_laplace.h` (predict / MAP / Woodbury); the stabilizer, `sample_queue` and `prior_store` are not in any of them. It survives in **`human_concept` alone**, which is still on the pre-AI2 lineage — one more axis of that agent's 5/17 shared score. `common/sample_queue/` is dead code with no caller at all. |
| **`common/dashboard/timeseries_plot.{h,cpp}`** (`rc::TimeSeriesPlot`) | ✅ **HOISTED (2026-06).** Pure Qt widget, identical between agents. |
| **`common/dashboard/custom_widget.h`** (`Custom_widget`) | ✅ **HOISTED (2026-06).** Title passed via the constructor (`new Custom_widget("Bottle Model — …")`), so one shared widget. |
| `specificworker_presence.cpp` | ✅ **converged 2026-08-16.** The state hooks are 1-line delegators (generated boilerplate, not worth sharing); the stream-gate ARITHMETIC is `common/stream_gate/stream_gate.h`; the `rc::owned::Spec` is declared ONCE at file scope; and the per-instance cleanup loop is GONE in all seven (redundant with `[Owns]`, and it deleted while the monitor was live). 1198 → 984 lines. |
| **`common/stream_gate/stream_gate.h`** · **`common/obj/convergence.h`** · **`common/dashboard/belief_series.h`** · **`common/dashboard/belief_certainty.h`** · **`common/birth_surprise/residual_field_reader.h`** | ✅ **HOISTED 2026-08-16 (Tier A).** All family-agnostic — they take a covariance / a sample / a z-band, never a belief unit, which is why `fill_certainty` serves the six instance agents AND cabinet's three run-shaped sites. Adoption is asserted BY SYMBOL in `tools/concept_audit.sh`, not by include path. |
| `common/diag_log/rotating_csv.h` | ✅ **universal since 2026-08-16.** No diagnostics CSV may open with `std::ios::trunc` — a restart erases the run that produced the fault. Persisted state rewritten in full is the one exception and declares itself (`DIAG-ROTATE: exempt`). |

### Capabilities — object-specific shape, opt-in (both reference agents now carry these)
| Capability | Unit(s) | bottle | table | cdsl impact |
|---|---|---|---|---|
| Active perception | `EpistemicPlanner` + `<Obj>Affordance` | ✅ (hidden-face) | ✅ (faces) | — |
| Live dashboards | `Custom_widget` + `TimeSeriesPlot` | ✅ | ✅ | Qt |
| Gated CSV (debug/monitor) | Fisher CSV (`*.FisherCsvPath`) + optional epistemic CSV (`Epistemic.CsvPath`) | ✅ both | ✅ Fisher | — |
| Validation (Webots GT / sweep) | `<Obj>Evaluator` | ✅ | — | `requires Webots2Robocomp` |

---

## 3. Naming scheme (enforce on every agent)

- Files `<obj>_*`; types `<Obj>*` in `namespace rc` (except `SpecificWorker`, framework-global).
- Config loader: free function `load_<obj>_config(const ConfigLoader&) -> <Obj>Config`.
- Worker orchestration entry: `process_<obj>_node(node)` **in `SpecificWorker`**.
- DSR write-back entry: `persist_<obj>_belief(...)` **in `<Obj>SceneGraph`**.
- Fitter public API (exactly): `bool ensure_instance(node, room)` (true on first create), `observe`,
  `run_inference`, `instances`, `forget_node`, `should_log` (NOT `should_log_<obj>`). Both reference
  agents conform as of 2026-06.
- Shutdown: `cleanup_owned_nodes()` (presence-coordinator path). Startup stale-sweep:
  `remove_owned_<obj>_nodes()` — both agents have it now (deletes leftover `<obj>_*` nodes at
  `initialize` so a crashed run's drifted node isn't adopted before scaffold re-creates from priors).

---

## 4. Generating `chair_concept`

1. Copy class (A) + (B) files from the closest sibling (table, since a chair has a footprint
   + legs + a back), rename `table`→`chair`, retype to `ChairModel`/`ChairState`.
2. Write the **only genuinely new code**: `ChairModel`'s SDF (seat slab + backrest + legs) and
   its `ChairState`/`ChairModelParams`; add `[[chairs]]` to `etc/object_priors.toml`; map the
   `chair` DSR node type + YOLO label in the ingestor's `select_for_chair`.
3. Opt into capabilities by including the unit + its config flag (dashboards: yes; epistemic +
   affordance: likely yes; evaluator: only if validating in Webots).
4. `.cdsl`: `options dsr` (+ `requires Webots2Robocomp` only if the evaluator is included).

The promise: **chair = ChairModel (the SDF) + priors + config, plus mechanical renames.**

---

## 5. Phased alignment plan

- **Phase 1 — taxonomy + this doc (done).** Establishes the contract and the (A)/(B)/(C)
  classification. Conclusion: the three "shared" files are object-coupled, not drifted, so a
  blind re-sync is a no-op/harmful — true sharing is a Phase-3 templating task.
- **Phase 2 — the real structural alignment. ✅ DONE (2026-06).** `BottleFitter` is now the
  pure-Fitter contract: write-back/eval moved to `SpecificWorker::process_bottle_node`, the
  `BottleEvaluator*` dropped from the fitter (it keeps `BottleSceneGraph*` for READS only —
  support surface / table-top / robot covariance). bottle and table are shape-identical.
- **Phase 3 — hoist generics to `common/`. ⏳ PARTIAL (2026-06).** ✅ The belief stabiliser is
  hoisted: `common/belief_stabilizer/belief_stabilizer.h` (`rc::BeliefStabilizer<N>`) — the Fisher
  filter + maturity stiffener + mask-confidence weight + counter-evidence CUSUM gate — is shared and
  used by both agents (table N=8, bottle N=5). ✅ `MaskIngestor` hoisted to `common/mask_ingestor`
  (genericised `select_nearest`). ✅ `SampleQueue` templated into `common/sample_queue` (per-object
  geometry policy + `diversity_admission` flag; A/B sanity-passed live). ✅ `TimeSeriesPlot` +
  `Custom_widget` hoisted to `common/dashboard`. **Phase 3 complete** — the only per-agent files left
  are genuinely object-specific (config/instance/model/fitter/scene_graph/affordance/epistemic_planner/
  prior_store/sample_queue_geometry) + opt-in capabilities.

## 6. Known asymmetries (not divergences — same components, different adoption depth)

- **Stabiliser usage depth.** table drives its acceptance gain through `BeliefStabilizer::compute_acceptance`;
  bottle currently exercises only the Fisher accumulators (diagnostic / posterior-σ), with the
  Kalman/CUSUM acceptance not yet wired (`BottleInstance::stab` comment). Wiring bottle to
  `compute_acceptance` is a deferred behaviour change, not a structural fix.
- **Epistemic CSV.** bottle has a dedicated gated `Epistemic.CsvPath` writer (worker-side, fresh ΔH);
  table logs its epistemic state on the affordance node + dashboard only. Either is fine; add to table
  for parity if offline affordance traces are wanted there too.
