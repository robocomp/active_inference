# Concept-Agent Lifecycle Contract

The **normative** contract every `<obj>_concept` agent must satisfy. `CONCEPT_AGENT_RECIPE.md` describes the
module *structure* (files, classes, DSR wiring); this file describes the *behaviour* — what an agent is
allowed to do to the shared scene graph and on what evidence.

Why it exists: the same defect class has been found and fixed three separate times in three agents
(silhouette existence "seeing through walls": refrigerator → door → table+cabinet), and a threshold that
deleted objects the sensor provably could not see (`ZedClearLosFloor`) survived in a hand-rolled copy of a
stage that already had a shared, correct implementation. Both were possible because the lifecycle was
structure without a contract. **A stage stated here is a stage a new agent — or a generator — cannot forget.**

Five stages: **CREATE · UPDATE · REMOVE · HISTORY · OWNERSHIP**.

---

## 1. CREATE — birth is tracker-only

**Required module:** `common/instance_tracker`. No agent may birth an instance from its own logic.

- Birth is **evidence-accumulating, not a counter**: a detection matures a *pending candidate* over
  `birth_frames`, and each frame contributes `Detection::birth_evidence` (default `1.0`).
- `birth_evidence` is the **only sanctioned hook for a birth prior**. It is continuous: a corroborated
  detection matures faster, a suspect one slower. `Tracker.BirthFusion` already uses it in the *raise*
  direction (residual-field corroboration); §4.2 uses it in the *lower* direction.
- `Detection::birthable = false` marks a detection that may **refine** an existing track but never birth one
  (e.g. a peripheral/360 sensor). Multi-sensor policy: **one sensor births, peripheral sensors only cue
  attention.**
- `birth_min_sep_m` from every existing track *and* every other pending candidate — anti-duplicate.

**Invariant:** an agent must never create a graph node for an instance the tracker has not birthed.

**Every node this fleet inserts carries its birth time.** `common/graph_provenance/creation_stamp.h`
(`rc::provenance::stamp_creation(*G, node)`), called immediately before `insert_node`, writes one instant in
two forms: `timestamp_creation` (uint64 ms since the Unix epoch — code subtracts) and `creation_datetime`
(local civil ISO-8601 with offset, `"2026-08-06T14:32:07.512+0200"` — the DSR graph viewer's attribute table
shows this one). This is not optional decoration: without it the graph answers "what exists" but never "since
when", and the CRDT's own per-attribute timestamps move on every write, so they date the last update, not the
birth. `rc::provenance::age_s(G, node)` reads the age back.

Written **once, before the node exists**, and never refreshed. A re-acquired object gets a NEW stamp — it is a
new node with a new id, and only its *name* carried over (§1's re-acquisition rule); "when did this object
first appear, across deaths" is an existence-belief question, not a node-provenance one.

Scope is every node, not just instances: affordance children, `room`/`wall`/`floor`, the controller's
temporary obstacles and the retina's channel nodes are stamped too, so a stale node in the shared graph
can always be dated against the run that leaked it.

---

## 2. UPDATE — a frame must earn the right to move geometry

**Required module:** `rc::ai::update` (`common/ai_belief/recursive_laplace.h`) with the agent's own
generative model. The Bayesian math lives in the engine exactly once.

**Admissibility contract.** A frame may touch the geometric mean only when it is

1. **resolvable** — enough un-cluttered surface to constrain the state (point mass, clutter fraction,
   truncation). *Not distance*: a dense far view beats a starved near one.
2. **centred** — the object is in the sensor's reliable region, not clipping the periphery.
3. **still** — ego-motion smear below the level at which the mask is a shared blur.

Otherwise the cycle is **`predict()`-only**: mean HELD, Σ inflates. Association and existence do **not** read
this gate, so an inadmissible frame still *confirms* and still *tracks* — only geometry is frozen.

This is the attention/fixation analogue: intake is suppressed outside a fixation, not merely down-weighted.
It is a deliberate, flagged threshold, justified because the graded alternative provably cannot do the job —
the engine saturates a frame's information at `Σc⁻¹`, a **nonzero** asymptote, so a bad frame is attenuated
but still moves the mean, and accumulation over hundreds of frames beats attenuation.

**Per-frame quality is expressed as covariance, never as a gate.** Range, obliquity, coverage and ego-motion
enter through the per-frame common-mode `Σc` (Woodbury-marginalised in the engine) so N correlated points
cannot collapse σ. Add new quality signals there, not as `if` statements.

**Observability covariates must not be circular.** A covariate that decides whether a DOF is observable must
not be derived from the point estimate of that same DOF — a wrong estimate then declares its own error
unobservable and protects it. Evaluate under the posterior (or from data) instead. *Live precedent:* the
chair's backrest-normal obliquity, computed from the belief's own yaw, reported `oblq_cos = 0.001` and froze
a yaw that was ~45° wrong.

---

## 3. REMOVE — evidence, never a miss-counter

**Required module:** `common/existence_belief` (`rc::exist`). One scalar log-odds `L` per instance, clamped
to `±L_max` so it stays **finite and recoverable**.

**The discipline, in order of precedence:**

| observation | verdict |
|---|---|
| occupancy (a return / a lit mask inside the volume) | **confirms** — holds `L` up |
| predicted-visible but absent | **removes** — the only removal evidence |
| occluded, out of FoV, or not probed (`n_reached == 0`) | **HOLDS** — never absence |

**Mandatory clauses:**

- **Absence is always weighted by P(detect | present).** A miss only argues for removal to the degree the
  sensor could have resolved a present object from here. The weight must scale the **saturated per-cycle
  log-odds**, not the raw pixel/point count — scaling the count is a no-op, because the count is large enough
  that even a 2%-resolvable view still lands at the full `±llr_occ` ceiling.
- **Line of sight must include walls.** A wall is not a YOLO class, so mask-based occluders alone let an
  object in the next room project into the frustum, land on unmasked wall pixels, and vote its own removal at
  full strength. Use `rc::occlusion::walls_block()` against the room polygon. *This is the clause whose
  absence caused the same bug three times.*
- **No pd floor.** Any constant added under P(detect) re-creates removal-without-evidence. If an unresolvable
  instance must eventually be reaped, route it to a **verification pull** (go look), not to a silent decay.
- **Debounce on evidence cycles**, not wall-clock, so mixed sensor rates do not bias it.

An agent may substitute the tracker's negative-information death (`death_frames`) *only* if it has no
existence channel at all; this must be recorded as a deviation in §6.

---

## 4. HISTORY — no constant may stand in for experience

The stage that makes a model *settle*. Two clauses, both instances of one move: **a constant becomes an
inferred, covariate-dependent latent with a hyperprior pull-back.** See `MODEL_HISTORY.md` for the theory.

### 4.1 No agent may use a constant process noise

**Why.** For a static model (`F = I`) the predict step is `Σ ← Σ + Q`, applied every frame regardless of
whether anything was observed. Two consequences follow, and neither depends on how long the object has been
tracked.

**(a) Unobserved, precision decays linearly in frames.** `Σ(n) = Σ₀ + Q·n`, so the time to double the
variance is `τ = Σ₀/Q`. With `AI2ProcessStdM = 0.005` (`Q = 2.5e-5 m²/frame`):

| | Σ | τ = Σ/Q |
|---|---|---|
| converged (σ_w ≈ 0.03 m) | 9.0e-4 m² | 36 frames ≈ **3.6 s** |
| prior (σ_w = 0.30 m) | 9.0e-2 m² | 3600 frames ≈ 6 min |

A converged belief returns to its prior after **≈ 5.9 minutes** of not being looked at — and that ceiling is
the same whether the object was tracked for thirty seconds or three hours, because nothing shrinks `Q`.

*Measured, not asserted* (`table_concept/etc/ai2_log.csv`, table_9 cycles 970→2290, 1320 predict-only cycles
with `npts = 0` throughout): predicted `σ_w = √(9.0e-4 + 2.5e-5·1320) = 0.184 m`, observed **0.170 m**. The
next sparse mask — 178 points at 2.39 m — then landed on that 5×-loosened belief and re-cut the table.

**(b) Observed, precision does not accumulate at all — it sits at a floor.** In information form the update is
`Σ⁻¹ ← Σ⁻¹ + i` (with `i` capped at `Σc⁻¹` by the engine's common-mode marginalisation). The fixed point of
one predict + one update, `1/y = 1/(y+i) + q`, is `q·y² + q·i·y − i = 0`, giving for `q·i ≪ 1`:

```
y ≈ √(i/q)        ⇒        σ_steady ≈ (q / i)^¼
```

**A ratio of two rates — elapsed time does not appear.** Evidence is discarded by `Q` at exactly the rate it
is acquired, so three hours of observation leaves the belief in the same state as thirty seconds.

⚠️ Note what is *not* claimed: `τ = Σ/Q` is **not** identical across instances — a converged belief has the
*shorter* τ (36 frames vs 3600), since a small variance doubles sooner. What is identical is `Q` itself, the
absolute forgetting rate, and hence the retention ceiling `Σ_prior/Q`. **Resistance to disturbance is a
function of *current* Σ; current Σ is pinned to a floor by (b) while observed, and decays to the prior by (a)
when not, at a rate independent of history.** Nothing an agent learns extends how long it stays resistant.

**The requirement.** Each instance carries a per-DOF log-volatility `ω` with `Q = exp(ω)`, inferred online
(`MODEL_HISTORY.md` §3). Both quantities above then improve without bound as evidence accrues — the floor
`(q/i)^¼` *and* the decay time `Σ/Q` — which is exactly what a constant `Q` holds fixed.

This is also why a Σ-clamp is not a substitute: `clamp_to_prior` caps the worst case of (a) and does nothing
about (b). Hysteresis must be **earned by evidence** rather than imposed by a bound, and recoverability then
comes from the hyperprior's own pull-back rather than from a threshold.

### 4.2 Every agent must record phantom events; it may consume a false-alarm prior

Classifiers make *confident* mistakes under partial occlusion, so score cannot filter them — but the
recurrence is learnable: the same place, seen from the same bearing, lies the same way every tour.

- **Record**, always: every birth and every death, with world cell, robot pose, view bearing, age at death,
  and the existence state at death (`p_detect`, in-FoV, central, fixated).
- **Consume**, optionally: a per-class false-alarm field `p_FA` indexed on **(world cell × view bearing)** —
  never place alone, because the failure is viewpoint-dependent and a place-only index would suppress a
  *genuine* object placed there from every direction. It lowers `Detection::birth_evidence` (§1); it must
  never set `birthable = false`. Raising the evidence bar keeps the memory falsifiable.
- **Attribution rule:** a birth→death event is evidence of *either* a detector false positive *or* a removal
  false positive. Only count it as a phantom event when the disconfirmation was **confident** (high
  `p_detect`, fixated, close). Learning from unconfident deaths teaches the agent its own removal bugs.

---

## 5. OWNERSHIP — one mechanism, one policy

**Required module:** `[Owns]` in `etc/config.toml` + the presence coordinator. No agent may hand-roll node
cleanup.

- `Owns.nodes` matches **exact names, or a prefix when the entry ends in `*`**. The agent's own node is named
  `"<agent name> <agent id>"` — an entry whose id does not match `[Agent].id` silently matches **nothing**.
- List **both** the agent node and the object glob (e.g. `["table_concept 7", "table*"]`). Omitting the glob
  leaks every object node on shutdown, or forces a bespoke sweep that duplicates the shared one.
- ★**AND THE BESPOKE SWEEP IS NOT MERELY REDUNDANT — IT RACES** (measured 2026-08-16, fixed in all seven).
  Six agents also walked `fitter_->instances()` deleting each node by id. The glob already covers that set
  (checked against the names each agent actually creates), and the loop ran its deletes while the presence
  MONITOR was still live. `cleanup_owned_nodes()` is now: `remove_stale_affordance_nodes()` first —
  affordances are the orphan-safe case `[Owns]` does not name — then
  `presence_coordinator_.cleanup_owned_nodes()`, which stops the monitor before deleting. Nothing in
  between. bottle_concept reached this form on 2026-07-31 and chair's config comment recorded the same
  conclusion; neither propagated, which is the argument for keeping it in this file rather than in a
  comment.
- Stop with **SIGTERM/SIGINT only**. `kill -9` cannot be caught, so cleanup never runs and every owned node
  leaks into the persistent graph.
- A **startup stale-sweep** is the safety net for a previous crash. Because nodes may still be syncing from
  the persistent server at `initialize()`, sweep again once on first entry to Operating.

---

## 6. Conformance audit (2026-07-31)

Derived by reading each agent, not inferred from includes. `—` = not applicable to that agent's sensing.

| agent | CREATE | UPDATE | admissibility | REMOVE | P(detect) | wall LoS | HISTORY record | OWNERSHIP |
|---|---|---|---|---|---|---|---|---|
| table | ✅ | ✅ | ✅ trunc + fixation | ✅ `rc::exist` | ✅ | ✅ shared helper | ✅ | ✅ |
| chair | ✅ | ✅ | ✅ trunc + confirm_only + fixation | ❌ **hand-rolled** | ✅ `zed_detectability` | — no silhouette channel | ✅ | ✅ *(fixed)* |
| bottle | ✅ | ✅ | ✅ confirm_only | ⚠️ tracker death only | — | — | ⚠️ unattributable | ✅ |
| cabinet | ✅ | ✅ | ✅ trunc | ✅ `rc::exist` | ✅ | ✅ shared helper | ✅ | ✅ |
| door | ✅ | ✅ | ✅ trunc + confirm_only | ✅ `rc::exist` | ✅ | ✅ shared helper | ✅ | ✅ *(fixed)* |
| refrigerator | ✅ | ✅ | ✅ trunc + confirm_only | ✅ `rc::exist` | ✅ | ✅ *(now shared)* | ✅ | ✅ |
| residual | ✅ | ✅ | — field model, no per-frame gate | ✅ `rc::exist` | — | — | ⚠️ unattributable | ✅ |
| human | ❌ | ❌ `belief_stabilizer` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

⚠️ **unattributable** = the agent records births and deaths, but retires on *divergence* (clutter), not on
sensor absence, so it has no `p_detect` to report. Those deaths carry an explicit
`"retire-diverged (no existence channel)"` note and **must be excluded** when testing the (place × bearing)
clustering hypothesis — they cannot distinguish a classifier phantom from a model mismatch.

*HISTORY §4.1 (inferred Q) is unimplemented in **all** agents by design — it ships after the baseline run.
The HISTORY column above tracks §4.2 recording only, which is the part that must be live during the baseline.*

**Fixed 2026-07-31 (this pass):**

- `door_concept` `[Owns]` read `"door_concept 20"` while `[Agent].id` is **15** — a stale copy from chair.
  Matching is by exact name, and the node is `"door_concept 15"`, so the entry matched **nothing** and the
  agent's own node leaked into the persistent graph on every graceful shutdown. Now `["door_concept 15",
  "door_*"]`.
- `chair_concept` `[Owns]` gained `"chair_*"`, so instance cleanup runs through the shared coordinator
  instead of only the bespoke `remove_owned_chair_nodes()` (kept as the crash safety net).
- `refrigerator_concept` now calls `rc::occlusion::walls_block()` instead of its hand-rolled crossing test.
  Note the self-occlusion guard changed form (ray-parameter slack → metric `own_wall_skip_m = 0.70`); the
  validated 7 m-through-a-wall case is unaffected.
- `door_config`'s `AI2ObliquityYawGain` documented as a **dead key** (loaded, read by nothing — a door has no
  free yaw DOF). Kept rather than deleted because the open-door leaf angle would land there.

**Open deviations, in priority order:**

1. **`human_concept` is on a different lineage entirely** — no `instance_tracker`, no `ai_belief`, no
   existence channel. Sole user of `common/belief_stabilizer`. Either migrate it or declare it out of contract
   explicitly; today it is neither.
2. **`chair_concept` hand-rolls REMOVE** (`exist_logodds` + private gains). This is where `ZedClearLosFloor`
   lived. Port to `rc::exist` — first item after the baseline.
3. **`bottle_concept` has no existence channel**, relying on tracker `death_frames`. Acceptable under §3 but
   should be a conscious declaration; it is why its phantom deaths are unattributable.
4. **Chair and door still write the legacy 5-column `<obj>_events.csv`** alongside the new
   `<obj>_phantom_events.csv`. Harmless (different files) but redundant — fold `log_tracker_event` into the
   shared writer once the baseline confirms the new schema captures everything wanted.

---

## 7. Checklist for a new agent

1. `common/instance_tracker` for CREATE; nothing else may birth.
2. `rc::ai::update`; per-frame quality into `Σc`; a stated admissibility gate; no circular observability covariates.
3. `common/existence_belief` for REMOVE; occupancy confirms / absence removes / occlusion HOLDS; absence
   weighted by P(detect) on the **saturated** delta; wall LoS via `rc::occlusion::walls_block`; no pd floor.
4. HISTORY: no constant `Q`; record phantom events from day one.
5. `[Owns]` lists the agent node **with the correct id** and the object glob; SIGTERM-only; startup stale-sweep.
6. Add a row to §6 and justify every `❌`.
