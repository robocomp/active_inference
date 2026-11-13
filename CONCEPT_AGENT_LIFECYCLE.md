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
- **Names and uids come from `common/instance_identity`** (2026-11-01): the next `<prefix>_<N>` is max(persisted counter in
  `etc/instance_identity.csv`, highest live N) + 1, written before use — never from a live-graph scan alone, which
  re-issued a dead instance's name after a restart. `object_uid = (agent_id << 40) | N`.
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
- **A SECOND AFFORDANCE PER OBJECT: `approach`** (2026-10-18, `common/pragmatic_affordance/object_approach.h`).
  An agent that adds `ObjectApproach` must call `approach.remove()` at **every** site it calls
  `affordance.remove()` (retire, death, merge), and `approach.on_node_deleted(id)` in `del_node_slot`. When the
  OBJECT node is deleted from outside, hand the approach to the next compute cycle to remove — never write from
  the slot. Its name `aff_<object>_approach` keeps the `aff_<prefix>` shape, so `remove_stale_affordances` sweeps
  it with no config change. Wired: chair.
- Stop with **SIGTERM/SIGINT only**. `kill -9` cannot be caught, so cleanup never runs and every owned node
  leaks into the persistent graph.
- A **startup stale-sweep** is the safety net for a previous crash. Because nodes may still be syncing from
  the persistent server at `initialize()`, sweep again once on first entry to Operating.

---

## 6. Conformance audit (2026-07-31)

Derived by reading each agent, not inferred from includes. `—` = not applicable to that agent's sensing.

| agent | CREATE | UPDATE | admissibility | REMOVE | P(detect) | wall LoS | HISTORY record | OWNERSHIP |
|---|---|---|---|---|---|---|---|---|
| table | ✅ resurrects from ltom recall | ✅ | ✅ trunc + fixation | ✅ `rc::exist` | ✅ | ✅ shared helper | ✅ | ✅ |
| chair | ✅ resurrects from ltom recall | ✅ | ✅ trunc + confirm_only + fixation | ❌ **hand-rolled** | ✅ shared envelope (`chair_detect.h`: `term_envelope` axes 1 × in_fov × central) | ✅ `walls_block` *(2026-10-31, below)* | ✅ | ✅ *(fixed)* |
| bottle | ✅ resurrects from ltom recall | ✅ | ✅ confirm_only | ⚠️ tracker death only | — | — | ⚠️ unattributable | ✅ |
| cabinet | ✅ resurrects from ltom recall | ✅ | ✅ trunc | ✅ `rc::exist` | ✅ | ✅ shared helper | ✅ | ✅ |
| door | ✅ birth by comparison (L > 0 on the door model; tracker enforces separation + association) + 2 justified non-tracker creations *(2026-11-07, below)* | ✅ | ✅ one trial per VIEW (view ledger), κ-weighted | ⚠️ costed deletion on the door's own belief, **not** `rc::exist` *(2026-11-07, below)* | ✅ | ✅ shared helper | ✅ | ✅ *(fixed; + `aff_door_look_*` 2026-10-21, below)* |
| refrigerator | ✅ resurrects from ltom recall | ✅ | ✅ trunc + confirm_only | ✅ `rc::exist` | ✅ | ✅ *(now shared)* | ✅ | ✅ |
| residual | ✅ | ✅ | — field model, no per-frame gate | ✅ `rc::exist` | — | — | ⚠️ unattributable | ✅ |
| human | ❌ | ❌ `belief_stabilizer` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| language_agent *(2026-10-12)* | — no instances | — no belief | — | — | — | — | — | ✅ |
| self_calibration *(2026-10-23, skeleton)* | — no instances | — no belief | — | — | — | — | — | ✅ |
| ltsm `recognition` / `recall_late` / `staged` *(2026-10-24)* | — no instances (decides which KNOWN room this is; births nothing) | — no belief | — | ❌ metric evidence only, no occupancy/absence channel | — | — | ❌ no phantom record | ✅ |
| retina `place_evidence` *(2026-10-24, PLANNED, not producing)* | — | — | — | — | — | — | ❌ not built | ❌ not yet owned |
| ltom_agent *(2026-11-01, observer + recall offers)* | — never writes an instance | — no belief | — | — | — | — | — | ✅ |

*2026-10-31 (perception cards, Task 9):* chair's wall LoS cell went from `—` to ✅. `ChairFitter::zed_detectability` now delegates to `chair_p_detect` (`chair_concept/src/chair_detect.h`): `term_envelope(roi_fill) × in_fov × central × walls_block(camera, chair, room polygon)`; a chair in another room gets P(detect) 0 and its absence is no longer charged. Retired `Existence.ZedEdgeOffset/ZedRangeFull/ZedRangeRef/ZedClearLosFloor`. Live check pending.

*2026-11-01 (ltom recall, step 3):* chair's CREATE gained **resurrection**: at a tracker birth, `chair_concept/src/recall_candidates.h` reads the `ltom_recall_<uid>` metaconcept nodes ltom offers under the current room (own agent's uid, `chair_<N>` not live) and `rc::recall::decide` (common/recall_match) picks the MAP hypothesis over {recalled candidates, new} on the room-frame centroid alone (measurement cov `DetectionNoiseM^2 I`, `Tracker.RecallPiMove` / `RecallLogPriorNew`). The newborn then takes that candidate's `chair_<N>` and `object_uid` instead of issuing new ones; a candidate is taken once per cycle. Birth stays tracker-only: resurrection only chooses the identity, never creates, delays or suppresses a birth. Logged as `RESURRECT chair_N uid=.. p=..`. Live check pending.

*2026-11-08 (working room follows the robot, chair only so far):* chair_concept no longer latches its room. `rc::room::follow_current_room` re-resolves it every cycle, so a birth is parented to the room the robot is in. REMOVE gained a channel: a chair whose `parent` is not that room is `left_behind` and is deleted, because its pose is not in this frame and fitting it would corrupt what ltom remembers; ltom offers it back on return. table, bottle, cabinet, refrigerator and hood still latch.

*2026-11-04 (ltom recall, step 4a):* the CREATE cell of table, chair, bottle, cabinet and refrigerator now reads "resurrects from ltom recall", and hood (no row of its own in this audit) does the same: a tracker birth may take the remembered instance's number and uid (`rc::choose_recall`, MAP over {recalled candidates, new}; birth itself stays tracker-only). Cabinet does it on both tracker birth paths (detection and residual-arm seed); `run_kitchen_model()` nodes carry no uid and are left out.

**language_agent** is not a concept agent: it births no instance and holds no belief, so CREATE…HISTORY do not apply
(`—`, not `❌`). It owns exactly one node, `chat_intention` (type `intention`, has_intention edge from `Shadow`), the
command channel to the controller: listed in `[Owns]` (`["language_agent 25", "chat_intention"]`, id matches `[Agent].id`),
stamped with `rc::provenance::stamp_creation`, swept at `initialize()` and again on first Operating, deleted by the
shared coordinator on SIGTERM/SIGINT (verified 2026-10-12: `[Graph] node 'chat_intention' deleted`). It connects no
DSR signal; the digest and the command status are polled on the main thread.

**self_calibration** (skeleton, 2026-10-23) is not a concept agent either: no instance, no belief, so CREATE…HISTORY are
`—`. It owns only its own agent node `self_calibration 27` (`[Owns] nodes = ["self_calibration 27"]`, id matches
`[Agent].id = 27`), deleted by the shared coordinator on SIGTERM (verified 5/5 on an isolated domain: `[Graph] node
'self_calibration 27' deleted`, exit 0). Presence is the shared `rc::presence::ConceptProtocol` (debounced Degraded,
grace 3000 ms; required `robot_concept`, `room_concept`); it connects no DSR signal. No ❌. When it later writes
proposal attributes on its own node, that node stays its only owned state (robot_concept remains the sole writer of
body→sensor RT edges).

**Room recognition (2026-10-24; spec `docs/superpowers/specs/2026-10-22-room-recognition-design.md`).** Built METRIC-FIRST by
decision: the appearance expert (retina `place_evidence` / PlaceMemory, plan Tasks 2 and 4) is DEFERRED and does not exist;
the recognizer READS a `place_evidence` node if one is published (`room_recognizer.cpp` sets the per-room `available` flag when the node is present), but nothing publishes it, so appearance is absent live. Nothing below depends on it.
- **ltsm `recognition`** (metaconcept, `recog_room_ids`, `recog_posterior`; `ltsm_agent/src/room_recognizer.{h,cpp}`,
  trace `recognition_trace.csv`) is ltsm-owned, removed on exit (`remove_owned()`) and swept at startup (`sweep_stale()`). It
  births no room: it only REQUESTS a swap or undo. **`recall_late`** (`recall_<room>`, subtype `room_recall`,
  `recall_pose_prior`) and the twin flag **`object_subtype "staged"`** are ltsm-owned too (`ltsm_agent/src/room_swap.{h,cpp}`).
  The only cross-agent write is `valid = false` by room_concept on a recall it rejected (it never deletes it). That is the
  late-adoption contract (`room_swap.h`; room_concept side `room_concept/src/room_scene_graph_late_recall.cpp`).
- **The undo window is the eviction fence.** The swap is reversible until the robot leaves the recalled room; then
  `on_room_left()` deletes the staged twin and the late recall and releases the hold. While staged, K's memory twin is
  read-only to eviction (`RoomEviction::hold`). Details in `ltsm_agent/EVICTION.md`.
- door_concept reads a late recall as a recalled branch and records `IdentityDrop` for the provisional room's doors
  (`door_concept/src/specificworker.cpp`); the controller's cross leg survives the rename (pinned by the anchored case in `controller/tests/test_door_passage.cpp`).
- **❌ justifications.** ltsm REMOVE: the recogniser has no occupancy/absence channel because it judges a hypothesis over rooms
  (a posterior from the pose seam), not an object's existence; the swap's own reversal is the undo above. ltsm HISTORY: no
  phantom record is kept; `recognition_trace.csv` records each decision's evidence but a false recognition is undone, not
  clustered, and there is no place×bearing hypothesis to test on it. retina `place_evidence`: PLANNED, not producing;
  every cell is ❌ because there is nothing to conform yet. Revisit the row when Tasks 2 and 4 are built (its node will need
  an `[Owns]` entry matching `[Agent].id`; if retina ever publishes it; ltsm sweeps a leftover at start). The retina PLANNED row is excluded from conformance counts.

**ltom_agent** (2026-11-01) is an OBSERVER that also OFFERS recalls: CREATE/UPDATE/REMOVE of instances are `—` (it never writes an object node and never decides
identity). It owns its agent node `ltom_agent 28` plus the `ltom_recall_<uid>` metaconcept nodes it publishes (RT children of the current room, never a proto-room) for remembered, absent objects (`[Owns] nodes = ["ltom_agent 28", "ltom_recall_*"]`, id matches `[Agent].id = 28`; crash recovery: `RecallPublisher::sync` reconciles to the wanted set every poll, plus a stale sweep at `initialize()` and first Operating), deleted by the
shared coordinator on SIGTERM (verified on isolated domain 64: `[Graph] node 'ltom_agent 28' deleted`). It connects no DSR signal (polls on the
main thread while Operating) and keeps its repertory in a file; a node disappearing while its owner is gone or Degraded is "owner exit", never death, and an instance still on the graph but not resolvable this poll (room deleted, no covariance, ...) is HELD, never counted as gone.

**door_concept CREATE (amended 2026-11-07, docs/superpowers/specs/2026-11-07-door-existence-model-design.md).** Birth is BY
COMPARISON, not by a streak: a door place has ONE belief `L = structural prior + carried evidence + crossing evidence + Σ views`
(`core/place_belief.h`), a pending candidate (a detection no live door explains) receives every frame through the same
likelihoods a live door does (`core/existence_update.h`), and the place is born when `L > 0` — one good hit view can do it,
a 3 m "door" or a candidate off any wall cannot (the place prior, `core/place_prior.h`: wall band, size 0.85 ± 0.12 x
2.05 ± 0.06, first door on a wall 0.4, each further one 0.01). The TRACKER still enforces what it always did: separation
(`Tracker.BirthMinSepM`), association, the claims of other concepts, the swept-arc explanation; `Tracker.BirthFrames` and the
birth bank's ripening are gone (the config loader REFUSES the retired keys: `door_config_retired.h`). **Deletion is costed:**
`L < −ln(C_delete / C_keep)` (`core/door_decision.h`; the costs are the robot's own experience in seconds, learned online and
kept in `etc/door_costs.csv`), never for a door being crossed, with no debounce, streak, timer or clamp. A removed place keeps
its row ("no door here", `removed_ms`) and is the prior of any candidate there — rows are never forgotten; the RAM ghost
registry and `forget_identity` are gone. Two creations do not go through the tracker, and neither is a birth in §1's sense:
both RESTORE a door that some earlier evidence already birthed, as a belief the model then judges like any other door:
1. **The carried door** (`release_room` / `recreate_lost_door` / `install_carried_door`, and the late carry in
   `step_room_following`): the doorway the robot has just crossed, recreated (or its proto-room entry mirror adopted)
   in the room entered **through the carried path**: `carried_belief(source face's row, crossing_logodds)` — the face's
   evidence, plus `ln(CrossPassablePrior / CrossLeak)` for the body having passed through the aperture, an EMPTY ledger (views
   are per room), the new room's prior entered by the first update; role `BeingCrossed`, so no view can delete it. The robot's
   body went through the aperture, so the door's existence was established by the crossing itself; from the new side an open
   door is the one thing the tracker cannot re-detect.
2. **Restored remembered doors** (`restore_remembered_doors`, `door_concept/src/door_identity_table.h`): on entering a
   room that is a RECALL — ltsm_agent announced it with a `recall_<room>` node (object_subtype `room_recall`) since
   the previous room switch — every door of that room in `etc/door_identities.csv` that has a remembered pose and is not a
   REMOVED row is recreated at its remembered room-frame pose, size and leaf memory, unless a door on that hole is already
   live (the carried door is never duplicated). Its belief IS the row's (`PlaceTable::restored_belief`): the stored view
   ledger and the evidence aged by the change hazard the place table LEARNS from revisits (`note_revisit`: a remembered door
   found again is "unchanged", one removed on a revisit is a change; Gamma prior, mean 1/600 s^-1) — not a fresh birth's prior
   and not certainty. Seen ⇒ confirmed, absent where it should be visible (detect_prob, wall line of sight) ⇒ deleted by the
   costed line, its row kept as REMOVED. *(2026-10-22, D-C round 2)* Its PLACEMENT prior is the memory's too: σ_s² = the
   row's `place_sd` ⊕ `Existence.BoundarySigmaM`, not the birth seed's 0.6 m. A fresh room never restores anything, whatever
   its name. Justification: the doors were birthed by evidence on an earlier visit and memory holds them; re-requiring a
   sighting made `cross door_1` fail in tourP2 because the 60 s exploration did not look at that wall. Rows of the older
   4-column table carry no pose and are skipped with a log line; rows without the ledger / visit / removed columns load with an
   empty ledger and the row's L kept.

**door_concept OWNERSHIP addition (2026-10-21): the look-at-a-remembered-door affordance.** For each identity row of
the current room whose door is NOT instantiated (identity-only or full), door_concept owns one epistemic affordance
`aff_door_look_<name>` (type `affordance`, `has_intention` from the ROOM node, `parent` = the room; contract = the
`door` servo contract, whose completion clauses the room cannot satisfy, so it ends by the producer's withdrawal or by
the contract timeout). It is placed by the shared NBV model (`common/nbv/viewpoint_score.h`, `door_look.h`) at the
standpoint that sees the remembered aperture and valued by the expected information gain about the door's existence
(p_detect · MI of a Bernoulli existence with the detector's hit/clutter rates). It is not a birth: no door node is
created. It is withdrawn when the door is re-born (identity match by name or hole), or when absence evidence from the
views the robot actually had — `rc::exist` accumulation weighted by P(detect) incl. wall line of sight, shared
`decide_removal` — rules it out (then not offered again this visit). Dropped on every room switch; swept by the
`aff_door` prefix (`rc::owned::remove_stale_affordances`) at startup and shutdown.

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
5. **door_concept's existence is its own model, not `rc::exist`** *(2026-11-07)*. The ⚠️ in the conformance row: the shared
   `rc::exist` channel (a HOLD-on-undetectable removal debounce over one silhouette) was replaced for the door by a generative
   model — one belief per place, one Bernoulli trial per VIEW per camera (hit/miss from `detect_prob`, which is 0 for a view
   that could not have reported the door, so unseen is never evidence), geometry and actuation as further channels in the same
   view record, and a costed decision. It keeps every §3 invariant (occupancy confirms, absence removes, occlusion/out-of-view
   HOLD, absence weighted by P(detect) with walls in the line of sight, no constant process noise) as properties of the model
   instead of a module, so a phantom the detector fires on in every frame is removed by distinct views at good fill (spec §5
   test 3) rather than reported "stalled". Justification for not using the shared module: its debounce is a streak (a timer in
   disguise) and its single silhouette cannot express the two cameras' different detectors; table/cabinet/refrigerator/residual
   still use it. `core/door_visibility.*` and `core/detector_llr.h` are the pieces another concept could take if it gets a
   second camera.

### 6.1 Object-kind axes (2026-10-18, DRAFT — classification only, nothing refactored yet)

Three **independent** questions about how a concept is individuated and how it changes. Each agent combines
its own answers; they are not four exclusive classes. The vocabulary is borrowed from linguistics (count vs
mass nouns, the animacy hierarchy). Computer vision draws the same line as *things* vs *stuff* (panoptic
segmentation). **Each agent declares its kind. The kind describes how we individuate the concept, not what it
intrinsically is**: shelving along a wall is mass, a freestanding shelf unit is count, and "a glass of water"
turns a mass into a count by packaging.

- **Individuation: count | mass.** A *count* concept has a natural unit, and its boundary is part of its
  identity (half a chair is not a chair). A *mass* concept is cumulative and divisive: any part of a counter is
  counter, and two abutting runs are one counter. Its identity comes from its **support** (the wall it runs
  along), not from its extent. Mass can be **per dimension**: a counter has a count-like cross-section (a tight
  prior on height and depth) and a mass-like length (a broad, support-bounded prior). A mass concept is
  rarely detectable without context, because detectors are trained on things, so its evidence is geometry,
  support and co-occurrence.
- **Kinematics: rigid | articulated.** An *articulated* concept carries latent joint states (door φ, drawer
  extension, fridge-door angle) and a parent/child part tree. Its affordances attach to joints. Visibility
  and absence must be computed **given the joint state**: an open leaf predicts the wall behind it is
  visible, and a shut one predicts it is not.
- **Agency: passive | animate.** An *animate* concept moves under its own policy. It is the one principled
  source of **intrinsic** process noise (§4.1 forbids constant Q for passive objects, whose state changes
  only when something acts on them). Occlusion should HOLD it more weakly, since it keeps moving while unseen.
  Eventually it needs an inferred intention, i.e. a model of another agent.

What each answer implies for the stages:

| | CREATE | UPDATE | REMOVE | HISTORY |
|---|---|---|---|---|
| **count** | tracker births one unit | bounded extent prior; pose + appearance identity | one `rc::exist` belief per unit | phantom = whole unit |
| **mass** | birth of an extent *cell* keyed to its support; growth/split/merge are UPDATES, not births/deaths | broad prior along mass dims, tight along count dims | per cell; absence trims, never deletes the whole | phantom = a cell; "re-key" churn is the mass-as-count symptom |
| **articulated** | parts are born under the parent | joint posterior; geometry predicted *through* the joint | absence evaluated under the joint posterior | joint changes recorded as events, not noise |
| **animate** | — | intrinsic, inferred process noise | occlusion HOLD decays with time unseen | trajectory history |

Current agents (read from code 2026-10-18; ⚠ = mismatch between the kind and how the agent treats it):

| agent | individuation | kinematics | agency | note |
|---|---|---|---|---|
| bottle | count | rigid | passive | |
| chair | count | rigid | passive | |
| table | count | rigid | passive | shape subtype is a count-like attribute |
| hood | count | rigid | passive | wall-mounted: count with a support parent |
| refrigerator | count | ⚠ rigid | passive | the door is only a *facing mode* (`resolve_front`); no door angle. Becomes articulated when the door is a joint |
| door | count | articulated (φ) | passive | the only agent with a joint posterior (`door_phi_posterior.h`); the phantom-door / "open door = missing wall" issue is the §6.1 visibility-given-joint rule |
| cabinet | **mass** (along the wall) | ⚠ rigid | passive | `cabinet_kitchen.h` already models runs as cells (`cell_mass`), and churn went to 0 after the kitchen-of-runs fix: the agent *discovered* it is mass. Drawers/doors are absent |
| residual | ⚠ count | rigid | passive | unexplained occupancy is closer to *stuff*, but it is born as box units |
| room | **mass** (walls, floor) | rigid | passive | no `instance_tracker`; the box cover of free space is a mass representation |
| human | count | articulated (skeleton) | **animate** | off-lineage (§6 deviation 1); its velocity/accel limits live in an output controller, not in an inferred Q |
| kitchen / ring metaconcepts | count (of an arrangement) | — | — | group members; a kitchen unit is a count wrapper over mass runs (packaging) |

Observations that motivated the draft. These are hypotheses, not decisions:

1. Every shared module in `common/` today (`instance_tracker`, `existence_belief`, `track/merge_instances`) is
   **count-shaped**. Cabinet and room implement mass behaviour privately. A shared mass/extent module (cells
   keyed to a support, with per-cell existence) is the main gap.
2. No shared joint module exists. Door φ is door-private. A drawer or fridge door would duplicate it.
3. Animate needs a different §4.1: inferred intrinsic Q is required, not forbidden. That should be stated, not
   left implicit in `human_concept`.

---

## 7. Checklist for a new agent

1. `common/instance_tracker` for CREATE; nothing else may birth.
2. `rc::ai::update`; per-frame quality into `Σc`; a stated admissibility gate; no circular observability covariates.
3. `common/existence_belief` for REMOVE; occupancy confirms / absence removes / occlusion HOLDS; absence
   weighted by P(detect) on the **saturated** delta; wall LoS via `rc::occlusion::walls_block`; no pd floor.
4. HISTORY: no constant `Q`; record phantom events from day one.
5. `[Owns]` lists the agent node **with the correct id** and the object glob; SIGTERM-only; startup stale-sweep.
6. Add a row to §6 and justify every `❌`.
7. Declare the concept's kind on the three §6.1 axes (count|mass per dimension, rigid|articulated, passive|animate)
   and add it to the §6.1 table.
