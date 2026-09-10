# Controller Epistemic Planner — Design

Design for an **interruptible, multi-step Expected-Free-Energy (EFE) planner** in the `controller`
agent: it selects, sequences, and executes the affordances that concept agents publish (table, bottle,
chair, human, room). Two hard requirements:

- **(a) Interruptible** — a higher-value affordance can preempt the one in progress.
- **(b) True multi-step EFE** — evaluate policies several steps ahead and execute richer,
  condition-controlled affordances, not one myopic pose at a time.

This is a **design document only** — no controller code changes land with it. It records the target
architecture and a staged implementation plan. It assumes the producer-side change already made in
`common/affordance_protocol.h`: affordances now carry an **object-relative viewpoint constraint**
(`ViewpointConstraint`: ranked faces + per-face gain + stand-off band + framing + Σ*), not an authoritative
world pose. See `../table_concept/TABLE.md` for the producer's NBV math.

---

## 1. The honest baseline (what exists today)

The pieces are mostly present but wired for a **single, sticky, myopic** target.

| Capability | State today | Location |
|---|---|---|
| EFE **selection** across affordances | Real, but **1-step and non-preemptive** | `common/affordance_manager/select_target` |
| **N-step arc-policy EFE rollout** | Fully written, **DEAD (not in build)** | `controller/src/epistemic_controller.cpp` |
| Object footprint enumeration | Works (`known_graph_obstacles_`) | `controller_obstacle_tracker.cpp` |
| Scene occupancy | **Polygon list only** (no sampleable grid); ESDF is local/transient | `controller_obstacle_tracker.cpp`, `trajectory_controller.h` |
| Collision test + target repair | Works (point-in-polygon; snap-to-free) | `controller_session.cpp:175-191` |
| Reachability | `plan_path` non-empty | `controller_session.cpp:225` |
| Execution primitives | `Reach`/`Servo`(lock-on)/`Orient`; **not preemptible** | `controller_session.cpp:306-601` |
| Hand-back / abort | `release_execution_claim` exists but is **never called** | `affordance_manager.cpp:404-432` |

**Selection objective today** (myopic, minimized):

```
G = λ_cost · nav_dist − epistemic_gain           (affordance_manager.cpp:450-464)
```

`epistemic_gain` is a common ΔH-nats currency across all producers. Two blockers:

1. **Non-preemptive.** If an incumbent is `Executing`, `select_target` returns it *before scoring any
   challenger* (`affordance_manager.cpp:493-522`). A higher-value affordance cannot get in.
2. **Myopic.** It ranks candidate poses one-shot; no horizon, no policy tree.

**Execution today** is a priority branch-cascade in `ControllerSession::execute_plan`
(`escape → no-plan → lockon → orient → MPPI`). Each "owns-the-base" branch **returns early**, so once
servoing/orienting/escaping starts, neither MPPI nor selection re-runs — **not preemptible mid-action**.

---

## 2. Design principles

1. **One objective, everywhere.** Selection, horizon, and interruption are all the *same* EFE comparison
   — interruptibility is not a special case, it is "the running policy is continuously re-scored against
   alternatives, with a commitment cost." No new priority scalars; keep the ΔH-nats currency.
2. **No thresholds where a covariance/precision belongs.** Commitment = a hysteresis *cost* in G
   (`switch_margin`), not an `if executing: don't switch` gate. "Done" = the producer's adequacy gap
   →0 (Σ reaches Σ*), not a tuned bound. (Consistent with the repo's modeling rule.)
3. **Split of knowledge.** The producer says *what view it needs* in the object frame; the controller
   owns *global feasibility* (occupancy + reachability) and resolves the concrete pose. Neither side
   duplicates the other's competence.
4. **Anticipation.** A policy's value includes how well each step *sets up the next* (boundary
   conditions), generalizing the existing one-step grasp→lift lookahead
   (`kinova EFE_CONTROLLER_MATH §4.11`) to the affordance-sequence level.

---

## 3. Architecture — four layers

```
        ┌──────────────────────────────────────────────────────────────┐
        │ (L1) SELECTION  — interruptible EFE arbitration over affordances│
        │      argmin_a G(a | belief, robot);  commit via switch_margin   │
        └───────────────┬──────────────────────────────────────────────┘
                        │ chosen affordance (or preempt signal)
        ┌───────────────▼──────────────────────────────────────────────┐
        │ (L2) HORIZON   — N-step policy EFE over affordance SEQUENCES    │
        │      resurrect epistemic_controller rollout; anticipatory glue  │
        └───────────────┬──────────────────────────────────────────────┘
                        │ next viewpoint constraint to realize
        ┌───────────────▼──────────────────────────────────────────────┐
        │ (L3) RESOLUTION — object-relative constraint → feasible pose    │
        │      faces × stand-offs → collision + reachability → best pose  │
        └───────────────┬──────────────────────────────────────────────┘
                        │ concrete (x,y,yaw) + contract
        ┌───────────────▼──────────────────────────────────────────────┐
        │ (L4) EXECUTION — preemptible primitives + clean teardown        │
        │      Reach/Servo/Orient that YIELD; release_execution_claim      │
        └──────────────────────────────────────────────────────────────┘
```

L1 and L4 deliver requirement (a); L2 delivers (b); L3 delivers the object-relative pivot.

---

## 4. (L1) Interruptible selection

**Change the arbiter from sticky to continuously-comparative.** Remove the incumbent-`Executing`
early-return (`affordance_manager.cpp:493-522`) and replace with an unconditional comparison:

```
G(a)          = λ_cost · nav_dist(a) − epistemic_value(a) − pragmatic_value(a)   // §5 adds pragmatic
incumbent i   = currently Executing affordance (if any)
challenger c* = argmin_{Offered ∪ Executing} G
switch  iff   G(i) − G(c*) > switch_margin      // commitment hysteresis (field already exists)
```

- `switch_margin` (already `select_switch_margin_`, config `AffordanceSwitchMargin`) is the commitment
  cost that prevents thrash — a challenger must be *meaningfully* better, not merely better.
- When a switch fires, emit a **preempt** for the incumbent (L4 tears it down + hands it back via
  `release_execution_claim`, so it returns to `Offered` for later, not `Consumed`).
- Producers already **decay `epistemic_gain` as Σ shrinks during execution** (`table_affordance.cpp:185-196`),
  so an incumbent naturally loses to a fresh high-ΔH challenger once it is nearly resolved — the
  interruption *emerges* from the value dynamics rather than a rule.

**Preempt path (the missing lifecycle):** add a single `preempt_current()` that (1) calls
`release_execution_claim` (sets `active=false` → `Offered`), and (2) triggers L4 teardown. Route any
external "attend now" trigger through the existing `command_queue_`/`enqueue_command` so it lands on the
control thread (never a DSR reader thread).

---

## 5. (L2) Multi-step EFE over affordance sequences

Generalize the **dead** `epistemic_controller.cpp` rollout from *navigation arcs* to *affordance
sequences*. A **policy** π is an ordered sequence of affordances/viewpoints; evaluate:

```
G(π) = Σ_{k=1..H}  [ w_cost · cost_k  −  w_epi · epistemic_k  −  w_prag · pragmatic_k ]  +  boundary(π)
```

- `epistemic_k` = expected ΔH from step k's view, using the producer's published per-face gains
  (already in nats) and their **adequacy-bounded** values — no new belief math on the controller side.
- `pragmatic_k` = goal-preference / task value. **This is the currently-missing term** at the affordance
  layer (today only `cost` and `epistemic` exist in `neg_efe`). Natural channel: the producer's published
  **Σ*** (precision demand) — reaching Σ* is the pragmatic payoff; distance-to-Σ* weights it.
- `boundary(π)` = **anticipation**: penalize a step whose end-state strands the next step near failure
  (e.g. a viewpoint that resolves table A but leaves the robot pointing away from the high-value table B).
  This is the one-step grasp→lift lookahead lifted to the sequence.
- Argmin over a **bounded** policy set (beam/greedy expansion, not a full tree — the horizon planner
  already generates a tractable candidate set). Reuse `generate_arc_policies`/`rollout_policy`/
  `evaluate_policy_efe` (`epistemic_controller.cpp:93-305`) as the rollout skeleton.

**Condition-controlled affordances** = the existing `Contract` machinery is already the vocabulary:
`GoalClause` predicates (AND-ed, held `stable_n`, `timeout_ms`, `on_fail`), stillness preconditions, and
the new object-relative constraint. The sequencer schedules affordances whose **preconditions** (a
predicate over the belief/graph) are met and whose **postconditions** set up the next — no new contract
type needed, just an optional `precondition` clause set mirroring `goal`.

**Cost of a policy step must be the RESOLVED cost.** Because poses are now resolved by L3 (collision-free,
reachable), `nav_dist` in the rollout should use the L3-resolved viewpoint (or a cheap lower bound during
rollout, exact for the committed step) — not the producer's hint pose.

---

## 6. (L3) Object-relative viewpoint resolution

Turn a `ViewpointConstraint` (parent object O; faces + gains; stand-off band; framing fill; Σ*) into a
concrete collision-free reachable base pose. **All the required data already exists** (see §1).

```
resolve(constraint, O):
  footprint ← known_graph_obstacles_[O.node_id]           // center, yaw, w, d  (already enumerated)
  best ← ∅
  for face in constraint.faces (ranked by gain):          // try high-gain faces first
      for standoff in [standoff_min .. standoff_max]:      // coarse sample of the band
          cand ← face_centre(footprint,face) + outward_normal·standoff   // NEW: face geometry helper
          heading ← atan2(O.cy − cand.y, O.cx − cand.x)
          if inside(inner_polygon_, cand)                  // room bounds (clearance-eroded)
             and free(cand, obstacle_polygons_ ⊕ robot_radius)   // NEW: inflate boxes by robot radius
             and plan_path(cand) non-empty:                // reachability (exists)
                score ← constraint.face_gain[face] − λ · nav_dist(cand)
                best ← argmax score
  return best  (or ∅ ⇒ this affordance is infeasible now → skip in L1/L2)
```

**Net-new pieces (small, self-contained):**
1. **Face geometry** — `face_centre`/`outward_normal` from an oriented box (trivial; producer already has
   the same math in `table_concept/epistemic_planner.cpp`).
2. **Candidate generator** — reuse the dead `epistemic_planner.cpp` `generate_candidates` grid/scoring as
   a skeleton (re-scope from localization-FIM to object-framing).
3. **Robot-radius inflation** of graph object boxes before the point-robot free-test (residual `grid`
   hulls are *already* inflated; graph boxes are **not** — must inflate for a fair test).
4. *(optional, later)* a **predictive framing model** so the controller can pick the stand-off that hits
   `framing_fill` a priori instead of only closed-loop after arrival. Until then, pick the mid-band
   stand-off and let the `Servo` lock-on drive `table_roi_fill` → `framing_fill` on arrival (works today).

**Why polygons, not the ESDF:** the ESDF is robot-frame, one-cycle, local — unusable for a faraway
candidate. Test candidates against the room-frame **polygon list** (available every cycle) instead. If a
true global costmap is ever needed, either have `residual_concept` publish the raw grid (attr/media plane)
or rasterize the polygons controller-side — out of scope here.

---

## 7. (L4) Preemptible execution + lifecycle

The correctness core. Every "owns-the-base" primitive must be able to **yield** between cycles, and a
switch must fully tear down per-action state.

- **Make `lockon_`, `Orient`, `escape` yield.** Each cycle, before the branch runs, check a `preempt`
  flag (set by L1). If set: stop the base, run teardown, clear the flag, fall through to re-selection.
- **Unified teardown** (today only `finalize_reached` does the full reset; `target_changed` does a
  partial one that **omits `lockon_`/`orient_`** — the known bug). Factor a single
  `reset_action_state()` covering: `current_plan_`, `last_target_info_`, `active_target_id_`,
  `active_contract_`, `feedback_node_id_`, `lockon_.reset()`, `orient_*`, `escape_*`, `stuck_since_ms_`,
  `last_mppi_*`. Call it from **both** `finalize_reached` (done) and `preempt_current()` (interrupted).
- **Hand-back vs consume.** On preemption call `release_execution_claim` (→ `Offered`, retryable). On
  completion keep `mark_reached` (→ `Completed`). On `on_fail=Abandon` timeout, release; on
  `on_fail=Consume`, mark reached. This gives the three distinct exits the protocol already models.
- **Graph write atomicity.** A switch writes `active=false` on A and `active=true` on B. Do both in the
  same control-thread cycle (never split across cycles) so another agent never observes two active
  claims or none. Both writes already go through `update_node` on the control thread.

---

## 8. Threading & DSR safety (hard constraints)

- **DSR update signals stay `Qt::QueuedConnection`** (main thread). `Qt::DirectConnection` corrupts the
  heap (FastDDS reader thread). Any new signal wiring obeys this (`specificworker.cpp:219-230`).
- **All planner work runs on `control_thread_`**, concurrent with DSR reader threads. An N-step rollout
  must **snapshot the graph once per cycle** (object footprints, affordance nodes, robot pose) into a
  plain struct and roll out against the snapshot — never query the live graph deep inside the rollout
  loop (contention + consistency).
- **External triggers via `command_queue_`** (`enqueue_command`), the existing marshalling path, so a
  "preempt now" never races `compute()`.

---

## 9. Contract / data touchpoints

- Consume the new `aff_view_*` attributes via `rc::affordance::read_viewpoint(node)` (added in
  `common/affordance_protocol.h`). If `object_relative` is false/absent, fall back to the legacy
  `epistemic_target_*` hint pose (backward compatible — nothing breaks before L3 lands).
- `epistemic_gain` / `epistemic_pending` / `active` protocol unchanged.
- Σ* (`aff_view_sigma_star`) becomes the **pragmatic-value channel** for §5 and the adequacy "done".

---

## 10. Staged implementation plan

Ordered so each stage is independently testable and low-risk before the next.

1. **Lifecycle + preemptible selection (delivers (a)).**
   - `reset_action_state()` unification + call it from `target_changed` (fixes the stranded-lockon bug).
   - `preempt_current()` = `release_execution_claim` + teardown.
   - Remove the incumbent early-return; add incumbent-vs-challenger with `switch_margin`.
   - Make `lockon_`/`Orient`/`escape` check the preempt flag and yield.
   - *Test:* two competing affordances; verify a higher-ΔH one preempts mid-servo and the old one returns
     to `Offered`, base motion stays smooth, no stranded state.

2. **Object-relative resolution (L3).**
   - `read_viewpoint` consumption + face geometry + candidate generator + robot-radius inflation.
   - Fall back to hint pose when absent.
   - *Test:* place an obstacle on the argmax face; verify the controller picks a feasible alternative
     face and frames the object; verify infeasible → affordance skipped, not stuck.

3. **Multi-step horizon EFE (L2, delivers (b)).**
   - Resurrect `epistemic_controller.cpp` into the build; re-scope rollout to affordance sequences;
     add the pragmatic (Σ*) term and the anticipatory `boundary(π)`.
   - Snapshot-based rollout on the control thread.
   - *Test:* three tables with different ΔH and geometry; verify the executed order minimizes sequence
     EFE (not greedy-nearest), and that a mid-sequence high-value arrival preempts correctly (L1 still on).

Each stage ships behind a config flag defaulting to today's behavior, so it can be A/B'd live.

---

## 11. Open questions / risks

- **Pragmatic value definition.** Σ* gives a precision-based pragmatic term for *perception* affordances.
  A future *manipulation* consumer (grasp/place — the controller is base-only today, no arm) will want a
  task-reward pragmatic term. Keep the term pluggable.
- **Rollout breadth.** Full policy trees are exponential; commit to beam/greedy with a small horizon
  (the dead planner uses `horizon_steps=20` at the *motion* level — at the *affordance* level H is 2–4).
  Log any truncation so "considered everything" is never implied when it wasn't.
- **Framing model.** Until a predictive framing model exists, stand-off is mid-band + closed-loop
  lock-on. Acceptable, but it means L3's "best pose" is framing-agnostic at selection time.
- **Preemption thrash under noisy gains.** `switch_margin` is the guard; may need the gain EMA the room
  planner already uses (`live_epistemic_gain` staleness/forgetting) to keep challenger scores stable.

---

## References

- Producer NBV math + the object-relative affordance: `../table_concept/TABLE.md`
- Shared affordance contract + `ViewpointConstraint`: `../common/affordance_protocol/affordance_protocol.h`
- Concept-agent pattern: `../CONCEPT_AGENT_RECIPE.md`
- Dead horizon-rollout skeleton to resurrect: `src/epistemic_controller.cpp`, `src/epistemic_planner.cpp`
- Selection arbiter to make preemptible: `../common/affordance_manager/affordance_manager.cpp`
- Anticipatory one-step precedent: `../kinova_controller/EFE_CONTROLLER_MATH.md` §4.11
