# FSM.md — the pick-and-place sequencer (upper layer)

This documents the **upper-layer state machine** that sequences a pick-and-place on top of
the continuous EFE-gradient controller (`EFE_CONTROLLER_MATH.md`). The EFE controller is the
*action* layer (turns a preferred EE pose into joint velocities); the FSM here is the *policy*
layer (decides which pose to pursue and when to advance). It lives in `SpecificWorker::compute()`
(`src/specificworker.cpp`).

Keep this in sync with the code; line/constant names below are the source of truth.

---

## 1. Two levels

**Outer (`Phase`, `specificworker.h`)** — lifecycle:

```
WaitingForStart → SendingRestPose → Homing → ActiveEFE
```

The agent commands its **own** homing to `rest_pose` rather than trusting the bridge, so the
inner loop always starts from a known posture. Everything below runs inside `ActiveEFE`.

**Inner (`GraspPhase`)** — the pick-and-place cycle, repeated `round_cycles` times:

```
Tracking → Inserting → Closing → Lifting →
PlaceMoving → PlaceLowering → PlaceReleasing → PlaceRetreating → (next rep)
```

---

## 2. State diagram

```
            ┌──────────────────────────── next rep ◄─────────────────────────┐
            ▼                                                                 │
  ┌──► TRACKING ──(seated & aligned, settle)──► INSERTING ──(at grasp pt)──► CLOSING
  │      │ │                                       │ ▲                          │
  │  no-progress                               tip-bumper                   (force held
  │  (~3 s, reactive)                          halt+shift                  GRASP_FORCE_HOLD)
  │      │ └─► [force_top_down_]──┐                │ └──(contact clears → re-seat)
  │      ▼                        ▼                ▼                             ▼
  │  give up / retry         TOP-DOWN grasp   (insert timeout → miss)        LIFTING
  │                                                                             │
  │                                                              (bottle-rise CONFIRM ✓
  │                                                               → sample place spot via
  │                                                                capability map)
  │                                                                             ▼
  └─ PLACE_RETREATING ◄─(release)─ PLACE_RELEASING ◄─(on-table+upright)─ PLACE_LOWERING ◄─(via)─ PLACE_MOVING
        back off along                open gripper                       lower, relaxed orient   carry (look-ahead blend)
        gripper +Z axis
```

Reactive branches (dashed in spirit): top-down re-plan, tip-bumper reflex, tilt reflex,
miss→retry/give-up. Every phase except Tracking has a timeout; Tracking has a fast
no-progress abort plus a slow jam watchdog.

---

## 3. State table

| State | What it does | Exit gate (the trigger) | Watchdog → action |
|---|---|---|---|
| **Tracking** | approach the standoff (skill-collapsed toward the grasp), gripper open, track the bottle belief | `e_pos < REACH_TOLERANCE_M` **and** `e_ang < grasp_align_tol_rad_` (**full-frame** orientation, see §5C), then settle `GRASP_SETTLE_TICKS·(1−c)` cycles | no-progress `TRACK_NOPROGRESS_TICKS` (~3 s) → reactive **top-down** or give up; jam `TRACK_TIMEOUT_TICKS` (~18 s) → teleport+miss |
| **Inserting** | ease in along the latched approach axis | `e_pos < REACH_TOLERANCE_M` **or** `f > GRASP_FORCE_THRESH` | `INSERT_TIMEOUT_TICKS` (~3 s) → miss; tip-bumper → reflex (halt/back-off/shift) |
| **Closing** | hold pose, close fingers, watch force | `f > GRASP_FORCE_THRESH` held `GRASP_FORCE_HOLD_TICKS` | `CLOSING_TIMEOUT_TICKS` (~2 s) → miss |
| **Lifting** | raise along the bottle long axis (`up_axis`) | `via_reached(e_pos)` → **bottle-rise confirm** (`rise > LIFT_CONFIRM_RISE_M`, `xy_gap < LIFT_CONFIRM_HOLD_M`) | `LIFT_TIMEOUT_TICKS` → miss |
| **PlaceMoving** | carry to place-hover (look-ahead blend) | `via_reached(e_pos)` | `PLACE_TIMEOUT_TICKS` (~6 s) |
| **PlaceLowering** | lower onto the spot (relaxed gripper orientation) | **bottle on table** (`Δz≈0` **and** `z−table_top < PLACE_ON_TABLE_M`) **and** upright (`tilt < PLACE_UPRIGHT_TOL_RAD`), held `PLACE_SETTLE_TICKS` | `PLACE_TIMEOUT_TICKS` |
| **PlaceReleasing** | open gripper, latch the retreat frame | hold `release_ticks_` cycles | — |
| **PlaceRetreating** | pure back-off along the gripper +Z axis | `e_pos < REACH_TOLERANCE_M` | `PLACE_TIMEOUT_TICKS` |

`c` = `skill_c()` = `confidence_` (when `precision_reweighting`).

Grasp targets (point + approach) are chosen by `compute_side_grasp_target()`; place spots by
`sample_place_spot()`, both pre-filtered by the **capability map** (`reach_lookup`, see §6).
A confirmed grasp is decided by **ground-truth bottle rise**, not finger force (force is a coarse gate).

---

## 4. Adjustable trigger variables

**Gates / tolerances**
`REACH_TOLERANCE_M`=0.02 · `grasp_align_tol_rad_` (cfg `grasp_align_tol_deg`=8°) ·
`GRASP_FORCE_THRESH`=3 N · `GRASP_FORCE_HOLD_TICKS`=3 · `INSERT_TOUCH_FORCE`=0.3 N ·
`LIFT_CONFIRM_RISE_M`=0.06 · `LIFT_CONFIRM_HOLD_M`=0.12 ·
`PLACE_ON_TABLE_M`=0.04 · `PLACE_UPRIGHT_TOL_RAD`=0.15 · `PLACE_SETTLE_TICKS`=5 · `release_ticks_`=8 (cfg)

**Timeouts (cycles @ 20 ms)**
`TRACK_NOPROGRESS_TICKS`=150 · `TRACK_TIMEOUT_TICKS`=900 · `INSERT_TIMEOUT_TICKS`=150 ·
`CLOSING_TIMEOUT_TICKS`=100 · `LIFT_TIMEOUT_TICKS` · `PLACE_TIMEOUT_TICKS`=300

**Speeds (m/s)**
Track `0.35·vscale` · `INSERT_VEL_MS`=0.05 · Lift `skilled_speed(0.20)` ·
Place `skilled_speed(0.18)` · `retreat_speed`

**Geometry**
`standoff_collapse`=0.6 · `blend_radius`=0.04 · `APPROACH_STANDOFF_M` · `LIFT_HEIGHT_M` · `BOTTLE_GRASP_HEIGHT_FRAC`

---

## 5. The logic that adjusts the triggers

### (A) Skill / confidence — continuous, learned
A single scalar `confidence_ = Π_m/(Π_m+Π_s) ∈ [0,1]`, surfaced as `skill_c()`, re-allocates
precision across the whole cycle.

- **Update:** confirmed grasp → `confidence_ += conf_gain` (0.15, cap 1.0); miss → `confidence_ ×= conf_decay`. Persisted across rounds (`confidence_path`).
- **Effects (novice → skilled):**
  - speeds `×(1 + speed_conf_gain·c)` and insert `×(1 + insert_conf_gain·c)` → **faster**;
  - `standoff_collapse·c` slides the Tracking waypoint toward the grasp → **less creep**;
  - `blend_radius·c` → **more coarticulation** (fluidity);
  - settle dwell `GRASP_SETTLE_TICKS·(1−c)` → **commits sooner**;
  - **orientation-commit gate widens with `c`** → the skilled approach commits at a looser
    `e_ang` and lets the insert finish the alignment → **cuts the orientation crawl** (the
    approach's dominant time cost); anticipatory, see (D);
  - observation period grows with `c` → **samples vision less** (closed-loop → open-loop).

One knob takes a novice (slow, careful, closed-loop) to a skilled agent (fast, collapsed, open-loop).

### (B) Reactive — event-driven, immediate
Overrides that fire on sensed conditions, independent of skill:
- Tracking **no-progress** (3 s) → `force_top_down_` (reactive grasp re-plan) or give up;
- Inserting **tip-bumper contact** → `tip_reflex` (halt, back off, lateral shift);
- **tilt reflex** during Tracking; **bottle-rise** as the authoritative grasp confirm;
- **capability-map** pre-filter on place-spot (and, optionally, grasp) selection.

### (C) Grasp-frame orientation — pin the FULL frame, not yaw-free
`compute_side_grasp_target()` returns a fully-determined grasp frame: tool **+Z** = approach
(`z_tool_des`), tool **+X** = finger-closing (`x_tool_des = z_bot × z_tool_des`), tool **+Y** = up.
The azimuth (which side to approach from) is chosen **anticipatorily** — see (D) below — to keep the
*downstream lift* manipulable (not just the forearm off the mast),
so **all three axes are committed** and the EFE step must pin the whole frame:
`make_params` sets `p.align_tool_y = false` ⇒ `efe_gradient` takes the full-frame branch
(`R_des = [x⟂, z×x, z]`), driving tool +X → `x_tool_des` so the jaws seat **⟂ the approach** and
straddle the body.

> **Do not re-enable `align_tool_y` for the grasp.** That "yaw-free" mode pins only tool +Y to the
> bottle axis and leaves the wrist **yaw to the null-space**; the finger-closing axis then settles
> ~90° off — **broadside** to the bottle — and the Tracking `e_ang` gate (which in that mode checks
> *only* tool +Y) still reads "aligned". Result: FK reports a clean seat while the gripper shoves the
> bottle over. This was the root cause of the new-mount **grasp ~1/10** blocker (2026-06-07); fix =
> `align_tool_y = false`. After the fix, fixed-spot reliability ≈ **6/10 with probe ON** (vs 0/10
> before), failures now dominated by the probe perturbations on the thin (28 mm) bottle, not
> orientation. The "any azimuth grasps a cylinder" intuition is true only for *choosing the side*;
> once the standoff is fixed the yaw must orient the jaws ⟂ the approach. See
> `[[grasp-yaw-free-misorient-bug]]`.

### (D) Anticipatory selection — each phase frames the next (EFE over the policy)
The sequencer's organising principle: a phase chooses its preference to minimise the **expected**
free energy of the *next* phase, not its own free energy now (`[[anticipatory-efe-sequencer]]`,
`EFE_CONTROLLER_MATH.md §4.11`). Two instances are live:

- **Grasp anticipates lift (azimuth selection).** Because the cylinder is symmetric, the approach
  azimuth is free. `compute_side_grasp_target()` sweeps candidate azimuths (±120°, 7.5° steps) and
  scores each by `min(μ_grasp, μ_lift)` — the *weakest* manipulability across grasp **and** the
  straight-up lift endpoint (`predict_reach` IK) — committing to the `argmax` (cached per episode,
  `grasp_azimuth_z_`). This stopped the arm grabbing in a pose it then couldn't lift from (μ
  collapsing `0.09→0`, the bottle twisting to 40°). An IK-free **real-μ standoff guard** rotates the
  azimuth + re-approaches if the *actual* μ at the standoff `< LIFT_MU_MIN`, backstopping IK
  over-scores. → **8/8 clean vertical lifts** vs ~0/6 greedy.
- **Lift holds orientation as a hard constraint.** `Lifting` pins the whole EE angular velocity to
  zero (`hard_level_hold`) so the rise is a pure translation — the infinite-precision limit of the
  orientation preference (`EFE §4.10`).

> **Next (in progress): the approach anticipates the insert.** The approach is **orientation-bound**
> — position reaches the standoff fast, then the camera-up frame *crawls* asymptotically to the tight
> commit gate (`grasp_align_tol_deg`), ~75% of the episode time, with `c_approach` already saturated
> so the *speed* knob is maxed. Since (i) the lift no longer cares about the residual cant and (ii)
> the insert runs the *same* orientation target and so finishes the alignment while travelling, the
> commit gate is being **wired to `c_approach`**: a skilled approach commits with a **looser**
> orientation gate and hands the residual to the insert — precision learning applied to the
> *orientation-commit tolerance* (a precision), not just speed.

---

## 6. Capability-map pre-filter

`compute_reach_map()` sweeps the table on a 5 cm grid, runs **multi-seed DLS-IK + manipulability**
per cell, and marks a cell usable only if the IK config also **clears the column and stays above
the table** — so the map bakes in the *static scene* (arm pose + SolidPipe + table). It is
**recomputed every startup** (`precompute_reach_map`, ~0.1 s for 629 cells × 4 seeds), saved to
`reach_map_path_`, and held in memory (`rm_mu_`). `reach_lookup(x,y)` returns the cell μ (or −1 if
unreachable/blocked) in O(1). `sample_place_spot()` uses it to skip unusable spots and pick the
highest-μ one — replacing the unreliable single-seed online IK. Recompute per *mission* when the
scene changes.

---

## 7. Learnable vs. fixed (annotation)

Candidates to expose to **online / opportunistic learning** (per-rep precision learning,
`[[online-opportunistic-learning]]`) vs. those to **keep fixed for safety**:

| Knob | Class | Why |
|---|---|---|
| `standoff_collapse` | **learnable** | fluidity/time trade-off; already skill-scheduled — learn the schedule |
| orientation-commit gate (`grasp_align_tol`) | **learnable** | the approach's *dominant* time cost; skilled → looser gate, insert finishes the alignment. A precision, anticipatory (set by what the insert tolerates) |
| `blend_radius` | **learnable** | coarticulation amount; swept optimum 0.04, but scene-dependent |
| per-leg speeds (Track/Lift/Place) | **learnable** | time vs. reliability; reward = cycle-time − SPARC |
| `release_ticks_` | **learnable** | place-tail time; bounded below by finger-clear time |
| settle dwells (`GRASP_SETTLE_TICKS`, `PLACE_SETTLE_TICKS`) | **learnable** | confidence-gated dwell length |
| `speed_conf_gain`, `insert_conf_gain`, `conf_gain` | **learnable (meta)** | shape the skill schedule itself |
| capability-map μ threshold for target acceptance | **learnable** | how conservative to be near the boundary |
| — | — | — |
| `GRASP_FORCE_THRESH`, `INSERT_TOUCH_FORCE` | **fixed (safety)** | contact gates; mis-set → crush/miss |
| `LIFT_CONFIRM_RISE_M`, `LIFT_CONFIRM_HOLD_M` | **fixed (safety)** | the ground-truth grasp confirm; the integrity check |
| `PLACE_UPRIGHT_TOL_RAD`, `PLACE_ON_TABLE_M` | **fixed (safety)** | releasing tilted / mid-air topples the object |
| `REACH_TOLERANCE_M` | **fixed** | the controller's deadband; couples to stability |
| all timeouts | **fixed (safety)** | watchdogs; learning them risks hangs |
| `recenter_sign`, reflex directions | **fixed (calibration)** | geometry, calibrate once |

**Principle:** learn the **fluidity/efficiency** knobs (speeds, collapse, blend, dwells) against a
`time + (−SPARC)` reward with a success/safety floor; keep the **integrity/safety** gates
(contact, confirm, upright, timeouts) fixed so exploration can never crush, drop, topple, or hang.

---

## 8. Where this is heading

Both the skill scheduler (A) and the reactive layer (B) are bolted onto a **hand-coded discrete
FSM**. The direction (now partly realised by (D)) replaces the fixed gates with a **continuous,
optimizable representation**: a small set of via-points with per-via **precisions** (option-c
preference field, `[[preference-field-prototype]]`) selected by a **predictive forward model** (the
capability map + `predict_reach`). Then the table's gates above become *learned parameters of one
representation* rather than ~30 hand-set constants — and each phase picks its target/precision by
**predicted expected free energy of the phases that follow** (D), with the reactive layer (B) as the
safety net.

The unifying view (`[[anticipatory-efe-sequencer]]`): a phase is a **Gaussian preference with a
precision structure** (hard constraints = infinite precision, `EFE §4.10`); the sequence is selected
by **EFE over the policy** so each phase frames the next (`EFE §4.11`); and the **coarticulated
handoff** — terminal velocity preference pointing into the next via, not zero — is the per-via
precision field. In this view *anticipation and execution-time reduction are the same property*: a
phase that ends where the next begins has neither stop-and-go nor dead-end recovery. The current
target is the **approach→insert** handoff (D, in progress): the approach's orientation-commit gate
is the single largest time cost, and it is precisely a precision to learn and hand off.
