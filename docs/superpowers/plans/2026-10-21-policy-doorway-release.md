# Local Policy Doorway Release (F1, F2, F4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A momentary doorway squeeze no longer ends a crossing; the route runs down the doorway's centre line.

**Architecture:** F2 adds stop as a scored release candidate inside `LocalPolicy`; F1 replaces the single-tick
hand-back with an accumulated, place-discounted reachability belief in the drive session; F4 adds the measured gap's
jambs to the cost-to-go's per-metre risk density.

**Tech Stack:** C++23, Eigen; standalone g++ tests in `controller/tests/` (header carries the build line); sim2d
scenarios (`controller/tests/sim2d/`, `test_policy_scenarios.cpp`).

**Spec:** `docs/superpowers/specs/2026-10-21-policy-doorway-release-design.md` (approved 2026-10-21).

## Global Constraints
- CLAUDE.md: no thresholds/gates — derive from model quantities; `and`/`or`/`not`; ★ WHY comments; `-j8` builds.
- `test_policy_scenarios` 11 scenarios × 20 seeds stay green with 0 contacts after every task.
- Commit ONLY your task's files by explicit pathspec (`git commit -- <paths>`): another task stream (door_concept)
  commits in this worktree concurrently; on `index.lock` wait and retry.
- No live runs inside tasks (the controller session runs the live check after Task 3).

## Review Focus
1. A genuinely blocked door must still end Unreachable (F1 must not turn every dead end into an endless hold).
2. Stop from rest has zero sweep: F2 must not make "stand still" the eternal best release (progress must resume).
3. F4 risk only along an ACTIVE measured gap for the current goal — never a stale/foreign gap; no effect away from doors.
4. Pose-discounting of F1 evidence under a frozen or jumping pose sample (rtD1b showed 1–5 frozen ticks).
5. The three call sites that hand back on `tick.stuck` (controller_policy_drive.cpp :523, :577, :705, specificworker.cpp:2292) all use the F1 belief.

---

### Task 1: F2 — stop is a release candidate
**Files:** `controller/src/local_policy/local_policy.{h,cpp}` (candidate set + release scoring, ~:347), tests
`controller/tests/test_local_policy.cpp`, new sim2d scenario (c) in `test_policy_scenarios.cpp`.
- [ ] Failing tests: a moving state where every motion candidate's swept P ≥ release_risk_max but the brake-to-rest
  sweep is below it → `Release` with v=w=0 (stop), not `Stuck`; at rest next tick → normal decision; sim2d (c).
- [ ] Implement (stop sweep = same latency + decel profile as every rollout; continuability from the rest pose). All
  controller tests + 11 scenarios green. Commit.

### Task 2: F1 — Unreachable as an accumulated, place-discounted belief
**Files:** new `controller/src/local_policy/reachability_belief.h` (+ test `controller/tests/test_reachability_belief.cpp`),
`controller/src/controller_policy_drive.cpp` (the hand-back sites), `controller/src/specificworker.cpp:~2292`; sim2d
scenarios (a) off-centre fast doorway → no hand-back, (b) blocked doorway → hand-back within the model-stated time.
- [ ] Failing tests: belief update (Stuck-at-rest raises, Stuck-moving barely, drivable tick lowers, motion discounts by
  the field resolution, frozen pose does not double-count); decision rule from the objective's terms; (a), (b).
- [ ] Implement; route every hand-back site through it; log the belief in the policy decision CSV. Tests green. Commit.

### Task 3: F4 — measured jambs in the route risk
**Files:** `controller/src/local_policy/cost_to_go.{h,cpp}` (risk density input), `controller/src/controller_policy_drive.cpp`
(feed the active gap: jamb endpoints, σ_lat, σ_s), tests `controller/tests/test_policy_cost_to_go.cpp`.
- [ ] Failing tests: with a 0.90 m gap and the Shadow hull, the cost-to-go's descent line through the doorway is the
  centre line (± one cell) from an approach 0.15 m off-centre at 20°; no risk added when no gap is active; field
  values away from the doorway unchanged.
- [ ] Implement; tests + 11 scenarios green; then the controller session runs `round_trips_door1.sh` (3 × 2 legs).
  Commit.

### Task 4: Mission `explore room_X` selects only affordances in room_X (user decision 2026-10-21)
**Files:** the selection path (`controller_world_model.cpp` `read_target_in_room` / the affordance selector it calls;
`controller_session.cpp` ~:984-992 where `command_explore_` opens the contest; `specificworker_mission.cpp` explore
step), tests in `controller/tests/`.
- [ ] Failing test: with an `explore room_1` order active, offered affordances targeting room_1 (room exploration, a door
  in room_1, a look at a remembered door of room_1, an object in room_1) are eligible; one whose target is in room_2 is
  never selected even with the highest gain; with no mission the behaviour is unchanged; cross steps unchanged.
- [ ] Implement: the explore order carries its room; the contest's candidate set is filtered by target room (resolve
  the target's room from its RT parent chain / the affordance's object; an unresolvable target is NOT in room_X).
  Log the excluded candidates once per change in the `[aff-select]` line. Tests green. Commit by pathspec.

### Task 5: an `explored(room)` belief, so `explore` ends when the room is known (user request 2026-10-21)
Today `until explored(room)` can never be satisfied (no graph attribute — specificworker_mission.cpp:18, :239), so every
`explore` runs its full budget, and an LTSM-recalled room is re-explored exactly like a new one (tourP2 leg 3).
**Files:** room_concept (publisher: the room node attribute — new cortex attribute `room_explored_p` via
`REGISTER_TYPE` per CLAUDE.md, the USER reinstalls cortex; until then the task stops at the header change + a stub
reader test), `controller/src/specificworker_mission.cpp` (the `explored` predicate and `until`), tests.
- [ ] Design note first (in the report, reviewed before code): `explored_p` is a BELIEF, not a threshold — derived from
  what room_concept already computes: the boundary actually SEEN (seen_frac / per-wall observation evidence) and the
  expected information gain left in `afford_room` vs its time cost. A recalled room starts from what LTSM restored (its
  layout + seen boundary), so a well-remembered room is already explored; a partly remembered one explores only the
  rest. The mission's `until explored(room)` resolves when continuing to explore no longer pays (EIG of afford_room below
  the objective's time cost) — no fixed fraction.
- [ ] Failing tests: predicate unknown → never satisfies (today); attribute present → satisfies by the rule; recalled-room
  fixture starts explored when its layout is complete. Implement; tests green; commit by pathspec. Live check rides on
  the next round-trip run (legs whose room is recalled should end explore early).

### Task 6: structured near-field clearance for the spline route optimiser (user decision 2026-10-21)
**Why (user):** the smooth, analytical, differentiable spline connecting robot and target is the right local↔global
bridge; it fails in NARROW passages because the noisy LiDAR ESDF / raw field gives the clearance term of the spline's
NEAR part no usable traction (in a 0.90 m doorway the margin is 1–3 cm per side, the size of the return noise; the
min-distance gradient flips between jambs, is flat across the opening and steep at the edges).
**Hook:** `RouteOptimizerConfig::distance` / `distance_gradient` (`controller/src/route_optimizer.h:66-70`) — the
optimiser is UNCHANGED; it is handed a better field.
**Files:** new `controller/src/near_field_clearance.{h,cpp}` (pure Eigen/STL), the place that builds the optimiser's
callables (find it: the ESDF/lidar distance wiring for the spline tracker), tests
`controller/tests/test_near_field_clearance.cpp`.
- [ ] Design (in the report, reviewed before wiring): near the robot (radius from the body + route look-ahead, stated),
  the field is the EXPECTED clearance to the FITTED surfaces — reuse the policy's local surface model
  (`local_policy/contact.h` surface samples, ends, isolated hits, each with its σ), and, when a measured gap is active
  for the goal, its two jamb segments with σ_lat/σ_s — i.e. the distance to the segments convolved with their
  Gaussian uncertainty (closed form for a segment), which is smooth and differentiable everywhere; far away, the
  existing ESDF. The two are joined by a smooth weight from the model (e.g. the near model's coverage/σ), not a
  switch radius. One obstacle model shared with the policy — no second fit.
- [ ] Failing tests: (a) between two parallel jamb segments 0.90 m apart with 1.5 cm return noise, the field's
  gradient points to the centre line everywhere in the opening and its maximum is on the centre line (± 1 cm), where
  the raw-ESDF field (same noisy returns) shows sign flips — assert both; (b) far from the robot the field equals the
  ESDF; (c) continuity of value and gradient across the near/far blend; (d) the optimiser, given the new field, puts
  the spline's near part on the doorway centre line from a 0.15 m-off-centre, 20° approach.
- [ ] Implement behind a config key `[Route] NearFieldClearance` (default false, so the old tracker is unchanged for the
  A/B). Tests + existing route optimiser/tracker tests green. Commit by pathspec.

### Task 7: three-arm A/B on the DOOR_1 round trips (controller session, live)
Same code, same start state (identity table, ltsm wiped, `round_trips_door1.sh`, 3 × 2 legs), N ≥ 2 runs per arm:
(A) old spline tracker (`[Policy] Enabled = false`, `NearFieldClearance = false`); (B) spline tracker with the structured
near field (`Enabled = false`, `NearFieldClearance = true`); (C) local policy with F1+F2+F4 (`Enabled = true`).
Report per arm: crossings completed / 6, time per crossing, contacts (bridge/bumper), stuck/hand-back events, minimum
measured jamb clearance in the doorway. The old chain is NOT removed whatever the result (user, 2026-10-21); the
result decides the default and whether a hybrid (route tracker + policy in narrow passages) is worth designing.

### Task 3b: the door leaf is a KNOWN obstacle, not unexplained residual (ruling 2026-10-21, rtD1c leg 5)
The open leaf is priced as hundreds of unexplained residual cells (density up to 1240 m/m) and can disconnect the
cross goal's free mask (rtD1c: field infinite ~105 s). door_concept already models it (MAP hinge, free edge, φ, σ).
**Files:** controller residual/field wiring (`controller_policy_drive.cpp` extra_fn/free_fn sources, the residual
reader), cost-to-go input, tests.
- [ ] Failing tests: an open leaf modelled by the door (hinge, free edge, φ, σ) + residual returns on it → the field
  routes around the free edge and stays finite from the near side; the leaf's cells no longer count as unexplained
  risk; a leaf with large σ_φ is priced by its swept uncertainty.
- [ ] Also (from F4): the cross goal line and the trap veto must come from the anchor opening with the gap weighted by its score (today a phantom gap still sets the goal line).
- [ ] Implement: returns explained by the door's leaf hypothesis are removed from the residual's unexplained mass and
  the leaf enters as a known obstacle (segment + σ) in free/extra; tests + scenarios green; commit by pathspec.
