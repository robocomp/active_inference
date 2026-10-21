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
