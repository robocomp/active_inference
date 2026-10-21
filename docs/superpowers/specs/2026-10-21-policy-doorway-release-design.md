# Local policy: doorway release and reachability (fixes 1, 2, 4)

Date: 2026-10-21 · Component: `controller/src/local_policy/` + `controller_policy_drive.cpp` (branch `feat/interaction-anchor`)
Status: approved by the user 2026-10-21 ("go with 1, 2 and 4"). Amends `2026-10-20-local-motion-policy-design.md`.

## 1. Why
Three round trips through DOOR_1 (runs rtD1a/rtD1b, three-room world). After the pose-age fix (3ae18a5b) the robot
crosses (leg 1 OK in 100 s) but leg 2 failed: 5 Stuck holds at normal latency, each followed on the NEXT tick by Drive
with ~70 eligible motions, yet each handed the crossing back as Unreachable and burned a mission retry. Analysis
(scratchpad `door1_stall_analysis.md`, "rtD1b follow-up"): the contact belief is TRUE — the gap detector's own
measurements put the robot 1–3 cm from a jamb, 6–12 cm off the gap centre, yawed 3–18° toward it. At 0.35 m/s every
release candidate (reverse and turn-in-place included) first coasts ~0.15 m (latency + braking at WheelDecel 0.5), so
all share a risky prefix and fail together (P 0.6–0.95). Three defects:
- **D1** `controller_policy_drive.cpp:577` (also :523, :705, specificworker.cpp:2292): ONE Stuck tick hands the goal back
  Unreachable. A Stuck is a statement about one tick from one moving state, not about reachability.
- **D2** `local_policy.cpp:347` "standing still is not a candidate": the motion that removes the shared coast is never
  scored, so Stuck-while-moving is declared where Release(stop) was available.
- **D4** the cost-to-go is indifferent between 1 cm and 15 cm of doorway clearance: `extra_fn` (residual body risk) does
  not see the measured jambs and `free` is the inscribed-radius test, so an oblique approach cuts the corner.

## 2. Decisions (approved fix set)
**F2 — stopping is a release candidate.** The brake-to-rest sweep from the current state (same latency + wheel-decel
profile as every rollout) is scored like any candidate (P(contact) over its sweep, continuability from the rest pose).
If it is the least risky releasable candidate the verdict is `Release(stop)`; the next tick decides again from rest.
`Stuck` then means "no candidate INCLUDING stop is releasable" — i.e. the robot cannot even stop safely, or it is at
rest and nothing moves it safely.

**F1 — Unreachable is an accumulated belief about a PLACE, not a tick.** Keep a log-odds `L_unreach` per goal for
"no safe motion exists from here". Each policy tick folds the evidence of that tick (Stuck at rest = strong for;
Stuck while moving = weak — the coast explains it; any tick with a releasable/drivable candidate = against), and the
accumulated evidence is DISCOUNTED as the pose moves, on the field's resolution (evidence is about the place it was
gathered at) — the same structure as `rc::exist`'s LLR fold and the Navigator's prediction monitor. The likelihoods
come from the policy's own quantities (the best candidate's P(contact), eligibility, v at the tick), not from new
constants. The goal is handed back when holding one more tick no longer pays: the expected cost of continuing (time,
weighted by P(unreachable)) exceeds the value of the goal under P(reachable) — expressed with the objective's
existing time/progress terms. No tick counters, no fixed "N stucks" rule.

**F4 — the measured doorway jambs enter the route's risk.** When a measured gap (anchor / `[gap]` detector: jamb
endpoints, σ_lat, σ_s) is active for the goal, add to the per-metre risk density the body's P(contact) at each cell's
lateral offset from the gap line (hull projected on the gap normal, σ from σ_lat and σ_s), over the doorway corridor
only. The cost-to-go then descends along the centre line; expected clearance ≈ (W − hull width)/2 per side.

## 3. Invariants and anti-goals
- No new thresholds/gates/clamps (CLAUDE.md). Every new number is derived from existing model quantities or stated
  with its measurement.
- The 11 sim2d scenarios × 20 seeds still pass (0 contacts); add scenarios: (a) off-centre fast doorway approach →
  crosses, never hands back; (b) a genuinely blocked doorway → hands back (F1 must still conclude Unreachable);
  (c) stop-is-the-only-safe-release → Release(stop), then progress.
- A genuine dead end must still be reported Unreachable within a bounded time (state the expected time from the
  model in the tests).
- Not in scope: F3 (continuability at the end of the hold) and F5 (mission/passage ordering).

## 4. Validation
Unit + sim2d as above; then live: DOOR_1 round trips (`round_trips_door1.sh`, 3 × 2 legs) — target 6/6 crossings with
no Unreachable hand-back on a passable door, and the policy decision CSV showing no single-tick hand-back.
