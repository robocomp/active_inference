# One local motion policy: drive safely, look ahead, never wedge

Date: 2026-10-20 · Component: `controller/` (branch `feat/interaction-anchor`) · Status: design, awaiting approval

## 1. Intent and why

The robot gets stuck. This morning (three-room tour, leg 3) it sat for 9 minutes 0.7 m from door_1's approach
standpoint, pressed side-on against a partition wall, its heading swinging ±70° while no motion was ever executed.
That was not a new bug. A catalogue of every stuck episode from try64 to today (≈60 runs) shows:

- Since the doorway was simplified to steer on the LiDAR-measured gap (try91), almost every remaining failure is in
  the **local motion layer**: in-place turns refused in confined space (≥8 legs, 4 fixes, still recurring), false
  "stuck" verdicts handing reachable targets back (Class E), and planner/standpoint approvals another layer then
  forbids (Class F).
- **The layers contradict each other.** The standpoint check says "it can TURN here" and the turn is refused at that
  same pose 124–188 times (try100/101). The planner refuses a corridor the door servo then drives. The stall judge
  and the returns net disagree about whether the robot is blocked.
- **Seven mechanisms can command the base** (relative motion, returns backoff, anchor/door servo, lock-on, orient,
  PLAIN tracker, escape) and ≈40 sites stop it. **Six vetoes** sit on top (ESDF clearance ceiling, speed floor,
  route speed limit, pose-uncertainty limiter, live-returns net, turn-room), each with its **own body model**
  (circumscribed disc, hull+0.05, hull∪mesh+0.05, r_circ+0.6, 0.9 m from centre). ≈70 config keys, ≈20 timers and
  ≈40 hard thresholds govern motion safety and recovery.
- **Two structural defects keep a wedge alive.** (1) Every all-zero command calls `path_controller.stop()`, which
  resets the tracker, which restarts with a pivot-to-tangent — the one motion a robot beside a wall cannot make.
  (2) Escape is diverted whenever the returns net bound in the stall window, so in exactly the confined cases escape
  never fires; backoff then picks the opposite turn, ends, re-plans, and the tracker turns back.
- **Wrong body.** The footprint is `shadow.stl` (correct, md5 06153a46…, the mesh Webots draws) ∪ FOUR mecanum
  wheels at (±0.21, −0.10/+0.16). The simulated Shadow has TWO drive wheels (cylinders r 0.10, width 0.05, centred
  x = ±0.259, y = 0) plus front/back ball casters (r 0.04, inside the body). The real outline reaches |x| = 0.284 at
  mid-length; the model stops at 0.272 and puts the protrusion at the corners.
- The ESDF that the speed ceiling reads deletes every return within ≈0.405 m of the centre (old disc self-filter), so
  near a wall that layer is blind and assumes the worst; nothing below z = 0.20 m reaches any safety layer, yet the
  drive wheels occupy z 0 … 0.167.

Each fix so far added a rule to one layer, which moved the disagreement to another. The user's ruling (2026-10-20):
**do not stack patches; replace the logic.**

## 2. What the user decided (verbatim intent)

1. "The robot should **evaluate its movements ahead** to avoid stuck situations."
2. "If it ends up in one, the only option is to **move back and turn a bit to release and then plan again**."
3. When backing up, it backs up **in whichever direction has the most room**.
4. The new policy **also drives the final stretch through a doorway** (the door servo goes away).
5. The body is **`shadow.stl` with the differential wheels, because that is what Webots draws** — the proto is the truth.
   This work is **simulator only** (user, 2026-10-20): the body is built for the simulated Shadow; the real robot's
   outline is out of scope (the 0.2464 m |x|max noted in `controller_robot_body.h` / ROBOT_GEOMETRY.md describes the
   old assembly, not the simulated robot).
6. Do not stack patches on patches; think about the cause and design one algorithm that drives safely without
   getting stuck.

## 3. Goals, non-goals, anti-goals

**Goals**
- ONE authority produces every base command (adv, rot) while the robot is driving to a place, through a doorway,
  turning to face something, or recovering.
- ONE body model, everywhere: planner, contact check, self-filter.
- Safety comes from predicting contact along each candidate motion with the true body, not from caps applied after
  the fact.
- The robot never commits to a motion from whose end pose it cannot continue (look-ahead).
- When it is trapped anyway: back up toward the most room, turn a little, re-plan — and the re-plan must not lead
  straight back into the same trap.
- Far fewer parameters, each with a physical meaning; no hidden thresholds.

**Non-goals (this step)**
- The global planner's algorithm (A* over x, y, heading on the grid) stays; it gets the new body.
- Door estimation (door_concept), the mission language, room estimation, the affordance protocol between agents.
- Omni/holonomic bases (P3Bot). The design must not preclude adding `side` later, but Shadow is differential.

**Anti-goals (would make this a failure even if it "works")**
- Keeping the old backoff / escape / make-room / stall-judge / returns-cap code alive "as a fallback". They are
  deleted once the new policy is validated; until then they are switched off, not layered.
- A new threshold to fix each new scenario.
- A policy that is safe only because it is slow or that refuses everything near walls (the corridor in front of
  door_1 is ~1 m wide; the robot must drive it).
- Validating only on live 20-minute runs (one defect per run, fixes never accumulate).

## 4. The algorithm

### 4.1 Body
- Footprint = XY outline of `webots-shadow/protos/meshes/shadow.stl` (z band of the robot that can touch the world)
  ∪ the two drive-wheel rectangles (x ∈ [0.234, 0.284] each side, y ∈ [−0.10, 0.10]) ∪ the casters (inside the body,
  included for completeness), as a **non-convex polygon**, read from / checked against `Shadow.proto` at start-up
  (the start-up report already diffs the footprint; it must now diff against the proto's wheels).
- Plus one safety margin (`FootprintSafetyMarginM`, physical meaning: unmodelled geometry). Same polygon in the
  planner, the policy and the LiDAR self-filter. The ESDF disc self-filter and the four-wheel model are removed.

### 4.2 Inputs each control cycle
- Live LiDAR returns (both planes, room-frame buffer re-expressed on the current pose, existing self-filter with the
  new body), each with range σ_r (`LidarRangeSigmaM`) and existence P_j.
- Robot pose and its **full SE(2) covariance** (today only two scalars reach the controller — extend
  `read_pose_uncertainty`), plus pose age.
- Current base velocity (odometry) and the base capability (accel/decel/max speeds, differential).
- A goal ("attractor") from the task layer, one of: a place (x, y) with optional final heading; a doorway (the
  measured gap line: centre, normal, width); a heading only (orient / lock-on).
- A **cost-to-go field** to that goal (4.4).

### 4.3 Candidate motions and scoring (the look-ahead)
Each cycle (on a fresh scan, ~20 Hz):
1. **Sample candidate commands** (v, ω) over the dynamically reachable window around the current velocity, forward
   AND reverse, including pure rotation and arcs of both signs. Each candidate is held for a horizon T and is
   followed by a full brake to rest at the base's real deceleration — so every candidate ends stopped, and "can I
   still stop safely" is part of it.
2. **Roll out** each candidate with differential-drive kinematics (latency compensated by the pose age) and **sweep
   the body polygon** along it. For each return compute the minimum signed distance to the swept body; with σ_r and
   the pose covariance (which grows with latency · speed) this gives P(contact) for the whole rollout.
   (The existing `lidar_return_speed_limit` / `anchor_obstacles_rot_limit` contact math is the seed: it becomes a
   likelihood inside the score, generalised from straight lines and in-place turns to arcs.)
3. **Continuability (the user's "evaluate movements ahead").** From each candidate's end pose, check coarsely
   whether the robot still has a way to make progress — i.e. a second, coarser set of candidates from that pose
   includes at least some with low contact probability and non-negative progress. A pose from which every
   continuation is blocked (side-on against a wall with the turn swept into it, nose into a corner) scores as a
   trap even if reaching it is contact-free. This is what would have stopped this morning's approach that drove the
   robot side-on into the partition.
4. **Score** = log P(no contact over rollout + brake) + log P(continuable from end pose) + progress, where progress is
   the decrease of the cost-to-go field from the start to the end pose (plus heading agreement when the goal has a
   heading). Speed is not capped separately: a faster arc has a longer brake and therefore a higher contact
   probability near obstacles, so slowing near walls falls out of the score.
5. **Act**: send the best candidate's (v, ω). Only one explicit safety parameter: candidates whose P(contact) exceeds
   a risk level ε are not eligible (a chance constraint; ε is the one place a threshold is kept, and it has a
   meaning: acceptable collision probability per decision).

Deadband: the base may not execute very small commands. The candidate set must not contain speeds below what the
base actually moves at (measure it; add it to `BaseCapability`), and odometry that does not move under a command it
should execute is evidence (4.6).

### 4.4 Cost-to-go and re-planning
- A geodesic distance field from the goal over the planner grid (free space for the body, the residual cells priced
  by their belief as today) gives progress everywhere around the robot, not only along one route. The A* route over
  (x, y, heading) is kept to decide reachability and to report Unreachable, and for display.
- For a doorway, the field is "distance to being through": along the gap's normal past the far side, with lateral
  agreement to the gap centre; the heading emerges.
- For heading-only goals, progress is heading agreement; the same contact and continuability terms apply (a turn
  beside a wall is evaluated, not assumed).

### 4.5 Stuck → release → re-plan (the only recovery)
- **Stuck** is a property of the score, not a timer: no eligible candidate makes positive progress, for as long as
  it takes the evidence to say so (the robot does not move and the field does not decrease). Not a separate judge.
- **Release**: choose the motion that backs the robot toward the most room — the reverse arcs (and in-place turns)
  scored by how much they increase the number of continuable candidates from their end pose (i.e. how much freedom
  they buy), still subject to the same contact probability. "Most room" is measured with the same machinery, not a
  separate probe. Execute it, then turn a little toward the goal if that turn is now continuable.
- **Re-plan** with the trap remembered: the region of the trap pose is added to the cost-to-go as a decaying cost
  (evidence that the body cannot get through from there), so the new plan does not lead back into it. If no plan
  exists without it, report Unreachable through the existing affordance protocol.

### 4.6 Unseen contact
If the base is commanded a motion it should execute (above deadband, eligible, P(contact) low) and odometry shows it
did not move, that is evidence of an obstacle the LiDAR does not see (below the 0.20 m band, glass, a wheel caught):
insert a contact hypothesis at the body edge in the direction of motion into the same returns set, with an existence
that grows with the evidence. The policy then avoids it like any other return; the release step handles the rest.

### 4.7 What is removed when the policy is validated
`clearance_speed_limit` + `clearance_ceiling.h`, the speed floor, `lidar_guard_command` / `cap_by_returns` /
turn-room (`anchor_make_turn_room`), returns backoff, escape (`begin_escape`/`step_escape`), the stall judge's
escape path, the anchor cross servo and jamb guards (the doorway becomes an attractor), `step_orient` / lock-on
motion (they become heading goals), relative-motion's own control law (becomes a place goal), the
pose-uncertainty speed limiter (folded into P(contact)), the elastic band and the PLAIN tracker's command path.
The four-wheel footprint and the ESDF disc self-filter.

## 5. Validation (before any live run)
A 2-D closed-loop harness that runs the **real policy code**: differential kinematics with the base's measured
limits and latency, ray-cast LiDAR with noise (reuse `test_jamb_cross.cpp`'s raycaster and scene), the true
non-convex body for contact, the measured pose-error process. Scenarios from the catalogue, each a test:
- tour3_1 leg 3: side-on to a wall in the ~1 m corridor in front of door_1, goal 0.7 m away behind the robot.
- try101 / try104: the corridor stub before door_1, left turns, 0.37 m from the east wall.
- try105 leg 5: the east jamb, 0.14 m off-centre, 0.37 m before the plane.
- Doorway crossing both ways with the leaf open at 90° on the near side and on the far side.
- A dead end it must release from (corner / cul-de-sac), a doorway too narrow for the body (must report
  Unreachable, not oscillate), furniture legs.
Oracles: zero contact; goal reached; never more than a bounded time without progress; release followed by a
different plan. Live: the three-room tour room1→room2→room1→room3→room1→room2→room1 with exploration between legs,
6/6, no stuck episode, then the round-trip script.

## 6. Constraints
- Same thread model: the policy runs in `compute()` on fresh scans; the output thread keeps sending the last command.
- Follow CLAUDE.md: no `DirectConnection`, `from_chars` for parsing, -j8 builds, no `kill -9`.
- One session edits the control stack.

## 7. Decisions made by the designer (Claude), not by the user — check these
- **The shared footprint is fixed at its source** (`common/robot_footprint`, `RobotFootprint::shadow()`): the
  four-wheel body is wrong for every agent, so every agent that self-filters with it gets the corrected body.
  Validation therefore includes the agents that use it (room_concept, retina, residual) — a short check that their
  self-filtered clouds do not change for the worse — not only the controller.
- **Unobserved space is not free by default.** The contact likelihood of a place the robot has not seen comes from
  the room-frame memory of returns and the residual/grid belief (what was seen free stays free, decaying); where
  there is no evidence at all it carries a prior occupancy. Reverse releases are scored with it like any motion.
- **Low obstacles come into scope.** Contact checks take returns from just above the floor (floor-segmented), not
  from 0.20 m, because the drive wheels span z 0 … 0.167 m; the odometry-based unseen-contact hypothesis (4.6)
  covers what is still invisible (glass, below the floor segmentation).
- **The door-crossing protocol stays; only its motion law is replaced.** The completion semantics (a crossing is
  "through" only if the robot was observed past the standoff; a claim that starts past the plane cannot complete),
  the phases, and everything door_concept / ltsm / mission retries read through the affordance protocol are kept as
  a protocol adapter beside the doorway attractor. The approach standpoint stays a separate stage before the
  doorway goal, because the door is opened from there. Goal tolerances keep their current values.
- **Every motion goes through the policy.** Exploration targets, mission/chat relative moves ("forward 1 m") and
  heading tasks become goals for it. Stop/hold stay as overrides (a zero command is always safe); simulator
  teleports are not base commands.

## 8. Left to the builder
- Sampling resolution, horizon T, how coarse the continuability check is — chosen to fit the cycle budget (~50 ms),
  measured, and stated.
- Exact form of the progress term and its single weight against the log-probabilities.
- Data structures for the swept-polygon distance (e.g. sampling the arc densely vs analytic sweep).
- Migration order inside the branch, as long as the old mechanisms are switched off (not layered) when the policy
  drives, and deleted after validation.
