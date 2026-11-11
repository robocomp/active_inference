# The controller drives with the spline chain; the look-ahead policy only handles the tight spots

Date: 2026-11-10. Status: DRAFT for the owner's review. Nothing is built.
Worktree: rt-build, controller.

## 1. Why

The owner on 2026-11-10:
- "Driving a robot using the 2 lidars 3D is a long time ago solved problem."
- "I like the planned_path -> optimized_spline -> local_spline_optimization_for_local_obstacles -> tight_spline_following.
  … no general policy will work for all situations so that one can be the one for long-run traverses. Narrow passages
  and stuck situations would require a different more specialized approach."

Today `Policy.Enabled = true` hands **every** motion to the look-ahead local policy (`local_policy/`, Navigator →
`decide()`). It was built to get the body through wedges, jambs and 0.9 m doorways, and it does. It has three hard
modes, Drive, Release and Stuck, each with its own objective, and it has been through at least eight rounds of fixes and
rulings. Run as the driver of every traverse, its mode switch produces absurd motion in open space:

- **clean38** (2026-11-10): a cross to DOOR_0 with the doorway about 8 m away in open space and P(contact) 0.
  - No contact-safe candidate showed positive two-step progress (the cross ran on a predicted anchor that moved 0.2–0.3 m
    per decision).
  - So the policy declared a wedge and entered **Release**, whose objective has **no progress term**: least contact risk,
    then continuability, with ties going to the *larger reverse travel*.
  - It reversed at 0.6 m/s from 8.0 to over 9 m from the goal, then stopped.
- **clean37**: in a recalled room, the policy drove "back up arcing right" for 16 s while exploring.
- **The mock's own catalogue already fails two scenarios today**: `dead_end_release` and `goal_behind_open_space` with OU
  pose drift. Its reverse breakdown counts reversals in open space with a forward alternative ("should be ~0").
- **The apartment stress suite** (§5, added today): results in §5.3.

Meanwhile the chain the owner describes **exists in this controller, complete**, and is switched off:

| stage | code | state today |
|---|---|---|
| planned path | `grid_planner.h` (A* over the map's free space, the exact footprint test) | used by both regimes |
| optimised spline | `route_spline.h` (C² uniform cubic B-spline, feasibility pull-back to the polyline) + `route_optimizer.h` (variational over the control polygon: bending + one-sided clearance + waypoint anchor + gauge) | `route_optimize = true`, but only reached when the policy is off |
| local spline optimisation for local obstacles | the **elastic band** (`band_enabled`): the same objective re-run every cycle on a 4 m window ahead of the robot, against the LIVE LiDAR ESDF; `near_field_clearance.h` gives it a structured near field (fitted surfaces, the doorway's jambs) where the raw ESDF has no traction | on: etc/config.toml sets `BandEnabled = true` (the code default is false), and the config notes PLAIN *requires* it, because the tracker itself does no avoidance |
| tight spline following | `trajectory_controller.h` PLAIN tracker (carrot on the arc-length coordinate, curvature-continuous command), blockage detection | used when `Policy.Enabled = false` |

## 2. Goals, non-goals, anti-goals

Goals
- G1. Long traverses (room to room, across a room, toward a far doorway or standpoint) are driven by the spline chain, with
  the band on (as the config already has it).
- G2. The look-ahead policy is a **specialist**. It runs only for a stretch the chain cannot drive: a narrow passage it
  cannot follow within its tracking error, or a stretch it has demonstrably failed on. Each such episode has a **local
  subgoal on the route**, and it ends there.
- G3. The hand-off between the two is a question about the model, not a switch on a symptom (§3.2).
- G4. Both regimes and the hand-off are exercised in the 2-D mock, in the real apartment geometry, before any live run.

Non-goals
- Rewriting either regime. The chain and the policy stay as they are. What changes is who drives when.
- The door crossing protocol (door_concept's approach/open/cross contracts, the anchor, the doorway field) is unchanged.

Anti-goals: what would make this a failure even if it works
- A hand-off that dithers between regimes, or new constants to stop it dithering (§3.3 says how it cannot).
- The policy back in charge of open space by another route.
- A live run used to discover what the mock could have shown.

## 3. The design

### 3.1 Traverse regime (the default)

Goal received (a place, a standpoint, a doorway's far side):
1. `grid_planner` plans on the map's free space, which fixes which way around obstacles the route goes.
2. `route_spline` and `route_optimizer` build the curve. It is feasible by construction (the pull-back).
3. Every cycle the **band** re-optimises the window ahead against the live ESDF and the near-field clearance model, so
   obstacles the map does not hold (chairs, people, an open leaf) bend the route instead of stopping the robot.
4. The PLAIN tracker follows the curve tightly. Speed comes from curvature and clearance, as today.

**A goal with a σ is a region, not a point** (proposed by language-agent-4d, chat_e2e2, 2026-11-10).
- A mission `go_to pose:room(x, y, θ, σ)` carries σ_xy, and the controller ignores it today.
- Live, the target was a remembered chair's own centre: the cost-to-go was infinite at the robot, the policy went into
  Release and backed up about 2 m, and the step failed `no_route` (dump `policy_field_dump_1794309704136.txt`). That is
  the same open-space Release failure as clean38.
- Here the planner and the field are seeded from every free cell within the goal's distribution, a Gaussian terminal
  cost of σ, so the route ends at the nearest reachable place near the object. No standoff constant.
- A point goal is σ → 0.

**A moving goal moves the route's end, not the robot.** On a predicted anchor (clean38) the goal is re-measured every
decision. The band window that reaches the goal re-anchors its last control points to the new estimate. The route is
re-planned only when the new goal's route takes a different way around an obstacle (A*'s output changes), not on every
measurement. The tracker's carrot, metres ahead, does not see centimetre noise at the far end.

### 3.2 The hand-off: can the chain drive the next stretch?

The chain's one assumption is that the body, following the curve with its tracking error, does not touch anything. So the
question is asked of exactly that, over the band's window:

    P_window = P(contact anywhere on the window | the curve, the tracker's lateral error N(0, σ_track²), the live surfaces)

It is computed with the **policy's own contact model** (`contact.h`, `log_p_no_contact_cells`), swept along the curve with
the footprint offset by the tracking error, so both regimes judge contact the same way.

- P_window ≤ risk_eps (the policy's existing chance constraint, 0.01): the chain drives.
- P_window > risk_eps: the stretch is **narrow for the chain**. The chain hands the policy a **local subgoal**: the first
  route point past the stretch where the window is clear again (a doorway's far side, the end of a squeeze). The policy
  runs one episode toward it.
- **Stuck is a failed prediction**, as in the policy (ProgressMonitor): the chain predicts arc-length progress over its
  horizon. If the realised progress falls within the measurement's resolution while the prediction was larger, the chain
  hands the policy the same kind of local subgoal from where the robot stands.

**Tight spaces never go to the chain** (owner, 2026-11-10): "the spline approach probably does not work in tight spaces due
to the instability of the gradient when pushed from equal distance sides, among other things."
- Between two near-equidistant surfaces (a 0.9 m doorway, a gap beside furniture) the clearance gradient flips side to side
  and is flat across the opening. The band and the optimiser have no stable push there, and `near_field_clearance.h`
  mitigates this without removing it.
- So a stretch is handed to the specialist because it is tight, **before** the chain is asked to drive it, not after it has
  struggled.
- P_window measures exactly that: in a passage whose free width barely exceeds the body, the tracking error alone gives
  contact mass on both sides.
- The band does not deform inside a stretch that has been handed over.
- **The specialist for tight spaces is its own policy.** The existing look-ahead policy (jambs, leaf, wedges) is today's
  candidate, and the mock's catalogue is its test. Whether it is the right specialist is a separate question.

σ_track is **measured, not chosen**: the tracker's lateral error from the controller's own logs (it records the carrot and
the pose), as a per-robot calibration. Until measured, it is flagged at the body's safety margin.

### 3.2b The hand-off as built and tested in the mock (`REGIME=hybrid`, 2026-11-10)

Four things the first draft got wrong, each found in the mock:
1. **Pose by pose, not one sweep.** The contact model's sweep form treats heading uncertainty as a rotation about the
   sweep's *first* pose (ρ·σ_yaw). That is right for a 1 s policy rollout, but over a 4 m route window it is about 0.2 m of
   invented uncertainty, and it handed off in open space (room_2, 1 m from anything).
2. **The worst segment decides, not a sum.** The tracking error is one offset shared along the window, so the window is as
   risky as its worst segment. Summing per-pose risks made any long window exceed risk_eps.
3. **Segments, not single poses.** The model judges the body swept between consecutive frames, so a one-pose "sweep" reads 0
   whatever is there. Measured: log P = 0.0000 for a pivot 8 cm from a wall.
4. **Hand over when the stretch is reached, not when it is seen.** Handing a doorway seen 4 m ahead to the policy gave it a
   far goal, and it retreated 5.4 m (clean38's failure reproduced inside the hybrid). The chain now drives up to the stretch.
   "Reached" means within the tracker's own preview (v·T_lag + W/2, the band's lead) plus the circumscribed radius.

As built:
- **The window** = the in-place turn from the robot's heading to the route's heading (PLAIN pivots there; that sweep is part
  of the next motion), then the route ahead every 0.1 m for 4 m.
- **Each consecutive pair** is judged by `log_p_no_contact_cells` on the live LiDAR cells (`denoise_returns`), with
  σ_track 0.05 m and 0.05 rad (★flagged; the mock measures 0.024–0.06 m RMS, the robot 0.086–0.153 m).
- **The window fails** iff some pair within reach has P(contact) > risk_eps (0.01, the policy's own bound).
- **Stall**: no 0.10 m of arc-length progress within one band window's travel time (4 m / v_max). Both hand over too.
- **The episode's goal**:
  - for a place goal, the first route point past the failing stretch whose next body length is clear;
  - for a doorway goal, the doorway itself (it is now near).
- **The episode** runs the unchanged Navigator and ends at Arrived, after which the chain re-plans from the robot's pose.
  If the policy finds the goal unreachable, the goal is unreachable.

### 3.3 Why it cannot dither

The policy's episode has a goal, the local subgoal, and ends only when it gets there (Arrived), hands back unreachable
(its reachability belief), or reports Stuck. The chain then replans from the robot's pose. A new hand-off needs a new
window failing its own test. No timer and no hysteresis constant is involved: the episode's end is its goal.

### 3.4 What the policy loses

The policy is never given a goal metres away in open space. Its Release mode, a wedge escape, then only meets the
situations it was built for. The mock's open-space reversing oracle (§5) must read 0 in the traverse scenarios.

## 4. What changes in code

- `controller_session` / `controller_policy_drive.cpp`: `Policy.Enabled` stops meaning "the policy drives everything". It
  becomes "the policy is available as the specialist". The session runs the chain, evaluates P_window each cycle, and
  starts and ends policy episodes with local subgoals.
- A function `window_contact_probability(route, s0, s1, σ_track, surfaces, body)` in the chain, reusing `contact.h`.
- The chain's progress monitor, reusing the policy's ProgressMonitor arithmetic on arc length.
- No change inside `local_policy/` or inside the route optimiser.

## 5. Tests: the mock

### 5.1 The mock

`controller/tests/test_policy_scenarios.cpp` with `tests/sim2d` is a closed loop at 20 Hz:
- both LiDARs at their real mounts (σ_r 0.02 m), re-expressed on the believed pose;
- plant mismatch: latency 60 ± 20 ms, deadband ×U[0.8, 1.2], slip 5 %;
- true-hull contact checked every 5 mm.

It runs the controller's own Navigator. To test this design it must also run the chain: `grid_planner`, `route_spline`,
`route_optimizer`, the band and the PLAIN tracker are all pure Eigen/STL. The hand-off in §3.2 then has to be tested
there.

### 5.2 The apartment stress suite (added 2026-11-10)

`tests/sim2d/apartment_threerooms.txt` is the real world, exported by `sim2d/export_apartment.py` from
`threerooms-piso.wbt`: walls and door frames at the LiDAR band. Scenarios, each over several seeds:
- the far doorway (clean38), still and on a jittering predicted anchor;
- each door crossed both ways from 2.5 m with a random heading;
- random place goals in each room;
- the goal 3 m straight behind, in each room;
- room_1 → room_3 through two doors;
- starting 8 cm from a wall, facing it;
- an unmapped chair on the line;
- DOOR_2's leaf open toward the robot;
- a 1.5 s LiDAR stall;
- a 0.3 m / 8° relocalisation jump;
- OU pose drift;
- a shut door, which must be declared unreachable.

Oracles:
- no contact;
- reached on every seed;
- no 20 s window without progress;
- **open-space reverse travel ≤ 0.5 m per run**;
- **never retreats more than 1.0 m** from the best distance-to-goal reached.

The last two are test acceptance bounds, not model parameters. A traverse in open space needs neither reversing nor
retreating.

### 5.3 Results on today's controller (policy drives everything)

Run of 2026-11-10, 6 seeds each:

| scenario | result | notes |
|---|---|---|
| far doorway, still / jittering anchor (clean38) | pass / pass | **clean38 is not reproduced**: the mock lacks whatever drove it live (the real anchor's errors, the leaf hypotheses, 3-D clutter: unknown) |
| each door both ways (6) | pass | at most 1 Release per run |
| random place goals, rooms 1–3 | pass | |
| goal 3 m behind, rooms 1 and 3 | pass | |
| goal 3 m behind, room 2 | **fail** | 1 of 6 seeds handed back as unreachable in open space |
| room_1 → room_3 traverse | pass | 22 s median |
| facing a wall at 8 cm | pass | |
| **unmapped chair on the line** | **fail** | 28 Releases, 25.6 s without progress, 1.49 m reversing in open space, path 2.8× the route: the stretch the band (local spline optimisation) is for |
| DOOR_2 leaf toward the robot | pass | |
| LiDAR stall 1.5 s | pass | after the mock was made to hold on Blind, as the controller does |
| relocalisation jump | pass | |
| **OU pose drift on the traverse** | **fail** | 2 of 6 not reached; Stuck on all 6; 117 Releases; up to 1.7 m reversing in open space; a **6.8 m retreat** |
| shut door | pass, but | unreachable on every seed, after 62 Releases, 54 s without progress and a path 4.5× long: it thrashes before it gives up |

The policy is reliable at what it was built for: doorways, jambs, a leaf, walls at 8 cm. It fails where a traverse meets
something the map does not hold (the chair) and where the pose drifts. In both cases it falls into Release cycles in
open space. Those are the traverse regime's cases.

### 5.3b The spline chain on the same suite (`REGIME=chain`, 2026-11-10)

`run_chain` in the mock drives the session's single-target chain with the live config's values:
- GridPlanner (cell 0.06, clearance 1.5 / 0.9);
- the route optimiser with `make_route_optimizer_config`'s settings;
- `TrajectoryController::compute(pose, cloud)` in PLAIN;
- the band with `step_route_band`'s window and anchors;
- the map's walls injected into the ESDF, as the session injects its room model.

Stated simplifications:
- a confirmed blockage re-plans around the blocked disc;
- a re-measured goal does not re-plan (§3.1).

Two mock errors of mine were found and fixed on the way:
- no model injection, which made walls within 0.44 m invisible;
- a re-plan on every goal re-measurement, which reset the tracker every tick.

6 seeds each:

| scenario | policy | chain |
|---|---|---|
| far doorway, still / jittering | pass / pass | pass / pass (no Release, no reversing) |
| door crossings, 6 | pass, 0.04–0.07 m min clearance | **5 of 6 pass** with 0.13 m; DOOR_0 northbound: 1 seed with contacts and a 3.4 m retreat |
| place goals, rooms 1–3 | pass | pass |
| goal behind, rooms 1 and 3 / room 2 | pass / 1 of 6 handed back | pass / the same seed fails |
| room_1 → room_3 traverse | pass, 21.8 s | pass, 29.7 s |
| **facing a wall at 8 cm** | pass | **fail**: it pivots in place and the body's corners sweep into the wall (nothing in the chain checks the footprint while turning) |
| **unmapped chair** | **fail**: 28 Releases, 1.5 m reversing | **pass**, 12.1 s, no reversing: the band bends the route around it |
| leaf toward the robot | pass | pass |
| LiDAR stall | pass | pass |
| **relocalisation jump 0.3 m / 8°** | pass | **fail**: thousands of contact ticks. The injected room model moves with the believed pose, so the walls it injects are 0.3 m off the real ones, and the chain trusts them |
| **OU pose drift** | **fail**: thrashes, 6.8 m retreat, no contacts | **fail**: contacts, 5.8 m retreat |
| shut door | unreachable on every seed, after 62 Releases | 1 of 6 got through by a detour; the rest were not declared unreachable (the mock's blockage handling is a simplification) |

Reading:
- **They are complementary, as the owner said.** The chain does the traverses better: no Release, no reversing, and it handles obstacles the map does not hold. The policy handles tight starts better: a wall at 8 cm.
- **Pose error is the chain's real weakness, and it is structural.** A model injected in the believed frame becomes a wrong wall when the pose is wrong, whereas the policy judges contact from LiDAR in its own body frame. So the hand-off test (§3.2) must use **live LiDAR surfaces**, never the injected model. And a relocalisation jump, or a pose covariance large against the window's clearance, must hand the stretch to the specialist.
- **Doorways in the 2-D mock do not show the gradient instability** the owner expects, except one DOOR_0 seed. The mock has perfect walls, no leaf hypotheses and no 3-D clutter, so this is not evidence that doorways are safe for the chain. Doorways stay with the specialist (§3.2).

### 5.3c All three regimes, final mock run (2026-11-10, 6 seeds, after the corrections below)

Corrections made before this run:
- a doorway counts as crossed only *through* the opening;
- DOOR_0 is approached from the north at 1.4 m;
- the shut-door oracle is "no contact, no thrash" (its far side is reachable by a detour);
- the apartment's pose drift is the robot's realistic 0.05 m / 2° (the catalogue's 0.5 m / 13° defeats any map-based driver);
- in the hybrid, the chain's confirmed blockage is a hand-off trigger, not an invented obstacle.

| scenario | policy | chain | hybrid (hand-offs) |
|---|---|---|---|
| far doorway, still / jittering | ok 14.9 / 15.6 s | ok 18.3 s | ok 18.3 s (6) |
| 6 door crossings | ok 4.5–7.8 s | ok 6.8–10.4 s | ok 8.2–10.4 s (6 each) |
| place goals, rooms 1–3 | ok | ok | ok (0) |
| goal behind, rooms 1 and 3 | ok | ok | ok (0) |
| goal behind, room 2 | 1 of 6 handed back | same seed | same seed (shared: probably the sample, not the driver) |
| room_1 → room_3 | ok 21.8 s | ok 29.7 s | **retreat 3.2 m**, 38 s (18) |
| facing a wall at 8 cm | ok | **4692 contact ticks** | ok 13.6 s (6) |
| unmapped chair | **28 Releases, 39 s** | ok 12.1 s | ok 12.1 s (0) |
| leaf toward / LiDAR stall | ok | ok | ok |
| relocalisation jump 0.3 m / 8° | ok 18.1 s | **8810 contact ticks** | 6/6 no contact, but **44 s, retreat 3.3 m** (15) |
| drift 0.05 m / 2° | ok 18.5 s | 6/6, retreat 6.1 m | 6/6 no contact, but **46 s, retreat 3.4 m** (20) |
| shut door | **0/6, 54 s thrash** | 1/6, retreat 5.8 m | 5/6 by detour, no contact, **a 49 s window without progress** |
| **failing scenarios** | **3** | **5** (2 with contacts) | **5** (none with contacts) |

Reading, honestly:
- **The hybrid is safe everywhere** (zero contacts in all 22 scenarios, every goal reached but the shared room-2 seed). It
  fixes each regime's worst failure: the chain's wall pivot and its pose-error crashes, and the policy's chair and
  shut-door thrash.
- **In this mock it is not yet better than the policy alone**, which fails 3. The hybrid's failures are inefficiency:
  hand-off churn in the tight middle of the apartment (the route from DOOR_0's pocket squeezes past room_1's partition end),
  and under pose error, where every re-plan threads offset doorways.
- **The mock does not reproduce the live failures that motivated this design** (clean38's Release reversal; language-agent's
  infinite field). Live, the policy alone has failed where the mock says it passes. So the mock's ranking cannot be the only
  evidence, and the two live failures must be added to the mock first.

Next, in order:
1. **Reproduce clean38 and the infinite-field go_to in the mock.** Until then the mock flatters the policy.
2. **Hand back early.** The episode ends when the window from the robot's pose is clear again, not at a far subgoal. This
   is the source of most of the hand-off churn.
3. **A planner route that avoids tight gaps when a wide one exists.** The clearance preference is weak against a 0.6 m
   squeeze.
4. **Pose error.** After a relocalisation jump the chain should re-plan on the corrected pose, and while pose uncertainty is
   large the specialist drives.

### 5.3d After the mock's wall bug and four design pieces (2026-11-10, 6 seeds, 24 scenarios)

**The mock had a bug that drove most of §5.3c's hybrid failures.** Its walls entered the chain's planner as 2 cm polygons on a
6 cm grid; `grid_planner` marks a cell only when its centre is inside a polygon (by design), so every interior wall was
invisible to the chain. It planned through room_1's north wall, the window test then handed the stretch to the policy,
and the policy (whose map had the wall) drove a 9 m detour to a sub-goal 1.3 m away. Live the planner's free space is the
room polygon itself, so this never happened on the robot. Walls now enter at half-width cell·√2/2. Items 2 and 3 of
§5.3c's "Next" were symptoms of this bug, not of the design.

**Two live failures added** (each from its field dump):
- `apt_goal_in_object` — "go to" a remembered chair's centre (dump 1794309704136, goal cell not free, field +inf). The
  mock now reproduces it: today's policy reverses **19.7 m** in open space and never arrives.
- `apt_phantom_strip` — the clean38 class (dump 1794309020589: the policy's own free map cuts a robot 0.77 m clear off the
  doorway; field +inf; Release). Modelled as residual hard cells across DOOR_0's approach that the world lacks. Not a
  faithful reproduction: the apartment gives the policy a detour, so its field stays finite and it passes.

**Four design pieces, built in the mock harness only:**
1. **Goal region (§3.1).** A target the body cannot stand on becomes the reachable place nearest it, judged by the chain's
   own feasibility (`gp.pose_free` at some heading) and map reachability. That is the Gaussian terminal cost's optimum
   for any σ when the terminal cost has no travel term. No standoff constant. `apt_goal_in_object`: 3.75 s, no reversing.
2. **The specialist measures progress along the chain's route.** Its field's free space is the union of the planner's
   clearance discs along the route, plus a disc at the robot reaching its projection. It is connected wherever the route
   runs, so the policy's own free map can no longer cut the robot off its goal (the mechanism of both live failures).
   Contact stays the policy's LiDAR model.
3. **The specialist tries first; the planner re-plans on map + live only when the specialist cannot pass.** The live
   marks are LiDAR cells farther than one planner cell from the map and earlier marks; they persist (live: they clear
   when seen free). Marking at the chain's first stall drew ghost walls across doors under pose error (measured: pose
   jump 3/6, 5.9 m retreats), hence this order. The specialist's stall clock reads its field as built when the episode
   started.
4. **The specialist's Arrived ends the run when its goal is the run's own goal** (harness bug: it looped 3602 hand-offs).

Also fixed: the goal-behind sampler put room_2's seed-0 goal beyond a wall (geodesic inf); it now requires the same room
and a clear straight line. The shut-door oracle now measures progress on the TRUE world (door shut) in every regime.
On the map's field, which believes the door open, every correct detour reads as retreat.

| scenario | policy (today) | chain | hybrid (hand-offs) |
|---|---|---|---|
| far doorway, still / jittering | ok 14.9 / 15.6 s | ok 18.3 s | ok 18.2 s (6) |
| 6 door crossings | ok 4.5–7.8 s | ok 6.8–10.6 s | ok 6.8–10.6 s (6 each) |
| place goals, goal behind (rooms 1–3) | ok | ok | ok (0) |
| room_1 → room_3 | ok 21.8 s | ok 24.1 s | ok 24.4 s (6), no retreat |
| facing a wall at 8 cm | ok 10.2 s | **4692 contact ticks** | ok 11.3 s (6) |
| unmapped chair | **1.5 m reversing, 39 s** | ok 12.2 s | ok 12.2 s (0) |
| leaf toward / LiDAR stall | ok | ok | ok |
| relocalisation jump 0.3 m / 8° | ok 18.1 s | **0/6, retreat 5.9 m** | ok 24.1 s (17), retreat 0.13 m |
| drift 0.05 m / 2° | ok 18.5 s | **1968 contact ticks** | ok 22.3 s (7), retreat 0.15 m |
| shut door (truth yardstick) | **0/6, 62 Releases** | ok 28.9 s | 6/6 by detour, 46.6 s; one seed 27.9 s without progress |
| goal in object (live: chat go_to) | **0/6, 19.7 m reversing** | no route | ok 3.75 s |
| phantom strip (clean38 class) | ok 17.5 s (detour) | ok 8.75 s | ok 8.75 s |
| **failing scenarios** | **3** | **4** (2 with contacts) | **1** (shut door, slow; no contacts anywhere) |

Reading:
- **The hybrid passes 23 of 24 with zero contacts in every scenario**, and fixes each regime's worst failure: the chain's
  wall pivot and pose-error crashes, the policy's chair reversing, shut-door thrash and target-in-object reversal.
- **It is slower than the policy alone in tight crossings** (door crossings 6.8–10.6 s against 4.5–7.8 s): the chain
  drives up to each door at its own pace and the hand-off costs a stop. That is the price of the chain in open space.
- **Open: the shut door.** The specialist creeps 0.1 m nearer the leaf, which restarts its 8 s stall clock, before the
  re-plan; 1.6× the chain alone. Faster would need the specialist to say "cannot pass" from its contact model directly
  (the stretch infeasible at every lateral offset), not from elapsed time.
- **Still not reproduced:** clean38 itself (the apartment offers the policy a detour). The class is covered by piece 2,
  which removes the mechanism (the policy's own free map) from the hybrid.

### 5.4 Acceptance for this design

All apartment scenarios pass with the two regimes. The existing wedge, jamb and doorway catalogue passes with the policy
as the specialist (its own oracles, unchanged).

## 6. For the owner to rule on

1. The hand-off criterion: P_window from the shared contact model against the existing risk_eps, plus the chain's failed
   progress prediction.
2. σ_track measured from logs; until then flagged at the body's safety margin.
3. (Withdrawn: the band is already on in etc/config.toml. My first draft misread the code default.)
4. Order: extend the mock to run the chain and the hand-off first; no live runs until the apartment suite passes.
