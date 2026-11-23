# Leg-1 door failures, 2026-11-23 (5 of 10 warm three-room tours) — root-cause diagnosis from logs + code

Paths are relative to `B=/home/pbustos/robocomp/components/active_inference`; runs are `B/room_concept/estimation_scripts/crossing/<run>/`.
No run, process control, edit or build was done; nothing here was reproduced. "unknown" = not established by the evidence.

## 1. What P(reach)=0 is, and what actually fails

**P(reach)=0.00 is NOT the cause.** `door_reach_prob` is door_concept's pose-only belief that the robot stands in the actuation band and faces the
opening (`door_concept/src/door_pragmatics.h:9,28,95-106`; printed by `controller/src/controller_session.cpp:758`). It is 0.00 in EVERY run at leg start,
successes included, because the robot is 3-9 m from the door ("body 8.77 m from the door, vantage 1.66 m", e.g. `s1_rs035_1/logs/controller.log:1788`;
`s1_rs035_3`: "P(reach)=0.00 ... body 7.86 m"). P(passable)=0.00 / clear span 0.00 at phi~0 just means the leaf is shut (it was shut by
`door.py close`, see `*.console` "DOOR_0 close accepted"). The `s1f_rs100_1` line quoted in the brief is simply the controller waiting at the vantage
step with the robot 5.58 m away — because it never moved (see 1c).

The four verdicts have different proximate causes:

a. **cross_unreachable (s1_rs035_2, s1f_rs035_1, and s1_rs035_1/s1_rs100_1 via step 2)** — the door ROUTE (pre -> ap -> post, `controller_door_passage.h`)
   is driven by the local policy; the policy hands back a waypoint as "cannot be reached" when `ReachabilityBelief` log-odds L exceed the bar
   (`controller/src/local_policy/reachability_belief.h:1-80`, `navigator.cpp:225-260`, message built at `controller_policy_drive.cpp:573-582`).
   The passage tolerates ONE such failure (rebuild) and the SECOND ends it `cross_unreachable`
   (`controller_door_passage.h:343-386`, `kMaxRouteFailures = 2`). Examples (log -> L jumps from the prior -4.6 to +4.5 in a few ticks, in free space):
   - `s1_rs035_2/logs/controller.log`: pt 9/14 at room (-0.77,0.34), "log-odds 4.51 > bar 0.06"; then pt 1/6 at (-0.78,0.50), 2.28 > -0.29 -> FAILED.
   - `s1f_rs035_1`: (-0.81,1.16) 5.58 > -0.30, then (-0.78,1.30) 5.11 > -0.44 -> FAILED.
   - `s1_rs100_1`: (-1.14,1.95) 2.42 > -0.24 ("release (prediction failed: predicted 0,023 m in 1,00 s, realized 0,008 m)"), then "the local policy has no route to it for 5.0 s".
   The belief's evidence is (i) "the plan realised less shortening than it predicted" (`reachability_belief.h` EVIDENCE/claim) and (ii) a count of
   "no eligible option escapes". Both grow when the base delivers less motion than the policy predicted, or when the policy is held (see section 3).
   In the successful runs the SAME event occurs once (s1_rs035_3: (-1.26,0.12) 2.98>0.06; s1_rs100_3; s1_rs100_2) and the single rebuild succeeds.
   Failure = two such events inside one passage. Count of "cannot be reached" inside leg 1: yesterday's 9 passing runs 0-1 each; today's 4 failures 2,2,4,2;
   today's 5 passes 0-1 (s1_rs035_3 1, _4 0, _5 0, s1_rs100_2 1, _3 1).
b. **approach_not_offered (s1_rs035_1)** — a follow-on, not an independent cause. After the passage failed (a), door_concept shows door_1 read OPEN
   (phi~1.7-1.8, P(passable)=0.95 at `s1_rs035_1/logs/controller.log:1794`) although the leaf was shut, so it withdraws `aff_door_1_approach`
   ("aff_door_1_cross not offered within 5.0 s ... handing back: back to aff_door_1_approach", then "approach not offered within 5.0 s again", lines 2243-2258).
   Why door_concept read phi~1.8 on a shut door from 7-9 m away: **unknown** (door_1 `ai2_log_phi.csv` post_mean jumps 0.09 -> 1.82 about 70 s before leg 1 while the
   robot is >7 m away; the same jump also occurs in successes, e.g. s1_rs100_2 at -69 s, so it does not separate runs).
c. **door_wont_open (s1_rs100_1)** — the robot ended at world (-4.60,7.25), i.e. in the wall line beside the doorway (successes cross at x ~ -4.8..-5.0, y 7.2
   from `gt_pose.tsv`, e.g. s1_rs035_3 (-4.94,7.48)). door_1 phi flips 2.0 <-> 1.45 with meas=0 for ~40 s. Root cause upstream (the same route failure in (a) at -1.14,1.95;
   then the step `open` is "satisfied" by the phantom open read and the cross route is built on a door that is shut). Details beyond that: unknown.
d. **s1f_rs100_1 (stuck 4+ min) is a DIFFERENT failure**: the robot never drove. `controller.log:639-641`: "STOPPED POLICY HOLD — hand-off to the policy (stall)" then
   "Blind — odometry stale (5371 ms old > CommandFreshnessTauMs 500)" and the age grows without bound to 85+ s (4285 occurrences; `controller_policy_drive.cpp:199-202`
   reads `robot_current_speed_timestamp`, written by robot_concept `robot_concept/src/specificworker.cpp:2305`). gt_pose.tsv: pose frozen at (-5.317,3.700) from the first
   minute. Webots/bridge was alive (bridge.out shows the setSpeedBase commands and 100 Hz Period lines). Which side stopped (robot_concept, pid 745880, running since 11-20 20:35, logging
   to pts/2 so unreadable; or the media/DSR path): **unknown** — needs robot_concept's console.

## 2. Failing vs succeeding at the same moment — what does NOT separate them

| variable | evidence | verdict |
|---|---|---|
| start pose | all 10 runs: teleport -5.33 3.66 0.58, landed within 1 mm (`teleport.txt`) | identical |
| fixture / remembered rooms+doors | all `fixture installed ... three_rooms_door0 (hwm 4)`, same restore messages (`*.console` lines 1-25), `fixture UNCHANGED` checks | identical |
| binaries | `agents.txt` md5 identical in all 7 checked today runs (existence-info worktree, 9 concept agents) | identical |
| learned motion calibration | `gt_error.csv` col calib_k_v/k_w/yaw at start = 1,1,0 and `carried`=0 in EVERY run, failures and passes (and 11-22 runs) | no leak (it is not carried; starts at priors) |
| P(reach), P(passable) at leg start | see 1 | not a separator |
| localisation error before/at leg 1 | displacement est-vs-world (gt_error.csv cols 5,6 vs 42,43), err@t0 0.02-0.25 m everywhere; leg-1 mean 0.02 (s1_rs035_2, a FAILURE) .. 0.63 (s1_rs100_1) | not a separator |
| robot pose when leg 1 begins | failures (-0.96,1.43) (-5.45,1.39) (-2.69,6.30) (-3.35,4.57); passes (-3.10,1.56) (-4.76,6.31) (-1.86,6.54) (-5.67,3.42) (-2.75,6.01) | no pattern |
| residual cell counts, door phi at leg start | similar | not a separator |

## 3. Variable that does separate (weakly) — CPU contention today

- A **TLC model-checker** (`java ... tlc2.TLC -workers 8`, pid 3355172, from another Claude session in the AuRo papers folder) was STARTED 2026-11-23 10:53:09 and still
  runs at ~694 % CPU (6 GB heap) on the 32-core box; `uptime` load average 17.5 / 25.5 / 22.6. It overlaps ALL ten of today's runs (first 11:31, last 14:10).
  Whether it, or any similar load, existed during the 11-22 evening runs: **unknown** (no per-run load record).
- Today's fleet is also larger: 9 concept agents + ltom_agent (`agents.txt`, 9 lines) vs c14 (4 lines: chair, table, nightstand, ltom_agent); but c3a_w2_* on 11-22 ran
  10 agents (cabinet-exit worktree) and passed, so agent count alone does not explain it.
- Controller rate (the `[CTRL] fps=` lines; policy holds below `Policy` minimum 10 Hz: "control rate N Hz below the minimum 10.0 Hz — holding", `controller.log`):
  | runs | median fps | p10 fps | median cpu | "below the minimum" lines | worst-period p90 |
  |---|---|---|---|---|---|
  | c14_r1/r2 (11-22) | 16.2 / 15.6 | 14.9 / 13.8 | 83 / 85 % | 0 / 9 | 97 / 96 ms |
  | c3a_w2_1/2 (11-22) | 14.1 / 14.2 | 12.7 / 12.5 | 78 / 147 % | 292 / 437 | 95 / 93 ms |
  | today FAIL s1_rs035_1/2, s1_rs100_1, s1f_rs035_1 | 13.8 / 12.2 / 12.7 / (52.5 outlier) | 11.4 / 10.9 / 11.0 / 10.7 | 118 / 114 / 112 / 119 % | 549 / 155 / 401 / 745 | 309 / 225 / 279 / 236 ms |
  | today PASS s1_rs035_3,4,5, s1_rs100_2,3 | 13.2 / 12.8 / 12.7 / 13.0 / 12.9 | 11.0 / 10.8 / 10.9 / 11.2 / 10.9 | 139 / 124 / 129 / 83 / 127 % | 121 / 103 / 189 / 190 / 393 | 93 / 192 / 61 / 54 / 269 ms |
  Today is ~10-15 % slower than 11-22 morning for everyone; the tail (worst-period p90) is larger in 5 of 5 failures (186-309 ms) than in 3 of 5 passes (54-93 ms). Weak, not a clean split.
- Webots itself runs slower than real time: `~/.local/logs/bridge.out` Period lines, nominal step 8 ms; mean per-run period 12.4-13.1 ms during the 11:31-12:41 runs
  (RTF ~0.6-0.65) and 15.5 ms for the 14:04 run, vs 9.4-9.6 ms when idle. Passing and failing runs have the same sim speed (12.36 fail / 12.88 fail / 12.46 pass /
  12.71 fail / 12.42 pass ...), so it is a background condition, not a separator. 11-22 sim speed: **unknown** (bridge.out was overwritten at the 11:11 restart).
- Mechanism by which a slow/starved sim and controller feed (a): the belief grades "realised vs predicted shortening" in wall time and the policy HOLDS below 10 Hz
  (`reachability_belief.h` EVIDENCE; hold counts `POLICY HOLD`: today fail 260/26/170/86, pass 124/98/126/134/268; yesterday c14 56/56/60). This is a plausible,
  NOT demonstrated, link.
- Reference rate rule from memory: "room/controller/retina at nominal rate or the run FAILS" (feedback-rates-before-features).

## 4. Changes since 11-22 evening, and persistent-state leaks

- **World file** (`git -C /home/pbustos/robocomp/components/webots-shadow diff worlds/piso/threerooms-piso.wbt`): DOOR_0 x -3.59333 -> -3.58277 (1 cm); DOOR_1 saved half open (position -1.2179; the
  bridge log "Door 'DOOR_1' swinging -1.21792 -> 0" shows it is closed by the script before every run); DOOR_2 hidden values only; Room2WallEastSouth y 9.75->9.76,
  size 1 -> 1.02; CHAIR_3 moved (-7.43,8.51) -> (-7.97,9.50) (room_2); NEW CHAIR_4 (-2.17,12.35), CHAIR_5 (0.58,7.61), CLOSET_0 cabinet (-7.38,7.2) on the room_1/room_2 wall, PortraitPainting,
  FloorLight, PhotoFrame; Toilet x -0.874 -> -0.544; mirror box size; the `bottle` DEF moved to (0.438,-0.575,-0.0095) (same as the pose the script sets every run); Shadow start pose in the file is
  (-6.32,3.69) but every run teleports to (-5.33,3.66,0.58) first. Nothing in room_1 along the door route changed except CLOSET_0 (far west wall, x -7.4) — not on the route, whether it matters to
  the remembered room_1 outline: **unknown**. Since both outcomes occur in the same world, the world edit cannot be the (sole) separator.
- **Bridge noise**: current `~/.local/logs/bridge.out` header: `SensorNoise.GyroBias = 0`, `[SensorNoise] ON ... gyro: sigma=0,002 rad/sqrt(s) bias=+0 rad/s`. The bridge was (re)started 11:11 (3 h 10 min up at 14:21,
  pid 3389791); all ten runs (first 11:31) are after the reset. Not a cause. (Note the bridge is NOT publishing ground truth: sensor noise is ON for all runs, as intended.)
- **motion_calib_state.csv** (`room_concept/etc/`, mtime 14:17, 13 KB, evidence rows): present, but each run's gt_error.csv shows the calibration starts at priors (k=1,1, yaw 0, carried 0), so it is not consumed. Not a leak.
- **C4 / piso.wbt leakage**: warm_start_tour.sh (`scripts/warm_start_tour.sh:34-51`) archives+replaces ltsm memory/history, door_identities.csv, door_phantom_memory.csv, place_maps, last_robot_pose.txt, room_name_hwm.txt.
  It does NOT touch: `room_concept/etc/motion_calib_state.csv`, `layouts/learnt/`, `scan_tape_*.bin`, `residual_concept/etc/*` (residual_room_poly.csv is output only, rewritten at each run: `residual_concept/src/specificworker.cpp:477`),
  the shared long-lived robot_concept process, or the other concept agents' etc/ files. I found no consumer of any of these that would carry C4 state into the three-room runs; the learnt layouts directory
  listing was empty. Residual risk: robot_concept (up since 11-20 20:35, state unknown) — see 1d.
- **Binaries**: same builds as 11-22 for controller/door_concept/retina/room_concept (mtimes 11-20 / 11-18 / 11-16 / 11-18); concept agents differ (today existence-info worktree 142b632d vs 11-22 fd0830ce / e5310a2a).

## 5. Ranked candidates

1. Environmental CPU starvation (TLC 8 workers, from 10:53) raising the controller's tail latency / hold rate, making the reachability belief fire spuriously twice per passage. Supported by timing + overlap; the
   split is not clean (5 passes under the same load). Causal link not demonstrated.
2. Intrinsic fragility, now exposed: `kMaxRouteFailures = 2` with one spurious "no motion escapes" per passage being already common (successes show 1 in 4 of 5 runs); a small rate shift tips the second.
3. For s1f_rs100_1 only: stalled odometry (robot_concept / DSR) — unknown origin.
4. World edits (CLOSET_0, extra chairs): no evidence of effect; unlikely.

## 6. Fix / workaround and cost

- Immediate, zero code: stop (SIGTERM) or `renice +19` / `taskset -c` the TLC to a few cores while running tours; rerun s1 x 5-6 to see if the failure rate returns to 0/11. Cost: ~12 min per tour.
- Record load per run (`/proc/loadavg`, top-5 CPU consumers) in `crossing.log` so this is no longer unknowable.
- If failures persist without TLC: (a) read robot_concept's console for 1d; (b) do not raise `kMaxRouteFailures` blindly — first trace what makes L jump to +4.5 in free space (the `unreach_L` columns in the policy CSV written at `controller_policy_drive.cpp:635`).
