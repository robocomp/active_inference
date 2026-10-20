# One Local Motion Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the controller's seven base-command authorities and six vetoes with one look-ahead policy that
scores (v, ω) arcs by P(no contact with the true body) × P(can continue from the end pose) × progress, and releases
(back up toward the most room, re-plan with the trap remembered) when no candidate makes progress.

**Architecture:** Pure, Eigen-only modules under `controller/src/local_policy/` (rollout + swept contact, cost-to-go
field, decision, unseen-contact evidence), unit-tested with standalone g++ programs, then validated closed-loop in a
2-D harness on the catalogued wedge scenarios, then wired into `ControllerSession` behind one switch that turns the
old chain OFF (never layered), validated live, and the old chain deleted.

**Tech Stack:** C++23, Eigen 3, standalone g++ test programs (the controller's existing test style), Python 3 +
numpy for the one offline footprint generator, RoboComp/DSR agent build via `cbuild -j8`.

**Spec:** `docs/superpowers/specs/2026-10-20-local-motion-policy-design.md` (approved 2026-10-20). Read it first.

## Global Constraints

- Simulator only: the body is the simulated Shadow (`webots-shadow/protos/meshes/shadow.stl` md5 06153a46… ∪ the
  two drive wheels of `Shadow.proto`: cylinders r 0.10, width 0.05, centred x = ±0.259, y = 0).
- One body everywhere: `rc::RobotFootprint::shadow()` (convex hull) + `FootprintSafetyMarginM` (default 0.05).
- Frames: `ControllerRobotPose::theta` has robot FORWARD at theta + π/2 (its transform's x axis is the robot's
  right). The policy's `Pose2::yaw` is the FORWARD heading (forward = (cos yaw, sin yaw)). Convert ONLY in the session
  adapter (Task 8): `yaw = theta + π/2`. The footprint's own forward axis is +y; use `contains_yaw` /
  `support_radius_yaw` (they apply the quarter turn) and never hand-roll it.
- Differential base: the policy outputs (v, ω) only; `side` = 0.
- The only threshold kept is the risk level ε (`Policy.RiskEps`, acceptable P(contact) per decision). Every other
  quantity is a model parameter with a physical meaning, stated where it is defined.
- No new `runtime_checked_*` DSR calls; `std::from_chars` for parsing; data files written with
  `imbue(std::locale::classic())`; no `Qt::DirectConnection`; stop agents with SIGTERM, never `kill -9`.
- Builds: `cbuild -j8` (check `pgrep -a cc1plus` first). Standalone tests: the g++ line in each test's header comment.
- Old mechanisms are switched OFF when the policy drives (Task 8) and DELETED after live validation (Task 10) —
  never kept as a fallback layer.
- Code style: C++23 containers/algorithms; `and`/`or`/`not` instead of `&&`/`||`/`!`; match the surrounding comment
  density (this codebase explains WHY in ★ comments).

## Review Focus

1. **Robot already overlapping a return at the start** (sd < 0, e.g. pressed into a wall or a self-return leak):
   only separating motions may be eligible; it must not freeze. → Task 2 test `start_inside_only_separating`,
   Task 4 test `pressed_into_wall_releases`.
2. **No fresh returns / LiDAR stalled** (empty or stale return set): the policy must not drive blind — it stops and
   says why. → Task 4 test `stale_returns_stop`.
3. **Goal unreachable** (doorway narrower than the body, goal inside furniture): reports Unreachable, never
   oscillates. → Task 6 scenario `narrow_doorway_unreachable`.
4. **Goal behind the robot in open space**: reaches it with a sensible path (≤ 1.5 × geodesic length), no
   reverse-forward dithering. → Task 6 scenario `goal_behind_open_space`.
5. **Pose jump mid-motion** (relocalisation shifts the pose 0.3 m / 10°): no contact, no lunge. → Task 6 scenario
   `pose_jump`.

---

### Task 1: The true Shadow body (two drive wheels)

**Files:**
- Create: `controller/tools/shadow_footprint.py` (offline generator: reads the binary STL, takes the XY outline of
  the vertices with z ≤ 1.45 m robot frame, adds the two wheel rectangles x ∈ [0.234, 0.284] each side, y ∈ [−0.10,
  0.10], hulls, simplifies within +0.5 % area, prints the C++ vertex list CCW, x right / y forward)
- Modify: `common/robot_footprint/robot_footprint.cpp:16-29` (`shadow()` vertices), `common/robot_footprint/robot_footprint.h:30-75,97-101`
  (provenance comments: two drive wheels from `Shadow.proto`, the four-mecanum history, the old 0.2464 m note is the
  old assembly), `controller/src/controller_robot_body.h` (same note)
- Test: `controller/tests/test_shadow_footprint.cpp`

**Interfaces:**
- Produces: `rc::RobotFootprint::shadow()` unchanged signature, new polygon. All later tasks use it as THE body.

- [ ] **Step 1: Write the failing test** `test_shadow_footprint.cpp` (header carries its g++ line, linking
  `../common/robot_footprint/robot_footprint.cpp ../common/robot_footprint/mesh_hull.cpp`):
  - `x_max` of `shadow().polygon()` ∈ [0.2835, 0.2845]
  - `contains({±0.280f, 0.f})` true and `contains({±0.280f, ±0.15f})` false (margin 0)
  - `contains({0.f, ±0.225f})` true (body front/back)
  - polygon is CCW, convex, origin inside; `RobotFootprint::self_test()` true
- [ ] **Step 2: Run it** — expect FAIL on `x_max` (old value 0.2716) and on `contains({0.28,0})`.
- [ ] **Step 3: Run the generator** `python3 controller/tools/shadow_footprint.py webots-shadow/protos/meshes/shadow.stl`
  and paste its vertex list into `shadow()`; update the comments listed above.
- [ ] **Step 4: Run the test** — expect PASS. Also run `test_lidar_body_guard`, `test_jamb_cross`,
  `test_anchor_corridor`, `test_grid_planner_curtain` (their header g++ lines) — record which numbers change;
  a test that encoded the old body's extent is updated with a one-line ★ note, never loosened silently.
- [ ] **Step 5: Rebuild every agent that compiles the footprint** (`controller`, `door_concept`, `robot_concept`)
  with `cbuild -j8`; run `room_concept/tools/wall_slam_selftest` once. Commit
  `footprint: Shadow's two drive wheels from Shadow.proto (was four mecanum wheels)`.

### Task 2: Rollout and swept contact probability

**Files:**
- Create: `controller/src/local_policy/policy_types.h`, `controller/src/local_policy/contact.h`, `controller/src/local_policy/contact.cpp`
- Test: `controller/tests/test_policy_contact.cpp`

**Interfaces:**
- Produces (namespace `rc::policy`):
  - `struct Pose2 { Eigen::Vector2f p; float yaw; };` — room frame, yaw = forward heading.
  - `struct Return { Eigen::Vector2f p; float sigma; float p_exist; };` — room frame; sigma = range σ (m).
  - `struct BaseLimits { float v_max, w_max, a_lin, d_lin, a_rot, d_rot, v_min_eff, w_min_eff; };` — the last two
    are the base's deadband (smallest speeds it actually executes; filled in Task 7).
  - `std::vector<Pose2> rollout(const Pose2& start, float v, float w, float hold_s, const BaseLimits& b, float dt);`
    — hold (v, w) for `hold_s`, then brake both axes to rest at d_lin / d_rot; poses every dt, differential
    kinematics (exact arc integration).
  - `float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
    float sigma_pose_xy, float sigma_pose_yaw);`
- Model (the one non-obvious algorithm): for each return j, `sd_j` = min over sweep poses of the signed distance from
  q_j to the body polygon (margin included) at that pose; σ_j² = σ_r² + σ_xy² + (ρ_j·σ_yaw)² with ρ_j the return's
  distance from the pose centre; P(contact_j) = p_exist·Φ(−sd_j/σ_j); result = Σ_j log(1 − P(contact_j)). Broad-phase:
  only returns within circumscribed radius + margin + path length + 3σ of the start pose. A return already inside the
  body at the START pose counts only if the sweep makes it deeper (sd decreases along the sweep) — separating motion
  out of an overlap is free.

- [ ] **Step 1: Write the failing tests** in `test_policy_contact.cpp` (g++ line in header):
  - `rollout_straight`: v 0.3, w 0, hold 1 s, d_lin 1.0 → end y-advance = 0.3 + 0.045 (± 0.005), final speed 0.
  - `rollout_turn_in_place`: v 0, w 1.0 → end yaw change = 1.0 + 0.5/d_rot·1.0² (± 0.02).
  - `wall_beside_parallel_free`: wall of returns along x = +0.40 (robot frame right), straight forward 1 s →
    log_p_no_contact > log(0.99).
  - `wall_beside_turn_hits`: same wall at x = +0.34, in-place turn 1 rad → log_p_no_contact < log(0.5).
  - `wall_ahead_fast_vs_slow`: wall 0.8 m ahead; v 0.35 scores lower than v 0.10.
  - `start_inside_only_separating`: one return 0.02 m inside the body's right flank; moving left (away) → > log(0.99);
    moving right (into it) → < log(0.5).
- [ ] **Step 2: Run** — expect compile FAIL (headers missing).
- [ ] **Step 3: Implement** `rollout` and `log_p_no_contact` in `contact.cpp` (Φ via `std::erfc`).
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `local_policy: rollout + swept-body contact probability`.

### Task 3: Cost-to-go field with trap memory

**Files:**
- Create: `controller/src/local_policy/cost_to_go.h`, `controller/src/local_policy/cost_to_go.cpp`
- Test: `controller/tests/test_policy_cost_to_go.cpp`

**Interfaces:**
- Produces:
  ```cpp
  class CostToGo {
  public:
      struct Grid { Eigen::Vector2f origin; float cell; int w, h; };
      void build(const Grid& g, const std::function<bool(const Eigen::Vector2f&)>& free,
                 const std::function<float(const Eigen::Vector2f&)>& extra_cost_m,
                 std::span<const Eigen::Vector2f> goal_points);
      float at(const Eigen::Vector2f& p) const;          // metres-equivalent; +inf where unreachable; bilinear
      bool reachable(const Eigen::Vector2f& p) const;
      void add_trap(const Eigen::Vector2f& p, float radius_m, float cost_m);   // applied at the next build()
      void decay_traps(float factor);                     // multiplies every trap cost; drops below 1e-3 m
  };
  ```
- `free(p)` is the planner's body-agnostic free test (distance to occupied > inscribed radius); `extra_cost_m` is the
  residual risk priced as today (`GridPlanner::risk_at`) plus traps. 8-connected Dijkstra (cost × √2 diagonals).

- [ ] **Step 1: Write the failing tests:** `l_corridor_geodesic` (L-shaped free space; field at the far arm's end ≈
  path length around the corner ± 1 cell, not the Euclidean distance); `walled_off_unreachable` (`reachable` false,
  `at` = inf); `trap_reroutes` (two parallel corridors to the goal; a trap of 2 m in the shorter makes the field at
  the start follow the longer); `goal_set_line` (goal = a segment; field 0 along it).
- [ ] **Step 2: Run** — compile FAIL.  **Step 3: Implement.**  **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** `local_policy: geodesic cost-to-go with trap memory`.

### Task 4: The decision (look-ahead, continuability, release)

**Files:**
- Create: `controller/src/local_policy/local_policy.h`, `controller/src/local_policy/local_policy.cpp`
- Test: `controller/tests/test_local_policy.cpp`

**Interfaces:**
- Consumes: Task 2 (`Pose2`, `Return`, `BaseLimits`, `rollout`, `log_p_no_contact`), Task 3 (`CostToGo`).
- Produces:
  ```cpp
  struct Goal { enum class Kind { Place, Doorway, Heading } kind; Eigen::Vector2f p{0,0};
                std::optional<float> yaw; Eigen::Vector2f normal{0,1}; float width = 0.f;
                float tol_xy = 0.25f, tol_yaw = 0.06f; };
  struct PolicyParams { float horizon_s = 1.0f, dt = 0.05f; int n_v = 7, n_w = 11, n_cont_v = 3, n_cont_w = 5;
                        float risk_eps = 0.01f, progress_scale_m = 0.10f, returns_max_age_s = 0.5f; };
  struct PolicyInput { Pose2 pose; float v_now = 0, w_now = 0, sigma_xy = 0, sigma_yaw = 0, returns_age_s = 0;
                       std::span<const Return> returns; const rc::RobotFootprint* body = nullptr; BaseLimits base;
                       const Goal* goal = nullptr; const CostToGo* field = nullptr; };
  struct Decision { enum class Mode { Drive, Release, Arrived, Stuck, Blind } mode; float v = 0, w = 0;
                    float log_p_safe = 0, log_p_cont = 0, progress_m = 0; int eligible = 0; std::string why; };
  Decision decide(const PolicyInput& in, const PolicyParams& p);
  ```
- Algorithm (spec §4.3–4.5): candidates = n_v × n_w grid over the dynamic window [v_now ± a_lin·horizon] ∩
  [−v_max, v_max] × [w_now ± a_rot·horizon] ∩ [−w_max, w_max], always including (0, ±w) and (−v, 0), excluding
  |v| ∈ (0, v_min_eff) and |w| ∈ (0, w_min_eff). Eligible ⇔ 1 − exp(log_p_safe) ≤ risk_eps. Progress = field(start) −
  field(end) (Place/Doorway; plus heading agreement when `yaw` set), heading agreement for Heading.
  log_p_cont(end) = log((k + 1)/(N + 2)), k = eligible coarse continuations (n_cont_v × n_cont_w) from the end pose
  with progress ≥ 0. Score = log_p_safe + log_p_cont + progress_m / progress_scale_m. Drive = best eligible with
  progress > 0. If none: **Release** = among eligible candidates with v ≤ 0, max (log_p_cont(end) −
  log_p_cont(now)); `why` names it. No eligible candidate at all → **Stuck** (v = w = 0). returns_age_s >
  returns_max_age_s → **Blind** (stop). Inside tolerances → **Arrived**.
  The ONE threshold is risk_eps; progress_scale_m is the distance worth one nat of safety (stated, tunable, not a gate).

- [ ] **Step 1: Write the failing tests** (static snapshots, no time loop):
  - `open_space_goal_ahead`: no returns, goal 3 m ahead → Drive, v ≥ 0.9·v_max, |w| < 0.1.
  - `side_on_wall_goal_behind_left` (tour3_1 leg 3): wall along x = +0.37 (robot right), goal 0.7 m away at
    bearing +150° → Drive or Release, and the chosen candidate's end pose has log_p_cont ≥ log(1/3); NOT a pure
    in-place turn whose sweep hits the wall.
  - `pressed_into_wall_releases`: wall 0.01 m inside the right flank, goal straight through the wall → Release with
    a motion that moves away from the wall.
  - `dead_end_releases`: nose 0.05 m from a wall in a 0.8 m cul-de-sac, goal through the wall → Release, v < 0.
  - `stale_returns_stop`: returns_age_s 1.0 → Blind, v = w = 0.
  - `arrived`: at goal within tol → Arrived.
  - `heading_goal_beside_wall`: Heading goal 90° right with a wall at x = +0.34 → never a candidate with
    P(contact) > risk_eps (the turn is evaluated, not assumed).
- [ ] **Step 2: Run** — FAIL.  **Step 3: Implement `decide`.**  **Step 4: Run** — PASS; print the per-call time
  (median over 100 calls with 300 returns) — must be ≤ 15 ms at -O2; if not, reduce n_cont_* / merge returns into
  σ_r cells first and record the numbers in the test header.
- [ ] **Step 5: Commit** `local_policy: look-ahead decision with continuability and release`.

### Task 5: Unseen-contact evidence

**Files:**
- Create: `controller/src/local_policy/unseen_contact.h` (header-only)
- Test: `controller/tests/test_unseen_contact.cpp`

**Interfaces:**
- Produces:
  ```cpp
  class UnseenContact {
  public:
      void update(float v_cmd, float w_cmd, float ds_meas, float dyaw_meas, float dt, const Pose2& pose,
                  const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m);
      std::optional<Return> hypothesis() const;   // at the body edge in the blocked direction, p_exist = σ(L)
      void reset();
  };
  ```
- Model: per tick, log-likelihood ratio of the measured displacement under "blocked" N(0, σ_odo) vs "free"
  N(v_cmd·dt, σ_odo) (only while |v_cmd| ≥ v_min_eff; rotation analogously), accumulated in L; moving as commanded
  drives L down. No timer.

- [ ] **Step 1: Failing tests:** `commanded_no_motion_grows` (v 0.2, ds 0, 20 ticks → p_exist > 0.9, hypothesis in
  front of the body); `moving_freely_none` (ds = v·dt → no hypothesis); `below_deadband_no_evidence` (v < v_min_eff
  → L unchanged); `reverse_blocked_behind`.
- [ ] **Steps 2–4:** run FAIL, implement, run PASS.  **Step 5: Commit** `local_policy: unseen contact from odometry`.

### Task 6: 2-D closed-loop harness and the catalogued scenarios

**Files:**
- Create: `controller/tests/sim2d/sim2d.h` (world of segments and boxes; 1440-beam raycast with Gaussian range noise
  σ 0.02; differential base with Task-2 `BaseLimits`, 60 ms command latency and a deadband; exact contact test of
  the true body against the world, sampled every 5 mm; optional OU pose-error process 0.5 m / 13° over ~20 s),
  `controller/tests/test_policy_scenarios.cpp` (driver: each tick → returns from raycast in room frame → `decide` →
  base; CostToGo built from the world's free space; on Release: `add_trap` at the pose where progress stopped,
  rebuild; Stuck/unreachable → scenario ends)
- Reuse: copy (do not include) the DOOR_1 boxes and raycaster from `controller/tests/test_jamb_cross.cpp:74-146`.

**Interfaces:**
- Consumes: Tasks 2–5 only (the harness never includes `ControllerSession`).
- Produces: `struct ScenarioResult { bool reached, unreachable; int contacts; float max_no_progress_s; float path_m; int releases; };`
  `ScenarioResult run_scenario(const sim2d::World&, Pose2 start, Goal goal, float time_limit_s, unsigned seed);`

- [ ] **Step 1: Write the scenarios as failing tests** (oracles: contacts == 0; reached unless stated; no window of
  more than 20 s without field decrease; after a Release the next route differs):
  - `tour3_1_leg3` — the room_1 corridor west of the x≈0 partition (room_1 frame): start (−0.26, 2.22), forward
    heading such that the robot's right flank is 0.02 m from the partition face, goal (−0.69, 2.84).
  - `try101_corridor` — 1 m corridor stub before DOOR_1, start 0.37 m from the east wall, goal the far side of
    the door (Doorway goal, leaf open 90° on the near side).
  - `try105_jamb` — 0.37 m before the plane, 0.14 m off-centre, heading 15° off the normal.
  - `door_crossing_both_ways` — leaf open 90° on the near side, then on the far side.
  - `dead_end_release` — enters a 0.8 m cul-de-sac on a stale map, must release and take the other way.
  - `narrow_doorway_unreachable` — 0.55 m gap → unreachable, contacts 0, no oscillation (≤ 2 releases).
  - `goal_behind_open_space` — path_m ≤ 1.5 × geodesic.
  - `pose_jump` — 0.3 m / 10° pose jump mid-corridor → contacts 0, reached.
- [ ] **Step 2: Run** — they fail only where the policy is wrong (Tasks 2–5 exist); fix the POLICY (not the
  scenario) until all pass, 20 seeds each. Record per-scenario time-to-goal in the test output.
- [ ] **Step 3: Commit** `local_policy: closed-loop 2-D harness, catalogued wedge scenarios pass`.

### Task 7: Policy inputs the controller does not provide yet

**Files:**
- Modify: `controller/src/controller_world_model.cpp:194-244` (`read_pose_uncertainty` returns the full 3×3 SE(2)
  covariance as `Eigen::Matrix3f cov` in the existing result struct; keep `xy_std_m`/`theta_std_rad` derived),
  `common/robot_capability/robot_capability.h:53-87` (`std::optional<float> min_linear_speed, min_rot_speed` = the
  measured deadband), the Shadow capability values (`robot_concept` config that publishes them),
  `controller/src/controller_obstacle_tracker.cpp:2257-2328` + `controller/etc/config.toml` (new key
  `Lidar.ContactMinHeightM`: returns for the policy start just above the floor instead of
  `TemporaryObstacleMinHeight` 0.20; the existing 0.20 band stays for the old consumers until Task 10)
- Create: `controller/tools/measure_deadband.py` (steps the base through small v and ω commands in Webots via the
  controller's remote channel, reads odometry, reports the smallest speeds that move), `controller/tools/floor_band.py`
  (histogram of return z for a parked robot on open floor → the value for `Lidar.ContactMinHeightM` = floor-return
  p99.9 + σ_r; write the measured number and the run id into the config comment)
- Test: extend `controller/tests/test_lidar_body_guard.cpp` with `low_returns_reach_policy` (a return at z 0.10 m
  reaches the policy's set, not the old 0.20 band)

- [ ] **Step 1:** failing test `low_returns_reach_policy`; **Step 2:** run FAIL; **Step 3:** implement the three
  inputs; **Step 4:** run the measurement scripts against the running simulator (fleet started with
  `run_crossing.sh <dir> here`, chair_concept excluded), put the measured values in config/capability with their
  run ids; **Step 5:** run tests PASS, `cbuild -j8` controller + robot_concept, commit
  `controller: full pose covariance, measured base deadband, low contact band for the policy`.

### Task 8: Wire the policy into the controller (old chain OFF)

**Files:**
- Create: `controller/src/controller_policy_drive.cpp` (session member functions below), `controller/src/local_policy/goal_adapter.h`
- Modify: `controller/src/controller_session.h` (members: `rc::policy::CostToGo policy_field_`,
  `rc::policy::UnseenContact unseen_`, `bool policy_enabled_`), `controller/src/controller_session.cpp:5648`
  (`execute_plan`: when `policy_enabled_`, dispatch to `drive_with_policy` BEFORE the escape / backoff / anchor
  cross / lock-on / orient / band / tracker branches, which are then not run), `:2757` (`ensure_current_plan`: escape
  and backoff branches skipped when enabled), `controller/src/specificworker.cpp:2194` (relative motion becomes a
  Place goal when enabled), `controller/etc/config.toml` (`[Policy] Enabled = true`, `RiskEps = 0.01`,
  `ProgressScaleM = 0.10`, `HorizonS = 1.0`)
- Test: `controller/tests/test_goal_adapter.cpp`

**Interfaces:**
- Consumes: Tasks 2–7.
- Produces:
  - `rc::policy::Pose2 to_policy_pose(const ControllerRobotPose& r);` — `{r.pos, r.theta + π/2}` (the ONLY place).
  - `rc::policy::Goal goal_for_target(const TargetInfo& t, const std::optional<rc::DoorwayGap>& gap, bool cross);`
    — standpoint/place (with heading if the affordance has one), Doorway from the measured gap during a cross
    (goal points = the segment 2 m past the gap centre along `normal`, width = gap width), Heading for orient/lock-on.
  - `bool ControllerSession::drive_with_policy(const ControllerRobotPose&, ControllerObstacleTracker&, ControllerMotionCommander&, rc::AffordanceManager&);`
    — builds returns (room-frame buffer, policy band, σ_r = `LidarRangeSigmaM`, existence as today, + `unseen_`
    hypothesis), rebuilds `policy_field_` when goal/world hash changes, calls `decide`, sends (v, ω) via
    `send_speed_command(v, 0, -ω_session_sign…)` (match the session's rot sign at controller_session.cpp:6607), on
    Release adds a trap and rebuilds, on Stuck/unreachable calls the existing `mark_reached(Unreachable)` path,
    on Arrived the existing `finalize_reached`. Writes one `[why]` verdict per mode change (motion_verdict.h) and a
    `policy_decisions.csv` row per cycle (mode, v, w, log_p_safe, log_p_cont, progress, eligible, why).
  - The door-crossing completion semantics are KEPT: extract the through/complete checks of `step_anchor_cross`
    (controller_session.cpp:3732-4130: through only if observed past the standoff, a claim that starts past the plane
    cannot complete, `rc::cross_continues_without_anchor`) into `bool cross_progress_complete(...)` used by
    `drive_with_policy`; the affordance protocol calls are unchanged.

- [ ] **Step 1:** failing `test_goal_adapter.cpp`: pose conversion (theta 0 → yaw π/2), Doorway goal points lie
  2 m past the gap along its normal, Heading goal for an orient target.
- [ ] **Step 2–4:** run FAIL, implement, run PASS; `cbuild -j8` controller.
- [ ] **Step 5: Live smoke test** in the three-room world (fleet without chair_concept): `cross door_2` from room_1,
  then back; `policy_decisions.csv` shows no mode flapping (< 1 change/s median); commit
  `controller: one local policy drives every motion (old chain off when [Policy] Enabled)`.

### Task 9: Live validation

**Files:**
- Modify: `room_concept/estimation_scripts/scripts/three_rooms_tour.sh` (already exists on main: legs
  door_2, door_2, door_1, door_1, door_2, door_2 with 60 s exploration; add the WT/WT_AGENTS defaults for this branch)

- [ ] **Step 1:** run the three-room tour twice (`three_rooms_tour.sh tourP1`, `tourP2`): pass = 6/6 legs both runs,
  `room_3` created behind door_1 (room-name fix on main), zero `Stuck` verdicts lasting > 20 s in
  `policy_decisions.csv`, no Webots contact (proximity log `body_clear_m` never < 0).
- [ ] **Step 2:** run `round_trips_mission.sh tryP3` (6 legs, two-room door): 6/6.
- [ ] **Step 3:** record results (per-leg times, releases, unreachable) in the run ledger
  `.superpowers/sdd/2026-10-18-interaction-anchor/progress.md` and commit any script change.

### Task 10: Delete the old chain

**Files:**
- Modify/delete (spec §4.7): `clearance_speed_limit` + `controller/src/clearance_ceiling.h` +
  `controller/tests/test_clearance_ceiling.cpp`, the 0.05 speed floor (trajectory_controller.cpp:89-90),
  `lidar_guard_command` / `cap_by_returns` / `anchor_make_turn_room`, returns backoff (`step_returns_backoff`,
  `lidar_return_backoff`), escape (`begin_escape`, `step_escape`), the stall judge's escape path, the anchor cross
  servo + jamb guards (keeping `cross_progress_complete`), `step_orient` / lock-on motion, relative motion's control
  law, `apply_uncertainty_speed_limit`, the elastic band, the PLAIN tracker's command path, the ESDF disc
  self-filter (trajectory_controller.cpp:1262-1312), `[Policy] Enabled` (always on), and every config key that only
  those read (list them in the commit message).
- Test: all controller test programs; tests that only exercised deleted code are deleted with it.

- [ ] **Step 1:** delete, build (`cbuild -j8`), run every remaining controller test program — all PASS.
- [ ] **Step 2:** rerun Task 6 scenarios (20 seeds) and one three-room tour — 6/6.
- [ ] **Step 3:** commit `controller: delete the cap/backoff/escape/servo chain — the local policy is the only authority`
  with the removed config keys listed.
