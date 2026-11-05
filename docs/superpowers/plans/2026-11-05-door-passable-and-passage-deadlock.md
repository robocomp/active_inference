# Door: opened door reads passable + no deadlock at the vantage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the two causes of the 2/54 failed tour legs (warm59, warm61): σ_w inflated by a hull corner the camera cannot see, and the vantage deadlock between "servo refused" and "route arrived".

**Architecture:** door_concept: log the width-error inputs (step 0); after a live measurement confirms it, make `DoorBelief::predicted_column_extent` predict only the VISIBLE hull. controller: a pure decision `vantage_action(...)` + a robot-frame re-aim goal on the anchor, wired at the "route keeps driving" branch.

**Tech Stack:** C++23, Eigen, standalone g++ test harnesses (each test file's header carries its build line), `cbuild` for the agents (`make -j8`).

**Spec:** `docs/superpowers/specs/2026-11-05-door-passable-and-passage-deadlock-design.md` (read it; fix (d) is the chosen one, (a)/(b)/(c) are rejected there).

## Global Constraints

- No new thresholds/gates/clamps/σ-floors (CLAUDE.md "no thresholds"). Model terms only; flag anything unavoidable.
- No change to `door_leaf_channel.h`. No new crossing stage; C1 (local policy drives the passage) stands. The door route is built ONCE per leg.
- `test_door_look_vs_approach` (1a–1d, 2) and `test_door_hull_midpoint` must keep passing UNEDITED. A failure there is a design conflict: stop and report.
- CSV columns are APPENDED, never inserted; write CSVs through the classic locale (existing sink already does).
- Use `and`/`or`/`not`; type-attributed DSR access only; never `Qt::DirectConnection`.
- Builds: `cbuild` / `make -C build -j8`; check `pgrep -a cc1plus` first. Commit on `feat/room-recognition` in the rt-build worktree; do NOT push.
- Webots is shared: message the peer session `plan-imagine-subjunctive-queries` (uds:/run/user/1000/cc-socks/30490.sock) before and after using it.

## Review Focus

1. Shut leaf, or leaf open toward the camera: the visible hull must equal today's hull (no regression) — Task 3 test `closed_and_near_side_unchanged`.
2. Camera exactly in the wall plane / behind the wall (grazing): visibility must not divide by zero or return NaN extents for a door that projects fine today — Task 3 test `grazing_view_finite`.
3. Re-aim when the anchor is stale (`fresh == false`, gap not seen): must NOT re-aim (that is the "no mouth" hold, unchanged) — Task 4 test `no_mouth_still_holds`.
4. Re-aim while the body is already past the plane (s ≥ 0): no re-aim; the engaged servo keeps the hole — Task 4 test `past_plane_never_reaims`.
5. A door genuinely narrower than the template still raises σ_w — covered by the unedited `test_door_look_vs_approach` 1b (run in Task 3).

---

### Task 1: Step-0 instrumentation — the width error's inputs in the log

**Files:**
- Modify: `door_concept/src/door_belief.h` (public accessor + struct near `model_error_var()`, ~l.649; member near `model_var_w_` ~l.685)
- Modify: `door_concept/src/door_belief.cpp:174-206` (`note_model_error` fills it)
- Modify: `door_concept/src/core/door_fit.h:70-80` (append columns) and `door_concept/src/core/door_fit.cpp` (~l.934 row)
- Modify: `door_concept/src/specificworker.cpp` (cross/approach `why` strings, ~l.2729 and the cross offer ~l.2807): append `σ_w {:.3f} (post {:.3f}, model {:.3f})`
- Test: `door_concept/tests/test_door_width_error_diag.cpp` (new; header comment with its g++ line like `test_door_hull_midpoint.cpp`)

**Interfaces:**
- Produces: `struct WidthErrorDiag { float r_w = NaN, r_w_jambs = NaN, J_w = NaN, R_w = NaN, var_model_w = 0, var_post_w = 0; bool valid = false; };` and `[[nodiscard]] const WidthErrorDiag& DoorBelief::last_width_error() const;`
  `r_w_jambs` = the same residual with the hull taken over the two jamb points only (no leaf corners).
- ai2_log.csv appended columns: `we_r_w,we_r_w_jambs,we_J_w,we_R_w,we_var_model_w,we_var_post_w`

- [ ] **Step 1: Write the failing test** `test_door_width_error_diag.cpp`: belief `{3.10, 1.0, 2.1, 0}`, leaf believed at φ = 2.09 swung to the FAR side, camera 1.3 m in front on the near side (reuse the `camera()`/`params()` pattern of `test_door_look_vs_approach.cpp`); mask = the jambs' projected extent ± 2 px. Asserts: `last_width_error().valid`; `std::isfinite(r_w) and std::isfinite(r_w_jambs)`; `std::abs(r_w_jambs) < 5.f` (px); `r_w > 50.f` (the far leaf edge widens today's hull — this is the warm59 signature); `var_post_w == covariance()(1,1)`.
- [ ] **Step 2: Build+run, expect a compile failure** (no `last_width_error`).
- [ ] **Step 3: Implement** the struct, member and accessor; fill it in `note_model_error` (set `valid` only after the early returns). Append the columns + row values; append the σ_w split to the two `why` strings.
- [ ] **Step 4: Run the new test + `test_door_hull_midpoint` + `test_door_look_vs_approach`** — all PASS. If `r_w > 50` fails, the far-leaf hypothesis is wrong on the bench: STOP and report (do not adjust the numbers).
- [ ] **Step 5: `cbuild` door_concept (-j8), commit** `door_concept: log the width-error inputs (step 0 of the door-passable spec)`.

### Task 2: Step-0 live measurement (GATE — the user/agent with Webots)

**Files:** none changed. Output: a short note appended to the spec under "Step 0 result" with the numbers.

- [ ] **Step 1:** Warn the peer session. Run `HOLD=1 LEGS="room_3 room_2 room_1" bash <rt-build>/room_concept/estimation_scripts/scripts/warm_start_tour.sh warm64 crossing/fixtures/three_rooms_door0` (from room_concept/estimation_scripts); let leg 3 reach DOOR_1 from room_2 and the `open` complete; `touch crossing/warm64/stop` 30 s after the leaf reaches its commanded angle.
- [ ] **Step 2:** From `crossing/warm64/ai2_log.csv`, door face of DOOR_1 in room_2, rows after the open: report medians of `we_r_w`, `we_r_w_jambs`, `sqrt(we_var_model_w)`, and door_open_prob from the cross/approach why-lines.
- [ ] **Step 3: Decision.** If sqrt(var_model_w) ≫ 0.1 m with r_w ≫ 0 and r_w_jambs ≈ 0 → Task 3. Otherwise STOP, record the numbers, report to the user. (If the leg passes without the plateau, record it and repeat once; the failure rate is ~4 %, so also accept a reproduction by placing the robot at the warm59 pose (−4.672, 10.629) facing the door.)
- [ ] **Step 4:** Tell the peer session Webots is free.

### Task 3: The predicted hull contains only what the camera can see (door_concept) — only if Task 2 confirms

**Files:**
- Modify: `door_concept/src/door_belief.cpp:132-151` (`predicted_column_extent`)
- Test: `door_concept/tests/test_door_hull_visibility.cpp` (new)

**Interfaces:**
- Consumes: `WidthErrorDiag` (Task 1) for the assertions.
- Produces: no signature change. Behaviour: a leaf corner whose segment to the camera crosses the wall line OUTSIDE the aperture [s, s+w] (i.e. the wall occludes it) is replaced by the point where that ray meets the wall line, clamped to the nearer jamb — so it can never widen the hull past the jamb it hides behind. Corners on the camera's side of the wall, or seen through the aperture, are unchanged. Applies to both `z` levels.

- [ ] **Step 1: Write the failing tests:**
  - `far_open_leaf_hidden`: the Task 1 scene → `last_width_error().r_w` within 5 px of `r_w_jambs`, and `sqrt(model_error_var().second) < 0.1f` after one `update` with that mask.
  - `closed_and_near_side_unchanged`: φ = 0.05, and φ = 1.2 on the camera's side → `predicted_column_extent` equal (±0.01 px) to a copy of today's function kept in the test.
  - `grazing_view_finite`: camera 0.05 m off the wall plane, 40 m along it (as test 1c) → finite extents, `model_error_var().second < 0.25f`.
  - `p_passable_recovers`: with the belief from `far_open_leaf_hidden`, `rc::door::passable_logit` on a φ curve peaked at 2.09 (σ 0.012), w 1.00, needed 0.70, σ_w = sqrt(covariance_reported()(1,1)) → sigmoid > 0.95. (warm59: 0.54.)
- [ ] **Step 2: Run, expect `far_open_leaf_hidden` and `p_passable_recovers` to FAIL, the other two to PASS.**
- [ ] **Step 3: Implement** the occlusion in `predicted_column_extent` (camera position = `-Rᵀ t` from `mb.zed_T_room`, as `note_model_error` already computes it).
- [ ] **Step 4: Run the 4 new tests + `test_door_hull_midpoint` + `test_door_look_vs_approach` + `test_door_width_error_diag`** — all PASS (the last one's `r_w > 50` assertion is expected to flip: update it to `std::abs(r_w - r_w_jambs) < 5` and say so in the commit).
- [ ] **Step 5: `cbuild` door_concept, commit** `door_concept: the predicted mask hull is the VISIBLE hull (a far-side open leaf hid behind the wall; warm59 P(passable) 0.54)`.

### Task 4: No deadlock at the vantage (controller) — independent of Tasks 1–3

**Files:**
- Modify: `controller/src/controller_anchor_servo.h` (add next to `anchor_cross_geometry`)
- Modify: `controller/src/controller_session.cpp:3740-3795` (the `front_of_open` block) and its why-line composer `observe_door_passage_why` (~l.648)
- Test: `controller/tests/test_vantage_action.cpp` (new; build line as `test_anchor_servo.cpp`)

**Interfaces:**
- Produces:
  - `enum class VantageAction { Servo, HoldNoMouth, ReAim, RouteDrives };`
  - `[[nodiscard]] VantageAction vantage_action(bool servo_engaged, bool open_belief, bool open_seen, bool near_side, bool within, bool facing, bool route_arrived);`
    Order: `servo_engaged` or all five checks → Servo; `not open_seen or not open_belief` → HoldNoMouth; `near_side and route_arrived and (not within or not facing)` → ReAim; else RouteDrives.
  - `struct ReAimGoal { Eigen::Vector2f p_robot; float yaw_robot; };` `[[nodiscard]] ReAimGoal reaim_on_anchor(const AnchorCrossGeometry& g);` → `p_robot = -g.e * g.a` (the anchor's axis at the current standoff), `yaw_robot = g.heading_err`.

- [ ] **Step 1: Write the failing tests:** `warm61_reaims` (open, seen, s −1.58, e −0.46, w 0.88 → within false, route_arrived → ReAim; and `reaim_on_anchor` with a = (1,0) gives p = (0.46, 0) ±1e-5, applying it makes |e| = 0); `no_mouth_still_holds` (seen false → HoldNoMouth even when route_arrived); `past_plane_never_reaims` (near_side false, not engaged → RouteDrives, never ReAim); `engaged_servo_wins` (servo_engaged → Servo regardless); `route_not_arrived_drives` (within false, route_arrived false → RouteDrives).
- [ ] **Step 2: Build+run, expect a compile failure.**
- [ ] **Step 3: Implement both functions** in `controller_anchor_servo.h`.
- [ ] **Step 4: Run, all PASS.**
- [ ] **Step 5: Wire it** in the `front_of_open` block: compute `route_arrived` as `not path_controller.is_active() or door_route_.progress() >= door_route_.length_to_vantage()` (use whatever the route already exposes for "reached the pre/vantage waypoint"; if nothing does, `not path_controller.is_active()`). On `ReAim`: convert `reaim_on_anchor` to the room frame with the tracker pose (`cpose`: p_room = cpose.pos + R(cpose.theta)·p_robot, yaw_room = cpose.theta + yaw_robot), call `policy_tick(rc::policy::place_goal(p_room, yaw_room, path_controller.goal_threshold(), gp.tol_yaw), "door re-aim " + leg.door, ...)`, set `cross_status_ = "re-aiming on the perceived opening: " + why_not`, log once per leg `[door-route] {door}: route arrived but not in front of the opening ({why_not}) — re-aiming on the anchor`, `return true`. HoldNoMouth/Servo/RouteDrives keep today's code paths.
- [ ] **Step 6: Why-line:** in `observe_door_passage_why`, when the cross affordance node exists and the passage holds a claim on it, say `'aff_X_cross' claimed, waiting: <cross_status_>` instead of "withdrawn".
- [ ] **Step 7: `cbuild` controller (-j8); run `test_vantage_action`, `test_anchor_servo`, `test_door_passage`, `test_door_route`; commit** `controller: re-aim on the anchor when the route has arrived but the opening is off-axis (warm61 9-min stall)`.

### Task 5: Live validation (Webots)

- [ ] **Step 1:** Warn the peer. Tours from `room_concept/estimation_scripts`: forward `LEGS="room_2 room_3 room_1"` warm65, backward `LEGS="room_3 room_2 room_1"` warm66, then warm67/68 the same.
- [ ] **Step 2: Pass:** 12/12 legs, no manual help, no stall > 60 s at a door; after every `open`, door_open_prob ≥ 0.95 within 5 s of the φ posterior mean first being within 0.1 rad of the commanded angle (ai2_log_phi.csv + why-lines). Any `[door-route] … re-aiming` line: report how long the re-aim took.
- [ ] **Step 3:** Free Webots; update `resume-11-05-after-relocaliser` memory and the spec's Validation section with the result; tell the user.
