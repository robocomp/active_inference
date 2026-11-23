# Room bench tail: re-run with the agent's own localisation against the promoted outline. ESTIMATE ONLY (nothing built or run), 2026-11-23

Read: `room_concept/tools/newroom_bench.h` (548 lines, `run_newroom`), `tools/wall_slam_selftest.cpp` (5521 lines; the `WS_NEWROOM` branch at 3535-3600+), the live pose path in `src/room_concept_update.cpp:334-385`,
`src/room_concept_wall_slam.cpp:403-445`, `src/room_concept_update_solve.cpp:50-80`, `src/room_scene_graph.cpp:4770-4790`, results in `estimation_scripts/explorer_rerun_1122/newroom_matterport594_f8000.csv`.

## 1. Is there a switch for agent-localisation mode?

**No, not in the newroom bench.** `run_newroom` takes its pose from `rc::slam::ScanSlam` every frame, before and after promotion (`newroom_bench.h:~201-205`, mirrored comment "POSE: plain scan-to-map SLAM, room_concept_update.cpp:358-364").
`RunConfig::with_sdf` (`wall_slam_selftest.cpp:504`, used at `:892` as `in.no_sdf = not cfg.with_sdf`) is an SDF switch for the OLDER box/wall-map bench (`run_wall_slam` through `rc::gn::Solver`, `room_gn_solver.h`); it is not read by `run_newroom`.
`WS_NEWROOM` env knobs are only `WS_NR_FRAMES, WS_NR_OUT, WS_NR_LAYOUTS` (+ the box-bench start knobs).

## 2. Important finding before building anything: the live refiner already uses the SLAM pose

In the agent, after promotion the room is GIVEN and the pose is solved against the outline's SDF (`adopt_map_layout_as_given`, `room_concept_wall_slam.cpp:403-445`: "the pose is solved against its SDF from here on (plain SLAM stops)"),
**but plain SLAM keeps running as a shadow, and the layout refiner accumulates its evidence at the shadow SLAM pose, "never the SDF pose: the evidence must not depend on the model it judges"** (`room_scene_graph.cpp:4777-4786`;
`record_slam_shadow`, `room_concept_wall_slam.cpp:403-420`, `r.slam_pose = plain_slam_.pose()`). So the bench's refinement stage, which scores layouts on evidence registered at plain-SLAM poses, **already does what the agent does**.
What the bench does NOT reproduce after promotion is (i) the pose the controller and the coverage explorer use to drive (agent: SDF-localised pose; bench: SLAM pose), so the trajectory, and therefore which cells get swept, can differ;
(ii) the pose the published outline is expressed in relative to the world. A "re-run with the agent's localiser" tests (i) only; it cannot change how the refiner weighs evidence.

## 3. Which rooms are below IoU 0.90, and is drift logged?

`newroom_matterport594_f8000.csv` logs per plan `pose_err_med_m` and `rot_err_med_deg` (median over frames of |SLAM - true|, map frame), plus `seen_frac, frames_outside, frames_blocked, iou_promo`. Of 594 plans, **20 have IoU < 0.90**
(overall median IoU 0.970; pose error median 0.0197 m, max 0.381 m; Spearman IoU vs pose error rho = -0.30, p = 1e-13, n = 594).
In the 20: pose error median 0.028 m (max 0.381 m), rotation error median 0.93 deg (max 3.2 deg); **4 of 20 have pose error > 0.1 m** (room 510: 0.381 m, 19: 0.246 m, 354: 0.150 m, 321: 0.131 m); the other 16 are <= 0.077 m, so their shortfall is
not localisation drift. `seen_frac < 1` in 11 of the 20 (coverage), est vertex count differs from the truth in 15 of 20 (over-segmentation: e.g. 6 -> 16, 12 -> 32). `iou_promo` (the outline stage alone) is already < 0.90 in 16 of the 20, i.e. the refiner
started low in most. Per-plan file: `explorer_rerun_1122/per_plan.csv`; plan ids of the 20: 92, 492, 305, 328, 458, 487, 484, 341, 510, 354, 74, 99, 227, 321, 19, 10, 285, 30, 111, 498.
Expected effect of the re-run: small for at most the 4 high-drift rooms (the localiser would pin the pose to the promoted outline, whatever the SLAM drift); for the other 16 none from drift.

## 4. What would have to change

1. `newroom_bench.h` (+ ~150-250 lines): after `startup_promoted`, build a `RoomModel` on the promoted outline (`model.update_polygon_vertices(...)`, as `adopt_map_layout_as_given`) and run the agent's SDF solve per frame (`rc::gn::Solver::solve` with `in.no_sdf = false`, a window, the
   odometry prior from the same motion-fusion calls the box bench uses, `RoomConcept::compute_motion_covariance/fuse_priors`, see `wall_slam_selftest.cpp:775-800`); feed the SDF pose to the controller/explorer while the SLAM pose keeps feeding `free_space`/refiner; log both and their disagreement (the agent's `slam_shadow.csv`).
2. `wall_slam_selftest.cpp` (+ ~20 lines): an env switch (e.g. `WS_NR_LOCALISE=1`) and the extra CSV columns.
3. Not reproducible without much more work (would stay approximations): the agent's early-exit gate, relocaliser, motion-prior learners, object anchors and the gauge hand-over `T_pub_int_`. The bench would therefore be "SDF localiser", not the whole agent.
Cheaper alternative worth stating: the live agent logs both poses (`tmp/slam_shadow.csv`: sdf vs slam, columns `d_pos_m`) so the real tours already give the disagreement (not read here).

## 5. Time

- Build: `wall_slam_selftest` is one very large TU (5.5 k lines + torch/room headers) plus link of `rc_room_estimator`; it is `EXCLUDE_FROM_ALL`. A TU does not parallelise, so `-j8` gives no speed-up on it: **my estimate 5-15 min per rebuild, a few GB**. Not measured (no build timing in the repo; `bin/wall_slam_selftest` is absent from `room_concept/bin`, only `room_concept` and `proto_mouth_graph_test`).
  A clean reconfigure/rebuild of dependent libs would be longer; check `pgrep -a cc1plus` first (CLAUDE.md OOM rule).
- Run: recorded `wall_s`: mean 0.69 s per plan, 411 s for all 594 (single process). Adding a GN-SDF solve per frame after promotion (median plan 365 frames; promotion typically ~100 frames in) at an assumed 5-20 ms per solve: +1-4 s per plan, i.e. **15-50 min for all 594**, about 2 min for the 20 sub-0.90 rooms. (The per-solve cost is an assumption; the box bench's measured cost was not looked up.)
- Hours: implement and compile-debug 4-6 h (3-5 rebuild cycles), validation on 5-10 rooms plus the 20 sub-0.90 rooms 1 h, full 594 run + paired analysis 1-2 h. **Total about 6-9 h (one working day).**

## 6. Main risk

That the result is uninformative by construction: the refiner's evidence is shadow-SLAM-anchored in the agent too (Section 2), only 4 of the 20 weak rooms drift, and a simplified in-bench SDF localiser is not the agent's localiser (no early exit, relocaliser, anchors),
so a "no change" outcome would not support a claim about the full system and a "change" could come from the bench's approximation. Second risk: the SDF solve against a WRONG outline (the 20 rooms are exactly those) can pull the driving pose and change the trajectory, so a paired comparison per plan (same seed) is required, not a pooled mean.
The paper claim most safely supported without this run: the refiner scores at the shadow pose by design (`room_scene_graph.cpp:4777`), and the logged pose error of the weak rooms (Section 3).
