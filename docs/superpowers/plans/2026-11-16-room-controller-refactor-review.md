# room_concept + controller: cleanup review and refactoring plan

Read-only review of `main` at `dbfb77ec` (2026-11-16). Nothing was edited, built or run.
All paths are relative to `/home/pbustos/robocomp/components/active_inference`.

**How "unused" was established.** Textual analysis over every tracked `*.cpp`, `*.h`, `*.py`, `*.sh`:
word-level `git grep -nw <name>` across the whole repository (not only the agent), plus the include graph
and `src/CMakeLists.txt`. An item is called dead only when the name occurs at its declaration/definition
and nowhere else. This cannot see a use built by a macro or a string (Qt `SIGNAL()`/`SLOT()` text, `.ui`
auto-connect); none of the names below has that shape, but it is the residual risk. The compiler is the
final check: delete, build, and the linker confirms. Anything I could not prove is marked **ask**.

Note: the shell `grep` on this machine is a wrapper that honours `.gitignore`; the commands below use
`git grep` so the owner gets the same answer.

---

## 1. Summary

| | room_concept | controller |
|---|---|---|
| `src/` files (`.cpp`/`.h`) | 159 | 116 |
| `src/` lines | 64,339 | 53,913 |
| comment-only lines | 17,391 (27 %) | 14,110 (26 %) |
| `src/` files over 800 lines | **16** | **16** |
| tools/tests C++ files over 800 | 4 (+3 Python) | 4 |
| largest file | `room_scene_graph.cpp` 4,873 | `controller_session.cpp` 9,567 |
| largest function | `load_room_config` 1,058; `Channel::plan_path` 1,023 | `build_planning_step` 1,388; `execute_plan` 1,074 |

Lines removable:

| Class | room_concept | controller |
|---|---|---|
| **High confidence** (dead: no caller, not compiled, or self-declared "delete me") | ~2,700 (1,842 are four tracked `.bak` files) | ~950 in `src/` + ~1,460 in `tests/` |
| **Ask** (gated off in the live config, no script flips it) | ~6,500 (box layout 4,389; object anchors ~1,300; Rerun 402; smaller ones) | ~1,900 (obstacle creation ~800; near-field arm ~460 + 652 test; unread diagnostics ~700) |
| Test code compiled into the agent (move, not delete) | 458 (two `*_test.cpp` in `src/`, not in CMake) | 1,969 (six `self_test()` bodies) |
| Kept by owner decision | legacy A/B arms in the estimator | the whole old drive chain (`Policy.Enabled=false`) |

Three things worth knowing before any plan:

1. **room_concept already has a refactor in progress** (`room_concept/docs/refactor_proposal_2026-10-13.md`).
   Steps 0–3 are done (lint, oracle, `rc_room_estimator` library, `room_concept.cpp` 9,444 → 15 TUs,
   `room_concept.h` 2,482 → 1,533). The plan below continues it rather than replacing it.
2. **The size guardrail has been outrun.** `tools/lint_allow.txt` allows `room_scene_graph.cpp` at 4,142; it is
   4,873 now. `free_space_polygon.h` 1,069 → 1,173, `room_concept.h` 1,533 → 1,562. Two new offenders are not
   listed at all (`reloc_search.cpp` 1,321, `layout_refiner.h` 873), and the list still names
   `src/mount_calibrator.cpp`, which no longer exists. The lint only warns at configure time, so nothing stopped it.
3. **controller has no guardrail and no test runner.** 47 tests in `controller/tests/` are built by hand from
   the `g++` line in each file's header; one of them no longer compiles (`bench_c1_door.cpp`, see 3a).

---

## 2. room_concept findings

### 2a. Dead code

| Item | Where | Evidence | Conf. | Action |
|---|---|---|---|---|
| Four backup files, tracked in git | `src/specificworker.cpp.bak.pre-gt` (1,064), `src/specificworker.h.bak.pre-gt` (219), `src/room_viewer.cpp.bak.pre-gtplot` (422), `src/room_viewer.h.bak.pre-gtplot` (137) | `git ls-files room_concept/src \| grep bak` lists all four; no CMake reference | high | remove |
| `EpistemicController` "Level 2" (arc policies, rollout, EFE, `plan()`, speed governor) | `src/epistemic_controller.cpp:48-354` (`set_lidar_obstacles` 27, `nearest_wall_distance_sq` 13, `set_localization_quality` 9, `generate_arc_policies` 62, `rollout_policy` 24, `evaluate_policy_efe` 101, `plan` 24), decls `epistemic_controller.h:107-131` | `git grep -nE "\bplan\(\)" -- room_concept` → only the definition. The three private functions and both setters occur at decl + def only. The agent uses the class only as a holder: `epistemic_planner()`, `set_room_bounds`, `set_room_polygon`, `set_robot_state`, `set_robot_footprint` | high | remove; keep `EpistemicPlanner` |
| Config reads feeding only that dead code | `src/room_config.cpp:1030-1074` (~23 `ec.*` keys: `NumArcCurvatures`, `HorizonSteps`, `Dt`, `MaxAdvSpeed`, `MaxRotSpeed`, `WEpistemic`, `WPragmatic`, `WHeading`, `WBoundary`, `KRot`, `GaussianSigma`, `SpeedHorizonS`, `Obstacle*`, `WObstacle`, `WallFilterMargin`, `BandwidthCoupling`, `SdfSafe`, `SdfDanger`, `GovernorAlphaMin`, …) and the same keys in `etc/config.toml [EpistemicController]` | consumers are the dead functions above | medium (check each `ec.` field for a second reader before deleting its key) | remove with the code |
| `RoomConcept::compute_velocity_adaptive_weights` (80 lines) | `src/room_concept_motion_prior.cpp:30`, decl `room_concept.h:1325` | `git grep -nw compute_velocity_adaptive_weights` → decl + def. `room_concept_solve.cpp:193` says it "preconditions Adam's" gradient; Adam was deleted in `db19b858`. Config `VelocityAdaptiveWeights = false` | high | remove, with `Params::velocity_adaptive_weights` and the config key |
| Declared, never defined or called | `RoomConcept::compute_sdf_median_abs` (`room_concept.h:1555`), `get_image_edge_stats` (`:559`), `CornerDetector::convex_rej_agree_mean` (`corner_detector.h:428`), `RoomSceneGraph::layout_publish_armed` (`room_scene_graph.h:422`), `EpistemicPlanner::mark_visited` (`epistemic_planner.h:474`), `CameraCoverage::unseen_length` (`camera_coverage.h:170`) | one occurrence each | high | remove |
| Defined, never called | `RoomConcept::get_loc_state` (`room_concept.cpp:84`, 7), `predict_pose_forward` (`room_concept_prediction.cpp:239`, 17), `Viewer2D::draw_all_trajectories` (`viewer_2d.cpp:710`, 36), `draw_score_grid` (`:1503`, 43), `draw_ior_grid` (`:1547`, 36), `center_on` (`:195`), `set_transform` (`:119`), `lidar_points_visible` (`:451`), `CameraVisualizer::project_points_to_image` (`camera_visualizer.cpp:716`, 50), `boxch::Channel::traversable` (`room_boxes_channel.cpp:380`, 16), `reloc::exit_prior` (`reloc_search.cpp:691`), `LikelihoodRaster::ll_at` (`:756`), `project_jacobian` + `project_jacobian_numeric` (`camera_jacobian.h:56,105`, 72), `px_per_rad_at` (`image_edge_ops.h:348`, 24), `p_continuous` (`door_crossing.h:267`, 13) | decl + def only, repo-wide | high | remove |
| Unused inline accessors | `is_loc_initialized` (`room_concept.h:124`), `governor_alpha`, `cell_scores`, `ior_cells` (`epistemic_planner.h:318,325`), `have_room_polygon`, `room_polygon_offset`, `driving_ingestor`, `driving_source` (`calib_channels.h:52-56`), `estimated_wheel_omega_scale`, `last_information`, `intake` (`motion_calibration.h:312,336,338`), `last_misfit_nats` (`room_boxes_channel.h:197`), `last_state` (`status_reporter.h:104`), `reset_tour_stats` (`corner_detector.h:300`), `verdict_name` (`calibration_intake.h:65`), `RerunLogger::is_enabled` (`rerun_logger.h:117`) | one occurrence each | high | remove |
| Members never read or written | `room_concept.h:1305 prev_lidar_ts_for_log_`, `:1308 slot_sdf_mse_str_`, `:1321 trans_noise_sample_buf_`; `room_scene_graph.h:415 sweep_last_ms_`; `specificworker.h:121 last_imu_sim_ts_`, `:145 last_image_edge_ms_`, `:201 self_target_active_`; `viewer_2d.h:304 capture_vertex_items_`, `:346 polygon_item_backup_`, `:353 furniture_draw_items_`, `:356 temp_obstacle_draw_items_`; `room_viewer.h:170 ts_plot_gt_`; `room_boxes_channel.h:220 refine_stall_`, `:428 init_scan_`; `epistemic_planner.h:565 rng_` | the identifier occurs once in `src/` + `tools/` (comments stripped) | high | remove |
| Config values read into a field nobody reads | `PredictPublish.enabled` → `PREDICT_PUBLISH_ENABLED` (`room_config.cpp:191`, `room_config.h:317`); `DSR.BootstrapTableEnabled` → `BOOTSTRAP_TABLE_ENABLED` (`room_config.cpp:1002`, `room_config.h:341`) | `git grep -nw` → loader + declaration only; both absent from `etc/config.toml` | high | remove |
| `Media.lidar_use_media` branch | `lidar_ingestor.cpp:53,277` | the false branch only logs "the DSR-graph path was removed — no LiDAR source" | high | remove the flag |
| Test programs living in `src/`, not in any CMake target | `src/calib_channel_test.cpp` (361), `src/calib_pivot_test.cpp` (97) | basename absent from every `CMakeLists.txt` and script | high that they are not built; **ask** whether to keep | move to `tools/` and add an `EXCLUDE_FROM_ALL` target, or delete |
| Tests of code that no longer exists here | `tools/tests/{lift.py,vb.h,test_vertex_offset_nuisance.cpp,README.md}` | they lift code out of `src/mount_lidar_pair.h`; `git ls-files \| grep mount_lidar_pair` is empty (mount calibration moved to `self_calibration`) | high | remove |
| Tools with no CMake target and no build line | `tests/lidar_mount_edge_live_test.cpp` (137), `tools/calib_estimator_selftest.cpp`, `calib_intake_selftest.cpp`, `proto_mouth_graph_test.cpp`, `reloc_proposer_test.cpp` | not in `src/CMakeLists.txt`; no `g++` line in the header. The two calib selftests are built and run by `tools/refactor_oracle.sh` itself (its standalone `g++` loop, lines 47 and 76) — **keep those two**. The other three have no builder that I could find | ask (for the three) | give each a CMake target or delete |
| Stale script reference | `estimation_scripts/scripts/run_crossing.sh:24` copies `tmp/reloc_doorway.csv`, `reloc_doorway_points.csv` | no writer in `room_concept/src` | high | drop from the `cp` |
| Tracked config backups | `etc/config_p3bot.toml.bak.{gn-4modes,pre-ee,pre-noEE,pre-preint}`, `etc/camera_calib*.PRE-ROBOT-KEY`, `etc/camera_calib.txt.MIXED-zed-ricoh-DISCARD` | tracked, referenced by nothing | medium | ask, then remove |

### 2b. Gated branches not taken in the live config

Evidence for every row: the value in `room_concept/etc/config.toml`, and
`git grep -nE "sed -i" -- room_concept/estimation_scripts/scripts` — the tour scripts edit exactly one
room_concept key, `MapMode` (`explore_tour.sh:21` and `explore_spike.sh:17` → `estimate`,
`warm_start_tour.sh:26` → `ltsm`; the committed value is `given`). **All three `MapMode` values are live.**
`StartupProto.Enabled`, `ProtoRoom.Enabled`, `ScanTape.Enabled` are `true` and live.

| Flag (live value) | Code behind it | Evidence / history | Action |
|---|---|---|---|
| `RoomShape.BoxLayout = false` | `room_boxes.{h,cpp}` (284 + 1,297), `room_boxes_channel.{h,cpp}` (440 + 2,368), call sites `room_concept_wall_slam.cpp:171-180`, `room_concept_wall_slam_after.cpp:245-286,390`, `room_concept.cpp:517-519`, 11 `RoomShape.Box*` keys | config comment (line 1157): "the wall map is the shipped estimator and the box layout is the A/B". Exercised by the bench (`WS_BOXES=1`, oracle cases `wss_box_room0/2/4`) | **kept by owner decision, confirm.** Decide before splitting: there is no point cutting 3,665 lines into six files if they are to be retired |
| `ObjectAnchor.enable = false` (since 09-17) | `object_anchor_{source,factor,types}.*` (511), `RoomSceneGraph::refresh_object_anchors` (128) + `log_table_landmarks` (112), `Viewer2D::draw_object_anchors` (~190), `Se2Landmark`/`LandmarkPrior` factors in `room_gn_solver.cpp`, 23 `ObjectAnchor.*` keys | comment: "OFF for layout-estimation tests: stale anchors from other rooms can pull the pose" | ask |
| `RoomConcept.RerunEnabled = false` | `rerun_logger.{h,cpp}` (402, own sender thread), `room_concept.cpp:187,342` | comment: "OFF: elimina ~3 ms/frame" | ask (likely remove) |
| `EpistemicController.ModeDisambiguation = false` | `EpistemicPlanner::mode_disambiguation_nats` (36) + `epistemic_planner.cpp:415-425` | "OFF until run live"; used by `reloc_selftest` section G | ask |
| `RoomConcept.MotionNoiseLearn = false` | `motion_noise_vc.h` learner, `room_concept_early_exit.cpp:78` | "OFF: live window posteriors break its assumption"; has its own selftest | ask |
| `RoomConcept.PreintZuptAsFactor = false` | `room_concept_update.cpp:225`, `se2_preintegration.h:226,400`, `window_types.h:32` | used by `gn_selftest.cpp:879` | ask |
| `RoomConcept.PredictedTicks = false` | `pose_publisher.cpp:571` ("mode b") | never flipped | ask |
| `RoomConcept.PreserveBootstrapRoom = false` | `pose_publisher.cpp:245,548`, `room_scene_graph.cpp:1575,1646`, `specificworker_presence.cpp:83`, `specificworker_startup.cpp:456` | "static-room phase" of an older design | ask (likely remove) |
| `RoomConcept.CalibPivotEnabled` (absent, default false) | `room_scene_graph.cpp:3015`, `calib_pivot.h` | — | ask |
| `RoomConcept.UseCommandVelocityPrior` (absent, default false) | `room_concept.cpp:48`, `room_concept_motion_prior.cpp:181-185` | — | ask |
| `RoomConcept.IncidenceAngleWeight` (absent, default false) | `room_obs_weights.cpp:61` | set true only in `gn_selftest.cpp:438` | ask |
| `RoomConcept.OdomSampleLog` (absent, default false) | `specificworker_slots.cpp:160` → `etc/odom_samples.csv` | reader exists: `tools/odom_whiteness.py` | keep |
| `RoomShape.WallLineFactor = false`, `AbsorbSchur = false`, `CommonModeGate = false`, `FreezeLayoutWhenPublishable = false`, `SpanKillRespectsSupport` (absent, false), `BoxConnected = false`, `BoxConnectedPrice` (absent, false) | `room_gn_solver.cpp:789,814,1080`; `wall_map.cpp:932,2284`; `room_concept_wall_slam_after.cpp:110`; `room_boxes_channel.cpp:1868-1870` | each is a bench arm (`WS_LINE`, `WS_NO_SCHUR`, `WS_CMODE`, `WS_CONNECT_PRICE` in `wall_slam_selftest.cpp`) | ask, one at a time |
| **Legacy arm of a settled A/B** — live value `true`, the `false` branch is the old behaviour: `AdaptiveCovEnabled`, `BoundaryFejSchur`, `HeadingFusion`, `HeadingNoiseLearning`, `MotionPreintegration`, `MotionNoiseProportional`, `PreintZuptOnPrediction`, `PreintZuptPredLearnRest`, `WindowStrideEnabled`, `SdfCurrentSlotOnly`, `ImuLinearInjection`, `OdomVarianceInjection`, `FarPointsWeight`, `SdfPolishOnEarlyExit`, `CornerEarlyExitCheck`, `HierPrecBoundaryEnabled` | spread over `room_concept_*.cpp`, `se2_preintegration.h`, `rest_density_learner.h` | The code defaults in `localizer_params.h` are the OLD values for most of them, and `reloc_selftest`/`gn_selftest` construct default `Params`. So the selftests run the legacy arms, not the live ones (the 10-17 note says the same: "the oracle does not cover live config") | **ask, and a prerequisite:** flip the code defaults to the live values and re-record the oracle first. Removing a legacy arm before that silently changes what the selftests test |

### 2c. Debug and diagnostic code

No `#if 0`, no `FIXME/HACK/XXX` in either agent. Two `TODO`s, both design notes
(`layout_estimator.h:26`, `room_concept.h:315`). Commented-out code: one block, `corner_detector.h:221-249`.

Diagnostic files written by room_concept. "Reader" = a tracked script or tool that names the file.

| File | Writer | Reader | Action |
|---|---|---|---|
| `tmp/proto_room.csv`, `tmp/map_layout.csv`, `tmp/promotion_select.csv`, `tmp/layout_refine.csv`, `tmp/refine_input.bin`, `tmp/slam_shadow.csv`, `tmp/door_filter.csv`, `tmp/sdf_localizer/gt_error.csv`, `etc/scan_tape_*.bin` | `room_scene_graph.cpp:1340,2164,2244,4835,4808`; `room_concept_wall_slam.cpp:418`; `specificworker.cpp:198`; `ground_truth_log.cpp:107`; `scan_tape.h` | `run_crossing.sh:24-27`, `grade_explore_tour.py`, `grade_campaign.py`, `regrade_anchor.py`, `reloc_tape_replay` | **keep — the tour/grading contract** |
| `etc/room_name_hwm.txt`, `etc/last_robot_pose.txt` | `room_scene_graph.cpp:1709`, `specificworker.cpp:536` | every tour script | keep (state, not diagnostics) |
| `tmp/layout_trace.csv`, `tmp/wall_geometry.csv`, `tmp/corner_probe.csv`, `tmp/corner_nis.csv`, `etc/pose_trace.csv`, `tmp/noise_innov/innov_*.csv`, `tmp/wall_input_*.bin`, `epistemic_select.jsonl`, `etc/odom_samples.csv` | `room_concept_wall_slam_after.cpp:329`, `room_concept_wall_slam.cpp:366,120`, `room_concept_update.cpp:420,459`, `pose_publisher.cpp:128`, `room_concept_early_exit.cpp:90`, `epistemic_planner.cpp:711` | `tools/score_layout_vs_gt.py`, `room2_analyze.py`, `analysis/corner_channel/*.py`, `noise_innov_*`, `scan_slam_replay`, `common/affordance_protocol/replay_selection.py` | keep |
| `etc/corner_stats.csv` | `room_concept_diagnostics.cpp:245` | no script, but the owner's notes say to read it before diagnosing a corner | keep |
| `tmp/sdf_localizer/log_*.csv` (`DebugLog = true`), `recovery_*.csv`, `flips_*.csv` | `room_concept_diagnostics.cpp:47,173`, `room_concept_update_solve.cpp:340` | `tools/nees_report.py`, `gt_analyze.py` and others read the `sdf_localizer` directory | keep; ask about `flips_*` |
| `etc/hessian_check.csv` (`HessianCheck = true`), `etc/hier_prec.csv`, `etc/optimizer_timing.csv`, `etc/publish_timing.csv`, `etc/resource_usage.csv`, `etc/pose_jumps.csv`, `etc/compute_timing.csv`, `tmp/startup_timing.csv`, `etc/wall_slam.csv`, `etc/wall_slam_events.csv`, `tmp/wall_residual.csv`, `tmp/calib_closures.csv`, `tmp/calib_producer.log`, `tmp/refine_free_origin.bin`, `tmp/crossing.csv`, `etc/image_edge.csv`, `etc/image_edge_triple_*.csv`, `etc/image_edge_triple_pose.csv`, `tmp/sdf_localizer/table_landmark_*.csv`, `etc/box_layout.csv` | `room_concept_diagnostics.cpp:288,341`, `room_concept.cpp:325`, `pose_publisher.cpp:80,205,520`, `specificworker.cpp:375`, `specificworker_startup.cpp:221`, `room_viewer.cpp:94`, `room_concept_wall_slam*.cpp`, `room_scene_graph.cpp:3069,3160,4016,4820`, `door_crossing.h:534`, `room_concept_image_edge.cpp:221,295,456`, `room_boxes_channel.cpp:2122` | **no tracked reader** (`git grep -lF <stem> -- '*.py' '*.sh' '*/tools/*'` is empty for each) | **ask.** Several are per-frame on the localiser thread; each one removed is rate given back. `HessianCheck = true` in the live config also runs a numerical check, not only a write |

Other:

- `room_concept.h` holds 21 `std::ofstream` members, `room_scene_graph.h` 8. Each re-implements open /
  `imbue(classic)` / header / flush. Consolidating them is in the plan (R9).
- Logging is mixed: 244 `qInfo/qWarning/qDebug` lines and 34 `std::print/printf/cout` lines. The tour scripts
  `grep` the log (`warm_start_tour.sh:78-86` looks for `MapMode=ltsm: layout of`); log text is part of the
  test contract and must not change during the refactor.
- `&&`/`||` occur on 242 lines (CLAUDE.md asks for `and`/`or`). Fix inside files as they are moved, not as a
  separate sweep.

### 2d. Files over 800 lines

| Lines | File | Lines | File |
|---|---|---|---|
| 4,873 | `src/room_scene_graph.cpp` | 1,173 | `src/free_space_polygon.h` |
| 4,040 | `src/wall_map.cpp` | 1,009 | `src/corner_detector.cpp` |
| 2,368 | `src/room_boxes_channel.cpp` | 975 | `src/wall_map.h` |
| 1,974 | `src/viewer_2d.cpp` | 911 | `src/room_viewer.cpp` |
| 1,562 | `src/room_concept.h` | 878 | `src/epistemic_planner.cpp` |
| 1,544 | `src/camera_visualizer.cpp` | 873 | `src/layout_refiner.h` |
| 1,383 | `src/room_gn_solver.cpp` | 5,471 | `tools/wall_slam_selftest.cpp` |
| 1,321 | `src/reloc_search.cpp` | 1,425 | `tools/reloc_selftest.cpp` |
| 1,297 | `src/room_boxes.cpp` | 1,315 | `tools/gn_selftest.cpp` |
| 1,278 | `src/room_config.cpp` | 938 | `tools/reloc_tape_replay.cpp` |

Python over 800: `estimation_scripts/scripts/protocol_mock.py` 1,474, `adoption_mock.py` 1,318,
`tools/room_tui.py` 1,167. Close to the limit: `room_concept_early_exit.cpp` 758, `specificworker_startup.cpp` 738.

Functions over 300 lines (from `tools/lint_function_lengths.py`): `load_room_config` 1,058,
`Channel::plan_path` 1,023, `CornerDetector::detect` 912, `WallMap::observe` 722, `re_derive` 669,
`try_splice` 641, `CameraVisualizer::draw_projections` 600, `SpecificWorker::initialize` 523,
`ImageEdgeSource::extract` 486, `write_robot_room_rt` 452, `dsr_update_affordance` 384,
`Channel::rebuild_from_free` 378, `Viewer2D::draw_wall_map` 378, `wall_slam_after_solve` 355,
`dsr_update_layout_affordance` 350, `RoomViewer::RoomViewer` 324, `run_image_edge_shadow` 322,
`RoomConcept::run` 313, `dsr_update_calibration` 312, `mount_pooled_solve` 302.

---

## 3. controller findings

### 3a. Dead code

| Item | Where | Evidence | Conf. | Action |
|---|---|---|---|---|
| **The measured doorway gap** | `src/doorway_gap.h` (567); `ControllerSession::update_doorway_gap` (`controller_session.cpp:4891`, 84); members `controller_session.h:1382-1425` (`gap_line_`, `gap_width_`, `gap_score_`, `gap_carry_pose_`, `gap_room_yaw_fix_`, …); 19 `on_gap` uses in `cross_protocol` | `controller_session.cpp:5052`: `const bool on_gap = false;` and the comment above it: "on_gap is false on every path below; the tracker's code (update_doorway_gap, gap_line_) is left for the controller clean-up to delete" (owner ruling 2026-10-31). `git grep -nw update_doorway_gap` → decl + def + comments. `doorway_gap.h` is included only by `controller_session.h` and two tests | high | remove. `local_policy/doorway_corridor.h` and `door_concept/src/door_opening_edges.h` mention it in comments only |
| Tests of that dead chain | `tests/test_doorway_gap.cpp` (554), `tests/test_jamb_cross.cpp` (392, a closed loop of "the session's cross chain with update_doorway_gap") | they include `../src/doorway_gap.h` | high | remove with it |
| `ControllerSession::band_cross_doorway_risk` + member `band_cross_risk_` | `controller_session.cpp:5255` (19), `controller_session.h:1475,1478` | decl + def only; the member occurs once | high | remove |
| `ControllerSession::anchor_clearance_speed_limit` | `controller_session.cpp:4807` (16), `.h:124` | decl + def + one comment in `lidar_body_guard.h:11` | high | remove |
| `ControllerObstacleTracker::scan_for_unmodelled_obstacles` | `controller_obstacle_tracker.cpp:1632` (117), `.h:202`, member `last_scan_ms_` | the only call is commented out (`controller_session.cpp:7237`): moved to `residual_concept` | high | remove |
| Defined, never called | `GridPlanner::why_cannot_turn` (`grid_planner.cpp:719`; `why_cannot_sweep` is the live one), `TrajectoryController::current_waypoint_room` (`trajectory_controller.cpp:366`), `refresh_lidar_field` (`:490`), `Navigator::downhill` (`local_policy/navigator.cpp:50`, 20), `ControllerObstacleTracker::object_label_prefix` (`controller_obstacle_tracker.cpp:120`), `ControllerSession::live_tracking` (`controller_session.cpp:8323`, 22 — its sampler `sample_live_tracking` IS called, so the accumulator runs every cycle for nobody) | decl + def only | high (medium for `live_tracking`: confirm no panel reads it) | remove |
| Unused inline accessors | `ControllerSession::affordance_view` (`controller_session.h:591`), `route_reverse` (`:280`); `TrajectoryController::align_yaw_tol`, `current_waypoint_index`, `get_path`, `esdf_model_points_dropped` (`trajectory_controller.h:113,218,219,250`); `MissionRunner::current_lap`, `has_click_target` (`controller_mission.h:391,359`); `GridPlanner::has_distance_field` (`grid_planner.h:371`); `RouteFailureMemory::has_record`; `ExplorationDrive::is_failed` (`exploration_drive.h:190`); `GoalAdapter::last_llr`, `last_p_correction`, `probe_llr` (`local_policy/goal_adapter.h:401-404`); `near_radius` (`near_field_clearance.h:107`); `ReachabilityBelief::p_unreachable` (`:165`); `Navigator::reachability` (`navigator.h:118`); `DoorPassage::exited` (`controller_door_passage.h:349`) | one occurrence each | high | remove |
| Members never used | `controller_session.h:691 reversal_captured_`, `:1046 last_carrot_collapse_log_ms_`, `:1194 aif_trace_next_ms_`, `:1213 returns_backoff_logged_`, `:1215 anchor_census_for_`, `:1216 anchor_band_logged_for_`, `:1282 anchor_servo_last_ms_`; `controller_obstacle_tracker.h:434-435` two `*_debug_report_` strings; `trackers/plain_tracker.h:99 reacquire_` | identifier occurs once in `src/` + `tools/` + `tests/` | high | remove |
| `MissionRunner::self_test` | `controller_mission.cpp:1284` (171) | no caller anywhere, including `tools/route_bench.cpp --self-test` | high | remove, or wire into `route_bench` |
| **Self-tests compiled into the agent** | `GridPlanner::self_test` (398), `route_optimizer_self_test` (525), `RouteFollower::self_test` (522), `RouteSpline::self_test` (206), `PlainTracker::self_test` (147) | called only from `tools/route_bench.cpp:399-406` and `tests/test_grid_planner_curtain.cpp:75` | high | **move** to `*_selftest.cpp`, linked by the tools only. Takes 1,798 lines out of the agent build and puts `route_follower.cpp` (590) and `route_spline.cpp` (406) under the limit |
| Migration monitor, self-declared temporary | `GridPlanner::pose_free_legacy`, `cell_free_legacy`, `orientation_census`, `offsets_legacy_` (`grid_planner.h:242-256,400,450`; `.cpp:48,65-120`); `ControllerSession::monitor_footprint_orientation` (`controller_session.cpp:5519`, 45, called every planning step at `:3257`) | header: "MIGRATION MONITOR (temporary; delete once the yaw correction is trusted)"; `controller_session.h:1065`: "delete with GridPlanner::pose_free_legacy" | medium | ask: is the yaw correction trusted? |
| A test that cannot compile | `tests/bench_c1_door.cpp` (516) | includes `../src/door_approach_law.h` and `../src/door_cross_aif.h`; neither exists (`git ls-files \| grep -E "door_approach_law\|door_cross_aif"` is empty) | high | remove or port |
| Config key that is never read | `etc/config.toml:927`, `[Transforms] TrackerUsesLatestPose = true` | the code reads `Controller.TrackerUsesLatestPose` (`specificworker.cpp:1074`); the file's value is ignored, the default `true` applies | high | move the key to `[Controller]` (same value, no behaviour change) |
| Leftovers of a regeneration, untracked | `src/specificworker.cpp.new`, `src/specificworker.h.new`, `src/CMakeLists.txt.new`, `etc/config.toml.new`, `src/build/`, `build_lateral_probe/`, `build-make-copilot/`, `build_test_migration/`, `CMakeCache.txt` in the source root | ignored by `.gitignore` | high | delete from the working tree |

### 3b. Gated branches

No tour script edits a controller key: `git grep -n "sed -i" -- room_concept/estimation_scripts/scripts`
touches only `room_concept` and `ltsm_agent` configs (`explore_tour.sh:72` only reads `MaxRoomS`).
The 2-D mock (`tests/test_policy_scenarios.cpp`, `REGIME=policy|chain|hybrid`) exercises the regimes offline.

| Flag (live value) | Code behind it | Evidence | Action |
|---|---|---|---|
| `Policy.Enabled = true` → the old chain is off | everything guarded by `not policy_enabled()`: escape (`detect_stuck`, `begin_escape`, `step_escape`, ~330), `step_returns_backoff`, the anchor servo (`controller_anchor_servo.h` 885 + `update_cross_anchor`, `note_anchor_block`), lock-on/orient motion, `step_route_band` (266), the clearance ceiling, the tracker's command path, the uncertainty limiter | `controller_session.h:198-202`: "The old chain is switched OFF, not deleted"; owner note 2026-10-21 ("keep the spline tracker until A/B") and 2026-11-10 (two regimes: chain for traverses, policy for tight spaces) | **kept by owner decision, confirm.** The plan gives each regime its own file instead of deleting either |
| `Policy.HybridChain` (absent, default false) | `controller_chain_hybrid.cpp` (255), `controller_chain_monitor.cpp` (232), `chain_monitor/` | the port of the owner's two-regime design, 2026-11-11; not live yet | keep (work in progress) |
| `ChainMonitor.MarksCarry` (absent, default false) | `controller_chain_monitor.cpp:80` | "off: no live noise model yet" | keep |
| `Route.NearFieldClearance = false`, `Route.BandDrivesCross = false` | `near_field_clearance.{h,cpp}` (458), `tests/test_near_field_clearance.cpp` (652), `controller_session.cpp:4074,5174,6701`, `band_cross.h` | config comment: arm B of the three-arm A/B (Task 6, 10-21); `band_drives_cross()` also requires `not policy_enabled`. Owner 11-10: near-field clearance "is a mitigation, not a fix" | **ask** — arm B looks abandoned; arm A (chain) and C (policy) are the two regimes |
| `Controller.ObstacleCreationEnabled = false` (since 2026-07-29) | `refresh_temporary_lidar_obstacle`, `create_temporary_lidar_obstacle`, `add_virtual_obstacle` and the temporary-obstacle lifecycle in `controller_obstacle_tracker.cpp` (≈ lines 36-794 and 1326-1500), ~30 `Controller.TemporaryObstacle*` keys loaded at `specificworker.cpp:1167-1218`; call sites `controller_session.cpp:7228,7975,9424` | comment: "residual_concept is now the sole obstacle source. Set back to true if it is down" | **ask** — it is a stated fallback, off for 3.5 months |
| `Controller.ControlMode = "plain"` | `trackers/pd_tracker.cpp` (206) | not dead: PD is PLAIN's fallback when there is no fitted curve (`trajectory_controller.cpp:725-766`) | keep |
| `LockOnEnabled`, `StuckRecoveryEnabled`, `BandEnabled`, `RouteContinuous`, `RouteOptimize`, `ProximityLogEnabled`, `DoorPassage.UseRoute` — all `true` | — | live | keep; the `false` arms are small |

### 3c. Debug and diagnostic code

The controller writes its diagnostics into its working directory, unconditionally, truncating at start.

| File | Writer | Reader | Action |
|---|---|---|---|
| `policy_field_dump_<ms>.txt` | `controller_policy_drive.cpp:358-385` — one file per new goal whose field exceeds 10× the straight line | none. **3,385 files (14 MB) are sitting in `controller/`.** Two were hand-copied into test comments (`tests/test_field_extent.cpp:6`, `test_policy_scenarios.cpp:287,491`) | **remove**, or write to one rotating file under `etc/runs/`. The 10× test is a threshold in a diagnostic, not in the model |
| `etc/exploration_drive.csv` | `specificworker_mission.cpp:918` | `explore_tour.sh:86`, `grade_explore_tour.py`, `door_concept/tools/door_costs_from_logs.py` | keep (tour contract) |
| `policy_decisions.csv` | `controller_policy_drive.cpp:630` | `estimation_scripts/scripts/ab_report.py` | keep |
| `tracker_diag.csv`, `approach_diag.csv`, `band_diag.csv`, `stall_events.csv`, `proximity_obstacles.csv`, `affordance_select.jsonl`, `standpoint_audit.jsonl`, `route_world.txt`, `route_geometry.csv`, `unreachable_world.txt`, `mission_metrics.csv`, `*_profile.csv`, `base_probe.csv`, `lidar_z_dump.csv` (armed by remote `zdump`) | `controller_session.cpp:2787,5351,6834,9290,7888,832,677,2738,2717,1997`; `controller_mission.cpp`; `specificworker.cpp:2557`; `controller_obstacle_tracker.cpp:2262` | `tools/lap_metrics.py`, `dead_columns.py`, `path_departure_monitor.py`, `approach_probe.py`, `standpoint_audit.py`, `adapt_L.py`, `measure_deadband.py`, `floor_band.py`, `route_bench`, `tracker_sim`, `replay_world`, `scripts/reversal_map.py`, `common/affordance_protocol/watch_dialogue.py`, `anchor_vs_truth.py` | keep |
| `cloud_wall_fit.csv` | `log_cloud_wall_fit` (`controller_session.cpp:2463`, 195 lines, 5 Hz; its own comment: "this searches, so it is not free") | none | **ask → remove.** A concluded investigation that still costs time in the loop |
| `residual_world.csv` | `log_residual_world` (`:2659`, 53, 5 Hz) | none | ask → remove |
| `overlay_lag_eval.csv`, `overlay_pairing.csv` | `update_overlay_extrapolation` (`:8533`, 166), `:1000` | none | ask. Careful: the extrapolation may also feed the display; only the CSV part is diagnostic |
| `plan_track.csv` | `:7450` (per cycle) | none | ask |
| `finalize_sites.jsonl` | `:8873` | none | ask |
| `base_commands.jsonl` | `controller_motion_commander.cpp:202` (one line per command) | none | ask |
| `route_events.csv`, `door_passage_world.txt`, `affordance_protocol.log` | `:2886`, `:4284`, `:6165` | none | ask |
| `chain_monitor.csv` | `controller_chain_monitor.cpp:30` | none | keep while the hybrid is being ported |
| `hold_events.csv`, `why_log.jsonl` | `specificworker.cpp:1521,1924` | none | ask |
| `canvas_health.csv`, `window_geometry.log` | `controller_display.cpp:681,35` | none; `log_geometry` is marked "DIAGNOSTIC (temporary)" | remove |

Other:

- 425 `std::print/println` lines against 39 Qt logging lines. `explore_tour.sh:62` and `warm_start_tour.sh`
  read `controller.log` to learn that a leg ended: the text of those lines is a contract.
- `controller_session.cpp` is 29 % comment-only lines (2,785), `controller_session.h` 40 %. Much of it is
  dated investigation history ("★★★MEASURED 2026-08-24 …", "SEVEN HYPOTHESES HAVE DIED HERE"). It is
  valuable, but it is history, not interface documentation. Moving it to a design note is the cheapest
  size reduction available and changes no code. **Ask** — the owner may want it next to the code.
- `&&`/`||` on 165 lines.
- `controller_mission.cpp:265` parses with `std::stoi` inside `try/catch`. Integers are locale-safe, but
  `std::from_chars` is the house rule; fix when the file is split.

### 3d. Files over 800 lines

| Lines | File | Lines | File |
|---|---|---|---|
| 9,567 | `src/controller_session.cpp` | 1,222 | `src/specificworker_mission.cpp` |
| 2,579 | `src/specificworker.cpp` | 1,206 | `src/controller_policy_drive.cpp` |
| 2,407 | `src/controller_obstacle_tracker.cpp` | 1,112 | `src/route_follower.cpp` |
| 1,630 | `src/controller_session.h` | 888 | `src/local_policy/local_policy.cpp` |
| 1,574 | `src/grid_planner.cpp` | 885 | `src/controller_anchor_servo.h` |
| 1,545 | `src/controller_door_passage.h` | 831 | `src/controller_display.cpp` |
| 1,456 | `src/controller_mission.cpp` | 2,304 | `tests/test_policy_scenarios.cpp` |
| 1,374 | `src/trajectory_controller.cpp` | 1,415 | `tests/test_door_passage.cpp` |
| 1,348 | `src/route_optimizer.cpp` | 1,203 | `tools/tracker_sim.cpp` |
| 1,344 | `src/viewer_2d.cpp` | 813 | `tests/test_anchor_servo.cpp` |

Functions over 300 lines: `build_planning_step` 1,388, `execute_plan` 1,074, `SpecificWorker::initialize` 570,
`drive_with_policy` 546, `policy_tick` 532, `sync_world_state` 528, `route_optimizer_self_test` 525,
`RouteFollower::self_test` 522, `load_params` 490, `read_obstacle_polygons` 475, `local_policy decide` 462,
`handle_lidar_points` 406, `GridPlanner::self_test` 398, `drive_point_target` 331, `assemble` 326,
`update_affordance_view` 321, `optimize_route` 319, `ControllerDisplay::present` 305.

---

## 4. Shared `common/` findings

Only the directories whose users are these two agents and nobody else (include scan over every tracked
source and CMake file):

| Directory | Users | Finding |
|---|---|---|
| `common/motion_calib/` (915 lines) | room_concept (`calib_channel.h:32`, `calib_pivot.h:34` include `scale_estimator.h`) | `closure_pivot.cpp`, `scale_replay.cpp`, `scale_estimator_test.cpp` are stand-alone programs with no CMake target; referenced by nothing. **ask** |
| `common/status_stream/` (882) | room_concept | live (`Status.Enable = true`, `tools/room_tui.py`). No dead code found |
| `common/robust_metrics/robust_metrics.{h,cpp}` (160) | controller | `density_squared`, `posterior_mean`, `robust_loss_type_name` have no caller outside the directory. Medium confidence. `robust_metrics_torch.*` belongs to `bottle_concept/tests`, not to these agents |
| `common/robot_capability/` (275) | both | live |

Duplicates between the two agents (not in `common/`, but they should be):

- `room_concept/src/timeseries_plot.{h,cpp}` (527) is a divergent copy of `common/dashboard/timeseries_plot.*`
  (466); `src/CMakeLists.txt` says the agent "deliberately does not link `ai_common::ui`". 266 differing lines.
  **ask** whether to merge.
- `corner_visibility.h` exists in both agents (218 and 116 lines, 143 differing lines), and
  `custom_widget.h` in both. Candidates for one shared header each.

---

## 5. Refactoring plan — room_concept

### 5.1 Current structure

- **Agent shell** — `SpecificWorker` in four files: `specificworker.cpp` (624: `compute`, pose file),
  `_startup.cpp` (738: `initialize` is one 523-line function that builds and wires everything),
  `_slots.cpp` (255: queued DSR slots), `_presence.cpp` (155).
- **Estimator** — `rc::RoomConcept`, one class over 15 TUs (`room_concept*.cpp`, 7,900 lines) with a 1,562-line
  header, plus solver (`room_gn_solver`, `room_model`, `room_obs_weights`), relocaliser (`reloc_*`), layout
  (`wall_map`, `wall_segmenter`, `room_boxes*`, `free_space_polygon.h`, `layout_refiner.h`, `layout_explorer`),
  corners and image edges, motion/calibration headers. Built once as the static library `rc_room_estimator`.
- **Graph adapter** — `RoomSceneGraph`, one class in `room_scene_graph.cpp` (4,873) + `_explored.cpp` +
  `_late_recall.cpp`: RT edge, proto-room at start-up and at a door crossing, promotion, recall, three
  affordances, calibration node, walls, landmarks, layout refiner. Plus `pose_publisher`, `proposal_adoption`,
  `door_apertures`, `calib_channels`, `ground_truth_log`, `status_reporter`.
- **Ingest** — `lidar_ingestor`, `imu_ingestor`, `camera_ingestor`, `camera_stream_hub` (own threads, media plane).
- **Planning** — `epistemic_planner` (live) inside `epistemic_controller` (dead shell, see 2a).
- **UI** — `viewer_2d`, `room_viewer`, `camera_visualizer`, `calibration_viewer`, `timeseries_plot`.
- **Config** — `room_config.cpp`: one 1,058-line loader.

### 5.2 Target structure

The layered target of the 10-13 proposal stands (`core / solver / layout / localizer / ui / agent`, the agent
being the only DSR user, `Localizer::step(FrameInputs)` so the bench runs the agent's code). What follows is
the file layout after the **mechanical** phase; classes keep their names and members.

| Now | Target files (approx. lines) | Responsibility |
|---|---|---|
| `room_scene_graph.cpp` 4,873 | `scene_publish.cpp` (update 194-434, absorb_reanchor; 450) · `room_rt_writer.cpp` (`write_robot_room_rt` 495-946 cut into pose / velocity-covariance / ring; 550) · `proto_room_crossing.cpp` (`step_proto_room_impl`, `write_proto_row` 1067-1415; 450) · `startup_proto.cpp` (birth, `find_recall`, `rearm_after_crossing`, arm, step, promote 1632-2391; 760) · `layout_affordance.cpp` (2416-2765; 400) · `explore_affordance.cpp` (`dsr_update_affordance`, `break_execution_stall` 3341-3871; 560) · `calib_affordance.cpp` (`ensure_calib_node`, `dsr_update_calibration`, entry relocalisation 2898-3339; 470) · `room_nodes.cpp` (create/reparent, walls, `sync_walls_to_polygon`, overlays, cleanup 2768-2877 + 4276-4723; 620) · `graph_landmarks.cpp` (table landmarks, object anchors 3923-4165; 280) · `layout_refine_step.cpp` (4772-4871 + `triangulate_polygon`; 220) | one DSR concern per file; still one class |
| `wall_map.cpp` 4,040, `wall_map.h` 975 | as in the 10-13 table (line numbers unchanged since): `wall_map_core.cpp` 750 · `_observe.cpp` 750 · `_splice.cpp` 700 · `_spurs.cpp` 350 · `_rederive.cpp` 800 → cut `re_derive` into trace / rectilinearise / judges · `_polygon.cpp` 500 · `wall_free_grid.cpp` 170; header → `wall_map_params.h` 430 · `wall_map_types.h` 160 · `wall_map.h` 340 | |
| `room_boxes_channel.cpp` 2,368, `room_boxes.cpp` 1,297 | **only if the owner keeps the box layout:** `box_path_planner.cpp` + `box_path_info.cpp` (`plan_path`, 1,023 lines) · `box_channel.cpp` 600 · `box_rebuild.cpp` 650 · `box_geometry.cpp` 420 · `box_register.cpp` 650 | |
| `room_gn_solver.cpp` 1,383 | `gn_factors_scan.cpp` · `gn_factors_pose.cpp` · `gn_factors_wall.cpp` (lines 45-696) · `gn_build.cpp` (`absorb_wall_observations`, `build_factors`; 450) · `gn_solve.cpp` (`newest_pose_marginal`, `solve`; 250) | |
| `reloc_search.cpp` 1,321 | `reloc_beam_model.cpp` (anonymous namespace 21-632: point/beam terms, `beam_pass`) · `reloc_likelihood.cpp` (geometry, raster, exact likelihood 632-872) · `reloc_search.cpp` (yaws, `polish`, `search`, priors 872-1321; 450) | |
| `room_config.cpp` 1,278 | `config_localizer.cpp` · `config_layout.cpp` · `config_imgedge.cpp` · `config_agent.cpp` · `config_planner.cpp` · `room_config.cpp` (overlays, 250) | one loader per key prefix; same keys, same order |
| `free_space_polygon.h` 1,173 | `polygon_ops.h` · `free_space_polygon.h` · `free_space_grid.h` · `raster_ops.h` · `manhattan_layout.h` (457-682) | |
| `layout_refiner.h` 873 | `layout_refiner.h` (class, `adopt_from_pool`, `propose_impl`) · `layout_refiner_candidates.h` (791-866 and helpers) | |
| `corner_detector.cpp` 1,009 | `corner_gather.cpp` 650 · `corner_associate.cpp` 300 | |
| `epistemic_planner.cpp` 878 | `planner_candidates.cpp` 350 · `planner_gain.cpp` 530 | |
| `viewer_2d.cpp` 1,974 | `viewer_2d.cpp` 700 · `_corners.cpp` 380 · `_layout.cpp` 500 · `_objects.cpp` 350 (after deleting the three dead draws) | |
| `camera_visualizer.cpp` 1,544 | `camera_visualizer.cpp` 450 · `camera_feeds.cpp` (DSR fetches, 300) · `camera_overlays.cpp` (`draw_projections` cut at its numbered layers, 650) · `camera_projection.cpp` 200 | |
| `room_viewer.cpp` 911 | `room_viewer.cpp` 500 · `room_viewer_build.cpp` (the 324-line constructor's widget building) | |
| `room_concept.h` 1,562 | `room_concept.h` (class, ~750) + `room_concept_state.h` (the nested state structs: `SearchEpisode`, `CornerVertexStats`, `PredictionState`, `LowObstacles`, `ImageEdgeStats`, …) | 36 % of the file is comment; see the question on comment history. It only drops well under 800 when the first collaborators are extracted (phase B) |
| `specificworker_startup.cpp` 738 | `wiring.cpp` (`initialize` as `make_*` steps in the same order) · `startup_presence.cpp` | |
| `tools/wall_slam_selftest.cpp` 5,471, `reloc_selftest.cpp`, `gn_selftest.cpp`, `reloc_tape_replay.cpp` | one file per bench mode / test section, sharing a `bench_common.h` | last, cannot affect the agent |

Interfaces after phase B (unchanged from the 10-13 proposal): `IObservationChannel` → `gn::IFactor`,
`ILayoutEstimator`, `Localizer::step`, `SceneSnapshot`, `CsvSink`, `LocalizerParams` in nine structs.

### 5.3 Ordered steps

Tests: **O** = `tools/refactor_oracle.sh` (28 offline cases, ~2.5 min, bit-exact except the chaotic ones);
**L** = `make -C build room_lint`; **T1** = one explore tour + `trace_check.py`; **T** = the full series
(6 explore + 4 warm on `three_rooms_door0`) + `trace_check.py`; **R** = nominal rate checked from
`etc/compute_timing.csv` / the status line during a tour.

| # | Step | Kind | Test |
|---|---|---|---|
| R0 | Repair the guardrail: rewrite `tools/lint_allow.txt` from the current tree (it is stale, see §1), add the two unlisted offenders, drop `mount_calibrator.cpp`; make the lint a hard failure in a pre-commit hook. Re-record the oracle on the current tree. Record the live baseline: the 10-tour series on `dbfb77ec` | guard | O green on record; T recorded |
| R1 | Delete what nothing references: the four `.bak` files, `tools/tests/` (lift.py set), the declared-never-defined functions, unused members and accessors of 2a | delete | build; O exact; L |
| R2 | Delete defined-never-called functions of 2a (including the three `Viewer2D` draws and `compute_velocity_adaptive_weights` with its Param and config key) | delete | build; O exact |
| R3 | Remove `EpistemicController` Level 2 and its config reads; keep the class as the thin owner of `EpistemicPlanner` (rename later). Remove the same keys from `etc/config.toml` so the unread-key report stays clean | delete | build; O; **T1** (touches config loading and the scene-graph constructor) |
| R4 | Remove dead config fields (`PREDICT_PUBLISH_ENABLED`, `BOOTSTRAP_TABLE_ENABLED`, `LIDAR_USE_MEDIA`) | delete | build; T1 can be shared with R3 |
| R5 | Move `calib_channel_test.cpp`, `calib_pivot_test.cpp` out of `src/` into `tools/` with `EXCLUDE_FROM_ALL` targets; give the orphan tools a target or delete them | move | they build and pass |
| R6 | **Owner decisions** from §7 (box layout, object anchors, Rerun, unread CSVs, legacy arms). Each accepted removal is its own commit. For the legacy arms: first flip the `Params` defaults to the live values and re-record the oracle, then remove one arm per commit | delete | O re-recorded then exact; T1 per removal that touches the live path; T after the batch |
| R7 | Split `room_scene_graph.cpp` into the ten TUs of 5.2, class unchanged, bodies verbatim. Anonymous-namespace helpers (lines 49-172, 1320-1333, 1450-1459) go to `room_scene_graph_detail.h` as `inline` | mech | build; L; **T** (this file is the tour's whole graph contract: `proto_room.csv`, RT edge, affordances) |
| R8 | Split `wall_map.cpp` / `wall_map.h` | mech | O exact (`wss_*`, `corner_gather_test`); T1 with `MapMode=estimate` |
| R9 | Split `room_gn_solver.cpp`, `reloc_search.cpp`, `corner_detector.cpp`, `epistemic_planner.cpp`, `free_space_polygon.h`, `layout_refiner.h` | mech | O exact (`gn_selftest`, `reloc_selftest`, `free_space_polygon_selftest`, `layout_explorer_selftest`) |
| R10 | Split `room_boxes*.cpp` — or delete, per R6 | mech | O exact (`wss_box_room*`) |
| R11 | Split `room_config.cpp` per prefix. Check: the start-up config table (`etc/config_effective.csv`) is identical before and after | mech | diff of `config_effective.csv`; T1 |
| R12 | Split the UI files (`viewer_2d`, `camera_visualizer`, `room_viewer`) and `specificworker_startup.cpp` | mech | build; start the agent once, windows open; T1 |
| R13 | Cut the functions over 300 lines into named phases, bodies verbatim, the way `update()` was cut on 10-17 (per-function A/B harness, outputs printed with `%a`) | mech | per-function A/B; O; T after the batch |
| R14 | `CsvSink` (open, `imbue(classic)`, header, flush in one place); each owner keeps its sinks | design, small | CSV files byte-identical on an oracle case; T1 |
| R15 | Split the four oversized tools | mech | O exact |
| — | **After R15 every file in room_concept is under 800 lines** except `room_concept.h`, which needs B1 | | |
| B1–B9 | Phase B = steps 7–15 of the 10-13 proposal, unchanged: extract collaborators (`DoorWeighting`, diagnostics, `Relocalizer`, image-edge channel, early-exit gate, window manager, wall-SLAM channel), `MotionCycle`, `Localizer::step`, `IObservationChannel`, `ILayoutEstimator`, agent adapters + `SceneSnapshot`, Params split, tests, delete the facade | design | each as written there; T after every one |

Build memory. `room_scene_graph.cpp` and `wall_map.cpp` are two of the three TUs CLAUDE.md names as the
heavy ones; ten and seven TUs respectively cut the per-`cc1plus` peak roughly in proportion. The risk runs
the other way too: ten TUs that all include `room_scene_graph.h` (DSR + Qt + the estimator headers) parse
those headers ten times, and eight at once under `-j8`. Measure `/usr/bin/time -v` on the largest new TU
after R7, and add `target_precompile_headers` for the DSR/Qt/Eigen set if the parallel peak rises.

Rates. The agent is built `-O3` without LTO, so a helper that was inlined inside one TU becomes a real call
once caller and callee are in different TUs. Keep small hot helpers in a header as `inline`; check R after
R7, R8 and R9.

---

## 6. Refactoring plan — controller

### 6.1 Current structure

- **`SpecificWorker`** — `specificworker.cpp` (2,579): constructor, `initialize` (570), `compute` (206),
  `load_params` (490, ~180 config keys), `control_loop` (262), LiDAR media recovery, mission command
  dispatch, door close, relative motion, chat inbox, base probe. `specificworker_mission.cpp` (1,222):
  `MissionHost`, the journal node, the exploration tour (`service_explore_tour` 243). `_presence.cpp` (77).
- **`ControllerSession`** — one class, 11,260 lines over `controller_session.cpp` (9,567),
  `controller_policy_drive.cpp` (1,206), `controller_chain_monitor.cpp` (232), `controller_chain_hybrid.cpp`
  (255), with a 1,630-line header that includes 35 others. It owns: world sync, target selection and
  standpoint resolution (`build_planning_step`), route building and optimisation, the door route, the point
  target, the anchor cross and the cross protocol, approach commitment and re-check, the affordance panel
  view, the elastic band, arrival and finalisation, `execute_plan` (the per-cycle dispatcher over **three
  drive regimes**), lock-on and orient, stuck detection and escape, and ~20 diagnostic writers.
- **Pure modules** (no Qt/DSR, unit-tested): `grid_planner`, `route_spline`, `route_follower`,
  `route_optimizer`, `trajectory_controller` + `trackers/`, `local_policy/` (policy, contact, cost-to-go,
  navigator, goal adapter, doorway models), `chain_monitor/`, `near_field_clearance`, and ~30 single-purpose
  headers (`stall_judge.h`, `motion_verdict.h`, `door_route.h`, `lidar_body_guard.h`, …).
- **Header-only classes that are really modules**: `controller_door_passage.h` (1,545, class `DoorPassage`),
  `controller_anchor_servo.h` (885).
- **Graph/IO**: `controller_world_model`, `controller_obstacle_tracker` (2,407: graph obstacles, temporary
  obstacles, LiDAR handling, RT twist checks), `controller_motion_commander`, `controller_camera_masks`,
  `mission_node`, `journal_node`, `chat_inbox`, `controller_robot_body`.
- **UI**: `controller_display`, `viewer_2d`, `controller_mission_panel`, status snapshot/text.

### 6.2 Target structure

**Phase A (mechanical, class unchanged)** — `controller_session.cpp` becomes one TU per responsibility:

| New file | Takes (current lines) | ~Lines |
|---|---|---|
| `session_world_sync.cpp` | `sync_world_state` 123-650, `observe_door_passage_why`, selection protocol notes 720-866 | 700 |
| `session_target.cpp`, `session_standpoint.cpp`, `session_plan_build.cpp` | `build_planning_step` 868-2255, cut at its own section comments into three phase functions | 3 × ~480 |
| `session_route.cpp` | optimizer config, `smooth_plan`, `build_route`, speed limits, `ensure_current_plan`, `drive_mission_route` 2321-2455, 2925-3519 | 730 |
| `session_door_route.cpp` | `build_door_route`, `update_chain_end`, `drive_door_route` 3541-4028 | 490 |
| `session_point_target.cpp` | `drive_point_target` 4033-4363 | 340 |
| `session_anchor_cross.cpp` | anchor reading and bookkeeping, `apply_anchor_standpoint`, `cross_hand_back`, `cross_protocol` 4375-5235 (minus the dead gap code) | 700 |
| `session_approach.cpp` | commitment, `recheck_standpoint_on_approach`, `resolve_target_contract` 5568-6116 | 560 |
| `session_affordance_view.cpp` | `note_protocol`, `update_affordance_view`, `update_display` 6147-6549 | 400 |
| `session_band.cpp` | `step_route_band`, band CSV 6564-6860 | 300 |
| `session_arrival.cpp` | unreachable reports, `consume_goal_reached`, `finalize_reached`, `skip_current_affordance`, `set_manual_target` 6883-7205, 8849-9133 | 620 |
| `session_execute.cpp`, `session_execute_chain.cpp` | `execute_plan` 7207-8280: the dispatcher and shared protocol / the old chain's tracker path with its proximity log and stuck checks | 500 + 580 |
| `session_tracking.cpp` | live tracking sample, dwell hits, servo reading, base speed, overlay extrapolation 8287-8698 | 420 |
| `session_lockon_orient.cpp` | `step_lockon`, `step_orient` 8725-8847 | 130 |
| `session_escape.cpp` | `detect_stuck`, `begin_escape`, `step_escape`, stall log 9165-9534 | 380 |
| `session_diagnostics.cpp` | the log writers that survive 3c | ≤ 600 |
| `policy_tick.cpp`, `policy_drive.cpp` | `controller_policy_drive.cpp` split at 660 | 660 + 550 |

`controller_session.h` (1,630): move the investigation history to a design note; move nested types to
`controller_session_types.h`; group members into per-responsibility state structs in their own headers
(`AnchorCrossState`, `BandState`, `EscapeState`, `PolicyDriveState`, `ApproachState`, `SessionDiagSinks`),
each included by `controller_session.h`. Target ≤ 600.

Other files:

| Now | Target |
|---|---|
| `specificworker.cpp` 2,579 | `specificworker.cpp` (ctor, `compute`, `control_loop`; 650) · `specificworker_init.cpp` (`initialize` as `make_*` steps; 600) · `controller_config.cpp` (`load_params` as a free function per section, 500) · `specificworker_commands.cpp` (mission dispatch, door close, relative motion, chat inbox, base probe, abort; 600) · `specificworker_lidar.cpp` (media poll/recover; 150) |
| `specificworker_mission.cpp` 1,222 | `mission_host.cpp` (500) · `specificworker_journal.cpp` (250) · `explore_tour_service.cpp` (450) |
| `controller_obstacle_tracker.cpp` 2,407 | `obstacle_tracker_graph.cpp` (`read_obstacle_polygons` cut into phases; 600) · `obstacle_tracker_temporary.cpp` (temporary lifecycle; 750 — or deleted, per the owner) · `obstacle_tracker_lidar.cpp` (`handle_lidar_points`; 450) · `obstacle_tracker_rt.cpp` (RT lead, twist checks; 250) · `obstacle_tracker_diag.cpp` (150) |
| `controller_door_passage.h` 1,545 | `controller_door_passage.h` (interface, 450) · `controller_door_passage.cpp` (step logic, 750) · `door_passage_view.h` (`View`, `State`, `Pre`; 250) |
| `controller_anchor_servo.h` 885 | `anchor_cross_geometry.h` (to line 478) · `anchor_jamb_guard.h` (479-885) |
| `grid_planner.cpp` 1,574 | `grid_planner.cpp` (world, footprint offsets, queries; 700) · `grid_planner_search.cpp` (`plan`; 450) · `grid_planner_selftest.cpp` (tools only) |
| `route_optimizer.cpp` 1,348 | `route_optimizer_assemble.cpp` (500) · `route_optimizer.cpp` (330) · `route_optimizer_selftest.cpp` (tools only) |
| `route_follower.cpp` 1,112, `route_spline.cpp` 612, `trackers/plain_tracker.cpp` 649 | each minus its `self_test` → 590 / 406 / 500 |
| `controller_mission.cpp` 1,456 | `mission_library.cpp` (parse/load/save) · `mission_runner.cpp` (start/stop/sample/summary) · `mission_report.cpp` (CSV, JSON, profile) |
| `trajectory_controller.cpp` 1,374 | `trajectory_controller.cpp` (`compute`, mode switch; 600) · `trajectory_esdf.cpp` (`build_esdf`, clearance queries; 400) · `trajectory_path.cpp` (relax, spline smoothing, carrot, blockage; 450) |
| `viewer_2d.cpp` 1,344 | `viewer_2d.cpp` (scene, events; 600) · `viewer_2d_world.cpp` (doors, room, obstacles, lidar; 450) · `viewer_2d_route.cpp` (path, approach; 350) |
| `local_policy/local_policy.cpp` 888 | `local_policy_candidates.cpp` · `local_policy_decide.cpp` (cut `decide`, 462) |
| `controller_display.cpp` 831 | `controller_display.cpp` · `controller_display_present.cpp` (`present`, 305) |
| tests/tools over 800 | `test_policy_scenarios.cpp` → harness header + one file per suite; `test_door_passage.cpp`, `test_anchor_servo.cpp`, `tracker_sim.cpp` by section |

**Phase B (design)** — make the three regimes objects instead of `if (policy_enabled())` threaded through
one class. This is the structural form of the owner's two-regime decision, and it deletes nothing:

```
struct DriveContext;                      // pose, plan, world, returns, clocks — built once per cycle
struct DriveResult { bool commanded; bool arrived; std::optional<Unreachable> unreachable; ... };

class IDriveRegime { virtual DriveResult drive(const DriveContext&) = 0; virtual void reset() = 0; };
class ChainDriver  : IDriveRegime   // planner → spline → band → PLAIN tracker, escape, backoff, anchor servo
class PolicyDriver : IDriveRegime   // local look-ahead policy (policy_tick / drive_with_policy)
class HybridDriver : IDriveRegime   // ChainDriver for a plain place, PolicyDriver for an episode (chain_hybrid)

class ArrivalProtocol   // consume_goal_reached, finalize_reached, dwell, confirming looks — shared
class CrossProtocol     // the cross's through / hand-back / timeouts — shared
class TargetResolver    // build_planning_step: selection, contract, standpoint
class SessionDiagnostics
class ControllerSession // owns the above; sync → resolve → regime.drive → protocols → display
```

Each regime owns its state (escape and band state leave the session for `ChainDriver`; the navigator, the
unreachable latch and the unseen-contact evidence for `PolicyDriver`). `Policy.Enabled` and
`Policy.HybridChain` then choose an object once at start-up.

### 6.3 Ordered steps

Tests: **U** = the unit tests in `controller/tests/` (after C0, one command); **M** =
`test_policy_scenarios` with `REGIME=policy`, `chain` and `hybrid` (24 main + 12 edge scenarios — the only
coverage of the chain and hybrid regimes, which the tours do not run); **B** = `route_bench --self-test` and
`tracker_sim`; **T1** / **T** / **R** as in 5.3 (R from the controller's `log_compute_perf` line).

| # | Step | Kind | Test |
|---|---|---|---|
| C0 | Guardrails. (i) A `tests/CMakeLists.txt` (or `run_tests.sh`) that builds every test from its header's own `g++` line and runs it; record which pass today. (ii) Reuse room_concept's `lint_file_lengths.sh` + allow-list for `controller/src`. (iii) Record M for the three regimes, B, and the live baseline (shared with R0) | guard | U, M, B recorded |
| C1 | Working-tree hygiene: delete the 3,385 `policy_field_dump_*.txt`, the `.new` files, the stray build directories; move `[Transforms] TrackerUsesLatestPose` to `[Controller]` | delete | start-up config table unchanged |
| C2 | Remove the doorway-gap code: `doorway_gap.h`, `update_doorway_gap`, the `gap_*` members, the `on_gap` branches (constant-false today), `test_doorway_gap.cpp`, `test_jamb_cross.cpp`. `CrossProtocol::on_gap`/`gap_score` stay only if a log column needs them — keep the columns, write the constant | delete | build; U; M; **T** (it is in the door cross, which every tour runs) |
| C3 | Remove the dead functions, accessors and members of 3a; remove `bench_c1_door.cpp` | delete | build; U |
| C4 | Move the five `self_test()` bodies into `*_selftest.cpp`, listed in `tools/CMakeLists.txt` only; delete or wire `MissionRunner::self_test` | move | B identical; agent builds |
| C5 | **Owner decisions** from §7: unread diagnostics, the migration monitor, obstacle creation, the near-field arm, the comment history. One commit each | delete | U; M for anything in the chain; T1 per live-path removal |
| C6 | Split `controller_door_passage.h` and `controller_anchor_servo.h` into interface + implementation **before** splitting the session: both are included by `controller_session.h`, so afterwards every session TU would re-parse them. Update the `g++` lines of the tests that include them | mech | U (`test_door_passage`, `test_anchor_servo`, `test_door_look_loop`, …) |
| C7 | Slim `controller_session.h`: history to a design note (if agreed), nested types and per-responsibility state structs into their own headers | mech (renames `x_` → `state_.x` are scripted; diff every touched function against git) | build; U; M; T1 |
| C8 | Split `controller_session.cpp` into the TUs of 6.2, bodies verbatim, a script that asserts every source line lands exactly once (the method used for `room_concept.cpp`). The four anonymous-namespace blocks (27-42, 52-86, 658-691, 2257-2310, 6129-6136) go to `controller_session_detail.h` | mech | build; M (3 regimes); **T**; R |
| C9 | Cut `build_planning_step` (1,388) and `execute_plan` (1,074) into named phases, bodies verbatim, sharing a per-cycle struct (as `UpdateCycle` did) | mech | M; T; R |
| C10 | Split `specificworker.cpp` and `specificworker_mission.cpp`. `load_params` becomes a free function; check `etc/config_effective.csv` is identical | mech | diff of `config_effective.csv`; T1 |
| C11 | Split `controller_obstacle_tracker.cpp`, `grid_planner.cpp`, `route_optimizer.cpp`, `trajectory_controller.cpp`, `controller_mission.cpp`, `local_policy.cpp`, `controller_policy_drive.cpp` | mech | B identical; U; M; T1 |
| C12 | Split `viewer_2d.cpp`, `controller_display.cpp` | mech | build; start once; T1 |
| C13 | Split the oversized tests and `tracker_sim.cpp` | mech | U, B |
| — | **After C13 every file in controller is under 800 lines** | | |
| C14 | `SessionDiagnostics` + the shared `CsvSink` | design, small | diagnostic files byte-identical on a mock run; T1 |
| C15 | `TargetResolver` out of the session | design | M; T |
| C16 | `ArrivalProtocol`, `CrossProtocol` as objects | design | M; T |
| C17 | `PolicyDriver` (the live regime first) | design | M `REGIME=policy`; T; R |
| C18 | `ChainDriver`, then `HybridDriver` | design | M `REGIME=chain` and `hybrid` — the tours cannot test these |

Build memory. `controller_session.cpp` is a single 9,567-line TU that includes DSR, Qt and 35 project headers;
it is very likely the controller's largest `cc1plus`. Sixteen TUs cut that peak, but each still pays for
the header set — hence C6 and C7 first, and a precompiled header for DSR/Qt/Eigen if the `-j8` peak rises.
Measure with `/usr/bin/time -v` before C6 and after C8.

Rates. Same LTO caveat as room_concept. The control loop is the rate-critical path: after C8 and C9 compare
the `log_compute_perf` line with the baseline. Removing `log_cloud_wall_fit`, `log_residual_world`,
`monitor_footprint_orientation` and the unread per-cycle CSVs in C5 can only help.

---

## 7. Risks and questions for the owner

### Risks

1. **The tours do not cover everything.** The series runs `MapMode=estimate` and `ltsm`, `Policy.Enabled=true`.
   It does not run `MapMode=given`, the old chain, the hybrid, the box layout, or object anchors. For the
   controller's chain and hybrid, the 2-D mock is the only regression test; for room_concept's offline
   paths, the oracle is. A step that touches one of those needs that test, not a tour.
2. **The oracle runs legacy arms.** `Params` defaults are the old values; the selftests use them. Flip the
   defaults to the live config and re-record before deleting any legacy arm (R6).
3. **Log lines and CSV columns are interfaces.** Tour scripts `grep` `room_concept.log` and `controller.log`;
   graders read `proto_room.csv` by column number (column 40 = promoted). No text or column changes in a
   mechanical step.
4. **Scripted cuts.** The 10-17 lesson: a cut by "next closing brace" deleted the tail of `start()`. Every
   split is done by a script that asserts line coverage, and every moved function is diffed against git.
5. **Thread ownership.** `RoomSceneGraph::update` runs on the localiser thread while `compute()` reads it on
   the GUI thread; only pinned landmarks are snapshotted so far (`301e7a98`). Splitting files does not change
   this, but phase B must finish `SceneSnapshot` before moving state. The `InnerEigenAPI` instance must stay
   the one the main thread owns. DSR connects stay queued; nothing new is connected.
6. **Init order** in both `initialize()` functions is load-bearing (media subscribers after the sensor node
   exists, publisher before reader threads). `make_*` steps keep the order line for line.
7. **Header-only tests.** Many tests compile a header with a one-line `g++` command. Turning a header into
   `.h` + `.cpp` (C6) breaks those lines unless they are updated in the same commit.
8. **Other sessions.** This checkout is shared with other Claude sessions and worktrees (`rt-build`,
   `interaction-anchor`). A file split conflicts with every open branch that touches the file. Merge or
   rebase the open branches before R7 and C8, and do each big split in one sitting.
9. **No model change is proposed.** No threshold, gate or clamp is added. The only items that could alter
   behaviour are flagged: removing per-cycle diagnostics (timing only), and the config-key move in C1
   (same effective value).

### Questions

Room_concept:
1. **Box layout** (`RoomShape.BoxLayout = false`, 4,389 lines + bench arms): keep as the A/B, or retire?
   This decides whether R10 is a split or a deletion.
2. **Object anchors** (`ObjectAnchor.enable = false` since 09-17, ~1,300 lines): keep?
3. **Rerun logger** (off, 402 lines, own thread): remove?
4. **Legacy arms** of the sixteen settled flags in 2b: which may go? And may the code defaults be flipped to
   the live values first?
5. `PreserveBootstrapRoom`, `PredictedTicks`, `CalibPivotEnabled`, `UseCommandVelocityPrior`,
   `ModeDisambiguation`, `MotionNoiseLearn`, `PreintZuptAsFactor`: any of these still wanted?
6. The ~20 diagnostic files with no tracked reader (2c): which do you still open by hand?
   Is `HessianCheck = true` still needed in the live config?
7. `src/calib_channel_test.cpp`, `calib_pivot_test.cpp` and the orphan tools: keep with a target, or delete?
8. Merge `room_concept/src/timeseries_plot.*` with `common/dashboard/`?

Controller:
9. **Arm B** (`Route.NearFieldClearance`, `Route.BandDrivesCross`, 458 lines + a 652-line test): abandoned?
10. **Obstacle creation** (`ObstacleCreationEnabled = false` since 07-29, ~800 lines and ~30 config keys):
    still wanted as the fallback when `residual_concept` is down?
11. **Old chain**: confirmed kept. Which of its parts does the hybrid need — the band, the escape, the returns
    back-off, the anchor servo? Anything the hybrid will never use could go once the hybrid is live.
12. The **migration monitor** (`pose_free_legacy`, `orientation_census`): is the yaw correction trusted?
13. Unread diagnostics in 3c (`cloud_wall_fit.csv`, `residual_world.csv`, `overlay_*`, `plan_track.csv`,
    `finalize_sites.jsonl`, `base_commands.jsonl`, `route_events.csv`, `hold_events.csv`, `why_log.jsonl`,
    `canvas_health.csv`): remove all, or name the ones to keep?
14. `policy_field_dump_*.txt`: remove the writer, or keep it bounded?
15. `live_tracking()` has no reader but its sampler runs every cycle: is the L-adaptation panel gone?

Both:
16. **Comment history.** About a quarter of both agents is comment-only lines, much of it dated investigation
    narrative. May it move to design notes next to the code (`controller/DESIGN_NOTES.md`,
    `ROOM_CONCEPT.md`)? It is the cheapest way under 800 for the two big headers, and the alternative is
    to keep it and split further.
17. Do tools and tests count for the 800-line rule (eight C++ files, three Python)? The plan assumes yes, last.
18. Order: room_concept first (guardrails exist, plan half done) or controller first (largest file, no
    guardrail, and the hybrid port is about to add to it)?
