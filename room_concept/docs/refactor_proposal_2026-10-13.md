# room_concept refactor proposal — 2026-10-13

Two parts: **(A)** my review of the analysis: what I verified, what I changed, and the order I recommend.
**(B)** Fable's full analysis, unchanged. It is the evidence base: every claim is cited by file:line.

## A. Review and recommendation

### Verified against the code (not taken on trust)
- **The thread race in §0/§1.4 is real and is a bug today, independent of the refactor.**
  - `RoomConcept::run()` calls `on_result_ready_()` directly on the localiser thread (`room_concept.cpp:1082`).
    That call reaches `PosePublisher::maybe_publish_corrected_pose`, and from there `scene_graph_->update(...)`
    (`pose_publisher.cpp:473`).
  - `refresh_object_anchors()` clears and refills `latest_pinned_landmarks_` on that thread
    (`room_scene_graph.cpp:3584-3590`).
  - Meanwhile `SpecificWorker::compute()` on the GUI thread iterates the same vector by const reference to draw it
    (`specificworker.cpp:311`, through `pinned_landmarks()` at `room_scene_graph.h:159`).
  - A reallocation during the draw is a use-after-free: the heap-smash class CLAUDE.md warns about, whose victim
    surfaces elsewhere.
  - The comment at `room_concept.cpp:1078-1081` still describes the QueuedConnection marshal that the 09-03 fix removed.
  - **Do step 4 (SceneSnapshot) first, as a standalone bug fix**, before any code moves.
- The bench reality (§0, §1.7) matches `src/CMakeLists.txt`: the selftests hand-list agent sources, and there is no
  library target.

### Amendments
1. **Compile memory: step 2 needs a precondition.**
   - Splitting `room_concept.cpp` into ~17 TUs that each still include `room_concept.h` (2,668 lines, 55 `torch`
     mentions) means 17 parses of the torch headers.
   - Under `-j8` that can RAISE the parallel peak, because eight torch TUs run at once. The cc1plus peak is
     dominated by torch and template instantiation, not by our own line count.
   - Before step 2:
     - (a) move the tensor-bearing members and the `torch/torch.h` include out of `room_concept.h`. This is
       Fable's step 3 and risk 6, brought forward.
     - (b) add `target_precompile_headers(... <torch/torch.h>)` on the solver/localizer libraries.
   - Measure each step with `/usr/bin/time -v` on the largest TU and `pgrep -a cc1plus` while building at -j8.
2. **Target sizes.** Several split targets in §3 sit at exactly 800. Aim for ≤ 650 so the next feature does not
   immediately push a file over the cap.
3. **Step 1 inconsistency.** §2.1 makes `rc_room_core` STATIC, while step 1 calls it INTERFACE. Start INTERFACE
   (header-only today) and turn it STATIC when the first pure `.cpp` lands (`ceiling_estimator.cpp`,
   `mount_reconcile.cpp`).
4. **Deletions in step 15 are separate decisions.** Each needs its own evidence first:
   - legacy grid search;
   - the Adam/L-BFGS shadows;
   - the differential test.

   Deleting one of these is not "refactoring". The memory notes record several cases where a "dead" path was the
   only witness of a defect.

### Recommended order
1. **Now, ~0.5 d:** SceneSnapshot plus the comment fix (Fable step 4). It removes a crash hazard.
2. **Guardrails, ~1.5 d:**
   - the lint with a shrinking allow-list;
   - the oracle artefacts recorded from one run;
   - library targets, with every selftest linking them (Fable steps 0-1).
3. **Untangle the header, ~1 d:** split it and confine torch (Fable step 3 + amendment 1).
4. **Mechanical splits, ~5 d:** Fable steps 2, 5 and 6. After these, every file is < 800 lines.
5. **Architecture, ~25 d:** Fable steps 7-15. Do them in this order:
   - `Localizer::step` (step 9) early, because it is what makes the bench run the agent's code;
   - then the channel and layout-estimator interfaces.

---

## B. Fable analysis (verbatim)

# room_concept — refactoring proposal (read-only analysis, 2026-10-13)

Scope: `room_concept/src` (126 files, 63 594 lines), its offline tools and the CMake that builds them.
Goal: every source file under 800 lines, organised so new sensor channels, layout estimators, room
types, multi-room/LTSM interaction and offline-bench testing are additive rather than surgical.
Everything below was read from the code; line numbers are from the working tree on
`wip/box-estimator-rework` (src is clean in `git status`).

---

## 0. Summary

* 17 files exceed 800 lines (+ `specificworker_startup.cpp` at 737). Together they are 38 k lines, 60 % of the agent.
* The agent is one god object, `rc::RoomConcept` (`room_concept.h:98-2668`, `room_concept.cpp` 9 444 lines):
  **176 `Params` fields, 72 `UpdateResult` fields, 355 data members (342 of them touched in the .cpp),
  187 member functions, 21 `std::ofstream` CSV writers, 17 mutexes/atomics**, and a single `update()` of
  **1 093 lines touching 103 members** (`room_concept.cpp:3510-4603`).
* Its header is the hub of the include graph: 13 files include `room_concept.h`, among them the
  *solver* (`room_gn_solver.h:49`), the observation weights (`room_obs_weights.h:17`), the status reporter,
  the ground-truth log and the LiDAR ingestor. Nothing under it can be compiled, tested or linked without
  torch, and the tests work around that with text surgery (`tools/tests/lift.py` copies structs out of
  `mount_lidar_pair.h` "so the test compiles with neither DSR nor Qt").
* The offline bench (`tools/wall_slam_selftest.cpp`, 5 363 lines) links `room_concept.cpp` but only to
  borrow types (`WindowSlot`, `BoundaryPrior`, `OdometryPrior`) and two helpers
  (`fuse_priors`, `compute_motion_covariance`, called on a dummy `RoomConcept motion_agent`,
  bench line 795). It drives `rc::gn::solve`, `WallMap`, `boxch::Channel` and `CornerDetector` itself,
  i.e. it **re-implements the per-frame loop that lives inside `RoomConcept::update()`**, because that loop
  cannot be called without the thread, the buffers and the DSR-facing callbacks. This is the structural
  reason behind the "the bench must run the agent's code" lesson.
* Five `EXCLUDE_FROM_ALL` selftests each hand-list 14-16 agent sources (`src/CMakeLists.txt:118-250`); the
  comments there record two link breakages from forgetting one. There is no library target.
* One real thread hazard found while mapping: `RoomSceneGraph::update()` (all DSR room/wall/affordance
  writes) runs on the **localiser thread** via
  `run()` → `on_result_ready_` (`room_concept.cpp:1082`) → `PosePublisher::maybe_publish_corrected_pose`
  (`specificworker_startup.cpp:482`) → `scene_graph_->update(...)` (`pose_publisher.cpp:473`),
  while `SpecificWorker::compute()` on the GUI thread reads `scene_graph_->pinned_landmarks()`
  (a `const std::vector&` to a member the localiser thread rewrites), `stable_frames()`,
  `room_node_created()` and calls `monitor_affordance()` with no lock (`specificworker.cpp:285-350`).
  The comment at `room_concept.cpp:1078-1081` still claims the callback "marshals the graph write to the
  main thread via a Qt::QueuedConnection"; `specificworker.cpp:325-335` says that marshal was removed
  on 2026-09-03. The refactor should fix this with a snapshot (see §2.4) before anything is moved.

The proposal: five static libraries layered by dependency (`room_core` → `room_solver` / `room_layout` →
`room_localizer` → `room_ui`, and the `room_concept` executable as the only DSR/Qt-widget consumer),
`RoomConcept` decomposed into a synchronous `Localizer::step(inputs) -> UpdateResult` plus ~10
collaborators with explicit state ownership, a channel interface that produces `rc::gn::IFactor`s (the
factor interface already exists, `room_gn_solver.h:97`), and a 15-step migration where the first five
steps are mechanical, each buildable, and already put every file under 800 lines.

---

## 1. Architecture as found

### 1.1 Components

| Area | Files (lines) | Notes |
|---|---|---|
| Agent shell (RoboComp/DSR) | `specificworker.{h,cpp}` 414/659, `specificworker_startup.cpp` 737, `specificworker_slots.cpp` 254, `specificworker_presence.cpp` 155 | `initialize()` is one 535-line function (`specificworker_startup.cpp:202-737`) that constructs and wires everything. |
| Localiser (god object) | `room_concept.{h,cpp}` 2 668 / 9 444 | See §1.3. |
| Solver | `room_gn_solver.{h,cpp}` 277/1 477, `room_model.{h,cpp}` 162/387 (torch SDF model), `room_obs_weights.h`, `reloc_search.{h,cpp}` 556, `scan_slam.h` 233 | GN/LM with an `IFactor` interface and **11 factor classes** (`room_gn_solver.cpp:45-760`); `make_layout` 787-1011, `build_factors` 1011-1238. Depends on `RoomConcept::WindowSlot/BoundaryPrior/Params`. |
| Layout estimation | `wall_segmenter` 286, `wall_map.{h,cpp}` 975/4 040, `room_boxes.{h,cpp}` 284/1 297, `room_boxes_channel.{h,cpp}` 440/2 368, `free_space_polygon.h` 1 069, `layout_refiner.h` 577, `layout_explorer.{h,cpp}` 421, `layout_estimator.h` 85 (type-only stub), `learnt_layout.h` | Pure Eigen/STL; `rc::status` prints only. |
| Corner / image-edge / anchors | `corner_detector.{h,cpp}` 534/1 009, `corner_visibility.h`, `image_edge_*` (types/ops/accumulate/source/factor ≈ 1 800), `object_anchor_*` ≈ 500 | Pure. `CornerDetector::detect()` is one 913-line function (`corner_detector.cpp:96-1009`). |
| Motion / calibration maths | `se2_preintegration.h` 531, `motion_calibration.h` 617, `motion_noise_*`, `wheel_noise_learner.h`, `rest_density_learner.h`, `pose_field_bias.h`, `surprise.h`, `localization_drift.h`, `drift_monitor.h`, `stride_span.h`, `calibration_estimator.h` 529, `calibration_intake.h`, `joint_calibration.h`, `joint_calib_monitor.h`, `mount_lidar_pair.h` 698, `mount_factors.h` 678, `camera_calibration.h` 562, `calib_pivot.h`, `calib_channel.h` 398, `lidar_mount.h` | Header-only, Eigen/STL. `camera_calibration.h` logs through `QDebug` (why `mount_replay` needs `Qt6::Core`, `src/CMakeLists.txt:259`). |
| Planning | `epistemic_planner.{h,cpp}` 581/878, `epistemic_controller.{h,cpp}` 355 | Pure; `evaluate_targets` 395-642, `select_target` 642-758 writes `epistemic_select.jsonl`. |
| DSR adapters | `room_scene_graph.{h,cpp}` 510/4 139 (210 DSR refs), `pose_publisher.cpp` 611, `mount_calibrator.cpp` 838 (19 DSR refs, 45 Qt), `calib_channels.cpp` 599, `ground_truth_log.cpp` 328, `object_anchor_source.cpp` 253, `door_apertures.cpp` 192, `status_reporter.cpp` 236 | All the graph reads/writes. |
| Sensor ingest (media plane) | `lidar_ingestor.{h,cpp}` 255/823, `imu_ingestor` 105, `camera_ingestor.{h,cpp}` 235/375, `camera_stream_hub.{h,cpp}` 236 | Own threads; `lidar_ingestor.h:41` includes `room_concept.h`. |
| Config | `room_config.{h,cpp}` 487/1 399 | `load_room_config` is one 1 180-line function (`room_config.cpp:76-1255`) with 215 parameter writes across prefixes `RoomConcept.`(204 keys) `RoomShape.`(57) `EpistemicController.`(54) `ImageEdge.`(51) `ObjectAnchor.`(23) `PredictPublish.` `Media.` `DSR.` `StartupProto.`; overlays at 1255-1399. |
| UI (Qt) | `viewer_2d.{h,cpp}` 427/1 974 (186 Qt refs), `room_viewer.{h,cpp}` 225/921, `camera_visualizer.{h,cpp}` 251/1 544 (36 DSR refs), `calibration_viewer.cpp` 495, `timeseries_plot.cpp` 405, `custom_widget.h` | `viewer_2d.cpp` polls DSR itself (`refresh_semantic_bboxes`, 1623-1750); `camera_visualizer.cpp` fetches RGB, intrinsics, walls and objects from DSR (360-649) and has a 601-line `draw_projections` (767-1368). |
| Diagnostics | `rerun_logger.cpp` 260 (own thread), `svg_room_loader.cpp`, 21 CSV streams inside `RoomConcept`, 8 inside `RoomSceneGraph` | |

### 1.2 Dependency sketch (as it is)

```
specificworker ─┬─ RoomConcept ─┬─ room_model (torch) ─┐
                │   (9.4k)      ├─ room_gn_solver ─────┤  room_gn_solver.h:49 → #include "room_concept.h"
                │               ├─ wall_map, room_boxes_channel, scan_slam, reloc_search
                │               ├─ corner_detector, image_edge_*, object_anchor_*
                │               └─ motion/calibration headers (14 of them)
                ├─ RoomSceneGraph (DSR) ── RoomConcept*, EpistemicController, LayoutRefiner, FreeSpacePolygon, CalibChannel, ObjectAnchorSource
                ├─ PosePublisher (DSR) ── RoomConcept&, RoomSceneGraph*
                ├─ LidarIngestor / ImuIngestor / CameraIngestor+Hub (media plane) ── RoomConcept&
                ├─ MountCalibrator (DSR+Qt), CalibChannels, GroundTruthLog, StatusReporter ── RoomConcept&
                ├─ RoomConfig (writes room_concept.params.*, epistemic params)
                └─ RoomViewer (Qt) ── Viewer2D (Qt+DSR), CameraVisualizer (Qt+DSR), CalibrationViewer
```

`room_concept.h` includes 27 local headers (`room_concept.h` include list) and is included by 13 files.
Two cycles of meaning, not of text: the solver needs the god class's nested types, and the god class
calls the solver; `room_obs_weights.h` needs `RoomConcept::Params` to weight points, and `room_concept.cpp`
re-declares the same three weight functions in an anonymous namespace (`room_concept.cpp:33-164` vs
`room_obs_weights.h:28-45`) — duplicated logic #1.

### 1.3 The god object, quantified

`rc::RoomConcept` (`room_concept.h:98-2668`):

| What | Count | Where |
|---|---|---|
| `Params` fields | 176 | `room_concept.h:101-784`, 38 `// =====` sections (window, boundary gate, hierarchical precision, far-point, incidence, early exit ×2, differential test, velocity-adaptive, dual-prior fusion, prior covariance, slip, online motion learning, relocalisation, optimizer selection, GN/LM, Adam, L-BFGS, covariance numerics, boundary gate B/C, FEJ+Schur, motion model, preintegration, strided window, innovation-adaptive, heading factors, wheel variance, corner detection, object anchors, wall-SLAM, debug, rerun) |
| `UpdateResult` fields | 72 | `room_concept.h:784-966`; carries `lidar_scan`, `corner_matches`, `segments`, `walls`, `seg_to_wall` vectors across the thread boundary every frame |
| Data members (trailing `_`) | 355 (342 used in the .cpp) | |
| Member functions | 187 | |
| CSV `std::ofstream` members | 21 | `recovery_log_ innov_csv_ kin_csv_ nis_csv_ resid_csv_ wallgeom_csv_ layout_csv_ probe_csv_ shadow_csv_ wall_input_rec_ wall_csv_ debug_log_ opt_csv_ flip_csv_ hier_prec_csv_ corner_stats_csv_ image_edge_csv_ mount_csv_ triple_csv_ triple_pose_csv_ gn_shadow_csv_` |
| Mutexes / atomics | 17 | `room_concept.h:1646-1726, 1795-1799, 2049-2050, 2121-2136, 2292-2307` |
| Nested classes | `WindowManager` (`room_concept.h:1868-1925`, impl `room_concept.cpp:8872-9444`, 570 lines, torch RFE loss), `LandmarkEstimate`, `SearchEpisode`, `CornerVertexStats`, `DiffTestStats`, `PredictionState`, `MountSnapshot`, `LowObstacles`, `FactorB`, `ImageEdgeStats`, `BoundaryPrior`, `WindowSlot` |

Feature areas inside `room_concept.cpp` (functions grouped by name; members = distinct data members referenced):

| Feature area | Lines | Members touched | Largest functions |
|---|---|---|---|
| update core (solve cycle) | 2 558 | 133 | `update` 3510-4603 (1 093 lines, 103 members), `try_prediction_early_exit` 5157-5719 (562), `run_gn_loop` 6035-6161, `run_gn_shadow` 7041-7156, `compute_posterior_covariance` 7319-7443, Adam/L-BFGS loops 7156-7319 |
| motion prior / odometry / IMU | 1 651 | 89 | `integrate_odometry_over_window` 8110-8663 (553), `build_motion_prior_selection` 4616-4871, `compute_odometry_prior` 7443-7538, `compute_measured_odometry_prior` 8663-8771, `fuse_priors` 8771-8808, `observe_innovation` 4948-5104, `feed_motion_calibrator` 7752-7848 |
| bootstrap / relocalisation | 1 342 | 36 | `legacy_grid_search_pose` 1856-2316 (460), `grid_search_initial_pose` 1378-1571, `bootstrap_initialization_from_lidar` 599-768, `step_rival_modes` 1760-1856, lost-belief 1655-1760 |
| image-edge shadow + mount self-calibration | 1 024 | 61 | `run_image_edge_shadow` 6161-6493 (332), `mount_pooled_solve` 6711-7015 (304), `log_triple_points` 6493-6711, `observe_kinematic_mount` 5104-5157 |
| lifecycle / thread / commands | 982 | 78 | `run` 782-1252 (470; the thread loop: drain commands, init, read LiDAR, snapshot buffers, `update`, publish, relocalise, differential test), `start/stop` 210-276, debug-log init 276-436 |
| wall-SLAM / layout bridge | 903 | 64 | `wall_slam_after_solve` 2854-3214 (360), `wall_slam_observe` 2528-2787, `reanchor_map_frame` 3214-3276, `set_polygon_room` 3276-3359, map-layout adoption 2812-2854 |
| boundary prior / hierarchical precision / window | 737 | 12 | `WindowManager::compute_rfe_loss` 8931-9243 (torch), `recompute_boundary_prior` 9243-9334, `marginalize_oldest` 9334-9444, `update_boundary_hyperprecision` 5719-5769 |
| doors | 39 | 1 | `filter_through_door` 2406, `door_point_weights` 2426 |

Cross-feature coupling (members shared between areas):

* update core ↔ motion prior: **47 shared members**, almost all `cyc_*` per-cycle scratch
  (`cyc_U_, cyc_dt_s_, cyc_dx_local_, cyc_imu_dtheta_, cyc_heading_cov_ …`) written by the prior
  builder and read when the result is extracted (`update` "EXTRACT RESULT FROM NEWEST SLOT", 4165-4245).
  These are not state; they are a missing `struct CycleMotion` return value.
* lifecycle ↔ update core: 23 (timing counters `last_t_*_ms_`, `last_loss_*`, `debug_log_`).
* wall-SLAM ↔ update core: 18 (`model_, window_mgr_, plain_slam_, map_ready_, doors_robot_, landmark_estimates_`).
* Touched by 4+ areas: `model_`, `window_mgr_`, `init_polygon_vertices_`, `doors_robot_` — these four are the
  true shared state; everything else is feature-private and can move with its feature.

torch is not confined: 315 `torch::` references in `room_concept.cpp` spread across every 500-line bucket
(peaks at 0-500: observation weights; 3500-5500: `update`/early exit; 8500-9444: `WindowManager` RFE
loss). Qt in `room_concept.cpp` is logging only (77 `QString`, 38 `qInfo`, 21 `qWarning`).

### 1.4 Threads and what crosses them

| Thread | Created at | Runs | Writes into shared state |
|---|---|---|---|
| GUI / main (Qt event loop) | RoboComp `compute()` timer | `SpecificWorker::compute` (`specificworker.cpp:112-420`), DSR slot `modify_node_attrs_slot` (connected queued, `specificworker_startup.cpp:466`), `calib_->pump_image_edges()` → `room_concept_.set_image_edges` (`calib_channels.cpp:438`), viewers | `latest_image_edges_` (mutex), door apertures (mutex), object anchors (mutex), commands (mutex+cv) |
| `localizer` | `room_concept.cpp:252` | `RoomConcept::run` → `update` → `on_result_ready_` → `PosePublisher::maybe_publish_corrected_pose` → **`RoomSceneGraph::update` (DSR writes, proto-room, affordances, calibration, layout refiner)** | `last_result_` (mutex), `derived_polygon_` (mutex), 21 CSVs (no mutex, loc-thread only), **RoomSceneGraph members read by the GUI thread without a lock** |
| `lidar-ingest` | `lidar_ingestor.cpp:109` | media drain, ceiling cap, low obstacles, mount factors → `notify_new_lidar`, `post_low_obstacles`, `mount_snapshot` | `HighLidarBuffer`, `low_obs_` (mutex), `mount_snap_` (mutex), atomics |
| `imu-ingest` | `imu_ingestor.cpp:17` | → `ImuBuffer`, `pose_pub_->publish_predicted_tick` (DSR write from the IMU thread) | |
| camera hub | `camera_stream_hub.cpp:52` | `CameraIngestor::on_frame` (grey conversion, absorb) | ingestor inbox |
| rerun sender | `rerun_logger.cpp:148` | | |
| FastDDS readers | library | DSR signals (queued), media callbacks | |

`UpdateResult` is the de-facto cross-thread snapshot (72 fields, copied under `result_mutex_`). There is
no equivalent snapshot for `RoomSceneGraph`: `pinned_landmarks()` returns a reference to a vector the
localiser thread reassigns in `refresh_object_anchors` (`room_scene_graph.cpp:3478`); `stable_frames_`,
`room_node_created_`, `startup_promoted_` are plain ints/bools read by `compute()`.

### 1.5 Other big files — responsibilities and oversize functions

* `room_scene_graph.cpp` (4 139): `update` 187-402 (frame tick: reanchor, proto-room steps, room node
  creation, RT write, calibration, affordances); `write_robot_room_rt` 450-859 (409 lines, 14 DSR);
  `step_proto_room_impl` 882-1073 (door-crossing proto-room, 16 DSR); startup proto 1372-1909
  (`birth/arm/step/promote_startup_proto`, `find_recall`, `rearm_after_crossing`); `dsr_update_layout_affordance`
  1932-2278 (346, explorer affordance: edge/level/re-evaluate/publish); `dsr_create_room_and_reparent`
  2278-2393; calibration affordance 2406-2782 (`ensure_calib_node`, `dsr_update_calibration` 309 lines, 8 CSV
  writes); `dsr_update_affordance` 2782-3168 (386, planner target protocol); `break_execution_stall` 3168-3314;
  landmarks/anchors 3363-3605; graph hygiene 3605-4060 (`dsr_create_wall_nodes` 171 lines / 36 DSR refs,
  `cleanup_room_graph_nodes`, overlays, `check_init_graph_is_valid`); `step_layout_refiner` 4060-4139.
* `wall_map.cpp` (4 040): `observe` 2064-2787 (723 lines: association, births, direction evidence,
  zero-evidence refutation, strict-Manhattan eviction, ghost sweep); `re_derive` 3116-3788 (672: trace,
  rectilinearise, surrender, one-energy adoption, waiver, margin judge); `try_splice` 1052-1694 (642: run
  replacement, replacement, wrap completion, spur currency); `decorate` 424-723; `try_spur_wraps` 1694-1919;
  `try_down_jumps` 1948-2043; FreeGrid/frontiers 2836-2994; `reanchor` 3885; `beam_loglik/forward_delta/referee`
  3971-4040. `wall_map.h` holds a 420-line `Params` (67-490).
* `room_boxes_channel.cpp` (2 368): `observe` 30-180; **`plan_path` 397-1421 (1 024 lines)** — an
  information-gain path planner over the box layout (lambda ladder `is_occ/is_unknown/is_free/model_sd/
  is_trav/is_escape/… info_nats/seg_free`) that has nothing to do with estimating boxes; `reproject`
  1421-1498; `snap_coplanar` 1498-1563; `rebuild_from_free` 1563-1947 (384); `polygon` 1986; `fit_gauge_ml`
  2027-2099; `step` 2137-2368.
* `room_boxes.cpp` (1 297): box SDF/likelihood 20-167, `Layout::polygon/polygon_cov` 247-427,
  `condense` 427-678 (250), `face_support/faces_observed` 678-760, `register_scan` + MDL cost 760-1297.
* `corner_detector.cpp`: `detect` 96-1009 in three phases — per-model-corner gather with explain-away
  and graded information matrix (≈128-736), Hungarian/PDA association (≈798-990), fold into run totals
  (≈992-1009).
* `room_gn_solver.cpp`: 11 `IFactor` classes 45-760 (Sdf, ImageEdge, Motion, Boundary, Se2Landmark,
  LandmarkPrior, WallLine, WallPoint, RoomWall, WallPrior, Theta0Prior), `make_layout` 787-1011,
  `build_factors` 1011-1238, `gradient_check` 1246, `newest_pose_marginal` 1274, `solve` 1329-1477.
* `viewer_2d.cpp`: `draw_wall_map` 1121-1503 (382), `draw_rgb_corners` 750-923, `draw_corners` 923-1075,
  planner grids 1503-1623, DSR semantic bboxes 1623-1782, anchors 1782-1974.
* `camera_visualizer.cpp`: DSR fetches 360-649, `predicted_camera_from_room` 649-716, `draw_projections`
  767-1368 (601 lines in 8 numbered layers), `update_frame` 1403-1522.
* `room_config.cpp`: see §1.1 — one function; the split points are the first key of each prefix
  (`RoomConcept.` 85, `ProtoRoom.` 129, `StartupProto.` 133, `PredictPublish.` 189, `RoomShape.` 223,
  `Media.` 388, `Overlay.` 412, `ObjectAnchor.` 836, `ImageEdge.` 882, `Platform.` 934, `Scenario.` 1007,
  `DSR.` 1115, `EpistemicController.` 1139).
* `mount_calibrator.cpp`: pure `reconcile_mount_nominal` 135-310 (6 DSR refs mixed in),
  `publish_mount_to_graph` 310-470 (DSR), pair log 470-581, `mount_pair_update` 581-785, `loop_closure_observe` 785.
* `lidar_ingestor.cpp`: `pump` 158-309, `update_ceiling_cap` 385-592 (207, pure statistics),
  `run_startup_geometry_check` 592-666, `low_obstacle_step` 716, `mount_factor_step` 741-817.
* `epistemic_planner.cpp`: candidates 125-213, information terms 231-395, `evaluate_targets` 395-642,
  `select_target` 642-758, IOR/attempts 758-878.
* `free_space_polygon.h` (1 069, header-only): polygon utilities 57-133, `FreeSpacePolygon` class 133-720,
  grid internals 720-870, static raster algorithms 874-1069 (`edt_2d`, `component`, `fill_holes`,
  `remove_pinches`, `trace_outer`, `merge_collinear`, `douglas_peucker_closed`).
* `specificworker_startup.cpp`: `initialize` 202-737 constructs, in order, `mount_` (261) → `pose_pub_`
  (265) → `calib_` (267) → `gt_log_` (276) → `rt_api_` (287) → `inner_eigen_` (301) → `scene_graph_` (302) →
  `lidar_ingestor_` (313) → `imu_ingestor_` (318) → run context (328) → viewer (423) → DSR connect (466) →
  `room_concept_.start()` (485) → presence/Operating-enter lambda (560-620) that starts ingestors.

### 1.6 Duplicated logic noticed

1. Observation weights: `room_concept.cpp:33-164` anonymous namespace vs `room_obs_weights.h:28-45` (same three functions, same signatures).
2. Three relocalisers (`grid_search_initial_pose`, `legacy_grid_search_pose`, `rc::reloc::search`) kept side by side; `legacy_grid_search_pose` (460 lines) is the retired lattice (memory: 3 fighting relocalisers → mixture search, `5772332`).
3. Three optimiser back-ends (`run_adam_loop`, `run_lbfgs_loop`, `run_gn_loop`) selected by `Params`; Adam/L-BFGS are now shadows (`shadow_single_step_adam`).
4. Door-aperture carrying into the robot frame is done in `RoomConcept` (`doors_robot_`) and again in `SpecificWorker::compute` (`specificworker.cpp:135-140`, `T * d.a`).
5. The bench's frame loop vs `RoomConcept::update` (see §0). `wall_slam_selftest.cpp:519-520` explicitly notes it re-does what `compute_motion_covariance` / `fuse_priors` do live.
6. CSV plumbing: 21 + 8 + `compute_csv_` + `dcsv` in `compute()` + `odom_sample_log_` each re-implement "open under tmp/, imbue classic, header, flush".
7. `room_scene_graph.cpp` and `viewer_2d.cpp` both classify DSR object nodes (`node_is_object_class` 47, `object_node_class` 1612).

### 1.7 What the offline tools link today

| Target (`src/CMakeLists.txt`) | Hand-listed agent sources | Needs |
|---|---|---|
| `gn_selftest` 118 | 15 | torch, Qt (via headers), DSR link line |
| `mp3d_replay` 149 | 15 | same |
| `reloc_selftest` 175 | 16 | same |
| `corner_gather_test` 200 | 1 | `${LIBS}` anyway |
| `wall_slam_selftest` 215 | 15 + 8 controller/common sources | same + controller |
| `motion_calib`, `mount_replay` 236-262 | 0 (header-only) | `mount_replay` needs `Qt6::Core` because `camera_calibration.h` uses `QDebug` |
| ~20 `g++ -std=c++23 …` one-liners in `tools/*.cpp` headers | header-only pieces | none; `tools/tests/lift.py` copies code out of `mount_lidar_pair.h` to escape Qt/DSR |

`estimation_scripts/scripts/run_campaign.sh` and `round_trips.sh` drive the live fleet and grade from
`tmp/*.csv` (`proto_room.csv` column 40 = promoted, `layout_refine.csv`, `map_layout.csv`,
`refine_input.bin`, `slam_shadow.csv`, `sdf_localizer/gt_error.csv`). Those CSV contracts are the
regression oracle the migration must preserve.

---

## 2. Target architecture

### 2.1 Layers and library targets

```
room_concept/src/
  core/        rc_room_core      STATIC   Eigen + STL only. No torch, Qt, DSR, OpenCV.
  solver/      rc_room_solver    STATIC   core + libtorch.   Model (SDF), GN/LM, factors, reloc search, scan_slam.
  layout/      rc_room_layout    STATIC   core only.         Segmenter, WallMap, boxes, free-space, refiner, explorer, planner.
  localizer/   rc_room_localizer STATIC   core+solver+layout. Localizer::step, channels, window, motion prior, reloc, diagnostics.
                                           std::thread allowed (LocalizerLoop). No Qt widgets, no DSR, no media.
  ui/          rc_room_ui        STATIC   Qt Widgets. Viewers draw from snapshots; no DSR.
  agent/       room_concept      EXE      DSR, media plane, RoboComp, presence. The only place `G->` appears.
```

Rule enforced by the build (each lib's `target_link_libraries` is explicit, as `common/CMakeLists.txt:13-15`
already does for `ai_common::*`): `core` links nothing; `solver` links `core` + `torch`; `layout` links `core`
(+ `ai_common::core` for `rc::status`); `localizer` links the three; `ui` links `localizer` + Qt6 Widgets/PrintSupport;
`agent` links everything + DSR + `ai_common::media/presence/affordance/robot/status`. Flags: `-g -O3 -fno-char8_t`
set per target, never `-march=native` (ABI note at `src/CMakeLists.txt:106-110` stays).

Every offline tool becomes `add_executable(x tools/x.cpp)` + `target_link_libraries(x rc_room_localizer)` —
no more hand-listed source lists, and no way to leave one out. `motion_calib`/`mount_replay` link `rc_room_core`
only (so `camera_calibration.h` must lose its `QDebug`).

### 2.2 Key interfaces

**(a) Observation channel → factors.** The factor interface already exists:

```cpp
// solver/factor.h  (moved from room_gn_solver.h:97-106, unchanged)
class IFactor { virtual float evaluate(const State&) const = 0;
                virtual float linearize(const State&, LinearSystem&) const = 0; };
```

Add the channel side in `localizer/channel.h`:

```cpp
struct FrameInputs {            // assembled once per cycle, on the localiser thread, from the buffers
  std::int64_t ts_ms;
  std::span<const Eigen::Vector3f> scan_robot;      // after door filter
  Eigen::Vector3f pose_pred;  Eigen::Matrix3f cov_pred;
  const MotionCycle& motion;                         // the former cyc_* scratch (see 2.3)
  std::span<const DoorAperture> doors_robot;
  std::optional<ImageEdgeObs> image_edges;           // taken from the inbox this cycle, or none
  std::span<const ObjectAnchorObs> anchors;
};
class IObservationChannel {
public:
  virtual ~IObservationChannel() = default;
  virtual std::string_view name() const = 0;
  virtual void observe(const FrameInputs&, const WindowView&) = 0;      // pre-solve: detect/associate
  virtual void emit(gn::FactorList&, const gn::VarIndex&) const = 0;   // factors for this solve
  virtual void after_solve(const gn::Result&, UpdateResult&) = 0;      // consume posterior, fill the view
  virtual void reanchor(const Eigen::Affine2f& T_new_old) {}
  virtual void reset() {}
};
```

Existing code maps onto it one-to-one: `SdfScanChannel` (Model SDF + `room_obs_weights`),
`CornerChannel` (`CornerDetector` + corner stats + hierarchical precision inputs), `ImageEdgeChannel`
(`image_edge_*`, shadow + mount pooled solve), `ObjectAnchorChannel` (`Se2LandmarkFactor`,
`landmark_estimates_`), `WallSlamChannel` (`WallMap` + `WallLine/WallPoint/RoomWall/WallPrior/Theta0Prior`
factors, `box_channel_`, `plain_slam_`), `MotionChannel` (`MotionFactor` from `MotionPriorBuilder`),
`BoundaryChannel` (`BoundaryFactor` from `WindowManager`). `gn::build_factors` (`room_gn_solver.cpp:1011-1238`)
becomes a loop over channels. A new sensor is a new channel file plus one line in the channel list.

**(b) Layout estimator.** `layout_estimator.h:9-27` is already the intended shape; make it a real interface:

```cpp
class ILayoutEstimator {
  virtual void observe(const LayoutObservation&) = 0;   // scan in map frame + pose + precision
  virtual bool step() = 0;                                // derive / re-derive; true if the polygon changed
  virtual LayoutView view() const = 0;                    // raw + published polygon, per-corner sigma, state
  virtual void reanchor(const Eigen::Affine2f&) = 0;
  virtual void adopt(const std::vector<Eigen::Vector2f>& given) = 0;
};
```

Implementations: `WallMapEstimator` (wraps `wallmap::WallMap`), `BoxCoverEstimator` (`boxch::Channel`),
`GivenLayout` (the `MapMode::Given` path), and later a `MultiRoomLayout` that holds one per passage for
LTSM. `RoomSceneGraph` publishes `LayoutView`, not `WallMap` internals.

**(c) The synchronous step and the thread loop.**

```cpp
class Localizer {                       // single-threaded, pure; owned by LocalizerLoop or by a bench
  UpdateResult step(const FrameInputs&);      // = today's update() minus thread/buffer handling
  void handle(const Command&);                // SetPolygon / SetPose / GridSearch
  void configure(const LocalizerParams&);
};
class LocalizerLoop {                   // the only std::thread in the library
  RunContext ctx; CommandQueue cmds; ResultMailbox mailbox;   // result_mutex_/last_result_/on_result_ready_
  void run();   // drain commands, read buffers, build FrameInputs, localizer.step(), publish to mailbox
};
```

The bench constructs a `Localizer`, feeds it `FrameInputs` from ray-cast scans, and reads `UpdateResult`.
That is the structural fix for "the bench must run the agent's code": the loop it re-implemented stops
existing outside `Localizer::step`.

**(d) Cross-thread snapshots.** Keep `UpdateResult` as the loc → everyone snapshot but strip it to POD +
vectors (no torch). Add `SceneSnapshot { stable_frames, room_node_created, startup_proto, readiness, pinned_landmarks, pinned_measured, internal_from_published }`
published by `RoomSceneGraph` under one mutex and read by `compute()`. Add `ChannelInbox<T>` (mutex + optional +
`take()`) for the four inbound mailboxes (`latest_image_edges_`, `latest_object_anchors_`, `latest_door_apertures_`, `low_obs_`).

**(e) Diagnostics.** `rc::diag::CsvSink` (open under `tmp/`, `imbue(std::locale::classic())`, header, row,
periodic flush) replaces 30 hand-rolled ofstreams; each collaborator owns its sinks.

**(f) Params.** `RoomConcept::Params` (176) → `LocalizerParams { WindowParams; MotionPriorParams; EarlyExitParams;
RelocParams (exists: rc::reloc::Params); SolverParams (gn::Options exists); CornerParams (exists in CornerDetector);
ImageEdgeParams; WallSlamParams (wallmap::Params exists, 420 lines → its own header); DebugParams; }`.
`load_room_config` becomes one loader per struct, one file per prefix.

### 2.3 How `RoomConcept` decomposes (state ownership)

| New object (file) | Takes from RoomConcept | Owns (thread) | Approx. lines |
|---|---|---|---|
| `LocalizerLoop` (`localizer/localizer_loop.cpp`) | `run` 782-1252, `start/stop` 210-276, commands, `RunContext`, `notify_new_lidar`, `record_*_ingress`, `ResultMailbox` | thread, queues (loc) | 600 |
| `Localizer` (`localizer/localizer.cpp`) | `update` 3510-4603 rewritten as `step()` calling the phases below; `set_initial_state/set_robot_pose/set_polygon_room`; owns `model_`, `window_mgr_`, pose belief, `init_polygon_vertices_` | loc | 700 |
| `MotionPriorBuilder` (`localizer/motion_prior.cpp` + `odometry_window.cpp`) | `build_motion_prior_selection`, `compute_odometry_prior`, `compute_measured_odometry_prior`, `integrate_*_over_window`, `fuse_priors`, `compute_motion_covariance`, `predict_step`, `observe_innovation`, `feed_motion_calibrator`, `score_surprise`, noise learners, `cyc_*` → returned `MotionCycle` | loc | 750 + 600 |
| `WindowManager` (`localizer/window_manager.cpp`, `boundary_prior.cpp`) | nested struct 1868-1925 + impl 8872-9444, `update_boundary_hyperprecision`, hier-prec state, `read/write_window_poses` | loc | 600 + 300 |
| `EarlyExitGate` (`localizer/early_exit.cpp`) | `try_prediction_early_exit` 5157-5719, `unopt_pos_var_`, EE counters, `EeReason` | loc | 600 |
| `Relocalizer` (`localizer/relocalizer.cpp`, `relocalizer_legacy.cpp`) | 533-782, 1252-2316 (bootstrap, yaw ambiguity, grid search, lost belief, rivals, episodes CSV) | loc; `rivals_` snapshot under its own mutex | 750 + 460 (legacy, candidate for deletion) |
| `WallSlamChannel` (`localizer/channels/wall_slam_channel.cpp`) | 2316-3359 wall-SLAM block, `wall_map_`, `box_channel_`, `plain_slam_`, map-layout adoption, `derived_polygon_` + mutex, 8 CSVs | loc; `LayoutView` snapshot | 750 |
| `ImageEdgeChannel` (`localizer/channels/image_edge_channel.cpp`, `mount_shadow.cpp`) | `run_image_edge_shadow`, `log_triple_points`, `imgedge_health`, `mount_pooled_solve`, `observe_kinematic_mount`, `mnt_*`, `factor_b_`, `image_edge_stats_`, inbox | loc; inbox mutex | 650 + 400 |
| `CornerChannel` (`localizer/channels/corner_channel.cpp`) | corner detection block of `update` (3824-4027), `accumulate_corner_stats`, `write_corner_stats_csv`, `landmark_estimates_` | loc | 400 |
| `ObjectAnchorChannel` | anchor inbox + `landmark_estimates_` fusion | loc | 200 |
| `DoorWeighting` (`localizer/door_weighting.cpp`) | `filter_through_door`, `door_point_weights`, `doors_robot_`, `ee_door_discount_` | loc; inbox mutex | 150 |
| `SolveBackends` (`localizer/solve_gn.cpp`, `solve_shadow.cpp`) | `run_gn_loop`, `run_gn_shadow`, `compute_posterior_covariance`, `autograd_hessian_3x3`, `log_hessian_check`, Adam/L-BFGS shadows | loc | 500 + 400 |
| `LocalizerDiagnostics` (`localizer/diagnostics.cpp`) | `init_debug_log`, `init_recovery_log`, `write_debug_tail`, `log_hier_prec_row`, opt timing, differential test | loc | 400 |
| `RoomConcept` (facade, temporary) | public API used by the agent (§1.4 table: `set_*`, `get_last_result`, `estimating`, `map_ready`, …) delegating to the above | | 300 → deleted in step 15 |

Thread affinity is written in each header ("loc-thread only unless stated"); every inbox is a
`ChannelInbox<T>`; nothing else in `rc_room_localizer` holds a mutex.

### 2.4 Agent layer decomposition

* `RoomSceneGraph` → six adapters, all DSR, all constructed by the agent: `RoomRtWriter` (RT edge, body
  calibration, predicted pose), `RoomNodeLifecycle` (create/reparent/walls/cleanup/overlays/validity),
  `ProtoRoomFlow` (startup proto + door-crossing proto + promotion + recall + learnt layout save),
  `ExploreAffordance` (`dsr_update_layout_affordance` + `dsr_update_affordance` + `break_execution_stall`),
  `CalibAffordance` (`ensure_calib_node` + `dsr_update_calibration`), `GraphLandmarks` (anchors, table
  landmarks, planner footprints). A thin `ScenePublisher::on_result(const UpdateResult&)` sequences them
  (today's `update` 187-402) and publishes the `SceneSnapshot` for the GUI.
* `MountCalibrator` → `core/mount_reconcile.cpp` (pure `reconcile_mount_nominal`, `mount_pair_update`,
  `loop_closure_observe`) + `agent/mount_publisher.cpp` (`publish_mount_to_graph`, pair log).
* `LidarIngestor` → keeps thread/media; `update_ceiling_cap` and `run_startup_geometry_check` move to
  `core/ceiling_estimator.cpp` (pure statistics, testable).
* `door_apertures.cpp` → `core/door_aperture.h` (geometry, mixture weights) + `agent/door_reader.cpp` (DSR walk).
* `SpecificWorker::initialize` → `agent/wiring.cpp` with one `make_*` per subsystem, in the documented order.
* Viewers draw from `UpdateResult` + `SceneSnapshot` + `LayoutView`; the DSR polls in `viewer_2d.cpp:1623-1782`
  and `camera_visualizer.cpp:360-649` move to `agent/viewer_feeds.cpp`, which hands the viewers plain structs.

### 2.5 Extension paths this buys

* New sensor channel: one `IObservationChannel` + optional `IFactor` + one `Params` struct + one config
  loader file; registered in `Localizer`'s channel list. No edit to `step()`.
* New layout estimator: one `ILayoutEstimator`; `WallSlamChannel` becomes a composition of estimators.
* New room type (e.g. non-Manhattan, curved): a `Model` variant behind `solver/room_model.h` and a
  layout estimator; the polygon contract (`LayoutView`) is unchanged.
* Multi-room / LTSM: `ProtoRoomFlow` already owns crossing; with `ILayoutEstimator` per room and the
  `metaconcept` passage node (memory: `ltsm-abstract-passage-node`), a `RoomRegistry` in the agent layer
  can hold several `Localizer` layouts without touching the localiser.
* Offline bench: links `rc_room_localizer`, calls `Localizer::step`; `tools/tests` lose `lift.py`.

---

## 3. File-by-file split table (every file > 800 lines, plus the 737-line startup)

Sizes are targets (± 15 %). "mech" = move code verbatim; "design" = signature/ownership change.

| File (lines) | New file | Moves (functions / line ranges) | Target | Kind |
|---|---|---|---|---|
| `room_concept.cpp` (9 444) | `localizer/obs_weights.cpp` | anon-namespace weights 24-176 (merge with `room_obs_weights.h`, delete duplicate) | 200 | mech |
| | `localizer/localizer_loop.cpp` | `start/stop` 210-276, `run` 782-1252, 436-533 accessors/ingress, 2316-2334 | 750 | mech → design (step 5) |
| | `localizer/diagnostics.cpp` | `init_debug_log` 276-403, `init_recovery_log` 403-436, `write_debug_tail` 7629-7654, `take_optimizer_timing` 7730-7752, `log_hier_prec_row` 5860-5882, `log_hessian_check` 5950-6035 | 450 | mech |
| | `localizer/relocalizer.cpp` | 533-782 (bootstrap, yaw, validate), 1252-1856 (search, episodes, lost belief, rivals), `evaluate_pose_fit` 3400-3429 | 800 | mech |
| | `localizer/relocalizer_legacy.cpp` | `legacy_grid_search_pose` 1856-2316 | 460 | mech (delete later) |
| | `localizer/channels/wall_slam_channel.cpp` | 2334-2396, 2445-2854 (`wall_slam_observe`, `record_wall_input`, `record_slam_shadow`, map adoption) | 650 | mech |
| | `localizer/channels/wall_slam_after_solve.cpp` | `wall_slam_after_solve` 2854-3214, `reanchor_map_frame` 3214-3276, `set_polygon_room` 3276-3359 | 520 | mech |
| | `localizer/door_weighting.cpp` | 2396-2445 | 60 | mech |
| | `localizer/localizer_step.cpp` | `update` 3510-4603 cut at its own `// =====` markers into `predict_phase`, `build_slot_phase` (3568-3788), `observe_phase` (corner/wall 3788-4027), `solve_phase` (4027-4162), `extract_phase` (4162-4354), `adapt_phase` (4244-4354), `finalize_phase` (4354-4603); `scan_linearize/scan_only_pose` 4871-4948, `compute_velocity_adaptive_weights` 3429-3510 | 800 | design |
| | `localizer/motion_prior.cpp` | 4603-4871, 7443-7538, 7654-7730, 7752-7966, 8663-8808 | 800 | mech |
| | `localizer/odometry_window.cpp` | `integrate_odometry_over_window` 8110-8663 split into IMU-sample preprocessing (8110-8380), preintegration (8380-8560), covariance/stride (8560-8663); `predict_step` 8029-8110, `integrate_velocity_over_window` 7966-8029 | 750 | mech |
| | `localizer/early_exit.cpp` | `try_prediction_early_exit` 5157-5719 (its three diagnostic blocks become helpers), `observe_innovation` 4948-5104 | 750 | mech |
| | `localizer/channels/image_edge_channel.cpp` | `run_image_edge_shadow` 6161-6493, `log_triple_points` 6493-6711, `imgedge_health` 7015-7041 | 600 | mech |
| | `localizer/channels/mount_shadow.cpp` | `mount_pooled_solve` 6711-7015, `observe_kinematic_mount` 5104-5157, `service_calibration` 7538-7629 | 500 | mech |
| | `localizer/solve_gn.cpp` | `run_gn_loop` 6035-6161, `run_gn_shadow` 7041-7156, `compute_posterior_covariance` 7319-7443, `autograd_hessian_3x3` 8844-8872 | 450 | mech |
| | `localizer/solve_shadow.cpp` | `run_adam_loop` 7156-7248, `run_lbfgs_loop` 7248-7319, `shadow_single_step_adam` 8808-8844, boundary hyper-precision 5719-5860, `boundary_weight_now` 5882-5950 | 450 | mech |
| | `localizer/window_manager.cpp` | `WindowManager::*` 8872-9444 | 580 | mech |
| `room_concept.h` (2 668) | `localizer/localizer_params.h` | `Params` 101-784 (then split per §2.2f) | 700 → 9×≤150 | mech → design |
| | `localizer/update_result.h` | `UpdateResult` 784-966, `OdometryPrior` 966-1028, `MotionPriorSelection`, `MotionIngressDebug` | 300 | mech |
| | `localizer/window_types.h` | `WindowSlot` 1400-1484, `BoundaryPrior` 1484-1533, `WindowManager` 1868-1925 | 250 | mech |
| | `localizer/localizer.h` | class with the collaborators of §2.3 | 400 | design |
| | per-collaborator headers | remaining member groups by `// =====` section | ≤300 each | mech |
| `room_scene_graph.cpp` (4 139) | `agent/scene_publisher.cpp` | `update` 187-402, `absorb_reanchor` 1209-1276, `mark_room_rt_not_current`, `published_polygon`, `forget_room`, snapshot | 500 | mech + snapshot (design) |
| | `agent/room_rt_writer.cpp` | `write_robot_room_rt` 450-859 (split: velocity/covariance block 630-760 → helper), `dsr_update_pose` 433, `dsr_publish_predicted_pose` 440, `dsr_publish_body_calibration` 402 | 550 | mech |
| | `agent/proto_room_flow.cpp` | 859-1209 (`step_proto_room[_impl]`, `write_proto_row`), 1372-1932 (startup proto, recall, rearm, promote, learnt save), `stamp_low_obstacles` 1785, `adopt_most_probable_layout` 1815 | 800 | mech |
| | `agent/explore_affordance.cpp` | `dsr_update_layout_affordance` 1932-2278, `dsr_update_affordance` 2782-3168, `break_execution_stall` 3168-3314, `on_affordance_attr_changed`, `update_planner_obstacle_footprints` 3314 | 800 → split again at 2278 if needed | mech |
| | `agent/calib_affordance.cpp` | `ensure_calib_node` 2406-2473, `dsr_update_calibration` 2473-2782 | 400 | mech |
| | `agent/room_node_lifecycle.cpp` | `dsr_create_room_and_reparent` 2278-2393, `re_parent_walls_onto_floor` 3645, `dsr_create_wall_nodes` 3684-3855, `cleanup_room_graph_nodes` 3855, `resolve_overlays_from_graph` 3888, `check_init_graph_is_valid` 4013, `sweep_foreign_rooms` 1313, `resolve_lidar_origin` 1337, `load_robot_body_dimensions_from_graph` 3605 | 650 | mech |
| | `agent/graph_landmarks.cpp` | `log_table_landmarks` 3363, `refresh_object_anchors` 3478, `step_layout_refiner` 4060-4139, `triangulate_polygon` 65-167 | 400 | mech |
| `wall_map.cpp` (4 040) | `layout/wall_map_core.cpp` | 13-424 (helpers, find/index, `to_map`, `update_theta0`, `precisions`, classify, Manhattan), `make_wall/intersect_walls/initialize_rect/update_existence` 723-1003, `merge_indistinguishable` 2787 | 750 | mech |
| | `layout/wall_map_observe.cpp` | `observe` 2064-2787 cut at its own blocks (association 2064-2300, births/direction 2300-2560, refutation+eviction+ghost sweep 2560-2787), `seed_extents_from_polygon` | 750 | mech |
| | `layout/wall_map_edit.cpp` | `splice_out/heal_order/try_splice` 1003-1694, `try_spur_wraps` 1694-1919, `mirror_backed`, `try_down_jumps` 1919-2043 | 800 → `wall_map_splice.cpp` (1003-1694) + `wall_map_spurs.cpp` (1694-2043) | mech |
| | `layout/wall_map_rederive.cpp` | `jump_delta_nats` 2994, `resolve_trial` 3078, `re_derive` 3116-3788 (trace 3116-3350, rectilinearise 3350-3560, judges 3560-3788), `edge_code_nats`, `repair_if_crossing`, `build_from` 3821 | 800 | mech |
| | `layout/wall_map_polygon.cpp` | `decorate` 424-723, `manhattan_polygon*`, `build_polygon_with`, `reanchor` 3885, `beam_loglik/forward_delta/referee` 3971-4040 | 500 | mech |
| | `layout/wall_free_grid.cpp` | `FreeGrid::*`, `frontiers`, `weak_matter`, `free_component` 2836-2994 | 170 | mech |
| `wall_map.h` (975) | `layout/wall_map_params.h` | `Params` 67-490 | 430 | mech |
| | `layout/wall_map_types.h` | 490-640 (`WallLandmark`, `Candidate`, `WallAssoc`, `Corner`, `Polygon`, `BirthInfo`, `FrameResult`, `DeathInfo`) | 160 | mech |
| | `layout/wall_map.h` | class 641-975 | 340 | mech |
| `room_boxes_channel.cpp` (2 368) | `layout/box_path_planner.cpp` | `plan_path` 397-1421, `obstacle_clearance/traversable/model_sdf` 357-397 — it is a planner; give it its own class `BoxPathPlanner(const Layout&, const FreeKeys&)` | 800 (split at the `info_nats` lambda 917 into `box_path_info.cpp`) | design (new class, same maths) |
| | `layout/box_channel.cpp` | `observe` 30-180, `reanchor`, `grow_free_box`, `simplify` 180-338, `observe_obstacles` 338-357, `step` 2137-2368, `open_csv`, `fuse` | 600 | mech |
| | `layout/box_rebuild.cpp` | `reproject` 1421, `snap_coplanar` 1498, `rebuild_from_free` 1563-1947, `layout_components` 1947, `polygon` 1986, `fit_gauge_ml` 2027 | 650 | mech |
| `room_boxes.cpp` (1 297) | `layout/box_geometry.cpp` | 14-427 (SDF, grad, faces, likelihoods, `Layout::inside/sdf/polygon/polygon_cov`) | 420 | mech |
| | `layout/box_register.cpp` | `condense` 427-678, `face_support/faces_observed` 678-760, `register_scan` + MDL 760-1297 | 650 | mech |
| `viewer_2d.cpp` (1 974) | `ui/viewer_2d.cpp` | ctor/view/robot/room/lidar/path/trajectory 26-750 | 730 | mech |
| | `ui/viewer_2d_corners.cpp` | `draw_rgb_corners` 750-923, `draw_corners` 923-1075, `draw_landmark_lines` 1075-1121 | 380 | mech |
| | `ui/viewer_2d_layout.cpp` | `draw_wall_map` 1121-1503, grids 1503-1623 | 500 | mech |
| | `ui/viewer_2d_objects.cpp` | `draw_object_anchors` 1782-1974 + bbox drawing (minus the DSR poll, which goes to `agent/viewer_feeds.cpp` 1612-1782) | 350 | mech + feed (design) |
| `camera_visualizer.cpp` (1 544) | `ui/camera_projection.cpp` (core-pure) | `RoomToCameraBasis` 37-120, `project_points_to_image` 716-767, `predicted_camera_from_room` maths 649-716 (without the DSR lookups) | 250 | design (split DSR out) |
| | `agent/camera_feeds.cpp` | `fetch_rgb_from_dsr` 360, `fetch_camera_intrinsics` 440, `get_room_corners_3d` 486, `get_dsr_object_boxes` 517, `get_dsr_wall_quads` 605 → fill a `CameraSceneFeed` struct | 300 | mech |
| | `ui/camera_visualizer.cpp` | widget, stream, timing 120-360, `update_frame` 1403-1544 | 450 | mech |
| | `ui/camera_overlays.cpp` | `draw_projections` 767-1368 split at its numbered layers (1-3 geometry 767-1007; 3.4-3.4d corner/triple/legend 1007-1224; 3.5-4 walls/objects 1224-1368), `draw_status_overlay` | 650 | mech |
| `room_gn_solver.cpp` (1 477) | `solver/factors_scan.cpp` (Sdf, ImageEdge 45-269), `solver/factors_pose.cpp` (Motion, Boundary, Se2Landmark, LandmarkPrior 269-535), `solver/factors_wall.cpp` (535-760), `solver/gn_build.cpp` (`make_layout`, `build_factors` 787-1238), `solver/gn_solve.cpp` (1238-1477) | | 230/270/230/450/250 | mech |
| `room_config.cpp` (1 399) | `agent/config/config_localizer.cpp` (`RoomConcept.*` 85-129, 490-836), `config_layout.cpp` (`RoomShape.*` 223-388, `ProtoRoom/StartupProto` 129-189), `config_imgedge.cpp` (`ImageEdge.*` 882-934, `ObjectAnchor.*` 836-882), `config_agent.cpp` (`PredictPublish/Media/DSR/Status` 189-223, 388-412, 1115-1139), `config_planner.cpp` (`EpistemicController.*` 1139-1255), `config_overlays.cpp` (412-490, 934-1115, 1255-1399) | | each ≤ 450 | mech (one `load_*` per `Params` struct) |
| `free_space_polygon.h` (1 069) | `core/polygon_ops.h` (57-133), `core/free_space_polygon.h` (133-720 class), `core/free_space_grid.h` (720-870), `core/raster_ops.h` (874-1069 static algorithms) | | 80/590/150/200 | mech |
| `corner_detector.cpp` (1 009) | `core/corner_gather.cpp` (phase 1: 96-736 → `gather_detections()`), `core/corner_associate.cpp` (phase 2+3: 736-1009 → `associate()`, `fold_stats()`), `set_model_corners` stays | | 650/300 | design (two helpers, one state struct) |
| `room_viewer.cpp` (921) | `ui/room_viewer.cpp` (ctor 74-410 trimmed by moving widget building 74-393 into `ui/room_viewer_build.cpp`), rest | | 500/400 | mech |
| `epistemic_planner.cpp` (878) | `layout/planner_candidates.cpp` (16-231 + 758-878 IOR/attempts), `layout/planner_gain.cpp` (231-758 information terms, evaluate, select) | | 350/530 | mech |
| `mount_calibrator.cpp` (838) | `core/mount_reconcile.cpp` (135-310 minus DSR, 581-838), `agent/mount_publisher.cpp` (26-135, 310-581) | | 450/400 | design (DSR out of the pure part) |
| `lidar_ingestor.cpp` (823) | `core/ceiling_estimator.cpp` (`update_ceiling_cap` 385-592, `run_startup_geometry_check` 592-666), `agent/lidar_ingestor.cpp` (rest) | | 300/520 | mech |
| `specificworker_startup.cpp` (737) | `agent/wiring.cpp` (`initialize` 202-485 as `make_*` functions), `agent/startup_presence.cpp` (presence lambda 485-737), shutdown/status 42-202 stays | | 350/300/200 | mech |

Net effect: the 17 oversize files become ~75 files of 150-800 lines; the largest translation unit drops
from 9 444 to ≈ 800 lines (compile memory per `cc1plus` roughly ÷10 for that TU).

---

## 4. Incremental migration plan

Each step leaves the agent building and running; each has a check. Oracle artefacts: (i) one recorded
live run (`tmp/sdf_localizer/*.csv`, `tmp/layout_trace.csv`, `tmp/proto_room.csv`, `tmp/slam_shadow.csv`,
`etc/compute_timing.csv`) compared column-wise; (ii) the five selftests' printed numbers; (iii)
`estimation_scripts` campaign IoU (`regrade_anchor.py`) within the measured CI (v3: .974 [.971, .978]).
Live runs are not bit-reproducible across threads, so (i) is compared as distributions (median/p95 of
`gt_error.csv`, counts of births/deaths); the selftests are the bit-for-bit oracle.

| # | Step | Kind | Check | Effort |
|---|---|---|---|---|
| 0 | Guardrails: `tools/lint_file_lengths.sh` (§5) wired into CMake configure; record oracle artefacts from one run + the five selftests on the current tree; add `-fsanitize` run via `run_asan.sh` as baseline. | mech | lint passes on an allow-list of the 17 current offenders (shrinks each step) | 0.5 d |
| 1 | CMake targets without moving files: `rc_room_core` (INTERFACE, headers), `rc_room_solver`, `rc_room_layout`, `rc_room_localizer`, `rc_room_ui` as STATIC libs over the existing `.cpp`s; agent and all 5 selftests link the libs; delete the hand-listed source lists. Flags per target (no `-march=native`). | mech | all 7 targets build; selftests print identical numbers; `pgrep cc1plus` peak unchanged | 1 d |
| 2 | Split `room_concept.cpp` into the 17 TUs of §3 **without changing the class** (same `room_concept.h`). Merge the duplicate weights into `room_obs_weights.cpp`. | mech | binary-identical selftest output; live run distribution match; TU peak memory drops | 1.5 d |
| 3 | Split `room_concept.h` into `localizer_params.h`, `update_result.h`, `window_types.h`, `room_concept.h`; make `room_gn_solver.h`, `room_obs_weights.h`, `status_reporter.h`, `ground_truth_log.h`, `pose_publisher.h`, `lidar_ingestor.h` include only the small headers. Remove `QDebug` from `camera_calibration.h` (use `rc::status`). | mech | builds; `mount_replay` no longer needs Qt; `lift.py` can be retired for `mount_lidar_pair.h` | 1 d |
| 4 | Fix the scene-graph thread hazard first: `SceneSnapshot` under a mutex, `compute()` reads it; `pinned_landmarks()` returns a copy from the snapshot. Correct the stale comment at `room_concept.cpp:1078`. | design (small) | ASan/TSan run clean on the live agent; no behaviour change | 0.5 d |
| 5 | Split `room_scene_graph.cpp` into the 7 TUs of §3 (still one class). | mech | identical `proto_room.csv`, RT trace, affordance JSONL on a replayed crossing (`estimation_scripts/scripts/round_trips.sh`) | 1 d |
| 6 | Split `wall_map.cpp/.h`, `room_boxes*.cpp`, `room_gn_solver.cpp`, `free_space_polygon.h`, `epistemic_planner.cpp`, `room_config.cpp` (per prefix), `viewer_2d.cpp`, `camera_visualizer.cpp` (TU split only), `specificworker_startup.cpp`, `lidar_ingestor.cpp`. After this step **every file is < 800 lines**. | mech | `wall_slam_selftest`, `gn_selftest`, `layout_explorer_selftest`, `free_space_polygon_selftest` identical; `common/config_report/audit_config_keys.py` reports the same key set; campaign IoU in CI | 3 d |
| 7 | Extract leaf collaborators from `RoomConcept` one per PR, moving members with their methods and mutexes: `DoorWeighting`, `LocalizerDiagnostics` (+ `CsvSink`), `Relocalizer`, `ImageEdgeChannel`/`MountShadow`, `EarlyExitGate`, `WindowManager` (un-nest), `WallSlamChannel`. `RoomConcept` keeps its public API and forwards. | mech-ish (ownership moves, no logic change) | per PR: selftests identical; live distributions match; member count of `RoomConcept` drops (track it) | 6-8 d |
| 8 | `MotionPriorBuilder`: replace the 47 `cyc_*` members by a returned `MotionCycle` struct; `integrate_odometry_over_window` split into its three stages. | design | `gn_selftest` motion section + `motion_calib --selftest` identical; `bench` motion arm (wall_slam_selftest 762-900) identical | 2 d |
| 9 | `Localizer::step(const FrameInputs&)`: lift `update()` out of the thread; `LocalizerLoop` builds `FrameInputs` from the buffers. `RoomConcept` becomes a facade over `LocalizerLoop`. | design | the bench's hand-rolled loop is replaced by `Localizer::step` on the same ray-cast inputs and must reproduce its previous IoU table (that comparison *is* the "bench runs the agent's code" check) | 3-4 d |
| 10 | `IObservationChannel`: adapt Sdf, Corner, ImageEdge, ObjectAnchor, WallSlam, Motion, Boundary; `gn::build_factors` iterates channels. | design | `gn_selftest` Jacobian/finite-difference checks pass per factor; identical loss breakdown CSV | 3 d |
| 11 | `ILayoutEstimator`: wrap `WallMap`, `boxch::Channel`, given-layout; `WallSlamChannel` composes them; `RoomSceneGraph` consumes `LayoutView`. `BoxPathPlanner` becomes its own class. | design | campaign IoU within CI; `layout_trace.csv` identical on replay | 3 d |
| 12 | Agent adapters: `ScenePublisher` + six DSR adapters (from step 5's TUs), `MountPublisher`/`mount_reconcile`, `door_reader`, `viewer_feeds`/`camera_feeds`, `wiring.cpp`. | design (small) | round trips 4/6 or better on `round_trips.sh`; ASan clean | 3 d |
| 13 | Params split (9 structs) + per-struct config loaders; `RoomConfig` keeps overlays. | mech | key audit identical; `etc/config.toml` unchanged | 1.5 d |
| 14 | Tests: unit tests per collaborator under `tools/tests` linking `rc_room_localizer` (EarlyExitGate on recorded rows, MotionPriorBuilder on `motion_calib` logs, DoorWeighting, CsvSink locale); CTest target `make check`. | new | `ctest` green; `lift.py` deleted | 2 d |
| 15 | Delete the `RoomConcept` facade, `relocalizer_legacy.cpp` (if the A/B in `reloc_selftest` still favours the mixture search), Adam/L-BFGS shadows if unused; rename `room_concept.{h,cpp}` away. | design | selftests + campaign unchanged | 1.5 d |

Steps 0-6 (≈ 8 working days) are mechanical and already satisfy the < 800-line goal; steps 7-15
(≈ 25 days) deliver the architecture. Steps 7-8 can run in parallel with 5-6 by two people because they
touch disjoint files after step 2.

---

## 5. Risks and guards

1. **Threading.** Members must move together with the mutex that guards them (`image_edges_mutex_` with
   `latest_image_edges_`, `wall_map_mutex_` with `derived_polygon_`/`map_layout_`, `rival_snapshot_mutex_`
   with the rivals snapshot, `low_obs_mutex_` with `low_obs_`). Rule: a collaborator header states its
   thread affinity; only `ChannelInbox<T>` and `ResultMailbox` may hold locks in `rc_room_localizer`.
   Keep `std::atomic` flags where they are read cross-thread today (`runtime_estimate_`, `map_ready_`,
   `grid_search_active_`, `slam_disagree_`, `measured_ceiling_*`). Verify with a TSan build of the agent
   on a 60 s run (the `ENABLE_ASAN` scaffolding at `src/CMakeLists.txt:72-116` and `run_asan.sh` show how
   to get a sanitizer past Ice's init order). Fix the §0 scene-graph race before moving anything (step 4).
2. **InnerEigen ownership (CLAUDE.md).** `inner_eigen_` is created on the main thread
   (`specificworker_startup.cpp:301`) and handed to `RoomSceneGraph`, which now runs on the localiser
   thread; the ts==0 cache invalidation slots fire on the owner's thread, which is fine, but the adapters
   must keep using the *main-thread-owned* instance and always check the optional. Do not create a
   per-thread instance inside the new `ScenePublisher`.
3. **Init order.** The order in `initialize()` (§1.5) is load-bearing (`PosePublisher` without a scene
   graph logs "NOTHING is being published", `pose_publisher.cpp:456-470`). `wiring.cpp` must preserve it;
   add a startup self-check that asserts every `set_*` hook was installed before `room_concept_.start()`
   (there are two start sites, 485 and 592 — keep both or document why one is enough).
4. **Qt moc.** `Viewer2D`, `RoomViewer`, `CameraVisualizer`, `CalibrationViewer`, `TimeseriesPlot` are
   `QObject`s. The `rc_room_ui` static library needs `set_target_properties(rc_room_ui PROPERTIES AUTOMOC ON)`
   and the generated `ui_localUI.h` include path; keep `custom_widget.h` listed as a source as today
   (`src/CMakeLists.txt:97`). Signals/slots between `Viewer2D` and `SpecificWorker` stay queued/auto
   (`specificworker_startup.cpp:682-683`).
5. **DSR signals.** Only one connect exists (`update_node_attr_signal`, `specificworker_startup.cpp:466`,
   default = queued). It stays in `agent/`; libraries below must not include `dsr/`. Enforce with
   `target_link_libraries` scoping and a grep in the lint script (`dsr/|DSR::|G->` forbidden under
   `core/ solver/ layout/ localizer/ ui/`).
6. **Compile memory (-j8 rule).** Splitting the 9 444-line TU reduces the per-`cc1plus` peak; the risk
   is the opposite direction — a new umbrella header that pulls torch into many TUs. Keep `torch/torch.h`
   behind `solver/` headers only; `UpdateResult` and channel headers carry no tensors (today
   `room_concept.h` has 47 `torch::` references — move the tensor members into the collaborators that use
   them). Check with `pgrep -a cc1plus` + `/usr/bin/time -v` on `localizer_step.cpp`.
7. **Behaviour drift.** Mechanical steps are diffed on selftest output; design steps on the campaign IoU
   CI and the recorded-run distributions. Keep seeds (`wall_rng_{12345}`, `room_concept.h:2120`).
   Known trap: the bench and the live agent disagree when one reads a file the other wrote
   (locale) — all new readers use `std::from_chars`, all writers `imbue(std::locale::classic())`;
   `CsvSink` enforces the latter in one place.
8. **ABI.** Static libs inherit no flags automatically; set `-fno-char8_t -O3 -g` per target, never
   `-march=native` or `EIGEN_MAX_ALIGN_BYTES`; after step 1 grep every `flags.make` under `build/`.
9. **Tools that bypass CMake** (the `g++ … -Isrc` one-liners): their include paths change with the
   directory layout. Keep `src/` as an include root and add forwarding headers for one release, or update
   the 20 header comments in the same PR as the move.
10. **Scope creep.** Steps 7-12 will expose dead code (legacy reloc, Adam/L-BFGS shadows, the differential
    test, `RecordWallInput`); delete only in step 15 after `reloc_selftest` and `gn_selftest` confirm.

### Lint / CI rule

`tools/lint_file_lengths.sh` (run at CMake configure via `execute_process(... RESULT_VARIABLE)` → `message(FATAL_ERROR)`,
and as a git `pre-commit` hook):

```bash
#!/usr/bin/env bash
# Fail if any source file exceeds 800 lines, any function exceeds 300, or a lower layer touches DSR/Qt widgets.
set -euo pipefail; cd "$(dirname "$0")/.."
bad=0
while read -r n f; do (( n > 800 )) && { echo "TOO LONG: $f ($n)"; bad=1; }; done \
  < <(find src -name '*.cpp' -o -name '*.h' | xargs wc -l | grep -v total)
# layering: nothing below agent/ may name DSR or widgets
grep -rnE 'dsr/|DSR::|G->|QWidget|QGraphics' src/core src/solver src/layout src/localizer 2>/dev/null && bad=1
# function length (brace-depth scan; python3 -I, no deps)
python3 -I tools/lint_function_lengths.py 300 src || bad=1
exit $bad
```

During migration the script reads an allow-list (`tools/lint_allow.txt`) that lists the current 17
offenders with their sizes; a file may only shrink, and the list must be empty by step 6.
