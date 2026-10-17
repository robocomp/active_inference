cmake_policy(SET CMP0074 NEW)

# Keep generated CMake changes from pushing an unsupported host dialect when templates are regenerated.
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD GREATER 23)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Xml PrintSupport REQUIRED)
find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras QUIET)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# AddressSanitizer (opt-in, separate build dir) — declared BEFORE add_subdirectory() so the
# ai_common targets are instrumented too; behind a library boundary they no longer pick this up
# from the directory-scope add_compile_options further down.
option(ENABLE_ASAN "Build room_concept with AddressSanitizer" OFF)

# Shared ai_common::* libraries (see common/CMakeLists.txt). room_concept has its OWN
# src/timeseries_plot.cpp, so it deliberately does not link ai_common::ui.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane (was 10 spliced IDL paths + an include_directories)
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::affordance   # affordance_manager
  ai_common::robot        # base CAPABILITY off the robot node — bounds the pose clamp
  ai_common::core         # header-only beliefs/geometry (object_anchor/, occlusion, …)
  ai_common::status       # rc::StatusStream: live status socket + events file for tools/room_tui.py
)

# ── Sources ────────────────────────────────────────────────────────────────────
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_slots.cpp
  ../src/specificworker_startup.cpp
  ../src/mount_calibrator.cpp
  ../src/pose_publisher.cpp
  ../src/calib_channels.cpp
  ../src/ground_truth_log.cpp
  ../src/specificworker_presence.cpp
  ../src/status_reporter.cpp
  ../src/viewer_2d.cpp
  ../src/svg_room_loader.cpp
  ../src/epistemic_controller.cpp
  ../src/room_scene_graph.cpp
  ../src/layout_explorer.cpp
  ../src/lidar_ingestor.cpp
  ../src/imu_ingestor.cpp
  ../src/room_viewer.cpp
  ../src/room_config.cpp
  ../src/timeseries_plot.cpp
  ../src/calibration_viewer.cpp
  ../src/object_anchor_source.cpp
  ../src/camera_ingestor.cpp
  ../src/camera_stream_hub.cpp
  ../src/camera_visualizer.cpp
  ../src/custom_widget.h
)

LIST(APPEND LIBS ${LIBS} Qt6PrintSupport Qt6Xml
            Boost::system Boost::filesystem)

# ── rc_room_estimator: the estimator, compiled ONCE ─────────────────────────────
# Refactor step 1 (docs/refactor_proposal_2026-10-13.md). These sources used to be listed in the agent AND
# hand-listed again in every selftest, so room_concept.cpp (9.4k lines, the heaviest TU) was compiled up to
# five times, and a new .cpp in the agent was a silent link error in whichever selftest was not updated
# (gn_selftest was unlinkable that way for days, 2026-09-13). Now the agent and every selftest link this
# one library: they cannot drift apart, and the bench runs exactly the code the agent runs.
# Pure estimator code: no Q_OBJECT, no DSR calls (door_apertures only includes DSR headers).
set(RC_ESTIMATOR_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/room_gn_solver.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_model.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_obs_weights.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_diagnostics.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_reloc_bootstrap.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_reloc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_wall_slam.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_wall_slam_after.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_update.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_update_solve.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_motion_prior.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_prediction.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_odometry_window.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_early_exit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_solve.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_image_edge.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_mount.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_concept_window.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reloc_search.cpp
  ${CMAKE_CURRENT_LIST_DIR}/door_apertures.cpp
  ${CMAKE_CURRENT_LIST_DIR}/object_anchor_factor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/image_edge_factor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/image_edge_source.cpp
  ${CMAKE_CURRENT_LIST_DIR}/corner_detector.cpp
  ${CMAKE_CURRENT_LIST_DIR}/wall_segmenter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/wall_map.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_boxes.cpp
  ${CMAKE_CURRENT_LIST_DIR}/room_boxes_channel.cpp
  ${CMAKE_CURRENT_LIST_DIR}/pointcloud_center_estimator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/rerun_logger.cpp
  ${CMAKE_CURRENT_LIST_DIR}/epistemic_planner.cpp
)
set(RC_EXTERNAL_LIBS ${LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash, e.g. camera_visualizer) or misaligns it (SIGBUS). libdsr is built
# with default Eigen alignment; the agents must too.
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t)

if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for room_concept (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()

# Defined AFTER add_definitions()/ENABLE_ASAN above: those are directory-scoped and only reach targets
# created after them — the library must be compiled with exactly the agent's flags (Eigen ABI, see above).
add_library(rc_room_estimator STATIC ${RC_ESTIMATOR_SOURCES})
target_include_directories(rc_room_estimator PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(rc_room_estimator PUBLIC ${RC_EXTERNAL_LIBS})
set(LIBS rc_room_estimator ${RC_EXTERNAL_LIBS})

# ── Refactor guardrail: file/function length + layering lint ──────────────────
# docs/refactor_proposal_2026-10-13.md §5. Grandfathered offenders live in tools/lint_allow.txt and may
# only shrink. At configure time a violation is a WARNING (so ongoing feature work is never blocked by a
# reconfigure); `make -C build room_lint` is the hard check, run before committing a refactor step.
set(RC_ROOM_ROOT ${CMAKE_CURRENT_LIST_DIR}/..)
execute_process(COMMAND ${RC_ROOM_ROOT}/tools/lint_file_lengths.sh
                OUTPUT_VARIABLE _rc_lint_out ERROR_VARIABLE _rc_lint_out RESULT_VARIABLE _rc_lint_rc)
if(NOT _rc_lint_rc EQUAL 0)
  message(WARNING "room_concept lint (tools/lint_file_lengths.sh):\n${_rc_lint_out}")
endif()
add_custom_target(room_lint COMMAND ${RC_ROOM_ROOT}/tools/lint_file_lengths.sh VERBATIM)

# ── gn_selftest ───────────────────────────────────────────────────────────────
# Offline validation of the Gauss-Newton backend (analytic Jacobians vs finite differences, and
# recovery of a known pose on a synthetic room). EXCLUDE_FROM_ALL: never built by a plain `make`,
# so it cannot affect the agent build. Build it with:  make -C build gn_selftest && ./bin/gn_selftest
add_executable(gn_selftest EXCLUDE_FROM_ALL
  ../tools/gn_selftest.cpp   # the estimator comes from rc_room_estimator (via LIBS)
)
target_include_directories(gn_selftest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(gn_selftest ${LIBS})
set_target_properties(gn_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── mp3d_replay ───────────────────────────────────────────────────────────────
# Drives the estimator from RECORDED MP3D-FPE scans (real Matterport depth + real trajectories)
# instead of simulated ones, and scores the closed polygon against the annotated corners.
# EXCLUDE_FROM_ALL. Build and run with:
#   make -C build mp3d_replay && ../bin/mp3d_replay datasets/mp3d_fpe
add_executable(mp3d_replay EXCLUDE_FROM_ALL
  ../tools/mp3d_replay.cpp
)
target_include_directories(mp3d_replay PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(mp3d_replay ${LIBS})
set_target_properties(mp3d_replay PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── reloc_selftest ────────────────────────────────────────────────────────────
# Offline A/B of relocalisation on INJECTED lost-robot episodes (flip, mirror, drift, uniform kidnap):
# today's grid_search_initial_pose() vs rc::reloc::search(). EXCLUDE_FROM_ALL. Build and run with:
#   make -C build reloc_selftest && ../bin/reloc_selftest
add_executable(reloc_selftest EXCLUDE_FROM_ALL
  ../tools/reloc_selftest.cpp
)
target_include_directories(reloc_selftest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(reloc_selftest ${LIBS})
set_target_properties(reloc_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── corner_gather_test ────────────────────────────────────────────────────────
# Offline check that the layout-wide explain-away removes a perpendicular face from a wall fit
# WITHOUT letting a collinear neighbour starve it (see the file header for the two live failures it
# isolates). EXCLUDE_FROM_ALL. Build and run both arms with:
#   make -C build corner_gather_test && ./bin/corner_gather_test \
#     && RC_CORNER_EXPLAIN_AWAY=1 ../bin/corner_gather_test
add_executable(corner_gather_test EXCLUDE_FROM_ALL
  ../tools/corner_gather_test.cpp
)
target_include_directories(corner_gather_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(corner_gather_test ${LIBS})
set_target_properties(corner_gather_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── wall_slam_selftest ────────────────────────────────────────────────────────
# Offline validation of the wall-SLAM layer (wall_segmenter + wall_map + the wall factor family in
# room_gn_solver): segmenter recall on ray-cast scans of a known L-shaped room, analytic Jacobians of
# the new factors vs finite differences, and the full estimate→closure→republish loop with pose and
# polygon error against the synthetic truth. EXCLUDE_FROM_ALL like gn_selftest. Build with:
#     make -C build wall_slam_selftest && ./bin/wall_slam_selftest
add_executable(wall_slam_selftest EXCLUDE_FROM_ALL
  ../tools/wall_slam_selftest.cpp   # the estimator comes from rc_room_estimator (via LIBS)
)
# ── THE BENCH DRIVES THE AGENT'S OWN FOLLOWER ────────────────────────────────────────────────
# `pursue()` was a six-line P controller written for the hand tour, whose waypoints were safe by
# construction. It steers at a POINT, never strafes, and knows nothing about walls — and once the
# bench grew a collision model it turned out to drive at a wall in 88 of 100 rooms.
# The controller agent already solved path following: RouteSpline + RouteFollower + PlainTracker,
# whose headers document the laps it cost to get right. They are pure Eigen/STL — "no Qt, no DSR"
# by their own top-of-file claim, verified — so the bench can link them directly instead of growing
# a second, worse follower. `controller_motion_commander` is the only DSR-bound piece and the bench
# does not need it: BaseModel::execute already applies the real hardware limits.
set(CTRL_SRC ${CMAKE_CURRENT_LIST_DIR}/../../controller/src)
# The WHOLE control chain, not a tracker in isolation: TrajectoryController owns the route, the
# ESDF it builds from the scan, the blockage/alignment logic and the Plain->PD fallback, and it IS
# the PathWorld/FieldWorld the trackers query. 1757 lines, zero DSR/Qt references.
target_sources(wall_slam_selftest PRIVATE
  ${CTRL_SRC}/route_optimizer.cpp
  ${CTRL_SRC}/route_spline.cpp
  ${CTRL_SRC}/route_follower.cpp
  ${CTRL_SRC}/trajectory_controller.cpp
  ${CTRL_SRC}/trackers/plain_tracker.cpp
  ${CTRL_SRC}/trackers/pd_tracker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../common/robot_footprint/robot_footprint.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../../common/robot_footprint/mesh_hull.cpp)
target_include_directories(wall_slam_selftest PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src ${CTRL_SRC})
target_link_libraries(wall_slam_selftest ${LIBS})
set_target_properties(wall_slam_selftest PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── motion_calib ──────────────────────────────────────────────────────────────
# Calibrates the preintegrated motion model's noise DENSITIES and SCALE errors from a logged run, by
# the same weighted least squares that maximises the evidence for those parameters. Standalone on
# purpose — it links nothing from the agent (no torch, no DSR), so it stays a seconds-long build and
# can be pointed at any archived log. EXCLUDE_FROM_ALL.
#   make -C build motion_calib && ../bin/motion_calib --selftest
#   make -C build motion_calib && ../bin/motion_calib tmp/sdf_localizer/<log>.csv
add_executable(motion_calib EXCLUDE_FROM_ALL ../tools/motion_calib.cpp)
target_include_directories(motion_calib PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
set_target_properties(motion_calib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)

# ── mount_replay ──────────────────────────────────────────────────────────────
# Arm 7 (VALIDATION_THREE_DEVICE_CORNERS §2b): replays a recorded drive under a deliberately wrong
# extrinsic, so the four injection legs are four analyses of ONE drive. Standalone like motion_calib
# — it includes mount_lidar_pair.h, which is Eigen + std only, so it links no torch and no DSR and
# builds in seconds. It calls the AGENT'S estimator rather than a replica of it, on purpose.
#   make -C build mount_replay && ../bin/mount_replay --help
add_executable(mount_replay EXCLUDE_FROM_ALL ../tools/mount_replay.cpp)
target_include_directories(mount_replay PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
# ★ QtCore ONLY, and only because camera_calibration.h logs through QDebug. The point of this target
#   is that it calls the AGENT'S OWN estimator rather than a replica of it, and the rebase-on-adopt
#   selftest needs rc::camcal::Estimator itself — the decision about whether a nominal change is
#   expressible as a mount correction lives there. QtCore is cheap; no widgets, no DSR, no torch, so
#   the target still builds in seconds.
find_package(Qt6 COMPONENTS Core REQUIRED)
target_link_libraries(mount_replay PRIVATE Qt6::Core)
set_target_properties(mount_replay PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bin)
