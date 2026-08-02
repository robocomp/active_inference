set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane (was 10 spliced IDL paths + an include_directories)
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::geometry     # robot_footprint + robust_metrics (pure Eigen; also used by tools/)
  ai_common::affordance   # affordance_manager
  ai_common::ui           # TimeSeriesPlot (compiled Q_OBJECT)
  ai_common::core         # header-only geometry/beliefs
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/controller_display.cpp
  ../src/controller_motion_commander.cpp
  ../src/controller_obstacle_model.cpp
  ../src/controller_obstacle_tracker.cpp
  ../src/controller_session.cpp
  ../src/controller_world_model.cpp
  ../src/controller_mission.cpp
  ../src/controller_mission_panel.cpp
  ../src/route_spline.cpp
  ../src/route_follower.cpp
  ../src/route_optimizer.cpp
  ../src/grid_planner.cpp
  ../src/trajectory_controller.cpp
  ../src/viewer_2d.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
