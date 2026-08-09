set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane (was 10 spliced IDL paths + an include_directories)
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::perception   # mask_ingestor
  ai_common::ui           # TimeSeriesPlot (compiled Q_OBJECT)
  ai_common::ui_headers   # custom_widget.h / evidence_monitor.h / belief_inspector.h (header-only)
  ai_common::core         # header-only beliefs (ai_belief/dof_spec.h, …)
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/specificworker_lifecycle.cpp
  ../src/specificworker_dashboard.cpp
  ../src/cabinet_config.cpp
  ../src/cabinet_scene_graph.cpp
  ../src/cabinet_lidar_ingestor.cpp
  ../src/cabinet_existence.cpp
  ../src/cabinet_fitter.cpp
  ../src/cabinet_projection.cpp
  ../src/cabinet_lidar_range_channel.cpp
  ../src/cabinet_model.cpp
  ../src/cabinet_belief.cpp
  ../src/epistemic_planner.cpp
  ../src/cabinet_affordance.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
