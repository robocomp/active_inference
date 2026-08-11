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
  ai_common::affordance   # affordance protocol, producer side
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/specificworker_lifecycle.cpp
  ../src/specificworker_dashboard.cpp
  ../src/hood_config.cpp
  ../src/hood_scene_graph.cpp
  ../src/hood_lidar_ingestor.cpp
  ../src/hood_existence.cpp
  ../src/hood_fitter.cpp
  ../src/hood_projection.cpp
  ../src/hood_rgb_ingestor.cpp
  ../src/hood_lidar_range_channel.cpp
  ../src/hood_model.cpp
  ../src/hood_belief.cpp
  ../src/epistemic_planner.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

# OpenCV — appearance-based FRONT (door) detection (HoodProjection::detect_front + the ZED RGB ingestor).
# Confined to hood_projection.cpp / hood_rgb_ingestor.cpp; the pure-Eigen belief stays OpenCV-free.
# Nothing under common/ includes <opencv2/...>, so this stays component-local.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})
