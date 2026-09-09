set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane — the ZED RGB the contour check scores against
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::perception   # mask_ingestor
  ai_common::ui           # TimeSeriesPlot (compiled Q_OBJECT)
  ai_common::ui_headers   # custom_widget.h / belief_inspector.h (header-only)
  ai_common::core         # header-only beliefs (ai_belief/dof_spec.h, …)
  ai_common::affordance   # affordance protocol, producer side
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/door_config.cpp
  ../src/door_scene_graph.cpp
  ../src/door_fitter.cpp

  ../src/door_semantic_field.cpp
  ../src/door_actuator.cpp
  ../src/door_world_registration.cpp
  ../generated/DoorControl.cpp
  ../../common/contour_edge/contour_edge_check.cpp
  ../../common/rgb_ingestor/rgb_ingestor.cpp
  ../src/door_model.cpp
  ../src/door_belief.cpp
  ../src/epistemic_planner.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

# OpenCV — the RGB contour check (common/contour_edge) + the shared ZED RGB ingestor. Confined to those
# two units; the pure-Eigen belief stays OpenCV-free. ⚠common/contour_edge DOES include <opencv2/...>,
# which breaks the old "nothing under common/ includes OpenCV" invariant hood's CMake notes rely on —
# any component linking ai_common's contour_edge must therefore pull OpenCV in, as here.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})

# Ice — RoboCompDoorControl client (generated/DoorControl.cpp + src/door_actuator.cpp). door_concept had
# NO Ice communications before this; its .cdsl still declares an empty Communications block, and the
# stubs were produced with slice2cpp from robocomp_interfaces/IDSLs/DoorControl.idsl rather than by
# regenerating the component (regeneration would rewrite genericworker/main and is not worth it for one
# client proxy).
LIST(APPEND LIBS ${LIBS} Ice++11)
