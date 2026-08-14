set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)
find_package(OpenCV REQUIRED)

# Shared ai_common::* libraries (see common/CMakeLists.txt).
# NOTE ai_common::status_overlay is deliberately NOT linked: it is the one target that pulls dsr_gui,
# whose AbstractGraphicViewer collides with a component's local viewer.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence         # agent presence monitor + coordinator
  ai_common::media          # LiDAR media plane (domain 7) — the one input not in the graph
  ai_common::core
  ${OpenCV_LIBS}
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/viewer_config.cpp
  ../src/scene_feed.cpp
  ../src/voxel_opengl_viewer.cpp
  ../src/obj_loader.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default.
# This agent calls get_transformation_matrix on every refresh, so it is squarely exposed.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
