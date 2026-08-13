set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Shared ai_common::* libraries (see common/CMakeLists.txt).
# NOTE ai_common::status_overlay is deliberately NOT linked here: it is the one target that pulls
# dsr_gui, whose AbstractGraphicViewer collides with this component's local one. ai_common::graph3d
# depends on dsr_api/dsr_core only, which is exactly what the scene builder needs.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::graph3d      # graph3d_scene_builder (stratified 3D graph view)
  ai_common::obj          # obj_loader for the display meshes
  ai_common::ui_headers   # viewers/gl_graph3d_viewer.h — header-only, mocced by OUR autogen
  ai_common::core         # graph3d/graph3d_scene.h (header-only)
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/viewer_config.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
# This agent calls it on every rebuild, so it is squarely exposed.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
