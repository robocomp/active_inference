set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/viewer_config.cpp
  # Stratified 3D graph view: z = abstraction stratum, x/y = true metric pose. The builder depends
  # on dsr_api ONLY (never dsr_gui — their AbstractGraphicViewer collides with the component-local
  # one), so it stays includable from anywhere; the renderer is header-only in common/viewers.
  ${ACTIVE_INFERENCE_COMMON_DIR}/graph3d/graph3d_scene_builder.cpp
  # Display meshes named by the nodes' mesh_path attr, loaded by gl_graph3d_viewer.h.
  ${ACTIVE_INFERENCE_COMMON_DIR}/obj/obj_loader.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/viewer_config.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/graph3d/graph3d_scene.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/graph3d/graph3d_scene_builder.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/viewers/gl_graph3d_viewer.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
# This agent calls it on every rebuild, so it is squarely exposed.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
