set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Shared ai_common::* libraries (see common/CMakeLists.txt).
# robot_concept is the only consumer of ai_common::status_overlay — it is the agent that shows the
# complete graph. That .cpp confines the dsr_gui includes to one TU (their AbstractGraphicViewer
# collides with the component-local one), and dsr_gui is a bare link name with no interface include
# dirs, so linking the target does not leak those headers into this component's own TUs.
# (The stratified 3D graph view moved to its own agent: active_inference/scene_graph_viewer.)
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media           # zero-copy media plane for RGBD payloads outside the DSR graph
  ai_common::presence        # agent presence monitor + coordinator
  ai_common::status_overlay  # agent-health node tinting (the only dsr_gui consumer under common/)
  ai_common::obj             # obj_loader for the display meshes
  ai_common::ui_headers      # common/viewers/gl_*.h — header-only, mocced by OUR autogen
  ai_common::core            # agent_state_publisher/agent_status.h (header-only)
)

# Sources set
LIST(APPEND SOURCES
  ../src/sensor_media_publisher.cpp
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash) or misaligns it (SIGBUS). libdsr uses default Eigen alignment; match it.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
