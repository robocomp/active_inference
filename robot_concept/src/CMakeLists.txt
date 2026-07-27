find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../../common/agent_presence_monitor/agent_presence_monitor.cpp
  ../../common/agent_presence_coordinator/agent_presence_coordinator.cpp
  # Observer half of the agent-health display: greys out agent nodes whose heartbeat went stale.
  # robot_concept only — it is the agent that shows the complete graph. Its .cpp confines the
  # dsr_gui includes to one TU (their AbstractGraphicViewer collides with the component-local one).
  ../../common/agent_state_publisher/agent_status_overlay.cpp
  # Stratified 3D graph view: z = abstraction stratum, x/y = true metric pose. The builder depends on
  # dsr_api ONLY (never dsr_gui — its AbstractGraphicViewer collides with the component-local one),
  # so it stays includable from anywhere; the renderer is header-only in common/viewers.
  ../../common/graph3d/graph3d_scene_builder.cpp
  ../../common/media_transport/media_transport.cpp
  ../../common/media_transport/generated/idl/image_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/image360_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image360_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../../common/obj/obj_loader.cpp
  ../src/sensor_media_publisher.cpp
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../../common/agent_presence_monitor/agent_presence_monitor.h
  ../../common/agent_presence_coordinator/agent_presence_coordinator.h
  ../../common/agent_state_publisher/agent_status.h
  ../../common/agent_state_publisher/agent_state_publisher.h
  ../../common/agent_state_publisher/agent_status_overlay.h
  ../../common/graph3d/graph3d_scene.h
  ../../common/graph3d/graph3d_scene_builder.h
  ../../common/viewers/gl_graph3d_viewer.h
  ../../common/media_transport/media_transport.h
  ../src/sensor_media_publisher.h
  ../src/specificworker.h
)

# Media transport plane: zero-copy DDS for RGBD payloads outside the DSR graph.
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash) or misaligns it (SIGBUS). libdsr uses default Eigen alignment; match it.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

