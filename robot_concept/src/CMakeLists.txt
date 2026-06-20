find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../../common/agent_presence_monitor/agent_presence_monitor.cpp
  ../../common/agent_presence_coordinator/agent_presence_coordinator.cpp
  ../../common/media_transport/media_transport.cpp
  ../../common/media_transport/generated/idl/image_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../../common/agent_presence_monitor/agent_presence_monitor.h
  ../../common/agent_presence_coordinator/agent_presence_coordinator.h
  ../../common/media_transport/media_transport.h
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

