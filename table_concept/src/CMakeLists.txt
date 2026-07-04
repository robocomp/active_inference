set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/table_config.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ../src/table_scene_graph.cpp
  ../src/table_lidar_ingestor.cpp
  # Zero-copy media plane (lidar3D consumer for the YOLO-independent range factor). Mirrors bottle_concept:
  # media_transport.cpp is type-agnostic but references all IDL frame types, so link every PubSubTypes pair.
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../src/table_fitter.cpp
  ../src/table_model.cpp
  ../src/table_belief.cpp
  ../src/epistemic_planner.cpp
  ../src/table_affordance.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/table_config.h
  ../src/table_instance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.h
  ../src/table_scene_graph.h
  ../src/table_fitter.h
  ../src/table_model.h
  ../src/table_belief.h
  ../src/epistemic_planner.h
  ../src/table_affordance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/custom_widget.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

# Media-plane headers (media_transport.h + generated IDL) — the lidar3D consumer factory.
include_directories(${ACTIVE_INFERENCE_COMMON_DIR}/media_transport)

