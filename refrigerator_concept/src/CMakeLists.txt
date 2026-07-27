set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/specificworker_lifecycle.cpp
  ../src/specificworker_dashboard.cpp
  ../src/refrigerator_config.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ../src/refrigerator_scene_graph.cpp
  ../src/refrigerator_lidar_ingestor.cpp
  ../src/refrigerator_existence.cpp
  # Zero-copy media plane (lidar3D consumer for the YOLO-independent range factor). Mirrors bottle_concept:
  # media_transport.cpp is type-agnostic but references all IDL frame types, so link every PubSubTypes pair.
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/lidar_plane_reader.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../src/refrigerator_fitter.cpp
  ../src/refrigerator_projection.cpp
  ../src/refrigerator_rgb_ingestor.cpp
  ../src/refrigerator_lidar_range_channel.cpp
  ../src/refrigerator_model.cpp
  ../src/refrigerator_belief.cpp
  ../src/epistemic_planner.cpp
  ../src/refrigerator_affordance.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/refrigerator_config.h
  ../src/refrigerator_instance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.h
  ../src/refrigerator_scene_graph.h
  ../src/refrigerator_fitter.h
  ../src/refrigerator_projection.h
  ../src/refrigerator_rgb_ingestor.h
  ../src/refrigerator_lidar_range_channel.h
  ../src/refrigerator_model.h
  ../src/refrigerator_belief.h
  ../src/refrigerator_dof.h
  ../src/epistemic_planner.h
  ../src/refrigerator_affordance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/ai_belief/dof_spec.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/custom_widget.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/evidence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/belief_inspector.h
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

# OpenCV — appearance-based FRONT (door) detection (RefrigeratorProjection::detect_front + the ZED RGB ingestor).
# Confined to refrigerator_projection.cpp / refrigerator_rgb_ingestor.cpp; the pure-Eigen belief stays OpenCV-free.
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})

