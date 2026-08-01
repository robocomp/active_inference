set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/controller_display.cpp
  ../src/controller_motion_commander.cpp
  ../src/controller_obstacle_model.cpp
  ../src/controller_obstacle_tracker.cpp
  ../src/controller_session.cpp
  ../src/controller_world_model.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.cpp
  ../src/controller_mission.cpp
  ../src/controller_mission_panel.cpp
  ../src/route_spline.cpp
  ../src/route_follower.cpp
  ../src/route_optimizer.cpp
  ../src/grid_planner.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robot_footprint/robot_footprint.cpp
  ../src/trajectory_controller.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ../src/viewer_2d.cpp
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
)

# Headers set
LIST(APPEND HEADERS
  ../src/controller_display.h
  ../src/controller_motion_commander.h
  ../src/controller_obstacle_model.h
  ../src/controller_obstacle_tracker.h
  ../src/controller_runtime_types.h
  ../src/controller_session.h
  ../src/controller_world_model.h
  ../src/specificworker.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.h
  ../src/custom_widget.h
  ../src/controller_mission.h
  ../src/controller_mission_panel.h
  ../src/route_spline.h
  ../src/route_follower.h
  ../src/route_optimizer.h
  ../src/grid_planner.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robot_footprint/robot_footprint.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.h
  ../src/viewer_2d.h
)


LIST(APPEND LIBS ${LIBS})

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

