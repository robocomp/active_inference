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
  ../src/room_path_planner.cpp
  ../src/trajectory_controller.cpp
  ../src/viewer_2d.cpp
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
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.h
  ../src/custom_widget.h
  ../src/room_path_planner.h
  ../src/viewer_2d.h
)


LIST(APPEND LIBS ${LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t  -DEIGEN_MAX_ALIGN_BYTES=0)

