set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ../src/room_path_planner.cpp
  ../src/trajectory_controller.cpp
  ../src/viewer_2d.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ../src/custom_widget.h
  ../src/room_path_planner.h
  ../src/viewer_2d.h
)


LIST(APPEND LIBS ${LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

