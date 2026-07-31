set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/door_config.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ../src/door_scene_graph.cpp
  ../src/door_fitter.cpp
  ../src/door_model.cpp
  ../src/door_belief.cpp
  ../src/epistemic_planner.cpp
  ../src/door_affordance.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/door_config.h
  ../src/door_instance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.h
  ../src/door_scene_graph.h
  ../src/door_fitter.h
  ../src/door_geometry.h
  ../src/door_model.h
  ../src/door_belief.h
  ../src/door_dof.h
  ../src/epistemic_planner.h
  ../src/door_affordance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/ai_belief/dof_spec.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/custom_widget.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/belief_inspector.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NEITHER -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: both change sizeof(optional<Eigen::Transform>)
# away from libdsr's default alignment (144) → get_transformation_matrix overruns the caller's stack
# slot → 0x17ff… SIGBUS in whatever agent shares the graph. Match libdsr's default. (See crash notes.)
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
