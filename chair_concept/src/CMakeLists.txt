set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/chair_config.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ../src/chair_scene_graph.cpp
  ../src/chair_fitter.cpp
  ../src/chair_model.cpp
  ../src/chair_belief.cpp
  ../src/epistemic_planner.cpp
  ../src/chair_affordance.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/chair_config.h
  ../src/chair_instance.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/mask_ingestor/mask_ingestor.h
  ../src/chair_scene_graph.h
  ../src/chair_fitter.h
  ../src/chair_model.h
  ../src/chair_belief.h
  ../src/epistemic_planner.h
  ../src/chair_affordance.h
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
