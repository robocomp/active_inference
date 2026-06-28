set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")
set(HUMAN_CORE_DIR              "${CMAKE_CURRENT_LIST_DIR}/../cpp/core")

cmake_policy(SET CMP0074 NEW)

if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD GREATER 23)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# human_concept is Eigen-only (no libtorch / CUDA): the generative model is the
# validated Eigen kinematic-model + Laplace estimator in cpp/core, and the belief
# stabiliser is header-only. So none of bottle/table's torch/robust_metrics wiring.
include_directories(${HUMAN_CORE_DIR})

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/human_config.cpp
  ../src/skeleton_source.cpp
  ../src/human_scene_graph.cpp
  ../src/human_fitter.cpp
  ../src/epistemic_planner.cpp
  ../src/human_affordance.cpp
  ../src/prior_store.cpp
  ${HUMAN_CORE_DIR}/human_kinematic_model.cpp
  ${HUMAN_CORE_DIR}/vfe_inference.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set (listed for AUTOMOC)
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/human_config.h
  ../src/human_instance.h
  ../src/skeleton_source.h
  ../src/human_scene_graph.h
  ../src/human_fitter.h
  ../src/epistemic_planner.h
  ../src/human_affordance.h
  ../src/prior_store.h
  ${HUMAN_CORE_DIR}/body18.h
  ${HUMAN_CORE_DIR}/human_kinematic_model.h
  ${HUMAN_CORE_DIR}/vfe_inference.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/timeseries_plot.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/dashboard/custom_widget.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/belief_stabilizer/belief_stabilizer.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_protocol/affordance_protocol.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)

# Force-keep libdsr_core (see bottle_concept rationale: --as-needed drops it on a light-DSR build).
add_link_options(-Wl,--no-as-needed)

# CRITICAL — match the prebuilt libdsr Eigen ABI (default alignment): NEITHER -march=native NOR
# -DEIGEN_MAX_ALIGN_BYTES (either changes sizeof(optional<Eigen::Transform>) → SIGBUS/stack-smash).
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
