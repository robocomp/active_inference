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

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed after the C++23 clamp above so
# the ai_common targets build at the same standard as the rest of this component (their
# target_compile_features is a FLOOR of cxx_std_23, so the clamp is honoured, not overridden).
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::ui           # TimeSeriesPlot (compiled Q_OBJECT)
  ai_common::ui_headers   # custom_widget.h (header-only)
  ai_common::core         # belief_stabilizer.h / affordance_protocol.h (header-only)
)

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
)

# Force-keep libdsr_core (see bottle_concept rationale: --as-needed drops it on a light-DSR build).
add_link_options(-Wl,--no-as-needed)

# CRITICAL — match the prebuilt libdsr Eigen ABI (default alignment): NEITHER -march=native NOR
# -DEIGEN_MAX_ALIGN_BYTES (either changes sizeof(optional<Eigen::Transform>) → SIGBUS/stack-smash).
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
