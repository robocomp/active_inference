set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Find PyTorch (same resolution strategy used in room_concept)
if(NOT Torch_DIR)
  if(EXISTS "/home/robocomp/software/libtorch/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "/home/robocomp/software/libtorch/share/cmake/Torch")
  else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if(TORCH_CMAKE_PREFIX_RESULT EQUAL 0 AND EXISTS "${TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
        set(Torch_DIR "${TORCH_CMAKE_PREFIX}/Torch")
      endif()
    endif()
  endif()
endif()

find_package(Torch REQUIRED)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/table_model.cpp
  ../src/sample_queue.cpp
  ../src/epistemic_planner.cpp
  ../src/prior_store.cpp
  ../src/table_affordance.cpp
  ../src/timeseries_plot.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics_torch.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/table_model.h
  ../src/sample_queue.h
  ../src/epistemic_planner.h
  ../src/prior_store.h
  ../src/table_affordance.h
  ../src/timeseries_plot.h
  ../src/custom_widget.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics_torch.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)


LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}")

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t  -DEIGEN_MAX_ALIGN_BYTES=0)

