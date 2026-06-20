set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

cmake_policy(SET CMP0074 NEW)

# Keep generated CMake changes from pushing an unsupported host dialect into
# libtorch/CUDA configure checks (CUDA-13.2 nvcc otherwise demands "CUDA26").
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD GREATER 23)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Find PyTorch (same resolution strategy used in table_concept / room_concept)
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
  ../src/bottle_model.cpp
  ../src/sample_queue.cpp
  ../src/prior_store.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics_torch.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/bottle_model.h
  ../src/sample_queue.h
  ../src/prior_store.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics_torch.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)


LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}")

# libdsr_core defines symbols that libdsr_{api,gui} only reference (UND). The
# framework links them under -Wl,--as-needed with core *ahead* of its consumers,
# so the linker drops core whenever no TU references it directly. table_concept
# survives only because its DSR-heavy TUs pull core in; force-keep all libs so a
# light-DSR build (or the current stub) links too.
add_link_options(-Wl,--no-as-needed)

# no-char8_t to avoid a problem with ICE Connection lib.
# EIGEN_MAX_ALIGN_BYTES=0: disable Eigen's over-alignment requirement so Eigen types reaching us via
# std::optional/temporaries (e.g. DSR Mat::RTMat from get_transformation_matrix) can't trip an
# aligned-load SIGBUS — neutralises the -march=native alignment hazard. Keep consistent across agents.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t -DEIGEN_MAX_ALIGN_BYTES=0)

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Instruments bottle_concept's OWN code to pin the Ctrl-C shutdown segfault. See run_asan.sh.
option(ENABLE_ASAN "Build bottle_concept with AddressSanitizer" OFF)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for bottle_concept (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()
