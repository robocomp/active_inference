cmake_policy(SET CMP0074 NEW)

# Keep generated CMake changes from pushing an unsupported host dialect into
# libtorch/CUDA configure checks when templates are regenerated.
if(NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD GREATER 23)
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Xml PrintSupport REQUIRED)
find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras QUIET)

# Find PyTorch
set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.2/bin/nvcc")

# Allow an external Torch_DIR/CMAKE_PREFIX_PATH to win, then try common local installs.
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

set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
find_package(Torch REQUIRED)
# Torch injects -D_GLIBCXX_USE_CXX11_ABI=0 via INTERFACE_COMPILE_OPTIONS on the imported `torch`
# target (TorchConfig.cmake), not via CMAKE_CXX_FLAGS, so it propagates at link time no matter what
# CMAKE_CXX_FLAGS says here. Strip it from the target itself: prebuilt libdsr_api/core/gui, Qt6, Ice
# and FastDDS are all built with the default (CXX11) ABI.
# (The ai_common::* targets never link torch, so they are not exposed to this either way.)
if(TARGET torch)
  get_target_property(_torch_compile_opts torch INTERFACE_COMPILE_OPTIONS)
  if(_torch_compile_opts)
    list(REMOVE_ITEM _torch_compile_opts "-D_GLIBCXX_USE_CXX11_ABI=0")
    set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS "${_torch_compile_opts}")
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# AddressSanitizer (opt-in, separate build dir) — declared BEFORE add_subdirectory() so the
# ai_common targets are instrumented too; behind a library boundary they no longer pick this up
# from the directory-scope add_compile_options further down.
option(ENABLE_ASAN "Build room_concept with AddressSanitizer" OFF)

# Shared ai_common::* libraries (see common/CMakeLists.txt). room_concept has its OWN
# src/timeseries_plot.cpp, so it deliberately does not link ai_common::ui.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane (was 10 spliced IDL paths + an include_directories)
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::affordance   # affordance_manager
  ai_common::core         # header-only beliefs/geometry (object_anchor/, occlusion, …)
)

# ── Sources ────────────────────────────────────────────────────────────────────
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/viewer_2d.cpp
  ../src/svg_room_loader.cpp
  ../src/room_concept.cpp
  ../src/room_model.cpp
  ../src/pointcloud_center_estimator.cpp
  ../src/epistemic_controller.cpp
  ../src/room_scene_graph.cpp
  ../src/lidar_ingestor.cpp
  ../src/room_viewer.cpp
  ../src/room_config.cpp
  ../src/epistemic_planner.cpp
  ../src/timeseries_plot.cpp
  ../src/corner_detector.cpp
  ../src/object_anchor_factor.cpp
  ../src/object_anchor_source.cpp
  ../src/rerun_logger.cpp
  ../src/camera_visualizer.cpp
  ../src/custom_widget.h
)

LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt6PrintSupport Qt6Xml
            Boost::system Boost::filesystem)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash, e.g. camera_visualizer) or misaligns it (SIGBUS). libdsr is built
# with default Eigen alignment; the agents must too.
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t)

if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for room_concept (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()
