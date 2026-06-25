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

find_package(Torch REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")


# ── Sources ────────────────────────────────────────────────────────────────────
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
  ../src/viewer_2d.cpp
  ../src/svg_room_loader.cpp
  ../src/room_concept.cpp
  ../src/room_model.cpp
  ../src/pointcloud_center_estimator.cpp
  ../src/epistemic_controller.cpp
  ../src/occupancy_belief.cpp
  ../src/room_scene_graph.cpp
  ../src/lidar_ingestor.cpp
  ../src/room_viewer.cpp
  ../src/room_config.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.cpp
  ../src/epistemic_planner.cpp
  ../src/timeseries_plot.cpp
  ../src/corner_detector.cpp
  ../src/rerun_logger.cpp
  ../src/camera_visualizer.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../src/custom_widget.h
)

# ── Headers ────────────────────────────────────────────────────────────────────
LIST(APPEND HEADERS
  ../src/specificworker.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
  ../src/viewer_2d.h
  ../src/svg_room_loader.h
  ../src/room_concept.h
  ../src/room_model.h
  ../src/pointcloud_center_estimator.h
  ../src/epistemic_controller.h
  ../src/room_scene_graph.h
  ../src/lidar_ingestor.h
  ../src/room_viewer.h
  ../src/room_config.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/affordance_manager/affordance_manager.h
  ../src/epistemic_planner.h
  ../src/occupancy_belief.h
  ../src/corner_visibility.h
  ../src/timeseries_plot.h
  ../src/corner_detector.h
  ../src/rerun_logger.h
  ../src/camera_visualizer.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/config/config_loader_utils.h
)

LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt6PrintSupport Qt6Xml
            Boost::system Boost::filesystem)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash, e.g. camera_visualizer) or misaligns it (SIGBUS). libdsr is built
# with default Eigen alignment; the agents must too.
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t)

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Instruments room_concept's OWN code so we catch the corrupting WRITE behind the bottle-churn
# segfault (the bad-free that surfaces in Qt paint), not just the later free. See run_asan.sh.
option(ENABLE_ASAN "Build room_concept with AddressSanitizer" OFF)
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

