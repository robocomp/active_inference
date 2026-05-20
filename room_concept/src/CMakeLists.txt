cmake_policy(SET CMP0074 NEW)

find_package(Qt6 COMPONENTS Xml PrintSupport REQUIRED)
find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras QUIET)

# Find PyTorch
set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
find_package(CUDAToolkit REQUIRED)

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

# Enable CUDA language so CMake initializes detection scripts
#enable_language(CUDA)
#find_package(Torch REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)


# ── Sources ────────────────────────────────────────────────────────────────────
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/viewer_2d.cpp
  ../src/svg_room_loader.cpp
  ../src/room_concept.cpp
  ../src/room_model.cpp
  ../src/pointcloud_center_estimator.cpp
  ../src/epistemic_controller.cpp
  ../src/epistemic_planner.cpp
  ../src/timeseries_plot.cpp
  ../src/corner_detector.cpp
  ../src/rerun_logger.cpp
  ../src/camera_visualizer.cpp
  ../src/custom_widget.h
)

# ── Headers ────────────────────────────────────────────────────────────────────
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/viewer_2d.h
  ../src/svg_room_loader.h
  ../src/room_concept.h
  ../src/room_model.h
  ../src/pointcloud_center_estimator.h
  ../src/epistemic_controller.h
  ../src/epistemic_planner.h
  ../src/corner_visibility.h
  ../src/timeseries_plot.h
  ../src/corner_detector.h
  ../src/rerun_logger.h
  ../src/camera_visualizer.h
)

LIST(APPEND LIBS ${LIBS} "${TORCH_LIBRARIES}" tbb Qt6PrintSupport Qt6Xml
            Boost::system Boost::filesystem)

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

