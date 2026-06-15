cmake_minimum_required(VERSION 3.16)
project(test_bottle_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Same CUDA/Torch resolution dance as the component (CUDA-13.2 nvcc demands a
# pinned dialect, otherwise libtorch's GPU probe fails to configure).
set(CUDAToolkit_ROOT "/usr/local/cuda-13.2")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT Torch_DIR)
  if(EXISTS "/home/robocomp/software/libtorch/share/cmake/Torch/TorchConfig.cmake")
    set(Torch_DIR "/home/robocomp/software/libtorch/share/cmake/Torch")
  endif()
endif()
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(test_bottle_fit
  test_bottle_fit.cpp
  ../src/bottle_model.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/robust_metrics/robust_metrics_torch.cpp
)
target_link_libraries(test_bottle_fit PRIVATE "${TORCH_LIBRARIES}" Eigen3::Eigen)
target_compile_options(test_bottle_fit PRIVATE -O2)
