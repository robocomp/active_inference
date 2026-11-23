# libnvblox_bridge.so — built on its own (CUDA), then linked by residual_nvblox as a plain shared library.
#   cmake -S nvblox_bridge -B nvblox_bridge/build -DNVBLOX_PREFIX=$HOME/software/nvblox/install && cmake --build nvblox_bridge/build -j
cmake_minimum_required(VERSION 3.22)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)   # RTX 5080 Laptop (shadow); must precede project()
endif()
project(nvblox_bridge LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(NVBLOX_PREFIX "$ENV{HOME}/software/nvblox/install" CACHE PATH "nvblox install prefix")
list(APPEND CMAKE_PREFIX_PATH ${NVBLOX_PREFIX})
find_package(CUDAToolkit REQUIRED)
find_package(gflags REQUIRED COMPONENTS nothreads_static)
find_package(glog REQUIRED)
find_package(nvblox REQUIRED)

add_library(nvblox_bridge SHARED nvblox_bridge.cu)
target_compile_options(nvblox_bridge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(nvblox_bridge PRIVATE nvblox::nvblox_lib)
set_target_properties(nvblox_bridge PROPERTIES
  CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden
  INSTALL_RPATH "${NVBLOX_PREFIX}/lib" BUILD_RPATH "${NVBLOX_PREFIX}/lib"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
