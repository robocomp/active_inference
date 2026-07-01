cmake_minimum_required(VERSION 3.16)
project(media_transport_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)

set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated/idl)

add_executable(media_bench
  bench/media_bench.cpp
  media_transport.cpp
  ${GEN_DIR}/image_framePubSubTypes.cxx
  ${GEN_DIR}/image_frameTypeObjectSupport.cxx
  ${GEN_DIR}/image360_framePubSubTypes.cxx
  ${GEN_DIR}/image360_frameTypeObjectSupport.cxx
  ${GEN_DIR}/lidar_framePubSubTypes.cxx
  ${GEN_DIR}/lidar_frameTypeObjectSupport.cxx
  ${GEN_DIR}/imu_framePubSubTypes.cxx
  ${GEN_DIR}/imu_frameTypeObjectSupport.cxx
)

target_include_directories(media_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(media_bench PRIVATE fastdds fastcdr)
