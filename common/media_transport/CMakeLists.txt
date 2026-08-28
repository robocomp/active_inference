cmake_minimum_required(VERSION 3.16)
project(media_transport_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)

# The IDL source list lives in ONE place, shared with ai_common::media (common/CMakeLists.txt),
# so a new FrameKind cannot be added to the agents and silently leave this bench unlinkable.
include(${CMAKE_CURRENT_SOURCE_DIR}/idl_sources.cmake)

add_executable(media_bench
  bench/media_bench.cpp
  media_transport.cpp
  ${AI_COMMON_IDL_SOURCES}
)

target_include_directories(media_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(media_bench PRIVATE fastdds fastcdr)

# Descriptor round-trip under the AGENT'S locale. It calls setlocale(LC_ALL,"") itself,
# because a standalone harness stays in "C" and the es_ES decimal-comma bug simply
# VANISHES -- the test and the agent would then answer different questions about the
# same string. Run it as:  LANG=es_ES.UTF-8 ./descriptor_roundtrip
add_executable(descriptor_roundtrip
  bench/descriptor_roundtrip.cpp
  media_transport.cpp
  ${AI_COMMON_IDL_SOURCES}
)
target_include_directories(descriptor_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(descriptor_roundtrip PRIVATE fastdds fastcdr)
