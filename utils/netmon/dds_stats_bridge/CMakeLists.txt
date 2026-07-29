cmake_minimum_required(VERSION 3.16)
project(dds_stats_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)

set(FASTDDS_SRC_TREE "$ENV{HOME}/software/Fast-DDS" CACHE PATH
    "Path to a Fast-DDS source checkout matching the installed libfastdds (for the statistics types generated code)")

add_executable(dds_stats_bridge
    main.cpp
    ${FASTDDS_SRC_TREE}/src/cpp/statistics/types/typesPubSubTypes.cxx
    ${FASTDDS_SRC_TREE}/src/cpp/statistics/types/typesTypeObjectSupport.cxx
)

target_include_directories(dds_stats_bridge PRIVATE
    ${FASTDDS_SRC_TREE}/src/cpp/statistics/types
)

target_link_libraries(dds_stats_bridge fastdds fastcdr)
