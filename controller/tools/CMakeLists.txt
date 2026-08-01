# Offline tools. They link the controller's own DSR-free modules directly (grid_planner, route_*,
# robot_footprint) — no Qt, no Ice, no libdsr — so a tool costs seconds to build and cannot drift away
# from what the component actually runs: it IS what the component runs.
set(AI_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Eigen3 QUIET NO_MODULE)

add_executable(route_bench
  route_bench.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/grid_planner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_spline.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_follower.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_optimizer.cpp
  ${AI_COMMON_DIR}/robot_footprint/robot_footprint.cpp
)
target_compile_features(route_bench PRIVATE cxx_std_23)
target_compile_options(route_bench PRIVATE -O3)
if (TARGET Eigen3::Eigen)
  target_link_libraries(route_bench PRIVATE Eigen3::Eigen)
else()
  target_include_directories(route_bench PRIVATE /usr/include/eigen3)
endif()
set_target_properties(route_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin")
