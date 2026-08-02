# Offline tools. They link the controller's own DSR-free modules directly (grid_planner, route_*)
# plus ai_common::geometry — no Qt, no Ice, no libdsr — so a tool costs seconds to build and cannot
# drift away from what the component actually runs: it IS what the component runs.
#
# ai_common::geometry (robot_footprint + robust_metrics) is the pure-Eigen half of common/; the
# DSR/Qt-dependent affordance_manager lives in ai_common::affordance and is deliberately not linked
# here. The target is created by generated/ -> src/CMakeLists.txt, which the parent
# CMakeLists.txt add_subdirectory()s before this one.
set(AI_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Eigen3 QUIET NO_MODULE)

add_executable(route_bench
  route_bench.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/grid_planner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_spline.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_follower.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_optimizer.cpp
)
target_compile_features(route_bench PRIVATE cxx_std_23)
target_compile_options(route_bench PRIVATE -O3)
target_link_libraries(route_bench PRIVATE ai_common::geometry)
if (TARGET Eigen3::Eigen)
  target_link_libraries(route_bench PRIVATE Eigen3::Eigen)
else()
  target_include_directories(route_bench PRIVATE /usr/include/eigen3)
endif()
set_target_properties(route_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin")

# Replays ONE recorded control cycle through the real TrajectoryController. It links the controller
# itself, which is possible only because that class depends on Eigen and robot_footprint and nothing
# else — no Qt, no Ice, no DSR.
add_executable(mppi_bench
  mppi_bench.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/trajectory_controller.cpp
)
target_compile_features(mppi_bench PRIVATE cxx_std_23)
target_compile_options(mppi_bench PRIVATE -O3 -fno-char8_t)
target_link_libraries(mppi_bench PRIVATE ai_common::geometry)
if (TARGET Eigen3::Eigen)
  target_link_libraries(mppi_bench PRIVATE Eigen3::Eigen)
else()
  target_include_directories(mppi_bench PRIVATE /usr/include/eigen3)
endif()
target_include_directories(mppi_bench PRIVATE /home/robocomp/robocomp/classes)
set_target_properties(mppi_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin")
