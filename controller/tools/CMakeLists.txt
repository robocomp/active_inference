# Offline tools. They link the controller's own DSR-free modules directly (grid_planner, route_*)
# plus ai_common::geometry — no Qt, no Ice, no libdsr — so a tool costs seconds to build and cannot
# drift away from what the component actually runs: it IS what the component runs.
#
# ai_common::geometry (robot_footprint + robust_metrics) is the pure-Eigen half of common/; the
# DSR/Qt-dependent affordance_manager lives in ai_common::affordance and is deliberately not linked
# here. The target is created by generated/ -> src/CMakeLists.txt, which the parent
# CMakeLists.txt add_subdirectory()s before this one.

find_package(Eigen3 QUIET NO_MODULE)

add_executable(route_bench
  route_bench.cpp
  # rc::mesh — the robot's silhouette read from its own mesh. Linked here (rather than via
  # ai_common::geometry alone) for the same reason every other module is: the bench must run the SAME
  # code the component does, and its self_test is the only place the es_ES locale trap is exercised.
  ${CMAKE_CURRENT_LIST_DIR}/../../common/robot_footprint/mesh_hull.cpp
  # PlainTracker rides along for its self_test() only — it depends on RouteSpline and the geometry lib
  # and nothing else, so --self-test can cover the arc-length projection alongside the route modules.
  ${CMAKE_CURRENT_LIST_DIR}/../src/trackers/plain_tracker.cpp
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

# Closed-loop tracker bench against the IDENTIFIED plant, plus the clearance-bound tests (the phantom
# response that reproduced the lap-4 stop). ★Was built by hand until 2026-08-05, which meant
# reconstructing the compile line before it could be re-run — the whole point of a harness is that it is
# one command away.
add_executable(tracker_sim
  tracker_sim.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/trackers/plain_tracker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/grid_planner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_spline.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_follower.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_optimizer.cpp
)
target_compile_features(tracker_sim PRIVATE cxx_std_23)
target_compile_options(tracker_sim PRIVATE -O2)
target_link_libraries(tracker_sim PRIVATE ai_common::geometry)
if (TARGET Eigen3::Eigen)
  target_link_libraries(tracker_sim PRIVATE Eigen3::Eigen)
else()
  target_include_directories(tracker_sim PRIVATE /usr/include/eigen3)
endif()
set_target_properties(tracker_sim PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../bin")

# Replays ONE recorded control cycle through the real TrajectoryController. It links the controller
# itself, which is possible only because that class depends on Eigen and robot_footprint and nothing
# else — no Qt, no Ice, no DSR.
add_executable(mppi_bench
  mppi_bench.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/trajectory_controller.cpp
  # TrajectoryController dispatches to these, so linking it means linking them.
  ${CMAKE_CURRENT_LIST_DIR}/../src/trackers/plain_tracker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/trackers/pd_tracker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/trackers/mppi_tracker.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_spline.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/route_optimizer.cpp
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
