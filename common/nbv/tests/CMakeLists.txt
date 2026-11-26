cmake_minimum_required(VERSION 3.16)
project(nbv_tests LANGUAGES CXX)

# Standalone test project for common/nbv (header-only except the robot silhouette). Run:
#   cmake -S common/nbv/tests -B build && make -C build && /usr/bin/ctest --test-dir build
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# rc::RobotFootprint (the body the standpoint check places) — the same sources as ai_common::geometry, pure Eigen.
add_library(nbv_test_footprint STATIC ../../robot_footprint/robot_footprint.cpp ../../robot_footprint/mesh_hull.cpp)
target_link_libraries(nbv_test_footprint PUBLIC Eigen3::Eigen)

# (standpoint_free_test moved to common/standpoint/tests/task8_mirror_test.cpp in K3.)
# countable_planner.h includes the shared standpoint oracle (common/standpoint), so the planner test links the footprint too.
add_executable(countable_planner_test ../countable_planner_test.cpp)
target_include_directories(countable_planner_test PRIVATE ..)
target_compile_options(countable_planner_test PRIVATE -O1)
target_link_libraries(countable_planner_test PRIVATE Eigen3::Eigen nbv_test_footprint)
add_test(NAME countable_planner_test COMMAND countable_planner_test)
add_executable(z_occlusion_test ../z_occlusion_test.cpp)
target_include_directories(z_occlusion_test PRIVATE ..)
target_compile_options(z_occlusion_test PRIVATE -O1)
target_link_libraries(z_occlusion_test PRIVATE Eigen3::Eigen)
add_test(NAME z_occlusion_test COMMAND z_occlusion_test)
