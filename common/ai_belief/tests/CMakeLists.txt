cmake_minimum_required(VERSION 3.16)
project(ai_belief_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Sensor-variance helpers (header-only common/ai_belief/sensor_var.h). Run: ./sensor_var_test
add_executable(sensor_var_test sensor_var_test.cpp)
target_include_directories(sensor_var_test PRIVATE ../..)
target_compile_options(sensor_var_test PRIVATE -O2)
target_link_libraries(sensor_var_test PRIVATE Eigen3::Eigen)   # the degenerate-geometry cases include viewpoint.h
add_test(NAME sensor_var_test COMMAND sensor_var_test)

# Viewpoint rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2), moved from
# countable_concept/tests. Header-only common/ai_belief/viewpoint.h; pure Eigen. Run: ./viewpoint_rho_test
add_executable(viewpoint_rho_test viewpoint_rho_test.cpp)
target_include_directories(viewpoint_rho_test PRIVATE ../..)
target_compile_options(viewpoint_rho_test PRIVATE -O2)
target_link_libraries(viewpoint_rho_test PRIVATE Eigen3::Eigen)
add_test(NAME viewpoint_rho_test COMMAND viewpoint_rho_test)
