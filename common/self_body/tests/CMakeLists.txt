cmake_minimum_required(VERSION 3.16)
project(self_body_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# SELF-BODY birth term: a mask on the robot's own top plate is not a world object. Needs the SHADOW footprint
# (robot_footprint + mesh_hull), so it is a CMake project; common/run_tests.sh builds and runs it explicitly.
add_executable(self_body_test self_body_test.cpp ../../robot_footprint/robot_footprint.cpp
                              ../../robot_footprint/mesh_hull.cpp)
target_include_directories(self_body_test PRIVATE ../.. )
target_compile_options(self_body_test PRIVATE -O2)
target_link_libraries(self_body_test PRIVATE Eigen3::Eigen)
add_test(NAME self_body_test COMMAND self_body_test)
