cmake_minimum_required(VERSION 3.16)
project(standpoint_tests LANGUAGES CXX)

# The shared standpoint oracle (K3). Run:
#   cmake -S common/standpoint/tests -B common/standpoint/tests/build && make -C common/standpoint/tests/build -j8 \
#     && /usr/bin/ctest --test-dir common/standpoint/tests/build
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

set(CTRL ${CMAKE_CURRENT_SOURCE_DIR}/../../../controller)
add_library(standpoint_test_footprint STATIC ../../robot_footprint/robot_footprint.cpp ../../robot_footprint/mesh_hull.cpp)
target_link_libraries(standpoint_test_footprint PUBLIC Eigen3::Eigen)

add_executable(standpoint_oracle_test standpoint_oracle_test.cpp)
target_compile_options(standpoint_oracle_test PRIVATE -O1 -Wall -Wextra)
target_link_libraries(standpoint_oracle_test PRIVATE Eigen3::Eigen standpoint_test_footprint)
add_test(NAME standpoint_oracle_test COMMAND standpoint_oracle_test)

# The controller's decisions against the golden recorded at BASE, and the oracle against the controller (controller/tests).
add_executable(standpoint_oracle_parity ${CTRL}/tests/test_standpoint_oracle_parity.cpp ${CTRL}/src/grid_planner.cpp
               ${CTRL}/src/local_policy/cost_to_go.cpp)
target_include_directories(standpoint_oracle_parity PRIVATE ${CTRL}/src)
target_compile_options(standpoint_oracle_parity PRIVATE -O2 -Wall -Wextra)
target_link_libraries(standpoint_oracle_parity PRIVATE Eigen3::Eigen standpoint_test_footprint)
add_test(NAME standpoint_oracle_parity COMMAND standpoint_oracle_parity ${CTRL}/tests/data/standpoint_oracle_golden.txt)
