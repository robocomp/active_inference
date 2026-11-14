cmake_minimum_required(VERSION 3.16)
project(countable_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# (The perception-card test is added by Task 4, parameterised over the class folders: no folder has a
# migrated [Concept]/[Model] config before then.)

# The bed's compound SDF: box ∪ headboard panel (2026-11-14). Run: ./headboard_sdf_test
add_executable(headboard_sdf_test headboard_sdf_test.cpp ../src/countable_belief.cpp)
target_include_directories(headboard_sdf_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(headboard_sdf_test PRIVATE -O2)
target_link_libraries(headboard_sdf_test PRIVATE Eigen3::Eigen)
add_test(NAME headboard_sdf_test COMMAND headboard_sdf_test)

