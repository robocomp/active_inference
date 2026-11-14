cmake_minimum_required(VERSION 3.16)
project(bed_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./bed_card_test
add_executable(bed_card_test bed_card_test.cpp)
target_include_directories(bed_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(bed_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(bed_card_test PRIVATE -O2)
target_link_libraries(bed_card_test PRIVATE Eigen3::Eigen)   # the I4 end-to-end case runs the evaluator
add_test(NAME bed_card_test COMMAND bed_card_test)

# The bed's compound SDF: box ∪ headboard panel (2026-11-14). Run: ./headboard_sdf_test
add_executable(headboard_sdf_test headboard_sdf_test.cpp ../src/bed_belief.cpp)
target_include_directories(headboard_sdf_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(headboard_sdf_test PRIVATE -O2)
target_link_libraries(headboard_sdf_test PRIVATE Eigen3::Eigen)
add_test(NAME headboard_sdf_test COMMAND headboard_sdf_test)

