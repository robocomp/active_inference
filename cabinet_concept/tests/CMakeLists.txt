cmake_minimum_required(VERSION 3.16)
project(cabinet_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./cabinet_card_test
add_executable(cabinet_card_test cabinet_card_test.cpp)
target_include_directories(cabinet_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(cabinet_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(cabinet_card_test PRIVATE -O2)
target_link_libraries(cabinet_card_test PRIVATE Eigen3::Eigen)
add_test(NAME cabinet_card_test COMMAND cabinet_card_test)
