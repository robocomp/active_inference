cmake_minimum_required(VERSION 3.16)
project(nightstand_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./nightstand_card_test
add_executable(nightstand_card_test nightstand_card_test.cpp)
target_include_directories(nightstand_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(nightstand_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(nightstand_card_test PRIVATE -O2)
target_link_libraries(nightstand_card_test PRIVATE Eigen3::Eigen)   # the I4 end-to-end case runs the evaluator
add_test(NAME nightstand_card_test COMMAND nightstand_card_test)
