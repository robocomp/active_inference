cmake_minimum_required(VERSION 3.16)
project(hood_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./hood_card_test
add_executable(hood_card_test hood_card_test.cpp)
target_include_directories(hood_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(hood_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(hood_card_test PRIVATE -O2)
add_test(NAME hood_card_test COMMAND hood_card_test)
