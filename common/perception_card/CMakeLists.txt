cmake_minimum_required(VERSION 3.16)
project(perception_card_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Tests are timing-sensitive in later tasks: always optimised.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# Header-only module. Run directly (ctest is unusable on this machine):  LANG=es_ES.UTF-8 ./card_test
add_executable(card_test tests/card_test.cpp)
target_include_directories(card_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)

enable_testing()
add_test(NAME card_test COMMAND card_test)
