cmake_minimum_required(VERSION 3.16)
project(chair_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./chair_card_test
add_executable(chair_card_test chair_card_test.cpp)
target_include_directories(chair_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(chair_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(chair_card_test PRIVATE -O2)
target_link_libraries(chair_card_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_card_test COMMAND chair_card_test)

# P(detect): the shared countable formula + wall line of sight, against a copy of the retired ramp. Run: ./chair_detect_test
add_executable(chair_detect_test chair_detect_test.cpp)
target_include_directories(chair_detect_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(chair_detect_test PRIVATE -O2)
target_link_libraries(chair_detect_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_detect_test COMMAND chair_detect_test)

# The manifest is the single source of the chair's size prior. Run: LANG=es_ES.UTF-8 ./chair_prior_test
add_executable(chair_prior_test chair_prior_test.cpp ../src/chair_belief.cpp)
target_include_directories(chair_prior_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(chair_prior_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(chair_prior_test PRIVATE -O2)
target_link_libraries(chair_prior_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_prior_test COMMAND chair_prior_test)
