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

add_executable(terms_test tests/terms_test.cpp)
target_include_directories(terms_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
add_test(NAME terms_test COMMAND terms_test)

# Needs Eigen (nbv/viewpoint_score.h). SensorModel comes from media_transport/sensor_model.h (DDS-free): no fastdds.
find_package(Eigen3 REQUIRED)
add_executable(evaluator_camera_test tests/evaluator_camera_test.cpp)
target_include_directories(evaluator_camera_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(evaluator_camera_test PRIVATE Eigen3::Eigen)
add_test(NAME evaluator_camera_test COMMAND evaluator_camera_test)

add_executable(evaluator_lidar_test tests/evaluator_lidar_test.cpp)
target_include_directories(evaluator_lidar_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(evaluator_lidar_test PRIVATE Eigen3::Eigen)
add_test(NAME evaluator_lidar_test COMMAND evaluator_lidar_test)

add_executable(evaluator_combine_test tests/evaluator_combine_test.cpp)
target_include_directories(evaluator_combine_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(evaluator_combine_test PRIVATE Eigen3::Eigen)
add_test(NAME evaluator_combine_test COMMAND evaluator_combine_test)

add_executable(card_io_test tests/card_io_test.cpp)
target_include_directories(card_io_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
add_test(NAME card_io_test COMMAND card_io_test)

add_executable(card_vs_probe tools/card_vs_probe.cpp)
target_include_directories(card_vs_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(card_vs_probe PRIVATE Eigen3::Eigen)

add_executable(card_vs_probe_test tests/card_vs_probe_test.cpp)
target_include_directories(card_vs_probe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(card_vs_probe_test PRIVATE Eigen3::Eigen)
add_test(NAME card_vs_probe_test COMMAND card_vs_probe_test)

add_executable(generative_test tests/generative_test.cpp)
target_include_directories(generative_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(generative_test PRIVATE Eigen3::Eigen)
add_test(NAME generative_test COMMAND generative_test)

add_executable(render_zed_test tests/render_zed_test.cpp)
target_include_directories(render_zed_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../media_transport)
target_link_libraries(render_zed_test PRIVATE Eigen3::Eigen)
add_test(NAME render_zed_test COMMAND render_zed_test)
