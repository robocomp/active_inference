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
add_executable(chair_prior_test chair_prior_test.cpp ../src/chair_belief.cpp ../src/chair_model.cpp)
target_include_directories(chair_prior_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(chair_prior_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(chair_prior_test PRIVATE -O2)
target_link_libraries(chair_prior_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_prior_test COMMAND chair_prior_test)

# The card's generative shape (closed primitives) against ChairModel's compound SDF. Run: LANG=es_ES.UTF-8 ./chair_shape_test
add_executable(chair_shape_test chair_shape_test.cpp ../src/chair_model.cpp)
target_include_directories(chair_shape_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(chair_shape_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(chair_shape_test PRIVATE -O2)
target_link_libraries(chair_shape_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_shape_test COMMAND chair_shape_test)

# Recall candidates read off ltom's recall nodes: DSR-linked, private DDS domain 65 (63 ltom, 62 episodic).
# Run: LANG=es_ES.UTF-8 ./recall_candidates_test
if ("$ENV{ROBOCOMP}" STREQUAL "")
  set(ENV{ROBOCOMP} "/home/robocomp/robocomp/")
endif ()
include($ENV{ROBOCOMP}/cmake/robocomp.cmake)
include($ENV{ROBOCOMP}/cmake/modules/qt.cmake)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_definitions("-DICE_CPP11_MAPPING")
find_package(Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)
add_definitions(-fno-char8_t)
add_executable(recall_candidates_test recall_candidates_test.cpp)
target_compile_definitions(recall_candidates_test PRIVATE CHAIR_TEST_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/../../ltom_agent/tests/ltom_root.json")
target_include_directories(recall_candidates_test PRIVATE ../src ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(recall_candidates_test
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})
set_target_properties(recall_candidates_test PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
