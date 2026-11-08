# Offline test of ltom_agent's pure repertory model (no DSR, no fleet).
#   cmake -S ltom_agent/tests -B ltom_agent/tests/build && make -C ltom_agent/tests/build -j8
#   ltom_agent/tests/build/test_repertory
cmake_minimum_required(VERSION 3.10)
project(ltom_agent_tests CXX)

# NO -march=native, NO -DEIGEN_MAX_ALIGN_BYTES=0 (house rule).
add_compile_options(-O2 -fmax-errors=1 -Wall -Wextra)
find_package(Qt6 REQUIRED COMPONENTS Core)

add_executable(test_repertory test_repertory.cpp ../src/repertory.cpp)
set_target_properties(test_repertory PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_executable(test_repertory_io test_repertory_io.cpp ../src/repertory_io.cpp ../src/repertory.cpp)
set_target_properties(test_repertory_io PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_link_libraries(test_repertory_io PRIVATE Qt6::Core)

# DSR-linked: the live-graph reader on a private DDS domain (63), modelled on episodic_agent/tests/CMakeLists.txt.
if ("$ENV{ROBOCOMP}" STREQUAL "")
  set(ENV{ROBOCOMP} "/home/robocomp/robocomp/")
endif ()
include($ENV{ROBOCOMP}/cmake/robocomp.cmake)
include($ENV{ROBOCOMP}/cmake/modules/qt.cmake)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_definitions("-DICE_CPP11_MAPPING")
find_package(Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)
add_definitions(-fno-char8_t)

add_executable(test_graph_sightings test_graph_sightings.cpp ../src/graph_sightings.cpp ../src/repertory.cpp)
target_compile_definitions(test_graph_sightings PRIVATE LTOM_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/ltom_root.json")
target_include_directories(test_graph_sightings PRIVATE ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(test_graph_sightings
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})
set_target_properties(test_graph_sightings PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_executable(test_recall_select test_recall_select.cpp ../src/recall_select.cpp ../src/repertory.cpp)
set_target_properties(test_recall_select PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_executable(test_recall_publisher test_recall_publisher.cpp ../src/recall_publisher.cpp ../src/recall_select.cpp ../src/repertory.cpp)
target_compile_definitions(test_recall_publisher PRIVATE LTOM_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/ltom_root.json")
target_include_directories(test_recall_publisher PRIVATE ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(test_recall_publisher
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})
set_target_properties(test_recall_publisher PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# The repertory projected onto the memory graph: DSR-linked, private DDS domain 63.
add_executable(test_memory_graph test_memory_graph.cpp ../src/memory_graph.cpp ../src/recall_publisher.cpp ../src/recall_select.cpp ../src/repertory.cpp)
target_compile_definitions(test_memory_graph PRIVATE LTOM_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/ltom_root.json")
target_include_directories(test_memory_graph PRIVATE ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(test_memory_graph
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})
set_target_properties(test_memory_graph PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# A left-drag pans the viewer on a quiet graph: the real GraphViewer, offscreen.
add_executable(test_graph_pan test_graph_pan.cpp)
target_compile_definitions(test_graph_pan PRIVATE LTOM_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/ltom_root.json")
target_include_directories(test_graph_pan PRIVATE ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(test_graph_pan
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES})
set_target_properties(test_graph_pan PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
