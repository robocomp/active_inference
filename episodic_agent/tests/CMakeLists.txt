# Offline test of episodic_agent's DSR adapters (no fleet; private DDS domains 60-62).
#   cmake -S episodic_agent/tests -B episodic_agent/tests/build && make -C episodic_agent/tests/build -j8
#   episodic_agent/tests/build/dsr_adapters_test
# Modelled on common/affordance_manager/tests/CMakeLists.txt.
cmake_minimum_required(VERSION 3.10)
project(episodic_agent_tests)

if ("$ENV{ROBOCOMP}" STREQUAL "")
  set(ENV{ROBOCOMP} "/home/robocomp/robocomp/")
endif ()
include($ENV{ROBOCOMP}/cmake/robocomp.cmake)
include($ENV{ROBOCOMP}/cmake/modules/qt.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_definitions("-DICE_CPP11_MAPPING")
find_package(Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)

# NO -march=native, NO -DEIGEN_MAX_ALIGN_BYTES=0: match the prebuilt libdsr's Eigen ABI (see src/CMakeLists.txt).
add_definitions(-O2 -fmax-errors=1 -fno-char8_t)

add_executable(dsr_adapters_test
  dsr_adapters_test.cpp
  ../src/dsr_fact_source.cpp
  ../src/em_graph.cpp
)
target_compile_definitions(dsr_adapters_test PRIVATE EM_ROOT_JSON="${CMAKE_CURRENT_LIST_DIR}/../em_root.json")
target_include_directories(dsr_adapters_test PRIVATE ../../common $ENV{ROBOCOMP}/classes)
target_link_libraries(dsr_adapters_test
  ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds osgDB OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets
  ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES}
)
set_target_properties(dsr_adapters_test PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# Pure (Qt Core only): the startup recompute / reseed / day-change folding of src/em_recorder.h.
add_executable(recorder_test recorder_test.cpp)
target_link_libraries(recorder_test Qt6::Core)
set_target_properties(recorder_test PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# Pure: the per-run metrics CSV of src/em_metrics.h (classic locale under es_ES).
add_executable(metrics_test metrics_test.cpp)
target_link_libraries(metrics_test Qt6::Core)
set_target_properties(metrics_test PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
