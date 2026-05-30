cmake_minimum_required(VERSION 3.10)
project(active_inference_common_tests)

enable_testing()

if ("$ENV{ROBOCOMP}" STREQUAL "")
  message(WARNING "ROBOCOMP variable not set. Using the default value: /home/robocomp/robocomp")
  set(ENV{ROBOCOMP} "/home/robocomp/robocomp/")
endif ()

if(NOT EXISTS $ENV{ROBOCOMP}/cmake)
  message(FATAL_ERROR "Couldn't find RoboComp in $ENV{ROBOCOMP}! Exiting...")
endif()

include($ENV{ROBOCOMP}/cmake/robocomp.cmake)
include($ENV{ROBOCOMP}/cmake/modules/qt.cmake)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_definitions("-DICE_CPP11_MAPPING")
find_package(Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "Version of g++ is lower than 11, g++ 11 or higher is required.")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    message(STATUS "Using g++ 11, setting C++20 standard.")
    set(CXX_STANDARD 20)
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(STATUS "Using g++ 12, setting C++23 standard.")
    set(CXX_STANDARD 23)
else()
    message(STATUS "Using g++ 13 or higher, setting the C++23 standard.")
    set(CXX_STANDARD 23)
endif()

add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

add_executable(affordance_protocol_test
  affordance_protocol_test.cpp
  ../affordance_manager.cpp
)
target_compile_definitions(affordance_protocol_test PRIVATE AFFORDANCE_TEST_SHADOW_JSON="${CMAKE_CURRENT_LIST_DIR}/../../robot_concept/shadow.json")
target_include_directories(affordance_protocol_test PRIVATE
  ..
  /home/pbustos/robocomp/components/cortex/api/include
  /home/pbustos/robocomp/components/cortex/core/include
  /home/pbustos/robocomp/components/cortex/gui/include
  /home/robocomp/robocomp/robocomp_core/classes
)
target_link_libraries(affordance_protocol_test
  ${LIBS}
  dsr_core
  dsr_gui
  dsr_api
  fastcdr
  fastdds
  osgDB
  OpenThreads
  Eigen3::Eigen
  QGLViewer-qt6
  Qt6OpenGLWidgets
  ${STATIC_LIBS}
  ${SPECIFIC_LIBS}
  ${QT_LIBRARIES}
  ${Ice_LIBRARIES}
)
set_target_properties(affordance_protocol_test
  PROPERTIES
  CXX_STANDARD ${CXX_STANDARD}
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

add_test(NAME affordance_protocol_test COMMAND affordance_protocol_test)