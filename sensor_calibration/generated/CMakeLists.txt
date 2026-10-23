cmake_minimum_required(VERSION 3.10)

#Select the compiler standard
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "Version of g++ is lower than 11, g++ 11 or higher is required.")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    message(STATUS "Using g++ 11, setting C++20 standard.")
    set(CMAKE_CXX_STANDARD 20)
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(STATUS "Using g++ 12, setting C++23 standard.")
    set(CMAKE_CXX_STANDARD 23)
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15)
    message(STATUS "Using g++ 13 or higher, setting the C++23 standard.")
    set(CMAKE_CXX_STANDARD 23)
else()
    message(STATUS "Using g++ 15 or higher, setting the C++26 standard.")
    set(CMAKE_CXX_STANDARD 26)
endif()


IF(NOT EXISTS $ENV{ROBOCOMP}/cmake)
  MESSAGE(FATAL_ERROR "Couldn't find RoboComp in $ENV{ROBOCOMP}! Exiting...")
ENDIF(NOT EXISTS $ENV{ROBOCOMP}/cmake)

INCLUDE($ENV{ROBOCOMP}/cmake/robocomp.cmake)
INCLUDE($ENV{ROBOCOMP}/cmake/modules/qt.cmake)
INCLUDE (../src/CMakeLists.txt)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
ADD_DEFINITIONS ("-DICE_CPP11_MAPPING")
FIND_PACKAGE( Ice REQUIRED COMPONENTS Ice++11 IceStorm++11)

# Sources set
SET (SOURCES
  ${SOURCES}
  $ENV{ROBOCOMP}/classes/rapplication/rapplication.cpp
  $ENV{ROBOCOMP}/classes/sigwatch/sigwatch.cpp
  $ENV{ROBOCOMP}/classes/grafcetStep/GRAFCETStep.cpp
  $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp
  main.cpp
  genericworker.cpp
)


#ROBOCOMP
ROBOCOMP_INITIALIZE( $ENV{ROBOCOMP}/ )
ROBOCOMP_IDSL_TO_ICE()
ROBOCOMP_ICE_TO_SRC()
SET (EXECUTABLE_OUTPUT_PATH ${RC_COMPONENT_DEVEL_PATH}/bin)

# Specify construction and link process
ADD_EXECUTABLE( sensor_calibration ${SOURCES} ${MOC_SOURCES} ${RC_SOURCES} ${UI_HEADERS} )
TARGET_LINK_LIBRARIES( sensor_calibration ${LIBS}  dsr_core dsr_gui dsr_api fastcdr fastdds OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES} )
INSTALL(FILES ${EXECUTABLE_OUTPUT_PATH}/sensor_calibration DESTINATION ${RC_COMPONENT_INSTALL_PATH}/bin/ PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE )

# Plan Task 2.2: input layer selftest. Private DDS domains only (graph 91, media 44); never the live ones.
#   make -C build sensor_inputs_test && ../bin/sensor_inputs_test
ADD_EXECUTABLE( sensor_inputs_test EXCLUDE_FROM_ALL ../tests/sensor_inputs_test.cpp ../src/sensor_inputs.cpp )
TARGET_COMPILE_DEFINITIONS( sensor_inputs_test PRIVATE SHADOW_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../../robot_concept/shadow.json" )
TARGET_LINK_LIBRARIES( sensor_inputs_test ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES} )
SET_TARGET_PROPERTIES( sensor_inputs_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )

# Trajectory harvester selftest (synthetic, no DDS/DSR): pure Eigen, links only ai_common::sensor_calib.
#   make -C build trajectory_harvester_test && ../bin/trajectory_harvester_test
ADD_EXECUTABLE( trajectory_harvester_test EXCLUDE_FROM_ALL ../tests/trajectory_harvester_test.cpp )
TARGET_LINK_LIBRARIES( trajectory_harvester_test ai_common::sensor_calib Eigen3::Eigen )
SET_TARGET_PROPERTIES( trajectory_harvester_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )

# Plan Task 2.3: proposal publisher selftest (pure parts + graph on PRIVATE domain 92).
#   make -C build proposal_publisher_test && ../bin/proposal_publisher_test
ADD_EXECUTABLE( proposal_publisher_test EXCLUDE_FROM_ALL ../tests/proposal_publisher_test.cpp )
TARGET_COMPILE_DEFINITIONS( proposal_publisher_test PRIVATE SHADOW_JSON="${CMAKE_CURRENT_SOURCE_DIR}/../../robot_concept/shadow.json" )
TARGET_LINK_LIBRARIES( proposal_publisher_test ${LIBS} dsr_core dsr_gui dsr_api fastcdr fastdds OpenThreads Eigen3::Eigen QGLViewer-qt6 Qt6OpenGLWidgets ${STATIC_LIBS} ${SPECIFIC_LIBS} ${QT_LIBRARIES} ${Ice_LIBRARIES} )
SET_TARGET_PROPERTIES( proposal_publisher_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin )
