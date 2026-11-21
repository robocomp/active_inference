cmake_minimum_required( VERSION 3.10 )
PROJECT( retina )

IF ( "$ENV{ROBOCOMP}" STREQUAL "")
  MESSAGE(WARNING "ROBOCOMP variable not set. Using the default value: /home/robocomp/robocomp")
  SET (ENV{ROBOCOMP} "/home/robocomp/robocomp/")
ENDIF ( "$ENV{ROBOCOMP}" STREQUAL "")

SET(RC_COMPONENT_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "component base path" FORCE)
SET(RC_COMPONENT_DEVEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "component binary path" FORCE)
#SET(RC_COMPONENT_DEVEL_PATH "${WORKSPACE_PATH}/devel"  CACHE STRING "component binary path")
SET(RC_COMPONENT_INSTALL_PATH $ENV{ROBOCOMP} CACHE STRING "component install path")

#commented as now devel is merged into source space
#get_filename_component( COMPONENT_NAME ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
#STRING(REPLACE "${COMPONENT_NAME}/" " " COMPONENT_NAME  ${RC_COMPONENT_PATH})
#STRING(STRIP ${COMPONENT_NAME} COMPONENT_NAME)
#SET(COMPONENT_NAME ${COMPONENT_NAME} CACHE STRING "component name" FORCE)

add_subdirectory(generated)

INSTALL(FILES etc/config DESTINATION ${RC_COMPONENT_INSTALL_PATH}/etc-default/ RENAME retina.conf )

#ADD_CUSTOM_TARGET(doc doxygen Doxyfile)
#ADD_CUSTOM_TARGET(installdoc mkdir -p ${RC_COMPONENT_INSTALL_PATH}/doc COMMAND cp -R doc/html ${RC_COMPONENT_INSTALL_PATH}/doc/retina)
# ── Standalone unit tests (opt-in; the agent build never compiles them) ─────────────────────────────
#   cmake -B build-tests -DRETINA_BUILD_TESTS=ON && make -C build-tests -j8 test_mask_dump && ctest --test-dir build-tests
# (or without cmake: tests/run_mask_dump_test.sh)
option(RETINA_BUILD_TESTS "Build retina's standalone unit tests (ctest)" OFF)
if (RETINA_BUILD_TESTS)
  enable_testing()
  find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
  find_package(Eigen3 REQUIRED)
  add_executable(test_mask_dump
    tests/test_mask_dump.cpp
    src/zed_frame_dump.cpp
    src/mask_dump_csv.cpp)
  target_compile_features(test_mask_dump PRIVATE cxx_std_23)
  target_include_directories(test_mask_dump PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/local/include)
  target_link_libraries(test_mask_dump PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
  add_test(NAME mask_dump COMMAND test_mask_dump)
endif()
