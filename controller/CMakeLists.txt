cmake_minimum_required( VERSION 3.10 )
PROJECT( controller )

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
# Offline benches (route_bench, mppi_bench, tracker_sim). ★They were never wired in: tools/CMakeLists.txt
# existed but nothing add_subdirectory()'d it, so `make tracker_sim` failed with "No rule to make target"
# — which is easy to miss when you grep the build output for "error:". Every bench was therefore built by
# hand, and a stale binary silently answered questions about source that had changed. Must come AFTER
# generated/, which creates the ai_common::geometry target these link.
add_subdirectory(tools)

INSTALL(FILES etc/config DESTINATION ${RC_COMPONENT_INSTALL_PATH}/etc-default/ RENAME controller.conf )

#ADD_CUSTOM_TARGET(doc doxygen Doxyfile)
#ADD_CUSTOM_TARGET(installdoc mkdir -p ${RC_COMPONENT_INSTALL_PATH}/doc COMMAND cp -R doc/html ${RC_COMPONENT_INSTALL_PATH}/doc/controller )
