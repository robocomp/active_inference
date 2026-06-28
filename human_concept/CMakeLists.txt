cmake_minimum_required( VERSION 3.10 )
PROJECT( human_concept )

IF ( "$ENV{ROBOCOMP}" STREQUAL "")
  MESSAGE(WARNING "ROBOCOMP variable not set. Using the default value: /home/robocomp/robocomp")
  SET (ENV{ROBOCOMP} "/home/robocomp/robocomp/")
ENDIF ( "$ENV{ROBOCOMP}" STREQUAL "")

SET(RC_COMPONENT_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "component base path" FORCE)
SET(RC_COMPONENT_DEVEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "component binary path" FORCE)
SET(RC_COMPONENT_INSTALL_PATH $ENV{ROBOCOMP} CACHE STRING "component install path")

add_subdirectory(generated)

INSTALL(FILES etc/config DESTINATION ${RC_COMPONENT_INSTALL_PATH}/etc-default/ RENAME human_concept.conf )
