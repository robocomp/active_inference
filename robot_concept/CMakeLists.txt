cmake_minimum_required( VERSION 3.10 )

option(ROBOT_CONCEPT_PREFER_GCC15 "Prefer gcc-15/g++-15 when no compiler is explicitly selected" ON)

if(ROBOT_CONCEPT_PREFER_GCC15 AND NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
  find_program(ROBOT_CONCEPT_GCC15 gcc-15)
  find_program(ROBOT_CONCEPT_GXX15 g++-15)
  if(ROBOT_CONCEPT_GCC15 AND ROBOT_CONCEPT_GXX15)
    message(STATUS "robot_concept: preferring GCC 15 toolchain: ${ROBOT_CONCEPT_GCC15}, ${ROBOT_CONCEPT_GXX15}")
    set(CMAKE_C_COMPILER "${ROBOT_CONCEPT_GCC15}" CACHE FILEPATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "${ROBOT_CONCEPT_GXX15}" CACHE FILEPATH "CXX compiler" FORCE)
  endif()
endif()

PROJECT( robot_concept )

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

INSTALL(FILES etc/config DESTINATION ${RC_COMPONENT_INSTALL_PATH}/etc-default/ RENAME robot_concept.conf )
INSTALL(FILES etc/config.toml DESTINATION ${RC_COMPONENT_INSTALL_PATH}/etc-default/ RENAME robot_concept.toml )

#ADD_CUSTOM_TARGET(doc doxygen Doxyfile)
#ADD_CUSTOM_TARGET(installdoc mkdir -p ${RC_COMPONENT_INSTALL_PATH}/doc COMMAND cp -R doc/html ${RC_COMPONENT_INSTALL_PATH}/doc/robot_concept )
