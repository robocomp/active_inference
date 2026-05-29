find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../../common/agent_presence_monitor/agent_presence_monitor.cpp
  ../src/specificworker.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../../common/agent_presence_monitor/agent_presence_monitor.h
  ../src/specificworker.h
)

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

