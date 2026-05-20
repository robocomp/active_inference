# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/room_path_planner.cpp
  ../src/trajectory_controller.cpp
  ../src/viewer_2d.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/custom_widget.h
  ../src/room_path_planner.h
  ../src/viewer_2d.h
)


LIST(APPEND LIBS ${LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

