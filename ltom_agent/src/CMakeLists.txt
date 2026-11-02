set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::core
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/repertory.cpp
  ../src/repertory_io.cpp
  ../src/graph_sightings.cpp
  ../src/recall_select.cpp
  ../src/recall_publisher.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/repertory.h
  ../src/repertory_io.h
  ../src/graph_sightings.h
  ../src/recall_select.h
  ../src/recall_publisher.h
)

LIST(APPEND LIBS ${LIBS})

# libdsr_core defines symbols that libdsr_{api,gui} only reference (UND); keep it (see episodic_agent/src).
add_link_options(-Wl,--no-as-needed)

# -march=native is REMOVED ON PURPOSE and must not come back: match the prebuilt libdsr's Eigen ABI
# (CLAUDE.md "Build"). -fno-char8_t avoids a problem with the ICE Connection lib.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
