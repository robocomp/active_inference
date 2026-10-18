set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::core         # header-only helpers (room_resolve) + common/episodic through its include dir
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/dsr_fact_source.cpp
  ../src/em_graph.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/dsr_fact_source.h
  ../src/em_graph.h
)

LIST(APPEND LIBS ${LIBS})

# libdsr_core defines symbols that libdsr_{api,gui} only reference (UND). The framework links them
# under -Wl,--as-needed with core AHEAD of its consumers, so the linker drops core whenever no TU
# references it directly. This agent barely touches DSR, so force-keep. (bottle_concept/src:40)
add_link_options(-Wl,--no-as-needed)

# The generated template shipped -O3 -march=native -flto=auto -fno-plt here; -march=native is
# REMOVED ON PURPOSE and must not come back.
# CRITICAL -- match the prebuilt libdsr's Eigen ABI: DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (->32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (->8): either changes
# sizeof(std::optional<Eigen::Transform>) against libdsr's 144, so get_transformation_matrix
# overruns the caller's stack slot (stack smash) or misaligns it (SIGBUS).
# -fno-char8_t avoids a problem with the ICE Connection lib.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
