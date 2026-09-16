# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/room_eviction.cpp
  ../src/memory_store.cpp
  ../src/passage_harvest.cpp
  ../src/passage_live.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/room_eviction.h
  ../src/memory_store.h
  ../src/passage_harvest.h
  ../src/passage_live.h
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
