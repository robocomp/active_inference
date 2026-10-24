# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/room_eviction.cpp
  ../src/memory_store.cpp
  ../src/passage_harvest.cpp
  ../src/passage_live.cpp
  ../src/room_recall.cpp
  ../src/topology_layout.cpp
  ../src/layout_optimizer.cpp
  ../src/layout_sidecar.cpp
  ../src/topology_view.cpp
  ../src/room_forget.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/room_eviction.h
  ../src/memory_store.h
  ../src/passage_harvest.h
  ../src/passage_live.h
  ../src/room_recall.h
  ../src/topology_layout.h
  ../src/layout_optimizer.h
  ../src/layout_sidecar.h
  ../src/topology_view.h
  ../src/room_forget.h
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

# Task S: offline test of the seam fusion (header-only). make -C build seam_fusion_test && ../bin/seam_fusion_test
add_executable(seam_fusion_test EXCLUDE_FROM_ALL ../tools/seam_fusion_test.cpp)
target_include_directories(seam_fusion_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
set_target_properties(seam_fusion_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

# The memory layout's MAP estimate on synthetic apartments (pure Eigen). make -C build layout_optimizer_test && ../bin/layout_optimizer_test
add_executable(layout_optimizer_test EXCLUDE_FROM_ALL ../tools/layout_optimizer_test.cpp ../src/layout_optimizer.cpp)
target_include_directories(layout_optimizer_test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
set_target_properties(layout_optimizer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)

# The MAP layout on REAL memory files vs the spanning tree (private DDS domain 9). make -C build layout_harness &&
# bin/layout_harness <ltsm_memory.json>...
add_executable(layout_harness EXCLUDE_FROM_ALL ../tools/layout_harness.cpp ../src/topology_layout.cpp
               ../src/layout_optimizer.cpp ../src/room_eviction.cpp)
target_include_directories(layout_harness PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src)
target_link_libraries(layout_harness dsr_core dsr_api dsr_gui fastcdr fastdds Qt6::Core Qt6::Gui Qt6::Widgets)
set_target_properties(layout_harness PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin)
