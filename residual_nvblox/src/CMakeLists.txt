set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Declared BEFORE add_subdirectory() so the ai_common targets are instrumented too — behind a
# library boundary they no longer pick this up from the directory-scope add_compile_options below.
option(ENABLE_ASAN "Build residual_nvblox with AddressSanitizer" OFF)

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
# residual_nvblox: nvblox lives behind libnvblox_bridge.so (CUDA, own Eigen), built first — see nvblox_bridge/CMakeLists.txt.
set(NVBLOX_BRIDGE_LIB "${CMAKE_CURRENT_LIST_DIR}/../nvblox_bridge/lib/libnvblox_bridge.so")
if(NOT EXISTS ${NVBLOX_BRIDGE_LIB})
  message(FATAL_ERROR "build nvblox_bridge first: cmake -S nvblox_bridge -B nvblox_bridge/build && cmake --build nvblox_bridge/build -j")
endif()
LIST(APPEND LIBS
  ${NVBLOX_BRIDGE_LIB}
  ai_common::media       # zero-copy media plane: lidar3D + ZED depth consumers (was 10 spliced IDL paths)
  ai_common::presence    # agent presence monitor + coordinator
  ai_common::core        # header-only beliefs/geometry
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/residual_config.cpp
  ../src/residual_lidar_ingestor.cpp
  ../src/residual_clusterer.cpp   # LiDAR residual filter → 3D DBSCAN → observations (replaces mask_ingestor)
  ../src/residual_occupancy_grid.cpp   # PHASE-0 REBUILD: fixed room-frame log-odds occupancy grid (safety layer)
  ../src/residual_nvblox_view.cpp   # residual_nvblox: live nvblox window (Nvblox.Display)
  ../src/residual_nvblox_map.cpp    # residual_nvblox: nvblox mapper, integration, view/3D dump
)

# libdsr_core defines symbols that libdsr_{api,gui} only reference (UND). The
# framework links them under -Wl,--as-needed with core *ahead* of its consumers,
# so the linker drops core whenever no TU references it directly. Force-keep all
# libs so a light-DSR build links too.
add_link_options(-Wl,--no-as-needed)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash) or misaligns it (SIGBUS). libdsr uses default Eigen alignment; match it.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for residual_nvblox (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()
