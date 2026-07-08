set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/residual_config.cpp
  ../src/residual_scene_graph.cpp
  ../src/residual_belief.cpp
  ../src/residual_fitter.cpp
  ../src/residual_lidar_ingestor.cpp
  ../src/residual_clusterer.cpp   # LiDAR residual filter → 3D DBSCAN → observations (replaces mask_ingestor)
  ../src/residual_occupancy_grid.cpp   # PHASE-0 REBUILD: fixed room-frame log-odds occupancy grid (safety layer)
  ../src/residual_zed_boost.cpp     # dense ZED-depth boost: project box → backproject region → likelihood
  ../src/residual_zed_ingestor.cpp  # ZED depth media-plane consumer
  # Zero-copy media plane (lidar3D consumer for the YOLO-independent range factor). Mirrors room_concept:
  # media_transport.cpp is type-agnostic but references all IDL frame types, so link every PubSubTypes pair.
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/media_transport.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/lidar_plane_reader.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/image360_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ${ACTIVE_INFERENCE_COMMON_DIR}/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../src/residual_model.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.cpp
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/residual_config.h
  ../src/residual_instance.h
  ../src/residual_scene_graph.h
  ../src/residual_belief.h
  ../src/residual_clusterer.h
  ../src/residual_zed_boost.h
  ../src/residual_zed_ingestor.h
  ../src/residual_fitter.h
  ../src/residual_model.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_monitor/agent_presence_monitor.h
  ${ACTIVE_INFERENCE_COMMON_DIR}/agent_presence_coordinator/agent_presence_coordinator.h
)

# libdsr_core defines symbols that libdsr_{api,gui} only reference (UND). The
# framework links them under -Wl,--as-needed with core *ahead* of its consumers,
# so the linker drops core whenever no TU references it directly. Force-keep all
# libs so a light-DSR build links too.
add_link_options(-Wl,--no-as-needed)

# Media plane (zero-copy DDS) headers — used by bottle_lidar_ingestor.
include_directories(${ACTIVE_INFERENCE_COMMON_DIR}/media_transport)

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# NEITHER -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr's 144 → get_transformation_matrix overruns the
# caller's stack slot (stack smash) or misaligns it (SIGBUS). libdsr uses default Eigen alignment; match it.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Instruments bottle_concept's OWN code to pin the Ctrl-C shutdown segfault. See run_asan.sh.
option(ENABLE_ASAN "Build bottle_concept with AddressSanitizer" OFF)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for bottle_concept (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()
