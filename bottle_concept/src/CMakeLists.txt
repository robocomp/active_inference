set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Declared BEFORE add_subdirectory() so the ai_common targets are instrumented too — behind a
# library boundary they no longer pick this up from the directory-scope add_compile_options below.
option(ENABLE_ASAN "Build bottle_concept with AddressSanitizer" OFF)

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy media plane (was 10 spliced IDL paths + an include_directories)
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::perception   # mask_ingestor
  ai_common::ui           # TimeSeriesPlot (compiled Q_OBJECT)
  ai_common::ui_headers   # custom_widget.h / belief_inspector.h — header-only, mocced by OUR autogen
  ai_common::core         # header-only beliefs (ai_belief/dof_spec.h, …)
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/bottle_config.cpp
  ../src/bottle_evaluator.cpp
  ../src/bottle_scene_graph.cpp
  ../src/bottle_belief.cpp
  ../src/bottle_fitter.cpp
  ../src/bottle_existence.cpp
  ../src/bottle_lidar_ingestor.cpp
  ../src/epistemic_planner.cpp
  ../src/bottle_affordance.cpp
  ../src/bottle_model.cpp
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
  message(STATUS "AddressSanitizer ENABLED for bottle_concept (-O1, frame pointers; -lasan-direct link)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init entry
  # into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # -lasan resolves __asan_* as UND dynamic symbols WITHOUT a preinit entry; run_asan.sh then strips
  # the libasan DT_NEEDED and LD_PRELOADs it, so ASan inits first while Ice's init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()
