set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sanitizer switches are declared HERE, before add_subdirectory(), so the ai_common targets are
# instrumented too — behind a library boundary they no longer pick this up from the directory-scope
# add_compile_options further down. (The if() blocks with the full rationale stay at the bottom.)
option(ENABLE_ASAN "Build retina with AddressSanitizer" OFF)
option(ENABLE_TSAN "Build retina with ThreadSanitizer" OFF)

# Shared ai_common::* libraries (see common/CMakeLists.txt). Placed FIRST, so these targets carry
# their own explicit flags instead of silently inheriting the ones set further down this file.
# NOTE these are STATIC archives, so the CMAKE_BUILD_RPATH/CMAKE_INSTALL_RPATH overwrite that the
# ONNX Runtime block below performs does not affect finding them at runtime.
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::media        # zero-copy DDS for RGBD pixels, carried OUT of the DSR graph
  ai_common::presence     # agent presence monitor + coordinator
  ai_common::core         # header-only beliefs/geometry
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/specificworker_viewers.cpp
  ../src/retina_params.cpp
  ../src/graph_publisher.cpp
  ../src/scene_processor.cpp
  ../src/media_plane_source.cpp
  ../src/zed_frame_aligner.cpp
  ../src/model_projection_overlay.cpp
  ../src/ricoh_projection_overlay.cpp
  ../src/yolo_processor.cpp
  ../src/yolo_seg_detector.cpp
  ../src/seg_stage.cpp
  ../src/pose_stage.cpp
  ../src/semantic_stage.cpp
  ../src/semantic_mask_stage.cpp
  ../src/semantic_stage_360.cpp
  ../src/sam2_stage.cpp
  ../src/sam2_processor.cpp
  ../src/bearing_stage.cpp
  ../src/depth_stage.cpp
  ../src/depth_processor.cpp
  ../src/depth_dataset.cpp
  ../src/depth_enrichment.cpp
  ../src/room_envelope_depth.cpp
  ../src/yolo_human.cpp
  ../src/yolo_semantic.cpp
  ../src/ricoh_source.cpp
  ../src/zed_source.cpp
  ../src/perception_worker.cpp
  ../src/onnx_providers.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/retina_params.h
  ../src/graph_publisher.h
  ../src/scene_processor.h
  ../src/media_plane_source.h
  ../src/zed_frame_aligner.h
  ../src/yolo_processor.h
  ../src/yolo_seg_detector.h
  ../src/ricoh_source.h
  ../src/zed_source.h
  ../src/perception_worker.h
  ../src/perception_stage.h
  ../src/seg_stage.h
  ../src/pose_stage.h
  ../src/semantic_stage.h
  ../src/semantic_mask_stage.h
  ../src/sam2_stage.h
  ../src/sam2_processor.h
  ../src/bearing_stage.h
  ../src/graph_object_box.h
  ../src/model_projection_overlay.h
  ../src/ricoh_projection_overlay.h
  ../src/depth_dataset.h
  ../src/depth_enrichment.h
  ../src/room_envelope_depth.h
)
# The 3-D GL view (voxel_opengl_viewer + obj_loader) MOVED OUT to the `viewer3d` agent — it drew other
# agents' products and its room dependency was gating this whole component behind localization. What
# remains here are the two CAMERA windows (ZED raster + ricoh panorama), raster QLabels needing no GL.

LIST(APPEND SOURCES ../src/yolo_viewer.cpp)
LIST(APPEND HEADERS ../src/yolo_viewer.h)
LIST(APPEND SOURCES ../src/image_popup_viewer.cpp)
LIST(APPEND HEADERS ../src/image_popup_viewer.h)

# TBB is required by std::execution::par_unseq (libstdc++ on Linux)
find_package(TBB REQUIRED)
LIST(APPEND LIBS ${LIBS} TBB::tbb)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime installation prefix")

set(_onnxruntime_hints)
if(ONNXRUNTIME_ROOT)
  list(APPEND _onnxruntime_hints "${ONNXRUNTIME_ROOT}")
endif()
list(APPEND _onnxruntime_hints
  "/usr/local/onnxruntime"
  "/opt/onnxruntime"
  "/opt/onnxruntime-gpu"
  "/usr"
  "/usr/local")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_onnxruntime_hints}
  PATH_SUFFIXES include)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${_onnxruntime_hints}
  PATH_SUFFIXES lib lib64)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to the unpacked/install prefix, "
    "for example: cmake -S . -B build -DONNXRUNTIME_ROOT=/opt/onnxruntime")
endif()

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
message(STATUS "Using ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
set(CMAKE_INSTALL_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")

LIST(APPEND LIBS ${LIBS} onnxruntime::onnxruntime)

# OpenCV (needed by YoloSegDetector)
find_package(OpenCV REQUIRED)
LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI: use DEFAULT alignment (alignof(Transform)=16).
# Do NOT add -march=native (→32) NOR -DEIGEN_MAX_ALIGN_BYTES=0 (→8): either changes
# sizeof(std::optional<Eigen::Transform>) (160 / 136) vs libdsr's 144, so get_transformation_matrix
# returning std::optional<Mat::RTMat> overruns the caller's stack slot (stack smash in
# build_graph_object_box) — or misaligns it (SIGBUS). libdsr is built with default Eigen alignment;
# we must too.
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t)

# AddressSanitizer (opt-in, separate build dir):
#   cmake -B build-asan -DENABLE_ASAN=ON && cmake --build build-asan -j
# Instruments retina's code AND intercepts global malloc/free, so it catches the bad
# free()/use-after-free that crashes paintAndFlush even when it happens inside the
# (non-instrumented) libdsr_gui — no need to ASan-build/install the shared lib.
# (option() is declared at the top of this file so ai_common sees it before add_subdirectory().)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for retina (-O1, frame pointers; preload-libasan link model)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Do NOT put -fsanitize=address on the LINK line. It makes libasan the first DT_NEEDED and the
  # first static initializer, which trips an Ice/IceStorm static-init-order SEGV in FactoryTable
  # during _dl_init (before main) -> the binary never starts. Instead keep the NORMAL link order
  # (Ice-safe) and leave the __asan_* symbols that the instrumented .o reference UNDEFINED in the
  # executable; they are resolved at runtime by LD_PRELOAD=libasan (run_asan.sh), which also makes
  # ASan initialise and install its malloc/free interceptors first. Net result: full redzone
  # checking of retina's OWN code (so we catch the corrupting WRITE, not just the later free)
  # AND a binary that actually boots.
  # Link libasan DIRECTLY (not via the -fsanitize driver). The driver injects an __asan_init
  # entry into .preinit_array, which runs BEFORE any shared-library init -> ASan is active during
  # libIceStorm's static init and trips a latent Ice FactoryTable init-order SEGV (before main).
  # Linking -lasan ourselves resolves the __asan_* symbols (as UND dynamic, satisfied by libasan)
  # WITHOUT a preinit entry. The run script then strips the libasan DT_NEEDED and LD_PRELOADs it,
  # so ASan initialises first (interceptors up) while the Ice/IceStorm init order stays normal.
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -lasan -Wl,--as-needed)
endif()

# ThreadSanitizer (opt-in, separate build dir) — to pin the bottle-join DATA RACE that corrupts the
# heap (random SEGVs in paint/cv::Mat/onnx). Same -ltsan-direct link model as ASan above so the
# binary boots past the Ice static-init wall; run_tsan.sh strips the libtsan DT_NEEDED + LD_PRELOADs
# it. Build against the TSan cortex (cmake passes LIBRARY_PATH to build-tsan libs).
#   cmake -B build-tsan -DENABLE_TSAN=ON && cmake --build build-tsan -j
# (option() is declared at the top of this file so ai_common sees it before add_subdirectory().)
if(ENABLE_TSAN)
  message(STATUS "ThreadSanitizer ENABLED for retina (frame pointers; -ltsan-direct link)")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -O1)
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -ltsan -Wl,--as-needed)
endif()

