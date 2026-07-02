find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../src/specificworker_viewers.cpp
  ../src/voxelizer_params.cpp
  ../src/graph_publisher.cpp
  ../../common/agent_presence_monitor/agent_presence_monitor.cpp
  ../../common/agent_presence_coordinator/agent_presence_coordinator.cpp
  ../../common/media_transport/media_transport.cpp
  ../../common/media_transport/generated/idl/image_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/image360_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image360_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
  ../src/scene_processor.cpp
  ../src/yolo_processor.cpp
  ../src/yolo_human.cpp
  ../src/yolo_semantic.cpp
  ../src/ricoh_yolo_worker.cpp
  ../src/onnx_providers.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/voxelizer_params.h
  ../src/graph_publisher.h
  ../src/scene_processor.h
  ../src/yolo_processor.h
  ../src/ricoh_yolo_worker.h
  ../src/graph_object_box.h
  ../../common/agent_presence_monitor/agent_presence_monitor.h
  ../../common/agent_presence_coordinator/agent_presence_coordinator.h
  ../../common/media_transport/media_transport.h
)

# Media transport plane: zero-copy DDS for RGBD pixels (carried OUT of the DSR graph).
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)

if(TARGET Qt6::OpenGL AND TARGET Qt6::OpenGLWidgets)
  LIST(APPEND SOURCES ../src/voxel_opengl_viewer.cpp)
  LIST(APPEND HEADERS ../src/voxel_opengl_viewer.h)
  LIST(APPEND LIBS Qt6::OpenGL Qt6::OpenGLWidgets)
else()
  message(WARNING "Qt OpenGL targets not found; building without voxel_opengl_viewer")
endif()

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
# Instruments voxelizer's code AND intercepts global malloc/free, so it catches the bad
# free()/use-after-free that crashes paintAndFlush even when it happens inside the
# (non-instrumented) libdsr_gui — no need to ASan-build/install the shared lib.
option(ENABLE_ASAN "Build voxelizer with AddressSanitizer" OFF)
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer ENABLED for voxelizer (-O1, frame pointers; preload-libasan link model)")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -O1)
  # Do NOT put -fsanitize=address on the LINK line. It makes libasan the first DT_NEEDED and the
  # first static initializer, which trips an Ice/IceStorm static-init-order SEGV in FactoryTable
  # during _dl_init (before main) -> the binary never starts. Instead keep the NORMAL link order
  # (Ice-safe) and leave the __asan_* symbols that the instrumented .o reference UNDEFINED in the
  # executable; they are resolved at runtime by LD_PRELOAD=libasan (run_asan.sh), which also makes
  # ASan initialise and install its malloc/free interceptors first. Net result: full redzone
  # checking of voxelizer's OWN code (so we catch the corrupting WRITE, not just the later free)
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
option(ENABLE_TSAN "Build voxelizer with ThreadSanitizer" OFF)
if(ENABLE_TSAN)
  message(STATUS "ThreadSanitizer ENABLED for voxelizer (frame pointers; -ltsan-direct link)")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -O1)
  add_link_options(-fno-omit-frame-pointer -Wl,--no-as-needed -ltsan -Wl,--as-needed)
endif()

