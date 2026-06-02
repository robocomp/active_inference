find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/specificworker_presence.cpp
  ../../common/agent_presence_monitor/agent_presence_monitor.cpp
  ../../common/agent_presence_coordinator/agent_presence_coordinator.cpp
  ../src/scene_processor.cpp
  ../src/lidar_track_attributor.cpp
  ../src/voxel_processor.cpp
  ../src/yolo_processor.cpp
  ../src/unified_voxel_grid.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/scene_processor.h
  ../src/lidar_track_attributor.h
  ../src/voxel_processor.h
  ../src/yolo_processor.h
  ../src/unified_voxel_grid.h
  ../src/graph_object_box.h
  ../../common/agent_presence_monitor/agent_presence_monitor.h
  ../../common/agent_presence_coordinator/agent_presence_coordinator.h
)

if(TARGET Qt6::OpenGL AND TARGET Qt6::OpenGLWidgets)
  LIST(APPEND SOURCES ../src/voxel_opengl_viewer.cpp)
  LIST(APPEND HEADERS ../src/voxel_opengl_viewer.h)
  LIST(APPEND LIBS Qt6::OpenGL Qt6::OpenGLWidgets)
else()
  message(WARNING "Qt OpenGL targets not found; building without voxel_opengl_viewer")
endif()

LIST(APPEND SOURCES ../src/yolo_viewer.cpp)
LIST(APPEND HEADERS ../src/yolo_viewer.h)

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
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

