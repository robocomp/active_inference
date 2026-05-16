find_package(Qt6 COMPONENTS 3DCore 3DRender 3DExtras QUIET)
find_package(Qt6 COMPONENTS OpenGL OpenGLWidgets QUIET)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/unified_voxel_grid.cpp
  ../src/yolo_seg_detector.cpp
  ../src/custom_widget.h
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/unified_voxel_grid.h
  ../src/yolo_seg_detector.h
)


if(TARGET Qt6::OpenGL AND TARGET Qt6::OpenGLWidgets)
  LIST(APPEND SOURCES ../src/voxel_opengl_viewer.cpp)
  LIST(APPEND HEADERS ../src/voxel_opengl_viewer.h)
  LIST(APPEND LIBS Qt6::OpenGL Qt6::OpenGLWidgets)
else()
  message(WARNING "Qt OpenGL targets not found; building without voxel_opengl_viewer")
endif()


# TBB is required by std::execution::par_unseq (libstdc++ on Linux)
find_package(TBB REQUIRED)
LIST(APPEND LIBS ${LIBS} TBB::tbb)

# ONNX Runtime (same path as the reference component)
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime installation")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}")
  message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
endif()
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)
LIST(APPEND LIBS ${LIBS} onnxruntime)

# OpenCV (needed by YoloSegDetector)
find_package(OpenCV REQUIRED)
LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-O3 -march=native -fmax-errors=1 -fno-char8_t)

