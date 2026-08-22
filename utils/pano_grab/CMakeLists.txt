cmake_minimum_required(VERSION 3.20)
project(pano_grab CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NO -march=native / -DEIGEN_MAX_ALIGN_BYTES here: they change Eigen's alignment away from the
# prebuilt libdsr's and crash with SIGBUS (see common/CMakeLists.txt and CLAUDE.md).
set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../common")
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)

find_package(OpenCV REQUIRED)
add_executable(pano_grab pano_grab.cpp)
target_link_libraries(pano_grab PRIVATE ai_common::media ${OpenCV_LIBS})
target_include_directories(pano_grab PRIVATE ${OpenCV_INCLUDE_DIRS})
