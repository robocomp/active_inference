cmake_minimum_required(VERSION 3.16)
project(compare_models LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# square-vs-round table model-selection harness. Eigen ONLY — the belief chain
# (table_belief.cpp + recursive_laplace.h + lidar_ray_factor.h) is pure Eigen; the round model is
# header-only (round_table_belief.h). Do NOT pull in Torch/DSR.
add_executable(compare_models
    compare_models.cpp
    ../src/table_belief.cpp)

target_include_directories(compare_models PRIVATE ../src ../../common)
target_link_libraries(compare_models PRIVATE Eigen3::Eigen)
target_compile_options(compare_models PRIVATE -O2)

# Measures the round-vs-square scoring ASYMMETRY that made a rectangular table read as round live
# (evaluate_shape refits only the round hypothesis), and checks the symmetric rule on both shapes.
add_executable(shape_asymmetry
    shape_asymmetry.cpp
    ../src/table_belief.cpp)

target_include_directories(shape_asymmetry PRIVATE ../src ../../common)
target_link_libraries(shape_asymmetry PRIVATE Eigen3::Eigen)
target_compile_options(shape_asymmetry PRIVATE -O2)

# The contour existence channel (table_contour.h + common/contour_edge) against a ray-cast synthetic
# scene: confirms a present table, refutes a blank floor, survives pose error and a floor-coloured top.
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
add_executable(contour_test
    contour_test.cpp
    ../../common/contour_edge/contour_edge_check.cpp
    ../../common/contour_edge/contour_edge_project.cpp
    ../../common/contour_edge/contour_depth_check.cpp)
target_include_directories(contour_test PRIVATE ../src ../../common ${OpenCV_INCLUDE_DIRS})
target_link_libraries(contour_test PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
target_compile_options(contour_test PRIVATE -O2)
