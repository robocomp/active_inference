cmake_minimum_required(VERSION 3.16)
project(nbv_tests LANGUAGES CXX)

# Standalone test project for common/nbv (header-only). Run: cmake -S . -B build && make -C build && ctest --test-dir build
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

foreach(t standpoint_free_test)
  add_executable(${t} ${t}.cpp)
  target_compile_options(${t} PRIVATE -O1 -Wall)
  target_link_libraries(${t} PRIVATE Eigen3::Eigen)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
# The two older standalone nbv tests live one level up and include their headers from there.
foreach(t countable_planner_test z_occlusion_test)
  add_executable(${t} ../${t}.cpp)
  target_include_directories(${t} PRIVATE ..)
  target_compile_options(${t} PRIVATE -O1)
  target_link_libraries(${t} PRIVATE Eigen3::Eigen)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
