# Offline test of ltom_agent's pure repertory model (no DSR, no fleet).
#   cmake -S ltom_agent/tests -B ltom_agent/tests/build && make -C ltom_agent/tests/build -j8
#   ltom_agent/tests/build/test_repertory
cmake_minimum_required(VERSION 3.10)
project(ltom_agent_tests CXX)

# NO -march=native, NO -DEIGEN_MAX_ALIGN_BYTES=0 (house rule).
add_compile_options(-O2 -fmax-errors=1 -Wall -Wextra)
find_package(Qt6 COMPONENTS Core)    # kept available for the persistence test (Task 3); not linked here

add_executable(test_repertory test_repertory.cpp ../src/repertory.cpp)
set_target_properties(test_repertory PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
