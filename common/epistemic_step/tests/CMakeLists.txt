cmake_minimum_required(VERSION 3.16)
project(epistemic_step_tests LANGUAGES CXX)

# Standalone: runs the real rc::epistemic::step with a stand-in instance. Needs the installed DSR headers/libs only to compile
# and link the (never executed) graph calls. Run: cmake -S . -B build && make -C build && /usr/bin/ctest --test-dir build
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
enable_testing()
find_package(Eigen3 REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

add_executable(epistemic_step_test epistemic_step_test.cpp)
target_include_directories(epistemic_step_test PRIVATE /usr/local/include)
target_link_directories(epistemic_step_test PRIVATE /usr/local/lib)
target_link_libraries(epistemic_step_test PRIVATE Eigen3::Eigen Qt6::Core Qt6::Widgets dsr_api dsr_core fastdds fastcdr)
add_test(NAME epistemic_step_test COMMAND epistemic_step_test)

# The offer cache (offer_cache.h; plan rev 2 F1.1): once per belief update serial / standing-world version, never in a cooldown.
add_executable(offer_cache_test offer_cache_test.cpp)
target_include_directories(offer_cache_test PRIVATE /usr/local/include)
target_link_directories(offer_cache_test PRIVATE /usr/local/lib)
target_link_libraries(offer_cache_test PRIVATE Eigen3::Eigen Qt6::Core Qt6::Widgets dsr_api dsr_core fastdds fastcdr)
add_test(NAME offer_cache_test COMMAND offer_cache_test)
