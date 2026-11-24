cmake_minimum_required(VERSION 3.16)
project(refrigerator_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./refrigerator_card_test
add_executable(refrigerator_card_test refrigerator_card_test.cpp)
target_include_directories(refrigerator_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(refrigerator_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(refrigerator_card_test PRIVATE -O2)
target_link_libraries(refrigerator_card_test PRIVATE Eigen3::Eigen)   # the I4 end-to-end case runs the evaluator
add_test(NAME refrigerator_card_test COMMAND refrigerator_card_test)

# Phase 0 Task 13: the planner on the shared core (common/nbv/countable_planner.h) vs a verbatim pre-task copy.
# Run: LANG=es_ES.UTF-8 ./planner_ab_test
add_executable(planner_ab_test planner_ab_test.cpp legacy_epistemic_planner.cpp ../src/epistemic_planner.cpp ../src/refrigerator_belief.cpp)
target_include_directories(planner_ab_test PRIVATE . ../src ../../common ../../common/media_transport)
target_compile_options(planner_ab_test PRIVATE -O2)
target_link_libraries(planner_ab_test PRIVATE Eigen3::Eigen)

# shared-engine Task 10: the LEGACY belief trajectory (Model.Engine = "legacy", the per-frame recursive-Laplace update),
# recorded with the sources BEFORE the port into golden/legacy_golden.bin and compared EXACTLY after it.
# Run: ./legacy_golden_test [--record]
add_executable(legacy_golden_test legacy_golden_test.cpp ../src/refrigerator_belief.cpp)
target_include_directories(legacy_golden_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(legacy_golden_test PRIVATE GOLDEN_FILE="${CMAKE_CURRENT_LIST_DIR}/golden/legacy_golden.bin")
target_compile_options(legacy_golden_test PRIVATE -O2)
target_link_libraries(legacy_golden_test PRIVATE Eigen3::Eigen)
add_test(NAME legacy_golden_test COMMAND legacy_golden_test)
