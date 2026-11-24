cmake_minimum_required(VERSION 3.16)
project(hood_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against etc/config.toml; header-only, no Qt/DSR. Run: LANG=es_ES.UTF-8 ./hood_card_test
add_executable(hood_card_test hood_card_test.cpp)
target_include_directories(hood_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(hood_card_test PRIVATE AGENT_DIR="${CMAKE_CURRENT_LIST_DIR}/..")
target_compile_options(hood_card_test PRIVATE -O2)
add_test(NAME hood_card_test COMMAND hood_card_test)

# Phase 0 Task 14: the planner on the shared core (common/nbv/countable_planner.h) vs a verbatim pre-task copy,
# plus the wall-mounted room-side assertion and the stand-off report (spec G3). Run: LANG=es_ES.UTF-8 ./planner_ab_test
add_executable(planner_ab_test planner_ab_test.cpp legacy_epistemic_planner.cpp ../src/epistemic_planner.cpp ../src/hood_belief.cpp)
target_include_directories(planner_ab_test PRIVATE . ../src ../../common ../../common/media_transport)
target_compile_options(planner_ab_test PRIVATE -O2)
target_link_libraries(planner_ab_test PRIVATE Eigen3::Eigen)

# shared-engine Task 10: the LEGACY belief trajectory (Model.Engine = "legacy", the per-frame recursive-Laplace update),
# recorded with the sources BEFORE the port into golden/legacy_golden.bin and compared EXACTLY after it.
# Run: ./legacy_golden_test [--record]
add_executable(legacy_golden_test legacy_golden_test.cpp ../src/hood_belief.cpp)
target_include_directories(legacy_golden_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(legacy_golden_test PRIVATE GOLDEN_FILE="${CMAKE_CURRENT_LIST_DIR}/golden/legacy_golden.bin")
target_compile_options(legacy_golden_test PRIVATE -O2)
target_link_libraries(legacy_golden_test PRIVATE Eigen3::Eigen)
add_test(NAME legacy_golden_test COMMAND legacy_golden_test)

# shared-engine Task 10: the hood belief on the shared persistent engine — 60 updates on a synthetic hanging hood (toy_model.h's
# scene builder, lifted) end within 3σ of the truth; the hanging beam part's Jacobian; the legacy switch. Run: ./hood_engine_test
add_executable(hood_engine_test hood_engine_test.cpp ../src/hood_belief.cpp)
target_include_directories(hood_engine_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(hood_engine_test PRIVATE -O2)
target_link_libraries(hood_engine_test PRIVATE Eigen3::Eigen)
add_test(NAME hood_engine_test COMMAND hood_engine_test)
