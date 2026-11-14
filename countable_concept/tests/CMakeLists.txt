cmake_minimum_required(VERSION 3.16)
project(countable_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# (The perception-card test is added by Task 4, parameterised over the class folders: no folder has a
# migrated [Concept]/[Model] config before then.)

# The bed's compound SDF: box ∪ headboard panel (2026-11-14). Run: ./headboard_sdf_test
add_executable(headboard_sdf_test headboard_sdf_test.cpp ../src/countable_belief.cpp)
target_include_directories(headboard_sdf_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(headboard_sdf_test PRIVATE -O2)
target_link_libraries(headboard_sdf_test PRIVATE Eigen3::Eigen)
add_test(NAME headboard_sdf_test COMMAND headboard_sdf_test)


# The back panel comes from the class manifest ([prior.headboard]); textual reader, no Qt. Run: LANG=es_ES.UTF-8 ./back_panel_test
add_executable(back_panel_test back_panel_test.cpp ../src/countable_belief.cpp)
target_include_directories(back_panel_test PRIVATE ../src ../../common ../../common/concept_manifest ../../common/media_transport)
target_compile_definitions(back_panel_test PRIVATE MANIFEST_DIR="${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest")
target_compile_options(back_panel_test PRIVATE -O2)
target_link_libraries(back_panel_test PRIVATE Eigen3::Eigen)
add_test(NAME back_panel_test COMMAND back_panel_test)

# Furniture priors (owner 2026-11-14), moved from nightstand_concept/tests; built with nightstand's parameters (no back panel).
add_executable(furniture_prior_test furniture_prior_test.cpp ../src/countable_belief.cpp)
target_include_directories(furniture_prior_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(furniture_prior_test PRIVATE -O2)
target_link_libraries(furniture_prior_test PRIVATE Eigen3::Eigen)
add_test(NAME furniture_prior_test COMMAND furniture_prior_test)
