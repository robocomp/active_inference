cmake_minimum_required(VERSION 3.16)
project(countable_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Perception card: make_card() against <class folder>/etc/config.toml; header-only, no Qt/DSR. Built ONCE, run per class
# folder (the agent dir is argv[1]). Run: LANG=es_ES.UTF-8 ./countable_card_test ../../bed_concept [--json]
add_executable(countable_card_test countable_card_test.cpp)
target_include_directories(countable_card_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(countable_card_test PRIVATE -O2)
target_link_libraries(countable_card_test PRIVATE Eigen3::Eigen)   # the I4 end-to-end case runs the evaluator
foreach(cls nightstand bed toilet)
  add_test(NAME countable_card_test_${cls} COMMAND countable_card_test ${CMAKE_CURRENT_LIST_DIR}/../../${cls}_concept)
  set_tests_properties(countable_card_test_${cls} PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")
endforeach()

# Task 4 config migration: every pre-countable value survives the section/key rename (old text read from git at BASE_REV,
# the commit before the migration), and a config with a [<X>Model]/[<X>Concept] section is refused. Compiles RoboComp's
# ConfigLoader (as generated/ does) so the check runs on the real loader. Run: LANG=es_ES.UTF-8 ./config_migration_test
if ("$ENV{ROBOCOMP}" STREQUAL "")
  set(ENV{ROBOCOMP} "/home/robocomp/robocomp/")
endif()
add_executable(config_migration_test config_migration_test.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(config_migration_test PRIVATE ../src ../../common/concept_manifest $ENV{ROBOCOMP}/classes)
target_compile_definitions(config_migration_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../.." BASE_REV="476000a8")
target_compile_options(config_migration_test PRIVATE -O2)
add_test(NAME config_migration_test COMMAND config_migration_test)
set_tests_properties(config_migration_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

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

# Display assets by class-folder convention: nightstand publishes its basecolor texture, bed/toilet none. Run: ./assets_test
add_executable(assets_test assets_test.cpp)
target_include_directories(assets_test PRIVATE ../src)
target_compile_definitions(assets_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
add_test(NAME assets_test COMMAND assets_test)

# Class evidence (spec 2026-11-14 countable class existence): shape log-evidence ratio + birth height density; header-only.
add_executable(class_evidence_test class_evidence_test.cpp ../src/countable_belief.cpp)
target_include_directories(class_evidence_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(class_evidence_test PRIVATE -O2)
target_link_libraries(class_evidence_test PRIVATE Eigen3::Eigen)
add_test(NAME class_evidence_test COMMAND class_evidence_test)

# Likelihood-only (H, w, h) summary beside the posterior (spec 2026-11-14 countable class existence §9: the class size
# prior is counted once). Same synthetic clouds as CountableBelief::self_test, bed parameters. Run: ./size_likelihood_test
add_executable(size_likelihood_test size_likelihood_test.cpp ../src/countable_belief.cpp)
target_include_directories(size_likelihood_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(size_likelihood_test PRIVATE -O2)
target_link_libraries(size_likelihood_test PRIVATE Eigen3::Eigen)
add_test(NAME size_likelihood_test COMMAND size_likelihood_test)

# The class factor of existence (spec 2026-11-14 countable class existence §3, §9): class_factor_step, apply_class_step and the
# product p_exists() = occupancy x class. Run: ./class_factor_test
add_executable(class_factor_test class_factor_test.cpp ../src/countable_belief.cpp)
target_include_directories(class_factor_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(class_factor_test PRIVATE -O2)
target_link_libraries(class_factor_test PRIVATE Eigen3::Eigen)
add_test(NAME class_factor_test COMMAND class_factor_test)

# [[shape.part]] of the class manifests + the countable layout on a fitted box (feeds the 3-D LiDAR existence carve).
# Run: LANG=es_ES.UTF-8 ./shape_parts_test
add_executable(shape_parts_test shape_parts_test.cpp)
target_include_directories(shape_parts_test PRIVATE ../src ../../common ../../common/concept_manifest)
target_compile_definitions(shape_parts_test PRIVATE MANIFEST_DIR="${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest")
target_compile_options(shape_parts_test PRIVATE -O2)
target_link_libraries(shape_parts_test PRIVATE Eigen3::Eigen)
add_test(NAME shape_parts_test COMMAND shape_parts_test)

# Offline carve replay (tools/carve_replay_rays.py -> RAYS.bin): scores simulated LiDAR sweeps with the agent's own
# carve_box and carve_parts through solid_delta. Run: ./carve_replay RAYS.bin ../../common/concept_manifest
add_executable(carve_replay ../tools/carve_replay.cpp)
target_include_directories(carve_replay PRIVATE ../src ../../common)
target_compile_options(carve_replay PRIVATE -O2)
target_link_libraries(carve_replay PRIVATE Eigen3::Eigen)

# Golden 40-frame fit trajectory (hex floats, bit-exact at -O2 for this compiler): the lidar-measures change must not move it.
add_executable(lidar_golden_test lidar_golden_test.cpp ../src/countable_belief.cpp)
target_include_directories(lidar_golden_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(lidar_golden_test PRIVATE -O2)
target_link_libraries(lidar_golden_test PRIVATE Eigen3::Eigen)
add_test(NAME lidar_golden_test COMMAND lidar_golden_test)
