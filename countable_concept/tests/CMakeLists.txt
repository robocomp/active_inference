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
target_link_libraries(config_migration_test PRIVATE Eigen3::Eigen)   # countable_beam_config.h (beam_config_problem) includes beam_ray.h
add_test(NAME config_migration_test COMMAND config_migration_test)
set_tests_properties(config_migration_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# The bed's compound SDF: box ∪ headboard panel (2026-11-14). Run: ./headboard_sdf_test
add_executable(headboard_sdf_test headboard_sdf_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(headboard_sdf_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(headboard_sdf_test PRIVATE -O2)
target_link_libraries(headboard_sdf_test PRIVATE Eigen3::Eigen)
add_test(NAME headboard_sdf_test COMMAND headboard_sdf_test)


# The back panel comes from the class manifest ([prior.headboard]); textual reader, no Qt. Run: LANG=es_ES.UTF-8 ./back_panel_test
add_executable(back_panel_test back_panel_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(back_panel_test PRIVATE ../src ../../common ../../common/concept_manifest ../../common/media_transport)
target_compile_definitions(back_panel_test PRIVATE MANIFEST_DIR="${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest")
target_compile_options(back_panel_test PRIVATE -O2)
target_link_libraries(back_panel_test PRIVATE Eigen3::Eigen)
add_test(NAME back_panel_test COMMAND back_panel_test)

# Furniture priors (owner 2026-11-14), moved from nightstand_concept/tests; built with nightstand's parameters (no back panel).
add_executable(furniture_prior_test furniture_prior_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
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
add_executable(class_evidence_test class_evidence_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(class_evidence_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(class_evidence_test PRIVATE -O2)
target_link_libraries(class_evidence_test PRIVATE Eigen3::Eigen)
add_test(NAME class_evidence_test COMMAND class_evidence_test)

# Likelihood-only (H, w, h) summary beside the posterior (spec 2026-11-14 countable class existence §9: the class size
# prior is counted once). Same synthetic clouds as CountableBelief::self_test, bed parameters. Run: ./size_likelihood_test
add_executable(size_likelihood_test size_likelihood_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(size_likelihood_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(size_likelihood_test PRIVATE -O2)
target_link_libraries(size_likelihood_test PRIVATE Eigen3::Eigen)
add_test(NAME size_likelihood_test COMMAND size_likelihood_test)

# The class factor of existence (spec 2026-11-14 countable class existence §3, §9): class_factor_step, apply_class_step and the
# product p_exists() = occupancy x class. Run: ./class_factor_test
add_executable(class_factor_test class_factor_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
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
add_executable(lidar_golden_test lidar_golden_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(lidar_golden_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(lidar_golden_test PRIVATE -O2)
target_link_libraries(lidar_golden_test PRIVATE Eigen3::Eigen)
add_test(NAME lidar_golden_test COMMAND lidar_golden_test)

# Per-ray membership weights on the ray factor + the weighted LiDAR selection (spec 2026-11-15 lidar-measures G2).
add_executable(lidar_weights_test lidar_weights_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(lidar_weights_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(lidar_weights_test PRIVATE -O2)
target_link_libraries(lidar_weights_test PRIVATE Eigen3::Eigen)
add_test(NAME lidar_weights_test COMMAND lidar_weights_test)

# The LiDAR-only frame (spec 2026-11-15 lidar-measures §7b, C1 = A): rays + one-sided constraints, no re-added class anchors.
add_executable(lidar_only_update_test lidar_only_update_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(lidar_only_update_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(lidar_only_update_test PRIVATE -O2)
target_link_libraries(lidar_only_update_test PRIVATE Eigen3::Eigen)
add_test(NAME lidar_only_update_test COMMAND lidar_only_update_test)

# The persistent shared-error latents (spec 2026-11-15 lidar-measures §7c.2): header-only common/ai_belief/persistent_update.h.
add_executable(persistent_update_test persistent_update_test.cpp)
target_include_directories(persistent_update_test PRIVATE ../../common)
target_compile_options(persistent_update_test PRIVATE -O2)
target_link_libraries(persistent_update_test PRIVATE Eigen3::Eigen)
add_test(NAME persistent_update_test COMMAND persistent_update_test)

# Mask membership m = pi_mask * Phi(s/sigma_px) (spec 2026-11-15 lidar-measures G2); OpenCV only for the distance transform.
find_package(OpenCV REQUIRED)
add_executable(mask_membership_test mask_membership_test.cpp ../src/countable_mask_membership.cpp)
target_include_directories(mask_membership_test PRIVATE ../src ${OpenCV_INCLUDE_DIRS})
target_compile_options(mask_membership_test PRIVATE -O2)
target_link_libraries(mask_membership_test PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
add_test(NAME mask_membership_test COMMAND mask_membership_test)

# Viewpoint rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2). Pure Eigen.
add_executable(viewpoint_rho_test viewpoint_rho_test.cpp ../src/countable_viewpoint.cpp)
target_include_directories(viewpoint_rho_test PRIVATE ../src ../../common)
target_compile_options(viewpoint_rho_test PRIVATE -O2)
target_link_libraries(viewpoint_rho_test PRIVATE Eigen3::Eigen)
add_test(NAME viewpoint_rho_test COMMAND viewpoint_rho_test)

# The countable belief with persistent shared errors and the class prior counted once (spec 2026-11-15 lidar-measures §7c.1–7c.2).
add_executable(persistent_belief_test persistent_belief_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(persistent_belief_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(persistent_belief_test PRIVATE GOLDEN_BIN="$<TARGET_FILE:lidar_golden_test>")
target_compile_options(persistent_belief_test PRIVATE -O2)
target_link_libraries(persistent_belief_test PRIVATE Eigen3::Eigen)
add_dependencies(persistent_belief_test lidar_golden_test)
add_test(NAME persistent_belief_test COMMAND persistent_belief_test)

# First-crossing LiDAR hits + fixed-precision one-sided constraints on the persistent path (spec 2026-11-15 lidar-measures
# §7c.6): factor unit cases + the 16-case creep set. Run: ./creep_set_test [sweeps=4000]
add_executable(creep_set_test creep_set_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(creep_set_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(creep_set_test PRIVATE -O2)
target_link_libraries(creep_set_test PRIVATE Eigen3::Eigen)
add_test(NAME creep_set_test COMMAND creep_set_test)

# live9 nightstand_9: b_chain absorbs 0.45 m on the second persistent update (root-cause brief bchain-jump, 2026-11-18).
add_executable(bchain_jump_test bchain_jump_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(bchain_jump_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(bchain_jump_test PRIVATE -O2)
target_link_libraries(bchain_jump_test PRIVATE Eigen3::Eigen)
add_test(NAME bchain_jump_test COMMAND bchain_jump_test)

# Per-tick transform memo (tick_memo.h): same key -> one call, new tick -> recompute. Header-only, no graph.
add_executable(tick_memo_test tick_memo_test.cpp)
target_include_directories(tick_memo_test PRIVATE ../src)
target_compile_options(tick_memo_test PRIVATE -O2)
add_test(NAME tick_memo_test COMMAND tick_memo_test)

# The smooth beam likelihood in one dimension (spec 2026-11-15 lidar-measures §7c.8 a): header-only common/ai_belief/beam_ray.h.
add_executable(beam_ray_test beam_ray_test.cpp)
target_include_directories(beam_ray_test PRIVATE ../../common)
target_compile_options(beam_ray_test PRIVATE -O2)
target_link_libraries(beam_ray_test PRIVATE Eigen3::Eigen)   # Task 17: oriented-box parts, 6x6 curvature
add_test(NAME beam_ray_test COMMAND beam_ray_test)

# Ray staging for the smooth beam model + the persistent label (spec 2026-11-15 lidar-measures §7c.8): header-only lidar_select.h.
add_executable(beam_staging_test beam_staging_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(beam_staging_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(beam_staging_test PRIVATE -O2)
target_link_libraries(beam_staging_test PRIVATE Eigen3::Eigen)
add_test(NAME beam_staging_test COMMAND beam_staging_test)

# The countable belief on the smooth beam model (spec 2026-11-15 lidar-measures §7c.8 a, b; flag on): kappa, reduction to
# 7c.7, occlusion, depth basin, silhouettes, parked floors, cost. Run: ./beam_belief_test
add_executable(beam_belief_test beam_belief_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp)
target_include_directories(beam_belief_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(beam_belief_test PRIVATE -O2)
target_link_libraries(beam_belief_test PRIVATE Eigen3::Eigen)
add_test(NAME beam_belief_test COMMAND beam_belief_test)

# The persistent path's rays as the fitter stages them (spec 7c.8 Pass "live staging"): a rendered mask, the agent's
# make_persistent_label and stage_beam_cycle (countable_beam_staging.h). Run: ./beam_live_staging_test
add_executable(beam_live_staging_test beam_live_staging_test.cpp ../src/countable_belief.cpp ../src/countable_viewpoint.cpp
               ../src/countable_mask_membership.cpp)
target_include_directories(beam_live_staging_test PRIVATE ../src ../../common ../../common/media_transport ${OpenCV_INCLUDE_DIRS})
target_compile_options(beam_live_staging_test PRIVATE -O2)
target_link_libraries(beam_live_staging_test PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
add_test(NAME beam_live_staging_test COMMAND beam_live_staging_test)
