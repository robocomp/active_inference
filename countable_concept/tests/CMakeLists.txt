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
# K1: refrigerator and hood run as countable classes from etc/countable.toml (their etc/config.toml is the legacy agent's).
foreach(cls refrigerator hood)
  add_test(NAME countable_card_test_${cls} COMMAND countable_card_test ${CMAKE_CURRENT_LIST_DIR}/../../${cls}_concept etc/countable.toml)
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

# The class manifests load through the agent's own calls (load_geometry, adopt_span, layer_spread) from the agent's cwd,
# under es_ES, on the real ConfigLoader.cpp (genericworker.h stubbed to it). Run: LANG=es_ES.UTF-8 ./manifest_load_test
add_executable(manifest_load_test manifest_load_test.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(manifest_load_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report $ENV{ROBOCOMP}/classes)
target_compile_definitions(manifest_load_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(manifest_load_test PRIVATE -O2)
add_test(NAME manifest_load_test COMMAND manifest_load_test)
set_tests_properties(manifest_load_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# K1: every countable class's config loads through the agent's own rc::load_countable_config (refrigerator / hood from
# etc/countable.toml), from the class folder, under es_ES, on the real ConfigLoader.cpp. Run: LANG=es_ES.UTF-8 ./config_load_test
add_executable(config_load_test config_load_test.cpp ../src/countable_config.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(config_load_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(config_load_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(config_load_test PRIVATE -O2)
target_link_libraries(config_load_test PRIVATE Eigen3::Eigen)
add_test(NAME config_load_test COMMAND config_load_test)
set_tests_properties(config_load_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# The bed's compound SDF: box ∪ headboard panel (2026-11-14). Run: ./headboard_sdf_test
add_executable(headboard_sdf_test headboard_sdf_test.cpp ../src/countable_belief.cpp)
target_include_directories(headboard_sdf_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(headboard_sdf_test PRIVATE -O2)
target_link_libraries(headboard_sdf_test PRIVATE Eigen3::Eigen)
add_test(NAME headboard_sdf_test COMMAND headboard_sdf_test)

# K1: a HANGING countable class (the hood): span, SDF, beam part, birth height evidence, select box. Run: ./hanging_box_test
add_executable(hanging_box_test hanging_box_test.cpp ../src/countable_belief.cpp)
target_include_directories(hanging_box_test PRIVATE ../src ../../common ../../common/media_transport ../../common/concept_manifest)
target_compile_options(hanging_box_test PRIVATE -O2)
target_link_libraries(hanging_box_test PRIVATE Eigen3::Eigen)
add_test(NAME hanging_box_test COMMAND hanging_box_test)


# K2a: one parts table (common/ai_belief/parts_geometry.h) gives the camera SDF, the beam parts and the carve; the toilet is a
# bowl and a tank; a solid = "box" class stays bit-identical. Run: ./parts_geometry_test ../../../common/concept_manifest
add_executable(parts_geometry_test parts_geometry_test.cpp ../src/countable_belief.cpp)
target_include_directories(parts_geometry_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(parts_geometry_test PRIVATE -O2)
target_link_libraries(parts_geometry_test PRIVATE Eigen3::Eigen)
add_test(NAME parts_geometry_test COMMAND parts_geometry_test ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)

# K2a-L: the class shape proportion as a rho = 1 latent — its prior (Delta^2/12), the Jq column, the box classes' engine without the
# slot, and the cliff gone (a prior-mean sweep on a synthetic stepped object moves the fit smoothly). Run: ./proportion_latent_test DIR
add_executable(proportion_latent_test proportion_latent_test.cpp ../src/countable_belief.cpp)
target_include_directories(proportion_latent_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(proportion_latent_test PRIVATE -O2)
target_link_libraries(proportion_latent_test PRIVATE Eigen3::Eigen)
add_test(NAME proportion_latent_test COMMAND proportion_latent_test ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)

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

# Per-ray membership weights on the ray factor + the weighted LiDAR selection (spec 2026-11-15 lidar-measures G2).
add_executable(lidar_weights_test lidar_weights_test.cpp ../src/countable_belief.cpp)
target_include_directories(lidar_weights_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(lidar_weights_test PRIVATE -O2)
target_link_libraries(lidar_weights_test PRIVATE Eigen3::Eigen)
add_test(NAME lidar_weights_test COMMAND lidar_weights_test)

# The LiDAR-only frame (spec 2026-11-15 lidar-measures §7b, C1 = A): rays + one-sided constraints, no re-added class anchors.
add_executable(lidar_only_update_test lidar_only_update_test.cpp ../src/countable_belief.cpp)
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

# The countable belief with persistent shared errors and the class prior counted once (spec 2026-11-15 lidar-measures §7c.1–7c.2).
add_executable(persistent_belief_test persistent_belief_test.cpp ../src/countable_belief.cpp)
target_include_directories(persistent_belief_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(persistent_belief_test PRIVATE GOLDEN_BIN="$<TARGET_FILE:lidar_golden_test>")
target_compile_options(persistent_belief_test PRIVATE -O2)
target_link_libraries(persistent_belief_test PRIVATE Eigen3::Eigen)
add_dependencies(persistent_belief_test lidar_golden_test)
add_test(NAME persistent_belief_test COMMAND persistent_belief_test)

# First-crossing LiDAR hits + fixed-precision one-sided constraints on the persistent path (spec 2026-11-15 lidar-measures
# §7c.6): factor unit cases + the 16-case creep set. Run: ./creep_set_test [sweeps=4000]
add_executable(creep_set_test creep_set_test.cpp ../src/countable_belief.cpp)
target_include_directories(creep_set_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(creep_set_test PRIVATE -O2)
target_link_libraries(creep_set_test PRIVATE Eigen3::Eigen)
add_test(NAME creep_set_test COMMAND creep_set_test)

# live9 nightstand_9: b_chain absorbs 0.45 m on the second persistent update (root-cause brief bchain-jump, 2026-11-18).
add_executable(bchain_jump_test bchain_jump_test.cpp ../src/countable_belief.cpp)
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
add_executable(beam_staging_test beam_staging_test.cpp ../src/countable_belief.cpp)
target_include_directories(beam_staging_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(beam_staging_test PRIVATE -O2)
target_link_libraries(beam_staging_test PRIVATE Eigen3::Eigen)
add_test(NAME beam_staging_test COMMAND beam_staging_test)

# The countable belief on the smooth beam model (spec 2026-11-15 lidar-measures §7c.8 a, b; flag on): kappa, reduction to
# 7c.7, occlusion, depth basin, silhouettes, parked floors, cost. Run: ./beam_belief_test
add_executable(beam_belief_test beam_belief_test.cpp ../src/countable_belief.cpp)
target_include_directories(beam_belief_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(beam_belief_test PRIVATE -O2)
target_link_libraries(beam_belief_test PRIVATE Eigen3::Eigen)
add_test(NAME beam_belief_test COMMAND beam_belief_test)

# The persistent path's rays as the fitter stages them (spec 7c.8 Pass "live staging"): a rendered mask, the agent's
# make_persistent_label and stage_beam_cycle (countable_beam_staging.h). Run: ./beam_live_staging_test
add_executable(beam_live_staging_test beam_live_staging_test.cpp ../src/countable_belief.cpp
               ../src/countable_mask_membership.cpp)
target_include_directories(beam_live_staging_test PRIVATE ../src ../../common ../../common/media_transport ${OpenCV_INCLUDE_DIRS})
target_compile_options(beam_live_staging_test PRIVATE -O2)
target_link_libraries(beam_live_staging_test PRIVATE Eigen3::Eigen ${OpenCV_LIBS})
add_test(NAME beam_live_staging_test COMMAND beam_live_staging_test)

# spec 7c.8 rev 3.3.14: birth size at the class means, position covariance from the cloud's spread. Run: ./birth_footprint_test
add_executable(birth_footprint_test birth_footprint_test.cpp)
target_include_directories(birth_footprint_test PRIVATE ../src)
target_compile_options(birth_footprint_test PRIVATE -O2)
target_link_libraries(birth_footprint_test PRIVATE Eigen3::Eigen)
add_test(NAME birth_footprint_test COMMAND birth_footprint_test)

# Shared-engine extraction (Task 3, spec 2026-11-23 shared-engine): a 40-update persistent trajectory (birth with the birth
# position covariance, camera / LiDAR / both, parked and moving, a mode rotation, GnTrace) recorded BEFORE the extraction in
# golden/engine_golden.bin and compared after it at 1e-9 relative. Run: ./engine_golden_test [--record]
add_executable(engine_golden_test engine_golden_test.cpp ../src/countable_belief.cpp)
target_include_directories(engine_golden_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_definitions(engine_golden_test PRIVATE GOLDEN_FILE="${CMAKE_CURRENT_LIST_DIR}/golden/engine_golden.bin")
target_compile_options(engine_golden_test PRIVATE -O2)
target_link_libraries(engine_golden_test PRIVATE Eigen3::Eigen)
add_test(NAME engine_golden_test COMMAND engine_golden_test)

# The flag-on fitter's persistent latent inputs (rc::persistent_latent_inputs, countable_latent_inputs.h): sig2_cam = the
# common-mode stds squared at every range / obliquity, the chain carried by e, not here (shared-engine Task 5). Run: ./latent_inputs_test
add_executable(latent_inputs_test latent_inputs_test.cpp)
target_include_directories(latent_inputs_test PRIVATE ../src)
target_compile_options(latent_inputs_test PRIVATE -O2)
add_test(NAME latent_inputs_test COMMAND latent_inputs_test)

# The replay tap (shared-engine plan revision 2, Task R1; src/countable_replay.h): a short persistent sequence dumped with the
# tap on replays bit-identically (θ and the joint P), every recorded struct member has a fields() line (Boost.PFR), the
# by-name reader keeps defaults for missing fields, and the bytes of one bed-sized update. Run: ./replay_test
find_package(Boost REQUIRED)   # header-only Boost.PFR, test only (the agent does not use it)
add_executable(replay_test replay_test.cpp ../src/countable_belief.cpp)
target_include_directories(replay_test PRIVATE ../src ../../common ../../common/media_transport ${Boost_INCLUDE_DIRS})
target_compile_options(replay_test PRIVATE -O2)
target_link_libraries(replay_test PRIVATE Eigen3::Eigen)
add_test(NAME replay_test COMMAND replay_test)

# Offline replay of a Model.ReplayDump dump through the current engine, one ai2-style row per update.
# Run: ./replay ../../../bed_concept/etc/replay/<node>_<run>_<serial>.bin > rows.csv
add_executable(replay ../tools/replay/replay.cpp ../src/countable_belief.cpp)
# -O3 as the agent builds countable_belief.cpp (build/generated flags.make): the same code at the same optimisation, so on an
# unchanged engine the tool's dth_live / dP_live against the agent's recorded results read 0.
target_include_directories(replay PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(replay PRIVATE -O3)
target_link_libraries(replay PRIVATE Eigen3::Eigen)

# CB1: the chair's birth placement from the first view (render-predicted cloud middle, derived σ, the free hypothesis's room-span
# prior) on reconstructed G1x first views, and the yaw-mode prior's wall-side face + adoption. Run: LANG=es_ES.UTF-8 ./birth_view_placement_test
add_executable(birth_view_placement_test birth_view_placement_test.cpp ../src/countable_belief.cpp ../src/countable_config.cpp
               $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(birth_view_placement_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(birth_view_placement_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(birth_view_placement_test PRIVATE -O2)
target_link_libraries(birth_view_placement_test PRIVATE Eigen3::Eigen)
add_test(NAME birth_view_placement_test COMMAND birth_view_placement_test)
set_tests_properties(birth_view_placement_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# CB1: re-derive a countable birth from a replay dump's first camera frame (BASE vs view-conditional placement) and replay the
# belief from it through the current engine (COUNTERFACTUAL). Not a ctest test (it reads tour data).
# Run: ./birth_placement ../../../chair_concept etc/countable.toml ../../../common/concept_manifest DUMP.bin [DUMP.bin ...]
add_executable(birth_placement ../tools/replay/birth_placement.cpp ../src/countable_belief.cpp ../src/countable_config.cpp
               $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(birth_placement PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_options(birth_placement PRIVATE -O3)
target_link_libraries(birth_placement PRIVATE Eigen3::Eigen)

# F1.7 (plan rev 2): a neighbour's size std until the published attribute (S1) — its class from the node name, Δ_w/√12 and
# Δ_h/√12 from that class's manifest. Run: ./neighbour_size_test ../../../common/concept_manifest
add_executable(neighbour_size_test neighbour_size_test.cpp)
target_include_directories(neighbour_size_test PRIVATE ../src ../../common)
target_compile_options(neighbour_size_test PRIVATE -O2)
add_test(NAME neighbour_size_test COMMAND neighbour_size_test ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)

# F1.4 (plan rev 2): the wall enters once — a toilet 17 cm off its wall, parked; σ along the normal stays at the value the facts
# entered once imply (stated in the test), it does not shrink with the number of updates. Run: ./toilet_wall_test
add_executable(toilet_wall_test toilet_wall_test.cpp ../src/countable_belief.cpp)
target_include_directories(toilet_wall_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(toilet_wall_test PRIVATE -O2)
target_link_libraries(toilet_wall_test PRIVATE Eigen3::Eigen)
add_test(NAME toilet_wall_test COMMAND toilet_wall_test)

# CW1: the persistent containment (no footprint corner past a room wall; countable_containment.h) — side-on, back-on, room corner,
# mid-room, each with the term off (= BASE) and on. Run: ./containment_test
add_executable(containment_test containment_test.cpp ../src/countable_belief.cpp)
target_include_directories(containment_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(containment_test PRIVATE -O2)
target_link_libraries(containment_test PRIVATE Eigen3::Eigen)
add_test(NAME containment_test COMMAND containment_test)
# CW1 fix round 3: the brief's side-on requirement (ii) on the parked items, registered as EXPECTED failures — each passes while it
# fails and FAILS loudly once the parked item is fixed (then remove WILL_FAIL). a_isolated: review I7; a_production: I7 + M5.
add_test(NAME containment_known_failure_a_isolated COMMAND containment_test --known-failure a_isolated)
set_tests_properties(containment_known_failure_a_isolated PROPERTIES WILL_FAIL TRUE)
add_test(NAME containment_known_failure_a_production COMMAND containment_test --known-failure a_production)
set_tests_properties(containment_known_failure_a_production PROPERTIES WILL_FAIL TRUE)

# F1 round 6.2: the birth seed's H is the class-prior MAP with the top of the points the BOX explains (box-vs-headboard shares),
# not the p95 of a cloud holding the headboard. Run: ./birth_seed_headboard_test
add_executable(birth_seed_headboard_test birth_seed_headboard_test.cpp ../src/countable_belief.cpp)
target_include_directories(birth_seed_headboard_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(birth_seed_headboard_test PRIVATE -O2)
target_link_libraries(birth_seed_headboard_test PRIVATE Eigen3::Eigen)
add_test(NAME birth_seed_headboard_test COMMAND birth_seed_headboard_test)

# F1 round 6.2: the bed_1 replay pin (gated on the objects_replay1 dump; SKIP without it). Bounds derived in the script.
add_test(NAME bed_replay_pin COMMAND python3 -I ${CMAKE_CURRENT_LIST_DIR}/bed_replay_pin.py $<TARGET_FILE:replay>
         ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)

# F1 round 6.3: the wall priors enter once per PHYSICAL wall (geometric identity; renumbering; A → B → A). Run: ./wall_identity_test
add_executable(wall_identity_test wall_identity_test.cpp ../src/countable_belief.cpp)
target_include_directories(wall_identity_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(wall_identity_test PRIVATE -O2)
target_link_libraries(wall_identity_test PRIVATE Eigen3::Eigen)
add_test(NAME wall_identity_test COMMAND wall_identity_test)

# P1 follow-up (shared-engine plan rev 2): the engine's offer gain at a GIVEN standpoint — at the planner's own standpoint it is
# the proposal's promised_nats bit for bit; a seen standpoint promises less than an unseen one; request lines parse locale-free.
# Run: ./gain_query_test
add_executable(gain_query_test gain_query_test.cpp ../src/countable_belief.cpp ../src/epistemic_planner.cpp
               ../../common/robot_footprint/robot_footprint.cpp ../../common/robot_footprint/mesh_hull.cpp)
target_include_directories(gain_query_test PRIVATE ../src ../../common ../../common/media_transport)
target_compile_options(gain_query_test PRIVATE -O2)
target_link_libraries(gain_query_test PRIVATE Eigen3::Eigen)
add_test(NAME gain_query_test COMMAND gain_query_test)

# K2b: the chair's NBV face ranking on the countable binary (seat/backrest/legs on the seat_top proportion): seen only from the
# front, every other face promises more and the planner's offer is not the front. Run: LANG=es_ES.UTF-8 ./chair_face_ranking_test DIR
add_executable(chair_face_ranking_test chair_face_ranking_test.cpp ../src/countable_belief.cpp ../src/epistemic_planner.cpp
               ../src/countable_config.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp
               ../../common/robot_footprint/robot_footprint.cpp ../../common/robot_footprint/mesh_hull.cpp)
target_include_directories(chair_face_ranking_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(chair_face_ranking_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(chair_face_ranking_test PRIVATE -O2)
target_link_libraries(chair_face_ranking_test PRIVATE Eigen3::Eigen)
add_test(NAME chair_face_ranking_test COMMAND chair_face_ranking_test ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)
set_tests_properties(chair_face_ranking_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# K2c: the table's symmetry (no quotient chart needed): the manifest's c2v and its invariance check, the parts SDF invariant under
# the group, the symmetric belief's point-mass mode posterior and within-sheet yaw variance. Run: LANG=es_ES.UTF-8 ./table_symmetry_test
add_executable(table_symmetry_test table_symmetry_test.cpp ../src/countable_belief.cpp ../src/countable_config.cpp
               $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(table_symmetry_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(table_symmetry_test PRIVATE MANIFEST_DIR="${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest"
                           REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(table_symmetry_test PRIVATE -O2)
target_link_libraries(table_symmetry_test PRIVATE Eigen3::Eigen)
add_test(NAME table_symmetry_test COMMAND table_symmetry_test)
set_tests_properties(table_symmetry_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# K2c: the table's NBV face ranking on the countable binary (top + legs on the top_under proportion, symmetry c2v): seen only from
# the −y face, every other face promises more and the planner's offer is not the seen face. Run: LANG=es_ES.UTF-8 ./table_face_ranking_test DIR
add_executable(table_face_ranking_test table_face_ranking_test.cpp ../src/countable_belief.cpp ../src/epistemic_planner.cpp
               ../src/countable_config.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp
               ../../common/robot_footprint/robot_footprint.cpp ../../common/robot_footprint/mesh_hull.cpp)
target_include_directories(table_face_ranking_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(table_face_ranking_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(table_face_ranking_test PRIVATE -O2)
target_link_libraries(table_face_ranking_test PRIVATE Eigen3::Eigen)
add_test(NAME table_face_ranking_test COMMAND table_face_ranking_test ${CMAKE_CURRENT_LIST_DIR}/../../common/concept_manifest)
set_tests_properties(table_face_ranking_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# F1 round 9 (a): which classes the birth seeds on a wall — only a declared back-face attachment, never c2v (rc::birth_on_wall),
# on every shipped countable config. Run: LANG=es_ES.UTF-8 ./birth_seed_rule_test
add_executable(birth_seed_rule_test birth_seed_rule_test.cpp ../src/countable_config.cpp $ENV{ROBOCOMP}/classes/ConfigLoader/ConfigLoader.cpp)
target_include_directories(birth_seed_rule_test PRIVATE manifest_stub ../src ../../common ../../common/concept_manifest
                           ../../common/config_report ../../common/media_transport $ENV{ROBOCOMP}/classes)
target_compile_definitions(birth_seed_rule_test PRIVATE REPO_DIR="${CMAKE_CURRENT_LIST_DIR}/../..")
target_compile_options(birth_seed_rule_test PRIVATE -O2)
target_link_libraries(birth_seed_rule_test PRIVATE Eigen3::Eigen)
add_test(NAME birth_seed_rule_test COMMAND birth_seed_rule_test)
set_tests_properties(birth_seed_rule_test PROPERTIES ENVIRONMENT "LANG=es_ES.UTF-8")

# S1: the published size std (size_std_m) against the belief's covariance for a floor, a hanging and a c2v class, and the
# neighbour reader's mapping + fallback. Run: ./size_std_test
add_executable(size_std_test size_std_test.cpp ../src/countable_belief.cpp)
target_include_directories(size_std_test PRIVATE ../src ../../common ../../common/media_transport ../../common/concept_manifest)
target_compile_options(size_std_test PRIVATE -O2)
target_link_libraries(size_std_test PRIVATE Eigen3::Eigen)
add_test(NAME size_std_test COMMAND size_std_test)
