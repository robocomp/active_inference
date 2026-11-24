cmake_minimum_required(VERSION 3.16)
project(ai_belief_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Eigen3 REQUIRED)

# Sensor-variance helpers (header-only common/ai_belief/sensor_var.h). Run: ./sensor_var_test
add_executable(sensor_var_test sensor_var_test.cpp)
target_include_directories(sensor_var_test PRIVATE ../..)
target_compile_options(sensor_var_test PRIVATE -O2)
target_link_libraries(sensor_var_test PRIVATE Eigen3::Eigen)   # the degenerate-geometry cases include viewpoint.h
add_test(NAME sensor_var_test COMMAND sensor_var_test)

# Viewpoint rho from the odometry path relative to the object (spec 2026-11-15 lidar-measures §7c.2), moved from
# countable_concept/tests. Header-only common/ai_belief/viewpoint.h; pure Eigen. Run: ./viewpoint_rho_test
add_executable(viewpoint_rho_test viewpoint_rho_test.cpp)
target_include_directories(viewpoint_rho_test PRIVATE ../..)
target_compile_options(viewpoint_rho_test PRIVATE -O2)
target_link_libraries(viewpoint_rho_test PRIVATE Eigen3::Eigen)
add_test(NAME viewpoint_rho_test COMMAND viewpoint_rho_test)

# The shared persistent engine (common/ai_belief/persistent_engine.h) on the minimal toy model (toy_model.h; shared-engine
# Task 4, Review Focus 1, 2): an unready BeamConfig is refused once and the camera still converges; an update that measures
# nothing leaves θ bit-identical and moves P only by the latents' prediction. Run: ./engine_no_beam_test, ./engine_predict_only_test
# Task 5: the shape latent (G4) cannot be averaged away; the parked test (G1) — σ stops shrinking at one standpoint, and a
# new one adds the analytic two-view drop. Run: ./engine_shape_latent_test, ./parked_synthetic_test
# Task 6: a neighbour's published belief as a foreign density (G3, Review Focus 5). Run: ./engine_neighbour_test
# Task 7: offers under the same cap (G5, Review Focus 4) — a return view promises little, a new one a lot; promised vs delivered;
# the face ranking. Run: ./offer_return_view_test
# F1.2: the inferred process noise is the EM statistic (a wide prior corrected is not motion; a shove is). Run: ./engine_volatility_test
# F1 round 6c: the move evidence is continuous in an uninformed direction (a biased gradient there is not a move). Run: ./jump_evidence_test
foreach(t engine_no_beam_test engine_predict_only_test engine_shape_latent_test parked_synthetic_test engine_neighbour_test
          offer_return_view_test engine_volatility_test jump_evidence_test)
    add_executable(${t} ${t}.cpp)
    target_include_directories(${t} PRIVATE ../..)
    target_compile_options(${t} PRIVATE -O2)
    target_link_libraries(${t} PRIVATE Eigen3::Eigen)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
