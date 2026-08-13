set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt). ai_common::kinematics carries the
# Pinocchio discovery (/opt/openrobots via robotpkg), the link, the common/kinematics include dir
# and -DCOAL_DISABLE_HPP_FCL_WARNINGS as PUBLIC properties — so none of the four is repeated here.
set(AI_COMMON_WITH_KINEMATICS ON)
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::kinematics   # Kinematics + fitted link capsules + URDF (shared with self_calibration)
  ai_common::core         # header-only (self_projection_capsules.h, …)
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/pick_and_place_fsm.cpp
  ../src/efe_gradient.cpp
  ../src/arm_belief_viewer_3d.cpp
)

# no-char8_t to avoid a problem with ICE Connection lib.
# NOTE: -march=native removed. Pinocchio's prebuilt .so (robotpkg) is built
# with default alignment; -march=native in our TUs would promote Eigen's
# alignment (AVX2/AVX-512 → 32-/64-byte) and shift pinocchio::Data's layout
# relative to the .so, breaking model.check(data) at runtime.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t)
