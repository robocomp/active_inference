# Pinocchio (rigid-body kinematics) — installed via robotpkg into /opt/openrobots
list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")
find_package(pinocchio REQUIRED)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/kinematics.cpp
  ../src/efe_gradient.cpp
  ../src/arm_belief_viewer_3d.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/kinematics.h
  ../src/efe_gradient.h
  ../src/arm_belief_viewer_3d.h
)


LIST(APPEND LIBS ${LIBS} pinocchio::pinocchio)

# no-char8_t to avoid a problem with ICE Connection lib.
# NOTE: -march=native removed. Pinocchio's prebuilt .so (robotpkg) is built
# with default alignment; -march=native in our TUs would promote Eigen's
# alignment (AVX2/AVX-512 → 32-/64-byte) and shift pinocchio::Data's layout
# relative to the .so, breaking model.check(data) at runtime.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t -DCOAL_DISABLE_HPP_FCL_WARNINGS)

