# Pinocchio (rigid-body kinematics) — installed via robotpkg into /opt/openrobots
list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")
find_package(pinocchio REQUIRED)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/pick_and_place_fsm.cpp
  ../src/kinematics.cpp
  ../src/efe_gradient.cpp
  ../src/arm_belief_viewer_3d.cpp
  ../src/self_projection_viewer.cpp
  # Zero-copy media plane (RGBD pixels carried OUT of the DSR graph) + the
  # threaded consumer that feeds the self-projection viewer off the control loop.
  ../../common/media_transport/media_transport.cpp
  ../../common/media_transport/media_source.cpp
  ../../common/media_transport/generated/idl/image_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/pick_and_place_fsm.h
  ../src/kinematics.h
  ../src/efe_gradient.h
  ../src/arm_belief_viewer_3d.h
  ../src/self_projection_viewer.h
  ../src/self_projection_types.h
  ../src/self_projection_capsules.h
  ../../common/media_transport/media_transport.h
  ../../common/media_transport/media_source.h
)

# Media transport plane: zero-copy DDS for RGB pixels (self-projection viewer).
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)
find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)
LIST(APPEND LIBS ${LIBS} fastdds fastcdr)

LIST(APPEND LIBS ${LIBS} pinocchio::pinocchio)

# no-char8_t to avoid a problem with ICE Connection lib.
# NOTE: -march=native removed. Pinocchio's prebuilt .so (robotpkg) is built
# with default alignment; -march=native in our TUs would promote Eigen's
# alignment (AVX2/AVX-512 → 32-/64-byte) and shift pinocchio::Data's layout
# relative to the .so, breaking model.check(data) at runtime.
add_definitions(-O3 -fmax-errors=1 -fno-char8_t -DCOAL_DISABLE_HPP_FCL_WARNINGS)

