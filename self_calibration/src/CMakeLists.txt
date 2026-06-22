# Shared rigid-body model (the SAME Kinematics + fitted capsules + URDF the
# controller uses) for FK + the reprojection silhouette. Pinocchio via robotpkg.
list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")
find_package(pinocchio REQUIRED)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/self_projection_viewer.cpp
  ../../common/kinematics/kinematics.cpp
  # Zero-copy media plane (RGB+depth carried OUT of the DSR graph). Discovered
  # from the graph via the MediaDescriptor on the camera ("zed") node.
  ../../common/media_transport/media_transport.cpp
  ../../common/media_transport/generated/idl/image_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/image_frameTypeObjectSupport.cxx
  # media_transport's WriterCore::init/make_type_support reference ALL FrameKind
  # PubSubTypes unconditionally, so even a subscriber-only consumer must link these.
  ../../common/media_transport/generated/idl/lidar_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/lidar_frameTypeObjectSupport.cxx
  ../../common/media_transport/generated/idl/imu_framePubSubTypes.cxx
  ../../common/media_transport/generated/idl/imu_frameTypeObjectSupport.cxx
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
  ../src/self_projection_viewer.h
  ../../common/kinematics/kinematics.h
  ../../common/kinematics/self_projection_capsules.h
  ../../common/media_transport/media_transport.h
)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/kinematics)
LIST(APPEND LIBS ${LIBS} pinocchio::pinocchio)

# Media transport plane: zero-copy DDS for RGBD pixels (self-calibration consumes
# the same plane the controller's self-projection viewer does).
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../common/media_transport)
find_package(fastcdr REQUIRED)
find_package(fastdds REQUIRED)
LIST(APPEND LIBS ${LIBS} fastdds fastcdr)

# no-char8_t avoids a problem with the ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI (default alignment). Do NOT add
# -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr (default=144) and SIGBUSes /
# stack-smashes DSR agents under graph churn. (The robocompdsl skeleton ships
# -march=native -flto here — removed for exactly this reason.)
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t -DCOAL_DISABLE_HPP_FCL_WARNINGS)
