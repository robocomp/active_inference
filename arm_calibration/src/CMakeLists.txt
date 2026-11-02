set(ACTIVE_INFERENCE_COMMON_DIR "${CMAKE_CURRENT_LIST_DIR}/../../common")

# Shared ai_common::* libraries (see common/CMakeLists.txt).
#   ai_common::kinematics — the SAME Kinematics + fitted capsules + URDF the controller uses;
#                           carries Pinocchio discovery (/opt/openrobots), the link, the include
#                           dir and -DCOAL_DISABLE_HPP_FCL_WARNINGS.
#   ai_common::media      — zero-copy DDS for RGBD pixels, discovered from the MediaDescriptor on
#                           the "zed" node; carries FastDDS and every FrameKind PubSubType (which
#                           WriterCore::init/make_type_support reference unconditionally).
set(AI_COMMON_WITH_KINEMATICS ON)
add_subdirectory(${ACTIVE_INFERENCE_COMMON_DIR} ${CMAKE_CURRENT_BINARY_DIR}/ai_common EXCLUDE_FROM_ALL)
LIST(APPEND LIBS
  ai_common::kinematics
  ai_common::media
  ai_common::core         # self_projection_capsules.h (header-only)
)

# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/self_projection_viewer.cpp
)

# no-char8_t avoids a problem with the ICE Connection lib.
# CRITICAL — match the prebuilt libdsr's Eigen ABI (default alignment). Do NOT add
# -march=native NOR -DEIGEN_MAX_ALIGN_BYTES=0: either changes
# sizeof(std::optional<Eigen::Transform>) vs libdsr (default=144) and SIGBUSes /
# stack-smashes DSR agents under graph churn. (The robocompdsl skeleton ships
# -march=native -flto here — removed for exactly this reason.)
add_definitions(-g -O3 -fmax-errors=1 -fno-char8_t)
