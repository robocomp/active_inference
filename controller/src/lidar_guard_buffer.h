#pragma once
// lidar_guard_buffer.h — the live-returns net read from the obstacle tracker's LiDAR buffer (Task 14, try76 fix).
// ONE accessor, used by ControllerSession::live_returns AND by tests/test_lidar_body_guard.cpp, so the test exercises the
// buffer read and the room→robot transform the session runs, not an imitation of them.
#include "lidar_body_guard.h"
#include "lidar_buffer_types.h"

#include <optional>

namespace rc
{
// The newest cloud (room frame, z-banded by the obstacle tracker's converter) into guard returns on the robot frame of
// `room_T_robot` (ControllerRobotPose::as_transform). nullopt = no cloud in the buffer.
[[nodiscard]] inline std::optional<LidarGuardReturns> lidar_guard_returns_from_buffer(LidarPointBuffer& buffer,
                                                                                      const Eigen::Affine2f& room_T_robot,
                                                                                      std::span<const Eigen::Vector2f> body,
                                                                                      float sigma_r, float reach)
{
    const auto [cloud_opt] = buffer.read_last();
    if (not cloud_opt.has_value()) return std::nullopt;
    const auto& [xs, ys, zs] = cloud_opt.value();
    return lidar_guard_returns_room(xs, ys, room_T_robot, body, sigma_r, reach);
}
}  // namespace rc
