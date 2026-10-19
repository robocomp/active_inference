#pragma once
// lidar_guard_buffer.h — the live-returns net read from the obstacle tracker's LiDAR buffer (Task 14, try76 fix).
// ONE accessor, used by ControllerSession::live_returns AND by tests/test_lidar_body_guard.cpp, so the test exercises the
// buffer read and the room→robot transform the session runs, not an imitation of them.
#include "lidar_body_guard.h"
#include "lidar_buffer_types.h"

#include <cmath>
#include <optional>
#include <span>
#include <vector>

namespace rc
{
// The newest cloud (room frame, z-banded by the obstacle tracker's converter) into guard returns on the robot frame of
// `room_T_robot` (ControllerRobotPose::as_transform). nullopt = no cloud in the buffer.
[[nodiscard]] inline std::optional<LidarGuardReturns> lidar_guard_returns_from_buffer(LidarPointBuffer& buffer,
                                                                                      const Eigen::Affine2f& room_T_robot,
                                                                                      std::span<const Eigen::Vector2f> body,
                                                                                      float sigma_r, float reach,
                                                                                      std::span<const Eigen::Vector2f> self_body = {},
                                                                                      std::span<const Eigen::Vector2f> true_body = {},
                                                                                      LidarWorldMemory* memory = nullptr)
{
    const auto [cloud_opt] = buffer.read_last();
    if (not cloud_opt.has_value()) return std::nullopt;
    const auto& [xs, ys, zs] = cloud_opt.value();
    return lidar_guard_returns_room(xs, ys, room_T_robot, body, sigma_r, reach, self_body, true_body, memory);
}

// Task 18: the newest cloud as plain robot-frame points at ANY range (the doorway gap needs the far room behind the opening:
// a return beyond the wall is what "a ray passes through" means), minus the robot itself — strictly inside the self
// footprint or the returns body, the same test lidar_guard_returns_from applies. No σ-cell merge: the gap's own scan bins it.
[[nodiscard]] inline std::optional<std::vector<Eigen::Vector2f>> lidar_robot_points_from_buffer(LidarPointBuffer& buffer,
                                                                                               const Eigen::Affine2f& room_T_robot,
                                                                                               std::span<const Eigen::Vector2f> body,
                                                                                               std::span<const Eigen::Vector2f> self_body = {})
{
    const auto [cloud_opt] = buffer.read_last();
    if (not cloud_opt.has_value()) return std::nullopt;
    const auto& [xs, ys, zs] = cloud_opt.value();
    const std::span<const Eigen::Vector2f> self = self_body.empty() ? body : self_body;
    const Eigen::Affine2f robot_from_room = room_T_robot.inverse();
    float r_circ = 0.f;   // beyond the bodies' circumscribed radius nothing is the robot (a cheap reject)
    for (const auto& v : self) r_circ = std::max(r_circ, v.norm());
    for (const auto& v : body) r_circ = std::max(r_circ, v.norm());
    std::vector<Eigen::Vector2f> out;
    const std::size_t n = std::min(xs.size(), ys.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (not std::isfinite(xs[i]) or not std::isfinite(ys[i])) continue;
        const Eigen::Vector2f q = robot_from_room * Eigen::Vector2f(xs[i], ys[i]);
        if (q.squaredNorm() <= r_circ * r_circ and (anchor_point_in_convex(self, q) or anchor_signed_distance(body, q) < 0.f)) continue;
        out.push_back(q);
    }
    return out;
}
}  // namespace rc
