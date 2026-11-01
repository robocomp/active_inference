#pragma once
/*
 * sim_door.h — "open door_1" in the SIMULATOR (2026-10-19, user): the chat asks webots-bridge's DoorControl to swing a
 * door of the current room. This is the simulator moving the door, not the robot pushing it; on the real robot there
 * are no robot_gt_* attributes and the tool refuses.
 *
 * WHICH WEBOTS DOOR, BY POSE (user: there are few doors). Our door is a room-frame pose; Webots' doors are world-frame
 * poses with opaque ids (DOOR_0, DOOR_1). The room frame is NOT the world frame (door_concept measured 6.45 m and
 * ~90° apart), so the door goes through the simulator's own witness of the robot:
 *     world_T_room = world_T_body(robot_gt_*) · room_T_body⁻¹
 * — the localiser's error, centimetres, in any room. tools/sim_door.py then takes the nearest advertised door and
 * refuses past SimDoor.match_max_m (doors here are >= 2.5 m apart).
 *
 * world_T_body_from_sim_gt is door_concept's (src/door_actuator.h, verified 2026-10-12 on 18031 rows): x, y are
 * metres in the Webots world; the angle is NOT the Webots yaw — the bridge sends -theta_w - pi/2. Keep both in step.
 *
 * Pure (Eigen only); tested in tests/test_sim_door.cpp.
 */

#include <cmath>
#include <numbers>

#include <Eigen/Dense>

namespace rc::chat
{

[[nodiscard]] inline Eigen::Matrix4d world_T_body_from_sim_gt(double x_m, double y_m, double raw_angle)
{
    const double th = -(raw_angle + std::numbers::pi / 2.0);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 0) = std::cos(th); M(0, 1) = -std::sin(th);
    M(1, 0) = std::sin(th); M(1, 1) = std::cos(th);
    M(0, 3) = x_m;          M(1, 3) = y_m;
    return M;
}

/// A room-frame point in the Webots world, through the robot: `room_T_body` is the robot's pose in the room.
[[nodiscard]] inline Eigen::Vector2d room_point_in_world(const Eigen::Vector2d& p_room, const Eigen::Matrix4d& room_T_body,
                                                         double gt_x, double gt_y, double gt_raw_angle)
{
    const Eigen::Matrix4d world_T_room = world_T_body_from_sim_gt(gt_x, gt_y, gt_raw_angle) * room_T_body.inverse();
    return (world_T_room * Eigen::Vector4d(p_room.x(), p_room.y(), 0.0, 1.0)).head<2>();
}

}   // namespace rc::chat
