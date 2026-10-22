// sim_door's pure side (src/sim_door.h): a door's room-frame centre in the Webots world, through the sim witness.
#include "../src/sim_door.h"

#include <cassert>
#include <print>

using namespace rc::chat;

namespace
{
Eigen::Matrix4d planar(double th, double x, double y)
{
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 0) = std::cos(th); M(0, 1) = -std::sin(th);
    M(1, 0) = std::sin(th); M(1, 1) = std::cos(th);
    M(0, 3) = x;            M(1, 3) = y;
    return M;
}
}   // namespace

int main()
{
    int checks = 0;
    // the bridge's angle is -theta_w - pi/2: a raw of -pi/2 is a body aligned with the world axes
    assert(world_T_body_from_sim_gt(1, 2, -std::numbers::pi / 2).isApprox(planar(0, 1, 2))); ++checks;

    // a room frame rotated ~90 deg and 6 m away from the world (door_concept measured exactly this), robot in it
    const Eigen::Matrix4d world_T_room = planar(std::numbers::pi / 2, -4.3, 4.8);
    const Eigen::Matrix4d room_T_body = planar(0.7, 1.0, -2.0);
    const Eigen::Matrix4d world_T_body = world_T_room * room_T_body;
    const double theta_w = std::atan2(world_T_body(1, 0), world_T_body(0, 0));
    const double raw = -theta_w - std::numbers::pi / 2;
    const Eigen::Vector2d door_room(-0.43, 4.61);
    const Eigen::Vector2d got = room_point_in_world(door_room, room_T_body, world_T_body(0, 3), world_T_body(1, 3), raw);
    const Eigen::Vector2d want = (world_T_room * Eigen::Vector4d(door_room.x(), door_room.y(), 0, 1)).head<2>();
    assert((got - want).norm() < 1e-9); ++checks;
    // …and the robot's own position comes back as robot_gt
    const Eigen::Vector2d at_robot = room_point_in_world(room_T_body.block<2, 1>(0, 3), room_T_body, world_T_body(0, 3),
                                                         world_T_body(1, 3), raw);
    assert((at_robot - world_T_body.block<2, 1>(0, 3)).norm() < 1e-9); ++checks;
    std::println("test_sim_door: {} checks passed", checks);
}
