/*
 * relative_pose.h — an object's pose in the ROBOT frame, predicted by odometry. SHARED, header-only.
 *
 * Room frames are used once, to initialise (from_room); after that the belief lives in the robot
 * frame and only the body twist moves it, so a room-frame jump cannot drag it along.
 * Body frame: forward = +y, side = +x (ROBOT_GEOMETRY.md); twist.linear = {side, adv, 0}.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

#include "rt_twist/rt_twist.h"

namespace rc::interaction
{

struct RelPose                          // the object frame expressed in the robot frame
{
    Eigen::Vector3f x{0, 0, 0};         // x, y (m), yaw (rad)
    Eigen::Matrix3f P = Eigen::Matrix3f::Identity();
    std::int64_t    stamp_ms = 0;
};

[[nodiscard]] inline float wrap_pi(float a)
{
    a = std::remainder(a, 2.f * float(M_PI));            // (-pi, pi]
    return a;
}

// The robot moves d = (side, adv)*dt and turns delta = rot*dt, so the object frame moves by the
// inverse: p' = R(-delta)(p - d), yaw' = yaw - delta.
// P <- J P Jt + G S Gt dt, with S = diag(var_adv, var_side, var_rot) from robot_current_speed_variance
// and G the derivative of x' w.r.t. the displacement (adv, side, rot) of the step. Zero variance adds nothing.
[[nodiscard]] inline RelPose predict(const RelPose& r, const rc::rt::Twist& tw,
                                     const Eigen::Vector3f& twist_var, float dt_s)
{
    const float side = tw.linear.x() * dt_s, adv = tw.linear.y() * dt_s, dth = tw.angular.z() * dt_s;
    const float c = std::cos(dth), s = std::sin(dth);
    Eigen::Matrix2f Rm; Rm << c, s, -s, c;                         // R(-delta)
    Eigen::Matrix2f dRm; dRm << -s, c, -c, -s;                     // d R(-delta) / d delta
    const Eigen::Vector2f q = r.x.head<2>() - Eigen::Vector2f(side, adv);

    RelPose o;
    o.x.head<2>() = Rm * q;
    o.x.z() = wrap_pi(r.x.z() - dth);
    o.stamp_ms = std::int64_t(tw.stamp_ms);

    Eigen::Matrix3f J = Eigen::Matrix3f::Identity();
    J.topLeftCorner<2, 2>() = Rm;
    Eigen::Matrix3f G = Eigen::Matrix3f::Zero();                   // columns: adv, side, rot
    G.block<2, 1>(0, 0) = -Rm.col(1);
    G.block<2, 1>(0, 1) = -Rm.col(0);
    G.block<2, 1>(0, 2) = dRm * q;
    G(2, 2) = -1.f;
    o.P = J * r.P * J.transpose() + G * twist_var.asDiagonal() * G.transpose() * dt_s;
    return o;
}

// Initialisation, the only use of room frames.
[[nodiscard]] inline RelPose from_room(const Eigen::Affine3d& room_T_robot, const Eigen::Affine3d& room_T_object,
                                       const Eigen::Matrix3f& P0)
{
    const Eigen::Affine3d rel = room_T_robot.inverse() * room_T_object;
    RelPose o;
    o.x = {float(rel.translation().x()), float(rel.translation().y()),
           wrap_pi(float(std::atan2(rel.linear()(1, 0), rel.linear()(0, 0))))};
    o.P = P0;
    return o;
}

}  // namespace rc::interaction
