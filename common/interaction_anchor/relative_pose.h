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

#include "../rt_twist/rt_twist.h"   // relative: pure readers (ltsm) carry no common/ include dir

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

// Mean update: the robot moves d = (side, adv) and turns dth, so the object frame moves by the
// inverse: p' = R(-dth)(p - d), yaw' = yaw - dth.
[[nodiscard]] inline Eigen::Vector3f move_mean(const Eigen::Vector3f& x, float side, float adv, float dth)
{
    const float c = std::cos(dth), s = std::sin(dth);
    Eigen::Matrix2f Rm; Rm << c, s, -s, c;                         // R(-dth)
    Eigen::Vector3f o;
    o.head<2>() = Rm * (x.head<2>() - Eigen::Vector2f(side, adv));
    o.z() = wrap_pi(x.z() - dth);
    return o;
}

// J = d x'/d x ; G = d x'/d (adv, side, rot displacement of the step), columns in that order.
inline void predict_jacobians(const Eigen::Vector3f& x, float side, float adv, float dth,
                              Eigen::Matrix3f& J, Eigen::Matrix3f& G)
{
    const float c = std::cos(dth), s = std::sin(dth);
    Eigen::Matrix2f Rm; Rm << c, s, -s, c;
    Eigen::Matrix2f dRm; dRm << -s, c, -c, -s;                     // d R(-dth) / d dth
    const Eigen::Vector2f q = x.head<2>() - Eigen::Vector2f(side, adv);
    J = Eigen::Matrix3f::Identity();
    J.topLeftCorner<2, 2>() = Rm;
    G = Eigen::Matrix3f::Zero();
    G.block<2, 1>(0, 0) = -Rm.col(1);
    G.block<2, 1>(0, 1) = -Rm.col(0);
    G.block<2, 1>(0, 2) = dRm * q;
    G(2, 2) = -1.f;
}

// P <- J P Jt + G S Gt dt^2. twist_var = velocity variances [var_adv, var_side, var_rot] in
// (m/s)^2, (m/s)^2, (rad/s)^2, from robot_current_speed_variance; the displacement covariance over
// one step is var*dt^2 (the velocity error is slip/scale-like, correlated within the step).
// Zero variance adds nothing.
[[nodiscard]] inline RelPose predict(const RelPose& r, const rc::rt::Twist& tw,
                                     const Eigen::Vector3f& twist_var, float dt_s)
{
    const float side = tw.linear.x() * dt_s, adv = tw.linear.y() * dt_s, dth = tw.angular.z() * dt_s;
    RelPose o;
    o.x = move_mean(r.x, side, adv, dth);
    o.stamp_ms = std::int64_t(tw.stamp_ms);
    Eigen::Matrix3f J, G;
    predict_jacobians(r.x, side, adv, dth, J, G);
    o.P = J * r.P * J.transpose() + G * twist_var.asDiagonal() * G.transpose() * (dt_s * dt_s);
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
