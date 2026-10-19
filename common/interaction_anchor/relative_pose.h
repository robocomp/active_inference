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

// ─── Task 12 (try73): THE ODOMETRY'S ROTATION SCALE IS UNCERTAIN, AND SHADOW'S IS MEASURED ────────────────────────────────
// A turn of Δθ is integrated from the wheels' rate; its SCALE error (wheel scrub on a differential base turning in place) is
// an error proportional to the turn, not white noise per second, and robot_current_speed_variance does not state it. The
// anchor's own frame records against Webots ground truth measure it over the initial ~5-rad spin of three runs: predicted /
// true rotation = 1.091 (try70), 1.079 (try71), 1.089 (try73) — the sim clock already corrected (SimClockRate). Without it the
// prediction claimed σ_yaw ≈ 1° after a 5-rad spin that left it 21° and 1.0 m off (try73 leg 1), and its overconfidence
// sealed it off from the beams that would have corrected it.
// So each step adds a rotation-displacement variance (k_rot·|Δθ|)², propagated through the SAME Jacobian G as the twist's
// own (its rot column: the anchor's lever arm about the robot), so lateral and depth grow with range. ⚑ k_rot is a PHYSICAL
// CALIBRATION of Shadow's odometry (the measured 7.9–9.1 %, rounded up), not a gate; the MEAN is not corrected by it (the
// sign and size of the bias are a property of the base and the floor, the variance is what the prediction owes).
inline constexpr float kRotScaleSigma = 0.09f;

// P <- J P Jt + G S Gt dt^2 + g_rot g_rotᵀ (k_rot·|Δθ|)². twist_var = velocity variances [var_adv, var_side, var_rot] in
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
    const float s_rot = kRotScaleSigma * std::abs(dth);            // the turn's scale error (see kRotScaleSigma)
    o.P += (s_rot * s_rot) * (G.col(2) * G.col(2).transpose());
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

// ─── Task 12 fix round 1 (I-1): WHAT AN UNSEEN SECOND COSTS THE CARRIED POSE ─────────────────────────────────────────────
// A pose carried while its object is NOT seen is the last sighting moved by odometry. Odometry adds its own variance with the
// MOTION (predict); what it cannot add is the error the last sighting already had and its covariance did not state — and
// that error is MEASURED: the anchor's published posterior under-covers its own error against Webots ground truth (Task 10
// calibration and the try66–try70 replays: median |lateral error| 0.055–0.125 m against a claimed σ_lat of ~0.02 m; median
// |yaw error| 0.5–6.9° against ~1°). Each frame that SEES the doorway re-tests the pose (a wrong one is refused by p_view);
// an unseen second re-tests nothing. So, while unseen, the carried covariance grows toward that measured discrepancy at the
// rate the predicted window is sized by: the full discrepancy after `horizon_s` (kAnchorPredictMs — the flagged horizon of
// "how long a prediction is trusted"). It grows inside the doorway band too (parked or spinning, where odometry adds ~0),
// so a long predicted window slows the servo instead of driving it on an 8σ pose at full speed.
// ⚑ MEASURED constants (not tuned): the across-leg medians of the |error| on published frames, isotropic in x/y (the depth
// error was not measured separately; the lateral one is the larger of the two in every replay).
inline constexpr float kUnseenSigmaXYM   = 0.08f;    // m
inline constexpr float kUnseenSigmaYawRad = 0.05f;   // rad (~2.9°)
inline void grow_unseen(RelPose& r, float dt_s, float horizon_s)
{
    if (not (dt_s > 0.f) or not (horizon_s > 0.f)) return;
    const float f = dt_s / horizon_s;
    r.P(0, 0) += f * kUnseenSigmaXYM * kUnseenSigmaXYM;     // isotropic in the plane: the same in robot and anchor axes
    r.P(1, 1) += f * kUnseenSigmaXYM * kUnseenSigmaXYM;
    r.P(2, 2) += f * kUnseenSigmaYawRad * kUnseenSigmaYawRad;
}

}  // namespace rc::interaction
