#pragma once

// ── THE SEAM BETWEEN TWO ROOMS, MEASURED TWO WAYS, FUSED BY COVARIANCE ──────────────────────────────────────
// T(room_new <- room_old) is read at eviction time, while both rooms still hang off the robot. Two independent
// measurements exist when the new room is a PROTO-room:
//
//   · the ROBOT CHAIN   T(new<-robot)·T(robot<-old). Exact if both RT edges are, but a proto's RT edge is a pose
//     in a frame the estimator was free to move, and a bad edge is silent: live crossing/gold1 (2026-10-22) put
//     the robot km away in a near-singular proto frame, the chain read (-14262.68, +1667.96, -1.13) and memory
//     stored it as the seam between two rooms 4 m apart.
//   · the CROSSED APERTURE. A proto's frame IS the aperture (x along a→b, y out of the room left behind;
//     room_concept door_crossing.h proto_frame), so T(new<-old) is the inverse of the aperture pose in the old
//     room — one door node, one pose and its covariance, no robot involved.
//
// They are combined by INFORMATION (inverse covariance). The sanity check is a model term, not a gate: the chain's
// covariance is inflated by (1 + d²/dof), d² being its Mahalanobis distance from the aperture seam under the two
// covariances together. A chain consistent with the door (d² ~ dof) loses a factor ~2 of information; one that
// contradicts it by ≫σ gets weight ~ dof/d² → 0 (the Student-t / adaptive-innovation form of "trust it less
// the more it disagrees"). No cutoff exists anywhere.
//
// Poses are SE(2) triples (x, y, yaw), covariances 3×3 in the same order, in the frame of the pose's parent.

#include <Eigen/Dense>
#include <cmath>

namespace ltsm::seam
{
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;

    inline double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

    inline Eigen::Matrix2d rot(double a)
    {
        Eigen::Matrix2d R;
        R << std::cos(a), -std::sin(a), std::sin(a), std::cos(a);
        return R;
    }

    /// Inverse of the pose p (parent<-child  ⇒  child<-parent).
    inline Vec3 inverse(const Vec3& p)
    {
        const Eigen::Vector2d t = -rot(-p.z()) * p.head<2>();
        return {t.x(), t.y(), wrap(-p.z())};
    }

    /// The proto frame in the OLD room, from the door node's pose there and the side the robot is on.
    /// The door's x axis is the wall tangent; the proto's y is the aperture normal pointing INTO the new room, i.e.
    /// toward the robot (which has crossed): the proto frame is the door frame turned by 0 or π, whichever makes its
    /// y axis look at the robot. The sign is a side test, not a tolerance.
    inline Vec3 proto_in_old(const Vec3& door_in_old, const Eigen::Vector2d& robot_in_old)
    {
        const Eigen::Vector2d y_door(-std::sin(door_in_old.z()), std::cos(door_in_old.z()));
        const bool toward = (robot_in_old - door_in_old.head<2>()).dot(y_door) >= 0.0;
        return {door_in_old.x(), door_in_old.y(), toward ? door_in_old.z() : wrap(door_in_old.z() + M_PI)};
    }

    /// T(new <- old) through the aperture, with its covariance propagated from the door pose's (x, y, yaw) covariance.
    struct Measured { Vec3 pose = Vec3::Zero(); Mat3 cov = Mat3::Identity(); };

    inline Measured aperture_seam(const Vec3& door_in_old, const Mat3& door_cov, const Eigen::Vector2d& robot_in_old)
    {
        const Vec3 p = proto_in_old(door_in_old, robot_in_old);
        // inverse: t' = -R(-θ)·t, θ' = -θ.  ∂t'/∂t = -R(-θ),  ∂t'/∂θ = R'(-θ)·t,  R'(a) = [[-sin a, -cos a],[cos a, -sin a]].
        const double s = std::sin(-p.z()), c = std::cos(-p.z());
        Eigen::Matrix2d dR;  dR << -s, -c, c, -s;
        Mat3 J = Mat3::Zero();
        J.topLeftCorner<2, 2>() = -rot(-p.z());
        J.block<2, 1>(0, 2)     = dR * p.head<2>();
        J(2, 2) = -1.0;
        return {inverse(p), J * door_cov * J.transpose()};
    }

    struct Fused
    {
        Vec3   pose = Vec3::Zero();
        Mat3   cov  = Mat3::Identity();
        double d2   = 0.0;      ///< Mahalanobis² of the chain from the aperture seam, before inflation
        double chain_weight = 1.0;   ///< 1/(1 + d²/dof): the factor the chain's information was scaled by
    };

    /// Information-weighted fusion of the robot-chain seam and the aperture seam.
    inline Fused fuse(const Measured& chain, const Measured& aperture)
    {
        constexpr double dof = 3.0;
        Vec3 d = chain.pose - aperture.pose;
        d.z() = wrap(d.z());
        const double d2 = d.dot((chain.cov + aperture.cov).ldlt().solve(d));
        const double w = 1.0 / (1.0 + d2 / dof);
        const Mat3 info = aperture.cov.inverse() + w * chain.cov.inverse();
        const Mat3 cov = info.inverse();
        Vec3 pose = aperture.pose + cov * (w * chain.cov.inverse()) * d;
        pose.z() = wrap(pose.z());
        return {pose, cov, d2, w};
    }
}   // namespace ltsm::seam
