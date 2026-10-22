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
// They are combined as an INLIER/OUTLIER MIXTURE, then by INFORMATION. The chain is either right (a draw from
// N(aperture seam, C_chain + C_aperture)) or garbage (uniform over the places the seam can be):
//
//     P(inlier | d²) = π N / (π N + (1-π) U)
//
//   · N  = (2π)^-3/2 |S|^-1/2 exp(-d²/2), S = C_chain + C_aperture, d the seam difference (yaw wrapped).
//   · U  = 1 / (π R² · 2π): the seam is the old room's origin seen from the aperture, which lies on the old room's
//     boundary, so the translation is within R of the aperture (R = the old room's radius from its origin, or the
//     door's distance from it when no outline is stored) and the yaw is anything. Geometry, not a tuning knob.
//   · π  = ½: a prior of ignorance. Neither the chain nor the door has a track record here; the live evidence is one
//     garbage chain in three seams (gold1), which does not support a tighter number. It matters only where N ~ U, i.e.
//     where the two measurements are within a few σ of being unresolvable from garbage; elsewhere the likelihood
//     ratio is ~e^±10 and π is irrelevant.
// The chain's information is scaled by P(inlier): ≈1 when it agrees with the door, ≈0 when it contradicts it by ≫σ,
// with log P(chain wrong) reported. No cutoff exists anywhere.
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

    // ── covariance-carrying SE(2) algebra (additive (x, y, yaw) parametrisation, as cortex stores RT covariance) ──
    inline Measured compose_cov(const Measured& a, const Measured& b)
    {
        const double th = a.pose.z();
        const double s = std::sin(th), c = std::cos(th);
        Eigen::Matrix2d dR;  dR << -s, -c, c, -s;
        Mat3 Ja = Mat3::Identity(), Jb = Mat3::Identity();
        Ja.block<2, 1>(0, 2) = dR * b.pose.head<2>();
        Jb.topLeftCorner<2, 2>() = rot(th);
        Measured m;
        m.pose << (a.pose.head<2>() + rot(th) * b.pose.head<2>()), wrap(a.pose.z() + b.pose.z());
        m.cov = Ja * a.cov * Ja.transpose() + Jb * b.cov * Jb.transpose();
        return m;
    }

    inline Measured inverse_cov(const Measured& p)
    {
        const double s = std::sin(-p.pose.z()), c = std::cos(-p.pose.z());
        Eigen::Matrix2d dR;  dR << -s, -c, c, -s;
        Mat3 J = Mat3::Zero();
        J.topLeftCorner<2, 2>() = -rot(-p.pose.z());
        J.block<2, 1>(0, 2)     = dR * p.pose.head<2>();
        J(2, 2) = -1.0;
        return {inverse(p.pose), J * p.cov * J.transpose()};
    }

    /// T(new<-old) through the robot: inverse(E_new) . E_old, E = T(robot<-room) with covariance in the robot frame.
    /// The two edges are independent estimates, so their covariances propagate through the composition Jacobians
    /// (never added raw: they live in different frames).
    inline Measured chain_seam(const Measured& e_new, const Measured& e_old)
    {
        return compose_cov(inverse_cov(e_new), e_old);
    }

    struct Fused
    {
        Vec3   pose = Vec3::Zero();
        Mat3   cov  = Mat3::Identity();
        double d2   = 0.0;           ///< Mahalanobis² of the chain from the aperture seam
        double p_inlier = 1.0;       ///< P(chain right | d²): the factor the chain's information is scaled by
        double log_p_wrong = 0.0;    ///< log P(chain garbage)
    };

    /// `radius` (m): the old room's extent, from its origin (see the header).
    inline Fused fuse(const Measured& chain, const Measured& aperture, double radius, double prior_inlier = 0.5)
    {
        Vec3 d = chain.pose - aperture.pose;
        d.z() = wrap(d.z());
        const Mat3 S = chain.cov + aperture.cov;
        const double d2 = d.dot(S.ldlt().solve(d));
        const double log_n = -1.5 * std::log(2.0 * M_PI) - 0.5 * std::log(S.determinant()) - 0.5 * d2;
        const double log_u = -std::log(M_PI * radius * radius * 2.0 * M_PI);
        const double lp = std::log(prior_inlier) + log_n, lq = std::log(1.0 - prior_inlier) + log_u;
        // logistic of (lp - lq), written to survive a -1e8 exponent
        const double m = std::max(lp, lq);
        const double p_in = std::exp(lp - m) / (std::exp(lp - m) + std::exp(lq - m));
        const Mat3 chain_info = p_in * chain.cov.inverse();
        const Mat3 cov = (aperture.cov.inverse() + chain_info).inverse();
        Vec3 pose = aperture.pose + cov * chain_info * d;
        pose.z() = wrap(pose.z());
        return {pose, cov, d2, p_in, (lq - m) - std::log(std::exp(lp - m) + std::exp(lq - m))};
    }
}   // namespace ltsm::seam
