#pragma once
/*
 * face_prior.h -- the covariance of a door FACE whose RT edge states none. ONE value for every reader in ltsm
 * (topology_layout / layout_optimizer seams, room_chain, room_recognizer, room_eviction).
 *
 * It is DATA, not a choice: the second moment of the faces that DO state a covariance. Over every saved ltsm memory
 * (2026-11-12: 1137 distinct stated faces in 875 files, scratchpad face_sigmas.py) the stated sigmas were
 *     along the wall   median 0.086 m   p10..p90 0.042..0.198   RMS 0.145 m
 *     normal to it     median 0.010 m   p90 0.010               RMS 0.020 m
 *     yaw              0.0173 rad on every face (door_concept writes a constant 1 deg; not a measurement)
 * A face that states nothing is one draw from that population, so its marginal is the scale mixture over it; the
 * Gaussian nearest to that mixture (moment matching = the KL projection) has covariance E[Sigma], i.e. the RMS sigmas.
 * The median would be over-confident exactly where the evidence is missing.
 *
 * The anisotropy is the point: a door is located well ACROSS its wall (the wall pins it) and poorly ALONG it (where the
 * opening starts). The face's x axis runs along the aperture (width_m along x), so the prior is diag(along, normal) in
 * the face's own axes, rotated into the frame its pose is expressed in.
 * Replaces three earlier constants (3 cm / 0.01 rad isotropic in room_chain + room_recognizer, 0.1 m / 0.03 rad in
 * topology_layout + room_eviction), none of which came from data.
 */
#include "seam_fusion.h"

namespace ltsm::seam
{
inline constexpr double kFaceSigmaAlong = 0.145;   ///< m, along the wall (the face's x)
inline constexpr double kFaceSigmaNormal = 0.020;  ///< m, across the wall (the face's y)
inline constexpr double kFaceSigmaYaw = 0.0173;    ///< rad

/// Covariance of an unstated face whose pose has yaw `face_yaw` in the frame the covariance is wanted in.
inline Mat3 unstated_face_cov(double face_yaw)
{
    const Eigen::Matrix2d R = rot(face_yaw);
    Mat3 C = Mat3::Zero();
    C.topLeftCorner<2, 2>() =
        R * Eigen::Vector2d(kFaceSigmaAlong * kFaceSigmaAlong, kFaceSigmaNormal * kFaceSigmaNormal).asDiagonal() * R.transpose();
    C(2, 2) = kFaceSigmaYaw * kFaceSigmaYaw;
    return C;
}

/// A stated (x, y, yaw) covariance expressed in a frame rotated by `frame_yaw` relative to the target frame, re-expressed
/// in the target frame (a parent-frame RT covariance read as a room-frame one).
inline Mat3 rotate_cov(const Mat3 &C, double frame_yaw)
{
    Mat3 J = Mat3::Identity();
    J.topLeftCorner<2, 2>() = rot(frame_yaw);
    return J * C * J.transpose();
}
}   // namespace ltsm::seam
