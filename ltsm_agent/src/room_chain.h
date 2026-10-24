#pragma once
/*
 * room_chain.h -- the METRIC evidence for "which known room am I in?", from two chains of seams that disagree.
 *
 * The robot has just crossed a door nobody had crossed before. Two independent routes put it in the frame of the
 * reference room (the start of the path):
 *
 *   the PATH chain   the seams of the passages the robot ACTUALLY crossed, then the odometry since the last of them;
 *   the OWN chain    for each remembered room k, the seams of the shortest passage path in memory from the reference
 *                    room to k, i.e. where memory says k lies in the reference frame.
 *
 * The robot is in k exactly when the first lands inside k's outline as the second places it. The two chains share no
 * seam wherever memory holds a loop, which is the whole content of the evidence: a closed loop is a check.
 * ★ The OWN chain of k never uses the LAST passage of the path (the door whose far side is being identified): going
 *   through it would assume the answer. Passages crossed earlier ARE usable, they are facts already.
 *
 * Covariances: a seam carries the covariance of its two door faces (the RT edge of each, when stated), else a prior of
 * ignorance (sigma 3 cm / 0.01 rad per face, the door-pose covariance the repo's seam tests already use). Memory stores
 * none today (cortex drops rt_covariance on JSON load), so the prior is what is normally read. Not a gate: a
 * covariance that enters the Gaussian whose mass inside the outline is the evidence.
 *
 * Mass inside an outline = deterministic Gauss-Hermite quadrature (9x9) over the (x, y) Gaussian. Where the quadrature
 * resolves no mass (the outline is further than ~4 sigma) the Gaussian tail bound exp(-d^2/2) is used, d the
 * Mahalanobis distance to the outline's edge, so the answer stays finite, monotone in the distance, and floor-free.
 */
#include "room_eviction.h"
#include "../../common/room_recognition/room_posterior.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <string>
#include <vector>

namespace ltsm
{
struct ChainPose { Eigen::Vector3d pose = Eigen::Vector3d::Zero(); Eigen::Matrix3d cov = Eigen::Matrix3d::Zero(); };   // SE(2) in the reference room

struct KnownRoomMetric
{
    std::uint64_t room_id = 0;
    std::string   mem_room;
    bool          chain_ok = false;       ///< a seam chain from the reference room to this one exists in memory
    Eigen::Affine2d T_ref_k = Eigen::Affine2d::Identity();   ///< room k in the reference room, from k's OWN chain
    Eigen::Matrix3d cov_T   = Eigen::Matrix3d::Zero();
    std::vector<Eigen::Vector2d> polygon_in_ref;
    double area = 0.0;
};

struct MetricQuery
{
    std::string reference_room;
    bool        robot_ok = false;         ///< every passage of the path resolved to a seam
    ChainPose   robot_in_ref;             ///< the robot via the path ACTUALLY taken + odometry since the last seam
    std::vector<KnownRoomMetric> rooms;
};

MetricQuery build_metric_query(DSR::DSRGraph &mem, const std::vector<std::string> &path_passages,
                               const std::string &path_start_room, const ChainPose &robot_in_last_room);

/// One entry per `q.rooms` (same order). `log_new_out` = log of the mass outside every known outline (0 when the
/// robot chain is unavailable: an unavailable expert is omitted, not zeroed).
std::vector<rc::recog::MetricEvidence> metric_evidence(const MetricQuery &q, float &log_new_out);
}   // namespace ltsm
