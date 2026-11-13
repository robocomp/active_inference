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
 * The crossed door has NO passage (that is when this runs), so the path chain ends with a FRESH seam from that door's
 * two faces, passed in; the OWN chains use only passages already in memory, so they never assume the answer.
 *
 * Covariances: a seam carries the covariance of its two door faces (the RT edge of each, when stated, rotated from the
 * wall's frame into the room's), else face_prior.h's (the population of stated faces: 0.145 m along the wall, 0.020 m
 * across, 1 deg). cortex restores rt_covariance on JSON load since 2026-11-12. Not a gate: a covariance that enters the
 * Gaussian whose mass inside the outline is the evidence.
 *
 * WITH THE MAP LAYOUT (owner 2026-11-12; `layout` non-null and solved for L this run): the reference room becomes L,
 * the robot in L is the fresh seam + odometry, and every room k is T(L <- k) from the layout with the joint covariance
 * of L and k. The layout already uses every passage, loops included, so the "two chains" check above is subsumed by
 * the posterior; rooms the layout does not hold fall back to the seam BFS.
 *
 * log_new = sum_k log(1 - P_inside_k), each P_inside_k under its own S = robot cov + cov_T(k). Treating the
 * room placements as independent is an approximation (they share seams); it is exact for disjoint, independently
 * placed outlines and conservative otherwise.
 *
 * Mass inside an outline = deterministic Gauss-Hermite quadrature (9x9) over the (x, y) Gaussian. The mass the rule cannot
 * resolve (an outline further than ~4.5 sigma) is added as exp(-(d^2 + r_max^2)/2), d the Mahalanobis distance to the
 * outline's edge: smooth and monotone in d, never -inf, no switch (see region_mass).
 */
#include "room_eviction.h"
#include "topology_layout.h"
#include "../../common/room_recognition/room_posterior.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cstdint>
#include <optional>
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
    bool        from_layout = false;      ///< rooms placed by the MAP layout, referenced to L (else seam chains from the start)
    bool        robot_ok = false;         ///< every passage of the path resolved to a seam
    ChainPose   robot_in_ref;             ///< the robot via the path ACTUALLY taken + odometry since the last seam
    std::vector<KnownRoomMetric> rooms;
};

/// `path_passages`: the passages the robot crossed, oldest first, ending at the last KNOWN room L (all in memory).
/// `seam_last_to_provisional`: the FRESH seam T(L <- provisional) with its covariance, measured from the crossed
///   door's two faces (its face in L, a memory node, and its live entry-mirror face in the provisional room). The crossed
///   door has no passage yet (that is the case this serves), so nothing in memory supplies it. nullopt => unmeasurable
///   => the metric expert is unavailable (all `available=false`, log_new 0). Pass identity/zero when the robot is still in L.
/// `robot_in_provisional`: the live robot pose in the PROVISIONAL room's frame (+ odometry covariance).
MetricQuery build_metric_query(DSR::DSRGraph &mem, const std::vector<std::string> &path_passages,
                               const std::string &path_start_room, const std::optional<ChainPose> &seam_last_to_provisional,
                               const ChainPose &robot_in_provisional, const topo::LayoutCache *layout = nullptr);

/// One entry per `q.rooms` (same order). `log_new_out` = log of the mass outside every known outline (0 when the
/// robot chain is unavailable: an unavailable expert is omitted, not zeroed).
std::vector<rc::recog::MetricEvidence> metric_evidence(const MetricQuery &q, float &log_new_out);
}   // namespace ltsm
