/*
 * common/media_transport/rt_extrapolate.h  —  efference-copy forward-extrapolation of the room<-robot pose.
 *
 * DSR's InterpolatedRT CLAMPS a query at the newest RT block (it never extrapolates velocity), so a LiDAR
 * scan whose stamp is ahead of room_concept's latest published pose (~90 ms of localization pipeline lag,
 * measured) is registered against a STALE robot pose → the cloud lags/shimmers during motion (worst under
 * rotation) and, for a consumer that clusters, its cluster positions jitter → tracker churn.
 *
 * This helper predicts the pose FORWARD from the newest RT block to the scan stamp using the body-frame
 * velocity room_concept already writes on the robot->room RT edge (rt_translation_velocity=[adv,side,0],
 * rt_rotation_euler_xyz_velocity=[0,0,rot]) — the SAME efference-copy trick the controller overlay and the
 * voxelizer mask deprojection use. Consumer-side only; no producer change.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

namespace rc::media
{

// Extrapolate `base` (the CLAMPED room<-robot pose at `target_stamp_ms`, i.e. the InterpolatedRT result,
// which equals the pose at the newest RT block when the scan outruns the RT) FORWARD to the scan stamp,
// writing the result into `out`. Returns true iff it extrapolated (scan ahead of the newest block AND the
// edge/velocities are present); otherwise leaves out == base and returns false. dt is clamped to max_dt_s.
// RTMat = Eigen::Transform<Scalar,3,Affine> (Mat::RTMat).
template <class RTMat>
inline bool extrapolate_room_T_robot(const std::shared_ptr<DSR::DSRGraph>& G,
                                     const std::string& room_name, const std::string& robot_name,
                                     std::uint64_t target_stamp_ms, float max_dt_s,
                                     const RTMat& base, RTMat& out)
{
    out = base;
    if (not G or not G->get_rt_api() or target_stamp_ms == 0)
        return false;
    const auto robot_n = G->get_node(robot_name);
    const auto room_n  = G->get_node(room_name);
    if (not robot_n.has_value() or not room_n.has_value())
        return false;
    auto edge = G->get_rt_api()->get_edge_RT(robot_n.value(), room_n->id());
    if (not edge.has_value())
        return false;
    const auto ts = G->get_attrib_by_name<rt_timestamps_att>(edge.value());
    const auto tv = G->get_attrib_by_name<rt_translation_velocity_att>(edge.value());
    const auto rv = G->get_attrib_by_name<rt_rotation_euler_xyz_velocity_att>(edge.value());
    if (not ts.has_value() or not tv.has_value() or tv->get().size() < 2
        or not rv.has_value() or rv->get().size() < 3)
        return false;

    std::uint64_t newest = 0;
    for (const auto t : ts->get())
        newest = std::max<std::uint64_t>(newest, static_cast<std::uint64_t>(t));
    if (newest == 0 or target_stamp_ms <= newest)
        return false;

    double dt = static_cast<double>(target_stamp_ms - newest) * 1e-3;
    dt = std::min(dt, static_cast<double>(max_dt_s));

    const double adv = tv->get()[0], side = tv->get()[1], rot = rv->get()[2];
    const Eigen::Matrix3d R = base.linear().template cast<double>();
    const double th  = std::atan2(R(1, 0), R(0, 0));
    const double dth = rot * dt;
    const double thm = th + 0.5 * dth;   // midpoint integration
    const double dx = (adv * std::cos(thm) - side * std::sin(thm)) * dt;
    const double dy = (adv * std::sin(thm) + side * std::cos(thm)) * dt;

    using Scalar = typename RTMat::Scalar;
    out = base;
    out.translation().x() += static_cast<Scalar>(dx);
    out.translation().y() += static_cast<Scalar>(dy);
    const Eigen::Matrix3d Rn = Eigen::AngleAxisd(dth, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R;
    out.linear() = Rn.template cast<Scalar>();
    return true;
}

}  // namespace rc::media
