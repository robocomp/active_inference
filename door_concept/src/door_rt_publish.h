/*
 * door_rt_publish.h — when door_concept rewrites a door's RT edge (DoorSceneGraph::write_rt_pose). Pure; tested by
 * tests/test_door_rt_publish.cpp.
 *
 * ★Task D-C (R2 review M-6): THE DEAD-BAND MEASURES HOW FAR THE APERTURE MOVED, NOT HOW FAR ITS CENTRE MOVED. The old rule
 * compared only the centre's translation with 5 cm, so a pure yaw re-fit about a fixed centre was never published — and the
 * restored DOOR_0, first initialised in the wrong wall (yaw 1.547 for −3.20), could be corrected by the fit and still leave
 * room_concept reading the 90°-off aperture. The aperture is a segment; its consumers read its two jambs. So the move is
 * the larger displacement of the two jambs, c ± (w/2)·(cos yaw, sin yaw), and it is compared with the SAME 5 cm the
 * translation used: one constant, in metres of the door's own geometry — a yaw change is published exactly when it moves
 * a jamb as far as a translation would have had to.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include <Eigen/Dense>

namespace rc::door
{

/// The rewrite dead-band (m): below this jamb displacement the RT edge is left as it is (gradient oscillation churn).
inline constexpr float kRtMinJambMoveM = 0.05f;

/// The larger displacement of the aperture's two jambs between the written pose and the new one.
[[nodiscard]] inline float aperture_jamb_move(float old_cx, float old_cy, float old_yaw,
                                              float cx, float cy, float yaw, float half_w)
{
    if (not std::isfinite(old_cx) or not std::isfinite(old_cy) or not std::isfinite(old_yaw))
        return std::numeric_limits<float>::infinity();   // never written
    const Eigen::Vector2f c0(old_cx, old_cy), c1(cx, cy);
    const Eigen::Vector2f u0(std::cos(old_yaw), std::sin(old_yaw)), u1(std::cos(yaw), std::sin(yaw));
    const float h = std::max(0.0f, half_w);
    const float a = ((c1 + h * u1) - (c0 + h * u0)).norm();
    const float b = ((c1 - h * u1) - (c0 - h * u0)).norm();
    return std::max(a, b);
}

[[nodiscard]] inline bool rt_write_due(float old_cx, float old_cy, float old_yaw, float cx, float cy, float yaw, float half_w)
{
    return aperture_jamb_move(old_cx, old_cy, old_yaw, cx, cy, yaw, half_w) >= kRtMinJambMoveM;
}

/// A door's room-frame aperture pose as the GRAPH currently publishes it.
struct PublishedPose { float cx, cy, yaw; };

/// The inverse of door_in_wall: the door's local RT (lx, ly along/across the wall, lyaw) under a wall whose RT puts its
/// midpoint at `mid` with tangent `wall_yaw` (room frame; floor->room is the identity) ⇒ the room-frame pose.
[[nodiscard]] inline PublishedPose door_room_pose(const Eigen::Vector2f& mid, float wall_yaw, float lx, float ly, float lyaw)
{
    const float c = std::cos(wall_yaw), s = std::sin(wall_yaw);
    return {mid.x() + c * lx - s * ly, mid.y() + s * lx + c * ly,
            std::remainder(wall_yaw + lyaw, 2.0f * static_cast<float>(M_PI))};
}

/// ★ROOM-TRANSITION RULE 1 (2026-10-29): room_concept moves a wall IN PLACE when the layout changes, and the door's RT
/// (its offset along the wall) goes with it — so the pose the graph publishes can leave the door's belief without the
/// belief moving. The dead-band therefore compares the belief with what the graph PUBLISHES when that can be read, and
/// with what this agent last wrote only when it cannot (no edge yet). A slid door is re-seated on the next cycle.
[[nodiscard]] inline bool rt_write_due_published(const std::optional<PublishedPose>& published,
                                                 float last_cx, float last_cy, float last_yaw,
                                                 float cx, float cy, float yaw, float half_w)
{
    if (published.has_value())
        return rt_write_due(published->cx, published->cy, published->yaw, cx, cy, yaw, half_w);
    return rt_write_due(last_cx, last_cy, last_yaw, cx, cy, yaw, half_w);
}

/// A planar rigid transform: p' = R(yaw)·p + t. Room-transition case 2: ltsm's proposal carries T(P<-K) on its RT edge; the
/// adopted room's frame changes P -> K, so what door_concept holds in P's frame moves by T(K<-P) = inverse.
struct Rigid2 { float tx = 0.0f, ty = 0.0f, yaw = 0.0f; };
[[nodiscard]] inline Rigid2 rigid_inverse(const Rigid2& T)
{
    const float c = std::cos(T.yaw), s = std::sin(T.yaw);
    return {-(c * T.tx + s * T.ty), -(-s * T.tx + c * T.ty), -T.yaw};
}
[[nodiscard]] inline Eigen::Vector2f rigid_apply(const Rigid2& T, const Eigen::Vector2f& p)
{
    const float c = std::cos(T.yaw), s = std::sin(T.yaw);
    return {c * p.x() - s * p.y() + T.tx, s * p.x() + c * p.y() + T.ty};
}
/// A pose (position + heading) seen in frame B, given the frame change `B_from_A`: the door's pose in the new frame.
[[nodiscard]] inline PublishedPose rigid_apply(const Rigid2& B_from_A, const PublishedPose& p)
{
    const auto q = rigid_apply(B_from_A, Eigen::Vector2f(p.cx, p.cy));
    return {q.x(), q.y(), std::remainder(p.yaw + B_from_A.yaw, 2.0f * static_cast<float>(M_PI))};
}

}  // namespace rc::door
