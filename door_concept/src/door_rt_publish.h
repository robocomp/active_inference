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

}  // namespace rc::door
