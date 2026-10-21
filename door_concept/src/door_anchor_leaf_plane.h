// door_anchor_leaf_plane.h — Task 5b item 2: the passage anchor's leaf hypotheses on the LEAF PLANE door_concept believes in.
// Header-only, Eigen only. tests/test_door_anchor_leaf_plane.cpp.
//
// anchor_leaf_hypotheses (common/interaction_anchor/anchor_leaf.h) hangs every hinge × swing hypothesis from the wall's
// mid-plane (face_y = ±half the thickness). door_concept's model puts the leaf at its own depth d off the wall line
// (DoorBeliefState::d, Task 5b) — 0.10-0.15 m on DOOR_0 — so the controller, which treats the anchor's leaf as a known
// obstacle, was told a different obstacle than the one door_concept fits, renders and removes doors by. One model: the
// hypotheses are translated rigidly onto the leaf plane, expressed in the anchor's frame.
// ★The anchor frame registers the WALL (door_anchor_update.h measures the opening in the wall it sees), so the offset is the
// leaf relative to that wall: −(d + reg_depth) along across_u (d > 0 out of the room = −across_u; the low LiDAR's wall
// registration reg_depth > 0 into the room = +across_u), and the anchor's y axis is sign·across_u (the robot on −y).
// No registration yet ⇒ relative to the polygon line (reg_depth 0); DoorFitState::wall_reg holds the last one (M-6).
// ★ALL FOUR hinge × swing hypotheses move by the SAME offset, the opposite-swing ones included (review M-6): what the swing
// changes is which FACE of the leaf the hinge pin sits on — anchor_leaf_hypotheses negates face_y = ±half the thickness
// about the leaf's own mid-plane for it — and which side the leaf turns toward; it does not change WHERE the leaf hangs in
// the wall. The plane is the leaf's (DoorBeliefState::d), one for every hypothesis, so the shift is rigid and common.
#pragma once

#include <vector>

#include <Eigen/Dense>

#include "../../common/interaction_anchor/anchor_leaf.h"

namespace rc::door
{

[[nodiscard]] inline float anchor_leaf_plane_y(float d, float reg_depth, bool reg_ok, float axis_sign)
{
    const float rel = d + (reg_ok and std::isfinite(reg_depth) ? reg_depth : 0.0f);   // leaf beyond the WALL, out of the room
    return -(axis_sign < 0.0f ? -1.0f : 1.0f) * rel;
}

inline void place_leaves_on_plane(std::vector<rc::interaction::AnchorLeaf>& leaves, float plane_y)
{
    if (plane_y == 0.0f) return;   // the wall's mid-plane: exactly the published hypotheses, untouched
    for (auto& l : leaves)
    {
        l.hinge.y() += plane_y;
        l.tip.y()   += plane_y;
    }
}

}  // namespace rc::door
