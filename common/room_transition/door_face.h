#pragma once
/*
 * door_face.h — the doorway as the two rooms' ONE attaching point (spec 2026-10-31-door-attachment-and-layout, D1/D2).
 *
 * A door node's frame takes its yaw from its WALL (x = the wall tangent), so its +y points into its room only when the
 * room's outline is counter-clockwise. A FACE is that frame turned by π if needed so that +y points INTO its own room,
 * decided locally: the point half the door's width along +y must lie inside the room's own outline. The two faces of one
 * doorway are then a half-turn apart, always, and one room is placed in the other by the exact composition
 *
 *     T(a ← b) = F_a · Rπ · F_b⁻¹
 *
 * Pure Eigen, header-only. Used by room_concept (recall placement) and ltsm (the layout optimizer's seams).
 */
#include <cmath>
#include <vector>

#include <Eigen/Geometry>

namespace rc::door_face
{
using Poly = std::vector<Eigen::Vector2f>;

/// Even-odd point-in-polygon (the room's outline; any winding).
[[nodiscard]] inline bool inside(const Poly& P, const Eigen::Vector2f& c)
{
    bool in = false;
    for (std::size_t i = 0, j = P.size() - 1; i < P.size(); j = i++)
        if (((P[i].y() > c.y()) != (P[j].y() > c.y()))
            and (c.x() < (P[j].x() - P[i].x()) * (c.y() - P[i].y()) / (P[j].y() - P[i].y()) + P[i].x()))
            in = not in;
    return in;
}

[[nodiscard]] inline Eigen::Affine2f half_turn()
{
    Eigen::Affine2f R = Eigen::Affine2f::Identity();
    R.linear() = Eigen::Rotation2Df(static_cast<float>(M_PI)).toRotationMatrix();
    return R;
}

/// The FACE of a door whose node pose in its room is `door` (room <- door): +y into the room of `outline`.
/// `half_width` = half the opening. An outline with fewer than 3 vertices leaves the pose as given.
[[nodiscard]] inline Eigen::Affine2f face(const Eigen::Affine2f& door, const Poly& outline, float half_width)
{
    if (outline.size() < 3 or inside(outline, door * Eigen::Vector2f(0.f, half_width))) return door;
    return door * half_turn();
}

/// Room b's frame in room a's, through one doorway: F_a, F_b its two faces (each in its own room's frame).
[[nodiscard]] inline Eigen::Affine2f attach(const Eigen::Affine2f& F_a, const Eigen::Affine2f& F_b)
{
    return F_a * half_turn() * F_b.inverse();
}
}  // namespace rc::door_face
