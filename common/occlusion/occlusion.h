/*
 * common/occlusion/occlusion.h  —  shared line-of-sight occlusion primitives (header-only, Eigen only).
 *
 * Every concept agent that runs an existence / vacate belief (door, chair, table, cabinet, refrigerator …)
 * needs the same question: "is the sensor's view of THIS object blocked, so absence of a detection is
 * EXPECTED rather than evidence the object is gone?" Two independent occluder classes are handled:
 *
 *   1. cone_blocks() — a CLOSER solid object (another instance, a detected mask: person, table, …) whose
 *      angular extent, seen from the camera, covers the target's bearing.
 *   2. walls_block()  — a room WALL segment crossing the 2D camera→target sightline (non-convex room, an
 *      object in another room, the robot around a corner). The target's OWN wall is skipped.
 *
 * Pure geometry: no DSR, no agent types. Each agent supplies the camera pose, the target, and its own list
 * of occluders / the room polygon. Extracted from door_concept::DoorFitter::los_occluded so door / chair /
 * table / cabinet / refrigerator share ONE implementation instead of copy-pasting the bearing-cone math.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

namespace rc::occlusion
{

// Does a CLOSER object block the line of sight from camera O to a target at range rc along unit bearing
// dc_unit? An occluder at range rs subtending half-extent `occ_half` covers a bearing cone of half-angle
// atan(occ_half / rs); it occludes when it is meaningfully closer (rs < rc − margin) AND the door's bearing
// falls inside that cone. All points in the room frame (3-D), ranges in metres.
inline bool cone_blocks(const Eigen::Vector3f& O, const Eigen::Vector3f& dc_unit, float rc,
                        const Eigen::Vector3f& occ_center, float occ_half, float occ_range, float margin)
{
    if (not std::isfinite(occ_range) or occ_range >= rc - margin)   // not meaningfully closer → cannot occlude
        return false;
    Eigen::Vector3f ds = occ_center - O;
    const float n = ds.norm();
    if (n < 1e-3f)
        return false;
    ds /= n;
    const float ang      = std::acos(std::clamp(dc_unit.dot(ds), -1.0f, 1.0f));       // bearing offset target↔occluder
    const float occ_half_ang = std::atan2(std::max(0.05f, occ_half), std::max(0.2f, occ_range));
    return ang < occ_half_ang;
}

// Does the 2D sightline cam→target cross any room-polygon wall segment OTHER than the target's OWN wall
// (a segment whose distance to the target is < own_wall_skip_m, i.e. the wall the target is embedded in)?
// True ⇒ a wall blocks the view. polygon = ordered room corners (room frame); < 3 corners ⇒ no walls.
inline bool walls_block(const Eigen::Vector2f& cam, const Eigen::Vector2f& target,
                        const std::vector<Eigen::Vector2f>& polygon, float own_wall_skip_m)
{
    const std::size_t np = polygon.size();
    if (np < 3)
        return false;
    const auto cross = [](const Eigen::Vector2f& p, const Eigen::Vector2f& q) { return p.x() * q.y() - p.y() * q.x(); };
    const Eigen::Vector2f d = target - cam;
    const float skip2 = own_wall_skip_m * own_wall_skip_m;
    for (std::size_t i = 0; i < np; ++i)
    {
        const Eigen::Vector2f a = polygon[i];
        const Eigen::Vector2f b = polygon[(i + 1) % np];
        const Eigen::Vector2f ab = b - a;
        const float len2 = ab.squaredNorm();
        const float tc = (len2 > 1e-8f) ? std::clamp((target - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        if ((target - (a + tc * ab)).squaredNorm() < skip2)
            continue;   // the target's OWN wall — never self-occludes
        const float rxs = cross(d, ab);
        if (std::abs(rxs) < 1e-9f)
            continue;   // parallel
        const float t = cross(a - cam, ab) / rxs;   // fraction along camera→target
        const float u = cross(a - cam, d)  / rxs;   // fraction along the wall segment
        if (t > 1e-3f and t < 1.0f - 1e-3f and u > 0.0f and u < 1.0f)
            return true;   // a wall crosses the sightline before the target
    }
    return false;
}

}  // namespace rc::occlusion
