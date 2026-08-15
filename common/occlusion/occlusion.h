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

// HOW MUCH of the line of sight from camera O to a target at range rc (unit bearing dc_unit) does a CLOSER
// object take? An occluder at range rs subtending half-extent `occ_half` covers a bearing cone of half-angle
// atan(occ_half / rs). Deep inside that cone ⇒ 1, at its edge ⇒ 0, outside or not meaningfully closer
// (rs >= rc − margin) ⇒ 0. All points in the room frame (3-D), ranges in metres.
//
// ★A STRENGTH, NOT A VERDICT — and chair_concept is why. It had this function written out by hand, returning
// [0,1] instead of a bool, with the reason in its own comment: "occlusion is a reason to trust absence LESS,
// not a licence to ignore it forever" (CONCEPT_AGENT_INVARIANTS mistake II — a gate must fail to HOLD, but it
// must not hold unconditionally and for ever). That is also this codebase's standing modelling rule: encode
// the effect as a continuous weight on the evidence, not as a hard switch that skips the cycle. The shared
// module had only the boolean, so the four agents on it can only skip. Linear in the bearing offset: the
// honest statement is "how much of my sightline does this thing take", and nothing justifies a sharper shape.
inline float cone_occlusion(const Eigen::Vector3f& O, const Eigen::Vector3f& dc_unit, float rc,
                            const Eigen::Vector3f& occ_center, float occ_half, float occ_range, float margin)
{
    if (not std::isfinite(occ_range) or occ_range >= rc - margin)   // not meaningfully closer → cannot occlude
        return 0.0f;
    Eigen::Vector3f ds = occ_center - O;
    const float n = ds.norm();
    if (n < 1e-3f)
        return 0.0f;
    ds /= n;
    const float ang          = std::acos(std::clamp(dc_unit.dot(ds), -1.0f, 1.0f));   // bearing offset target↔occluder
    const float occ_half_ang = std::atan2(std::max(0.05f, occ_half), std::max(0.2f, occ_range));
    return std::clamp(1.0f - ang / std::max(1e-4f, occ_half_ang), 0.0f, 1.0f);
}

// The boolean form the existing callers use: "is the bearing inside the cone at all". Expressed in terms of
// the strength so the two can never disagree — an agent still on the verdict gets exactly what it had.
inline bool cone_blocks(const Eigen::Vector3f& O, const Eigen::Vector3f& dc_unit, float rc,
                        const Eigen::Vector3f& occ_center, float occ_half, float occ_range, float margin)
{
    return cone_occlusion(O, dc_unit, rc, occ_center, occ_half, occ_range, margin) > 0.0f;
}

// The half-extent an axis-aligned footprint presents ACROSS the camera ray — the number cone_occlusion wants
// for a detected mask used as an occluder.
//
// ★NOT THE BBOX HALF-DIAGONAL, and chair_concept measured what that costs. The diagonal models a 2.4 m table
// as a sphere of ~1.3 m radius, whose bearing cone swallows every chair AT that table — so a phantom chair in
// a dining set was permanently "occluded" by the furniture it was born among, held its absence forever and
// froze at L = −1.95 for 300+ cycles while the robot stared straight at it and the detector reported 0.53.
// What blocks a sightline is the width the occluder presents ACROSS it, which is what this projects.
inline float across_ray_half_extent(const Eigen::Vector3f& O, const Eigen::Vector3f& occ_center,
                                    const Eigen::Vector3f& bbox_min, const Eigen::Vector3f& bbox_max)
{
    const Eigen::Vector2f ext = (bbox_max - bbox_min).head<2>();
    const Eigen::Vector2f d   = (occ_center - O).head<2>();
    if (d.norm() < 1e-6f)
        return 0.5f * ext.norm();          // degenerate: nothing to project onto
    const Eigen::Vector2f ray = d.normalized();
    const Eigen::Vector2f perp(-ray.y(), ray.x());
    return 0.5f * (std::abs(ext.x() * perp.x()) + std::abs(ext.y() * perp.y()));
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
