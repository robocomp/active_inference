/*
 * bed_self_body.h — the ROBOT is an object too: a mask point that lies on the robot's own body is evidence
 * of the robot, not of a bed. Pure Eigen, header-only, unit-tested (tests/self_body_test.cpp).
 *
 * WHY (live, 2026-11-14, objects_live3): bed_5 was born 0.19 m in front of the robot's centre, on no bed. The
 * ZED hangs at body (0, -0.075, +0.945) looking straight ahead with a 78° vertical FoV, so its bottom image
 * rows (-39°) graze the front of the robot's own top plate (z ≈ 0.79, front edge y = +0.23): the visible
 * strip is y ∈ [0.156, 0.230] across the 0.44 m body width. That is exactly the cloud bed_5 was born from —
 * centroid z 0.78, ext_major 0.405 x ext_minor 0.055, ROBOT-FIXED (0.18–0.20 m dead ahead while the robot
 * turned and stopped) — labelled "bed" 0.34 / "bathtub" 0.37 by YOLOE. Nothing in the birth evidence asked
 * whether the points could belong to the world at all: range weight is 1 below 2.5 m, the residual-fusion
 * term is one-sided (m = 0 ⇒ ×1), and the room envelope / foreign-claim supports know nothing of the robot.
 *
 * THE MODEL. Each support point is generated either by the world or by the robot's body. The body is a known
 * prism: its convex ground silhouette (rc::RobotFootprint, robot frame x-right / y-forward) extruded over its
 * height. A floor-standing object shorter than the robot (every bed) can never put a surface point inside that
 * prism, so P(point is world | p) is the probability that p is OUTSIDE the silhouette given the point's own
 * position noise σ:  P_world = Φ(s/σ), s = signed distance to the silhouette (> 0 outside). The mask's world
 * fraction  f = mean_i P_world(p_i)  is how much of this detection is evidence for an external object, and
 * multiplies the per-frame birth evidence exactly like the room-envelope support and p_unclaimed already do.
 * Continuous, no cut-off: a real bed mask that bleeds a few pixels onto the plate keeps f ≈ 1; a mask that is
 * the plate accrues ~a sixth of an observation per frame (σ = 3 cm) and cannot mature in a normal stop.
 *
 * Same physics as residual_concept's SELF-BODY sensor term (residual_occupancy_grid.h: hit weight Φ(s/σ)),
 * applied to the mask channel at CREATE. The points must be in the ROBOT frame; the caller carries them there
 * by the static body←camera mount, so no localisation error enters σ.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Dense>

namespace rc::self_body
{

// Signed distance (m) from p to the boundary of a CONVEX, counter-clockwise polygon: > 0 outside, < 0 inside.
// An empty / degenerate polygon has no inside ⇒ +inf (every point is world).
inline float signed_distance(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return std::numeric_limits<float>::infinity();
    float d2 = std::numeric_limits<float>::max();
    bool inside = true;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
        const Eigen::Vector2f ap = p - a;
        if (ab.x() * ap.y() - ab.y() * ap.x() < 0.0f)   // right of a CCW edge ⇒ outside
            inside = false;
        const float len2 = ab.squaredNorm();
        const float t = len2 > 1e-12f ? std::clamp(ap.dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        d2 = std::min(d2, (ap - t * ab).squaredNorm());
    }
    const float d = std::sqrt(d2);
    return inside ? -d : d;
}

// P(this point was produced by the world, not by the robot's body) = Φ(s/σ).
inline float p_world(float signed_dist_m, float sigma_m)
{
    if (not std::isfinite(signed_dist_m))
        return 1.0f;
    const float sg = std::max(sigma_m, 1e-4f);
    return 0.5f * std::erfc(-signed_dist_m / (sg * std::sqrt(2.0f)));
}

// The detection's WORLD FRACTION ∈ [0, 1]: mean P_world over its support points, given in the ROBOT frame
// (x right, y forward, origin at the rotation centre — the frame of rc::RobotFootprint). No points ⇒ 1 (the
// term asserts nothing it has not measured).
inline float world_fraction(const std::vector<Eigen::Vector2f>& poly, std::span<const Eigen::Vector3f> pts_robot,
                            float sigma_m)
{
    if (pts_robot.empty() or poly.size() < 3)
        return 1.0f;
    // Broad phase: beyond the circumscribed radius + 6σ a point is world to float precision.
    float r_circ = 0.0f;
    for (const auto& v : poly) r_circ = std::max(r_circ, v.norm());
    const float far2 = (r_circ + 6.0f * sigma_m) * (r_circ + 6.0f * sigma_m);
    double acc = 0.0;
    for (const auto& p : pts_robot)
    {
        const Eigen::Vector2f q{p.x(), p.y()};
        acc += q.squaredNorm() > far2 ? 1.0 : static_cast<double>(p_world(signed_distance(poly, q), sigma_m));
    }
    return static_cast<float>(acc / static_cast<double>(pts_robot.size()));
}

}  // namespace rc::self_body
