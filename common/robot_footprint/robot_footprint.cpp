/*
 * robot_footprint.cpp — see robot_footprint.h
 */

#include "robot_footprint.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace rc
{

RobotFootprint RobotFootprint::shadow()
{
    // XY convex hull of the BODY ∪ the FOUR WHEELS, simplified 74 → 18 vertices (+0.3% area).
    // Robot frame: x right, y forward. CCW. Provenance and the wheel-axis caveat are in the header.
    RobotFootprint f;
    f.poly_ = {
        {-0.2716f, -0.1000f}, {-0.2536f, -0.1436f}, {-0.1727f, -0.2257f}, {-0.0390f, -0.2299f},
        {+0.0324f, -0.2300f}, {+0.1684f, -0.2259f}, {+0.1770f, -0.2226f}, {+0.2536f, -0.1436f},
        {+0.2716f, -0.1000f}, {+0.2716f, +0.1600f}, {+0.2536f, +0.2036f}, {+0.2154f, +0.2214f},
        {+0.1278f, +0.2298f}, {-0.1295f, +0.2300f}, {-0.2154f, +0.2214f}, {-0.2536f, +0.2036f},
        {-0.2658f, +0.1860f}, {-0.2716f, +0.1600f},
    };
    return f;
}

float RobotFootprint::inscribed_radius() const
{
    // Distance from the origin to the nearest EDGE LINE. Valid because the polygon is convex and contains the
    // origin; for a concave shape this would need the nearest point on the boundary instead.
    float best = std::numeric_limits<float>::max();
    const std::size_t n = poly_.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& a = poly_[i];
        const auto& b = poly_[(i + 1) % n];
        const Eigen::Vector2f e = b - a;
        const float len = e.norm();
        if (len < 1e-9f) continue;
        best = std::min(best, std::abs(e.x() * (0.f - a.y()) - e.y() * (0.f - a.x())) / len);
    }
    return (best == std::numeric_limits<float>::max() ? 0.f : best) + safety_margin_m_;
}

float RobotFootprint::circumscribed_radius() const
{
    float best = 0.f;
    for (const auto& p : poly_) best = std::max(best, p.norm());
    return best + safety_margin_m_;
}

bool RobotFootprint::contains(const Eigen::Vector2f& p_robot) const
{
    // Convex containment: the point must be on the interior side of every edge. The safety margin pushes each
    // edge outward along its own normal, which for a convex polygon is exactly a uniform offset.
    const std::size_t n = poly_.size();
    if (n < 3) return false;
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& a = poly_[i];
        const auto& b = poly_[(i + 1) % n];
        const Eigen::Vector2f e = b - a;
        const float len = e.norm();
        if (len < 1e-9f) continue;
        // CCW winding ⇒ interior is to the LEFT of a→b ⇒ cross > 0 inside.
        const float cross = e.x() * (p_robot.y() - a.y()) - e.y() * (p_robot.x() - a.x());
        if (cross / len < -safety_margin_m_) return false;
    }
    return true;
}

std::vector<Eigen::Vector2f> RobotFootprint::at_pose(const Eigen::Vector2f& pos, float theta) const
{
    // Offset the polygon outward by the safety margin FIRST (in the robot frame, along each vertex's own
    // direction from the origin — exact for a convex hull about an interior origin), then place it.
    const float c = std::cos(theta), s = std::sin(theta);
    std::vector<Eigen::Vector2f> out;
    out.reserve(poly_.size());
    for (const auto& p : poly_)
    {
        Eigen::Vector2f q = p;
        if (safety_margin_m_ > 0.f)
        {
            const float len = p.norm();
            if (len > 1e-6f) q = p + (safety_margin_m_ / len) * p;
        }
        out.emplace_back(pos.x() + c * q.x() - s * q.y(), pos.y() + s * q.x() + c * q.y());
    }
    return out;
}

float RobotFootprint::support_radius(float theta, const Eigen::Vector2f& dir_world) const
{
    const float n = dir_world.norm();
    if (n < 1e-6f or poly_.empty()) return circumscribed_radius();   // no direction ⇒ worst case
    const Eigen::Vector2f d = dir_world / n;
    const float c = std::cos(theta), s = std::sin(theta);
    float best = 0.f;
    for (const auto& p : poly_)
    {
        const Eigen::Vector2f w(c * p.x() - s * p.y(), s * p.x() + c * p.y());   // vertex in the world frame
        best = std::max(best, w.dot(d));
    }
    // The margin is a uniform outward offset, so it adds to the support in every direction.
    return best + safety_margin_m_;
}

std::vector<Eigen::Vector2i> RobotFootprint::cell_offsets(float cell_size, float theta) const
{
    std::vector<Eigen::Vector2i> out;
    const float cs = std::max(1e-3f, cell_size);
    const float r = circumscribed_radius();
    const int span = static_cast<int>(std::ceil(r / cs)) + 1;
    const float c = std::cos(-theta), s = std::sin(-theta);   // world offset → robot frame
    for (int iy = -span; iy <= span; ++iy)
        for (int ix = -span; ix <= span; ++ix)
        {
            // CONSERVATIVE: include the cell if ANY of its corners or its centre lies in the footprint, so a
            // cell the robot merely clips still counts as overlapping. Under-reporting here would be a
            // collision; over-reporting costs at most one cell of pessimism at the boundary.
            bool hit = false;
            for (const auto& [dx, dy] : {std::pair{0.5f, 0.5f}, std::pair{0.0f, 0.0f}, std::pair{1.0f, 0.0f},
                                         std::pair{0.0f, 1.0f}, std::pair{1.0f, 1.0f}})
            {
                const float wx = (static_cast<float>(ix) + dx) * cs;
                const float wy = (static_cast<float>(iy) + dy) * cs;
                if (contains({c * wx - s * wy, s * wx + c * wy})) { hit = true; break; }
            }
            if (hit) out.emplace_back(ix, iy);
        }
    return out;
}

bool RobotFootprint::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    auto f = shadow();
    const float ins = f.inscribed_radius(), cir = f.circumscribed_radius();
    std::printf("  shadow footprint: %zu verts  inscribed %.3f m  circumscribed %.3f m\n",
                f.polygon().size(), ins, cir);
    check(f.polygon().size() >= 8, "footprint must have a real outline");
    check(ins > 0.21f and ins < 0.25f, "inscribed radius must match the measured mesh+wheels (~0.230 m)");
    check(cir > 0.31f and cir < 0.34f, "circumscribed radius must match the measured mesh+wheels (~0.325 m)");
    check(ins < cir, "inscribed must be smaller than circumscribed");

    // The origin is the rotation centre and must be inside; a point beyond the circumscribed radius cannot be.
    check(f.contains({0.f, 0.f}), "the rotation centre must be inside the footprint");
    check(not f.contains({cir + 0.05f, 0.f}), "a point beyond the circumscribed radius must be outside");
    // Just inside the inscribed disc is inside at EVERY bearing — that is what inscribed means.
    for (int i = 0; i < 16; ++i)
    {
        const float a = 6.2831853f * i / 16.f;
        check(f.contains({0.98f * ins * std::cos(a), 0.98f * ins * std::sin(a)}),
              "every point inside the inscribed disc must be inside the footprint");
    }

    // The safety margin must grow both radii by exactly itself — it is a uniform offset, not a scale.
    f.set_safety_margin(0.10f);
    std::printf("  with 0.10 m margin: inscribed %.3f  circumscribed %.3f\n",
                f.inscribed_radius(), f.circumscribed_radius());
    check(std::abs(f.inscribed_radius() - (ins + 0.10f)) < 1e-4f, "margin must add exactly to the inscribed radius");
    check(std::abs(f.circumscribed_radius() - (cir + 0.10f)) < 1e-4f, "margin must add exactly to circumscribed");
    f.set_safety_margin(0.f);

    // ROTATION INVARIANCE. A footprint is a rigid body: rotating it cannot change how much area it covers.
    // This is the property an inflated-disc model throws away, and the reason a 0.47 m robot was being denied
    // gaps it fits through — so it is worth asserting rather than assuming.
    const auto n0 = f.cell_offsets(0.05f, 0.f).size();
    const auto n45 = f.cell_offsets(0.05f, 0.7853982f).size();
    const auto n90 = f.cell_offsets(0.05f, 1.5707963f).size();
    std::printf("  cell coverage at 0.05 m: %zu (0 deg) %zu (45 deg) %zu (90 deg)\n", n0, n45, n90);
    check(n0 > 50, "a 0.48 m footprint must cover a plausible number of 5 cm cells");
    const auto lo = std::min({n0, n45, n90}), hi = std::max({n0, n45, n90});
    check(static_cast<float>(hi - lo) / static_cast<float>(hi) < 0.20f,
          "cell coverage must be roughly rotation-invariant (rasterisation jitter only)");

    // at_pose must be a rigid transform: side lengths preserved under translation + rotation.
    const auto a = f.at_pose({0.f, 0.f}, 0.f), b = f.at_pose({3.f, -2.f}, 1.1f);
    check(a.size() == b.size(), "at_pose must preserve the vertex count");
    bool edges_ok = true;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float la = (a[(i + 1) % a.size()] - a[i]).norm();
        const float lb = (b[(i + 1) % b.size()] - b[i]).norm();
        if (std::abs(la - lb) > 1e-4f) edges_ok = false;
    }
    check(edges_ok, "at_pose must be RIGID (edge lengths unchanged by pose)");

    // SUPPORT RADIUS: direction-dependent body extent, and the reason the MPPI can stop using a disc.
    // It must lie between the inscribed and circumscribed radii in every direction, equal the circumscribed
    // radius toward the furthest vertex, and be invariant to rotating body and direction together.
    {
        bool bounded = true, consistent = true;
        for (int i = 0; i < 24; ++i)
        {
            const float a = 6.2831853f * i / 24.f;
            const Eigen::Vector2f d(std::cos(a), std::sin(a));
            const float r = f.support_radius(0.f, d);
            if (r < ins - 1e-4f or r > cir + 1e-4f) bounded = false;
            // rotate body and query direction together by +0.7 rad — a rigid body cannot change extent
            const Eigen::Vector2f d2(std::cos(a + 0.7f), std::sin(a + 0.7f));
            if (std::abs(f.support_radius(0.7f, d2) - r) > 1e-3f) consistent = false;
        }
        check(bounded, "support radius must lie between the inscribed and circumscribed radii");
        check(consistent, "support radius must be invariant to rotating body and direction together");
        std::printf("  support radius: +x %.3f  +y %.3f  diagonal %.3f  (inscribed %.3f, circumscribed %.3f)\n",
                    f.support_radius(0.f, {1.f, 0.f}), f.support_radius(0.f, {0.f, 1.f}),
                    f.support_radius(0.f, {0.707f, 0.707f}), ins, cir);
    }

    // The point of the whole exercise: the robot fits gaps far narrower than the stacked margins allowed.
    std::printf("  passable gap: %.3f m (2x inscribed). The six stacked margins demanded ~0.95 m.\n", 2.f * ins);

    std::printf("RobotFootprint::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
