/*
 * robot_footprint.cpp — see robot_footprint.h
 */

#include "robot_footprint.h"
#include "mesh_hull.h"

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

// ── A BODY BUILT AT RUNTIME ─────────────────────────────────────────────────────────────────────
std::optional<RobotFootprint> RobotFootprint::from_polygon(std::vector<Eigen::Vector2f> poly,
                                                           std::string *why)
{
    const auto fail = [why](const char *reason) -> std::optional<RobotFootprint>
    { if (why) *why = reason; return std::nullopt; };

    if (poly.size() < 3) return fail("fewer than 3 vertices");

    const float area = rc::mesh::polygon_area(poly);
    // ★WINDING IS NOT COSMETIC. contains() tests every edge with the same sign; a clockwise polygon passes
    // that test for exactly the points OUTSIDE it, so a mis-wound body reports the whole world as inside
    // itself and nothing as free. Reject rather than silently reverse: a caller handing this class a CW
    // polygon has a bug worth seeing.
    if (area <= 0.f) return fail("polygon is clockwise or degenerate (area <= 0)");

    // The origin IS the rotation centre by this class's contract, so it must be inside the body. A mesh drawn
    // about its bounding-box corner, or about a tool tip, fails here instead of producing a body that swings
    // through walls when the robot turns.
    for (std::size_t i = 0, n = poly.size(); i < n; ++i)
    {
        const Eigen::Vector2f &a = poly[i], &b = poly[(i + 1) % n];
        if ((b.x() - a.x()) * (0.f - a.y()) - (b.y() - a.y()) * (0.f - a.x()) < 0.f)
            return fail("the origin is OUTSIDE the polygon — the mesh is not drawn about the rotation centre");
    }

    RobotFootprint f;
    f.poly_ = std::move(poly);
    f.source_ = "runtime:polygon";
    return f;
}

std::optional<RobotFootprint> RobotFootprint::from_obj(const std::string &path, float yaw_offset_rad,
                                                       MeshReport &report)
{
    report = MeshReport{};
    report.path = path;
    report.yaw_offset_rad = yaw_offset_rad;

    std::string err;
    // The whole mesh, every height: the 2-D projection of a rigid body does not depend on where the mesh puts
    // its floor, and the two robots' meshes disagree about that. See mesh_hull.h.
    const auto r = rc::mesh::hull_from_obj(path, -std::numeric_limits<float>::max(),
                                           std::numeric_limits<float>::max(), 24, 0.01f, &err);
    if (not r.has_value()) { report.reason = err; return std::nullopt; }

    report.vertices = r->vertices;
    report.vertices_rejected = r->vertices_rejected;
    report.bb_min = r->bb_min;
    report.bb_max = r->bb_max;
    report.hull_raw = r->hull_raw.size();
    report.hull_simplified = r->hull.size();
    report.area_growth_frac = r->area_growth_frac;

    // ★A SINGLE REJECTED LINE CONDEMNS THE FILE. Either it is not the format we think it is, or the parse is
    // broken — and a broken float parse is exactly the locale bug mesh_hull exists to be immune to. Hulling
    // "whatever survived" would produce a plausible, wrong, smaller robot.
    if (r->vertices_rejected > 0)
    {
        report.reason = std::to_string(r->vertices_rejected) + " 'v' lines failed to parse — wrong format, "
                        "or a broken float parse (see mesh_hull.h on LC_NUMERIC)";
        return std::nullopt;
    }

    // MESH -> ROBOT frame. The caller owns this angle; this class cannot tell a body from the same body
    // turned 90 degrees, which is why the number is explicit and logged rather than inferred here.
    const Eigen::Rotation2Df R(yaw_offset_rad);
    std::vector<Eigen::Vector2f> poly;
    poly.reserve(r->hull.size());
    for (const auto &p : r->hull) poly.push_back(R * p);

    std::string why;
    auto f = from_polygon(std::move(poly), &why);
    if (not f.has_value()) { report.reason = why; return std::nullopt; }

    report.area_m2      = rc::mesh::polygon_area(f->poly_);
    report.inscribed    = f->inscribed_radius();
    report.circumscribed = f->circumscribed_radius();
    for (const auto &p : f->poly_)
    {
        report.x_max = std::max(report.x_max, std::abs(p.x()));
        report.y_max = std::max(report.y_max, std::abs(p.y()));
        report.centroid += p;
    }
    report.centroid /= static_cast<float>(f->poly_.size());

    // ── DOES THIS LOOK LIKE A MOBILE BASE AT ALL? ────────────────────────────────────────────────
    // The only thresholds in this file, and they are not tuning: they separate "a robot" from "a mesh in
    // millimetres" or "a parse that produced rubbish". ★A MESH IN THE WRONG UNITS IS REJECTED, NEVER
    // AUTO-SCALED — a silent x0.001 turns a wrong file into a plausible robot.
    if (report.circumscribed < 0.15f or report.circumscribed > 1.00f)
    {
        report.reason = "circumscribed radius " + std::to_string(report.circumscribed)
                      + " m is not a mobile base (expected 0.15..1.00 m; wrong units are NOT auto-scaled)";
        return std::nullopt;
    }
    if (report.inscribed < 0.05f or report.inscribed >= report.circumscribed)
    {
        report.reason = "degenerate hull: inscribed " + std::to_string(report.inscribed)
                      + " m vs circumscribed " + std::to_string(report.circumscribed) + " m";
        return std::nullopt;
    }
    // A hull whose centroid is far from the origin is drawn about something other than the rotation centre.
    // The origin-inside test above is the hard bound; this is the soft one that catches a near-miss.
    if (report.centroid.norm() > 0.10f)
    {
        report.reason = "hull centroid is " + std::to_string(report.centroid.norm())
                      + " m from the origin — the mesh is not drawn about the rotation centre";
        return std::nullopt;
    }

    char buf[512];
    std::snprintf(buf, sizeof(buf), "mesh:%s yaw%+.0f", path.c_str(),
                  yaw_offset_rad * 180.0 / M_PI);
    f->source_ = buf;
    report.ok = true;
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

        // ── (N) ★THE FRAME PIN. This is the one assertion that catches a wrong mesh->robot yaw, and a wrong
    // yaw is the failure that INVERTS this whole change: on P3Bot, +90 deg makes the body 22 mm wider than
    // the compiled hull (correct), while 0 makes it 45 mm NARROWER — the same file, with the error moved
    // onto the axis along which a corridor closes. "Reaches further sideways than forwards" is a statement
    // no aspect ratio can fake and a 90 deg error always breaks, so it is asserted directly rather than
    // inferred from extents. The mesh path is optional: if it is not there (a tools-only checkout), say so
    // rather than passing silently — a test that cannot run has not run.
    {
        const char *mesh = "../robot_concept/meshes/p3bot.obj";
        MeshReport rep;
        auto body = from_obj(mesh, static_cast<float>(M_PI_2), rep);
        if (not body.has_value())
            std::printf("  frame pin: SKIPPED (%s) — this test is only meaningful with the mesh present\n",
                        rep.reason.c_str());
        else
        {
            const float forward  = body->support_radius(0.f, {0.f, 1.f});
            const float sideways = body->support_radius(0.f, {1.f, 0.f});
            std::printf("  frame pin: p3bot yaw+90 reaches %.4f m FORWARD, %.4f m SIDEWAYS\n",
                        forward, sideways);
            check(sideways > forward + 0.05f,
                  "★P3Bot is a long-and-narrow base: in the ROBOT frame it must reach further SIDEWAYS "
                  "than FORWARD. If this fails the mesh->robot yaw is wrong by a quarter turn");
            check(std::abs(sideways - 0.2938f) < 0.005f, "...and sideways is the mesh's own |y|max, 0.2938");
            check(std::abs(forward - 0.2266f) < 0.005f, "...and forward is its |x|max, 0.2266");
            // The same file at yaw 0 must be the transpose — proof the difference is the ANGLE, not the file.
            MeshReport rep0;
            auto wrong = from_obj(mesh, 0.f, rep0);
            check(wrong.has_value() and wrong->support_radius(0.f, {0.f, 1.f}) > forward + 0.05f,
                  "★at yaw 0 the SAME mesh reaches further forward — which is what a quarter-turn error "
                  "looks like, and why the angle is stated in config rather than guessed");
        }
    }

    // (N+1) A mesh in the wrong units must be REJECTED, never auto-scaled: a silent x0.001 turns a wrong
    // file into a plausible robot.
    {
        std::vector<Eigen::Vector2f> huge{{-300.f, -200.f}, {300.f, -200.f}, {300.f, 200.f}, {-300.f, 200.f}};
        std::string why;
        check(from_polygon(huge, &why).has_value(),
              "from_polygon itself accepts any convex CCW polygon — the units check belongs to from_obj");
        // ...and the winding/origin guards do their job.
        std::vector<Eigen::Vector2f> cw{{-1.f, -1.f}, {-1.f, 1.f}, {1.f, 1.f}, {1.f, -1.f}};
        check(not from_polygon(cw, &why).has_value(), "★a CLOCKWISE polygon must be refused, not reversed: "
              "contains() would then report the whole world as inside the robot");
        std::vector<Eigen::Vector2f> off{{1.f, 1.f}, {2.f, 1.f}, {2.f, 2.f}, {1.f, 2.f}};
        check(not from_polygon(off, &why).has_value(),
              "★a polygon not containing the ORIGIN is not drawn about the rotation centre");
    }

    std::printf("RobotFootprint::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
