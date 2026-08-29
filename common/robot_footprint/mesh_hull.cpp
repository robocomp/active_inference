/*
 * mesh_hull.cpp — see mesh_hull.h
 */

#include "mesh_hull.h"

#include <algorithm>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>

namespace rc::mesh
{

namespace
{
// One `v` line: three floats, whitespace separated, LOCALE-INDEPENDENT. See the header — this is the whole
// reason the file exists rather than a call into one of the two OBJ readers already in the repo.
bool parse_v_line(const char *p, const char *end, Eigen::Vector3f &out)
{
    for (int k = 0; k < 3; ++k)
    {
        while (p < end and (*p == ' ' or *p == '\t')) ++p;
        if (p >= end) return false;
        float value = 0.f;
        const auto [next, ec] = std::from_chars(p, end, value);
        if (ec != std::errc{}) return false;
        if (not std::isfinite(value)) return false;
        out[k] = value;
        p = next;
    }
    return true;
}

float cross_z(const Eigen::Vector2f &o, const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
}
}   // namespace

std::optional<Vertices> read_obj_vertices(const std::string &path, std::string *err)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (not f.is_open())
    {
        if (err) *err = "cannot open '" + path + "'";
        return std::nullopt;
    }

    Vertices out;
    out.bb_min.setConstant(std::numeric_limits<float>::max());
    out.bb_max.setConstant(std::numeric_limits<float>::lowest());
    // shadow.obj is 12 MB / 141k vertices, so reserve rather than grow one push at a time.
    out.v.reserve(1 << 14);

    std::string line;
    while (std::getline(f, line))
    {
        // Only `v ` — not `vn`, not `vt`, not `vp`. The trailing space is the whole discriminator.
        if (line.size() < 2 or line[0] != 'v' or (line[1] != ' ' and line[1] != '\t')) continue;
        ++out.v_lines;
        Eigen::Vector3f p;
        if (not parse_v_line(line.data() + 1, line.data() + line.size(), p))
        {
            ++out.v_rejected;
            continue;
        }
        out.v.push_back(p);
        out.bb_min = out.bb_min.cwiseMin(p);
        out.bb_max = out.bb_max.cwiseMax(p);
    }

    if (out.v.empty())
    {
        if (err) *err = "no parseable 'v' lines in '" + path + "' (" + std::to_string(out.v_lines)
                      + " seen, " + std::to_string(out.v_rejected) + " rejected)";
        return std::nullopt;
    }
    return out;
}

std::vector<Eigen::Vector2f> convex_hull_2d(std::vector<Eigen::Vector2f> pts)
{
    if (pts.size() < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
              { return a.x() < b.x() or (a.x() == b.x() and a.y() < b.y()); });
    pts.erase(std::unique(pts.begin(), pts.end(),
                          [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
                          { return a.x() == b.x() and a.y() == b.y(); }),
              pts.end());
    if (pts.size() < 3) return pts;

    const std::size_t n = pts.size();
    std::vector<Eigen::Vector2f> hull(2 * n);
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i)                       // lower
    {
        while (k >= 2 and cross_z(hull[k - 2], hull[k - 1], pts[i]) <= 0.f) --k;
        hull[k++] = pts[i];
    }
    for (std::size_t i = n - 1, t = k + 1; i > 0; --i)        // upper
    {
        while (k >= t and cross_z(hull[k - 2], hull[k - 1], pts[i - 1]) <= 0.f) --k;
        hull[k++] = pts[i - 1];
    }
    hull.resize(k > 0 ? k - 1 : 0);                           // last point == first
    return hull;
}

float polygon_area(const std::vector<Eigen::Vector2f> &poly)
{
    if (poly.size() < 3) return 0.f;
    double a = 0.0;
    for (std::size_t i = 0, n = poly.size(); i < n; ++i)
    {
        const Eigen::Vector2f &p = poly[i], &q = poly[(i + 1) % n];
        a += static_cast<double>(p.x()) * q.y() - static_cast<double>(q.x()) * p.y();
    }
    return static_cast<float>(0.5 * a);
}

Simplified simplify_hull_outward(const std::vector<Eigen::Vector2f> &hull,
                                 std::size_t max_verts,
                                 float max_area_growth_frac)
{
    Simplified out{.poly = hull, .area_growth_frac = 0.f};
    if (hull.size() < 4) return out;

    const float area0 = polygon_area(hull);
    if (area0 <= 0.f) return out;
    max_verts = std::max<std::size_t>(3, max_verts);

    // Removing vertex i replaces the two edges (i-1,i) and (i,i+1) by the intersection of their LINES. That
    // point lies OUTSIDE the polygon for a convex hull, so the result contains the original — which is the
    // whole safety property. Cost is the area of the triangle that gets added; remove the cheapest first.
    while (out.poly.size() > max_verts)
    {
        const std::size_t n = out.poly.size();
        if (n <= 3) break;
        std::size_t best = n;
        float best_cost = std::numeric_limits<float>::max();
        Eigen::Vector2f best_point = Eigen::Vector2f::Zero();

        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f &a = out.poly[(i + n - 2) % n];
            const Eigen::Vector2f &b = out.poly[(i + n - 1) % n];
            const Eigen::Vector2f &c = out.poly[(i + 1) % n];
            const Eigen::Vector2f &d = out.poly[(i + 2) % n];
            const Eigen::Vector2f u = b - a, w = d - c;
            const float det = u.x() * w.y() - u.y() * w.x();
            if (std::abs(det) < 1e-9f) continue;              // parallel edges: no finite intersection
            const float t = ((c.x() - b.x()) * w.y() - (c.y() - b.y()) * w.x()) / det;
            const Eigen::Vector2f x = b + t * u;
            // The intersection must lie forward of both edges, or this is not the outward corner.
            if (t <= 0.f) continue;
            const float cost = std::abs(cross_z(b, x, c)) * 0.5f;
            if (cost < best_cost) { best_cost = cost; best = i; best_point = x; }
        }
        if (best == n) break;
        if ((best_cost + polygon_area(out.poly) - area0) / area0 > max_area_growth_frac) break;

        std::vector<Eigen::Vector2f> next;
        next.reserve(out.poly.size() - 1);
        const std::size_t drop_prev = (best + n - 1) % n;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i == best) continue;                          // the vertex being removed
            if (i == drop_prev) { next.push_back(best_point); continue; }   // ...and its neighbour moves out
            next.push_back(out.poly[i]);
        }
        out.poly = std::move(next);
    }

    out.area_growth_frac = (polygon_area(out.poly) - area0) / area0;
    return out;
}

std::optional<HullResult> hull_from_obj(const std::string &path,
                                        float z_lo, float z_hi,
                                        std::size_t max_verts,
                                        float max_area_growth_frac,
                                        std::string *err)
{
    const auto verts = read_obj_vertices(path, err);
    if (not verts.has_value()) return std::nullopt;

    std::vector<Eigen::Vector2f> flat;
    flat.reserve(verts->v.size());
    for (const auto &p : verts->v)
        if (p.z() >= z_lo and p.z() <= z_hi) flat.emplace_back(p.x(), p.y());

    if (flat.size() < 3)
    {
        if (err) *err = "fewer than 3 vertices in the slab [" + std::to_string(z_lo) + ", "
                      + std::to_string(z_hi) + "]";
        return std::nullopt;
    }

    HullResult r;
    r.bb_min = verts->bb_min;
    r.bb_max = verts->bb_max;
    r.vertices = static_cast<long>(verts->v.size());
    r.vertices_rejected = verts->v_rejected;
    r.hull_raw = convex_hull_2d(std::move(flat));
    if (r.hull_raw.size() < 3)
    {
        if (err) *err = "hull is degenerate (fewer than 3 vertices)";
        return std::nullopt;
    }
    r.area_raw = polygon_area(r.hull_raw);
    const auto s = simplify_hull_outward(r.hull_raw, max_verts, max_area_growth_frac);
    r.hull = s.poly;
    r.area_growth_frac = s.area_growth_frac;
    r.area = polygon_area(r.hull);
    return r;
}

// ── SELF-TEST ───────────────────────────────────────────────────────────────────────────────────
bool self_test()
{
    bool ok = true;
    const auto check = [&ok](bool cond, const char *what)
    {
        if (not cond) { std::printf("  FAIL: %s\n", what); ok = false; }
    };

    // (1) ★THE LOCALE TRAP, REPRODUCED RATHER THAN ASSUMED. This is the one test in the repo that reads a
    // decimal-point float the way the agent does. If the caller has not activated a comma locale the test
    // still passes — but it then proves nothing, so it says so instead of claiming a pass it did not earn.
    {
        const char *decimal = std::setlocale(LC_NUMERIC, nullptr);
        const bool comma_locale = decimal != nullptr and std::string_view(decimal) != "C"
                                                     and std::string_view(decimal) != "POSIX";
        const std::string line = "v -0.1886818 0.1865224 0.2369575";
        Eigen::Vector3f p = Eigen::Vector3f::Zero();
        check(parse_v_line(line.data() + 1, line.data() + line.size(), p), "a 'v' line must parse");
        check(std::abs(p.x() + 0.1886818f) < 1e-6f, "x must survive the decimal POINT in full precision");
        check(std::abs(p.y() - 0.1865224f) < 1e-6f, "y must survive it too");
        check(std::abs(p.z() - 0.2369575f) < 1e-6f, "...and z");
        std::printf("  locale trap: LC_NUMERIC='%s' -> %s\n", decimal ? decimal : "?",
                    comma_locale ? "COMMA locale ACTIVE, the trap is genuinely exercised"
                                 : "C locale — test ran but proves nothing; call setlocale(LC_ALL,\"\") first");
    }

    // (2) A square must hull to itself, CCW, and its area must be right.
    {
        const std::vector<Eigen::Vector2f> square{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        auto h = convex_hull_2d(square);
        check(h.size() == 4, "a square hulls to 4 vertices");
        check(std::abs(polygon_area(h) - 4.f) < 1e-5f, "...of area 4, POSITIVE (CCW)");
        // Interior points must not appear.
        auto with_inner = square;
        with_inner.emplace_back(0.f, 0.f);
        with_inner.emplace_back(0.5f, 0.2f);
        check(convex_hull_2d(with_inner).size() == 4, "interior points must not enter the hull");
    }

    // (3) Rotation equivariance: hull(R p) == R hull(p). This is what says the hull carries no axis
    // convention of its own — the mesh->robot yaw is the CALLER's to apply, and a bug here would look
    // exactly like a wrong yaw.
    {
        std::vector<Eigen::Vector2f> pts;
        for (int i = 0; i < 40; ++i)
        {
            const float t = 2.f * static_cast<float>(M_PI) * static_cast<float>(i) / 40.f;
            pts.emplace_back(0.30f * std::cos(t), 0.22f * std::sin(t));
        }
        const auto h0 = convex_hull_2d(pts);
        for (int k = 1; k < 12; ++k)
        {
            const float a = 2.f * static_cast<float>(M_PI) * static_cast<float>(k) / 12.f;
            const Eigen::Rotation2Df R(a);
            std::vector<Eigen::Vector2f> rot;
            rot.reserve(pts.size());
            for (const auto &p : pts) rot.push_back(R * p);
            const auto hr = convex_hull_2d(rot);
            check(hr.size() == h0.size(), "rotating the input must not change the hull's vertex count");
            check(std::abs(polygon_area(hr) - polygon_area(h0)) < 1e-4f,
                  "...nor its area — the hull must be equivariant, not opinionated about axes");
        }
    }

    // (4) ★SIMPLIFICATION IS A STRICT SUPERSET. Every raw vertex must still be inside the simplified
    // polygon. A simplifier that cut a corner would hand the planner a body smaller than the robot, which
    // is the one failure mode this whole module exists to make impossible.
    {
        std::vector<Eigen::Vector2f> pts;
        for (int i = 0; i < 60; ++i)
        {
            const float t = 2.f * static_cast<float>(M_PI) * static_cast<float>(i) / 60.f;
            pts.emplace_back(0.28f * std::cos(t), 0.23f * std::sin(t));
        }
        const auto raw = convex_hull_2d(pts);
        // ★TWO BOUNDS, AND WHICHEVER BINDS FIRST WINS — the vertex budget and the area cap. The first
        // version of this test asserted the budget was always reached and FAILED here at 14 verts / +3.05%,
        // which was the code being right: one more removal would have cost more than the 5% allowed. The
        // contract is "never grow more than the cap", not "always hit the count".
        const auto s = simplify_hull_outward(raw, 10, 0.05f);
        check(s.poly.size() >= 3 and s.poly.size() <= raw.size(), "simplification may only REMOVE vertices");
        check(s.area_growth_frac >= -1e-6f, "area may only GROW — outward-only is the safety property");
        check(s.area_growth_frac <= 0.05f + 1e-6f, "...and never past the stated cap");
        // With a cap that cannot bind, the budget must be the thing that stops it.
        const auto loose = simplify_hull_outward(raw, 10, 10.f);
        check(loose.poly.size() == 10, "with the area cap slack, the vertex budget is what binds");
        for (const auto &p : raw)
        {
            bool inside = true;
            for (std::size_t i = 0, n = s.poly.size(); i < n; ++i)
                if (cross_z(s.poly[i], s.poly[(i + 1) % n], p) < -1e-5f) { inside = false; break; }
            check(inside, "★every raw hull vertex must remain INSIDE the simplified polygon");
            if (not inside) break;
        }
        std::printf("  simplify: %zu -> %zu verts, area %+.3f%%\n",
                    raw.size(), s.poly.size(), 100.f * s.area_growth_frac);
    }

    std::printf("rc::mesh::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}   // namespace rc::mesh
