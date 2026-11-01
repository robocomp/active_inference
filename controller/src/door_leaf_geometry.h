// door_leaf_geometry.h — pure geometry for drawing a door as an ARTICULATED object on the 2-D canvas: the aperture (the
// gap in the wall) plus the LEAF hung from its believed hinge, opened by the believed angle phi toward the believed side.
// Room frame, metres. No DSR, no Qt. Test: tests/test_door_leaf_draw.cpp.
//
// WHAT THE WORLD MODEL CARRIES (verified 2026-11-01): the door node has the aperture (RT room->door: centre + wall yaw;
// width_m) and door_phi_rad. HINGE jamb and SWING side are NOT on the door node: door_concept holds them as log-odds and
// publishes them only on the interaction anchor "<door>@robot" (anchor_leaf.h, hinge x swing hypotheses with P). So a door
// with an anchor is drawn with its believed leaf hypotheses; a door without one is drawn with phi and "side unknown"
// (the four hinge x swing candidates, faint).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace rc::doordraw
{
using V2 = Eigen::Vector2f;

struct Leaf { V2 hinge, tip; float p = 1.f; float sigma_phi = 0.f; };

// A leaf from the aperture alone. u = unit along the wall, c = aperture centre, w = width. hinge_sgn = -1: hinge at the
// c - w/2 u jamb (leaf shuts toward +u), +1: at c + w/2 u. swing_sgn = +1/-1: toward +n / -n (n = u rotated +90 deg).
[[nodiscard]] inline Leaf leaf_from_aperture(V2 c, V2 u, float w, int hinge_sgn, int swing_sgn, float phi)
{
    const V2 n(-u.y(), u.x());
    const float sgn = hinge_sgn < 0 ? 1.f : -1.f;   // direction along u that the leaf shuts toward
    Leaf l;
    l.hinge = c + float(hinge_sgn) * 0.5f * w * u;
    l.tip = l.hinge + w * (sgn * std::cos(phi) * u + float(swing_sgn) * std::sin(phi) * n);
    return l;
}

// An anchor-frame point (origin = opening centre, x along the wall = (cos yaw, sin yaw) in the ROBOT frame, y = its normal)
// into the room frame through robot->room transform T. The same mapping the anchor overlay uses for the jambs.
[[nodiscard]] inline V2 anchor_to_room(const V2 &p, const V2 &centre_robot, float yaw, const Eigen::Affine2f &T)
{
    const V2 a(std::cos(yaw), std::sin(yaw)), n(-a.y(), a.x());
    return T * (centre_robot + p.x() * a + p.y() * n);
}

// The phi +- sigma wedge about the hinge: the arc of the free edge from phi - sigma to phi + sigma (rotating the hinge->tip
// vector by -sigma .. +sigma), as a polygon hinge, arc points... Empty when sigma is 0.
[[nodiscard]] inline std::vector<V2> wedge(const V2 &hinge, const V2 &tip, float sigma, int steps = 8)
{
    std::vector<V2> out;
    if (not (sigma > 0.f)) return out;
    const V2 d = tip - hinge;
    out.push_back(hinge);
    for (int k = 0; k <= steps; ++k)
    {
        const float a = -sigma + 2.f * sigma * float(k) / float(steps);
        out.push_back(hinge + V2(std::cos(a) * d.x() - std::sin(a) * d.y(), std::sin(a) * d.x() + std::cos(a) * d.y()));
    }
    return out;
}

// Does the aperture [c - w/2 u, c + w/2 u] run past the END of its wall? The wall is the polygon edge nearest the centre
// among edges roughly parallel to u (|cos| > 0.9). Returns the overhang lengths (m) past the c - w/2 u end and the
// c + w/2 u end, 0 when within (or when no such edge exists). Display only: never used to move the aperture.
struct Overhang { float lo = 0.f, hi = 0.f; };
[[nodiscard]] inline Overhang aperture_overhang(V2 c, V2 u, float w, const std::vector<V2> &poly)
{
    Overhang best;
    float best_d = 1e9f;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n and n >= 3; ++i)
    {
        const V2 a = poly[i], b = poly[(i + 1) % n];
        const V2 e = b - a;
        const float L = e.norm();
        if (not (L > 1e-3f) or std::abs(e.dot(u)) / L < 0.9f) continue;
        const V2 t = e / L;
        const float dist = std::abs((c - a).x() * t.y() - (c - a).y() * t.x());
        if (dist >= best_d) continue;
        best_d = dist;
        const float s0 = (c - 0.5f * w * u - a).dot(t), s1 = (c + 0.5f * w * u - a).dot(t);
        const float lo = std::min(s0, s1), hi = std::max(s0, s1);
        const bool flip = t.dot(u) < 0.f;   // t runs against u: the "lo end" along u is the high end along t
        const float over_lo_t = std::max(0.f, -lo), over_hi_t = std::max(0.f, hi - L);
        best = flip ? Overhang{over_hi_t, over_lo_t} : Overhang{over_lo_t, over_hi_t};
    }
    return best_d < 0.5f ? best : Overhang{};
}
}  // namespace rc::doordraw
