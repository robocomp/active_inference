// countable_containment.h — the prior's SUPPORT on the persistent path (CW1): no part of the object's footprint lies past a room
// wall. Pure geometry, header-only, shared by the belief (CountableEngineModel::accumulate_containment), the replay tool's columns
// and the tests.
//
// The region R the footprint must lie in is the room polygon (room frame, any shape, convex or not) when the frame carries one,
// else the half-plane on the room side of the staged wall line, else nothing (the term is inert). For a footprint corner p the
// signed distance to R is d(p) = +dist(p, ∂R) inside, −dist(p, ∂R) outside; it is continuous across ∂R (0 on it).
//
// THE MODEL (fix round 1). The REAL object does not cross the TRUE walls: a hard support g_real ≥ δ_j for every wall j, where the
// true wall j is the staged line moved by δ_j along its inward normal. Two errors separate the box corner from that support:
//   · the box surface is the real surface up to the model's surface error, N(0, σ_s²), σ_s = Model.ModelSigmaM (the σ every
//     point's R and the LiDAR first-crossing already carry);
//   · the wall j is where the room put it up to δ_j ~ N(δ̄_j, v_j): for the STAGED wall the wall latent's carried belief (its
//     predicted mean and variance, handed in by the engine, fixed over the update); for every other wall the staged wall's σ²
//     with mean 0 — ⚠ STAND-IN, flagged: room_concept publishes no per-edge covariance (the agent reads only
//     delimiting_polygon_x/y; WallRef::sigma_m is the fitter's 0.02 m), so every edge carries the staged σ. Not invented: the
//     same value the staged wall and the beam's foreign density use.
// The PROFILE (MAP) likelihood of the corner under the support, jointly over the real corner and δ_j, is
//     E = ½ · min(0, d_k − δ̄_j)² / (σ_s² + v_j)
// (minimise ½(g − g_r)²/σ_s² + ½(δ − δ̄)²/v subject to g_r ≥ δ: for g < δ̄ the optimum splits the violation in proportion to the
// two variances and leaves exactly this). So each wall's error is counted ONCE, as its own marginal, inside the term, and no state
// is moved by it: the term is a prior on θ only (engine selector 0), so δ's carried mean is never pushed without information
// (review I1). The profile, not the marginal: the marginal −log Φ((d − δ̄)/σ) has a nonzero gradient on the legal side and,
// re-applied every update, would push a flush object into the room until d ≈ 3σ; the profile's fixed-point set is exactly
// "every corner legal".
//
// The staged wall's mean δ̄ moves the staged edge of the polygon (both of its vertices, so the adjacent walls meet it there), and
// the signed distance is to that polygon. The variance v at a corner outside is the variance of the nearest boundary point along
// the corner's way back in u: v_j on edge j's interior (u = n_j), Σ_j (u·n_j)² v_j at a vertex (both edges). For a right-angled
// room corner the two agree on the vertex region's boundary (u·n_other = 0 there); for other angles v jumps by cos²(angle)·v_other
// across it — flagged, a geometric approximation of the vertex's displacement. At a vertex v depends on θ through u, and the
// gradient carries that term (fix round 2, review M-a); the curvature stays Gauss–Newton. ⚠ shifted_polygon moves both staged
// vertices along the staged n: that keeps each adjacent wall's LINE only for right-angled corners; otherwise it rotates the adjacent
// wall by about δ̄/L (review M-b), flagged with the vertex-variance note.
//
// Corner switching. Each corner is its own smooth term with its own Jacobian. The support-function gap of one wall,
// g(θ) = min_k (p_k − x_w)·n_w, is what the sum reduces to while one corner penetrates; when two corners penetrate (a face through
// the wall) both terms are active, so the energy stays C¹ across the tie instead of switching its gradient from one corner to the
// other. ⚠ The two corners' surface errors are treated as independent, so a whole face through a wall gets 2λ, while a rigid face's
// error is largely common-mode (review M1); harmless while the term is outside P, stated. A corner inside R contributes exactly 0
// (value, gradient, curvature): a mid-room object is untouched, bit for bit. Outside a polygon the gradient points from the corner
// to its nearest boundary point, which is a vertex past a convex room corner — the shortest way back in across BOTH walls — and an
// edge's normal elsewhere. ⚠ Only the footprint's four corners are tested (review M4): a reflex room vertex poking into a footprint
// EDGE with every corner inside is not penalised (a generic support would also test the polygon's vertices against the box).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace rc::containment
{

// The footprint's corners, counter-clockwise from (+w/2, +h/2) in the local frame: c + R(yaw)·(sx·w/2, sy·h/2).
struct Corner
{
    Eigen::Vector2f p;
    float sx, sy;                                            // the corner's signs in the local frame
};
inline std::array<Corner, 4> corners(float cx, float cy, float w, float h, float yaw)
{
    const Eigen::Vector2f ux(std::cos(yaw), std::sin(yaw)), uy(-std::sin(yaw), std::cos(yaw)), c(cx, cy);
    std::array<Corner, 4> out;
    constexpr std::array<std::array<float, 2>, 4> sg{{{1.0f, 1.0f}, {-1.0f, 1.0f}, {-1.0f, -1.0f}, {1.0f, -1.0f}}};
    for (std::size_t k = 0; k < 4; ++k)
        out[k] = {c + sg[k][0] * 0.5f * w * ux + sg[k][1] * 0.5f * h * uy, sg[k][0], sg[k][1]};
    return out;
}

// ∂p/∂θ for θ = (cx, cy, H, w, h, yaw): a 2×6 Jacobian (H does not move the footprint).
inline Eigen::Matrix<float, 2, 6> corner_jacobian(const Corner& k, float w, float h, float yaw)
{
    const Eigen::Vector2f ux(std::cos(yaw), std::sin(yaw)), uy(-std::sin(yaw), std::cos(yaw));
    Eigen::Matrix<float, 2, 6> J = Eigen::Matrix<float, 2, 6>::Zero();
    J(0, 0) = 1.0f; J(1, 1) = 1.0f;
    J.col(3) = 0.5f * k.sx * ux;
    J.col(4) = 0.5f * k.sy * uy;
    J.col(5) = k.sx * 0.5f * w * uy - k.sy * 0.5f * h * ux;  // d ux/dyaw = uy, d uy/dyaw = −ux
    return J;
}

// Even-odd containment of q in the polygon.
inline bool inside_polygon(const Eigen::Vector2f& q, const std::vector<Eigen::Vector2f>& poly)
{
    bool in = false;
    const std::size_t n = poly.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f& b = poly[j];
        if ((a.y() > q.y()) != (b.y() > q.y()) and q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x())
            in = not in;
    }
    return in;
}

// Signed distance of q to the region (see the header) and its gradient ∂d/∂q, with the nearest boundary feature: edge i (from
// poly[i] to poly[i+1]) and t, the foot's parameter on it (0 or 1 ⇒ a vertex). Polygon: ≥ 3 vertices; else the half-plane
// {x : (x − wp)·wn ≥ 0} (wn unit) when has_wall (edge −1); else +inf (no region: inert).
struct Signed
{
    float d = std::numeric_limits<float>::infinity();
    Eigen::Vector2f grad = Eigen::Vector2f::Zero();
    int edge = -1;
    float t = 0.5f;
};
inline Signed signed_distance(const Eigen::Vector2f& q, const std::vector<Eigen::Vector2f>& poly, bool has_wall,
                              const Eigen::Vector2f& wp, const Eigen::Vector2f& wn)
{
    Signed s;
    if (poly.size() >= 3)
    {
        float best = std::numeric_limits<float>::infinity();
        Eigen::Vector2f foot = q;
        for (std::size_t i = 0, n = poly.size(); i < n; ++i)
        {
            const Eigen::Vector2f a = poly[i], ab = poly[(i + 1) % n] - a;
            const float len2 = ab.squaredNorm();
            const float t = len2 > 0.0f ? std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
            const Eigen::Vector2f f = a + t * ab;
            const float d2 = (q - f).squaredNorm();
            if (d2 < best) { best = d2; foot = f; s.edge = int(i); s.t = t; }
        }
        const float dist = std::sqrt(best);
        if (inside_polygon(q, poly)) { s.d = dist; return s; }   // inside: only the value is read (the term is 0 there)
        s.d = -dist;
        if (dist > 0.0f) s.grad = (foot - q) / dist;             // d = −|q − foot| ⇒ ∂d/∂q = (foot − q)/|q − foot|
        return s;
    }
    if (has_wall) { s.d = (q - wp).dot(wn); s.grad = wn; }
    return s;
}

// The inward unit normal of polygon edge i (from poly[i] to poly[i+1]), by the polygon's orientation (signed area).
inline Eigen::Vector2f inward_normal(const std::vector<Eigen::Vector2f>& poly, int i)
{
    double area2 = 0.0;
    for (std::size_t k = 0, n = poly.size(); k < n; ++k)
    {
        const Eigen::Vector2f& a = poly[k];
        const Eigen::Vector2f& b = poly[(k + 1) % n];
        area2 += double(a.x()) * double(b.y()) - double(b.x()) * double(a.y());
    }
    const Eigen::Vector2f e = poly[(std::size_t(i) + 1) % poly.size()] - poly[std::size_t(i)];
    Eigen::Vector2f n(-e.y(), e.x());                        // left of the edge: inside for a counter-clockwise polygon
    if (area2 < 0.0) n = -n;
    const float l = n.norm();
    return l > 0.0f ? Eigen::Vector2f(n / l) : n;
}

// The staged polygon with the staged edge moved by δ̄ along its inward normal (both vertices: the adjacent walls meet it there).
inline std::vector<Eigen::Vector2f> shifted_polygon(std::vector<Eigen::Vector2f> poly, int staged_edge, const Eigen::Vector2f& n,
                                                    float delta)
{
    const int m = int(poly.size());
    if (m >= 3 and staged_edge >= 0 and staged_edge < m and delta != 0.0f)
    {
        poly[std::size_t(staged_edge)] += delta * n;
        poly[std::size_t((staged_edge + 1) % m)] += delta * n;
    }
    return poly;
}

// The smallest signed distance over the footprint's corners (a diagnostic: < 0 ⇒ some corner is past a wall).
inline float min_corner_distance(float cx, float cy, float w, float h, float yaw, const std::vector<Eigen::Vector2f>& poly,
                                 bool has_wall, const Eigen::Vector2f& wp, const Eigen::Vector2f& wn)
{
    float m = std::numeric_limits<float>::infinity();
    for (const auto& k : corners(cx, cy, w, h, yaw)) m = std::min(m, signed_distance(k.p, poly, has_wall, wp, wn).d);
    return m;
}

}   // namespace rc::containment
