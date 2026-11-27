// countable_containment.h — the prior's SUPPORT on the persistent path (CW1): no part of the object's footprint lies past a room
// wall. Pure geometry, header-only, shared by the belief (CountableBelief::accumulate_containment), the replay tool's columns and
// the tests.
//
// The region R the footprint must lie in is the room polygon (room frame, any shape, convex or not) when the frame carries one,
// else the half-plane on the room side of the staged wall line, else nothing (the term is inert). For a footprint corner p the
// signed distance to R is d(p) = +dist(p, ∂R) inside, −dist(p, ∂R) outside; it is continuous across ∂R (0 on it).
//
// The term: E = Σ_k ½ (min(0, d(p_k(θ))) / σ_s)², k over the footprint's four corners, σ_s = the model's surface std
// (Model.ModelSigmaM, the same σ the points' R and the LiDAR first-crossing carry). Derivation: the REAL object does not cross the
// TRUE wall, a hard support g_real ≥ 0; the box's corner is the real corner up to the model's surface error N(0, σ_s²); the
// profile (MAP) likelihood of the box corner under that support is N's value at the nearest legal point, ½(min(0, g)/σ_s)². The
// wall's own position error is not in σ_s: it is the wall latent δ (the engine evaluates the term at θ + B_w b), so σ_wall is
// counted once.
//
// Corner switching. Each corner is its own smooth term with its own Jacobian. The support-function gap of one wall,
// g(θ) = min_k (p_k − x_w)·n_w, is what the sum reduces to while one corner penetrates; when two corners penetrate (a face through
// the wall) both terms are active, so the energy stays C¹ across the tie instead of switching its gradient from one corner to the
// other. A corner inside R contributes exactly 0 (value, gradient, curvature): a mid-room object is untouched, bit for bit.
// Outside a polygon the gradient points from the corner to its nearest boundary point, which is a vertex past a convex room
// corner — the shortest way back in across BOTH walls — and an edge's normal elsewhere.
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

// Signed distance of q to the region (see the header) and its gradient ∂d/∂q. Polygon: ≥ 3 vertices; else the half-plane
// {x : (x − wp)·wn ≥ 0} (wn unit) when has_wall; else +inf (no region: inert).
struct Signed { float d = std::numeric_limits<float>::infinity(); Eigen::Vector2f grad = Eigen::Vector2f::Zero(); };
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
            if (d2 < best) { best = d2; foot = f; }
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

// The smallest signed distance over the footprint's corners (a diagnostic: < 0 ⇒ some corner is past a wall).
inline float min_corner_distance(float cx, float cy, float w, float h, float yaw, const std::vector<Eigen::Vector2f>& poly,
                                 bool has_wall, const Eigen::Vector2f& wp, const Eigen::Vector2f& wn)
{
    float m = std::numeric_limits<float>::infinity();
    for (const auto& k : corners(cx, cy, w, h, yaw)) m = std::min(m, signed_distance(k.p, poly, has_wall, wp, wn).d);
    return m;
}

}   // namespace rc::containment
