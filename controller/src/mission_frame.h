#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// MISSION FRAME TRANSFER (2026-10-10) — a mission recorded in one room frame, run in another.
//
// An estimated room's frame is the robot's pose when the room was born, so the same apartment gets a different
// frame on every run; the surveyed layout has yet another. A mission therefore stores the ROOM OUTLINE it was
// recorded in, and at start the controller registers that outline against the current room's
// delimiting_polygon and moves the waypoints with the result:
//   1. coarse — dominant wall direction of each outline (edge-length-weighted circular mean of 4θ), so the four
//      candidate rotations of a Manhattan room are k·90°; each candidate centred on the area centroids;
//   2. refine — TRANSLATION-only ICP from points sampled along the stored outline to the current outline's edges (the
//      rotation is fixed by the wall directions, which are far more precise than a free-rotation fit);
//   3. pick   — the candidate with the best IoU (rasterised). The IoU is returned so the caller can refuse a
//      mission recorded in a different room.
// Pure: no Qt, no DSR.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace rc::mission_frame
{
using Poly = std::vector<Eigen::Vector2f>;

inline bool inside(const Eigen::Vector2f& q, const Poly& p)
{
    bool in = false;
    for (std::size_t i = 0, j = p.size() - 1; i < p.size(); j = i++)
        if (((p[i].y() > q.y()) != (p[j].y() > q.y()))
            and (q.x() < (p[j].x() - p[i].x()) * (q.y() - p[i].y()) / (p[j].y() - p[i].y()) + p[i].x()))
            in = not in;
    return in;
}
inline Eigen::Vector2f area_centroid(const Poly& p)
{
    double A = 0, cx = 0, cy = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto& a = p[i]; const auto& b = p[(i + 1) % p.size()];
        const double c = static_cast<double>(a.x()) * b.y() - static_cast<double>(b.x()) * a.y();
        A += c; cx += (a.x() + b.x()) * c; cy += (a.y() + b.y()) * c;
    }
    if (std::abs(A) < 1e-12) return p.empty() ? Eigen::Vector2f::Zero() : p.front();
    return {static_cast<float>(cx / (3.0 * A)), static_cast<float>(cy / (3.0 * A))};
}
inline float dominant_angle(const Poly& p)
{
    double zr = 0, zi = 0;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const Eigen::Vector2f d = p[(i + 1) % p.size()] - p[i];
        const double L = d.norm(), a = 4.0 * std::atan2(d.y(), d.x());
        zr += L * std::cos(a); zi += L * std::sin(a);
    }
    return static_cast<float>(0.25 * std::atan2(zi, zr));
}
inline Eigen::Vector2f nearest_on(const Poly& p, const Eigen::Vector2f& q)
{
    Eigen::Vector2f best = p.front(); float bd = 1e30f;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const Eigen::Vector2f a = p[i], ab = p[(i + 1) % p.size()] - a;
        const float t = std::clamp((q - a).dot(ab) / std::max(ab.squaredNorm(), 1e-12f), 0.f, 1.f);
        const Eigen::Vector2f c = a + t * ab;
        if (const float d = (c - q).squaredNorm(); d < bd) { bd = d; best = c; }
    }
    return best;
}
inline float iou(const Poly& a, const Poly& b, float cell = 0.05f)
{
    Eigen::Vector2f lo = a.front(), hi = lo;
    for (const auto* P : {&a, &b}) for (const auto& v : *P) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
    long I = 0, U = 0;
    for (float x = lo.x(); x <= hi.x(); x += cell)
        for (float y = lo.y(); y <= hi.y(); y += cell)
        {
            const bool ia = inside({x, y}, a), ib = inside({x, y}, b);
            I += (ia and ib); U += (ia or ib);
        }
    return U > 0 ? static_cast<float>(I) / static_cast<float>(U) : 0.f;
}

struct Result { Eigen::Matrix2f R = Eigen::Matrix2f::Identity(); Eigen::Vector2f t = Eigen::Vector2f::Zero(); float iou = 0.f; float yaw = 0.f; };

/// Rigid transform taking points of the frame `from` was recorded in into the frame of `to`.
inline Result register_outlines(const Poly& from, const Poly& to)
{
    Result best;
    if (from.size() < 3 or to.size() < 3) return best;
    Poly samples;   // points along `from`, every 5 cm
    for (std::size_t i = 0; i < from.size(); ++i)
    {
        const Eigen::Vector2f a = from[i], b = from[(i + 1) % from.size()];
        const int n = std::max(1, static_cast<int>((b - a).norm() / 0.05f));
        for (int k = 0; k < n; ++k) samples.push_back(a + (b - a) * (static_cast<float>(k) / static_cast<float>(n)));
    }
    const Eigen::Vector2f cf = area_centroid(from), ct = area_centroid(to);
    const float base = dominant_angle(to) - dominant_angle(from);
    for (int k = 0; k < 4; ++k)
    {
        float th = base + static_cast<float>(k) * 0.5f * static_cast<float>(M_PI);
        Eigen::Matrix2f R = Eigen::Rotation2Df(th).toRotationMatrix();
        Eigen::Vector2f t = ct - R * cf;
        for (int it = 0; it < 25; ++it)   // rigid ICP, point-to-edge correspondences
        {
            // ROBUST (Cauchy, scale = kCauchy): outline pieces with no counterpart (a corner one run saw and the other
            // did not) stop pulling — plain least squares put the room 26 cm off against ground truth.
            constexpr float kCauchy = 0.10f;   // m: the outline's own placement noise (wall-band snap scale)
            Eigen::Vector2f num = Eigen::Vector2f::Zero(); float den = 0.f;
            for (const auto& smp : samples)
            {
                const Eigen::Vector2f p = R * smp + t, q = nearest_on(to, p);
                const float r2 = (q - p).squaredNorm() / (kCauchy * kCauchy);
                const float w = 1.f / (1.f + r2);
                num += w * (q - p); den += w;
            }
            if (den > 0.f) t += num / den;
        }
        Poly moved; moved.reserve(from.size());
        for (const auto& v : from) moved.push_back(R * v + t);
        if (const float s = iou(moved, to); s > best.iou)
            best = {R, t, s, std::atan2(R(1, 0), R(0, 0))};
    }
    return best;
}
}   // namespace rc::mission_frame
