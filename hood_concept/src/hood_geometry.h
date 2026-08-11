/*
 * hood_geometry.h — pure footprint-geometry + uncertainty helpers for hood_concept (header-only).
 *
 * Free functions in rc::geom shared by the worker's dashboard, convergence, and instance-merge paths (split
 * across specificworker*.cpp translation units): a scalar belief-uncertainty readout, and oriented-rectangle
 * footprint overlap (corners → Sutherland–Hodgman clip → overlap-area ratio) used to detect two instances
 * fitted to the same physical hood. No state, no DSR — pure math on the belief covariance + the fitted state.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "hood_instance.h"   // rc::HoodInstance (belief covariance)
#include "hood_model.h"      // rc::HoodState

namespace rc::geom {

// Scalar model-uncertainty readout for model_uncertainty_att / the dashboard: the sum of the belief's per-DOF
// posterior stds over position (cx,cy) + size (w,h), in metres, from the AI2 covariance Σ over
// [cx,cy,H,w,h,yaw]. Shrinks as the robot gathers viewpoints. 0 before the belief is seeded.
inline float belief_uncertainty(const rc::HoodInstance& inst)
{
    if (not inst.ai2_initialized)
        return 0.0f;
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    return sd(0) + sd(1) + sd(3) + sd(4);
}

// Two hoods cannot share physical space. The footprint is the oriented rectangle (cx,cy,w,h,yaw) in the room
// plane; these helpers compute the overlap area between two footprints so the merge operator can collapse
// duplicate instances. Corners are returned CCW (local order (-,-),(+,-),(+,+),(-,+)).
inline std::array<Eigen::Vector2f, 4> footprint_corners(const rc::HoodState& s)
{
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const Eigen::Vector2f ex(c, sn), ey(-sn, c), ctr(s.cx, s.cy);
    const float hw = 0.5f * s.w, hh = 0.5f * s.h;
    return { ctr - hw * ex - hh * ey, ctr + hw * ex - hh * ey,
             ctr + hw * ex + hh * ey, ctr - hw * ex + hh * ey };
}

inline float poly_area(const std::vector<Eigen::Vector2f>& p)
{
    if (p.size() < 3) return 0.0f;
    float a = 0.0f;
    for (std::size_t i = 0, n = p.size(); i < n; ++i)
    {
        const auto& u = p[i]; const auto& v = p[(i + 1) % n];
        a += u.x() * v.y() - v.x() * u.y();
    }
    return 0.5f * std::abs(a);
}

// Sutherland–Hodgman: clip the subject polygon against the convex CCW clip rectangle.
inline std::vector<Eigen::Vector2f> clip_poly(std::vector<Eigen::Vector2f> subj,
                                              const std::array<Eigen::Vector2f, 4>& clip)
{
    for (int e = 0; e < 4 and not subj.empty(); ++e)
    {
        const Eigen::Vector2f a = clip[e], b = clip[(e + 1) % 4], d1 = b - a;
        const auto inside = [&](const Eigen::Vector2f& p)
        { return d1.x() * (p.y() - a.y()) - d1.y() * (p.x() - a.x()) >= 0.0f; };
        std::vector<Eigen::Vector2f> out;
        for (std::size_t i = 0, n = subj.size(); i < n; ++i)
        {
            const Eigen::Vector2f cur = subj[i], prv = subj[(i + n - 1) % n];
            const bool ci = inside(cur), pi = inside(prv);
            const auto isect = [&]() -> Eigen::Vector2f
            {
                const Eigen::Vector2f d2 = cur - prv;
                const float den = d2.x() * d1.y() - d2.y() * d1.x();
                const float t = std::abs(den) < 1e-12f ? 0.0f
                    : ((a.x() - prv.x()) * d1.y() - (a.y() - prv.y()) * d1.x()) / den;
                return prv + t * d2;
            };
            if (ci) { if (not pi) out.push_back(isect()); out.push_back(cur); }
            else if (pi) out.push_back(isect());
        }
        subj.swap(out);
    }
    return subj;
}

// Overlap area as a fraction of the SMALLER footprint (1.0 = one hood fully inside the other).
inline float footprint_overlap_ratio(const rc::HoodState& a, const rc::HoodState& b)
{
    const auto ca = footprint_corners(a), cb = footprint_corners(b);
    const auto inter = clip_poly(std::vector<Eigen::Vector2f>(ca.begin(), ca.end()), cb);
    const float ai = poly_area(inter);
    const float amin = std::min(poly_area({ca.begin(), ca.end()}), poly_area({cb.begin(), cb.end()}));
    return amin > 1e-6f ? ai / amin : 0.0f;
}

}  // namespace rc::geom
