/*
 * footprint.h — oriented-rectangle footprint geometry. SHARED, header-only.
 *
 * Extracted 2026-08-13 from FIVE BYTE-IDENTICAL COPIES (hood · cabinet · table · refrigerator · door),
 * measured at 100.0% pairwise identical lines with the object's name normalised away. Not "similar" —
 * identical, character for character, in five files.
 *
 * ★WHY THIS ONE MATTERS MORE THAN ITS SIZE SUGGESTS. It answers "are these two instances the same physical
 * object?" for merge_overlapping_instances — a LIFECYCLE DECISION, and the same profile decide_removal had
 * before it drifted three ways: small, copied, and consulted at a moment nobody watches. Five copies that
 * agree today is not a state, it is a coincidence with a date on it.
 *
 * ⚠AND ONE COPY HAS ALREADY DIVERGED. residual_concept computes the same quantity by SAMPLING a 6x6 grid of
 * the smaller rectangle and counting points inside the larger — 8.3% similar to these five, and a different
 * answer: a coarse Monte-Carlo estimate against an exact polygon clip, with quantisation of 1/36 and a bias
 * that depends on which rectangle is smaller. It is not an object-concept agent so it is out of scope here,
 * but it is the proof that this function does drift when left copied.
 *
 * THE SEAM, the same one support_bank uses: the agent hands over its footprint as five numbers. Each object
 * names its extents differently — a box has (w, h), a run has (L, d), a bottle has a radius — and that
 * mapping is the only per-object part. Everything after it is plane geometry.
 *
 * Pure: Eigen + the standard library. No DSR, no state type, no config.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

namespace rc::geom
{

// An oriented rectangle in the room plane. The agent builds one from its own state — a box from (w,h), a
// cabinet run from (L,d), a cylinder from (2r,2r) — and that adapter is the whole per-object surface.
struct Footprint
{
    float cx = 0.0f, cy = 0.0f;   // centre (room frame, m)
    float w  = 0.0f, h  = 0.0f;   // extents along the local +x and +y axes (m)
    float yaw = 0.0f;             // rotation of the local frame (rad)
};

// Corners CCW, local order (-,-), (+,-), (+,+), (-,+).
inline std::array<Eigen::Vector2f, 4> corners(const Footprint& s)
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

// Sutherland–Hodgman: clip the subject polygon against the convex CCW clip rectangle. EXACT — which is the
// point of having one copy: the alternative that grew in residual_concept samples a 6x6 grid and quantises
// the answer to 1/36, with a bias depending on which rectangle it picked as the smaller.
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

// Overlap area as a fraction of the SMALLER footprint (1.0 = one fully inside the other). This is the number
// merge_overlapping_instances tests against Tracker.MergeOverlap.
inline float overlap_ratio(const Footprint& a, const Footprint& b)
{
    const auto ca = corners(a), cb = corners(b);
    const auto inter = clip_poly(std::vector<Eigen::Vector2f>(ca.begin(), ca.end()), cb);
    const float ai = poly_area(inter);
    const float amin = std::min(poly_area({ca.begin(), ca.end()}), poly_area({cb.begin(), cb.end()}));
    return amin > 1e-6f ? ai / amin : 0.0f;
}

// A CIRCULAR footprint in the room plane. Not every concept is a rectangle: a bottle is a cylinder, and
// approximating it by its bounding square answers a different question — a square overestimates the area by
// 4/pi and, worse, makes the overlap depend on a yaw a cylinder does not have.
struct Circle
{
    float cx = 0.0f, cy = 0.0f;
    float r  = 0.0f;
};

// Overlap area as a fraction of the SMALLER circle (1.0 = one fully inside the other) — the same number
// overlap_ratio(Footprint, Footprint) reports, for the same purpose (collapsing two instances fitted to one
// physical object). EXACT: the classic two-circle lens, not a sampled or bounding-box estimate.
inline float overlap_ratio(const Circle& a, const Circle& b)
{
    const float ra = std::max(a.r, 1e-4f), rb = std::max(b.r, 1e-4f);
    const float d  = std::hypot(a.cx - b.cx, a.cy - b.cy);
    if (d >= ra + rb)           return 0.0f;   // disjoint
    if (d <= std::abs(ra - rb)) return 1.0f;   // smaller fully inside the larger
    const float d1 = (d * d + ra * ra - rb * rb) / (2.0f * d);   // a-centre → radical line
    const float d2 = d - d1;
    const float seg_a = ra * ra * std::acos(std::clamp(d1 / ra, -1.0f, 1.0f))
                        - d1 * std::sqrt(std::max(0.0f, ra * ra - d1 * d1));
    const float seg_b = rb * rb * std::acos(std::clamp(d2 / rb, -1.0f, 1.0f))
                        - d2 * std::sqrt(std::max(0.0f, rb * rb - d2 * d2));
    const float amin  = std::numbers::pi_v<float> * std::min(ra, rb) * std::min(ra, rb);
    return amin > 1e-9f ? std::clamp((seg_a + seg_b) / amin, 0.0f, 1.0f) : 0.0f;
}

}  // namespace rc::geom
