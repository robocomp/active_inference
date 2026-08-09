/*
 * cabinet_lshape_split.h — split a multi-corner 'cabinet' mask into its N perpendicular room-axis arms
 * (header-only, pure).
 *
 * A cabinet RUN is a straight box; it cannot bend. So when ONE YOLO-sem 'cabinet' mask wraps a kitchen
 * that turns one or more corners (an L = 2 arms, a U = 3 arms, a galley = 2 parallel arms), it MUST be
 * several runs. This splits such a mask's support points into its room-axis arms ONCE, geometrically —
 * the worker then replaces the one slice with N sub-slices, and each run is born/fitted from a CLEAN
 * single-arm cloud by the ordinary single-run machinery (no per-frame, per-instance corner arbitration,
 * which is where the observe-time split kept failing).
 *
 * Method (Manhattan, greedy line-peeling). Kitchen runs are axis-aligned, so every arm is either a
 * Y-RUN (a run along +y: a thin, dense band in X spanning a Y-range → a tall peak in the X-histogram)
 * or an X-RUN (a run along +x: a thin band in Y → a tall peak in the Y-histogram). We DISCOVER the arm
 * lines by peeling: repeatedly take the single densest X- or Y-bin among the not-yet-claimed points as
 * the next arm's line, claim the points within arm_halfwidth of it, and continue until the residue is
 * below min_arm_pts. A straight run yields exactly one line (residue then vanishes) ⇒ not split. Once
 * the lines are known, EVERY point is (re)assigned to its nearest arm line — so corner points shared by
 * two arms land on the right one regardless of peel order. ≥2 lines ⇒ the mask is split.
 *
 * Pure geometry, no DSR. Unit-tested by lshape_self_test().
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace rc {

struct LShapeParams
{
    float bin_m           = 0.15f;   // histogram bin for locating each arm's line (m)
    int   min_arm_pts     = 500;     // an arm (and the residue that keeps peeling) must exceed this
    float arm_halfwidth_m = 0.45f;   // half-width of an arm's footprint band (≈ half a carcass depth), m
};

// One room-axis arm. axis: 0 = Y-run (line x = offset, runs along +y); 1 = X-run (line y = offset).
struct MaskArm
{
    std::vector<int> idx;                        // absolute indices into the support-points vector
    Eigen::Vector2f  c = Eigen::Vector2f::Zero(); // arm centroid (room frame)
    int              axis   = 0;
    float            offset = 0.0f;              // the arm line's coordinate (x for a Y-run, y for an X-run)
};

struct LShapeResult
{
    bool                 is_split = false;       // ≥2 arms found (the mask bends / has parallel runs)
    std::vector<MaskArm> arms;
};

// Split support_points[begin,end) into its N perpendicular room-axis arms (see file header).
inline LShapeResult lshape_split(const std::vector<Eigen::Vector3f>& pts,
                                 std::size_t begin, std::size_t end, const LShapeParams& p)
{
    LShapeResult out;
    if (end <= begin or end > pts.size()) return out;
    const float inv  = 1.0f / std::max(1e-3f, p.bin_m);
    const float half = std::max(p.bin_m, p.arm_halfwidth_m);
    const auto  min_arm = static_cast<std::size_t>(std::max(1, p.min_arm_pts));

    // ── 1. DISCOVER arm lines by greedy peeling of the densest axis-aligned bin. ──────────────────
    struct Line { int axis; float offset; };
    std::vector<Line> lines;
    std::vector<int>  remaining;
    remaining.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) remaining.push_back(static_cast<int>(i));

    while (remaining.size() >= min_arm)
    {
        std::unordered_map<int, int> hx, hy;
        for (const int i : remaining)
        {
            ++hx[static_cast<int>(std::floor(pts[i].x() * inv))];
            ++hy[static_cast<int>(std::floor(pts[i].y() * inv))];
        }
        const auto peak = [](const std::unordered_map<int, int>& h)
        {
            int best_bin = 0, best_cnt = -1;
            for (const auto& [b, c] : h) if (c > best_cnt) { best_cnt = c; best_bin = b; }
            return std::pair<int, int>{best_bin, best_cnt};
        };
        const auto [xbin, xcnt] = peak(hx);
        const auto [ybin, ycnt] = peak(hy);
        const bool y_run = xcnt >= ycnt;                                  // densest in X ⇒ a run along +y
        const int   axis = y_run ? 0 : 1;
        const float off  = (static_cast<float>(y_run ? xbin : ybin) + 0.5f) * p.bin_m;

        // Claim the points whose perpendicular coordinate is within half a footprint of this line.
        std::vector<int> keep, band;
        keep.reserve(remaining.size());
        for (const int i : remaining)
        {
            const float perp = y_run ? pts[i].x() : pts[i].y();
            if (std::abs(perp - off) <= half) band.push_back(i);
            else                              keep.push_back(i);
        }
        if (band.size() < min_arm) break;    // the strongest remaining line is not itself a full arm ⇒ stop
        lines.push_back({axis, off});
        remaining = std::move(keep);
    }

    if (lines.size() < 2) return out;        // a straight run (one line) or nothing ⇒ not a multi-arm mask

    // ── 2. FINAL assignment: every point → its nearest arm line (perpendicular distance). ──────────
    out.arms.resize(lines.size());
    for (std::size_t k = 0; k < lines.size(); ++k)
    { out.arms[k].axis = lines[k].axis; out.arms[k].offset = lines[k].offset; }
    std::vector<Eigen::Vector2f> sum(lines.size(), Eigen::Vector2f::Zero());
    for (std::size_t i = begin; i < end; ++i)
    {
        int   best = 0;
        float bestd = std::numeric_limits<float>::max();
        for (std::size_t k = 0; k < lines.size(); ++k)
        {
            const float perp = lines[k].axis == 0 ? pts[i].x() : pts[i].y();
            const float dd   = std::abs(perp - lines[k].offset);
            if (dd < bestd) { bestd = dd; best = static_cast<int>(k); }
        }
        out.arms[best].idx.push_back(static_cast<int>(i));
        sum[best] += Eigen::Vector2f(pts[i].x(), pts[i].y());
    }
    for (std::size_t k = 0; k < out.arms.size(); ++k)
        if (not out.arms[k].idx.empty())
            out.arms[k].c = sum[k] / static_cast<float>(out.arms[k].idx.size());

    // Drop arms that lost all their mass to a neighbour in the final assignment (keep ≥2 to stay split).
    std::erase_if(out.arms, [&](const MaskArm& a) { return a.idx.size() < min_arm; });
    out.is_split = out.arms.size() >= 2;
    return out;
}

inline bool lshape_self_test()
{
    bool ok = true;
    const auto check = [&](bool c, const char* w) { if (not c) { std::printf("[lshape::self_test] FAIL: %s\n", w); ok = false; } };
    LShapeParams pp; pp.min_arm_pts = 100; pp.arm_halfwidth_m = 0.30f;

    const auto band_x = [](std::vector<Eigen::Vector3f>& v, float x, float y0, float y1, int n)
    { for (int i = 0; i <= n; ++i) { const float y = y0 + (y1 - y0) * i / n;
        for (int w = -1; w <= 1; ++w) v.emplace_back(x + 0.08f * w, y, 0.4f); } };     // Y-run at x
    const auto band_y = [](std::vector<Eigen::Vector3f>& v, float y, float x0, float x1, int n)
    { for (int i = 0; i <= n; ++i) { const float x = x0 + (x1 - x0) * i / n;
        for (int w = -1; w <= 1; ++w) v.emplace_back(x, y + 0.08f * w, 0.4f); } };     // X-run at y

    // L: a long vertical arm at x≈-8 (y 4.4–6.5) + a horizontal arm at y≈6.3 (x -7.7…-6.6).
    { std::vector<Eigen::Vector3f> L; band_x(L, -8.0f, 4.4f, 6.5f, 200); band_y(L, 6.3f, -7.7f, -6.6f, 90);
      const auto r = lshape_split(L, 0, L.size(), pp);
      check(r.is_split and r.arms.size() == 2, "L-shaped mask → 2 arms");
    }
    // U: two parallel Y-runs (x≈-8 and x≈-6.25) joined by an X-run at y≈6.66 (a galley/U kitchen).
    { std::vector<Eigen::Vector3f> U; band_x(U, -8.0f, 4.4f, 6.6f, 200); band_x(U, -6.25f, 4.9f, 6.6f, 170);
      band_y(U, 6.66f, -7.9f, -6.5f, 120);
      const auto r = lshape_split(U, 0, U.size(), pp);
      check(r.is_split and r.arms.size() == 3, "U-shaped mask → 3 arms");
      // exactly two Y-runs (axis 0) and one X-run (axis 1)
      int ny = 0, nx = 0; for (const auto& a : r.arms) (a.axis == 0 ? ny : nx)++;
      check(ny == 2 and nx == 1, "U → two Y-runs + one X-run");
    }
    // Single straight vertical run ⇒ NOT split (one line, residue vanishes).
    { std::vector<Eigen::Vector3f> single; band_x(single, -8.0f, 4.0f, 7.0f, 200);
      check(not lshape_split(single, 0, single.size(), pp).is_split, "a straight run is NOT split");
    }

    if (ok) std::printf("[lshape::self_test] all checks passed\n");
    return ok;
}

}  // namespace rc
