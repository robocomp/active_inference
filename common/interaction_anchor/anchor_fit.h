#pragma once
/*
 * anchor_fit.h — the controller's door-fit model (P(body fits), the clear gap a swung leaf leaves), moved VERBATIM
 * from controller/src/controller_anchor_servo.h so chat_agent's `imagine` answers with the SAME model.
 * Header-only, pure std; no controller / DSR / Qt dependency.
 */

#include <algorithm>
#include <cmath>
#include <span>

#include "anchor_leaf.h"

namespace rc
{

// Φ, the standard normal CDF.
[[nodiscard]] inline float anchor_std_normal_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

// P(the body fits through the opening at the current lateral offset). σ_lat = 0 (a certain opening) is the limit
// of the expression: a step at slack 0.
[[nodiscard]] inline float anchor_p_fit(float width, float body_half_width, float e, float sigma_lat)
{
    const float slack = 0.5f * width - body_half_width - std::abs(e);
    if (sigma_lat <= 0.f) return slack > 0.f ? 1.f : (slack < 0.f ? 0.f : 0.5f);
    return anchor_std_normal_cdf(slack / sigma_lat);
}

// ── THE CLEAR GAP THE SWUNG LEAF LEAVES (Task 12 fix A) ──────────────────────────────────────────────────────────────
// The leaf hangs from one jamb; its centre line's footprint ALONG THE WALL, grown by its half-thickness, is
// [min(H.x, T.x) − t, max(H.x, T.x) + t] (anchor_leaf.h; anchor frame). Whatever side it swings to, the body passes through
// both half-spaces of the doorway, so the passage the robot can use is the part of the opening [−w/2, +w/2] that footprint
// does not cover: the interval beside the leaf, up to the far jamb. A leaf at 90° takes its thickness off the hinge side;
// an ajar leaf takes its projection L·cos φ; one swung past 90° folds back over the wall and takes only its thickness.
// The CROSS LINE is that interval's centre and P(body fits) its width — the same geometry, no new level. No leaf ⇒ the
// opening itself (centre 0, width w).
struct AnchorGap { float centre = 0.f, width = 0.f; };
[[nodiscard]] inline AnchorGap anchor_clear_gap(float width, const rc::interaction::AnchorLeaf* leaf)
{
    const float half = 0.5f * width;
    if (leaf == nullptr or not (width > 0.f)) return {0.f, std::max(0.f, width)};
    const float lo_leaf = std::min(leaf->hinge.x(), leaf->tip.x()) - leaf->half_thick;
    const float hi_leaf = std::max(leaf->hinge.x(), leaf->tip.x()) + leaf->half_thick;
    float lo = -half, hi = half;
    if (leaf->hinge.x() < 0.f) lo = std::clamp(hi_leaf, -half, half);   // hung from the −x jamb: the gap is to its +x
    else                       hi = std::clamp(lo_leaf, -half, half);   // hung from the +x jamb: the gap is to its −x
    return {0.5f * (lo + hi), std::max(0.f, hi - lo)};
}

[[nodiscard]] inline float anchor_p_fit_leaves(float width, std::span<const rc::interaction::AnchorLeaf> leaves,
                                               float body_half_width, float e, float sigma_lat)
{
    if (leaves.empty()) return anchor_p_fit(width, body_half_width, e, sigma_lat);
    float p = 0.f;
    for (const auto& l : leaves)
    {
        const auto gap = anchor_clear_gap(width, &l);
        p += l.p * anchor_p_fit(gap.width, body_half_width, e - gap.centre, sigma_lat);
    }
    return std::clamp(p, 0.f, 1.f);
}

} // namespace rc
