/*
 * detectability.h — the detector's INVERSE model: P(detect | geometry). Shared, header-only.
 *
 * One question, two consumers that were answering it separately and inconsistently:
 *
 *   "Given where the robot is standing, could this detector have seen this object at all?"
 *
 *   · REMOVE — absence is evidence of removal only in proportion to P(detect). Without this term, "no mask"
 *     from a place the detector could never have fired reads as "the object is gone".
 *   · CREATE / NBV — the best viewpoint is the one that MAXIMISES P(detect). The stand-off then stops being a
 *     hand-picked framing constant and becomes the argmax of this model.
 *
 * Deriving both from ONE model is the point. They were previously two half-models that disagreed: removal had
 * only a FAR term ((ref/range)^p) and the planner had only a framing constant, so nothing in the system knew
 * that standing TOO CLOSE also destroys detectability — and a robot driven nose-to-nose with a fridge got no
 * mask, which the removal channel scored as absence and deleted it. The near half was missing on both sides.
 *
 * THE PHYSICS. A segmentation network needs the object to be
 *   · big enough to segment  — below a few percent of the frame there are too few pixels for a mask, and
 *   · small enough to FIT    — as the projection approaches the frame the mask truncates at the border and the
 *                              network loses the whole-object context it classifies on.
 * So P(detect) is UNIMODAL in the projected fill fraction, not monotonic in range. Two logistics, no cut-off:
 *
 *     p(fill) = σ((fill − min_fill)/soft) · σ((max_fill − fill)/soft)
 *
 * times the fraction of the object the frustum and other objects actually leave visible.
 *
 * `fill` is the SAME quantity the agents already publish as roi_fill: max(w_px/W, h_px/H), the projected extent
 * as a fraction of the image. An object of circumscribed radius R at distance d, in a camera of horizontal FoV
 * `hfov`, subtends fill = 2·atan(R/d)/hfov — which inverts to the stand-off below.
 *
 * ★CALIBRATION. min_fill/max_fill are properties of the DETECTOR + camera, not tuning knobs, and they are
 * measurable: log (roi_fill, mask-present) pairs over a tour and fit the two logistics to the empirical
 * detection rate. The defaults below are a conservative envelope, not a measurement, and should be replaced by
 * one. Until then they are honest about being a prior on the detector rather than a fact about it.
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace rc::detect
{

// The detector's operating envelope in projected-fill units. Physical properties of detector+camera.
struct DetectorEnvelope
{
    float min_fill = 0.10f;   // below this the object is too few pixels to segment reliably
    float max_fill = 0.60f;   // above this it crowds/overflows the frame: mask truncates, context is lost
    float soft     = 0.06f;   // roll-off width of both shoulders (continuous; nothing is ever hard-cut)
    bool  valid() const { return max_fill > min_fill and soft > 1e-4f; }
};

// P(detect | present, geometry) ∈ [0,1], from the projected bbox's TWO axes.
//   fill_max / fill_min — the larger / smaller of (Δcol/W, Δrow/H). roi_fill is the max of the pair.
//   visible_frac        — fraction of the object the frustum + occluders leave visible (in_fov_frac)
//
// ★THE TWO SHOULDERS ACT ON DIFFERENT AXES. Collapsing them onto the max alone is a modelling ERROR, not a
// simplification, because the physics says which axis limits which failure:
//
//   · "too few pixels to segment" is limited by the SMALLER axis. A door seen edge-on is a 5 cm-wide sliver:
//     it is 2 m TALL, so its max-fill looks perfectly framed, but there is nothing across it to segment. No
//     detector fires on that, and no amount of height rescues it.
//   · "overflows the frame, mask truncates, context lost" is limited by the LARGER axis — that is the one
//     that runs off the edge first.
//
// With the max alone the two are conflated and an edge-on view is indistinguishable from a face-on one.
// MEASURED on the live 0.70 x 0.05 x 2.00 m door leaf: all four faces scored p_detect 0.970, including the
// two EDGE faces, so the NBV sent the robot along the wall — a place the door cannot be seen from at all.
// Splitting the axes drops those faces to ~0 and leaves the panel faces untouched.
//
// This adds NO new parameter: the same min_fill/max_fill/soft, applied to the axis each one actually governs.
// For a compact object fill_min ≈ fill_max and it reduces EXACTLY to the previous model, so it is a strict
// generalisation — and it stays fittable, since the agents log both axes precisely for this.
inline float p_detect(float fill_max, float fill_min, float visible_frac, const DetectorEnvelope& e)
{
    if (not e.valid())
        return std::clamp(visible_frac, 0.0f, 1.0f);
    const auto sig = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const float p_size = sig((fill_min - e.min_fill) / e.soft);   // enough pixels ACROSS the short axis
    const float p_fit  = sig((e.max_fill - fill_max) / e.soft);   // the long axis still fits, with context
    return p_size * p_fit * std::clamp(visible_frac, 0.0f, 1.0f);
}

// Single-axis overload for a caller that has only roi_fill (the max) — chiefly the removal channels, which
// read a measured roi_fill off the instance.
//
// ★OPTIMISTIC for slivers by construction: it assumes the object is compact, so an edge-on or grazing view
// scores as if it were face-on. Prefer the two-axis form wherever both axes are available.
//
// ★★AND THE OPTIMISM IS THE DANGEROUS DIRECTION ON THE REMOVAL SIDE — not the safe one. In the existence
// channels p_detect interpolates the per-cycle log-odds from confirm-only toward FULL absence:
//     log_odds_delta = d_conf + p_detect · (d_full − d_conf)
// so p_detect → 1 admits the whole absence swing and p_detect → 0 is a pure HOLD (table_existence.cpp says
// exactly this). A too-HIGH p_detect therefore pushes HARDER toward deleting the object. This is the shape
// behind the documented over-removals: a 7 m through-a-wall deletion at p_detect 0.126, and a table removed
// from a 2 %-resolvable view. So on a sliver view this overload does not merely lose accuracy — it charges
// an absence that the geometry could never have resolved.
//
// It is kept because switching the removal channels to the pair is coupled to the UNCALIBRATED envelope:
// with min_fill 0.10 now tested against the SHORT axis, a tall box at a good vertical framing sits near or
// below that shoulder, so the pair form roughly halves p_detect at every genuinely good viewpoint. Halving
// removal sensitivity fleet-wide would trade a documented over-removal bug for a documented under-removal one
// (immortal phantoms) on the strength of a prior we already know is wrong. CALIBRATE FIRST, then switch the
// removal channels to the two-axis form together.
inline float p_detect(float fill, float visible_frac, const DetectorEnvelope& e)
{
    return p_detect(fill, fill, visible_frac, e);
}

// The fill at which P(detect) peaks — the framing the NBV should aim for. Found by a coarse scan rather than
// asserted as (min+max)/2, so it stays correct if the shoulders are ever given different widths.
inline float best_fill(const DetectorEnvelope& e)
{
    if (not e.valid())
        return 0.35f;
    float best = e.min_fill, best_p = -1.0f;
    for (int i = 0; i <= 200; ++i)
    {
        const float f = static_cast<float>(i) / 200.0f;
        const float p = p_detect(f, 1.0f, e);
        if (p > best_p) { best_p = p; best = f; }
    }
    return best;
}


// The usable stand-off BAND: the range over which P(detect) stays above `frac` of its peak. The near end is
// what stops a controller closing in until the object overflows; the far end is where the mask gets too small.
inline void standoff_band(float R_circumscribed_m, float hfov_rad, const DetectorEnvelope& e,
                          float frac, float& near_m, float& far_m)
{
    const float peak = p_detect(best_fill(e), 1.0f, e);
    const float want = std::clamp(frac, 0.0f, 0.99f) * peak;
    const auto d_of = [&](float f)   // R / (fill * tan(hfov/2)) — the pinhole inversion
    { return R_circumscribed_m / (std::max(1e-3f, f) * std::tan(0.5f * hfov_rad)); };
    float lo = best_fill(e), hi = best_fill(e);
    for (int i = 0; i <= 200; ++i)   // widen outward while the envelope still clears `want`
    {
        const float f = static_cast<float>(i) / 200.0f;
        if (p_detect(f, 1.0f, e) >= want) { lo = std::min(lo, f); hi = std::max(hi, f); }
    }
    near_m = d_of(hi);   // the LARGEST fill is the CLOSEST distance
    far_m  = d_of(lo);
}

}  // namespace rc::detect
