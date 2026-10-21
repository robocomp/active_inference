// door_view_pool.h — the image-geometry existence evidence (leaf contour + depth + doorway frame), POOLED BY VIEW NOVELTY
// (Task 5b, D4). Header-only, Eigen only: no DSR, no OpenCV.
//
// ★WHAT WAS WRONG. rc::exist::contour_evidence turns ONE cycle's verdict v (the summed contour/depth/frame excess) into
// llr·tanh(v), llr = log(pd/pc) = one confident observation, and the door integrated that EVERY cycle. A camera at 10 Hz
// that has not moved re-measures the same image statistic of the same view: those frames are one observation repeated,
// and their noise averages — it does not add. On the recorded tours the verdict saturated (|ΔL| ≈ 2.83) on 95 % of cycles
// and its sign flipped with the frame noise, so a correct, shut DOOR_0 random-walked between the +4 clamp and the −4 floor
// in a few frames (tourR1 door_4: 45 cycles at the floor; tourR2 door_2 too), and whether it survived depended on the
// noise sequence — any perturbation of the fit upstream (the recess prior's σ) decided its life.
//
// ★THE MODEL. The frames of one view are one observation of that view; its verdict is their mean; a new view is a new,
// independent observation. Between the two, how much a frame is "the same view" is the STATIONARITY s ∈ [0, 1] of the
// observer relative to the view being pooled — exactly ExistenceBelief::note_viewpoint's: a Gaussian in the bearing change
// measured in units of the door's own angular half-size atan(R/d), and the range change in units of R. That scale is the
// displacement at which a different aspect of the object comes into view, so it adds no tuned constant. Per frame:
//     n ← s·n + 1,   m ← (s·n_old·m + v)/n,   view ← the n-weighted centroid of the pooled viewpoints
//     what the view says:  c = llr·tanh(m)
//     ΔL = c − s·counted                 (less what the view already contributed, to the extent it IS the same view)
//     counted ← s·counted + (the change ΔL actually made to L)
// ★`counted` is what the view REALLY moved L, not what it asked for (commit()): L is clamped at ±L_max, so a view that
// said +2.8 while L sat at the +4 clamp moved nothing, and taking back +2.8 it never added would turn the view's
// revision into a −5.6 swing. Measured: tourR1 door_4 went +4 → −0.28 in one frame that way before this.
// s = 1 throughout ⇒ the frames telescope to ONE observation at the mean verdict; s = 0 ⇒ every frame is new and counts
// in full (the old behaviour). Symmetric: confirmation and refutation are pooled alike, so it is not a ratchet.
#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>

namespace rc::door
{

struct ViewPool
{
    bool  have = false;
    float bearing = 0.0f, range = 0.0f;   // the pooled view: n-weighted centroid of the viewpoints (around the door)
    float n = 0.0f;                       // effective number of frames in the pooled view
    float mean = 0.0f;                    // their mean verdict
    float counted = 0.0f;                 // log-odds this view has actually moved L so far (commit)
    float last_s = 0.0f;                  // the last frame's stationarity (diagnostic)
    float pending = 0.0f;                 // the last add()'s ΔL (commit corrects `counted` by what was realised)

    // One cycle's verdict `v` seen from `obs` of a door centred at `tgt` with half-width `R`; `llr` = the log-odds of one
    // confident observation. Returns the log-odds to integrate this cycle. A non-finite verdict or geometry ⇒ 0, held.
    float add(const Eigen::Vector2f& obs, const Eigen::Vector2f& tgt, float R, float v, float llr)
    {
        const Eigen::Vector2f dv = obs - tgt;   // the viewpoint, seen from the door
        const float d = dv.norm();
        if (not std::isfinite(v) or not std::isfinite(d) or not std::isfinite(llr) or d < 1e-3f)
            return 0.0f;
        const float b  = std::atan2(dv.y(), dv.x());
        const float Rr = std::max(R, 1e-3f);
        float s = 0.0f;
        if (have)
        {
            const float half = std::max(std::atan2(Rr, d), 1e-3f);   // the door's angular half-size from here
            const float db = std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>);
            const float dr = d - range;
            const float u2 = (db / half) * (db / half) + (dr / Rr) * (dr / Rr);
            s = std::exp(-std::min(u2, 60.0f));
        }
        last_s = s;
        const float n_old = s * n;
        n = n_old + 1.0f;
        mean = (n_old * mean + v) / n;
        if (have)
        {
            bearing += std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>) / n;
            range   += (d - range) / n;
        }
        else
        {
            bearing = b; range = d; have = true;
        }
        const float c = llr * std::tanh(mean);
        pending = c - s * counted;        // less the part of the old contribution that is this same view
        counted = c;                      // assumed realised in full until commit() says otherwise
        return pending;
    }
    // The same pooling for a channel whose per-look evidence is ALREADY a log-odds increment (the silhouette: P(detect)·the
    // mask's log-likelihood ratio — door_existence.cpp): the view's contribution is the MEAN increment of its looks, one
    // observation's worth, instead of llr·tanh(mean verdict). Task 5b item 4, fix round I-2: both existence channels pool
    // by the SAME view model at the SAME scale, so neither can out-run the other on a gradual approach (a silhouette
    // damped by a growing run length while the image channel renewed at (1 − s) per frame walked tourR1's DOOR_1, read
    // occupied 150/30 by the silhouette, to the floor).
    float add_logodds(const Eigen::Vector2f& obs, const Eigen::Vector2f& tgt, float R, float dL_look)
    {
        const Eigen::Vector2f dv = obs - tgt;
        const float d = dv.norm();
        if (not std::isfinite(dL_look) or not std::isfinite(d) or d < 1e-3f)
            return 0.0f;
        const float b  = std::atan2(dv.y(), dv.x());
        const float Rr = std::max(R, 1e-3f);
        float s = 0.0f;
        if (have)
        {
            const float half = std::max(std::atan2(Rr, d), 1e-3f);
            const float db = std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>);
            const float dr = d - range;
            s = std::exp(-std::min((db / half) * (db / half) + (dr / Rr) * (dr / Rr), 60.0f));
        }
        last_s = s;
        const float n_old = s * n;
        n = n_old + 1.0f;
        mean = (n_old * mean + dL_look) / n;
        if (have)
        {
            bearing += std::remainder(b - bearing, 2.0f * std::numbers::pi_v<float>) / n;
            range   += (d - range) / n;
        }
        else
        {
            bearing = b; range = d; have = true;
        }
        pending = mean - s * counted;
        counted = mean;
        return pending;
    }

    // After integrating add()'s ΔL: what it actually changed L by (after the clamp). Without a call, the full ΔL is assumed.
    void commit(float realized_dL) { counted += realized_dL - pending; pending = realized_dL; }
};

}  // namespace rc::door
