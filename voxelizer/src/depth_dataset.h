#pragma once

/*
 * depth_dataset.h
 *
 * Dataset + fitted correction map for the ricoh monocular-depth branch.
 *
 * WHY THIS EXISTS. yolo26l-depth predicts the room's STRUCTURE well — measured against helios LiDAR
 * over the same panorama pixels, Pearson r = +0.95 (etc/ricoh_depth_lidar.csv). What it does not
 * predict is scale, in two separate ways:
 *   1. Every strip is scale-and-shift invariant on its own, so the six views disagree by orders of
 *      magnitude (median predicted depth ranged 2.35 m to 685 m against a LiDAR truth of 1.1-3.9 m).
 *   2. Its log-depth is COMPRESSED: fitting log_range = a*log_model + b gives a ~= 0.22 consistently
 *      across all views and both projections, i.e. the model's log range spans ~4.5x reality. The
 *      reverse-regression bracket [a, a/r^2] is [0.22, 0.24], so that is a real property of the
 *      model, not regression dilution from a noisy predictor.
 * Both are corrected by a fitted map rather than by retraining — the structure is already right.
 *
 * ★WHY THE MAP NEEDS A REAL DATASET, not a per-cycle fit. helios is a near-horizontal scanner at
 * ~1.1 m, so its returns land in a thin stripe near the equator: widening the estimated band from
 * ±37.5° to ±60° added essentially ZERO samples (per-view n stayed ~2900-3700). And one standing
 * pose spans only 1.13-4.28 m of range = 0.58 decades. A map fitted from that is anchored on a
 * narrow horizontal stripe at short range and then extrapolated over the whole frame and out past
 * 10 m. Collecting across poses and rooms is what makes the fit mean anything, which is why
 * collection is a deliberate act (a button) rather than something that happens silently.
 */

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rc::depth
{

// One LiDAR-anchored observation: what the model said at a pixel, and what the LiDAR measured there.
struct DepthSample
{
    float         log_model = 0.f;   // model output at (u,v) — log depth, per-view arbitrary scale
    float         log_range = 0.f;   // ln(helios range) at the same pixel — the truth
    float         s         = 0.f;   // horizontal position within the view, normalised to [-1,1]
    float         t         = 0.f;   // vertical position in the panorama, normalised to [-1,1]
    std::uint8_t  view      = 0;
};

// One capture: all samples from a single panorama, plus the pose that produced it. The pose is what
// makes de-duplication possible — two captures from the same spot are one observation of the room,
// however many frames apart they were taken.
struct DepthFrame
{
    std::uint64_t            stamp_ms = 0;
    float                    rx = 0.f, ry = 0.f, rtheta = 0.f;   // robot pose in the room
    std::vector<DepthSample> samples;
};

inline constexpr int kMaxViews = 16;

// The correction: log_range ~= a*log_model + b[view] + cs*s^2 + ct*t^2.
//
// ONE SHARED EXPONENT `a`, a PER-VIEW OFFSET `b`. That split is not a guess — `a` measured 0.18-0.26
// across every view and both projections while `b` swung -0.43..+0.37, so the compression is a
// property of the MODEL and the offset is a property of the VIEW. Fitting `a` per view instead would
// spend six parameters on one number and let each view's thin sample stripe pull it around.
//
// The two quadratic terms absorb the smooth radial error the overlay shows (warm edges, cool centre)
// without pretending to more spatial detail than a horizon-stripe dataset can support. They are the
// first thing to drop if the fit looks over-parameterised.
inline constexpr int kSplineKnots = 8;

struct DepthFitMap
{
    bool  valid = false;
    float a  = 1.0f;                            // base slope of the piecewise-linear response
    std::array<float, kMaxViews> b{};
    float cs = 0.f;
    float ct = 0.f;                             // DEPRECATED, never applied — see apply_with()
    // ★Piecewise-linear (hinge) response in log_model: y = a*lm + Σ hinge[k]*max(0, lm − knot[k]).
    // The relation is smoothly NONLINEAR in log space, and a straight line leaves ~2 points of error
    // on the table. Held-out by frame: linear 11.3% err / δ1 77.4%, 2 knots 9.6%/84.3%, 8 knots
    // 9.3%/85.0%, 16 knots 9.3%/85.0% — it SATURATES at 8, so that is the size. Knots are stored
    // because apply() cannot recover them from the data.
    std::array<float, kSplineKnots> knot{}, hinge{};
    int   n_knots = 0;
    int   n_views = 0;
    // Provenance — a map is only as good as what it was fitted on, so it travels with its own stats.
    long  n_samples = 0, n_frames = 0;
    float resid_rms = 0.f;      // RMS about the POOLED fit (b held constant per view)
    // ★RMS about the fit as it is ACTUALLY USED — b re-solved per (frame, view), which is what the
    // runtime does every tick. resid_rms is the pessimistic number: it charges the fit for scale drift
    // that the live anchor cancels. Reporting only the pooled one made the map look far worse than the
    // system performs (±40% vs ±25% measured over 406 frames), so both travel together.
    float resid_anchored = 0.f;
    float r         = 0.f;      // correlation of fitted vs measured log_range
    // ★What a person can act on. resid_* are RMS in LOG space and are dominated by the tail; these are
    // the standard monocular-depth measures on the ANCHORED prediction, and they are what the UI shows.
    // Both are honest and they answer different questions — quote med_rel for "how good usually", and
    // look at the tail separately, because the median hides it.
    float med_rel   = 0.f;      // median |d−d*|/d*  — "typical error", the headline number
    float med_abs_m = 0.f;      // the same in metres
    float delta125  = 0.f;      // fraction within 1.25x (δ1 in the depth literature)
    float range_lo = 0.f, range_hi = 0.f;   // the metre span the fit is actually supported over

    [[nodiscard]] float apply(float log_model, int view, float s, float t) const
    {
        const float bb = (view >= 0 and view < kMaxViews) ? b[static_cast<std::size_t>(view)] : 0.f;
        return apply_with(log_model, bb, s, t);
    }
    // Same correction with the offset supplied by the caller. ★THIS is the form that actually works:
    // measured over 406 frames, `a` is stable (0.199 pooled vs 0.205 within-frame) but `b` swings
    // ±0.25 log frame to frame — up to 12x in metres inside ONE view — because the model is
    // scale-invariant PER IMAGE and every frame is a new image. Holding b fixed dumps all of that into
    // the residual (±40%); re-solving it each frame from that frame's LiDAR recovers ±25%. So the
    // DATASET owns a/cs/ct, and the LIDAR owns b, every frame.
    // `t` is accepted but IGNORED. The old ct*t^2 term is gone: the dataset's t spans −0.14..+0.67
    // against a band of ±0.667, so the ENTIRE UPPER HALF (ceiling) had zero LiDAR samples and ct was
    // pure extrapolation there — a −28% correction with nothing constraining it. Dropping it measured
    // slightly BETTER (11.1% vs 11.3%). The parameter stays in the signature so call sites keep their
    // meaning, and so it can be reinstated the moment synthetic ceiling/floor samples give it support.
    [[nodiscard]] float apply_with(float log_model, float b_view, float s, float /*t*/) const
    {
        return b_view + base(log_model, s);
    }
    // Everything except the per-view offset — what the online anchor must average to solve b.
    [[nodiscard]] float base(float log_model, float s) const
    {
        float y = a * log_model + cs * s * s;
        for (int k = 0; k < n_knots and k < kSplineKnots; ++k)
            y += hinge[static_cast<std::size_t>(k)]
               * std::max(0.f, log_model - knot[static_cast<std::size_t>(k)]);
        return y;
    }

    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

// Per-frame, per-view scale offsets solved from this frame's LiDAR with `a` held at its dataset value.
// One division per view over sums the correlation pass already accumulates — the cheapest possible way
// to track a quantity that is re-drawn every frame.
struct DepthAnchor
{
    std::array<float, kMaxViews> b{};
    std::array<long,  kMaxViews> n{};
    std::array<bool,  kMaxViews> ok{};
    [[nodiscard]] bool any() const
    { return std::any_of(ok.begin(), ok.end(), [](bool v) { return v; }); }
};

class DepthDataset
{
public:
    // Append a capture. Returns false if the pose is not NOVEL enough vs the last kept frame — that
    // is the whole point of collecting by hand: a robot standing still would otherwise bury the set
    // under thousands of copies of one viewpoint and the fit would silently describe that spot.
    bool add_frame(DepthFrame&& f, float min_move_m = 0.10f, float min_turn_rad = 0.087f);

    // Append to `path` (kept across runs, so a session ADDS to the set rather than replacing it).
    bool append_csv(const std::string& path);
    bool load_csv(const std::string& path);

    // Drop near-duplicate poses across the WHOLE set, not just consecutively — several passes over
    // the same room leave duplicates that no capture-time check can see. Returns frames removed.
    std::size_t dedup(float min_move_m = 0.10f, float min_turn_rad = 0.087f);

    [[nodiscard]] DepthFitMap fit(int n_views) const;

    [[nodiscard]] std::size_t frame_count()  const { return frames_.size(); }
    [[nodiscard]] std::size_t sample_count() const;
    void clear() { frames_.clear(); pending_ = 0; }

private:
    std::vector<DepthFrame> frames_;
    std::size_t             pending_ = 0;   // frames not yet written by append_csv
};

}   // namespace rc::depth
