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

// Where a sample's `log_range` came from. The fit treats both identically except for PRECISION; the
// quality metrics are computed on MEASURED rows only, so the headline numbers stay comparable across
// a dataset that has been enriched and one that has not.
inline constexpr std::uint8_t kSrcLidar    = 0;   // helios range at that pixel — a MEASUREMENT
inline constexpr std::uint8_t kSrcEnvelope = 1;   // room-belief envelope ray-cast — a BELIEF (see depth_enrichment.h)

// One anchored observation: what the model said at a pixel, and what the anchor says the range is there.
struct DepthSample
{
    float         log_model = 0.f;   // model output at (u,v) — log depth, per-view arbitrary scale
    float         log_range = 0.f;   // ln(range) at the same pixel — the anchor
    float         s         = 0.f;   // horizontal position within the view, normalised to [-1,1]
    float         t         = 0.f;   // vertical position in the panorama, normalised to [-1,1]
    std::uint8_t  view      = 0;
    std::uint8_t  src       = kSrcLidar;
    // ★COMMON-MODE GROUP, frame-local. 0 = this sample's error is independent of every other one.
    // Non-zero ⇒ it shares ONE unknown displacement with every other sample carrying the same id in
    // the SAME frame (all the pixels that landed on one believed wall move together when that wall
    // moves). See `h` and the Woodbury block in fit().
    std::uint16_t region    = 0;
    // ★PRECISION, not a count. w = σ_lidar² / σ_independent², i.e. this row's weight RELATIVE to a
    // LiDAR row, which is 1.0 by definition. The normal equations accumulate w·xxᵀ / w·y·x, so a row
    // the generative model trusts half as much counts half as much. Default 1.0 ⇒ an untouched
    // dataset fits bit-identically to before weights existed.
    float         w         = 1.f;
    // ★COMMON-MODE SD of this row, in the same σ_lidar-relative log units. The region's covariance is
    // D + h·hᵀ with D = diag(1/w): one rank-1 direction shared by the whole region. h is per-SAMPLE
    // (not constant) because the same plane displacement δn produces δlnR = δn/(R·cosθ), which varies
    // over the region with range and incidence. 0 ⇒ no common mode.
    float         h         = 0.f;
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
    // ★ct*t² is BACK, but only when the data earns it. It was dropped because a LiDAR-only dataset has
    // no samples above the horizon (t < 0), so the term was pure extrapolation over the whole ceiling
    // half of the band. The offline enrichment pass (depth_enrichment.h) supplies exactly those rows
    // from the room belief's ceiling, and whether they are ENOUGH is decided by Bayesian model
    // selection at fit time, not by anyone's opinion: ct is kept iff ΔBIC = t_ratio² − ln(N_eff) > 0.
    // `ct_active` is false on every map fitted before this existed and on every LiDAR-only refit, so
    // the runtime behaviour is unchanged unless enrichment actually happened AND paid for itself.
    float ct = 0.f;
    bool  ct_active = false;
    float ct_delta_bic = 0.f;                   // provenance: how decisively the data chose (nats-ish)
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
    long  n_synth   = 0;        // of which SYNTHETIC (room-envelope). 0 ⇒ a pure LiDAR fit.
    // Where the fit is supported in the VERTICAL band coordinate, which is the whole reason the
    // enrichment exists: a LiDAR-only set spans t ≈ −0.14..+0.67 against a band of ±0.667.
    float t_lo = 0.f, t_hi = 0.f;
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
    // `t` is applied ONLY when `ct_active` — see the field. A LiDAR-only dataset spans t ≈ −0.14..+0.67
    // against a band of ±0.667, so the ENTIRE UPPER HALF (ceiling) has no measured samples and ct is
    // pure extrapolation there (a −28% correction with nothing constraining it; dropping it measured
    // slightly BETTER, 11.1% vs 11.3%). Enrichment fills that half from the room belief's ceiling, and
    // fit() then decides by ΔBIC whether the term is identifiable. Unset ⇒ this is exactly the old
    // t-ignoring form, bit for bit.
    [[nodiscard]] float apply_with(float log_model, float b_view, float s, float t) const
    {
        return b_view + base(log_model, s, t);
    }
    // Everything except the per-view offset — what the online anchor must average to solve b.
    [[nodiscard]] float base(float log_model, float s, float t) const
    {
        float y = a * log_model + cs * s * s;
        if (ct_active)
            y += ct * t * t;
        for (int k = 0; k < n_knots and k < kSplineKnots; ++k)
            y += hinge[static_cast<std::size_t>(k)]
               * std::max(0.f, log_model - knot[static_cast<std::size_t>(k)]);
        return y;
    }

    bool save(const std::string& path) const;
    bool load(const std::string& path);
};

// Number of free parameters of the map for `n_views`: a + hinge[n_knots] + cs + [ct] + b[n_views].
[[nodiscard]] int map_n_params(const DepthFitMap& m, int n_views);

// The design row for one sample — the SINGLE definition, used by both the least-squares fit and the
// information selector. They must agree exactly: the selector scores how much a frame would sharpen
// the very parameters the fit solves, so a divergence here would silently make the two disagree about
// what is informative. `x` is resized as needed.
void design_row(const DepthSample& s, const DepthFitMap& m, int n_views, Eigen::VectorXd& x);

// ★EPISTEMIC FRAME SELECTION. Admits a frame by the INFORMATION it adds about the map parameters,
// not by how far the robot moved. Pose novelty is only a proxy: it cannot tell that the 300th view of
// the same wall adds nothing, nor that a long corridor sightline from an already-visited spot is
// exactly what the fit is starving for (the dataset spans ~0.87 decades of range and the spline is
// barely constrained at its ends).
//
// Because the map is LINEAR IN ITS PARAMETERS given the knots, the calculus is exact and cheap:
// with Λ = Σ xᵢxᵢᵀ the parameter information, a candidate frame contributing ΔΛ = Σⱼ xⱼxⱼᵀ carries
//     ΔI = ½·(log det(Λ + ΔΛ) − log det Λ)     nats
// which is the mutual information between that frame and the parameters under a Gaussian posterior
// (D-optimality). P is ~16, so this is microseconds per frame.
//
// ★D-OPTIMALITY CANNOT DISTINGUISH LEVERAGE FROM CORRUPTION. A frame with a bad pose, bad
// registration, or a person walking through has HIGH leverage and therefore scores as maximally
// informative — a naive rule would preferentially collect exactly the frames that poison the fit. So
// a frame must also be CONSISTENT with the current map before its novelty is believed: novelty is
// earned by data that agrees, not merely by data that is unusual. See `consistent()`.
class InfoSelector
{
public:
    // Start from a ridge εI so log det is finite on an empty set, then fold in whatever is already
    // collected — otherwise the first frames of a session look informative merely because the
    // selector has forgotten the existing dataset.
    void reset(int n_params, double ridge = 1e-6);
    void accumulate(const std::vector<DepthSample>& samples, const DepthFitMap& m, int n_views);
    [[nodiscard]] bool ready() const { return lambda_.rows() > 0; }

    // Information this frame would add, in nats. 0 if not ready.
    [[nodiscard]] double gain(const std::vector<DepthSample>& samples, const DepthFitMap& m,
                              int n_views) const;
    // Median |log_range − predicted| for the frame under `m` with the offset solved per view on the
    // frame itself — i.e. "does this frame behave like the model expects". Large ⇒ suspect the DATA,
    // not the model. Returns -1 when `m` is not valid (nothing to be consistent with yet).
    [[nodiscard]] double residual(const std::vector<DepthSample>& samples, const DepthFitMap& m,
                                  int n_views) const;

    [[nodiscard]] double logdet() const { return logdet_; }
    [[nodiscard]] long   n_accepted() const { return n_accepted_; }

private:
    Eigen::MatrixXd lambda_;
    double          logdet_    = 0.0;   // cached log det(lambda_)
    long            n_accepted_ = 0;
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
    // ★Writes the ORIGINAL 10-column schema. Enrichment never round-trips through here.
    bool append_csv(const std::string& path);
    // ★Reads BOTH schemas. The original header
    //     frame,stamp_ms,rx,ry,rtheta,view,s,t,log_model,log_range
    // and the EXTENDED one written by save_csv(), which adds src,region,w,h. Detection is on the
    // HEADER, never on field count: a row that simply ran out of columns must default to w=1 (a
    // measured, unit-weight sample), and silently reading a missing `w` as 0 would zero the weight of
    // every legacy row — the fit would come out empty with no error anywhere.
    bool load_csv(const std::string& path);
    // Write the WHOLE set in the extended schema (synthetic rows and their precisions included), so an
    // enrichment pass can be cached instead of re-running two neural networks over every frame.
    bool save_csv(const std::string& path) const;

    // Drop near-duplicate poses across the WHOLE set, not just consecutively — several passes over
    // the same room leave duplicates that no capture-time check can see. Returns frames removed.
    std::size_t dedup(float min_move_m = 0.10f, float min_turn_rad = 0.087f);

    // `measured_only` refits over the kSrcLidar rows alone. That is the HONEST A/B for enrichment:
    // the same code, the same frames, the same knot placement rule — only the synthetic rows removed.
    [[nodiscard]] DepthFitMap fit(int n_views, bool measured_only = false) const;

    // ── Offline enrichment hooks (rc::depth::DatasetEnricher) ───────────────────────────────────
    // Attach SYNTHETIC rows to the frame carrying `stamp_ms`. This is the ONLY mutation the
    // enrichment pass performs: the measured rows are never touched, re-derived or re-scaled, so a
    // pass that produces nothing leaves the dataset exactly as it found it. False ⇒ no such frame.
    bool add_samples_to_frame(std::uint64_t stamp_ms, const std::vector<DepthSample>& extra);
    [[nodiscard]] const DepthFrame& frame(std::size_t i) const;

    // Samples of the most recently added frame — for folding it into the InfoSelector after the
    // dataset has taken ownership (add_frame moves the frame in).
    [[nodiscard]] const std::vector<DepthSample>& last_frame_samples() const
    {
        static const std::vector<DepthSample> none;
        return frames_.empty() ? none : frames_.back().samples;
    }
    [[nodiscard]] const std::vector<DepthSample>& frame_samples(std::size_t i) const
    {
        static const std::vector<DepthSample> none;
        return i < frames_.size() ? frames_[i].samples : none;
    }
    [[nodiscard]] std::size_t frame_count()  const { return frames_.size(); }
    [[nodiscard]] std::size_t sample_count() const;
    void clear() { frames_.clear(); pending_ = 0; }

private:
    std::vector<DepthFrame> frames_;
    std::size_t             pending_ = 0;   // frames not yet written by append_csv
};

}   // namespace rc::depth
