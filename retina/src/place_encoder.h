#pragma once

/*
 * place_encoder.h — DINOv2 ViT-S/14 over the ricoh 360 panorama, pooled into per-azimuth-sector
 * appearance descriptors.
 *
 * ONE forward pass per panorama, not one per sector. The 1920x960 equirect (2:1) is resized to
 * 448x224 (also 2:1, so no aspect distortion), the ViT produces a 16x32 patch grid at 11.25 deg per
 * column, and the 32 columns are grouped into n_sectors azimuth bins. A yaw change is then exactly a
 * cyclic permutation of that array — which is the entire reason this reads a panorama and not the
 * forward camera.
 *
 * OUTPUT LAYOUT: (1 + n_sectors) * dim floats — the CLS token first, then sector 0..S-1 in PANORAMA
 * COLUMN (sensor) order. Not de-rotated into the room frame: doing that at capture would bake a
 * possibly-wrong yaw estimate into the descriptors themselves.
 *
 * ★ CLS IS A PREFILTER, NOT PART OF THE SCORE (see place_map.h). It is emitted because a single
 * yaw-invariant dot product per keyframe is a very cheap way to shortlist, and for no other reason.
 *
 * ★ THE THREE POOLING PARAMETERS ARE MEASURED, NOT CHOSEN — and the obvious defaults are wrong.
 * Sweep over 5 panoramas x 15 rolls (2026-08-28, tools/export_dinov2.py):
 *
 *     band     centre  pool_p |  roll acc   peak margin
 *     (0,16)   no      1.0    |    0.57       +0.017      <- "just mean-pool the whole image"
 *     (6,12)   no      1.0    |    1.00       +0.066
 *     (6,12)   yes     3.0    |    1.00       +0.354      <- shipped
 *
 * and the ordering is the SAME for place discrimination (similarity contrast between frames <2 s and
 * >60 s apart, 90 frames / 2399 s: +0.046 naive vs +0.169 tuned), so it is not a trade-off.
 *
 *   - BAND. Measured per-row |mean over azimuth| (1.0 = that elevation is constant around the
 *     circle, so carries no bearing information): rows 0-4 ceiling 0.88-0.93, rows 7-11 horizon and
 *     near floor 0.70-0.77, rows 14-15 0.92-0.95. Those bottom rows are THE ROBOT'S OWN BODY — the
 *     ricoh looks down onto it, so the same rigid object appears in every panorama from every
 *     position in the room. Half the image was nuisance.
 *   - CENTERING. |mean token| over a frame is 0.656: DINOv2 tokens share a large common component
 *     carrying no positional information, and a mean over L2-normalised tokens inherits it, so every
 *     sector-to-sector cosine starts near 0.9 and the match peak drowns in it. ★ Per-FRAME, not a
 *     dataset mean: rolling a panorama permutes sectors but leaves the frame mean unchanged, so
 *     per-frame centering is exactly rotation-invariant and cannot break the circular shift.
 *   - GeM p=3. The standard objection is that GeM assumes non-negative features; the sign-preserving
 *     form used here sidesteps that, and once the common mode is gone it is worth +0.12 of margin.
 *
 * ★ PREPROCESSING MUST MATCH tools/export_dinov2.py EXACTLY. The one that bites is the resize
 * filter: 1920 -> 448 is a 4.3x downscale, where cv::INTER_AREA corresponds to torchvision
 * antialias=True and cv::INTER_LINEAR does not. A mismatch there presents as "the model is bad".
 */

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Ort { class Env; class Session; }

namespace rc::place
{

struct EncoderConfig
{
    std::string model_path = "models/dinov2/dinov2_vits14_reg_448x224.onnx";
    bool  use_gpu    = true;
    bool  use_trt    = true;
    int   input_w    = 448;
    int   input_h    = 224;
    int   n_sectors  = 16;
    float pool_p     = 3.0f;
    int   band_lo    = 6;      // patch rows [lo, hi) kept; see the measurement above
    int   band_hi    = 12;
    bool  center     = true;
    /// ★ SOFT SECTOR BINNING. 0 = hard bins (each patch column belongs to exactly one sector).
    /// > 0 = raised-cosine weighting with half-width sector_soft * cols_per_sector, so adjacent
    /// sectors overlap and a sub-sector rotation moves weight SMOOTHLY instead of jumping a boundary.
    ///
    /// ★ WHY IT EXISTS (measured 2026-08-28). Rolling ONE panorama — position identical by
    /// construction, so every drop is pure quantisation — gives a SCALLOP in peak similarity:
    ///        roll     0.00d  5.62d  11.25d  16.88d  22.50d   depth
    ///        hard     1.000  0.830   0.773   0.800   0.912   0.227
    ///        2.0x     1.000  0.878   0.906   0.856   0.923   0.144
    /// Maxima at integer sector multiples, minimum at the half-sector offset. This is why P1
    /// (yaw invariance) fails on real data at 0.977 vs 0.653 while P2 (yaw RECOVERY) passes at 99%:
    /// the argmax still lands correctly, the peak VALUE is just depressed.
    ///
    /// ★ AND IT IS NOT COSMETIC. rho(sim) in place_map.h maps similarity to a POSITIONAL spread, so a
    /// query that happens to sit mid-sector is treated as if it were further away than it is. The
    /// similarity conflates place with sub-sector yaw offset, and soft binning removes 36% of that.
    ///
    /// ★ DEFAULT 0 ON PURPOSE: the preliminary decay/tau numbers in etc/config.toml were measured
    /// with HARD bins, and shipping a different default would silently invalidate them. Turning this
    /// to 2.0 is the FIRST thing to try in the Phase-3 ablation. The residual scallop after that is
    /// patch-grid quantisation (a ViT is not shift-equivariant below its 14 px patch) and no pooling
    /// choice removes it.
    float sector_soft = 0.0f;
};

class PlaceEncoder
{
public:
    explicit PlaceEncoder(const EncoderConfig& cfg);
    ~PlaceEncoder();

    [[nodiscard]] bool ready() const { return ready_; }
    [[nodiscard]] const std::string& why_not_ready() const { return why_not_; }
    [[nodiscard]] const EncoderConfig& config() const { return cfg_; }
    [[nodiscard]] int dim() const { return dim_; }
    [[nodiscard]] int grid_rows() const { return gh_; }
    [[nodiscard]] int grid_cols() const { return gw_; }
    /// (1 + n_sectors) * dim
    [[nodiscard]] std::size_t descriptor_size() const
    { return std::size_t(1 + cfg_.n_sectors) * std::size_t(dim_); }

    /// Encode one BGR equirect panorama. Returns descriptor_size() floats, or empty on failure.
    /// `raw_grid_out`, if non-null, receives the un-pooled gh*gw*dim patch grid — logging THAT rather
    /// than the pooled vector is what makes n_sectors / pool_p / band an offline ablation instead of
    /// a fresh robot drive per option.
    [[nodiscard]] std::vector<float> encode(const cv::Mat& bgr_panorama,
                                            std::vector<float>* raw_grid_out = nullptr);

    /// Pool a raw patch grid with the given parameters. Static so place_eval can re-pool logged grids
    /// under different settings without an ONNX session.
    [[nodiscard]] static std::vector<float> pool(const std::vector<float>& grid, int gh, int gw,
                                                 int dim, const EncoderConfig& cfg);

    /*
     * ★ STARTUP SELF-TEST, and it is worth the one extra forward pass. Encode the panorama and a copy
     * rolled by exactly one sector; the recovered circular shift must be +1. This catches, at startup
     * rather than three weeks later, the three highest-severity silent failures: a transposed patch
     * grid (sectors become elevation bands), an off-by-n_register token slice (grid scrambled), and a
     * pooling band that collapsed. None of them crash and all of them still produce plausible numbers.
     */
    [[nodiscard]] bool self_test(const cv::Mat& bgr_panorama, std::string* detail);

private:
    EncoderConfig                cfg_;
    std::unique_ptr<Ort::Env>    env_;
    std::unique_ptr<Ort::Session> session_;
    struct Impl;
    std::unique_ptr<Impl>        impl_;      // session options + strdup'd io names
    int  dim_ = 0, gh_ = 0, gw_ = 0;
    bool ready_ = false;
    std::string why_not_;
};

}   // namespace rc::place
