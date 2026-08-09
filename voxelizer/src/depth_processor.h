#pragma once

/*
 * depth_processor.h
 *
 * Monocular DEPTH branch of the voxelizer perception stack (yolo26l-depth, task=depth, head=Depth).
 * Input [1,3,768,768] f32, output [1,1,768,768] f32 = the NATURAL LOG of depth, so metres = exp(v).
 *
 * ★READ THIS BEFORE USING THE NUMBERS FOR GEOMETRY. The model is scale-and-shift invariant per
 * INPUT IMAGE, not metric across inputs — it is distilled from a Depth-Anything-style relative-depth
 * teacher (metadata: "depth-mega-v11-imagenet"). Measured 2026-08-02 by running the exact 3-strip
 * geometry below and comparing the SAME global panorama columns where two strips overlap:
 *
 *     strips 0|1  median 5.58 m vs 19.59 m   → 2.78x
 *     strips 1|2  median 16.71 m vs  6.49 m  → 0.45x     (log-difference sigma ~0.4)
 *
 * Identical pixels, depths off by 2-3x, and the DISAGREEMENT IS NOT A CONSTANT OFFSET (the sigma):
 * the predicted shape moves with the surrounding context too. So each strip carries its own arbitrary
 * scale, and the strips CANNOT simply be pasted into one continuous depth field. (That measurement
 * used a flat texture, the only natural image to hand, which gives the model no real depth cues — the
 * mechanism is real, the magnitude is a worst case. Re-measure on a live panorama.)
 *
 * Consequences, encoded in this file:
 *   - The map stores the RAW log-depth exactly as each strip produced it, plus `strip_id` saying which
 *     strip owns each column. Nothing is blended across a seam, because a blend would manufacture a
 *     continuity the model never claimed.
 *   - compose_depth_canvas() normalises PER STRIP. The colour ramp is therefore comparable WITHIN a
 *     strip and meaningless ACROSS one — which is exactly what the model supports, drawn honestly.
 *   - Turning this into metric depth needs an external anchor per strip. The one already in the
 *     pipeline is the helios LiDAR that Ricoh.mask_depth reprojects into the panorama: a least-squares
 *     fit of (a, b) in log space per strip against those hits makes the strips metric AND mutually
 *     consistent, closing the seams by construction rather than by blending. Not done here — this
 *     stage is DISPLAY-ONLY by design.
 *
 * Equirectangular caveat: a strip spans 360/n_strips degrees of azimuth of an EQUIRECT panorama, which
 * is not a perspective image. A depth model trained on perspective photos is being asked to do a
 * geometric task on warped input, and the distortion grows toward the poles — hence the elevation band
 * (see Depth360Config::band_half_elev_deg), which both cuts the worst distortion and makes the strip
 * near-square so it fills the 768 input instead of being letterboxed away.
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rc::depth
{

// One panorama's depth estimate, strip-wise. Panorama-sized so it overlays the source 1:1.
struct DepthMap
{
    // CV_32FC1: natural log of the model's depth, raw. NaN wherever no strip wrote (outside the
    // elevation band, or a strip that failed). exp() gives metres ON THAT STRIP'S OWN SCALE.
    cv::Mat log_depth;
    // CV_8UC1: index of the strip that owns each pixel, kNoStrip where log_depth is NaN. This is what
    // makes the per-strip scale explicit to every consumer instead of an invisible trap.
    cv::Mat strip_id;
    int     n_strips = 0;
    int     band_y0  = 0;   // [band_y0, band_y1) = the panorama rows the band covers
    int     band_y1  = 0;

    [[nodiscard]] bool empty() const noexcept { return log_depth.empty(); }
};

inline constexpr unsigned char kNoStrip = 255;

// 360 slicing. Mirrors Detection360Config (yolo_processor.h) so the two 360 paths read alike, plus the
// elevation band that depth needs and detection does not.
struct Depth360Config
{
    // ── Projection ────────────────────────────────────────────────────────────────────────────────
    // ★GNOMONIC is the reason this works at all. Feeding raw equirect crops (gnomonic=false) was
    // measured 2026-08-02 to produce depth ANTI-CORRELATED with geometry: on a lab panorama the model
    // called the wall at 3-5 m the nearest thing in the strip and the floor 1.4 m under the robot the
    // farthest, with both band edges collapsing to "far". Equirect smears content horizontally toward
    // the poles into low-frequency mush, and a depth head resolves textureless expanses as sky. The
    // model is not broken — it was being handed something that is not a photograph.
    //
    // So each strip is rendered as a VIRTUAL PINHOLE CAMERA looking along the strip's azimuth: build
    // the ray for every pixel of a square gnomonic image, sample the panorama through it (cv::remap,
    // maps cached), run the model on that, then map the depth back to equirect through the inverse
    // maps. The model now sees an actual perspective image, which is what it was trained on.
    bool  gnomonic = true;
    // Horizontal = vertical FOV of each virtual camera, degrees (the view is square, side = the model's
    // input_size). 0 ⇒ derive as 360/n_strips * 1.25, giving a 25% context margin beyond the strip's own
    // azimuth span, clamped to 140 (a rectilinear projection diverges at 180 and is already brutally
    // stretched past ~120). ★A gnomonic view is only as good as it is NARROW — this is what makes
    // n_strips matter far more here than it does for seg: 6 strips ⇒ 75°, close to a normal lens;
    // 3 strips ⇒ 140°, an extreme wide angle that reintroduces much of the distortion.
    float gnomonic_fov_deg = 0.0f;
    // Monocular depth datasets overwhelmingly encode Z-DEPTH (distance along the optical axis), not
    // range along the ray. Converting is exact and, because the head emits LOG depth, purely ADDITIVE:
    // log_range = log_z + 0.5*log(1 + xn^2 + yn^2) for normalised image coords xn,yn. It matters at
    // the corners of a wide view (at 70 deg off-axis, range is ~2.9x z), which is precisely where the
    // reprojection then places those pixels. Off ⇒ treat the output as range already.
    bool  zdepth_to_range = true;

    int   n_strips   = 3;
    // Circular-pad margin, in panorama columns, added to BOTH sides of a strip. It is CONTEXT ONLY:
    // the model sees strip_w + 2*overlap columns, and only the central strip_w are kept, so a column
    // is never predicted twice and no seam blending is needed. Default 64 is chosen with the default
    // band so a strip is exactly 768 px wide = the model input width ⇒ NO horizontal resampling.
    int   overlap_px = 64;
    // Half-height of the kept elevation band, degrees about the equator. 60 deg on a 960-row panorama
    // keeps rows [160, 800) = 640 tall; with the default overlap that is a 768x640 strip, which
    // letterboxes into 768x768 at scale 1.0 (vertical padding only, no scaling at all).
    float band_half_elev_deg = 60.0f;
};

// ONNX wrapper. Owns its own session (same plumbing as YoloSemanticSegmenter / YoloSegDetector) so the
// depth model can be loaded, gated and failed independently of every other model.
class DepthEstimator
{
public:
    DepthEstimator(const std::string& model_path, int input_size, bool use_gpu, bool use_trt);
    ~DepthEstimator();

    DepthEstimator(const DepthEstimator&)            = delete;
    DepthEstimator& operator=(const DepthEstimator&) = delete;

    // Letterbox `bgr` into input_size^2, run, unletterbox, resize back to bgr.size().
    // Returns CV_32FC1 log-depth (OWNED buffer), or an empty Mat on failure.
    [[nodiscard]] cv::Mat infer_log_depth(const cv::Mat& bgr) const;

    [[nodiscard]] int input_size() const noexcept { return input_size_; }

private:
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions           session_opts_;
    std::vector<char*>            input_names_, output_names_;
    std::vector<const char*>      input_names_cstr_, output_names_cstr_;
    int                           input_size_ = 768;
};

// Thin processor: owns the estimator (lazily, like YoloSemanticProcessor) and provides the 360 strip
// orchestration + a viewer overlay.
class DepthProcessor
{
public:
    struct Config
    {
        std::string model_path    = "models/yolo26/yolo26l-depth.onnx";
        int         input_size    = 768;   // must match the exported imgsz
        bool        use_gpu       = true;
        bool        use_trt       = false;
        bool        verbose_debug = false;
    };

    void configure(const Config& config);
    [[nodiscard]] bool ready() const noexcept { return estimator_ != nullptr; }

    // Slice the panorama into strips, run the model on each, and assemble the per-strip log-depth.
    // NOT continuous across seams — see the file header. NON-const because it caches the gnomonic
    // remap tables; called only from DepthStage::run, i.e. the worker thread, never from main.
    [[nodiscard]] DepthMap estimate_360(const cv::Mat& panorama_bgr, const Depth360Config& cfg);

    // Whole-frame estimate for a PERSPECTIVE image (the ZED). One inference, no strips, no elevation
    // band — a pinhole frame is what the model was trained on, so none of the 360 machinery applies.
    // ★No z→range correction here: the ZED's own `depth` is camera Z along the optical axis, so
    // leaving the model in z-depth keeps the two directly comparable. Converting one and not the
    // other would inject a cos-off-axis error that looks exactly like model error.
    [[nodiscard]] DepthMap estimate_perspective(const cv::Mat& bgr) const;

    // Blend a per-strip-normalised TURBO ramp over a copy of `base_bgr` (near = warm). `alpha` is the
    // depth layer's weight in [0,1]. const + argument-only: the main thread calls this on a processor
    // the ricoh worker is concurrently running inference on, so it must touch NO member state
    // (CLAUDE.md cv::Mat rule / the compose-passthrough convention the other stages follow).
    // `metric` switches the ramp from PER-STRIP percentile normalisation (relative, the only honest
    // rendering of an uncorrected map) to a FIXED metres scale spanning [lo_m, hi_m]. Use it only once
    // a correction has been applied — then every view shares one scale, the seams stop meaning
    // anything, and the six-panel patchwork becomes a single field. Seam lines are drawn only in the
    // relative mode, where reading across a seam really is a mistake.
    [[nodiscard]] cv::Mat compose_depth_canvas(const cv::Mat& base_bgr, const DepthMap& map,
                                               float alpha = 0.65f, bool metric = false,
                                               float lo_m = 0.4f, float hi_m = 8.0f) const;

private:
    // One virtual pinhole view: the maps that sample the panorama into it, and the maps that put its
    // result back. Both are pure functions of (panorama size, n_strips, fov, input_size), so they are
    // built once and reused — a remap table is ~2 x 768 x 768 floats per direction, cheap to keep,
    // expensive to rebuild every frame.
    struct StripMaps
    {
        cv::Mat fwd_x, fwd_y;    // gnomonic pixel -> panorama sample coords (CV_32FC1, S x S)
        cv::Mat inv_x, inv_y;    // strip's equirect pixel -> gnomonic sample coords (strip_w x band_h)
        cv::Mat inv_valid;       // CV_8UC1: equirect pixels this view actually covers
        cv::Mat range_corr;      // CV_32FC1, S x S: additive log(range/z) term, zero when disabled
    };

    [[nodiscard]] DepthMap estimate_equirect(const cv::Mat& panorama_bgr, const Depth360Config& cfg) const;
    [[nodiscard]] DepthMap estimate_gnomonic(const cv::Mat& panorama_bgr, const Depth360Config& cfg);
    // Rebuild maps_ if the geometry changed; returns false if the config cannot be satisfied.
    bool ensure_maps(int pano_w, int pano_h, const Depth360Config& cfg);

    Config                          config_;
    std::unique_ptr<DepthEstimator> estimator_;

    // Map cache + the geometry it was built for. WORKER THREAD ONLY (estimate_360 is non-const and
    // only DepthStage::run calls it); compose_depth_canvas is const and never touches this, which is
    // what keeps the main thread's overlay call race-free.
    std::vector<StripMaps> maps_;
    int   maps_w_ = 0, maps_h_ = 0, maps_n_ = 0, maps_s_ = 0;
    float maps_fov_ = 0.f;
    bool  maps_zcorr_ = false;
};

// Best-fit alignment of a model log-depth map against a MEASURED metric depth image, solving
// log(measured) ≈ a·log_model + b by least squares over the valid pixels. This is the "minimum free
// energy" alignment: whatever residual survives it is STRUCTURE the model got wrong, not scale it was
// never given. Returns false if too few valid pixels.
//
// Solving BOTH a and b (rather than reusing the fitted map's a) is deliberate for this diagnostic:
// hundreds of thousands of dense pixels over the full depth range make it far better conditioned than
// the LiDAR horizon stripe the map is fitted from, and it isolates residual shape from residual scale.
struct DepthAlign
{
    float a = 1.f, b = 0.f;
    long  n = 0;
    float med_abs_rel = 0.f;   // median |d−d*|/d* after alignment
    float med_abs_m   = 0.f;
    float delta125    = 0.f;
};
[[nodiscard]] bool align_to_measured(const DepthMap& model, const cv::Mat& measured_m,
                                     float min_m, float max_m, DepthAlign& out);

// Corrected model depth in METRES, same size as the map. NaN where the model has no estimate.
[[nodiscard]] cv::Mat apply_align(const DepthMap& model, const DepthAlign& al);

// Signed difference render: BLACK where the aligned model agrees with the measurement, red where the
// model reads FARTHER than truth, blue where it reads NEARER. Black-at-zero is the point — a
// perfectly aligned model shows an all-black frame, so error is visible as light rather than hue.
[[nodiscard]] cv::Mat compose_difference(const cv::Mat& model_m, const cv::Mat& measured_m,
                                         float span_m = 1.0f);

}   // namespace rc::depth
