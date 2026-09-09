#pragma once

/*
 * yolo_semantic.h
 *
 * Semantic-segmentation branch of the retina perception stack. Runs a YOLO
 * semantic-segmentation ONNX model (e.g. yolo26l-sem) on the same RGB frame the
 * instance-seg detector consumes and returns a DENSE per-pixel class-label map at
 * the original image resolution — one integer class id per pixel.
 *
 * This is semantically distinct from YoloSegDetector (yolo*-seg): that model emits a
 * sparse set of per-INSTANCE masks (proto coefficients → one binary mask per detected
 * object). A semantic model instead emits a single dense label field that tiles the
 * whole image (every pixel gets exactly one class, including background), with no
 * notion of separate object instances. The output tensor is therefore a logit volume
 * [1, C, h, w] (C = #classes), and the label of a pixel is argmax over the class axis.
 *
 * Self-contained ONNX session (mirrors YoloSegDetector / YoloPoseDetector plumbing) so
 * it stays decoupled from the rest of the perception stack and can be gated on its own.
 */

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rc::semantic
{

// Pixel value used in the label map for "no confident class" (argmax score below the
// confidence threshold). 255 keeps the label map a compact CV_8UC1 while staying clear
// of the 80 COCO ids; raise to a wider type if a model ever has ≥255 classes.
inline constexpr unsigned char IGNORE_LABEL = 255;

// Dense semantic result for one RGB frame.
struct SemanticMap
{
    // CV_8UC1, original image size. Each pixel = class id in [0, num_classes) or IGNORE_LABEL.
    cv::Mat labels;
    // CV_32FC1, original image size. Per-pixel argmax score (softmax prob in [0,1] when the
    // model output is logits we normalize, else the raw winning channel value). Empty if
    // the caller asked to skip the score map.
    cv::Mat scores;
    // ★HOW MUCH OF `labels` THE MODEL ACTUALLY RAN OVER, in pixels. 0 ⇒ all of it — the ZED case, and
    // any producer predating this field. SemanticStage360 is why it exists: it infers ONE strip and
    // pastes it into a full-panorama canvas, so `labels` is 3× larger than what was looked at, and a
    // consumer sizing anything off labels.cols*labels.rows measures a 120° observation against a 360°
    // image. Nobody chose that ratio — it appeared when the canvas grew under the knob (see
    // SemanticMaskStage::run, where a hood-sized region was the thing it silently cost).
    // A pixel COUNT, not a rect: the scheduled strips need not be contiguous.
    long long inferred_area_px = 0;

    // ─── GRADED CLASS POSTERIORS (empty unless the model exposes them) ────────────────────────────
    // ★WHY. A *-sem export collapses its 150-way softmax to a uint8 argmax INSIDE the ONNX graph, so
    // a door at P(door)=0.44 losing to P(wall)=0.45 is emitted identically to one the model never
    // saw. Measured on 1108 apartment frames: on frames yielding NO door mask the median margin
    // P(wall)-P(door) at the winning pixel is 0.0084, and 69% are within 0.10. The argmax reports a
    // coin flip as certainty, and the existence channel then charges that silence as confident
    // absence. `probs` is the recovered posterior (see tools/expose_semantic_logits.py).
    //
    // ★NOT frame-sized, ON PURPOSE. These stay at the MODEL's native resolution (the classifier head
    // is 80x80 for a 640 letterbox), already cropped to the active non-padded region. Frame-sizing
    // K channels would be ~37 MB/frame of allocation churn for information the model never had at
    // that resolution; at native it is ~128 kB. Sample with prob_at(), which owns the mapping - the
    // writer and the reader are in different files and this is only coherent if they agree.
    //
    // ★The channel order is declared BY THE MODEL (metadata "prob_class_ids"), never by config: a
    // reordered export against a stale config key would silently read P(door) out of the cabinet
    // channel with nothing to catch it.
    std::vector<int>     prob_class_ids;   // ADE20K class id of each channel of `probs`
    std::vector<cv::Mat> probs;            // CV_32FC1, native res, active region only. May be empty.
    cv::Size             probs_src_size;   // the frame size `probs` maps onto (== labels.size())

    // Channel index of `class_id` in `probs`, or -1 if this model does not expose it.
    [[nodiscard]] int prob_channel(int class_id) const
    {
        for (std::size_t i = 0; i < prob_class_ids.size(); ++i)
            if (prob_class_ids[i] == class_id)
                return static_cast<int>(i);
        return -1;
    }

    // Posterior of `class_id` at FRAME pixel (x, y), bilinear. Returns `fallback` when the model
    // exposes no posterior for it — so a caller written against a graded model still runs, at the
    // old behaviour, against the ungraded one.
    [[nodiscard]] float prob_at(int class_id, int x, int y, float fallback = -1.0f) const
    {
        const int c = prob_channel(class_id);
        if (c < 0 or probs.empty() or probs[static_cast<std::size_t>(c)].empty()
            or probs_src_size.width <= 0 or probs_src_size.height <= 0)
            return fallback;
        const cv::Mat& m = probs[static_cast<std::size_t>(c)];
        // Map frame -> native with the same convention decode() used to resize `labels`: the active
        // region spans the full frame, so it is a straight scale, sampled at pixel centres.
        const float fx = (static_cast<float>(x) + 0.5f) * static_cast<float>(m.cols)
                       / static_cast<float>(probs_src_size.width) - 0.5f;
        const float fy = (static_cast<float>(y) + 0.5f) * static_cast<float>(m.rows)
                       / static_cast<float>(probs_src_size.height) - 0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, m.cols - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, m.rows - 1);
        const int x1 = std::min(x0 + 1, m.cols - 1);
        const int y1 = std::min(y0 + 1, m.rows - 1);
        const float ax = std::clamp(fx - static_cast<float>(x0), 0.0f, 1.0f);
        const float ay = std::clamp(fy - static_cast<float>(y0), 0.0f, 1.0f);
        const float v00 = m.at<float>(y0, x0), v01 = m.at<float>(y0, x1);
        const float v10 = m.at<float>(y1, x0), v11 = m.at<float>(y1, x1);
        return (v00 * (1.0f - ax) + v01 * ax) * (1.0f - ay)
             + (v10 * (1.0f - ax) + v11 * ax) * ay;
    }

    // Every exposed class at FRAME pixel (x, y) as (class_id, P), sorted most-probable first. Empty
    // when the model is ungraded. This is the readout that makes the actual defect visible: at a pixel
    // the argmax calls `wall`, it shows by HOW MUCH door lost — a 0.443/0.378 coin flip and a 0.99/0.01
    // landslide are indistinguishable in the label map, and only the first should ever be read as
    // "the classifier did not resolve this". A NaN entry means "not looked at" (a 360 strip that did
    // not run this frame) and is deliberately NOT the same as a low probability.
    [[nodiscard]] std::vector<std::pair<int, float>> probs_at(int x, int y) const
    {
        std::vector<std::pair<int, float>> out;
        if (not graded())
            return out;
        out.reserve(prob_class_ids.size());
        for (std::size_t i = 0; i < prob_class_ids.size(); ++i)
            out.emplace_back(prob_class_ids[i], prob_at(prob_class_ids[i], x, y));
        std::ranges::sort(out, [](const auto& a, const auto& b)
        {
            if (std::isnan(a.second)) return false;   // NaN ("not looked at") sinks to the bottom
            if (std::isnan(b.second)) return true;
            return a.second > b.second;
        });
        return out;
    }

    // True when this map carries recovered posteriors (i.e. a graded export is loaded).
    [[nodiscard]] bool graded() const noexcept { return not probs.empty(); }
};

class YoloSemanticSegmenter
{
public:
    explicit YoloSemanticSegmenter(const std::string& model_path,
                                   const std::vector<std::string>& class_names = {},
                                   float conf_thresh = 0.25f,
                                   int input_size = 640,
                                   bool use_gpu = true,
                                   bool use_trt = false);
    ~YoloSemanticSegmenter();

    YoloSemanticSegmenter(const YoloSemanticSegmenter&) = delete;
    YoloSemanticSegmenter& operator=(const YoloSemanticSegmenter&) = delete;
    YoloSemanticSegmenter(YoloSemanticSegmenter&&) = default;
    YoloSemanticSegmenter& operator=(YoloSemanticSegmenter&&) = default;

    // Run the model and return the per-pixel label map. `want_scores` controls whether the
    // (more expensive to materialize) per-pixel score map is filled in.
    [[nodiscard]] SemanticMap segment(const cv::Mat& image, bool is_rgb = false,
                                      bool want_scores = false) const;

    void set_conf_threshold(float t) noexcept { conf_thresh_ = t; }
    [[nodiscard]] float conf_threshold() const noexcept { return conf_thresh_; }
    [[nodiscard]] int input_size() const noexcept { return input_size_; }
    [[nodiscard]] int num_classes() const noexcept { return static_cast<int>(class_names_.size()); }
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept { return class_names_; }
    [[nodiscard]] const std::string& label_for(int class_id) const;

private:
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions           session_opts_;
    std::vector<char*>            input_names_;
    std::vector<char*>            output_names_;
    std::vector<const char*>      input_names_cstr_;
    std::vector<const char*>      output_names_cstr_;
    std::vector<std::string>      class_names_;
    float conf_thresh_;
    int   input_size_;
    mutable bool logged_layout_ = false;   // one-shot output-layout diagnostic (see segment())
    // ADE20K class id per channel of the model's `class_probs` output, read from the model's own
    // metadata at load time. Empty ⇒ an ungraded export ⇒ everything below degrades to the old path.
    std::vector<int> prob_class_ids_;

    struct LetterboxResult
    {
        std::vector<float> tensor;
        float scale;
        int   pad_left;
        int   pad_top;
    };

    [[nodiscard]] LetterboxResult preprocess(const cv::Mat& rgb_image) const;

    // Optional graded outputs, located by NAME in segment() (never by index — output order is an
    // exporter detail). Default-constructed ⇒ ungraded model ⇒ decode() takes exactly the old path.
    // ★No default ARGUMENT on decode(): there is one call site, and a silent "" default would let a
    // future caller drop the posteriors without the compiler saying so.
    struct GradedOutputs
    {
        const void* class_probs = nullptr;          // [1,K,h,w]
        const void* top_prob    = nullptr;          // [1,1,h,w], max over ALL classes
        ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        int k = 0, h = 0, w = 0;
        [[nodiscard]] bool valid() const { return class_probs and k > 0 and h > 0 and w > 0; }
    };

    // Turn the model output (a float logit volume OR an integer dense class-id map — dispatched on
    // `elem_type`) into a label (+ optional score) map at letterbox resolution, then crop the active
    // (non-padded) region and resize back to `orig_size`.
    [[nodiscard]] SemanticMap decode(const void* data,
                                     ONNXTensorElementDataType elem_type,
                                     const std::vector<int64_t>& shape,
                                     const cv::Size& orig_size,
                                     float scale, int pad_left, int pad_top,
                                     bool want_scores,
                                     const GradedOutputs& graded) const;

    [[nodiscard]] static std::vector<std::string> default_class_names();
};

// Thin processor: owns the segmenter (lazily, like YoloProcessor) + config gating, and
// provides a colourised overlay for the viewer.
class YoloSemanticProcessor
{
public:
    struct Config
    {
        std::string model_path = "yolo26l-sem-ade20k.onnx";   // ADE20K-150 semantic weights
        float conf_thresh = 0.25f;
        int   input_size = 640;
        bool  use_gpu = true;
        bool  use_trt = false;
        bool  verbose_debug = false;
        std::vector<std::string> class_names;   // empty → default COCO-80
    };

    YoloSemanticProcessor() = default;

    void configure(const Config& config);
    [[nodiscard]] bool ready() const noexcept { return segmenter_.has_value(); }

    // Dense per-pixel labels for the frame. Empty SemanticMap if not configured / frame empty.
    // Caches the result so a decimated caller can reuse the last map on skipped cycles.
    [[nodiscard]] SemanticMap segment(const cv::Mat& rgb_frame, bool want_scores = false);

    [[nodiscard]] const SemanticMap& last_map() const noexcept { return last_map_; }

    // Class-id → name table (ADE20K-150). Empty until configured.
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept
    {
        static const std::vector<std::string> empty;
        return segmenter_ ? segmenter_->class_names() : empty;
    }

    // Blend the label map over a BGR copy of rgb_frame (one stable colour per class id).
    [[nodiscard]] cv::Mat compose_semantic_canvas(const cv::Mat& rgb_frame,
                                                  const SemanticMap& map) const;

private:
    Config config_;
    std::optional<YoloSemanticSegmenter> segmenter_;
    SemanticMap last_map_;
};

// Heat overlay of ONE class's posterior on `img` (CV_8UC3). Returns a fresh Mat; `img` is never
// written. `img_is_rgb` says which order the CALLER's canvas is in — the ZED popup works in RGB and
// the panorama popup in BGR, and applyColorMap always emits BGR, so without being told the two
// windows would paint the same probability in two different colours and only one of them would be
// the colormap. It is not cosmetic: the point of a fixed colormap is that a colour means a number.
//
// ★ALPHA IS THE PROBABILITY ITSELF, not a constant. A fixed blend would paint a P=0.05 pixel and a
// P=0.95 pixel with equal conviction, which is the very confusion this channel exists to remove: the
// point of the graded export is that the image should look uncertain where the model is. So a pixel
// is tinted in proportion to how strongly the class is believed there, and the underlying frame shows
// through untouched where it is not believed at all.
// ★NaN ⇒ NOT LOOKED AT, drawn as a visible hatch rather than as zero. On the 360 panorama only the
// strips scheduled this frame carry a posterior; painting the rest as P=0 would show the model
// confidently denying a door in a place it never inspected.
[[nodiscard]] cv::Mat compose_prob_canvas(const cv::Mat& img, const SemanticMap& map, int class_id,
                                          bool img_is_rgb);

}  // namespace rc::semantic
