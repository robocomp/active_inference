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

#include <memory>
#include <optional>
#include <string>
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

    struct LetterboxResult
    {
        std::vector<float> tensor;
        float scale;
        int   pad_left;
        int   pad_top;
    };

    [[nodiscard]] LetterboxResult preprocess(const cv::Mat& rgb_image) const;

    // Turn the model output (a float logit volume OR an integer dense class-id map — dispatched on
    // `elem_type`) into a label (+ optional score) map at letterbox resolution, then crop the active
    // (non-padded) region and resize back to `orig_size`.
    [[nodiscard]] SemanticMap decode(const void* data,
                                     ONNXTensorElementDataType elem_type,
                                     const std::vector<int64_t>& shape,
                                     const cv::Size& orig_size,
                                     float scale, int pad_left, int pad_top,
                                     bool want_scores) const;

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

}  // namespace rc::semantic
