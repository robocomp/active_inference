#pragma once

// YOLO segmentation ONNX engine (model load + inference + mask decode). Split out of yolo_processor.h
// so the heavy ONNX Runtime / mask-decode machinery lives on its own; YoloProcessor (yolo_processor.h)
// is the thin façade that owns one of these and adds label filtering / tray suppression / 360 tiling.

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <span>
#include <string>
#include <vector>

struct SegDetection
{
    cv::Rect bbox;
    int class_id;
    std::string label;
    float confidence;
    cv::Mat mask;
    // Runner-up class for the same mask (argmax #2). Lets a downstream whitelist recover a detection whose
    // TOP class is rejected but whose 2nd is accepted and within a small confidence margin. -1/"" if none
    // (e.g. end2end models emit a single class per row, so no runner-up is available).
    int second_class_id = -1;
    std::string second_label;
    float second_confidence = 0.f;
};

class YoloSegDetector
{
public:
    explicit YoloSegDetector(const std::string& model_path,
                             const std::vector<std::string>& class_names = {},
                             float conf_thresh = 0.25f,
                             float iou_thresh = 0.45f,
                             int input_size = 640,
                             bool use_gpu = true,
                             bool use_trt = false);

    ~YoloSegDetector();

    YoloSegDetector(const YoloSegDetector&) = delete;
    YoloSegDetector& operator=(const YoloSegDetector&) = delete;
    YoloSegDetector(YoloSegDetector&&) = default;
    YoloSegDetector& operator=(YoloSegDetector&&) = default;

    [[nodiscard]] std::vector<SegDetection> detect(const cv::Mat& image, bool is_rgb = false) const;

    void set_conf_threshold(float t) noexcept { conf_thresh_ = t; }
    void set_iou_threshold(float t) noexcept { iou_thresh_ = t; }

    // ─── NEAR-MISS PROBE ──────────────────────────────────────────────────────────────────────────
    // ★WHY. Everything below conf_thresh_ is discarded here and never reaches any consumer, so a
    // detection that ALMOST fired is indistinguishable from the object not being there — and the
    // removal channel then charges that frame as absence. The rows are already computed (the model
    // emits a fixed 300 regardless of threshold), so capturing the ones in [probe_floor, conf_thresh)
    // costs one comparison and no inference. Read it to answer "did YOLO nearly see it?", which no
    // log in the fleet can answer today.
    // Off unless probe_floor > 0. Not published anywhere: this is evidence ABOUT the detector.
    struct NearMiss { int class_id; float confidence; cv::Rect bbox; };
    void set_probe_floor(float f) noexcept { probe_floor_ = f; }
    [[nodiscard]] const std::vector<NearMiss>& last_near_misses() const noexcept { return near_misses_; }

    [[nodiscard]] float conf_threshold() const noexcept { return conf_thresh_; }
    [[nodiscard]] float iou_threshold() const noexcept { return iou_thresh_; }
    [[nodiscard]] int input_size() const noexcept { return input_size_; }
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept { return class_names_; }

private:
    float probe_floor_ = 0.0f;                 // 0 ⇒ the near-miss probe is off
    mutable std::vector<NearMiss> near_misses_;   // rebuilt per detect(); mutable because detect() is const

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions session_opts_;
    std::vector<char*> input_names_;
    std::vector<char*> output_names_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    std::vector<std::string> class_names_;
    float conf_thresh_;
    float iou_thresh_;
    int input_size_;

    struct LetterboxResult
    {
        std::vector<float> tensor;
        float scale;
        int pad_left;
        int pad_top;
    };

    struct RawDetection
    {
        cv::Rect2f bbox;
        int class_id;
        float confidence;
        std::vector<float> mask_coeff;
        int second_class_id = -1;      // argmax #2 over class scores (multi-class raw path only)
        float second_confidence = 0.f;
    };

    [[nodiscard]] LetterboxResult preprocess(const cv::Mat& rgb_image) const;
    [[nodiscard]] std::vector<RawDetection> parse_detections(const float* data,
                                                             const std::vector<int64_t>& shape,
                                                             const cv::Size& orig_size,
                                                             float scale,
                                                             int pad_left,
                                                             int pad_top) const;
    [[nodiscard]] std::vector<RawDetection> parse_detections_end2end(const float* data,
                                                                     const std::vector<int64_t>& shape,
                                                                     const cv::Size& orig_size,
                                                                     float scale,
                                                                     int pad_left,
                                                                     int pad_top,
                                                                     int num_protos) const;
    [[nodiscard]] std::vector<int> nms(const std::vector<RawDetection>& dets) const;
    [[nodiscard]] cv::Mat decode_mask(std::span<const float> coeff,
                                      const float* proto_data,
                                      int proto_h,
                                      int proto_w,
                                      const cv::Rect& bbox,
                                      const cv::Size& orig_size,
                                      float scale,
                                      int pad_left,
                                      int pad_top) const;
    [[nodiscard]] static float iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept;
    [[nodiscard]] static std::vector<std::string> default_class_names();
};
