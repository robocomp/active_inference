#pragma once

#include <genericworker.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <optional>
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

    [[nodiscard]] float conf_threshold() const noexcept { return conf_thresh_; }
    [[nodiscard]] float iou_threshold() const noexcept { return iou_thresh_; }
    [[nodiscard]] int input_size() const noexcept { return input_size_; }
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept { return class_names_; }

private:
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

class YoloProcessor
{
public:
    struct Config
    {
        std::string model_path = "yolo11-seg.onnx";
        float conf_thresh = 0.25f;
        float iou_thresh = 0.45f;
        int input_size = 640;
        bool use_gpu = true;
        bool use_trt = true;
        int mask_erode_kernel = 2;
        bool mask_tray = true;
        int tray_mask_ref_width = 1280;
        int tray_mask_ref_height = 720;
        std::vector<cv::Point> tray_mask_polygon_px;
        std::vector<std::string> accepted_labels;
        bool verbose_debug = false;
    };

    YoloProcessor() = default;

    void configure(const Config& config);

    std::vector<SegDetection> detect_segmentation(const RoboCompCameraRGBDSimple::TRGBD& rgbd);
    cv::Mat apply_tray_mask(const cv::Mat& rgb_frame) const;
    cv::Mat compose_detection_canvas(const cv::Mat& rgb_frame,
                                     const std::vector<SegDetection>& detections) const;

private:
    std::vector<cv::Point> get_tray_mask_polygon(const cv::Size& image_size) const;
    std::string normalize_yolo_label(const std::string& label) const;
    bool is_accepted_yolo_label(const std::string& label) const;
    void postprocess_yolo_detections(std::vector<SegDetection>& detections) const;

    Config config_;
    std::optional<YoloSegDetector> detector_;
};