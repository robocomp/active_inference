#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
//  Result types
// ═══════════════════════════════════════════════════════════════════════════

/// One detected instance returned by YoloSegDetector::detect().
struct SegDetection
{
    cv::Rect   bbox;        ///< Bounding box in original image coordinates.
    int        class_id;    ///< COCO / model class index.
    std::string label;      ///< Human-readable class name.
    float      confidence;  ///< Detection confidence in [0, 1].
    cv::Mat    mask;        ///< Binary mask (CV_8UC1, same size as input image).
                            ///  Pixels belonging to the instance are 255, rest 0.
};

// ═══════════════════════════════════════════════════════════════════════════
//  YoloSegDetector
//
//  Runs a YOLO*-seg ONNX model (YOLOv8-seg / YOLO11-seg format) on a single
//  RGB or BGR cv::Mat and returns per-instance segmentation results.
//
//  YOLO-seg output layout (two tensors):
//    out0: [1, 4 + num_classes + num_protos, num_anchors]
//           └─ first 4       : cx, cy, w, h (letterbox space)
//           └─ next C        : per-class confidence
//           └─ last P (=32)  : mask prototype coefficients
//    out1: [1, num_protos, mask_h, mask_w]   (prototype mask bank)
//
//  Mask decoding:
//    mask_logits = coefficients (1×P) × protos.reshape(P, mask_h×mask_w)
//    mask_sigmoid = sigmoid(mask_logits).reshape(mask_h, mask_w)
//    → crop to letterbox region, resize to original size, threshold @ 0.5
// ═══════════════════════════════════════════════════════════════════════════

class YoloSegDetector
{
public:
    /// \param model_path   Path to the .onnx YOLO-seg model file.
    /// \param class_names  Optional class list. Defaults to 80-class COCO.
    /// \param conf_thresh  Minimum confidence to keep a detection.
    /// \param iou_thresh   NMS IoU threshold.
    /// \param input_size   Square input side (model must have been exported at this size).
    /// \param use_gpu      Try to use CUDA EP; falls back to CPU on failure.
    /// \param use_trt      Also register TensorRT EP (requires libnvinfer to be installed).
    explicit YoloSegDetector(const std::string&              model_path,
                             const std::vector<std::string>& class_names = {},
                             float                           conf_thresh  = 0.25f,
                             float                           iou_thresh   = 0.45f,
                             int                             input_size   = 640,
                             bool                            use_gpu      = true,
                             bool                            use_trt      = false);

    ~YoloSegDetector();

    // Non-copyable, movable.
    YoloSegDetector(const YoloSegDetector&)            = delete;
    YoloSegDetector& operator=(const YoloSegDetector&) = delete;
    YoloSegDetector(YoloSegDetector&&)                 = default;
    YoloSegDetector& operator=(YoloSegDetector&&)      = default;

    /// Run inference on \p image (BGR or RGB — see \p is_rgb).
    /// Returns one SegDetection per instance passing conf/NMS filtering.
    /// \param image   Input image (CV_8UC3).
    /// \param is_rgb  True if the image is already RGB; false (default) for BGR.
    [[nodiscard]] std::vector<SegDetection>
    detect(const cv::Mat& image, bool is_rgb = false) const;

    void set_conf_threshold(float t) noexcept { conf_thresh_ = t; }
    void set_iou_threshold(float t)  noexcept { iou_thresh_  = t; }

    [[nodiscard]] float conf_threshold() const noexcept { return conf_thresh_; }
    [[nodiscard]] float iou_threshold()  const noexcept { return iou_thresh_;  }
    [[nodiscard]] int   input_size()     const noexcept { return input_size_;  }
    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept { return class_names_; }

private:
    // ── ONNX Runtime handles ──────────────────────────────────────────────
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::SessionOptions           session_opts_;

    // Owned C-strings for input/output name arrays
    std::vector<char*>       input_names_;
    std::vector<char*>       output_names_;
    // Non-owning views passed to Session::Run
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;

    // ── Config ────────────────────────────────────────────────────────────
    std::vector<std::string> class_names_;
    float conf_thresh_;
    float iou_thresh_;
    int   input_size_;   // e.g. 640

    // ── Preprocessing ─────────────────────────────────────────────────────
    /// Letterbox-resize to (input_size_ × input_size_), normalize to [0,1],
    /// convert to CHW float tensor. Returns the float buffer plus padding info.
    struct LetterboxResult
    {
        std::vector<float> tensor;   // CHW, float32, [0,1]
        float scale;                 // resize scale applied to original image
        int   pad_left;
        int   pad_top;
    };
    [[nodiscard]] LetterboxResult preprocess(const cv::Mat& rgb_image) const;

    // ── Postprocessing ────────────────────────────────────────────────────
    struct RawDetection
    {
        cv::Rect2f         bbox;          // in original-image space
        int                class_id;
        float              confidence;
        std::vector<float> mask_coeff;    // length == num_protos (32)
    };

    /// Parse output tensor 0 (detections + mask coefficients).
    [[nodiscard]] std::vector<RawDetection>
    parse_detections(const float*            data,
                     const std::vector<int64_t>& shape,
                     const cv::Size&         orig_size,
                     float                   scale,
                     int                     pad_left,
                     int                     pad_top) const;

    /// Parse end2end (NMS-baked) format: [1, max_det, 4+1+1+P]
    /// Each row: x1,y1,x2,y2 (letterbox), conf, class_id, coeff×P
    [[nodiscard]] std::vector<RawDetection>
    parse_detections_end2end(const float*                data,
                             const std::vector<int64_t>& shape,
                             const cv::Size&             orig_size,
                             float                       scale,
                             int                         pad_left,
                             int                         pad_top,
                             int                         num_protos) const;

    /// Non-maximum suppression; returns surviving indices into \p dets.
    [[nodiscard]] std::vector<int>
    nms(const std::vector<RawDetection>& dets) const;

    /// Decode a single instance mask from prototype bank.
    /// \param coeff      mask coefficients (length P).
    /// \param proto_data raw float data of output tensor 1, shape [1, P, mh, mw].
    /// \param proto_h    prototype mask height (e.g. 160).
    /// \param proto_w    prototype mask width  (e.g. 160).
    /// \param bbox       detection bounding box in original-image coordinates.
    /// \param orig_size  original image size (used to crop & resize back).
    /// \param scale      letterbox scale.
    /// \param pad_left   letterbox left padding.
    /// \param pad_top    letterbox top padding.
    [[nodiscard]] cv::Mat
    decode_mask(std::span<const float>  coeff,
                const float*            proto_data,
                int                     proto_h,
                int                     proto_w,
                const cv::Rect&         bbox,
                const cv::Size&         orig_size,
                float                   scale,
                int                     pad_left,
                int                     pad_top) const;

    [[nodiscard]] static float iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept;

    [[nodiscard]] static std::vector<std::string> default_class_names();
};
