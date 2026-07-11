#pragma once

#include "yolo_seg_detector.h"   // SegDetection + YoloSegDetector (ONNX engine)

#include <opencv2/opencv.hpp>

#include <optional>
#include <string>
#include <vector>

// 360-panorama detection config (YoloProcessor::detect_segmentation_360). Free-standing, not nested
// in YoloProcessor: a default *argument* referencing a nested class's own default *member*
// initializers can't be evaluated before the enclosing class is complete (GCC rejects it outright).
struct Detection360Config
{
    int   n_strips   = 3;
    int   overlap_px = 80;    // circular-padded margin so a seam-straddling object survives whole
    float merge_iou  = 0.5f;  // cross-strip box IoU above which two strips' boxes are the same object
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
        // Drop a whole detection when its rectangular ROI (bbox) intersects the tray region and the
        // overlap covers at least this fraction of the bbox area. 0 ⇒ drop on ANY intersection
        // (strongest — the tray is circular but ROIs are rectangular, so any touch is suspect).
        float tray_drop_fraction = 0.0f;
        std::vector<std::string> accepted_labels;
        bool verbose_debug = false;
    };

    YoloProcessor() = default;

    void configure(const Config& config);

    std::vector<SegDetection> detect_segmentation(const cv::Mat& rgb_frame);
    // Runs detection on the given frame and applies postprocess (label normalization, accepted-label
    // filtering, tray-region suppression). detect_segmentation forwards the clean frame here.
    std::vector<SegDetection> detect_segmentation_on(const cv::Mat& rgb_frame);
    // 360-panorama detection: splits a wide equirectangular/cylindrical frame into n_strips vertical
    // strips (mitigates the aspect-ratio distortion of squashing e.g. 1920x960 straight to 640x640),
    // runs the SAME model/session on each, and merges duplicate detections across strip seams. Output
    // boxes are in FULL-PANORAMA pixel coordinates (column canonicalized into [0, panorama width)).
    // No tray-phantom suppression (that's calibrated to the zed camera's mount, not applicable here).
    std::vector<SegDetection> detect_segmentation_360(const cv::Mat& panorama_rgb,
                                                       const Detection360Config& cfg = Detection360Config{});
    cv::Mat compose_detection_canvas(const cv::Mat& rgb_frame,
                                     const std::vector<SegDetection>& detections) const;

private:
    std::vector<cv::Point> get_tray_mask_polygon(const cv::Size& image_size) const;
    std::string normalize_yolo_label(const std::string& label) const;
    bool is_accepted_yolo_label(const std::string& label) const;
    // Label-normalize + accepted-label filter + target-mask erode. Shared by the zed path (which then
    // ALSO applies tray-phantom suppression) and the ricoh strip path (which does not — different
    // camera mount, the tray polygon calibration doesn't apply).
    void normalize_filter_and_erode(std::vector<SegDetection>& detections) const;
    void postprocess_yolo_detections(std::vector<SegDetection>& detections) const;
    [[nodiscard]] std::vector<SegDetection> detect_on_strip(const cv::Mat& rgb_strip) const;

    Config config_;
    std::optional<YoloSegDetector> detector_;

    // Cached rasterized tray polygon (image-sized 8UC1, 255 inside the tray). Rebuilt only when the
    // image size changes. Used by postprocess to erase tray-region pixels from every detection mask.
    mutable cv::Mat  tray_mask_cache_;
    mutable cv::Size tray_mask_cache_size_{0, 0};
};
