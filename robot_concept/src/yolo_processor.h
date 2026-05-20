#pragma once

#include <genericworker.h>

#include <string>
#include <vector>

#include "yolo_seg_detector.h"

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