#include "yolo_processor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <print>

void YoloProcessor::configure(const Config& config)
{
    config_ = config;
    for (auto& label : config_.accepted_labels)
        label = normalize_yolo_label(label);
    std::sort(config_.accepted_labels.begin(), config_.accepted_labels.end());
    config_.accepted_labels.erase(std::unique(config_.accepted_labels.begin(), config_.accepted_labels.end()),
                                  config_.accepted_labels.end());

    detector_.emplace(config_.model_path,
                      std::vector<std::string>{},
                      config_.conf_thresh,
                      config_.iou_thresh,
                      config_.input_size,
                      config_.use_gpu,
                      config_.use_trt);

    if (config_.verbose_debug)
    {
        std::println("[YOLO] effective flags: use_gpu={} use_trt={}", config_.use_gpu, config_.use_trt);
        std::println("YOLO-seg detector ready: {}", config_.model_path);
    }
}

std::vector<SegDetection> YoloProcessor::detect_segmentation(const RoboCompCameraRGBDSimple::TRGBD& rgbd)
{
    if (!detector_.has_value() || rgbd.image.width <= 0 || rgbd.image.height <= 0)
        return {};

    const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
    const cv::Mat masked_rgb_frame = apply_tray_mask(rgb_frame);
    auto detections = detector_->detect(masked_rgb_frame, true);
    postprocess_yolo_detections(detections);
    return detections;
}

std::vector<cv::Point> YoloProcessor::get_tray_mask_polygon(const cv::Size& image_size) const
{
    if (!config_.mask_tray || config_.tray_mask_polygon_px.size() < 3
        || image_size.width <= 0 || image_size.height <= 0
        || config_.tray_mask_ref_width <= 0 || config_.tray_mask_ref_height <= 0)
        return {};

    const float scale_x = static_cast<float>(image_size.width) / static_cast<float>(config_.tray_mask_ref_width);
    const float scale_y = static_cast<float>(image_size.height) / static_cast<float>(config_.tray_mask_ref_height);

    std::vector<cv::Point> polygon;
    polygon.reserve(config_.tray_mask_polygon_px.size());
    for (const auto& point : config_.tray_mask_polygon_px)
    {
        polygon.emplace_back(
            std::clamp(static_cast<int>(std::lround(static_cast<float>(point.x) * scale_x)), 0, image_size.width - 1),
            std::clamp(static_cast<int>(std::lround(static_cast<float>(point.y) * scale_y)), 0, image_size.height - 1));
    }

    return polygon;
}

cv::Mat YoloProcessor::apply_tray_mask(const cv::Mat& rgb_frame) const
{
    if (rgb_frame.empty())
        return {};

    const auto polygon = get_tray_mask_polygon(rgb_frame.size());
    if (polygon.size() < 3)
        return rgb_frame.clone();

    cv::Mat masked = rgb_frame.clone();
    const std::vector<std::vector<cv::Point>> polygons{polygon};
    cv::fillPoly(masked, polygons, cv::Scalar(0, 0, 0));
    return masked;
}

std::string YoloProcessor::normalize_yolo_label(const std::string& label) const
{
    std::string normalized = label;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "dining table")
        return "table";
    if (normalized == "tv")
        return "monitor";
    return normalized;
}

bool YoloProcessor::is_accepted_yolo_label(const std::string& label) const
{
    if (config_.accepted_labels.empty())
        return true;

    return std::find(config_.accepted_labels.begin(), config_.accepted_labels.end(), label)
        != config_.accepted_labels.end();
}

void YoloProcessor::postprocess_yolo_detections(std::vector<SegDetection>& detections) const
{
    for (auto& detection : detections)
        detection.label = normalize_yolo_label(detection.label);

    std::erase_if(detections, [&](const SegDetection& detection)
    {
        return !is_accepted_yolo_label(detection.label);
    });

    for (auto& detection : detections)
    {
        const bool is_target = detection.label == "table" || detection.label == "chair" || detection.label == "monitor";
        if (!is_target || detection.mask.empty())
            continue;

        cv::Mat eroded;
        const int kernel = std::max(1, config_.mask_erode_kernel);
        cv::erode(detection.mask, eroded, cv::Mat(), cv::Point(-1, -1), kernel);
        detection.mask = eroded;
    }
}

cv::Mat YoloProcessor::compose_detection_canvas(const cv::Mat& rgb_frame,
                                                const std::vector<SegDetection>& detections) const
{
    cv::Mat canvas;
    cv::cvtColor(rgb_frame, canvas, cv::COLOR_RGB2BGR);

    static const std::array<cv::Scalar, 20> palette = {{
        {220, 20, 60}, {119, 11, 32}, {0, 0, 142}, {0, 0, 230}, {106, 0, 228},
        {0, 60, 100}, {0, 80, 100}, {0, 0, 192}, {250, 170, 30}, {100, 170, 30},
        {220, 220, 0}, {175, 116, 175}, {250, 0, 30}, {165, 42, 42}, {255, 77, 255},
        {0, 226, 252}, {182, 182, 255}, {0, 82, 0}, {120, 166, 157}, {110, 76, 0},
    }};

    for (const auto& detection : detections)
    {
        const cv::Scalar& color = palette[static_cast<std::size_t>(detection.class_id) % palette.size()];

        if (!detection.mask.empty())
        {
            cv::Mat color_layer(canvas.size(), CV_8UC3, color);
            cv::Mat mask_bin;
            cv::threshold(detection.mask, mask_bin, 127, 255, cv::THRESH_BINARY);
            cv::Mat blended;
            cv::addWeighted(canvas, 0.55, color_layer, 0.45, 0.0, blended);
            blended.copyTo(canvas, mask_bin);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(canvas, contours, -1, color, 1, cv::LINE_AA);
        }

        cv::rectangle(canvas, detection.bbox, color, 2);

        const std::string text = std::format("{} {:.2f}", detection.label, detection.confidence);
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale = 0.65;
        const int thickness = 2;
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(text, font, scale, thickness, &baseline);

        const int label_pad = 4;
        const int box_x0 = std::clamp(detection.bbox.x, 0, std::max(0, canvas.cols - 1));
        const int box_y0 = std::clamp(detection.bbox.y, 0, std::max(0, canvas.rows - 1));
        const int label_w = std::min(canvas.cols, text_size.width + 2 * label_pad);
        const int label_h = text_size.height + baseline + 2 * label_pad;
        const bool place_above = box_y0 >= label_h;
        const int label_x0 = std::clamp(box_x0, 0, std::max(0, canvas.cols - label_w));
        const int label_y0 = place_above ? (box_y0 - label_h) : std::min(box_y0 + 2, std::max(0, canvas.rows - label_h));
        const cv::Rect label_rect(label_x0, label_y0, label_w, std::min(label_h, canvas.rows - label_y0));

        cv::rectangle(canvas, label_rect, color, cv::FILLED);
        cv::putText(canvas, text,
                    cv::Point(label_rect.x + label_pad, label_rect.y + label_rect.height - baseline - label_pad),
                    font, scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
    }

    return canvas;
}