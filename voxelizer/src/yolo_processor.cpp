#include "yolo_processor.h"
#include "onnx_providers.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>
#include <numeric>
#include <print>
#include <stdexcept>

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

std::vector<SegDetection> YoloProcessor::detect_segmentation(const cv::Mat& rgb_frame)
{
    if (rgb_frame.empty())
        return {};
    // No tray black-out: run on the clean frame; tray-region detections are dropped in postprocess.
    return detect_segmentation_on(rgb_frame);
}

std::vector<SegDetection> YoloProcessor::detect_segmentation_on(const cv::Mat& rgb_frame)
{
    if (!detector_.has_value() || rgb_frame.empty()
        || rgb_frame.cols <= 0 || rgb_frame.rows <= 0)
        return {};

    auto detections = detector_->detect(rgb_frame, true);
    postprocess_yolo_detections(detections);
    return detections;
}

std::vector<SegDetection> YoloProcessor::detect_on_strip(const cv::Mat& rgb_strip) const
{
    if (!detector_.has_value() || rgb_strip.empty())
        return {};

    auto detections = detector_->detect(rgb_strip, true);
    normalize_filter_and_erode(detections);   // no tray-phantom drop — zed-only calibration
    return detections;
}

namespace
{
// Circular-pad columns on both sides by wrapping from the opposite edge, so a strip extracted from
// the padded image already contains its seam neighbourhood contiguously — no modular arithmetic
// needed anywhere else in the 360 pipeline (including the panorama's own 0/width wraparound seam,
// which this handles the same way as the two internal strip cuts).
cv::Mat circular_pad_columns(const cv::Mat& img, int pad)
{
    if (pad <= 0 || img.cols <= 0)
        return img;
    const int p = std::min(pad, img.cols);
    const cv::Mat left  = img(cv::Rect(img.cols - p, 0, p, img.rows));
    const cv::Mat right = img(cv::Rect(0, 0, p, img.rows));
    cv::Mat padded;
    cv::hconcat(std::vector<cv::Mat>{left, img, right}, padded);
    return padded;
}

float rect_iou(const cv::Rect& a, const cv::Rect& b)
{
    const cv::Rect inter = a & b;
    if (inter.width <= 0 || inter.height <= 0)
        return 0.0f;
    const float inter_area = static_cast<float>(inter.area());
    const float union_area = static_cast<float>(a.area() + b.area()) - inter_area;
    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

// The linear "extended" global coordinate space (see detect_segmentation_360) correctly places every
// strip EXCEPT the one wraparound pair: the last strip's tail (global >= width, wrapped-around true
// columns near 0) and the first strip's head (global < strip_w, true columns near 0) represent the
// SAME physical columns but sit ~width apart numerically — a plain rect_iou would miss that overlap.
// Try the box shifted by ±width too and take the best match; a real duplicate only ever needs one of
// the three alignments, and a full-width shift never spuriously matches an unrelated pair (object
// boxes are always far smaller than the panorama width).
float rect_iou_with_wrap(const cv::Rect& a, const cv::Rect& b, int width)
{
    float best = rect_iou(a, b);
    cv::Rect shifted = b;
    shifted.x = b.x + width;
    best = std::max(best, rect_iou(a, shifted));
    shifted.x = b.x - width;
    best = std::max(best, rect_iou(a, shifted));
    return best;
}

// Lift a strip-local segmentation mask (full panorama height, strip_px wide) back into full-panorama
// column coordinates, so a kept 360 detection carries a mask aligned with its global bbox (which the
// popup's compose_detection_canvas can then draw). The strip occupies global columns
// [global_offset, global_offset + strip_px) modulo `width`; since strip_px < width it wraps the 0/width
// seam at most once, so a single contiguous copy plus (optionally) one wrapped tail suffices.
cv::Mat lift_strip_mask_to_panorama(const cv::Mat& strip_mask, int global_offset, int width)
{
    if (strip_mask.empty() || width <= 0)
        return {};
    cv::Mat pano = cv::Mat::zeros(strip_mask.rows, width, strip_mask.type());
    const int sp = std::min(strip_mask.cols, width);
    int start = global_offset % width;
    if (start < 0) start += width;
    const int head = std::min(sp, width - start);       // columns that fit before the seam
    strip_mask(cv::Rect(0, 0, head, strip_mask.rows))
        .copyTo(pano(cv::Rect(start, 0, head, strip_mask.rows)));
    if (const int tail = sp - head; tail > 0)           // remainder wraps to column 0
        strip_mask(cv::Rect(head, 0, tail, strip_mask.rows))
            .copyTo(pano(cv::Rect(0, 0, tail, strip_mask.rows)));
    return pano;
}
} // namespace

std::vector<SegDetection> YoloProcessor::detect_segmentation_360(const cv::Mat& panorama_rgb,
                                                                  const Detection360Config& cfg)
{
    if (!detector_.has_value() || panorama_rgb.empty() || cfg.n_strips <= 0)
        return {};

    const int width = panorama_rgb.cols;
    const int strip_w = width / cfg.n_strips;
    if (strip_w <= 0)
        return {};
    const int overlap = std::clamp(cfg.overlap_px, 0, strip_w / 2);

    // padded[0] corresponds to true column (width - overlap); padded[overlap] corresponds to true
    // column 0. So a strip's global offset (true/global column that its own local column 0 maps to)
    // is `s*strip_w - overlap` — a plain linear integer, negative for the first strip and running past
    // `width` for the last, with no special-casing needed for the panorama's wraparound seam.
    const cv::Mat padded = circular_pad_columns(panorama_rgb, overlap);
    const int strip_px = strip_w + 2 * overlap;

    struct GlobalDet { SegDetection det; cv::Rect global_bbox; int global_offset; };
    std::vector<GlobalDet> all;

    for (int s = 0; s < cfg.n_strips; ++s)
    {
        const int local_x0 = s * strip_w;
        if (local_x0 + strip_px > padded.cols)
            continue;   // width not exactly divisible by n_strips — last sliver dropped, not sliced short

        const cv::Mat strip = padded(cv::Rect(local_x0, 0, strip_px, padded.rows));
        const int global_offset = s * strip_w - overlap;

        for (auto& d : detect_on_strip(strip))
        {
            cv::Rect gbox = d.bbox;
            gbox.x += global_offset;
            all.push_back(GlobalDet{std::move(d), gbox, global_offset});
        }
    }

    // Cross-strip merge: an object can appear whole in at most two ADJACENT strips (the ones sharing
    // the overlap region it sits in), so a same-label, high-IoU box from a different strip is the same
    // physical object, not a coincidence — greedy keep-highest-confidence, same idea as normal NMS but
    // across strips instead of across raw proposals within one image.
    std::sort(all.begin(), all.end(), [](const GlobalDet& a, const GlobalDet& b)
              { return a.det.confidence > b.det.confidence; });

    std::vector<SegDetection> merged;
    for (auto& g : all)
    {
        const bool duplicate = std::any_of(merged.begin(), merged.end(), [&](const SegDetection& kept)
        {
            return kept.label == g.det.label
                && rect_iou_with_wrap(kept.bbox, g.global_bbox, width) >= cfg.merge_iou;
        });
        if (duplicate)
            continue;

        SegDetection out = std::move(g.det);
        // Lift the strip-local mask into panorama columns so it aligns with the global bbox (the popup's
        // compose_detection_canvas draws it as a silhouette). The bbox is canonicalized to [0,width) below;
        // the mask spans the whole panorama width already, so it needs no further wrap fix-up.
        out.mask = lift_strip_mask_to_panorama(out.mask, g.global_offset, width);
        out.bbox = g.global_bbox;   // extended coords for now (may be <0 or >=width); canonicalized below
        merged.push_back(std::move(out));
    }

    for (auto& d : merged)
    {
        int x = d.bbox.x % width;
        if (x < 0)
            x += width;
        d.bbox.x = x;
    }

    return merged;
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

void YoloProcessor::normalize_filter_and_erode(std::vector<SegDetection>& detections) const
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

void YoloProcessor::postprocess_yolo_detections(std::vector<SegDetection>& detections) const
{
    normalize_filter_and_erode(detections);

    // Tray phantom removal: the robot's visible tray, if segmented, reads as a "table". So a detection
    // labelled "table" whose ROI overlaps the tray zone IS that phantom — drop it. We no longer black
    // out the pixels (YOLO runs on the clean frame); we just reject the table phantom here. Other labels
    // (e.g. a bottle the robot holds in the tray) are kept. The bbox is rectangular and the tray is
    // curved, so we test the actual tray pixels inside the ROI (not rect-vs-rect). tray_drop_fraction =
    // min ROI/tray overlap to drop (0 ⇒ any intersection). The tray raster is cached (size is fixed).
    if (config_.mask_tray)
    {
        cv::Size img;
        for (const auto& d : detections)
            if (!d.mask.empty()) { img = d.mask.size(); break; }

        if (img.width > 0 && img.height > 0 && tray_mask_cache_size_ != img)
        {
            const auto tray_poly = get_tray_mask_polygon(img);
            tray_mask_cache_ = cv::Mat();
            if (tray_poly.size() >= 3)
            {
                tray_mask_cache_ = cv::Mat::zeros(img, CV_8UC1);
                cv::fillPoly(tray_mask_cache_, std::vector<std::vector<cv::Point>>{tray_poly}, cv::Scalar(255));
            }
            tray_mask_cache_size_ = img;
        }

        if (!tray_mask_cache_.empty())
        {
            const cv::Rect img_rect(0, 0, tray_mask_cache_.cols, tray_mask_cache_.rows);
            const double drop_frac = std::clamp(config_.tray_drop_fraction, 0.0f, 1.0f);
            std::erase_if(detections, [&](const SegDetection& d) -> bool
            {
                if (d.label != "table")
                    return false;   // only the phantom the tray itself would produce

                const cv::Rect roi = d.bbox & img_rect;
                if (roi.width <= 0 || roi.height <= 0)
                    return false;

                const double inter = cv::countNonZero(tray_mask_cache_(roi));
                if (inter <= 0.0)
                    return false;   // table ROI does not touch the tray → real table, keep

                const double bbox_frac = inter / (static_cast<double>(roi.width) * roi.height);
                return bbox_frac >= drop_frac;   // tray phantom → drop bbox + label + mask
            });
        }
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

YoloSegDetector::YoloSegDetector(const std::string& model_path,
                                 const std::vector<std::string>& class_names,
                                 float conf_thresh,
                                 float iou_thresh,
                                 int input_size,
                                 bool use_gpu,
                                 bool use_trt)
    : conf_thresh_(conf_thresh)
    , iou_thresh_(iou_thresh)
    , input_size_(input_size)
{
    class_names_ = class_names.empty() ? default_class_names() : class_names;

    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloSegDetector");

        session_opts_.SetIntraOpNumThreads(1);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        rc::onnx::append_gpu_providers(session_opts_, use_gpu, use_trt, "[YoloSegDetector]");

        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts_);

        Ort::AllocatorWithDefaultOptions allocator;

        const std::size_t n_in = session_->GetInputCount();
        input_names_.reserve(n_in);
        input_names_cstr_.reserve(n_in);
        for (std::size_t i = 0; i < n_in; ++i)
        {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(strdup(name.get()));
            input_names_cstr_.push_back(input_names_.back());
        }

        const std::size_t n_out = session_->GetOutputCount();
        output_names_.reserve(n_out);
        output_names_cstr_.reserve(n_out);
        for (std::size_t i = 0; i < n_out; ++i)
        {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(strdup(name.get()));
            output_names_cstr_.push_back(output_names_.back());
        }

        std::cout << "[YoloSegDetector] Loaded: " << model_path
                  << "  inputs=" << n_in << "  outputs=" << n_out << '\n';
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("[YoloSegDetector] ONNX error: ") + e.what());
    }
}

YoloSegDetector::~YoloSegDetector()
{
    for (char* n : input_names_)
        free(n);
    for (char* n : output_names_)
        free(n);
}

YoloSegDetector::LetterboxResult YoloSegDetector::preprocess(const cv::Mat& rgb_image) const
{
    const float scale_x = static_cast<float>(input_size_) / rgb_image.cols;
    const float scale_y = static_cast<float>(input_size_) / rgb_image.rows;
    const float scale = std::min(scale_x, scale_y);

    const int new_w = static_cast<int>(std::round(rgb_image.cols * scale));
    const int new_h = static_cast<int>(std::round(rgb_image.rows * scale));
    const int pad_l = (input_size_ - new_w) / 2;
    const int pad_t = (input_size_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(rgb_image, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    cv::Mat lb(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(lb(cv::Rect(pad_l, pad_t, new_w, new_h)));

    cv::Mat lb_f;
    lb.convertTo(lb_f, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(lb_f, channels);

    const int stride = input_size_ * input_size_;
    std::vector<float> tensor(3 * stride);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + c * stride, channels[c].data, stride * sizeof(float));

    return {std::move(tensor), scale, pad_l, pad_t};
}

std::vector<YoloSegDetector::RawDetection> YoloSegDetector::parse_detections(const float* data,
                                                                              const std::vector<int64_t>& shape,
                                                                              const cv::Size& orig_size,
                                                                              float scale,
                                                                              int pad_left,
                                                                              int pad_top) const
{
    const int64_t F = shape[1];
    const int64_t A = shape[2];

    const int num_classes = static_cast<int>(class_names_.size());
    const int num_protos = static_cast<int>(F) - 4 - num_classes;

    if (num_protos <= 0)
        throw std::runtime_error("[YoloSegDetector] Unexpected feature count in output tensor 0. Check model and class_names size.");

    std::vector<RawDetection> raws;
    raws.reserve(256);

    for (int64_t a = 0; a < A; ++a)
    {
        const float cx = data[0 * A + a];
        const float cy = data[1 * A + a];
        const float bw = data[2 * A + a];
        const float bh = data[3 * A + a];

        float best_score = 0.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c)
        {
            const float s = data[(4 + c) * A + a];
            if (s > best_score)
            {
                best_score = s;
                best_cls = c;
            }
        }
        if (best_score < conf_thresh_ || best_cls < 0)
            continue;

        std::vector<float> coeff(static_cast<std::size_t>(num_protos));
        for (int p = 0; p < num_protos; ++p)
            coeff[static_cast<std::size_t>(p)] = data[(4 + num_classes + p) * A + a];

        auto unpad = [&](float v, float pad, float sc, int max_v) -> float
        {
            return std::clamp((v - pad) / sc, 0.0f, static_cast<float>(max_v - 1));
        };
        const float x1 = unpad(cx - bw * 0.5f, static_cast<float>(pad_left), scale, orig_size.width);
        const float y1 = unpad(cy - bh * 0.5f, static_cast<float>(pad_top), scale, orig_size.height);
        const float x2 = unpad(cx + bw * 0.5f, static_cast<float>(pad_left), scale, orig_size.width);
        const float y2 = unpad(cy + bh * 0.5f, static_cast<float>(pad_top), scale, orig_size.height);

        if (x2 <= x1 || y2 <= y1)
            continue;

        raws.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best_cls, best_score, std::move(coeff)});
    }

    return raws;
}

std::vector<YoloSegDetector::RawDetection> YoloSegDetector::parse_detections_end2end(const float* data,
                                                                                      const std::vector<int64_t>& shape,
                                                                                      const cv::Size& orig_size,
                                                                                      float scale,
                                                                                      int pad_left,
                                                                                      int pad_top,
                                                                                      int num_protos) const
{
    const int64_t N = shape[1];
    const int64_t F = shape[2];

    auto unpad = [&](float v, float pad, float sc, int maxv) -> float
    {
        return std::clamp((v - pad) / sc, 0.0f, static_cast<float>(maxv - 1));
    };

    std::vector<RawDetection> raws;
    raws.reserve(static_cast<std::size_t>(N));

    for (int64_t d = 0; d < N; ++d)
    {
        const float* row = data + d * F;
        const float conf = row[4];
        if (conf < conf_thresh_)
            continue;

        const int cls = static_cast<int>(row[5]);

        const float ox1 = unpad(row[0], static_cast<float>(pad_left), scale, orig_size.width);
        const float oy1 = unpad(row[1], static_cast<float>(pad_top), scale, orig_size.height);
        const float ox2 = unpad(row[2], static_cast<float>(pad_left), scale, orig_size.width);
        const float oy2 = unpad(row[3], static_cast<float>(pad_top), scale, orig_size.height);

        if (ox2 <= ox1 || oy2 <= oy1)
            continue;

        std::vector<float> coeff(row + 6, row + 6 + num_protos);
        raws.push_back({cv::Rect2f(ox1, oy1, ox2 - ox1, oy2 - oy1), cls, conf, std::move(coeff)});
    }
    return raws;
}

float YoloSegDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept
{
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(a.x + a.width, b.x + b.width);
    const float iy2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0f, ix2 - ix1);
    const float inter_h = std::max(0.0f, iy2 - iy1);
    const float inter = inter_w * inter_h;
    if (inter <= 0.0f)
        return 0.0f;

    const float uni = a.width * a.height + b.width * b.height - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

std::vector<int> YoloSegDetector::nms(const std::vector<RawDetection>& dets) const
{
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return dets[i].confidence > dets[j].confidence; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int> keep;

    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        const int ii = idx[i];
        if (suppressed[ii])
            continue;
        keep.push_back(ii);
        for (std::size_t j = i + 1; j < idx.size(); ++j)
        {
            const int jj = idx[j];
            if (suppressed[jj])
                continue;
            if (dets[ii].class_id == dets[jj].class_id && iou(dets[ii].bbox, dets[jj].bbox) > iou_thresh_)
                suppressed[jj] = true;
        }
    }
    return keep;
}

cv::Mat YoloSegDetector::decode_mask(std::span<const float> coeff,
                                     const float* proto_data,
                                     int proto_h,
                                     int proto_w,
                                     const cv::Rect& bbox,
                                     const cv::Size& orig_size,
                                     float scale,
                                     int pad_left,
                                     int pad_top) const
{
    const int proto_pix = proto_h * proto_w;
    std::vector<float> logits(static_cast<std::size_t>(proto_pix), 0.0f);

    for (int p = 0; p < static_cast<int>(coeff.size()); ++p)
    {
        const float c = coeff[static_cast<std::size_t>(p)];
        const float* pp = proto_data + p * proto_pix;
        for (int px = 0; px < proto_pix; ++px)
            logits[static_cast<std::size_t>(px)] += c * pp[px];
    }

    for (float& v : logits)
        v = 1.0f / (1.0f + std::exp(-v));

    cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, logits.data());
    cv::Mat mask_lb;
    cv::resize(mask_proto, mask_lb, {input_size_, input_size_}, 0, 0, cv::INTER_LINEAR);

    const int new_w = static_cast<int>(std::round(orig_size.width * scale));
    const int new_h = static_cast<int>(std::round(orig_size.height * scale));
    const cv::Rect active_rect(pad_left, pad_top, new_w, new_h);
    cv::Mat mask_active = mask_lb(active_rect).clone();

    cv::Mat mask_full;
    cv::resize(mask_active, mask_full, orig_size, 0, 0, cv::INTER_LINEAR);

    cv::Mat mask_bin = cv::Mat::zeros(orig_size, CV_8UC1);
    const cv::Rect bbox_clamped(
        std::clamp(bbox.x, 0, orig_size.width - 1),
        std::clamp(bbox.y, 0, orig_size.height - 1),
        std::clamp(bbox.x + bbox.width, 0, orig_size.width) - std::clamp(bbox.x, 0, orig_size.width - 1),
        std::clamp(bbox.y + bbox.height, 0, orig_size.height) - std::clamp(bbox.y, 0, orig_size.height - 1));

    if (bbox_clamped.width > 0 && bbox_clamped.height > 0)
    {
        cv::Mat roi_float = mask_full(bbox_clamped);
        cv::Mat roi_bin;
        cv::threshold(roi_float, roi_bin, 0.5, 255.0, cv::THRESH_BINARY);
        roi_bin.convertTo(roi_bin, CV_8UC1);
        roi_bin.copyTo(mask_bin(bbox_clamped));
    }

    return mask_bin;
}

std::vector<SegDetection> YoloSegDetector::detect(const cv::Mat& image, bool is_rgb) const
{
    if (image.empty())
    {
        std::cerr << "[YoloSegDetector] detect() called with empty image\n";
        return {};
    }

    cv::Mat rgb;
    if (is_rgb)
        rgb = image;
    else
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const cv::Size orig_size = rgb.size();
    auto [tensor, scale, pad_l, pad_t] = preprocess(rgb);

    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    const auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info,
                                                              tensor.data(),
                                                              tensor.size(),
                                                              input_shape.data(),
                                                              input_shape.size());

    std::vector<Ort::Value> outputs;
    try
    {
        outputs = session_->Run(Ort::RunOptions{nullptr},
                                input_names_cstr_.data(), &input_tensor, 1,
                                output_names_cstr_.data(), output_names_cstr_.size());
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "[YoloSegDetector] Inference error: " << e.what() << '\n';
        return {};
    }

    if (outputs.size() < 2)
    {
        std::cerr << "[YoloSegDetector] Expected ≥2 output tensors (detections + protos), got "
                  << outputs.size() << '\n';
        return {};
    }

    const auto det_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto proto_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const float* det_data = outputs[0].GetTensorData<float>();
    const float* proto_data = outputs[1].GetTensorData<float>();
    const int proto_h = static_cast<int>(proto_shape[2]);
    const int proto_w = static_cast<int>(proto_shape[3]);
    const int num_protos = static_cast<int>(proto_shape[1]);

    const bool is_end2end = (det_shape[2] == static_cast<int64_t>(4 + 1 + 1 + num_protos));

    std::vector<RawDetection> raws;
    if (is_end2end)
        raws = parse_detections_end2end(det_data, det_shape, orig_size, scale, pad_l, pad_t, num_protos);
    else
        raws = parse_detections(det_data, det_shape, orig_size, scale, pad_l, pad_t);

    if (raws.empty())
        return {};

    const std::vector<int> keep = is_end2end
        ? [&]
          {
              std::vector<int> v(raws.size());
              std::iota(v.begin(), v.end(), 0);
              return v;
          }()
        : nms(raws);

    std::vector<SegDetection> results;
    results.reserve(keep.size());

    for (const int k : keep)
    {
        const RawDetection& rd = raws[static_cast<std::size_t>(k)];

        const cv::Rect bbox(static_cast<int>(rd.bbox.x),
                            static_cast<int>(rd.bbox.y),
                            static_cast<int>(rd.bbox.width),
                            static_cast<int>(rd.bbox.height));

        cv::Mat mask = decode_mask(rd.mask_coeff,
                                   proto_data, proto_h, proto_w,
                                   bbox, orig_size, scale, pad_l, pad_t);

        const std::string lbl = (rd.class_id < static_cast<int>(class_names_.size()))
            ? class_names_[static_cast<std::size_t>(rd.class_id)]
            : "unknown";

        results.push_back({bbox, rd.class_id, lbl, rd.confidence, std::move(mask)});
    }

    return results;
}

std::vector<std::string> YoloSegDetector::default_class_names()
{
    return {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck",
        "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
        "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
        "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
        "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
        "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
        "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"
    };
}