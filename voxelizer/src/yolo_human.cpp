#include "yolo_human.h"
#include "onnx_providers.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <print>
#include <stdexcept>

namespace rc::human_pose
{

// Raw (pre-NMS) pose row: 4 bbox(cxcywh) + 1 conf + 17*(x,y,conf).
inline constexpr int POSE_FEATURES = 5 + NUM_COCO_KP * 3;      // = 56
// End2end (NMS baked in) row: 4 bbox(xyxy) + 1 conf + 1 class + 17*(x,y,conf).
inline constexpr int POSE_FEATURES_E2E = 6 + NUM_COCO_KP * 3;  // = 57

YoloPoseDetector::YoloPoseDetector(const std::string& model_path,
                                   float conf_thresh,
                                   float iou_thresh,
                                   int input_size,
                                   bool use_gpu,
                                   bool use_trt)
    : conf_thresh_(conf_thresh)
    , iou_thresh_(iou_thresh)
    , input_size_(input_size)
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloPoseDetector");
        session_opts_.SetIntraOpNumThreads(1);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        rc::onnx::append_gpu_providers(session_opts_, use_gpu, use_trt, "[YoloPoseDetector]");

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
        std::cout << "[YoloPoseDetector] Loaded: " << model_path
                  << "  inputs=" << n_in << "  outputs=" << n_out << '\n';
    }
    catch (const Ort::Exception& e)
    {
        throw std::runtime_error(std::string("[YoloPoseDetector] ONNX error: ") + e.what());
    }
}

YoloPoseDetector::~YoloPoseDetector()
{
    for (char* n : input_names_)
        free(n);
    for (char* n : output_names_)
        free(n);
}

YoloPoseDetector::LetterboxResult YoloPoseDetector::preprocess(const cv::Mat& rgb_image) const
{
    const float scale = std::min(static_cast<float>(input_size_) / rgb_image.cols,
                                 static_cast<float>(input_size_) / rgb_image.rows);
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

std::vector<PoseDetection> YoloPoseDetector::parse(const float* data,
                                                   const std::vector<int64_t>& shape,
                                                   const cv::Size& orig_size,
                                                   float scale, int pad_left, int pad_top) const
{
    // Two export layouts: end2end (NMS baked, [1,N,57] = bbox xyxy + conf + class + kp) — what the
    // ultralytics `nms=True` export emits — or raw pre-NMS ([1,56,A] / [1,A,56], bbox cxcywh + conf + kp).
    if (shape.size() != 3)
        throw std::runtime_error("[YoloPoseDetector] expected a rank-3 output tensor");

    const int64_t d1 = shape[1];
    const int64_t d2 = shape[2];

    auto unpad = [&](float v, float pad, int maxv) -> float
    {
        return std::clamp((v - pad) / scale, 0.0f, static_cast<float>(maxv - 1));
    };
    // Fill the 17 COCO keypoints starting at feature `kp_base`. `at(f, idx)` reads feature f of det idx.
    auto read_kp = [&](auto&& at, int kp_base, int64_t idx, PoseDetection& det)
    {
        for (int k = 0; k < NUM_COCO_KP; ++k)
        {
            const int b = kp_base + k * 3;
            const float kx = (at(b + 0, idx) - static_cast<float>(pad_left)) / scale;
            const float ky = (at(b + 1, idx) - static_cast<float>(pad_top)) / scale;
            det.kp[static_cast<std::size_t>(k)] = {kx, ky, at(b + 2, idx)};
        }
    };

    std::vector<PoseDetection> raws;
    raws.reserve(64);

    // ── End2end: [1, N, 57] (row-major) or transposed. Already NMS'd; rows are conf-sorted with
    //    zero-conf padding at the tail (filtered by the conf gate). bbox is xyxy in letterbox space. ──
    if (d1 == POSE_FEATURES_E2E || d2 == POSE_FEATURES_E2E)
    {
        const bool row_major = (d2 == POSE_FEATURES_E2E);
        const int64_t N = row_major ? d1 : d2;
        auto at = [&](int f, int64_t i) -> float
        {
            return row_major ? data[i * POSE_FEATURES_E2E + f] : data[f * N + i];
        };
        for (int64_t i = 0; i < N; ++i)
        {
            const float conf = at(4, i);
            if (conf < conf_thresh_)
                continue;
            const float x1 = unpad(at(0, i), static_cast<float>(pad_left), orig_size.width);
            const float y1 = unpad(at(1, i), static_cast<float>(pad_top), orig_size.height);
            const float x2 = unpad(at(2, i), static_cast<float>(pad_left), orig_size.width);
            const float y2 = unpad(at(3, i), static_cast<float>(pad_top), orig_size.height);
            if (x2 <= x1 || y2 <= y1)
                continue;
            PoseDetection det;
            det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
            det.confidence = conf;
            read_kp(at, /*kp_base=*/6, i, det);   // after bbox(4) + conf(1) + class(1)
            raws.push_back(std::move(det));
        }
        return raws;
    }

    // ── Raw pre-NMS: [1, 56, A] (feature-major) or [1, A, 56] (anchor-major). bbox is cxcywh. ──
    bool feature_major = false;
    int64_t A = 0;
    if (d1 == POSE_FEATURES) { feature_major = true;  A = d2; }
    else if (d2 == POSE_FEATURES) { feature_major = false; A = d1; }
    else
        throw std::runtime_error("[YoloPoseDetector] output feature count != 56/57 — not a 17-kp pose model");

    auto at = [&](int f, int64_t a) -> float
    {
        return feature_major ? data[f * A + a] : data[a * POSE_FEATURES + f];
    };
    for (int64_t a = 0; a < A; ++a)
    {
        const float conf = at(4, a);
        if (conf < conf_thresh_)
            continue;

        const float cx = at(0, a), cy = at(1, a), bw = at(2, a), bh = at(3, a);
        const float x1 = unpad(cx - bw * 0.5f, static_cast<float>(pad_left), orig_size.width);
        const float y1 = unpad(cy - bh * 0.5f, static_cast<float>(pad_top), orig_size.height);
        const float x2 = unpad(cx + bw * 0.5f, static_cast<float>(pad_left), orig_size.width);
        const float y2 = unpad(cy + bh * 0.5f, static_cast<float>(pad_top), orig_size.height);
        if (x2 <= x1 || y2 <= y1)
            continue;

        PoseDetection det;
        det.bbox = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                            cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        det.confidence = conf;
        read_kp(at, /*kp_base=*/5, a, det);   // after bbox(4) + conf(1)
        raws.push_back(std::move(det));
    }
    return raws;
}

float YoloPoseDetector::iou(const cv::Rect& a, const cv::Rect& b) noexcept
{
    const float inter = static_cast<float>((a & b).area());
    if (inter <= 0.0f)
        return 0.0f;
    const float uni = static_cast<float>(a.area() + b.area()) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> YoloPoseDetector::nms(const std::vector<PoseDetection>& dets) const
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
            if (!suppressed[jj] && iou(dets[ii].bbox, dets[jj].bbox) > iou_thresh_)
                suppressed[jj] = true;
        }
    }
    return keep;
}

std::vector<PoseDetection> YoloPoseDetector::detect(const cv::Mat& image, bool is_rgb) const
{
    if (!session_ || image.empty())
        return {};

    cv::Mat rgb;
    if (is_rgb)
        rgb = image;
    else
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const auto lb = preprocess(rgb);
    const std::array<int64_t, 4> in_shape = {1, 3, input_size_, input_size_};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(lb.tensor.data()), lb.tensor.size(),
        in_shape.data(), in_shape.size());

    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 input_names_cstr_.data(), &input, 1,
                                 output_names_cstr_.data(), 1);
    if (outputs.empty())
        return {};

    const float* out = outputs[0].GetTensorData<float>();
    const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    auto raws = parse(out, shape, rgb.size(), lb.scale, lb.pad_left, lb.pad_top);
    const auto keep = nms(raws);

    std::vector<PoseDetection> result;
    result.reserve(keep.size());
    for (int i : keep)
        result.push_back(std::move(raws[static_cast<std::size_t>(i)]));
    return result;
}

// ---------------------------------------------------------------------------------------

void YoloHumanProcessor::configure(const Config& config)
{
    config_ = config;
    detector_.emplace(config_.model_path, config_.conf_thresh, config_.iou_thresh,
                      config_.input_size, config_.use_gpu, config_.use_trt);
    if (config_.verbose_debug)
        std::println("[YoloHuman] pose detector ready: {}", config_.model_path);
}

std::vector<PoseDetection> YoloHumanProcessor::detect_poses(const cv::Mat& rgb_frame)
{
    if (!detector_.has_value() || rgb_frame.empty())
        return {};
    return detector_->detect(rgb_frame, /*is_rgb=*/false);
}

cv::Mat YoloHumanProcessor::compose_pose_canvas(const cv::Mat& rgb_frame,
                                                const std::vector<PoseDetection>& poses) const
{
    if (rgb_frame.empty())
        return {};
    cv::Mat canvas = rgb_frame.clone();

    // COCO-17 skeleton edges (ultralytics convention).
    static constexpr std::array<std::pair<int, int>, 19> EDGES = {{
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
        {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10},
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6},
    }};
    const float kp_thresh = 0.30f;
    for (const auto& p : poses)
    {
        cv::rectangle(canvas, p.bbox, cv::Scalar(0, 200, 255), 2);
        for (const auto& [a, b] : EDGES)
        {
            const auto& ka = p.kp[static_cast<std::size_t>(a)];
            const auto& kb = p.kp[static_cast<std::size_t>(b)];
            if (ka.conf < kp_thresh || kb.conf < kp_thresh)
                continue;
            cv::line(canvas, {static_cast<int>(ka.x), static_cast<int>(ka.y)},
                     {static_cast<int>(kb.x), static_cast<int>(kb.y)}, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }
        for (const auto& k : p.kp)
            if (k.conf >= kp_thresh)
                cv::circle(canvas, {static_cast<int>(k.x), static_cast<int>(k.y)}, 3, cv::Scalar(0, 0, 255), -1);
    }
    return canvas;
}

}  // namespace rc::human_pose
