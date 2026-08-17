#include "yolo_seg_detector.h"
#include "onnx_providers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <print>
#include <span>
#include <stdexcept>

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

        float best_score = 0.0f, second_score = 0.0f;
        int best_cls = -1, second_cls = -1;
        for (int c = 0; c < num_classes; ++c)
        {
            const float s = data[(4 + c) * A + a];
            if (s > best_score)
            {
                second_score = best_score;   second_cls = best_cls;
                best_score = s;              best_cls = c;
            }
            else if (s > second_score)
            {
                second_score = s;            second_cls = c;
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

        raws.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), best_cls, best_score, std::move(coeff),
                        second_cls, second_score});
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

    // ─── NEAR-MISS PROBE: the rows we are about to throw away ─────────────────────────────────────
    // Collected BEFORE parsing so it sees the raw model output, and only for the end2end layout (the
    // legacy layout's per-class scores would need the same argmax the parser does, and that path is
    // not what runs). These are detections in [probe_floor_, conf_thresh_) — the ones that make a
    // frame read as "the object is not there" when the truth is "the detector nearly fired".
    near_misses_.clear();
    if (probe_floor_ > 0.0f and is_end2end)
    {
        const int64_t N = det_shape[1], F = det_shape[2];
        for (int64_t d = 0; d < N; ++d)
        {
            const float* row = det_data + d * F;
            const float conf = row[4];
            if (conf < probe_floor_ or conf >= conf_thresh_)
                continue;
            const auto unpad = [&](float v, float pad, int maxv)
            { return std::clamp((v - pad) / scale, 0.0f, static_cast<float>(maxv - 1)); };
            const float x1 = unpad(row[0], static_cast<float>(pad_l), orig_size.width);
            const float y1 = unpad(row[1], static_cast<float>(pad_t), orig_size.height);
            const float x2 = unpad(row[2], static_cast<float>(pad_l), orig_size.width);
            const float y2 = unpad(row[3], static_cast<float>(pad_t), orig_size.height);
            if (x2 <= x1 or y2 <= y1)
                continue;
            near_misses_.push_back({static_cast<int>(row[5]), conf,
                                    cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                             static_cast<int>(x2 - x1), static_cast<int>(y2 - y1))});
        }
    }

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

        const auto name_of = [&](int cls) -> std::string
        {
            return (cls >= 0 and cls < static_cast<int>(class_names_.size()))
                ? class_names_[static_cast<std::size_t>(cls)] : std::string{};
        };
        const std::string lbl = name_of(rd.class_id).empty() ? "unknown" : name_of(rd.class_id);

        results.push_back({bbox, rd.class_id, lbl, rd.confidence, std::move(mask),
                           rd.second_class_id, name_of(rd.second_class_id), rd.second_confidence});
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
