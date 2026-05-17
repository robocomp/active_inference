#include "yolo_seg_detector.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <span>
#include <stdexcept>

// ═══════════════════════════════════════════════════════════════════════════
//  Construction / destruction
// ═══════════════════════════════════════════════════════════════════════════

YoloSegDetector::YoloSegDetector(const std::string&              model_path,
                                 const std::vector<std::string>& class_names,
                                 float conf_thresh,
                                 float iou_thresh,
                                 int   input_size,
                                 bool  use_gpu,
                                 bool  use_trt)
    : conf_thresh_(conf_thresh)
    , iou_thresh_(iou_thresh)
    , input_size_(input_size)
{
    class_names_ = class_names.empty() ? default_class_names() : class_names;

    auto can_use_tensorrt = []() -> bool
    {
        // Probe the core TensorRT runtime + builder resource so ORT does not segfault
        // later when creating the builder.
        struct HandleGuard
        {
            void* h = nullptr;
            ~HandleGuard() { if (h) dlclose(h); }
        } nvinfer, builder_res;

        auto try_open_any = [](std::initializer_list<const char*> names) -> void*
        {
            for (const char* name : names)
            {
                if (void* h = dlopen(name, RTLD_LAZY | RTLD_LOCAL); h)
                    return h;
            }
            return nullptr;
        };

        nvinfer.h = try_open_any({"libnvinfer.so", "libnvinfer.so.10", "libnvinfer.so.10.9.0"});
        if (!nvinfer.h)
        {
            std::cerr << "[YoloSegDetector] TensorRT runtime missing (libnvinfer.so): "
                      << dlerror() << "\n";
            return false;
        }

        builder_res.h = try_open_any({
            "libnvinfer_builder_resource.so",
            "libnvinfer_builder_resource.so.10",
            "libnvinfer_builder_resource.so.10.9.0"
        });

        // TensorRT 10.16+ may package builder resources as split SM/PTX libraries
        // (e.g., libnvinfer_builder_resource_sm86.so.10.16.1) without a generic soname.
        if (!builder_res.h)
        {
            const std::array<std::filesystem::path, 3> probe_dirs = {
                "/usr/lib/x86_64-linux-gnu",
                "/lib/x86_64-linux-gnu",
                "/usr/local/cuda-13.2/lib64"
            };
            for (const auto& dir : probe_dirs)
            {
                std::error_code ec;
                if (!std::filesystem::exists(dir, ec) || ec)
                    continue;
                for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
                {
                    if (ec) break;
                    if (!entry.is_regular_file(ec) || ec)
                        continue;
                    const auto name = entry.path().filename().string();
                    if (name.rfind("libnvinfer_builder_resource_", 0) == 0 &&
                        name.find(".so") != std::string::npos)
                    {
                        builder_res.h = dlopen(entry.path().c_str(), RTLD_LAZY | RTLD_LOCAL);
                        if (builder_res.h)
                            break;
                    }
                }
                if (builder_res.h)
                    break;
            }
        }

        if (!builder_res.h)
        {
            std::cerr << "[YoloSegDetector] TensorRT builder resources missing "
                      << "(libnvinfer_builder_resource.so): " << dlerror() << "\n";
            return false;
        }

        return true;
    };

    auto prefer_system_tensorrt_stack = []()
    {
        // Force the process to bind against the system TensorRT 10.16 stack before
        // ONNX Runtime's TensorRT EP gets a chance to pull an older CUDA-12 install
        // from the global linker cache.
        static const std::array<const char*, 3> libs = {
            "/usr/lib/x86_64-linux-gnu/libnvinfer.so.10",
            "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10",
            "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10"
        };

        for (const char* lib : libs)
        {
            if (void* h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL); h == nullptr)
            {
                std::cerr << "[YoloSegDetector] Failed to preload TensorRT library "
                          << lib << ": " << dlerror() << "\n";
            }
        }
    };

    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloSegDetector");

        session_opts_.SetIntraOpNumThreads(1);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu)
        {
            if (use_trt)
            {
                if (can_use_tensorrt())
                {
                    prefer_system_tensorrt_stack();
                    try
                    {
                        OrtTensorRTProviderOptions trt{};
                        trt.device_id                    = 0;
                        trt.trt_fp16_enable              = 1;
                        trt.trt_engine_cache_enable      = 1;
                        trt.trt_engine_cache_path        = ".trt_cache";
                        trt.trt_max_partition_iterations = 1000;
                        trt.trt_min_subgraph_size        = 1;
                        session_opts_.AppendExecutionProvider_TensorRT(trt);
                        std::cout << "[YoloSegDetector] TensorRT EP registered (FP16, cache=.trt_cache)\n";
                    }
                    catch (const Ort::Exception& e)
                    {
                        std::cerr << "[YoloSegDetector] TensorRT EP unavailable ("
                                  << e.what() << "), using CUDA only\n";
                    }
                }
                else
                {
                    std::cerr << "[YoloSegDetector] TensorRT disabled due to missing shared libraries; using CUDA only\n";
                }
            }
            try
            {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                session_opts_.AppendExecutionProvider_CUDA(cuda);
                std::cout << "[YoloSegDetector] CUDA EP registered\n";
            }
            catch (const Ort::Exception& e)
            {
                std::cerr << "[YoloSegDetector] CUDA EP unavailable ("
                          << e.what() << "), falling back to CPU\n";
            }
        }
        else
        {
            std::cout << "[YoloSegDetector] Using CPU EP\n";
        }

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
    for (char* n : input_names_)  free(n);
    for (char* n : output_names_) free(n);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Preprocessing
// ═══════════════════════════════════════════════════════════════════════════

YoloSegDetector::LetterboxResult
YoloSegDetector::preprocess(const cv::Mat& rgb_image) const
{
    // Compute letterbox scale and padding
    const float scale_x = static_cast<float>(input_size_) / rgb_image.cols;
    const float scale_y = static_cast<float>(input_size_) / rgb_image.rows;
    const float scale   = std::min(scale_x, scale_y);

    const int new_w = static_cast<int>(std::round(rgb_image.cols * scale));
    const int new_h = static_cast<int>(std::round(rgb_image.rows * scale));
    const int pad_l = (input_size_ - new_w) / 2;
    const int pad_t = (input_size_ - new_h) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(rgb_image, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);

    // Letterbox (grey padding = 114 as per Ultralytics default)
    cv::Mat lb(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(lb(cv::Rect(pad_l, pad_t, new_w, new_h)));

    // To float [0, 1]
    cv::Mat lb_f;
    lb.convertTo(lb_f, CV_32FC3, 1.0 / 255.0);

    // HWC → CHW interleaved copy
    std::vector<cv::Mat> channels(3);
    cv::split(lb_f, channels);

    const int stride = input_size_ * input_size_;
    std::vector<float> tensor(3 * stride);
    for (int c = 0; c < 3; ++c)
        std::memcpy(tensor.data() + c * stride, channels[c].data, stride * sizeof(float));

    return {std::move(tensor), scale, pad_l, pad_t};
}

// ═══════════════════════════════════════════════════════════════════════════
//  Postprocessing — parse detection tensor
// ═══════════════════════════════════════════════════════════════════════════

std::vector<YoloSegDetector::RawDetection>
YoloSegDetector::parse_detections(const float*                data,
                                   const std::vector<int64_t>& shape,
                                   const cv::Size&             orig_size,
                                   float                       scale,
                                   int                         pad_left,
                                   int                         pad_top) const
{
    // shape: [1, num_features, num_anchors]
    // num_features = 4 + num_classes + num_protos
    const int64_t F = shape[1];   // feature rows
    const int64_t A = shape[2];   // anchors

    // Determine number of classes and prototype coefficients.
    // num_protos is canonically 32 for YOLO-seg; we derive it as F - 4 - num_classes.
    // We discover num_classes from the registry; if empty default to 80.
    const int num_classes = static_cast<int>(class_names_.size());
    const int num_protos  = static_cast<int>(F) - 4 - num_classes;

    if (num_protos <= 0)
        throw std::runtime_error("[YoloSegDetector] Unexpected feature count in output tensor 0. "
                                 "Check model and class_names size.");

    std::vector<RawDetection> raws;
    raws.reserve(256);

    for (int64_t a = 0; a < A; ++a)
    {
        // Box (centre-format, letterbox space)
        const float cx = data[0 * A + a];
        const float cy = data[1 * A + a];
        const float bw = data[2 * A + a];
        const float bh = data[3 * A + a];

        // Best class
        float best_score = 0.0f;
        int   best_cls   = -1;
        for (int c = 0; c < num_classes; ++c)
        {
            const float s = data[(4 + c) * A + a];
            if (s > best_score) { best_score = s; best_cls = c; }
        }
        if (best_score < conf_thresh_ || best_cls < 0) continue;

        // Mask coefficients
        std::vector<float> coeff(static_cast<std::size_t>(num_protos));
        for (int p = 0; p < num_protos; ++p)
            coeff[static_cast<std::size_t>(p)] = data[(4 + num_classes + p) * A + a];

        // Convert box to original-image coordinates
        auto unpad = [&](float v, float pad, float sc, int max_v) -> float {
            return std::clamp((v - pad) / sc, 0.0f, static_cast<float>(max_v - 1));
        };
        const float x1 = unpad(cx - bw * 0.5f, static_cast<float>(pad_left), scale, orig_size.width);
        const float y1 = unpad(cy - bh * 0.5f, static_cast<float>(pad_top),  scale, orig_size.height);
        const float x2 = unpad(cx + bw * 0.5f, static_cast<float>(pad_left), scale, orig_size.width);
        const float y2 = unpad(cy + bh * 0.5f, static_cast<float>(pad_top),  scale, orig_size.height);

        if (x2 <= x1 || y2 <= y1) continue;

        raws.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1),
                        best_cls, best_score, std::move(coeff)});
    }

    return raws;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Postprocessing — end2end (NMS-baked) parser
//  Output shape: [1, max_det, 4+1+1+P]
//  Each row: x1 y1 x2 y2 (letterbox float) | conf | class_id | coeff×P
// ═══════════════════════════════════════════════════════════════════════════

std::vector<YoloSegDetector::RawDetection>
YoloSegDetector::parse_detections_end2end(const float*                data,
                                           const std::vector<int64_t>& shape,
                                           const cv::Size&             orig_size,
                                           float                       scale,
                                           int                         pad_left,
                                           int                         pad_top,
                                           int                         num_protos) const
{
    // shape: [1, max_det, 4+1+1+P]
    const int64_t N = shape[1];   // max detections (rows)
    const int64_t F = shape[2];   // features per detection (cols)

    auto unpad = [&](float v, float pad, float sc, int maxv) -> float {
        return std::clamp((v - pad) / sc, 0.0f, static_cast<float>(maxv - 1));
    };

    std::vector<RawDetection> raws;
    raws.reserve(static_cast<std::size_t>(N));

    for (int64_t d = 0; d < N; ++d)
    {
        const float* row = data + d * F;
        const float conf = row[4];
        if (conf < conf_thresh_) continue;

        const int cls = static_cast<int>(row[5]);

        const float ox1 = unpad(row[0], static_cast<float>(pad_left), scale, orig_size.width);
        const float oy1 = unpad(row[1], static_cast<float>(pad_top),  scale, orig_size.height);
        const float ox2 = unpad(row[2], static_cast<float>(pad_left), scale, orig_size.width);
        const float oy2 = unpad(row[3], static_cast<float>(pad_top),  scale, orig_size.height);

        if (ox2 <= ox1 || oy2 <= oy1) continue;

        std::vector<float> coeff(row + 6, row + 6 + num_protos);
        raws.push_back({cv::Rect2f(ox1, oy1, ox2 - ox1, oy2 - oy1),
                        cls, conf, std::move(coeff)});
    }

    return raws;
}
// ═══════════════════════════════════════════════════════════════════════════

float YoloSegDetector::iou(const cv::Rect2f& a, const cv::Rect2f& b) noexcept
{
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(a.x + a.width,  b.x + b.width);
    const float iy2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0f, ix2 - ix1);
    const float inter_h = std::max(0.0f, iy2 - iy1);
    const float inter   = inter_w * inter_h;
    if (inter <= 0.0f) return 0.0f;

    const float uni = a.width * a.height + b.width * b.height - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

std::vector<int>
YoloSegDetector::nms(const std::vector<RawDetection>& dets) const
{
    // Sort by confidence descending
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int i, int j){ return dets[i].confidence > dets[j].confidence; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int>  keep;

    for (std::size_t i = 0; i < idx.size(); ++i)
    {
        const int ii = idx[i];
        if (suppressed[ii]) continue;
        keep.push_back(ii);
        for (std::size_t j = i + 1; j < idx.size(); ++j)
        {
            const int jj = idx[j];
            if (suppressed[jj]) continue;
            if (dets[ii].class_id == dets[jj].class_id &&
                iou(dets[ii].bbox, dets[jj].bbox) > iou_thresh_)
                suppressed[jj] = true;
        }
    }
    return keep;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mask decoding
// ═══════════════════════════════════════════════════════════════════════════

cv::Mat YoloSegDetector::decode_mask(std::span<const float>  coeff,
                                      const float*            proto_data,
                                      int                     proto_h,
                                      int                     proto_w,
                                      const cv::Rect&         bbox,
                                      const cv::Size&         orig_size,
                                      float                   scale,
                                      int                     pad_left,
                                      int                     pad_top) const
{
    const int   P         = static_cast<int>(coeff.size());
    const int   proto_pix = proto_h * proto_w;

    // mask_logits[y*proto_w + x] = sum_p( coeff[p] * proto[p, y, x] )
    // proto layout: [P, proto_h, proto_w]  (CHW, already without batch dim)
    std::vector<float> logits(static_cast<std::size_t>(proto_pix), 0.0f);

    for (int p = 0; p < P; ++p)
    {
        const float c   = coeff[static_cast<std::size_t>(p)];
        const float* pp = proto_data + p * proto_pix;
        for (int px = 0; px < proto_pix; ++px)
            logits[static_cast<std::size_t>(px)] += c * pp[px];
    }

    // Sigmoid in-place
    for (float& v : logits)
        v = 1.0f / (1.0f + std::exp(-v));

    // Wrap as cv::Mat (proto_h × proto_w)
    cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, logits.data());

    // Correct decode pipeline (matches Ultralytics):
    //   1. proto (160×160) → letterbox space (input_size × input_size)
    //   2. crop the active (non-padded) region
    //   3. resize active region → orig_size

    // Step 1: upsample to full letterbox
    cv::Mat mask_lb;
    cv::resize(mask_proto, mask_lb, {input_size_, input_size_}, 0, 0, cv::INTER_LINEAR);

    // Step 2: crop out padding — active image occupies [pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    const int new_w = static_cast<int>(std::round(orig_size.width  * scale));
    const int new_h = static_cast<int>(std::round(orig_size.height * scale));
    const cv::Rect active_rect(pad_left, pad_top, new_w, new_h);
    cv::Mat mask_active = mask_lb(active_rect).clone();

    // Step 3: resize to original image size
    cv::Mat mask_full;
    cv::resize(mask_active, mask_full, orig_size, 0, 0, cv::INTER_LINEAR);

    // Threshold → binary mask (full image size, bbox-clipped)
    cv::Mat mask_bin = cv::Mat::zeros(orig_size, CV_8UC1);
    const cv::Rect bbox_clamped(
        std::clamp(bbox.x,               0, orig_size.width  - 1),
        std::clamp(bbox.y,               0, orig_size.height - 1),
        std::clamp(bbox.x + bbox.width,  0, orig_size.width)  - std::clamp(bbox.x, 0, orig_size.width  - 1),
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

// ═══════════════════════════════════════════════════════════════════════════
//  detect() — main entry point
// ═══════════════════════════════════════════════════════════════════════════

std::vector<SegDetection>
YoloSegDetector::detect(const cv::Mat& image, bool is_rgb) const
{
    if (image.empty())
    {
        std::cerr << "[YoloSegDetector] detect() called with empty image\n";
        return {};
    }

    // ── 1. Convert colour space ───────────────────────────────────────────
    cv::Mat rgb;
    if (is_rgb)
        rgb = image;
    else
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    const cv::Size orig_size = rgb.size();

    // ── 2. Preprocess ─────────────────────────────────────────────────────
    auto [tensor, scale, pad_l, pad_t] = preprocess(rgb);

    // ── 3. Build ONNX input tensor ────────────────────────────────────────
    const std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    const auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        tensor.data(), tensor.size(),
        input_shape.data(), input_shape.size());

    // ── 4. Inference ──────────────────────────────────────────────────────
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
        std::cerr << "[YoloSegDetector] Expected ≥2 output tensors (detections + protos), "
                  << "got " << outputs.size() << '\n';
        return {};
    }

    // ── 5. Parse detection output ─────────────────────────────────────────
    const auto det_shape  = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto proto_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const float* det_data   = outputs[0].GetTensorData<float>();
    const float* proto_data = outputs[1].GetTensorData<float>();
    const int proto_h = static_cast<int>(proto_shape[2]);
    const int proto_w = static_cast<int>(proto_shape[3]);
    const int num_protos = static_cast<int>(proto_shape[1]);

    // ── Diagnostics (printed once) ─────────────────────────────────────────
    {
        static std::once_flag diag_flag;
        std::call_once(diag_flag, [&]()
        {
            std::cerr << "[YOLO DIAG] out0 shape: [" << det_shape[0]
                      << ", " << det_shape[1] << ", " << det_shape[2] << "]\n";
            std::cerr << "[YOLO DIAG] out1 shape: [" << proto_shape[0] << ", "
                      << proto_shape[1] << ", " << proto_shape[2] << ", " << proto_shape[3] << "]\n";
            const bool e2e = (det_shape[2] == 4 + 1 + 1 + num_protos);
            std::cerr << "[YOLO DIAG] format=" << (e2e ? "end2end-NMS" : "standard-raw")
                      << "  num_protos=" << num_protos
                      << "  conf_thresh=" << conf_thresh_ << "\n";
            // Print first row with conf > 0 to see if boxes are in letterbox (0-640)
            // or original-image space (values > 640 likely means already scaled)
            if (e2e)
            {
                const int64_t N = det_shape[1], F2 = det_shape[2];
                for (int64_t d = 0; d < N; ++d)
                {
                    const float* row = det_data + d * F2;
                    if (row[4] > 0.01f)
                    {
                        std::cerr << "[YOLO DIAG] first det: x1=" << row[0] << " y1=" << row[1]
                                  << " x2=" << row[2] << " y2=" << row[3]
                                  << " conf=" << row[4] << " cls=" << row[5]
                                  << "  image=" << orig_size.width << "x" << orig_size.height
                                  << "  input_size=" << input_size_ << "\n";
                        break;
                    }
                }
            }
        });
    }

    // Detect layout: end2end has shape [1, max_det, 4+1+1+P]; standard has [1, 4+C+P, anchors]
    const bool is_end2end = (det_shape[2] == static_cast<int64_t>(4 + 1 + 1 + num_protos));

    std::vector<RawDetection> raws;
    if (is_end2end)
        raws = parse_detections_end2end(det_data, det_shape, orig_size, scale, pad_l, pad_t, num_protos);
    else
        raws = parse_detections(det_data, det_shape, orig_size, scale, pad_l, pad_t);

    if (raws.empty()) return {};

    // NMS: skip for end2end (model already applied it)
    const std::vector<int> keep = is_end2end
        ? [&]{ std::vector<int> v(raws.size()); std::iota(v.begin(), v.end(), 0); return v; }()
        : nms(raws);

    // ── 8. Assemble results ───────────────────────────────────────────────
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

// ═══════════════════════════════════════════════════════════════════════════
//  Default COCO 80-class names
// ═══════════════════════════════════════════════════════════════════════════

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
