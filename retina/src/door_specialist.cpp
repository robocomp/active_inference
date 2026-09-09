#include "door_specialist.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <print>

#include "onnx_providers.h"   // rc::onnx::append_gpu_providers — the same EP setup every stage uses

namespace rc::doors
{

DoorSpecialist::DoorSpecialist() = default;
DoorSpecialist::~DoorSpecialist() = default;

bool DoorSpecialist::configure(const Config& cfg)
{
    cfg_ = cfg;
    session_.reset();
    if (cfg_.model_path.empty())
        return false;
    try
    {
        env_  = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DoorSpecialist");
        opts_ = std::make_unique<Ort::SessionOptions>();
        // ★THREADS DEPEND ON WHERE IT RUNS. Every other detector here pins 1 intra-op thread because it
        // runs on the GPU and the CPU work is just marshalling. This one is CPU-bound by default (see
        // the note in configure()'s caller): 1 thread would make a 640^2 yolo11s pass cost ~0.3 s and
        // stall the ZED worker. 4 is a compromise — enough to keep a call near 0.1 s, few enough not to
        // starve the seven GPU sessions' host-side threads.
        opts_->SetIntraOpNumThreads(cfg_.use_gpu ? 1 : 4);
        opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // ⚠CPU BY DEFAULT, AND THIS IS NOT A PREFERENCE. retina already holds SEVEN ONNX sessions on the
        // CUDA EP (seg, pose, semantic, sam2, depth, ricoh-depth, place). Adding an eighth aborted the
        // whole agent on its first inference — "Invalid handle. Cannot load symbol cublasLtGetVersion",
        // then SIGABRT, which no try/catch can intercept because it is abort() and not an exception.
        // The specialist is small (37 MB, yolo11s) and runs at ~1-2 Hz inside silences, so CPU is a real
        // option where it would not be for a per-frame model. Set use_gpu=true again only after the
        // cuBLASLt loading problem is understood — a crash loop costs more than the latency.
        rc::onnx::append_gpu_providers(*opts_, cfg_.use_gpu, cfg_.use_trt, "[DoorSpecialist]");
        session_ = std::make_unique<Ort::Session>(*env_, cfg_.model_path.c_str(), *opts_);

        Ort::AllocatorWithDefaultOptions alloc;
        input_name_  = session_->GetInputNameAllocated(0, alloc).get();
        output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

        // The model names its own classes; a config copy of them could drift out of step with the export.
        try
        {
            auto meta = session_->GetModelMetadata();
            if (auto v = meta.LookupCustomMetadataMapAllocated("names", alloc); v)
            {
                const std::string s{v.get()};
                std::vector<std::string> parsed;
                for (std::size_t i = 0; i < s.size(); )
                {
                    const auto q1 = s.find('\'', i);
                    if (q1 == std::string::npos) break;
                    const auto q2 = s.find('\'', q1 + 1);
                    if (q2 == std::string::npos) break;
                    parsed.push_back(s.substr(q1 + 1, q2 - q1 - 1));
                    i = q2 + 1;
                }
                if (not parsed.empty())
                    class_names_ = std::move(parsed);
            }
        }
        catch (const std::exception&) { /* names are a convenience; the ids still work */ }

        std::println("[DoorSpecialist] loaded {} ({} classes)", cfg_.model_path, class_names_.size());
        return true;
    }
    catch (const std::exception& e)
    {
        std::println("[DoorSpecialist] disabled — failed to load {}: {}", cfg_.model_path, e.what());
        session_.reset();
        return false;
    }
}

std::vector<DoorDetection> DoorSpecialist::detect(const cv::Mat& bgr) const
{
    std::vector<DoorDetection> out;
    if (not ready() or bgr.empty() or bgr.type() != CV_8UC3)
        return out;

    const int S = cfg_.input_size;
    // Letterbox: preserve aspect, pad to square. The offsets are kept so boxes can be mapped back —
    // getting this wrong shifts every box by the pad and looks like a calibration error downstream.
    const float scale = std::min(static_cast<float>(S) / bgr.cols, static_cast<float>(S) / bgr.rows);
    const int new_w = static_cast<int>(std::round(bgr.cols * scale));
    const int new_h = static_cast<int>(std::round(bgr.rows * scale));
    const int pad_l = (S - new_w) / 2, pad_t = (S - new_h) / 2;

    cv::Mat resized, canvas(S, S, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(pad_l, pad_t, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> input(static_cast<std::size_t>(3) * S * S);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < S; ++y)
        {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            float* dst = input.data() + (static_cast<std::size_t>(c) * S + y) * S;
            for (int x = 0; x < S; ++x)
                dst[x] = row[x][c] / 255.0f;
        }

    try
    {
        const std::array<std::int64_t, 4> shape{1, 3, S, S};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(),
                                                      shape.data(), shape.size());
        const char* in_names[]  = {input_name_.c_str()};
        const char* out_names[] = {output_name_.c_str()};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        if (outputs.empty())
            return out;

        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto dims = info.GetShape();
        // end2end layout: [1, N, 6] = x1, y1, x2, y2, score, class — already NMS'd inside the graph.
        if (dims.size() != 3 or dims[2] < 6)
            return out;
        const float* d = outputs[0].GetTensorData<float>();
        const std::int64_t n = dims[1], stride = dims[2];

        for (std::int64_t i = 0; i < n; ++i)
        {
            const float* row = d + i * stride;
            const float score = row[4];
            // The rows are score-sorted, so the first one below the floor ends the useful list.
            if (not std::isfinite(score) or score < cfg_.score_floor)
                break;
            // Undo the letterbox. Clamped to the frame: a box straddling the border is still a door,
            // and dropping it would silently bias the specialist against edge-of-frame doors — exactly
            // the framing that gets more common as the robot closes on one.
            const float x1 = (row[0] - pad_l) / scale, y1 = (row[1] - pad_t) / scale;
            const float x2 = (row[2] - pad_l) / scale, y2 = (row[3] - pad_t) / scale;
            const int cx1 = std::clamp(static_cast<int>(std::lround(x1)), 0, bgr.cols - 1);
            const int cy1 = std::clamp(static_cast<int>(std::lround(y1)), 0, bgr.rows - 1);
            const int cx2 = std::clamp(static_cast<int>(std::lround(x2)), 0, bgr.cols - 1);
            const int cy2 = std::clamp(static_cast<int>(std::lround(y2)), 0, bgr.rows - 1);
            if (cx2 <= cx1 or cy2 <= cy1)
                continue;
            const int cls = static_cast<int>(std::lround(row[5]));
            DoorDetection det;
            det.bbox = cv::Rect(cx1, cy1, cx2 - cx1, cy2 - cy1);
            det.confidence = score;
            det.class_id = cls;
            det.label = (cls >= 0 and cls < static_cast<int>(class_names_.size()))
                            ? class_names_[static_cast<std::size_t>(cls)] : std::to_string(cls);
            out.push_back(std::move(det));
        }
    }
    catch (const std::exception& e)
    {
        std::println("[DoorSpecialist] inference failed: {}", e.what());
    }
    return out;
}

}   // namespace rc::doors
