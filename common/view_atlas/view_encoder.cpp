#include "view_encoder.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <print>
#include <vector>

namespace rc::atlas
{
ViewEncoder::ViewEncoder(EncoderConfig cfg) : cfg_(std::move(cfg))
{
    try
    {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ViewEncoder");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(cfg_.threads);
        opts.SetGraphOptimizationLevel(cfg_.graph_opt_all ? GraphOptimizationLevel::ORT_ENABLE_ALL
                                                          : GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, cfg_.model_path.c_str(), opts);
    }
    catch (const Ort::Exception& e)
    {
        std::println("[ViewEncoder] cannot load {}: {}", cfg_.model_path, e.what());
        session_.reset();
    }
}

ViewEncoder::~ViewEncoder() = default;

std::optional<Eigen::VectorXf> ViewEncoder::encode(const cv::Mat& rgb224) const
{
    if (not session_ or rgb224.rows != 224 or rgb224.cols != 224 or rgb224.type() != CV_8UC3) return std::nullopt;
    static constexpr float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> t(3 * 224 * 224);
    for (int y = 0; y < 224; ++y)
        for (int x = 0; x < 224; ++x)
        {
            const auto p = rgb224.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c)
                t[std::size_t(c) * 224 * 224 + std::size_t(y) * 224 + std::size_t(x)] = (float(p[c]) / 255.f - mean[c]) / stdv[c];
        }
    try
    {
        const std::array<std::int64_t, 4> shape{1, 3, 224, 224};
        const auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor<float>(mem, t.data(), t.size(), shape.data(), shape.size());
        const char* in_names[] = {"pixel_values"};
        const char* out_names[] = {"embedding"};
        auto outs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        const float* o = outs[0].GetTensorData<float>();
        Eigen::VectorXf v = Eigen::Map<const Eigen::VectorXf>(o, 384);
        const float n = v.norm();
        if (not (n > 0.f)) return std::nullopt;
        return v / n;
    }
    catch (const Ort::Exception& e)
    {
        std::println("[ViewEncoder] run failed: {}", e.what());
        return std::nullopt;
    }
}
} // namespace rc::atlas
