/*
 * view_encoder.h — DINOv3 ViT-S/16 crop encoder, ONNX Runtime CPU (spec: VIEW_ATLAS.md §2). One instance
 * per agent, owned by the agent's encoder worker thread (ORT sessions are not shared across agents).
 */
#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace Ort { struct Env; struct Session; }

namespace rc::atlas
{
struct EncoderConfig
{
    std::string model_path = "models/dinov3/dinov3_vits16_crop224.onnx";
    int  threads = 2;
    bool graph_opt_all = true;   // Task 4 decides: false if ENABLE_ALL broke parity
};

class ViewEncoder
{
public:
    explicit ViewEncoder(EncoderConfig cfg);
    ~ViewEncoder();
    [[nodiscard]] bool ok() const noexcept { return session_ != nullptr; }
    /// rgb224: CV_8UC3 RGB 224x224 (make_crop().rgb224). Returns the unit 384-d embedding, or nothing.
    [[nodiscard]] std::optional<Eigen::VectorXf> encode(const cv::Mat& rgb224) const;
private:
    EncoderConfig cfg_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
};
} // namespace rc::atlas
