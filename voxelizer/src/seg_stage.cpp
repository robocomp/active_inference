#include "seg_stage.h"

#include <opencv2/imgproc.hpp>
#include <print>

namespace rc
{

SegStage::SegStage(const YoloProcessor::Config& cfg, const Detection360Config& cfg360)
    : cfg360_(cfg360)
{
    try
    {
        yolo_.configure(cfg);
        ready_ = true;
    }
    catch (const std::exception& e)
    {
        std::println("[SegStage] disabled — failed to load model {}: {}", cfg.model_path, e.what());
    }
}

void SegStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (!ready_ || in.rgbd.rgb.empty())
        return;
    if (in.is_360)
    {
        // Ricoh panorama is carried BGR (popup-native); the 360 detector wants RGB — convert here only.
        cv::Mat rgb;
        cv::cvtColor(in.rgbd.rgb, rgb, cv::COLOR_BGR2RGB);
        out.masks = yolo_.detect_segmentation_360(rgb, cfg360_);
    }
    else
        out.masks = yolo_.detect_segmentation(in.rgbd.rgb);   // ZED frame (same BGR convention as before)
}

} // namespace rc
