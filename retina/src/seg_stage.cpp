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
    if (!ready_ || in.rgbd.bgr.empty())
        return;
    if (in.is_360)
    {
        // Ricoh panorama is carried BGR (popup-native); the 360 detector wants RGB — convert here only.
        cv::Mat rgb;
        cv::cvtColor(in.rgbd.bgr, rgb, cv::COLOR_BGR2RGB);
        out.masks = [&]
        {
            // Round-robin the panorama: segment strips_per_frame_ strips this frame, advancing the
            // window so every direction is revisited every n_strips/strips_per_frame_ frames. A fixed
            // rotation rather than "look where the objects are" — see the note on next_strip_.
            Detection360Config c = cfg360_;
            if (strips_per_frame_ > 0 and strips_per_frame_ < c.n_strips and c.n_strips > 0)
            {
                c.strips.reserve(static_cast<std::size_t>(strips_per_frame_));
                for (int k = 0; k < strips_per_frame_; ++k)
                    c.strips.push_back((next_strip_ + k) % c.n_strips);
                next_strip_ = (next_strip_ + strips_per_frame_) % c.n_strips;
            }
            return yolo_.detect_segmentation_360(rgb, c);
        }();
    }
    else
        out.masks = yolo_.detect_segmentation(in.rgbd.bgr);   // ZED frame (same BGR convention as before)
}

} // namespace rc
