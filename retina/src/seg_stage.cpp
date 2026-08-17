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
            // Round-robin the panorama. The window is decided HERE, once per frame, and published on the
            // shared schedule for the other panorama stages to read — see strip_schedule.h for why they
            // must not each keep their own counter.
            Detection360Config c = cfg360_;
            if (sched_)
                c.strips = sched_->advance(c.n_strips, strips_per_frame_);
                out.strips_looked = c.strips;   // snapshot for consumers off this thread (see PerceptionResult)
            return yolo_.detect_segmentation_360(rgb, c);
        }();
    }
    else
        out.masks = yolo_.detect_segmentation(in.rgbd.bgr);   // ZED frame (same BGR convention as before)
}

} // namespace rc
