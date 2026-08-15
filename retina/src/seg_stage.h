#pragma once

// Instance-segmentation stage → PerceptionResult::masks. Wraps a YoloProcessor (own ONNX session).
// Handles both the ZED frame (detect_segmentation) and the ricoh 360 panorama (detect_segmentation_360,
// 3-strip) based on PerceptionFrame::is_360.

#include "perception_stage.h"

namespace rc
{

class SegStage : public Stage
{
public:
    // Configures its own YoloProcessor. cfg360 is used only for is_360 frames. ready() reflects load.
    explicit SegStage(const YoloProcessor::Config& cfg, const Detection360Config& cfg360 = {});

    const char* name() const override { return "seg"; }
    bool ready() const override { return ready_; }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    // Draw detections onto a canvas (ricoh popup / viewer) — passthrough to the owned processor.
    cv::Mat compose(const cv::Mat& rgb, const std::vector<SegDetection>& dets) const
    { return yolo_.compose_detection_canvas(rgb, dets); }

private:
    YoloProcessor      yolo_;
    Detection360Config cfg360_;
    bool               ready_ = false;
};

} // namespace rc
