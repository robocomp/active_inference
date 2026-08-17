#pragma once

// Instance-segmentation stage → PerceptionResult::masks. Wraps a YoloProcessor (own ONNX session).
// Handles both the ZED frame (detect_segmentation) and the ricoh 360 panorama (detect_segmentation_360,
// 3-strip) based on PerceptionFrame::is_360.

#include <memory>

#include "perception_stage.h"
#include "strip_schedule.h"

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

    // The detector behind this stage, for the near-miss probe (see YoloSegDetector::NearMiss). Exposed
    // rather than wrapped because the probe is diagnostics: it must not grow an API of its own.
    YoloProcessor&       processor()       noexcept { return yolo_; }
    const YoloProcessor& processor() const noexcept { return yolo_; }

    // How many strips of the panorama to segment per frame. <=0 or >= n_strips = all (original
    // behaviour). 1 = pure round-robin: a third of the cost, full 360 coverage every n_strips frames.
    // `sched` is SHARED with the other panorama stages: seg runs first and ADVANCES it, everyone else
    // READS it, so the semantic labels always describe the same 60 degrees as the instance masks.
    void set_strips_per_frame(int n, std::shared_ptr<StripSchedule> sched)
    { strips_per_frame_ = n; sched_ = std::move(sched); }

private:
    YoloProcessor      yolo_;
    Detection360Config cfg360_;
    bool               ready_ = false;
    // ── ROUND-ROBIN over the panorama ────────────────────────────────────────────────────────────
    // ★A FIXED ROTATION, NOT A GREEDY ONE, AND THAT IS THE POINT. The obvious "look where the objects
    // are" starves the empty strips, and a NEW object appearing in a starved direction is precisely
    // what a peripheral channel exists to catch. A plain rotation gives every direction a guaranteed
    // revisit period (n_strips frames), which is the property the confirmation consumers need:
    // bottle/chair/door use this channel to confirm STATIC objects, so a strip that stops being
    // visited stops confirming what is in it. Opportunistic weighting can be layered on top later,
    // but only with an age term that keeps this guarantee.
    int                            strips_per_frame_ = 0;   // 0 = all
    std::shared_ptr<StripSchedule> sched_;                   // shared; seg advances it, others read
};

} // namespace rc
