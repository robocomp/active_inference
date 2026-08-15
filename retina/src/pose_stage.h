#pragma once

// Human-pose stage → PerceptionResult::poses (BODY_18 skeletons). Wraps a YoloHumanProcessor (own ONNX
// session). Runs every `decimation`-th frame; between runs it fills the slot with the held-last cache so
// the overlay never flickers. poses_fresh marks the cycles the model actually ran (gates skeleton publish).

#include "perception_stage.h"

#include <memory>

namespace rc
{

class PoseStage : public Stage
{
public:
    PoseStage(const rc::human_pose::YoloHumanProcessor::Config& cfg, int decimation);

    const char* name() const override { return "pose"; }
    bool ready() const override { return human_ && human_->ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    // Draw skeletons onto a canvas (viewer) — passthrough to the owned processor. null-safe via ready().
    rc::human_pose::YoloHumanProcessor* processor() { return human_.get(); }

private:
    std::unique_ptr<rc::human_pose::YoloHumanProcessor> human_;
    int           decimation_ = 1;
    std::uint64_t counter_ = 0;
};

} // namespace rc
