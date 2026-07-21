#pragma once

// Dense semantic-segmentation stage → PerceptionResult::semantic (ADE20K-150 per-pixel class map). Wraps
// a YoloSemanticProcessor (own ONNX session). Runs every `decimation`-th frame; reuses the held-last map
// between runs. Currently a viewer overlay only (no DSR publish yet — the Publisher gains a route later).

#include "perception_stage.h"

#include <memory>

namespace rc
{

class SemanticStage : public Stage
{
public:
    // `want_scores` fills SemanticMap::scores (per-pixel argmax confidence) — needed to score the
    // semantic-derived instance masks (SemanticMaskStage); leave false for the viewer-only path.
    SemanticStage(const rc::semantic::YoloSemanticProcessor::Config& cfg, int decimation,
                  bool want_scores = false);

    const char* name() const override { return "semantic"; }
    bool ready() const override { return sem_ && sem_->ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    // Compose the dense class-map underlay + label table for the viewer — passthrough. null-safe via ready().
    rc::semantic::YoloSemanticProcessor* processor() { return sem_.get(); }

private:
    std::unique_ptr<rc::semantic::YoloSemanticProcessor> sem_;
    int           decimation_ = 1;
    bool          want_scores_ = false;
    std::uint64_t counter_ = 0;
};

} // namespace rc
