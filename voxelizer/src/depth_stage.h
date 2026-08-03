#pragma once

// Monocular-depth stage → PerceptionResult::depth. Wraps a DepthProcessor (own ONNX session).
//
// 360 ONLY, on purpose. The stage runs the 3-strip equirect path and no-ops on a perspective frame:
// the ZED already has real metric depth from the media plane, so a relative-depth guess there would
// be strictly worse than the measurement it would sit next to. Read depth_processor.h before using
// the output for anything geometric — it is per-strip scale-invariant, hence DISPLAY-ONLY today.

#include "depth_processor.h"
#include "perception_stage.h"

namespace rc
{

class DepthStage : public Stage
{
public:
    DepthStage(const depth::DepthProcessor::Config& cfg, const depth::Depth360Config& cfg360,
               int decimation = 1);

    const char* name() const override { return "depth"; }
    bool ready() const override { return ready_; }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    // Overlay passthrough for the Ricoh popup. const + argument-only, like every other stage's compose:
    // the MAIN thread calls this while this worker thread may be inside run().
    [[nodiscard]] cv::Mat compose(const cv::Mat& bgr, const depth::DepthMap& map, float alpha,
                                  bool metric = false, float lo_m = 0.4f, float hi_m = 8.0f) const
    { return depth_.compose_depth_canvas(bgr, map, alpha, metric, lo_m, hi_m); }

private:
    depth::DepthProcessor  depth_;
    depth::Depth360Config  cfg360_;
    int                    decimation_ = 1;   // run the model every Nth frame; hold the last map between
    std::uint64_t          counter_    = 0;
    depth::DepthMap        last_map_;         // worker-thread only; deep-copied out in run()
    bool                   ready_      = false;
};

}   // namespace rc
