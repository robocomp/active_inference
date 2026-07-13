#include "semantic_stage.h"

#include <algorithm>
#include <print>

namespace rc
{

SemanticStage::SemanticStage(const rc::semantic::YoloSemanticProcessor::Config& cfg, int decimation)
    : decimation_(std::max(1, decimation))
{
    try
    {
        sem_ = std::make_unique<rc::semantic::YoloSemanticProcessor>();
        sem_->configure(cfg);
    }
    catch (const std::exception& e)
    {
        std::println("[SemanticStage] disabled — failed to load model {}: {}", cfg.model_path, e.what());
        sem_.reset();
    }
}

void SemanticStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (!ready() || in.rgbd.rgb.empty())
        return;
    const bool run_now = (counter_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (run_now)
        (void) sem_->segment(in.rgbd.rgb);   // refreshes last_map_ (reused between runs)
    out.semantic = sem_->last_map();
    out.semantic_fresh = run_now;
}

} // namespace rc
