#include "depth_stage.h"

#include <algorithm>
#include <print>

namespace rc
{

DepthStage::DepthStage(const depth::DepthProcessor::Config& cfg, const depth::Depth360Config& cfg360,
                       int decimation)
    : cfg360_(cfg360), decimation_(std::max(1, decimation))
{
    depth_.configure(cfg);
    ready_ = depth_.ready();
    if (not ready_)
        std::println("[DepthStage] disabled — model {} did not load", cfg.model_path);
    else
        std::println("[DepthStage] ready — {} strips, overlap {} px, band ±{:.0f}° , decimation {}",
                     cfg360_.n_strips, cfg360_.overlap_px, cfg360_.band_half_elev_deg, decimation_);
}

void DepthStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (not ready_ or in.rgbd.bgr.empty())
        return;
    if (not in.is_360)
        return;   // ZED has measured metric depth already — see the header.

    const bool run_now = (counter_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (run_now)
        last_map_ = depth_.estimate_360(in.rgbd.bgr, cfg360_);   // panorama is carried BGR

    if (last_map_.empty())
        return;

    // DEEP-COPY across the worker→main boundary (CLAUDE.md cv::Mat rule, same as SemanticStage):
    // last_map_ is reused between decimated runs, so a shallow copy would hand the main thread a
    // handle onto a buffer this thread overwrites on the next inference.
    out.depth = depth::DepthMap{ last_map_.log_depth.clone(), last_map_.strip_id.clone(),
                                 last_map_.n_strips, last_map_.band_y0, last_map_.band_y1 };
    out.depth_fresh = run_now;
}

}   // namespace rc
