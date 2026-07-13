#include "pose_stage.h"

#include <algorithm>
#include <print>

namespace rc
{

PoseStage::PoseStage(const rc::human_pose::YoloHumanProcessor::Config& cfg, int decimation)
    : decimation_(std::max(1, decimation))
{
    try
    {
        human_ = std::make_unique<rc::human_pose::YoloHumanProcessor>();
        human_->configure(cfg);
    }
    catch (const std::exception& e)
    {
        std::println("[PoseStage] disabled — failed to load model {}: {}", cfg.model_path, e.what());
        human_.reset();
    }
}

void PoseStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (!ready() || in.rgbd.rgb.empty())
        return;
    const bool run_now = (counter_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (run_now)
        (void) human_->detect_poses(in.rgbd.rgb, in.stamp);   // refreshes last_poses_ (held across misses)
    out.poses = human_->last_poses();
    out.poses_fresh = run_now;
}

} // namespace rc
