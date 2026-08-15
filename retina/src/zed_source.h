#pragma once

// Pull source for the ZED PerceptionWorker. Each call drains the timestamp-aligned RGBD off the media
// plane (via SceneProcessor → MediaPlaneSource/ZedFrameAligner), dedups on the capture stamp, and builds
// a PerceptionFrame: the RGBD (rgb already an owned clone) + room_T_zed resolved at the capture stamp
// (ts!=0 → no InnerEigenAPI cache; own instance). Runs on the worker thread — moves the RGBD drain +
// transform off the main compute tick.

#include "perception_stage.h"   // rc::PerceptionFrame

#include <cstdint>
#include <memory>
#include <optional>

class SceneProcessor;
namespace DSR { class DSRGraph; class InnerEigenAPI; }

namespace rc
{

class ZedSource
{
public:
    ZedSource(SceneProcessor* scene, std::shared_ptr<DSR::DSRGraph> graph);
    ~ZedSource();

    std::optional<PerceptionFrame> operator()();

private:
    SceneProcessor*                     scene_;
    std::shared_ptr<DSR::DSRGraph>      graph_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;   // own instance (ts!=0 use only)
    std::uint64_t                       last_stamp_ = 0;
};

} // namespace rc
