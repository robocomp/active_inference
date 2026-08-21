#pragma once

// Pull source for the ZED PerceptionWorker. Each call drains the timestamp-aligned RGBD off the media
// plane (via SceneProcessor → MediaPlaneSource/ZedFrameAligner), dedups on the capture stamp, and builds
// a PerceptionFrame: the RGBD (rgb already an owned clone) + room_T_zed resolved at the capture stamp
// (ts!=0 → no InnerEigenAPI cache; own instance). Runs on the worker thread — moves the RGBD drain +
// transform off the main compute tick.

#include "perception_stage.h"   // rc::PerceptionFrame

#include <chrono>
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
    // Loss probe (see the note in operator()). Counts, per 5 s: frames actually taken, polls that found
    // nothing new, and grabs PASSED OVER by the latest-wins aligner — the three numbers that say whether
    // the missing third is the worker being busy or the buffer not committing.
    std::uint64_t probe_hits_ = 0, probe_polls_idle_ = 0, probe_skipped_ = 0;
    std::uint64_t probe_jump_sum_ = 0, probe_jump_max_ = 0;
    std::chrono::steady_clock::time_point probe_last_{};
};

} // namespace rc
