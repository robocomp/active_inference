#pragma once

// Pull source for the ricoh 360 PerceptionWorker. Each call polls the panorama media plane, dedups on the
// source stamp, and (if new) builds a 360 PerceptionFrame: BGR panorama (popup-native; SegStage converts
// to RGB for the model), plus room_T_ricoh resolved at the panorama stamp (ts!=0 → no InnerEigenAPI cache;
// own instance) with the az-tune applied. Runs on the worker thread.

#include "perception_stage.h"   // rc::PerceptionFrame

#include <cstdint>
#include <memory>
#include <optional>

class SceneProcessor;
namespace DSR { class DSRGraph; class InnerEigenAPI; }

namespace rc
{

class RicohSource
{
public:
    RicohSource(SceneProcessor* scene, std::shared_ptr<DSR::DSRGraph> graph, float azimuth_tune_deg);
    ~RicohSource();

    std::optional<PerceptionFrame> operator()();

private:
    SceneProcessor*                     scene_;
    std::shared_ptr<DSR::DSRGraph>      graph_;
    std::unique_ptr<DSR::InnerEigenAPI> inner_eigen_;   // own instance (ts!=0 use only)
    float                               azimuth_tune_deg_;
    std::uint64_t                       last_stamp_ = 0;
};

} // namespace rc
