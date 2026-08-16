#pragma once

// Dense semantic segmentation of the 360 PANORAMA, one scheduled strip at a time.
//
// ★WHY IT EXISTS AT ALL. cabinet_concept and hood_concept get their labels from YOLO-sem (ADE20K,
// published as class_id = 1000 + ade_id by SemanticMaskStage), and the ricoh's YOLO-seg accepted_labels
// list contains neither "cabinet" nor "hood". So those two agents could never receive a peripheral
// detection: their ricoh code — 94 lines each, shared now, commented, maintained — was unable to fire
// even once. This is the stage that makes their peripheral channel exist.
//
// ★HOW IT AVOIDS DUPLICATING THE EXTRACTION. It does NOT produce detections. It runs the model on the
// scheduled strip and pastes that strip's label map back into a FULL-PANORAMA-SIZED map, everything
// else IGNORE_LABEL. The existing SemanticMaskStage then extracts connected components from it in
// GLOBAL panorama coordinates, and BearingStage turns those into azimuths — neither of them needing to
// know a strip was involved. One paste buys the whole downstream path unchanged.
//
// ★AND IT SHARES THE STRIP WINDOW rather than keeping its own counter. Seg advances the schedule, this
// reads it. Two private counters would start in phase and drift the moment either stage is skipped or
// decimated once, after which the semantic labels describe a different 60 degrees than the instance
// masks do — silently, with both stages looking correct in isolation. See strip_schedule.h.
//
// NOT PUBLISHED AS MASKS. The peripheral channel consumes a BEARING; a 360 mask carries no usable
// geometry (its range comes from a sparse LiDAR depth-fill at best) and every consumer already refuses
// to fit it. So the product of this stage is an azimuth, and the pixels are thrown away.

#include <memory>

#include "perception_stage.h"
#include "strip_schedule.h"

namespace rc
{

class SemanticStage360 : public Stage
{
public:
    // `sched` is the SHARED window (seg advances it, this reads it). `want_scores` fills the per-pixel
    // confidence SemanticMaskStage uses to score its components — on by default here because a
    // peripheral detection with no confidence cannot be weighted by the consumers.
    SemanticStage360(const rc::semantic::YoloSemanticProcessor::Config& cfg,
                     std::shared_ptr<StripSchedule> sched,
                     int n_strips,
                     int decimation = 1,
                     bool want_scores = true);

    const char* name() const override { return "semantic360"; }
    bool ready() const override { return sem_ && sem_->ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    rc::semantic::YoloSemanticProcessor* processor() { return sem_.get(); }

private:
    std::unique_ptr<rc::semantic::YoloSemanticProcessor> sem_;
    std::shared_ptr<StripSchedule>                       sched_;
    int           n_strips_    = 3;
    int           decimation_  = 1;
    bool          want_scores_ = true;
    std::uint64_t counter_     = 0;
};

}   // namespace rc
