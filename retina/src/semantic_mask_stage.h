#pragma once

// Semantic-instance-mask stage. Runs AFTER SemanticStage (needs PerceptionResult::semantic) and BEFORE
// Sam2Stage in the worker's stage list. It turns the dense ADE20K-150 class field into per-object instance
// masks for a configured set of furniture classes (cabinet/hood/shelf/door) that YOLO-seg does NOT detect,
// scores each like a YOLO-seg mask, DROPS any region already covered by a YOLO-seg mask (YOLO is
// authoritative), and APPENDS the survivors into PerceptionResult::masks. From there they ride the
// unchanged SAM2-refine swap and the graph_publisher deprojection — same 'masks' node, same format.
//
// One mask per connected component (touching pixels of a class = one instance). Extraction only runs on a
// FRESH semantic cycle (out.semantic_fresh) so the label map matches the current frame's depth at publish.

#include "perception_stage.h"   // Stage, PerceptionFrame/Result, SegDetection, rc::semantic::SemanticMap

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rc
{

class SemanticMaskStage : public Stage
{
public:
    // `accepted_classes` = resolved (ADE20K class id, canonical label) pairs — resolved once in
    // initialize() from the model's class table, so the stage never needs the name table. `min_area_frac`
    // drops components smaller than that fraction of the frame; `overlap_drop_frac` drops a component that
    // is at least that covered by a YOLO-seg mask; `morph_kernel` (≤1 = off) denoises the class field;
    // `score_default` is the confidence used when the per-pixel semantic score map is unavailable.
    SemanticMaskStage(std::vector<std::pair<int, std::string>> accepted_classes,
                      float min_area_frac, float overlap_drop_frac, int morph_kernel, float score_default);

    const char* name() const override { return "semantic_masks"; }
    bool ready() const override { return not accepted_.empty(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

private:
    std::vector<std::pair<int, std::string>> accepted_;   // (ade_id, label)
    float          min_area_frac_    = 0.003f;
    float          overlap_drop_frac_= 0.5f;
    int            morph_kernel_     = 5;
    float          score_default_    = 0.5f;
    std::uint64_t  run_count_        = 0;   // for the throttled per-class summary log
};

} // namespace rc
