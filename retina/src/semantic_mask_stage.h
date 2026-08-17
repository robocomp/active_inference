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

    // ─── DROP-STAGE ATTRIBUTION ───────────────────────────────────────────────────────────────────
    // ★WHY. "YOLO missed it" is an inference, not an observation: a region can be correctly labelled by
    // the model and still never reach a consumer because THIS stage dropped it. Measured 2026-08-16, that
    // is exactly what happened to the hood — the ADE20K pass labelled it every time and min_area deleted
    // it, because the threshold was being taken over a canvas 3× larger than the strip inferred. Nothing
    // anywhere recorded that, so the loss was indistinguishable from a detector failure.
    // Per class, per frame: how many components the labelling produced and where each one died.
    struct ClassDrops
    {
        std::string label;
        int  n_components   = 0;   // connected components found for this class
        int  kept           = 0;
        int  dropped_area   = 0;   // below min_area (the one that bit the hood)
        int  dropped_yolo   = 0;   // ≥ overlap_drop_frac covered by a YOLO-seg mask (YOLO priority)
        int  largest_dropped_area = 0;   // how CLOSE the biggest casualty was to surviving
        bool class_skipped  = false;     // total class pixels < min_area ⇒ no CC pass at all
    };
    [[nodiscard]] const std::vector<ClassDrops>& last_drops() const noexcept { return drops_; }
    [[nodiscard]] int last_min_area() const noexcept { return last_min_area_; }
    [[nodiscard]] long long last_inferred_px() const noexcept { return last_inferred_px_; }

private:
    std::vector<ClassDrops> drops_;
    int       last_min_area_    = 0;
    long long last_inferred_px_ = 0;

    std::vector<std::pair<int, std::string>> accepted_;   // (ade_id, label)
    float          min_area_frac_    = 0.003f;
    float          overlap_drop_frac_= 0.5f;
    int            morph_kernel_     = 5;
    float          score_default_    = 0.5f;
    std::uint64_t  run_count_        = 0;   // for the throttled per-class summary log
};

} // namespace rc
