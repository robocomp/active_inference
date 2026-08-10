/*
 * bottle_existence.h — bottle_concept's EXISTENCE channel (the shared rc::exist policy).
 *
 * Mirrors table_existence.h. Until now bottle was the ONE object agent with no existence channel at all:
 * it removed instances on `tracker_death_frames` (a miss counter) and on divergence, which invariant 5 of
 * CONCEPT_AGENT_INVARIANTS.md forbids — "removal is a decision on L, never a miss counter". This supplies L.
 *
 * TWO CHANNELS, both reduced to the ONE policy   ΔL = p_vis · log[P(outcome|exists)/P(outcome|¬exists)]:
 *
 *   · CAMERA (mask clock) — bottle has no silhouette projector (table/refrigerator project pixel-level
 *     silhouettes; bottle never grew one), so the outcome is the coarser but honest "was a bottle mask
 *     ASSOCIATED to this instance this cycle". What makes it a likelihood ratio rather than a miss counter is
 *     the weight: p_vis is the detector's own P(detect | present, geometry) at the ACTUAL camera pose —
 *     computed with rc::nbv::predicted_fill_axes + visible_fraction + expected_p_detect, i.e. literally the
 *     same expression the epistemic planner MAXIMISES when it chooses where to look. One model, two consumers:
 *     the planner puts the stand-off at its argmax and this weights absence by it. Out of frustum, occluded, or
 *     too far to resolve ⇒ p_vis → 0 ⇒ HOLD, with no gate anywhere.
 *
 *   · LiDAR (sweep clock) — the shared rc::exist::carve_box, unchanged from table/refrigerator/residual. A
 *     beam that returns inside the cylinder confirms; one that passes through carves it away; one that stops
 *     short is occluded and says nothing. Occlusion-aware by construction, which is what makes it the right
 *     partner for a camera channel that has no occlusion test of its own beyond visible_fraction.
 *
 * ★THE ENVELOPE IS THE FLEET PRIOR AND IT IS WRONG FOR A BOTTLE — but MEASURE what that costs before
 * assuming. min_fill = 0.10 was fitted on furniture; a 7 cm-wide bottle never gets its short axis above
 * ~0.09 fill at any range, so the "enough pixels across" shoulder never saturates and p_detect is CAPPED.
 * Swept against the shipped envelope (zed 68°×41°, bottle r=3.5 cm h=20 cm on a 0.75 m table):
 *
 *     d (m)    0.4     0.6     0.8     1.0     1.5     2.0     3.0
 *     p_detect 0.026   0.267   0.359   0.312   0.252   0.226   0.201
 *     cycles   125     13      10      11      13      15      17     (L: +4 → the −1.99 removal boundary)
 *
 * So the channel is NOT the toothless "holds rather than removes" it would be easy to assume — removal is
 * comfortably REACHABLE (invariant 5's `max_absence_per_run ≥ 2·L_max` check passes at every usable range).
 * What the bad envelope actually costs is a ~3× SLOWDOWN (p_detect caps at 0.36 instead of approaching 1),
 * which the debounce inherits: `existence_remove_frames` counts IDEAL observations, so 15 of them is ~42 real
 * cycles. The 0.4 m column is the model working, not failing — at that range fill_max 0.76 is past the
 * overflow shoulder, the mask truncates, and absence is correctly worth almost nothing.
 *
 * Fix it with a config edit, not code: fit BottleModel.DetectMinFill/MaxFill/Soft from this agent's own
 * ai2_log via common/detectability/tools/fit_envelope. Invariant 6 asks for a MEASURED per-object envelope;
 * this declares that bottle's is not measured yet, and now also what that is worth in cycles.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "bottle_config.h"
#include "bottle_instance.h"

namespace rc {

class BottleFitter;

class BottleExistence
{
public:
    explicit BottleExistence(BottleConfig& cfg) : cfg_(cfg) {}

    // Everything the two channels need, gathered by the worker (which owns the graph handles and the
    // LiDAR ingestor) so this stays a pure evidence→decision step.
    struct Inputs
    {
        DSR::DSRGraph*      G           = nullptr;
        DSR::InnerEigenAPI* inner_eigen = nullptr;
        bool fresh_masks = false;                              // a mask frame arrived this cycle (camera clock)
        // Capture stamp of that mask frame. The camera pose MUST be read at this instant, not at ts=0:
        // p_detect answers "could THIS frame have resolved the bottle", so a latest-pose read judges the
        // frame from wherever the robot has since driven. 0 ⇒ no stamp published ⇒ fall back to latest.
        std::uint64_t masks_stamp_ms = 0;
        const std::vector<Eigen::Vector3f>* sweep = nullptr;   // fresh room-frame sweep, or nullptr (LiDAR clock)
        Eigen::Vector3f origin = Eigen::Vector3f::Zero();      // that sweep's sensor origin, room frame
        const std::vector<Eigen::Vector2f>* room_polygon = nullptr;   // walls occlude too (may be empty/null)
    };

    // Integrate both channels into every instance's existence log-odds and hand back the ones whose volume is
    // demonstrably empty (debounced by LOOKS, not cycles). Never deletes anything itself — the worker owns the
    // affordance→instance→node teardown ordering, exactly as it does for a tracker DEATH.
    void update_and_remove(BottleFitter& fitter, const Inputs& in,
                           const std::function<void(std::uint64_t, BottleInstance&)>& on_remove);

private:
    BottleConfig& cfg_;
};

}  // namespace rc
