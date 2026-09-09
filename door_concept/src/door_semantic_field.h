/*
 * door_semantic_field.h — read retina's published GRADED CLASS POSTERIOR and sample it under the door's
 * own projected silhouette.
 *
 * WHY THIS EXISTS.
 *
 * The removal channel's only question is "could this look have resolved whether the door is there", and
 * until now it answered with geometry alone: resolvability x in_fov x central, all three of which IMPROVE
 * as a face-on door gets closer. Measured over one approach (198 distinct viewpoints, 2026-09-08) the
 * detector does the opposite — it fires 90% of the time at roi_fill 0.20-0.25 and 23% at 0.30-0.35 — so
 * absence was charged ~4x harder than the truth exactly where the door was hardest to see, and a door in
 * plain view was deleted at 4 m.
 *
 * The dense argmax label map cannot help: a door losing to `wall` by 0.02 and one the model never saw are
 * byte-identical in it. retina now publishes the posterior the argmax discards (semantic_class_probs, K
 * planes at the classifier's native resolution), and THIS is what a concept agent needs, because it can be
 * sampled exactly where the agent believes its own object is — which retina cannot do, since it does not
 * know where anybody believes anything.
 *
 * ★WHAT IS SAMPLED, AND WHY IT IS A CONTRAST AND NOT AN ABSOLUTE.
 * At 4.2 m the field reads P(door)=0.265 against P(wall)=0.676 over a plainly visible door: the classifier
 * is not undecided, it is confidently wrong. So a rule of the form "P(door) is high ⇒ believe" would fail
 * on the very case that motivated this. What survives is WHERE the door-ness is: if the projected contour
 * is markedly more door-like than the rest of the image, that is evidence the door is where the belief
 * says, even while the argmax calls it a wall. The statistic is therefore
 *
 *      contrast = mean P(class) over the contour  vs  mean P(class) over the whole field
 *
 * a likelihood ratio between "door-ness is concentrated here" and "door-ness is spread uniformly". It
 * needs no threshold and no calibration to have the right SIGN.
 *
 * ★NOT A TOP-DOWN LOOP. The belief chooses WHERE to sample a field retina computed without any knowledge
 * of it; it cannot change what the field says, and a contour over a genuinely doorless wall gets a
 * contrast at or below 1 and is refuted. This is the distinction that the removed table loop failed: there
 * the belief suppressed evidence before the estimator saw it, so it could never be contradicted.
 *
 * Read-only. Nothing here writes to the graph.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <dsr/api/dsr_api.h>

namespace rc
{

// ADE20K-150 class id for `door`. ★A constant here rather than a config key on purpose: retina publishes
// the ids alongside the data (semantic_prob_class_ids) and refresh() matches against THOSE, so this is only
// the question being asked, never a second copy of the answer. A model that stopped exposing door would
// make refresh() return false rather than silently read a neighbouring class.
inline constexpr int door_semantic_class_id() { return 14; }

class SemanticProbField
{
public:
    // Pull the latest field off the 'semantic' node. Returns false (and invalidates) when the node, the
    // attributes, or the class this agent follows are absent — which is the ungraded-model case, and must
    // leave every caller behaving exactly as it did before this channel existed.
    bool refresh(DSR::DSRGraph* G, int class_id);

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] std::uint64_t stamp_ms() const noexcept { return stamp_ms_; }

    // P(class) at a FRAME pixel, nearest-cell. The planes map onto the same frame as semantic_labels, so
    // this is a straight scale — the same convention retina's own sampler uses (yolo_semantic.h prob_at).
    // Returns a negative value when the pixel is outside the frame or the field is invalid, so a caller
    // can tell "not measured" from "measured low" without a sentinel that looks like a probability.
    [[nodiscard]] float at(int u, int v) const noexcept;

    // Mean P(class) over the ENTIRE field — the background this channel's contrast is taken against.
    // Computed once per refresh, not per door: it is a property of the frame.
    [[nodiscard]] float background_mean() const noexcept { return bg_mean_; }

    [[nodiscard]] int frame_w() const noexcept { return frame_w_; }
    [[nodiscard]] int frame_h() const noexcept { return frame_h_; }

private:
    bool               valid_ = false;
    std::vector<float> plane_;            // the followed class's plane only, pw_ * ph_
    int                pw_ = 0, ph_ = 0;  // native (classifier) resolution
    int                frame_w_ = 0, frame_h_ = 0;
    float              bg_mean_ = 0.0f;
    std::uint64_t      stamp_ms_ = 0;
};

}   // namespace rc
