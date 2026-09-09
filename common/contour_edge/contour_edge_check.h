/*
 * contour_edge_check.h — does the IMAGE agree that an object's projected silhouette is there?
 *
 * WHY THIS IS DIFFERENT FROM EVERYTHING ELSE WE TRIED.
 *
 * Every existence channel in this agent's history ultimately asks a CLASSIFIER: the ADE20K argmax, then
 * its recovered posterior, then a fine-tuned door detector. All three answer "what class is this?", and
 * the first two go blind together because the second is derived from the first — measured 2026-09-09,
 * P(door) over a plainly visible closed door falls 0.995 → 0.138 → 0.048 as the robot closes, while the
 * network calls it `wall` at 0.676.
 *
 * This asks a question no classifier is involved in: the belief predicts a quadrilateral in the image,
 * and a real object bounded by that quadrilateral must produce INTENSITY GRADIENT along it, oriented
 * across it. A door's jamb, lintel and leaf edge are in the RGB whatever a network chooses to call those
 * pixels. So a successful boundary match is evidence of the same kind a mask is — it says something is
 * there, with that shape, at that place — and it survives exactly the situation that kills the semantic
 * channels.
 *
 * ★THE SCORE IS AGAINST A LOCAL CONTROL, NOT AN ABSOLUTE. Raw edge energy is meaningless on its own: a
 * cluttered wall is full of gradient and a blank one has none, so a fixed cutoff would be a per-scene
 * constant in disguise. The same silhouette is therefore re-scored at CONTROL positions — displaced
 * sideways along the wall by roughly its own width — and the statistic is how much better the believed
 * position does than its own neighbours in the same frame, same exposure, same wall:
 *
 *      support = s_true / (s_true + s_control)      0.5 = no better than a displaced copy
 *
 * That is the scale the posterior version lacked. There it was taken against the WHOLE frame, which is
 * mostly floor and ceiling where P(door) ≈ 0.015, so the ratio saturated at 0.98 for any contour that
 * touched anything door-ish and the damping was effectively always maximal. A local control cannot
 * saturate that way: the neighbouring wall is as edge-rich as the door's surroundings.
 *
 * ★AND IT MUST BE ABLE TO FAIL. A contour over blank wall scores at or below its controls and is
 * refuted. That is what makes this a test rather than a shield, and it is the property the belief-driven
 * channels must have to be allowed at all — the belief chooses WHERE to look, the image decides what is
 * found.
 *
 * ⚠WHAT IT CANNOT DO. It confirms "a rectangular thing with these borders is here", not "a door is
 * here". A door-shaped panel, a poster, a cupboard front will all pass. It is deliberately a weaker
 * claim than a classifier's, and it is the right one for EXISTENCE — the question the removal channel
 * actually asks is whether the object is still there, not what it is called.
 */

#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace rc::edges
{

struct ContourEdgeScore
{
    // ★EXCESS IS THE EVIDENCE; `support` is only for reading. (s_true - s_control) in units of the
    // frame's own mean gradient. A DIFFERENCE, not a ratio, because a ratio deliberately discards
    // magnitude — and that cost a door on 2026-09-09: on one frame s_true collapsed 51.5 -> 5.2 while
    // the controls also fell (9.7 -> 7.8), so a near-blank region produced support = 0.400, which the
    // consumer read as a confident refutation and deleted a door in plain view. 5.2 vs 7.8 is not a
    // refutation, it is an absence of measurement wearing one's clothes.
    // Dividing by the FRAME's mean gradient makes it dimensionless and self-calibrating: it scales with
    // the scene's own contrast, so a dim corridor and a bright room are read on the same axis without
    // any tuned constant. Same event as excess = -0.26: weak, correctly.
    float excess    = 0.0f;
    float support   = 0.5f;   // s_true / (s_true + s_control); 0.5 = indistinguishable. DIAGNOSTIC ONLY.
    float s_true    = 0.0f;   // mean across-boundary gradient on the believed contour
    float s_control = 0.0f;   // same, averaged over the control placements
    float frame_ref = 0.0f;   // mean |grad| over the whole frame — the scale `excess` is expressed in
    int   n_samples = 0;      // boundary samples actually scored (0 ⇒ nothing measured, NOT "no support")
    int   n_controls = 0;     // control placements that survived (fell inside the frame). 0 ⇒ no comparison
};

// `gray`: CV_8UC1 or CV_8UC3 (converted internally). `poly`: the projected silhouette, in image pixels,
// as a closed polygon (4 points for a door face). `controls`: the same polygon at displaced positions.
// Returns n_samples == 0 when the contour is too small or entirely outside the frame — a caller must
// treat that as "not measured" and leave its belief untouched.
[[nodiscard]] ContourEdgeScore contour_edge_support(const cv::Mat& gray,
                                                    const std::vector<cv::Point>& poly,
                                                    const std::vector<std::vector<cv::Point>>& controls);

// Build control polygons by shifting `poly` sideways by ±k widths of its own bounding box. Sideways
// only: a door's neighbours along the wall are the honest comparison, while shifting vertically would
// straddle floor and ceiling and compare against a different kind of surface entirely.
[[nodiscard]] std::vector<std::vector<cv::Point>> make_side_controls(const std::vector<cv::Point>& poly,
                                                                     int frame_w, int frame_h);

}   // namespace rc::edges
