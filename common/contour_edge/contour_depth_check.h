/*
 * contour_depth_check.h — does the DEPTH plane agree that the believed silhouette is a real surface,
 * standing where the belief says it stands?
 *
 * ★WHAT THIS ADDS OVER THE RGB CONTOUR CHECK, which is not "more evidence of the same kind".
 * contour_edge_check.h states its own limit plainly: it confirms "a rectangular thing with these borders
 * is here", so a poster, a door-shaped panel or a cupboard front all pass it. That limit is intrinsic —
 * intensity gradient is produced by paint as readily as by geometry. Depth is the observable that
 * separates them: a photograph of a door has the borders and none of the step.
 *
 * ★AND IT IS NOT AN EDGE DETECTOR. Running Sobel on a depth image and calling the result "a depth edge"
 * would be the same measurement again in a second modality, and it would fire on every occlusion
 * boundary in the scene. The belief predicts something far stronger than "a discontinuity here": it
 * predicts a DEPTH at every point of its own silhouette. So the question asked is the model's own —
 *
 *      is the surface at the depth I predicted, and does the world recede behind its boundary?
 *
 * Both halves are needed and neither is a threshold:
 *
 *   agreement  A = exp(−½·((d_in − d̂)/σ_d)²)     the surface is where the belief said it is
 *   recession  R = Φ((d_out − d_in)/σ_step)       what lies beyond the boundary is farther away
 *
 * with σ_d the predicted-depth uncertainty (belief position σ ⊕ a sensor term that grows with range,
 * because a stereo camera's depth error does) and σ_step the noise on a difference of two depths. The
 * product is a per-sample likelihood in [0,1]: 1 for a surface exactly where predicted with clear
 * recession behind its edge, ~0 when either half fails, and everything between graded continuously. A
 * belief that is 30 cm off does not fail a test, it scores lower — which is what lets the SAME machinery
 * confirm a good pose and refuse a bad one without a cutoff separating them.
 *
 * ★THIS CHANNEL IS ABSOLUTE, AND THAT IS WHY IT IS SHAPED DIFFERENTLY FROM THE RGB ONE.
 * The RGB check must score against a displaced copy of itself, because intensity gradient has no scale
 * of its own: a cluttered wall is full of it and a blank one has none, so only "better than its own
 * neighbours" means anything. Depth has a scale. Metres are metres, and the belief names one, so the
 * comparison is with the PREDICTION rather than with a neighbourhood.
 *
 * That difference is not cosmetic: measured on the first build of this file, a control-relative depth
 * statistic COULD NOT REFUTE. With the object gone, the believed contour scored 0 (nothing at the
 * predicted depth) — and so did every control, for the same reason — so the difference was 0.000 and
 * the channel fell silent on precisely the observation that matters most: we can see straight past the
 * place the object is supposed to be. A channel that can only confirm is a ratchet, and this file's
 * consumer (rc::exist) exists because a ratchet kept a phantom alive for 2885 cycles.
 *
 * ★THE PER-SAMPLE VERDICT, then, is signed by construction:
 *
 *      v = 2·A·R − 1        ∈ [−1, +1]
 *
 * and every case it has to separate falls out of it with nothing added:
 *
 *   surface at the predicted depth, world recedes behind the edge   A≈1 R≈1  ⇒ v ≈ +1   CONFIRM
 *   nothing there — we see the background straight through it       A≈0      ⇒ v ≈ −1   REFUTE
 *   a flat wall AT the predicted depth (a phantom drawn on it)      A≈1 R≈.5 ⇒ v ≈  0   ABSTAIN
 *   surface there, but nothing readable beyond its edge             A≈1 R=.5 ⇒ v ≈  0   ABSTAIN
 *
 * The third line is the one worth dwelling on, because it is the failure the RGB channel needs controls
 * to avoid and this one does not: a fridge-shaped region of a wall at exactly the believed distance
 * agrees perfectly on depth, and is refused anyway — not by a null, but because a thing with no space
 * behind its boundary is not a free-standing object. R does that work, per sample, with no comparison
 * region and no cutoff.
 *
 * ★CONTROLS ARE STILL SCORED, as a DIAGNOSTIC ONLY (`s_control`). They answer "would a displaced copy
 * have done as well?", which is worth having in the log when a verdict looks too good — a large surface
 * at a uniform distance is exactly the case where it would. Nothing branches on them here.
 *
 * ★NO RETURN IS NOT RECESSION. A depth image is full of holes — dark surfaces, glass, the open doorway
 * itself. It is tempting to read "nothing beyond the boundary" as "the background is infinitely far",
 * and for a doorway that is even true; but it is indistinguishable from a sensor dropout, and a channel
 * that scores its own blind spots as confirmation is a shield, not a test. An invalid d_out therefore
 * scores R = 0.5, saying nothing. An invalid d_in is not scored at all: the sample was never measured.
 */

#pragma once

#include <opencv2/core.hpp>

#include "contour_edge_project.h"

namespace rc::edges
{

struct ContourDepthScore
{
    // ★VERDICT IS THE EVIDENCE: mean of (2·A·R − 1) over the scored samples, in [−1, +1]. Signed, so it
    // can refute; bounded, so it is already on the same dimensionless axis as the RGB channel's
    // `excess` and the two may be SUMMED before the common-mode cap — see rc::exist::contour_evidence.
    // 0 means "this frame says nothing", which is a different statement from "no".
    float verdict   = 0.0f;
    float s_true    = 0.0f;   // mean A·R on the believed contour, [0,1] — the unsigned form, for reading
    float s_control = 0.0f;   // the same over the control placements. DIAGNOSTIC ONLY; nothing uses it
    float mean_bias_m = 0.0f; // mean (d_in − d̂) over scored samples: the belief's depth error, SIGNED.
                              // Diagnostic, but the informative one — a channel scoring low because the
                              // object is 40 cm nearer than believed is a fit problem, not an absence.
    int   n_samples  = 0;     // boundary samples with a usable inside-depth (0 ⇒ NOT MEASURED)
    int   n_controls = 0;     // control placements that produced at least one sample
    int   n_no_return = 0;    // samples whose OUTSIDE depth was invalid (scored neutral)
};

struct ContourDepthParams
{
    // σ_d = hypot(sigma_depth_m, sigma_depth_rel · d̂). The absolute term is the belief's own position
    // uncertainty projected on the view ray; the relative one is the stereo camera's — depth error grows
    // with range because disparity resolution does. Callers that track a real position σ should pass it.
    float sigma_depth_m   = 0.10f;
    float sigma_depth_rel = 0.02f;
    // Noise on (d_out − d_in), a difference of two independent depth reads near an edge — larger than a
    // single read's noise, and deliberately generous: the point of R is its SIGN and grading, not a
    // sharp verdict on how deep the recession is.
    float sigma_step_m    = 0.08f;
    // How far off the predicted line to read the two depths, in pixels. Must clear the projection error
    // the RGB channel also allows for (pose covariance, fitted extent, RT lag) or a correct prediction
    // 2 px off would read the object's own surface on BOTH sides and score zero recession everywhere.
    int   offset_px_min   = 3;
    int   offset_px_max   = 6;
    // Maps a contour pixel to a depth pixel, for the common case that the depth plane is delivered at a
    // different resolution than the RGB the contour was projected into. 1.0 = same frame.
    float depth_scale_x   = 1.0f;
    float depth_scale_y   = 1.0f;
};

// `depth_img`: CV_32F in METRES, with non-finite or ≤ 0 meaning no return (decode Z16 millimetres to this
// before calling — 0 there means no return and must not arrive as a valid 0.0 m). `face`/`controls` come
// from contour_edge_project.h and must carry per-vertex ranges; range is interpolated along each edge in
// INVERSE depth, which is what is linear in image space.
//
// n_samples == 0 means the contour could not be measured — outside the frame, or no valid depth along
// it. A caller must treat that as "no evidence", never as a refutation.
[[nodiscard]] ContourDepthScore contour_depth_support(const cv::Mat& depth_img,
                                                      const Contour& face,
                                                      const std::vector<Contour>& controls,
                                                      const ContourDepthParams& p = {});

}   // namespace rc::edges
