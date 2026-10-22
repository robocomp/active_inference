/*
 * door_frame_evidence.h — POSITIVE existence evidence from the doorway FRAME (two jambs + the lintel).
 *
 * ★WHY. Every existence channel door_concept had looked at the LEAF: the mask silhouette (marginalised
 * over phi) and the RGB/depth contour of the projected leaf face. The leaf is the one part of a door that
 * MOVES. Once it swings far enough — and above all once it swings OUT of the room the robot is in — the
 * leaf is behind the wall plane, seen at best edge-on through the gap, and every leaf channel goes silent
 * or reads absence. Live try32 (2026-10-12): the crossed door, carried into room_2 with its leaf at ~95°
 * opening OUT of room_2, went L +3.6 -> -4.0 in ~60 evidence cycles and room_2 lost its only door, while
 * the camera was looking straight at a plainly visible doorway (retina/etc/zed_frames/1791834215054.jpg).
 *
 * The APERTURE is the part that does not move. Its frame — the two jamb edges and the lintel edge, the
 * aperture rectangle's boundary in the wall plane minus the sill — is an OCCLUDING boundary between the
 * wall and whatever is in the hole, at a place the belief predicts exactly (inst.aperture: wall_O, wall_u,
 * s, w, h, floor_z). It is in the image whether the leaf is shut, open into the room or open out of it.
 *
 * ★THE MODEL (per frame sample i, a point P_i on a jamb or the lintel, its image normal n_i pointing INTO
 * the aperture):
 *
 *   visibility    v_i  = [P_i projects in the image] · [no wall crosses the sightline] · [no non-door mask
 *                        covers it] · [no other door in front] · (1 − P_φ(own leaf crosses the sightline))
 *                        — the last factor marginalised over the SAME phi posterior the silhouette uses, so
 *                        a leaf swung toward the camera hides the jamb behind it to exactly the degree the
 *                        belief says it does. Nothing is gated: v_i is a weight.
 *
 *   RGB (relative) g_i  = across-boundary gradient at P_i (rc::edges::across_boundary_gradient_at — the
 *                        identical arithmetic the leaf contour uses), and g_ik the same at P_i slid along
 *                        the WALL by k·w (AWAY from the hole for a jamb — kFrameJambOutward — and ±kControlOffsets for the
 *                        lintel). Sliding along the wall is a valid null for
 *                        the frame at ANY phi, because the frame lies IN the wall plane — the defect that
 *                        made the leaf contour's wall-slid null invalid (a swung leaf leaves the plane) does
 *                        not exist here.
 *                        excess = (Σ v_i g_i / Σ v_i − Σ v_i c_ik g_ik / Σ v_i c_ik) / frame_ref
 *
 *   depth (relative) R_i = Φ((d_ap − d_wall − Δ_plane) / σ_step), v^d_i = 2R_i − 1, where d_wall / d_ap are
 *                        the upper-quartile depth over a window on the WALL / APERTURE side of the edge (long
 *                        enough to reach past the proud casing and the registration error — see
 *                        build_frame_probe), and Δ_plane what the wall plane alone predicts between them. The aperture is a
 *                        HOLE: whatever is in it (the next room, a leaf swung away, a leaf recessed in the
 *                        reveal) is at or behind the wall plane, so R ≥ ½ for a real doorway and R ≈ ½ for a
 *                        blank wall. Weighted by v_i · (1 − P_φ(own leaf in FRONT of the aperture-side read)),
 *                        because a leaf swung toward the camera is the one thing that legitimately sits in
 *                        front of the hole.
 *                        ★RELATIVE ON PURPOSE. The leaf's depth half compares d_in with the PREDICTED d̂ and
 *                        is OFF in config because the ZED carries a ~20 cm common-mode bias at these ranges
 *                        (door_concept/etc/config.toml, ContourDepthCheck). A difference of two reads taken
 *                        3-6 px apart is blind to a common-mode bias, so this half does not inherit it.
 *
 *   frame verdict  f   = vis_frac · (excess + depth_verdict),   vis_frac = Σ v_i / N_attempted
 *
 * vis_frac is P(the probe could resolve the frame | it exists): a frame half out of view, or mostly hidden,
 * speaks in proportion — the lifecycle's "absence weighted by visibility" applied to this channel, with
 * walls included. N_attempted counts the WHOLE frame, so a sliver at the image border cannot speak with the
 * authority of a full view.
 *
 * ★SYMMETRIC, NOT CONFIRM-ONLY. A confirm-only channel fed zero-mean noise is a RATCHET: E[max(0, x)] > 0,
 * so a phantom on a blank wall would be pushed up a little every cycle. On a blank wall both halves are
 * zero-mean (the jamb lines and their slid copies sample the same paint; the two sides of the "edge" are at
 * the same depth), so the symmetric channel neither confirms nor refutes a phantom there — removal stays
 * with the mask silhouette, the channel with the detectability model. It CAN refute (a strong edge in a
 * control and none on the predicted frame), which is the honest reading of "the doorway is not where you
 * say it is".
 *
 * Pure: no DSR, no agent state. Projection and visibility are the caller's callbacks (the caller owns the
 * near-clip, the transform stamp and the occluder model), so this is unit-testable on saved frames —
 * door_concept/tests/test_door_frame_evidence.cpp.
 */

#pragma once

#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "door_geometry.h"
#include "../../common/contour_edge/contour_edge_check.h"
#include "../../common/contour_edge/contour_edge_project.h"
#include "../../common/contour_edge/contour_depth_check.h"

namespace rc::door
{

// One slid copy of a frame sample (the null).
struct FrameControl
{
    cv::Point2f px{};
    cv::Point2f n{};         // unit image normal (sign irrelevant for the gradient)
    float       w = 0.0f;    // its own visibility weight; 0 ⇒ not usable
};

// One point of the predicted frame, already projected.
struct FrameSample
{
    cv::Point2f px{};
    cv::Point2f n_in{};      // unit image normal pointing INTO the aperture (from the wall side)
    float       depth_m = -1.0f;   // predicted camera-forward depth (diagnostic; the depth half is relative)
    float       w_vis   = 0.0f;    // v_i  — see the header
    float       w_depth = 0.0f;    // 1 − P(own leaf in front of the aperture-side read)
    // d(forward depth)/d(pixel) of the WALL PLANE along n_in at this sample, from the model. The two depth
    // reads straddle the edge by ~(offset_px_min + offset_px_max) px, so on a wall seen obliquely they differ
    // by that much even with NO hole: the null for the step is this plane gradient, not zero.
    float       plane_ddepth_dpx = 0.0f;
    // Length (px, along n_in) of the depth half's read window on EACH side of the edge: the projected
    // casing width + registration σ (see build_frame_probe / score_frame). 0 ⇒ the bare 3-6 px offsets.
    float       side_window_px = 0.0f;
    std::array<FrameControl, rc::edges::kControlOffsets.size()> ctl{};
};

struct FrameProbe
{
    std::vector<FrameSample> samples;   // samples that projected inside the image with v_i > 0
    int   n_total = 0;                  // samples attempted along the WHOLE frame (both jambs + lintel)
    float vis_sum = 0.0f;               // Σ v_i
    [[nodiscard]] float vis_frac() const { return n_total > 0 ? vis_sum / static_cast<float>(n_total) : 0.0f; }
};

struct FrameScore
{
    float excess   = 0.0f;   // RGB: (s_true − s_control) / frame_ref, visibility-weighted. 0 if no control
    float s_true   = 0.0f, s_control = 0.0f;
    int   n_edge   = 0;      // samples scored by the RGB half (0 ⇒ NOT MEASURED)
    int   n_ctl    = 0;      // control evaluations that contributed
    float depth_verdict = 0.0f;   // mean (2R − 1), weighted. 0 if not measured
    int   n_depth  = 0;
    float vis_frac = 0.0f;   // copied from the probe
    // The frame's contribution to the image-geometry verdict that rc::exist::contour_evidence integrates.
    [[nodiscard]] float verdict() const { return vis_frac * (excess + depth_verdict); }
    [[nodiscard]] int   n_samples() const { return std::max(n_edge, n_depth); }
};

// Visibility weight in [0,1] of a room-frame point (1 = nothing between the camera and it).
using PointVisibility = std::function<float(const Eigen::Vector3f&)>;

// ★THE JAMB NULL SLIDES OUTWARD ONLY. The leaf contour's null — the same shape slid ±1.0 and ±1.6 widths
// along its plane — is right for a closed quad, but a jamb slid +1.0 w toward the hole lands EXACTLY on the
// OPPOSITE jamb: a quarter of every jamb's controls sampled the real frame, so the null contained the signal
// (live: the shut door scored t=135 against c=53; from inside room_2, seen obliquely with one jamb in view, the
// slid copy found the other, real, edge and the excess went NEGATIVE, −1.2, on a door in plain view). A jamb's
// controls go away from the hole, onto the wall beside it (clear of a casing ~0.1 w), at the same magnitudes
// where they overlap the old set. The lintel keeps the symmetric ±1.0/±1.6: slid along the wall it lands beside
// the opening, never on it.
// ★Task D-C round 2: AND BEYOND WHERE THE APERTURE MAY REALLY BE. The frame channel scores the belief's placement mixed over
// s ~ N(ŝ, σ_s²) on the Gauss-Hermite nodes ±√3σ_s (door_existence.cpp); a control nearer than that to the believed jamb can
// sit ON the real one when the placement is uncertain. Each jamb control slides kFrameJambOutward·w PLUS √3·σ_s
// (`placement_sigma_m`, the fit's own posterior along the wall): at a settled σ_s of a few cm it is the old set; at a
// restored door's 0.1-0.6 m the null moves out with the uncertainty instead of sampling the signal.
inline constexpr std::array<float, 4> kFrameJambOutward{0.4f, 0.7f, 1.0f, 1.6f};

// Sampling resolution (quadrature, not a decision constant): 30 per jamb, 14 along the lintel — the same
// densities the leaf silhouette uses vertically / horizontally.
inline constexpr int kFrameJambSamples   = 30;
inline constexpr int kFrameLintelSamples = 14;

// Build the projected frame probe. `visible(P)` is the occlusion weight of the frame point itself;
// `aperture_side_clear(P)` is 1 − P(own leaf in front of a point just inside the hole next to P).
//
// `casing_m`: width of the CASING lining the opening (DoorConcept.FrameCasingM); `registration_sigma_m`: how
// far the believed edge may sit from the real one (Existence.BoundarySigmaM); `placement_sigma_m`: the belief's own σ_s
// along the wall (the jamb controls clear √3 of it, see kFrameJambOutward). ★A DOORWAY'S EDGE IS NOT A
// KNIFE-EDGE IN THE WALL. A real opening is lined with frame boards that stand PROUD of the wall face (Webots'
// Door.proto: 5 cm boards inside a 1 m gap, 5 cm proud of both faces; a real architrave ~1-2 cm). Read 3-6 px
// inside the predicted edge, the "hole" side lands on that board — NEARER than the wall — and the step reads
// "the hole is in front of the wall": live try34, from inside room_2, dep −0.45 on a real open doorway, which
// is Φ(−0.05/0.08)·2 − 1 for a 5 cm proud board. Each side is therefore read over a window reaching past
// casing + registration error (score_frame). Both 0 ⇒ the bare 3-6 px offsets of the leaf contour check.
[[nodiscard]] FrameProbe build_frame_probe(const Aperture& ap,
                                           const rc::edges::PointProjector& project,
                                           const PointVisibility& visible,
                                           const PointVisibility& aperture_side_clear,
                                           int img_w, int img_h, float casing_m = 0.0f,
                                           float registration_sigma_m = 0.0f, float placement_sigma_m = 0.0f);

// Score a probe against one frame. `prep` may be null/invalid (RGB half not measured); `depth` may be null
// or empty (depth half not measured). `dp` supplies the side-read offsets, σ_step and the RGB→depth scale.
[[nodiscard]] FrameScore score_frame(const FrameProbe& probe,
                                     const rc::edges::PreparedFrame* prep,
                                     const cv::Mat* depth,
                                     const rc::edges::ContourDepthParams& dp = {});

// Does leaf L cross the 3-D sightline O→P before reaching P? Intersections within the leaf's own thickness
// of P do not count — that is P lying ON the leaf's edge (the hinge jamb), not the leaf standing in front.
[[nodiscard]] bool leaf_blocks_sightline(const LeafPose& L, const Eigen::Vector3f& O, const Eigen::Vector3f& P);

}   // namespace rc::door
