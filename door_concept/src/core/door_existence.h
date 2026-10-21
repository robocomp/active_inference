// door_existence.h — the EXISTENCE slice of the DoorCore (Task 5 of .superpowers/sdd/2026-10-21-door-core-replay):
// the per-door existence belief, updated once per NEW mask frame from the pixel silhouette (the only channel that may
// remove a door), the image-geometry evidence (the leaf contour's RGB/depth halves + the doorway frame), P(detect) and
// the two static priors (the wall band, the minimum height); then the shared removal decision. DSR-free.
//
// ★MOVED, NOT REWRITTEN. DoorCore::update_existence (door_existence.cpp) is SpecificWorker::update_existence_beliefs
// (specificworker.cpp as of 8a1abeac) and the three probes below are DoorFitter's compute_silhouette_existence /
// compute_frame_probe / door_view_obliquity / aperture_band_offset, with their ★ comments. What changed is ONLY where
// each value comes from:
//   · the mask packet: the core's MaskSource (the packet the ingestor delivered — door_cycle_io.h MaskPacket);
//   · room ← zed and the camera: the snapshot (CameraModel::room_T_zed, CameraIntrinsics) recorded at the TOP of the
//     cycle, instead of a ts=0 Interpolated cortex query made inside each probe, mid-cycle, and DSR::CameraAPI. The
//     projection is CameraAPI::project's pinhole arithmetic, line for line (camera_project below). ★A named difference,
//     the same class as Task 4's ego-motion: if the RT moved between the top of the cycle and the existence update
//     (a few ms, robot moving) the probes see the cycle-top pose;
//   · the graded semantic posterior: SemanticField below, rebuilt from the recorded door plane (the input's
//     semantic_state / semantic, captured right after the worker refreshed rc::SemanticProbField) with that class's
//     arithmetic (background mean, nearest-cell lookup);
//   · the RGB / depth frames: ImageSource below — live, the world stages the ingestors' frames (stage_images); on a tape,
//     the recorded gray ROI patches + the full-frame gradient reference (frame_ref) and the depth patches;
//   · the stdout lines and the existence CSV row: formatted here (classic locale) and handed to the diag sink; the
//     Existence / REMOVE decisions to the decision sink; the removal's graph half (remember_ghost, the affordance
//     teardown, the node delete) to the world (TrackerWorld::before_remove / remove_door);
//   · the input recorder's image ROIs (which pixels each check sampled): ImageUseNote / ImageRoiNote to the sinks, at
//     the point the old code called InputSnapshot::note_images / note_roi.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "door_cycle_io.h"
#include "../door_config.h"
#include "../door_fit_state.h"
#include "../door_room_band.h"
#include "../door_frame_evidence.h"
#include "../../../common/contour_edge/contour_edge_project.h"   // rc::edges::ContourSet

namespace rc::door::core
{

// Pixel-level silhouette existence evidence for ONE door (see compute_silhouette_existence below; moved from door_fitter.h).
// Feeds rc::exist::mask_evidence — the shared channel table/chair use — so removal is a Bayesian decision on
// P(exists), never a miss-counter or a detectability heuristic.
struct DoorSilhouette
{
    float e_occ  = 0.0f;       // predicted samples LIT by a "door" mask     ⇒ still there
    float e_free = 0.0f;       // predicted samples lit by NOTHING           ⇒ predicted-but-absent ("gone")
    int   n_total      = 0;    // silhouette samples attempted (the whole panel face)
    int   n_detectable = 0;    // samples inside the REAL camera frustum and un-occluded (0 ⇒ not probed ⇒ HOLD)
    int   n_central    = 0;    // detectable samples in the central image region (the robot is LOOKING at it)
    // Silhouette centroid accumulators (image px) + the frame size, for the size-invariant central_frac
    // below. Summed over the DETECTABLE samples only, so an occluded or off-frustum part of the object
    // cannot drag the centroid toward a region the camera never saw.
    double sum_col = 0.0, sum_row = 0.0;
    int    img_w = 0, img_h = 0;
    // ★THE BELIEF'S OWN PROJECTED CONTOUR — the four corners of the leaf face as THIS agent projected
    // them, with THIS agent's camera and transform, plus the null it is scored against and the DEPTH it
    // predicted at each corner. Retained so the contour checks run on exactly the contour the existence
    // channel is defending. retina draws a similar rectangle for the human, but from the DSR node's
    // oriented box and a transform pinned to a different stamp — close, never identical, and a defence
    // measured on a not-quite-right contour is not a defence.
    //
    // ★THE CONTROLS ARE BUILT IN 3-D, ALONG THE LEAF'S OWN PLANE, then projected — not displaced in
    // image pixels. The contour check needs a null of the form "the same shape somewhere it could
    // equally have been". Displacing along the wall IN THE IMAGE is that null only while the leaf lies
    // in the wall; once it swings, the displaced copies stop sampling wall and start sampling the open
    // doorway and the room beyond, both edge-rich, so s_true < s_control and the channel votes to DELETE
    // the door precisely when it is open (measured: dL −1.4 to −2.6 per cycle at phi 35-45°). Displacing
    // along the leaf's OWN +x (hinge → free edge) and reprojecting is the same null at phi = 0 and stays
    // valid at every other angle, so the channel never has to stand down. That removes the abstention
    // rule, which was gating on a phi that is not reliable enough to gate on.
    //
    // Empty `contour.face` when any corner fell behind the camera ⇒ NOT MEASURED, never a refutation.
    // Construction is shared: common/contour_edge/contour_edge_project.h.
    rc::edges::ContourSet contour;
    int   n_occluded   = 0;    // in-frustum samples hidden behind a nearer NON-door mask
    // ── PROVENANCE OF THE OCC/FREE SPLIT ────────────────────────────────────────────────────────────
    // Whether e_occ/e_free above are the marginal over the phi posterior or a single-angle rendering,
    // and how concentrated that posterior was. Reported because a consumer must be able to tell an
    // absence the model is confident about from one it could only average over — and because a silent
    // switch between the two would make every earlier log incomparable with every later one.
    // ★THE MARGINAL COUNTS ARE KEPT SEPARATE FROM THE MODAL ONES ON PURPOSE. `n_detectable`,
    // `n_central`, `n_cells` and the centroid sums all describe ONE rendering — the leaf at phi_est —
    // and resolvability()/central_frac() divide by them. Overwriting n_detectable with a weighted
    // average across angles while leaving n_cells and the centroid at the modal pose would put a
    // numerator and a denominator from two different measurements into one ratio, which is the exact
    // shape of error that has produced three false findings in this project. So the marginal lives in
    // its own fields and is consulted only where absence is weighed.
    bool  phi_marginalised = false;
    int   phi_n_hyp        = 0;      // opening angles carrying weight
    float phi_w_max        = 0.0f;   // largest normalised weight; ~1/n_hyp = flat, ~1 = identified
    float n_det_marg   = -1.0f;      // E[detectable samples] over the phi posterior; <0 = not computed
    float n_total_marg = -1.0f;      // E[attempted samples]  over the phi posterior; <0 = not computed
    // Did the sensor look at this door AT ALL — under any opening angle the data still permits? The
    // HOLD branch turns on this and not on the modal count: a leaf whose fitted angle happens to point
    // it out of frame has not thereby become unobserved, and charging that as "not probed" would freeze
    // a phantom just as surely as charging it as absence would delete a real door.
    bool probed() const { return phi_marginalised ? n_det_marg > 0.5f : det_w > 0.0f; }
    // Σ over the detectable samples of 1 − P(the sample lies past the room-side face, i.e. is seen THROUGH
    // the opening) (door_room_band.h). Equal to n_detectable for a leaf that stays in the room or in its own
    // aperture plane; smaller for a leaf swung OUT through the aperture, a view the silhouette model does not
    // represent. e_occ/e_free are weighted the same way, and in_fov_frac() divides THIS by n_total.
    float det_w        = 0.0f;
    int   n_cells      = 0;    // DISTINCT pixel cells the detectable silhouette covers (see resolvability)
    float mean_range_m = 0.0f; // mean camera→sample distance over the detectable samples
    // ── POSTERIOR SAMPLED UNDER THE CONTOUR (retina's semantic_class_probs) ──────────────────────
    // Accumulated over the DETECTABLE samples only: a sample the camera could not have seen carries no
    // information about what the classifier thought there. `field_n` 0 ⇒ the field was unavailable
    // (ungraded model, or retina down) and every consumer must behave exactly as it did before.
    double field_sum = 0.0;    // Σ P(door) over detectable samples
    float  field_max = 0.0f;   // max P(door) over detectable samples
    int    field_n   = 0;
    float  field_bg  = 0.0f;   // mean P(door) over the WHOLE field this frame — the comparison population
    // Mean P(door) under the contour. Absolute value is NOT the signal (the classifier reads a plainly
    // visible door at 0.265 while calling it a wall at 0.676); the CONTRAST against field_bg is.
    float field_mean() const { return field_n > 0 ? static_cast<float>(field_sum / field_n) : 0.0f; }
    // How much of the door the sensor could actually have seen from here. Absence is evidence of removal only
    // in proportion to this — the rest is epistemic surprise ("I cannot resolve this from here"), not absence.
    float in_fov_frac() const
    {
        if (phi_marginalised and n_total_marg > 0.0f)
            return std::clamp(n_det_marg / n_total_marg, 0.0f, 1.0f);
        return n_total > 0 ? det_w / static_cast<float>(n_total) : 0.0f;
    }
    // ★★CENTRALITY IS ABOUT WHERE THE OBJECT IS, NOT HOW MUCH OF IT FITS IN A BOX.
    // The old form counted the FRACTION of silhouette samples landing inside a 50%x50% central box, which
    // is a SIZE measurement wearing an attention label: an object bigger than the box can never score well
    // from ANY position. Measured on hood 2026-08-17 (n=22 323 in-view cycles):
    //     roi_fill 0.00-0.25  central_frac med 0.441
    //     roi_fill 0.25-0.50  central_frac med 0.080
    //     roi_fill 0.50-0.75  central_frac med 0.000
    // and p_detect = envelope(fill) * in_fov_frac * central_frac, where the envelope PEAKS around
    // fill 0.3-0.6 — so the two factors pull against each other, and hardest exactly where the view is
    // best. Net effect: median p_detect 0.078 while the detector actually fired on 38% of those cycles.
    // Since absence AND confirmation are both scaled by p_detect, that silenced the camera in both
    // directions and left the LiDAR carve to decide alone (ex_p 0.959 -> 0.053 on hood_1).
    // ★The size-invariant question is where the silhouette's CENTROID sits relative to the principal
    // point. chair_concept already scored it this way (1 - centroid_radius); this brings the rest into
    // line with the one that had it right, rather than inventing a sixth variant.
    // ★NOTE central_region_frac_ no longer participates: a normalised radius has no box to size.
    float central_frac() const
    {
        if (n_detectable <= 0 or img_w <= 0 or img_h <= 0) return 0.0f;
        const float cx = static_cast<float>(sum_col / n_detectable);
        const float cy = static_cast<float>(sum_row / n_detectable);
        const float xn = (cx - 0.5f * img_w) / (0.5f * img_w);   // [-1,1], 0 = principal point
        const float yn = (cy - 0.5f * img_h) / (0.5f * img_h);
        return std::clamp(1.0f - std::hypot(xn, yn), 0.0f, 1.0f);
    }
    // RESOLVABILITY ∈ (0,1]: the fraction of the sampling budget that lands in DISTINCT image cells — i.e. how
    // many independent pixels the panel subtends per unit of model. Scale-free and reference-free: it is 1 when
    // every sample resolves separately (close, face-on) and falls as samples collapse together (far away, or
    // foreshortened edge-on). This one covariate replaces the old ZedRangeFull/ZedRangeRef falloff AND the
    // obliquity ramp, with no tuning constant and — unlike that ramp, which returned exactly 0 below cos 0.20
    // and made a grazing phantom permanently unjudgeable — it never reaches zero while the door is in view.
    float resolvability() const { return n_detectable > 0 ? static_cast<float>(n_cells) / n_detectable : 0.0f; }
};

/// rc::SemanticProbField (door_semantic_field.h) over the RECORDED door plane: the same background mean and the same
/// nearest-cell lookup. Fresh replaces the plane, Unchanged keeps it, Invalid invalidates it — what refresh() left in
/// the worker's field each cycle (InputSnapshot::capture_semantic records exactly those three outcomes).
class SemanticField
{
public:
    void apply(SemanticState state, const std::optional<SemanticPlane>& plane);
    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] float at(int u, int v) const noexcept;
    [[nodiscard]] float background_mean() const noexcept { return bg_mean_; }
private:
    bool valid_ = false;
    std::vector<float> plane_;
    int pw_ = 0, ph_ = 0, frame_w_ = 0, frame_h_ = 0;
    float bg_mean_ = 0.0f;
};

/// The RGB and depth frames the existence checks score. Two feeds, one reader:
///  · LIVE — TrackerWorld::stage_images() hands the ingestors' frames (BGR, decoded depth in metres) after their pump;
///    frame_ref() is then nullopt and prepare_frame computes it from the full frame, as the old code did.
///  · TAPE — apply() rebuilds each frame from the recorded ROI patches on a canvas of the full frame's size (gray 0 /
///    depth NaN outside them) and keeps the recorded whole-frame gradient reference, which the patches alone cannot
///    give. "nullopt = unchanged" (door_cycle_io.h): a frame not recorded this cycle keeps the last one, as the ingestor
///    does. ★Exact inside the patches: the recorder crops with a margin (InputSnapshot::kRoiMarginPx) that covers the
///    blur + Sobel + sample search, so every gradient and depth the checks read equals the full frame's.
class ImageSource
{
public:
    void apply(const std::optional<ImageFrame>& rgb_gray, const std::optional<ImageFrame>& depth);
    void stage_live(const cv::Mat& bgr, const cv::Mat& depth_m);
    [[nodiscard]] const cv::Mat& rgb() const { return rgb_; }
    [[nodiscard]] const cv::Mat& depth() const { return depth_; }
    [[nodiscard]] std::optional<float> frame_ref() const { return frame_ref_; }
private:
    cv::Mat rgb_, depth_;
    std::optional<float> frame_ref_;
};

/// What the input recorder must know about the frames a cycle's checks read (InputSnapshot::note_images' arguments that
/// the core decides): whether the RGB half was prepared, its whole-frame gradient reference, whether depth was read.
struct ImageUseNote
{
    bool rgb_used = false;
    float frame_ref = 0.0f;
    bool depth_used = false;
};
/// One ROI a check sampled (InputSnapshot::note_roi): a polygon's boundary, or one pixel, with the check's extra reach.
struct ImageRoiNote
{
    std::vector<cv::Point> poly;           // empty ⇒ `point`
    cv::Point2f point{0.f, 0.f};
    int  extra_px = 0;
    bool rgb = false, depth = false;
};

/// DSR::CameraAPI::project for a PINHOLE camera, line for line: centre = width / 2 (unsigned, stored as float, passed
/// on as int), u = fx·X/Y + cx, v = −fy·Z/Y + cy (Y forward, Z up). Doubles where CameraAPI computes in double.
[[nodiscard]] inline Eigen::Vector2d camera_project(const CameraIntrinsics& c, const Eigen::Vector3d& p)
{
    const float centre_x = static_cast<float>(c.width / 2u), centre_y = static_cast<float>(c.height / 2u);
    const int cx = static_cast<int>(centre_x), cy = static_cast<int>(centre_y);
    Eigen::Vector2d proj;
    proj << c.focal_x * p.x() / p.y() + cx, -c.focal_y * p.z() / p.y() + cy;
    return proj;
}

/// What the probes read besides the door itself: the camera, the mask packet, the room, the other doors.
struct ExistenceScene
{
    const rc::DoorConfig& cfg;
    std::optional<Mat4d> room_T_zed;                       // nullopt ⇒ no camera pose ⇒ nothing projectable
    const CameraIntrinsics& cam;
    const MaskPacket& pkt;
    const std::vector<Eigen::Vector2f>& polygon;           // the containment polygon (< 3 vertices = none)
    rc::door::RoomBand band;
    std::span<const std::pair<std::uint64_t, rc::DoorFitState*>> doors;   // every live instance (occluders)
};

/// The camera is usable: present, a pinhole (the zed — CameraAPI's 360 models are not reproduced here) of non-zero size.
[[nodiscard]] bool camera_usable(const CameraIntrinsics& c);
[[nodiscard]] DoorSilhouette compute_silhouette_existence(const ExistenceScene& sc, const rc::DoorFitState& inst,
                                                          const SemanticField* field);
[[nodiscard]] rc::door::FrameProbe compute_frame_probe(const ExistenceScene& sc, const rc::DoorFitState& inst);
[[nodiscard]] float door_view_obliquity(const std::optional<Mat4d>& room_T_zed, const rc::DoorFitState& inst);
[[nodiscard]] float aperture_band_offset(const std::vector<Eigen::Vector2f>& polygon, const rc::DoorFitState& inst);

/// ★Task D-C (rtD1i door_5): THE DOOR'S IMAGE CHANNELS UNDER THE ROBOT POSE'S UNCERTAINTY. Every image channel predicts
/// WHERE the door is in the frame from the robot↔room pose; under a bearing error e (rad) the prediction is shifted by e,
/// and a window of angular width a = 2·atan(half-diagonal / range) still overlaps the true door by max(0, 1 − |e|/a). With
/// e ~ N(0, σ_b²), σ_b² = σ_yaw² + σ_xy²/range² (σ_xy² the largest eigenvalue of the published RT covariance's position
/// block — conservative, frame-free):
///   kappa     — the expected overlap, closed form (2Φ(a/σ_b) − 1) − (2σ_b/(a√(2π)))·(1 − e^{−a²/(2σ_b²)}): 1 for a
///               well-localised robot, → a/(σ_b√(2π)) → 0 as the pose spreads. It multiplies P(detect) and the image
///               evidence: a look is about the door only in that proportion.
///   spread_m  — σ_b·range, the lateral spread of the door's predicted place. The pose error is COMMON MODE: one wrong pose
///               is wrong for every frame taken under it, so frames whose viewpoints differ by less than this spread are not
///               independent looks. It widens the view pools' novelty scale (door_view_pool.h) — under a 4 m spread the
///               whole room is one view, worth κ of one observation, instead of hundreds of new ones.
/// No covariance on the input ⇒ {1, 0} (the old behaviour). No gate: the covariance does it.
struct PoseOverlap { float kappa = 1.0f; float spread_m = 0.0f; };
[[nodiscard]] PoseOverlap pose_overlap_factor(const std::optional<Mat6d>& rt_cov, const std::optional<Mat4d>& room_T_zed,
                                              const Eigen::Vector2f& door_xy, float half_diag_m);

/// The header line of etc/door_existence_log.csv (the adapter writes it once, when it opens the file).
extern const char* const kExistenceCsvHeader;

}  // namespace rc::door::core
