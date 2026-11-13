// door_existence.h — the EXISTENCE slice of the DoorCore: the inputs the existence update reads and the update's plumbing.
// The model itself is the generative model of docs/superpowers/plans/2026-11-07-door-existence-model.md (existence_update.h:
// one belief per door place, evidence per view; door_decision.h: the costed decision). DSR-free. What lives here:
//   · ExistenceScene: the camera, the mask packet, the room and the other doors, from the snapshot recorded at the TOP of the
//     cycle (CameraModel::room_T_zed, CameraIntrinsics) - the projection is CameraAPI::project's pinhole arithmetic, line for
//     line (camera_project below);
//   · SemanticField: the graded semantic posterior, rebuilt from the recorded door plane;
//   · ImageSource: the RGB / depth frames - live, the world stages the ingestors' frames (stage_images); on a tape, the recorded
//     gray ROI patches + the full-frame gradient reference (frame_ref) and the depth patches;
//   · ImageUseNote / ImageRoiNote: the input recorder's image ROIs (which pixels each check sampled);
//   · pose_overlap_factor: kappa, the image channels under the robot pose's uncertainty;
//   · the stdout lines and the two CSV headers (door_existence_log.csv, door_views_log.csv).
// (The old silhouette / doorway-frame / obliquity probes that fed the pre-model absence channel are gone.)
#pragma once
#include "../../../common/detectability/detectability.h"   // rc::detect — the shared detector envelope
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
#include "../../../common/perception_card/terms.h"            // rc::card::term_* — the P(detect) factor formulas (SHARED)

namespace rc::door::core
{

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
    /// `stamp_ms` = the RGB frame's capture stamp (0 = unknown: every staging counts as a new frame).
    void stage_live(const cv::Mat& bgr, const cv::Mat& depth_m, std::uint64_t stamp_ms = 0);
    [[nodiscard]] const cv::Mat& rgb() const { return rgb_; }
    /// Increments on every NEW RGB frame (a tape's recorded frame, or a live frame with a new stamp): the existence update's
    /// geometry channel runs once per RGB frame (spec §3.4, ruling R3).
    [[nodiscard]] std::uint64_t rgb_generation() const { return rgb_gen_; }
    [[nodiscard]] const cv::Mat& depth() const { return depth_; }
    [[nodiscard]] std::optional<float> frame_ref() const { return frame_ref_; }
private:
    cv::Mat rgb_, depth_;
    std::optional<float> frame_ref_;
    std::uint64_t rgb_gen_ = 0, last_stamp_ = 0;
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
///               independent looks. It widens the view ledger's kernel (view_ledger.h) — under a 4 m spread the
///               whole room is one view, worth κ of one observation, instead of hundreds of new ones.
/// No covariance on the input ⇒ {1, 0} (the old behaviour). No gate: the covariance does it.
struct PoseOverlap { float kappa = 1.0f; float spread_m = 0.0f; };
[[nodiscard]] PoseOverlap pose_overlap_factor(const std::optional<Mat6d>& rt_cov, const std::optional<Mat4d>& room_T_zed,
                                              const Eigen::Vector2f& door_xy, float half_diag_m);

/// The header line of etc/door_existence_log.csv (the adapter writes it once, when it opens the file).
extern const char* const kExistenceCsvHeader;
/// The header line of etc/door_views_log.csv: one row per camera per frame that joins a view of a door or a pending candidate
/// (place_<key>), all fields THIS frame's (q nan = a ricoh with no model); tools/fit_detector_envelopes.py averages a view's rows.
extern const char* const kViewsCsvHeader;

}  // namespace rc::door::core
