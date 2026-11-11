// existence_update.h — Task 8 of the door-existence model: ONE FRAME OF EVIDENCE INTO ONE PLACE BELIEF. The helpers
// DoorCore::update_existence (door_existence.cpp) and the birth path (door_core.cpp) share, so a pending candidate and a
// live door are judged by the same likelihoods (spec §3.6: birth and survival use the same model). DSR-free.
//
// Per place θ and fresh frame (spec §3.4/§3.5, task-8 brief):
//   · for each camera that ran (ZED: a fresh mask packet; ricoh: a fresh ricoh frame and its model): door_view →
//     detect_prob → ViewLedger::observe → add_bernoulli with the association weight of this frame's detection of θ;
//   · geometry (a fresh RGB frame, mask or not): the leaf contour's across-boundary gradient against the SAME frame's
//     control placements (slid along the leaf's own plane) → add_geometry with μ0, s² from the controls and μ1 = μ0 + d·s
//     (d = 2, the flagged starting value — no bench statistic of a true door exists in tests/data; fixed within a session);
//   · a commanded actuation seen satisfied → add_actuation once;
//   · recompute every touched view.
// Nothing here decides; door_decision.h does, on PlaceBelief::L().
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "camera_model.h"
#include "detector_model.h"
#include "door_existence.h"
#include "door_visibility.h"
#include "place_belief.h"
#include "../../../common/contour_edge/contour_edge_check.h"

namespace rc::door::core
{
/// One detection of this frame as the existence model sees it (the core's DetectionView: its room xy and variance).
struct FrameDetection
{
    Vec2f xy = Vec2f::Zero();
    float xy_var = -1.0f;      // < 0 ⇒ the classic detection noise
    int   sensor = 0;          // 0 ZED, else ricoh
    float w = 1.0f;            // its reliability: YOLO confidence x (ricoh) the wall/variance weight x P(unclaimed by another concept)
    bool admissible = true;    // a frame the fit would refuse (truncated, or moving off-centre: DoorCore::frame_admissible) — a
                               // detection there may neither confirm nor refute the place it lands on: that camera's frame is no trial
};

/// What every place shares on one fresh frame.
struct ExistenceFrame
{
    const ExistenceScene& scene;                 // the ZED scene: camera pose, intrinsics, packet, room, other doors
    const rc::DoorConfig& cfg;
    const DetectorTable& det;
    bool zed_ran = false;                        // a fresh mask packet: the ZED detector processed a frame
    bool ricoh_ran = false;                      // a fresh ricoh frame in the packet AND a ricoh model + mount
    // A fresh ricoh frame with NO ricoh model or mount (a v3/v4 tape, a graph without the node): its detections still count as hits —
    // a detection proves the door was reportable, and q cancels in a hit's ratio (R9) — but nothing is a miss (q → 0⁺: exactly 0,
    // Review Focus 4). room_T_ricoh is then the body pose (the best origin known).
    bool ricoh_hits_only = false;
    std::optional<Mat4d> room_T_ricoh;
    std::optional<CameraSpec> zed_spec, ricoh_spec;
    const rc::edges::PreparedFrame* rgb = nullptr;   // a fresh RGB frame (nullptr ⇒ no geometry this cycle)
    std::optional<Mat6d> rt_cov;
    bool geometry_may_refute = false;            // Existence.GeometryMayRefute (§8.5: false until benched)
    float geo_mu1_sd = 2.0f;                     // μ1 = μ0 + geo_mu1_sd · s (★STARTING VALUE, fixed in-session; refit offline)
    float rotation_w = 1.0f;                     // what a frame taken at this camera turn rate can say (rotation_reliability): joins kappa, for hits and misses alike
    std::span<const FrameDetection> dets;        // this frame's door detections (association weights)
    std::function<void(const std::vector<cv::Point>&)> note_poly;   // the input recorder: a polygon the geometry channel read
};

/// One frame of one camera joining one view: what etc/door_views_log.csv carries (tools/fit_detector_envelopes.py groups the
/// rows by (door, camera, view_id) into one trial per view) and what the detector learner counts.
struct ViewRow
{
    int camera = 0;            // 0 zed, 1 ricoh
    int view_id = -1;
    float q = 0.f, fill_h = 0.f, fill_v = 0.f, vis = 0.f, kappa = 0.f, aspect = 0.f;   // THIS frame's
    float hit = 0.f;           // THIS frame's association weight (0 = a miss) — the same frame as q / fill / vis
    bool q_known = true;       // false: a ricoh frame with no model (q → 0⁺): logged, never a learning trial
    float sup_x = std::numeric_limits<float>::quiet_NaN(), sup_l = std::numeric_limits<float>::quiet_NaN();   // camera == 2 (`support`): the fit's x and ℓ_sup(x)
};

/// The leaf face's projected contour + its controls in the ZED (the construction the old silhouette probe used, moved
/// out so the geometry channel runs with no mask at all). Empty face ⇒ not measurable from here.
[[nodiscard]] rc::edges::ContourSet leaf_contour(const ExistenceScene& sc, const rc::DoorFitState& inst);

/// The association weight of this frame's detections of source `sensor` to a place at xy with position variance `var`:
/// κ · max over gated detections of w · exp(−½ m²) (m² ≤ Tracker.GateMahalanobis), 0 if none. κ carries the pose: under a
/// lost pose a detection is about this place only in proportion to the image overlap (Review Focus 1).
[[nodiscard]] float association_weight(std::span<const FrameDetection> dets, int sensor, const Vec2f& xy, float var,
                                       float kappa, const rc::DoorConfig& cfg);

/// One `open` claim's evidence for the door's existence (clean5: 18 attempts at a phantom and L never moved). `requested`: our open request went out for
/// this claim. `no_door_here`: the request could not go out because nothing that can be opened is at this place. `satisfied`: the door was SEEN open.
/// An attempt the body never got to make (the claim ended unreachable with no request) says nothing about the door. One likelihood ratio, both ways.
[[nodiscard]] inline float open_attempt_llr(bool requested, bool no_door_here, bool satisfied, float llr)
{
    if (requested and satisfied) return llr;
    if (not satisfied and (requested or no_door_here)) return -llr;
    return 0.0f;
}

/// One frame into one place. `self_sensor` ≥ 0 forces the hit weight of that sensor to κ · self_w (a candidate's own detection,
/// self_w its reliability).
/// Returns the trace (NaN for every channel that did not run) and appends the touched views' rows.
ExistenceTrace observe_place(PlaceBelief& b, const rc::DoorFitState& theta, const ExistenceFrame& fr, int visit,
                             float actuation_llr, std::vector<ViewRow>* rows, int self_sensor = -1, float self_w = 0.f,
                             std::optional<FitSupport> support = std::nullopt);

/// The structural prior of θ in this room (place_prior.h). `doors_on_wall_before`: how many doors were on θ's wall before it.
/// The size is scored on the FITTED width/height with their posterior sds (a candidate: what the fit will make of its blob) under
/// `size`, the door fit's own width/height prior (one size prior for one quantity, place_prior.h). `aperture_sd`: the sd of θ's
/// aperture position along the wall normal (the band term is the marginal over it, place_prior.h).
[[nodiscard]] PlacePrior prior_of(const rc::DoorFitState& theta, const std::vector<Vec2f>& polygon, bool proto,
                                  const rc::door::RoomBand& band, int doors_on_wall_before, float width_sd, float height_sd,
                                  bool size_resolved, const SizePrior& size, float aperture_sd);

/// The wall group (place_prior.h wall_groups) θ's aperture lies on: the group of the polygon edge nearest to it; -1 = none.
[[nodiscard]] int wall_of(const Vec2f& xy, const std::vector<Vec2f>& polygon, const std::vector<int>& groups);

/// A pending candidate's θ: the typical door (`height` = the fit's height prior, leaf shut) at the blob, along the nearest wall,
/// with the blob's measured width (its extent along that wall). With no polygon the blob's own principal direction is the wall.
[[nodiscard]] rc::DoorFitState typical_door_at(const Vec2f& xy, float width, float height, const std::vector<Vec2f>& polygon);

/// ★DETECTOR RATES ARE FIXED WITHIN A SESSION (Task 8 fix round 3, ruling): update_existence and decide use the door_detectors.csv
/// values unchanged. In-session, a phantom's misses are explained equally by "door absent" and "hit rate low" — not identifiable from
/// one robot's few trials on uncertain doors — so the rates are learned OFFLINE, across sessions, by tools/fit_detector_envelopes.py
/// from the per-view rows (etc/door_views_log.csv). Nothing here learns them.

}  // namespace rc::door::core
