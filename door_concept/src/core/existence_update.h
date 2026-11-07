// existence_update.h — Task 8 of the door-existence model: ONE FRAME OF EVIDENCE INTO ONE PLACE BELIEF. The helpers
// DoorCore::update_existence (door_existence.cpp) and the birth path (door_core.cpp) share, so a pending candidate and a
// live door are judged by the same likelihoods (spec §3.6: birth and survival use the same model). DSR-free.
//
// Per place θ and fresh frame (spec §3.4/§3.5, task-8 brief):
//   · for each camera that ran (ZED: a fresh mask packet; ricoh: a fresh ricoh frame and its model): door_view →
//     detect_prob → ViewLedger::observe → add_bernoulli with the association weight of this frame's detection of θ;
//   · geometry (a fresh RGB frame, mask or not): the leaf contour's across-boundary gradient against the SAME frame's
//     control placements (slid along the leaf's own plane) → add_geometry with μ0, s² from the controls and μ1 = μ0 + d·s
//     (d learned from doors with L > 0, starting value 2 — no bench statistic of a true door exists in tests/data);
//   · a commanded actuation seen satisfied → add_actuation once;
//   · recompute every touched view.
// Nothing here decides; door_decision.h does, on PlaceBelief::L().
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <tuple>
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
    float w = 1.0f;            // its reliability: YOLO confidence x (ricoh) the wall-agreement / range-variance weight
};

/// What every place shares on one fresh frame.
struct ExistenceFrame
{
    const ExistenceScene& scene;                 // the ZED scene: camera pose, intrinsics, packet, room, other doors
    const rc::DoorConfig& cfg;
    const DetectorTable& det;
    bool zed_ran = false;                        // a fresh mask packet: the ZED detector processed a frame
    bool ricoh_ran = false;                      // a fresh ricoh frame in the packet AND a ricoh model + mount
    std::optional<Mat4d> room_T_ricoh;
    std::optional<CameraSpec> zed_spec, ricoh_spec;
    const rc::edges::PreparedFrame* rgb = nullptr;   // a fresh RGB frame (nullptr ⇒ no geometry this cycle)
    std::optional<Mat6d> rt_cov;
    bool geometry_may_refute = false;            // Existence.GeometryMayRefute (§8.5: false until benched)
    float geo_mu1_sd = 2.0f;                     // μ1 = μ0 + geo_mu1_sd · s (learned online)
    std::span<const FrameDetection> dets;        // this frame's door detections (association weights)
    std::function<void(const std::vector<cv::Point>&)> note_poly;   // the input recorder: a polygon the geometry channel read
};

/// The per-view numbers etc/door_views_log.csv carries (tools/fit_detector_envelopes.py reads them).
struct ViewRow
{
    int camera = 0;            // 0 zed, 1 ricoh
    int view_id = -1;
    float q = 0.f, fill_h = 0.f, fill_v = 0.f, vis = 0.f, kappa = 0.f, aspect = 0.f, hit = 0.f;   // hit: the view's mean weight
};

/// The leaf face's projected contour + its controls in the ZED (the construction compute_silhouette_existence used, moved
/// out so the geometry channel runs with no mask at all). Empty face ⇒ not measurable from here.
[[nodiscard]] rc::edges::ContourSet leaf_contour(const ExistenceScene& sc, const rc::DoorFitState& inst);

/// The association weight of this frame's detections of source `sensor` to a place at xy with position variance `var`:
/// κ · max over gated detections of w · exp(−½ m²) (m² ≤ Tracker.GateMahalanobis), 0 if none. κ carries the pose: under a
/// lost pose a detection is about this place only in proportion to the image overlap (Review Focus 1).
[[nodiscard]] float association_weight(std::span<const FrameDetection> dets, int sensor, const Vec2f& xy, float var,
                                       float kappa, const rc::DoorConfig& cfg);

/// One frame into one place. `self_sensor` ≥ 0 forces the hit weight of that sensor to κ · self_w (a candidate's own detection,
/// self_w its reliability).
/// Returns the trace (NaN for every channel that did not run) and appends the touched views' rows.
ExistenceTrace observe_place(PlaceBelief& b, const rc::DoorFitState& theta, const ExistenceFrame& fr, int visit,
                             float actuation_llr, std::vector<ViewRow>* rows, int self_sensor = -1, float self_w = 0.f);

/// The structural prior of θ in this room (place_prior.h). `doors_on_wall_before`: how many doors were on θ's wall before it.
/// The size is scored on the FITTED width/height with their posterior sds (a candidate: the blob's width, the typical height).
[[nodiscard]] PlacePrior prior_of(const rc::DoorFitState& theta, const std::vector<Vec2f>& polygon, bool proto,
                                  const rc::door::RoomBand& band, int doors_on_wall_before, float width_sd, float height_sd,
                                  bool size_resolved);

/// The wall group (place_prior.h wall_groups) θ's aperture lies on: the group of the polygon edge nearest to it; -1 = none.
[[nodiscard]] int wall_of(const Vec2f& xy, const std::vector<Vec2f>& polygon, const std::vector<int>& groups);

/// A pending candidate's θ: the typical door (2.05 m tall, leaf shut) at the blob, along the nearest wall, with the
/// blob's measured width (its extent along that wall). With no polygon the blob's own principal direction is the wall.
[[nodiscard]] rc::DoorFitState typical_door_at(const Vec2f& xy, float width, const std::vector<Vec2f>& polygon);

/// ONLINE DETECTOR LEARNING (spec §8.4): ONE TRIAL PER VIEW, never per frame. A view of camera c contributes its aggregate:
/// y = its mean hit weight, g = its mean front-end factor q / hit_rate (= env · aspect · vis · κ). The hit rate is the
/// Bernoulli rate through that front end, learned from the views of doors believed in (L > 0):
///     hit_rate = (h0 · n0 + Σ y) / (n0 + Σ g),       n0 = the loaded model's n_hit_trials + 1 (a compiled default: 1 view);
/// the false-alarm rate from the views of places whose row was later REMOVED (their hits were the detector's false alarms):
///     false_alarm = (f0 + Σ y_fa) / (1 + Σ g_fa).
/// A view's trial is REPLACED as its aggregate grows (the ledger's own rule), so a parked stare stays one trial. The circularity
/// (records come from doors the same model believes in) is accepted and stated in the spec; the tests guard against drift.
/// An estimate outside the detector's domain (0 < false_alarm < hit_rate < 1) is not adopted — a probability domain, not a gate.
class DetectorLearner
{
public:
    void set_prior(const DetectorTable& t) { prior_ = t; }
    /// The view's trial for a door with L > 0 (alive = true) or not (alive = false: kept, not counted for the hit rate).
    void note(std::uint64_t place, int cam, std::size_t view, float y, float g, bool alive)
    { trials_[{place, cam, view}] = {y, g, alive}; }
    /// The place was removed: every trial it ever had becomes a false-alarm trial.
    void retire(std::uint64_t place)
    {
        for (auto it = trials_.begin(); it != trials_.end();)
            if (std::get<0>(it->first) == place) { fa_[std::get<1>(it->first)].y += it->second.y; fa_[std::get<1>(it->first)].g += it->second.g; it = trials_.erase(it); }
            else ++it;
    }
    /// The learned table (falls back per camera to the prior where an estimate leaves the domain).
    [[nodiscard]] DetectorTable table() const
    {
        DetectorTable t = prior_;
        for (int c = 0; c < 2; ++c)
        {
            double sy = 0.0, sg = 0.0;
            for (const auto& [k, v] : trials_)
                if (std::get<1>(k) == c and v.alive) { sy += v.y; sg += v.g; }
            const DetectorModel& p = c == 0 ? prior_.zed : prior_.ricoh;
            DetectorModel m = p;
            const double n0 = static_cast<double>(p.n_hit_trials) + 1.0;
            m.hit_rate = static_cast<float>((p.hit_rate * n0 + sy) / (n0 + sg));
            m.n_hit_trials = static_cast<float>(static_cast<double>(p.n_hit_trials) + sg);
            m.false_alarm = static_cast<float>((p.false_alarm + fa_[c].y) / (1.0 + fa_[c].g));
            if (detector_in_domain(m)) (c == 0 ? t.zed : t.ricoh) = m;
        }
        return t;
    }
    [[nodiscard]] double n_views(int cam) const
    { double n = 0; for (const auto& [k, v] : trials_) if (std::get<1>(k) == cam and v.alive) n += 1; return n; }
    [[nodiscard]] double n_false_alarm_g(int cam) const { return fa_[cam].g; }
private:
    struct Trial { float y = 0.f, g = 0.f; bool alive = false; };
    struct Sum { double y = 0.0, g = 0.0; };
    DetectorTable prior_ = default_detectors();
    std::map<std::tuple<std::uint64_t, int, std::size_t>, Trial> trials_;
    std::array<Sum, 2> fa_{};
};

/// The geometry channel's present-hypothesis mean, in units of the controls' spread: μ1 = μ0 + d·s. Learned from the views of
/// doors with L > 0 (their standardised statistic z = (x̄ − μ0) / s), one per view, replaced as the view grows; the starting
/// value d0 = 2 with weight 1 view (no bench statistic of a true door exists in tests/data — flagged as the starting value).
class GeometryLearner
{
public:
    void note(std::uint64_t place, std::size_t view, float z) { if (std::isfinite(z)) z_[{place, view}] = z; }
    void retire(std::uint64_t place) { std::erase_if(z_, [&](const auto& kv) { return kv.first.first == place; }); }
    [[nodiscard]] float d() const { double s = kD0; for (const auto& [k, z] : z_) s += z; return static_cast<float>(s / (1.0 + static_cast<double>(z_.size()))); }
    [[nodiscard]] std::size_t n() const { return z_.size(); }
private:
    static constexpr double kD0 = 2.0;   // ★STARTING VALUE (task-8 context: "else μ0 + 2·s"), learned online
    std::map<std::pair<std::uint64_t, std::size_t>, float> z_;
};
}  // namespace rc::door::core
