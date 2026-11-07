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
    float geo_mu1_sd = 2.0f;                     // μ1 = μ0 + geo_mu1_sd · s (learned online)
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

/// ONLINE DETECTOR LEARNING (spec §8.4): ONE TRIAL PER VIEW, never per frame — an EM step. A view of camera c is one trial: y = the
/// fraction of its frames with an associated detection (UNWEIGHTED: the association weight already scales hit_llr), g = its mean
/// front-end factor q / hit_rate (= env · aspect · vis · κ). Whether the trial was a trial of the detector ON A DOOR or on clutter
/// is not known; its RESPONSIBILITY is the place's existence posterior r = σ(L) (fix round 2, ruling). So each trial counts r into
/// the hit-rate pool and 1 − r into the false-alarm pool — a miss on a door that probably does not exist says almost nothing about
/// the hit rate, and nothing is cut off:
///     hit_rate    = (h0 · n0 + Σ r y) / (n0 + Σ r g),            n0 = the loaded hit_rate row's n   (absent / 0: 1 view)
///     false_alarm = (f0 · m0 + Σ (1−r) y) / (m0 + Σ (1−r) g),    m0 = the loaded false_alarm row's n (absent / 0: 1 view)
/// A view's trial is REPLACED as its frames accumulate (the ledger's own rule) and re-weighted whenever r changes; a REMOVED door's
/// trials are frozen at its last r (it keeps counting, it stops changing); a merged duplicate's are dropped. Pending candidates
/// contribute nothing. Frames whose q is unknown (a ricoh with no model: q → 0⁺) are no trials. Sums are incremental: table() is
/// O(1). An estimate outside the detector's domain (0 < false_alarm < hit_rate < 1) is not adopted — a probability domain, not a gate.
class DetectorLearner
{
public:
    void set_prior(const DetectorTable& t) { prior_ = t; dirty_ = true; }
    /// One frame of camera `cam` joining `view` of door `place`: hit = a detection was associated; g = q / hit_rate.
    void frame(std::uint64_t place, int cam, std::size_t view, bool hit, float g)
    {
        auto& P = places_[place];
        Trial& t = P.trials[{cam, view}];
        add(cam, t, P.r, -1.0);
        t.n += 1.0; t.hits += hit ? 1.0 : 0.0; t.sg += g;
        add(cam, t, P.r, +1.0);
        dirty_ = true;
    }
    /// The place's existence responsibility r = σ(L) (re-weights every trial it has).
    void set_responsibility(std::uint64_t place, double r)
    {
        auto it = places_.find(place);
        if (it == places_.end() or it->second.r == r) return;
        for (const auto& [k, t] : it->second.trials) { add(k.first, t, it->second.r, -1.0); add(k.first, t, r, +1.0); }
        it->second.r = r;
        dirty_ = true;
    }
    /// The door was REMOVED: its trials stay in both pools at its last responsibility, frozen.
    void retire(std::uint64_t place) { places_.erase(place); }
    /// The place is gone without a verdict (a merged duplicate): its trials are dropped.
    void forget(std::uint64_t place)
    {
        auto it = places_.find(place);
        if (it == places_.end()) return;
        for (const auto& [k, t] : it->second.trials) add(k.first, t, it->second.r, -1.0);
        places_.erase(it);
        dirty_ = true;
    }
    [[nodiscard]] static double prior_weight(float n) { return n > 0.0f ? static_cast<double>(n) : 1.0; }
    [[nodiscard]] const DetectorTable& table() const
    {
        if (not dirty_) return table_;
        table_ = prior_;
        for (int c = 0; c < 2; ++c)
        {
            const DetectorModel& p = c == 0 ? prior_.zed : prior_.ricoh;
            DetectorModel m = p;
            const double n0 = prior_weight(p.n_hit_trials), m0 = prior_weight(p.n_fa_trials);
            m.hit_rate = static_cast<float>((p.hit_rate * n0 + hit_[c].y) / (n0 + hit_[c].g));
            m.n_hit_trials = static_cast<float>(n0 + hit_[c].g);
            m.false_alarm = static_cast<float>((p.false_alarm * m0 + fa_[c].y) / (m0 + fa_[c].g));
            m.n_fa_trials = static_cast<float>(m0 + fa_[c].g);
            if (detector_in_domain(m)) (c == 0 ? table_.zed : table_.ricoh) = m;
        }
        dirty_ = false;
        return table_;
    }
    /// Σ y and Σ g of one live place's trials of camera `cam`, unweighted (for tests and the readout).
    [[nodiscard]] std::pair<double, double> place_sums(std::uint64_t place, int cam) const
    {
        double y = 0, g = 0;
        if (const auto it = places_.find(place); it != places_.end())
            for (const auto& [k, t] : it->second.trials) if (k.first == cam) { y += t.y(); g += t.g(); }
        return {y, g};
    }
    [[nodiscard]] double n_views(int cam) const
    {
        double n = 0;
        for (const auto& [id, P] : places_) for (const auto& [k, t] : P.trials) if (k.first == cam) n += P.r;
        return n;
    }
    [[nodiscard]] std::size_t n_places() const { return places_.size(); }
private:
    struct Trial
    {
        double n = 0, hits = 0, sg = 0;
        [[nodiscard]] double y() const { return n > 0 ? hits / n : 0.0; }
        [[nodiscard]] double g() const { return n > 0 ? sg / n : 0.0; }
    };
    struct Place { double r = 0.0; std::map<std::pair<int, std::size_t>, Trial> trials; };
    struct Sum { double y = 0.0, g = 0.0; };
    void add(int cam, const Trial& t, double r, double sgn)
    {
        hit_[cam].y += sgn * r * t.y(); hit_[cam].g += sgn * r * t.g();
        fa_[cam].y += sgn * (1.0 - r) * t.y(); fa_[cam].g += sgn * (1.0 - r) * t.g();
    }
    DetectorTable prior_ = default_detectors();
    std::map<std::uint64_t, Place> places_;
    std::array<Sum, 2> hit_{}, fa_{};
    mutable DetectorTable table_ = default_detectors();
    mutable bool dirty_ = true;
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
