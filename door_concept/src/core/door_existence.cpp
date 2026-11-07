// door_existence.cpp — see door_existence.h. The probes are DoorFitter's (door_fitter.cpp as of 8a1abeac) and
// DoorCore::update_existence is SpecificWorker::update_existence_beliefs (specificworker.cpp as of 8a1abeac), moved
// with their comments; only the sources of the values and the destinations of the outputs changed (door_existence.h).
#include "door_existence.h"
#include "door_core.h"
#include "door_visibility.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <locale>
#include <numbers>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/imgproc.hpp>
#include <Eigen/Eigenvalues>

#include "../../../common/instance_tracker/birth_evidence.h"   // rc::birth::observation_weight (SHARED): the ricoh channel's P(detect)
#include "../door_geometry.h"
#include "../door_leaf_visibility.h"                     // leaf_closedness / ricoh_detect_scale (Task D2)
#include "../door_phi_posterior.h"                                // room_walls_beside, phi_field_nuisance (one own-wall definition)
#include "../../../common/occlusion/occlusion.h"                 // rc::occlusion::{cone_blocks, walls_block} (SHARED)
#include "../../../common/contour_edge/contour_edge_check.h"     // rc::edges::PreparedFrame / contour_edge_support
#include "../../../common/contour_edge/contour_depth_check.h"    // rc::edges::contour_depth_support
#include "../../../common/detectability/detectability.h"         // rc::detect — the YOLO inverse model (SHARED)
#include "../../../common/existence_belief/existence_belief.h"   // rc::exist (SHARED)

namespace rc::door::core
{
namespace
{
using K = rc::door::DoorDecisionKind;

// MaskCells, build_mask_cells and own_wall_edges live in door_visibility.{h,cpp} (shared with door_view).

}  // namespace

// ★Task 8 (spec §3.8): ONE ROW PER DOOR PER FRESH FRAME, every channel's contribution THIS cycle; `nan` = not computed this
// cycle (never a stale value). L = prior_band + prior_base + prior_size + carried + crossing + ledger. zed_dl / ric_dl / geo_dl /
// act_dl: what each channel changed the ledger sum by this cycle; view_id / ric_view_id: the ledger record the frame joined;
// zed_q / ric_q: P(detector reports the door from this view); zed_hit / ric_hit: this frame's association weight (0 = a miss);
// geo_x / geo_mu0 / geo_mu1 / geo_s: the contour statistic, the controls' mean, the present-hypothesis mean and the controls' sd.
const char* const kExistenceCsvHeader =
    "cycle,node,L,p_exists,cx,cy,role,delete_line,view_id,ric_view_id,"
    "prior_band,prior_base,prior_size,carried,crossing,ledger,n_views,"
    "zed_dl,ric_dl,geo_dl,act_dl,zed_q,ric_q,zed_hit,ric_hit,kappa,geo_x,geo_mu0,geo_mu1,geo_s\n";

// Ruling R8: the per-view rows tools/fit_detector_envelopes.py reads (its column names, exactly). Every field is
// THIS frame's (hit = its association weight, 0 = a miss); door = the node name or place_<key> of a pending candidate.
const char* const kViewsCsvHeader = "cycle,door,camera,view_id,q,fill_h,fill_v,vis,kappa,aspect,hit\n";

// ─── SemanticField: rc::SemanticProbField over the recorded plane ───────────────────────────────────────────────────

void SemanticField::apply(SemanticState state, const std::optional<SemanticPlane>& plane)
{
    if (state == SemanticState::Unchanged)
        return;                                   // the field the worker holds is the last refresh's
    valid_ = false;
    if (state != SemanticState::Fresh or not plane.has_value())
        return;
    const auto& p = plane.value();
    if (p.pw <= 0 or p.ph <= 0 or p.frame_w <= 0 or p.frame_h <= 0
        or p.plane.size() < static_cast<std::size_t>(p.pw) * static_cast<std::size_t>(p.ph))
        return;
    plane_ = p.plane;
    // Background = the whole frame's mean door-ness. NaN cells are "not looked at" (a 360 strip the
    // scheduler skipped) and are excluded rather than counted as zero (SemanticProbField::refresh, unchanged).
    double sum = 0.0;
    int    n   = 0;
    for (const float v : plane_)
        if (std::isfinite(v)) { sum += v; ++n; }
    if (n == 0)
        return;
    bg_mean_ = static_cast<float>(sum / n);
    pw_ = p.pw; ph_ = p.ph; frame_w_ = p.frame_w; frame_h_ = p.frame_h;
    valid_ = true;
}

float SemanticField::at(int u, int v) const noexcept
{
    if (not valid_ or u < 0 or v < 0 or u >= frame_w_ or v >= frame_h_)
        return -1.0f;
    // Frame pixel → native cell, cell centres, nearest (SemanticProbField::at, unchanged).
    const int cx = std::clamp(static_cast<int>((static_cast<float>(u) + 0.5f) * pw_   // clamp: pixel index kept inside the image array (a domain guard), not a threshold
                                               / static_cast<float>(frame_w_)), 0, pw_ - 1);
    const int cy = std::clamp(static_cast<int>((static_cast<float>(v) + 0.5f) * ph_   // clamp: pixel index kept inside the image array (a domain guard), not a threshold
                                               / static_cast<float>(frame_h_)), 0, ph_ - 1);
    return plane_[static_cast<std::size_t>(cy) * static_cast<std::size_t>(pw_) + static_cast<std::size_t>(cx)];
}

// ─── ImageSource ─────────────────────────────────────────────────────────────────────────────────────────────────────

void ImageSource::apply(const std::optional<ImageFrame>& rgb_gray, const std::optional<ImageFrame>& depth)
{
    if (rgb_gray.has_value() and rgb_gray->format == PixelFormat::Gray8 and rgb_gray->width > 0 and rgb_gray->height > 0)
    {
        cv::Mat canvas(rgb_gray->height, rgb_gray->width, CV_8UC1, cv::Scalar(0));
        for (const auto& p : rgb_gray->patches)
        {
            if (p.w <= 0 or p.h <= 0 or p.bytes.size() < static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h)) continue;
            const cv::Rect r = cv::Rect(p.x0, p.y0, p.w, p.h) & cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (r != cv::Rect(p.x0, p.y0, p.w, p.h)) continue;
            cv::Mat src(p.h, p.w, CV_8UC1, const_cast<std::uint8_t*>(p.bytes.data()));
            src.copyTo(canvas(r));
        }
        rgb_ = canvas;
        frame_ref_ = rgb_gray->frame_ref;
        ++rgb_gen_;
    }
    if (depth.has_value() and depth->format == PixelFormat::DepthF32 and depth->width > 0 and depth->height > 0)
    {
        cv::Mat canvas(depth->height, depth->width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
        for (const auto& p : depth->patches)
        {
            if (p.w <= 0 or p.h <= 0
                or p.bytes.size() < static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h) * sizeof(float)) continue;
            const cv::Rect r = cv::Rect(p.x0, p.y0, p.w, p.h) & cv::Rect(0, 0, canvas.cols, canvas.rows);
            if (r != cv::Rect(p.x0, p.y0, p.w, p.h)) continue;
            cv::Mat src(p.h, p.w, CV_32FC1);
            std::memcpy(src.data, p.bytes.data(), static_cast<std::size_t>(p.w) * static_cast<std::size_t>(p.h) * sizeof(float));
            src.copyTo(canvas(r));
        }
        depth_ = canvas;
    }
    // (DepthZ16 is never written: the recorder stores the ingestor's DECODED frame, DepthF32.)
}

void ImageSource::stage_live(const cv::Mat& bgr, const cv::Mat& depth_m, std::uint64_t stamp_ms)
{
    if (not bgr.empty() and (stamp_ms == 0 or stamp_ms != last_stamp_)) ++rgb_gen_;
    last_stamp_ = stamp_ms;
    rgb_ = bgr;
    depth_ = depth_m;
    frame_ref_.reset();
}

// ─── the pose-uncertainty overlap (Task D-C, door_existence.h) ─────────────────────────────────────────────────────

PoseOverlap pose_overlap_factor(const std::optional<Mat6d>& rt_cov, const std::optional<Mat4d>& room_T_zed,
                                const Eigen::Vector2f& door_xy, float half_diag_m)
{
    if (not rt_cov.has_value() or not room_T_zed.has_value() or not rt_cov->allFinite())
        return {};
    const Eigen::Vector2d cam(room_T_zed->coeff(0, 3), room_T_zed->coeff(1, 3));
    const double range = std::max(1e-2, (door_xy.cast<double>() - cam).norm());
    const Eigen::Matrix2d Pxy = rt_cov->topLeftCorner<2, 2>();
    const double var_xy = std::max(0.0, Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(Pxy).eigenvalues().maxCoeff());
    const double var_b = std::max(0.0, rt_cov->coeff(5, 5)) + var_xy / (range * range);
    if (not (var_b > 0.0))
        return {};
    const double sb = std::sqrt(var_b);
    return {rc::card::term_kappa(var_b, range, static_cast<double>(half_diag_m)), static_cast<float>(sb * range)};
}

// ─── the probes (DoorFitter's, moved) ────────────────────────────────────────────────────────────────────────────────

bool camera_usable(const CameraIntrinsics& c)
{
    return c.present and c.projection_model == 0 /*CameraAPI::ProjectionModel::Pinhole*/ and c.width > 0 and c.height > 0;
}

// ═══ THE UPDATE (Task 8 of the door-existence model: rewritten, no branch of the old function ported) ═══════════════
// ONE belief per door place: L = prior + carried + crossing + Σ_views ℓ_v (place_belief.h). Per fresh frame (a new mask packet,
// a fresh ricoh frame in it, or a new RGB frame — ruling R3) every door's and every pending candidate's views receive this
// frame through the SAME likelihoods (existence_update.h), and the costed decision (door_decision.h) creates and deletes.

namespace
{
std::string fmt_or_dash(float v) { return std::isfinite(v) ? std::format("{:+.2f}", v) : std::string("—"); }
const char* role_name(Role r) { return r == Role::BeingCrossed ? "crossing" : r == Role::Route ? "route" : "ordinary"; }
}  // namespace

std::vector<FrameDetection> DoorCore::frame_detections() const
{
    std::vector<FrameDetection> v;
    v.reserve(dets_.size());
    for (std::size_t i = 0; i < dets_.size(); ++i)
        v.push_back({dets_[i].xy, dets_[i].xy_var, dets_[i].sensor, i < det_w_.size() ? det_w_[i] : 1.0f,
                     i < det_adm_.size() ? det_adm_[i] != 0 : true});
    return v;
}

void DoorCore::with_frame(const DoorCycleInput& in, bool zed_ran, bool use_rgb, const std::function<void(const ExistenceFrame&)>& f)
{
    const auto& cfg = cfg_.door;
    const auto& pkt = masks_.packet();
    const auto states = world_->fit_states();
    const std::optional<Mat4d> room_T_zed = CameraModel::room_T_zed(in.robot);
    const ExistenceScene scene{cfg, room_T_zed, in.camera, pkt, fit_.room_polygon(), fit_.room_band(), states};
    rc::edges::PreparedFrame rgb_prep;
    if (use_rgb and cfg.rgb_contour_check and not images_.rgb().empty())
    {
        rgb_prep = rc::edges::prepare_frame(images_.rgb());
        if (images_.frame_ref().has_value()) rgb_prep.frame_ref = images_.frame_ref().value();
    }
    if (sinks_.image_use) sinks_.image_use({rgb_prep.valid(), rgb_prep.frame_ref, false});
    const std::vector<FrameDetection> dets = frame_detections();
    const DetectorTable& det = detectors_;   // fixed within the session (fix round 3)
    ExistenceFrame fr{scene, cfg, det};
    if (camera_usable(in.camera)) fr.zed_spec = camera_spec(in.camera);
    fr.zed_ran = zed_ran and fr.zed_spec.has_value();
    if (zed_ran and in.ricoh.present and in.ricoh.width > 0 and in.ricoh.height > 0 and ricoh_fresh(pkt)
        and in.robot.room_T_body.has_value() and in.robot.body_T_ricoh.has_value())
    {
        fr.ricoh_ran = true;
        fr.ricoh_spec = camera_spec(in.ricoh);
        fr.room_T_ricoh = *in.robot.room_T_body * *in.robot.body_T_ricoh;
    }
    else if (zed_ran and ricoh_fresh(pkt) and in.robot.room_T_body.has_value())
    {
        fr.ricoh_hits_only = true;
        fr.room_T_ricoh = *in.robot.room_T_body;
    }
    fr.rgb = rgb_prep.valid() ? &rgb_prep : nullptr;
    fr.rt_cov = in.robot.rt_cov;
    fr.geometry_may_refute = cfg.exist_geometry_may_refute;
    // fr.geo_mu1_sd stays at its starting value (μ1 = μ0 + 2s): fixed within the session like the detector rates (fix round 3)
    fr.dets = dets;
    if (sinks_.image_roi)
        fr.note_poly = [this](const std::vector<cv::Point>& poly) { sinks_.image_roi({poly, {}, 0, true, false}); };
    f(fr);
}

// The per-frame view rows to etc/door_views_log.csv (one row per camera per frame joining a view; the fit groups them by view).
void DoorCore::log_view_rows(const std::string& place, const std::vector<ViewRow>& rows, DoorCycleOutput& out)
{
    for (const auto& r : rows)
    {
        std::ostringstream row;
        row.imbue(std::locale::classic());
        row << ex_dbg_ << ',' << place << ',' << (r.camera == 0 ? "zed" : "ricoh") << ',' << r.view_id << ','
            << (r.q_known ? r.q : std::numeric_limits<float>::quiet_NaN()) << ',' << r.fill_h << ',' << r.fill_v << ',' << r.vis << ','
            << r.kappa << ',' << r.aspect << ',' << r.hit << '\n';
        diag("door_views_log.csv", row.str(), out);
    }
}

// C_keep's samples (spec §8.2): an executing episode on one of a door's affordances that ends without the robot crossing it.
// An episode spans the cycles in which ANY of the door's aff_<door>_* affordances is executed; it is crossed when the door is
// backed by a crossing fact while it lasts, or the room changes under it.
void DoorCore::track_episodes(const DoorCycleInput& in)
{
    std::set<std::string> now;
    for (const auto& e : in.executing)
        for (const auto& a : in.affordances)
            if (a.node_id == e.to and a.name.starts_with("aff_door_"))
                if (const auto k = a.name.rfind('_'); k != std::string::npos and k > 4) now.insert(a.name.substr(4, k - 4));
    const auto room = world_->room_id();
    for (const auto& n : now)
        episodes_.try_emplace(n, Episode{in.t_wall_ms, false, room});
    for (auto& [name, ep] : episodes_)
    {
        if (ep.room != room) ep.crossed = true;
        for (const auto& [id, st] : world_->fit_states())
            if (st->node_name == name and std::ranges::contains(in.backed_faces, id)) ep.crossed = true;
    }
    for (auto it = episodes_.begin(); it != episodes_.end();)
        if (not now.contains(it->first))
        {
            if (not it->second.crossed and in.t_wall_ms > it->second.start_wall_ms)
                costs_.add(CostEstimator::Kind::Keep, static_cast<float>(in.t_wall_ms - it->second.start_wall_ms) * 1e-3f, "online");
            it = episodes_.erase(it);
        }
        else ++it;
}

// The candidates (spec §3.6, birth by comparison): every birthable detection that no live door's gate explains belongs to a
// pending place; each candidate in the room receives this frame like a door does, and its verdict is the birth evidence.
void DoorCore::prepare_candidates(const DoorCycleInput& in, const std::vector<rc::TrackView>& tracks, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    det_candidate_.assign(dets_.size(), -1);
    for (auto& d : dets_) d.birth_evidence = 0.0f;
    if (not info_.new_frame)
        return;
    const auto gated = [&](const Vec2f& xy, float var)
    {
        for (const auto& t : tracks)
        {
            const Vec2f e = xy - t.xy;
            if (t.has_cov)
            {
                const Eigen::Matrix2f S = t.cov + (var >= 0.0f ? var : cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m)
                                                  * Eigen::Matrix2f::Identity();
                if (const float m2 = e.dot(S.inverse() * e); std::isfinite(m2) and m2 <= cfg.tracker_gate_mahalanobis) return true;
            }
            else if (e.squaredNorm() <= cfg.tracker_gate_fallback_m * cfg.tracker_gate_fallback_m) return true;
        }
        return false;
    };
    // a candidate a live door now explains is that door (born, restored or carried meanwhile): it is no longer pending
    std::erase_if(candidates_, [&](const Candidate& c) { return c.room == room_name_ and gated(c.xy, c.var); });
    const auto& pkt = masks_.packet();
    const auto& polygon = fit_.room_polygon();
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    std::unordered_map<std::size_t, int> self_det;   // candidate index → the detection that is its own this frame
    for (std::size_t d = 0; d < dets_.size(); ++d)
    {
        const auto& dv = dets_[d];
        if (not dv.birthable or gated(dv.xy, dv.xy_var)) continue;
        const float var = dv.xy_var >= 0.0f ? dv.xy_var : rn2;
        int best = -1; float best_m2 = cfg.tracker_gate_mahalanobis;
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            if (candidates_[c].room != room_name_) continue;
            const float m2 = (candidates_[c].xy - dv.xy).squaredNorm() / std::max(1e-9f, candidates_[c].var + var);
            if (m2 <= best_m2) { best_m2 = m2; best = static_cast<int>(c); }
        }
        if (best < 0)
        {
            // A new place: the typical door at the blob, with the blob's width along its wall (its support's extent).
            const rc::DoorFitState probe = typical_door_at(dv.xy, cfg.door_prior_w_m, polygon);
            const Vec2f u = probe.aperture.wall_u;
            float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
            for (const auto& p : masks_.support(pkt.slices[static_cast<std::size_t>(dv.slice_index)]))
            { const float a = Vec2f(p.x(), p.y()).dot(u); lo = std::min(lo, a); hi = std::max(hi, a); }
            const float width = hi > lo ? hi - lo : cfg.door_prior_w_m;
            Candidate c;
            c.key = ++next_candidate_key_;
            c.room = room_name_;
            c.xy = dv.xy;
            c.var = var;
            c.theta = typical_door_at(dv.xy, width, polygon);
            // ★The memory of this place is the candidate's starting evidence (spec §3.7/§8.3): a matched row continues its
            //  ledger, aged by the change hazard; a REMOVED row ("no door here") pushes it down by exactly its aged L.
            if (const PlaceRow* row = places_.match(room_name_, dv.xy, Eigen::Matrix2f::Identity() * var, cfg.tracker_gate_mahalanobis))
            {
                c.belief.ledger = row->ledger;
                c.belief.carried = places_.aged_L(*row, in.t_wall_ms) - row->ledger.sum();
                c.seeded_from_removed_ms = row->removed_wall_ms;
                // A REVISIT of a remembered place (the hazard is learned from these): a detection at a place remembered as a door is
                // "found"; at a place remembered as "no door here" the world has CHANGED.
                if (row->last_seen_wall_ms > 0)
                    places_.note_revisit(not row->removed_wall_ms.has_value(), double(in.t_wall_ms - row->last_seen_wall_ms) * 1e-3);
            }
            candidates_.push_back(std::move(c));
            best = static_cast<int>(candidates_.size() - 1);
        }
        det_candidate_[d] = best;
        self_det.try_emplace(static_cast<std::size_t>(best), static_cast<int>(d));
    }
    // every pending candidate in this room receives the frame (hits from its own detection, misses where it should be seen)
    const auto groups = wall_groups(polygon, fit_.room_band().sigma_m);
    std::unordered_map<int, int> doors_on_wall;
    for (const auto& [id, st] : world_->fit_states())
        if (not st->is_bearing_hypothesis) ++doors_on_wall[wall_of(st->aperture.centre_xy(), polygon, groups)];
    const Costs costs = costs_.costs(cfg.exist_max_room_s);
    std::vector<char> exists(candidates_.size(), 0);
    with_frame(in, true, false, [&](const ExistenceFrame& fr)
    {
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            Candidate& cand = candidates_[c];
            if (cand.room != room_name_) continue;
            const int wall = wall_of(cand.theta.aperture.centre_xy(), polygon, groups);
            cand.belief.prior = prior_of(cand.theta, polygon, fit_.room_is_proto(), fit_.room_band(),
                                         wall >= 0 ? doors_on_wall[wall] : 0, cfg.tracker_detection_noise_m, 0.0f, true);
            const auto it = self_det.find(c);
            const int self_sensor = it == self_det.end() ? -1 : (dets_[static_cast<std::size_t>(it->second)].sensor != 0 ? 1 : 0);
            const float self_w = it == self_det.end() ? 0.0f : det_w_[static_cast<std::size_t>(it->second)];
            std::vector<ViewRow> rows;
            observe_place(cand.belief, cand.theta, fr, visit_, 0.0f, &rows, self_sensor, self_w);
            log_view_rows(std::format("place_{}", cand.key), rows, out);   // logged (no selection by birth), never learned from
            const Verdict v = decide(cand.belief.L(), Role::Ordinary, costs);
            exists[c] = v.exists ? 1 : 0;
            if (it != self_det.end() and (cand.belief.ledger.views().size() <= 1 or ex_dbg_ % 60 == 0))   // its first views, then 1 in 60
                diag("stdout", std::format("door_concept: [candidate] place_{} at ({:.2f},{:.2f}) w={:.2f} L={:+.2f} (band {:+.2f} base {:+.2f} "
                                           "size {:+.2f} carried {:+.2f} views {}) -> {}\n", cand.key, cand.xy.x(), cand.xy.y(),
                                           cand.theta.aperture.w, cand.belief.L(), cand.belief.prior.band, cand.belief.prior.base_rate,
                                           cand.belief.prior.size, cand.belief.carried, cand.belief.ledger.views().size(),
                                           v.exists ? "EXISTS" : "not born"), out);
            // ★A pending candidate below the line is simply NOT BORN (fix round 1, review I-3): it has no node to remove, so it
            //  writes no REMOVED row, books no C_delete and retires no false-alarm trials. Only a removed DOOR does those.
        }
    });
    // ruling R2: the model decides. ONE detection per candidate carries its verdict; the candidate's other detections this frame (the
    // other camera, a split mask) may not seed a second birth of the same place (the tracker's promoted candidates are not in its
    // same-frame duplicate test), so they are not birthable this frame — they still associate.
    for (std::size_t d = 0; d < dets_.size(); ++d)
        if (const int c = det_candidate_[d]; c >= 0)
        {
            const bool own = self_det.at(static_cast<std::size_t>(c)) == static_cast<int>(d);
            dets_[d].birth_evidence = own and exists[static_cast<std::size_t>(c)] ? 1.0f : 0.0f;
            if (not own) dets_[d].birthable = false;
        }
}

void DoorCore::update_existence(const DoorCycleInput& in, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    const auto& pkt = masks_.packet();
    const bool zed_fresh = pkt.valid and static_cast<int>(pkt.frame_id) != exist_last_mask_frame_;
    const bool rgb_fresh = cfg.rgb_contour_check and not images_.rgb().empty() and images_.rgb_generation() != last_rgb_gen_;
    if (not zed_fresh and not rgb_fresh)
        return;
    if (zed_fresh) exist_last_mask_frame_ = static_cast<int>(pkt.frame_id);
    last_rgb_gen_ = images_.rgb_generation();
    ++ex_dbg_;

    // The door being crossed (the controller's `executing` names its cross) or backed by a crossing fact: Role::BeingCrossed —
    // its evidence still accumulates, only its deletion is barred (spec §3.3; the decision rule carries the protection).
    std::set<std::string> being_crossed;
    for (const auto& e : in.executing)
        for (const auto& a : in.affordances)
            if (a.node_id == e.to and a.name.starts_with("aff_") and a.name.ends_with("_cross"))
                being_crossed.insert(a.name.substr(4, a.name.size() - 4 - 6));

    const auto states = world_->fit_states();
    const auto& polygon = fit_.room_polygon();
    const auto groups = wall_groups(polygon, fit_.room_band().sigma_m);
    for (const auto& [id, st] : states) door_seq_.try_emplace(id, next_door_seq_++);
    const Costs costs = costs_.costs(cfg.exist_max_room_s);
    std::vector<std::uint64_t> to_remove;
    std::vector<float> L_before(states.size());
    with_frame(in, zed_fresh, rgb_fresh, [&](const ExistenceFrame& fr)
    {
        std::vector<int> wall(states.size(), -1);
        for (std::size_t i = 0; i < states.size(); ++i)
            wall[i] = wall_of(states[i].second->aperture.centre_xy(), polygon, groups);
        for (std::size_t i = 0; i < states.size(); ++i)
        {
            const auto& [id, instp] = states[i];
            rc::DoorFitState& inst = *instp;
            inst.exist_trace = {};
            L_before[i] = inst.belief.L();
            if (inst.is_bearing_hypothesis) continue;   // no depth: no place to judge
            inst.role = door_role(inst.node_name, in.route_faces, being_crossed.contains(inst.node_name) or inst.fact_backed);
            // the structural prior, REPLACED (never accumulated); a newborn not yet fitted keeps its candidate's
            if (inst.ai2_initialized)
            {
                int before = 0;
                for (std::size_t j = 0; j < states.size(); ++j)
                    if (j != i and wall[j] == wall[i] and wall[i] >= 0 and not states[j].second->is_bearing_hypothesis
                        and door_seq_[states[j].first] < door_seq_[id])
                        ++before;
                const auto C = inst.ai2_belief.covariance();
                inst.belief.prior = prior_of(inst, polygon, fit_.room_is_proto(), fit_.room_band(), before,
                                             std::sqrt(std::max(0.0f, C(1, 1))), std::sqrt(std::max(0.0f, C(2, 2))), true);
            }
            const float act = std::exchange(inst.pending_actuation_llr, 0.0f);
            std::vector<ViewRow> rows;
            // ★A newborn's belief already holds its birth frame (its candidate observed it): it may not enter twice (fix round 1).
            ExistenceFrame fr_door = fr;
            if (const auto of = observed_frame_.find(id); of != observed_frame_.end() and of->second == pkt.frame_id)
                fr_door.zed_ran = fr_door.ricoh_ran = fr_door.ricoh_hits_only = false;
            inst.exist_trace = observe_place(inst.belief, inst, fr_door, visit_, act, &rows);
            // A door restored from a remembered row is a REVISIT of that place: its first detection finds it again (the learned hazard's
            // "unchanged" sample); its removal below is the "changed" one. Counted once.
            if (inst.remembered_seen_ms.has_value() and (inst.exist_trace.zed_hit > 0.0f or inst.exist_trace.ric_hit > 0.0f))
            {
                places_.note_revisit(true, double(in.t_wall_ms - *inst.remembered_seen_ms) * 1e-3);
                inst.remembered_seen_ms.reset();
            }
            if (act != 0.0f and not std::isfinite(inst.exist_trace.act_dl))
                inst.pending_actuation_llr += act;   // no view and no ZED pose this cycle: the evidence waits (fix round 1)
            const float L = inst.belief.L();
            const Verdict v = decide(L, inst.role, costs);
            inst.exist_trace.delete_line = v.delete_line;
            if (v.remove)
                to_remove.push_back(id);
            log_view_rows(inst.node_name, rows, out);
        }
    });

    // Decision log: every door's belief after this update (the input of the REMOVE below). a = L, b = P, c = x, d = y.
    for (const auto& [id, instp] : states)
        if (const auto& inst = *instp; not inst.is_bearing_hypothesis)
        {
            rc::door::DoorDecision d{out.cycle, out.t_ms, static_cast<std::uint16_t>(K::Existence), id, inst.node_name,
                                     inst.belief.L(), inst.belief.p(), inst.model.state().cx, inst.model.state().cy, {}};
            if (sinks_.decision) sinks_.decision(d);
            out.decisions.push_back(std::move(d));
        }
    // The readout: one CSV row per door per fresh frame (spec §3.8), stdout throttled (1 in 60, plus every cycle on which a
    // door moved by more than one nat or is removed) — the CSV is the record, stdout the operator's view.
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        const auto& [id, instp] = states[i];
        const auto& inst = *instp;
        if (inst.is_bearing_hypothesis) continue;
        const auto& b = inst.belief;
        const auto& t = inst.exist_trace;
        const auto& ms = inst.model.state();
        std::ostringstream ex_csv;
        ex_csv.imbue(std::locale::classic());
        ex_csv << ex_dbg_ << ',' << inst.node_name << ',' << b.L() << ',' << b.p() << ',' << ms.cx << ',' << ms.cy << ','
               << role_name(inst.role) << ',' << t.delete_line << ',' << t.zed_view << ',' << t.ric_view << ','
               << b.prior.band << ',' << b.prior.base_rate << ',' << b.prior.size << ',' << b.carried << ',' << b.crossing << ','
               << b.ledger.sum() << ',' << b.ledger.views().size() << ',' << t.zed_dl << ',' << t.ric_dl << ',' << t.geo_dl << ','
               << t.act_dl << ',' << t.zed_q << ',' << t.ric_q << ',' << t.zed_hit << ',' << t.ric_hit << ',' << t.kappa << ','
               << t.geo_x << ',' << t.geo_mu0 << ',' << t.geo_mu1 << ',' << t.geo_s << '\n';
        diag("door_existence_log.csv", ex_csv.str(), out);
        const bool removing = std::ranges::contains(to_remove, id);
        if (ex_dbg_ % 60 == 0 or removing or std::abs(b.L() - L_before[i]) > 1.0f)
            diag("stdout", std::format("door_concept: [existence] {} L={:+.2f} p={:.2f} role={} delete_line={} | prior band={:+.2f} "
                                       "base={:+.2f} size={:+.2f} carried={:+.2f} cross={:+.2f} | views={} | zed dl={} q={} hit={} "
                                       "| ric dl={} q={} hit={} | geo dl={} | act dl={} | kappa={}\n",
                                       inst.node_name, b.L(), b.p(), role_name(inst.role), fmt_or_dash(t.delete_line),
                                       b.prior.band, b.prior.base_rate, b.prior.size, b.carried, b.crossing,
                                       b.ledger.views().size(), fmt_or_dash(t.zed_dl), fmt_or_dash(t.zed_q), fmt_or_dash(t.zed_hit),
                                       fmt_or_dash(t.ric_dl), fmt_or_dash(t.ric_q), fmt_or_dash(t.ric_hit), fmt_or_dash(t.geo_dl),
                                       fmt_or_dash(t.act_dl), fmt_or_dash(t.kappa)), out);
    }
    if (ex_dbg_ % 600 == 0)
    {
        const auto& T = detectors_;
        const auto c = costs_.costs(cfg.exist_max_room_s);
        diag("stdout", std::format("door_concept: [existence] model (fixed this session): zed hit={:.3f} fa={:.4f} (n={:.0f}) | ricoh "
                                   "hit={:.3f} fa={:.4f} (n={:.0f}) | C_keep={:.1f}s (n={}) C_delete={:.1f}s (n={})\n",
                                   T.zed.hit_rate, T.zed.false_alarm, T.zed.n_hit_trials, T.ricoh.hit_rate, T.ricoh.false_alarm,
                                   T.ricoh.n_hit_trials, c.keep_s,
                                   costs_.n(CostEstimator::Kind::Keep), c.delete_ordinary_s, costs_.n(CostEstimator::Kind::DeleteOrdinary)), out);
    }

    // The removals: the decision is the core's; the graph half is the world's (before_remove / remove_door), in the order the
    // decision log has always had. The place's row is kept with its removal time: "no door here" (spec §3.7).
    for (const std::uint64_t id : to_remove)
    {
        const auto it = std::ranges::find(states, id, &std::pair<std::uint64_t, rc::DoorFitState*>::first);
        if (it == states.end()) continue;
        const rc::DoorFitState& inst = *it->second;
        const float L = inst.belief.L();
        diag("stdout", std::format("door_concept: [existence] REMOVE id={} {} (L={:.2f} < delete line {:.2f}, role {})\n", id,
                                   inst.node_name, L, inst.exist_trace.delete_line, role_name(inst.role)), out);
        PlaceRow r;
        r.room = room_name_; r.name = inst.node_name; r.xy = inst.aperture.centre_xy();
        const float sd2 = inst.ai2_initialized ? std::max(0.0f, inst.ai2_belief.covariance()(0, 0)) : cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
        r.cov = Eigen::Matrix2f::Identity() * sd2;
        r.yaw = inst.aperture.yaw(); r.width = inst.aperture.w; r.height = inst.aperture.h; r.depth = inst.aperture.thickness;
        r.L_evidence = inst.belief.L_evidence(); r.ledger = inst.belief.ledger; r.visit = visit_;
        r.last_seen_wall_ms = in.t_wall_ms; r.removed_wall_ms = in.t_wall_ms;
        places_.upsert(std::move(r));
        // The robot is back in this room, the place was in view (the door was judged on views with q > 0) and the remembered door is
        // gone: a CHANGE since the row was last seen - the learned hazard's other sample.
        if (inst.remembered_seen_ms.has_value())
            places_.note_revisit(false, double(in.t_wall_ms - *inst.remembered_seen_ms) * 1e-3);
        observed_frame_.erase(id);
        world_->before_remove(id);
        const auto& s = inst.model.state();
        record(K::Remove, id, s.cx, s.cy, std::format("logodds {:.2f}", L), inst.node_name, out);
        out.deletes.push_back({id, "existence"});
        world_->remove_door(id, L);
    }
}

}  // namespace rc::door::core
