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
#include <map>
#include <numeric>
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

namespace rc::door::core
{
namespace
{
using K = rc::door::DoorDecisionKind;

// MaskCells, build_mask_cells and own_wall_edges live in door_visibility.{h,cpp} (shared with door_view).

}  // namespace

// ★Task 8 (spec §3.8): ONE ROW PER DOOR PER FRESH FRAME, every channel's contribution THIS cycle; `nan` = not computed this
// cycle (never a stale value). L = prior_band + prior_base + prior_size + prior_extent (last column) + carried + crossing + (physical part of ledger) + vision (spec §3.1b; Sc, Sp and the saturation term are its last columns). zed_dl / ric_dl / geo_dl /
// act_dl: what each channel changed the ledger sum by this cycle; view_id / ric_view_id: the ledger record the frame joined;
// zed_q / ric_q: P(detector reports the door from this view); zed_hit / ric_hit: this frame's association weight (0 = a miss);
// geo_x / geo_mu0 / geo_mu1 / geo_s: the contour statistic, the controls' mean, the present-hypothesis mean and the controls' sd.
const char* const kExistenceCsvHeader =
    "cycle,node,L,p_exists,cx,cy,role,delete_line,view_id,ric_view_id,"
    "prior_band,prior_base,prior_size,carried,crossing,ledger,n_views,"
    "zed_dl,ric_dl,geo_dl,act_dl,zed_q,ric_q,zed_hit,ric_hit,kappa,geo_x,geo_mu0,geo_mu1,geo_s,sup_x,sup_dl,Sc,Sp,sat_term,sup_c,sup_q,prior_extent\n";

// Ruling R8: the per-view rows tools/fit_detector_envelopes.py reads (its column names, exactly). Every field is
// THIS frame's (hit = its association weight, 0 = a miss); door = the node name or place_<key> of a pending candidate.
const char* const kViewsCsvHeader = "cycle,door,camera,view_id,q,fill_h,fill_v,vis,kappa,aspect,hit,sup_x,sup_l\n";   // sup_x/sup_l: the fit-support row (camera `support`), nan on the detector rows

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
// The size prior of the existence term: the fit's own (Door.Prior*, one prior for one quantity) today; the typical real door
// (Existence.Typical*) under the one-model flag, which keeps the fit's prior untouched (spec §3.4).
SizePrior existence_size_prior(const rc::DoorConfig& cfg)
{
    if (cfg.exist_one_model)
        return SizePrior{cfg.exist_typical_w_m, cfg.exist_typical_w_std, cfg.exist_typical_h_m, cfg.exist_typical_h_std};
    return SizePrior{cfg.door_prior_w_m, cfg.door_prior_w_std, cfg.door_prior_h_m, cfg.door_prior_h_std};
}
}  // namespace

// A live door's position variance along its wall: the detection noise plus the fit's variance of the aperture centre (s + w/2).
static float place_var(const rc::DoorFitState& st, float rn2)
{
    if (not st.ai2_initialized) return rn2;
    const auto& C = st.ai2_belief.covariance();
    return rn2 + std::max(0.0f, C(0, 0) + 0.25f * C(1, 1) + C(0, 1));
}

// ONE-MODEL (S3): two labels at one hole COMPETE instead of being separated or merged. The "another door on this wall" cost of a
// place is charged by the OTHER place's existence probability times how DISTINCT the two places are: 1 - exp(-m^2/2), m^2 the squared
// Mahalanobis separation of the two places under their summed position variances (the same comparison the masks' responsibilities make).
// Two places on one hole (m -> 0) are one door under two labels and charge each other nothing; two real doors a metre apart (m >> 1) charge
// the full second-door cost. Continuous, no radius. Doors weigh their own along-wall variance; candidates their detection variance.
std::vector<float> DoorCore::other_p_on_wall(int wall, const std::vector<Vec2f>& polygon, const std::vector<int>& groups,
                                             std::uint64_t self_door, std::uint64_t self_cand, const Vec2f& self_xy, float self_var, bool doors_only) const
{
    std::vector<float> r;
    if (wall < 0) return r;
    const float rn2 = cfg_.door.tracker_detection_noise_m * cfg_.door.tracker_detection_noise_m;
    const auto distinct = [&](const Vec2f& xy, float var)
    { return place_distinctness((xy - self_xy).squaredNorm(), var + self_var); };
    for (const auto& [id, st] : world_->fit_states())
    {
        if (id == self_door or st->is_bearing_hypothesis) continue;
        const auto it = last_p_door_.find(id);
        const Vec2f c = st->aperture.centre_xy();
        if (it == last_p_door_.end() or wall_of(c, polygon, groups) != wall) continue;
        r.push_back(it->second * distinct(c, place_var(*st, rn2)));
    }
    for (const Candidate& c : candidates_)
    {
        if (doors_only or c.key == self_cand or c.room != room_name_) continue;
        const auto it = last_p_cand_.find(c.key);
        if (it != last_p_cand_.end() and wall_of(c.theta.aperture.centre_xy(), polygon, groups) == wall)
            r.push_back(it->second * distinct(c.xy, c.var));
    }
    return r;
}

// ═══ ONE-MODEL: THE WALL'S COUNT POSTERIOR (final review 1) ═══════════════════════════════════════════════════════════════
// The per-wall base rate of every door and candidate of the room, from the EXACT marginal of each member under the wall's count prior,
// computed jointly from every member's evidence as it stands NOW (wall_count_base_rates, wall_base_rate.h). It replaces the earlier
// "charge the others' previous-cycle p": that was an undamped Jacobi sweep, which refused every seed of a wall by its sibling seeds and
// oscillated at four members; the posterior has nothing to iterate. Members on one hole (same_place) are one slot, alternatives of each other.
void DoorCore::assign_wall_counts()
{
    if (not cfg_.door.exist_one_model or world_ == nullptr) return;
    const auto& polygon = fit_.room_polygon();
    const auto& groups = groups_cached();
    const float rn2 = cfg_.door.tracker_detection_noise_m * cfg_.door.tracker_detection_noise_m;
    struct Member { PlaceBelief* b; Vec2f xy; float var; int wall; Candidate* cand; };
    std::vector<Member> ms;
    for (const auto& [id, st] : world_->fit_states())
    {
        if (st->is_bearing_hypothesis) continue;
        const Vec2f c = st->aperture.centre_xy();
        ms.push_back({&st->belief, c, place_var(*st, rn2), wall_of(c, polygon, groups), nullptr});
    }
    for (Candidate& c : candidates_)
        if (c.room == room_name_)
            ms.push_back({&c.belief, c.xy, c.var, wall_of(c.theta.aperture.centre_xy(), polygon, groups), &c});
    if (ms.empty()) return;
    const double b_ref = std::log(rates_.first.mean() / (1.0 - rates_.first.mean()));
    // ln O_i: the member's own evidence with the base rate taken out (L evaluated at the first-door rate, minus that rate)
    std::vector<double> lo(ms.size());
    for (std::size_t i = 0; i < ms.size(); ++i)
    {
        PlaceBelief& b = *ms[i].b;
        const float held = b.prior.base_rate;
        b.prior.base_rate = static_cast<float>(b_ref);
        lo[i] = static_cast<double>(b.L()) - b_ref;
        b.prior.base_rate = held;
    }
    std::map<int, std::vector<std::size_t>> by_wall;   // a member with no wall is alone on its own (negative key)
    int alone = -1;
    for (std::size_t i = 0; i < ms.size(); ++i)
        by_wall[ms[i].wall >= 0 ? ms[i].wall : alone--].push_back(i);
    for (const auto& [w, idx] : by_wall)
    {
        // slots: members on one hole (the MAP reading of their separation, place_distinctness < 1/2) are alternatives of each other
        std::vector<int> slot(idx.size());
        std::iota(slot.begin(), slot.end(), 0);
        const auto find = [&](int a) { while (slot[static_cast<std::size_t>(a)] != a) a = slot[static_cast<std::size_t>(a)]; return a; };
        for (std::size_t a = 0; a < idx.size(); ++a)
            for (std::size_t b = a + 1; b < idx.size(); ++b)
                if (same_place((ms[idx[a]].xy - ms[idx[b]].xy).squaredNorm(), ms[idx[a]].var + ms[idx[b]].var))
                    slot[static_cast<std::size_t>(find(static_cast<int>(b)))] = find(static_cast<int>(a));
        std::vector<int> root(idx.size()), compact(idx.size(), -1);
        int n_slots = 0;
        for (std::size_t a = 0; a < idx.size(); ++a)
        {
            const int r = find(static_cast<int>(a));
            if (compact[static_cast<std::size_t>(r)] < 0) compact[static_cast<std::size_t>(r)] = n_slots++;
            root[a] = compact[static_cast<std::size_t>(r)];
        }
        std::vector<double> lw(idx.size());
        for (std::size_t a = 0; a < idx.size(); ++a) lw[a] = lo[idx[a]];
        const std::vector<double> base = wall_count_base_rates(rates_, lw, root);
        for (std::size_t a = 0; a < idx.size(); ++a)
            ms[idx[a]].b->prior.base_rate = static_cast<float>(base[a]);
    }
    // the refusal's base rate of a candidate: the LIVE doors' charge only (Candidate::base_open)
    for (const Member& m : ms)
        if (m.cand != nullptr)
        {
            const std::vector<float> r_live = other_p_on_wall(m.wall, polygon, groups, ~std::uint64_t{0}, ~std::uint64_t{0}, m.xy, m.var, /*doors_only=*/true);
            m.cand->base_open = wall_base_rate(rates_, r_live);
        }
}

std::vector<FrameDetection> DoorCore::frame_detections() const
{
    std::vector<FrameDetection> v;
    v.reserve(dets_.size());
    for (std::size_t i = 0; i < dets_.size(); ++i)
        v.push_back({dets_[i].xy, dets_[i].xy_var, dets_[i].sensor, i < det_w_.size() ? det_w_[i] : 1.0f,
                     i < det_adm_.size() ? det_adm_[i] != 0 : true});
    return v;
}

// ═══ ONE-MODEL: the four causes of every detection (spec §3.2; door_mask_causes.h) ═══════════════════════════════════════
// The log of the probability that a door is BORN at `xy`: the place prior of the typical door there (band, extent, per-wall base
// rate charged by the live doors' last-cycle p; size unresolved = 0). Candidates are not counted: they ARE the new-door hypothesis.
double DoorCore::log_birth(const Vec2f& xy, float sd) const
{
    const auto& cfg = cfg_.door;
    const auto& polygon = fit_.room_polygon();
    const auto& groups = groups_cached();
    const rc::DoorFitState th = typical_door_at(xy, cfg.door_prior_w_m, cfg.door_prior_h_m, polygon);
    const int wall = wall_of(th.aperture.centre_xy(), polygon, groups);
    // the live doors only (a candidate IS the new-door hypothesis, not another door), charged by distinctness as everywhere else
    const std::vector<float> r_others = other_p_on_wall(wall, polygon, groups, ~std::uint64_t{0}, ~std::uint64_t{0}, xy, sd * sd, /*doors_only=*/true);
    const PlacePrior pp = prior_of(th, polygon, fit_.room_is_proto(), fit_.room_band(), 0, cfg.tracker_detection_noise_m, cfg.door_prior_h_std, false,
                                   existence_size_prior(cfg), sd, sd, sd, std::numeric_limits<float>::quiet_NaN(), &rates_, r_others);
    return -std::log1p(std::exp(-static_cast<double>(pp.total())));   // log of the logistic: stable for either sign
}

// wall_groups of the current outline, built once per outline (it was rebuilt on every log_birth / wall_length_at call: several per mask per cycle).
// Keyed on the outline itself, so a changed outline or sigma rebuilds it.
const std::vector<int>& DoorCore::groups_cached() const
{
    const auto& polygon = fit_.room_polygon();
    const float sg = fit_.room_band().sigma_m;
    if (not (groups_key_poly_ == polygon and groups_key_sigma_ == sg))
    {
        groups_key_poly_ = polygon; groups_key_sigma_ = sg;
        groups_cache_ = wall_groups(polygon, sg);
    }
    return groups_cache_;
}

// Λ: the length of the wall (its wall_groups group) that `xy` lies on; births and spurious masks are uniform along it (spec §3.4).
// kept: no polygon / no wall (an unknown room): 5 m, a typical wall; it enters as a log and cancels between new door and clutter.
double DoorCore::wall_length_at(const Vec2f& xy) const
{
    const auto& polygon = fit_.room_polygon();
    const auto& groups = groups_cached();
    if (const int wg = wall_of(xy, polygon, groups); wg >= 0)
    {
        double len = 0.0;
        for (std::size_t k = 0, n = polygon.size(); k < n; ++k)
            if (groups[k] == wg) len += (polygon[(k + 1) % n] - polygon[k]).norm();
        if (len > 0.0) return len;
    }
    return 5.0;
}

bool DoorCore::live_door_explains(const std::vector<Vec2f>& polygon, const rc::door::RoomBand& band, bool room_is_proto,
                                  std::span<const std::uint64_t> room_doors, const Vec2f& xy, float var) const
{
    if (world_ == nullptr or polygon.size() < 3 or room_doors.empty()) return false;
    const auto& cfg = cfg_.door;
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    const std::vector<int> groups = wall_groups(polygon, band.sigma_m);   // the CALLER's room, not the core's (it takes its outline up at the next step)
    const auto of_room = [&](std::uint64_t id) { return std::ranges::find(room_doors, id) != room_doors.end(); };
    const rc::DoorFitState th = typical_door_at(xy, cfg.door_prior_w_m, cfg.door_prior_h_m, polygon);
    const int wall = wall_of(th.aperture.centre_xy(), polygon, groups);
    const float sd = std::sqrt(std::max(1e-9f, var));
    // log b - log Lambda of a NEW door at the place: the room's live doors on its wall charge the base rate, weighted by distinctness (as log_birth)
    std::vector<float> r_others;
    for (const auto& [id, st] : world_->fit_states())
    {
        if (not of_room(id) or st->is_bearing_hypothesis) continue;
        const auto it = last_p_door_.find(id);
        const Vec2f c = st->aperture.centre_xy();
        if (wall >= 0 and c.allFinite() and wall_of(c, polygon, groups) == wall)
            r_others.push_back((it != last_p_door_.end() ? it->second : 1.0f) * place_distinctness((c - xy).squaredNorm(), place_var(*st, rn2) + var));
    }
    const PlacePrior pp = prior_of(th, polygon, room_is_proto, band, 0, cfg.tracker_detection_noise_m, cfg.door_prior_h_std, false,
                                   existence_size_prior(cfg), sd, sd, sd, std::numeric_limits<float>::quiet_NaN(), &rates_, r_others);
    double len = 0.0;
    if (const int wg = wall_of(xy, polygon, groups); wg >= 0)
        for (std::size_t k = 0, n = polygon.size(); k < n; ++k)
            if (groups[k] == wg) len += (polygon[(k + 1) % n] - polygon[k]).norm();
    if (not (len > 0.0)) len = 5.0;   // kept: as wall_length_at (no wall under xy: a typical 5 m wall)
    const double log_new = -std::log1p(std::exp(-static_cast<double>(pp.total()))) - std::log(len);
    for (const auto& [id, st] : world_->fit_states())
    {
        if (not of_room(id) or st->is_bearing_hypothesis) continue;
        const auto it = last_p_door_.find(id);
        const double p = it != last_p_door_.end() ? double(it->second) : 1.0;   // not yet judged (born, restored or carried this cycle): taken as certain, as the candidates' test does
        const Vec2f c = st->aperture.centre_xy();
        if (not c.allFinite()) continue;
        if (place_explained_by(p, (c - xy).squaredNorm(), double(place_var(*st, rn2)) + double(var), log_new)) return true;
    }
    return false;
}

void DoorCore::compute_mask_causes(const DoorCycleInput& /*in*/)
{
    mask_causes_.clear();
    mask_log_birth_.clear();
    mask_leaf_.clear();
    const auto& cfg = cfg_.door;
    if (not cfg.exist_one_model or dets_.empty() or not info_.new_frame)
        return;
    const auto& pkt = masks_.packet();
    const auto& polygon = fit_.room_polygon();
    const auto groups = wall_groups(polygon, fit_.room_band().sigma_m);
    // the live doors as causes (those with a belief); r = their previous-cycle p, so nothing depends on this cycle's own update
    std::vector<DoorCauseInput> doors;
    struct Along { Vec2f centre, u; float var; };
    std::unordered_map<std::uint64_t, Along> door_along;
    const rc::DoorBeliefParams* bp = nullptr;
    for (const auto& [id, st] : world_->fit_states())
    {
        if (st->is_bearing_hypothesis or not st->ai2_initialized) continue;
        DoorCauseInput d;
        d.id = id;
        d.belief = &st->ai2_belief;
        d.leaf = &st->leaf_channel;
        const auto it = last_p_door_.find(id);
        const double r = it != last_p_door_.end() ? it->second : st->belief.p();
        d.log_r = std::log(std::max(1e-9, r));   // max: the domain of a log (r is a probability, 0 is not a belief)
        d.place_var = std::pow(std::max(0.0f, st->ai2_belief.recess_sd()), 2.0f) + 0.5f * (st->chain_cov_xx + st->chain_cov_yy);
        d.phi_max_rad = cfg.phi_max_rad;
        door_along[id] = {st->aperture.centre_xy(), st->aperture.wall_u, std::max(0.0f, st->ai2_belief.covariance()(0, 0)) + st->chain_cov_xx};
        doors.push_back(d);
        if (bp == nullptr) bp = &st->ai2_belief.params();
    }
    const rc::DoorBeliefParams bp_default{};
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    mask_causes_.resize(dets_.size());
    mask_log_birth_.assign(dets_.size(), -std::numeric_limits<double>::infinity());
    mask_leaf_.assign(dets_.size(), std::nullopt);
    const rc::door::RoomBand band = fit_.room_band();
    const SizePrior typical = existence_size_prior(cfg);
    for (std::size_t i = 0; i < dets_.size(); ++i)
    {
        const auto& dv = dets_[i];
        const auto& sl = pkt.slices[static_cast<std::size_t>(dv.slice_index)];
        const auto span = masks_.support(sl);
        if (span.empty()) continue;
        // a stride, so a 3000-point mask costs what a 250-point one does (cost only: the mean per-point score is unchanged by it)
        const std::size_t stride = std::max<std::size_t>(1, span.size() / 256);
        std::vector<Eigen::Vector3f> pts;
        pts.reserve(span.size() / stride + 1);
        for (std::size_t k = 0; k < span.size(); k += stride) pts.push_back(span[k]);
        const DetectorModel& dm = dv.sensor != 0 ? detectors_.ricoh : detectors_.zed;
        const float var = dv.xy_var >= 0.0f ? dv.xy_var : rn2;
        const double log_b = log_birth(dv.xy, std::sqrt(var));
        // Λ: the length of the wall (its wall_groups group) the mask lies on; births and spurious masks are uniform along it.
        const double wall_len = wall_length_at(dv.xy);
        mask_log_birth_[i] = log_b;
        MaskCauseModel model;
        model.log_birth = log_b;
        model.log_kappa = std::log(std::max(1e-9, static_cast<double>(dm.false_alarm)));
        model.log_pd_new = std::log(std::max(1e-9, static_cast<double>(dm.hit_rate)));
        model.along_len_m = wall_len;
        for (auto& d : doors)
        {
            d.log_pd = model.log_pd_new;
            const auto it = door_along.find(d.id);
            d.along_offset_m = (dv.xy - it->second.centre).dot(it->second.u);
            d.along_var = it->second.var + var;
        }
        // ★THE LEAF MODE OF THE NEW-DOOR CAUSE (spec §3.6; door_leaf_birth.h): the mask is a leaf seen from outside its frame wall.
        std::vector<rc::door::LeafApertureHyp> hyps;
        std::vector<double> hyp_log_b;   // log b of each hypothesis (the probability a door is born at its aperture)
        // the mask's own depth noise (a ricoh LiDAR-depth slice: depth_var > 0) widens how well its ends are known against the wall (the fit's R does the same)
        const rc::door::RoomBand mband{band.wall_thickness_m, std::hypot(band.sigma_m, std::sqrt(std::max(0.0f, sl.depth_var)))};
        if (polygon.size() >= 3)
            if (const auto seg = rc::door::plan_segment(pts); seg.ok)
            {
                const Vec2f centre = 0.5f * (seg.a + seg.b), dir = (seg.b - seg.a) / seg.length;
                if (const int e = rc::door::detail::nearest_edge(polygon, centre); e >= 0)
                {
                    const Vec2f uw = (polygon[(static_cast<std::size_t>(e) + 1) % polygon.size()] - polygon[static_cast<std::size_t>(e)]).normalized();
                    const float theta = std::acos(std::min(1.0f, std::abs(dir.dot(uw))));   // 0 flush with its wall, pi/2 perpendicular
                    const double g_ap = rc::door::aperture_angle_likelihood(theta, rc::door::segment_angle_sd(seg.length, mband.sigma_m));
                    model.log_aperture_factor = std::log(std::max(g_ap, 1e-300));   // numerical floor: the log of an underflowed likelihood
                    hyps = rc::door::apertures_from_leaf(seg.a, seg.b, seg.height, polygon, groups, typical, mband, &rates_);
                    const double log_open = std::log(std::max(1e-9, static_cast<double>(cfg.ricoh_birth_p_open)));   // P(open): the one open prior the agent has
                    const double lid_max = std::log(static_cast<double>(detectors_.lidar.through_open) / detectors_.lidar.through_wall);   // a sweep's term is a mean per-bin log ratio: at most this
                    const double log_floor = model.log_kappa - std::log(std::max(1e-9, wall_len)) - 20.0;   // numerical reach floor, as causes_of_mask's
                    model.leaf_log_w.assign(hyps.size(), -std::numeric_limits<double>::infinity());
                    hyp_log_b.assign(hyps.size(), -std::numeric_limits<double>::infinity());
                    for (std::size_t h = 0; h < hyps.size(); ++h)
                    {
                        auto& hy = hyps[h];
                        const Vec2f hc = hy.aperture.centre_xy();
                        // the live doors on the hypothesis' wall charge the base rate by distinctness, as everywhere (S5/S3); the seed rate is replaced
                        const std::vector<float> r_o = other_p_on_wall(hy.wall, polygon, groups, ~std::uint64_t{0}, ~std::uint64_t{0}, hc, rn2, /*doors_only=*/true);
                        const double lw = hy.log_w - wall_base_rate(rates_, {}) + wall_base_rate(rates_, r_o);
                        const double lb = -std::log1p(std::exp(-lw));
                        // a hole holds one door: a NEW door at a hole a live door already stands in has the mass the live door leaves (1 − r · overlap)
                        const double hole_free = this->hole_free(hy.aperture, groups);
                        // the leaf is NOT SHUT against ITS frame wall (the angle to that wall, not to the wall nearest the mask): 1 - aperture angular there
                        const float th_h = std::acos(std::min(1.0f, std::abs(dir.dot(hy.aperture.wall_u))));
                        const double log_not_h = std::log1p(-std::min(rc::door::aperture_angle_likelihood(th_h, rc::door::segment_angle_sd(seg.length, mband.sigma_m)), 1.0 - 1e-16));
                        double v = lb - std::log(wall_length_at(hc)) + log_open + log_not_h + std::log(std::max(hole_free, 1e-300));   // 1e-300: numerical floor on a log
                        // the φ prior the fit uses (uniform on [0, phi_max]): a wrong-side twin at φ = π − θ beyond the hinge's range is impossible, in range it pays the density
                        if (not (hy.leaf.phi <= cfg.phi_max_rad)) continue;
                        v -= std::log(std::max(1e-3f, cfg.phi_max_rad));   // 1e-3: a numerical floor on the log
                        if (not std::isfinite(v) or v + std::max(0.0, lid_max) + model.log_pd_new < log_floor) continue;
                        if (leaf_sweep_.present())
                        {
                            // the opening is evidence for the HOLE: the bins a live door's span holds are that door's, the others count in full
                            const auto claims = live_claims(hy.aperture, groups);
                            const std::pair<float, float> pt{hy.leaf.phi, 1.0f};
                            v += rc::door::opening_view_llr(hy.aperture, registered(hy.aperture), detectors_.lidar.through_open, detectors_.lidar.through_wall,
                                                            std::span<const std::pair<float, float>>(&pt, 1), claims);
                        }
                        model.leaf_log_w[h] = v;
                        hyp_log_b[h] = lb;
                    }
                }
            }
        const float R = cfg.ai2_sigma_base_m * cfg.ai2_sigma_base_m + std::max(0.0f, sl.depth_var);
        mask_causes_[i] = causes_of_mask(dv.slice_index, pts, R, doors, model, bp != nullptr ? *bp : bp_default);
        {   // one-model: the doors on this mask's list of causes compete for it
            std::vector<NodeRef> nodes;
            for (const auto& cz : mask_causes_[i].causes)
                if ((cz.kind == CauseKind::Aperture or cz.kind == CauseKind::Leaf) and mask_causes_[i].rho_door(cz.door) >= rc::door::hlook::kWeightFloor)
                    nodes.push_back({'d', cz.door});
            note_competition(nodes);
        }
        if (const int lh = mask_causes_[i].new_leaf_hyp; lh >= 0)
        {
            mask_leaf_[i] = hyps[static_cast<std::size_t>(lh)];
            mask_log_birth_[i] = hyp_log_b[static_cast<std::size_t>(lh)];   // a candidate here is born at the leaf's aperture: its own place prior
        }
    }
}

// ONE-MODEL: the LiDAR opening of a pending candidate's aperture span: the newest sweep registered to it, scored against solid wall (spec §3.3).
// P(open) is the leaf's own when the candidate was born from a side-on leaf (its phi is the leaf's measured angle), else the birth prior of a
// place with no leaf belief (Birth.RicohOpenPrior, the one open prior the agent has) as a one-point leaf curve with that clear fraction.
std::optional<LidarOpening> DoorCore::candidate_lidar(const rc::DoorFitState& theta, bool leaf_born, const std::vector<int>& groups) const
{
    if (not cfg_.door.exist_one_model or not leaf_sweep_.present()) return std::nullopt;
    const float cf = leaf_born ? 1.0f - std::cos(theta.leaf.phi) : std::clamp(cfg_.door.ricoh_birth_p_open, 0.0f, 1.0f);   // clamp: domain of a share
    const std::pair<float, float> pt{std::acos(std::clamp(1.0f - cf, -1.0f, 1.0f)), 1.0f};
    // the opening is evidence for the HOLE: a bin a live door's span holds is that door's (per-bin freedom), so a duplicate shifted along an open
    // door gets nothing for the half it shares and pays for its own half reading wall
    const auto claims = live_claims(theta.aperture, groups);
    const auto& reg = registered(theta.aperture);
    return LidarOpening{rc::door::opening_view_llr(theta.aperture, reg, detectors_.lidar.through_open, detectors_.lidar.through_wall,
                                                   std::span<const std::pair<float, float>>(&pt, 1), claims),
                        reg.origin.head<2>()};
}

// The live doors' claims on this aperture's wall line (their spans in the aperture's own along-wall coordinate, with their last-cycle existence).
std::vector<rc::door::OpeningClaim> DoorCore::live_claims(const rc::door::Aperture& ap, const std::vector<int>& groups) const
{
    std::vector<rc::door::OpeningClaim> out;
    const auto& polygon = fit_.room_polygon();
    const int wall = wall_of(ap.centre_xy(), polygon, groups);
    for (const auto& [id, st] : world_->fit_states())
    {
        if (st->is_bearing_hypothesis) continue;
        const auto it = last_p_door_.find(id);
        if (it == last_p_door_.end() or wall_of(st->aperture.centre_xy(), polygon, groups) != wall) continue;
        const float c = (st->aperture.centre_xy() - ap.wall_O).dot(ap.wall_u);
        out.push_back({c - 0.5f * st->aperture.w, c + 0.5f * st->aperture.w, it->second});
    }
    return out;
}

// One registration of the newest sweep per aperture per cycle (the hypothesis loop and the candidates ask for the same apertures); cleared by step().
const rc::door::RegisteredSweep& DoorCore::registered(const rc::door::Aperture& ap) const
{
    for (const auto& [k, r] : reg_cache_)
        if (k.wall_O == ap.wall_O and k.wall_u == ap.wall_u and k.s == ap.s and k.w == ap.w and k.h == ap.h) return r;
    reg_cache_.emplace_back(ap, rc::door::register_sweep(ap, leaf_sweep_.origin(), leaf_sweep_.points()));
    return reg_cache_.back().second;
}

// The share of the aperture's span NO live door stands in: Π_i (1 − r_i · overlap_i) over the live doors on the same wall, r_i the door's last-cycle
// existence probability and overlap_i the fraction of the (mean) width the two spans share along the wall, max(0, 1 − |Δ| / w̄). Continuous (no radius).
double DoorCore::hole_free(const rc::door::Aperture& ap, const std::vector<int>& groups) const
{
    const auto& polygon = fit_.room_polygon();
    const int wall = wall_of(ap.centre_xy(), polygon, groups);
    double free = 1.0;
    for (const auto& [id, st] : world_->fit_states())
    {
        if (st->is_bearing_hypothesis) continue;
        const auto it = last_p_door_.find(id);
        if (it == last_p_door_.end() or wall_of(st->aperture.centre_xy(), polygon, groups) != wall) continue;
        const double wbar = 0.5 * (ap.w + st->aperture.w);
        const double d = std::abs((st->aperture.centre_xy() - ap.centre_xy()).dot(ap.wall_u));
        free *= 1.0 - static_cast<double>(it->second) * std::max(0.0, 1.0 - d / std::max(1e-3, wbar));
    }
    return free;
}

std::vector<FrameDetection> DoorCore::place_detections(std::uint64_t door, int candidate) const
{
    std::vector<FrameDetection> v = frame_detections();
    if (not cfg_.door.exist_one_model)
        return v;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        double rho = 0.0;   // no causes computed for this detection (not a fresh frame): it says nothing about this place
        if (i < mask_causes_.size() and not mask_causes_[i].causes.empty())
        {
            if (door != 0) rho = mask_causes_[i].hit_door(door);
            else if (candidate >= 0 and i < det_candidate_.size() and det_candidate_[i] == candidate) rho = mask_causes_[i].hit_new();
        }
        v[i].w *= static_cast<float>(rho);
    }
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
    // fr.geo_mu1_sd stays at its starting value (μ1 = μ0 + 2s) unless one-model: then the fitted table's value (0 = the channel carries no information)
    if (cfg.exist_one_model) fr.geo_mu1_sd = detectors_.geometry.mu1_sd;
    fr.rotation_w = rotation_reliability(ego_.ang_radps, cfg.ai2_ang_lever_m, cfg.ai2_motion_ref_mps);
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
        row << ex_dbg_ << ',' << place << ',' << (r.camera == 0 ? "zed" : r.camera == 2 ? "support" : "ricoh") << ',' << r.view_id << ','
            << (r.q_known ? r.q : std::numeric_limits<float>::quiet_NaN()) << ',' << r.fill_h << ',' << r.fill_v << ',' << r.vis << ','
            << r.kappa << ',' << r.aspect << ',' << r.hit << ',' << r.sup_x << ',' << r.sup_l << '\n';
        diag("door_views_log.csv", row.str(), out);
    }
}

// C_keep's samples (spec §8.2) - LOGGED to etc/door_cost_events.csv for tools/door_costs_from_logs.py; the costs are FIXED within a session: an executing episode on one of a door's affordances that ends without the robot crossing it.
// An episode spans the cycles in which ANY of the door's aff_<door>_* affordances is executed; it is crossed when the door is
// backed by a crossing fact while it lasts, or the room changes under it.
void DoorCore::track_episodes(const DoorCycleInput& in, DoorCycleOutput& out)
{
    std::set<std::string> now, crossing;
    for (const auto& e : in.executing)
        for (const auto& a : in.affordances)
            if (a.node_id == e.to and a.name.starts_with("aff_door_"))
                if (const auto k = a.name.rfind('_'); k != std::string::npos and k > 4)
                {
                    now.insert(a.name.substr(4, k - 4));
                    if (a.name.ends_with("_cross")) crossing.insert(a.name.substr(4, k - 4));
                }
    const auto room = world_->room_id();
    for (const auto& n : now)
        episodes_.try_emplace(n, Episode{in.t_wall_ms, false, room, room_name_});
    for (auto& [name, ep] : episodes_)
    {
        ep.crossing = ep.crossing or crossing.contains(name);
        if (ep.room != room) ep.crossed = true;
        for (const auto& [id, st] : world_->fit_states())
            if (st->node_name == name)
            {
                if (std::ranges::contains(in.backed_faces, id)) ep.crossed = true;
                if (ep.crossing and ep.room != room) st->belief.note_crossing();   // the room changed under the door's own cross: pending proof (place_belief.h), tape or live
                ep.x = st->aperture.centre_xy().x(); ep.y = st->aperture.centre_xy().y();   // the door's last place while the episode ran
            }
    }
    for (auto it = episodes_.begin(); it != episodes_.end();)
        if (not now.contains(it->first))
        {
            const Episode& ep = it->second;
            if (ep.crossed)   // the crossing: the in-data ground truth that the door existed (the tool's C_delete join, kCostEventsHeader)
                diag("door_cost_events.csv", std::format("crossed,{},{},{},{},{},0,{}\n", in.t_wall_ms, ep.room_name, it->first, ep.x, ep.y,
                                                         it->first), out);
            else if (in.t_wall_ms > ep.start_wall_ms)
                diag("door_cost_events.csv", std::format("keep_episode,{},{},{},nan,nan,{},{}\n", in.t_wall_ms, room_name_, it->first,
                                                         static_cast<double>(in.t_wall_ms - ep.start_wall_ms) * 1e-3, it->first), out);
            it = episodes_.erase(it);
        }
        else ++it;
}

// The candidates (spec §3.6, birth by comparison): every birthable detection that no live door's gate explains belongs to a
// pending place; each candidate in the room receives this frame like a door does, and its verdict is the birth evidence.
void DoorCore::prepare_candidates(const DoorCycleInput& in, const std::vector<rc::TrackView>& tracks, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    const SizePrior size_prior = existence_size_prior(cfg);
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
    // (one-model: no gate. A live door explains a candidate when the door's cause outweighs a new door's at the candidate's place:
    //  log r_i − ½ m² against log b, the same comparison the masks' responsibilities make.)
    const auto live_door_outweighs_new_at = [&](const Vec2f& cxy, float cvar)
    {
        // a place's existence against a NEW place at the same spot, as DENSITIES along the wall: r N(m; 0, var) against b / Λ
        const double lb = log_birth(cxy, std::sqrt(std::max(1e-9f, cvar))) - std::log(wall_length_at(cxy));
        for (const auto& t : tracks)
        {
            const Vec2f e = cxy - t.xy;
            const float cv = t.has_cov ? 0.5f * (t.cov(0, 0) + t.cov(1, 1)) : cfg.tracker_gate_fallback_m * cfg.tracker_gate_fallback_m;
            const double m2 = e.squaredNorm() / std::max(1e-9f, cv + cvar);
            const auto it = last_p_door_.find(t.id);
            if (std::log(std::max(1e-9, double(it != last_p_door_.end() ? it->second : 1.0f))) - 0.5 * m2
                - 0.5 * std::log(2.0 * std::numbers::pi * std::max(1e-9f, cv + cvar)) >= lb) return true;
        }
        return false;
    };
    const auto live_door_outweighs_new = [&](const Candidate& c) { return live_door_outweighs_new_at(c.xy, c.var); };
    std::erase_if(candidates_, [&](const Candidate& c)
                  { return c.room == room_name_ and (cfg.exist_one_model ? live_door_outweighs_new(c) : gated(c.xy, c.var)); });
    const auto& pkt = masks_.packet();
    const auto& polygon = fit_.room_polygon();
    // The typical door at the place, with the width of this frame's blob along its wall (its support's extent).
    const auto measure_candidate_width = [&](Candidate& c, const MaskSlice& slice)
    {
        const Vec2f u = typical_door_at(c.xy, cfg.door_prior_w_m, cfg.door_prior_h_m, polygon).aperture.wall_u;
        float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
        for (const auto& p : masks_.support(slice))
        { const float a = Vec2f(p.x(), p.y()).dot(u); lo = std::min(lo, a); hi = std::max(hi, a); }
        c.size_resolved = hi > lo;   // no extent in the support => no size estimate => the size term is 0 (spec: nothing measured pays nothing)
        c.width_meas = c.size_resolved ? hi - lo : cfg.door_prior_w_m;
        c.theta = typical_door_at(c.xy, c.width_meas, cfg.door_prior_h_m, polygon);
        // a clipped mask's width is a LOWER BOUND: its precision falls with the clipped fraction (continuously, no gate)
        const float untrunc = std::max(1e-3f, 1.0f - slice.trunc_frac);   // clamp-free domain guard: avoids /0 at full truncation
        c.width_sd = cfg.tracker_detection_noise_m / std::sqrt(untrunc);
    };
    const float rn2 = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    std::unordered_map<std::size_t, int> self_det;   // candidate index → the detection that is its own this frame
    for (std::size_t d = 0; d < dets_.size(); ++d)
    {
        const auto& dv = dets_[d];
        // one-model: no gate. The mask belongs to a pending place when that place's cause outweighs a new place (log r_c − ½ m² against
        // log b), and a new place is made for it unless a live door already owns it better (the mask's MAP cause is a door).
        const bool one = cfg.exist_one_model and d < mask_causes_.size() and not mask_causes_[d].causes.empty();
        if (not dv.birthable or (not one and gated(dv.xy, dv.xy_var))) continue;
        if (cfg.exist_one_model and not one) continue;   // no causes computed for it (not a fresh frame): it says nothing
        const float var = dv.xy_var >= 0.0f ? dv.xy_var : rn2;
        const Vec2f pxy = cfg.exist_one_model ? place_xy(d) : dv.xy;   // a side-on leaf: where its door's APERTURE is (spec §3.6), else the mask
        // a side-on leaf whose frame-wall hole a live door already outweighs a new place in is that door's leaf (its own Leaf cause explains it): no new place
        // (the comparison that retires a candidate there the next cycle, applied before it is born instead of after)
        if (cfg.exist_one_model and d < mask_leaf_.size() and mask_leaf_[d].has_value() and live_door_outweighs_new_at(pxy, dv.xy_var >= 0.0f ? dv.xy_var : rn2))
            continue;
        int best = -1; float best_m2 = cfg.tracker_gate_mahalanobis;
        if (one)
        {
            double best_score = mask_log_birth_[d] - std::log(wall_length_at(pxy));   // a candidate must beat a new place (densities along the wall)
            for (std::size_t c = 0; c < candidates_.size(); ++c)
            {
                if (candidates_[c].room != room_name_) continue;
                const double m2 = (candidates_[c].xy - pxy).squaredNorm() / std::max(1e-9f, candidates_[c].var + var);
                const auto it = last_p_cand_.find(candidates_[c].key);
                const double score = std::log(std::max(1e-9, double(it != last_p_cand_.end() ? it->second : 1.0f))) - 0.5 * m2
                                     - 0.5 * std::log(2.0 * std::numbers::pi * std::max(1e-9f, candidates_[c].var + var));
                if (score >= best_score) { best_score = score; best = static_cast<int>(c); }
            }
            double door_best = 0.0;
            for (const auto& [id, st] : world_->fit_states()) door_best = std::max(door_best, mask_causes_[d].rho_door(id));
            if (best < 0 and door_best > mask_causes_[d].rho_kind(CauseKind::NewDoor))
                continue;   // an existing door explains this mask better than a new place would

        }
        else
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            if (candidates_[c].room != room_name_) continue;
            const float m2 = (candidates_[c].xy - dv.xy).squaredNorm() / std::max(1e-9f, candidates_[c].var + var);
            if (m2 <= best_m2) { best_m2 = m2; best = static_cast<int>(c); }
        }
        if (best < 0)
        {
            // A new place. Its width is measured below, with every later frame's.
            Candidate c;
            c.key = ++next_candidate_key_;
            c.room = room_name_;
            c.xy = pxy;
            c.var = var;
            // ★The memory of this place is the candidate's starting evidence (spec §3.7/§8.3): a matched row continues its
            //  ledger, aged by the change hazard; a REMOVED row ("no door here") pushes it down by exactly its aged L.
            if (const PlaceRow* row = places_.match(room_name_, pxy, Eigen::Matrix2f::Identity() * var, cfg.tracker_gate_mahalanobis))
            {
                places_.restore_into(c.belief, *row, in.t_wall_ms);
                c.seeded_from_removed_ms = row->removed_wall_ms;
                c.seeded_name = row->name;   // its revisit sample is taken if it is BORN (door_core.cpp), never from one detection
            }
            candidates_.push_back(std::move(c));
            best = static_cast<int>(candidates_.size() - 1);
        }
        // ★THE SIZE TERM IS THIS FRAME'S, as the band term is (replaced, never accumulated): one narrow glimpse of a door half hidden must not
        //  hold it back for the candidate's life. A candidate's first detection of the frame measures it.
        if (not self_det.contains(static_cast<std::size_t>(best)))
        {
            Candidate& cb = candidates_[static_cast<std::size_t>(best)];
            measure_candidate_width(cb, pkt.slices[static_cast<std::size_t>(dv.slice_index)]);
            if (cfg.exist_one_model and d < mask_leaf_.size() and mask_leaf_[d].has_value())
            {   // the blob's extent along the wall is the leaf's FOOTPRINT, not a width: the door is the hypothesis' aperture, its width the leaf's length (fused with the typical size)
                const auto& hy = *mask_leaf_[d];
                cb.theta.aperture = hy.aperture; cb.theta.leaf = hy.leaf;
                cb.width_meas = hy.aperture.w; cb.size_resolved = true; cb.width_sd = cfg.tracker_detection_noise_m; cb.leaf_born = true;
            }
        }
        det_candidate_[d] = best;
        if (cfg.exist_one_model and d < mask_causes_.size())
        {   // one-model: this mask is explained by the candidate and by every door on its list of causes: they compete for it
            std::vector<NodeRef> nodes;
            if (mask_causes_[d].rho_kind(CauseKind::NewDoor) >= rc::door::hlook::kWeightFloor) nodes.push_back({'c', candidates_[static_cast<std::size_t>(best)].key});
            for (const auto& cz : mask_causes_[d].causes)
                if ((cz.kind == CauseKind::Aperture or cz.kind == CauseKind::Leaf) and mask_causes_[d].rho_door(cz.door) >= rc::door::hlook::kWeightFloor)
                    nodes.push_back({'d', cz.door});
            note_competition(nodes);
        }
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
            // ★THE MEASUREMENT PAYS THE SIZE TERM (2026-11-09, clean19-21): a candidate has a measured width and no fit, so its size evidence is
            //  N(measured; prior mean, prior var + measurement var). Scoring the measurement after the prior had pulled it counted the prior
            //  twice and cancelled the evidence: 0.36 m became 0.95 m and paid -0.20 nats, against -4.7. cand.theta holds the measured width
            //  and the prior's height (which then pays 0). Without a size estimate (no extent in the support) the size term is 0.
            const rc::DoorFitState& th = cand.theta;
            float h_sd = cfg.door_prior_h_std;
            // ★THE PLACE'S REMEMBERED HEIGHT IS THE CANDIDATE'S HEIGHT MEASUREMENT (2026-11-13, clean60 door_7; place_table.h
            //  remembered_height): the blob gives no height, so the place it was seeded from supplies the one its door's fit measured,
            //  aged by the change hazard. A place remembered door-tall pays ~0; a place refuted by its height (the GlassBay pane, 2.62 m)
            //  pays that refutation here instead of being born to be removed again. No row, no height: θ's (the prior's: 0).
            float h_meas = std::numeric_limits<float>::quiet_NaN();
            if (not cand.seeded_name.empty())
                if (const PlaceRow* row = places_.find(room_name_, cand.seeded_name); row != nullptr)
                    if (const auto mh = places_.remembered_height(*row, in.t_wall_ms, cfg.door_prior_h_std); mh.has_value())
                    { h_meas = mh->height; h_sd = mh->sd; }
            const float w_sd = cand.width_sd;
            // one-model: the base rate is the wall's joint count posterior, assigned below once every member has seen this frame (assign_wall_counts);
            // band, size and extent are replaced here, the base rate stays what the last assignment left
            const float held_base = cand.belief.prior.base_rate;
            cand.belief.prior = prior_of(th, polygon, fit_.room_is_proto(), fit_.room_band(),
                                         wall >= 0 ? doors_on_wall[wall] : 0, w_sd, h_sd, cand.size_resolved, size_prior,
                                         std::sqrt(cand.var),    // no fit yet: the detection's own place sd (isotropic, so along the normal too)
                                         std::sqrt(cand.var), std::sqrt(cand.var),   // …and along the wall, at both jambs
                                         h_meas, nullptr, {});
            if (cfg.exist_one_model) cand.belief.prior.base_rate = held_base;
            if (cand.size_resolved) cand.best_size = std::max(cand.best_size, cand.belief.prior.size);
            const auto it = self_det.find(c);
            const int self_sensor = it == self_det.end() ? -1 : (dets_[static_cast<std::size_t>(it->second)].sensor != 0 ? 1 : 0);
            float self_w = it == self_det.end() ? 0.0f : det_w_[static_cast<std::size_t>(it->second)];
            std::vector<ViewRow> rows;
            if (cfg.exist_one_model)
            {   // the candidate sees the detections as ITS cause's responsibility: no gate, no one-detection rule for the hits (S8, S9)
                if (it != self_det.end())
                    self_w *= static_cast<float>(mask_causes_[static_cast<std::size_t>(it->second)].hit_new());
                const std::vector<FrameDetection> place = place_detections(0, static_cast<int>(c));
                ExistenceFrame fr_c = fr;
                fr_c.dets = place;
                observe_place(cand.belief, cand.theta, fr_c, visit_, 0.0f, &rows, self_sensor, self_w, std::nullopt,
                              candidate_lidar(cand.theta, cand.leaf_born, groups));
            }
            else
            observe_place(cand.belief, cand.theta, fr, visit_, 0.0f, &rows, self_sensor, self_w);
            log_view_rows(std::format("place_{}", cand.key), rows, out);   // logged (no selection by birth), never learned from
        }
        // one-model: every member of the room has seen this frame; the wall's count posterior gives each its base rate, then the verdicts
        if (cfg.exist_one_model) assign_wall_counts();
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            Candidate& cand = candidates_[c];
            if (cand.room != room_name_) continue;
            const auto it = self_det.find(c);
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
            // kept: ONE detection per candidate carries the birth verdict (S8) even under the one-model flag: the tracker's own same-frame
            // duplicate test does not see its promoted candidates, so two verdicts at one place would be two births. The HITS of the
            // candidate's other detections are already counted by their responsibility (place_detections).
            dets_[d].birth_evidence = own and exists[static_cast<std::size_t>(c)] ? 1.0f : 0.0f;
            if (not own) dets_[d].birthable = false;
        }
}

namespace
{
float candidate_R(const DoorCore::Candidate& c) { return 0.5f * std::hypot(c.theta.aperture.w, c.theta.aperture.h); }
}

void DoorCore::retire_wall_seeds(const std::string& room)
{
    std::erase_if(candidates_, [&](const Candidate& c) { return c.room == room and c.wall_seed; });
}

bool DoorCore::hypothesise_place(const std::string& room, const Vec2f& xy, bool wall_seed)
{
    if (room != room_name_) return false;
    const auto& cfg = cfg_.door;
    const float var = cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
    for (const auto& c : candidates_)
        if (c.room == room and (c.xy - xy).squaredNorm() / std::max(1e-9f, c.var + var) <= cfg.tracker_gate_mahalanobis) return true;   // already a question
    Candidate c;
    c.key = ++next_candidate_key_;
    c.room = room;
    c.xy = xy;
    c.var = var;
    c.width_meas = cfg.door_prior_w_m;
    c.wall_seed = wall_seed;
    c.size_resolved = false;   // nothing measured with the door's own detector yet: the size term is 0
    c.width_sd = cfg.tracker_detection_noise_m;
    c.theta = typical_door_at(xy, cfg.door_prior_w_m, cfg.door_prior_h_m, fit_.room_polygon());
    candidates_.push_back(std::move(c));
    return true;
}

static float question_L(const DoorCore::Candidate& c)
{
    return c.belief.L() - c.belief.prior.size + std::max(c.best_size, c.belief.prior.size);
}

// One-model: the question judged under the REFUSAL's base rate (Candidate::base_open): the place's evidence and the live doors' charge, not the
// probability the wall's count posterior shares among its undecided places.
static float question_L_open(const DoorCore::Candidate& c)
{
    return std::isfinite(c.base_open) ? question_L(c) - c.belief.prior.base_rate + c.base_open : question_L(c);
}

void DoorCore::note_competition(std::vector<NodeRef> nodes) { competes_.note(std::move(nodes)); }

void DoorCore::refresh_competition()
{
    if (not cfg_.door.exist_one_model) return;
    std::map<NodeRef, int> now;
    const auto costs = costs_.costs(cfg_.door.exist_max_room_s);
    for (const auto& [id, st] : world_->fit_states())
        if (st != nullptr and not st->is_bearing_hypothesis)
        {
            const auto v = decide(st->belief.L(), st->role, costs);
            now[{'d', id}] = (v.exists ? 1 : 0) + (v.remove ? 2 : 0);
        }
    for (const auto& c : candidates_) now[{'c', c.key}] = candidate_is_open(c) ? 1 : 0;
    competes_.refresh(now);
}

std::vector<rc::door::hlook::Alt> DoorCore::leaf_alternatives(const Candidate& c) const
{
    std::vector<rc::door::hlook::Alt> out;
    const auto& polygon = fit_.room_polygon();
    if (not cfg_.door.exist_one_model or not c.leaf_born or polygon.size() < 3) return out;
    const rc::door::LeafPose lp = rc::door::leaf_pose(c.theta.aperture, c.theta.leaf);
    const Vec2f end_a = lp.hinge_xy, end_b = lp.hinge_xy + lp.ex * (2.f * lp.half_w);
    const auto& groups = groups_cached();
    const auto hyps = rc::door::apertures_from_leaf(end_a, end_b, c.theta.aperture.h, polygon, groups, existence_size_prior(cfg_.door), fit_.room_band(), &rates_);
    std::vector<double> lw(hyps.size(), -std::numeric_limits<double>::infinity());
    for (std::size_t h = 0; h < hyps.size(); ++h)
    {
        if (not (hyps[h].leaf.phi <= cfg_.door.phi_max_rad)) continue;   // beyond the hinge's range: no support (as the births')
        double v = hyps[h].log_w;
        if (leaf_sweep_.present())
        {
            const std::pair<float, float> pt{hyps[h].leaf.phi, 1.0f};
            v += rc::door::opening_view_llr(hyps[h].aperture, registered(hyps[h].aperture), detectors_.lidar.through_open, detectors_.lidar.through_wall,
                                            std::span<const std::pair<float, float>>(&pt, 1), live_claims(hyps[h].aperture, groups));
        }
        lw[h] = v;
    }
    // the candidate's own hypothesis: the one whose aperture is nearest to its hole
    std::size_t own = hyps.size();
    float best = std::numeric_limits<float>::max();
    for (std::size_t h = 0; h < hyps.size(); ++h)
        if (const float d = (hyps[h].aperture.centre_xy() - c.theta.aperture.centre_xy()).norm(); d < best and std::isfinite(lw[h])) { best = d; own = h; }
    if (own == hyps.size()) return out;
    for (std::size_t h = 0; h < hyps.size(); ++h)
        if (h != own and std::isfinite(lw[h]))
            out.push_back({hyps[h].aperture, hyps[h].leaf, lw[h] - lw[own]});
    return out;
}

bool DoorCore::candidate_is_open(const Candidate& c) const
{
    // ★The question is judged on the widest the blob ever looked (clean25): the size term is this frame's, and one clipped mask of width 0 sank
    //  L to -11.9, read as "refused", and ended the tour. A mask's width is a lower bound under occlusion, so a narrow frame says nothing against
    //  a place that once looked door-sized; a place that only ever looks narrow keeps its low best term and is refused.
    return not decide(cfg_.door.exist_one_model ? question_L_open(c) : question_L(c), Role::Ordinary, costs_.costs(cfg_.door.exist_max_room_s)).remove;
}

float DoorCore::candidate_question_p(const Candidate& c) const
{
    return 1.0f / (1.0f + std::exp(-question_L(c)));
}

bool DoorCore::candidate_view_is_new(const Candidate& c, const Vec2f& standpoint) const
{
    return c.belief.ledger.find_view(standpoint, c.theta.aperture.centre_xy(), c.theta.aperture.across_u(), candidate_R(c), 0.0f, visit_) == ViewLedger::npos;
}

bool DoorCore::candidate_site_untried(const Candidate& c, const ViewLedger& tried, const Vec2f& site) const
{
    return tried.find_view(site, c.theta.aperture.centre_xy(), c.theta.aperture.across_u(), candidate_R(c), 0.0f, visit_) == ViewLedger::npos;
}

void DoorCore::mark_candidate_site_tried(const Candidate& c, ViewLedger& tried, const Vec2f& site) const
{
    tried.observe(site, c.theta.aperture.centre_xy(), c.theta.aperture.across_u(), candidate_R(c), 0.0f, visit_);
}

void DoorCore::update_existence(const DoorCycleInput& in, DoorCycleOutput& out)
{
    const auto& cfg = cfg_.door;
    const SizePrior size_prior = existence_size_prior(cfg);
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
    std::vector<std::vector<ViewRow>> door_rows(states.size());
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
            inst.role = door_role(inst.node_name, in.route_faces, being_crossed.contains(inst.node_name) or inst.fact_backed or inst.belief.crossing_protected());   // a crossed or crossing door: no delete line
            // the structural prior, REPLACED (never accumulated); a newborn not yet fitted keeps its candidate's
            if (inst.ai2_initialized)
            {
                int before = 0;
                if (not cfg.exist_one_model)   // the integer rule (birth order); the one-model base rate is the wall's count posterior (assign_wall_counts)
                    for (std::size_t j = 0; j < states.size(); ++j)
                        if (j != i and wall[j] == wall[i] and wall[i] >= 0 and not states[j].second->is_bearing_hypothesis
                            and door_seq_[states[j].first] < door_seq_[id])
                            ++before;
                const float held_base = inst.belief.prior.base_rate;
                const auto C = inst.ai2_belief.covariance();
                // σ_ap, the aperture's sd ALONG THE WALL NORMAL: the fit's belief is [s, w, h, d] in its wall frame, and its only
                // across-wall state is d, the door plane's depth off the wall line; its posterior sd (Σ(3,3), recess_sd()) is the
                // fit's own uncertainty of where across the wall the door is. One long-range view leaves it near its prior
                // (sqrt(T²/12 + σ_b²)); a well-measured door collapses it.
                inst.belief.prior = prior_of(inst, polygon, fit_.room_is_proto(), fit_.room_band(), before,
                                             std::sqrt(std::max(0.0f, C(1, 1))), std::sqrt(std::max(0.0f, C(2, 2))), true, size_prior,
                                             inst.ai2_belief.recess_sd(),
                                             std::sqrt(std::max(0.0f, C(0, 0))),                                // the near jamb: s
                                             std::sqrt(std::max(0.0f, C(0, 0) + C(1, 1) + 2.0f * C(0, 1))),   // the far jamb: s + w
                                             std::numeric_limits<float>::quiet_NaN(), nullptr, {});
                if (cfg.exist_one_model) inst.belief.prior.base_rate = held_base;
            }
            const float act = std::exchange(inst.pending_actuation_llr, 0.0f);
            std::vector<ViewRow> rows;
            // ★A newborn's belief already holds its birth frame (its candidate observed it): it may not enter twice (fix round 1).
            ExistenceFrame fr_door = fr;
            if (const auto of = observed_frame_.find(id); of != observed_frame_.end() and of->second == pkt.frame_id)
                fr_door.zed_ran = fr_door.ricoh_ran = fr_door.ricoh_hits_only = false;
            // ★The fit's own support, consumed once: the first fit of a door runs in the same cycle as its birth (after this update) and is
            //  scored here on the next frame. No fit since the last update → no term (door_fit.cpp sets it only when a fit ran).
            std::vector<FrameDetection> place_dets;   // one-model: the detections as this door's responsibility (no gate)
            if (cfg.exist_one_model)
            {
                place_dets = place_detections(id, -1);
                fr_door.dets = place_dets;
            }
            inst.exist_trace = observe_place(inst.belief, inst, fr_door, visit_, act, &rows, -1, 0.0f, std::exchange(inst.pending_fit_support, std::nullopt),
                                             std::exchange(inst.pending_lidar, std::nullopt));
            // A door that descends from a remembered row: its first detection in this visit finds the place again (the learned
            // hazard's "unchanged" sample; one per place per visit - PlaceTable::sample_revisit). The sighting moves to now.
            if (not inst.place_row_name.empty() and (inst.exist_trace.zed_hit > 0.0f or inst.exist_trace.ric_hit > 0.0f))
                if (PlaceRow* row = places_.find(room_name_, inst.place_row_name); row != nullptr)
                {
                    places_.sample_revisit(*row, visit_, not row->removed_wall_ms.has_value(), in.t_wall_ms);
                    row->sighted_wall_ms = in.t_wall_ms;
                }
            if (act != 0.0f and not std::isfinite(inst.exist_trace.act_dl))
                inst.pending_actuation_llr += act;   // no view and no ZED pose this cycle: the evidence waits (fix round 1)
            door_rows[i] = std::move(rows);
        }
        // one-model: every door has seen this frame; the wall's count posterior gives each its base rate, then the verdicts
        if (cfg.exist_one_model) assign_wall_counts();
        for (std::size_t i = 0; i < states.size(); ++i)
        {
            const auto& [id, instp] = states[i];
            rc::DoorFitState& inst = *instp;
            if (inst.is_bearing_hypothesis) continue;
            const float L = inst.belief.L();
            const Verdict v = decide(L, inst.role, costs);
            inst.exist_trace.delete_line = v.delete_line;
            if (v.remove)
                to_remove.push_back(id);
            log_view_rows(inst.node_name, door_rows[i], out);
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
               << t.geo_x << ',' << t.geo_mu0 << ',' << t.geo_mu1 << ',' << t.geo_s << ',' << t.sup_x << ',' << t.sup_dl << ','
               << b.Sc() << ',' << b.Sp() << ',' << b.saturation_term() << ',' << t.sup_c << ',' << t.sup_q << ',' << b.prior.extent << '\n';
        diag("door_existence_log.csv", ex_csv.str(), out);
        const bool removing = std::ranges::contains(to_remove, id);
        if (ex_dbg_ % 60 == 0 or removing or std::abs(b.L() - L_before[i]) > 1.0f)
            diag("stdout", std::format("door_concept: [existence] {} L={:+.2f} p={:.2f} role={} delete_line={} | prior band={:+.2f} "
                                       "base={:+.2f} size={:+.2f} extent={:+.2f} carried={:+.2f} cross={:+.2f} Sc={:+.2f} Sp={:+.2f} term={:+.2f} | views={} | zed dl={} q={} hit={} "
                                       "| ric dl={} q={} hit={} | geo dl={} | sup dl={} x={} c={} on={} | act dl={} | kappa={}{}\n",
                                       inst.node_name, b.L(), b.p(), role_name(inst.role), fmt_or_dash(t.delete_line),
                                       b.prior.band, b.prior.base_rate, b.prior.size, b.prior.extent, b.carried, b.crossing, b.Sc(), b.Sp(), b.saturation_term(),
                                       b.ledger.views().size(), fmt_or_dash(t.zed_dl), fmt_or_dash(t.zed_q), fmt_or_dash(t.zed_hit),
                                       fmt_or_dash(t.ric_dl), fmt_or_dash(t.ric_q), fmt_or_dash(t.ric_hit), fmt_or_dash(t.geo_dl),
                                       fmt_or_dash(t.sup_dl), fmt_or_dash(t.sup_x), fmt_or_dash(t.sup_c), fmt_or_dash(t.sup_q), fmt_or_dash(t.act_dl), fmt_or_dash(t.kappa),
                                       cfg.exist_one_model ? std::format(" | lid dl={}", fmt_or_dash(t.lid_dl)) : std::string{}), out);
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
        // The robot is back in this room, the place was in view (the door was judged on views with q > 0) and the remembered door is
        // gone: a CHANGE since the row was last sighted - one sample per place per visit (taken BEFORE the removed row replaces it).
        if (not inst.place_row_name.empty())
            if (const PlaceRow* row = places_.find(room_name_, inst.place_row_name); row != nullptr)
                places_.sample_revisit(*row, visit_, false, in.t_wall_ms);
        diag("door_cost_events.csv", std::format("remove,{},{},{},{},{},0,{}\n", in.t_wall_ms, room_name_, inst.node_name,
                                                 inst.aperture.centre_xy().x(), inst.aperture.centre_xy().y(), inst.node_name), out);
        PlaceRow r;
        r.room = room_name_; r.name = inst.node_name; r.xy = inst.aperture.centre_xy();
        const float sd2 = inst.ai2_initialized ? std::max(0.0f, inst.ai2_belief.covariance()(0, 0)) : cfg.tracker_detection_noise_m * cfg.tracker_detection_noise_m;
        r.cov = Eigen::Matrix2f::Identity() * sd2;
        r.yaw = inst.aperture.yaw(); r.width = inst.aperture.w; r.height = inst.aperture.h; r.depth = inst.aperture.thickness;
        if (inst.ai2_initialized) r.height_sd = std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(2, 2)));   // θ = [s, w, h, d]: what a candidate here re-reads (remembered_height)
        r.L_evidence = inst.belief.L_evidence(); r.physical = float(inst.belief.Sp()); r.crossed = inst.belief.crossed; r.ledger = inst.belief.ledger; r.visit = visit_;
        r.last_seen_wall_ms = in.t_wall_ms; r.sighted_wall_ms = in.t_wall_ms; r.removed_wall_ms = in.t_wall_ms;
        places_.upsert(std::move(r));
        observed_frame_.erase(id);
        world_->before_remove(id);
        const auto& s = inst.model.state();
        record(K::Remove, id, s.cx, s.cy, std::format("logodds {:.2f}", L), inst.node_name, out);
        out.deletes.push_back({id, "existence"});
        world_->remove_door(id, L);
    }

    // This cycle's p of every live door and candidate: the NEXT update's "other doors" (one-model per-wall rate; birth order plays no part).
    if (cfg.exist_one_model)
    {
        last_p_door_.clear();
        last_p_cand_.clear();
        for (const auto& [id, st] : world_->fit_states())
            if (not st->is_bearing_hypothesis) last_p_door_[id] = st->belief.p();
        for (const Candidate& c : candidates_)
            last_p_cand_[c.key] = c.belief.p();
    }
}

}  // namespace rc::door::core
