/*
 * cabinet_fitter.cpp — the active-inference fit core for cabinet_concept (see cabinet_fitter.h).
 *
 * Implements the per-"cabinet_*" instance lifecycle and the AI2 full-covariance belief update: ensure_instance
 * (birth-seed + RT/prior warm-start + NaN sanitize), observe_slice (mask-cloud → candidate/residual SDF split
 * + per-slice R inputs), and run_inference (lazy footprint-moment birth, voxel-bank ingest, one CabinetBelief
 * update with range/motion covariance, the step-bound divergence net, FE-surprise attention, orientation-mode
 * resolution, and write-back into the legacy CabinetState). Collaborates with MaskIngestor, CabinetSceneGraph,
 * CabinetProjection, CabinetLidarRangeChannel, and the header-only voxel bank; SpecificWorker owns orchestration.
 */

#include "cabinet_fitter.h"
#include "cabinet_voxel_bank.h"
#include "../../common/object_anchor/object_anchor_contract.h"
#include "../../common/object_anchor/ray_anisotropic_cov.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <print>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rc {

// ─── Construction & chain covariance ──────────────────────────────────────────────────────────────

CabinetFitter::CabinetFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         CabinetConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         CabinetSceneGraph* scene_graph)
    : G_(graph), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph),
      projection_(std::make_unique<CabinetProjection>(graph, inner_eigen, mask_ingestor))
{
    // (No symmetry-quotient chart: a run has no w<->h fold. Its only symmetry is the box's 180° yaw
    // ambiguity, folded geometrically in CabinetBelief::canonicalize against the room interior.)
}

// Enable Part-B chain-covariance propagation from source_frame (no-op unless a gaussian API + frame are given).
void CabinetFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void CabinetFitter::set_object_observation(bool enabled, std::string robot_frame)
{
    obs_robot_frame_ = std::move(robot_frame);
    obs_enabled_     = enabled and (inner_eigen_ != nullptr) and not obs_robot_frame_.empty();
}

// Build the object-anchor observation z_o for room_concept's landmark factor.
//
// z_o MUST be independent of the robot pose the localizer is estimating, or the factor just
// re-anchors to the last pose (the residual is ~0 at the current estimate). So we take the cabinet's
// RAW camera-frame mask centroid (this frame's ZED measurement, no localization in it) and carry it
// to the robot base by the STATIC body←zed extrinsic (ts=0, a fixed calibrated mount). Position-only:
// a single view's yaw is biased (the grazing/obliquity problem), so we publish [x,y] and let the
// consumer treat it as a 2-DOF landmark. Gated OFF by default.
void CabinetFitter::compute_object_observation(CabinetInstance& inst)
{
    inst.obs_robot_valid = false;
    if (not obs_enabled_ or not inner_eigen_ or not mask_ingestor_)
        return;
    const auto& packet = mask_ingestor_->packet();
    if (packet.support_points_cam.empty())
        return;

    // This frame's ZED slice assigned to this cabinet (the pinhole camera cloud; skip ricoh depth_var>0).
    const MaskIngestor::MaskSlice* zed_slice = nullptr;
    for (const int idx : inst.assigned_mask_idxs)
    {
        if (idx < 0 or idx >= static_cast<int>(packet.slices.size())) continue;
        const auto& sl = packet.slices[idx];
        if (sl.depth_var > 0.0f) continue;                      // ricoh / lidar-depth mask, not the ZED frame
        if (sl.support_end > sl.support_begin) { zed_slice = &sl; break; }
    }
    if (zed_slice == nullptr)
        return;

    // Centroid of the RAW camera-frame support points — the pose-INDEPENDENT sensor measurement.
    const std::size_t b = std::min<std::size_t>(zed_slice->support_begin, packet.support_points_cam.size());
    const std::size_t e = std::min<std::size_t>(zed_slice->support_end,   packet.support_points_cam.size());
    if (e <= b)
        return;
    Eigen::Vector3d c_cam = Eigen::Vector3d::Zero();
    for (std::size_t k = b; k < e; ++k)
        c_cam += packet.support_points_cam[k].cast<double>();
    c_cam /= static_cast<double>(e - b);

    // STATIC camera→base extrinsic (ts=0, rigid mount) — carries NO localization pose ⇒ z_o ⟂ robot pose.
    const auto m = inner_eigen_->get_transformation_matrix(obs_robot_frame_, obs_cam_frame_, 0);
    if (not m.has_value())
        return;
    const Mat::Vector3d p = m.value() * c_cam;
    inst.obs_robot       = {static_cast<float>(p.x()), static_cast<float>(p.y()), 0.0f};  // position-only

    // Anisotropic measurement covariance R_o (body frame): loose along the viewing ray (near-face/partial-
    // view depth bias), tight ⊥. Ray origin = camera optical centre in body = the extrinsic translation.
    // box_yaw in body = belief yaw (room) + yaw(body←room). See CABINET_TRIANGULATION.md.
    {
        const Mat::Vector3d cam_o = m.value() * Mat::Vector3d(0, 0, 0);
        const Eigen::Vector2f cam_xy{static_cast<float>(cam_o.x()), static_cast<float>(cam_o.y())};
        const Eigen::Vector2f cen_xy{inst.obs_robot.x(), inst.obs_robot.y()};
        const auto bs = inst.ai2_belief.state();
        float box_yaw_body = bs.yaw;
        if (const auto br = inner_eigen_->get_transformation_matrix(obs_robot_frame_, "room", 0); br.has_value())
            box_yaw_body = bs.yaw + static_cast<float>(std::atan2(br.value().linear()(1, 0),
                                                                  br.value().linear()(0, 0)));
        inst.obs_robot_cov = rc::object_anchor::ray_anisotropic_cov(
            cen_xy, cam_xy, {0.5f * bs.L, 0.5f * bs.d}, box_yaw_body,
            std::max(0.1f, inst.last_range), std::clamp(inst.dbg_obliquity_cos, 0.f, 1.f),
            rc::object_anchor::RayCovParams{});

        // ★ ALONG-RUN inflation — mandatory for a run, and NOT what ray_anisotropic_cov models.
        // That helper declares the VIEWING RAY untrustworthy (near-face depth bias). A run has a
        // second, larger unobservability that is independent of where the camera is: sliding the
        // centre ALONG a long uniform run barely changes anything observable, so the along-axis
        // coordinate of the centroid carries almost no information. Publishing an anchor without
        // this would tell room_concept we know the along-wall coordinate to ~centimetres and would
        // poison localization — precisely the failure the lateral/yaw information is valuable enough
        // to be worth avoiding. Variance grows with the run's own length (an L≈0 run is a point
        // landmark and keeps full information; a 4 m run is a line landmark).
        {
            const float run_c = std::cos(box_yaw_body), run_s = std::sin(box_yaw_body);
            const Eigen::Vector2f a(run_c, run_s);              // run axis in the body frame
            const float sigma_along = 0.5f * std::max(0.0f, bs.L);   // ~half-length: honest, not tuned
            inst.obs_robot_cov += (sigma_along * sigma_along) * (a * a.transpose());
        }
    }
    inst.obs_robot_valid = true;
}

// ─── Nearest room wall (feeds the wall-flush factor) ─────────────────────────────────────────────

// Closest point on each polygon edge, keeping the nearest. The inward normal is chosen by pointing it
// at the room interior (the polygon centroid) rather than assuming a winding order, so a polygon
// authored either CW or CCW — and the apartamento layout's concave jogs — both behave.
// Group the polygon's edges into collinear-merged WALLS; cache each edge's canonical wall id (the start
// edge of its collinear run, found by walking backward while the neighbour stays parallel to within 3°).
void CabinetFitter::rebuild_wall_ids()
{
    const std::size_t n = room_polygon_.size();
    wall_seg_id_.assign(n, -1);
    if (n < 2) return;
    constexpr float kColinCos = 0.99863f;   // cos(3°)
    const auto dir = [&](std::size_t i)
    { return (room_polygon_[(i + 1) % n] - room_polygon_[i]).normalized(); };
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f ui = dir(i);
        std::size_t start = i;
        for (std::size_t step = 0; step + 1 < n; ++step)
        {
            const std::size_t prev = (start + n - 1) % n;
            if (std::abs(dir(prev).dot(ui)) < kColinCos) break;
            start = prev;
        }
        wall_seg_id_[i] = static_cast<int>(start);
    }
}

// Nearest wall to a point: canonical id, distance, unit direction. Lean per-point version for the wall-split.
CabinetFitter::PointWall CabinetFitter::point_wall(const Eigen::Vector2f& q) const
{
    PointWall out;
    const std::size_t n = room_polygon_.size();
    if (n < 2 or wall_seg_id_.size() != n) return out;
    float best = std::numeric_limits<float>::max();
    std::size_t bi = n;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-8f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const float d2 = (q - (a + t * ab)).squaredNorm();
        if (d2 < best) { best = d2; bi = i; }
    }
    if (bi == n) return out;
    out.id   = wall_seg_id_[bi];
    out.dist = std::sqrt(best);
    out.dir  = (room_polygon_[(bi + 1) % n] - room_polygon_[bi]).normalized();
    return out;
}

// Build the full WallRef (foot, inward normal, collinear-merged segment corners, canonical seg_id) for a
// GIVEN polygon edge, projecting q onto it. Shared by nearest_wall (argmin edge) and wall_ref_by_seg_id
// (committed edge). The segment walk collinear-merges an authored jog/door-notch into ONE segment so
// accumulate_extent doesn't clamp the run at a phantom interior corner; a true corner (≈90°) stops it.
WallRef CabinetFitter::build_wall_ref(std::size_t edge_i, const Eigen::Vector2f& q) const
{
    WallRef w;
    const std::size_t n = room_polygon_.size();
    if (n < 2 or edge_i >= n) return w;
    const Eigen::Vector2f& a = room_polygon_[edge_i];
    const Eigen::Vector2f  b = room_polygon_[(edge_i + 1) % n];
    const Eigen::Vector2f  ab = b - a;
    const float len2 = ab.squaredNorm();
    if (len2 < 1e-8f) return w;
    const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
    const Eigen::Vector2f foot = a + t * ab;
    Eigen::Vector2f nrm(-ab.y(), ab.x());
    nrm.normalize();
    if (nrm.dot(room_interior_ - foot) < 0.0f) nrm = -nrm;   // point INTO the room
    w.ok = true; w.p = foot; w.n = nrm; w.sigma_m = cfg_.wall_sigma_m;

    constexpr float kColinCos = 0.99863f;   // cos(3°): edges within 3° are the "same" wall
    const auto edge_dir = [&](std::size_t i)
    { return (room_polygon_[(i + 1) % n] - room_polygon_[i]).normalized(); };
    const Eigen::Vector2f u0 = edge_dir(edge_i);
    std::size_t lo = edge_i, hi = edge_i;   // edge indices; segment spans vertex lo .. vertex (hi+1)
    for (std::size_t step = 0; step + 1 < n; ++step)
    { const std::size_t prev = (lo + n - 1) % n; if (std::abs(edge_dir(prev).dot(u0)) < kColinCos) break; lo = prev; }
    for (std::size_t step = 0; step + 1 < n; ++step)
    { const std::size_t nxt = (hi + 1) % n; if (std::abs(edge_dir(nxt).dot(u0)) < kColinCos) break; hi = nxt; }
    w.a = room_polygon_[lo];
    w.b = room_polygon_[(hi + 1) % n];
    w.has_segment = true;
    w.seg_id = (wall_seg_id_.size() == n) ? wall_seg_id_[edge_i] : static_cast<int>(lo);
    return w;
}

// One KitchenWall per canonical wall id (collinear-merged corners + inward normal). Stage 0 cell geometry.
std::vector<rc::KitchenWall> CabinetFitter::kitchen_walls() const
{
    std::vector<rc::KitchenWall> out;
    const std::size_t n = room_polygon_.size();
    if (n < 2 or wall_seg_id_.size() != n) return out;
    std::unordered_set<int> seen;
    for (std::size_t i = 0; i < n; ++i)
    {
        const int sid = wall_seg_id_[i];
        if (sid < 0 or seen.count(sid)) continue;
        seen.insert(sid);
        const WallRef wr = build_wall_ref(i, room_polygon_[i]);   // corners a,b + inward normal for this wall
        if (not wr.ok or not wr.has_segment) continue;
        rc::KitchenWall w;
        w.a = wr.a; w.b = wr.b; w.W = (wr.b - wr.a).norm();
        if (w.W < 1e-4f) continue;
        w.u = (wr.b - wr.a) / w.W; w.n = wr.n; w.seg_id = sid;
        out.push_back(w);
    }
    return out;
}

WallRef CabinetFitter::nearest_wall(const Eigen::Vector2f& q) const
{
    const std::size_t n = room_polygon_.size();
    if (n < 2) return WallRef{};                    // no room model ⇒ inert ⇒ treated as free-standing
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f  ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-8f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const float dist2 = (q - (a + t * ab)).squaredNorm();
        if (dist2 < best) { best = dist2; best_i = i; }
    }
    return (best_i < n) ? build_wall_ref(best_i, q) : WallRef{};
}

// The committed wall's nearest edge (by seg_id), projected onto for q. Returns ok=false if the id is gone.
WallRef CabinetFitter::wall_ref_by_seg_id(int seg_id, const Eigen::Vector2f& q) const
{
    const std::size_t n = room_polygon_.size();
    if (n < 2 or wall_seg_id_.size() != n or seg_id < 0) return WallRef{};
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (wall_seg_id_[i] != seg_id) continue;
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f  ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-8f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const float dist2 = (q - (a + t * ab)).squaredNorm();
        if (dist2 < best) { best = dist2; best_i = i; }
    }
    return (best_i < n) ? build_wall_ref(best_i, q) : WallRef{};
}

// Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the cabinet centre; store it on the instance.
//
// Transform the centre to the measurement frame and back to room with ZERO input cov, so InnerGaussianAPI
// returns exactly the chain contribution (Σ_chain from each RT edge's rt_covariance), pinned to the mask
// capture stamp. The cabinet is fit in room but its position stays conditional on the robot pose
// (camera→robot→room), so this per-frame SHARED localization error feeds the belief common-mode.
void CabinetFitter::compute_chain_cov(CabinetInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the cabinet centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution
    // (Σ_chain from each RT edge's rt_covariance), pinned to the mask capture stamp. The cabinet is fit in
    // room but its position is still conditional on the robot pose (camera→robot→room), so this applies.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, 0.0);   // cabinet node origin = base on the floor (z=0)
    const auto c_src = inner_eigen_->transform(chain_src_frame_, centre, "room", inst.last_mask_timestamp_ms);
    if (not c_src.has_value())
        return;
    DSR::GaussianPoint3D gp;
    gp.mean = c_src.value();
    gp.covariance = DSR::Cov3d::Zero();
    const auto g = gaussian_->transform_point("room", gp, chain_src_frame_, inst.last_mask_timestamp_ms);
    if (not g.has_value())
        return;
    inst.chain_cov_xx = static_cast<float>(g->covariance(0, 0));
    inst.chain_cov_yy = static_cast<float>(g->covariance(1, 1));
}

// ─── Instance lifecycle ───────────────────────────────────────────────────────────────────────────

// Create the instance for a "cabinet_*" node if absent; return true only the first time it is created.
//
// Warm-starts from the node's size attribs and the room→cabinet RT edge, then OVERRIDES with the tracker
// birth seed when present (the RT edge written at birth is not reliably queryable this same cycle — it reads
// 0,0 — which would freeze the model at the origin and cause endless re-births). Sanitizes any non-finite
// field before it can poison the SDF and lock the optimizer.
bool CabinetFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    CabinetState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.L = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.d = v.value();
    // height_m is the CARCASS height (z1−z0); the RT edge (read below) supplies z0, so reconstruct z1.
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.z1 = init_state.z0 + v.value();

    // Read RT pose from room→cabinet edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // Tracker birth seed: authoritative for a freshly-born instance. The room→cabinet RT edge written at
    // birth is not reliably queryable this same cycle (it reads as 0,0), and the warm-start would then
    // freeze the model at the origin forever → the tracker never associates and re-births endlessly.
    // A residual-born run carries a FULL room-axis seed of its arm — commit cx,cy,yaw,L so its footprint
    // is correct immediately (the tracker's footprint-claim uses it next cycle) and mark it residual_born.
    // The seed is kept (not erased) so the lazy belief init below applies the SAME arm rather than the
    // shared slice's dominant (parent) arm.
    const bool has_full_seed = birth_full_seeds_.find(node.id()) != birth_full_seeds_.end();
    if (has_full_seed)
    {
        const RunSeed& fs = birth_full_seeds_.at(node.id());
        init_state.cx = fs.cx; init_state.cy = fs.cy; init_state.yaw = fs.yaw;
        init_state.L  = std::max(0.10f, fs.L);
        std::print("[{}] residual birth-seed applied → cx={:.2f} cy={:.2f} yaw={:.2f} L={:.2f}\n",
                   node.name(), fs.cx, fs.cy, fs.yaw, fs.L);
    }
    else if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        // Seed the vertical band from the detection centroid so a WALL-unit birth (centroid z≈1.7) starts
        // HIGH, not on the floor. Otherwise the newborn's track-z is base-like and the z_gate blocks the
        // unit's own mask from ever associating/fitting (see cabinet_scene_graph birth). Keep the carcass
        // height; centre the band on the detection centroid.
        const float h = std::max(0.10f, init_state.z1 - init_state.z0);
        init_state.z0 = std::max(0.0f, it->second.z() - 0.5f * h);
        init_state.z1 = init_state.z0 + h;
        std::print("[{}] birth-seed applied → cx={:.2f} cy={:.2f} z0={:.2f}\n",
                   node.name(), init_state.cx, init_state.cy, init_state.z0);
        birth_seeds_.erase(it);
    }
    else
        std::print("[{}] NO birth-seed (id={}) → init cx={:.2f} cy={:.2f}\n",
                   node.name(), node.id(), init_state.cx, init_state.cy);

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF and lock the
    // optimizer; replace any non-finite field with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("cabinet_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.L, 1.0f, "L");
        fix(init_state.d, cfg_.base_depth_m, "d");
        fix(init_state.z0, 0.0f, "z0");
        fix(init_state.z1, cfg_.base_z1_m, "z1");
        if (init_state.z1 <= init_state.z0)
            init_state.z1 = init_state.z0 + cfg_.base_z1_m;
    }

    CabinetInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.residual_born = has_full_seed;   // gets the footprint-claim slice feed in run_instance_tracker

    inst.model = CabinetModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("cabinet_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ──────────────────────────────────────────────────────────────────────────────────

// Build an observation from ONE assigned mask slice: latch that slice's R inputs, then SDF-split its cloud.
//
// Classify-don't-destroy split into candidate (near the current surface) vs residual (off-surface) points.
// The caller (process_cabinet_node) invokes this once per assigned slice and runs a belief update for each —
// sequential fusion that keeps every sensor's R and common-mode separate. RICOH slices (depth_var>0) are
// BEARING-ONLY and return empty (never fitted); they drive only the attention path.
CabinetFitter::CabinetObservation CabinetFitter::observe_slice(CabinetInstance& inst, int slice_index)
{
    CabinetObservation observation;
    const auto& masks_packet = mask_ingestor_->packet();
    if (not masks_packet.valid or slice_index < 0
        or slice_index >= static_cast<int>(masks_packet.slices.size()))
        return observation;

    const auto& slice = masks_packet.slices[slice_index];
    // RICOH (depth_var>0) is BEARING-ONLY now: a peripheral 360 detection has a reliable DIRECTION but a biased
    // centroid/extent (partial oblique view), so it must NEVER touch pose/extent/birth — it drove duplicate
    // births + drift/inflation. It is handled entirely by the attention path (process_ricoh_bearings): an
    // unassigned ricoh bearing raises attention so the robot seeks a good ZED view. Return empty here (never
    // fitted). Defensive — run_instance_tracker also skips ricoh from assignment, so this normally isn't reached.
    if (slice.depth_var > 0.0f)
        return observation;   // has_fresh_data=false ⇒ no fit
    // A slice assigned to this cabinet ⇒ detection is alive; latch its per-slice R inputs.
    inst.frames_since_detection = 0;
    inst.last_mask_confidence   = slice.confidence;
    inst.dbg_pkt_frame_id       = masks_packet.frame_id;       // provenance of THIS fit (instrumentation)
    inst.dbg_pkt_ts_ms          = masks_packet.timestamp_ms;
    inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
    // Ego-motion capture-corruption + depth uncertainty for THIS mask (AI2 obs precision / bias gate; 0 if the
    // producer predates the feature, or for a dense zed slice). See MASK_MOTION_CORRUPTION.md / depth_projection.
    inst.last_motion_var      = slice.motion_var;
    inst.last_motion_dotd     = slice.motion_dotd;
    inst.last_trunc_frac      = slice.trunc_frac;
    inst.last_centroid_radius = slice.centroid_radius;
    inst.last_range           = slice.range;
    inst.last_depth_var       = slice.depth_var;   // mask depth uncertainty → R (ricoh lidar-depth masks)
    // FRAME ego-motion (shared, this whole capture) → common-mode next: |v|, |ω| of the camera twist + dt.
    const auto& tw = masks_packet.cam_twist;   // [vx,vy,vz,wx,wy,wz] (optical frame); zeros if absent
    inst.last_ego_v  = std::hypot(std::hypot(tw[0], tw[1]), tw[2]);
    inst.last_ego_w  = std::hypot(std::hypot(tw[3], tw[4]), tw[5]);
    inst.last_ego_dt = masks_packet.frame_dt_s;

    const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
    const std::size_t end   = std::min(slice.support_end,   masks_packet.support_points.size());

    // Simple SDF split into on-surface (candidate) vs off-surface (residual). An L-corner mask is separated
    // UPSTREAM into two single-arm sub-slices (SpecificWorker::split_lshaped_cabinet_masks + cabinet_lshape_
    // split.h), so this run always sees a clean single-arm cloud — no per-instance corner arbitration here.
    observation.candidate_pts.reserve(end > begin ? end - begin : 0);
    observation.residual_pts.reserve(end > begin ? end - begin : 0);
    for (std::size_t i = begin; i < end; ++i)
    {
        const auto& p = masks_packet.support_points[i];
        const float sdf = inst.model.sdf_point(p);
        if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
            observation.candidate_pts.push_back(p);
        else
            observation.residual_pts.push_back(p);
    }

    const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
    observation.has_fresh_data   = total > 0.0f;

    if (should_log(inst))
        std::print("[{}] masks={} slice={} label='{}' conf={:.2f} dvar={:.4f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                   inst.node_name, masks_packet.frame_id, slice_index, slice.label, slice.confidence, slice.depth_var,
                   end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                   slice.centroid.x(), slice.centroid.y(), slice.centroid.z());
    return observation;
}

// Fallback when the tracker assigned NO mask slice this cycle: always a stale observation (age the belief).
CabinetFitter::CabinetObservation CabinetFitter::observe(CabinetInstance& inst, const DSR::Node& node)
{
    // No mask slice was assigned this cycle. There is no node-attrib sensing path (nothing writes
    // candidate/residual points onto the node), so this is always a stale observation: has_fresh_data=false
    // ⇒ run_inference ages the belief (predict-only) instead of fitting.
    (void)inst; (void)node;
    return CabinetObservation{};
}

// True on the config-driven log period (one in cabinet_log_period_frames cycles).
bool CabinetFitter::should_log(const CabinetInstance& inst) const
{
    if (not cfg_.verbose_log) return false;   // quiet terminal by default (the CSV keeps every cycle)
    const int period = std::max(1, cfg_.cabinet_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Inference ────────────────────────────────────────────────────────────────────────────────────

// One recursive full-covariance belief update (or age-only step) for this instance; returns the free energy.
//
// Lazy first-frame init (snap centre/height to the cloud, footprint-moment birth of w/h/yaw), voxel-bank
// ingest, then the range/motion covariance and the CabinetBelief update guarded by a step-bound divergence net,
// FE-surprise attention baseline, and orientation-mode resolution. On a stale frame it ages the belief
// (Σ grows on the agent's clock) instead of freezing. Result is written back into the legacy CabinetState.
float CabinetFitter::run_inference(CabinetInstance& inst, const CabinetObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy init: warm-start the belief from the model state, but on the FIRST frame snap the centre/height
    // to the observed cloud — a box far from the points would see them all as clutter (zero gradient) and
    // never converge (the AI2 analogue of the legacy cold-start snap).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        CabinetBeliefState s0{m.cx, m.cy, m.yaw, m.L, m.d, m.z0, m.z1};
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            std::vector<float> zs; zs.reserve(npts);
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v)
            { for (const auto& p : v) { sum += p; zs.push_back(p.z()); } };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            s0.cx = sum.x() / npts; s0.cy = sum.y() / npts;
            const std::size_t k = static_cast<std::size_t>(0.95f * (zs.size() - 1));
            std::nth_element(zs.begin(), zs.begin() + k, zs.end());
            s0.z1 = zs[k];   // observed top face

            // Birth seed from the cloud's own principal axis (see CabinetBelief::seed_from_points):
            // start the box near the run's real position, orientation and length. A box seeded small
            // and axis-aligned would route the rest of the run to clutter and — because the per-point
            // mixture cannot grow a run — would never recover; only the censored extent factor could,
            // and much more slowly. Seeding well is what makes the first few frames converge.
            {
                std::vector<Eigen::Vector3f> pts;
                pts.reserve(static_cast<std::size_t>(npts));
                pts.insert(pts.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
                pts.insert(pts.end(), observation.residual_pts.begin(), observation.residual_pts.end());
                if (const auto sd = CabinetBelief::seed_from_points(pts, cfg_.seed_room_axis_snap); sd.ok)
                {
                    s0.cx = sd.cx; s0.cy = sd.cy;
                    s0.L  = std::max(0.10f, sd.L);
                    s0.z0 = std::max(0.0f, sd.z0);
                    s0.z1 = std::max(s0.z0 + 0.10f, sd.z1);
                    // Depth is NOT seeded from the cloud: the back face is never observed, so the
                    // cloud's minor extent measures the visible near-face thickness, not the carcass.
                    // Leave d at the tier's carcass prior and let the wall factor resolve it.
                    //
                    // Commit the orientation only when the cloud is clearly elongated. A nearly square
                    // patch (a single close-up fragment) has an ill-defined principal axis that swings
                    // ±90° on noise; leave yaw at its RT/default seed and let the wall-parallel factor
                    // settle it once the run has length.
                    if (sd.aniso > 0.10f)
                        s0.yaw = sd.yaw;
                }
                // Residual-born run: the seed above came from the WHOLE shared slice, whose dominant arm
                // is the parent's — override with the stored arm seed so this box commits to ITS arm. The
                // per-instance SDF split (observe_slice) then feeds it only the arm's points; the parent's
                // arm falls to residual. Consumed once here.
                if (const auto fit = birth_full_seeds_.find(inst.node_id); fit != birth_full_seeds_.end())
                {
                    const RunSeed& fs = fit->second;
                    s0.cx = fs.cx; s0.cy = fs.cy; s0.yaw = fs.yaw; s0.L = std::max(0.10f, fs.L);
                    birth_full_seeds_.erase(fit);
                }
            }
        }
        CabinetBeliefParams p;
        p.sigma_base_m    = cfg_.ai2_sigma_base_m;
        p.clutter_frac    = cfg_.ai2_clutter_frac;
        p.clutter_scale_m = cfg_.ai2_clutter_scale_m;
        p.prior_L_std     = cfg_.ai2_prior_size_std;
        p.process_std_m   = cfg_.ai2_process_std_m;
        p.process_std_yaw = cfg_.ai2_process_std_yaw;
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
        p.gn_iters        = cfg_.ai2_gn_iters;
        // Run-specific structural terms (see cabinet_belief.h).
        p.wall_precision          = cfg_.wall_precision;
        p.wall_reach_m            = cfg_.wall_reach_m;
        p.wall_parallel_precision = cfg_.wall_parallel_precision;
        p.room_axis_precision     = cfg_.room_axis_precision;
        p.room_axis_capture_rad   = cfg_.room_axis_capture_rad;
        p.extent_precision        = cfg_.extent_precision;
        p.free_space_precision    = cfg_.free_space_precision;   // VACATE: closes the one-sided extent
        p.base_tier = {cfg_.base_depth_m, cfg_.base_depth_std, cfg_.base_z0_m, cfg_.base_z0_std,
                       cfg_.base_z1_m,    cfg_.base_z1_std};
        p.wall_tier = {cfg_.wall_depth_m, cfg_.wall_depth_std, cfg_.wall_z0_m, cfg_.wall_z0_std,
                       cfg_.wall_z1_m,    cfg_.wall_z1_std};
        inst.ai2_belief = CabinetBelief(s0, p);
        inst.ai2_belief.set_room_interior(room_interior_);   // for the 180° C2v yaw fold

        // Seed the DISCRETE tier at its MAP given the birth cloud's vertical band. The seed z-band
        // (s0.z0,z1) already comes from the cloud (seed_from_points, 2%/98% z), so pick whichever tier
        // prior it lies closer to. A wall unit's carcass sits at z≈1.4–2.1; born as Base (the default),
        // its Base carcass prior (z0_mean=0) then drags z0 to the FLOOR faster than resolve_tier climbs
        // back — the "upper cabinet on the floor" symptom. This is a MAP over the discrete mode from the
        // evidence, exactly what resolve_tier does per-frame; here we just start it in the right mode.
        {
            const float d_base = std::abs(s0.z0 - p.base_tier.z0_mean) + std::abs(s0.z1 - p.base_tier.z1_mean);
            const float d_wall = std::abs(s0.z0 - p.wall_tier.z0_mean) + std::abs(s0.z1 - p.wall_tier.z1_mean);
            if (d_wall < d_base)
                inst.ai2_belief.set_tier(CabinetTier::Wall);
        }
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
    {
        rc::voxel_bank::ingest(inst, observation.candidate_pts, observation.residual_pts, cfg_);   // keep the viewer's voxel bank fed
        inst.last_residual_pts = observation.residual_pts;   // model-unexplained points for the viewer / residual-birth
    }

    const auto now = std::chrono::steady_clock::now();

    // Freeze-vs-age on stale: no fresh mask this cycle. Historically the belief just froze (Σ held —
    // information-filter axiom), so a dead mask/ZED feed read downstream as a confident-but-stale cabinet.
    // With AI2AgeNominalDtS>0 we instead AGE it: run the predict-only step on the AGENT's clock so Σ grows
    // by Q·(dt/dt_nominal) for as long as the stream is silent, and the propagated RT covariance reflects
    // how old the evidence is. No gate, no emergency flag — the generative-model form of a stale sensor.
    if (not observation.has_fresh_data)
    {
        if (cfg_.ai2_age_nominal_dt_s > 0.0f and inst.last_belief_touch.time_since_epoch().count() != 0)
        {
            const float dt = std::chrono::duration<float>(now - inst.last_belief_touch).count();
            inst.ai2_belief.inflate_for_age(dt, cfg_.ai2_age_nominal_dt_s);
        }
        inst.last_belief_touch = now;
        projection_->compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD the last free energy — no new mask ≠ FE 0 (the fit is unchanged)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting (motion-free). Even at zero camera motion, deprojection noise grows with
    // distance AND a far mask subtends a tiny angle, so pose — orientation most of all — becomes
    // unobservable: a 7 m view should confirm existence but never rotate a converged cabinet. The
    // motion×distance term is already in last_motion_var (the voxelizer interaction matrix carries 1/Z),
    // but that vanishes when still; this is the missing static part. Pure continuous covariance growth (no
    // gate): the per-frame information CAP (common-mode) rises with range, so the frame's yaw gain against a
    // converged prior shrinks smoothly toward zero. mask_range = mean camera→mask depth Z, from the producer.
    const float range         = std::max(0.0f, inst.last_range);
    const float lat_std       = cfg_.ai2_range_noise_lat_per_m * range;   // m   (lateral deprojection)
    const float yaw_std       = cfg_.ai2_range_noise_yaw_per_m * range;   // rad (orientation lever-arm ∝ 1/range)
    const float size_std      = cfg_.ai2_range_noise_size_per_m * range;  // m   (extent unresolved at range)
    const float range_lat_var = lat_std * lat_std;
    const float range_yaw_var = yaw_std * yaw_std;
    const float range_size_var = size_std * size_std;

    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m
                  + std::max(0.0f, inst.last_motion_var) + std::max(0.0f, inst.last_depth_var) + range_lat_var;

    // Truncation gate: a mask clipped by the image border has a chopped silhouette → it biases the fit
    // (shrinks/displaces the model). Above tolerance, skip the geometric update (predict only) — but
    // keep the instance (association ran upstream). Ego-motion corruption is handled by CONTINUOUS covariance
    // (moment_extra_var motion term + the mode evidence_weight), not a gate.
    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac;

    // Pose-chain covariance at the cabinet centre (cx,cy) — the per-frame SHARED localization error. Fed
    // into the belief as part of the common-mode so the frame's information saturates (calibrated σ).
    // Computed once here (before the update) and reused for the published RT cov below.
    compute_chain_cov(inst);

    // ── Rogue-mask yaw instrumentation (NO effect on the fit) ──────────────────────────────────────
    // Snapshot yaw at cycle entry and compute OBLIQUITY — the grazing-view covariate the truncation gate does
    // not measure. A near-horizontal camera→tabletop ray foreshortens the top face, so its 2D footprint (hence
    // yaw/extent) is biased even with a full in-frame mask. |cos(incidence)| = |ray·ẑ|/|ray|: 1 top-down, →0
    // grazing. Completeness + the per-channel yaw split are filled after the update. See the [yaw-jump] log.
    inst.dbg_yaw_pre = inst.ai2_belief.state().yaw;
    inst.dbg_dyaw_points = 0.0f; inst.dbg_dyaw_moment = 0.0f; inst.dbg_dyaw_flip = 0.0f; inst.dbg_completeness = 1.0f;
    Eigen::Vector3f cam_origin_room = Eigen::Vector3f::Zero();   // for the anisotropic per-point R (Stage 1)
    Eigen::Matrix4f zed_T_room = Eigen::Matrix4f::Identity();    // room→zed: transform points to the CAMERA frame
    bool have_cam = false;
    if (const auto rTz = projection_->room_T_zed_matrix(inst.last_mask_timestamp_ms); rTz)
    {
        const Eigen::Vector3d cam = rTz->block<3, 1>(0, 3);   // camera origin in the room frame
        cam_origin_room = cam.cast<float>(); have_cam = true;
        zed_T_room = rTz->inverse().cast<float>();            // for the camera-frame azimuth (tilt-state identifiability)
        const auto& s0b = inst.ai2_belief.state();
        const Eigen::Vector3d ray(s0b.cx - cam.x(), s0b.cy - cam.y(), s0b.z1 - cam.z());   // camera→run top
        const double rn = ray.norm();
        inst.dbg_obliquity_cos = (rn > 1e-6) ? static_cast<float>(std::abs(ray.z()) / rn) : 1.0f;
    }

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated / rejected cycle took no measurement)
    if (gated)
        inst.ai2_belief.predict();   // Σ inflates, mean unchanged
    else
    {
        CabinetFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        if (have_cam)
            frame.cam_origin = cam_origin_room;
        // The room wall for the flush factor — the term that makes the never-observed back face (and
        // hence the depth) identifiable. PERSISTENT commitment: choose the wall ONCE (a cabinet run does
        // not migrate between walls) and reuse it every frame. The old per-frame nearest-wall argmin
        // flip-flopped between two walls as a tilted/overgrown run's back face drifted near a corner —
        // positive feedback that produced the oblique-drift + impossible-depth failures. A free-standing
        // run still commits to its nearest wall, but its flush weight stays ~0 (gap ≫ reach), so the
        // choice is inert. The back centre remains the flush reference within the committed wall.
        const Eigen::Vector2f back_c = inst.ai2_belief.state().back_centre();
        // Commit only AFTER the fit has settled: during the noisy birth phase yaw/d (hence back_centre)
        // are not yet converged, so an early commit could lock onto the wrong wall. Until then use the
        // per-frame nearest wall (the drift-feedback it can cause needs many frames and is cut off once
        // we commit). `frames_converged` counts consecutive small-Δ frames.
        constexpr int kWallCommitFrames = 5;
        if (inst.committed_wall_seg_id < 0 and inst.frames_converged >= kWallCommitFrames)
            inst.committed_wall_seg_id = nearest_wall(back_c).seg_id;
        frame.wall = (inst.committed_wall_seg_id >= 0) ? wall_ref_by_seg_id(inst.committed_wall_seg_id, back_c)
                                                       : nearest_wall(back_c);
        if (not frame.wall.ok)                        // committed wall gone (polygon changed) → re-commit
        {
            frame.wall = nearest_wall(back_c);
            inst.committed_wall_seg_id = frame.wall.seg_id;
        }
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var;   // range adds to the SHARED position error (cap)
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var;
        // EGO-MOTION → common-mode ("still to update"): the shared ego-DISPLACEMENT this capture, (gain·|v|·dt)²
        // on position/size and (gain·|ω|·dt)² on yaw. A moving frame thus loses its authority to move the mean
        // (Woodbury) and only confirms existence; a near-still frame updates fully. Continuous, no motion gate.
        {
            const float g  = cfg_.ai2_common_mode_motion_gain;
            const float dt = std::max(0.0f, inst.last_ego_dt);
            // POSITION uses the DEPTH-AMPLIFIED shared corruption motion_dotd = Z·‖ṡ‖ (m/s), NOT the raw
            // camera |v|: at ~3 m range a small ego-motion moves the DEPROJECTED mask a lot, so |v|·dt (mm)
            // is far too small to bite at slow driving — motion_dotd·dt is the real shared position error.
            const float dp = g * inst.last_motion_dotd * dt;   // shared position displacement (m)
            const float da = g * inst.last_ego_w       * dt;   // shared yaw displacement (rad)
            frame.ego_motion_pos_var = dp * dp;
            frame.ego_motion_yaw_var = da * da;
        }
        // Obliquity yaw cap: at an edge-on (grazing) view the tabletop cloud is ~1-D along the near edge, so yaw
        // is barely observable and the per-point GN snaps between the box's symmetric orientations (r_π / w↔h —
        // the CSV flips). Grow the SHARED yaw variance as the view grazes (|cos(incidence)|→0 ⇒ 1/cos→∞), so a
        // grazing frame confirms the cabinet but cannot rotate it — the same continuous-covariance form as the range
        // term, keyed on view angle instead of distance. Validated live 2026-07-11 → always on; kObliquityYawGain
        // is a candidate for derivation from the deprojection Jacobian (replace the tuned coefficient with physics).
        constexpr float kObliquityYawGain = 0.05f;   // ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing
        const float oblq_cos          = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);
        const float obliquity_yaw_std = kObliquityYawGain * (1.0f / oblq_cos - 1.0f);   // 0 at top-down
        // FootprintResidual on ⇒ the depth tilt is an ESTIMATED N=7 state (no per-frame yaw cap — that ratcheted
        // and needed a fragile sweet-spot); chain_cov_yaw carries only the WHITE range term. Else the legacy tuned
        // obliquity+range form. (The tilt STATE replaces both the cap and the obliquity/range yaw gains.)
        frame.chain_cov_yaw = cfg_.footprint_residual
            ? range_yaw_var
            : range_yaw_var + obliquity_yaw_std * obliquity_yaw_std;   // range + grazing-view cap
        frame.chain_cov_size = range_size_var;                     // ...nor RESHAPE/inflate — geometry freezes afar
        // Footprint-moment SHARED per-frame variance: ego-motion mask corruption + a GENTLE range term (a global
        // footprint fit is far more range-robust than a single boundary point, so a coefficient << the per-point
        // AI2RangeNoiseSizePerM). This makes the moment ACCUMULATE across frames and back off when the robot is
        // moving, instead of snapping the belief to each frame's noisy/corrupted footprint (the live wander).
        // The ego-motion term uses motion_dotd (camera→mask relative motion): a "going-away/rotation" frame yields
        // a degraded/split mask, and this inflates the moment variance so it CANNOT reshape the established fit
        // (the observed reshape/rotate came in exactly on those frames). motion_var (interaction-matrix) is often
        // under-reported for split masks, so motion_dotd is the load-bearing signal here.
        const float moment_range_std  = cfg_.footprint_moment_range_per_m * range;
        const float moment_motion_std = cfg_.footprint_moment_motion_gain * std::abs(inst.last_motion_dotd);
        // Obliquity → moment variance: a grazing/foreshortened view biases the 2D inertia tensor (the SAME cause
        // that spoils the per-point yaw, but on the moment channel — where the CSV rogue rotations actually came
        // in). Mirror the per-point obliquity cap onto moment_extra_var so an edge-on frame confirms the cabinet but
        // cannot rotate/reshape it via the moment. Reuses oblq_cos computed above for chain_cov_yaw. 0 = OFF.
        (void) moment_range_std; (void) moment_motion_std;   // no moment channel on a run (single box)
        // YOLO-independent LiDAR range channel: stage returns landing on the legs/rim. No-op if precision==0
        // or no fresh sweep. The shared factor (accumulate_lidar_rays<6> in CabinetBelief::accumulate_extra)
        // sphere-traces this belief's own SDF, so the same call the bottle uses drops in unchanged.
        lidar_channel_.feed(inst, frame);

        // Stage 1 (PRECISION_AS_INFORMATION.md): replace the scalar per-point R with the anisotropic deprojection
        // noise projected on the SDF normal. Ego-motion variance is preserved as an isotropic floor (Stage 2 will
        // subsume it into the nuisance Jacobian). No-op unless the flag is on and we know the camera origin — then
        // a grazing view's yaw-carrying points get huge R → the per-point GN cannot rotate a converged cabinet,
        // WITHOUT the obliquity/range yaw gains above (which this is designed to make redundant).
        // (the footprint residual runs inside ai2_belief.update via accumulate_footprint; its tilt→yaw common-mode
        //  was folded into frame.chain_cov_yaw above so the engine Woodbury caps the total yaw information)

        // Divergence safety net (mirrors bottle): snapshot state+Σ, run the update, and if the centre teleports
        // beyond a physical bound in one frame (corrupted mask cloud / one-sided LiDAR runaway → the cx=−200m
        // event) REJECT it — restore the snapshot, widen Σ via a predict so the next good frame re-associates,
        // and accrue frames_diverged. A non-finite state is treated the same. 0 disables. NOT a magic gate: a
        // static cabinet cannot physically move max_step_m in one frame, so such a step is definitionally spurious.
        const CabinetBelief pre_belief = inst.ai2_belief;   // value copy (state + Σ + prior + flip_evidence)
        const auto&       ps         = pre_belief.state();
        energy = inst.ai2_belief.update(frame);
        const auto& ns = inst.ai2_belief.state();
        // Full-state jump: centre (cx,cy,H) AND EXTENT (w,h). A static cabinet can't grow/shrink its extent by
        // max_step_m in one frame any more than it can teleport — a close-range OVER-SEGMENTED mask (floor/wall
        // points) + the grow-only coverage term used to inflate w→5 m in a couple of frames, unguarded because
        // the old step omitted w,h. Now that blow-up is a definitionally-spurious frame → rejected.
        const float step = std::sqrt((ns.cx - ps.cx) * (ns.cx - ps.cx) + (ns.cy - ps.cy) * (ns.cy - ps.cy)
                                   + (ns.z0 - ps.z0) * (ns.z0 - ps.z0) + (ns.z1 - ps.z1) * (ns.z1 - ps.z1)
                                   + (ns.L  - ps.L)  * (ns.L  - ps.L)  + (ns.d  - ps.d)  * (ns.d  - ps.d));
        const bool bad = not (std::isfinite(ns.cx) and std::isfinite(ns.cy) and std::isfinite(ns.z0)
                              and std::isfinite(ns.z1) and std::isfinite(ns.L) and std::isfinite(ns.d)
                              and std::isfinite(ns.yaw));
        if (cfg_.max_step_m > 0.0f and (bad or step > cfg_.max_step_m))
        {
            std::print("[{}] AI2 step-bound REJECT: state moved {:.2f}m (>{:.2f}){} — outlier frame dropped (L={:.2f} d={:.2f})\n",
                       inst.node_name, step, cfg_.max_step_m, bad ? " [non-finite]" : "", ns.L, ns.d);
            inst.ai2_belief = pre_belief;     // reject the corrupted update (restore state + Σ)
            inst.ai2_belief.predict();        // widen Σ so the next good frame re-associates
            energy = inst.dbg_energy;         // HOLD last FE — a rejected outlier frame took no valid measurement
            ++inst.frames_diverged;
        }
        else
        {
            inst.frames_diverged = 0;
            // Log the TRUE free energy (mean −log mixture likelihood, clutter included) at the converged state —
            // the engine's return zeroed misfit via clutter (read ≈0 even on a bad fit). This is a meaningful F.
            energy = inst.ai2_belief.mean_energy(frame.points, inst.ai2_belief.state(), R);
            // FE SURPRISE (attention trigger): baseline tracks DOWN fast (consolidate a better fit) / UP slow (a
            // sustained rise = the cabinet moved shows as surprise before the baseline accepts it); surprise = the
            // smoothed positive gap FE−baseline. Updated only on a genuine measurement (this accepted branch).
            if (inst.fe_baseline < 0.0f)
                inst.fe_baseline = energy;
            else
            {
                const float a = (energy < inst.fe_baseline) ? cfg_.fe_baseline_adapt_down
                                                            : cfg_.fe_baseline_adapt_up;
                inst.fe_baseline += a * (energy - inst.fe_baseline);
            }
            const float gap = std::max(0.0f, energy - inst.fe_baseline);
            inst.fe_surprise += cfg_.fe_surprise_smooth * (gap - inst.fe_surprise);
        }
        // Near-square yaw disambiguation: sequential Bayesian comparison of the two orientation modes
        // (current vs the w↔h swap ≡ 90° rotation). Owns the GENUINE mode flip so the per-frame MAP no
        // longer SNAPS 90° on extent noise; the reported yaw uncertainty (yaw_marginal_var) stays honest
        // until an orbit resolves it. See CABINET.md.
        // Ego-motion reliability: a moving frame's mask must barely vote on the discrete w↔h mode (split/degraded
        // masks during motion were flipping it). Continuous down-weight, not a gate. Static (dotd≈0) → weight 1.
        const float mref = std::max(1e-3f, cfg_.orientation_motion_ref);
        const float dotd = std::abs(inst.last_motion_dotd);
        const float mode_evidence_weight = 1.0f / (1.0f + (dotd / mref) * (dotd / mref));
        // Discrete TIER mode {base unit, wall unit}: a Laplace filter cannot hold the genuinely bimodal
        // prior on (z0,z1), so the tier rides outside the Gaussian as a sequential model comparison —
        // the same construction the table uses for its w↔h mode. The motion down-weight applies for
        // the same reason it does there: a corrupted/split mask must not get a vote on a discrete latent.
        if (inst.ai2_belief.resolve_tier(frame.points, R, mode_evidence_weight) and should_log(inst))
            std::print("[{}] TIER switch → {} | tier_ev={:.3f} p_alt={:.2f}\n",
                       inst.node_name,
                       inst.ai2_belief.tier() == CabinetTier::Base ? "base" : "wall",
                       inst.ai2_belief.tier_evidence(), inst.ai2_belief.tier_posterior());

        // ── Per-cycle yaw / geometry instrumentation ──────────────────────────────────────────────
        // A run has ONE geometric channel (the per-point GN pass) plus the wall factor, so there is no
        // three-way channel attribution to make (the table splits per-point / moment / 90°-flip). What
        // matters here instead is whether the WALL is the thing moving the run — log the flush gap and
        // the applied precision alongside the yaw move. LOG-ONLY: no effect on the fit.
        {
            const auto  wrap    = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
            const auto& st      = inst.ai2_belief.state();
            const float d_total = wrap(st.yaw - inst.dbg_yaw_pre);
            inst.dbg_dyaw_points = d_total;
            constexpr float kYawJumpLogRad = 0.087f;   // ≈5°: diagnostic PRINT trigger only
            constexpr float kRad2Deg       = 57.29578f;
            if (std::abs(d_total) > kYawJumpLogRad)
                std::print("[yaw-jump] {} dpsi={:.1f}deg | L={:.2f} d={:.2f} wall_gap={:.3f} wall_lam={:.0f} "
                           "span={:.2f}/{} rays={} | trunc={:.2f} mvar={:.4f} dotd={:.2f} range={:.2f} npts={}\n",
                           inst.node_name, d_total * kRad2Deg, st.L, st.d,
                           inst.ai2_belief.last_wall_gap(), inst.ai2_belief.last_wall_lambda(),
                           inst.ai2_belief.last_span_obs(), inst.ai2_belief.last_span_pts(),
                           inst.ai2_belief.last_lidar_rays(),
                           inst.last_trunc_frac, inst.last_motion_var, inst.last_motion_dotd, inst.last_range, npts);
        }
    }

    // Write the belief back into CabinetState so all downstream publish/viewer/RT code is unchanged.
    // For a run the two parameterisations are identical (no derived legs to reconstruct).
    const auto& bs = inst.ai2_belief.state();
    CabinetState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.yaw = bs.yaw;
    ms.L  = bs.L;  ms.d  = bs.d;  ms.z0 = bs.z0; ms.z1 = bs.z1;
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;

    projection_->compute_projected_roi(inst);
    // (chain cov already computed above, before the belief update)

    // Object-anchor observation z_o: settled fit expressed in the localizer's base frame (gated OFF).
    compute_object_observation(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} dotd={:.2f} trunc={:.2f}{} tier={} | FE={:.2f} base={:.2f} surprise={:.2f} | cx={:.3f} cy={:.3f} ψ={:.3f} L={:.3f} d={:.3f} z=[{:.3f},{:.3f}] | σ(L,d,z1)mm=({:.0f},{:.0f},{:.0f}) | wall gap={:.3f} λ={:.0f} axis={:.2f}° span={:.2f}/{} | lidar {}/{} bp{} resid={:.3f}m vac={} div={}\n",
                   inst.node_name, npts, R, inst.last_motion_dotd, inst.last_trunc_frac, gated ? " GATED" : "",
                   inst.ai2_belief.tier() == CabinetTier::Base ? "base" : "wall",
                   energy, inst.fe_baseline, inst.fe_surprise,
                   bs.cx, bs.cy, bs.yaw, bs.L, bs.d, bs.z0, bs.z1,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(6, 6))),
                   inst.ai2_belief.last_wall_gap(), inst.ai2_belief.last_wall_lambda(),
                   57.2958f * inst.ai2_belief.last_axis_resid(),
                   inst.ai2_belief.last_span_obs(), inst.ai2_belief.last_span_pts(),
                   inst.dbg_lidar_rays, inst.dbg_lidar_raw, inst.dbg_lidar_bpearl_rays, inst.dbg_lidar_resid_m,
                   inst.ai2_belief.last_vacate_beams(),
                   inst.frames_diverged);

    // Wall-segment domain diagnostic: is the corner clamp BINDING? seg=1 ⇒ live; corners t=[tlo,thi] are the
    // wall endpoints projected onto the run axis; shi = observed +u span end (grows PAST thi ⇒ should censor);
    // qhi = +u retract residual (box end past the corner ⇒ should pull back). slices = masks fed to THIS run.
    if (should_log(inst))
        std::print("[{}] SEG seg={} t=[{:.2f},{:.2f}] shi={:.2f} qhi={:.2f} | slices_fed={}\n",
                   inst.node_name, inst.ai2_belief.last_seg_active(),
                   inst.ai2_belief.last_seg_tlo(), inst.ai2_belief.last_seg_thi(),
                   inst.ai2_belief.last_seg_shi(), inst.ai2_belief.last_seg_qhi(),
                   static_cast<int>(inst.assigned_mask_idxs.size()));

    // EvidenceMonitor snapshot: persist this frame's fit evidence (else it dies with the local observation).
    inst.dbg_cand_pts  = static_cast<int>(observation.candidate_pts.size());
    inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());
    inst.dbg_energy    = energy;
    inst.dbg_R         = R;
    inst.dbg_gated     = gated;

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

// Append one AI2 belief row (state + Σ-diag std + mask R/bias/trunc + mode evidence + LiDAR diag) to the CSV.
void CabinetFitter::log_ai2_csv(const CabinetInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        // Header MUST match the body row below field-for-field. It had drifted: the state block still
        // named the old 6-DOF layout (cx,cy,H,w,h,yaw) while the body writes the 7-DOF state
        // (cx,cy,yaw,L,d,z0,z1), and several stale moment_/completeness columns were never emitted — so
        // every column past cy parsed under the wrong name. Rewritten to the current body exactly.
        ai2_csv_ << "cycle,node,pkt_fid,pkt_ts,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,motion_dotd,ego_v,ego_w,ego_dt,trunc_frac,range,"
                 << "cx,cy,yaw,L,d,z0,z1,std_cx,std_cy,std_yaw,std_L,std_d,std_z0,std_z1,"
                 << "tier_ev,p_alt,lidar_rays,lidar_raw,lidar_bpearl,lidar_resid_m,lidar_meanz,lidar_topz,lidar_floorz,lidar_cov_ang,"
                 << "dyaw_points,obliquity_cos,wall_gap,wall_lambda,span_obs,span_pts,span_lidar_rays,"
                 << "ex_L,ex_p,ex_locc,ex_lfree,ex_lfree_eff,ex_ln,ex_socc,ex_sfree,ex_sfree_eff,ex_sndet,ex_streak,"
                 << "ex_pdetect,ex_central,ex_verify,ex_wantsverify,"
                 << "axis_resid,cand_pts,resid_pts\n";   // + Manhattan-yaw residual (rad) & candidate/residual split (merge diag)
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };   // posterior std (m / rad)
    // std_yaw is now the MARGINAL (mode-entropy-inflated) yaw std; std_yaw_within is the within-mode Σ(5,5).
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ','
             << inst.dbg_pkt_frame_id << ',' << inst.dbg_pkt_ts_ms << ','
             << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ','
             << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_motion_dotd << ','
             << inst.last_ego_v << ',' << inst.last_ego_w << ',' << inst.last_ego_dt << ','
             << inst.last_trunc_frac << ',' << inst.last_range << ','
             << s.cx << ',' << s.cy << ',' << s.yaw << ',' << s.L << ',' << s.d << ',' << s.z0 << ',' << s.z1 << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ','
             << sd(5) << ',' << sd(6) << ','
             << inst.ai2_belief.tier_evidence() << ',' << inst.ai2_belief.tier_posterior() << ','
             << inst.dbg_lidar_rays << ',' << inst.dbg_lidar_raw << ',' << inst.dbg_lidar_bpearl_rays << ',' << inst.dbg_lidar_resid_m << ','
             << inst.dbg_lidar_meanz_m << ',' << inst.dbg_lidar_topz_m << ',' << inst.dbg_lidar_floorz_m << ','
             << inst.dbg_lidar_cov_ang << ','
             << inst.dbg_dyaw_points << ',' << inst.dbg_obliquity_cos << ','
             << inst.ai2_belief.last_wall_gap()  << ',' << inst.ai2_belief.last_wall_lambda() << ','
             << inst.ai2_belief.last_span_obs()  << ',' << inst.ai2_belief.last_span_pts()    << ','
             << inst.ai2_belief.last_lidar_rays() << ','
             << inst.existence.logodds() << ',' << inst.existence.p_exists() << ','
             << inst.dbg_ex_lidar_occ << ',' << inst.dbg_ex_lidar_free << ',' << inst.dbg_ex_lidar_free_eff << ',' << inst.dbg_ex_lidar_n << ','
             << inst.dbg_ex_sil_occ << ',' << inst.dbg_ex_sil_free << ',' << inst.dbg_ex_sil_free_eff << ',' << inst.dbg_ex_sil_ndet << ','
             << inst.existence_remove_streak << ','
             << inst.dbg_ex_pdetect << ',' << inst.dbg_ex_central << ','
             << inst.verify_surprise << ',' << (inst.wants_verification ? 1 : 0) << ','
             << inst.ai2_belief.last_axis_resid() << ',' << inst.dbg_cand_pts << ',' << inst.dbg_resid_pts << '\n';   // + axis residual (rad) & candidate/residual split (merge diag)
    ai2_csv_.flush();
}

// ─── Factory helpers ──────────────────────────────────────────────────────────────────────────────

// Build the CabinetModel params from config (the SDF split band for candidate/residual classification).
CabinetModelParams CabinetFitter::make_model_params() const
{
    CabinetModelParams p;
    p.sigma_obs = cfg_.sigma_obs;   // top/leg SDF split band for the candidate/residual classification
    return p;
}

}  // namespace rc
