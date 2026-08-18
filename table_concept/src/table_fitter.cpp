/*
 * table_fitter.cpp — the active-inference fit core for table_concept (see table_fitter.h).
 *
 * Implements the per-"table_*" instance lifecycle and the AI2 full-covariance belief update: ensure_instance
 * (birth-seed + RT/prior warm-start + NaN sanitize), observe_slice (mask-cloud → candidate/residual SDF split
 * + per-slice R inputs), and run_inference (lazy footprint-moment birth, support-bank ingest, one TableBelief
 * update with range/motion covariance, the step-bound divergence net, FE-surprise attention, orientation-mode
 * resolution, and write-back into the legacy TableState). Collaborates with MaskIngestor, TableSceneGraph,
 * TableProjection, TableLidarRangeChannel, and the header-only support bank; SpecificWorker owns orchestration.
 */

#include "table_fitter.h"

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "table_support_bank.h"
#include "round_table_belief.h"   // round hypothesis for the shape model-selection (evaluate_shape)
#include "../../common/object_anchor/object_anchor_contract.h"
#include "../../common/object_anchor/ray_anisotropic_cov.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <print>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rc {

// ─── Construction & chain covariance ──────────────────────────────────────────────────────────────

TableFitter::TableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         TableConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         TableSceneGraph* scene_graph)
    : G_(graph), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph),
      projection_(std::make_unique<TableProjection>(graph, inner_eigen, mask_ingestor))
{
    TableBeliefState::use_quotient = cfg.quotient_chart;   // C2v symmetry-quotient optimisation chart (global mode)
    projection_->set_central_region_frac(cfg.central_region_frac);
}

// Enable Part-B chain-covariance propagation from source_frame (no-op unless a gaussian API + frame are given).
void TableFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void TableFitter::set_object_observation(bool enabled, std::string robot_frame)
{
    obs_robot_frame_ = std::move(robot_frame);
    obs_enabled_     = enabled and (inner_eigen_ != nullptr) and not obs_robot_frame_.empty();
}

// Build the object-anchor observation z_o for room_concept's landmark factor.
//
// z_o MUST be independent of the robot pose the localizer is estimating, or the factor just
// re-anchors to the last pose (the residual is ~0 at the current estimate). So we take the table's
// RAW camera-frame mask centroid (this frame's ZED measurement, no localization in it) and carry it
// to the robot base by the STATIC body←zed extrinsic (ts=0, a fixed calibrated mount). Position-only:
// a single view's yaw is biased (the grazing/obliquity problem), so we publish [x,y] and let the
// consumer treat it as a 2-DOF landmark. Gated OFF by default.
void TableFitter::compute_object_observation(TableInstance& inst)
{
    inst.obs_robot_valid = false;
    if (not obs_enabled_ or not inner_eigen_ or not mask_ingestor_)
        return;
    const auto& packet = mask_ingestor_->packet();
    if (packet.support_points_cam.empty())
        return;

    // This frame's ZED slice assigned to this table (the pinhole camera cloud; skip ricoh depth_var>0).
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
    // box_yaw in body = belief yaw (room) + yaw(body←room). See TABLE_TRIANGULATION.md.
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
            cen_xy, cam_xy, {0.5f * bs.w, 0.5f * bs.h}, box_yaw_body,
            std::max(0.1f, inst.last_range), std::clamp(inst.dbg_obliquity_cos, 0.f, 1.f),
            rc::object_anchor::RayCovParams{});
    }
    inst.obs_robot_valid = true;
}

// Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the table centre; store it on the instance.
//
// Transform the centre to the measurement frame and back to room with ZERO input cov, so InnerGaussianAPI
// returns exactly the chain contribution (Σ_chain from each RT edge's rt_covariance), pinned to the mask
// capture stamp. The table is fit in room but its position stays conditional on the robot pose
// (camera→robot→room), so this per-frame SHARED localization error feeds the belief common-mode.
void TableFitter::compute_chain_cov(TableInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the table centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution
    // (Σ_chain from each RT edge's rt_covariance), pinned to the mask capture stamp. The table is fit in
    // room but its position is still conditional on the robot pose (camera→robot→room), so this applies.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, 0.0);   // table node origin = base on the floor (z=0)
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

// Create the instance for a "table_*" node if absent; return true only the first time it is created.
//
// Warm-starts from the node's size attribs and the room→table RT edge, then OVERRIDES with the tracker
// birth seed when present (the RT edge written at birth is not reliably queryable this same cycle — it reads
// 0,0 — which would freeze the model at the origin and cause endless re-births). Sanitizes any non-finite
// field before it can poison the SDF and lock the optimizer.
bool TableFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    TableState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w            = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.h            = v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.table_height = v.value();

    // Read RT pose from room→table edge
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

    // Tracker birth seed: authoritative for a freshly-born instance. The room→table RT edge written at
    // birth is not reliably queryable this same cycle (it reads as 0,0), and the warm-start would then
    // freeze the model at the origin forever → the tracker never associates and re-births endlessly.
    if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        std::print("[{}] birth-seed applied → cx={:.2f} cy={:.2f}\n", node.name(), init_state.cx, init_state.cy);
        birth_seeds_.erase(it);
    }
    else
        std::print("[{}] NO birth-seed (id={}) → init cx={:.2f} cy={:.2f}\n",
                   node.name(), node.id(), init_state.cx, init_state.cy);

    init_state.leg_length = std::max(0.05f, init_state.table_height - TableModel::TOP_THICKNESS);

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF and lock the
    // optimizer; replace any non-finite field with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("table_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.w, 1.0f, "w");
        fix(init_state.h, 0.6f, "h");
        fix(init_state.table_height, 0.75f, "table_height");
        fix(init_state.leg_inset, TableModel::LEG_RADIUS, "leg_inset");  // FROZEN at outer edge
        init_state.leg_length = std::max(0.05f, init_state.table_height - TableModel::TOP_THICKNESS);
    }

    TableInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();

    inst.model = TableModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name(), "table");

    instances_.emplace(node.id(), std::move(inst));
    std::print("table_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ──────────────────────────────────────────────────────────────────────────────────

// Build an observation from ONE assigned mask slice: latch that slice's R inputs, then SDF-split its cloud.
//
// Classify-don't-destroy split into candidate (near the current surface) vs residual (off-surface) points.
// The caller (process_table_node) invokes this once per assigned slice and runs a belief update for each —
// sequential fusion that keeps every sensor's R and common-mode separate. RICOH slices (depth_var>0) are
// BEARING-ONLY and return empty (never fitted); they drive only the attention path.
TableFitter::TableObservation TableFitter::observe_slice(TableInstance& inst, int slice_index)
{
    TableObservation observation;
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
    // A slice assigned to this table ⇒ detection is alive; latch its per-slice R inputs.
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
    // Appearance (DISPLAY ONLY — deliberately NOT part of `observation`, so no fit can ever see it). This
    // is a sidecar belief on the instance whose only consumer is the mesh tint published to the retina.
    inst.appearance.update(slice.color_chroma, slice.color_var, slice.color_neff);

    const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
    const std::size_t end   = std::min(slice.support_end,   masks_packet.support_points.size());

    observation.candidate_pts.reserve(end > begin ? end - begin : 0);
    observation.residual_pts.reserve(end > begin ? end - begin : 0);
    for (std::size_t i = begin; i < end; ++i)
    {
        const auto& p = masks_packet.support_points[i];
        // ★A POINT ANOTHER OBJECT ALREADY EXPLAINS IS NOT EVIDENCE FOR THIS ONE (SHARED, common/exclusion).
        // Occam, stated locally: growing this object's extent to cover it buys no likelihood and costs
        // complexity, so the fit stops there of its own accord — no clamp, no arbitration, and an ABUTTING
        // neighbour pays nothing because it does not overlap. The point's z is part of the question: a claim
        // is a VOLUME, so a hood over a worktop shares its footprint without sharing its space.
        // Measured 2026-08-17: bottle_2 walked its radius to 4.51 m, converging on a flat disc at worktop
        // height — the worktop's own returns, admitted as evidence of bottle.
        if (foreign_claims_ and rc::exclusion::explained_by_other(p.x(), p.y(), p.z(), *foreign_claims_))
        {
            // NOT gated on a verbose flag: this is the one line that says whether the rule is running at
            // all, which is the first thing anyone asks. Throttled rather than silenced.
            if ((n_explained_away_++ % 200) == 0)
                std::print("[{}] [exclusion] point dropped — already explained by another object "
                           "({} so far this run)\n", inst.node_name, n_explained_away_);
            continue;
        }
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
TableFitter::TableObservation TableFitter::observe(TableInstance& inst, const DSR::Node& node)
{
    // No mask slice was assigned this cycle. There is no node-attrib sensing path (nothing writes
    // candidate/residual points onto the node), so this is always a stale observation: has_fresh_data=false
    // ⇒ run_inference ages the belief (predict-only) instead of fitting.
    (void)inst; (void)node;
    return TableObservation{};
}

// True on the config-driven log period (one in table_log_period_frames cycles).
bool TableFitter::should_log(const TableInstance& inst) const
{
    const int period = std::max(1, cfg_.table_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// May this mask frame move geometry? — see table_fitter.h. The FIXATION gate's conditions, evaluated on a raw
// slice (no instance yet): RESOLVABLE (point mass + truncation), CENTRED (the fovea) and STILL (ego-motion mask
// smear). Each condition is disabled by the same config key that disables it in the fit, so the two predicates
// can never drift apart. Unlike the fit's version there is no roi_valid escape — a birth candidate always has a
// mask centroid, so centring is always judgeable.
bool TableFitter::frame_admissible(const MaskIngestor::MaskSlice& sl) const
{
    if (sl.trunc_frac > cfg_.ai2_trunc_gate_frac)
        return false;                                    // truncated ⇒ extent is a lower bound ⇒ never births
    if (not cfg_.fixation_enabled)
        return true;
    const int npts = static_cast<int>(sl.support_end - sl.support_begin);
    const bool resolvable = (cfg_.fixation_min_pts   <= 0    or npts >= cfg_.fixation_min_pts)
                       and (cfg_.fixation_max_trunc <= 0.0f or sl.trunc_frac <= cfg_.fixation_max_trunc)
                       and (cfg_.fixation_range_m   <= 0.0f or (sl.range > 0.0f and sl.range <= cfg_.fixation_range_m));
    const bool centred = cfg_.fixation_centre_frac <= 0.0f or sl.centroid_radius <= cfg_.fixation_centre_frac;
    const bool still   = cfg_.fixation_still_dotd  <= 0.0f or std::abs(sl.motion_dotd) <= cfg_.fixation_still_dotd;
    return resolvable and centred and still;
}

// ─── Inference ────────────────────────────────────────────────────────────────────────────────────

// One recursive full-covariance belief update (or age-only step) for this instance; returns the free energy.
//
// Lazy first-frame init (snap centre/height to the cloud, footprint-moment birth of w/h/yaw), support-bank
// ingest, then the range/motion covariance and the TableBelief update guarded by a step-bound divergence net,
// FE-surprise attention baseline, and orientation-mode resolution. On a stale frame it ages the belief
// (Σ grows on the agent's clock) instead of freezing. Result is written back into the legacy TableState.
float TableFitter::run_inference(TableInstance& inst, const TableObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy init: warm-start the belief from the model state, but on the FIRST frame snap the centre/height
    // to the observed cloud — a box far from the points would see them all as clutter (zero gradient) and
    // never converge (the AI2 analogue of the legacy cold-start snap).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        TableBeliefState s0{m.cx, m.cy, m.table_height, m.w, m.h, m.yaw};
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
            s0.H = zs[k];   // observed top face

            // Birth seed (w,h,yaw) from the footprint second-moment: start the box near the mask's real size and
            // orientation so the per-point mixture + legs associate immediately, instead of a small axis-aligned
            // box that sees the true rim/legs as clutter and never rotates. Same statistic as the per-frame factor;
            // gated on footprint_moment_precision so the baseline (flag OFF) births exactly as before. Skip on a
            // TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
            {
                std::vector<Eigen::Vector3f> pts;
                pts.reserve(static_cast<std::size_t>(npts));
                pts.insert(pts.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
                pts.insert(pts.end(), observation.residual_pts.begin(), observation.residual_pts.end());
                const float z_hi = s0.H + 3.0f * cfg_.ai2_sigma_base_m;
                const float z_lo = s0.H - TableModel::TOP_THICKNESS - 3.0f * cfg_.ai2_sigma_base_m;
                if (const auto mom = TableBelief::footprint_moment(pts, z_lo, z_hi); mom.ok)
                {
                    s0.cx = mom.cx; s0.cy = mom.cy;   // top-plane centroid (sharper than the leg-biased full-cloud mean)
                    s0.w = std::max(0.10f, mom.ext_major);
                    s0.h = std::max(0.10f, mom.ext_minor);
                    // Only commit the orientation when the footprint is clearly anisotropic. A near-square
                    // footprint has an ill-defined principal axis (phi swings ±90° on noise), so birthing yaw
                    // from a single frame would seed a random ±90°; leave yaw at its RT/default seed and let the
                    // per-frame moment (aniso-weighted) + resolve_orientation settle it once evidence accrues.
                    const float sum = mom.ext_major + mom.ext_minor;
                    if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)
                        s0.yaw = mom.phi;
                }
            }
        }
        TableBeliefParams p;
        p.sigma_base_m    = cfg_.ai2_sigma_base_m;
        p.clutter_frac    = cfg_.ai2_clutter_frac;
        p.clutter_scale_m = cfg_.ai2_clutter_scale_m;
        p.prior_size_std  = cfg_.ai2_prior_size_std;
        p.pixel_sigma_over_f     = cfg_.pixel_sigma_over_f;
        p.depth_sigma0_m         = cfg_.depth_sigma0_m;
        p.depth_sigma_range_coef = cfg_.depth_sigma_range_coef;
        p.model_sigma_m          = cfg_.model_sigma_m;
        p.footprint_residual     = cfg_.footprint_residual;   // a1′+a2′: 2-D footprint residual + shared depth-affine
        p.depth_bias_std         = cfg_.depth_bias_std;
        p.depth_scale_std        = cfg_.depth_scale_std;
        p.process_std_m   = cfg_.ai2_process_std_m;
        p.process_std_yaw = cfg_.ai2_process_std_yaw;
        p.process_std_extent_m   = cfg_.ai2_process_std_extent_m;    // 0 ⇒ a table's dimensions are constant
        p.clamp_sigma_to_prior   = cfg_.ai2_clamp_sigma_to_prior;    // ageing saturates at Σ₀, never diverges
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
        p.gn_iters        = cfg_.ai2_gn_iters;
        p.coverage_precision  = cfg_.coverage_precision;
        p.coverage_robust_c_m = cfg_.coverage_robust_c_m;
        p.free_space_precision = cfg_.free_space_precision;
        p.footprint_moment_precision = cfg_.footprint_moment_precision;
        p.footprint_moment_completeness_gain = cfg_.footprint_moment_completeness_gain;
        p.footprint_moment_min_completeness  = cfg_.footprint_moment_min_completeness;
        p.top_thickness   = TableModel::TOP_THICKNESS;
        p.leg_radius      = TableModel::LEG_RADIUS;
        inst.ai2_belief = TableBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
    {
        rc::support_bank::ingest(inst, observation.candidate_pts, observation.residual_pts, cfg_);
        // ★NOT for a viewer — nothing ever read the published attribute. The bank is how this agent
        // gathers 3D points from its own masks over many frames so evaluate_shape() can DISCRIMINATE
        // SHAPES (round vs square) from an accumulated cloud rather than one frame's partial view.
        inst.last_residual_pts = observation.residual_pts;   // model-unexplained points for the viewer layer
    }

    const auto now = std::chrono::steady_clock::now();

    // Freeze-vs-age on stale: no fresh mask this cycle. Historically the belief just froze (Σ held —
    // information-filter axiom), so a dead mask/ZED feed read downstream as a confident-but-stale table.
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
        inst.dbg_gate_fresh = false;   // no mask reached the fit ⇒ the gate flags below are STALE this cycle
        projection_->compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD the last free energy — no new mask ≠ FE 0 (the fit is unchanged)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting (motion-free). Even at zero camera motion, deprojection noise grows with
    // distance AND a far mask subtends a tiny angle, so pose — orientation most of all — becomes
    // unobservable: a 7 m view should confirm existence but never rotate a converged table. The
    // motion×distance term is already in last_motion_var (the retina interaction matrix carries 1/Z),
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

    // COVERAGE (completeness) common-mode — the view-quality half of "close enough to reshape". Range says how
    // far the table is; coverage says how much of it this viewpoint actually spans. Both must be good before a
    // frame may re-cut the geometry, and the live failure was a CLOSE view with poor coverage (see the block
    // comment on ai2_coverage_size_gain in table_config.h). σ = gain·(1/c − 1): identically 0 at c ≥ 1, so a
    // full view behaves exactly as before; unbounded as the footprint fragments, so a sliver confirms but
    // cannot reshape. c is last cycle's measured completeness (static table ⇒ one cycle stale is fine).
    const float cover      = std::clamp(inst.last_completeness, cfg_.ai2_coverage_min, 1.0f);
    const float cover_defect = 1.0f / cover - 1.0f;                        // 0 at full coverage
    const float cov_size_var = std::pow(cfg_.ai2_coverage_size_gain * cover_defect, 2.0f);   // m²  (w,h,H)
    const float cov_yaw_var  = std::pow(cfg_.ai2_coverage_yaw_gain  * cover_defect, 2.0f);   // rad² (yaw)
    const float cov_pos_var  = std::pow(cfg_.ai2_coverage_pos_gain  * cover_defect, 2.0f);   // m²  (cx,cy)

    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m
                  + std::max(0.0f, inst.last_motion_var) + std::max(0.0f, inst.last_depth_var) + range_lat_var;

    // ── Ego-motion → COMMON-MODE: the "be still to UPDATE, else CONFIRM" fixation term (like the VOR / an
    // animal taking in detail only during fixations). A moving frame's mask is smeared/displaced by ego-motion
    // by ≈ effective-lag · motion-speed — a per-mask SHARED error that per-point R averages away (N points
    // collapse it). Route it into the per-frame COMMON-MODE instead: the engine's Woodbury marginalisation
    // then caps the frame's authority to move the GEOMETRY MEAN (size/pose/yaw), so geometric updates
    // concentrate at stillness while a moving frame contributes CONFIRMATION only (existence/association don't
    // read this). motion_dotd = Z·‖ṡ‖ (m/s) from the retina. CONTINUOUS, no gate: at dotd→0 the term
    // vanishes (a still frame updates fully); the gains are ~effective-lag (s), 0 disables a channel.
    const float mot_dotd    = std::abs(inst.last_motion_dotd);
    const float mot_pos_var = std::pow(cfg_.motion_cm_pos_gain  * mot_dotd, 2.0f);   // m²  (cx,cy)
    const float mot_size_var= std::pow(cfg_.motion_cm_size_gain * mot_dotd, 2.0f);   // m²  (w,h,H)
    const float mot_yaw_var = std::pow(cfg_.motion_cm_yaw_gain  * mot_dotd, 2.0f);   // rad² (yaw)

    // Truncation gate: a mask clipped by the image border has a chopped silhouette → it biases the fit
    // (shrinks/displaces the model). Above tolerance, skip the geometric update (predict only) — but
    // keep the instance (association ran upstream). Ego-motion corruption is handled by CONTINUOUS covariance
    // (moment_extra_var motion term + the mode evidence_weight), not a gate.
    const bool trunc_gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac;

    // ── FIXATION gate (attention): only a CLOSE, CENTRED, STILL view may touch the geometry ───────────
    // See the block comment on TableConfig::fixation_enabled for why the graded common-mode terms could not
    // do this: the engine saturates a frame's information at Σc⁻¹, a NONZERO asymptote, so a bad frame is
    // attenuated but still moves the mean — and accumulation over hundreds of frames beats attenuation.
    // A fixation is judged from the PREDICTED projection (roi_* is last cycle's, recomputed below at the same
    // pre-update belief) + the retina's ego-motion smear. Each condition is independently disable-able.
    // roi_valid==false (no projection yet, e.g. a newborn's first cycle) cannot judge centring → do not let it
    // block; the range and stillness conditions still apply.
    const float roi_off      = std::hypot(inst.roi_offset_x, inst.roi_offset_y);
    // RESOLVABLE (was a hard RANGE cut until 2026-07-30): gate on the mask quality that actually determines
    // whether this frame can resolve the geometry — point mass and truncation — not on distance. Range is a
    // proxy that had the physics backwards: a dense far stare beats a starved near glance. Kept as a
    // diagnostic and as an optional extra condition (fixation_range_m > 0 re-enables the old cut).
    const bool  fix_close    = (cfg_.fixation_min_pts   <= 0    or npts >= cfg_.fixation_min_pts)
                           and (cfg_.fixation_max_trunc <= 0.0f or inst.last_trunc_frac <= cfg_.fixation_max_trunc)
                           and (cfg_.fixation_range_m   <= 0.0f or (range > 0.0f and range <= cfg_.fixation_range_m));
    const bool  fix_centred  = cfg_.fixation_centre_frac <= 0.0f or (not inst.roi_valid)
                                                                or roi_off <= cfg_.fixation_centre_frac;
    const bool  fix_still    = cfg_.fixation_still_dotd  <= 0.0f or mot_dotd <= cfg_.fixation_still_dotd;
    const bool  fixated      = fix_close and fix_centred and fix_still;
    inst.dbg_fix_close = fix_close; inst.dbg_fix_centred = fix_centred;
    inst.dbg_fix_still = fix_still; inst.dbg_fixated = fixated;
    inst.dbg_trunc_gated = trunc_gated;   // split by mechanism for the existence channel (see table_instance.h)
    inst.dbg_gate_fresh  = true;          // …and mark the verdict as computed from THIS frame's mask
    // Outside a fixation take the SAME path as a truncated mask: predict() only, mean HELD. Association and
    // existence do not read this, so the table is still confirmed and tracked — only its geometry is frozen.
    const bool gated = trunc_gated or (cfg_.fixation_enabled and not fixated);
    if (cfg_.fixation_enabled and not fixated and not trunc_gated and should_log(inst))
        std::print("[{}] [fixation] SUPPRESSED — resolvable={} (npts={}>={} trunc={:.2f}<={:.2f}) centred={} (off={:.2f}<={:.2f}) still={} (dotd={:.3f}<={:.3f}) | range={:.2f}\n",
                   inst.node_name, fix_close, npts, cfg_.fixation_min_pts, inst.last_trunc_frac,
                   cfg_.fixation_max_trunc, fix_centred, roi_off, cfg_.fixation_centre_frac,
                   fix_still, mot_dotd, cfg_.fixation_still_dotd, range);

    // Pose-chain covariance at the table centre (cx,cy) — the per-frame SHARED localization error. Fed
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
        const Eigen::Vector3d ray(s0b.cx - cam.x(), s0b.cy - cam.y(), s0b.H - cam.z());   // camera→tabletop centre
        const double rn = ray.norm();
        inst.dbg_obliquity_cos = (rn > 1e-6) ? static_cast<float>(std::abs(ray.z()) / rn) : 1.0f;
    }

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated / rejected cycle took no measurement)
    if (gated)
        inst.ai2_belief.predict();   // Σ inflates, mean unchanged
    else
    {
        TableFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        // Ray geometry (needed by the footprint residual + the anisotropic R later).
        if (cfg_.footprint_residual and have_cam)
        { frame.cam_origin = cam_origin_room; frame.has_rays = true; }
        // Camera-frame azimuth per point (about the optical axis; ZED frame x-right, y-depth) for the N=7 depth-
        // tilt STATE — computed from the TRUE extrinsic, NOT the belief's drifting table-centre, so a persistent
        // tilt is coherent across viewpoints (the identifiability the tilt estimate needs). See PRECISION_AS_INFO.
        if (cfg_.footprint_residual and have_cam)
        {
            frame.point_azim.reserve(frame.points.size());
            for (const auto& p : frame.points)
            {
                const Eigen::Vector4f pc = zed_T_room * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
                frame.point_azim.push_back(std::atan2(pc.x(), pc.y()));   // signed azimuth about the optical axis
            }
        }
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var + mot_pos_var + cov_pos_var;   // pose-chain + range + EGO-MOTION + COVERAGE
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var + mot_pos_var + cov_pos_var;
        // Obliquity yaw cap: at an edge-on (grazing) view the tabletop cloud is ~1-D along the near edge, so yaw
        // is barely observable and the per-point GN snaps between the box's symmetric orientations (r_π / w↔h —
        // the CSV flips). Grow the SHARED yaw variance as the view grazes (|cos(incidence)|→0 ⇒ 1/cos→∞), so a
        // grazing frame confirms the table but cannot rotate it — the same continuous-covariance form as the range
        // term, keyed on view angle instead of distance. Validated live 2026-07-11 → always on.
        //
        // THIS IS THE LIVE YAW CAP on the shipped (FootprintResidual == false) branch below — a tuned coefficient,
        // deliberately kept. Its intended derived replacement, TableBelief::tilt_yaw_common_mode() (a2′), is WIP and
        // NOT wired here: the fitter never calls it, and its self_test drift check is disabled. So on the
        // FootprintResidual == true branch there is currently NO obliquity/tilt yaw cap wired in (chain_cov_yaw
        // carries only the white range term) — that branch is experimental. Do not delete kObliquityYawGain until
        // tilt_yaw_common_mode is called here and validated. See table_belief.cpp::tilt_yaw_common_mode.
        constexpr float kObliquityYawGain = 0.05f;   // ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing
        const float oblq_cos          = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);
        const float obliquity_yaw_std = kObliquityYawGain * (1.0f / oblq_cos - 1.0f);   // 0 at top-down
        // FootprintResidual on ⇒ the depth tilt is an ESTIMATED N=7 state (no per-frame yaw cap — that ratcheted
        // and needed a fragile sweet-spot); chain_cov_yaw carries only the WHITE range term. Else the legacy tuned
        // obliquity+range form. (The tilt STATE replaces both the cap and the obliquity/range yaw gains.)
        frame.chain_cov_yaw = mot_yaw_var + cov_yaw_var + (cfg_.footprint_residual
            ? range_yaw_var
            : range_yaw_var + obliquity_yaw_std * obliquity_yaw_std);   // range + grazing cap + EGO-MOTION + COVERAGE
        // range freezes afar + EGO-MOTION freezes while moving + COVERAGE freezes on a partial view. The coverage
        // term is what stops a close-but-fragmentary look from re-cutting w/h (the live +0.76 m h jump at 0.75 m).
        frame.chain_cov_size = range_size_var + mot_size_var + cov_size_var;
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
        // Obliquity → moment YAW variance: a grazing/foreshortened view biases the 2D inertia tensor's principal
        // AXES (the SAME cause that spoils the per-point yaw, but on the moment channel — where the CSV rogue
        // rotations actually came in). Mirror the per-point obliquity cap so an edge-on frame confirms the table
        // but cannot ROTATE it via the moment. Reuses oblq_cos computed above for chain_cov_yaw. 0 = OFF.
        //
        // ★2026-08-06: this goes on the YAW row ONLY (moment_yaw_extra_var), not on w/h. A tabletop is viewed at
        // cos≈0.10-0.26 by a robot-mounted camera essentially ALWAYS — it is the normal condition, not an
        // exceptional one — so charging the extent rows for it put a permanent 0.43 m σ on the ONE channel that
        // measures the footprint scale, and the live table froze at its birth 0.47×0.61 m while its own moment
        // read 0.90×0.81 m (ai2_log.csv table_1: completeness 2.54, moment K decayed to 0.004). The extent needs
        // no obliquity guard: foreshortening can only UNDER-report an extent and the grow-only guard in
        // apply_footprint_moment already drops that direction.
        const float moment_oblq_std   = cfg_.obliquity_moment_gain * (1.0f / oblq_cos - 1.0f);
        frame.moment_extra_var = std::max(0.0f, inst.last_motion_var)
                               + moment_range_std * moment_range_std
                               + moment_motion_std * moment_motion_std;
        frame.moment_yaw_extra_var = moment_oblq_std * moment_oblq_std;
        // YOLO-independent LiDAR range channel: stage returns landing on the legs/rim. No-op if precision==0
        // or no fresh sweep. The shared factor (accumulate_lidar_rays<6> in TableBelief::accumulate_extra)
        // sphere-traces this belief's own SDF, so the same call the bottle uses drops in unchanged.
        lidar_channel_.feed(inst, frame);

        // Stage 1 (PRECISION_AS_INFORMATION.md): replace the scalar per-point R with the anisotropic deprojection
        // noise projected on the SDF normal. Ego-motion variance is preserved as an isotropic floor (Stage 2 will
        // subsume it into the nuisance Jacobian). No-op unless the flag is on and we know the camera origin — then
        // a grazing view's yaw-carrying points get huge R → the per-point GN cannot rotate a converged table,
        // WITHOUT the obliquity/range yaw gains above (which this is designed to make redundant).
        // (the footprint residual runs inside ai2_belief.update via accumulate_footprint; its tilt→yaw common-mode
        //  was folded into frame.chain_cov_yaw above so the engine Woodbury caps the total yaw information)

        // Divergence safety net (mirrors bottle): snapshot state+Σ, run the update, and if the centre teleports
        // beyond a physical bound in one frame (corrupted mask cloud / one-sided LiDAR runaway → the cx=−200m
        // event) REJECT it — restore the snapshot, widen Σ via a predict so the next good frame re-associates,
        // and accrue frames_diverged. A non-finite state is treated the same. 0 disables. NOT a magic gate: a
        // static table cannot physically move max_step_m in one frame, so such a step is definitionally spurious.
        const TableBelief pre_belief = inst.ai2_belief;   // value copy (state + Σ + prior + flip_evidence)
        const auto&       ps         = pre_belief.state();
        energy = inst.ai2_belief.update(frame);
        const auto& ns = inst.ai2_belief.state();
        // Full-state jump: centre (cx,cy,H) AND EXTENT (w,h). A static table can't grow/shrink its extent by
        // max_step_m in one frame any more than it can teleport — a close-range OVER-SEGMENTED mask (floor/wall
        // points) + the grow-only coverage term used to inflate w→5 m in a couple of frames, unguarded because
        // the old step omitted w,h. Now that blow-up is a definitionally-spurious frame → rejected.
        const float step = std::sqrt((ns.cx - ps.cx) * (ns.cx - ps.cx) + (ns.cy - ps.cy) * (ns.cy - ps.cy)
                                   + (ns.H  - ps.H)  * (ns.H  - ps.H)
                                   + (ns.w  - ps.w)  * (ns.w  - ps.w) + (ns.h - ps.h) * (ns.h - ps.h));
        const bool bad = not (std::isfinite(ns.cx) and std::isfinite(ns.cy) and std::isfinite(ns.H)
                              and std::isfinite(ns.w) and std::isfinite(ns.h) and std::isfinite(ns.yaw));
        if (cfg_.max_step_m > 0.0f and (bad or step > cfg_.max_step_m))
        {
            std::print("[{}] AI2 step-bound REJECT: state moved {:.2f}m (>{:.2f}){} — outlier frame dropped (w={:.2f} h={:.2f})\n",
                       inst.node_name, step, cfg_.max_step_m, bad ? " [non-finite]" : "", ns.w, ns.h);
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
            // sustained rise = the table moved shows as surprise before the baseline accepts it); surprise = the
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
        // until an orbit resolves it. See TABLE.md.
        // Ego-motion reliability: a moving frame's mask must barely vote on the discrete w↔h mode (split/degraded
        // masks during motion were flipping it). Continuous down-weight, not a gate. Static (dotd≈0) → weight 1.
        const float mref = std::max(1e-3f, cfg_.orientation_motion_ref);
        const float dotd = std::abs(inst.last_motion_dotd);
        const float mode_evidence_weight = 1.0f / (1.0f + (dotd / mref) * (dotd / mref));
        // The quotient chart makes the w↔h mode a continuous, single-valued coordinate — the discrete flip
        // accumulator is unrepresentable and unnecessary (skipped). Legacy path keeps resolve_orientation.
        if (not rc::TableBeliefState::use_quotient
            and inst.ai2_belief.resolve_orientation(frame.points, R, mode_evidence_weight) and should_log(inst))
            std::print("[{}] orientation mode FLIP (w↔h) | flip_ev={:.3f} p_alt={:.2f}\n",
                       inst.node_name, inst.ai2_belief.flip_evidence(), inst.ai2_belief.mode_posterior());

        // ── Rogue-mask yaw attribution (instrumentation) ──────────────────────────────────────────
        // Split this cycle's yaw move across the three channels (per-point GN / footprint-moment / 90° flip)
        // and, on a jump beyond kYawJumpLogRad (LOG-ONLY diagnostic trigger — NOT a control gate), dump the
        // covariate vector so we can see WHICH frame geometry drove it: obliquity + completeness (the gate's
        // blind spots) alongside anisotropy / motion / trunc / range. This is how we CATCH the rogue frames.
        {
            const auto  wrap    = [](float a) { return std::remainder(a, 2.0f * static_cast<float>(M_PI)); };
            const auto& st      = inst.ai2_belief.state();
            const float yaw_now = st.yaw;
            const float y_pts   = inst.ai2_belief.dbg_yaw_after_points();
            const float y_mom   = inst.ai2_belief.dbg_yaw_after_moment();
            inst.dbg_dyaw_points = wrap(y_pts - inst.dbg_yaw_pre);
            inst.dbg_dyaw_moment = wrap(y_mom - y_pts);
            inst.dbg_dyaw_flip   = wrap(yaw_now - y_mom);
            const float d_total  = wrap(yaw_now - inst.dbg_yaw_pre);
            const float em = inst.ai2_belief.dbg_moment_ext_major(), en = inst.ai2_belief.dbg_moment_ext_minor();
            inst.dbg_completeness = (st.w * st.h > 1e-4f and em > 0.0f) ? (em * en) / (st.w * st.h) : 1.0f;
            inst.last_completeness = inst.dbg_completeness;   // persist for NEXT cycle's coverage common-mode
            constexpr float kYawJumpLogRad = 0.087f;   // ≈5°: diagnostic PRINT trigger only (no fit effect)
            constexpr float kRad2Deg       = 57.29578f;
            if (std::abs(d_total) > kYawJumpLogRad)
                std::print("[yaw-jump] {} dpsi={:.1f}deg (pts={:.1f} mom={:.1f} flip={:.1f}) | oblq_cos={:.2f} compl={:.2f} "
                           "aniso={:.2f} r_yaw={:.3f} | trunc={:.2f} mvar={:.4f} dotd={:.2f} range={:.2f} npts={}\n",
                           inst.node_name, d_total * kRad2Deg, inst.dbg_dyaw_points * kRad2Deg,
                           inst.dbg_dyaw_moment * kRad2Deg, inst.dbg_dyaw_flip * kRad2Deg,
                           inst.dbg_obliquity_cos, inst.dbg_completeness,
                           inst.ai2_belief.dbg_moment_aniso(), inst.ai2_belief.dbg_moment_r_yaw(),
                           inst.last_trunc_frac, inst.last_motion_var, inst.last_motion_dotd, inst.last_range, npts);
        }
    }

    // Write the belief back into the legacy TableState so all downstream publish/viewer/RT code is
    // unchanged. Legs are derived: leg_length = H − TOP_THICKNESS, inset frozen at the outer edge.
    const auto& bs = inst.ai2_belief.state();
    TableState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.table_height = bs.H; ms.w = bs.w; ms.h = bs.h; ms.yaw = bs.yaw;
    ms.leg_length = std::max(0.05f, bs.H - TableModel::TOP_THICKNESS);
    ms.leg_inset  = TableModel::LEG_RADIUS;
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;

    projection_->compute_projected_roi(inst);
    // (chain cov already computed above, before the belief update)

    // Object-anchor observation z_o: settled fit expressed in the localizer's base frame (gated OFF).
    compute_object_observation(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} dotd={:.2f} trunc={:.2f}{} mom={} | FE={:.2f} base={:.2f} surprise={:.2f} | cx={:.3f} cy={:.3f} H={:.3f} w={:.3f} h={:.3f} ψ={:.3f} | σ(w,h,H)mm=({:.0f},{:.0f},{:.0f}) | lidar {}/{} bp{} resid={:.3f}m topz={:.3f}(H={:.3f}) floorz={:.3f} covA={:.2f} div={}\n",
                   inst.node_name, npts, R, inst.last_motion_dotd, inst.last_trunc_frac, gated ? " GATED" : "",
                   inst.ai2_belief.last_moment_pts(),
                   energy, inst.fe_baseline, inst.fe_surprise,
                   bs.cx, bs.cy, bs.H, bs.w, bs.h, bs.yaw,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(2, 2))),
                   inst.dbg_lidar_rays, inst.dbg_lidar_raw, inst.dbg_lidar_bpearl_rays, inst.dbg_lidar_resid_m,
                   inst.dbg_lidar_topz_m, bs.H, inst.dbg_lidar_floorz_m, inst.dbg_lidar_cov_ang, inst.frames_diverged);

    // EvidenceMonitor snapshot: persist this frame's fit evidence (else it dies with the local observation).
    inst.dbg_cand_pts  = static_cast<int>(observation.candidate_pts.size());
    inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());
    inst.dbg_energy    = energy;
    inst.dbg_R         = R;
    inst.dbg_gated     = gated;

    log_ai2_csv(inst, npts, R, gated, energy);

    // Periodic shape model-selection (round vs square) on the accumulated cloud → inst.subtype.
    evaluate_shape(inst);
    return energy;
}

// Seed for the shape hypothesis test, derived from the CLOUD alone: xy centroid, a high-quantile top height,
// and the mean top-band radius as a size hint (w/h split 1.2/0.8 so the fit has an orientation to resolve).
// Mirrors tests/compare_models' init_from_cloud. Both hypotheses start here — see evaluate_shape on why.
static TableBeliefState shape_seed_from_cloud(const std::vector<Eigen::Vector3f>& pts)
{
    double sx = 0.0, sy = 0.0;
    std::vector<float> zs; zs.reserve(pts.size());
    for (const auto& p : pts) { sx += p.x(); sy += p.y(); zs.push_back(p.z()); }
    const float cx = static_cast<float>(sx / static_cast<double>(pts.size()));
    const float cy = static_cast<float>(sy / static_cast<double>(pts.size()));
    std::ranges::sort(zs);
    const float H = zs[static_cast<std::size_t>(0.90 * static_cast<double>(zs.size() - 1))];   // ~tabletop
    float rad = 0.0f; int n = 0;
    for (const auto& p : pts)
        if (std::abs(p.z() - H) < 0.08f) { rad += std::hypot(p.x() - cx, p.y() - cy); ++n; }
    rad = std::clamp(n > 0 ? rad / static_cast<float>(n) * 1.3f : 0.5f, 0.2f, 1.5f);  // mean radius → size hint
    return {cx, cy, H, 1.2f * rad, 0.8f * rad, 0.0f};
}

// SECOND seed: the top band's own principal axes and extent. The mean-radius hint above is a mean over a
// FILLED top, so it under-seeds a large footprint by roughly half — and the box then settles into a local
// optimum where the outlying points are written off as CLUTTER and never pull it open, while the
// single-parameter disc escapes and wins. Measured on the live bank this agent dumped
// (/tmp/round_table.xyz, 401 pts off the 1.26 x 0.69 m table): from the mean-radius seed the box reaches
// 0.61 x 0.68 and LOSES (lbf +0.030 → round); from this one it reaches 1.12 x 0.10 and WINS (lbf −0.595 →
// square). Same points, same budget, opposite verdicts — which is why evaluate_shape runs BOTH.
static TableBeliefState shape_seed_from_extent(const std::vector<Eigen::Vector3f>& pts)
{
    const TableBeliefState c = shape_seed_from_cloud(pts);
    const auto in_band = [&](const Eigen::Vector3f& p) { return std::abs(p.z() - c.H) < 0.08f; };
    double mx = 0.0, my = 0.0; int n = 0;
    for (const auto& p : pts) if (in_band(p)) { mx += p.x(); my += p.y(); ++n; }
    if (n < 8) return c;
    mx /= n; my /= n;
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (const auto& p : pts)
        if (in_band(p))
        {
            const double dx = p.x() - mx, dy = p.y() - my;
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
    sxx /= n; syy /= n; sxy /= n;
    const double tr = sxx + syy, det = sxx * syy - sxy * sxy;
    const double root = std::sqrt(std::max(0.0, 0.25 * tr * tr - det));
    // A uniformly filled side of length L has variance L²/12 along that side ⇒ L = sqrt(12·var).
    const float w = std::clamp(static_cast<float>(std::sqrt(12.0 * std::max(1e-6, 0.5 * tr + root))), 0.2f, 3.0f);
    const float h = std::clamp(static_cast<float>(std::sqrt(12.0 * std::max(1e-6, 0.5 * tr - root))), 0.2f, 3.0f);
    return {static_cast<float>(mx), static_cast<float>(my), c.H, w, h,
            static_cast<float>(0.5 * std::atan2(2.0 * sxy, sxx - syy))};
}

// Round-vs-square shape hypothesis test by free energy / model evidence (CONCEPT_AGENT_RECIPE.md §6, the
// "evidence over hypotheses, not a threshold" pattern; validated offline in tests/compare_models). Every
// cfg_.shape_eval_period cycles, once the ownership-gated support bank has enough points, fit BOTH shape
// models — square(box top + 4 legs) and round(disc top + 4 ring legs, MATCHED primitive cardinality so the
// mixture-prior baseline cancels and only disc-vs-box TOP shape is measured) — to the SAME accumulated cloud,
// with the SAME R, the SAME seed and the SAME iteration budget, and accumulate a BOUNDED sequential
// log-Bayes-factor (E_square − E_round). subtype flips at the zero boundary — no tuned cutoff. The clamp lets
// a converged run RECANT if a fuller view turns the evidence.
//
// ★A MODEL COMPARISON IS ONLY A COMPARISON IF BOTH SIDES ARE SCORED THE SAME WAY. Until 2026-08-17 this
// refitted the ROUND hypothesis to the cloud and scored the SQUARE one at the LIVE TRACKING BELIEF's state —
// a state fitted to the mask stream under priors, process noise and common-mode inflation, NOT to this cloud.
// So one hypothesis was optimised on the very data it was scored on and the other was not, and every
// transient error the tracker carried (a still-converging extent, an ego-motion-inflated size) was charged to
// the SQUARE alone and read as evidence for ROUND. LIVE SYMPTOM (table_3, a 1.26 x 0.69 m RECTANGULAR table):
// classified round for ~2000 cycles, then flipped to square with NO change in the round model and no change
// in the object — e_round sat pinned at 5.3085 (converged, refitted every time) while e_square walked from
// 5.97 down to 5.02 as the tracker's extent converged from 1.30x0.61 to 1.26x0.69. The verdict was measuring
// TRACKER CONVERGENCE, not shape. Measured in tests/shape_asymmetry: on a rectangular cloud a mid-convergence
// tracker gave lbf +1.45 (ROUND, wrong); refitting both from a cloud-derived seed gives -0.63 (square) and —
// the property that matters — the SAME -0.63 from every tracker state, because the shape verdict is now a
// function of the CLOUD alone. That also fixes the mirror case: the round cloud's true margin is only
// ~+0.01 nats/point (a box circumscribing a disc pays nothing for its empty corners — there is no free-space
// term in either model), so ANY procedural asymmetry is one to two orders of magnitude larger than the signal
// being measured and simply overwrites it.
void TableFitter::evaluate_shape(TableInstance& inst)
{
    if (cfg_.shape_eval_period <= 0) return;                              // gate disabled
    if (++inst.shape_eval_ctr < cfg_.shape_eval_period) return;
    inst.shape_eval_ctr = 0;
    const auto& bank = inst.support_bank_pts;
    if (static_cast<int>(bank.size()) < cfg_.shape_eval_min_points) return;

    // Stride-subsample to the point budget. This is a BATCH refit on the main thread every
    // shape_eval_period cycles, so its cost is a hitch in the belief loop; both hypotheses score the SAME
    // points, which makes the comparison PAIRED and its result nearly independent of the count (see
    // TableConfig::shape_fit_max_points for the measurements).
    std::vector<Eigen::Vector3f> sub;
    if (const auto cap = static_cast<std::size_t>(std::max(1, cfg_.shape_fit_max_points)); bank.size() > cap)
    {
        sub.reserve(cap);
        const double step = static_cast<double>(bank.size()) / static_cast<double>(cap);
        for (std::size_t i = 0; i < cap; ++i) sub.push_back(bank[static_cast<std::size_t>(i * step)]);
    }
    const std::vector<Eigen::Vector3f>& cloud = sub.empty() ? bank : sub;

    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m;        // SAME R both sides → normaliser cancels
    const int   iters = std::max(1, cfg_.shape_fit_iters);
    TableFrame  f; f.points = cloud;

    // Square hypothesis: a fresh belief with the PLAIN SDF-mixture parameters — the same mixture, clutter
    // floor and geometry constants the round model has, and none of the tracking-only factors (footprint
    // residual, coverage/free-space/moment precisions, process noise) that have no counterpart on the round
    // side and would make this an unequal contest again.
    TableBeliefParams sp;
    sp.sigma_base_m = cfg_.ai2_sigma_base_m; sp.clutter_frac = cfg_.ai2_clutter_frac;
    sp.clutter_scale_m = cfg_.ai2_clutter_scale_m; sp.prior_size_std = cfg_.ai2_prior_size_std;
    sp.top_thickness = TableModel::TOP_THICKNESS; sp.leg_radius = TableModel::LEG_RADIUS;
    RoundTableParams rp; rp.sigma_base_m = cfg_.ai2_sigma_base_m; rp.clutter_frac = cfg_.ai2_clutter_frac;
    rp.clutter_scale_m = cfg_.ai2_clutter_scale_m; rp.prior_size_std = cfg_.ai2_prior_size_std;

    // MULTI-START, both hypotheses, both seeds: each model is judged at its BEST (lowest-energy) fit. A
    // model comparison is a comparison of the two models' best accounts of the data; score each side at
    // whatever local optimum one arbitrary seed happened to reach and the verdict measures optimiser luck
    // instead. On the live bank that failure alone flipped the answer (see shape_seed_from_extent).
    float e_square = std::numeric_limits<float>::max(), e_round = std::numeric_limits<float>::max();
    for (const auto& seed : {shape_seed_from_cloud(cloud), shape_seed_from_extent(cloud)})
    {
        TableBelief square(seed, sp);
        for (int it = 0; it < iters; ++it) { square.update(f); square.resolve_orientation(cloud, R); }
        if (const float e = square.mean_energy(cloud, square.state(), R); e < e_square)
        {
            e_square = e;
            // What the hypothesis actually claimed, so the verdict can be read back against the object.
            inst.dbg_shape_sq_w = square.state().w;
            inst.dbg_shape_sq_h = square.state().h;
        }

        RoundTableBelief round({seed.cx, seed.cy, seed.H, 0.25f * (seed.w + seed.h)}, rp, RoundBase::Ring);
        for (int it = 0; it < iters; ++it) round.update(f);
        if (const float e = round.mean_energy(cloud, round.state(), R); e < e_round)
        {
            e_round = e;
            inst.dbg_shape_rd_r = round.state().radius;
        }
    }

    // Bounded sequential accumulation of the per-evaluation log-Bayes-factor (>0 ⇒ round explains it better).
    // ★NOT A SUM WHEN THE ALPHA IS SET. What is being re-scored is the accumulated support-bank cloud —
    // largely the SAME points at every evaluation — so adding the log-Bayes-factor each time counts one
    // body of evidence over and over: confidence grows with DWELL TIME rather than with information, the
    // accumulator pins at the clamp, and recanting then costs as many evaluations as saturating did. Live
    // symptom: the robot looking straight at a SQUARE table that stays classified round and "does not go
    // back". Same lesson ring_metaconcept already recorded for its own log-odds ("static furniture
    // re-observed is not new evidence; summing saturated the clamp on cycle 1"), and the same shape as the
    // existence channel's frame-correlation fix earlier today.
    //
    // With alpha > 0 the statistic tracks the CURRENT evidence on the CURRENT cloud with a time constant of
    // ~1/alpha evaluations, so a fuller view turns the verdict in bounded time no matter how long the old
    // one stood. alpha = 0 keeps the legacy sum for an A/B.
    const float lbf = e_square - e_round;
    inst.dbg_shape_lbf = lbf; inst.dbg_e_square = e_square; inst.dbg_e_round = e_round;
    inst.dbg_shape_pts = static_cast<int>(cloud.size());
    if (cfg_.shape_evidence_ema_alpha > 0.0f)
    {
        const float a = std::clamp(cfg_.shape_evidence_ema_alpha, 0.0f, 1.0f);
        inst.shape_evidence += a * (lbf - inst.shape_evidence);
    }
    else
        inst.shape_evidence += lbf;
    inst.shape_evidence = std::clamp(inst.shape_evidence,
                                     -cfg_.shape_evidence_clamp, cfg_.shape_evidence_clamp);
    const std::string prev = inst.subtype;
    inst.subtype = inst.shape_evidence > 0.0f ? "round" : "square";
    if (inst.subtype != prev)
        std::print("[{}] shape → {} (log-BF acc={:.2f}; e_sq={:.3f} [{:.2f}x{:.2f}] e_round={:.3f} [r={:.2f}] "
                   "on {} pts)\n",
                   inst.node_name, inst.subtype, inst.shape_evidence, e_square, inst.dbg_shape_sq_w,
                   inst.dbg_shape_sq_h, e_round, inst.dbg_shape_rd_r, cloud.size());
}

// Append one AI2 belief row (state + Σ-diag std + mask R/bias/trunc + mode evidence + LiDAR diag) to the CSV.
void TableFitter::log_ai2_csv(const TableInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        rc::diag::open_rotating(ai2_csv_, cfg_.ai2_csv_path);
        ai2_csv_.imbue(std::locale::classic());   // ★Qt imbues the GLOBAL locale, so operator<< inserts THOUSANDS
                                            // SEPARATORS into integers (pkt_ts 1785763853131 → "1,785,763,853,131"),
                                            // splitting one CSV field into five. Field counts then vary per row and
                                            // the whole log is unreadable by column name — every value past the
                                            // first big integer is shifted, which silently invalidates any analysis.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,pkt_fid,pkt_ts,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,motion_dotd,trunc_frac,range,"
                 << "cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw,"
                 << "std_yaw_within,flip_ev,p_alt,lidar_rays,lidar_raw,lidar_bpearl,lidar_resid_m,lidar_meanz,lidar_topz,lidar_floorz,lidar_cov_ang,"
                 << "dyaw_points,dyaw_moment,dyaw_flip,obliquity_cos,completeness,moment_aniso,moment_r_yaw,"
                 // ★The SHAPE decision and what produced it. It previously reached the outside world only
                 // through mesh_path (round_table.obj vs table.obj), so a discrimination that was failing
                 // could be seen ONLY by looking at the rendered mesh.
                 //
                 // ⚠THESE NAMES AND THE VALUES BELOW ARE TWO LISTS THAT MUST AGREE BY POSITION, and on the
                 // first run they did not: the values were emitted two fields early, so bank_pts (an int
                 // capped at SupportBankMaxPoints) was being read under e_square. The field COUNTS matched
                 // exactly — 94 and 94 — so no count check could have caught it. What caught it was 4000
                 // appearing where a free energy belongs. Same failure as the audit matrix whose column
                 // labels drifted out of order while every value read "ok": a mislabelled instrument is
                 // worse than none, because it is believed. If this file is touched again, the durable fix
                 // is to emit the header FROM the value list rather than maintaining two.
                 "shape_round,shape_evidence,shape_lbf,e_square,e_round,shape_sq_w,shape_sq_h,shape_rd_r,shape_pts,bank_pts,"
                 << "mom_major,mom_minor,mom_phi,mom_pts,"   // RAW footprint statistic (basin diagnosis)   // rogue-mask diag
                 << "ex_L,ex_p,ex_locc,ex_lfree,ex_lfree_eff,ex_ln,ex_socc,ex_sfree,ex_sfree_eff,ex_sndet,ex_streak,"
                 << "ex_pdetect,ex_central,ex_verify,ex_wantsverify,"   // existence-removal + verification-gate diag
                 << "ex_q,ex_qn,ex_lllr,"   // MEASURED clutter prior: shell rate q, its beam count, the LiDAR ΔL admitted
                 << "fixated,fix_close,fix_centred,fix_still,roi_off,roi_fill,roi_off_x,roi_off_y,"   // FIXATION (attention) gate diag
                 // ── DETECTOR-ENVELOPE CALIBRATION ────────────────────────────────────────────────────
                 // The columns needed to FIT rc::detect::DetectorEnvelope (min_fill/max_fill/soft) from real
                 // data instead of the current admitted prior. A row is written on EVERY cycle — including
                 // no-mask ones, via log_existence_cycle(npts=0) — so the NEGATIVES are present, which is what
                 // makes a detection-rate fit possible at all.
                 //   roi_fill_h/v : the two axes behind roi_fill's max. The model consumes only the max, but a
                 //                  grazing view is a tall thin sliver whose max is large, so its misses would
                 //                  otherwise drag the fitted max_fill shoulder down for the wrong reason.
                 //   det_alive    : did YOLO actually fire this cycle. `npts` is NOT a substitute — it
                 //                  conflates "no mask" with "mask present but gated/rejected".
                 //   roi_valid    : drop rows where the projection is degenerate, or garbage fill pollutes it.
                 << "roi_fill_h,roi_fill_v,roi_valid,det_alive,frames_since_det,"
                 // NBV emission: what the planner proposed THIS cycle (the affordance node holds the
                 // FROZEN pose during a claim, so it cannot answer this). nbv_vfov=0 ⇒ the camera
                 // model was still incomplete when the proposal was made.
                 << "nbv_standoff,nbv_tx,nbv_ty,nbv_pdetect,nbv_fill,nbv_vfov,nbv_gain_raw,nbv_gain_pub,ricoh_det,ricoh_attn\n";
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
             << inst.last_trunc_frac << ',' << inst.last_range << ','
             << s.cx << ',' << s.cy << ',' << s.H << ',' << s.w << ',' << s.h << ',' << s.yaw << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ','
             << std::sqrt(std::max(0.0f, inst.ai2_belief.yaw_marginal_var())) << ','
             << sd(5) << ',' << inst.ai2_belief.flip_evidence() << ',' << inst.ai2_belief.mode_posterior() << ','
             << inst.dbg_lidar_rays << ',' << inst.dbg_lidar_raw << ',' << inst.dbg_lidar_bpearl_rays << ',' << inst.dbg_lidar_resid_m << ','
             << inst.dbg_lidar_meanz_m << ',' << inst.dbg_lidar_topz_m << ',' << inst.dbg_lidar_floorz_m << ','
             << inst.dbg_lidar_cov_ang << ','
             << inst.dbg_dyaw_points << ',' << inst.dbg_dyaw_moment << ',' << inst.dbg_dyaw_flip << ','
             << inst.dbg_obliquity_cos << ',' << inst.dbg_completeness << ','
             << inst.ai2_belief.dbg_moment_aniso() << ',' << inst.ai2_belief.dbg_moment_r_yaw() << ','   // rogue-mask diag
             << (inst.subtype == "round" ? 1 : 0) << ',' << inst.shape_evidence << ','
             << inst.dbg_shape_lbf << ',' << inst.dbg_e_square << ',' << inst.dbg_e_round << ','
             << inst.dbg_shape_sq_w << ',' << inst.dbg_shape_sq_h << ',' << inst.dbg_shape_rd_r << ','
             << inst.dbg_shape_pts << ',' << inst.support_bank_pts.size() << ','
             << inst.ai2_belief.dbg_moment_ext_major() << ',' << inst.ai2_belief.dbg_moment_ext_minor() << ','
             << inst.ai2_belief.dbg_moment_phi() << ',' << inst.ai2_belief.dbg_moment_pts() << ','
             << inst.existence.logodds() << ',' << inst.existence.p_exists() << ','
             << inst.dbg_ex_lidar_occ << ',' << inst.dbg_ex_lidar_free << ',' << inst.dbg_ex_lidar_free_eff << ',' << inst.dbg_ex_lidar_n << ','
             << inst.dbg_ex_sil_occ << ',' << inst.dbg_ex_sil_free << ',' << inst.dbg_ex_sil_free_eff << ',' << inst.dbg_ex_sil_ndet << ','
             << inst.existence_debounce.streak << ','
             << inst.dbg_ex_pdetect << ',' << inst.dbg_ex_central << ','
             << inst.verify_surprise << ',' << (inst.wants_verification ? 1 : 0) << ','   // existence-removal + verification-gate diag
             << inst.dbg_ex_clutter_q << ',' << inst.dbg_ex_clutter_n << ',' << inst.dbg_ex_lidar_llr << ','
             << (inst.dbg_fixated ? 1 : 0) << ',' << (inst.dbg_fix_close ? 1 : 0) << ','
             << (inst.dbg_fix_centred ? 1 : 0) << ',' << (inst.dbg_fix_still ? 1 : 0) << ','
             << std::hypot(inst.roi_offset_x, inst.roi_offset_y) << ',' << inst.roi_fill << ','
             // Split the centring radius into its axes: the hypothesis is that roi_off is dominated by
             // off_y because a 0.73 m tabletop seen from a ~1.2 m camera sits LOW in the frame — i.e. the
             // fovea test is biased against tables by MOUNTING GEOMETRY, not by inattention.
             << inst.roi_offset_x << ',' << inst.roi_offset_y << ','   // FIXATION (attention) gate diag
             // Detector-envelope calibration — see the header comment for why each column is here.
             << inst.roi_fill_h << ',' << inst.roi_fill_v << ','
             << (inst.roi_valid ? 1 : 0) << ',' << (inst.detection_alive ? 1 : 0) << ','
             << inst.frames_since_detection << ','
             << inst.dbg_nbv_standoff << ',' << inst.dbg_nbv_target_x << ',' << inst.dbg_nbv_target_y << ','
             << inst.dbg_nbv_pdetect << ',' << inst.dbg_nbv_fill << ',' << inst.dbg_nbv_vfov << ','
             << inst.dbg_nbv_gain_raw << ',' << inst.dbg_nbv_gain_pub
             // Peripheral (ricoh-360) attention: unassigned 360 detections of this label this
             // cycle. It was computed and shown nowhere — the whole peripheral channel was
             // invisible in every agent's log, which is why 'is it producing anything?' could
             // not be answered offline for cabinet/hood, the two it exists for.
             << ',' << ricoh_dets_ << ',' << ricoh_attention_ << '\n';
    ai2_csv_.flush();
}

// ─── Factory helpers ──────────────────────────────────────────────────────────────────────────────

// Build the TableModel params from config (the SDF split band for candidate/residual classification).
TableModelParams TableFitter::make_model_params() const
{
    TableModelParams p;
    p.sigma_obs = cfg_.sigma_obs;   // top/leg SDF split band for the candidate/residual classification
    return p;
}

}  // namespace rc
