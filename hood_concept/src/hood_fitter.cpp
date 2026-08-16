/*
 * hood_fitter.cpp — the active-inference fit core for hood_concept (see hood_fitter.h).
 *
 * Implements the per-"hood_*" instance lifecycle and the AI2 full-covariance belief update: ensure_instance
 * (birth-seed + RT/prior warm-start + NaN sanitize), observe_slice (mask-cloud → candidate/residual SDF split
 * + per-slice R inputs), and run_inference (lazy footprint-moment birth, support-bank ingest, one HoodBelief
 * update with range/motion covariance, the step-bound divergence net, FE-surprise attention, orientation-mode
 * resolution, and write-back into the legacy HoodState). Collaborates with MaskIngestor, HoodSceneGraph,
 * HoodProjection, HoodLidarRangeChannel, and the header-only support bank; SpecificWorker owns orchestration.
 */

#include "hood_fitter.h"
#include "hood_support_bank.h"
#include "../../common/object_anchor/object_anchor_contract.h"
#include "../../common/object_anchor/ray_anisotropic_cov.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <locale>
#include <print>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rc {

// ─── Construction & chain covariance ──────────────────────────────────────────────────────────────

HoodFitter::HoodFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         HoodConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         HoodSceneGraph* scene_graph)
    : G_(graph), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph),
      projection_(std::make_unique<HoodProjection>(graph, inner_eigen, mask_ingestor))
{
    HoodBeliefState::use_quotient = cfg.quotient_chart;   // C2v symmetry-quotient optimisation chart (global mode)
    projection_->set_central_region_frac(cfg.central_region_frac);
    projection_->set_front_params(cfg.front_min_face_area_px, cfg.front_min_confidence);
}

// ─── Ego-motion "be-still-to-update" (ported 1:1 from chair_concept) ─────────────────────────────────

// Robot/camera speed in the room frame from the transform chain (producer-independent). Uses the SAME room_T_zed
// extrinsic the fit uses (HoodProjection), pinned to the CURRENT pose (ts=0). Call once per compute cycle.
void HoodFitter::update_ego_motion()
{
    const auto M = projection_->room_T_zed_matrix(0);   // camera→room (ts=0 = current robot pose)
    const auto now = std::chrono::steady_clock::now();
    if (not M.has_value())
    {
        have_prev_cam_ = false;   // pose chain unavailable → can't judge motion; reset the baseline
        return;
    }
    const Eigen::Vector3f pos(static_cast<float>(M->coeff(0, 3)),
                              static_cast<float>(M->coeff(1, 3)),
                              static_cast<float>(M->coeff(2, 3)));
    const Eigen::Vector3f fwd(static_cast<float>(M->coeff(0, 1)),   // zed +y is the depth/forward axis
                              static_cast<float>(M->coeff(1, 1)),
                              static_cast<float>(M->coeff(2, 1)));
    if (have_prev_cam_)
    {
        const float dt = std::max(1e-3f, std::chrono::duration<float>(now - prev_cam_tp_).count());
        ego_lin_mps_ = (pos - prev_cam_pos_).norm() / dt;
        const float fa = prev_cam_fwd_.norm(), fb = fwd.norm();
        const float cang = (fa > 1e-6f and fb > 1e-6f)
            ? std::clamp(prev_cam_fwd_.dot(fwd) / (fa * fb), -1.0f, 1.0f) : 1.0f;
        ego_ang_radps_ = std::acos(cang) / dt;
    }
    prev_cam_pos_ = pos; prev_cam_fwd_ = fwd; prev_cam_tp_ = now; have_prev_cam_ = true;
}

// Robust combined ego-motion magnitude (m/s): the per-mask corruption speed OR'd with the robot's own measured
// speed (linear + a lever-arm conversion of angular), so it works whether or not the producer populated motion_dotd.
float HoodFitter::motion_magnitude(const HoodInstance& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear/distortion), → 1 at
// centroid radius ai2_periph_ref. This is the "well-centred masks stay trustworthy" lever, expressed continuously.
float HoodFitter::periphery_penalty(const HoodInstance& inst) const
{
    const float ref = std::max(1e-6f, cfg_.ai2_periph_ref);
    const float r = std::max(0.0f, inst.last_centroid_radius) / ref;
    return std::clamp(r * r, 0.0f, 1.0f);
}

// "Be-still-to-update": true ⇒ this frame may only CONFIRM (predict-only), never move/reshape the geometry mean.
// Robot linear/angular speed above the still-level, OR the mask's own ego-motion corruption (motion_dotd) above
// its still-level. EXCEPTION: a well-centred mask (near the principal point) is trustworthy even while moving.
// Is THIS raw mask good enough to move geometry? Byte-for-byte the admissibility the UPDATE path applies to an
// existing instance (the trunc gate + confirm_only), but evaluated on a slice that has no instance yet — every
// input is already on the slice (motion_dotd, centroid_radius, trunc_frac) plus this cycle's ego speeds.
//
// ★A frame that may not MOVE an existing belief must not CREATE one. Birth used to admit frames the fit itself
// would refuse — a smeared mask taken while the robot drives past, or one clipped at the image border — which is
// how furniture appeared in passing. One rule, defined once, applied to both ends of the lifecycle.
bool HoodFitter::frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const
{
    if (sl.trunc_frac > cfg_.ai2_trunc_gate_frac)
        return false;                                   // truncated ⇒ extent is a lower bound ⇒ never births
    if (not cfg_.ai2_motion_confirm_only)
        return true;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(sl.motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return true;                                    // robot still ⇒ admissible
    // Moving: only a well-CENTRED mask is trusted (the robot is looking AT it), exactly as in confirm_only.
    return cfg_.ai2_moving_update_center_radius >= 0.0f
       and sl.centroid_radius <= cfg_.ai2_moving_update_center_radius;
}

bool HoodFitter::confirm_only(const HoodInstance& inst) const
{
    if (not cfg_.ai2_motion_confirm_only)
        return false;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(inst.last_motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return false;
    if (cfg_.ai2_moving_update_center_radius >= 0.0f
        and inst.last_centroid_radius <= cfg_.ai2_moving_update_center_radius)
        return false;   // well-centred mask → trust it even while moving
    return true;        // moving AND the mask is off-centre → confirmation only
}

// ─── Appearance FRONT (door) resolver ───────────────────────────────────────────────────────────────

// Project the fitted box into the newest ZED RGB, score door-ness per visible face, and fold the winning cue
// into the belief's discrete door-mode resolver. No-op unless the feature is on and a FRESH RGB frame is staged;
// the per-instance capture-stamp gate stops a stalled/duplicated frame from re-integrating the same appearance.
void HoodFitter::resolve_front_from_rgb(HoodInstance& inst, float mode_evidence_weight)
{
    inst.last_front_conf = 0.0f;
    if (not cfg_.front_detect_enabled or not rgb_fresh_ or rgb_frame_.empty())
        return;
    // Freshness-as-precision: skip a frame we already folded in (same capture stamp). stamp==0 (unstamped
    // producer) falls back to the per-cycle rgb_fresh_ flag, which the worker only sets on a genuinely new frame.
    if (rgb_stamp_ms_ != 0 and rgb_stamp_ms_ == inst.last_front_stamp_ms)
        return;
    inst.last_front_stamp_ms = rgb_stamp_ms_;

    // Project the FRESHLY-UPDATED belief box (not the not-yet-written-back model state) into the RGB.
    const auto& bs = inst.ai2_belief.state();
    HoodState box = inst.model.state();
    box.cx = bs.cx; box.cy = bs.cy; box.z_top = bs.H; box.extent = cfg_.vertical_extent_m;
    box.w = bs.w; box.h = bs.h; box.yaw = bs.yaw;
    const auto cue = projection_->detect_front(box, rgb_frame_, rgb_stamp_ms_);
    if (not cue.has_value())
        return;
    // Where this look came FROM, for the accumulator's novelty budget (see FrontCue::view_bearing_rad).
    // ★PINNED TO THE CAPTURE STAMP, not ts=0. Asking for the LATEST camera pose files a look under wherever
    // the camera is NOW, so while the robot moves the novelty bookkeeping mis-registers itself: it can credit
    // an exhausted viewpoint as fresh and charge a genuinely new one as spent. detect_front already pins its
    // own projection to this same stamp (hood_projection.cpp), so using ts=0 here also meant the cue
    // and its provenance disagreed about which camera pose produced them.
    if (const auto M = projection_->room_T_zed_matrix(rgb_stamp_ms_); M.has_value())
    {
        const float camx = static_cast<float>((*M)(0, 3)), camy = static_cast<float>((*M)(1, 3));
        const_cast<FrontCue&>(*cue).view_bearing_rad = std::atan2(camy - bs.cy, camx - bs.cx);
    }
    inst.last_front_conf    = cue->confidence;
    inst.last_front_bearing = cue->bearing_rad;
    // evidence_weight = ego-motion reliability (a moving frame's smeared door face barely votes on the discrete
    // orientation); the belief additionally weights by the cue confidence. Adopt-on-argmax is inside the belief.
    const bool flipped = inst.ai2_belief.resolve_front(*cue, mode_evidence_weight);
    if (cfg_.front_log and (flipped or should_log(inst)))
        std::print("[{}] FRONT cue bearing={:.3f} conf={:.2f} w={:.2f} → {} | door_mode={} p={:.2f} σyaw={:.2f}deg\n",
                   inst.node_name, cue->bearing_rad, cue->confidence, mode_evidence_weight,
                   flipped ? "ADOPTED door flip" : "held", inst.ai2_belief.front_mode(),
                   inst.ai2_belief.front_confidence(),
                   std::sqrt(inst.ai2_belief.yaw_marginal_var()) * 57.29578f);
}

// Enable Part-B chain-covariance propagation from source_frame (no-op unless a gaussian API + frame are given).
void HoodFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void HoodFitter::set_object_observation(bool enabled, std::string robot_frame)
{
    obs_robot_frame_ = std::move(robot_frame);
    obs_enabled_     = enabled and (inner_eigen_ != nullptr) and not obs_robot_frame_.empty();
}

// Build the object-anchor observation z_o for room_concept's landmark factor.
//
// z_o MUST be independent of the robot pose the localizer is estimating, or the factor just
// re-anchors to the last pose (the residual is ~0 at the current estimate). So we take the hood's
// RAW camera-frame mask centroid (this frame's ZED measurement, no localization in it) and carry it
// to the robot base by the STATIC body←zed extrinsic (ts=0, a fixed calibrated mount). Position-only:
// a single view's yaw is biased (the grazing/obliquity problem), so we publish [x,y] and let the
// consumer treat it as a 2-DOF landmark. Gated OFF by default.
void HoodFitter::compute_object_observation(HoodInstance& inst)
{
    inst.obs_robot_valid = false;
    if (not obs_enabled_ or not inner_eigen_ or not mask_ingestor_)
        return;
    const auto& packet = mask_ingestor_->packet();
    if (packet.support_points_cam.empty())
        return;

    // This frame's ZED slice assigned to this hood (the pinhole camera cloud; skip ricoh depth_var>0).
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
    // box_yaw in body = belief yaw (room) + yaw(body←room). See HOOD_TRIANGULATION.md.
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

// Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the hood centre; store it on the instance.
//
// Transform the centre to the measurement frame and back to room with ZERO input cov, so InnerGaussianAPI
// returns exactly the chain contribution (Σ_chain from each RT edge's rt_covariance), pinned to the mask
// capture stamp. The hood is fit in room but its position stays conditional on the robot pose
// (camera→robot→room), so this per-frame SHARED localization error feeds the belief common-mode.
void HoodFitter::compute_chain_cov(HoodInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the hood centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution
    // (Σ_chain from each RT edge's rt_covariance), pinned to the mask capture stamp. The hood is fit in
    // room but its position is still conditional on the robot pose (camera→robot→room), so this applies.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, 0.0);   // hood node origin = base on the floor (z=0)
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

// Create the instance for a "hood_*" node if absent; return true only the first time it is created.
//
// Warm-starts from the node's size attribs and the room→hood RT edge, then OVERRIDES with the tracker
// birth seed when present (the RT edge written at birth is not reliably queryable this same cycle — it reads
// 0,0 — which would freeze the model at the origin and cause endless re-births). Sanitizes any non-finite
// field before it can poison the SDF and lock the optimizer.
bool HoodFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    HoodState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w            = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.h            = v.value();
    // ★height_m is the drawn EXTENT and the RT origin is the body's BASE — that is the publish convention
    // (see HoodSceneGraph::write_rt_pose). Reading height_m back as the TOP inverted the round-trip: after
    // the publish was corrected to emit the extent, an ADOPTED node came back with z_top = 0.50 m, i.e. a
    // collapsed box on the floor. Read the extent as the extent, and recover the top from the RT z below.
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.extent = v.value();

    // Read RT pose from room→hood edge
    if (room_node_id_ != 0)
    {
        if (const auto edge = G_->get_edge(room_node_id_, node.id(), "RT"); edge.has_value())
        {
            if (const auto tr = G_->get_attrib_by_name<rt_translation_att>(edge.value()); tr.has_value())
            {
                const auto& tvec = tr.value().get();
                if (tvec.size() >= 2) { init_state.cx = tvec[0]; init_state.cy = tvec[1]; }
                // The RT origin is the body's BASE (z0), so the top is base + extent — the inverse of what
                // write_rt_pose publishes. A floor-anchored read (z_top = z) would put the body underground.
                if (tvec.size() >= 3) init_state.z_top = tvec[2] + init_state.extent;
            }
            if (const auto rot = G_->get_attrib_by_name<rt_rotation_euler_xyz_att>(edge.value()); rot.has_value())
            {
                const auto& rvec = rot.value().get();
                if (rvec.size() >= 3) init_state.yaw = rvec[2];
            }
        }
    }

    // Tracker birth seed: authoritative for a freshly-born instance. The room→hood RT edge written at
    // birth is not reliably queryable this same cycle (it reads as 0,0), and the warm-start would then
    // freeze the model at the origin forever → the tracker never associates and re-births endlessly.
    // Only the XY is seeded here. All cold-start GEOMETRY (centre, height, footprint, yaw) is owned by
    // run_inference's lazy snap, which overwrites it from the observed cloud on the first frame with points —
    // anything set here beyond xy would simply be discarded one frame later.
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

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF and lock the
    // optimizer; replace any non-finite field with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("hood_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.yaw, 0.0f, "yaw");
        // ★Fall back to THIS object's priors, not the parent's. These were 0.60×0.60×1.70 — a fridge, and a
        // fourth disagreeing height at that (the config/manifest say 0.90 × 0.50, top 2.05, extent 0.50). A
        // hood recovering from a corrupt attribute was re-seeded as a small square box on the floor.
        fix(init_state.w, cfg_.ai2_prior_footprint_m, "w");
        fix(init_state.h, cfg_.ai2_prior_depth_m > 0.0f ? cfg_.ai2_prior_depth_m
                                                        : cfg_.ai2_prior_footprint_m, "h");
        fix(init_state.extent, cfg_.vertical_extent_m, "extent");
        fix(init_state.z_top, cfg_.ai2_prior_height_m, "z_top");
    }

    HoodInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();

    inst.model = HoodModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name(), "hood");

    instances_.emplace(node.id(), std::move(inst));
    std::print("hood_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ──────────────────────────────────────────────────────────────────────────────────

// Build an observation from ONE assigned mask slice: latch that slice's R inputs, then SDF-split its cloud.
//
// Classify-don't-destroy split into candidate (near the current surface) vs residual (off-surface) points.
// The caller (process_hood_node) invokes this once per assigned slice and runs a belief update for each —
// sequential fusion that keeps every sensor's R and common-mode separate. RICOH slices (depth_var>0) are
// BEARING-ONLY and return empty (never fitted); they drive only the attention path.
HoodFitter::HoodObservation HoodFitter::observe_slice(HoodInstance& inst, int slice_index)
{
    HoodObservation observation;
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
    // A slice assigned to this hood ⇒ detection is alive; latch its per-slice R inputs.
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

    const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
    const std::size_t end   = std::min(slice.support_end,   masks_packet.support_points.size());

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
HoodFitter::HoodObservation HoodFitter::observe(HoodInstance& inst, const DSR::Node& node)
{
    // No mask slice was assigned this cycle. There is no node-attrib sensing path (nothing writes
    // candidate/residual points onto the node), so this is always a stale observation: has_fresh_data=false
    // ⇒ run_inference ages the belief (predict-only) instead of fitting.
    (void)inst; (void)node;
    return HoodObservation{};
}

// True on the config-driven log period (one in hood_log_period_frames cycles).
bool HoodFitter::should_log(const HoodInstance& inst) const
{
    const int period = std::max(1, cfg_.hood_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Room walls (wall-flush factor; simplified from cabinet_concept — no run/seg/collinear-merge) ─────

// Nearest room-polygon edge to a room-frame point q, as a WallRef {foot on the edge, UNIT normal pointing
// INTO the room, wall sigma}. Returns ok=false when no room polygon is known ⇒ the wall factor is inert
// (the fridge is treated as free-standing). The inward normal is the one that points toward room_interior_.
WallRef HoodFitter::nearest_wall(const Eigen::Vector2f& q) const
{
    WallRef w;
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 2) return w;   // no model ⇒ inert ⇒ free-standing
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    Eigen::Vector2f best_foot = Eigen::Vector2f::Zero();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f  ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-8f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const Eigen::Vector2f foot = a + t * ab;
        const float dist2 = (q - foot).squaredNorm();
        if (dist2 < best) { best = dist2; best_i = i; best_foot = foot; }
    }
    if (best_i >= n) return w;
    const Eigen::Vector2f& a = room_polygon_[best_i];
    const Eigen::Vector2f  ab = room_polygon_[(best_i + 1) % n] - a;
    Eigen::Vector2f nrm(-ab.y(), ab.x());
    nrm.normalize();
    if (nrm.dot(room_interior_ - best_foot) < 0.0f) nrm = -nrm;   // point INTO the room
    w.ok = true; w.p = best_foot; w.n = nrm; w.sigma_m = 0.02f;
    w.length_m = ab.norm();   // the segment's length ⇒ how well its DIRECTION is known (see WallRef)
    return w;
}

// Continuous interior support at q (see the header). Even-odd containment test against the room polygon, then a
// smooth falloff over one fridge half-depth measured from the boundary — so a fridge legitimately pushed against
// a wall keeps most of its support, while a detection at or past the boundary has none. No hard cutoff.
float HoodFitter::interior_support(const Eigen::Vector2f& q) const
{
    const std::size_t n = room_polygon_.size();
    if (not have_room_geometry_ or n < 3) return 1.0f;   // no room model ⇒ the room asserts nothing
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f& b = room_polygon_[j];
        if ((a.y() > q.y()) != (b.y() > q.y()) and
            q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x())
            inside = not inside;
    }
    if (not inside) return 0.0f;                          // outside the layout: the room explains it, not a fridge
    const auto w = nearest_wall(q);
    if (not w.ok) return 1.0f;
    const float d = (q - w.p).norm();                     // distance in from the boundary
    // Reuse the flush factor's own scale: within AI2WallReachM of the boundary you are effectively AT the wall
    // (that is precisely what that constant means), so no new knob. A real fridge's mask centroid sits well
    // beyond it — a front-only view is ~a depth in — and keeps ~full support.
    const float reach = std::max(1e-3f, cfg_.ai2_wall_reach_m);
    return 1.0f - std::exp(-(d * d) / (2.0f * reach * reach));
}

// ─── Inference ────────────────────────────────────────────────────────────────────────────────────

// One recursive full-covariance belief update (or age-only step) for this instance; returns the free energy.
//
// Lazy first-frame init (snap centre/height to the cloud, footprint-moment birth of w/h/yaw), support-bank
// ingest, then the range/motion covariance and the HoodBelief update guarded by a step-bound divergence net,
// FE-surprise attention baseline, and orientation-mode resolution. On a stale frame it ages the belief
// (Σ grows on the agent's clock) instead of freezing. Result is written back into the legacy HoodState.
float HoodFitter::run_inference(HoodInstance& inst, const HoodObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy init: warm-start the belief from the model state, but on the FIRST frame snap the centre/height
    // to the observed cloud — a box far from the points would see them all as clutter (zero gradient) and
    // never converge (the AI2 analogue of the legacy cold-start snap).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        HoodBeliefState s0{m.cx, m.cy, m.z_top, m.w, m.h, m.yaw};

        // The cold-start cloud is THIS FRAME's mask only.
        //
        // ★Do not union the candidate's probation burst in here. It was tried (common/birth_fragment) and
        // measurably made birth WORSE: unioning ~8 frames of room-frame points spreads them by the robot-pose
        // error accumulated over the window, and the footprint moment reads that spread as extent. Live A/B on
        // the same fridge — single frame born at w=0.669 h=0.600, already the converged values; burst union
        // born at w=0.915 (+37%) h=0.494, then ~255 cycles crawling back to w=0.673 h=0.626. The give-away is
        // that ext_MAJOR inflated too: a genuine second face lifts only ext_minor, blur stretches both. This is
        // Khronos's spatio-temporal local consistency (arXiv:2402.13817 eq. 8) failing over a birth window —
        // and at BirthFrames=8 / ~10 Hz the robot cannot reach a second face anyway, so the union can only ever
        // be the same face smeared. See [[hood-birth-burst-smear]].
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

            // Birth seed (w, yaw) from the footprint second-moment of the BODY cloud: start the box near the
            // mask's real width and orientation so the per-point mixture associates immediately.
            //
            // ★The band is the whole box, NOT a top slab. The table-concept original sampled
            // [H − TOP_THICKNESS − 3σ, H + 3σ] — for a table that IS the tabletop (the full footprint), but for a
            // solid fridge it is a ~9 cm sliver at 1.9 m seen grazing, so its minor extent collapsed and line
            // `h = max(0.10, ext_minor)` birthed EVERY fridge as a 10 cm-deep slab (live: every birth in
            // etc/ai2_log.csv starts h=0.10). See [[hood-table-geometry-churn]].
            //
            // DEPTH (h) is not seeded from a single view at all: one camera pose sees ONE vertical face, whose
            // footprint projects to a line, so ext_minor measures mask noise rather than the fridge's depth. Depth
            // is UNOBSERVED until the cloud spans it — exactly the condition the fit's depth-observability prior
            // already tests (ai2_depth_obs_band_m). Below that span h stays at the footprint prior mean and the
            // AI2DepthUnobsPrecision term holds it there; above it the view genuinely resolves depth, so use it.
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
            {
                std::vector<Eigen::Vector3f> pts;
                pts.reserve(static_cast<std::size_t>(npts));
                pts.insert(pts.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
                pts.insert(pts.end(), observation.residual_pts.begin(), observation.residual_pts.end());
                // Body band: the full observed height minus the floor-contact and above-top over-segmentation
                // tails, which are cloud ARTEFACTS (not fridge surface) and would bias the footprint outward.
                const float z_hi = s0.H + 3.0f * cfg_.ai2_sigma_base_m;
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
                if (const auto mom = HoodBelief::footprint_moment(pts, z_lo, z_hi); mom.ok)
                {
                    s0.cx = mom.cx; s0.cy = mom.cy;
                    s0.w = std::max(0.10f, mom.ext_major);          // major axis = the seen face's width (reliable)
                    s0.h = (mom.ext_minor > cfg_.ai2_depth_obs_band_m) ? mom.ext_minor
                                                                       : cfg_.ai2_prior_footprint_m;
                    // Expected: "prior". From ONE camera pose the footprint is a line, so ext_minor is mask
                    // noise (live: ~0.03 against a 0.10 band) and depth correctly defers to the prior. A birth
                    // reporting DEPTH RESOLVED from a single frame means the cloud is not one face — check
                    // ext_major against the fridge's real width before believing it.
                    std::print("[{}] cold-start footprint: {} pts → ext_major={:.3f} ext_minor={:.3f} "
                               "(band={:.3f}) → h={:.3f} [{}]\n",
                               inst.node_name, npts, mom.ext_major, mom.ext_minor,
                               cfg_.ai2_depth_obs_band_m, s0.h,
                               mom.ext_minor > cfg_.ai2_depth_obs_band_m ? "DEPTH RESOLVED" : "prior");
                    // Only commit the orientation when the footprint is clearly anisotropic. A near-square
                    // footprint has an ill-defined principal axis (phi swings ±90° on noise), so birthing yaw
                    // from a single frame would seed a random ±90°; leave yaw at its RT/default seed and let
                    // resolve_orientation settle it once evidence accrues.
                    const float sum = mom.ext_major + mom.ext_minor;
                    if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)
                        s0.yaw = mom.phi;
                }
            }
        }
        HoodBeliefParams p;
        p.sigma_base_m    = cfg_.ai2_sigma_base_m;
        p.clutter_frac    = cfg_.ai2_clutter_frac;
        p.clutter_scale_m = cfg_.ai2_clutter_scale_m;
        p.prior_size_std  = cfg_.ai2_prior_size_std;
        p.prior_footprint_m   = cfg_.ai2_prior_footprint_m;
        p.prior_depth_m       = cfg_.ai2_prior_depth_m;
        p.prior_footprint_std = cfg_.ai2_prior_footprint_std;
        p.prior_height_m      = cfg_.ai2_prior_height_m;
        p.prior_height_std    = cfg_.ai2_prior_height_std;
        p.depth_unobs_precision = cfg_.ai2_depth_unobs_precision;
        p.depth_obs_band_m      = cfg_.ai2_depth_obs_band_m;
        p.top_no_float_precision = cfg_.ai2_top_no_float_precision;
        p.top_no_float_margin_m  = cfg_.ai2_top_no_float_margin_m;
        p.wall_precision          = cfg_.ai2_wall_precision;
        p.wall_parallel_precision = cfg_.ai2_wall_parallel_precision;
        p.wall_reach_m            = cfg_.ai2_wall_reach_m;
        p.wall_flush_prior        = cfg_.ai2_wall_flush_prior;
        p.wall_no_cross_precision = cfg_.ai2_wall_no_cross_precision;
        p.wall_no_cross_margin_m  = cfg_.ai2_wall_no_cross_margin_m;
        // The wall may only explain away mask points to the extent the room itself is trusted: with no room
        // polygon pushed this session there is nothing to explain away with, so the component is OFF and the
        // mixture is box+clutter exactly as before. (When room_concept publishes a map-trust scalar, multiply
        // here — a poorly-localized room must return the fridge hypothesis its freedom, not silently veto it.)
        p.wall_explain_frac    = have_room_geometry_ ? cfg_.ai2_wall_explain_frac : 0.0f;
        p.wall_explain_sigma_m = cfg_.ai2_wall_explain_sigma_m;
        p.door_clearance_gain  = cfg_.ai2_door_clearance_gain;
        p.volatility_infer     = cfg_.ai2_volatility_infer;
        p.volatility_lr        = cfg_.ai2_volatility_lr;
        p.volatility_sigma     = cfg_.ai2_volatility_sigma;
        p.pixel_sigma_over_f     = cfg_.pixel_sigma_over_f;
        p.depth_sigma0_m         = cfg_.depth_sigma0_m;
        p.depth_sigma_range_coef = cfg_.depth_sigma_range_coef;
        p.model_sigma_m          = cfg_.model_sigma_m;
        p.footprint_residual     = cfg_.footprint_residual;   // a1′+a2′: 2-D footprint residual + shared depth-affine
        p.depth_bias_std         = cfg_.depth_bias_std;
        p.depth_scale_std        = cfg_.depth_scale_std;
        p.process_std_m   = cfg_.ai2_process_std_m;
        p.process_std_yaw = cfg_.ai2_process_std_yaw;
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
        p.top_thickness   = HoodModel::TOP_THICKNESS;
        p.leg_radius      = HoodModel::LEG_RADIUS;
        // "Is this really a fridge?" plausibility + short-height prior parameters (mis-detection filter).
        p.plaus_aspect_scale      = cfg_.plaus_aspect_scale;
        p.plaus_size_scale        = cfg_.plaus_size_scale;
        p.plaus_height_min        = cfg_.plaus_height_min;
        p.plaus_height_soft       = cfg_.plaus_height_soft;
        p.plaus_fe_ref            = cfg_.plaus_fe_ref;
        p.plaus_fe_scale          = cfg_.plaus_fe_scale;
        p.plaus_alt_size_scale    = cfg_.plaus_alt_size_scale;
        p.plaus_height_prior_gain = cfg_.fridge_filter_enabled ? cfg_.plaus_height_prior_gain : 0.0f;
        inst.ai2_belief = HoodBelief(s0, p);
        inst.ai2_initialized = true;
        std::print("[{}] cold-start state: cx={:.2f} cy={:.2f} H={:.2f} w={:.2f} h={:.2f} yaw={:.2f}\n",
                   inst.node_name, s0.cx, s0.cy, s0.H, s0.w, s0.h, s0.yaw);
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
    // information-filter axiom), so a dead mask/ZED feed read downstream as a confident-but-stale hood.
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
        inst.dbg_gate_fresh = false;   // no mask reached the fit ⇒ the gate verdict below is STALE this cycle
        projection_->compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD the last free energy — no new mask ≠ FE 0 (the fit is unchanged)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting (motion-free). Even at zero camera motion, deprojection noise grows with
    // distance AND a far mask subtends a tiny angle, so pose — orientation most of all — becomes
    // unobservable: a 7 m view should confirm existence but never rotate a converged hood. The
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
    // Use the ROBUST combined motion signal (motion_magnitude = max(|motion_dotd|, ego_lin + lever·ego_ang)),
    // scaled by the off-axis periphery penalty — the SAME signal the discrete confirm_only gate reads, so the
    // continuous common-mode and the discrete gate share one motion measure (matches chair_concept). A still OR
    // well-centred mask ⇒ ~0 common-mode (full authority); moving AND peripheral ⇒ large common-mode (confirm only).
    const float mot_mag     = motion_magnitude(inst);
    const float mot_periph  = periphery_penalty(inst);
    const float mot_pos_var = std::pow(cfg_.motion_cm_pos_gain  * mot_mag, 2.0f) * mot_periph;   // m²  (cx,cy)
    const float mot_size_var= std::pow(cfg_.motion_cm_size_gain * mot_mag, 2.0f) * mot_periph;   // m²  (w,h,H)
    const float mot_yaw_var = std::pow(cfg_.motion_cm_yaw_gain  * mot_mag, 2.0f) * mot_periph;   // rad² (yaw)

    // Gate → CONFIRMATION-ONLY (predict only; age Σ, hold the geometry mean; association ran upstream). Two causes:
    //  (1) TRUNCATION: a mask clipped by the image border has a chopped silhouette that biases the fit.
    //  (2) "BE-STILL-TO-UPDATE" (confirm_only): the robot is MOVING and the mask is off-centre — a moving frame's
    //      mask is a shared smear whose centroid is unreliable, so it may only CONFIRM the fridge, not move/reshape
    //      it (a well-centred mask is trusted even while moving — the exception). The continuous common-mode above
    //      is the graceful backstop; this discrete gate concentrates geometry updates at stillness (matches chair).
    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac or confirm_only(inst);

    // Pose-chain covariance at the hood centre (cx,cy) — the per-frame SHARED localization error. Fed
    // into the belief as part of the common-mode so the frame's information saturates (calibrated σ).
    // Computed once here (before the update) and reused for the published RT cov below.
    compute_chain_cov(inst);

    // ── Rogue-mask yaw instrumentation (NO effect on the fit) ──────────────────────────────────────
    // Snapshot yaw at cycle entry and compute OBLIQUITY — the grazing-view covariate the truncation gate does
    // not measure. A near-horizontal camera→hoodtop ray foreshortens the top face, so its 2D footprint (hence
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
        const Eigen::Vector3d ray(s0b.cx - cam.x(), s0b.cy - cam.y(), s0b.H - cam.z());   // camera→hoodtop centre
        const double rn = ray.norm();
        inst.dbg_obliquity_cos = (rn > 1e-6) ? static_cast<float>(std::abs(ray.z()) / rn) : 1.0f;
    }

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated / rejected cycle took no measurement)
    if (gated)
        inst.ai2_belief.predict();   // Σ inflates, mean unchanged
    else
    {
        HoodFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        // Robust observed cloud TOP (z-p97): the top-no-float factor caps the box top here so a floor-anchored H
        // cannot ratchet ABOVE the mask. p97 (not max) ignores a handful of stray high points.
        if (not frame.points.empty())
        {
            std::vector<float> zt; zt.reserve(frame.points.size());
            for (const auto& p : frame.points) zt.push_back(p.z());
            const std::size_t kt = static_cast<std::size_t>(0.97f * (zt.size() - 1));
            std::nth_element(zt.begin(), zt.begin() + kt, zt.end());
            frame.z_top_obs = zt[kt];
            // Down-weight points ABOVE the robust top: they are almost surely mask OVER-SEGMENTATION (wall / cabinet
            // above the fridge) and would otherwise ratchet the floor-anchored box top up to claim them (covering a
            // point is "free"). Grow their measurement variance with height above the top (R += (σ_per_m·above)²) so
            // they fade from the fit — a covariance keyed on the covariate, not a hard z-clip. The two-sided
            // top-anchor then pins H to the real top unopposed.
            if (cfg_.ai2_top_overseg_sigma_per_m > 0.0f)
                for (std::size_t i = 0; i < frame.points.size(); ++i)
                {
                    const float above = frame.points[i].z() - frame.z_top_obs;
                    if (above > 0.0f)
                    {
                        const float s = cfg_.ai2_top_overseg_sigma_per_m * above;
                        frame.R[i] += s * s;
                    }
                }
        }
        // Ray geometry (needed by the footprint residual + the anisotropic R later).
        if (cfg_.footprint_residual and have_cam)
        { frame.cam_origin = cam_origin_room; frame.has_rays = true; }
        // Camera-frame azimuth per point (about the optical axis; ZED frame x-right, y-depth) for the N=7 depth-
        // tilt STATE — computed from the TRUE extrinsic, NOT the belief's drifting hood-centre, so a persistent
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
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var + mot_pos_var;   // pose-chain + range + EGO-MOTION
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var + mot_pos_var;
        // Obliquity yaw cap: at an edge-on (grazing) view the hoodtop cloud is ~1-D along the near edge, so yaw
        // is barely observable and the per-point GN snaps between the box's symmetric orientations (r_π / w↔h —
        // the CSV flips). Grow the SHARED yaw variance as the view grazes (|cos(incidence)|→0 ⇒ 1/cos→∞), so a
        // grazing frame confirms the hood but cannot rotate it — the same continuous-covariance form as the range
        // term, keyed on view angle instead of distance. Validated live 2026-07-11 → always on.
        //
        // THIS IS THE LIVE YAW CAP on the shipped (FootprintResidual == false) branch below — a tuned coefficient,
        // deliberately kept. Its intended derived replacement, HoodBelief::tilt_yaw_common_mode() (a2′), is WIP and
        // NOT wired here: the fitter never calls it, and its self_test drift check is disabled. So on the
        // FootprintResidual == true branch there is currently NO obliquity/tilt yaw cap wired in (chain_cov_yaw
        // carries only the white range term) — that branch is experimental. Do not delete kObliquityYawGain until
        // tilt_yaw_common_mode is called here and validated. See hood_belief.cpp::tilt_yaw_common_mode.
        constexpr float kObliquityYawGain = 0.05f;   // ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing
        const float oblq_cos          = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);
        const float obliquity_yaw_std = kObliquityYawGain * (1.0f / oblq_cos - 1.0f);   // 0 at top-down
        // FootprintResidual on ⇒ the depth tilt is an ESTIMATED N=7 state (no per-frame yaw cap — that ratcheted
        // and needed a fragile sweet-spot); chain_cov_yaw carries only the WHITE range term. Else the legacy tuned
        // obliquity+range form. (The tilt STATE replaces both the cap and the obliquity/range yaw gains.)
        frame.chain_cov_yaw = mot_yaw_var + (cfg_.footprint_residual
            ? range_yaw_var
            : range_yaw_var + obliquity_yaw_std * obliquity_yaw_std);   // range + grazing-view cap + EGO-MOTION
        frame.chain_cov_size = range_size_var + mot_size_var;      // range freezes afar + EGO-MOTION freezes while moving
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
        // in). Mirror the per-point obliquity cap onto moment_extra_var so an edge-on frame confirms the hood but
        // cannot rotate/reshape it via the moment. Reuses oblq_cos computed above for chain_cov_yaw. 0 = OFF.
        const float moment_oblq_std   = cfg_.obliquity_moment_gain * (1.0f / oblq_cos - 1.0f);
        frame.moment_extra_var = std::max(0.0f, inst.last_motion_var)
                               + moment_range_std * moment_range_std
                               + moment_motion_std * moment_motion_std
                               + moment_oblq_std * moment_oblq_std;
        // YOLO-independent LiDAR range channel: stage returns landing on the legs/rim. No-op if precision==0
        // or no fresh sweep. The shared factor (accumulate_lidar_rays<6> in HoodBelief::accumulate_extra)
        // sphere-traces this belief's own SDF, so the same call the bottle uses drops in unchanged.
        lidar_channel_.feed(inst, frame);

        // WALL-FLUSH factor: give the interior (picks the BACK face) to the belief and stage the nearest room
        // wall to the fridge's BACK-face centre. Inert (ok=false) unless a room polygon was pushed this session.
        if (have_room_geometry_)
        {
            inst.ai2_belief.set_room_interior(room_interior_);
            const auto& bs_now = inst.ai2_belief.state();
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
        }

        // Stage 1 (PRECISION_AS_INFORMATION.md): replace the scalar per-point R with the anisotropic deprojection
        // noise projected on the SDF normal. Ego-motion variance is preserved as an isotropic floor (Stage 2 will
        // subsume it into the nuisance Jacobian). No-op unless the flag is on and we know the camera origin — then
        // a grazing view's yaw-carrying points get huge R → the per-point GN cannot rotate a converged hood,
        // WITHOUT the obliquity/range yaw gains above (which this is designed to make redundant).
        // (the footprint residual runs inside ai2_belief.update via accumulate_footprint; its tilt→yaw common-mode
        //  was folded into frame.chain_cov_yaw above so the engine Woodbury caps the total yaw information)

        // Divergence safety net (mirrors bottle): snapshot state+Σ, run the update, and if the centre teleports
        // beyond a physical bound in one frame (corrupted mask cloud / one-sided LiDAR runaway → the cx=−200m
        // event) REJECT it — restore the snapshot, widen Σ via a predict so the next good frame re-associates,
        // and accrue frames_diverged. A non-finite state is treated the same. 0 disables. NOT a magic gate: a
        // static hood cannot physically move max_step_m in one frame, so such a step is definitionally spurious.
        const HoodBelief pre_belief = inst.ai2_belief;   // value copy (state + Σ + prior + flip_evidence)
        const auto&       ps         = pre_belief.state();
        energy = inst.ai2_belief.update(frame);
        const auto& ns = inst.ai2_belief.state();
        // Full-state jump: centre (cx,cy,H) AND EXTENT (w,h). A static hood can't grow/shrink its extent by
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
            // sustained rise = the hood moved shows as surprise before the baseline accepts it); surprise = the
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

            // "IS THIS REALLY A FRIDGE?" plausibility is now judged EVERY cycle from the CURRENT fitted shape in
            // the worker's singleton/existence step (specificworker_lifecycle.cpp), NOT only here on an accepted
            // mask-update. Reason: a mis-detection that has diverged to a cabinet shape (w≫h, short) then coasts
            // out-of-FoV would otherwise FREEZE its birth-time positive evidence and stay immortal. Judging the
            // shape per-cycle lets it keep losing plausibility so the singleton decays it. (fridge_plausibility is
            // shape-only — aspect·size·height, FE unused — so it needs no fresh mask.)
        }
        // Near-square yaw disambiguation: sequential Bayesian comparison of the two orientation modes
        // (current vs the w↔h swap ≡ 90° rotation). Owns the GENUINE mode flip so the per-frame MAP no
        // longer SNAPS 90° on extent noise; the reported yaw uncertainty (yaw_marginal_var) stays honest
        // until an orbit resolves it. See HOOD.md.
        // Ego-motion reliability: a moving frame's mask must barely vote on the discrete w↔h mode (split/degraded
        // masks during motion were flipping it). Continuous down-weight, not a gate. Static (dotd≈0) → weight 1.
        const float mref = std::max(1e-3f, cfg_.orientation_motion_ref);
        const float dotd = std::abs(inst.last_motion_dotd);
        const float mode_evidence_weight = 1.0f / (1.0f + (dotd / mref) * (dotd / mref));
        // The quotient chart makes the w↔h mode a continuous, single-valued coordinate — the discrete flip
        // accumulator is unrepresentable and unnecessary (skipped). Legacy path keeps resolve_orientation.
        if (not rc::HoodBeliefState::use_quotient
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

        // Appearance FRONT (door) resolver: project the freshly-updated box into the live ZED RGB and fold the
        // door-ness cue into the belief's discrete door-mode accumulator. Runs on an ACCEPTED update only (the
        // belief yaw/w/h it may adopt is propagated by the model write-back below). Reuses the ego-motion weight.
        resolve_front_from_rgb(inst, mode_evidence_weight);
    }

    // Write the belief back into the legacy HoodState so all downstream publish/viewer/RT code is
    // unchanged. The vertical span comes from the belief's top plus the declared extent — see HoodState.
    const auto& bs = inst.ai2_belief.state();
    HoodState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.z_top = bs.H; ms.extent = cfg_.vertical_extent_m;
    ms.w = bs.w; ms.h = bs.h; ms.yaw = bs.yaw;
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
    inst.dbg_gate_fresh = true;   // computed from THIS frame's mask — see hood_instance.h

    log_ai2_csv(inst, npts, R, gated, energy);

    // Periodic shape model-selection (round vs square) on the accumulated cloud → inst.subtype.
    evaluate_shape(inst);
    return energy;
}

// STRIPPED for hood_concept: a hood has a single fixed CLASS (a solid box) — there is no
// round/square shape hypothesis to select. The subtype is the constant class discriminator "hood"
// (set once at birth + republished by the scene-graph). Kept as a no-op so the call site / header are
// unchanged. (table_concept's round-vs-square model-evidence test lived here; not applicable to one box.)
void HoodFitter::evaluate_shape(HoodInstance& inst)
{
    inst.subtype = "hood";
}

// Append one AI2 belief row (state + Σ-diag std + mask R/bias/trunc + mode evidence + LiDAR diag) to the CSV.
void HoodFitter::log_ai2_csv(const HoodInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        ai2_csv_.imbue(std::locale::classic());   // ★Qt imbues the global locale, which inserts THOUSANDS SEPARATORS
                                            // into integers (pkt_ts 1785763853131 -> "1,785,763,853,131"), splitting
                                            // one CSV field into five and making the whole log unparseable by column.
                                            // Pin "C" so the log is machine-readable regardless of the UI locale.
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,pkt_fid,pkt_ts,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,motion_dotd,trunc_frac,range,"
                 << "cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw,"
                 // door-mode resolver state. These REPLACE flip_ev/p_alt, which were flip_evidence() /
                 // mode_posterior() — hardcoded `return 0.0f` stubs left from table_concept's w↔h mode machinery
                 // that this asymmetric-box model does not use. They logged 0 for every row of every run, so the
                 // ONE thing that can resolve yaw on a square footprint was completely uninstrumented.
                 << "std_yaw_within,front_mode,front_conf,cue_conf,cue_bearing,wall_resp,wall_gap,wall_lambda,"
                 "wall_par_lambda,wall_misalign_deg,"
                 << "lidar_rays,lidar_raw,lidar_bpearl,lidar_resid_m,lidar_meanz,lidar_topz,lidar_floorz,lidar_cov_ang,"
                 << "dyaw_points,dyaw_moment,dyaw_flip,obliquity_cos,completeness,moment_aniso,moment_r_yaw,"
                 << "mom_major,mom_minor,mom_phi,mom_pts,"   // RAW footprint statistic (basin diagnosis)   // rogue-mask diag
                 << "ex_L,ex_p,ex_locc,ex_lfree,ex_lfree_eff,ex_ln,ex_socc,ex_sfree,ex_sfree_eff,ex_sndet,ex_streak,"
                 << "ex_pdetect,ex_central,ex_verify,ex_wantsverify,"
                 << "omega_w,omega_yaw,tau_w,tau_yaw,"   // HISTORY: inferred log-volatility + retention (frames)   // existence-removal + verification-gate diag
                 // DETECTOR-ENVELOPE CALIBRATION. min_fill/max_fill/soft are an admitted PRIOR, never measured,
                 // and every stand-off in the fleet now inherits them. These are the columns the offline fit
                 // needs: (roi_fill, det_alive) is the (framing, hit) pair, ex_p above weights the trial so a
                 // miss caused by the object not being there is not charged to the envelope, and the two axes
                 // let the fit control for grazing views — see roi_fill_h/roi_fill_v in hood_instance.h
                 // for why the max alone biases the result.
                 << "roi_fill,roi_fill_h,roi_fill_v,roi_valid,det_alive,frames_since_det,"
                 // What the NBV actually proposed (lags the state columns by one cycle; see the instance).
                 // nbv_vfov=0 ⇒ the sensor model was incomplete and NO proposal was made that cycle.
                 << "nbv_standoff,nbv_tx,nbv_ty,nbv_pdetect,nbv_vfov,nbv_gain,ricoh_det,ricoh_attn\n";
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
             << sd(5) << ','
             // front_mode/front_conf = the belief's adopted door quarter-turn + its posterior peakedness;
             // cue_conf/cue_bearing = what detect_front actually returned THIS frame (0 ⇒ no cue was emitted:
             // no RGB frame, <2 faces qualified, or the margin fell under FrontMinConfidence).
             << inst.ai2_belief.front_mode() << ',' << inst.ai2_belief.front_confidence() << ','
             << inst.last_front_conf << ',' << inst.last_front_bearing << ','
             << inst.ai2_belief.last_wall_resp() << ','   // mean per-point WALL responsibility (→1 ⇒ it IS a wall)
             // ★THE TWO NUMBERS THAT DECIDE THE "BITES THE WALL" QUESTION, and the code has been computing
             // and discarding them all along. wall_resp is the per-point MIXTURE responsibility — it says the
             // belief knows a wall is there, and says nothing about where the BOX ended up. The geometry is:
             //   wall_gap    = (back_centre − wall.p)·(inward normal): >0 the back face is inside the room,
             //                 0 flush, <0 the back face has CROSSED the wall.
             //   wall_lambda = the flush factor's APPLIED precision. 0 means the prior is inert this cycle
             //                 (no wall staged, or the flush weight has decayed to nothing) — in which case
             //                 nothing is holding the back face and the gap is merely where the points fell.
             // Reported symptom is a bite of ~HALF THE DEPTH (0.24 m of 0.485), and half-depth is exactly the
             // offset between the box CENTRE and its back face — so gap ≈ 0 with the box still overlapping the
             // wall would mean the drawn/published geometry is offset by half a depth, while gap ≈ −0.24 with
             // lambda ≈ 0 would mean the flush prior never fired. The two have nothing in common as fixes.
             << inst.ai2_belief.last_wall_gap() << ','
             << inst.ai2_belief.last_wall_lambda() << ','
             << inst.ai2_belief.last_wall_par_lambda() << ','
             << 57.2958f * inst.ai2_belief.last_wall_misalign() << ','   // degrees off the wall
             << inst.dbg_lidar_rays << ',' << inst.dbg_lidar_raw << ',' << inst.dbg_lidar_bpearl_rays << ',' << inst.dbg_lidar_resid_m << ','
             << inst.dbg_lidar_meanz_m << ',' << inst.dbg_lidar_topz_m << ',' << inst.dbg_lidar_floorz_m << ','
             << inst.dbg_lidar_cov_ang << ','
             << inst.dbg_dyaw_points << ',' << inst.dbg_dyaw_moment << ',' << inst.dbg_dyaw_flip << ','
             << inst.dbg_obliquity_cos << ',' << inst.dbg_completeness << ','
             << inst.ai2_belief.dbg_moment_aniso() << ',' << inst.ai2_belief.dbg_moment_r_yaw() << ','   // rogue-mask diag
             << inst.ai2_belief.dbg_moment_ext_major() << ',' << inst.ai2_belief.dbg_moment_ext_minor() << ','
             << inst.ai2_belief.dbg_moment_phi() << ',' << inst.ai2_belief.dbg_moment_pts() << ','
             << inst.existence.logodds() << ',' << inst.existence.p_exists() << ','
             << inst.dbg_ex_lidar_occ << ',' << inst.dbg_ex_lidar_free << ',' << inst.dbg_ex_lidar_free_eff << ',' << inst.dbg_ex_lidar_n << ','
             << inst.dbg_ex_sil_occ << ',' << inst.dbg_ex_sil_free << ',' << inst.dbg_ex_sil_free_eff << ',' << inst.dbg_ex_sil_ndet << ','
             << inst.existence_debounce.streak << ','
             << inst.dbg_ex_pdetect << ',' << inst.dbg_ex_central << ','
             << inst.verify_surprise << ',' << (inst.wants_verification ? 1 : 0) << ','
             << inst.ai2_belief.log_volatility()(3) << ',' << inst.ai2_belief.log_volatility()(5) << ','
             << inst.ai2_belief.retention_frames()(3) << ',' << inst.ai2_belief.retention_frames()(5) << ','   // existence-removal + verification-gate diag
             << inst.roi_fill << ',' << inst.roi_fill_h << ',' << inst.roi_fill_v << ','
             << (inst.roi_valid ? 1 : 0) << ',' << (inst.detection_alive ? 1 : 0) << ','
             << inst.frames_since_detection << ','
             << inst.dbg_nbv_standoff << ',' << inst.dbg_nbv_target_x << ',' << inst.dbg_nbv_target_y << ','
             << inst.dbg_nbv_pdetect << ',' << inst.dbg_nbv_vfov << ','
             << inst.dbg_nbv_gain
             // Peripheral (ricoh-360) attention: unassigned 360 detections of this label this
             // cycle. It was computed and shown nowhere — the whole peripheral channel was
             // invisible in every agent's log, which is why 'is it producing anything?' could
             // not be answered offline for cabinet/hood, the two it exists for.
             << ',' << ricoh_dets_ << ',' << ricoh_attention_ << '\n';
    ai2_csv_.flush();
}

// ─── Factory helpers ──────────────────────────────────────────────────────────────────────────────

// Build the HoodModel params from config (the SDF split band for candidate/residual classification).
HoodModelParams HoodFitter::make_model_params() const
{
    HoodModelParams p;
    p.sigma_obs = cfg_.sigma_obs;   // top/leg SDF split band for the candidate/residual classification
    return p;
}

// ─── One-shot fitter self-check: motion_magnitude robustness + the "be-still-to-update" confirm_only gate ─────
// Pure: builds a fitter with a LOCAL default config and a null graph (update_ego_motion is not called, so no DSR is
// needed) and drives ego speeds directly. Mirrors chair_concept's ported policy 1:1.
bool HoodFitter::self_test()
{
    bool all = true;
    HoodConfig cfg;   // defaults: AI2MotionConfirmOnly=true, still-levels, ang-lever=2.0, center-radius=0.35
    HoodFitter f(nullptr, nullptr, cfg, nullptr, nullptr);

    // (1) motion_magnitude returns the ROBOT-TWIST term when the producer never populated motion_dotd (=0), and
    //     the |motion_dotd| term when THAT dominates. Producer-independence is the whole point of the max().
    HoodInstance inst;
    inst.last_motion_dotd = 0.0f;                                  // producer gave nothing
    f.ego_lin_mps_ = 0.30f; f.ego_ang_radps_ = 0.0f;
    const float mm_lin = f.motion_magnitude(inst);                 // = ego_lin
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.10f;
    const float mm_ang = f.motion_magnitude(inst);                 // = ang_lever · ego_ang
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.0f; inst.last_motion_dotd = 0.50f;
    const float mm_dotd = f.motion_magnitude(inst);                // = |motion_dotd|
    const bool ok_mm = std::abs(mm_lin  - 0.30f) < 1e-4f
                   and std::abs(mm_ang  - cfg.ai2_ang_lever_m * 0.10f) < 1e-4f
                   and std::abs(mm_dotd - 0.50f) < 1e-4f;
    std::print("HoodFitter::self_test [motion_magnitude]   {}  lin={:.3f} ang={:.3f} dotd={:.3f}\n",
               ok_mm ? "PASS" : "FAIL", mm_lin, mm_ang, mm_dotd);
    all = all and ok_mm;

    // (2) confirm_only ("only update if the robot is still"): MOVING+off-centre ⇒ true; STILL ⇒ false;
    //     MOVING+well-centred ⇒ false (the trusted-centre exception).
    HoodInstance ci; ci.last_motion_dotd = 0.0f;
    f.ego_lin_mps_ = 0.30f; f.ego_ang_radps_ = 0.0f; ci.last_centroid_radius = 0.60f;   // moving & off-centre
    const bool co_move_off = f.confirm_only(ci);
    f.ego_lin_mps_ = 0.0f;  f.ego_ang_radps_ = 0.0f; ci.last_motion_dotd = 0.0f;         // still
    const bool co_still = f.confirm_only(ci);
    f.ego_lin_mps_ = 0.30f; ci.last_centroid_radius = 0.20f;                              // moving & well-centred
    const bool co_move_centre = f.confirm_only(ci);
    const bool ok_co = co_move_off and not co_still and not co_move_centre;
    std::print("HoodFitter::self_test [confirm_only]       {}  move_off={} still={} move_centre={}\n",
               ok_co ? "PASS" : "FAIL", co_move_off, co_still, co_move_centre);
    all = all and ok_co;

    // (3) The gate's EFFECT: confirm_only ⇒ predict-only ⇒ the geometry MEAN is HELD, whereas a still frame's
    //     update MOVES it toward the cloud. This is exactly run_inference's branch: if (gate) predict(); else update().
    HoodBeliefParams P;
    const HoodBeliefState s0{0.0f, 0.0f, 1.70f, 0.60f, 0.60f, 0.0f};
    std::mt19937 rng(777); std::normal_distribution<float> nz(0.0f, 0.01f);
    HoodFrame fr;                                          // a box cloud offset from the seed by (0.15,-0.10)
    for (int i = 0; i < 1200; ++i)
    {
        const int fc = i & 3; const float u = (i % 60) / 60.0f - 0.5f, z = (i % 100) / 100.0f * 1.7f;
        float lx, ly;
        if      (fc == 0) { lx =  0.30f; ly = u * 0.60f; }
        else if (fc == 1) { lx = -0.30f; ly = u * 0.60f; }
        else if (fc == 2) { ly =  0.30f; lx = u * 0.60f; }
        else              { ly = -0.30f; lx = u * 0.60f; }
        fr.points.push_back({0.15f + lx + nz(rng), -0.10f + ly + nz(rng), z + nz(rng)});
    }
    fr.R.assign(fr.points.size(), P.sigma_base_m * P.sigma_base_m);
    HoodBelief bpred(s0, P);
    bpred.predict();                                              // confirm-only branch → mean unchanged
    const auto mp = bpred.state();
    const float d_pred = std::hypot(mp.cx - s0.cx, mp.cy - s0.cy);
    HoodBelief bupd(s0, P);
    for (int it = 0; it < 12; ++it) bupd.update(fr);              // still branch → mean tracks the cloud
    const auto mu = bupd.state();
    const float d_upd = std::hypot(mu.cx - s0.cx, mu.cy - s0.cy);
    const bool ok_branch = d_pred < 1e-4f and d_upd > 0.05f;
    std::print("HoodFitter::self_test [confirm-only hold]  {}  |dPredict|={:.4f} |dUpdate|={:.4f}\n",
               ok_branch ? "PASS" : "FAIL", d_pred, d_upd);
    all = all and ok_branch;

    // (4) Depth observability, the assumption the cold-start snap rests on: a footprint moment from ONE
    //     viewpoint CANNOT resolve depth. One camera pose sees a single vertical face, whose footprint
    //     projects to a line, so ext_minor measures mask noise (~0.03) and stays under the band — which is why
    //     run_inference does h = (ext_minor > depth_obs_band) ? ext_minor : prior and lands on the prior.
    //     The union of two perpendicular faces is what it would take to clear the band. This is also why the
    //     probation burst is NOT unioned in: it never contains a second face at BirthFrames=8, so anything
    //     that clears the band there is pose-error blur, not depth (see the ★ note in run_inference).
    {
        std::mt19937 rng(4242);
        std::uniform_real_distribution<float> U01(0.0f, 1.0f);
        std::normal_distribution<float> jitter(0.0f, 0.01f);            // mask/de-projection noise
        const float H_gt = 1.85f, W_gt = 0.70f, D_gt = 0.60f, yaw_gt = 0.40f;
        const float cg = std::cos(yaw_gt), sg = std::sin(yaw_gt);
        const auto to_room = [&](float lx, float ly, float z) -> Eigen::Vector3f
        { return {0.20f + cg * lx - sg * ly, -0.30f + sg * lx + cg * ly, z}; };
        // The +y face: a line in footprint, spanning the WIDTH only.
        std::vector<Eigen::Vector3f> face_y, face_x;
        for (int i = 0; i < 600; ++i)
            face_y.push_back(to_room((U01(rng) - 0.5f) * W_gt + jitter(rng),
                                     0.5f * D_gt + jitter(rng), U01(rng) * H_gt));
        // The +x face: perpendicular, spanning the DEPTH — only a second viewpoint ever sees this.
        for (int i = 0; i < 600; ++i)
            face_x.push_back(to_room(0.5f * W_gt + jitter(rng),
                                     (U01(rng) - 0.5f) * D_gt + jitter(rng), U01(rng) * H_gt));

        const float z_lo = 0.03f, z_hi = H_gt + 0.03f;
        const auto m_single = HoodBelief::footprint_moment(face_y, z_lo, z_hi);
        std::vector<Eigen::Vector3f> both = face_y;
        both.insert(both.end(), face_x.begin(), face_x.end());
        const auto m_union = HoodBelief::footprint_moment(both, z_lo, z_hi);

        const float band = cfg.ai2_depth_obs_band_m;
        const bool ok_depth = m_single.ok and m_union.ok
                          and m_single.ext_minor < band       // one face   → depth UNRESOLVED, h stays at prior
                          and m_union.ext_minor  > band       // two faces  → only then is depth observable
                          and std::abs(m_union.ext_major - m_single.ext_major) < 0.30f;  // width must AGREE:
                          // a true second face lifts only the minor axis. Blur stretches BOTH — that is how
                          // the live burst gave itself away (ext_major 0.669 -> 0.915 while claiming depth).
        std::print("HoodFitter::self_test [depth observ.]     {}  ext_minor: single={:.3f} union={:.3f}"
                   "  (band={:.3f})  ext_major: single={:.3f} union={:.3f}\n",
                   ok_depth ? "PASS" : "FAIL", m_single.ext_minor, m_union.ext_minor, band,
                   m_single.ext_major, m_union.ext_major);
        all = all and ok_depth;
    }

    return all;
}

}  // namespace rc
