/*
 * door_fitter.cpp — the active-inference fit core for door_concept (AI2 recursive-Laplace belief).
 */

#include "door_fitter.h"

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "door_support_bank.h"   // rc::support_bank:: adapter (SHARED bank)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <print>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rc {

DoorFitter::DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         DoorConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         DoorSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{}

DoorBeliefParams DoorFitter::make_belief_params() const
{
    DoorBeliefParams p;
    p.sigma_base_m       = cfg_.ai2_sigma_base_m;
    p.clutter_frac       = cfg_.ai2_clutter_frac;
    p.clutter_scale_m    = cfg_.ai2_clutter_scale_m;
    p.floor_z            = cfg_.ai2_floor_z;
    p.floor_std          = cfg_.ai2_floor_std;
    p.thickness          = cfg_.door_thickness_m;
    // s tracks localisation jitter (ai2_process_std_m); w,h are rigid size DOFs (ai2_process_std_size).
    p.process_std_s      = cfg_.ai2_process_std_m;
    p.process_std_w      = cfg_.ai2_process_std_size;
    p.process_std_h      = cfg_.ai2_process_std_size;
    p.common_mode_s_std  = cfg_.ai2_common_mode_pos_std;
    p.common_mode_wh_std = cfg_.ai2_common_mode_size_std;
    // Strong panel priors (w,h) + broad along-wall offset prior (s). tpl_{w,h} = the fixed non-drifting
    // template anchor mean (see DoorBelief::accumulate_extra).
    p.prior_s_std        = cfg_.door_prior_s_std;
    p.prior_w_std        = cfg_.door_prior_w_std;
    p.prior_h_std        = cfg_.door_prior_h_std;
    p.tpl_w              = cfg_.door_prior_w_m;
    p.tpl_h              = cfg_.door_prior_h_m;
    p.gn_iters           = cfg_.ai2_gn_iters;
    // ── Leaf articulation (openable door, M0) ──
    // The ONE place the leaf state is authored, so the M0 pin cannot drift. With Openable.Enabled false
    // phi is the literal 0.0f, which makes every geometry consumer reduce EXACTLY to the wall-plane
    // behaviour (see door_geometry.h). phi becomes a fitted DOF in M1; hinge/swing hypotheses are M2.
    p.leaf.phi   = cfg_.openable_enabled ? cfg_.openable_phi_init : 0.0f;
    p.leaf.hinge = (cfg_.openable_enabled and cfg_.openable_hinge_side == 1) ? door::HingeSide::Far
                                                                            : door::HingeSide::Near;
    p.leaf.swing = (cfg_.openable_enabled and cfg_.openable_swing_dir < 0.0f) ? -1.0f : +1.0f;
    return p;
}

// ─── The SINGLE authoring point for a door's geometry ────────────────────────────────────────────
// Refresh the cached aperture / leaf / leaf_pose from the belief, then write the room-frame read-back
// into DoorState. Every geometry consumer in the agent reads one of these; nothing reconstructs the
// panel rectangle for itself. Called from ensure_instance (pre-belief), the bearing seed, and the
// per-cycle write-back — the only three places a door's shape can change.
void DoorFitter::refresh_geometry(DoorInstance& inst)
{
    DoorState ms = inst.model.state();
    if (inst.ai2_initialized)
    {
        inst.aperture  = inst.ai2_belief.aperture();
        inst.leaf      = inst.ai2_belief.params().leaf;
        inst.leaf_pose = inst.ai2_belief.leaf_pose();
        ms.cz        = inst.ai2_belief.cz();
        ms.w         = inst.ai2_belief.width();
        ms.h         = inst.ai2_belief.height();
        ms.thickness = inst.ai2_belief.thickness();
        const Eigen::Vector2f ap = inst.ai2_belief.center_xy();
        ms.ap_cx = ap.x(); ms.ap_cy = ap.y(); ms.ap_yaw = inst.ai2_belief.yaw();
    }
    else
    {
        // Pre-belief (a node adopted from the graph, or a fresh birth before its first mask): the only
        // geometry we have is the published box, so derive the leaf pose from that. Identical result.
        // The wall frame is not resolved yet, so the aperture is seeded as the box itself (u = the box's
        // own heading, near edge at s = 0); run_inference re-authors it properly on the first fit.
        inst.leaf_pose = door::leaf_pose_from_box(ms.cx, ms.cy, ms.cz, ms.yaw, ms.w, ms.h, ms.thickness);
        inst.leaf      = {};
        inst.aperture  = {};
        inst.aperture.wall_u = inst.leaf_pose.ex;
        inst.aperture.wall_O = inst.leaf_pose.hinge_xy;
        inst.aperture.s = 0.0f; inst.aperture.w = ms.w; inst.aperture.h = ms.h;
        inst.aperture.floor_z = ms.cz; inst.aperture.thickness = ms.thickness;
        ms.ap_cx = ms.cx; ms.ap_cy = ms.cy; ms.ap_yaw = ms.yaw;
    }
    ms.cx  = inst.leaf_pose.centre_xy.x();
    ms.cy  = inst.leaf_pose.centre_xy.y();
    ms.yaw = inst.leaf_pose.yaw();
    ms.phi = inst.leaf.phi;
    inst.model.set_state(ms);
}

// Associate q (room frame) to the nearest room-polygon wall. The polygon has chamfered corners / door
// notches that split one physical wall into several short edges, so — like cabinet's build_wall_ref — the
// nearest edge is COLLINEAR-MERGED into its logical wall (walk both ways while neighbours stay within ~3°):
// near corner O = the merged run's start corner, along-wall unit u = (far−O)/|far−O|, len = the run length.
// s is measured along u from O; the belief clamps the panel to [0, len].
DoorFitter::WallFrame DoorFitter::nearest_wall(const Eigen::Vector2f& q) const
{
    WallFrame w;
    const std::size_t n = room_polygon_.size();
    if (n < 3)
        return w;
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f  ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-9f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        const float d2 = (q - (a + t * ab)).squaredNorm();
        if (d2 < best) { best = d2; best_i = i; }
    }
    if (best_i >= n)
        return w;

    // Collinear-merge the argmin edge with its neighbours (dir within ~3°) into one logical wall.
    constexpr float kColinCos = 0.99863f;   // cos(3°)
    const auto edge_dir = [&](std::size_t i)
    { return (room_polygon_[(i + 1) % n] - room_polygon_[i]).normalized(); };
    const Eigen::Vector2f d0 = edge_dir(best_i);
    std::size_t lo = best_i, hi = best_i;
    for (std::size_t step = 0; step + 1 < n; ++step)   // walk backward while collinear
    {
        const std::size_t prev = (lo + n - 1) % n;
        if (std::abs(edge_dir(prev).dot(d0)) < kColinCos) break;
        lo = prev;
    }
    for (std::size_t step = 0; step + 1 < n; ++step)   // walk forward while collinear
    {
        const std::size_t nxt = (hi + 1) % n;
        if (nxt == lo or std::abs(edge_dir(nxt).dot(d0)) < kColinCos) break;
        hi = nxt;
    }
    const Eigen::Vector2f a   = room_polygon_[lo];
    const Eigen::Vector2f far = room_polygon_[(hi + 1) % n];
    const Eigen::Vector2f ab  = far - a;
    const float len = ab.norm();
    if (len < 1e-6f)
        return w;
    w.O = a; w.u = ab / len; w.len = len; w.ok = true;
    return w;
}

void DoorFitter::seed_bearing_hypothesis(DoorInstance& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                          float nominal_range, float along_std, float across_std, float yaw_std)
{
    // Bearing-only birth does not fit the wall-anchored door model (a door is placed by wall association,
    // not by a peripheral ray). Kept as a dormant stub (cfg_.bearing_birth_enabled defaults OFF): seed a
    // broad panel on the ray so the scene-graph/viewer show the hypothesis, and flag it for the Orient
    // affordance. A subsequent depth mask replaces it with a wall-associated fit in run_inference.
    (void) along_std; (void) across_std; (void) yaw_std;
    DoorBeliefParams p = make_belief_params();
    p.wall_O = robot_xy;
    p.wall_u = {std::cos(azimuth), std::sin(azimuth)};
    p.prior_s_std = std::max(p.prior_s_std, along_std);   // range unknown → broad along the ray
    DoorBeliefState s0;
    s0.s = std::max(0.0f, nominal_range);   // along the ray from the robot
    s0.w = cfg_.door_prior_w_m;
    s0.h = cfg_.door_prior_h_m;
    inst.ai2_belief = DoorBelief(s0, p);
    inst.ai2_initialized       = true;
    inst.is_bearing_hypothesis = true;
    inst.hypothesis_azimuth    = azimuth;   // Orient affordance target yaw = the bearing to look toward
    // Write the room-frame read-back into the model so the scene-graph publish / viewer show it on the ray.
    refresh_geometry(inst);
}

void DoorFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = enabled and (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void DoorFitter::compute_chain_cov(DoorInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the door centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution,
    // pinned to the mask capture stamp.
    const auto& s = inst.model.state();
    const Mat::Vector3d centre(s.cx, s.cy, s.cz);
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

// ─── Instance lifecycle ──────────────────────────────────────────────────────

bool DoorFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    DoorState init_state;
    init_state.cx        = 0.0f;
    init_state.cy        = 0.0f;
    init_state.yaw       = 0.0f;
    init_state.w         = cfg_.door_prior_w_m;
    init_state.h         = cfg_.door_prior_h_m;
    init_state.thickness = cfg_.door_thickness_m;

    // Standard DSR geometry attrs → panel dims: width_m→w, depth_m→thickness, height_m→h.
    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.w         = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.thickness = v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.h         = v.value();

    // Read RT pose from room→door edge
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

    // Tracker birth seed: a freshly born node's room→door RT may not compose this cycle, so prefer the
    // detection XY the tracker handed us (consumed once) over a possibly-0,0 RT read.
    if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        birth_seeds_.erase(it);
    }

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF; replace any non-finite field
    // with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* name)
        {
            if (!std::isfinite(v))
            {
                std::print("door_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.cz, 0.0f, "cz");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.w, cfg_.door_prior_w_m, "w");
        fix(init_state.h, cfg_.door_prior_h_m, "h");
        fix(init_state.thickness, cfg_.door_thickness_m, "thickness");
    }

    DoorInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = DoorModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name(), "door");
    // Seed the cached geometry from the published box: the belief is not initialised yet, but observe()
    // needs a valid leaf_pose for its candidate/residual split on this very cycle.
    refresh_geometry(inst);

    instances_.emplace(node.id(), std::move(inst));
    std::print("door_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ─────────────────────────────────────────────────────────────

DoorFitter::DoorObservation DoorFitter::observe(DoorInstance& inst, const DSR::Node& node)
{
    DoorObservation observation;

    // Detection-aliveness ages every cycle; a fresh door mask below resets it to 0.
    if (inst.frames_since_detection < 1000000) ++inst.frames_since_detection;

    // Primary path: YOLO "masks" (room frame), masks-only. Classify-don't-destroy SDF split keeps
    // inliers as candidates and the rest as residuals that drive model expansion.
    const auto& masks_packet = mask_ingestor_->packet();
    if (masks_packet.valid && masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance = ONLY the §3.1 gated assignment. No greedy-nearest fallback: when this
        // instance has no association this frame (its door occluded / another instance won the slice), it
        // must FREEZE, not grab the nearest door's mask — that fallback teleported a far-born instance onto
        // a near door when its own door was momentarily undetected (→ merge, so the far door never
        // persisted). Freeze-on-no-association is the information-filter axiom; mirrors table_concept.
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this door on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
            inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
            inst.last_motion_var  = slice.motion_var;     // AI2 ego-motion / range / truncation channels
            inst.last_motion_dotd = slice.motion_dotd;
            inst.last_trunc_frac  = slice.trunc_frac;
            inst.last_range       = slice.range;
            inst.last_centroid_radius = slice.centroid_radius;   // image-centredness (moving-update exception)
            inst.last_depth_var   = slice.depth_var;             // 0=ZED, >0=ricoh LiDAR-depth → downweights the fit (added to R)
            // MINIMUM-HEIGHT evidence: the top of the observed support (room frame; floor = 0), accumulated
            // ONLY from views that actually saw the top. A border-clipped mask (trunc_frac → 1) reports a
            // top that is merely a lower bound, so it carries no height information — weight it out
            // continuously rather than gating, and never let it drag obs_top_z down. See DoorInstance.
            if (slice.has_depth and slice.bbox_max.allFinite())
            {
                const float top = slice.bbox_max.z();
                const float wgt = std::clamp(1.0f - slice.trunc_frac, 0.0f, 1.0f);
                inst.obs_top_last = top;
                if (wgt > 0.0f)
                {
                    constexpr float kEwma = 0.05f;   // smoothing over untruncated views (a measurement filter)
                    const float a = kEwma * wgt;
                    inst.obs_top_z    = std::isnan(inst.obs_top_z) ? top : (1.0f - a) * inst.obs_top_z + a * top;
                    inst.obs_top_conf = (1.0f - a) * inst.obs_top_conf + a;
                }
            }
            const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
            const std::size_t end = std::min(slice.support_end, masks_packet.support_points.size());

            std::vector<Eigen::Vector3f> candidate_pts;
            std::vector<Eigen::Vector3f> residual_pts;
            candidate_pts.reserve(end > begin ? end - begin : 0);
            residual_pts.reserve(end > begin ? end - begin : 0);

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
                // Same SDF the belief fits with (door_geometry.h), evaluated at the cached leaf pose —
                // there is no longer a second, independently-derived panel SDF that could disagree.
                const float sdf = door::leaf_sdf(inst.leaf_pose, p);
                if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
                    candidate_pts.push_back(p);
                else
                    residual_pts.push_back(p);
            }

            observation.has_fresh_data = true;
            observation.candidate_pts = std::move(candidate_pts);
            observation.residual_pts = std::move(residual_pts);

            if (!observation.candidate_pts.empty() || !observation.residual_pts.empty())
            {
                const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
                observation.explanation_ratio = total > 0.0f
                    ? static_cast<float>(observation.candidate_pts.size()) / total : 0.0f;

                inst.last_masks_frame_seen = masks_packet.frame_id;

                if (should_log(inst))
                    std::print("[{}] masks={} label='{}' conf={:.2f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                               inst.node_name, masks_packet.frame_id, slice.label, slice.confidence,
                               end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                               slice.centroid.x(), slice.centroid.y(), slice.centroid.z());
                return observation;
            }
        }
    }

    // Fallback: candidate/residual point attributes written directly on the node.
    int last_frame = -1;
    if (const auto v = G_->get_attrib_by_name<last_sensing_frame_att>(node); v.has_value())
        last_frame = v.value();

    observation.has_fresh_data = (last_frame > inst.last_frame_seen);
    if (not observation.has_fresh_data)
        return observation;

    inst.last_frame_seen = last_frame;
    observation.candidate_pts = mask_ingestor_->read_pts_attrib(node, "candidate_pts_att");
    observation.residual_pts  = mask_ingestor_->read_pts_attrib(node, "residual_pts_att");

    if (const auto v = G_->get_attrib_by_name<explanation_ratio_att>(node); v.has_value())
        observation.explanation_ratio = v.value();

    if (should_log(inst))
        std::print("[{}] ↓ frame={} cands={} resid={} expl={:.2f}\n",
                   inst.node_name, last_frame,
                   observation.candidate_pts.size(), observation.residual_pts.size(),
                   observation.explanation_ratio);
    return observation;
}

bool DoorFitter::should_log(const DoorInstance& inst) const
{
    const int period = std::max(1, cfg_.door_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── AI2 inference (shared recursive-Laplace belief; mirrors table run_inference) ───────────────
float DoorFitter::run_inference(DoorInstance& inst, const DoorObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy belief init: associate the door to the nearest room-polygon wall (fixes yaw / lateral / floor),
    // then seed the along-wall offset s from the mask centroid. w,h start at their strong template priors;
    // the belief works in the wall frame θ=[s,w,h].
    if (not inst.ai2_initialized)
    {
        // RE-ACQUISITION first: this node is a door that was removed and has now come back at the same place,
        // so it resumes the belief it had converged to (state + Σ) rather than cold-starting at the template
        // priors and re-fitting from scratch. See DoorFitter::note_reacquire.
        if (const auto rs = restore_seeds_.find(inst.node_id); rs != restore_seeds_.end())
        {
            inst.ai2_belief = rs->second;
            inst.ai2_initialized = true;
            restore_seeds_.erase(rs);
            std::print("door_concept: [{}] RE-ACQUIRED — resuming converged belief (s={:.2f} w={:.2f} h={:.2f})\n",
                       inst.node_name, inst.ai2_belief.state().s, inst.ai2_belief.state().w, inst.ai2_belief.state().h);
        }
    }
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        Eigen::Vector2f centroid_xy(m.cx, m.cy);
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v) { for (const auto& p : v) sum += p; };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            centroid_xy = (sum / static_cast<float>(npts)).head<2>();
        }
        DoorBeliefParams p = make_belief_params();
        DoorBeliefState s0;
        s0.w = cfg_.door_prior_w_m;
        s0.h = cfg_.door_prior_h_m;
        // Wall association: the belief's centre = O + (s + w/2)·u, so seed s so the panel centre lands on the
        // detection centroid. Off the wall (no polygon) → keep the identity frame and s = along-x of centroid.
        if (const WallFrame wf = nearest_wall(centroid_xy); wf.ok)
        {
            p.wall_O = wf.O; p.wall_u = wf.u; p.wall_len = wf.len;
            s0.s = std::clamp((centroid_xy - wf.O).dot(wf.u) - 0.5f * s0.w, 0.0f, std::max(0.0f, wf.len - s0.w));
        }
        else
            s0.s = centroid_xy.x() - 0.5f * s0.w;
        inst.ai2_belief = DoorBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
    {
        ingest_observation_support(inst, observation);
        // The glance paid off: a real depth mask arrived → this is no longer a bearing-only hypothesis. The
        // belief update below collapses the broad along-ray Σ to the observed position; the affordance
        // switches from Orient to the normal one next cycle.
        inst.is_bearing_hypothesis = false;
    }

    const auto now = std::chrono::steady_clock::now();

    if (not observation.has_fresh_data)   // stale: age (Σ grows on the agent clock) unless AI2AgeNominalDtS<=0 (freeze)
    {
        if (cfg_.ai2_age_nominal_dt_s > 0.0f and inst.last_belief_touch.time_since_epoch().count() != 0)
        {
            const float dt = std::chrono::duration<float>(now - inst.last_belief_touch).count();
            inst.ai2_belief.inflate_for_age(dt, cfg_.ai2_age_nominal_dt_s);
        }
        inst.last_belief_touch = now;
        inst.dbg_gate_fresh = false;   // no mask reached the fit ⇒ the gate flags below are STALE this cycle
        compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD last FE — an aged cycle took no measurement (no new energy)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting + ego-motion downweight (mirror table): a far / moving view widens the
    // along-wall common-mode so it can only CONFIRM the door's position, not slide it; continuous, no gate.
    const float range         = std::max(0.0f, inst.last_range);
    const float range_lat_var = (cfg_.ai2_range_noise_lat_per_m * range) * (cfg_.ai2_range_noise_lat_per_m * range);
    // depth_var (0 for a ZED slice, >0 for a ricoh LiDAR-depth slice) enters R in the SAME currency as motion_var:
    // a ricoh point's along-ray depth uncertainty inflates its measurement noise, so it barely moves the belief
    // mean (ZED drives geometry, ricoh only confirms). Mirrors table_concept's R = σ²+motion_var+depth_var+range.
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m + std::max(0.0f, inst.last_motion_var)
                    + std::max(0.0f, inst.last_depth_var) + range_lat_var;
    // The wall fixes yaw/lateral/floor, so there is no orientation to observe — the backrest-obliquity yaw
    // cap (chair) is gone. Only the along-wall offset s and the panel size w,h are inferred.
    inst.dbg_obliquity_cos = 1.0f;

    // "Be-still-to-update" invariant: a truncated view (gated) OR a MOVING robot may only CONFIRM the door, never
    // move/reshape it. A moving frame's mask is a shared smear whose centroid is unreliable — predict-only here
    // (mean held, Σ carries its one-step Q); the existence belief still confirms it (its mask reset stops vacate).
    const bool trunc_gated  = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac;
    const bool motion_gated = confirm_only(inst);
    const bool gated = trunc_gated or motion_gated;
    inst.dbg_gated = gated;   // the existence channel needs this verdict (removal requires an admissible frame)
    // …split by mechanism, and stamped as computed from THIS frame's mask — see DoorInstance.
    inst.dbg_trunc_gated = trunc_gated; inst.dbg_motion_gated = motion_gated; inst.dbg_gate_fresh = true;
    compute_chain_cov(inst);

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated cycle takes no measurement)
    if (gated)
        inst.ai2_belief.predict();
    else
    {
        DoorFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        // AIF "be-still-to-update" as CONTINUOUS PRECISION: a frame's authority to MOVE the mean is capped via the
        // per-frame common-mode (NOT per-point R), by a variance that grows with MOTION × OFF-AXIS position. Still
        // (motion→0) OR well-centred (periphery→0) ⇒ ~0 common-mode ⇒ full authority; moving AND peripheral ⇒ large
        // common-mode ⇒ the frame can only CONFIRM. No gate — "confirmation-only" is the precision→0 limit. The
        // periphery factor is why a centred mask stays trustworthy while moving (the user's exception, emergent).
        // The along-wall position uncertainty is the pose-chain XY variance projected onto u, plus range/motion.
        const float motion_mag  = motion_magnitude(inst);
        const float periph      = periphery_penalty(inst);
        const float mot_pos_var = std::pow(cfg_.motion_cm_pos_gain * motion_mag, 2.0f) * periph;   // m² (along-wall s)
        const Eigen::Vector2f u = inst.ai2_belief.params().wall_u;
        const float chain_s = u.x() * u.x() * inst.chain_cov_xx + u.y() * u.y() * inst.chain_cov_yy;   // proj onto u
        frame.chain_cov_s = chain_s + range_lat_var + mot_pos_var;
        inst.ai2_belief.update(frame);   // MAP mean + posterior Σ; its surface-only return is NOT the FE (below)
        // NOTE: refine_extent (coverage/extent likelihood) DISABLED — coverage without a free-space
        // counter-force is positive feedback: it inflated the footprint to cover contamination/neighbours
        // (seat → 2–5.8 m) and spawned phantom instances. Kept in the belief for reference; see Fable review.

        // FREE ENERGY = the clutter-INCLUSIVE mixture NLL, NOT the engine's surface-only return: a misfit point
        // routes to clutter (r_surface≈0) and contributes ≈0 to the surface energy, so a badly-fit door would
        // read F≈0 — blind to exactly the errors that matter (TABLE.md §3, "do NOT reintroduce"). mixture_nll
        // includes the clutter term so F RISES with misfit; it is the same quantity association/orientation use.
        energy = inst.ai2_belief.mixture_nll(frame.points, inst.ai2_belief.state(), R);

        // (No orientation resolution: the containing wall fixes the door's yaw — there is no front/back
        // 180° ambiguity to correct.)

        // FE-surprise attention (TABLE.md §9): baseline tracks DOWN fast (consolidate a better fit) / UP slow (a
        // sustained rise = the door moved surfaces as surprise before the baseline accepts it); surprise = the
        // smoothed positive gap F−baseline. Updated only on this accepted-measurement branch.
        if (inst.fe_baseline < 0.0f)
            inst.fe_baseline = energy;
        else
        {
            const float a = (energy < inst.fe_baseline) ? cfg_.ai2_fe_baseline_adapt_down
                                                        : cfg_.ai2_fe_baseline_adapt_up;
            inst.fe_baseline += a * (energy - inst.fe_baseline);
        }
        const float gap = std::max(0.0f, energy - inst.fe_baseline);
        inst.fe_surprise += cfg_.ai2_fe_surprise_smooth * (gap - inst.fe_surprise);

        // Clutter diagnostic: how much of the assigned mask the model can't explain (off-model points —
        // e.g. the table bleeding into a door mask, which drags the centroid). High + a position jump ⇒
        // contamination the clutter component didn't fully reject.
        inst.last_clutter_frac = inst.ai2_belief.clutter_fraction(frame.points, R);
    }
    inst.dbg_energy = energy;   // remember for the next gated/aged cycle to HOLD
    inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());   // HELD, see door_instance.h

    // Write belief → cached aperture/leaf geometry + room-frame DoorState (the single authoring point).
    const auto& bs = inst.ai2_belief.state();
    const Eigen::Vector2f c = inst.ai2_belief.center_xy();
    refresh_geometry(inst);
    const DoorState& ms = inst.model.state();

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;
    compute_projected_roi(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} clutter={:.0f}% R={:.4f} range={:.2f} trunc={:.2f}{} | FE={:.2f} base={:.2f} surprise={:.2f} | s={:.2f} w={:.2f} h={:.2f} @({:.2f},{:.2f}) ψ={:.2f}\n",
                   inst.node_name, npts, 100.0f * inst.last_clutter_frac, R, range, inst.last_trunc_frac, gated ? " GATED" : "",
                   energy, inst.fe_baseline, inst.fe_surprise,
                   bs.s, bs.w, bs.h, c.x(), c.y(), ms.yaw);

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

void DoorFitter::log_ai2_csv(const DoorInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        rc::diag::open_rotating(ai2_csv_, cfg_.ai2_csv_path);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,trunc_frac,range,clutter_frac,"
                 << "s,w,h,cx,cy,yaw,std_s,std_w,std_h,phi\n";   // phi APPENDED so the existing columns stay byte-identical
    }
    const auto& s = inst.ai2_belief.state();
    const Eigen::Vector2f c = inst.ai2_belief.center_xy();   // APERTURE centre (see DoorBelief) — unchanged by phi
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_trunc_frac << ',' << inst.last_range << ',' << inst.last_clutter_frac << ','
             << s.s << ',' << s.w << ',' << s.h << ','
             << c.x() << ',' << c.y() << ',' << inst.ai2_belief.yaw() << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << inst.leaf.phi << '\n';
    ai2_csv_.flush();
}

// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> DoorFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", pose_ts_ms);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", 0);
    if (not (rtb.has_value() and btz.has_value()))
        return std::nullopt;
    const auto to_mat4 = [](const Mat::RTMat& T)
    {
        Eigen::Matrix4d m;
        const auto& s = T.matrix();
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

void DoorFitter::update_ego_motion()
{
    const auto M = room_T_zed_matrix();   // camera→room (ts=0 = current robot pose)
    const auto now = std::chrono::steady_clock::now();
    if (not M.has_value())
    {
        have_prev_cam_ = false;           // pose chain unavailable → can't judge motion; reset baseline
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

// Combined ego-motion magnitude (m/s): the per-mask corruption speed OR'd with the robot's own measured speed
// (linear + a lever-arm conversion of angular), so it works whether or not the producer populated motion_dotd.
float DoorFitter::motion_magnitude(const DoorInstance& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear/distortion), → 1 at
// centroid radius ai2_periph_ref. This is the "well-centred masks stay trustworthy" lever, expressed continuously.
float DoorFitter::periphery_penalty(const DoorInstance& inst) const
{
    const float ref = std::max(1e-6f, cfg_.ai2_periph_ref);
    const float r = std::max(0.0f, inst.last_centroid_radius) / ref;
    return std::clamp(r * r, 0.0f, 1.0f);
}

// ─── Silhouette existence (pixel-level) ─────────────────────────────────────────────────────────

// PIXEL-LEVEL silhouette existence evidence — see DoorSilhouette / DoorInstance::existence.
//
// Projects the door PANEL FACE (a w×h rectangle in the wall plane) into the ZED and, over the predicted
// samples, splits into: lit by a "door" mask (occupancy), lit by a non-door mask (occlusion ⇒ excluded from
// the detectable footprint), or lit by nothing (ABSENCE — the "gone" signal that fires even on a frame where
// YOLO produced no door mask at all). A sample that does not land inside the real image is NOT detectable, so
// n_detectable==0 means "the door was never looked at this frame" and rc::exist HOLDs.
//
// This is what makes turning the robot around harmless: the old scheme asked a bearing-free range term how
// detectable the door was and got ≈0.4 for a door squarely BEHIND the camera, then charged absence evidence
// against it every frame until it died. Here the frustum test is the detectability.
DoorSilhouette DoorFitter::compute_silhouette_existence(const DoorInstance& inst)
{
    DoorSilhouette out;
    if (not inner_eigen_ or not inst.ai2_initialized)
        return out;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return out;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return out;
    }
    const auto Mopt = room_T_zed_matrix();   // camera→room, current pose
    if (not Mopt.has_value())
        return out;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();

    const float W    = static_cast<float>(camera_api_->get_width());
    const float Himg = static_cast<float>(camera_api_->get_height());
    if (W <= 0.f or Himg <= 0.f)
        return out;

    if (not mask_ingestor_)
        return out;
    const auto& pkt = mask_ingestor_->packet();
    if (not pkt.valid or pkt.mask_pixels.empty())
        return out;

    // Hashed pixel-cell coverage of the current YOLO foreground, split door (occupancy) vs other (occluder).
    // A CELL-px cell absorbs mask-boundary jitter and makes membership O(1). Key packs the two cell indices.
    // CELL is a discretization constant (quadrature), not a belief gate — no config key.
    constexpr float CELL = 6.0f;
    const auto key = [&](float col, float row) -> std::int64_t
    {
        const std::int64_t kx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t ky = static_cast<std::int64_t>(std::floor(row / CELL));
        return (kx << 32) ^ (ky & 0xffffffffLL);
    };
    std::unordered_set<std::int64_t> door_cells, occluder_cells;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "door") ? door_cells : occluder_cells;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }

    const auto& s = inst.model.state();
    const float hw = inst.leaf_pose.half_w;

    // Camera position (room frame) for the GEOMETRIC occlusion tests below.
    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)),
                            static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const float occ_margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);

    // Classify ONE LEAF-face sample (local x hinge→free edge, z absolute): project it, then vote.
    //
    // ★ The sample comes from door::leaf_point at the instance's CURRENT leaf pose. It used to be spelled
    // out here as a rectangle in the WALL PLANE, which meant that the moment a door actually opened, its
    // predicted silhouette landed where the leaf no longer was: every sample came back unlit, e_free
    // spiked, and the existence channel deleted the door BECAUSE it opened. With phi pinned at 0 (M0) this
    // yields the identical samples; when phi becomes a fitted DOF in M1 it follows the leaf for free.
    std::unordered_set<std::int64_t> covered_cells;   // distinct cells the DETECTABLE silhouette occupies
    double range_sum = 0.0;
    const auto classify = [&](float lx, float lz)
    {
        ++out.n_total;                                                  // one sample of the WHOLE panel
        const Eigen::Vector3f Ps = door::leaf_point(inst.leaf_pose, lx, 0.0f, lz);   // face at mid-thickness
        const Eigen::Vector4d Pr(Ps.x(), Ps.y(), Ps.z(), 1.0);
        const Eigen::Vector4d Pc = zed_T_room * Pr;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        if (Y <= 0.20) return;                                          // behind / at the near clip
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(X, Y, Z));
        const float col = static_cast<float>(uv.x()), row = static_cast<float>(uv.y());
        if (col < 0.f or col >= W or row < 0.f or row >= Himg) return;  // out of the REAL frustum ⇒ not detectable
        const std::int64_t k = key(col, row);
        if (occluder_cells.contains(k) and not door_cells.contains(k))
        { ++out.n_occluded; return; }                                   // a nearer MASKED object hides it ⇒ no vote
        // GEOMETRIC occluders, which carry no YOLO mask and so are invisible to the cell test above:
        // (a) a room WALL crossing this sightline (the robot is around a corner / the door is in another
        //     room's wall) — the structural case, and the one that would otherwise read as clean absence;
        // (b) another door instance's panel standing in front of this sample.
        // Per SAMPLE, not per instance: a partially-hidden door keeps the visible part of its footprint and
        // votes with it, instead of the old all-or-nothing `continue` that froze a phantom indefinitely.
        const Eigen::Vector3f Pw(static_cast<float>(Pr.x()), static_cast<float>(Pr.y()), static_cast<float>(Pr.z()));
        if (rc::occlusion::walls_block(Oxy, {Pw.x(), Pw.y()}, room_polygon_, /*own_wall_skip_m=*/0.30f))
        { ++out.n_occluded; return; }
        {
            Eigen::Vector3f dc = Pw - O;
            const float rc_len = dc.norm();
            if (rc_len > 1e-3f)
            {
                dc /= rc_len;
                for (const auto& [jid, jinst] : instances_)
                {
                    if (jid == inst.node_id or not jinst.ai2_initialized) continue;
                    const auto& js = jinst.model.state();
                    const Eigen::Vector3f Cj(js.cx, js.cy, js.cz + 0.5f * js.h);
                    if (rc::occlusion::cone_blocks(O, dc, rc_len, Cj, 0.5f * std::max(js.w, js.thickness),
                                                   (Cj - O).norm(), occ_margin))
                    { ++out.n_occluded; return; }
                }
            }
        }
        ++out.n_detectable;
        covered_cells.insert(k);
        const float f = central_region_frac_, g = 1.0f - central_region_frac_;
        // Silhouette centroid over ALL detectable samples — the size-invariant input to
        // central_frac(). Deliberately OUTSIDE the central-box test below: it must describe
        // where the whole visible object sits, not only the part already inside the box.
        out.sum_col += col;
        out.sum_row += row;
        out.img_w = static_cast<int>(W);
        out.img_h = static_cast<int>(Himg);
        if (col > f * W and col < g * W and row > f * Himg and row < g * Himg)
            ++out.n_central;                                            // the robot is looking AT it
        range_sum += std::sqrt(X * X + Y * Y + Z * Z);
        if (door_cells.contains(k)) out.e_occ  += 1.0f;                 // still there
        else                        out.e_free += 1.0f;                 // predicted-but-absent
    };

    // Regular grid over the panel face. NX/NZ are numeric SAMPLING RESOLUTION (a quadrature density for the
    // occupancy/detectability counts), not decision thresholds — denser is smoother at linear cost. A door is
    // much taller than wide, so the vertical grid is finer.
    constexpr int NX = 14, NZ = 30;
    for (int ix = 0; ix < NX; ++ix)
        for (int iz = 0; iz < NZ; ++iz)
            classify((-1.0f + 2.0f * (ix + 0.5f) / NX) * hw, s.cz + s.h * (iz + 0.5f) / NZ);

    out.n_cells = static_cast<int>(covered_cells.size());
    if (out.n_detectable > 0)
        out.mean_range_m = static_cast<float>(range_sum / out.n_detectable);
    return out;
}

// |cos| of the camera→leaf ray against the LEAF's face normal. 1 = square-on, 0 = grazing. Diagnostic only
// (a log column) — see the note in door_fitter.h. Keyed on the leaf, not the wall, so it stays meaningful
// once phi is fitted: an open door presents its face at a different angle than its aperture does.
float DoorFitter::door_view_obliquity(const DoorInstance& inst) const
{
    if (not inst.ai2_initialized)
        return 1.0f;
    const auto M = room_T_zed_matrix();   // camera→room (current pose)
    if (not M.has_value())
        return 1.0f;                      // no extrinsic → can't judge → don't suppress
    const Eigen::Vector2f cam(static_cast<float>(M->coeff(0, 3)), static_cast<float>(M->coeff(1, 3)));
    Eigen::Vector2f r = inst.leaf_pose.centre_xy - cam;
    if (r.norm() < 1e-6f)
        return 1.0f;
    r.normalize();
    return std::clamp(std::abs(r.dot(inst.leaf_pose.ey)), 0.0f, 1.0f);   // ey = the leaf's face normal
}

bool DoorFitter::frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const
{
    if (sl.trunc_frac > cfg_.ai2_trunc_gate_frac)
        return false;
    if (not cfg_.ai2_motion_confirm_only)
        return true;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(sl.motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return true;
    return cfg_.ai2_moving_update_center_radius >= 0.0f
       and sl.centroid_radius <= cfg_.ai2_moving_update_center_radius;
}

bool DoorFitter::confirm_only(const DoorInstance& inst) const
{
    if (not cfg_.ai2_motion_confirm_only)
        return false;
    const bool moving = ego_lin_mps_   > cfg_.ai2_still_lin_mps
                     or ego_ang_radps_ > cfg_.ai2_still_ang_radps
                     or std::abs(inst.last_motion_dotd) > cfg_.ai2_still_dotd;
    if (not moving)
        return false;
    // EXCEPTION: a well-centred mask (near the principal point) is trustworthy even while moving → allow the update.
    if (cfg_.ai2_moving_update_center_radius >= 0.0f
        and inst.last_centroid_radius <= cfg_.ai2_moving_update_center_radius)
        return false;
    return true;   // moving AND the mask is off-centre → confirmation only
}

bool DoorFitter::point_in_room(const Eigen::Vector2f& q, float margin_m) const
{
    const std::size_t n = room_polygon_.size();
    if (n < 3)
        return true;                       // no trusted polygon → unknown room → impose no prior

    // Ray-cast parity test (even-odd rule) for strict interior.
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f& b = room_polygon_[j];
        if (((a.y() > q.y()) != (b.y() > q.y())) and
            (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x()))
            inside = not inside;
    }
    if (inside)
        return true;
    if (margin_m <= 0.0f)
        return false;

    // Outside the polygon: accept only if within margin_m of the boundary (wall-hugging door, centroid noise).
    float best2 = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = room_polygon_[i];
        const Eigen::Vector2f ab = room_polygon_[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        const float t = (len2 > 1e-8f) ? std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        best2 = std::min(best2, (q - (a + t * ab)).squaredNorm());
    }
    return best2 <= margin_m * margin_m;
}

void DoorFitter::compute_projected_roi(DoorInstance& inst)
{
    inst.roi_valid = false;
    if (not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room)
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();   // room point → camera frame

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float H  = static_cast<float>(camera_api_->get_height());
    if (fx <= 0.f || fy <= 0.f || W <= 0.f || H <= 0.f)
        return;
    const float cx_px = W * 0.5f, cy_px = H * 0.5f;

    // Project the 8 LEAF corners into the image (door_geometry.h — the ROI must frame the panel the camera
    // actually sees, so this follows the leaf, not the aperture). Camera convention matches the producer:
    // X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    {
        for (const auto& corner : door::leaf_corners(inst.leaf_pose))
        {
            const Eigen::Vector4d Pr(corner.x(), corner.y(), corner.z(), 1.0);
            const Eigen::Vector4d Pc = zed_T_room * Pr;
            const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
            if (Y <= 0.20) continue;   // skip corners at/near the image plane: X/Y explodes there
            ++in_front;
            const float col = cx_px + static_cast<float>(X / Y) * fx;
            const float row = cy_px - static_cast<float>(Z / Y) * fy;
            min_col = std::min(min_col, col); max_col = std::max(max_col, col);
            min_row = std::min(min_row, row); max_row = std::max(max_row, row);
        }
    }

    if (in_front < 4)   // need most of the box in front of the camera to trust the ROI
        return;

    const float roi_cx = 0.5f * (min_col + max_col);
    const float roi_cy = 0.5f * (min_row + max_row);
    const float off_x = (roi_cx - cx_px) / (0.5f * W);   // [-1,1], 0 = centred
    const float off_y = (roi_cy - cy_px) / (0.5f * H);
    const float fill  = std::max((max_col - min_col) / W, (max_row - min_row) / H);
    // Reject degenerate projections (robot too close / a corner grazing the image plane → the bbox
    // explodes). Mark invalid and clamp so consumers/logs never see garbage.
    const bool sane = std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(fill)
                      && std::abs(off_x) < 3.0f && std::abs(off_y) < 3.0f && fill < 4.0f;
    inst.roi_offset_x = std::clamp(off_x, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(off_y, -3.0f, 3.0f);
    inst.roi_fill     = std::clamp(fill, 0.0f, 4.0f);
    inst.roi_valid    = sane;
}

// Delegates to the SHARED bank (common/support_bank via door_support_bank.h). This agent's own answer
// is the EXTENT + the two vertical allowances, and nothing else — see door_support_bank.h::extent_of.
void DoorFitter::ingest_observation_support(DoorInstance& inst, const DoorObservation& observation)
{
    rc::support_bank::ingest(inst, observation.candidate_pts, observation.residual_pts, cfg_);
}
// ─── Factory helpers ─────────────────────────────────────────────────────────

DoorModelParams DoorFitter::make_model_params() const
{
    DoorModelParams p;
    p.sigma_obs = cfg_.sigma_obs;
    return p;
}

}  // namespace rc
