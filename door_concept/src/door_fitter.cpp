/*
 * door_fitter.cpp — the active-inference fit core for door_concept (AI2 recursive-Laplace belief).
 */

#include "door_fitter.h"
#include "door_standpoint.h"
#include "door_phi_posterior.h"   // the phi channel as pure functions (tested offline)

#include "../../common/exclusion/exclusion.h"   // rc::exclusion — the SHARED no-two-objects rule

#include "../../common/diag_log/rotating_csv.h"   // keep the previous run instead of wiping it
#include "door_support_bank.h"   // rc::support_bank:: adapter (SHARED bank)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <limits>
#include <cstdint>
#include <print>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <array>
#include <opencv2/imgproc.hpp>   // fillConvexPoly / bitwise_and — the rasterised phi score
#include "../../common/rt_query_probe/rt_query_probe.h"
#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_room_frame — the room is room_1/room_2/…, never the literal "room"

namespace
{
// Hashed pixel-cell coverage of the current YOLO foreground, split door (occupancy) vs every other label
// (occluder). A CELL-px cell absorbs mask-boundary jitter and makes membership O(1); the key packs the two
// cell indices. CELL is a discretization constant (quadrature), not a belief gate — no config key.
// Used by BOTH the leaf silhouette and the aperture-frame probe, so "what hides a door" is one definition.
struct MaskCells
{
    static constexpr float CELL = 6.0f;
    std::unordered_set<std::int64_t> door, occluder;
    [[nodiscard]] static std::int64_t key(float col, float row)
    {
        const std::int64_t kx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t ky = static_cast<std::int64_t>(std::floor(row / CELL));
        return (kx << 32) ^ (ky & 0xffffffffLL);
    }
    // A pixel covered by some NON-door mask and by no door mask: a nearer masked object hides it.
    [[nodiscard]] bool hides(float col, float row) const
    {
        const std::int64_t k = key(col, row);
        return occluder.contains(k) and not door.contains(k);
    }
};

template <class Packet>
MaskCells build_mask_cells(const Packet& pkt)
{
    MaskCells c;
    if (not pkt.valid)
        return c;
    for (const auto& sl : pkt.slices)
    {
        const std::size_t b = std::min(sl.pixel_begin, pkt.mask_pixels.size());
        const std::size_t e = std::min(sl.pixel_end,   pkt.mask_pixels.size());
        auto& dst = (sl.label == "door") ? c.door : c.occluder;
        for (std::size_t i = b; i < e; ++i)
            dst.insert(MaskCells::key(pkt.mask_pixels[i].x(), pkt.mask_pixels[i].y()));
    }
    return c;
}
}   // namespace

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
    // ★PHI FIRST. Every projection below is built from the leaf pose, and the leaf pose is a function of
    // phi — so the angle has to be estimated before anything reads it, or the whole cycle runs on last
    // cycle's door.
    estimate_phi(inst);

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
        ms.phi   = inst.phi_est;   // no longer pinned: the UI and the graph report what is estimated
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

// Associate the door to the room-polygon wall that holds its aperture: rc::door::associate_wall (door_standpoint.h)
// — one vote per observed point for the wall its sightline from the camera leaves the room through (or, with no
// usable camera, for its nearest wall), collinear-merged into the logical wall, normal INTO the room whatever the
// polygon's winding. ★Why sightlines and not the centroid (2026-10-12): an open leaf at a corner lies in the line
// of the PERPENDICULAR wall, so the centroid's nearest edge is that wall and the door is fitted as a closed one
// in it (door_24: a swung leaf behind room_1's partition became a phantom door in the partition, and its
// approach standpoint landed outside the room). See the header of door_standpoint.h.
DoorFitter::WallFrame DoorFitter::associate_wall(std::span<const Eigen::Vector2f> pts_xy,
                                                 const std::optional<Eigen::Vector2f>& cam_xy) const
{
    WallFrame w;
    if (const auto r = door::associate_wall(room_polygon_, pts_xy, cam_xy); r.ok)
    { w.O = r.O; w.u = r.u; w.len = r.len; w.ok = true; }
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
    const auto c_src = inner_eigen_->transform(chain_src_frame_, centre, rc::room::current_room_frame(*G_), inst.last_mask_timestamp_ms);
    if (not c_src.has_value())
        return;
    DSR::GaussianPoint3D gp;
    gp.mean = c_src.value();
    gp.covariance = DSR::Cov3d::Zero();
    const auto g = gaussian_->transform_point(rc::room::current_room_frame(*G_), gp, chain_src_frame_, inst.last_mask_timestamp_ms);
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
            // MINIMUM-HEIGHT evidence: the top of the observed support (room frame; floor = 0).
            //
            // ★A MASK'S TOP IS A LOWER BOUND ON THE DOOR'S TOP, AND THE ESTIMATOR MUST BE ONE-SIDED.
            // Every way of seeing less than the whole object — border clipping, occlusion, and (the one
            // that bit us) the segmenter returning the door in PIECES — pushes the observed top DOWN.
            // Nothing pushes it up: you cannot see more door than there is. So a low reading is evidence
            // about the view, never about the door, and it must not move the estimate.
            //
            // The comment here already said "never let it drag obs_top_z down" — the symmetric EWMA
            // below did exactly that, and it is what killed door_17 on 2026-09-09. YOLO-sem split the
            // door into two masks; the assigned slice was the LOWER piece, topping out at ~1.27 m;
            // obs_top_z decayed 2.09 -> 2.03 -> 1.91 -> 1.78 -> ... -> 1.47, crossed MinHeightM 1.80,
            // and the min-height prior then removed a door in plain view at exactly -1.5 nats/cycle
            // (exist_short_gain) — while the silhouette channel was charging ZERO absence
            // (free_eff 0.00) and the RGB contour channel was affirming it at +2.83 every cycle. Only
            // trunc_frac was guarded, and a fragment is not border-clipped: it is complete and small.
            //
            // ⚠THE TRADE, stated: a one-sided estimator cannot come back down, so a single spurious
            // tall mask pins the door tall for ever and this prior can no longer fire on it. That is
            // the correct direction to fail — the prior exists to reject things DEMONSTRABLY too short
            // to walk through, and shortness can only be demonstrated by seeing the whole object.
            // Believing a fragment about the top is how it deletes real doors.
            if (slice.has_depth and slice.bbox_max.allFinite())
            {
                const float top = slice.bbox_max.z();
                const float wgt = std::clamp(1.0f - slice.trunc_frac, 0.0f, 1.0f);
                inst.obs_top_last = top;
                if (wgt > 0.0f)
                {
                    constexpr float kEwma = 0.05f;   // confidence accrual rate over untruncated views
                    const float a = kEwma * wgt;
                    // The best lower bound is the tallest one seen: a running MAX. ★It used to EWMA UP from
                    // the first reading (a = 0.05), and that is the same bug as door_17 entering from the other
                    // end: a door first seen in a fragment starts LOW and crawls up more slowly than the
                    // confidence below accrues. try46 (2026-10-17), door_1 re-seen from room_2 after a crossing:
                    // seeded at 0.85 m, had crawled to only 1.40 m when conf reached MinHeightConf 0.30, and the
                    // min-height prior deleted an open door the robot had just driven through.
                    if (std::isnan(inst.obs_top_z) or top > inst.obs_top_z)
                        inst.obs_top_z = top;
                    // Confidence still accumulates on EVERY untruncated look: we did examine the door,
                    // whatever the piece showed. Otherwise a run of fragments would leave the prior
                    // unable to act at all, which is a different failure from the one being fixed.
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

    // Lazy belief init: associate the door to the room-polygon wall it was seen through (fixes yaw / lateral / floor),
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
        std::vector<Eigen::Vector2f> pts_xy;   // plan-view mask points: each votes for the wall it was seen through
        if (npts > 0)
        {
            pts_xy.reserve(static_cast<std::size_t>(npts));
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v)
            { for (const auto& p : v) { sum += p; pts_xy.emplace_back(p.head<2>()); } };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            centroid_xy = (sum / static_cast<float>(npts)).head<2>();
        }
        else
            pts_xy.push_back(centroid_xy);
        // The camera the points were seen from (room frame, current pose; ALWAYS check the optional). Without
        // it every point votes for its nearest wall — the old rule, which is what a missing pose can support.
        std::optional<Eigen::Vector2f> cam_xy;
        if (const auto M = room_T_zed_matrix(); M.has_value())
            cam_xy = Eigen::Vector2f(static_cast<float>(M->coeff(0, 3)), static_cast<float>(M->coeff(1, 3)));
        DoorBeliefParams p = make_belief_params();
        DoorBeliefState s0;
        s0.w = cfg_.door_prior_w_m;
        s0.h = cfg_.door_prior_h_m;
        // Wall association: the belief's centre = O + (s + w/2)·u, so seed s so the panel centre lands on the
        // detection centroid. Off the wall (no polygon) → keep the identity frame and s = along-x of centroid.
        if (const WallFrame wf = associate_wall(pts_xy, cam_xy); wf.ok)
        {
            p.wall_O = wf.O; p.wall_u = wf.u; p.wall_len = wf.len;
            s0.s = std::clamp((centroid_xy - wf.O).dot(wf.u) - 0.5f * s0.w, 0.0f, std::max(0.0f, wf.len - s0.w));
        }
        else
            s0.s = centroid_xy.x() - 0.5f * s0.w;
        if (inst.leaf_side_hint.has_value())
        {
            // The hinge is the aperture edge nearer the remembered hinge point; the swing is the side of THIS wall the
            // remembered opening direction points to. Both follow from geometry, so a reversed wall frame flips them.
            const auto& [hinge_pt, open_dir] = *inst.leaf_side_hint;
            const Eigen::Vector2f near_edge = p.wall_O + s0.s * p.wall_u;
            const Eigen::Vector2f far_edge  = near_edge + s0.w * p.wall_u;
            const Eigen::Vector2f n(-p.wall_u.y(), p.wall_u.x());
            p.leaf.hinge = (hinge_pt - near_edge).norm() <= (hinge_pt - far_edge).norm() ? door::HingeSide::Near
                                                                                           : door::HingeSide::Far;
            p.leaf.swing = n.dot(open_dir) >= 0.0f ? +1.0f : -1.0f;
            std::print("door_concept: [{}] leaf side inherited from the other room: hinge {} swing {:+.0f} (opens {} this room)\n",
                       inst.node_name, p.leaf.hinge == door::HingeSide::Near ? "near" : "far", p.leaf.swing,
                       p.leaf.swing > 0.0f ? "INTO" : "OUT OF");
            inst.leaf_side_hint.reset();
        }
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
        // ★THE LEAF ANGLE IS A LIDAR QUANTITY; A MISSING CAMERA MASK MUST NOT FREEZE IT (2026-10-12).
        // estimate_phi() has had no camera precondition since 2026-09-14, but it is only reached through
        // refresh_geometry(), and this early return skipped that — so a door the detector stopped labelling
        // kept its last phi, and its last phi_curve, for ever. Live try15: the crossed door's mirror door_29
        // lost its mask at 10:23:59, sat at phi = 2.09 rad through a manual CLOSE and a re-OPEN, and published
        // door_open_prob = 0 from a curve nobody had refreshed. The aperture belief still ages (above) and is
        // not re-fitted; only the articulated coordinate is re-estimated, from where the rays stop.
        refresh_geometry(inst);
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
        // ★NO FREEZE SWITCH HERE. What used to gate this update is now stated in the generative model
        // instead: the aperture's along-wall process noise is ZERO (door_belief.h process_std_s), so its
        // precision grows with evidence and its mean stops being dragged by a swinging leaf without any
        // moment at which the code changes behaviour. A boolean freeze also had a defect this does not:
        // it keyed on model_stable, whose convergence test INCLUDES phi, so the aperture unfroze exactly
        // when the leaf started moving — the one time it most needed to hold still.
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
                 << "s,w,h,cx,cy,yaw,std_s,std_w,std_h,phi,"
                 // ★APPENDED, never inserted: an existing reader keyed on column position keeps working.
                 // ★phi_sigma IS THE COLUMN THAT MAKES phi READABLE. An angle without its width cannot be
                 // told apart from a guess, and "off_plane" says whether the LiDAR branch had anything a
                 // CLOSED leaf could not explain: a shut leaf lies exactly in the wall plane, so it is
                 // geometrically indistinguishable from the wall itself and LiDAR can only ever see an
                 // OPEN one. A high off_plane with a low phi, or a low one with a high phi, is a
                 // contradiction the angle alone cannot show.
                 << "leaf_pts,w_sdf,w_mask,w_edge,phi_sigma,phi_measured,off_plane,phi_cv,"
                 << "ray_hit0,ray_expl0,ray_signed0,"
                 << "phi_ninf,phi_beta,phi_lat\n";
    }
    const auto& s = inst.ai2_belief.state();
    const Eigen::Vector2f c = inst.ai2_belief.center_xy();   // APERTURE centre (see DoorBelief) — unchanged by phi
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_trunc_frac << ',' << inst.last_range << ',' << inst.last_clutter_frac << ','
             << s.s << ',' << s.w << ',' << s.h << ','
             << c.x() << ',' << c.y() << ',' << inst.ai2_belief.yaw() << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << inst.leaf.phi << ','
             << inst.dbg_leaf_pts << ',' << inst.dbg_w_sdf << ',' << inst.dbg_w_mask << ','
             << inst.dbg_w_edge << ',' << inst.phi_sigma << ',' << (inst.phi_measured ? 1 : 0) << ','
             << inst.dbg_leaf_off_plane << ',' << inst.dbg_phi_cv << ','
             << inst.dbg_ray_hit0 << ',' << inst.dbg_ray_expl0 << ','
             << inst.dbg_ray_signed0 << ','
             << inst.dbg_phi_ninf << ',' << inst.dbg_phi_beta << ',' << inst.dbg_phi_lat << '\n';
    ai2_csv_.flush();
}

// One row per estimate_phi() call — EVERY call, mask or not. The ai2 row above is written only on the mask
// path, so the LiDAR-only cycles (no detection: exactly the ones where the leaf angle collapsed in try42)
// left no trace. Path = the ai2 path with "_phi" before the extension; off when that is off.
void DoorFitter::log_phi_csv(const DoorInstance& inst, const rc::ai::LidarRays& rays,
                             const std::vector<std::pair<float, float>>& loglik,
                             float prior_mean, float prior_sigma, float post_mean, float post_sigma, bool measured)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not phi_trace_csv_.is_open())
    {
        std::string p = cfg_.ai2_csv_path;
        const auto dot = p.rfind('.');
        p = (dot == std::string::npos) ? p + "_phi" : p.substr(0, dot) + "_phi" + p.substr(dot);
        rc::diag::open_rotating(phi_trace_csv_, p);
        if (not phi_trace_csv_.is_open()) return;
        phi_trace_csv_ << "t_ms,node,ox,oy,nrays,off_plane,ninf,beta,lat,prior_mean,prior_sigma,post_mean,post_sigma,measured,"
                       << "hit0,expl0,signed0,loglik\n";
    }
    const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count();
    phi_trace_csv_ << t_ms << ',' << inst.node_name << ',' << leaf_origin_.x() << ',' << leaf_origin_.y() << ','
                   << rays.endpoints.size() << ',' << inst.dbg_leaf_off_plane << ',' << inst.dbg_phi_ninf << ','
                   << inst.dbg_phi_beta << ',' << inst.dbg_phi_lat << ',' << prior_mean << ',' << prior_sigma << ','
                   << post_mean << ',' << post_sigma << ',' << (measured ? 1 : 0) << ','
                   << inst.dbg_ray_hit0 << ',' << inst.dbg_ray_expl0 << ',' << inst.dbg_ray_signed0 << ',';
    for (std::size_t k = 0; k < loglik.size(); ++k)
        phi_trace_csv_ << (k ? ";" : "") << loglik[k].second;
    phi_trace_csv_ << '\n';
}

// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> DoorFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin the moving room→body hop to the frame's capture stamp (Nearest); keep the rigid body→zed mount
    // at latest (it carries only a bootstrap stamp — a pinned query would fail). ts=0 → current pose.
    // ★LISTENING ONLY — no behaviour change. Asks cortex what this query DID (see
    // common/rt_query_probe/rt_query_probe.h). A timestamped query that falls outside the ring
    // returns the end block and is indistinguishable from a success, so a fitter cannot currently
    // tell whether it is placing detections at the pose the camera actually had. One throttled line
    // per 15 s appends a row to etc/rt_query_probe.csv saying whether this call site ever clamps,
    // by how much, and — the column that makes it readable — how fast the robot was moving in the
    // same window. A clamp only costs geometry while the robot moves; the controller's rate goes
    // 1-3% parked to 35% moving, so a clamp share without a motion column cannot be interpreted.
    static rc::rtprobe::Probe rt_probe{"door_fitter room<-body"};
    DSR::RT_API::TimeQueryInfo rt_info;
    const auto rtb = inner_eigen_->get_transformation_matrix(rc::room::current_room_frame(*G_), "body", pose_ts_ms, "RT",
                                                             DSR::RT_API::TimeQuery::Interpolated,
                                                             &rt_info);
    rt_probe.note(rt_info, G_);
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
// ─── M1: PHI AS A FITTED DOF ─────────────────────────────────────────────────────────────────────
//
// Score candidate leaf angles against the door mask that is actually in the image, and take the best.
// Three things make this honest rather than a way of believing whatever we commanded:
//
//  1. ★THE COMMAND IS A PRIOR, NOT AN ANSWER. While a swing we requested is in flight, the expected
//     angle is known exactly — start, target and the provider's advertised rate — so it centres the
//     search and wins ties. But every candidate is still scored against the image, and if the door did
//     not move the evidence says so and the estimate stays put. That is the difference between
//     anticipating a change you caused and asserting it.
//
//  2. ★NO EVIDENCE ⇒ NO MOVEMENT OF THE ESTIMATE. With no door mask in the frame every candidate
//     scores 0 and the search is uninformative; the estimate then follows the command prior if one is
//     live, and otherwise holds. It must never drift on an empty scoreboard — a flat likelihood is not
//     a measurement.
//
//  3. ★RATE-LIMITED. A leaf is a rigid body on a hinge: it cannot jump 90 degrees between frames. The
//     step cap is the provider's own swing rate (or a generic bound when nobody is actuating), so a
//     single bad frame cannot teleport the model onto a spurious match.
void DoorFitter::estimate_phi(DoorInstance& inst)
{
    // ★CLEAR FIRST, BEFORE ANY EARLY RETURN. The curve is consumed by the silhouette channel in the
    // SAME cycle; leaving last cycle's behind on a frame that could not score phi would marginalise
    // this frame's absence over a stale posterior — evidence carried forward under the name of a
    // measurement that was never taken. An empty curve is the honest state and the consumer handles it.
    inst.phi_curve.clear();
    if (not inst.ai2_initialized)
        return;
    // ★NO CAMERA PRECONDITION. This used to return when the camera, its pose or the mask stream was
    // missing — so a door out of the camera's view could not have its leaf angle estimated at all, even
    // with LiDAR rays passing straight through its aperture. The leaf angle is a LiDAR quantity now.

    // Where the command says the leaf should be right now. Pure prediction from our own request.
    const auto now = std::chrono::steady_clock::now();
    float phi_prior = inst.phi_est;
    bool  have_prior = false;
    // ★HOW FAR COULD THE LEAF HAVE SWUNG SINCE WE LAST LOOKED? That, and nothing else, is the width of
    // the prior. A door is a passive object: between two frames it moves by at most (its own maximum
    // rate) x (the elapsed time), whether the mover is our actuator or a person. So sigma comes from the
    // rate the provider advertised — or the same generic 2 rad/s the step limiter below already uses —
    // times a MEASURED interval, not a per-cycle constant. Long gap, wide prior; fast loop, tight one;
    // first ever look, no prior at all.
    const float swing_rate = inst.phi_cmd_rate > 1e-3f ? inst.phi_cmd_rate : 2.0f;
    const float dt_phi = inst.phi_last_t.time_since_epoch().count() == 0
                       ? std::numeric_limits<float>::infinity()
                       : std::chrono::duration<float>(now - inst.phi_last_t).count();
    inst.phi_last_t = now;
    // Never tighter than the search grid can resolve (that would be a claim about phi finer than the
    // measurement) and never wider than the full range (which is the same as having no prior at all).
    // ★★★THE PRIOR WIDTH COMPOUNDS — IT IS A PREDICT STEP, NOT A FRESH GAUSSIAN EACH CYCLE (2026-09-13).
    // A flat likelihood leaves the posterior equal to the prior; a prior re-centred on the current estimate
    // with a per-cycle width would freeze the value and reassert it with a confidence it never earned
    // (measured: sigma 5.00, 5.00, 5.01 while the estimate froze at 47 deg on a shut door). Compounding the
    // previous posterior's width with this interval's swing makes ignorance ACCUMULATE. Never tighter than
    // the grid resolves, never wider than the range. See rc::door::phi_prior_sigma.
    const float sigma_phi = rc::door::phi_prior_sigma(inst.phi_sigma, swing_rate, dt_phi,
                                                      cfg_.phi_step_rad, cfg_.phi_max_rad);
    if (inst.phi_cmd_active)
    {
        const float dt = std::chrono::duration<float>(now - inst.phi_cmd_t0).count();
        const float span = inst.phi_cmd_to - inst.phi_cmd_from;
        if (inst.phi_cmd_rate > 1e-3f and std::abs(span) > 1e-3f)
        {
            const float travelled = std::min(std::abs(span), inst.phi_cmd_rate * dt);
            phi_prior = inst.phi_cmd_from + std::copysign(travelled, span);
            if (travelled >= std::abs(span) - 1e-4f)
                inst.phi_cmd_active = false;   // the swing should be over; evidence owns it from here
        }
        else
            phi_prior = inst.phi_cmd_to;
        have_prior = true;
    }

    // ─── THE CAMERA DOES NOT SCORE THE LEAF ANGLE (removed 2026-09-14) ─────────────────────────────
    // ★The semantic "door" mask labels the doorway FRAME, not the leaf. Once the leaf swings out of the
    // aperture the mask still covers the opening, so mask overlap kept voting "shut" on an open door.
    // Measured live 2026-09-14, door_19, robot driven through it: at the chosen angle the mask supplied
    // 0.166 of the weight against the LiDAR's 0.038, phi parked at 5 deg, door_open_prob fell to 8e-27,
    // and room_concept could not register the crossing. The contour channel meant to rescue it
    // contributed 0.000 over the whole run.
    // ★AND THE MASK WAS A PRECONDITION: with no door pixels every hypothesis was skipped, the curve came
    // out empty, and the LiDAR likelihood below was never consulted. The camera stays what it is good at:
    // EXISTENCE evidence (compute_silhouette_existence). The leaf angle comes from where rays stop.

    // ─── THE HINGE BRANCH: free energy over phi alone, against LiDAR points on the leaf ──────────
    // ★THE SAME MINIMISATION THE AGENT ALREADY RUNS, RESTRICTED TO THE ONE ARTICULATED COORDINATE. The
    // aperture is held because a hole in a wall cannot move, so the swing has nowhere to go but into the
    // parameter that swings — which is what breaks the s<->phi degeneracy rather than arbitrating it.
    // ★SELECTION IS BY THE MODEL, NOT BY A RADIUS. A leaf swung 90 deg reaches a full aperture-width out
    // from the wall, so the agent's 0.5 m ownership circle about the aperture excludes exactly the points
    // that carry the angle. What belongs to this door is what the LEAF COULD REACH: the swept disc of the
    // hinge, plus a band for thickness and range noise. That is a statement about the kinematics, not a
    // tuned neighbourhood.
    // ─── RAYS THAT INTERROGATE THIS DOORWAY ───────────────────────────────────────────────────────
    // ★SELECT BY THE PATH, NOT BY THE ENDPOINT. The rays that matter most are the ones that went THROUGH:
    // their endpoints are in the next room, metres from the door, so any endpoint-proximity test discards
    // precisely the evidence that an aperture is open. A ray belongs to this doorway if its SEGMENT
    // passes through the cylinder the leaf can sweep — whether it stopped there or flew on.
    // The search grid. ★RANGE AND RESOLUTION BOTH DECLARED; the step COUNT is derived from them.
    const float PHI_MIN = 0.0f;
    const float PHI_MAX = std::max(0.1f, cfg_.phi_max_rad);
    const int   NSTEP   = std::clamp(
        static_cast<int>(std::lround(PHI_MAX / std::max(0.01f, cfg_.phi_step_rad))) + 1, 5, 64);

    // ★THE ALONG-WALL SLIDE IS A NUISANCE OVER ITS HONEST SUPPORT (+-w/2), and the ray selection follows it
    // (door_phi_posterior.h has the why: a carried mirror sat 0.23-0.40 m from the real doorway).
    rc::DoorBelief::FieldNuisance nz = rc::door::phi_field_nuisance(inst.aperture);
    // ★THE SLIDE'S PRIOR IS WHERE THE BELIEF PUTS THE APERTURE (2026-10-13): N(0, var_s) from the aperture's own
    // posterior. A fitted door (σ_s ≈ 6-11 cm) cannot be "explained" by a shut leaf slid 0.55 m onto the wall beside it
    // (try44: every collapse to phi≈0 had lat = -0.55); a freshly carried mirror, whose placement IS uncertain, keeps
    // the wide support that ±w/2 was introduced for.
    if (inst.ai2_initialized)
        nz.lat_sigma = std::sqrt(std::max(1e-8f, inst.ai2_belief.covariance()(0, 0)));   // 0 = known, never "uniform"
    rc::ai::LidarRays rays;
    if (leaf_pts_ and not leaf_pts_->empty() and inst.ai2_initialized)
    {
        // ★SELECT BY THE PATH, NOT BY THE ENDPOINT — a slab through the aperture plus the jamb band; see
        // rc::door::select_doorway_rays (the selection used to live here, unchanged but for the slide).
        auto sel = rc::door::select_doorway_rays(inst.aperture, leaf_origin_, *leaf_pts_,
                                                 std::max(std::abs(nz.lat_lo), std::abs(nz.lat_hi)),
                                                 cfg_.lidar_bpearl_precision);
        rays = std::move(sel.rays);
        // ★THE DIAGNOSTIC, TAKEN BEFORE ANY MODEL CHANGE. Tolerance is 3 sigma of the range noise: a ray
        // that stops within that of where a SHUT leaf would be is explained by one. On a closed door this
        // ratio should be near 1; anything near 0 says the selection is not looking at the leaf.
        {
            const auto ex = inst.ai2_belief.phi_ray_explain(rays, 0.0f, 3.0f * rays.robust_c_m);
            inst.dbg_ray_hit0     = ex.n_hit;
            inst.dbg_ray_expl0    = ex.n_explained;
            inst.dbg_ray_signed0  = ex.mean_signed_m;
        }
        inst.dbg_leaf_pts       = static_cast<int>(rays.endpoints.size());
        inst.dbg_leaf_off_plane = sel.off_plane;
    }
    else
    {
        inst.dbg_leaf_pts = 0;
        inst.dbg_leaf_off_plane = 0;
    }

    // ─── THE CURVE: LiDAR likelihood × the swing prior ─────────────────────────────────────────────
    // ★THE FULL-FIELD RESPONSE (2026-09-16), NOW TEMPERED BY WHAT THE FRAME DISCRIMINATES (2026-10-13).
    // DoorBelief::phi_field_loglik: same per-ray density as phi_field_likelihood, the slide marginalised over
    // +-w/2, and the joint log-likelihood raised to n_eff/n_inf — a frame that sees a sliver of the leaf
    // carries a sliver of evidence instead of a delta.
    std::vector<std::pair<float, float>> loglik;
    if (not rays.endpoints.empty())
    {
        auto fr = inst.ai2_belief.phi_field_loglik(rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        loglik = std::move(fr.loglik);
        inst.dbg_phi_ninf  = fr.n_inf;
        inst.dbg_phi_beta  = fr.beta;
        inst.dbg_phi_lat   = fr.lat_map;
    }
    else
    {
        inst.dbg_phi_ninf = 0.0f;
        inst.dbg_phi_beta = 1.0f;
    }

    // ★THE PRIOR MULTIPLIES THE LIKELIHOOD, IN LOGS. A flat likelihood leaves the prior standing; a peaked
    // one moves it; and no likelihood, however far from the prior, can underflow the posterior to nothing
    // (the old linear product did, and the "no curve" branch then threw the prior away — see
    // rc::door::phi_posterior_step).
    const auto post = rc::door::phi_posterior_step(loglik, PHI_MIN, PHI_MAX, NSTEP, phi_prior, sigma_phi);
    inst.phi_curve  = post.curve;
    inst.dbg_phi_cv = post.cv;
    inst.dbg_w_sdf  = post.w_lik_best;
    inst.dbg_w_mask = 0.0f;   // kept in the log schema; the camera no longer votes on phi
    inst.dbg_w_edge = 0.0f;

    // ─── COLLAPSE THE POSTERIOR: a MEAN and a WIDTH, always ────────────────────────────────────────
    // ★THE WIDTH IS THE POINT, and ★NO ARGMAX ANYWHERE (the argmax of a nearly-flat curve is noise; chasing
    // it slid the aperture along the wall, r = -1.000, 34 cm against 41 deg).
    // ★THE CARRIED BELIEF IS THE POSTERIOR'S (mean, sigma) — NO CLAMP ON THE MEAN (2026-10-13). The mean used
    // to move at most swing_rate*dt per cycle while sigma took the posterior's width, so the NEXT prior was
    // centred many sigma from the evidence and fought it (harness (c): first look at an open door,
    // P(passable) 1.00 -> 0.10 -> 1.00). "A hinge cannot jump" is already the predict step: the prior's width
    // grows by exactly what the leaf could have swung in dt (phi_prior_sigma), so a single frame moves the
    // mean far only if its evidence outweighs that prior — and, tempered by what it actually observed
    // (phi_field_loglik), a near-empty frame cannot.
    if (not post.curve.empty())
    {
        inst.phi_sigma = post.sigma;
        inst.phi_est   = post.mean;
    }
    else
        inst.phi_sigma = PHI_MAX - PHI_MIN;   // not even a prior: HOLD the estimate, width = the whole range
    inst.phi_measured = post.measured;
    log_phi_csv(inst, rays, loglik, phi_prior, sigma_phi, post.mean, post.sigma, post.measured);
    // phi_support was the camera mask overlap at the claimed angle; -1 = "the camera does not score phi".
    inst.phi_support = -1.0f;

    inst.phi_est = std::clamp(inst.phi_est, PHI_MIN, PHI_MAX);
    inst.leaf.phi = inst.phi_est;
    if (inst.ai2_initialized)
        inst.ai2_belief.set_leaf_phi(inst.phi_est);
}

DoorSilhouette DoorFitter::compute_silhouette_existence(const DoorInstance& inst,
                                                        const rc::SemanticProbField* field)
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
    // Shared with compute_frame_probe — see MaskCells.
    const MaskCells cells = build_mask_cells(pkt);
    const auto key = [&cells](float col, float row) { return cells.key(col, row); };
    const auto& door_cells     = cells.door;
    const auto& occluder_cells = cells.occluder;

    const auto& s = inst.model.state();
    const float hw = inst.leaf_pose.half_w;

    // Camera position (room frame) for the GEOMETRIC occlusion tests below.
    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)),
                            static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const float occ_margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);

    // ── THE LEAF BEYOND THE APERTURE (door_room_band.h) ─────────────────────────────────────────────────
    // A leaf swung OUT of the room the camera is in lies in the NEXT room. What reaches the camera of it is
    // a sliver seen edge-on THROUGH the hole, and the P(detect) below (envelope × obliquity × resolvability
    // × in-view × central) is a model of a leaf in THIS room — it does not represent that view. Live try34:
    // the crossed door, carried into room_2 with its leaf at ~95-110° out of room_2, scored free=112-122 of
    // 120 detectable samples (the part within walls_block's 0.30 m own-wall skip; the rest was already
    // wall-occluded) on a door that plainly exists. The model has no term for the opening's jambs/casing as
    // occluders of what lies past the room-side face, so each sample carries weight 1 − P(it lies past that
    // face), measured from the APERTURE LINE: a sample on it (a shut leaf, the hinge edge) weighs exactly 1,
    // one a few σ out weighs ~0. A weight, not a skip: occupancy and absence both scale by it, and
    // so does the visible fraction (det_w), so a door whose whole leaf is beyond is simply not probed by
    // this channel — the frame channel (compute_frame_probe) is what speaks for it then.
    const auto band = room_band();
    const float delta_ap = has_room_polygon()
                         ? rc::door::signed_boundary_distance(room_polygon_, inst.aperture.centre_xy())
                         : std::numeric_limits<float>::quiet_NaN();
    const bool  band_known = std::isfinite(delta_ap);
    const Eigen::Vector2f n_out = band_known
        ? rc::door::outward_normal(room_polygon_, inst.aperture.centre_xy(), inst.aperture.across_u(), band)
        : Eigen::Vector2f::Zero();
    const float r0 = rc::door::face_offset_from_aperture(delta_ap);
    const auto room_weight = [&](const Eigen::Vector3f& P) -> float
    {
        if (not band_known) return 1.0f;
        const float a = (Eigen::Vector2f(P.x(), P.y()) - inst.aperture.wall_O).dot(n_out);
        return 1.0f - rc::door::p_beyond_far_face(a, r0, band.sigma_m);
    };

    // Classify ONE LEAF-face sample (local x hinge→free edge, z absolute): project it, then vote.
    //
    // ★ The sample comes from door::leaf_point at the instance's CURRENT leaf pose. It used to be spelled
    // out here as a rectangle in the WALL PLANE, which meant that the moment a door actually opened, its
    // predicted silhouette landed where the leaf no longer was: every sample came back unlit, e_free
    // spiked, and the existence channel deleted the door BECAUSE it opened. With phi pinned at 0 (M0) this
    // yields the identical samples; when phi becomes a fitted DOF in M1 it follows the leaf for free.
    std::unordered_set<std::int64_t> covered_cells;   // distinct cells the DETECTABLE silhouette occupies
    double range_sum = 0.0;
    // ★PARAMETERISED BY THE LEAF POSE, AND BY WHERE IT ACCUMULATES. One pass renders the panel at ONE
    // opening angle; the caller runs it once per candidate phi and combines. `extras` marks the pass
    // whose by-products describe the scene for everybody else (centroid, semantic field, covered cells,
    // range) — those are reported at the modal pose, because they answer "where is it on screen", which
    // is a statement about one drawing and not something to average over hypotheses.
    const auto classify = [&](const door::LeafPose& L, DoorSilhouette& out, bool extras,
                              float lx, float lz)
    {
        ++out.n_total;                                                  // one sample of the WHOLE panel
        const Eigen::Vector3f Ps = door::leaf_point(L, lx, 0.0f, lz);   // face at mid-thickness
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
        const float w_room = room_weight(Pw);                           // 1 − P(seen through the opening)
        if (not (w_room > 0.0f))
        { ++out.n_occluded; return; }                                   // wholly through the opening: not this view
        ++out.n_detectable;
        out.det_w += w_room;
        if (extras) covered_cells.insert(k);
        const float f = central_region_frac_, g = 1.0f - central_region_frac_;
        // Silhouette centroid over ALL detectable samples — the size-invariant input to
        // central_frac(). Deliberately OUTSIDE the central-box test below: it must describe
        // where the whole visible object sits, not only the part already inside the box.
        out.sum_col += col;
        out.sum_row += row;

        // Sample the graded posterior right here, at a pixel we have just established the camera could
        // actually see. Doing it in this loop rather than from a bounding box is the point: the contour
        // is the door's own projected shape, so the statistic is about the door and not about the
        // rectangle around it.
        if (extras and field != nullptr and field->valid())
            if (const float p = field->at(static_cast<int>(col), static_cast<int>(row)); p >= 0.0f)
            {
                out.field_sum += p;
                out.field_max = std::max(out.field_max, p);
                ++out.field_n;
            }
        out.img_w = static_cast<int>(W);
        out.img_h = static_cast<int>(Himg);
        if (col > f * W and col < g * W and row > f * Himg and row < g * Himg)
            ++out.n_central;                                            // the robot is looking AT it
        if (extras) range_sum += std::sqrt(X * X + Y * Y + Z * Z);
        if (door_cells.contains(k)) out.e_occ  += w_room;               // still there
        else                        out.e_free += w_room;               // predicted-but-absent
    };

    // Regular grid over the panel face. NX/NZ are numeric SAMPLING RESOLUTION (a quadrature density for the
    // occupancy/detectability counts), not decision thresholds — denser is smoother at linear cost. A door is
    // much taller than wide, so the vertical grid is finer.
    constexpr int NX = 14, NZ = 30;
    const auto sweep = [&](const door::LeafPose& L, DoorSilhouette& acc, bool extras)
    {
        for (int ix = 0; ix < NX; ++ix)
            for (int iz = 0; iz < NZ; ++iz)
                classify(L, acc, extras,
                         (-1.0f + 2.0f * (ix + 0.5f) / NX) * hw, s.cz + s.h * (iz + 0.5f) / NZ);
    };

    // The modal pass: the leaf where phi_est puts it. This is the drawing — face_px, the centroid, the
    // semantic-field statistics and the covered cells all come from here, and so does everything the UI
    // and the logs show.
    sweep(inst.leaf_pose, out, /*extras=*/true);
    out.n_cells = static_cast<int>(covered_cells.size());

    // ── ABSENCE IS MARGINALISED OVER phi, NOT CONDITIONED ON ITS ARGMAX ──────────────────────────────
    // ★WHY. The silhouette is the only channel permitted to remove a door, and until now it rendered the
    // leaf at a single fitted angle. Measured 2026-09-10 over one approach, the phi likelihood peaks at
    // 0.007-0.245 — flat — so that angle is chosen from noise; twice on that run the leaf was projected
    // where the panel was not, occ fell 207 -> 0, free_eff rose 0 -> 33 and L ran +4.00 -> -4.00 in 13
    // cycles. The door was not gone; the model had merely guessed its hinge angle wrong. Charging a
    // removal to an unidentified nuisance parameter is the defect, and picking the peak better, or
    // smoothing it, would not fix it — the information is not there to be extracted.
    // ★THE FIX IS THE STANDARD ONE: integrate the nuisance out. e_free under P(phi | mask) instead of
    // e_free at phi-hat. A sample counts as absent to the extent that it is absent at EVERY angle the
    // data still permits, so a flat curve nearly cancels the absence (the panel is SOMEWHERE, we just
    // cannot say where) while a sharply-peaked one charges it in full. There is no flatness test and no
    // angle gate: the sharpness of the curve does the weighting by itself, which is what makes this a
    // model term and not another threshold.
    // ★THE WEIGHTS ARE THE POSTERIOR the angle estimate itself is drawn from — image support times the
    // persistence/command prior — so the absence is marginalised over exactly what this agent believes
    // about the leaf, and the two consumers of phi cannot disagree. An empty curve means the frame
    // carried no door mask to score phi against, and then there is nothing to marginalise: the modal
    // pass stands, which is exactly the old behaviour on exactly the frames where it was never at fault
    // (with no mask anywhere, every angle is equally unlit and the marginal equals the point estimate).
    double w_sum = 0.0;
    for (const auto& [phi, sup] : inst.phi_curve)
        w_sum += std::max(0.0f, sup);
    if (inst.phi_curve.size() > 1 and w_sum > 0.0)
    {
        double m_occ = 0.0, m_free = 0.0, m_total = 0.0, m_det = 0.0;
        for (const auto& [phi, sup] : inst.phi_curve)
        {
            const double w = std::max(0.0f, sup) / w_sum;
            if (w <= 0.0) continue;
            door::LeafState ls = inst.leaf;
            ls.phi = phi;
            DoorSilhouette acc;
            sweep(door::leaf_pose(inst.aperture, ls), acc, /*extras=*/false);
            m_occ     += w * acc.e_occ;
            m_free    += w * acc.e_free;
            m_total   += w * acc.n_total;
            m_det     += w * acc.det_w;
        }
        // Only the ABSENCE/OCCUPANCY evidence is replaced. The modal counts stay as the modal pass left
        // them, because resolvability() and central_frac() are ratios built from that one rendering.
        out.e_occ  = static_cast<float>(m_occ);
        out.e_free = static_cast<float>(m_free);
        out.n_det_marg   = static_cast<float>(m_det);
        out.n_total_marg = static_cast<float>(m_total);
        out.phi_marginalised = true;
        out.phi_n_hyp = static_cast<int>(inst.phi_curve.size());
        out.phi_w_max = 0.0f;
        for (const auto& [phi, sup] : inst.phi_curve)
            out.phi_w_max = std::max(out.phi_w_max, static_cast<float>(std::max(0.0f, sup) / w_sum));
    }

    if (out.n_detectable > 0)
        out.mean_range_m = static_cast<float>(range_sum / out.n_detectable);

    // ★THE LEAF-FACE QUAD, projected with the SAME camera and the SAME transform as every sample above.
    // This is what the RGB and depth contour checks are scored on, so the thing being defended and the
    // thing being measured are one contour. Order is a closed loop (bottom-hinge, bottom-free, top-free,
    // top-hinge) — a polygon whose points are not in loop order produces a bow-tie whose "edges" cross
    // the object and would score whatever happens to lie under the diagonals.
    //
    // ★THE CONSTRUCTION NOW LIVES IN common/contour_edge/contour_edge_project.h, unchanged in behaviour:
    // the same four corners, the same ±1/±1.6 slide along the leaf's OWN plane in 3-D, the same
    // all-or-nothing rejection when a corner falls behind the camera. It moved because the trap it
    // encodes is not door-specific — a control displaced in IMAGE pixels stops being a like-for-like
    // comparison the moment the object leaves the surface its neighbours are made of, and every concept
    // agent adopting this channel would otherwise rediscover that the expensive way (we did: the channel
    // voted to DELETE the door, hardest when it was open). It also returns the per-vertex DEPTH the
    // projection already computed, which is what the depth channel tests the belief against.
    {
        const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
        {
            const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
            if (Pc.y() <= 0.20) return std::nullopt;                       // a corner behind the camera
            const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
            if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
            rc::edges::ProjectedVertex v;
            v.px = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
            // The ZED depth plane stores the camera-frame FORWARD coordinate, not the Euclidean norm
            // (retina deprojects with `py = depth`). Handing back a norm here would read high off-axis
            // and look exactly like a door believed slightly too far away.
            v.depth_m = static_cast<float>(Pc.y());
            return v;
        };
        const float half_h = inst.leaf_pose.half_h;
        const float cz     = inst.leaf_pose.centre_z;
        const auto corner = [&](float lx, float lz) -> Eigen::Vector3d
        { return door::leaf_point(inst.leaf_pose, lx, 0.0f, lz).cast<double>(); };
        const std::array<Eigen::Vector3d, 4> quad{
            corner(-hw, cz - half_h), corner(hw, cz - half_h),
            corner( hw, cz + half_h), corner(-hw, cz + half_h)};
        // Slide direction: the leaf's own +x (hinge → free edge), so a displaced copy stays in the leaf's
        // plane at any phi. Span is the leaf's full width.
        const Eigen::Vector3d along = corner(1.0f, cz) - corner(0.0f, cz);
        out.contour = rc::edges::project_quad(quad, along, 2.0f * hw, project);
    }
    if (field != nullptr and field->valid())
        out.field_bg = field->background_mean();
    return out;
}

// ── THE DOORWAY FRAME: the part of a door that does not move ────────────────────────────────────────────
// Projects the aperture's jambs + lintel and weights each sample by whether the camera could see it. The
// scoring against the image and the reasoning behind the statistic live in door_frame_evidence.h; this
// method owns only what needs the agent: the camera, the transform, the masks, the walls, the other doors
// and the leaf's opening-angle posterior.
//
// Visibility per frame point P (a weight, never a skip-the-cycle switch):
//   · outside the real frustum / behind the near clip                 → 0
//   · a NON-door mask covers its pixel (a person, a chair in front)   → 0   (same cells as the silhouette)
//   · a room WALL crosses the sightline (rc::occlusion::walls_block)  → 0   (own wall skipped, as everywhere)
//   · another door instance stands in front (cone_blocks)             → 0
//   · OUR OWN LEAF crosses the sightline                              → 1 − P_φ(blocked), marginalised over
//     the same phi posterior the silhouette's absence is marginalised over, so the two cannot disagree
//     about where the leaf is. A leaf swung toward the camera hides the jamb behind it; one swung away
//     (out of this room — the try32 case) hides nothing, and the frame speaks in full.
rc::door::FrameProbe DoorFitter::compute_frame_probe(const DoorInstance& inst)
{
    if (not inner_eigen_ or not inst.ai2_initialized)
        return {};
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return {};
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return {};
    }
    const auto Mopt = room_T_zed_matrix();   // camera→room, current pose — the SAME transform the silhouette uses
    if (not Mopt.has_value())
        return {};
    const Eigen::Matrix4d zed_T_room = Mopt.value().inverse();
    const int W = camera_api_->get_width(), H = camera_api_->get_height();
    if (W <= 0 or H <= 0)
        return {};

    // No mask packet ⇒ no KNOWN masked occluder: the cells are empty and that factor is 1. The frame is a
    // geometric probe of the RGB image, so it does not need masks to exist — only to know what hides it.
    const MaskCells cells = mask_ingestor_ ? build_mask_cells(mask_ingestor_->packet()) : MaskCells{};

    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)), static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const Eigen::Vector2f Oxy(O.x(), O.y());
    const float occ_margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);

    // The leaf hypotheses and their weights: the phi posterior when there is one, else the modal pose.
    std::vector<std::pair<door::LeafPose, float>> leaves;
    {
        double w_sum = 0.0;
        for (const auto& [phi, sup] : inst.phi_curve)
            w_sum += std::max(0.0f, sup);
        if (inst.phi_curve.size() > 1 and w_sum > 0.0)
            for (const auto& [phi, sup] : inst.phi_curve)
            {
                const float w = static_cast<float>(std::max(0.0f, sup) / w_sum);
                if (w <= 0.0f) continue;
                door::LeafState ls = inst.leaf;
                ls.phi = phi;
                leaves.emplace_back(door::leaf_pose(inst.aperture, ls), w);
            }
        else
            leaves.emplace_back(inst.leaf_pose, 1.0f);
    }
    const auto own_leaf_clear = [&](const Eigen::Vector3f& P) -> float
    {
        float blocked = 0.0f;
        for (const auto& [L, w] : leaves)
            if (door::leaf_blocks_sightline(L, O, P))
                blocked += w;
        return std::clamp(1.0f - blocked, 0.0f, 1.0f);
    };

    const auto project = [&](const Eigen::Vector3d& Pr) -> std::optional<rc::edges::ProjectedVertex>
    {
        const Eigen::Vector4d Pc = zed_T_room * Pr.homogeneous();
        if (Pc.y() <= 0.20) return std::nullopt;                       // behind / at the near clip
        const Eigen::Vector2d uv = camera_api_->project(Eigen::Vector3d(Pc.x(), Pc.y(), Pc.z()));
        if (not std::isfinite(uv.x()) or not std::isfinite(uv.y())) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
        v.depth_m = static_cast<float>(Pc.y());                        // ZED depth = forward coordinate
        return v;
    };

    // The frame is scored where the belief puts the aperture. (Tried 2026-10-13 and NOT adopted: seating it on
    // the room polygon's face. With a real casing standing proud of the wall the visible occluding edge is the
    // casing's front, which is where the fit already puts the aperture — try34's carried door sat 6 cm in front
    // of room_2's face, a 5 cm proud Webots casing — and the synthetic scene in tests/test_door_frame_evidence.cpp
    // scored the seated probe WORSE in 3 of 5 views, down to +0.16 from +28 on an oblique view.)
    const door::Aperture& ap_seen = inst.aperture;

    const auto visible = [&](const Eigen::Vector3f& P) -> float
    {
        const auto v = project(P.cast<double>());
        if (not v.has_value()) return 0.0f;
        if (v->px.x < 0.f or v->px.y < 0.f or v->px.x >= static_cast<float>(W) or v->px.y >= static_cast<float>(H))
            return 0.0f;
        if (cells.hides(v->px.x, v->px.y))
            return 0.0f;
        if (rc::occlusion::walls_block(Oxy, {P.x(), P.y()}, room_polygon_, /*own_wall_skip_m=*/0.30f))
            return 0.0f;
        Eigen::Vector3f dc = P - O;
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
                    return 0.0f;
            }
        }
        return own_leaf_clear(P);
    };
    return rc::door::build_frame_probe(ap_seen, project, visible, own_leaf_clear, W, H, cfg_.frame_casing_m,
                                       cfg_.exist_boundary_sigma_m);
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

float DoorFitter::aperture_band_offset(const DoorInstance& inst) const
{
    if (not has_room_polygon())
        return std::numeric_limits<float>::quiet_NaN();
    const auto& ms = inst.model.state();
    return rc::door::signed_boundary_distance(room_polygon_, Eigen::Vector2f(ms.ap_cx, ms.ap_cy));
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
    inst.roi_fill_h   = std::clamp((max_col - min_col) / W, 0.0f, 4.0f);
    inst.roi_fill_v   = std::clamp((max_row - min_row) / H, 0.0f, 4.0f);
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
