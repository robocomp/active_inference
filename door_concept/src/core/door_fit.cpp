// door_fit.cpp — see door_fit.h. MOVED from door_fitter.cpp (e902bf7f) with its comments: ensure_instance's DSR-free
// half, observe, run_inference, estimate_phi, refresh_geometry, associate_wall, seed_bearing_hypothesis,
// compute_projected_roi, compute_chain_cov, the two CSV rows, confirm_only / motion_magnitude / periphery_penalty.
#include "door_fit.h"
#include <tuple>
#include "door_core.h"                  // point_in_polygon — the containment rule, shared with the birth test
#include "../door_standpoint.h"
#include "../door_phi_posterior.h"      // the phi channel as pure functions (tested offline)
#include "../door_opening_edges.h"      // try96: wall registration, the opening's through-span, the registration offset
#include "../door_support_bank.h"       // rc::support_bank:: adapter (SHARED bank)

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <locale>
#include <numeric>
#include <sstream>

namespace rc::door::core
{
namespace
{
// Fix round 1 (M-1): the hinge / swing log-odds are relative to the belief's leaf; a (re-)created belief may hang it the other
// way, and then the same evidence reads with the opposite sign.
void resign_leaf_llrs(rc::DoorFitState& inst)
{
    const auto& L = inst.ai2_belief.params().leaf;
    if (inst.hinge_llr_ref.has_value() and *inst.hinge_llr_ref != L.hinge) inst.hinge_llr = -inst.hinge_llr;
    if (inst.swing_llr_ref.has_value() and *inst.swing_llr_ref * L.swing < 0.0f) inst.swing_llr = -inst.swing_llr;
    inst.hinge_llr_ref = L.hinge;
    inst.swing_llr_ref = L.swing;
}

// ─── cortex InnerGaussianAPI, the parts transform_point uses (cortex/api/dsr_inner_gaussian_api.cpp) ───────────────
using RTMat    = Eigen::Transform<double, 3, Eigen::Affine>;   // Mat::RTMat
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Cov6d    = Eigen::Matrix<double, 6, 6>;

Eigen::Matrix3d skew_symmetric(const Eigen::Vector3d& vector)
{
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -vector.z();
    skew(0, 2) = vector.y();
    skew(1, 0) = vector.z();
    skew(1, 2) = -vector.x();
    skew(2, 0) = -vector.y();
    skew(2, 1) = vector.x();
    return skew;
}

Cov6d adjoint_matrix(const RTMat& transform)
{
    Cov6d adjoint = Cov6d::Zero();
    const auto rotation = transform.rotation();
    const auto translation = transform.translation();
    adjoint.block<3, 3>(0, 0) = rotation;
    adjoint.block<3, 3>(0, 3) = skew_symmetric(translation) * rotation;
    adjoint.block<3, 3>(3, 3) = rotation;
    return adjoint;
}

RTMat pose_vector_to_rtmat(const Vector6d& pose)
{
    auto rtmat = RTMat::Identity();
    rtmat.translate(pose.head<3>());
    rtmat.rotate(Eigen::AngleAxisd(pose(3), Eigen::Vector3d::UnitX()) *
                 Eigen::AngleAxisd(pose(4), Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(pose(5), Eigen::Vector3d::UnitZ()));
    return rtmat;
}

Vector6d rtmat_to_pose_vector(const RTMat& rtmat)
{
    Vector6d pose;
    const auto angles = rtmat.rotation().eulerAngles(0, 1, 2);
    pose << rtmat.translation().x(), rtmat.translation().y(), rtmat.translation().z(),
            angles.x(), angles.y(), angles.z();
    return pose;
}

template<int OutDim, int InDim, typename Function, typename Difference>
Eigen::Matrix<double, OutDim, InDim> numerical_jacobian(Function function, Difference difference, double epsilon = 1e-6)
{
    Eigen::Matrix<double, OutDim, InDim> jacobian = Eigen::Matrix<double, OutDim, InDim>::Zero();
    Eigen::Matrix<double, InDim, 1> delta = Eigen::Matrix<double, InDim, 1>::Zero();
    for (int column = 0; column < InDim; ++column)
    {
        delta.setZero();
        delta(column) = epsilon;
        const auto positive = function(delta);
        const auto negative = function(-delta);
        jacobian.col(column) = difference(positive, negative) / (2.0 * epsilon);
    }
    return jacobian;
}
}  // namespace

std::optional<Eigen::Matrix3d> chain_point_cov(const std::vector<RtChainStep>& recorded, const Eigen::Vector3d& p_room)
{
    if (recorded.empty())
        return std::nullopt;
    // cortex's PathSteps for (dest = room, orig = zed): the zed side ascending, then the room side descending
    // (inverted) — exactly the REVERSE of the recorded order (input_snapshot.cpp room_zed_chain).
    struct Step { RTMat edge_mean; Cov6d edge_covariance; bool invert; };
    std::vector<Step> steps;
    steps.reserve(recorded.size());
    for (auto it = recorded.rbegin(); it != recorded.rend(); ++it)
    {
        RTMat T;
        T.matrix() = it->T;
        steps.push_back({T, it->cov.value_or(Cov6d::Zero()), it->inverted});
    }
    // get_transformation_gaussian
    Vector6d mean = Vector6d::Zero();
    Cov6d covariance = Cov6d::Zero();
    auto prefix = RTMat::Identity();
    for (const auto& step : steps)
    {
        auto operation_covariance = step.edge_covariance;
        if (step.invert)
        {
            const auto inverse_map = adjoint_matrix(step.edge_mean);
            operation_covariance = inverse_map * operation_covariance * inverse_map.transpose();
        }
        covariance += adjoint_matrix(prefix.inverse()) * operation_covariance * adjoint_matrix(prefix.inverse()).transpose();
        const auto operation = step.invert ? step.edge_mean.inverse() : step.edge_mean;
        prefix = operation * prefix;
    }
    mean = rtmat_to_pose_vector(prefix);
    // the zed-frame point compute_chain_cov transformed the door centre into (inner_eigen_->transform(zed, c, room, ts))
    const Eigen::Vector3d point = prefix.inverse() * p_room;
    const Eigen::Matrix3d point_cov = Eigen::Matrix3d::Zero();
    // transform_point
    const auto rtmat = pose_vector_to_rtmat(mean);
    const auto transform_jacobian = numerical_jacobian<3, 6>(
        [&](const Vector6d& delta) { return (rtmat * pose_vector_to_rtmat(delta) * point.homogeneous()).eval(); },
        [](const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) { return lhs - rhs; });
    const auto point_jacobian = numerical_jacobian<3, 3>(
        [&](const Eigen::Vector3d& delta) { return (rtmat * (point + delta).homogeneous()).eval(); },
        [](const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) { return lhs - rhs; });
    Eigen::Matrix3d out = transform_jacobian * covariance * transform_jacobian.transpose()
                        + point_jacobian * point_cov * point_jacobian.transpose();
    return out;
}

std::string phi_csv_path(const std::string& ai2_csv_path)
{
    std::string p = ai2_csv_path;
    const auto dot = p.rfind('.');
    return (dot == std::string::npos) ? p + "_phi" : p.substr(0, dot) + "_phi" + p.substr(dot);
}

rc::DoorBeliefParams DoorFit::make_belief_params() const
{
    rc::DoorBeliefParams p;
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
    // Task 5b: the leaf plane's depth d beyond the wall line: mean 0 (the free-space polygon closes ON a shut leaf), variance
    // T²/12 + σ_b² (where in the wall's thickness T the leaf hangs relative to the wall beside it, plus the polygon's
    // registration error σ_b — the same T and σ_b as the wall band). See DoorBeliefParams::recess_prior_m.
    {
        const auto band = room_band();
        p.recess_prior_m   = 0.0f;
        p.recess_prior_std = cfg_.door_recess_prior_std >= 0.0f
                           ? cfg_.door_recess_prior_std
                           : std::sqrt(band.wall_thickness_m * band.wall_thickness_m / 12.0f + band.sigma_m * band.sigma_m);
    }
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

rc::DoorModelParams DoorFit::make_model_params() const
{
    rc::DoorModelParams p;
    p.sigma_obs = cfg_.sigma_obs;
    return p;
}

// ─── The SINGLE authoring point for a door's geometry ────────────────────────────────────────────
// Refresh the cached aperture / leaf / leaf_pose from the belief, then write the room-frame read-back
// into DoorState. Every geometry consumer in the agent reads one of these; nothing reconstructs the
// panel rectangle for itself. Called from ensure_instance (pre-belief), the bearing seed, and the
// per-cycle write-back — the only three places a door's shape can change.
void DoorFit::refresh_geometry(rc::DoorFitState& inst)
{
    // ★PHI FIRST. Every projection below is built from the leaf pose, and the leaf pose is a function of
    // phi — so the angle has to be estimated before anything reads it, or the whole cycle runs on last
    // cycle's door.
    estimate_phi(inst);

    rc::DoorState ms = inst.model.state();
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
        // geometry we have is the published box (door_model.h pre_belief_geometry — shared with the DoorCore's
        // replay world, which must build a newborn exactly like this).
        const rc::PreBeliefGeometry g = rc::pre_belief_geometry(ms);
        inst.leaf_pose = g.leaf_pose;
        inst.leaf      = g.leaf;
        inst.aperture  = g.aperture;
    }
    rc::read_back_leaf(ms, inst.leaf_pose, inst.leaf);
    inst.model.set_state(ms);
}

// Associate the door to the room-polygon wall that holds its aperture: rc::door::associate_wall (door_standpoint.h)
// — one vote per observed point for the wall its sightline from the camera leaves the room through (or, with no
// usable camera, for its nearest wall), collinear-merged into the logical wall, normal INTO the room whatever the
// polygon's winding. ★Why sightlines and not the centroid (2026-10-12): an open leaf at a corner lies in the line
// of the PERPENDICULAR wall, so the centroid's nearest edge is that wall and the door is fitted as a closed one
// in it (door_24: a swung leaf behind room_1's partition became a phantom door in the partition, and its
// approach standpoint landed outside the room). See the header of door_standpoint.h.
DoorFit::WallFrame DoorFit::associate_wall(std::span<const Eigen::Vector2f> pts_xy,
                                           const std::optional<Eigen::Vector2f>& cam_xy) const
{
    WallFrame w;
    // A doorway recess no deeper than the wall belongs to the wall (door_standpoint.h, wall_run_of_edge).
    if (const auto r = door::associate_wall(room_polygon_, pts_xy, cam_xy, room_band().wall_thickness_m); r.ok)
    { w.O = r.O; w.u = r.u; w.len = r.len; w.ok = true; }
    return w;
}

void DoorFit::seed_bearing_hypothesis(rc::DoorFitState& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                      float nominal_range, float along_std, float across_std, float yaw_std)
{
    // Bearing-only birth does not fit the wall-anchored door model (a door is placed by wall association,
    // not by a peripheral ray). Kept as a dormant stub (cfg_.bearing_birth_enabled defaults OFF): seed a
    // broad panel on the ray so the scene-graph/viewer show the hypothesis, and flag it for the Orient
    // affordance. A subsequent depth mask replaces it with a wall-associated fit in run_inference.
    (void) along_std; (void) across_std; (void) yaw_std;
    rc::DoorBeliefParams p = make_belief_params();
    p.wall_O = robot_xy;
    p.wall_u = {std::cos(azimuth), std::sin(azimuth)};
    p.prior_s_std = std::max(p.prior_s_std, along_std);   // range unknown → broad along the ray
    rc::DoorBeliefState s0;
    s0.s = std::max(0.0f, nominal_range);   // along the ray from the robot
    s0.w = cfg_.door_prior_w_m;
    s0.h = cfg_.door_prior_h_m;
    inst.ai2_belief = rc::DoorBelief(s0, p);
    inst.ai2_initialized       = true;
    inst.is_bearing_hypothesis = true;
    inst.hypothesis_azimuth    = azimuth;   // Orient affordance target yaw = the bearing to look toward
    // Write the room-frame read-back into the model so the scene-graph publish / viewer show it on the ray.
    refresh_geometry(inst);
}

void DoorFit::compute_chain_cov(rc::DoorFitState& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    inst.chain_cov_xy = 0.0f;
    if (not cfg_.rt_cov_add_chain or masks_ == nullptr)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the door centre: transform it to the measurement frame,
    // then back to room with ZERO input cov — InnerGaussianAPI returns exactly the chain contribution,
    // pinned to the mask capture stamp. ★Over the chain the snapshot recorded WITH the packet whose stamp this is
    // (last_mask_timestamp_ms is set from that very packet in observe(), on this same fresh cycle).
    const auto& s = inst.model.state();
    const auto g = chain_point_cov(masks_->packet().room_zed_chain, Eigen::Vector3d(s.cx, s.cy, s.cz));
    if (not g.has_value())
        return;
    inst.chain_cov_xx = static_cast<float>((*g)(0, 0));
    inst.chain_cov_yy = static_cast<float>((*g)(1, 1));
    inst.chain_cov_xy = static_cast<float>((*g)(0, 1));
}

// ─── Instance lifecycle ──────────────────────────────────────────────────────

void DoorFit::init_instance(rc::DoorFitState& inst, std::uint64_t id, std::string name, rc::DoorState init_state)
{
    // Tracker birth seed: a freshly born node's room→door RT may not compose this cycle, so prefer the
    // detection XY the tracker handed us (consumed once) over a possibly-0,0 RT read.
    if (auto it = birth_seeds_.find(id); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        birth_seeds_.erase(it);
    }

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF; replace any non-finite field
    // with a safe default before it reaches the model.
    {
        const auto fix = [&](float& v, float fallback, const char* field)
        {
            if (not std::isfinite(v))
            {
                say(std::format("door_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", field, name, fallback));
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

    inst = rc::DoorFitState{};
    inst.node_id   = id;
    inst.node_name = std::move(name);
    inst.model     = rc::DoorModel(init_state, make_model_params());
    // Seed the cached geometry from the published box: the belief is not initialised yet, but observe()
    // needs a valid leaf_pose for its candidate/residual split on this very cycle.
    refresh_geometry(inst);
}

// ─── Observation ─────────────────────────────────────────────────────────────

DoorObservation DoorFit::observe(rc::DoorFitState& inst)
{
    DoorObservation observation;

    // Detection-aliveness ages every cycle; a fresh door mask below resets it to 0.
    if (inst.frames_since_detection < 1000000) ++inst.frames_since_detection;

    // Primary path: YOLO "masks" (room frame), masks-only. Classify-don't-destroy SDF split keeps
    // inliers as candidates and the rest as residuals that drive model expansion.
    if (masks_ == nullptr)
        return observation;
    const auto& masks_packet = masks_->packet();
    if (masks_packet.valid and masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance = ONLY the §3.1 gated assignment. No greedy-nearest fallback: when this
        // instance has no association this frame (its door occluded / another instance won the slice), it
        // must FREEZE, not grab the nearest door's mask — that fallback teleported a far-born instance onto
        // a near door when its own door was momentarily undetected (→ merge, so the far door never
        // persisted). Freeze-on-no-association is the information-filter axiom; mirrors table_concept.
        std::optional<MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[static_cast<std::size_t>(inst.assigned_mask_idx)];
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this door on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_detection_wall_ms = t_wall_ms_;   // Task D-C: the age of what the identity row will remember
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
            if (slice.source == 0)   // Task D-C review C1: the mask's sides (DoorObservation::mask_cols)
            {
                const std::size_t pb = std::min<std::size_t>(slice.pixel_begin, masks_packet.mask_pixels.size());
                const std::size_t pe = std::min<std::size_t>(slice.pixel_end, masks_packet.mask_pixels.size());
                if (pe > pb)
                {
                    float c0 = std::numeric_limits<float>::infinity(), c1 = -std::numeric_limits<float>::infinity();
                    for (std::size_t i = pb; i < pe; ++i)
                    { c0 = std::min(c0, masks_packet.mask_pixels[i].x()); c1 = std::max(c1, masks_packet.mask_pixels[i].x()); }
                    observation.mask_cols = std::make_pair(c0, c1);
                }
            }
            const std::size_t begin = std::min<std::size_t>(slice.support_begin, masks_packet.support_points.size());
            const std::size_t end = std::min<std::size_t>(slice.support_end, masks_packet.support_points.size());

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
                // ★The claim set is the DoorCore's ONE per cycle (Task 4b: the worker's second graph walk is gone).
                if (claims_ != nullptr and rc::exclusion::explained_by_other(p.x(), p.y(), p.z(), *claims_))
                {
                    // NOT gated on a verbose flag: this is the one line that says whether the rule is running at
                    // all, which is the first thing anyone asks. Throttled rather than silenced.
                    if ((n_explained_away_++ % 200) == 0)
                        say(std::format("[{}] [exclusion] point dropped — already explained by another object "
                                        "({} so far this run)\n", inst.node_name, n_explained_away_));
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

            if (not observation.candidate_pts.empty() or not observation.residual_pts.empty())
            {
                const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
                observation.explanation_ratio = total > 0.0f
                    ? static_cast<float>(observation.candidate_pts.size()) / total : 0.0f;

                inst.last_masks_frame_seen = masks_packet.frame_id;

                if (should_log(inst))
                    say(std::format("[{}] masks={} label='{}' conf={:.2f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                                    inst.node_name, masks_packet.frame_id, slice.label, slice.confidence,
                                    end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                                    slice.centroid.x(), slice.centroid.y(), slice.centroid.z()));
                return observation;
            }
        }
    }

    // (Was: a fallback reading candidate/residual points written directly on the door node — last_sensing_frame_att,
    // candidate_pts_att. No agent writes them on a door node, so `last_frame > last_frame_seen` was always -1 > -1:
    // no fresh data. That is what this returns.)
    observation.has_fresh_data = false;
    return observation;
}

bool DoorFit::should_log(const rc::DoorFitState& inst) const
{
    const int period = std::max(1, cfg_.door_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── AI2 inference (shared recursive-Laplace belief; mirrors table run_inference) ───────────────
float DoorFit::run_inference(rc::DoorFitState& inst, const DoorObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy belief init: associate the door to the room-polygon wall it was seen through (fixes yaw / lateral / floor),
    // then seed the along-wall offset s from the mask centroid. w,h start at their strong template priors;
    // the belief works in the wall frame θ=[s,w,h].
    if (not inst.ai2_initialized)
    {
        // RE-ACQUISITION first: this node is a door that was removed and has now come back at the same place,
        // so it resumes the belief it had converged to (state + Σ) rather than cold-starting at the template
        // priors and re-fitting from scratch. See DoorFit::note_reacquire.
        if (const auto rs = restore_seeds_.find(inst.node_id); rs != restore_seeds_.end())
        {
            inst.ai2_belief = rs->second;
            inst.ai2_initialized = true;
            resign_leaf_llrs(inst);   // M-1
            restore_seeds_.erase(rs);
            say(std::format("door_concept: [{}] RE-ACQUIRED — resuming converged belief (s={:.2f} w={:.2f} h={:.2f})\n",
                            inst.node_name, inst.ai2_belief.state().s, inst.ai2_belief.state().w, inst.ai2_belief.state().h));
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
        // ★Task D-C (rtD1i door_5): ONLY POINTS THAT WERE SEEN VOTE BY THEIR SIGHTLINE. With no mask points (a door RESTORED
        //  from memory, a carried door) the "point" is the remembered place, and a ray from wherever the camera stands now
        //  to it is not a sightline anything was seen along: re-entering room_1 behind the partition, that ray crossed the
        //  partition first and every restore of DOOR_0 was initialised IN THE PARTITION (yaw 1.547 for a remembered −3.20),
        //  which room_concept then reused as its aperture. With no observation the place is the only evidence: the door
        //  belongs to the wall it lies on (the nearest-wall rule).
        std::optional<Eigen::Vector2f> cam_xy;
        if (room_T_zed_.has_value() and npts > 0)
            cam_xy = Eigen::Vector2f(static_cast<float>(room_T_zed_->coeff(0, 3)), static_cast<float>(room_T_zed_->coeff(1, 3)));
        rc::DoorBeliefParams p = make_belief_params();
        rc::DoorBeliefState s0;
        s0.w = inst.seed_wh_from_box ? m.w : cfg_.door_prior_w_m;
        s0.h = inst.seed_wh_from_box ? m.h : cfg_.door_prior_h_m;
        inst.seed_wh_from_box = false;
        // Wall association: the belief's centre = O + (s + w/2)·u, so seed s so the panel centre lands on the
        // detection centroid. Off the wall (no polygon) → keep the identity frame and s = along-x of centroid.
        WallFrame wf = associate_wall(pts_xy, cam_xy);
        if (npts == 0 and std::isfinite(m.ap_yaw))   // review M-1: a remembered door's pose (place AND yaw) picks its wall
            if (const auto r = door::associate_wall_remembered(room_polygon_, centroid_xy, m.ap_yaw, 0.5f * s0.w,
                                                               room_band().wall_thickness_m); r.ok)
            { wf.O = r.O; wf.u = r.u; wf.len = r.len; wf.ok = true; }
        if (wf.ok)
        {
            p.wall_O = wf.O; p.wall_u = wf.u; p.wall_len = wf.len;
            s0.s = (centroid_xy - wf.O).dot(wf.u) - 0.5f * s0.w;   // not clamped to the run: see apply_constraints
            s0.d = p.recess_prior_m;   // 0: see DoorBeliefParams::recess_prior_m
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
            say(std::format("door_concept: [{}] leaf side inherited from the other room: hinge {} swing {:+.0f} (opens {} this room)\n",
                            inst.node_name, p.leaf.hinge == door::HingeSide::Near ? "near" : "far", p.leaf.swing,
                            p.leaf.swing > 0.0f ? "INTO" : "OUT OF"));
            inst.leaf_side_hint.reset();
        }
        inst.ai2_belief = rc::DoorBelief(s0, p);
        inst.ai2_initialized = true;
        resign_leaf_llrs(inst);   // M-1
    }

    if (observation.has_fresh_data)
    {
        rc::support_bank::ingest(inst, observation.candidate_pts, observation.residual_pts, cfg_);
        // The glance paid off: a real depth mask arrived → this is no longer a bearing-only hypothesis. The
        // belief update below collapses the broad along-ray Σ to the observed position; the affordance
        // switches from Orient to the normal one next cycle.
        inst.is_bearing_hypothesis = false;
    }

    const auto now = now_;   // ★the cycle's clock (door_fit.h): was steady_clock::now(), read here mid-cycle

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
    // …split by mechanism, and stamped as computed from THIS frame's mask — see DoorFitState.
    inst.dbg_trunc_gated = trunc_gated; inst.dbg_motion_gated = motion_gated; inst.dbg_gate_fresh = true;
    compute_chain_cov(inst);

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated cycle takes no measurement)
    if (gated)
        inst.ai2_belief.predict();
    else
    {
        // ★Task D-C round 2: ONE VIEW, ONE OBSERVATION — FOR THE WHOLE FRAME (DoorFrame::weight). The mask of a parked camera
        //  is the same mask every frame (the sim tapes: its midpoint column varies by 0.00-0.10 px while parked), seen through
        //  the same pose error, so re-adding it each frame is the same observation counted hundreds of times. Fix round 1
        //  credited only the mask's SIDES by novelty and still added every frame's POINTS at full weight: rtD1i counted ~40
        //  frames of one parked view taken while room_concept's heading was 8.7° wrong (it published ~1°) as 40 independent
        //  looks — s settled 0.46 m off at σ 0.067 and, with no process noise on s, never recovered; the later right views'
        //  evidence went into the registration offset b instead. Each frame is credited by its NOVELTY 1 − sim against the
        //  last credited view, sim the view similarity door_view_pool.h uses (bearing from the door in units of its angular
        //  half-size, range in units of its half-diagonal): a still camera adds nothing, a camera that moved by a door's
        //  half-size adds a full observation. A frame without a camera pose cannot be placed in a view and counts once.
        float credit = 1.0f;
        std::optional<Eigen::Vector2f> view_cam;
        if (room_T_zed_.has_value())
        {
            view_cam = Eigen::Vector2f(static_cast<float>(room_T_zed_->coeff(0, 3)), static_cast<float>(room_T_zed_->coeff(1, 3)));
            const Eigen::Vector2f dc = inst.aperture.centre_xy();
            const float Rd = std::max(1e-3f, 0.5f * std::hypot(inst.aperture.w, inst.aperture.h));
            float novelty = 1.0f;
            if (inst.bearing_view_cam.has_value())
            {
                const Eigen::Vector2f v0 = *inst.bearing_view_cam - dc, v1 = *view_cam - dc;
                const float d1 = std::max(1e-3f, v1.norm());
                const float half = std::max(std::atan2(Rd, d1), 1e-3f);
                const float db = std::remainder(std::atan2(v1.y(), v1.x()) - std::atan2(v0.y(), v0.x()), 2.0f * std::numbers::pi_v<float>);
                const float dr = d1 - v0.norm();
                novelty = 1.0f - std::exp(-std::min((db / half) * (db / half) + (dr / Rd) * (dr / Rd), 60.0f));
            }
            // Credit only the novelty not yet credited since the anchor view (as door_view_pool.h's `counted`), so a slow
            // drift adds up to one observation per door-size of motion instead of vanishing frame by frame; once the view
            // is wholly new the anchor moves there.
            credit = std::max(0.0f, novelty - inst.bearing_view_credit);
            if (not inst.bearing_view_cam.has_value() or novelty >= 1.0f - 1e-3f)
            { inst.bearing_view_cam = view_cam; inst.bearing_view_credit = 0.0f; }
            else
                inst.bearing_view_credit = std::max(inst.bearing_view_credit, novelty);
        }
        inst.dbg_view_credit = credit;
        rc::DoorFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.weight = credit;
        frame.R.assign(frame.points.size(), credit > 0.0f ? R / credit : R);
        // Task D-C review C1 / round 2: the mask's sides (DoorFrame::bearings, rc::MaskBearings) — the ZED pinhole and the
        // cycle's camera pose. A side within one mask cell (6 px, door_existence.cpp MaskCells::CELL) of the image border
        // is CENSORED — the door may continue past it. Variance per side (px²): the per-point position noise R at the door's
        // range, carried to the image (fx²·R/r²), plus the column's own quantisation (1/12 px²), independent between the
        // two sides; and the pose chain at the door perpendicular to the ray (fx²·σ⊥²/r²), ONE error moving both sides — the
        // fix-round-1 variance had no pose term at all. Both scaled by the frame's view credit, as everything it says is.
        if (credit > 0.0f and observation.mask_cols.has_value() and room_T_zed_.has_value() and cam_.present
            and cam_.projection_model == 0 and cam_.width > 0 and cam_.focal_x > 0.0f)
        {
            rc::MaskBearings mb;
            mb.zed_T_room = room_T_zed_->inverse().topRows<3>().cast<float>();
            mb.fx = cam_.focal_x;
            mb.cx = static_cast<float>(static_cast<int>(static_cast<float>(cam_.width / 2u)));   // camera_project's centre
            mb.jamb_depth = inst.wall_reg.ok ? inst.wall_reg.depth : 0.0f;
            constexpr float kCellPx = 6.0f;
            const float W = static_cast<float>(cam_.width);
            const auto [c0, c1] = *observation.mask_cols;
            mb.lo = c0 > kCellPx ? c0 : std::numeric_limits<float>::quiet_NaN();
            mb.hi = c1 < W - 1.0f - kCellPx ? c1 : std::numeric_limits<float>::quiet_NaN();
            const Eigen::Vector2f ray = inst.aperture.centre_xy() - *view_cam;
            const float brange = std::max(0.3f, ray.norm());
            const Eigen::Vector2f perp = Eigen::Vector2f(-ray.y(), ray.x()) / std::max(1e-6f, ray.norm());
            const float chain_perp = perp.x() * perp.x() * inst.chain_cov_xx + 2.0f * perp.x() * perp.y() * inst.chain_cov_xy
                                   + perp.y() * perp.y() * inst.chain_cov_yy;
            const float px_per_m = mb.fx / brange;
            mb.var = (px_per_m * px_per_m * R + 1.0f / 12.0f) / credit;
            mb.var_common = px_per_m * px_per_m * std::max(0.0f, chain_perp) / credit;
            frame.bearings = mb;
        }
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
        const float chain_s = u.x() * u.x() * inst.chain_cov_xx + 2.0f * u.x() * u.y() * inst.chain_cov_xy
                            + u.y() * u.y() * inst.chain_cov_yy;   // proj onto u
        // Task D-C round 2: the across-wall projection for d (DoorFrame::chain_cov_n).
        const float chain_n = u.y() * u.y() * inst.chain_cov_xx - 2.0f * u.x() * u.y() * inst.chain_cov_xy
                            + u.x() * u.x() * inst.chain_cov_yy;
        frame.chain_cov_s = chain_s + range_lat_var + mot_pos_var;
        frame.chain_cov_n = chain_n + range_lat_var + mot_pos_var;
        // try96: the mask is displaced by the same registration offset b as the LiDAR (one pose registers both): the pose
        // chain + range term is b's (door_opening_edges.h), so the frame keeps only its own motion smear, and the fit is
        // re-expressed as a measurement of s + b (DoorBelief::update_sharing_offset).
        const bool share_offset = inst.along_wall.init;
        if (share_offset)
            frame.chain_cov_s = mot_pos_var;   // (b is ALONG the wall only: the across common mode keeps the chain)
        // ★NO FREEZE SWITCH HERE. What used to gate this update is now stated in the generative model
        // instead: the aperture's along-wall process noise is ZERO (door_belief.h process_std_s), so its
        // precision grows with evidence and its mean stops being dragged by a swinging leaf without any
        // moment at which the code changes behaviour. A boolean freeze also had a defect this does not:
        // it keyed on model_stable, whose convergence test INCLUDES phi, so the aperture unfroze exactly
        // when the leaf started moving — the one time it most needed to hold still.
        // Task 15: the mask is explained under the HINGE MARGINAL — the leaf from the believed jamb with P = σ(hinge_llr), and
        // from the other — so a wrong hinge cannot slide the aperture a width to make its leaf fit (try75 leg 1).
        inst.ai2_belief.set_hinge_prob(1.0f / (1.0f + std::exp(-inst.hinge_llr)));
        inst.ai2_belief.set_leaf_phi_sigma(inst.phi_sigma);   // …and under the leaf ANGLE's own uncertainty
        if (credit <= 0.0f)
            inst.ai2_belief.predict();   // nothing new from this view: the belief carries its one-step Q only
        else if (share_offset)
            inst.ai2_belief.update_sharing_offset(frame, inst.along_wall.b, inst.along_wall.Pb, inst.along_wall.c);
        else
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
    inst.dbg_resid_pts = static_cast<int>(observation.residual_pts.size());   // HELD, see door_fit_state.h

    // Write belief → cached aperture/leaf geometry + room-frame DoorState (the single authoring point).
    const auto& bs = inst.ai2_belief.state();
    const Eigen::Vector2f c = inst.ai2_belief.center_xy();
    refresh_geometry(inst);
    const rc::DoorState& ms = inst.model.state();

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;
    compute_projected_roi(inst);

    if (should_log(inst))
        say(std::format("[{}] AI2 npts={} clutter={:.0f}% R={:.4f} range={:.2f} trunc={:.2f}{} | FE={:.2f} base={:.2f} surprise={:.2f} | s={:.2f} w={:.2f} h={:.2f} @({:.2f},{:.2f}) ψ={:.2f}\n",
                        inst.node_name, npts, 100.0f * inst.last_clutter_frac, R, range, inst.last_trunc_frac, gated ? " GATED" : "",
                        energy, inst.fe_baseline, inst.fe_surprise,
                        bs.s, bs.w, bs.h, c.x(), c.y(), ms.yaw));

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

void DoorFit::log_ai2_csv(const rc::DoorFitState& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty() or not diag_)
        return;
    // Formatted through the classic locale, as the old ofstream (global C++ locale "C") formatted it.
    std::ostringstream row;
    row.imbue(std::locale::classic());
    const auto& s = inst.ai2_belief.state();
    const Eigen::Vector2f c = inst.ai2_belief.center_xy();   // APERTURE centre (see DoorBelief) — unchanged by phi
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    row << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
        << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_trunc_frac << ',' << inst.last_range << ',' << inst.last_clutter_frac << ','
        << s.s << ',' << s.w << ',' << s.h << ','
        << c.x() << ',' << c.y() << ',' << inst.ai2_belief.yaw() << ','
        << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << inst.leaf.phi << ','
        << inst.dbg_leaf_pts << ',' << inst.dbg_w_sdf << ',' << inst.dbg_w_mask << ','
        << inst.dbg_w_edge << ',' << inst.phi_sigma << ',' << (inst.phi_measured ? 1 : 0) << ','
        << inst.dbg_leaf_off_plane << ',' << inst.dbg_phi_cv << ','
        << inst.dbg_ray_hit0 << ',' << inst.dbg_ray_expl0 << ','
        << inst.dbg_ray_signed0 << ','
        << inst.dbg_phi_ninf << ',' << inst.dbg_phi_beta << ',' << inst.dbg_phi_lat << ','
        << inst.ai2_belief.recess() << ',' << inst.ai2_belief.recess_sd() << '\n';
    diag_({"ai2_log.csv", row.str()});
}

// One row per estimate_phi() call — EVERY call, mask or not. The ai2 row above is written only on the mask
// path, so the LiDAR-only cycles (no detection: exactly the ones where the leaf angle collapsed in try42)
// left no trace. Path = the ai2 path with "_phi" before the extension; off when that is off.
void DoorFit::log_phi_csv(const rc::DoorFitState& inst, const rc::ai::LidarRays& rays,
                          const std::vector<std::pair<float, float>>& loglik,
                          float prior_mean, float prior_sigma, float post_mean, float post_sigma, bool measured)
{
    if (cfg_.ai2_csv_path.empty() or not diag_)
        return;
    std::ostringstream row;
    row.imbue(std::locale::classic());
    const Eigen::Vector3f origin = leaf_sweep_present() ? sweep_->origin() : Eigen::Vector3f::Zero();
    // t_ms: the cycle's wall clock (the input's t_wall_ms) — was system_clock::now() read here (diagnostic column).
    row << t_wall_ms_ << ',' << inst.node_name << ',' << origin.x() << ',' << origin.y() << ','
        << rays.endpoints.size() << ',' << inst.dbg_leaf_off_plane << ',' << inst.dbg_phi_ninf << ','
        << inst.dbg_phi_beta << ',' << inst.dbg_phi_lat << ',' << prior_mean << ',' << prior_sigma << ','
        << post_mean << ',' << post_sigma << ',' << (measured ? 1 : 0) << ','
        << inst.dbg_ray_hit0 << ',' << inst.dbg_ray_expl0 << ',' << inst.dbg_ray_signed0 << ','
        << inst.ai2_belief.state().s << ',' << std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(0, 0))) << ','
        << (inst.dbg_reg.ok ? 1 : 0) << ',' << inst.dbg_reg.eps << ',' << inst.dbg_reg.depth << ','
        << inst.dbg_reg.n_inliers << ',' << inst.dbg_along.n_runs << ',' << inst.dbg_along.z << ','
        << inst.dbg_along.R << ',' << inst.dbg_along.nu << ',' << inst.dbg_along.beta << ','
        << inst.dbg_along.p_detect << ',' << inst.along_wall.b << ','
        << std::sqrt(std::max(0.0f, inst.along_wall.Pb)) << ','
        << inst.hinge_llr << ',' << inst.swing_llr << ',' << inst.leaf_channel.views.views.size() << ','
        << inst.leaf_channel.views.epoch << ',' << inst.dbg_leaf_p_ok << ',' << inst.dbg_leaf_pi_move << ','
        << inst.leaf_channel.hazard() << ',';
    for (std::size_t k = 0; k < loglik.size(); ++k)
        row << (k ? ";" : "") << loglik[k].second;
    row << '\n';
    diag_({"ai2_log_phi.csv", row.str()});
}

// Combined ego-motion magnitude (m/s): the per-mask corruption speed OR'd with the robot's own measured speed
// (linear + a lever-arm conversion of angular), so it works whether or not the producer populated motion_dotd.
float DoorFit::motion_magnitude(const rc::DoorFitState& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear/distortion), → 1 at
// centroid radius ai2_periph_ref. This is the "well-centred masks stay trustworthy" lever, expressed continuously.
float DoorFit::periphery_penalty(const rc::DoorFitState& inst) const
{
    const float ref = std::max(1e-6f, cfg_.ai2_periph_ref);
    const float r = std::max(0.0f, inst.last_centroid_radius) / ref;
    return std::clamp(r * r, 0.0f, 1.0f);
}

bool DoorFit::confirm_only(const rc::DoorFitState& inst) const
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
void DoorFit::estimate_phi(rc::DoorFitState& inst)
{
    // ★CLEAR FIRST, BEFORE ANY EARLY RETURN. The curve is consumed by the silhouette channel in the
    // SAME cycle; leaving last cycle's behind on a frame that could not score phi would marginalise
    // this frame's absence over a stale posterior — evidence carried forward under the name of a
    // measurement that was never taken. An empty curve is the honest state and the consumer handles it.
    inst.phi_curve.clear();
    inst.phi_curve_log.clear();
    if (not inst.ai2_initialized)
        return;
    // ★NO CAMERA PRECONDITION. This used to return when the camera, its pose or the mask stream was
    // missing — so a door out of the camera's view could not have its leaf angle estimated at all, even
    // with LiDAR rays passing straight through its aperture. The leaf angle is a LiDAR quantity now.

    // Where the command says the leaf should be right now. Pure prediction from our own request.
    const auto now = now_;   // ★the cycle's clock (door_fit.h): was steady_clock::now(), read here mid-cycle
    float phi_prior = inst.phi_est;
    bool  have_prior = false;
    // ★HOW FAR COULD THE LEAF HAVE SWUNG SINCE WE LAST LOOKED? That, and nothing else, is the width of
    // the prior. A door is a passive object: between two frames it moves by at most (its own maximum
    // rate) x (the elapsed time), whether the mover is our actuator or a person. So sigma comes from the
    // rate the provider advertised — or the same generic 2 rad/s the step limiter below already uses —
    // times a MEASURED interval, not a per-cycle constant. Long gap, wide prior; fast loop, tight one;
    // first ever look, no prior at all.
    // ★ try102 (2026-10-19): THAT WIDTH IS NOW ONLY THE COMMANDED SWING'S. Without a command the leaf is HELD and moved at
    //   an inferred hazard (rc::door::leaf_channel_step): the generic 2 rad/s diffusion that stood here widened σ by
    //   0.2 rad per unmeasured cycle, so a parked robot 5.5 m out forgot an open door in ~1.5 s — phi read the uniform's
    //   mean 60° and P(passable) its mass above 72.5° (0.33-0.37) for 40 s, twice (ai2_log_phi.csv rows 4264-4301).
    const float swing_rate = inst.phi_cmd_rate > 1e-3f ? inst.phi_cmd_rate : 2.0f;   // the COMMANDED motion's rate
    const float dt_phi = inst.phi_last_t.time_since_epoch().count() == 0
                       ? std::numeric_limits<float>::infinity()
                       : std::chrono::duration<float>(now - inst.phi_last_t).count();
    inst.phi_last_t = now;
    // While our own command runs: the commanded trajectory, its width compounding with the swing it could have made in dt
    // (never tighter than the grid resolves, never wider than the range — rc::door::phi_prior_sigma, 2026-09-13).
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
    // (Task 11 fix round 1: one definition, rc::door::phi_grid, shared with the passage anchor's φ curve.)
    const auto grid_phi = rc::door::phi_grid(cfg_.phi_max_rad, cfg_.phi_step_rad);
    const float PHI_MIN = grid_phi.lo;
    const float PHI_MAX = grid_phi.hi;
    const int   NSTEP   = grid_phi.n;

    // The low-LiDAR sweep (core/sensing.h LeafSweep): the newest room-frame sweep and its sensor.
    const std::vector<Eigen::Vector3f>* leaf_pts = leaf_sweep_present() ? &sweep_->points() : nullptr;
    const Eigen::Vector3f leaf_origin = leaf_sweep_present() ? sweep_->origin() : Eigen::Vector3f::Zero();

    // ★THE ALONG-WALL SLIDE IS A NUISANCE OVER ITS HONEST SUPPORT (+-w/2), and the ray selection follows it
    // (door_phi_posterior.h has the why: a carried mirror sat 0.23-0.40 m from the real doorway).
    rc::DoorBelief::FieldNuisance nz = rc::door::phi_field_nuisance(inst.aperture);
    // ★THE SLIDE'S PRIOR IS WHERE THE BELIEF PUTS THE APERTURE (2026-10-13): N(0, var_s) from the aperture's own
    // posterior. A fitted door (σ_s ≈ 6-11 cm) cannot be "explained" by a shut leaf slid 0.55 m onto the wall beside it
    // (try44: every collapse to phi≈0 had lat = -0.55); a freshly carried mirror, whose placement IS uncertain, keeps
    // the wide support that ±w/2 was introduced for.
    if (inst.ai2_initialized)
        nz.lat_sigma = std::sqrt(std::max(1e-8f, inst.ai2_belief.covariance()(0, 0)));   // 0 = known, never "uniform"
    // try97: the room the sensor stands in is part of the scene — its walls, less the door's own (rc::door::room_walls_beside).
    // The field explains returns on them under every leaf hypothesis (so the selection keeps returns short of the doorway
    // plane: a leaf swung toward the sensor is seen, and a corridor wall is not taken for one), and the registration
    // attributes them to their own wall instead of the door's.
    if (has_room_polygon())
    {
        nz.room_walls = rc::door::room_walls_beside(inst.aperture, room_polygon_, nz.depth_lo, nz.depth_hi);
        // Task D-C: the room walls are placed with the polygon face's own uncertainty (Existence.BoundarySigmaM — the σ the
        // wall-band prior reads), a common-mode offset per wall marginalised in the field, as the door's own wall's depth is.
        nz.room_wall_sigma = cfg_.exist_boundary_sigma_m;
        std::tie(nz.wall_seg_lo, nz.wall_seg_hi) = rc::door::door_wall_segment(inst.aperture, room_polygon_, nz.depth_lo, nz.depth_hi);
    }
    rc::ai::LidarRays rays;
    // ★try96: THE SWEEP IS REGISTERED TO THE WALL IT SEES before any doorway likelihood (door_opening_edges.h). A heading /
    //   across-wall error of the room's pose misregisters the cloud in the room frame (try96: 14-20°, 0.3-0.45 m for ~30 s ⇒
    //   the hole appeared 1.4 m along the wall); the rotation and the across offset are measured from the wall's own
    //   returns beside the opening and undone about the sensor. Unregistrable (few wall returns) ⇒ the sweep as given.
    rc::door::RegisteredSweep reg_sweep;
    inst.dbg_reg = {};
    if (leaf_pts and not leaf_pts->empty() and inst.ai2_initialized)
    {
        reg_sweep = rc::door::register_sweep(inst.ai2_belief.aperture(), leaf_origin, *leaf_pts, nz.room_walls);
        inst.dbg_reg = reg_sweep.reg;
        // The model's copy (M-6) holds the last registration while nothing contradicts it — and is RESET when a sweep of
        // this door arrived, its field of view COVERED the wall window beside the aperture, and the wall did NOT register
        // (fix round 2): the wall stopped being registrable, so an old offset is no longer evidence. ★ A sweep that does not
        // cover the window (robot turned away) looked at nothing about it ⇒ held, like no sweep at all.
        if (reg_sweep.reg.ok or reg_sweep.reg.wall_in_view)
            inst.wall_reg = reg_sweep.reg.ok ? reg_sweep.reg : door::WallRegistration{};
    }
    // ★Task 5b: THE FIELD RUNS IN THE WALL-REGISTERED FRAME, the leaf plane's depth d in the ROOM frame. Registration moved
    // the sweep by reg.depth along across_u so the wall it sees lies on the polygon line; d is measured from that line in
    // the room frame (the mask's frame), so the part of d that is the wall's own registration offset is already undone in
    // the sweep. In the registered frame the leaf stands d + reg.depth beyond the wall line (reg.depth > 0 is INTO the
    // room, d > 0 out of it). One plane, expressed in the frame each sensor is in — not a second estimate.
    rc::DoorBelief field_belief = inst.ai2_belief;
    if (reg_sweep.reg.ok)
    {
        rc::DoorBeliefState st = field_belief.state();
        st.d += reg_sweep.reg.depth;
        field_belief.set_state(st);
    }
    if (leaf_pts and not leaf_pts->empty() and inst.ai2_initialized)
    {
        // ★SELECT BY THE PATH, NOT BY THE ENDPOINT — a slab through the aperture plus the jamb band; see
        // rc::door::select_doorway_rays (the selection used to live here, unchanged but for the slide).
        auto sel = rc::door::select_doorway_rays(inst.aperture, reg_sweep.origin, reg_sweep.pts,
                                                 std::max(std::abs(nz.lat_lo), std::abs(nz.lat_hi)),
                                                 cfg_.lidar_bpearl_precision);
        rays = std::move(sel.rays);
        // ★THE DIAGNOSTIC, TAKEN BEFORE ANY MODEL CHANGE. Tolerance is 3 sigma of the range noise: a ray
        // that stops within that of where a SHUT leaf would be is explained by one. On a closed door this
        // ratio should be near 1; anything near 0 says the selection is not looking at the leaf.
        {
            const auto ex = field_belief.phi_ray_explain(rays, 0.0f, 3.0f * rays.robust_c_m);
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
    // ★THE HINGE IS MARGINALISED (Task 11, try69): the leaf is also scored hung from the OTHER jamb, the curve below is
    // the hinge mixture, and the hinge's own log-odds accumulates — see rc::door::phi_hinge_marginal for why one config-
    // pinned hinge made an open door read shut from room_2. The geometry then hangs from the MAP hinge.
    // ⚑ Fix round 1 (cost): the second jamb doubles phi_field_loglik, so it is scored every cycle only while the hinge
    // MATTERS (hinge_question_live: the passage anchor serves this door, or one of its affordances is claimed) and on a
    // stride of kHingeStride cycles otherwise — a compute budget, not a model level. The log-odds accumulates, so a
    // stride only slows learning the hinge for a door nobody is about to use.
    constexpr int kHingeStride = 5;
    nz.other_hinge = inst.hinge_question_live or (inst.phi_calls++ % kHingeStride == 0);
    // Task 20: the SWING is marginalised the same way and on the same budget (rc::door::phi_leaf_marginal) — 4 hypotheses
    // when both are scored. ⚑ Compute budget, as above: same stride, not a model level.
    nz.other_swing = nz.other_hinge;
    std::vector<std::pair<float, float>> ll_this, ll_other, ll_swing, ll_both;
    if (not rays.endpoints.empty())
    {
        auto fr = field_belief.phi_field_loglik(rays, PHI_MIN, PHI_MAX, NSTEP, nz);
        inst.dbg_phi_ninf  = fr.n_inf;
        inst.dbg_phi_beta  = fr.beta;
        inst.dbg_phi_lat   = fr.lat_map;
        // ★try96: the field's slide marginal is NO LONGER fused into s. Task 15 fused it every cycle as an independent
        //   measurement in the room frame; under a 30-s localisation error the frames of that stretch, sharing ONE pose
        //   error, were counted as hundreds of observations and dragged the aperture 1.1 m along its wall (and, past
        //   ±w/2, out of reach of the true hole for ever). It also integrates every leaf hypothesis uniformly — the
        //   other-hinge / other-swing mirrors that "fit at many offsets". s now takes its LiDAR evidence from the opening's
        //   through-span with a persistent registration offset (below, after the leaf posterior). fr.lat_loglik stays a
        //   nuisance of the φ curve only.
        ll_this  = std::move(fr.loglik);
        ll_other = std::move(fr.loglik_other);
        ll_swing = std::move(fr.loglik_swing);
        ll_both  = std::move(fr.loglik_both);
    }
    else
    {
        inst.dbg_phi_ninf = 0.0f;
        inst.dbg_phi_beta = 1.0f;
    }
    // ★ Fix round 1: THE PASSAGE ANCHOR'S BEAMS, as a φ likelihood on this same grid (door_anchor_open.h), ADDED — never
    //   fused into P(passable): the passability marginalisation stays passable_logit's, over the joint posterior. Each
    //   anchor frame is consumed once. Its hinge axis is the one it was computed under; swapped if the MAP jamb flipped.
    if (inst.anchor_phi_ll.size() == static_cast<std::size_t>(NSTEP)
        and inst.anchor_phi_ll_other.size() == static_cast<std::size_t>(NSTEP))
    {
        const bool same = inst.anchor_phi_hinge == inst.ai2_belief.params().leaf.hinge;
        const auto& a_this  = same ? inst.anchor_phi_ll : inst.anchor_phi_ll_other;
        const auto& a_other = same ? inst.anchor_phi_ll_other : inst.anchor_phi_ll;
        const bool have_leaf = ll_this.size() == static_cast<std::size_t>(NSTEP);
        const bool have_other = ll_other.size() == static_cast<std::size_t>(NSTEP);
        // Task 20: the anchor frame scores the swing through its own log-odds (below), not through these curves; the
        // field's other-swing curves lack its evidence, so this frame's swing mixture is left to the anchor's log-odds.
        ll_swing.clear();
        ll_both.clear();
        for (int k = 0; k < NSTEP; ++k)
        {
            const float phi = PHI_MIN + (PHI_MAX - PHI_MIN) * static_cast<float>(k) / static_cast<float>(NSTEP - 1);
            if (not have_leaf) ll_this.emplace_back(phi, 0.0f);
            ll_this[static_cast<std::size_t>(k)].second += a_this[static_cast<std::size_t>(k)];
        }
        // The other jamb is joined only when the leaf channel scored it too (or there is no leaf frame at all): a
        // mixture of a two-source curve with a one-source one would weigh the hinges on different evidence.
        if (have_other or not have_leaf)
        {
            if (not have_other)
                for (int k = 0; k < NSTEP; ++k)
                    ll_other.emplace_back(PHI_MIN + (PHI_MAX - PHI_MIN) * static_cast<float>(k) / static_cast<float>(NSTEP - 1), 0.0f);
            for (int k = 0; k < NSTEP; ++k) ll_other[static_cast<std::size_t>(k)].second += a_other[static_cast<std::size_t>(k)];
        }
    }
    inst.anchor_phi_ll.clear();
    inst.anchor_phi_ll_other.clear();
    // ★ Task 12 fix D: the swing's evidence from the anchor's beams (the only channel that scores both swings), consumed once;
    //   its sign is relative to the swing it was computed under. ★ try102: it is THIS FRAME's swing evidence, added to the
    //   field's own for the same frame and contaminated ONCE with it (the anchor and the field read the same sweep), never
    //   accumulated on its own.
    double anchor_swing_delta = 0.0;
    if (std::isfinite(inst.anchor_swing_llr))
    {
        const float cur = inst.ai2_belief.params().leaf.swing;
        anchor_swing_delta = (inst.anchor_phi_swing * cur > 0.0f ? 1.0 : -1.0) * double(inst.anchor_swing_llr);
        inst.anchor_swing_llr = std::numeric_limits<float>::quiet_NaN();
    }
    // ─── THE LEAF CHANNEL (try102, rc::door::leaf_channel_step): hinge × swing × φ for this frame ─────────────────────
    //   ε-contaminated frames, evidence pooled per VIEWPOINT (a parked robot adds to one bounded term), and φ HELD unless
    //   moved (inferred hazard) or commanded. The geometry hangs from the MAP hinge and swings to the MAP side (⚑ P = ½, the
    //   MAP of a binary posterior — a decision level, as before).
    rc::door::LeafChannelInput lin;
    lin.bb = std::move(ll_this); lin.ob = std::move(ll_other); lin.bs = std::move(ll_swing); lin.os = std::move(ll_both);
    lin.anchor_swing_delta = anchor_swing_delta;
    lin.have_sensor = leaf_pts != nullptr;   // leaf_origin is the sweep's sensor, in the room frame the aperture is in
    lin.sensor = leaf_origin.head<2>();
    lin.dt = dt_phi;
    lin.cmd_active = have_prior;
    lin.cmd_mean = phi_prior;
    lin.cmd_sigma = sigma_phi;
    lin.phi_min = PHI_MIN; lin.phi_max = PHI_MAX; lin.nstep = NSTEP;
    lin.view_rad = cfg_.phi_step_rad;
    const auto lr = rc::door::leaf_channel_step(inst.leaf_channel, inst.ai2_belief, inst.hinge_llr, inst.swing_llr,
                                                inst.phi_est, inst.phi_sigma, lin);
    inst.hinge_llr_ref = inst.ai2_belief.params().leaf.hinge;
    inst.swing_llr_ref = inst.ai2_belief.params().leaf.swing;
    inst.dbg_leaf_pi_move = static_cast<float>(lr.pi_move);
    inst.dbg_leaf_p_ok    = static_cast<float>(lr.p_ok);
    if (lr.swing_flipped)
        say(std::format("door_concept: [{}] SWING FLIPPED — the leaf now opens {} the wall's normal (log-odds {:+.1f} over {} "
                        "viewpoints; phi {:.0f} deg)\n", inst.node_name,
                        inst.ai2_belief.params().leaf.swing > 0.0f ? "along" : "against", inst.swing_llr,
                        inst.leaf_channel.views.views.size(), inst.phi_est * 57.2958f));
    if (lr.hinge_flipped)
        // ★ The live check (fix round 1): this line must come BEFORE the aperture's σ_s collapses (σ_s printed here).
        say(std::format("door_concept: [{}] HINGE FLIPPED — now the {} jamb, the leaf is better explained hung there "
                        "(log-odds {:+.1f} over {} viewpoints; phi {:.0f} deg; aperture s {:.2f} m σ_s {:.3f} m)\n",
                        inst.node_name, inst.ai2_belief.params().leaf.hinge == door::HingeSide::Near ? "near" : "far",
                        inst.hinge_llr, inst.leaf_channel.views.views.size(), inst.phi_est * 57.2958f,
                        inst.ai2_belief.state().s, std::sqrt(std::max(0.0f, inst.ai2_belief.covariance()(0, 0)))));
    const auto& post = lr.post;
    inst.phi_curve  = post.curve;
    inst.phi_curve_log = post.log_curve;
    inst.dbg_phi_cv = post.cv;
    inst.dbg_w_sdf  = post.w_lik_best;
    inst.dbg_w_mask = 0.0f;   // kept in the log schema; the camera no longer votes on phi
    inst.dbg_w_edge = 0.0f;

    // ─── THE CARRIED BELIEF: the channel wrote phi_est / phi_sigma — the posterior's MEAN and WIDTH, never an argmax (the
    // argmax of a nearly-flat curve is noise; chasing it slid the aperture along the wall, r = -1.000, 34 cm against 41 deg),
    // never a clamped mean (2026-10-13: a rate-limited mean fought its own posterior, P(passable) 1.00 -> 0.10 -> 1.00).
    inst.phi_measured = post.measured;

    // ─── WHERE ALONG THE WALL: the opening's edges, with the registration offset the frames share (try96) ────────────
    // rc::door::along_wall_update (door_opening_edges.h): the through-span of the REGISTERED sweep is a measurement of
    // s (the aperture contains it), associated against clutter with P(the door shows a span) = E[1 − cos φ] under the
    // posterior just formed, and fused with the scan's along-wall registration offset b as a persistent state.
    // σ0² = the pose chain's along-wall variance + the static range term — the common mode the mask frames carry.
    inst.dbg_along = {};
    if (inst.ai2_initialized and leaf_pts and not leaf_pts->empty())
    {
        const auto ap = inst.ai2_belief.aperture();
        const auto runs = rc::door::measure_openings(ap, reg_sweep.origin, reg_sweep.pts);
        const Eigen::Vector2f u = ap.wall_u;
        const float chain_s = u.x() * u.x() * inst.chain_cov_xx + u.y() * u.y() * inst.chain_cov_yy;
        const float range = (ap.centre_xy() - reg_sweep.origin.head<2>()).norm();
        const float rl = cfg_.ai2_range_noise_lat_per_m * range;
        const float dt_aw = inst.along_wall_t.time_since_epoch().count() == 0
                          ? 0.0f : std::chrono::duration<float>(now - inst.along_wall_t).count();
        inst.along_wall_t = now;
        inst.dbg_along = rc::door::along_wall_update(inst.ai2_belief, inst.along_wall, runs, reg_sweep.reg,
                                                     rc::door::clear_fraction(post.curve), chain_s + rl * rl, range, dt_aw);
    }
    log_phi_csv(inst, rays, lr.frame_loglik, lr.prior_mean, lr.prior_sigma, post.mean, post.sigma, post.measured);
    // phi_support was the camera mask overlap at the claimed angle; -1 = "the camera does not score phi".
    inst.phi_support = -1.0f;

    inst.phi_est = std::clamp(inst.phi_est, PHI_MIN, PHI_MAX);
    inst.leaf.phi = inst.phi_est;
    if (inst.ai2_initialized)
        inst.ai2_belief.set_leaf_phi(inst.phi_est);
}

// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────
// The camera is the snapshot's: room ← zed recorded at the top of the cycle and the zed intrinsics (was the ts=0
// cortex query and DSR::CameraAPI, read here mid-cycle).
void DoorFit::compute_projected_roi(rc::DoorFitState& inst)
{
    inst.roi_valid = false;
    if (not cam_.present)
        return;

    if (not room_T_zed_.has_value())
        return;
    const Eigen::Matrix4d zed_T_room = room_T_zed_.value().inverse();   // room point → camera frame

    const float fx = cam_.focal_x;
    const float fy = cam_.focal_y;
    const float W  = static_cast<float>(cam_.width);
    const float H  = static_cast<float>(cam_.height);
    if (fx <= 0.f or fy <= 0.f or W <= 0.f or H <= 0.f)
        return;
    const float cx_px = W * 0.5f, cy_px = H * 0.5f;

    // Project the LEAF box into the image (door_geometry.h — the ROI must frame the panel the camera actually
    // sees, so this follows the leaf, not the aperture). Camera convention matches the producer:
    // X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    // ★ CLIPPED, NOT SKIPPED (2026-10-18). Corners behind a near plane used to be dropped, and with ≥ 4 left
    //   the bbox of the survivors was used: a corner just in front of the plane has X/Y exploding, so a leaf
    //   swung toward the camera read roi_fill ≈ 3-4 image widths, the detector envelope gave p_detect ≈ 1e-20,
    //   and NO absence was ever charged — misplaced doors after a pose flip (live 10-18: door_2 condemned at
    //   p 0.018 for ever, debounce streak 0). With fewer than 4 the function returned and roi_fill kept a
    //   stale value. Now each of the 12 box edges is clipped to the near plane (and fill is measured
    //   in a camera aimed at the door, see below); nothing in front ⇒ fill 0.
    constexpr double kNear = 0.20;
    std::array<Eigen::Vector3d, 8> P{};
    {
        const auto corners = door::leaf_corners(inst.leaf_pose);
        for (std::size_t i = 0; i < corners.size(); ++i)
            P[i] = (zed_T_room * Eigen::Vector4d(corners[i].x(), corners[i].y(), corners[i].z(), 1.0)).head<3>();
    }
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    const auto add = [&](const Eigen::Vector3d& q)
    {
        const float col = cx_px + static_cast<float>(q.x() / q.y()) * fx;
        const float row = cy_px - static_cast<float>(q.z() / q.y()) * fy;
        min_col = std::min(min_col, col); max_col = std::max(max_col, col);
        min_row = std::min(min_row, row); max_row = std::max(max_row, row);
    };
    for (const auto& q : P) if (q.y() >= kNear) { add(q); ++in_front; }
    for (int i = 0; i < 8; ++i)   // corner k = 4·ix + 2·iy + iz: an edge joins two corners one bit apart
        for (const int bit : {1, 2, 4})
        {
            if (i & bit) continue;
            const Eigen::Vector3d& a = P[i]; const Eigen::Vector3d& b = P[i | bit];
            if ((a.y() >= kNear) == (b.y() >= kNear)) continue;
            const double t = (kNear - a.y()) / (b.y() - a.y());
            add(a + t * (b - a));
        }
    if (in_front == 0)   // the whole leaf is behind the camera: not projectable, nothing to see
    {
        inst.roi_fill = inst.roi_fill_h = inst.roi_fill_v = 0.f;
        return;
    }
    // Centring (door_roi_offset, the base's err_vec) keeps the near-clipped box in the REAL camera, so a door
    // off to one side still says which way to turn; |offset| > 1 ⇒ the door's centre is off the image.
    const float roi_cx = 0.5f * (min_col + max_col);
    const float roi_cy = 0.5f * (min_row + max_row);
    const float off_x = (roi_cx - cx_px) / (0.5f * W);   // [-1,1] on the image, 0 = centred
    const float off_y = (roi_cy - cy_px) / (0.5f * H);

    // ★ FILL IS THE DOOR'S APPARENT SIZE, NOT ITS FOOTPRINT IN WHATEVER THE ZED HAPPENS TO FACE (2026-10-18).
    //   The detector envelope asks "is this door big enough, and not too big, to be segmented at this range?"
    //   — a property of the door and its distance. Read off the live ZED frame it was a property of the GAZE:
    //   live, door_3 (a phantom beside the robot) projected entirely off the image, roi_fill came out 1
    //   (fill_h 0, fill_v 1), p_env 0.0013, and in 21000 cycles its removal streak reached 0.6 while the
    //   panoramic silhouette — the channel that charges the absence, with its own field of view in
    //   in_fov_frac — had 403 of 420 samples on it. So the leaf box is projected into a VIRTUAL camera with
    //   the ZED's intrinsics, aimed at the box centre: the size it would have if looked at.
    Eigen::Vector3d ctr = Eigen::Vector3d::Zero();
    for (const auto& q : P) ctr += q;
    ctr /= static_cast<double>(P.size());
    float fh = 0.f, fv = 0.f;
    if (ctr.norm() > 1e-6)
    {
        const Eigen::Vector3d fwd = ctr.normalized();
        Eigen::Vector3d right = fwd.cross(Eigen::Vector3d::UnitZ());
        if (right.norm() < 1e-6) right = Eigen::Vector3d::UnitX();   // door straight above/below: any roll
        right.normalize();
        const Eigen::Vector3d up = right.cross(fwd);
        float lo_c = 1e9f, hi_c = -1e9f, lo_r = 1e9f, hi_r = -1e9f;
        const auto add_v = [&](const Eigen::Vector3d& q)
        {
            const double X = q.dot(right), Y = q.dot(fwd), Z = q.dot(up);
            const float col = static_cast<float>(X / Y) * fx, row = -static_cast<float>(Z / Y) * fy;
            lo_c = std::min(lo_c, col); hi_c = std::max(hi_c, col);
            lo_r = std::min(lo_r, row); hi_r = std::max(hi_r, row);
        };
        std::array<Eigen::Vector3d, 8> V{};
        for (std::size_t i = 0; i < P.size(); ++i) V[i] = P[i];
        int n_front = 0;
        for (const auto& q : V) if (q.dot(fwd) >= kNear) { add_v(q); ++n_front; }
        for (int i = 0; i < 8; ++i)
            for (const int bit : {1, 2, 4})
            {
                if (i & bit) continue;
                const double ya = V[i].dot(fwd), yb = V[i | bit].dot(fwd);
                if ((ya >= kNear) == (yb >= kNear)) continue;
                add_v(V[i] + (kNear - ya) / (yb - ya) * (V[i | bit] - V[i]));
            }
        if (n_front > 0) { fh = (hi_c - lo_c) / W; fv = (hi_r - lo_r) / H; }
    }
    const bool sane = std::isfinite(off_x) and std::isfinite(off_y) and std::isfinite(fh) and std::isfinite(fv);
    inst.roi_offset_x = std::clamp(sane ? off_x : 0.f, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(sane ? off_y : 0.f, -3.0f, 3.0f);
    inst.roi_fill_h   = sane ? std::clamp(fh, 0.f, 4.f) : 0.f;
    inst.roi_fill_v   = sane ? std::clamp(fv, 0.f, 4.f) : 0.f;
    inst.roi_fill     = std::max(inst.roi_fill_h, inst.roi_fill_v);
    inst.roi_valid    = sane;
}

}  // namespace rc::door::core
