/*
 * chair_fitter.cpp — the active-inference fit core for chair_concept (AI2 recursive-Laplace belief).
 */

#include "chair_fitter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

ChairFitter::ChairFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         ChairConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         ChairSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{}

ChairBeliefParams ChairFitter::make_belief_params() const
{
    ChairBeliefParams p;
    p.tpl_seat_w = cfg_.tracker_birth_seat_w;   // fixed standard-chair template (pose-only belief)
    p.tpl_seat_d = cfg_.tracker_birth_seat_d;
    p.tpl_seat_h = cfg_.tracker_birth_seat_h;
    p.tpl_back_h = cfg_.tracker_birth_back_h;
    p.sigma_base_m         = cfg_.ai2_sigma_base_m;
    p.clutter_frac         = cfg_.ai2_clutter_frac;
    p.clutter_scale_m      = cfg_.ai2_clutter_scale_m;
    p.process_std_m        = cfg_.ai2_process_std_m;
    p.process_std_yaw      = cfg_.ai2_process_std_yaw;
    p.floor_z              = cfg_.ai2_floor_z;
    p.floor_std            = cfg_.ai2_floor_std;
    p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
    p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
    p.gn_iters             = cfg_.ai2_gn_iters;
    return p;
}

void ChairFitter::seed_bearing_hypothesis(ChairInstance& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                          float nominal_range, float along_std, float across_std, float yaw_std)
{
    ChairBeliefState s0;
    s0.cx  = robot_xy.x() + nominal_range * std::cos(azimuth);
    s0.cy  = robot_xy.y() + nominal_range * std::sin(azimuth);
    s0.yaw = 0.0f;                                    // unknown; the broad yaw_std below owns that
    inst.ai2_belief = ChairBelief(s0, make_belief_params());
    inst.ai2_belief.seed_bearing(robot_xy, azimuth, along_std, across_std, yaw_std);
    inst.ai2_initialized       = true;
    inst.is_bearing_hypothesis = true;
    inst.hypothesis_azimuth    = azimuth;   // Orient affordance target yaw = the bearing to look toward
    // Write the mean into the model so the scene-graph publish / viewer show the hypothesis on the ray.
    ChairState ms = inst.model.state();
    ms.cx = s0.cx; ms.cy = s0.cy; ms.yaw = s0.yaw;
    inst.model.set_state(ms);
}

void ChairFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = enabled and (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

void ChairFitter::compute_chain_cov(ChairInstance& inst)
{
    inst.chain_cov_xx = 0.0f;
    inst.chain_cov_yy = 0.0f;
    if (not chain_cov_enabled_ or not gaussian_ or not inner_eigen_)
        return;
    // Localization/chain term J·Σ_chain·Jᵀ at the chair centre: transform it to the measurement frame,
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

bool ChairFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)
{
    room_node_id_ = room_id;
    if (instances_.count(node.id()))
        return false;

    ChairState init_state;
    init_state.cx  = 0.0f;
    init_state.cy  = 0.0f;
    init_state.yaw = 0.0f;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.seat_w = v.value();
    if (auto v = G_->get_attrib_by_name<depth_m_att> (node); v.has_value()) init_state.seat_d = v.value();
    // height_m carries the OVERALL height (seat_h + back_h, see ChairSceneGraph); recover seat_h by
    // subtracting back_h (default) — else seat_h reads back as the full 0.9 m and the model fits a giant chair.
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value())
        init_state.seat_h = std::max(0.10f, v.value() - init_state.back_h);

    // Read RT pose from room→chair edge
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

    // Tracker birth seed: a freshly born node's room→chair RT may not compose this cycle, so prefer the
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
                std::print("chair_concept: WARNING non-finite {} for '{}' → reset to {:.3f}\n", name, node.name(), fallback);
                v = fallback;
            }
        };
        fix(init_state.cx, 0.0f, "cx");
        fix(init_state.cy, 0.0f, "cy");
        fix(init_state.cz, 0.0f, "cz");
        fix(init_state.yaw, 0.0f, "yaw");
        fix(init_state.seat_w, 0.45f, "seat_w");
        fix(init_state.seat_d, 0.45f, "seat_d");
        fix(init_state.seat_h, 0.45f, "seat_h");
        fix(init_state.back_h, 0.45f, "back_h");
    }

    ChairInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = ChairModel(init_state, make_model_params());
    inst.affordance.init(G_, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("chair_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ─────────────────────────────────────────────────────────────

ChairFitter::ChairObservation ChairFitter::observe(ChairInstance& inst, const DSR::Node& node)
{
    ChairObservation observation;

    // Detection-aliveness ages every cycle; a fresh chair mask below resets it to 0.
    if (inst.frames_since_detection < 1000000) ++inst.frames_since_detection;

    // Primary path: YOLO "masks" (room frame), masks-only. Classify-don't-destroy SDF split keeps
    // inliers as candidates and the rest as residuals that drive model expansion.
    const auto& masks_packet = mask_ingestor_->packet();
    if (masks_packet.valid && masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance = ONLY the §3.1 gated assignment. No greedy-nearest fallback: when this
        // instance has no association this frame (its chair occluded / another instance won the slice), it
        // must FREEZE, not grab the nearest chair's mask — that fallback teleported a far-born instance onto
        // a near chair when its own chair was momentarily undetected (→ merge, so the far chair never
        // persisted). Freeze-on-no-association is the information-filter axiom; mirrors table_concept.
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this chair on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
            inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
            inst.last_motion_var  = slice.motion_var;     // AI2 ego-motion / range / truncation channels
            inst.last_motion_dotd = slice.motion_dotd;
            inst.last_trunc_frac  = slice.trunc_frac;
            inst.last_range       = slice.range;
            inst.last_centroid_radius = slice.centroid_radius;   // image-centredness (moving-update exception)
            inst.last_depth_var   = slice.depth_var;             // 0=ZED, >0=ricoh LiDAR-depth → downweights the fit (added to R)
            const std::size_t begin = std::min(slice.support_begin, masks_packet.support_points.size());
            const std::size_t end = std::min(slice.support_end, masks_packet.support_points.size());

            std::vector<Eigen::Vector3f> candidate_pts;
            std::vector<Eigen::Vector3f> residual_pts;
            candidate_pts.reserve(end > begin ? end - begin : 0);
            residual_pts.reserve(end > begin ? end - begin : 0);

            for (std::size_t i = begin; i < end; ++i)
            {
                const auto& p = masks_packet.support_points[i];
                const float sdf = inst.model.sdf_point(p);
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

bool ChairFitter::should_log(const ChairInstance& inst) const
{
    const int period = std::max(1, cfg_.chair_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── AI2 inference (shared recursive-Laplace belief; mirrors table run_inference) ───────────────
float ChairFitter::run_inference(ChairInstance& inst, const ChairObservation& observation)
{
    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy belief init from the model state; on the FIRST frame snap cx,cy→centroid and cz/seat_h→z-pctiles
    // (the belief is a LOCAL filter — the tracker birth provides the coarse yaw + size seed).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        ChairBeliefState s0{m.cx, m.cy, m.yaw};   // pose-only belief (size is the fixed template)
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            std::vector<float> zs; zs.reserve(static_cast<std::size_t>(npts));
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v)
            { for (const auto& p : v) { sum += p; zs.push_back(p.z()); } };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            s0.cx = sum.x() / npts; s0.cy = sum.y() / npts;
            std::sort(zs.begin(), zs.end());
            const float zmid = zs[static_cast<std::size_t>(0.55f * (zs.size() - 1))];   // ~seat-top band (yaw split)

            // Coarse yaw from the backrest offset: the backrest sits on the −local_y seat edge, so the
            // vector from the seat centroid to the upper (backrest) points runs along −local_y ⇒
            // yaw = atan2(dx, −dy). Seeds the (weakly-observed, near-square-seat) yaw near the answer so
            // the belief doesn't dwell at the wrong seed for hundreds of frames before the backrest wins.
            Eigen::Vector2f seat_c = Eigen::Vector2f::Zero(), back_c = Eigen::Vector2f::Zero();
            int n_seat = 0, n_back = 0;
            const float back_z = zmid + 0.10f, seat_lo = zmid - 0.10f;
            const auto split = [&](const std::vector<Eigen::Vector3f>& v)
            {
                for (const auto& p : v)
                {
                    if (p.z() > back_z)       { back_c += p.head<2>(); ++n_back; }
                    else if (p.z() > seat_lo) { seat_c += p.head<2>(); ++n_seat; }
                }
            };
            split(observation.candidate_pts); split(observation.residual_pts);
            if (n_back > 20 and n_seat > 20)
            {
                seat_c /= static_cast<float>(n_seat); back_c /= static_cast<float>(n_back);
                const Eigen::Vector2f d = back_c - seat_c;
                if (d.norm() > 0.05f)   // enough backrest offset to be a meaningful heading
                    s0.yaw = std::atan2(d.x(), -d.y());
            }
        }
        inst.ai2_belief = ChairBelief(s0, make_belief_params());
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
    {
        ingest_observation_voxels(inst, observation);
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
        compute_projected_roi(inst);
        return inst.dbg_energy;   // HOLD last FE — an aged cycle took no measurement (no new energy)
    }
    // Fresh path: update()/predict() below carry their own one-step Q, so just reset the age clock here.
    inst.last_belief_touch = now;

    // Static range weighting + ego-motion downweight (mirror table): a far view widens the common-mode
    // (binding yaw cap) so it can't rotate the chair; continuous, no gate.
    const float range         = std::max(0.0f, inst.last_range);
    const float range_lat_var = (cfg_.ai2_range_noise_lat_per_m * range) * (cfg_.ai2_range_noise_lat_per_m * range);
    const float range_yaw_var = (cfg_.ai2_range_noise_yaw_per_m * range) * (cfg_.ai2_range_noise_yaw_per_m * range);
    // depth_var (0 for a ZED slice, >0 for a ricoh LiDAR-depth slice) enters R in the SAME currency as motion_var:
    // a ricoh point's along-ray depth uncertainty inflates its measurement noise, so it barely moves the belief
    // mean (ZED drives geometry, ricoh only confirms). Mirrors table_concept's R = σ²+motion_var+depth_var+range.
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m + std::max(0.0f, inst.last_motion_var)
                    + std::max(0.0f, inst.last_depth_var) + range_lat_var;

    // Obliquity of the view onto the BACKREST (the chair's yaw-carrying surface, a VERTICAL plate whose normal
    // is horizontal, unlike the table's horizontal top). Backrest sits on the −local_y edge ⇒ its room-frame
    // outward normal is n̂ = (sin ψ, −cos ψ). obliquity_cos = |r̂_xy·n̂| ∈ [0,1]: 1 = facing it square-on, →0 =
    // viewing it edge-on (yaw barely observable). Feeds the shared yaw variance below (§D). Needs the camera
    // origin; default 1.0 (no cap) when the extrinsic isn't available this frame.
    inst.dbg_obliquity_cos = 1.0f;
    if (const auto rTz = room_T_zed_matrix(inst.last_mask_timestamp_ms); rTz.has_value())
    {
        const auto& s0b = inst.ai2_belief.state();
        Eigen::Vector2f r_xy(s0b.cx - static_cast<float>(rTz->coeff(0, 3)),
                             s0b.cy - static_cast<float>(rTz->coeff(1, 3)));   // horizontal camera→chair
        if (r_xy.norm() > 1e-6f)
        {
            r_xy.normalize();
            const Eigen::Vector2f n_back(std::sin(s0b.yaw), -std::cos(s0b.yaw));   // backrest outward normal (room xy)
            inst.dbg_obliquity_cos = std::abs(r_xy.dot(n_back));
        }
    }
    // Grow the SHARED yaw variance as the backrest grazes (1/cos−1): continuous covariance, no gate. THRESHOLD
    // (flagged, physical): clamp obliquity_cos to [0.05,1] so 1/cos stays finite at a perfectly edge-on view.
    // ⚠ gain is table's value, UNVALIDATED for chair (different surface) — tune from the logged obliquity_cos.
    const float oblq_cos = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);
    const float obliquity_yaw_std = cfg_.ai2_obliquity_yaw_gain * (1.0f / oblq_cos - 1.0f);

    // "Be-still-to-update" invariant: a truncated view (gated) OR a MOVING robot may only CONFIRM the chair, never
    // move/reshape it. A moving frame's mask is a shared smear whose centroid is unreliable — predict-only here
    // (mean held, Σ carries its one-step Q); the existence belief still confirms it (its mask reset stops vacate).
    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac or confirm_only(inst);
    compute_chain_cov(inst);

    float energy = inst.dbg_energy;   // default = HOLD last FE (a gated cycle takes no measurement)
    if (gated)
        inst.ai2_belief.predict();
    else
    {
        ChairFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        // AIF "be-still-to-update" as CONTINUOUS PRECISION: a frame's authority to MOVE the mean is capped via the
        // per-frame common-mode (NOT per-point R), by a variance that grows with MOTION × OFF-AXIS position. Still
        // (motion→0) OR well-centred (periphery→0) ⇒ ~0 common-mode ⇒ full authority; moving AND peripheral ⇒ large
        // common-mode ⇒ the frame can only CONFIRM. No gate — "confirmation-only" is the precision→0 limit. The
        // periphery factor is why a centred mask stays trustworthy while moving (the user's exception, emergent).
        const float motion_mag  = motion_magnitude(inst);
        const float periph      = periphery_penalty(inst);
        const float mot_pos_var = std::pow(cfg_.motion_cm_pos_gain * motion_mag, 2.0f) * periph;   // m²  (cx,cy)
        const float mot_yaw_var = std::pow(cfg_.motion_cm_yaw_gain * motion_mag, 2.0f) * periph;   // rad² (yaw)
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var + mot_pos_var;
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var + mot_pos_var;
        frame.chain_cov_yaw = range_yaw_var + obliquity_yaw_std * obliquity_yaw_std + mot_yaw_var;   // range + grazing cap (§D) + ego-motion
        inst.ai2_belief.update(frame);   // MAP mean + posterior Σ; its surface-only return is NOT the FE (below)
        // NOTE: refine_extent (coverage/extent likelihood) DISABLED — coverage without a free-space
        // counter-force is positive feedback: it inflated the footprint to cover contamination/neighbours
        // (seat → 2–5.8 m) and spawned phantom instances. Kept in the belief for reference; see Fable review.

        // FREE ENERGY = the clutter-INCLUSIVE mixture NLL, NOT the engine's surface-only return: a misfit point
        // routes to clutter (r_surface≈0) and contributes ≈0 to the surface energy, so a badly-fit chair would
        // read F≈0 — blind to exactly the errors that matter (TABLE.md §3, "do NOT reintroduce"). mixture_nll
        // includes the clutter term so F RISES with misfit; it is the same quantity association/orientation use.
        energy = inst.ai2_belief.mixture_nll(frame.points, inst.ai2_belief.state(), R);

        // 180° yaw-flip: a chair's only front/back asymmetry is the backrest, so the fit can lock 180° off
        // when the backrest was weakly seen at seed time, and no gradient step makes the π jump once yaw is
        // confidently converged. resolve_orientation() runs a SEQUENTIAL test (accumulated yaw-mode NLL
        // advantage) — it corrects a sustained-wrong orientation but ignores single partial-view frames. The
        // ego-motion weight w=1/(1+(dotd/ref)²) down-votes smeared/moving frames (the flips arrived on those).
        const float mref = std::max(1e-3f, cfg_.ai2_orientation_motion_ref);
        const float dotd = std::abs(inst.last_motion_dotd);
        const float mode_evidence_weight = 1.0f / (1.0f + (dotd / mref) * (dotd / mref));
        if (inst.ai2_belief.resolve_orientation(frame.points, R, mode_evidence_weight) and should_log(inst))
            std::print("[{}] AI2 yaw 180° FLIP corrected (sustained backrest evidence)\n", inst.node_name);

        // FE-surprise attention (TABLE.md §9): baseline tracks DOWN fast (consolidate a better fit) / UP slow (a
        // sustained rise = the chair moved surfaces as surprise before the baseline accepts it); surprise = the
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
        // e.g. the table bleeding into a chair mask, which drags the centroid). High + a position jump ⇒
        // contamination the clutter component didn't fully reject.
        inst.last_clutter_frac = inst.ai2_belief.clutter_fraction(frame.points, R);
    }
    inst.dbg_energy = energy;   // remember for the next gated/aged cycle to HOLD

    // Write belief → legacy ChairState so downstream publish/viewer/RT code is unchanged.
    const auto& bs = inst.ai2_belief.state();
    ChairState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.yaw = bs.yaw;
    ms.cz = inst.ai2_belief.cz();                     // pinned floor
    ms.seat_w = inst.ai2_belief.seat_w(); ms.seat_d = inst.ai2_belief.seat_d();   // fixed template dims
    ms.seat_h = inst.ai2_belief.seat_h(); ms.back_h = inst.ai2_belief.back_h();
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;
    compute_projected_roi(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} clutter={:.0f}% R={:.4f} range={:.2f} oblq={:.2f} trunc={:.2f}{} | FE={:.2f} base={:.2f} surprise={:.2f} | cx={:.2f} cy={:.2f} ψ={:.2f} σψ_rep={:.0f}° (template sw={:.2f} sh={:.2f})\n",
                   inst.node_name, npts, 100.0f * inst.last_clutter_frac, R, range, inst.dbg_obliquity_cos, inst.last_trunc_frac, gated ? " GATED" : "",
                   energy, inst.fe_baseline, inst.fe_surprise,
                   bs.cx, bs.cy, bs.yaw, std::sqrt(std::max(0.0f, inst.ai2_belief.covariance_reported()(2, 2))) * 57.2958f,
                   inst.ai2_belief.seat_w(), inst.ai2_belief.seat_h());

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

void ChairFitter::log_ai2_csv(const ChairInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,npts,gated,energy,fe_baseline,fe_surprise,R,motion_var,depth_var,trunc_frac,range,obliquity_cos,clutter_frac,"
                 << "cx,cy,yaw,seat_w,seat_d,seat_h,back_h,std_cx,std_cy,std_yaw,std_yaw_rep\n";
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_depth_var << ',' << inst.last_trunc_frac << ',' << inst.last_range << ',' << inst.dbg_obliquity_cos << ',' << inst.last_clutter_frac << ','
             << s.cx << ',' << s.cy << ',' << s.yaw << ','
             << inst.ai2_belief.seat_w() << ',' << inst.ai2_belief.seat_d() << ',' << inst.ai2_belief.seat_h() << ',' << inst.ai2_belief.back_h() << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << std::sqrt(std::max(0.0f, inst.ai2_belief.covariance_reported()(2, 2))) << '\n';
    ai2_csv_.flush();
}

// ─── RGB-mask ROI projection (active-perception aid) ──────────────────────────

std::optional<Eigen::Matrix4d> ChairFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
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

void ChairFitter::update_ego_motion()
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
float ChairFitter::motion_magnitude(const ChairInstance& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear/distortion), → 1 at
// centroid radius ai2_periph_ref. This is the "well-centred masks stay trustworthy" lever, expressed continuously.
float ChairFitter::periphery_penalty(const ChairInstance& inst) const
{
    const float ref = std::max(1e-6f, cfg_.ai2_periph_ref);
    const float r = std::max(0.0f, inst.last_centroid_radius) / ref;
    return std::clamp(r * r, 0.0f, 1.0f);
}

float ChairFitter::frame_reliability(const ChairInstance& inst) const
{
    // u = (motion / motion_ref) × periphery ∈ [0,1]: only high when the frame is BOTH moving AND off-axis.
    const float u = std::clamp(motion_magnitude(inst) / std::max(1e-3f, cfg_.ai2_motion_ref_mps), 0.0f, 1.0f)
                    * periphery_penalty(inst);
    return 1.0f - u;
}

float ChairFitter::zed_detectability(const ChairInstance& inst) const
{
    if (not inst.roi_valid)
        return 0.0f;
    // Off-centre falloff: YOLO/depth degrade toward the ZED image edge. roi_offset is normalised (0=centred,
    // ±1=image edge). Reliable near the axis, → 0 at the configured edge reach.
    const float off = std::hypot(inst.roi_offset_x, inst.roi_offset_y);
    const float pd_center = std::clamp(1.0f - off / std::max(1e-3f, cfg_.exist_zed_edge_offset), 0.0f, 1.0f);
    // Range falloff: a far chair is a small, unreliable ZED detection → its ABSENCE is weak evidence. 1 within
    // ZedRangeFull, smoothly → 0 by ZedRangeRef (beyond which ZED absence says nothing about existence).
    const float r = std::max(0.0f, inst.last_range);
    const float span = std::max(1e-3f, cfg_.exist_zed_range_ref - cfg_.exist_zed_range_full);
    const float pd_range = std::clamp((cfg_.exist_zed_range_ref - r) / span, 0.0f, 1.0f);
    return pd_center * pd_range;
}

bool ChairFitter::confirm_only(const ChairInstance& inst) const
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

bool ChairFitter::point_in_room(const Eigen::Vector2f& q, float margin_m) const
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

    // Outside the polygon: accept only if within margin_m of the boundary (wall-hugging chair, centroid noise).
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

bool ChairFitter::los_occluded(const ChairInstance& inst) const
{
    const auto Mopt = room_T_zed_matrix();   // camera→room
    if (not Mopt.has_value())
        return false;                        // no extrinsic → can't judge occlusion → let vacate proceed
    const Eigen::Vector3f O(static_cast<float>(Mopt->coeff(0, 3)),
                            static_cast<float>(Mopt->coeff(1, 3)),
                            static_cast<float>(Mopt->coeff(2, 3)));
    const auto& s = inst.model.state();
    const Eigen::Vector3f C(s.cx, s.cy, s.cz + 0.5f * (s.seat_h + s.back_h));   // chair mid-height
    Eigen::Vector3f dc = C - O;
    const float rc = dc.norm();
    if (rc < 1e-3f)
        return false;
    dc /= rc;

    const float margin = std::max(0.0f, cfg_.exist_occlusion_margin_m);   // occluder must be at least this closer
    // An object at range rs subtending a half-extent `half` covers a bearing cone of half-angle atan(half/rs).
    // The chair is occluded if a CLOSER object's cone contains the chair's bearing.
    const auto blocks = [&](const Eigen::Vector3f& Cs, float half, float rs) -> bool
    {
        if (not std::isfinite(rs) or rs >= rc - margin)   // not meaningfully closer → cannot occlude
            return false;
        Eigen::Vector3f ds = Cs - O;
        const float n = ds.norm();
        if (n < 1e-3f)
            return false;
        ds /= n;
        const float ang      = std::acos(std::clamp(dc.dot(ds), -1.0f, 1.0f));   // camera-bearing offset chair↔occluder
        const float occ_half = std::atan2(std::max(0.05f, half), std::max(0.2f, rs));
        return ang < occ_half;
    };

    // (a) other chair instances (always known, even when undetected this frame).
    for (const auto& [jid, jinst] : instances_)
    {
        if (jid == inst.node_id or not jinst.ai2_initialized)
            continue;
        const auto& js = jinst.model.state();
        const Eigen::Vector3f Cj(js.cx, js.cy, js.cz + 0.5f * (js.seat_h + js.back_h));
        if (blocks(Cj, 0.5f * std::max(js.seat_w, js.seat_d), (Cj - O).norm()))
            return true;
    }
    // (b) any other object DETECTED this frame (table, person, …) via its mask slice geometry.
    if (mask_ingestor_)
    {
        const auto& pkt = mask_ingestor_->packet();
        if (pkt.valid)
            for (const auto& sl : pkt.slices)
            {
                if (not sl.has_depth or not sl.centroid.allFinite() or not sl.bbox_max.allFinite())
                    continue;
                const float half = 0.5f * (sl.bbox_max - sl.bbox_min).head<2>().norm();
                const float rs   = (sl.range > 0.0f) ? sl.range : (sl.centroid - O).norm();
                if (blocks(sl.centroid, half, rs))
                    return true;
            }
    }
    return false;
}

void ChairFitter::compute_projected_roi(ChairInstance& inst)
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

    // Project the 8 box corners (top slab + footprint at floor) into the image. Camera convention
    // matches the producer: X=right, Y=forward(depth), Z=up ⇒ col=cx+X/Y·fx, row=cy−Z/Y·fy.
    const auto& s = inst.model.state();
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float hw = s.seat_w * 0.5f, hh = s.seat_d * 0.5f;
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    for (const int ix : {-1, 1})
        for (const int iy : {-1, 1})
            for (const float z : {s.cz, s.cz + s.seat_h + s.back_h})
            {
                const float lx = static_cast<float>(ix) * hw, ly = static_cast<float>(iy) * hh;
                const Eigen::Vector4d Pr(s.cx + c * lx - sn * ly, s.cy + sn * lx + c * ly, z, 1.0);
                const Eigen::Vector4d Pc = zed_T_room * Pr;
                const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
                if (Y <= 0.20) continue;   // skip corners at/near the image plane: X/Y explodes there
                ++in_front;
                const float col = cx_px + static_cast<float>(X / Y) * fx;
                const float row = cy_px - static_cast<float>(Z / Y) * fy;
                min_col = std::min(min_col, col); max_col = std::max(max_col, col);
                min_row = std::min(min_row, row); max_row = std::max(max_row, row);
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

// ─── Voxel bank (chair-owned historical memory) ──────────────────────────────

std::uint64_t ChairFitter::voxel_key(const Eigen::Vector3f& point, float quantization_m)
{
    const float q = std::max(1e-4f, quantization_m);
    const int ix = static_cast<int>(std::floor(point.x() / q));
    const int iy = static_cast<int>(std::floor(point.y() / q));
    const int iz = static_cast<int>(std::floor(point.z() / q));

    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
    mix(static_cast<std::uint64_t>(ix));
    mix(static_cast<std::uint64_t>(iy));
    mix(static_cast<std::uint64_t>(iz));
    return h;
}

void ChairFitter::ingest_observation_voxels(ChairInstance& inst, const ChairObservation& observation)
{
    std::size_t inserted = 0;
    std::size_t rejected_foreign = 0;
    const auto max_points = static_cast<std::size_t>(std::max(1, cfg_.voxel_bank_max_points));

    auto ingest = [&](const std::vector<Eigen::Vector3f>& src)
    {
        for (const auto& p : src)
        {
            if (inst.voxel_bank_pts.size() >= max_points)
                break;
            if (not is_voxel_owned_by_chair(inst, p))
            {
                ++rejected_foreign;
                continue;
            }
            const auto key = voxel_key(p, cfg_.voxel_bank_quantization_m);
            if (inst.voxel_bank_keys.insert(key).second)
            {
                inst.voxel_bank_pts.push_back(p);
                ++inserted;
            }
        }
    };

    ingest(observation.candidate_pts);
    ingest(observation.residual_pts);

    if (inserted > 0 && should_log(inst))
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   inst.node_name, inserted, inst.voxel_bank_pts.size(), max_points, rejected_foreign);
}

bool ChairFitter::is_voxel_owned_by_chair(const ChairInstance& inst, const Eigen::Vector3f& point) const
{
    const auto& s = inst.model.state();

    // XY ownership gate: chair-centered radius with a configurable margin.
    const float half_diag = 0.5f * std::sqrt(s.seat_w * s.seat_w + s.seat_d * s.seat_d);
    const float gate_radius = std::max(1.0f, half_diag + cfg_.voxel_select_radius_margin_m);
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate to reject floor / distant clutter points in mixed scenes.
    const float z_min = -0.05f;
    const float z_max = s.cz + s.seat_h + s.back_h + cfg_.voxel_select_height_margin_m;
    return point.z() >= z_min && point.z() <= z_max;
}

// ─── Factory helpers ─────────────────────────────────────────────────────────

ChairModelParams ChairFitter::make_model_params() const
{
    ChairModelParams p;
    p.sigma_obs = cfg_.sigma_obs;
    return p;
}

}  // namespace rc
