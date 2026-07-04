/*
 * table_fitter.cpp — the active-inference fit core for table_concept.
 */

#include "table_fitter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

TableFitter::TableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         TableConfig& cfg,
                         MaskIngestor* mask_ingestor,
                         TableSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{
}

void TableFitter::set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled)
{
    gaussian_          = gaussian;
    chain_src_frame_   = std::move(source_frame);
    chain_cov_enabled_ = enabled and (gaussian_ != nullptr) and not chain_src_frame_.empty();
}

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

// ─── Instance lifecycle ──────────────────────────────────────────────────────

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
    inst.affordance.init(G_, node.id(), node.name());

    instances_.emplace(node.id(), std::move(inst));
    std::print("table_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

// ─── Observation ─────────────────────────────────────────────────────────────

TableFitter::TableObservation TableFitter::observe(TableInstance& inst, const DSR::Node& node)
{
    TableObservation observation;

    // Detection-aliveness ages every cycle; a fresh table mask below resets it to 0.
    if (inst.frames_since_detection < 1000000) ++inst.frames_since_detection;

    // Primary path: YOLO "masks" (room frame), masks-only. Classify-don't-destroy SDF split keeps
    // inliers as queue anchors (candidates) and the rest as residuals that drive model expansion.
    const auto& masks_packet = mask_ingestor_->packet();
    if (masks_packet.valid && masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance = the tracker's gated assignment ONLY. A frame where the tracker did not
        // associate a detection to this instance carries no trustworthy mask for it (feeding the nearest
        // blob would cross-contaminate instances and update on unconfirmed evidence), so we skip — the
        // belief just freezes that cycle (information-filter axiom). assigned_mask_idx is set each cycle by
        // run_instance_tracker; -1 means "no association this frame".
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this table on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
            inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
            // Ego-motion capture-corruption for this mask (AI2 obs precision / bias gate; 0 if producer
            // predates the feature). See MASK_MOTION_CORRUPTION.md.
            inst.last_motion_var      = slice.motion_var;
            inst.last_motion_dotd     = slice.motion_dotd;
            inst.last_trunc_frac      = slice.trunc_frac;
            inst.last_centroid_radius = slice.centroid_radius;
            inst.last_range           = slice.range;
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

bool TableFitter::should_log(const TableInstance& inst) const
{
    const int period = std::max(1, cfg_.table_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Inference ───────────────────────────────────────────────────────────────

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
        }
        TableBeliefParams p;
        p.sigma_base_m    = cfg_.ai2_sigma_base_m;
        p.clutter_frac    = cfg_.ai2_clutter_frac;
        p.clutter_scale_m = cfg_.ai2_clutter_scale_m;
        p.prior_size_std  = cfg_.ai2_prior_size_std;
        p.process_std_m   = cfg_.ai2_process_std_m;
        p.process_std_yaw = cfg_.ai2_process_std_yaw;
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.common_mode_yaw_std  = cfg_.ai2_common_mode_yaw_std;
        p.gn_iters        = cfg_.ai2_gn_iters;
        p.top_thickness   = TableModel::TOP_THICKNESS;
        p.leg_radius      = TableModel::LEG_RADIUS;
        inst.ai2_belief = TableBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
    {
        ingest_observation_voxels(inst, observation);   // keep the viewer's voxel bank fed
        inst.last_residual_pts = observation.residual_pts;   // model-unexplained points for the viewer layer
    }

    // Freeze-on-stale: no fresh mask ⇒ don't touch the belief (information-filter axiom).
    if (not observation.has_fresh_data)
    {
        compute_projected_roi(inst);
        return 0.0f;
    }

    // Static range weighting (motion-free). Even at zero camera motion, deprojection noise grows with
    // distance AND a far mask subtends a tiny angle, so pose — orientation most of all — becomes
    // unobservable: a 7 m view should confirm existence but never rotate a converged table. The
    // motion×distance term is already in last_motion_var (the voxelizer interaction matrix carries 1/Z),
    // but that vanishes when still; this is the missing static part. Pure continuous covariance growth (no
    // gate): the per-frame information CAP (common-mode) rises with range, so the frame's yaw gain against a
    // converged prior shrinks smoothly toward zero. mask_range = mean camera→mask depth Z, from the producer.
    const float range         = std::max(0.0f, inst.last_range);
    const float lat_std       = cfg_.ai2_range_noise_lat_per_m * range;   // m   (lateral deprojection)
    const float yaw_std       = cfg_.ai2_range_noise_yaw_per_m * range;   // rad (orientation lever-arm ∝ 1/range)
    const float range_lat_var = lat_std * lat_std;
    const float range_yaw_var = yaw_std * yaw_std;

    // Observation precision R = σ_base² + ego-motion var + static range var (per-point random part).
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m
                  + std::max(0.0f, inst.last_motion_var) + range_lat_var;

    // Truncation gate: a mask clipped by the image border has a chopped silhouette → it biases the fit
    // (shrinks/displaces the model). Above tolerance, skip the geometric update (predict only) — but
    // keep the instance (association ran upstream). The ego-motion lag bias is handled upstream by the
    // voxelizer pose extrapolation, so no separate motion/bias gate is needed here.
    const bool gated = inst.last_trunc_frac > cfg_.ai2_trunc_gate_frac;

    // Pose-chain covariance at the table centre (cx,cy) — the per-frame SHARED localization error. Fed
    // into the belief as part of the common-mode so the frame's information saturates (calibrated σ).
    // Computed once here (before the update) and reused for the published RT cov below.
    compute_chain_cov(inst);

    float energy = 0.0f;
    if (gated)
        inst.ai2_belief.predict();   // Σ inflates, mean unchanged
    else
    {
        TableFrame frame;
        frame.points.reserve(static_cast<std::size_t>(npts));
        frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
        frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
        frame.R.assign(frame.points.size(), R);
        frame.chain_cov_xx  = inst.chain_cov_xx + range_lat_var;   // range adds to the SHARED position error (cap)
        frame.chain_cov_yy  = inst.chain_cov_yy + range_lat_var;
        frame.chain_cov_yaw = range_yaw_var;                       // the binding term: far view can't rotate
        // YOLO-independent LiDAR range channel: stage returns landing on the legs/rim. No-op if precision==0
        // or no fresh sweep. The shared factor (accumulate_lidar_rays<6> in TableBelief::accumulate_extra)
        // sphere-traces this belief's own SDF, so the same call the bottle uses drops in unchanged.
        feed_lidar(inst, frame);
        energy = inst.ai2_belief.update(frame);
        // Near-square yaw disambiguation: sequential Bayesian comparison of the two orientation modes
        // (current vs the w↔h swap ≡ 90° rotation). Owns the GENUINE mode flip so the per-frame MAP no
        // longer SNAPS 90° on extent noise; the reported yaw uncertainty (yaw_marginal_var) stays honest
        // until an orbit resolves it. See TABLE_FIT_AI2.md.
        if (inst.ai2_belief.resolve_orientation(frame.points, R) and should_log(inst))
            std::print("[{}] orientation mode FLIP (w↔h) | flip_ev={:.3f} p_alt={:.2f}\n",
                       inst.node_name, inst.ai2_belief.flip_evidence(), inst.ai2_belief.mode_posterior());
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

    compute_projected_roi(inst);
    // (chain cov already computed above, before the belief update)

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} dotd={:.2f} trunc={:.2f}{} | cx={:.3f} cy={:.3f} H={:.3f} w={:.3f} h={:.3f} ψ={:.3f} | σ(w,h,H)mm=({:.0f},{:.0f},{:.0f})\n",
                   inst.node_name, npts, R, inst.last_motion_dotd, inst.last_trunc_frac, gated ? " GATED" : "",
                   bs.cx, bs.cy, bs.H, bs.w, bs.h, bs.yaw,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(2, 2))));

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

void TableFitter::log_ai2_csv(const TableInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,npts,gated,energy,R,motion_var,motion_dotd,trunc_frac,range,"
                 << "cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw,"
                 << "std_yaw_within,flip_ev,p_alt\n";
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };   // posterior std (m / rad)
    // std_yaw is now the MARGINAL (mode-entropy-inflated) yaw std; std_yaw_within is the within-mode Σ(5,5).
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_motion_dotd << ','
             << inst.last_trunc_frac << ',' << inst.last_range << ','
             << s.cx << ',' << s.cy << ',' << s.H << ',' << s.w << ',' << s.h << ',' << s.yaw << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ','
             << std::sqrt(std::max(0.0f, inst.ai2_belief.yaw_marginal_var())) << ','
             << sd(5) << ',' << inst.ai2_belief.flip_evidence() << ',' << inst.ai2_belief.mode_posterior() << '\n';
    ai2_csv_.flush();
}

// ─── YOLO-independent LiDAR first-hit range channel ────────────────────────────
// Select this cycle's returns landing on THIS table and stage them on frame.lidar. The box is anchored on
// the FRESH mask-cloud centroid (XY) — not the fitted state, which a diverging fit would drag into empty
// space, starving LiDAR exactly when it is most needed — and sized from the BIRTH footprint (not the fitted
// w/h, so a blown-up extent can't explode the region). The vertical band is floor-referenced [−m, birth_H+m]
// so it deliberately spans the LEGS and the tabletop RIM: those are the unbiased, segmentation-independent
// surfaces that attack the mask-erosion under-size. Final membership is the factor's own sphere-trace hit
// test (a ray must actually cross the model SDF), so this box is only a work bound + neighbour reject.
void TableFitter::feed_lidar(TableInstance& inst, TableFrame& frame) const
{
    inst.dbg_lidar_rays = 0;
    inst.dbg_lidar_raw  = 0;
    inst.dbg_lidar_resid_m = -1.0f;
    if (cfg_.lidar_precision <= 0.0f or not lidar_have_sweep_ or lidar_sweep_room_.empty()
        or frame.points.empty())
        return;

    // Anchor XY on this cycle's fresh mask-cloud centroid.
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : frame.points) c += p;
    c /= static_cast<float>(frame.points.size());

    const auto& s = inst.ai2_belief.state();
    // Fixed footprint from the BIRTH dims (never the fitted w/h). Circumscribed horizontal radius so the box
    // is rotation-agnostic (yaw need not be resolved to select). Vertical band is floor-referenced.
    const float m    = cfg_.lidar_select_margin_m;
    const float rxy  = 0.5f * std::sqrt(cfg_.tracker_birth_width_m * cfg_.tracker_birth_width_m
                                      + cfg_.tracker_birth_depth_m * cfg_.tracker_birth_depth_m) + m;
    const float rxy2 = rxy * rxy;
    const float z_lo = -m;                                   // floor (room z≈0), minus a little slack
    const float z_hi = cfg_.tracker_birth_height_m + m;      // tabletop, plus margin
    // Generous diagnostic box (1.5× horizontal) — "is a return anywhere near this table?" — kept below the
    // birth min-separation so it can't grab a neighbouring table's returns.
    const float rraw2 = (1.5f * rxy) * (1.5f * rxy);
    int raw = 0;

    frame.lidar.endpoints.clear();
    frame.lidar.endpoints.reserve(256);
    double resid_sum = 0.0;
    for (const auto& p : lidar_sweep_room_)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        const float dh2 = dx * dx + dy * dy;
        if (dh2 <= rraw2 and p.z() >= z_lo and p.z() <= z_hi) ++raw;   // generous "near?" count
        if (dh2 > rxy2) continue;
        if (p.z() < z_lo or p.z() > z_hi) continue;
        frame.lidar.endpoints.push_back(p);
        resid_sum += std::abs(inst.ai2_belief.sdf_compound(p, s));     // |dist to CURRENT model surface|
    }
    inst.dbg_lidar_raw  = raw;
    inst.dbg_lidar_rays = static_cast<int>(frame.lidar.endpoints.size());
    if (inst.dbg_lidar_rays > 0)
        inst.dbg_lidar_resid_m = static_cast<float>(resid_sum / inst.dbg_lidar_rays);
    if (frame.lidar.endpoints.empty())
        return;

    // Precision = base, DOWN-WEIGHTED by sparse coverage (a handful of noisy returns must not swing extent).
    // No range fade: a LiDAR hit is a real metric surface point at any range — sparsity is the only decay.
    const float coverage = std::min(1.0f, static_cast<float>(inst.dbg_lidar_rays)
                                        / std::max(1.0f, cfg_.lidar_coverage_n0));
    frame.lidar.origin     = lidar_origin_room_;
    frame.lidar.precision  = cfg_.lidar_precision * coverage;
    frame.lidar.robust_c_m = cfg_.lidar_robust_c_m;
}

// ─── Camera extrinsic (room_T_zed) ─────────────────────────────────────────────

std::optional<Eigen::Matrix4d> TableFitter::room_T_zed_matrix(std::uint64_t pose_ts_ms) const
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

void TableFitter::compute_projected_roi(TableInstance& inst)
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
    const float hw = s.w * 0.5f, hh = s.h * 0.5f;
    float min_col = 1e9f, min_row = 1e9f, max_col = -1e9f, max_row = -1e9f;
    int in_front = 0;
    for (const int ix : {-1, 1})
        for (const int iy : {-1, 1})
            for (const float z : {0.0f, s.table_height})
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
    // Reject degenerate projections (robot too close / a corner grazing the image plane → the
    // bbox explodes to absurd offsets). Beyond a sane bound the ROI is unusable for centring:
    // mark invalid (the controller then keeps sweeping / treats framing as unknown) and clamp the
    // stored values so consumers/logs never see garbage.
    const bool sane = std::isfinite(off_x) && std::isfinite(off_y) && std::isfinite(fill)
                      && std::abs(off_x) < 3.0f && std::abs(off_y) < 3.0f && fill < 4.0f;
    inst.roi_offset_x = std::clamp(off_x, -3.0f, 3.0f);
    inst.roi_offset_y = std::clamp(off_y, -3.0f, 3.0f);
    inst.roi_fill     = std::clamp(fill, 0.0f, 4.0f);
    inst.roi_valid    = sane;
}

// ─── Voxel bank (table-owned historical memory) ──────────────────────────────

std::uint64_t TableFitter::voxel_key(const Eigen::Vector3f& point, float quantization_m)
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

void TableFitter::ingest_observation_voxels(TableInstance& inst, const TableObservation& observation)
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
            if (not is_voxel_owned_by_table(inst, p))
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

bool TableFitter::is_voxel_owned_by_table(const TableInstance& inst, const Eigen::Vector3f& point) const
{
    const auto& s = inst.model.state();

    // XY ownership gate: table-centered radius with a configurable margin.
    const float half_diag = 0.5f * std::sqrt(s.w * s.w + s.h * s.h);
    const float gate_radius = std::max(1.0f, half_diag + cfg_.voxel_select_radius_margin_m);
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate to reject floor / distant clutter points in mixed scenes.
    const float z_min = -0.05f;
    const float z_max = s.table_height + cfg_.voxel_select_height_margin_m;
    return point.z() >= z_min && point.z() <= z_max;
}

// ─── Factory helpers ─────────────────────────────────────────────────────────

TableModelParams TableFitter::make_model_params() const
{
    TableModelParams p;
    p.sigma_obs = cfg_.sigma_obs;   // top/leg SDF split band for the candidate/residual classification
    return p;
}

}  // namespace rc
