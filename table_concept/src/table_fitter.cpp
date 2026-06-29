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
    // Cap torch intra-op parallelism: the per-box FE fit is tiny (a few k points), so torch spreading it
    // across every core is pure overhead + contention — it grabbed ~20 cores for a single table. A small
    // pool is faster here and frees the machine. (Set TableModel.TorchThreads; default 2.)
    set_torch_threads(cfg_.torch_threads);
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

// ─── TableBeliefPolicy ───────────────────────────────────────────────────────

float TableFitter::TableBeliefPolicy::clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float TableFitter::TableBeliefPolicy::lerp(float start, float end, float gain)
{
    return start + gain * (end - start);
}

float TableFitter::TableBeliefPolicy::wrap_angle(float angle)
{
    while (angle > M_PIf) angle -= 2.0f * M_PIf;
    while (angle < -M_PIf) angle += 2.0f * M_PIf;
    return angle;
}

float TableFitter::TableBeliefPolicy::angle_lerp(float start, float end, float gain)
{
    return wrap_angle(start + gain * wrap_angle(end - start));
}

TableState TableFitter::TableBeliefPolicy::apply_observability_warm_start(
    const TableState& previous,
    const TableState& raw,
    const std::array<float, 8>& kalman_gain)
{
    // SYMMETRIC precision-weighted acceptance for EVERY DOF (the asymmetric size ratchet is retired). Each
    // DOF's lerp is driven by its Kalman gain K_j = obs_j/(Y_pred_j + obs_j) from the information filter —
    // a well-observed DOF (large Y_pred) barely moves; a high-information view still moves it. Size needs no
    // grow-only hack: the extent term works over the ACCUMULATED RFE queue (which holds the full observed
    // span, so a partial current view can't shrink it), and the gain (mature size → small) + the ρ-EMA
    // (transient bursts) damp it. So the box relaxes an over-grow toward the consistent span AND resists a
    // partial-view collapse — symmetrically, with no high-side noise accumulation.
    const auto& K = kalman_gain;
    TableState accepted = raw;
    accepted.cx           = lerp(previous.cx,           raw.cx,           K[0]);
    accepted.cy           = lerp(previous.cy,           raw.cy,           K[1]);
    accepted.w            = lerp(previous.w,            raw.w,            K[2]);
    accepted.h            = lerp(previous.h,            raw.h,            K[3]);
    accepted.table_height = lerp(previous.table_height, raw.table_height, K[4]);
    accepted.leg_length   = lerp(previous.leg_length,   raw.leg_length,   K[5]);
    accepted.yaw          = angle_lerp(previous.yaw,    raw.yaw,          K[6]);
    accepted.leg_inset    = lerp(previous.leg_inset,    raw.leg_inset,    K[7]);
    return accepted;
}

float TableFitter::TableBeliefPolicy::update_warm_confidence(
    float previous_confidence,
    const TableConfig& cfg,
    const std::array<float, 6>& coverage,
    int point_count,
    int residual_count,
    float residual_precision)
{
    constexpr float kCoverageEps = 1e-3f;

    const float cov_px = coverage[0];
    const float cov_nx = coverage[1];
    const float cov_py = coverage[2];
    const float cov_ny = coverage[3];

    const float rho_x = std::min(cov_px, cov_nx) / (std::max(cov_px, cov_nx) + kCoverageEps);
    const float rho_y = std::min(cov_py, cov_ny) / (std::max(cov_py, cov_ny) + kCoverageEps);
    const float pts_span = std::max(1e-3f, cfg.warm_pts_max - cfg.warm_pts_min);
    const float rho_pts = clamp01((static_cast<float>(point_count) - cfg.warm_pts_min) / pts_span);
    const float residual_ratio = clamp01(residual_precision * static_cast<float>(residual_count) /
                                         static_cast<float>(std::max(1, point_count + residual_count)));

    const float bilateral_support = 0.5f * (rho_x + rho_y);
    const float coverage_evidence = rho_pts * lerp(0.15f, 1.0f, bilateral_support);
    const float evidence = cfg.warm_confidence_coverage_gain * coverage_evidence +
                           cfg.warm_confidence_residual_gain * residual_ratio;

    const float updated = cfg.warm_confidence_decay * previous_confidence +
                          (1.0f - cfg.warm_confidence_decay) * evidence;
    return clamp01(updated);
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

    // Anchor the size prior to this table's actual prior geometry, else the KL size term
    // pulls every table toward the hardcoded defaults regardless of the loaded prior/observations.
    TableModelParams mparams = make_model_params();
    mparams.prior_w            = init_state.w;
    mparams.prior_h            = init_state.h;
    mparams.prior_table_height = init_state.table_height;
    // Born-from-detection: the Birth* dims (seed) double as the size prior MEAN, at this precision, so the
    // size is pinned near standard furniture under weak early views instead of being reshaped freely by
    // the first partial mask ("go with the prior": same mean as the seed, with a confidence behind it).
    mparams.prior_size_std = cfg_.tracker_birth_size_std;

    inst.model     = TableModel(init_state, mparams);
    inst.queue     = SampleQueue<TableModel>(make_queue_params());
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
            inst.last_motion_bias     = slice.motion_bias;
            inst.last_trunc_frac      = slice.trunc_frac;
            inst.last_centroid_radius = slice.centroid_radius;
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

float TableFitter::run_inference_ai2(TableInstance& inst, const TableObservation& observation)
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
        p.gn_iters        = cfg_.ai2_gn_iters;
        p.top_thickness   = TableModel::TOP_THICKNESS;
        p.leg_radius      = TableModel::LEG_RADIUS;
        inst.ai2_belief = TableBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);   // keep the viewer's voxel bank fed

    // Freeze-on-stale: no fresh mask ⇒ don't touch the belief (information-filter axiom).
    if (not observation.has_fresh_data)
    {
        compute_projected_roi(inst);
        return 0.0f;
    }

    // Observation precision R = σ_base² + mask_motion_var (the ego-motion downweight; symmetric/random
    // error). One scalar per mask in v1 — the per-DOF split is a later step.
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m + std::max(0.0f, inst.last_motion_var);

    // Bias gate: a known timing offset shifts the whole mask systematically (a bias, not noise). Above
    // tolerance, skip the geometric update (predict only) — but keep the instance (association ran
    // upstream). Table is cm-scale ⇒ a looser gate than bottle.
    const bool gated = inst.last_motion_bias > cfg_.ai2_motion_bias_gate_m;

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
        energy = inst.ai2_belief.update(frame);
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
    compute_chain_cov(inst);

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} bias={:.3f}{} | cx={:.3f} cy={:.3f} H={:.3f} w={:.3f} h={:.3f} ψ={:.3f} | σ(w,h,H)mm=({:.0f},{:.0f},{:.0f})\n",
                   inst.node_name, npts, R, inst.last_motion_bias, gated ? " GATED" : "",
                   bs.cx, bs.cy, bs.H, bs.w, bs.h, bs.yaw,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(2, 2))));

    log_ai2_csv(inst, npts, R, gated, energy);
    return energy;
}

float TableFitter::run_inference(TableInstance& inst, const TableObservation& observation)
{
    if (cfg_.use_ai2)
        return run_inference_ai2(inst, observation);

    inst.queue.begin_cycle();

    ++inst.settle_maturity;   // climbs each cycle; step_queue_update resets it on a new-evidence burst

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);

    if (observation.has_fresh_data and not inst.voxel_bank_pts.empty())
    {
        // Cold-start: on first observation snap model & prior to the voxel centroid so gradient
        // descent begins at the right place rather than the prior.
        if (inst.matched_frames == 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            const auto& centroid_source = observation.candidate_pts.empty() ? inst.voxel_bank_pts
                                                                             : observation.candidate_pts;
            for (const auto& p : centroid_source) sum += p;
            const Eigen::Vector3f cen = sum / static_cast<float>(centroid_source.size());
            auto s  = inst.model.state();
            s.cx = cen.x();
            s.cy = cen.y();
            inst.model.set_state(s);
            inst.model.set_prior(s);   // zero KL so the data term dominates from the start
            std::print("[{}] cold-start snap → cx={:.2f} cy={:.2f} ({} pts)\n",
                       inst.node_name, s.cx, s.cy, centroid_source.size());
            inst.matched_frames = cfg_.min_frames_before_historical + cfg_.historical_warmup_frames + 1;
        }
        else
            ++inst.matched_frames;

        step_queue_update(inst, inst.voxel_bank_pts,
                          TableBeliefPolicy::clamp01(observation.explanation_ratio));
    }

    const float explanation_confidence = TableBeliefPolicy::clamp01(observation.explanation_ratio);
    float residual_precision = TableBeliefPolicy::clamp01(inst.warm_confidence * (1.0f - explanation_confidence));
    if (observation.has_fresh_data && observation.candidate_pts.empty() && !observation.residual_pts.empty())
        residual_precision = std::max(residual_precision, 0.10f);   // only residuals: keep a floor

    feed_silhouette(inst);   // set the RGB-mask contour rays before the gradient step

    // Model evidence ρ = mean inlier responsibility of the FRESH mask under the CURRENT belief: how well
    // the box explains what we just saw. exp(−½(SDF/σ_ρ)²) per point → ρ∈(0,1]. A box that is too small
    // leaves the mask points far outside → low ρ → the stabiliser's predict loosens the size precision so
    // the belief yields and the extent term grows it. This (with the always-on optimiser) is what makes the
    // fit respond to a mask beyond its span — replacing the freeze/re-open/CUSUM machinery. Computed over
    // the fresh observation, NOT the near-surface-by-construction queue anchors.
    if (observation.has_fresh_data)
    {
        const float inv2s2 = 0.5f / std::max(1e-6f, cfg_.evidence_sigma_m * cfg_.evidence_sigma_m);
        double sum = 0.0; std::size_t n = 0;
        const auto accum = [&](const std::vector<Eigen::Vector3f>& pts)
        { for (const auto& p : pts) { const float d = inst.model.sdf_point(p); sum += std::exp(-d * d * inv2s2); ++n; } };
        accum(observation.candidate_pts);
        accum(observation.residual_pts);
        const float rho_raw = n > 0 ? static_cast<float>(sum / static_cast<double>(n)) : 1.0f;

        // Gate the loosening by the observation's RELIABILITY (YOLO-confidence weight): only a TRUSTED view
        // may falsify the belief. A far / low-confidence mask whose points miss the box is a bad OBSERVATION,
        // not evidence the (static) table moved — so it must NOT loosen position/yaw (the instability seen on
        // mask_w≈0.2–0.3 views). ρ_eff = 1 − w(conf)·(1−ρ_raw): w→0 ⇒ ρ_eff→1 (precision persists, stays put);
        // w→1 with a genuinely contradicting view ⇒ ρ_eff→ρ_raw (loosens, yields).
        const float c0 = std::clamp(cfg_.mask_conf_floor, 0.0f, 0.99f);
        const float cr = std::max(c0 + 1e-3f, cfg_.mask_conf_ref);
        const float w  = std::pow(TableBeliefPolicy::clamp01((inst.last_mask_confidence - c0) / (cr - c0)),
                                  std::max(0.1f, cfg_.mask_conf_power));
        const float rho_frame = 1.0f - w * (1.0f - rho_raw);

        // Temporal integration: loosen only on SUSTAINED low evidence, not a transient noisy burst. EMA
        // over ~1/(1−α) fresh frames; a brief dip barely moves it, a persistent mismatch pulls it down.
        const float a = std::clamp(cfg_.evidence_ema_alpha, 0.0f, 0.999f);
        inst.evidence_ema = a * inst.evidence_ema + (1.0f - a) * rho_frame;
        inst.stab.model_evidence = inst.evidence_ema;
    }
    else
        inst.stab.model_evidence = 1.0f;   // no fresh data → don't loosen (freeze-on-stale governs updates)

    const float free_energy = step_model_update(inst, observation.residual_pts,
                                                observation.candidate_pts, residual_precision,
                                                observation.has_fresh_data);

    // Active-perception aids for the controller's lock-on search: where the model projects in the
    // image (centring target) + whether YOLO is currently firing here (dwell/lock signal).
    compute_projected_roi(inst);
    compute_chain_cov(inst);   // Part B: localization/chain cov added to the published RT cov
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;

    // Per-DOF observation info ("times viewed") accumulates only on frames with a FRESH mask, so it
    // hardens the belief by views, not wall-clock.
    if (observation.has_fresh_data)
    {
        // Fisher information filter: fold this viewpoint's (confidence-weighted) per-DOF observation
        // info into the accumulators (predict Q-bleed + update + normalised "equivalent views"), via the
        // shared stabiliser. The weighting + Kalman/CUSUM already ran in accept_table_belief. info_w/info_h
        // mirror the w/h accumulators for the diagnostic log line below.
        stabilizer_.accumulate(inst.stab);
        inst.info_w = inst.stab.fisher_info[2];   // DOF index 2 = w
        inst.info_h = inst.stab.fisher_info[3];   // DOF index 3 = h
        inst.last_residual_pts = observation.residual_pts;   // hold for the voxelizer residual layer
    }

    const auto& s = inst.model.state();
    // Data-only posterior std (mm) from the accumulated Fisher precision — shrinks as evidence
    // gathers; ∞ (shown as -1) until a DOF is first observed. Diagnostic for the EFE epistemic value.
    const auto post_std_mm = [&](int j) -> float {
        return BeliefStabilizer<8>::posterior_std_milli(inst.stab, j);
    };
    if (should_log(inst))
        std::print("[{}] FE={:.4f}  cx={:.3f} cy={:.3f}  w={:.3f} h={:.3f} H={:.3f} L={:.3f} ψ={:.3f} inset={:.3f}  pts={} sil={} silres={:.4f} info=({:.1f},{:.1f}) σ(w,h)mm=({:.1f},{:.1f})\n",
                   inst.node_name, free_energy,
                   s.cx, s.cy, s.w, s.h, s.table_height, s.leg_length, s.yaw, s.leg_inset,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()),
                   inst.model.silhouette_ray_count(), inst.model.silhouette_residual(),
                   inst.info_w, inst.info_h, post_std_mm(2), post_std_mm(3));
    if (should_log(inst))
        std::print("[{}] roi: valid={} offset=({:.2f},{:.2f}) fill={:.2f} | detect alive={} conf={:.2f} since={}\n",
                   inst.node_name, inst.roi_valid, inst.roi_offset_x, inst.roi_offset_y, inst.roi_fill,
                   inst.detection_alive, inst.last_mask_confidence, inst.frames_since_detection);

    log_fisher_csv(inst, observation.has_fresh_data, free_energy,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()),
                   inst.model.silhouette_residual());

    return free_energy;
}

void TableFitter::log_ai2_csv(const TableInstance& inst, int npts, float R, bool gated, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;
    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open()) { cfg_.ai2_csv_path.clear(); return; }
        ai2_csv_ << "cycle,node,npts,gated,energy,R,motion_var,motion_bias,trunc_frac,"
                 << "cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw\n";
    }
    const auto& s = inst.ai2_belief.state();
    const auto& S = inst.ai2_belief.covariance();
    const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };   // posterior std (m / rad)
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << npts << ',' << (gated ? 1 : 0) << ','
             << energy << ',' << R << ',' << inst.last_motion_var << ',' << inst.last_motion_bias << ','
             << inst.last_trunc_frac << ','
             << s.cx << ',' << s.cy << ',' << s.H << ',' << s.w << ',' << s.h << ',' << s.yaw << ','
             << sd(0) << ',' << sd(1) << ',' << sd(2) << ',' << sd(3) << ',' << sd(4) << ',' << sd(5) << '\n';
    ai2_csv_.flush();
}

void TableFitter::log_fisher_csv(const TableInstance& inst, bool fresh, float free_energy,
                                 int point_count, float silres)
{
    if (cfg_.fisher_csv_path.empty())
        return;

    static const std::array<const char*, 8> kDof =
        {"cx", "cy", "w", "h", "H", "leg", "yaw", "inset"};

    if (not fisher_csv_.is_open())
    {
        fisher_csv_.open(cfg_.fisher_csv_path, std::ios::out | std::ios::trunc);
        if (not fisher_csv_.is_open())
        {
            std::print("table_concept: [fisher] cannot open CSV '{}'\n", cfg_.fisher_csv_path);
            cfg_.fisher_csv_path.clear();   // disable further attempts
            return;
        }
        fisher_csv_ << "cycle,node,fresh,fe,warm_conf,settle_maturity,frames_converged,pts,silres,mask_conf,mask_w";
        for (const auto* d : kDof) fisher_csv_ << ",state_" << d;
        for (const auto* d : kDof) fisher_csv_ << ",obs_"   << d;   // this frame's raw Fisher diag
        for (const auto* d : kDof) fisher_csv_ << ",acc_"   << d;   // normalised accumulated info (drives stiffness)
        for (const auto* d : kDof) fisher_csv_ << ",raw_"   << d;   // un-normalised accumulated precision Σ⁻¹
        for (const auto* d : kDof) fisher_csv_ << ",std_"   << d;   // posterior std (mm for lengths, mrad for yaw); -1 = unobserved
        for (const auto* d : kDof) fisher_csv_ << ",gain_"  << d;   // per-DOF Kalman acceptance gain (the calibrated stiffness)
        // Size-bias diagnostic: observed half-extent at 3 percentile pairs vs fitted half-w/h
        // (= state_w/2, state_h/2), the 2-98 span's local centre offset, and the top/leg point split.
        fisher_csv_ << ",ext_hw02,ext_hh02,ext_hw05,ext_hh05,ext_hw10,ext_hh10"
                    << ",ext_offx,ext_offy,n_top,n_leg,n_pts";
        // (D) Counter-evidence gate: per-DOF accumulator S_j (signed excess-bands) and the gate decision
        // (-1 rejected/locked, 0 coherent passthrough, +1 unlocked/snapped).
        for (const auto* d : kDof) fisher_csv_ << ",ce_"  << d;
        for (const auto* d : kDof) fisher_csv_ << ",ceg_" << d;
        fisher_csv_ << ",w_view";   // viewpoint-novelty weight applied to this frame's obs info
        fisher_csv_ << '\n';
    }

    const auto s = inst.model.state().to_array();
    fisher_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << (fresh ? 1 : 0) << ','
                << free_energy << ',' << inst.warm_confidence << ',' << inst.settle_maturity << ','
                << inst.frames_converged << ',' << point_count << ',' << silres << ','
                << inst.last_mask_confidence << ',' << inst.stab.last_mask_conf_weight;
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << s[j];
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.last_obs_info[j];
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.fisher_info[j];
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.fisher_info_raw[j];
    for (int j = 0; j < 8; ++j)
        fisher_csv_ << ',' << BeliefStabilizer<8>::posterior_std_milli(inst.stab, j);
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.last_kalman_gain[j];
    const auto& ed = inst.last_extent_diag;
    fisher_csv_ << ',' << ed.half_ex[0] << ',' << ed.half_ey[0]
                << ',' << ed.half_ex[1] << ',' << ed.half_ey[1]
                << ',' << ed.half_ex[2] << ',' << ed.half_ey[2]
                << ',' << ed.off_x << ',' << ed.off_y
                << ',' << ed.n_top << ',' << ed.n_leg << ',' << ed.n_total;
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.counter_evidence[j];
    for (int j = 0; j < 8; ++j) fisher_csv_ << ',' << inst.stab.last_ce_gate[j];
    fisher_csv_ << ',' << inst.last_view_novelty;
    fisher_csv_ << '\n';
    fisher_csv_.flush();   // flush each row so a plot can tail the file during a live run
}

void TableFitter::step_queue_update(TableInstance& inst,
                                    const std::vector<Eigen::Vector3f>& candidate_pts,
                                    float observation_precision)
{
    const auto sdf_vals = inst.model.compute_sdf(candidate_pts);
    const float precision = std::max(0.05f, observation_precision);
    Eigen::Matrix2f robot_cov = scene_graph_->read_robot_covariance(room_node_id_);
    // Low explanatory adequacy ⇒ low sensory precision ⇒ inflate capture covariance.
    robot_cov /= precision;
    const int q_before = inst.queue.size();
    inst.queue.insert(candidate_pts, sdf_vals, robot_cov, inst.model, inst.matched_frames);
    const int admitted = inst.queue.size() - q_before;
    if (should_log(inst))
        std::print("[{}] queue: admitted={} size={} obs_precision={:.2f}\n",
                   inst.node_name, admitted, inst.queue.size(), observation_precision);
}

float TableFitter::step_model_update(TableInstance& inst,
                                     const std::vector<Eigen::Vector3f>& residual_pts,
                                     const std::vector<Eigen::Vector3f>& current_pts,
                                     float residual_precision,
                                     bool fresh_observation)
{
    const TableState previous_state = inst.model.state();

    const auto evidence = compose_belief_evidence(inst, residual_pts, current_pts, residual_precision);
    if (not evidence.has_evaluation())
    {
        refresh_table_memory(inst);
        inst.last_queue_metrics = inst.queue.metrics();
        return inst.model.compute_free_energy({}, {});
    }

    float free_energy;
    if (not fresh_observation)
    {
        // Information-filter axiom: no new measurement → no belief update. Re-optimising the same frozen
        // anchor set every cycle and re-accepting through a noisy per-frame Kalman gain injects jitter into
        // the accepted state and FE. Recompute FE at the unchanged state for reporting only (queue weights
        // still age, so this drifts only as slowly as the anchor mass decays).
        inst.last_fe_terms = inst.model.compute_free_energy_decomposition(
            evidence.eval_pts, evidence.eval_weights,
            static_cast<std::size_t>(evidence.historical_anchor_count));
        free_energy = inst.last_fe_terms.total_fe;
    }
    else
    {
        evolve_table_belief(inst, evidence);
        free_energy = accept_table_belief(inst, previous_state, evidence);
    }
    refresh_table_memory(inst);
    inst.last_queue_metrics = inst.queue.metrics();

    const auto& fe = inst.last_fe_terms;
    const auto& qm = inst.last_queue_metrics;
    if (should_log(inst))
        std::print("[{}] objective: Lc={:.4f} Lh={:.4f} P={:.4f} FE={:.4f} | anchors={}/{} mass={:.2f} "
                   "rfe=({:.4f},{:.4f},{:.4f}) occ={:.2f} edge={:.2f} | acc={} rej[warm={},sdf={},cap={},bin={}] "
                   "evict[bin={},rfe={}] mean|sdf|={:.4f}\n",
                   inst.node_name, fe.likelihood_current, fe.likelihood_historical, fe.prior, fe.total_fe,
                   qm.anchor_count, qm.capacity, qm.effective_weight_mass,
                   qm.rfe_mean, qm.rfe_p50, qm.rfe_p90, qm.bin_occupancy_ratio, qm.edge_anchor_ratio,
                   qm.counters.accepted_new, qm.counters.rejected_warmup, qm.counters.rejected_sdf,
                   qm.counters.rejected_frame_cap, qm.counters.rejected_bin_rank,
                   qm.counters.evicted_bin_rank, qm.counters.evicted_rfe, qm.mean_abs_sdf);
    return free_energy;
}

std::vector<Eigen::Vector3f> TableFitter::gate_to_top_band(
    const std::vector<Eigen::Vector3f>& pts, const TableModel& model) const
{
    if (!cfg_.top_band_gate_enabled)
        return pts;
    const float H    = model.state().table_height;
    const float band = std::max(0.0f, cfg_.top_band_m);
    std::vector<Eigen::Vector3f> out;
    out.reserve(pts.size());
    for (const auto& p : pts)
        if (std::abs(p.z() - H) <= band)
            out.push_back(p);
    return out;
}

TableFitter::TableBeliefEvidence TableFitter::compose_belief_evidence(
    const TableInstance& inst,
    const std::vector<Eigen::Vector3f>& residual_pts_in,
    const std::vector<Eigen::Vector3f>& current_pts_in,
    float residual_precision) const
{
    // Gate the live observation streams to the table-top band before they enter the fit/extent so
    // off-surface points (floor/under-table/clutter) can't inflate w/h or corrupt the leg/height fit.
    // The queue anchors are already SDF-filtered near the surface, so they're left untouched.
    const std::vector<Eigen::Vector3f> residual_pts = gate_to_top_band(residual_pts_in, inst.model);
    const std::vector<Eigen::Vector3f> current_pts  = gate_to_top_band(current_pts_in, inst.model);

    TableBeliefEvidence evidence;
    evidence.fit_pts = inst.queue.points();
    evidence.fit_weights = inst.queue.weights();
    evidence.eval_pts = evidence.fit_pts;
    evidence.eval_weights = evidence.fit_weights;
    evidence.historical_anchor_count = static_cast<int>(evidence.fit_pts.size());
    evidence.residual_count = static_cast<int>(residual_pts.size());
    evidence.residual_precision = residual_precision;

    for (const auto& residual_pt : residual_pts)
    {
        evidence.eval_pts.push_back(residual_pt);
        evidence.eval_weights.push_back(1.0f);
        if (residual_precision > 1e-3f)
        {
            evidence.fit_pts.push_back(residual_pt);
            evidence.fit_weights.push_back(residual_precision);
        }
    }

    // Live-observation drive: the queue subsamples to a handful of top-face anchors (Lc=0), so the
    // gradient barely sees the current frame and w/h/leg_inset are degenerate/noisy. Feed a bounded
    // strided subsample of THIS frame's points (top + rim + legs) straight into the fit at full
    // precision so the model is pulled onto the actual silhouette/legs every frame.
    const int cap = std::max(0, cfg_.max_direct_fit_points);
    if (cap > 0 && !current_pts.empty())
    {
        const std::size_t stride = std::max<std::size_t>(1, current_pts.size() / static_cast<std::size_t>(cap));
        for (std::size_t i = 0; i < current_pts.size(); i += stride)
        {
            evidence.fit_pts.push_back(current_pts[i]);
            evidence.fit_weights.push_back(1.0f);
            evidence.eval_pts.push_back(current_pts[i]);
            evidence.eval_weights.push_back(1.0f);
        }
    }

    evidence.trusted_point_count = static_cast<int>(evidence.fit_pts.size());
    return evidence;
}

void TableFitter::evolve_table_belief(TableInstance& inst, const TableBeliefEvidence& evidence)
{
    // Optimise on every fresh frame (no convergence freeze): the belief stabiliser controls how much the
    // accepted state actually moves, so the raw fit can always respond to evidence (e.g. grow to a mask
    // that extends past the current span) while the accepted belief stays stable on a settled table.
    if (evidence.can_optimize())
    {
        auto observer = [&](int /*iter*/, const TableState& state, const FreeEnergyDecomposition& /*terms*/)
        {
            TableModel shadow = inst.model;
            shadow.set_state(state);
            inst.queue.refresh_scores(shadow);
        };
        // Decay the GNC start scale over the instance lifetime: wide for the first
        // robust_gnc_decay_cycles (coarse alignment pulls the box over far points), then ramped
        // to the sharp target so the converged pose is no longer re-widened each frame (which
        // otherwise prevents it from ever settling — stable stuck at 0).
        const float gnc_target = cfg_.robust_loss_scale;
        float gnc_start = cfg_.robust_gnc_start_scale;
        if (cfg_.robust_gnc_decay_cycles > 0)
        {
            const float ramp = std::max(0.0f, 1.0f - static_cast<float>(inst.processed_cycles) /
                                                      static_cast<float>(cfg_.robust_gnc_decay_cycles));
            gnc_start = gnc_target + (cfg_.robust_gnc_start_scale - gnc_target) * ramp;
        }
        inst.model.gradient_step(evidence.fit_pts, evidence.fit_weights,
                                 static_cast<std::size_t>(evidence.historical_anchor_count), observer, gnc_start);
    }
    else
    {
        inst.model.compute_free_energy(evidence.fit_pts, evidence.fit_weights);
        inst.queue.refresh_scores(inst.model);
    }
}

void TableFitter::refresh_stabilizer_params()
{
    // Table DOF order = [cx, cy, w, h, H, leg, yaw, inset]: yaw is the periodic DOF; cx,cy are position.
    StabilizerLayout<8> layout;
    layout.yaw_index = 6;
    layout.is_position[0] = true;
    layout.is_position[1] = true;
    stabilizer_.set_layout(layout);

    // The belief model: Fisher filter + Kalman stiffness + maturity + quality rescale + mask-conf
    // weighting, all hardwired on. Stability/plasticity is precision (the evidence-scaled Kalman gain),
    // NOT the CUSUM gate — that is retired (ce_gate=false), subsumed by model_evidence ρ. Only numeric
    // tunings remain configurable.
    StabilizerParams p;
    p.fisher_filter_enabled = true;
    p.kalman_stiffness      = true;
    p.maturity_stiffness    = true;
    p.info_decay            = cfg_.fisher_info_decay;
    p.info_quality_rescale  = true;
    p.info_peak_decay        = cfg_.fisher_peak_decay;
    p.info_peak_ratchet      = cfg_.fisher_peak_ratchet;
    p.info_peak_ema          = cfg_.fisher_peak_ema;
    p.views_half            = cfg_.fisher_views_half;
    p.process_std_len       = cfg_.fisher_process_std_m;
    p.process_std_ang       = cfg_.fisher_process_std_yaw;
    p.process_std_pos       = cfg_.process_std_pos_m;   // static-centre lock (cx,cy): small Q vs the size DOFs
    p.mask_conf_weight      = true;
    p.mask_conf_floor       = cfg_.mask_conf_floor;
    p.mask_conf_ref         = cfg_.mask_conf_ref;
    p.mask_conf_power       = cfg_.mask_conf_power;
    p.ce_gate               = false;   // CUSUM retired — see above (ρ is the falsifiability mechanism)
    stabilizer_.set_params(p);
}

float TableFitter::accept_table_belief(TableInstance& inst,
                                       const TableState& previous_state,
                                       const TableBeliefEvidence& evidence)
{
    const auto finite_state = [](const TableState& state)
    {
        return std::isfinite(state.cx) && std::isfinite(state.cy) &&
               std::isfinite(state.w) && std::isfinite(state.h) &&
               std::isfinite(state.table_height) && std::isfinite(state.leg_length) &&
               std::isfinite(state.yaw) && std::isfinite(state.leg_inset);
    };

    const auto revert = [&]() -> float
    {
        inst.model.set_state(previous_state);
        inst.model.set_prior(previous_state);
        inst.last_fe_terms = inst.model.compute_free_energy_decomposition(
            evidence.eval_pts, evidence.eval_weights,
            static_cast<std::size_t>(evidence.historical_anchor_count));
        return inst.last_fe_terms.total_fe;
    };

    const TableState raw_state = inst.model.state();
    if (!finite_state(raw_state))
        return revert();

    // (A2) FE-spike guard: a fresh-but-contaminated frame (clutter / partial-or-wrong mask) produces a
    // raw-fit FE far above this instance's running level. Don't let it move the belief — revert (keeps the
    // state, recomputes FE for reporting) BEFORE the stabiliser, so the gate can't unlock onto garbage.
    // The information-filter axiom for "fresh but untrustworthy". Baseline updated below on accepted frames.
    if (std::isfinite(inst.fe_baseline) and inst.fe_baseline > 1e-3f)
    {
        const float raw_fe = inst.model.compute_free_energy_decomposition(
            evidence.eval_pts, evidence.eval_weights,
            static_cast<std::size_t>(evidence.historical_anchor_count)).total_fe;
        if (std::isfinite(raw_fe) and raw_fe > cfg_.bad_fit_fe_ratio * inst.fe_baseline)
        {
            if (should_log(inst))
                std::print("[{}] bad-fit frame rejected: raw_fe={:.2f} > {:.1f}× baseline {:.2f}\n",
                           inst.node_name, raw_fe, cfg_.bad_fit_fe_ratio, inst.fe_baseline);
            return revert();
        }
    }

    // Measure this viewpoint's observation Fisher information (per DOF) at the raw fit, and derive
    // the per-DOF Kalman gain K_j = obs_j / (Y_pred_j + obs_j) from the calibrated filter (Y_pred =
    // previous posterior precision after the Q-bleed predict). Always computed (logged); only used as
    // the acceptance stiffness when fisher_kalman_stiffness is on. The filter STATE update (predict +
    // posterior) stays in run_inference's fresh-mask block — here we only READ the prior precision.
    // Per-DOF belief stabiliser (shared rc::BeliefStabilizer): weight this viewpoint's observation
    // Fisher info by the mask confidence, derive the maturity-stiffened Kalman acceptance gain from the
    // Q-bleed filter, and run the CUSUM/SPRT gate (reject lone surprises, ease onto sustained coherent
    // ones). The filter STATE accumulation (predict + posterior) stays in run_inference's fresh block.
    refresh_stabilizer_params();
    auto obs_info = inst.model.observation_information(evidence.eval_pts, evidence.eval_weights);
    // Viewpoint-novelty gate: a static robot re-observes the SAME partial view every frame, so folding
    // each frame's Fisher info in as independent evidence makes the belief overconfident about geometry it
    // never saw (the occluded far depth h collapses to mm-precision). Attenuate the info by how far the
    // camera has moved since the last accumulated vantage — a redundant same-pose frame contributes only
    // the floor (sensor-noise averaging), so with the predict Q-bleed the steady-state precision reflects
    // viewpoint DIVERSITY, not dwell time. Under-observed DOFs stay honestly loose until the robot orbits.
    if (cfg_.view_novelty_gate)
    {
        float w_view = 1.0f;
        if (const auto M = room_T_zed_matrix(inst.last_mask_timestamp_ms); M.has_value())
        {
            const Eigen::Vector3f cam(static_cast<float>((*M)(0, 3)),
                                      static_cast<float>((*M)(1, 3)),
                                      static_cast<float>((*M)(2, 3)));
            const float scale = std::max(1e-3f, cfg_.view_novelty_scale_m);
            const float floor = std::clamp(cfg_.view_novelty_floor, 0.0f, 1.0f);
            const float delta = inst.has_view_pos ? (cam - inst.last_view_pos).norm()
                                                  : std::numeric_limits<float>::max();
            w_view = floor + (1.0f - floor) * std::clamp(delta / scale, 0.0f, 1.0f);
            if (not inst.has_view_pos or delta >= scale)   // genuinely new vantage → register it
            {
                inst.last_view_pos = cam;
                inst.has_view_pos  = true;
            }
        }
        inst.last_view_novelty = w_view;
        for (auto& v : obs_info)
            v *= w_view;
    }
    stabilizer_.weight_observation(inst.stab, obs_info, inst.last_mask_confidence);
    stabilizer_.compute_acceptance(inst.stab, raw_state.to_array(), previous_state.to_array());

    const auto coverage = inst.queue.face_coverage(inst.model);

    inst.warm_confidence = TableBeliefPolicy::update_warm_confidence(
        inst.warm_confidence, cfg_, coverage,
        evidence.trusted_point_count, evidence.residual_count, evidence.residual_precision);

    TableState accepted_state = TableBeliefPolicy::apply_observability_warm_start(
        previous_state, raw_state, inst.stab.last_kalman_gain);
    if (!finite_state(accepted_state))
        return revert();

    // Canonicalise the footprint to w ≥ h (fold yaw into a π/2 wedge). A box has a 90° symmetry —
    // (w,h,ψ) ≡ (h,w,ψ+π/2) — so without a convention a yaw flip silently swaps w↔h, and with the size
    // RATCHET (grow-only) each flip ratchets BOTH dims up → the table inflates toward square → yaw gets
    // ever more ambiguous → more flips (the runaway). Enforcing w≥h makes the representation unique so a
    // flip maps back to the SAME (w,h,ψ); the w/h Fisher accumulators are swapped too so precision follows
    // its dimension. Converges in ≤1 swap then holds (the optimiser won't recross w=h once canonical).
    if (accepted_state.w < accepted_state.h)
    {
        std::swap(accepted_state.w, accepted_state.h);
        accepted_state.yaw = TableBeliefPolicy::wrap_angle(accepted_state.yaw + 0.5f * M_PIf);
        BeliefStabilizer<8>::swap_dofs(inst.stab, 2, 3);   // DOF 2 = w, 3 = h
    }

    inst.model.set_state(accepted_state);
    // Do NOT update prior_ to accepted_state — keep prior anchored to the previous frame's state so
    // the state-transition prior (lambda_state/pos/angle) acts as a true per-frame dampener.

    // Footprint-extent / point-attribution diagnostic at the accepted state (for the w/h size bias).
    inst.last_extent_diag = inst.model.extent_diagnostics(evidence.eval_pts);

    inst.last_fe_terms = inst.model.compute_free_energy_decomposition(
        evidence.eval_pts, evidence.eval_weights,
        static_cast<std::size_t>(evidence.historical_anchor_count));
    const float free_energy = inst.last_fe_terms.total_fe;
    if (!std::isfinite(free_energy))
        return revert();

    // Update the FE-spike baseline on ACCEPTED frames only (good evidence), so a contaminated frame that
    // was reverted above never raises the bar that detects the next one.
    inst.fe_baseline = std::isfinite(inst.fe_baseline)
        ? (1.0f - cfg_.fe_baseline_ema) * inst.fe_baseline + cfg_.fe_baseline_ema * free_energy
        : free_energy;

    if (should_log(inst))
        std::print("[{}] warm-start: conf={:.2f} mask_conf={:.2f} w={:.2f} rho_x={:.2f} rho_y={:.2f} residual={} trusted_pts={} residual_precision={:.2f} raw(w={:.3f},h={:.3f},psi={:.3f}) accepted(w={:.3f},h={:.3f},psi={:.3f})\n",
                   inst.node_name, inst.warm_confidence, inst.last_mask_confidence, inst.stab.last_mask_conf_weight,
                   std::min(coverage[0], coverage[1]) / (std::max(coverage[0], coverage[1]) + 1e-3f),
                   std::min(coverage[2], coverage[3]) / (std::max(coverage[2], coverage[3]) + 1e-3f),
                   evidence.residual_count, evidence.trusted_point_count, evidence.residual_precision,
                   raw_state.w, raw_state.h, raw_state.yaw,
                   accepted_state.w, accepted_state.h, accepted_state.yaw);

    return free_energy;
}

void TableFitter::refresh_table_memory(TableInstance& inst)
{
    const Eigen::Matrix2f robot_cov = scene_graph_->read_robot_covariance(room_node_id_);
    inst.queue.update_rfe(inst.model, robot_cov);
}

// ─── RGB-mask silhouette ──────────────────────────────────────────────────────

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

void TableFitter::feed_silhouette(TableInstance& inst)
{
    inst.model.clear_silhouette();
    if (cfg_.mask_precision <= 0.0f or not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }

    const auto& packet = mask_ingestor_->packet();
    const auto slice = mask_ingestor_->select_nearest(Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().table_height), "table");
    if (not slice.has_value() or slice->pixel_end <= slice->pixel_begin
        or slice->pixel_end > packet.mask_pixels.size())
        return;

    // Back-project the contour through the pose AT THE MASK'S CAPTURE TIME, not the current pose — a
    // moving base between capture and consumption would otherwise cast the rays from the wrong origin
    // and bias the silhouette pull on w/h/yaw/pos. ts=0 (old producer) → falls back to the current pose.
    const auto Mopt = room_T_zed_matrix(packet.timestamp_ms);   // room_T_zed (camera→room) at capture time
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d& M = Mopt.value();
    const Eigen::Vector3f C(static_cast<float>(M(0, 3)), static_cast<float>(M(1, 3)), static_cast<float>(M(2, 3)));

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float cx_px = static_cast<float>(camera_api_->get_width())  * 0.5f;
    const float cy_px = static_cast<float>(camera_api_->get_height()) * 0.5f;

    // Contour = per-row min/max column (left/right edges) + per-column min/max row (near/far edges),
    // so all four top-rectangle edges are sampled.
    std::unordered_map<int, std::pair<float, float>> row_mm, col_mm;
    for (std::size_t i = slice->pixel_begin; i < slice->pixel_end; ++i)
    {
        const float col = packet.mask_pixels[i].x();
        const float row = packet.mask_pixels[i].y();
        const int ri = static_cast<int>(row), ci = static_cast<int>(col);
        if (auto it = row_mm.find(ri); it == row_mm.end()) row_mm.emplace(ri, std::pair{col, col});
        else { it->second.first = std::min(it->second.first, col); it->second.second = std::max(it->second.second, col); }
        if (auto it = col_mm.find(ci); it == col_mm.end()) col_mm.emplace(ci, std::pair{row, row});
        else { it->second.first = std::min(it->second.first, row); it->second.second = std::max(it->second.second, row); }
    }

    std::vector<Eigen::Vector3f> dirs;
    dirs.reserve((row_mm.size() + col_mm.size()) * 2);
    const auto add = [&](float col, float row)
    {
        // d_cam = ((col-cx)/fx, 1, (cy-row)/fy); rotate to room (rotation block of M).
        const double dcx = (col - cx_px) / fx, dcz = (cy_px - row) / fy;
        const double rx = M(0, 0) * dcx + M(0, 1) + M(0, 2) * dcz;
        const double ry = M(1, 0) * dcx + M(1, 1) + M(1, 2) * dcz;
        const double rz = M(2, 0) * dcx + M(2, 1) + M(2, 2) * dcz;
        if (rz < -1e-2)   // ray must descend to intersect the table-top plane below the camera
            dirs.emplace_back(static_cast<float>(rx), static_cast<float>(ry), static_cast<float>(rz));
    };
    for (const auto& [r, mm] : row_mm) { add(mm.first, static_cast<float>(r)); add(mm.second, static_cast<float>(r)); }
    for (const auto& [c, mm] : col_mm) { add(static_cast<float>(c), mm.first); add(static_cast<float>(c), mm.second); }

    // Report how stale the consumed mask is: lag from its capture stamp to now (same wall-clock epoch as
    // the producer's frame stamp). A large lag while the base is moving is exactly what the capture-time
    // pose pinning above corrects — this quantifies how much it's buying. Throttled; only with a stamp.
    if (packet.timestamp_ms > 0 and should_log(inst))
    {
        const auto now_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const long long lag_ms = static_cast<long long>(now_ms) - static_cast<long long>(packet.timestamp_ms);
        std::print("[{}] silhouette: mask capture→now lag={} ms (pose pinned to capture) rays={}\n",
                   inst.node_name, lag_ms, dirs.size());
    }

    if (not dirs.empty())
        inst.model.set_silhouette(C, std::move(dirs), slice->confidence);
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
    p.sigma_obs          = cfg_.sigma_obs;
    p.lambda_size        = cfg_.lambda_size;
    p.lambda_pos         = cfg_.lambda_pos;
    p.lambda_extent      = cfg_.lambda_extent;
    p.extent_pct_lo      = cfg_.extent_pct_lo;
    p.extent_pct_hi      = cfg_.extent_pct_hi;
    p.lambda_state       = cfg_.lambda_state;
    p.lambda_angle       = cfg_.lambda_angle;
    p.prior_size_std     = cfg_.prior_size_std;
    p.optimization_iters = cfg_.optimization_iters;
    p.optimization_lr    = cfg_.optimization_lr;
    p.grad_clip          = cfg_.grad_clip;
    p.optimizer_type     = cfg_.optimizer_type;
    p.sgd_momentum       = cfg_.sgd_momentum;
    p.robust_loss        = cfg_.robust_loss;
    p.robust_loss_scale  = cfg_.robust_loss_scale;
    p.robust_gnc_start_scale = cfg_.robust_gnc_start_scale;
    p.mask_precision     = cfg_.mask_precision;
    p.sil_tangent_samples = cfg_.sil_tangent_samples;
    p.fisher_grad_clamp  = cfg_.fisher_grad_clamp;
    return p;
}

SampleQueueParams TableFitter::make_queue_params() const
{
    SampleQueueParams p;
    p.num_angle_bins               = cfg_.num_angle_bins;
    p.num_z_bins                   = cfg_.num_z_bins;
    p.max_per_bin                  = cfg_.max_per_bin;
    p.sdf_threshold_for_storage    = cfg_.sdf_threshold_for_storage;
    p.min_frames_before_historical = cfg_.min_frames_before_historical;
    p.historical_warmup_frames     = cfg_.historical_warmup_frames;
    p.max_new_points_per_frame     = cfg_.max_new_points_per_frame;
    p.rfe_alpha                    = cfg_.rfe_alpha;
    p.rfe_max_threshold            = cfg_.rfe_max_threshold;
    p.rfe_weight_gain              = cfg_.rfe_weight_gain;
    p.min_anchor_weight            = cfg_.min_anchor_weight;
    p.edge_bonus_weight            = cfg_.edge_bonus_weight;
    p.edge_proximity_threshold     = cfg_.edge_proximity_threshold;
    p.diversity_admission          = true;   // table: spatial-diversity admission (preserve pre-template behaviour)
    return p;
}

}  // namespace rc
