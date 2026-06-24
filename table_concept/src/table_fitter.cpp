/*
 * table_fitter.cpp — the active-inference fit core for table_concept.
 */

#include "table_fitter.h"

#include <algorithm>
#include <cmath>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

TableFitter::TableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                         DSR::InnerEigenAPI* inner_eigen,
                         TableConfig& cfg,
                         const std::vector<TablePrior>& priors,
                         MaskIngestor* mask_ingestor,
                         TableSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg), priors_(priors),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{}

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
    const TableModelParams& /*params*/,
    const TableConfig& cfg,
    float confidence,
    const std::array<float, 6>& coverage,
    int point_count,
    float settle_gain,
    float info_w,
    float info_h)
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

    // A full top-face view (high rho_pts) constrains w/h directly from the silhouette extent, even
    // with no bilateral vertical-side coverage. Without this, an oversized box reads rho_x/rho_y≈0
    // (rim points sit far inside the too-wide side planes) → freeze → it can never shrink: a
    // deadlock. When point evidence is strong, trust size from points and release the freeze.
    const bool strong_evidence = rho_pts >= cfg.warm_size_pts_release;
    const float rho_pos = rho_pts * std::max(rho_x, rho_y);
    const float rho_size_x = strong_evidence ? std::max(rho_pts * rho_x, rho_pts) : rho_pts * rho_x;
    const float rho_size_y = strong_evidence ? std::max(rho_pts * rho_y, rho_pts) : rho_pts * rho_y;
    const float rho_vertical = rho_pts;
    const float yaw_support = clamp01(0.25f * std::max(rho_x, rho_y) + 0.75f * std::sqrt(rho_x * rho_y));

    // Evidence-hardening: a well-observed extent (high accumulated per-face info) STIFFENS — its
    // acceptance gain shrinks toward 0, so re-encountered masks barely move it (belief→knowledge).
    // An unobserved face has info≈0 → stiffness≈1 → it stays plastic until first seen.
    const float ih = std::max(1e-3f, cfg.warm_info_half);
    const float stiff_w = ih / (ih + info_w);
    const float stiff_h = ih / (ih + info_h);

    const float lambda_pos = settle_gain * lerp(cfg.warm_lambda_pos_base + cfg.warm_lambda_pos_gain * rho_pos, 0.95f, confidence);
    const float lambda_size_x = settle_gain * stiff_w * lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_size_x, 0.95f, confidence);
    const float lambda_size_y = settle_gain * stiff_h * lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_size_y, 0.95f, confidence);
    const float lambda_vertical = settle_gain * lerp(cfg.warm_lambda_size_base + cfg.warm_lambda_size_gain * rho_vertical, 0.90f, confidence);
    const float lambda_yaw = settle_gain * lerp(cfg.warm_lambda_yaw_base + cfg.warm_lambda_yaw_gain * (rho_pts * yaw_support), 0.70f, confidence);

    const float effective_side_min = cfg.warm_coverage_min_side * (1.0f - 0.6f * confidence);
    const float effective_rho_freeze = cfg.warm_rho_freeze * (1.0f - 0.7f * confidence);

    const bool freeze_x = !strong_evidence && (std::min(cov_px, cov_nx) < effective_side_min || rho_x < effective_rho_freeze);
    const bool freeze_y = !strong_evidence && (std::min(cov_py, cov_ny) < effective_side_min || rho_y < effective_rho_freeze);

    TableState accepted = raw;
    accepted.cx = lerp(previous.cx, raw.cx, lambda_pos);
    accepted.cy = lerp(previous.cy, raw.cy, lambda_pos);
    accepted.w = freeze_x ? previous.w : lerp(previous.w, raw.w, lambda_size_x);
    accepted.h = freeze_y ? previous.h : lerp(previous.h, raw.h, lambda_size_y);
    accepted.table_height = lerp(previous.table_height, raw.table_height, lambda_vertical);
    accepted.leg_length = lerp(previous.leg_length, raw.leg_length, lambda_vertical);
    accepted.yaw = angle_lerp(previous.yaw, raw.yaw, lambda_yaw);
    // leg_inset was previously accepted raw (unsmoothed) → it swung every frame. Damp it like the
    // other size DOFs so it settles with maturity.
    accepted.leg_inset = lerp(previous.leg_inset, raw.leg_inset, 0.5f * (lambda_size_x + lambda_size_y));

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
        fix(init_state.leg_inset, 0.045f, "leg_inset");
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
    {
        const auto it = std::find_if(priors_.begin(), priors_.end(),
                                     [&](const TablePrior& prior){ return prior.node_name == node.name(); });
        if (it != priors_.end())
            mparams.prior_size_std = std::max(mparams.prior_size_std, it->sigma_size);
    }

    inst.model     = TableModel(init_state, mparams);
    inst.queue     = SampleQueue(make_queue_params());
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
        const auto selected_mask = mask_ingestor_->select_for_table(inst);
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this table on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
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

                if (should_log_table(inst))
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

    if (should_log_table(inst))
        std::print("[{}] ↓ frame={} cands={} resid={} expl={:.2f}\n",
                   inst.node_name, last_frame,
                   observation.candidate_pts.size(), observation.residual_pts.size(),
                   observation.explanation_ratio);
    return observation;
}

bool TableFitter::should_log_table(const TableInstance& inst) const
{
    const int period = std::max(1, cfg_.table_log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

// ─── Inference ───────────────────────────────────────────────────────────────

float TableFitter::run_inference(TableInstance& inst, const TableObservation& observation)
{
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

    // A fresh mask whose contour disagrees with the (possibly frozen) model — high silhouette
    // residual — is NEW evidence: re-open the converged fit and re-energise the acceptance gain so
    // the silhouette term can actually pull w/h/pose onto the mask (e.g. expand an undersized box),
    // instead of merely reporting the mismatch. Mirrors the queue-admission re-open.
    if (inst.frames_converged >= cfg_.K_stable
        && inst.model.silhouette_ray_count() > 0
        && inst.model.silhouette_residual() > cfg_.sil_reopen_residual_m)
    {
        if (should_log_table(inst))
            std::print("[{}] silhouette re-open: silres={:.3f} > {:.3f} → reopen fit\n",
                       inst.node_name, inst.model.silhouette_residual(), cfg_.sil_reopen_residual_m);
        inst.frames_converged = cfg_.K_stable / 2;
        inst.settle_maturity = 0;
    }

    const float free_energy = step_model_update(inst, observation.residual_pts,
                                                observation.candidate_pts, residual_precision);

    // Active-perception aids for the controller's lock-on search: where the model projects in the
    // image (centring target) + whether YOLO is currently firing here (dwell/lock signal).
    compute_projected_roi(inst);
    inst.detection_alive = inst.frames_since_detection < cfg_.detection_alive_max_frames;

    // Per-DOF observation info ("times viewed") accumulates only on frames with a FRESH mask, so it
    // hardens w/h by views, not wall-clock. info_w ← x-faces, info_h ← y-faces (saturating).
    if (observation.has_fresh_data)
    {
        const auto cov = inst.queue.face_coverage(inst.model);
        const float dm = std::max(1.0f, cfg_.delta_min);
        inst.info_w += TableBeliefPolicy::clamp01(std::max(cov[0], cov[1]) / dm);
        inst.info_h += TableBeliefPolicy::clamp01(std::max(cov[2], cov[3]) / dm);
        inst.last_residual_pts = observation.residual_pts;   // hold for the voxelizer residual layer
    }

    const auto& s = inst.model.state();
    if (should_log_table(inst))
        std::print("[{}] FE={:.4f}  cx={:.3f} cy={:.3f}  w={:.3f} h={:.3f} H={:.3f} L={:.3f} ψ={:.3f} inset={:.3f}  pts={} sil={} silres={:.4f} info=({:.0f},{:.0f})\n",
                   inst.node_name, free_energy,
                   s.cx, s.cy, s.w, s.h, s.table_height, s.leg_length, s.yaw, s.leg_inset,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()),
                   inst.model.silhouette_ray_count(), inst.model.silhouette_residual(),
                   inst.info_w, inst.info_h);
    if (should_log_table(inst))
        std::print("[{}] roi: valid={} offset=({:.2f},{:.2f}) fill={:.2f} | detect alive={} conf={:.2f} since={}\n",
                   inst.node_name, inst.roi_valid, inst.roi_offset_x, inst.roi_offset_y, inst.roi_fill,
                   inst.detection_alive, inst.last_mask_confidence, inst.frames_since_detection);

    return free_energy;
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
    if (admitted > 0 && inst.frames_converged >= cfg_.K_stable)
        inst.frames_converged = cfg_.K_stable / 2;
    // A burst of NET new anchors = a genuinely new viewpoint (new faces fill empty bins). Static
    // churn nets ~0 (admit≈evict), so this ignores noise. Re-energise the acceptance gain so the
    // model adapts to the new evidence, then re-settles.
    if (admitted >= cfg_.warm_reopen_admit)
        inst.settle_maturity = 0;
    if (should_log_table(inst))
        std::print("[{}] queue: admitted={} size={} obs_precision={:.2f}\n",
                   inst.node_name, admitted, inst.queue.size(), observation_precision);
}

float TableFitter::step_model_update(TableInstance& inst,
                                     const std::vector<Eigen::Vector3f>& residual_pts,
                                     const std::vector<Eigen::Vector3f>& current_pts,
                                     float residual_precision)
{
    const TableState previous_state = inst.model.state();

    const auto evidence = compose_belief_evidence(inst, residual_pts, current_pts, residual_precision);
    if (not evidence.has_evaluation())
    {
        refresh_table_memory(inst);
        inst.last_queue_metrics = inst.queue.metrics();
        return inst.model.compute_free_energy({}, {});
    }

    evolve_table_belief(inst, evidence);
    const float free_energy = accept_table_belief(inst, previous_state, evidence);
    refresh_table_memory(inst);
    inst.last_queue_metrics = inst.queue.metrics();

    const auto& fe = inst.last_fe_terms;
    const auto& qm = inst.last_queue_metrics;
    if (should_log_table(inst))
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

TableFitter::TableBeliefEvidence TableFitter::compose_belief_evidence(
    const TableInstance& inst,
    const std::vector<Eigen::Vector3f>& residual_pts,
    const std::vector<Eigen::Vector3f>& current_pts,
    float residual_precision) const
{
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
    // Freeze gradient descent once converged to prevent oscillation; step_queue_update unlocks it
    // automatically when new points arrive.
    if (inst.frames_converged < cfg_.K_stable && evidence.can_optimize())
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

    const auto coverage = inst.queue.face_coverage(inst.model);

    inst.warm_confidence = TableBeliefPolicy::update_warm_confidence(
        inst.warm_confidence, cfg_, coverage,
        evidence.trusted_point_count, evidence.residual_count, evidence.residual_precision);

    // Acceptance-gain damping driven by settle_maturity: a counter that climbs each cycle and is
    // reset ONLY by a genuine new-evidence burst (net new queue anchors ⇒ a fresh viewpoint), in
    // step_queue_update. High gain while immature (startup or just after the controller moves the
    // robot) → adapts fast; decays to a low floor once mature → smooth lock. It deliberately does
    // NOT react to frames_converged / per-frame state jitter, which would spike the gain on noise
    // and make the fit jerky.
    const float maturity = TableBeliefPolicy::clamp01(static_cast<float>(inst.settle_maturity) /
                                                      static_cast<float>(std::max(1, cfg_.warm_settle_cycles)));
    const float settle_gain = TableBeliefPolicy::lerp(1.0f, cfg_.warm_settle_floor, maturity);

    const TableState accepted_state = TableBeliefPolicy::apply_observability_warm_start(
        previous_state, raw_state, inst.model.params(), cfg_, inst.warm_confidence,
        coverage, evidence.trusted_point_count, settle_gain, inst.info_w, inst.info_h);
    if (!finite_state(accepted_state))
        return revert();

    inst.model.set_state(accepted_state);
    // Do NOT update prior_ to accepted_state — keep prior anchored to the previous frame's state so
    // the state-transition prior (lambda_state/pos/angle) acts as a true per-frame dampener.

    inst.last_fe_terms = inst.model.compute_free_energy_decomposition(
        evidence.eval_pts, evidence.eval_weights,
        static_cast<std::size_t>(evidence.historical_anchor_count));
    const float free_energy = inst.last_fe_terms.total_fe;
    if (!std::isfinite(free_energy))
        return revert();

    if (should_log_table(inst))
        std::print("[{}] warm-start: conf={:.2f} rho_x={:.2f} rho_y={:.2f} residual={} trusted_pts={} residual_precision={:.2f} raw(w={:.3f},h={:.3f},psi={:.3f}) accepted(w={:.3f},h={:.3f},psi={:.3f})\n",
                   inst.node_name, inst.warm_confidence,
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

std::optional<Eigen::Matrix4d> TableFitter::room_T_zed_matrix() const
{
    if (not inner_eigen_)
        return std::nullopt;
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", 0);
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
    const auto slice = mask_ingestor_->select_for_table(inst);
    if (not slice.has_value() or slice->pixel_end <= slice->pixel_begin
        or slice->pixel_end > packet.mask_pixels.size())
        return;

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room)
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

    if (inserted > 0 && should_log_table(inst))
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
    return p;
}

}  // namespace rc
