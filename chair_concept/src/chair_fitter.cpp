/*
 * chair_fitter.cpp — the active-inference fit core for chair_concept.
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
                         const std::vector<ChairPrior>& priors,
                         MaskIngestor* mask_ingestor,
                         ChairSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg), priors_(priors),
      mask_ingestor_(mask_ingestor), scene_graph_(scene_graph)
{}

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

// ─── ChairBeliefPolicy ───────────────────────────────────────────────────────

float ChairFitter::ChairBeliefPolicy::clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float ChairFitter::ChairBeliefPolicy::lerp(float start, float end, float gain)
{
    return start + gain * (end - start);
}

float ChairFitter::ChairBeliefPolicy::wrap_angle(float angle)
{
    while (angle > M_PIf) angle -= 2.0f * M_PIf;
    while (angle < -M_PIf) angle += 2.0f * M_PIf;
    return angle;
}

float ChairFitter::ChairBeliefPolicy::angle_lerp(float start, float end, float gain)
{
    return wrap_angle(start + gain * wrap_angle(end - start));
}

ChairState ChairFitter::ChairBeliefPolicy::apply_observability_warm_start(
    const ChairState& previous,
    const ChairState& raw,
    const ChairModelParams& /*params*/,
    const ChairConfig& /*cfg*/,
    float /*confidence*/,
    const std::array<float, 6>& /*coverage*/,
    int /*point_count*/,
    float settle_gain,
    float /*info_w*/,
    float /*info_h*/,
    const std::array<float, 8>* kalman_gain)
{
    // Generic per-DOF acceptance lerp. Drive each DOF directly by its per-DOF Kalman gain
    // K_j = obs_info_j / (Y_pred_j + obs_info_j) from the information filter (compute_acceptance);
    // a settle_gain fallback if the filter is off. Chair DOF order = [cx,cy,cz,yaw,seat_w,seat_d,seat_h,back_h].
    auto prev_a = previous.to_array();
    auto raw_a  = raw.to_array();
    std::array<float, 8> acc{};
    for (int j = 0; j < 8; ++j)
    {
        float g = kalman_gain ? (*kalman_gain)[j] : settle_gain;
        float d = raw_a[j] - prev_a[j];
        if (j == 3) d = std::remainder(d, 2.0f * static_cast<float>(M_PI));   // yaw wrap
        acc[j] = prev_a[j] + g * d;
    }
    acc[2] = prev_a[2];   // cz anchored — keep the support/floor value (set by the support step)
    return ChairState::from_array(acc);
}

float ChairFitter::ChairBeliefPolicy::update_warm_confidence(
    float previous_confidence,
    const ChairConfig& cfg,
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
    // back_h has no dedicated DSR attr — seed it from the matching prior (default 0.45) so the
    // height_m round-trip below recovers seat_h correctly.
    if (const auto it = std::find_if(priors_.begin(), priors_.end(),
            [&](const ChairPrior& pr){ return pr.node_name == node.name(); }); it != priors_.end())
        init_state.back_h = it->back_h;
    // height_m carries the OVERALL height (seat_h + back_h, see ChairSceneGraph); recover seat_h by
    // subtracting back_h — else seat_h reads back as the full 0.9 m and the model fits a giant chair.
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

    // Sanitize: a NaN/Inf from a corrupted RT edge would poison the SDF and lock the
    // optimizer; replace any non-finite field with a safe default before it reaches the model.
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

    // Anchor the size prior to this chair's actual prior geometry, else the KL size term
    // pulls every chair toward the hardcoded defaults regardless of the loaded prior/observations.
    ChairModelParams mparams = make_model_params();
    mparams.prior_seat_w = init_state.seat_w;
    mparams.prior_seat_d = init_state.seat_d;
    mparams.prior_seat_h = init_state.seat_h;
    mparams.prior_back_h = init_state.back_h;
    {
        const auto it = std::find_if(priors_.begin(), priors_.end(),
                                     [&](const ChairPrior& prior){ return prior.node_name == node.name(); });
        if (it != priors_.end())
            mparams.prior_size_std = std::max(mparams.prior_size_std, it->sigma_size);
    }

    inst.model     = ChairModel(init_state, mparams);
    inst.queue     = SampleQueue<ChairModel>(make_queue_params());
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
    // inliers as queue anchors (candidates) and the rest as residuals that drive model expansion.
    const auto& masks_packet = mask_ingestor_->packet();
    if (masks_packet.valid && masks_packet.frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance: the tracker's gated assignment when present (multi-instance safe),
        // else the legacy greedy nearest-to-our-centre.
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = masks_packet.slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        else
            selected_mask = mask_ingestor_->select_nearest(Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().seat_h), "chair");
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO fired for this chair on a fresh frame → detection is alive.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence = slice.confidence;
            inst.last_mask_timestamp_ms = masks_packet.timestamp_ms;   // chain-cov pinning (Part B)
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

// ─── Inference ───────────────────────────────────────────────────────────────

float ChairFitter::run_inference(ChairInstance& inst, const ChairObservation& observation)
{
    inst.queue.begin_cycle();

    ++inst.settle_maturity;   // climbs each cycle; step_queue_update resets it on a new-evidence burst

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);

    // Cold-start: on the FIRST frame with data, snap model & prior to the observed mask centroid so
    // gradient descent begins on the chair. Use candidates if any, else queue anchors, else the RAW
    // residual support points — so it fires even when the model starts so far (e.g. a missing RT seed)
    // that nothing matched and the queue is still empty. Without the residual fallback the snap is
    // gated behind a bank that can never fill while the model sits at the origin.
    if (observation.has_fresh_data and inst.matched_frames == 0)
    {
        const auto& src = not observation.candidate_pts.empty() ? observation.candidate_pts
                        : not inst.voxel_bank_pts.empty()       ? inst.voxel_bank_pts
                                                                 : observation.residual_pts;
        if (not src.empty())
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            for (const auto& p : src) sum += p;
            const Eigen::Vector3f cen = sum / static_cast<float>(src.size());
            auto s  = inst.model.state();
            s.cx = cen.x();
            s.cy = cen.y();
            inst.model.set_state(s);
            inst.model.set_prior(s);   // zero KL so the data term dominates from the start
            std::print("[{}] cold-start snap → cx={:.2f} cy={:.2f} ({} pts)\n",
                       inst.node_name, s.cx, s.cy, src.size());
            inst.matched_frames = cfg_.min_frames_before_historical + cfg_.historical_warmup_frames + 1;
        }
    }
    else if (observation.has_fresh_data)
        ++inst.matched_frames;

    if (observation.has_fresh_data and not inst.voxel_bank_pts.empty())
        step_queue_update(inst, inst.voxel_bank_pts,
                          ChairBeliefPolicy::clamp01(observation.explanation_ratio));

    const float explanation_confidence = ChairBeliefPolicy::clamp01(observation.explanation_ratio);
    float residual_precision = ChairBeliefPolicy::clamp01(inst.warm_confidence * (1.0f - explanation_confidence));
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
        if (should_log(inst))
            std::print("[{}] silhouette re-open: silres={:.3f} > {:.3f} → reopen fit\n",
                       inst.node_name, inst.model.silhouette_residual(), cfg_.sil_reopen_residual_m);
        inst.frames_converged = cfg_.K_stable / 2;
        inst.settle_maturity = 0;
    }

    // Resolve the discrete seat-square ORIENTATION ambiguity before the gradient refine. A near-square
    // seat + 4-corner legs are symmetric under 90°/180°; only the BACKREST breaks the tie, and local
    // gradient descent can lodge in the wrong well (e.g. a 180° flip). The robust loss SATURATES the
    // backrest mismatch in the full FE (far points read as outliers), so instead compare the RAW mean
    // |SDF| of only the ELEVATED points (the backrest band, above the seat) across the four cardinal
    // orientations — that signal is ~10× stronger and un-saturated. Hard-snap to the minimum, BEFORE
    // step_model_update captures previous_state, so the acceptance lerp follows the flip. Skips
    // (keeps the current heading) when too few elevated points are visible (backrest occluded), so an
    // edge-on view cannot flip-flop it.
    if (observation.has_fresh_data)
    {
        const ChairState s0 = inst.model.state();
        const float z_floor = s0.cz + s0.seat_h + 0.05f;     // just above the seat top → backrest band
        // Use the FULL support set (candidate AND residual): when the model is mis-oriented the true
        // backrest points fall far from the misplaced model backrest and land in residual, so a
        // candidate-only set would miss the very signal that reveals the error.
        std::vector<Eigen::Vector3f> hi;
        hi.reserve(observation.candidate_pts.size() + observation.residual_pts.size());
        for (const auto& p : observation.candidate_pts) if (p.z() > z_floor) hi.push_back(p);
        for (const auto& p : observation.residual_pts)  if (p.z() > z_floor) hi.push_back(p);

        if (hi.size() >= 16)
        {
            const auto mean_abs_sdf = [&](const ChairState& s) -> float
            {
                inst.model.set_state(s);
                const auto d = inst.model.compute_sdf(hi);
                float acc = 0.f;
                for (const float v : d) acc += std::abs(v);
                return acc / static_cast<float>(d.size());
            };
            std::array<float, 4> score{};
            int best_k = 0;
            for (int k = 0; k < 4; ++k)
            {
                ChairState t = s0;
                t.yaw = s0.yaw + static_cast<float>(k) * (static_cast<float>(M_PI) * 0.5f);
                score[static_cast<std::size_t>(k)] = mean_abs_sdf(t);
                if (score[static_cast<std::size_t>(k)] < score[static_cast<std::size_t>(best_k)]) best_k = k;
            }
            // DIAGNOSTIC: append the four backrest-band scores to a CSV every fresh frame for offline
            // analysis. score[0] = current orientation; the lowest score = best-explained = true
            // heading. File truncated once per process launch. t_ms is a steady-clock stamp.
            {
                static std::ofstream csv = []
                {
                    std::ofstream f("etc/chair_yaw_probe.csv", std::ios::trunc);
                    f << "t_ms,node,yaw,hi,sdf0,sdf90,sdf180,sdf270,best_deg\n";
                    return f;
                }();
                if (csv)
                {
                    const long t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::steady_clock::now().time_since_epoch()).count();
                    csv << t_ms << ',' << inst.node_name << ',' << s0.yaw << ',' << hi.size() << ','
                        << score[0] << ',' << score[1] << ',' << score[2] << ',' << score[3] << ','
                        << best_k * 90 << '\n';
                    csv.flush();
                }
            }
            ChairState s = s0;
            if (best_k != 0 and score[static_cast<std::size_t>(best_k)] < score[0] * 0.85f)
                s.yaw = s0.yaw + static_cast<float>(best_k) * (static_cast<float>(M_PI) * 0.5f);
            inst.model.set_state(s);   // restore s0 (no flip) or commit the flip
        }
    }

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
        if (cfg_.fisher_filter_enabled)
        {
            // Fisher information filter: fold this viewpoint's (confidence-weighted) per-DOF observation
            // info into the accumulators (predict Q-bleed + update + normalised "equivalent views"), via
            // the shared stabiliser. The weighting + Kalman/CUSUM already ran in accept_chair_belief.
            stabilizer_.accumulate(inst.stab);
            inst.info_w = inst.stab.fisher_info[4];   // DOF index 4 = seat_w
            inst.info_h = inst.stab.fisher_info[5];   // DOF index 5 = seat_d
        }
        else
        {
            // Legacy face-coverage PROXY (kept for A/B): info_w ← x-faces, info_h ← y-faces.
            const auto cov = inst.queue.face_coverage(inst.model);
            const float dm = std::max(1.0f, cfg_.delta_min);
            inst.info_w += ChairBeliefPolicy::clamp01(std::max(cov[0], cov[1]) / dm);
            inst.info_h += ChairBeliefPolicy::clamp01(std::max(cov[2], cov[3]) / dm);
        }
        inst.last_residual_pts = observation.residual_pts;   // hold for the voxelizer residual layer
    }

    const auto& s = inst.model.state();
    // Data-only posterior std (mm) from the accumulated Fisher precision — shrinks as evidence
    // gathers; ∞ (shown as -1) until a DOF is first observed. Diagnostic for the EFE epistemic value.
    const auto post_std_mm = [&](int j) -> float {
        return BeliefStabilizer<8>::posterior_std_milli(inst.stab, j);
    };
    if (should_log(inst))
        std::print("[{}] FE={:.4f}  cx={:.3f} cy={:.3f} cz={:.3f}  sw={:.3f} sd={:.3f} sh={:.3f} bh={:.3f} ψ={:.3f}  pts={} sil={} silres={:.4f} info=({:.1f},{:.1f}) σ(sw,sd)mm=({:.1f},{:.1f})\n",
                   inst.node_name, free_energy,
                   s.cx, s.cy, s.cz, s.seat_w, s.seat_d, s.seat_h, s.back_h, s.yaw,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()),
                   inst.model.silhouette_ray_count(), inst.model.silhouette_residual(),
                   inst.info_w, inst.info_h, post_std_mm(4), post_std_mm(5));
    if (should_log(inst))
        std::print("[{}] roi: valid={} offset=({:.2f},{:.2f}) fill={:.2f} | detect alive={} conf={:.2f} since={}\n",
                   inst.node_name, inst.roi_valid, inst.roi_offset_x, inst.roi_offset_y, inst.roi_fill,
                   inst.detection_alive, inst.last_mask_confidence, inst.frames_since_detection);

    log_fisher_csv(inst, observation.has_fresh_data, free_energy,
                   inst.queue.size() + static_cast<int>(observation.residual_pts.size()),
                   inst.model.silhouette_residual());

    return free_energy;
}

void ChairFitter::log_fisher_csv(const ChairInstance& inst, bool fresh, float free_energy,
                                 int point_count, float silres)
{
    if (cfg_.fisher_csv_path.empty())
        return;

    static const std::array<const char*, 8> kDof =
        {"cx", "cy", "cz", "yaw", "seat_w", "seat_d", "seat_h", "back_h"};

    if (not fisher_csv_.is_open())
    {
        fisher_csv_.open(cfg_.fisher_csv_path, std::ios::out | std::ios::trunc);
        if (not fisher_csv_.is_open())
        {
            std::print("chair_concept: [fisher] cannot open CSV '{}'\n", cfg_.fisher_csv_path);
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
    fisher_csv_ << '\n';
    fisher_csv_.flush();   // flush each row so a plot can tail the file during a live run
}

void ChairFitter::step_queue_update(ChairInstance& inst,
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
    if (should_log(inst))
        std::print("[{}] queue: admitted={} size={} obs_precision={:.2f}\n",
                   inst.node_name, admitted, inst.queue.size(), observation_precision);
}

float ChairFitter::step_model_update(ChairInstance& inst,
                                     const std::vector<Eigen::Vector3f>& residual_pts,
                                     const std::vector<Eigen::Vector3f>& current_pts,
                                     float residual_precision,
                                     bool fresh_observation)
{
    const ChairState previous_state = inst.model.state();

    const auto evidence = compose_belief_evidence(inst, residual_pts, current_pts, residual_precision);
    if (not evidence.has_evaluation())
    {
        refresh_chair_memory(inst);
        inst.last_queue_metrics = inst.queue.metrics();
        return inst.model.compute_free_energy({}, {});
    }

    float free_energy;
    if (cfg_.freeze_belief_on_stale and not fresh_observation)
    {
        // (A) No new measurement → don't update the belief. Re-optimising the same frozen anchor set
        // every cycle and re-accepting through a noisy per-frame Kalman gain is exactly what injects
        // jitter into the accepted state and FE. The information-filter axiom: no observation, no
        // update. Recompute FE at the unchanged state for reporting only (queue weights still age, so
        // this drifts only as slowly as the anchor mass decays — not the per-cycle re-fit noise).
        inst.last_fe_terms = inst.model.compute_free_energy_decomposition(
            evidence.eval_pts, evidence.eval_weights,
            static_cast<std::size_t>(evidence.historical_anchor_count));
        free_energy = inst.last_fe_terms.total_fe;
    }
    else
    {
        evolve_chair_belief(inst, evidence);
        free_energy = accept_chair_belief(inst, previous_state, evidence);
    }
    refresh_chair_memory(inst);
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

std::vector<Eigen::Vector3f> ChairFitter::gate_to_top_band(
    const std::vector<Eigen::Vector3f>& pts, const ChairModel& model) const
{
    if (!cfg_.top_band_gate_enabled)
        return pts;
    const float H    = model.state().seat_h;
    const float band = std::max(0.0f, cfg_.top_band_m);
    std::vector<Eigen::Vector3f> out;
    out.reserve(pts.size());
    for (const auto& p : pts)
        if (std::abs(p.z() - H) <= band)
            out.push_back(p);
    return out;
}

ChairFitter::ChairBeliefEvidence ChairFitter::compose_belief_evidence(
    const ChairInstance& inst,
    const std::vector<Eigen::Vector3f>& residual_pts_in,
    const std::vector<Eigen::Vector3f>& current_pts_in,
    float residual_precision) const
{
    // Gate the live observation streams to the chair-top band before they enter the fit/extent so
    // off-surface points (floor/under-chair/clutter) can't inflate w/h or corrupt the leg/height fit.
    // The queue anchors are already SDF-filtered near the surface, so they're left untouched.
    const std::vector<Eigen::Vector3f> residual_pts = gate_to_top_band(residual_pts_in, inst.model);
    const std::vector<Eigen::Vector3f> current_pts  = gate_to_top_band(current_pts_in, inst.model);

    ChairBeliefEvidence evidence;
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

void ChairFitter::evolve_chair_belief(ChairInstance& inst, const ChairBeliefEvidence& evidence)
{
    // Freeze gradient descent once converged to prevent oscillation; step_queue_update unlocks it
    // automatically when new points arrive.
    if (inst.frames_converged < cfg_.K_stable && evidence.can_optimize())
    {
        auto observer = [&](int /*iter*/, const ChairState& state, const FreeEnergyDecomposition& /*terms*/)
        {
            ChairModel shadow = inst.model;
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
        // [fit-probe] wall-clock cost of the torch autograd gradient_step. Aggregated avg/max + the
        // per-call point count, printed every ~2 s, so we can see what fraction of the 100 ms compute
        // budget the fit eats (→ decide if an analytic-Jacobian / Gauss-Newton rewrite is worth it).
        const auto fit_t0 = std::chrono::steady_clock::now();
        inst.model.gradient_step(evidence.fit_pts, evidence.fit_weights,
                                 static_cast<std::size_t>(evidence.historical_anchor_count), observer, gnc_start);
        {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - fit_t0).count();
            static double acc = 0.0, mx = 0.0;
            static long   n = 0, pts_sum = 0;
            static auto   last = std::chrono::steady_clock::now();
            acc += ms; mx = std::max(mx, ms); ++n; pts_sum += static_cast<long>(evidence.fit_pts.size());
            if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(2))
            {
                std::print("[fit-probe] gradient_step avg={:.2f}ms max={:.2f}ms calls={} avg_pts={} iters={}\n",
                           acc / static_cast<double>(n), mx, n,
                           pts_sum / std::max(1L, n), inst.model.params().optimization_iters);
                acc = 0.0; mx = 0.0; n = 0; pts_sum = 0; last = now;
            }
        }
    }
    else
    {
        inst.model.compute_free_energy(evidence.fit_pts, evidence.fit_weights);
        inst.queue.refresh_scores(inst.model);
    }
}

void ChairFitter::refresh_stabilizer_params()
{
    // Chair DOF order = [cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h]: yaw periodic; cx,cy position.
    StabilizerLayout<8> layout;
    layout.yaw_index = 3;
    layout.is_position[0] = true;   // cx
    layout.is_position[1] = true;   // cy
    stabilizer_.set_layout(layout);

    StabilizerParams p;
    p.fisher_filter_enabled = cfg_.fisher_filter_enabled;
    p.kalman_stiffness      = cfg_.fisher_kalman_stiffness;
    p.maturity_stiffness    = cfg_.fisher_maturity_stiffness;
    p.info_decay            = cfg_.fisher_info_decay;
    p.info_quality_rescale  = cfg_.fisher_quality_rescale;
    p.info_peak_decay        = cfg_.fisher_peak_decay;
    p.info_peak_ratchet      = cfg_.fisher_peak_ratchet;
    p.info_peak_ema          = cfg_.fisher_peak_ema;
    p.views_half            = cfg_.fisher_views_half;
    p.process_std_len       = cfg_.fisher_process_std_m;
    p.process_std_ang       = cfg_.fisher_process_std_yaw;
    p.mask_conf_weight      = cfg_.mask_conf_weight;
    p.mask_conf_floor       = cfg_.mask_conf_floor;
    p.mask_conf_ref         = cfg_.mask_conf_ref;
    p.mask_conf_power       = cfg_.mask_conf_power;
    p.ce_gate               = cfg_.counter_evidence_gate;
    p.ce_band_len           = cfg_.counter_evidence_band_m;
    p.ce_band_ang           = cfg_.counter_evidence_band_rad;
    p.ce_base_pos           = cfg_.counter_evidence_base_pos;
    p.ce_lambda_pos         = cfg_.counter_evidence_lambda_pos;
    p.ce_base_size          = cfg_.counter_evidence_base_size;
    p.ce_lambda_size        = cfg_.counter_evidence_lambda_size;
    p.ce_decay              = cfg_.counter_evidence_decay;
    p.ce_step_cap           = cfg_.counter_evidence_step_cap;
    p.ce_unlock_deflate     = cfg_.counter_evidence_unlock_deflate;
    stabilizer_.set_params(p);
}

float ChairFitter::accept_chair_belief(ChairInstance& inst,
                                       const ChairState& previous_state,
                                       const ChairBeliefEvidence& evidence)
{
    const auto finite_state = [](const ChairState& state)
    {
        return std::isfinite(state.cx) && std::isfinite(state.cy) &&
               std::isfinite(state.cz) && std::isfinite(state.yaw) &&
               std::isfinite(state.seat_w) && std::isfinite(state.seat_d) &&
               std::isfinite(state.seat_h) && std::isfinite(state.back_h);
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

    const ChairState raw_state = inst.model.state();
    if (!finite_state(raw_state))
        return revert();

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
    const auto obs_info = inst.model.observation_information(evidence.eval_pts, evidence.eval_weights);
    stabilizer_.weight_observation(inst.stab, obs_info, inst.last_mask_confidence);
    stabilizer_.compute_acceptance(inst.stab, raw_state.to_array(), previous_state.to_array());

    const auto coverage = inst.queue.face_coverage(inst.model);

    inst.warm_confidence = ChairBeliefPolicy::update_warm_confidence(
        inst.warm_confidence, cfg_, coverage,
        evidence.trusted_point_count, evidence.residual_count, evidence.residual_precision);

    // Acceptance-gain damping driven by settle_maturity: a counter that climbs each cycle and is
    // reset ONLY by a genuine new-evidence burst (net new queue anchors ⇒ a fresh viewpoint), in
    // step_queue_update. High gain while immature (startup or just after the controller moves the
    // robot) → adapts fast; decays to a low floor once mature → smooth lock. It deliberately does
    // NOT react to frames_converged / per-frame state jitter, which would spike the gain on noise
    // and make the fit jerky.
    const float maturity = ChairBeliefPolicy::clamp01(static_cast<float>(inst.settle_maturity) /
                                                      static_cast<float>(std::max(1, cfg_.warm_settle_cycles)));
    const float settle_gain = ChairBeliefPolicy::lerp(1.0f, cfg_.warm_settle_floor, maturity);

    const ChairState accepted_state = ChairBeliefPolicy::apply_observability_warm_start(
        previous_state, raw_state, inst.model.params(), cfg_, inst.warm_confidence,
        coverage, evidence.trusted_point_count, settle_gain, inst.info_w, inst.info_h,
        stabilizer_.kalman_active() ? &inst.stab.last_kalman_gain : nullptr);
    if (!finite_state(accepted_state))
        return revert();

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

    if (should_log(inst))
        std::print("[{}] warm-start: conf={:.2f} mask_conf={:.2f} w={:.2f} rho_x={:.2f} rho_y={:.2f} residual={} trusted_pts={} residual_precision={:.2f} raw(w={:.3f},h={:.3f},psi={:.3f}) accepted(w={:.3f},h={:.3f},psi={:.3f})\n",
                   inst.node_name, inst.warm_confidence, inst.last_mask_confidence, inst.stab.last_mask_conf_weight,
                   std::min(coverage[0], coverage[1]) / (std::max(coverage[0], coverage[1]) + 1e-3f),
                   std::min(coverage[2], coverage[3]) / (std::max(coverage[2], coverage[3]) + 1e-3f),
                   evidence.residual_count, evidence.trusted_point_count, evidence.residual_precision,
                   raw_state.seat_w, raw_state.seat_d, raw_state.yaw,
                   accepted_state.seat_w, accepted_state.seat_d, accepted_state.yaw);

    return free_energy;
}

void ChairFitter::refresh_chair_memory(ChairInstance& inst)
{
    const Eigen::Matrix2f robot_cov = scene_graph_->read_robot_covariance(room_node_id_);
    inst.queue.update_rfe(inst.model, robot_cov);
}

// ─── RGB-mask silhouette ──────────────────────────────────────────────────────

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

void ChairFitter::feed_silhouette(ChairInstance& inst)
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
    const auto slice = mask_ingestor_->select_nearest(Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().seat_h), "chair");
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
        if (rz < -1e-2)   // ray must descend to intersect the chair-top plane below the camera
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

SampleQueueParams ChairFitter::make_queue_params() const
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
    p.diversity_admission          = true;   // chair: spatial-diversity admission (preserve pre-template behaviour)
    return p;
}

}  // namespace rc
