/*
 * bottle_fitter.cpp — the active-inference fit core for bottle_concept.
 */

#include "bottle_fitter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

BottleFitter::BottleFitter(std::shared_ptr<DSR::DSRGraph> graph,
                           DSR::InnerEigenAPI* inner_eigen,
                           BottleConfig& cfg,
                           const std::vector<BottlePrior>& priors,
                           MaskIngestor* perception,
                           BottleSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg), priors_(priors),
      mask_ingestor_(perception), scene_graph_(scene_graph)
{}

void BottleFitter::update_support_surface(BottleInstance& inst)
{
    // Decide the surface the bottle rests on (room vs a table) BEFORE ingest/fit, so the surface
    // filter (is_voxel_owned_by_bottle) and the post-fit cz anchor both have it. MAP over {room,
    // every table_N} using the OBSERVED base (low-percentile z of the owned voxel bank — NOT the
    // anchored cz, which would be circular) + an oriented footprint + vertical-support test. A
    // re-parent is committed only after support_commit_cycles of agreement (hysteresis).
    const auto& s0 = inst.model.state();
    float base_z;
    if (not inst.voxel_bank_pts.empty())
    {
        std::vector<float> zs;
        zs.reserve(inst.voxel_bank_pts.size());
        for (const auto& p : inst.voxel_bank_pts) zs.push_back(p.z());
        const std::size_t k = zs.size() / 10;          // 10th percentile → robust base, rejects outliers
        std::nth_element(zs.begin(), zs.begin() + k, zs.end());
        base_z = zs[k];
    }
    else
        base_z = s0.cz - 0.5f * s0.height;             // pre-evidence fallback: model base

    const auto dec = scene_graph_->decide_support_surface(s0.cx, s0.cy, base_z, room_node_id_);
    if (inst.parent_id == 0)                            // uninitialised → adopt immediately
    {
        inst.parent_id = dec.parent_id;
        inst.parent_name = dec.parent_name;
    }
    if (dec.parent_id == inst.parent_id)
    {
        inst.support_challenger_id = 0;
        inst.support_challenger_count = 0;
    }
    else                                                // a different surface is winning → debounce
    {
        if (dec.parent_id == inst.support_challenger_id)
            ++inst.support_challenger_count;
        else
        {
            inst.support_challenger_id = dec.parent_id;
            inst.support_challenger_count = 1;
        }
        if (inst.support_challenger_count >= cfg_.support_commit_cycles)
        {
            std::print("[{}] re-parent {} → {} (support won {} cycles)\n",
                       inst.node_name, inst.parent_name, dec.parent_name, inst.support_challenger_count);
            inst.parent_id = dec.parent_id;
            inst.parent_name = dec.parent_name;
            inst.support_reparent_pending = true;
            inst.support_challenger_id = 0;
            inst.support_challenger_count = 0;
        }
    }
    // Anchor reflects the COMMITTED parent: table_top_of returns NaN for the room (no z anchor).
    inst.table_top_z = scene_graph_->table_top_of(inst.parent_id);
}

BottleFitter::BottleObservation BottleFitter::observe(BottleInstance& inst,
                                                      const DSR::Node& node)
{
    BottleObservation observation;

    // Detection aliveness ticks every cycle; a fresh selected bottle mask below resets it to 0.
    ++inst.frames_since_detection;

    // Primary path: YOLO masks (room frame). Classify-don't-destroy SDF split —
    // inliers become queue anchors, the rest drive model expansion.
    if (mask_ingestor_->packet().valid and mask_ingestor_->packet().frame_id > inst.last_masks_frame_seen)
    {
        // Mask for this instance: the tracker's gated assignment when present (multi-instance safe),
        // else the legacy greedy nearest-to-our-centre.
        std::optional<MaskIngestor::MaskSlice> selected_mask;
        if (const auto& sl = mask_ingestor_->packet().slices;
            inst.assigned_mask_idx >= 0 and inst.assigned_mask_idx < static_cast<int>(sl.size()))
            selected_mask = sl[inst.assigned_mask_idx];
        else
            selected_mask = mask_ingestor_->select_nearest(
                Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().cz), "bottle");
        if (selected_mask.has_value())
        {
            const auto& slice = selected_mask.value();
            // YOLO produced a bottle mask this frame → detection is alive; record its confidence.
            inst.frames_since_detection = 0;
            inst.last_mask_confidence   = slice.confidence;
            inst.last_mask_timestamp_ms = mask_ingestor_->packet().timestamp_ms;   // chain-cov pinning (Part B)
            const std::size_t begin = std::min(slice.support_begin, mask_ingestor_->packet().support_points.size());
            const std::size_t end   = std::min(slice.support_end,   mask_ingestor_->packet().support_points.size());

            std::vector<Eigen::Vector3f> candidate_pts;
            std::vector<Eigen::Vector3f> residual_pts;
            candidate_pts.reserve(end > begin ? end - begin : 0);
            residual_pts.reserve(end > begin ? end - begin : 0);

            for (std::size_t i = begin; i < end; ++i)
            {
                const auto& p = mask_ingestor_->packet().support_points[i];
                const float sdf = inst.model.sdf_point(p);
                if (std::abs(sdf) < cfg_.sdf_threshold_for_storage)
                    candidate_pts.push_back(p);
                else
                    residual_pts.push_back(p);
            }

            // Fresh-frame observation centroid (room frame) — the CV motion filter's position
            // measurement. It FOLLOWS the bottle (it's this frame's points), unlike the raw SDF fit
            // which the historical queue drags toward the old location. De-projection applied at use.
            if (not candidate_pts.empty() or not residual_pts.empty())
            {
                Eigen::Vector3f sum = Eigen::Vector3f::Zero();
                for (const auto& p : candidate_pts) sum += p;
                for (const auto& p : residual_pts)  sum += p;
                inst.last_obs_centroid = sum / static_cast<float>(candidate_pts.size() + residual_pts.size());
            }

            observation.has_fresh_data = true;
            observation.candidate_pts = std::move(candidate_pts);
            observation.residual_pts  = std::move(residual_pts);

            if (not observation.candidate_pts.empty() or not observation.residual_pts.empty())
            {
                const float total = static_cast<float>(observation.candidate_pts.size() + observation.residual_pts.size());
                observation.explanation_ratio = total > 0.0f
                    ? static_cast<float>(observation.candidate_pts.size()) / total
                    : 0.0f;

                inst.last_masks_frame_seen = mask_ingestor_->packet().frame_id;

                if (should_log(inst))
                    std::print("[{}] masks={} label='{}' conf={:.2f} support={} cand={} resid={} centroid=({:.2f},{:.2f},{:.2f})\n",
                               inst.node_name, mask_ingestor_->packet().frame_id, slice.label, slice.confidence,
                               end - begin, observation.candidate_pts.size(), observation.residual_pts.size(),
                               slice.centroid.x(), slice.centroid.y(), slice.centroid.z());
                return observation;
            }
        }
    }

    // Fallback: candidate/residual point attributes written directly on the node.
    observation.candidate_pts = mask_ingestor_->read_pts_attrib(node, "candidate_pts_att");
    observation.residual_pts  = mask_ingestor_->read_pts_attrib(node, "residual_pts_att");
    observation.has_fresh_data = not observation.candidate_pts.empty() or not observation.residual_pts.empty();
    return observation;
}

float BottleFitter::run_inference(BottleInstance& inst, const BottleObservation& observation)
{
    update_support_surface(inst);   // resting-surface decision + table-top z anchor (before ingest/fit)

    inst.queue.begin_cycle();

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);

    const bool have_obs = not observation.candidate_pts.empty() or not observation.residual_pts.empty();
    if (observation.has_fresh_data and (not inst.voxel_bank_pts.empty() or have_obs))
    {
        // Cold-start: snap model & prior to the first observation centroid so
        // gradient descent begins at the right place rather than the prior.
        if (inst.matched_frames == 0)
        {
            // Seed from the freshest mask points: candidate first, else RESIDUAL, else the bank. The
            // candidate/residual split is gated by the CURRENT (possibly stale) model — when the model is
            // far (a fresh start or a move-experiment teleport) ALL mask points land in residual, so a
            // candidate-only seed would deadlock (it never moves to where the bottle actually is). The
            // residual points ARE the current mask, so their centroid is the true bottle location.
            const auto& src = not observation.candidate_pts.empty() ? observation.candidate_pts
                            : not observation.residual_pts.empty()  ? observation.residual_pts
                                                                     : inst.voxel_bank_pts;
            if (src.empty())
                return inst.prev_free_energy;   // genuinely nothing to seed from yet — retry next frame
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            for (const auto& p : src) sum += p;
            const Eigen::Vector3f cen = sum / static_cast<float>(src.size());
            auto s = inst.model.state();
            s.cx = cen.x(); s.cy = cen.y(); s.cz = cen.z();
            // De-project the visible-arc centroid onto the cylinder AXIS. The camera sees only the
            // near arc, so the centroid lies ~one radius camera-ward of the axis; snapping the seed
            // there biases the whole fit forward (the symmetric SDF is depth-degenerate for a one-
            // sided cloud, so this seed/prior decides depth). Push the seed away from the camera by
            // frac·radius in the horizontal plane (axis is vertical → bias is purely radial in XY).
            float deproj = 0.0f;
            if (const auto Mopt = room_T_zed_matrix();
                Mopt.has_value() and cfg_.seed_deproject_frac > 0.0f and s.radius > 0.0f)
            {
                const Eigen::Vector2f cam_xy(static_cast<float>((*Mopt)(0, 3)),
                                             static_cast<float>((*Mopt)(1, 3)));
                const Eigen::Vector2f ray = Eigen::Vector2f(s.cx, s.cy) - cam_xy;   // camera → centroid (horizontal)
                if (ray.norm() > 1e-4f)
                {
                    deproj = cfg_.seed_deproject_frac * s.radius;
                    const Eigen::Vector2f back = deproj * ray.normalized();
                    s.cx += back.x(); s.cy += back.y();   // axis pushed AWAY from the camera by ~radius
                }
            }
            inst.model.set_state(s);
            inst.model.set_prior(s);   // zero KL so the data term dominates from the start
            // Move-experiment re-seed: the model has just snapped to the NEW pose's mask centroid, so the
            // ownership gate is now correct — purge the stale bank (it holds the PREVIOUS pose's voxels;
            // the bank is append-only and would otherwise drag the fit back). Refills from the new pose
            // over the next cycles. Without this the fit is stranded near the prior pose (gate never
            // admits the moved-away voxels) — the sweep "pins" the fit and inflates the error vs GT.
            if (inst.reseed_requested)
            {
                inst.voxel_bank_pts.clear();
                inst.voxel_bank_keys.clear();
                inst.reseed_requested = false;
            }
            std::print("[{}] cold-start snap → ({:.2f},{:.2f},{:.2f}) ({} pts, de-proj {:+.3f} m away from cam)\n",
                       inst.node_name, s.cx, s.cy, s.cz, src.size(), deproj);
            inst.matched_frames = cfg_.min_frames_before_historical + cfg_.historical_warmup_frames + 1;
        }
        else
            ++inst.matched_frames;

        // Combined per-observation precision = geometric self-consistency (explanation_ratio) ×
        // detection confidence (YOLO score). Both must be high for the frame to count fully; a
        // low-score mask widens the resulting Laplace covariance instead of hardening the belief.
        step_queue_update(inst, inst.voxel_bank_pts,
                          std::clamp(observation.explanation_ratio, 0.0f, 1.0f)
                              * mask_confidence_weight(inst.last_mask_confidence));
    }

    const float explanation_confidence = std::clamp(observation.explanation_ratio, 0.0f, 1.0f);
    const float conf_w = mask_confidence_weight(inst.last_mask_confidence);
    float residual_precision = 0.0f;
    if (observation.has_fresh_data and observation.candidate_pts.empty() and not observation.residual_pts.empty())
        residual_precision = 0.10f * conf_w;   // only residuals: keep a floor so the model can expand
    else
        residual_precision = std::clamp(1.0f - explanation_confidence, 0.0f, 1.0f) * conf_w;

    feed_silhouette(inst);   // attach the RGB-mask edge rays before the model + covariance update
    const float free_energy = step_model_update(inst, observation.residual_pts, residual_precision);

    // Fold this viewpoint's Fisher information into the cross-frame filter — fresh masks only, so the
    // belief hardens by VIEWS, not wall-clock (a frozen scene must not keep accruing precision).
    if (cfg_.fisher_filter_enabled and observation.has_fresh_data)
        update_fisher_filter(inst);

    const auto& s = inst.model.state();
    const int point_count = inst.queue.size() + static_cast<int>(observation.residual_pts.size());
    // Data-only posterior std (mm) from the accumulated Fisher precision — shrinks as evidence
    // gathers; -1 until a DOF is first observed. Diagnostic for the EFE epistemic value / P_bottle.
    const auto post_std_mm = [&](int j) -> float {
        return BeliefStabilizer<5>::posterior_std_milli(inst.stab, j);
    };
    if (should_log(inst))
        std::print("[{}] FE={:.4f}  c=({:.3f},{:.3f},{:.3f}) r={:.3f} h={:.3f}  pts={} σ(r,h)mm=({:.1f},{:.1f})\n",
                   inst.node_name, free_energy, s.cx, s.cy, s.cz, s.radius, s.height,
                   point_count, post_std_mm(3), post_std_mm(4));

    log_fisher_csv(inst, observation.has_fresh_data, free_energy, point_count);

    // Hang the bottle from the table: standing on the surface fixes cz = table_top + height/2, so z is
    // determined (not fitted) — removes the z DOF and the systematic z error the free fit carried. The
    // returned FE is the pre-anchor value (as before); the anchored state is what the worker writes back.
    if (std::isfinite(inst.table_top_z))
    {
        auto s_anchor = inst.model.state();
        s_anchor.cz = inst.table_top_z + 0.5f * s_anchor.height;
        inst.model.set_state(s_anchor);
    }

    return free_energy;
}

std::optional<Eigen::Matrix4d> BottleFitter::room_T_zed_matrix() const
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
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) m(i, j) = s(i, j);   // element-wise: no aligned load
        return m;
    };
    return to_mat4(rtb.value()) * to_mat4(btz.value());
}

void BottleFitter::feed_silhouette(BottleInstance& inst)
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

    const auto slice = mask_ingestor_->select_nearest(Eigen::Vector3f(inst.model.state().cx, inst.model.state().cy, inst.model.state().cz), "bottle");
    if (not slice.has_value() or slice->pixel_end <= slice->pixel_begin
        or slice->pixel_end > mask_ingestor_->packet().mask_pixels.size())
        return;

    const auto Mopt = room_T_zed_matrix();   // room_T_zed (camera→room), plain 4×4
    if (not Mopt.has_value())
        return;
    const Eigen::Matrix4d& M = Mopt.value();
    const double Cx = M(0, 3), Cy = M(1, 3);   // camera centre (translation) in room

    const float fx = camera_api_->get_focal_x();
    const float fy = camera_api_->get_focal_y();
    const float cx_px = static_cast<float>(camera_api_->get_width())  * 0.5f;
    const float cy_px = static_cast<float>(camera_api_->get_height()) * 0.5f;

    // Per image row, the min & max column = the mask's left/right occluding-contour edges.
    std::unordered_map<int, std::pair<float, float>> row_minmax;
    for (std::size_t i = slice->pixel_begin; i < slice->pixel_end; ++i)
    {
        const float col = mask_ingestor_->packet().mask_pixels[i].x();
        const int   row = static_cast<int>(mask_ingestor_->packet().mask_pixels[i].y());
        auto it = row_minmax.find(row);
        if (it == row_minmax.end()) row_minmax.emplace(row, std::pair{col, col});
        else { it->second.first = std::min(it->second.first, col); it->second.second = std::max(it->second.second, col); }
    }

    std::vector<Eigen::Vector2f> dirs;
    dirs.reserve(row_minmax.size() * 2);
    const auto add_edge = [&](float col, float row)
    {
        // d_cam = ((col-cx)/fx, 1, (cy-row)/fy); rotate to room (rotation only, no translation).
        const double dx = (col - cx_px) / fx, dz = (cy_px - row) / fy;
        const double rx = M(0, 0) * dx + M(0, 1) + M(0, 2) * dz;
        const double ry = M(1, 0) * dx + M(1, 1) + M(1, 2) * dz;
        dirs.emplace_back(static_cast<float>(rx), static_cast<float>(ry));
    };
    for (const auto& [row, mm] : row_minmax)
    {
        add_edge(mm.first,  static_cast<float>(row));
        add_edge(mm.second, static_cast<float>(row));
    }
    if (not dirs.empty())
        inst.model.set_silhouette(Eigen::Vector2f(static_cast<float>(Cx), static_cast<float>(Cy)),
                                  std::move(dirs), slice->confidence);
}

float BottleFitter::mask_confidence_weight(float confidence) const
{
    if (not cfg_.mask_conf_weight)
        return 1.0f;
    // w = clamp01((conf−floor)/(ref−floor))^power — identical to BeliefStabilizer::weight_observation
    // so the Laplace fit covariance and the Fisher stabiliser down-weight a low-score mask the same way.
    const float floor = std::clamp(cfg_.mask_conf_floor, 0.0f, 0.99f);
    const float ref   = std::max(floor + 1e-3f, cfg_.mask_conf_ref);
    return std::pow(std::clamp((confidence - floor) / (ref - floor), 0.0f, 1.0f),
                    std::max(0.1f, cfg_.mask_conf_power));
}

void BottleFitter::step_queue_update(BottleInstance& inst,
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
    // New points from a fresh view → unlock the optimizer so it can re-converge.
    if (admitted > 0 and inst.frames_converged >= cfg_.K_stable)
        inst.frames_converged = cfg_.K_stable / 2;
}

float BottleFitter::step_model_update(BottleInstance& inst,
                                      const std::vector<Eigen::Vector3f>& residual_pts,
                                      float residual_precision)
{
    const BottleState previous_state = inst.model.state();

    // Compose evidence: historical queue anchors + this frame's residuals.
    std::vector<Eigen::Vector3f> fit_pts = inst.queue.points();
    std::vector<float>           fit_weights = inst.queue.weights();
    const std::size_t historical_anchor_count = fit_pts.size();

    for (const auto& r : residual_pts)
        if (residual_precision > 1e-3f)
        {
            fit_pts.push_back(r);
            fit_weights.push_back(residual_precision);
        }

    if (fit_pts.empty())
    {
        inst.last_queue_metrics = inst.queue.metrics();
        return inst.model.compute_free_energy({}, {});
    }

    float free_energy = previous_state.cx;  // placeholder, overwritten below
    if (inst.frames_converged < cfg_.K_stable)
    {
        // Refresh queue scores against the moving model during descent.
        auto observer = [&](int, const BottleState& state, const FreeEnergyDecomposition&)
        {
            BottleModel shadow = inst.model;
            shadow.set_state(state);
            inst.queue.refresh_scores(shadow);
        };
        free_energy = inst.model.gradient_step(fit_pts, fit_weights, historical_anchor_count, observer);
    }
    else
    {
        free_energy = inst.model.compute_free_energy(fit_pts, fit_weights);
        inst.queue.refresh_scores(inst.model);
    }

    // Reject a non-finite optimum (NaN SDF gradient) — revert to the last good state.
    const auto& s = inst.model.state();
    const bool finite = std::isfinite(s.cx) and std::isfinite(s.cy) and std::isfinite(s.cz) and
                        std::isfinite(s.radius) and std::isfinite(s.height) and std::isfinite(free_energy);
    if (not finite)
    {
        inst.model.set_state(previous_state);
        inst.model.set_prior(previous_state);
        free_energy = inst.model.compute_free_energy(fit_pts, fit_weights);
    }

    inst.last_fe_terms = inst.model.compute_free_energy_decomposition(fit_pts, fit_weights, historical_anchor_count);
    inst.last_queue_metrics = inst.queue.metrics();

    // Measure this frame's per-DOF observation Fisher information at the fitted state on the same
    // evidence used for the fit — the calibrated, anisotropic "how well is each DOF seen". Stored
    // raw; run_inference folds it into the cross-frame filter on fresh-mask frames only.
    if (cfg_.fisher_filter_enabled)
        inst.stab.last_obs_info = inst.model.observation_information(fit_pts, fit_weights);

    // Apply the stabiliser to the ACCEPTED fit: the maturity-stiffened per-DOF Kalman gain + CUSUM gate
    // lerp the raw gradient fit back toward the committed belief, so a well-seen DOF (esp. the
    // depth-degenerate radius) barely moves per frame and the perceived pose stops jerking — while a
    // sustained coherent move still unlocks via the gate. Uses the prior accumulated precision (the
    // accumulate() that folds THIS frame runs afterwards, in update_fisher_filter). Off → raw fit.
    if (cfg_.fisher_filter_enabled and cfg_.stabilizer_acceptance and finite)
    {
        refresh_stabilizer_params();
        stabilizer_.weight_observation(inst.stab, inst.stab.last_obs_info, inst.last_mask_confidence);
        const auto raw  = inst.model.state().to_array();
        const auto prev = previous_state.to_array();
        stabilizer_.compute_acceptance(inst.stab, raw, prev);

        std::array<float, 5> accepted;
        for (int j = 0; j < 5; ++j)
            accepted[j] = prev[j] + inst.stab.last_kalman_gain[j] * (raw[j] - prev[j]);

        // Movable object: cx,cy TRACK via the CV filter using the raw per-frame fit as the measurement
        // (the maturity-locked gain above froze them, so a moved bottle spawned a new instance). cz
        // stays table-anchored downstream; radius/height keep the stabiliser's hardening. The CV
        // covariance inflates during motion → the InstanceTracker gate widens and still associates.
        if (cfg_.dynamics_model == "constant_velocity")
        {
            // Measurement = the fresh-frame observation centroid (FOLLOWS the bottle), de-projected onto
            // the cylinder axis (the visible front-arc centroid sits ~1 radius camera-ward — same
            // correction as the cold-start seed). Falls back to the raw fit before the first observation.
            Eigen::Vector2f zc(raw[0], raw[1]);
            if (inst.last_obs_centroid.squaredNorm() > 0.0f)
            {
                zc = {inst.last_obs_centroid.x(), inst.last_obs_centroid.y()};
                if (const auto Mopt = room_T_zed_matrix();
                    Mopt.has_value() and cfg_.seed_deproject_frac > 0.0f and inst.model.state().radius > 0.0f)
                {
                    const Eigen::Vector2f cam_xy(static_cast<float>((*Mopt)(0, 3)), static_cast<float>((*Mopt)(1, 3)));
                    const Eigen::Vector2f ray = zc - cam_xy;
                    if (ray.norm() > 1e-4f)
                        zc += (cfg_.seed_deproject_frac * inst.model.state().radius) * ray.normalized();
                }
            }
            const double mv = static_cast<double>(cfg_.cv_meas_std) * cfg_.cv_meas_std;
            const std::vector<double> z   = {zc.x(), zc.y()};
            const std::vector<double> mvv = {mv, mv};
            if (not inst.motion.initialized())
            {
                inst.motion.configure(cfg_.cv_accel_std, cfg_.cv_init_vel_std);
                inst.motion.init(z, mvv);
            }
            else if (inst.last_masks_frame_seen != inst.last_motion_frame)   // a NEW frame's measurement
            {
                double dt = (inst.last_mask_timestamp_ms > inst.last_motion_ts_ms and inst.last_motion_ts_ms > 0)
                            ? static_cast<double>(inst.last_mask_timestamp_ms - inst.last_motion_ts_ms) / 1000.0
                            : static_cast<double>(cfg_.cv_dt_default_s);
                dt = std::clamp(dt, 0.01, 0.5);
                inst.motion.predict(dt);
                inst.motion.correct(z, mvv);
            }
            inst.last_motion_ts_ms = inst.last_mask_timestamp_ms;
            inst.last_motion_frame = inst.last_masks_frame_seen;
            accepted[0] = static_cast<float>(inst.motion.position(0));
            accepted[1] = static_cast<float>(inst.motion.position(1));
        }

        inst.model.set_state(BottleState::from_array(accepted));

        free_energy = inst.model.compute_free_energy(fit_pts, fit_weights);
        inst.last_fe_terms = inst.model.compute_free_energy_decomposition(fit_pts, fit_weights, historical_anchor_count);
    }

    // Convergence bookkeeping — AFTER the stabiliser, on the COMMITTED free energy. prev_free_energy
    // holds last frame's post-stabiliser FE, so this compares committed-vs-committed. Doing it on the
    // pre-stabiliser raw gradient optimum (before this block) pinned frames_converged at 0 forever: the
    // stabiliser holds the committed state away from the per-frame raw optimum, so raw(t) and
    // committed(t−1) never agreed within fe_eps even when the committed belief was perfectly stable.
    if (std::abs(free_energy - inst.prev_free_energy) < cfg_.fe_eps)
        inst.frames_converged = std::min(inst.frames_converged + 1, cfg_.K_stable);
    else
        inst.frames_converged = 0;

    return free_energy;
}

void BottleFitter::refresh_stabilizer_params()
{
    StabilizerParams p;
    p.info_decay      = cfg_.fisher_info_decay;
    p.process_std_len = cfg_.fisher_process_std_m;
    p.mask_conf_weight = cfg_.mask_conf_weight;   // YOLO score → per-DOF obs-info weight (see BottleConfig)
    p.mask_conf_floor  = cfg_.mask_conf_floor;
    p.mask_conf_ref    = cfg_.mask_conf_ref;
    p.mask_conf_power  = cfg_.mask_conf_power;
    p.kalman_stiffness = cfg_.stabilizer_acceptance;   // drive the accepted fit from the Kalman gain
    // CUSUM/SPRT gate: bands sized to the bottle's cm scale. Applied to the fit when stabilizer_acceptance
    // is on; otherwise it runs diagnostically (dashboard counter-evidence panel stays live).
    p.ce_gate         = true;
    p.ce_band_len     = 0.005f;   // 5 mm: radius/height innovations within this are "coherent"
    StabilizerLayout<5> layout;   // yaw_index = -1 (no periodic DOF); cx,cy are position
    layout.is_position[0] = true;
    layout.is_position[1] = true;
    stabilizer_.set_layout(layout);
    stabilizer_.set_params(p);
}

void BottleFitter::update_fisher_filter(BottleInstance& inst)
{
    // Fold this frame's per-DOF Fisher diagonal (inst.stab.last_obs_info) into the cross-frame
    // accumulators via the shared stabiliser: normalised "equivalent views" + information-filter
    // predict (Q-bleed) + update. 5 DOF, all lengths (no yaw).
    refresh_stabilizer_params();

    // When the stabiliser is NOT applied to the fit, run the CUSUM diagnostically (innovation = this
    // fresh fit vs the previous one) so the dashboard's counter-evidence panel stays live. When it IS
    // applied, compute_acceptance already ran in step_model_update (innovation vs the committed belief).
    if (not cfg_.stabilizer_acceptance)
    {
        const auto cur = inst.model.state().to_array();
        if (inst.has_prev_diag)
            stabilizer_.compute_acceptance(inst.stab, cur, inst.prev_diag_state);
        inst.prev_diag_state = cur;
        inst.has_prev_diag   = true;
    }

    stabilizer_.accumulate(inst.stab);
}

void BottleFitter::log_fisher_csv(const BottleInstance& inst, bool fresh, float free_energy, int point_count)
{
    if (cfg_.fisher_csv_path.empty())
        return;

    static const std::array<const char*, 5> kDof = {"cx", "cy", "cz", "r", "h"};

    if (not fisher_csv_.is_open())
    {
        fisher_csv_.open(cfg_.fisher_csv_path, std::ios::out | std::ios::trunc);
        if (not fisher_csv_.is_open())
        {
            std::print("bottle_concept: [fisher] cannot open CSV '{}'\n", cfg_.fisher_csv_path);
            cfg_.fisher_csv_path.clear();   // disable further attempts
            return;
        }
        fisher_csv_ << "cycle,node,fresh,fe,frames_converged,pts,conf,conf_w";   // YOLO score + applied weight w
        for (const auto* d : kDof) fisher_csv_ << ",state_" << d;
        for (const auto* d : kDof) fisher_csv_ << ",obs_"   << d;   // this frame's raw Fisher diag
        for (const auto* d : kDof) fisher_csv_ << ",acc_"   << d;   // normalised accumulated info (stiffener)
        for (const auto* d : kDof) fisher_csv_ << ",raw_"   << d;   // un-normalised accumulated precision Σ⁻¹
        for (const auto* d : kDof) fisher_csv_ << ",std_"   << d;   // posterior std (mm); -1 = unobserved
        for (const auto* d : kDof) fisher_csv_ << ",gain_"  << d;   // per-DOF Kalman gain (diagnostic; not applied)
        for (const auto* d : kDof) fisher_csv_ << ",ce_"    << d;   // CUSUM counter-evidence Sⱼ (diagnostic)
        for (const auto* d : kDof) fisher_csv_ << ",ceg_"   << d;   // gate: -1 reject / 0 passthrough / +1 unlock
        fisher_csv_ << ",vx,vy,motion_var";   // CV motion filter: velocity (m/s) + cx position variance
        fisher_csv_ << ",chain_xx,chain_yy,rtcov_xx,rtcov_yy";   // Part B: localization/chain cov + total published RT cov (m²)
        fisher_csv_ << '\n';
    }

    const auto s = inst.model.state().to_array();
    fisher_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << (fresh ? 1 : 0) << ','
                << free_energy << ',' << inst.frames_converged << ',' << point_count << ','
                << inst.last_mask_confidence << ',' << inst.stab.last_mask_conf_weight;
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << s[j];
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.last_obs_info[j];
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.fisher_info[j];
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.fisher_info_raw[j];
    for (int j = 0; j < 5; ++j)
        fisher_csv_ << ',' << BeliefStabilizer<5>::posterior_std_milli(inst.stab, j);
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.last_kalman_gain[j];
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.counter_evidence[j];
    for (int j = 0; j < 5; ++j) fisher_csv_ << ',' << inst.stab.last_ce_gate[j];
    fisher_csv_ << ',' << (inst.motion.initialized() ? inst.motion.velocity(0) : 0.0)
                << ',' << (inst.motion.initialized() ? inst.motion.velocity(1) : 0.0)
                << ',' << (inst.motion.initialized() ? inst.motion.pos_var(0)  : 0.0);
    fisher_csv_ << ',' << inst.dbg_chain_cov_xx << ',' << inst.dbg_chain_cov_yy
                << ',' << inst.dbg_rtcov_xx     << ',' << inst.dbg_rtcov_yy;
    fisher_csv_ << '\n';
    fisher_csv_.flush();   // flush each row so a plot can tail the file during a live run
}

void BottleFitter::ingest_observation_voxels(BottleInstance& inst, const BottleObservation& observation)
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
            if (not is_voxel_owned_by_bottle(inst, p))
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

    if (inserted > 0 and should_log(inst))
        std::print("[{}] voxel-bank: +{} total={} (cap={}) reject_foreign={}\n",
                   inst.node_name, inserted, inst.voxel_bank_pts.size(), max_points, rejected_foreign);
}

bool BottleFitter::is_voxel_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const
{
    const auto& s = inst.model.state();

    // XY ownership gate: centred on the live pose, but sized by the FIXED prior radius — NOT the live
    // estimate. Gating on s.radius is a feedback loop: depth points just outside the cylinder get
    // admitted → support a larger radius → widen the gate next frame → "invent" radius from edge-pixel
    // depth noise (the pose-3/6 inflation). A fixed gate caps how far depth alone can grow the bottle;
    // the mask silhouette owns the actual radius.
    const float gate_radius = cfg_.prior_radius + cfg_.voxel_select_radius_margin_m;
    const float dx = point.x() - s.cx;
    const float dy = point.y() - s.cy;
    if (std::hypot(dx, dy) > gate_radius)
        return false;

    // Height gate around the cylinder span.
    float z_min = s.cz - s.height * 0.5f - cfg_.voxel_select_height_margin_m;
    const float z_max = s.cz + s.height * 0.5f + cfg_.voxel_select_height_margin_m;
    // Surface filter: when standing on a table, reject points at/below the table top (+1 cm slack to
    // keep the base ring). These are table-surface deprojections the mask depth-gate let in; they drag
    // the depth/centroid and inflate the lateral spread.
    if (std::isfinite(inst.table_top_z))
        z_min = std::max(z_min, inst.table_top_z + 0.01f);
    return point.z() >= z_min and point.z() <= z_max;
}

std::uint64_t BottleFitter::voxel_key(const Eigen::Vector3f& point, float quantization_m)
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

bool BottleFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_node_id)
{
    room_node_id_ = room_node_id;   // latch every cycle (the support decision + parent default read it)
    if (instances_.count(node.id()))
        return false;

    BottleState init_state;
    init_state.radius = cfg_.prior_radius;
    init_state.height = cfg_.prior_height;

    if (auto v = G_->get_attrib_by_name<width_m_att> (node); v.has_value()) init_state.radius = 0.5f * v.value();
    if (auto v = G_->get_attrib_by_name<height_m_att>(node); v.has_value()) init_state.height = v.value();

    // Read the bottle's current room-frame pose via the RT tree (parent-agnostic: works whether the
    // bottle is parented to the room or the table — room←…←bottle is composed for us).
    if (inner_eigen_)
    {
        if (const auto p = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), node.name(), 0);
            p.has_value())
        {
            init_state.cx = static_cast<float>(p->x());
            init_state.cy = static_cast<float>(p->y());
            init_state.cz = static_cast<float>(p->z());
        }
    }

    // Tracker birth seed: authoritative XY for a freshly-born instance, independent of the same-cycle RT
    // composition (which can read as 0,0 right after birth). Consumed once.
    if (auto it = birth_seeds_.find(node.id()); it != birth_seeds_.end())
    {
        init_state.cx = it->second.x();
        init_state.cy = it->second.y();
        birth_seeds_.erase(it);
    }

    // No checkpoints: the model ALWAYS cold-starts at the fresh masks-detected pose (init_state, set
    // above) and the cold-start centroid snap. Persisted fits reloaded as a stale, drifted start that
    // could deadlock the fit past the voxel-ownership gate (cand=0, "model won't move") — removed.

    // Sanitize non-finite fields so a bad detection can't poison the SDF.
    const auto fix = [&](float& v, float fallback)
    {
        if (not std::isfinite(v)) v = fallback;
    };
    fix(init_state.cx, 0.0f);
    fix(init_state.cy, 0.0f);
    fix(init_state.cz, cfg_.prior_radius > 0.f ? 0.85f : 0.85f);
    fix(init_state.radius, cfg_.prior_radius);
    fix(init_state.height, cfg_.prior_height);

    BottleModelParams mparams = make_model_params();
    // The size PRIOR is a fixed generative-model belief ("a bottle is ~2 cm"), NOT the live estimate.
    // Anchoring it to init_state.radius would let a diverged radius become its own prior (size_energy≈0
    // there) → locked forever. A single depth view cannot observe radius (only the front arc), so the
    // prior must govern that direction; keep it pinned to config.
    mparams.prior_radius = cfg_.prior_radius;
    mparams.prior_height = cfg_.prior_height;
    {
        const auto it = std::find_if(priors_.begin(), priors_.end(),
                                     [&](const BottlePrior& pr){ return pr.node_name == node.name(); });
        if (it != priors_.end())
            mparams.prior_size_std = std::max(mparams.prior_size_std, it->sigma_size);
    }

    BottleInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = BottleModel(init_state, mparams);
    inst.queue     = SampleQueue<BottleModel>(make_queue_params());
    // RT-tree parent (table when hung from it, else room) — write_rt_pose writes in the parent frame.
    inst.parent_id = G_->get_attrib_by_name<parent_att>(node).value_or(room_node_id_);
    if (const auto pn = G_->get_node(inst.parent_id); pn.has_value())
        inst.parent_name = pn.value().name();

    inst.affordance.init(G_, node.id(), node.name());   // epistemic hidden-face affordance for this bottle

    instances_.emplace(node.id(), std::move(inst));
    std::print("bottle_concept: created instance for node '{}' id={}\n", node.name(), node.id());
    return true;
}

bool BottleFitter::should_log(const BottleInstance& inst) const
{
    const int period = std::max(1, cfg_.log_period_frames);
    return (inst.processed_cycles % period) == 0;
}

BottleModelParams BottleFitter::make_model_params() const
{
    BottleModelParams p;
    p.sigma_obs          = cfg_.sigma_obs;
    p.lambda_size        = cfg_.lambda_size;
    p.lambda_pos         = cfg_.lambda_pos;
    p.lambda_state       = cfg_.lambda_state;
    p.prior_radius       = cfg_.prior_radius;
    p.prior_height       = cfg_.prior_height;
    p.prior_size_std     = cfg_.prior_size_std;
    p.optimization_iters = cfg_.optimization_iters;
    p.optimization_lr    = cfg_.optimization_lr;
    p.grad_clip          = cfg_.grad_clip;
    p.optimizer_type     = cfg_.optimizer_type;
    p.sgd_momentum       = cfg_.sgd_momentum;
    p.robust_loss        = cfg_.robust_loss;
    p.robust_loss_scale  = cfg_.robust_loss_scale;
    p.mask_precision     = cfg_.mask_precision;
    p.cov_eff_scale      = cfg_.cov_eff_scale;
    p.lambda_freespace   = cfg_.lambda_freespace;
    p.freespace_margin   = cfg_.freespace_margin;
    return p;
}

SampleQueueParams BottleFitter::make_queue_params() const
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
    p.z_bin_size                   = cfg_.z_bin_size;
    p.diversity_admission          = false;  // bottle: smallest-|SDF| admission (preserve pre-template behaviour)
    return p;
}

}  // namespace rc
