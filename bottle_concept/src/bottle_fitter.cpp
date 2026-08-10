/*
 * bottle_fitter.cpp — the active-inference fit core for bottle_concept.
 */

#include "bottle_fitter.h"
#include "bottle_dof.h"         // kBottleDofs: the AI2 CSV column names

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <print>
#include <unordered_map>
#include <utility>

namespace rc {

BottleFitter::BottleFitter(std::shared_ptr<DSR::DSRGraph> graph,
                           DSR::InnerEigenAPI* inner_eigen,
                           BottleConfig& cfg,
                           MaskIngestor* perception,
                           BottleSceneGraph* scene_graph)
    : G_(std::move(graph)), inner_eigen_(inner_eigen), cfg_(cfg),
      mask_ingestor_(perception), scene_graph_(scene_graph)
{}

// ─── Ego-motion "be-still-to-update" (mirrors refrigerator/chair confirm_only) ───────────────────────
// Robot/camera speed in the room frame from the transform chain (producer-independent). Uses the SAME
// room_T_zed extrinsic the fit uses, pinned to the CURRENT pose (ts=0 = latest). Call once per compute cycle
// on the MAIN thread. ALWAYS checks the returned optional; bails (leaves ego_* unchanged) if the chain is gone.
void BottleFitter::update_ego_motion()
{
    const auto M = room_T_zed_matrix(0);   // camera→room (ts=0 = current robot pose)
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
float BottleFitter::motion_magnitude(const BottleInstance& inst) const
{
    return std::max(std::abs(inst.last_motion_dotd),
                    ego_lin_mps_ + cfg_.ai2_ang_lever_m * ego_ang_radps_);
}

// "Be-still-to-update": true ⇒ this frame may only CONFIRM (predict-only), never move/reshape the geometry mean.
// Robot linear/angular speed above the still-level, OR the mask's own ego-motion corruption (motion_dotd) above
// its still-level. A bottle is a yaw-symmetric CYLINDER, so this governs position + size only (no yaw channel).
bool BottleFitter::confirm_only(const BottleInstance& inst) const
{
    if (not cfg_.ai2_motion_confirm_only)
        return false;
    return ego_lin_mps_   > cfg_.ai2_still_lin_mps
        or ego_ang_radps_ > cfg_.ai2_still_ang_radps
        or std::abs(inst.last_motion_dotd) > cfg_.ai2_still_dotd;
}

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
            inst.last_motion_dotd       = slice.motion_dotd;   // ego-motion smear speed → common-mode fixation
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

// The fit: one recursive full-covariance belief update per fresh frame, on the shared recursive-Laplace
// engine (bottle_belief.h), with the occluding-contour silhouette factor pinning the depth-degenerate
// radius. Static belief (a moved bottle is re-acquired via the tracker gate widening on stale predicts, not
// a CV transition). Writes the posterior back into inst.model so all downstream publish/viewer/RT code
// (which reads inst.model.state()) is unchanged.
float BottleFitter::run_inference(BottleInstance& inst, const BottleObservation& observation)
{
    update_support_surface(inst);   // resting-surface decision + table-top z anchor (reads scene_graph_)

    const int npts = static_cast<int>(observation.candidate_pts.size() + observation.residual_pts.size());

    // Lazy init: warm-start the belief from the model state, but on the FIRST frame snap the centre to the
    // observed cloud — a cylinder far from the points would see them all as clutter (zero gradient) and
    // never converge (the AI2 analogue of the legacy cold-start snap).
    if (not inst.ai2_initialized)
    {
        const auto& m = inst.model.state();
        BottleBeliefState s0{m.cx, m.cy, m.cz, m.radius, m.height};
        if (npts > 0)
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            const auto scan = [&](const std::vector<Eigen::Vector3f>& v) { for (const auto& p : v) sum += p; };
            scan(observation.candidate_pts); scan(observation.residual_pts);
            const Eigen::Vector3f cen = sum / static_cast<float>(npts);
            s0.cx = cen.x(); s0.cy = cen.y(); s0.cz = cen.z();
        }
        BottleBeliefParams p;
        p.sigma_base_m         = cfg_.ai2_sigma_base_m;
        p.clutter_frac         = cfg_.ai2_clutter_frac;
        p.clutter_scale_m      = cfg_.ai2_clutter_scale_m;
        p.prior_pos_std        = cfg_.ai2_prior_pos_std;
        p.prior_size_std       = cfg_.ai2_prior_size_std;
        p.process_std_m        = cfg_.ai2_process_std_m;
        p.process_std_moved_m  = cfg_.ai2_process_std_moved_m;
        p.process_std_size_m   = cfg_.ai2_process_std_size_m;
        p.common_mode_pos_std  = cfg_.ai2_common_mode_pos_std;
        p.common_mode_size_std = cfg_.ai2_common_mode_size_std;
        p.gn_iters             = cfg_.ai2_gn_iters;
        p.min_radius           = BottleModel::MIN_RADIUS;
        p.min_height           = BottleModel::MIN_HEIGHT;
        inst.ai2_belief = BottleBelief(s0, p);
        inst.ai2_initialized = true;
    }

    if (observation.has_fresh_data)
        ingest_observation_voxels(inst, observation);   // keep the viewer's voxel bank fed

    // ★MOTION REQUIRES A CAUSE — set BEFORE any predict path, because every one of them draws Q from here.
    // A bottle does not move by itself, so its position is volatile only to the degree something able to
    // move it is in contact. This single number does all the work: Q_pos mixes toward the carried regime,
    // Sigma_pos grows with it, and the instance tracker — which gates association on squared Mahalanobis
    // against exactly that block — widens its window in step. The "gate that opens when the robot sees a
    // human grab it" is therefore not a gate at all; it is the transition model telling the truth, and
    // association inherits it for free.
    inst.mover_p = mover_belief(inst);
    inst.ai2_belief.set_mover_belief(inst.mover_p);

    // No fresh mask: a bottle is MOVABLE and can leave the FoV, so — unlike a static table — its POSITION
    // information DECAYS without evidence. Inflate ONLY the position block of Σ each unseen frame (its size
    // is constant while unseen); the tracker gates on this Σ, so the gate WIDENS across a look-away and the
    // re-acquired detection (at a shifted apparent position) associates instead of spawning a rebirth,
    // WITHOUT loosening the learned radius/height. Σ is bounded in practice: the negative-information death
    // gate retires the instance long before it grows unusable.
    const auto now = std::chrono::steady_clock::now();
    if (not observation.has_fresh_data)
    {
        // Age the position Σ. Default q_scale=1 (historic one-Q-per-unseen-cycle); with AI2AgeNominalDtS>0
        // scale by real elapsed time so Σ_pos reflects how long the bottle has been unseen on the agent clock.
        float q_scale = 1.0f;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f and inst.last_belief_touch.time_since_epoch().count() != 0)
            q_scale = std::chrono::duration<float>(now - inst.last_belief_touch).count() / cfg_.ai2_age_nominal_dt_s;
        inst.ai2_belief.predict_stale(q_scale);
        inst.last_belief_touch = now;
        update_expected_visible(inst);
        return inst.prev_free_energy;
    }
    inst.last_belief_touch = now;   // fresh path: update() below carries its own Q; reset the age clock

    // Observation precision R = σ_base² (per-point random part); the shared-mask/localization error is
    // carried separately by the common-mode (Woodbury) saturation, not folded into per-point R.
    // Range-scaled precision (perceive precisely when close, leave untouched when far): R grows with the
    // camera→object sensing distance beyond `RangeNearM`, so distant frames carry ~no info and the belief
    // holds. range_scale >= 1; == 1 within grasp range or when disabled.
    const float range_scale = range_precision_scale(inst);
    const float R = cfg_.ai2_sigma_base_m * cfg_.ai2_sigma_base_m * range_scale;

    BottleFrame frame;
    frame.points.reserve(static_cast<std::size_t>(npts));
    frame.points.insert(frame.points.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
    frame.points.insert(frame.points.end(), observation.residual_pts.begin(), observation.residual_pts.end());
    frame.R.assign(frame.points.size(), R);
    frame.precision_scale = range_scale;   // inflate the common-mode with range — the BINDING cap on the mean
    // Ego-motion → COMMON-MODE fixation ("be still to UPDATE, else CONFIRM"): a moving frame's mask is
    // smeared/displaced by ego-motion by ≈ effective-lag·speed — a per-mask SHARED error that per-point R
    // averages away. Route it into the frame's common-mode (position AND size), so the Woodbury cap freezes
    // how far a moving frame can move/reshape the bottle while existence/association still update. CONTINUOUS,
    // no gate: at motion_dotd→0 both terms vanish (a still frame refines fully); gains 0 disable a channel.
    const float mot_dotd     = std::abs(inst.last_motion_dotd);
    const float mot_pos_var  = std::pow(cfg_.motion_cm_pos_gain  * mot_dotd, 2.0f);   // m²  (cx,cy)
    const float mot_size_var = std::pow(cfg_.motion_cm_size_gain * mot_dotd, 2.0f);   // m²  (radius,height)
    // Pose-chain (localization) shared error at the bottle centre, computed in write_rt_pose the previous
    // cycle (slowly varying) — fed into the common-mode so the frame's information saturates (calibrated σ).
    frame.chain_cov_xx   = std::max(0.0f, inst.dbg_chain_cov_xx) + mot_pos_var;   // pose-chain + EGO-MOTION
    frame.chain_cov_yy   = std::max(0.0f, inst.dbg_chain_cov_yy) + mot_pos_var;
    frame.chain_cov_size = mot_size_var;                                          // EGO-MOTION freezes size while moving

    // Occluding-contour silhouette (mask edge rays) — the term that pins the depth-degenerate radius a
    // symmetric cylinder's depth SDF cannot (see bottle-sdf-depth-bias). feed_silhouette computes the
    // room-frame edge rays into inst.model; copy them onto the frame so the belief's accumulate_extra folds
    // the tangent factor. Weighted by cfg_.mask_precision × the YOLO-score reliability (a weak mask must
    // not over-tighten radius). 0 (MaskPrecision off) leaves the AI2 fit on depth alone.
    feed_silhouette(inst);
    if (cfg_.mask_precision > 0.0f and inst.model.silhouette_ray_count() > 0)
    {
        frame.sil_cam_xy    = inst.model.silhouette_cam_xy();
        frame.sil_dirs      = inst.model.silhouette_dirs();
        // Silhouette also fades with range (its radius-pinning must not over-tighten from a distant view).
        frame.sil_precision = cfg_.mask_precision * mask_confidence_weight(inst.model.silhouette_conf()) / range_scale;
    }

    // YOLO-INDEPENDENT LiDAR range channel: sensor-fusion evidence that pins metric depth + radius along the
    // viewing ray (the direction the mask is blind to). Selects this cycle's sweep returns near the instance,
    // down-weighted by sparse coverage + the range fade.
    feed_lidar(inst, frame, range_scale);

    // Ego-motion "be-still-to-update" DISCRETE gate (mirrors refrigerator/chair confirm_only). When the robot is
    // clearly MOVING (ego lin/ang speed OR the mask's own motion_dotd above the still-level), a motion-smeared mask
    // is a shared displacement whose per-point R averages away — it may only CONFIRM the bottle (existence +
    // association still run), never move/reshape a converged one. Take the PREDICT-ONLY branch (Σ inflates, geometry
    // mean HELD) instead of a full geometry update. The continuous common-mode above is the graceful backstop; this
    // discrete gate concentrates GEOMETRY updates at stillness. A static bottle loses nothing by not updating while
    // moving. Bottle is a yaw-symmetric CYLINDER, so the gate governs position + size (radius,height) only — no yaw.
    const bool gated = confirm_only(inst);

    bool  frame_rejected = false;
    float energy;
    float clut = inst.last_clutter_frac;   // hold last on a gated (no-measurement) cycle
    if (gated)
    {
        inst.ai2_belief.predict();          // confirmation only: Σ inflates, geometry mean UNCHANGED
        energy = inst.prev_free_energy;     // hold last FE — a confirm-only frame took no geometry measurement
    }
    else
    {
        // Physical step-bound (fix 2a). A corrupted mask cloud (e.g. mask points deprojected to garbage during
        // fast robot motion) can produce ONE wild GN step that teleports the centre tens of metres; once that
        // happens LiDAR's restoring returns fall outside its robust kernel (huge residual → ~0 weight) and the fit
        // can't recover → runaway. A static-scene bottle cannot physically move MaxStepM in one frame, so a frame
        // whose net centre move exceeds it is an OUTLIER FRAME: reject its update (restore state+Σ), widen the
        // position Σ like a look-away, and mark it no-data so persistent corruption still retires via frames_diverged.
        const BottleBeliefState pre_state = inst.ai2_belief.state();
        const BottleBelief      pre_belief = inst.ai2_belief;   // full snapshot (state + Σ)

        inst.ai2_belief.update(frame);   // MAP mean + posterior Σ; its surface-only return is NOT the FE (below)

        if (cfg_.max_step_m > 0.0f)
        {
            const auto& ns = inst.ai2_belief.state();
            const float dx = ns.cx - pre_state.cx, dy = ns.cy - pre_state.cy, dz = ns.cz - pre_state.cz;
            if (std::sqrt(dx * dx + dy * dy + dz * dz) > cfg_.max_step_m)
            {
                std::print("[{}] AI2 step-bound REJECT: centre moved {:.2f}m (>{:.2f}) — outlier frame dropped\n",
                           inst.node_name, std::sqrt(dx * dx + dy * dy + dz * dz), cfg_.max_step_m);
                inst.ai2_belief = pre_belief;         // reject the corrupted update (restore state + Σ)
                inst.ai2_belief.predict_stale();      // widen position Σ so the next good frame re-associates
                frame_rejected = true;                // explained nothing → drives frames_diverged (retirement)
            }
        }

        // FREE ENERGY = the clutter-INCLUSIVE mixture NLL (rises with misfit), NOT the engine's surface-only
        // responsibility-weighted energy (which reads ≈0 on a bad fit because misfit points route to clutter — the
        // TABLE.md §3 / recipe-invariant-#1 bug that made a diverged fit read F≈0). CLUTTER-FRACTION is the honest
        // "explains none of its data" signal that REPLACES the old energy==0 all-clutter divergence sentinel.
        energy = inst.ai2_belief.mixture_nll(frame.points, inst.ai2_belief.state(), R);
        clut   = inst.ai2_belief.clutter_fraction(frame.points, R);
        inst.last_clutter_frac = clut;
    }

    // Write the belief back into the legacy BottleState so all downstream publish/viewer/RT code is
    // unchanged. Hang the bottle from the table when a support surface is known (cz = table_top + h/2),
    // matching the legacy path's z-anchor.
    const auto& bs = inst.ai2_belief.state();
    BottleState ms = inst.model.state();
    ms.cx = bs.cx; ms.cy = bs.cy; ms.cz = bs.cz; ms.radius = bs.radius; ms.height = bs.height;
    if (std::isfinite(inst.table_top_z))
        ms.cz = inst.table_top_z + 0.5f * ms.height;
    inst.model.set_state(ms);

    ++inst.matched_frames;
    inst.frames_since_detection = 0;
    inst.detection_alive = true;

    // Convergence / divergence / surprise bookkeeping — ONLY on a genuine geometry measurement. A confirm-only
    // (gated) cycle took no measurement (predict-only), so it must NOT accrue frames_diverged (which would retire
    // a perfectly good bottle while the robot is merely moving) nor be misread as converged: leave both counters
    // untouched (matches refrigerator/chair, whose gated branch touches neither).
    if (not gated)
    {
        // GUARD: a fit that explains NONE of its data (drifted/inflated off the cloud) has a HIGH-but-STABLE F (all
        // points to the clutter floor) and would be MISread as "converged" while the state diverges (observed live:
        // radius ran 0.06→0.93 m with frames_converged marching to K_stable). The honest "explains its data" test is
        // the mean clutter responsibility: `explained` iff clutter_fraction < clutter_diverge_frac AND not rejected.
        // (The old sentinel used the surface-only energy==0, only reachable because that energy was blind to
        // misfit; on the clutter-inclusive F, F never hits 0, so the sentinel moves to clutter_fraction.)
        const bool explained = std::isfinite(energy) and not frame_rejected
                               and clut < cfg_.clutter_diverge_frac;
        if (explained and std::abs(energy - inst.prev_free_energy) < cfg_.fe_eps)
            inst.frames_converged = std::min(inst.frames_converged + 1, cfg_.K_stable);
        else
            inst.frames_converged = 0;
        // Divergence persistence: an unexplained fit accrues; the worker retires the instance past
        // cfg_.diverged_retire_frames so it can't keep writing a garbage model.
        inst.frames_diverged = explained ? 0 : inst.frames_diverged + 1;

        // FE-surprise attention (TABLE.md §9): baseline tracks DOWN fast (consolidate a better fit) / UP slow (a
        // sustained rise = the bottle moved surfaces as surprise before the baseline accepts it); surprise = the
        // smoothed positive gap F−baseline. Updated only on an explained (accepted-measurement) frame.
        if (explained)
        {
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
        }
    }

    update_expected_visible(inst);   // negative-information death gate (persist out-of-FoV)

    if (should_log(inst))
        std::print("[{}] AI2 npts={} R={:.4f} | c=({:.3f},{:.3f},{:.3f}) r={:.3f} h={:.3f} | σ(r,h)mm=({:.1f},{:.1f}) | lidar {}/{} raw resid={:.3f}m | F={:.3f} base={:.3f} surprise={:.3f} clut={:.0f}% div={} rscale={:.1f}\n",
                   inst.node_name, npts, R, bs.cx, bs.cy, bs.cz, bs.radius, bs.height,
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(3, 3))),
                   1000.f * std::sqrt(std::max(0.f, inst.ai2_belief.covariance()(4, 4))),
                   inst.dbg_lidar_rays, inst.dbg_lidar_raw, inst.dbg_lidar_resid_m,
                   energy, inst.fe_baseline, inst.fe_surprise, 100.0f * inst.last_clutter_frac, inst.frames_diverged, range_scale);

    log_ai2_csv(inst, npts, R, energy);
    return energy;
}

void BottleFitter::log_ai2_csv(const BottleInstance& inst, int point_count, float R, float energy)
{
    if (cfg_.ai2_csv_path.empty())
        return;

    // Column names come from the shared DOF table (bottle_dof.h) so the CSV header and the dashboard's
    // BeliefInspector rows can no longer drift apart.
    const auto& kDof = kBottleDofs;

    if (not ai2_csv_.is_open())
    {
        ai2_csv_.open(cfg_.ai2_csv_path, std::ios::out | std::ios::trunc);
        if (not ai2_csv_.is_open())
        {
            std::print("bottle_concept: [ai2] cannot open CSV '{}'\n", cfg_.ai2_csv_path);
            cfg_.ai2_csv_path.clear();   // disable further attempts
            return;
        }
        ai2_csv_ << "cycle,node,pts,R,energy,fe_baseline,fe_surprise,clutter_frac,frames_converged,sil_rays";   // sil_rays = silhouette edge rays folded
        for (const auto& d : kDof) ai2_csv_ << ",state_" << d.name;
        for (const auto& d : kDof) ai2_csv_ << ",std_"   << d.name;   // posterior std (m) = sqrt(Σ_jj)
        ai2_csv_ << ",chain_xx,chain_yy,lidar_rays,lidar_raw,lidar_resid_m,frames_diverged,nbv_gain,mover_p";
        ai2_csv_ << '\n';
    }

    const auto s = inst.ai2_belief.state().vec();
    const auto& S = inst.ai2_belief.covariance();
    ai2_csv_ << inst.processed_cycles << ',' << inst.node_name << ',' << point_count << ','
             << R << ',' << energy << ',' << inst.fe_baseline << ',' << inst.fe_surprise << ',' << inst.last_clutter_frac << ',' << inst.frames_converged << ','
             << inst.model.silhouette_ray_count();
    for (int j = 0; j < 5; ++j) ai2_csv_ << ',' << s[j];
    for (int j = 0; j < 5; ++j) ai2_csv_ << ',' << std::sqrt(std::max(0.0f, S(j, j)));
    ai2_csv_ << ',' << std::max(0.0f, inst.dbg_chain_cov_xx) << ',' << std::max(0.0f, inst.dbg_chain_cov_yy);
    ai2_csv_ << ',' << inst.dbg_lidar_rays << ',' << inst.dbg_lidar_raw << ',' << inst.dbg_lidar_resid_m << ',' << inst.frames_diverged;
    ai2_csv_ << ',' << inst.dbg_nbv_gain;   // the PUBLISHED gain the controller ranks on
    // ★The CAUSE column. std_cx/std_cy rising with mover_p ~ 0 is not a moving bottle, it is a bug: the
    // position may only become volatile when something able to move it is in contact. Logged beside the
    // sigmas precisely so the two can never be read apart.
    ai2_csv_ << ',' << inst.mover_p;
    ai2_csv_ << '\n';
    ai2_csv_.flush();
}

std::optional<Eigen::Matrix4d> BottleFitter::room_T_zed_matrix(std::uint64_t timestamp_ms) const
{
    if (not inner_eigen_)
        return std::nullopt;
    // Pin to the frame's capture stamp (0 = latest): with the robot rotating, the LATEST camera pose
    // differs from the one at capture, so a latest-pose de-projection swings the measurement (~±12 cm)
    // and makes a STATIC bottle drift + split. Using the capture pose keeps it stable.
    const auto rtb = inner_eigen_->get_transformation_matrix("room", "body", timestamp_ms);
    const auto btz = inner_eigen_->get_transformation_matrix("body", "zed", timestamp_ms);
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

// Pre-select the sweep returns that plausibly lie on THIS bottle — within (radius + margin) horizontally of
// the current centre estimate and within (h/2 + margin) vertically — then stage them onto the frame's LiDAR
// channel. Pre-selection just bounds the work and keeps a distant object's returns out; final membership is
// the factor's own geometric test (a ray must actually cross the model's SDF), not a distance gate here.
float BottleFitter::range_precision_scale(const BottleInstance& inst) const
{
    if (cfg_.range_near_m <= 0.0f)
        return 1.0f;
    const auto T = room_T_zed_matrix();   // room<-zed; translation col = camera origin in room
    if (not T.has_value())
        return 1.0f;
    const auto& s = inst.ai2_belief.state();
    const double dx = (*T)(0, 3) - s.cx, dy = (*T)(1, 3) - s.cy, dz = (*T)(2, 3) - s.cz;
    const float range = static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
    return std::pow(std::max(1.0f, range / cfg_.range_near_m), cfg_.range_precision_power);
}

void BottleFitter::feed_lidar(BottleInstance& inst, BottleFrame& frame, float range_scale) const
{
    inst.dbg_lidar_rays = 0;
    inst.dbg_lidar_raw  = 0;
    inst.dbg_lidar_resid_m = -1.0f;
    if (cfg_.lidar_precision <= 0.0f or not lidar_have_sweep_ or lidar_sweep_room_.empty()
        or frame.points.empty())
        return;

    // ANCHOR the selection on THIS CYCLE'S FRESH mask-cloud centroid, NOT the model estimate. A model-anchored
    // box FOLLOWS a diverging fit into empty space → 0 rays exactly when LiDAR is most needed (observed live
    // 2026-07-03: centre ran to −91 m, rays→0, no brake). Anchored on the observation, LiDAR keeps selecting
    // the returns at the object's REAL location and can pull a runaway back. Box size is the PHYSICAL bottle
    // footprint (prior radius/height + margin), not the fitted radius, so a blown-up radius can't explode the
    // region either. (Echoes the CV-filter lesson: gate on the fresh obs, not the dragged fit.)
    Eigen::Vector3f c = Eigen::Vector3f::Zero();
    for (const auto& p : frame.points) c += p;
    c /= static_cast<float>(frame.points.size());

    const auto& s = inst.ai2_belief.state();
    const float rxy = cfg_.prior_radius + cfg_.lidar_select_margin_m;          // fixed footprint, NOT fitted r
    const float hz  = 0.5f * cfg_.prior_height + cfg_.lidar_select_margin_m;
    const float rxy2 = rxy * rxy;
    // Generous diagnostic box (1.6× horizontal, 2× vertical): counts returns NEAR the bottle that the tight
    // box may miss. Kept below the inter-bottle spacing so it can't grab a neighbour's returns.
    const float rraw2 = (1.6f * rxy) * (1.6f * rxy);
    const float hraw  = 2.0f * hz;
    int raw = 0;

    frame.lidar.endpoints.clear();
    frame.lidar.endpoints.reserve(64);
    double resid_sum = 0.0;
    for (const auto& p : lidar_sweep_room_)
    {
        const float dx = p.x() - c.x(), dy = p.y() - c.y();
        const float dh2 = dx * dx + dy * dy;
        if (dh2 <= rraw2 and std::abs(p.z() - c.z()) <= hraw) ++raw;   // generous — "is a return anywhere near?"
        if (dh2 > rxy2) continue;
        if (std::abs(p.z() - c.z()) > hz) continue;
        frame.lidar.endpoints.push_back(p);
        resid_sum += std::abs(inst.ai2_belief.sdf_cylinder(p, s));   // |dist to CURRENT model surface| (diag)
    }
    inst.dbg_lidar_raw = raw;
    // Frame-sanity diagnostic (drives the smoke-test): #returns landing on this bottle + how far they sit
    // from the current surface. Few rays / large residual ⇒ wrong LidarFrameNode (mount double-applied).
    inst.dbg_lidar_rays = static_cast<int>(frame.lidar.endpoints.size());
    if (inst.dbg_lidar_rays > 0)
        inst.dbg_lidar_resid_m = static_cast<float>(resid_sum / inst.dbg_lidar_rays);
    if (frame.lidar.endpoints.empty())
        return;

    // Precision = base, DOWN-WEIGHTED by sparse coverage (few noisy returns must not swing radius; bottle_2
    // finding) and by the range fade (far → low precision → object left untouched). rays/N0 capped at 1.
    const float coverage = std::min(1.0f, static_cast<float>(inst.dbg_lidar_rays) / std::max(1.0f, cfg_.lidar_coverage_n0));
    frame.lidar.origin     = lidar_origin_room_;
    frame.lidar.precision  = cfg_.lidar_precision * coverage / std::max(1.0f, range_scale);
    frame.lidar.robust_c_m = cfg_.lidar_robust_c_m;
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
    // w = clamp01((conf−floor)/(ref−floor))^power — a weak mask scales down the silhouette precision so it
    // can't over-tighten the depth-degenerate radius. The score WIDENS Σ but the SDF geometry sets its SHAPE.
    const float floor = std::clamp(cfg_.mask_conf_floor, 0.0f, 0.99f);
    const float ref   = std::max(floor + 1e-3f, cfg_.mask_conf_ref);
    return std::pow(std::clamp((confidence - floor) / (ref - floor), 0.0f, 1.0f),
                    std::max(0.1f, cfg_.mask_conf_power));
}

void BottleFitter::update_expected_visible(BottleInstance& inst)
{
    inst.expected_visible = false;   // permanence default: only retire if CONFIRMED in-frame & undetected
    if (not inner_eigen_)
        return;
    if (not camera_api_)
    {
        const auto zed = G_->get_node("zed");
        if (not zed.has_value()) return;
        camera_api_ = G_->get_camera_api(zed.value());
        if (not camera_api_) return;
    }
    const auto& s = inst.model.state();
    const auto p = inner_eigen_->transform("zed", Mat::Vector3d(s.cx, s.cy, s.cz), "room", 0);
    if (not p.has_value())
        return;
    const double depth = p->y();   // camera frame: x=right, y=forward(depth), z=up (see feed_silhouette)
    if (depth <= 0.05)
        return;                    // behind/at the camera → not in the frustum
    const float fx = camera_api_->get_focal_x(), fy = camera_api_->get_focal_y();
    const float W  = static_cast<float>(camera_api_->get_width());
    const float H  = static_cast<float>(camera_api_->get_height());
    const double col = p->x() * fx / depth + 0.5 * W;
    const double row = 0.5 * H - p->z() * fy / depth;
    inst.expected_visible = (col >= 0.0 and col <= W and row >= 0.0 and row <= H);
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

    BottleInstance inst;
    inst.node_id   = node.id();
    inst.node_name = node.name();
    inst.model     = BottleModel(init_state, mparams);
    // RT-tree parent (table when hung from it, else room) — write_rt_pose writes in the parent frame.
    inst.parent_id = G_->get_attrib_by_name<parent_att>(node).value_or(room_node_id_);
    if (const auto pn = G_->get_node(inst.parent_id); pn.has_value())
        inst.parent_name = pn.value().name();

    // "bottle" selects the execution contract (Servo lock-on bound to the bottle detection attrs).
    inst.affordance.init(G_, node.id(), node.name(), "bottle");

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
    p.prior_radius       = cfg_.prior_radius;
    p.prior_height       = cfg_.prior_height;
    p.prior_size_std     = cfg_.prior_size_std;
    return p;
}


// See bottle_fitter.h. The cause that licenses position volatility.
float BottleFitter::mover_belief(const BottleInstance& inst) const
{
    if (not cfg_.motion_requires_cause or not G_ or inner_eigen_ == nullptr)
        return 1.0f;   // feature off => behave exactly as before: the position is always free to move

    const auto& st = inst.ai2_belief.state();
    const float reach = std::max(0.05f, cfg_.mover_reach_m);
    float best = 0.0f;
    for (const auto& person : G_->get_nodes_by_type("person"))
    {
        const auto c = inner_eigen_->transform("room", Mat::Vector3d(0.0, 0.0, 0.0), person.name(), 0);
        if (not c.has_value())
            continue;   // un-resolvable pose => this mover cannot be reasoned about
        const float dx = static_cast<float>(c->x()) - st.cx;
        const float dy = static_cast<float>(c->y()) - st.cy;
        const float d  = std::hypot(dx, dy);
        best = std::max(best, std::exp(-0.5f * (d / reach) * (d / reach)));
    }
    return std::clamp(best, 0.0f, 1.0f);
}

}  // namespace rc
