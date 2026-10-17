// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// One localiser cycle: RoomConcept::update.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{

    RoomConcept::UpdateResult RoomConcept::update(
                const LidarData &lidar,
                const std::vector<VelocityCommand> &velocity_history,
                const std::vector<OdometryReading> &odometry_history)
    {
        t_update_start_ = std::chrono::high_resolution_clock::now();
        UpdateResult res;
        if (lidar.first.empty() || model_ == nullptr)
            return res;

        res.lidar_scan = lidar.first;   // store scan for synchronized visualization

        // ===== MANUAL RESET: skip optimization during settle period =====
        if (manual_reset_frames_ > 0)
        {
            manual_reset_frames_--;
            const auto state = model_->get_state();
            res.ok = true;
            res.state = state;
            res.robot_pose.translation() = Eigen::Vector2f(state[2], state[3]);
            res.robot_pose.linear() = Eigen::Rotation2Df(state[4]).toRotationMatrix();
            res.covariance = current_covariance;
            res.timestamp_ms = lidar.second;
            last_update_result = res;
            last_lidar_timestamp = lidar.second;
            return res;
        }

        // ===== ORIENTATION SEARCH ON FIRST UPDATE =====
        if (needs_orientation_search_)
        {
            const auto state = model_->get_state();
            const float best_phi = find_best_initial_orientation(lidar.first, state[2], state[3], state[4]);
            model_->robot_theta = best_phi;
            needs_orientation_search_ = false;
        }

        // ===== PREDICTION =====
        const auto motion_prior_selection = build_motion_prior_selection(velocity_history, odometry_history, lidar);
        const auto &selected_prior = motion_prior_selection.selected_prior;
        const PredictionState prediction = predict_step(model_, selected_prior, true);

        if (prediction.have_propagated)
            current_covariance = prediction.propagated_cov;

        const auto pred_pos = motion_prior_selection.predicted_pos;
        const auto pred_theta = motion_prior_selection.predicted_theta;
        last_pred_pos_ = pred_pos;      // kept for UpdateResult on BOTH return paths
        last_pred_theta_ = pred_theta;

        // ===== BUILD NEW WINDOW SLOT =====
        const auto& all_points = lidar.first;
        std::vector<Eigen::Vector3f> sampled_points;
        if (static_cast<int>(all_points.size()) > params.max_lidar_points)
        {
            const int stride = static_cast<int>(all_points.size()) / params.max_lidar_points;
            sampled_points.reserve(params.max_lidar_points);
            for (size_t i = 0; i < all_points.size(); i += stride)
                sampled_points.push_back(all_points[i]);
        }
        else
            sampled_points = all_points;

        const ScanPoints& scan_pts = sampled_points;
        tracking_step_count_++;

        // Motion constraint for the new slot.
        // Prefer measured odometry (encoder/IMU) when available — it is more accurate than
        // the commanded velocity, especially during rotation where wheel slip can differ from
        // the commanded angular rate.  The initial pose (pred_pos, pred_theta) was already
        // computed from the fused estimate inside apply_dual_prior_fusion, so using the same
        // measured delta here removes the command/measured inconsistency that previously caused
        // Adam to fight the motion factor during turns.
        Eigen::Vector3f slot_odom_delta = Eigen::Vector3f::Zero();
        Eigen::Matrix3f slot_motion_cov = Eigen::Matrix3f::Identity() * params.default_slot_motion_cov;
        if (selected_prior.valid)
        {
            slot_odom_delta = selected_prior.delta_pose;
            slot_motion_cov = selected_prior.covariance_eigen;
        }

        // ---- Strided window: decide ADMIT vs REPLACE, and span the motion factor accordingly -------
        // See Params::window_stride_enabled. The per-frame delta above describes ONE inter-frame gap.
        // While we are replacing the newest slot instead of appending, the motion factor that slot
        // carries must describe the whole interval back to the last ADMITTED slot — otherwise the
        // window's oldest-to-newest chain is constrained by a fraction of the motion that actually
        // happened, which under-constrains exactly the DOF this change exists to fix.
        bool stride_replace = false;
        const Eigen::Vector3f frame_odom_delta = slot_odom_delta;   // THIS frame's motion, for the recorder
        if (params.window_stride_enabled)
        {
            if (stride_has_admitted_ and window_mgr_.size() > 1)
            {
                const float travel = std::hypot(pred_pos.x() - stride_last_admitted_[0],
                                                pred_pos.y() - stride_last_admitted_[1]);
                const float turn = std::abs(std::remainder(pred_theta - stride_last_admitted_[2],
                                                           2.f * static_cast<float>(M_PI)));
                stride_replace = (travel < params.window_min_travel_m
                                  and turn < params.window_min_turn_rad);
            }
            // ★ What the factor spans (stride_span.h). The newest slot is always the PREVIOUS frame, so a
            //   REPLACED frame links to the newest's predecessor and carries newest ⊕ this frame, while an
            //   APPENDED frame links to the previous frame and carries this frame only. Until 2026-10-05
            //   one accumulator served both and an appended slot carried the whole stride as one frame:
            //   ±3-9 cm correction pairs at every admission. The preintegrated form is CHAINED (chain()
            //   keeps the transport term that `+=` on the covariance drops).
            const rc::preint::Interval *frame_preint =
                (params.motion_preintegration and selected_prior.has_preint) ? &selected_prior.preint : nullptr;
            const auto &span = stride_replace ? stride_span_.replace(slot_odom_delta, slot_motion_cov, frame_preint)
                                              : stride_span_.append(slot_odom_delta, slot_motion_cov, frame_preint);
            slot_odom_delta = span.delta;
            slot_motion_cov = span.cov;
            // covariance() is called exactly HERE, once, on the interval the motion factor will
            // actually carry — never per frame (see the warning on Interval::covariance()).
            if (params.motion_preintegration and span.preint.samples > 0)
                slot_motion_cov = span.preint.covariance();
        }

        // Mirror what this slot's motion factor actually got, for BOTH debug-log writers (the
        // early-exit one is in another function and can only see members).
        last_slot_motion_cov_ = slot_motion_cov;
        if (params.motion_preintegration and params.window_stride_enabled and stride_span_.current().preint.samples > 0)
        {
            last_preint_samples_    = stride_span_.current().preint.samples;
            last_preint_duration_s_ = stride_span_.current().preint.duration_s;
        }
        else if (params.motion_preintegration and selected_prior.has_preint)
        {
            last_preint_samples_    = selected_prior.preint.samples;
            last_preint_duration_s_ = selected_prior.preint.duration_s;
        }
        else { last_preint_samples_ = 0; last_preint_duration_s_ = 0.f; }

        WindowSlot new_slot;
        new_slot.pose = Eigen::Vector3f(pred_pos.x(), pred_pos.y(), pred_theta);
        new_slot.lidar_points = scan_pts;
        new_slot.odometry_delta = slot_odom_delta;
        new_slot.motion_cov = slot_motion_cov;
        new_slot.pose_cov = current_covariance;
        new_slot.timestamp_ms = lidar.second;
        // ── THE CONNECTION, missing until 2026-08-26 ──────────────────────────────────────────────
        // pump_image_edges() has been extracting contours and calling set_image_edges() every tick,
        // and NOTHING read them: `image_edges()` had zero call sites and `new_slot.image_edges` was
        // never assigned, so Slot::image_edges was empty for every slot ever built. Every consumer
        // guards on `obs.empty()` / `image_edges.empty()` and so all of them silently did nothing —
        // the shadow monitor, the driving factor, and both loss terms. `drive = true` would have
        // been inert too. A feature that is enabled, extracting, and disconnected looks exactly like
        // a feature that is working and has nothing to say.
        if (params.image_edge.enable)
            new_slot.image_edges = take_image_edges();

        new_slot.motion_prec = slot_motion_cov.inverse();
        // ── The rest hypothesis, as a constraint on the STATE ────────────────────────────────────
        // A zero-delta motion factor between this slot and the last: "you did not move", with the
        // interval's own precision. The solver then weighs it against "you moved by Delta" and the
        // MEAN goes where the evidence says — which the covariance-shaping form could never do,
        // because the preintegrator never writes the mean.
        if (params.zupt_as_factor and selected_prior.has_preint)
        {
            const Eigen::Matrix3f R = selected_prior.preint.zupt_covariance();
            if (R(0, 0) > 0.f and R(2, 2) > 0.f)
            {
                new_slot.zupt_prec = R.inverse();
            }
        }

        rec_stride_replace_ = stride_replace;   // RecordWallInput (diagnostic only)
        rec_odom_delta_     = frame_odom_delta;   // v2: per FRAME (v1 recorded the stride accumulator)
        bool window_slid = false;
        if (stride_replace)
        {
            // Not enough motion to be worth a slot: overwrite the newest so the current frame is still
            // represented (and still optimized) without consuming the window's span. Nothing is
            // marginalized and nothing is dropped, so the older slots keep their separation.
            window_mgr_.newest() = std::move(new_slot);
        }
        else
        {
            // FEJ+Schur: marginalize the dropping slot BEFORE append() pops it, while both x₀ (front)
            // and its Markov blanket x₁ (window[1]) are still live at their converged linearization
            // points. append() then just pops (fej_schur=true suppresses the legacy mu re-anchoring),
            // and the post-optimization recompute_boundary_prior() below is skipped.
            // Wall-SLAM: the slot about to be dropped hands its wall observations to the map's carried
            // information — the ONLY place that information may grow (see absorb_wall_observations).
            // ... unless the layout is frozen. This is the ONLY place a wall's information grows,
            // so skipping it is what actually holds the geometry still; the guards above only stop
            // the structure from changing around it.
            if (searching() and not window_mgr_.empty()
                and static_cast<int>(window_mgr_.size()) >= params.rfe_window_size)
            {
                gn::Input ain; ain.params = &params; ain.walls = &wall_map_;
                const auto& front = window_mgr_.window.front();
                gn::absorb_wall_observations(ain, front, front.pose);
            }
            if (params.boundary_fej_schur)
                window_mgr_.marginalize_oldest(*model_, params);

            window_slid = window_mgr_.append(std::move(new_slot), params.rfe_window_size,
                                             params.boundary_mu_quality_threshold,
                                             params.boundary_fej_schur);
            if (params.window_stride_enabled)
            {
                stride_last_admitted_ = Eigen::Vector3f(pred_pos.x(), pred_pos.y(), pred_theta);
                stride_has_admitted_  = true;   // stride_span_ already restarted from this frame (append)
            }
        }
        window_mgr_.subsample_old_slots(params.rfe_max_lidar_per_old_slot);

        // ── OPEN DOORWAYS, CARRIED INTO THE ROBOT FRAME ONCE PER SCAN ───────────────────────────
        // The worker writes the apertures in ROOM coordinates (it owns the shared InnerEigenAPI); the
        // scan is in ROBOT coordinates, and so is every consumer of the observation weights. Carrying
        // the two or three segments across once here is cheaper than carrying thousands of points the
        // other way, and it makes the beam origin the origin, so the weight function needs no pose.
        // The pose used is the newest window slot's, i.e. the linearisation point — which is exactly
        // right, because these weights are detached constants AT that point by contract.
        // Empty snapshot (every door shut, or none known) ⇒ empty here ⇒ no cost at all downstream.
        doors_robot_.clear();
        if (const auto doors_room = door_apertures(); not doors_room.empty() and not window_mgr_.empty())
        {
            const Eigen::Vector3f& npa = window_mgr_.newest().pose;
            const Eigen::Rotation2Df R_inv(-npa[2]);
            const Eigen::Vector2f t(npa[0], npa[1]);
            doors_robot_.reserve(doors_room.size());
            // Seat each aperture on the room's OWN boundary (the polygon the SDF scores against, same frame) before
            // carrying it across: the crossing test runs at the mouth, the leaf δ behind it (DoorApertures::seat_in_room).
            const auto poly = current_room_polygon();
            for (const auto& d : doors_room)
            {
                DoorAperture r = d;
                DoorApertures::seat_in_room(r, poly);
                if (const auto it = door_recess_logged_.find(r.id);
                    it == door_recess_logged_.end() or std::abs(it->second - r.recess) > 0.5f * params.rfe_obs_sigma)
                {   // log on change of the seat by half the SDF noise scale — a reporting cadence, not a model gate
                    door_recess_logged_[r.id] = r.recess;
                    qInfo().noquote() << QString("[room][doors] %1 seated on the room boundary: mouth (%2,%3)-(%4,%5), "
                                                 "leaf line %6 m behind it, p_open %7")
                                             .arg(QString::fromStdString(r.name))
                                             .arg(r.a.x(), 0, 'f', 2).arg(r.a.y(), 0, 'f', 2)
                                             .arg(r.b.x(), 0, 'f', 2).arg(r.b.y(), 0, 'f', 2)
                                             .arg(r.recess, 0, 'f', 3).arg(r.p_open, 0, 'f', 2);
                }
                r.a = R_inv * (r.a - t);
                r.b = R_inv * (r.b - t);
                doors_robot_.push_back(std::move(r));
            }
        }
        window_mgr_.doors_newest_robot = &doors_robot_;

        // ===== WALL-SLAM OBSERVATION (Estimate mode) =====
        if (estimating())
        {
            Eigen::Vector3f wall_pose = window_mgr_.newest().pose;
            // ── PLAIN SLAM OWNS THE POSE WHILE THE ROOM IS LEARNT (scan_slam.h) ──────────────────────────────
            // The scan is registered against the occupancy map of the scans already registered, from the odometry
            // prediction; the result REPLACES the newest slot's pose, and the solve below is skipped. The wall map
            // then observes at a pose it did not produce — it can no longer drag the pose it is built from.
            if (plain_slam_driving() or map_layout_adopted_)   // after the switch: SHADOW only (pose untouched)
            {
                if (plain_slam_seed_)
                {
                    plain_slam_.reset();
                    plain_slam_.set_pose(wall_pose - rec_odom_delta_);   // step() adds this frame's delta back
                    plain_slam_seed_ = false;
                }
                std::vector<Eigen::Vector2f> xy;
                xy.reserve(sampled_points.size());
                for (const auto& q : sampled_points) xy.emplace_back(q.x(), q.y());
                const auto t_slam0 = std::chrono::steady_clock::now();
                const Eigen::Vector3f slam_pose = plain_slam_.step(rec_odom_delta_, xy);
                last_t_slam_ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_slam0).count();
                if (plain_slam_driving())
                {
                    wall_pose = slam_pose;
                    window_mgr_.newest().pose = wall_pose;
                }
            }
            const auto t_wall0 = std::chrono::steady_clock::now();
            wall_slam_observe(sampled_points, wall_pose, lidar.second);
            last_t_wall_ms_ = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_wall0).count();
        }

        // ===== CORNER DETECTION (optional, controlled by EnableCornerTracking) =====
        if (params.enable_corner_tracking && !init_polygon_vertices_.empty())
        {
            const Eigen::Vector3f& pa = window_mgr_.newest().pose;
            const float cx = pa[0], cy = pa[1], cth = pa[2];

            // current_covariance feeds the association gate: a poorly-localized robot associates
            // permissively, a sharply-localized one refuses a neighbouring corner outright.
            auto det = corner_detector_.detect(all_points, cx, cy, cth, current_covariance);
            res.corners_in_fov = det.corners_in_fov;
            res.corner_matches = det.matches;

            // ── NIS and every input to it → tmp/corner_nis.csv, one row per frame ────────────
            // This was a terminal line and it had grown to two dense rows of numbers nobody could
            // read while a tour was running. A file is the right place: it keeps the whole history
            // rather than the throttled 1-in-20 sample, and it can be plotted.
            // NIS/dof averages 1.0 when the covariance is honest; nis_pre is measured BEFORE the
            // gate and against the corner each detection was formed from, so it is the calibration
            // number. nis_acc comes from the chosen assignment — an argmin, biased low — and
            // diagnoses assignment quality only. The sigma_* columns split S into its three terms
            // so a value below 1 can be attributed; the input columns say which factor of the
            // propagation carries it.
            // ⚠ imbued with the classic locale: this machine runs es_ES and a comma decimal
            // separator would make the file unreadable by from_chars (CLAUDE.md).
            if (det.nis_pre_n > 0)
            {
                if (not nis_csv_.is_open())
                {
                    std::filesystem::create_directories("tmp");
                    nis_csv_.open("tmp/corner_nis.csv", std::ios::out | std::ios::trunc);
                    if (nis_csv_.is_open())
                    {
                        nis_csv_.imbue(std::locale::classic());
                        nis_csv_ << "frame,nis_pre,n,over,tour_nis_pre,tour_se,tour_n,tour_over_frac,"
                                    "tour_nis_acc,sigma_det,sigma_pred,sigma_model,resid_mean,resid_max,"
                                    "ori_mean,ori_min,line_sigma_phi_deg,line_sigma_d_m,npts,resid_sig_m,"
                                    "lever_m,corner_angle_deg,corners_in_fov,accepted,rej_dist\n";
                    }
                }
                if (nis_csv_.is_open())
                {
                    const auto& T = corner_detector_.tour_stats();
                    nis_csv_ << nis_log_tick_++ << ','
                             << det.nis_pre_mean() << ',' << det.nis_pre_n << ',' << det.nis_pre_over << ','
                             << T.nis_pre_mean() << ',' << T.nis_pre_se() << ',' << T.nis_pre_n << ','
                             << (T.nis_pre_n ? static_cast<float>(T.nis_pre_over) / T.nis_pre_n : 0.f) << ','
                             << T.nis_acc_mean() << ','
                             << T.s_det_sigma() << ',' << T.s_pred_sigma() << ',' << T.s_model_sigma() << ','
                             << det.resid_mean << ',' << det.resid_max << ','
                             << T.ori_mean() << ',' << T.ori_min << ','
                             << T.sphi_deg() << ',' << T.sd_m() << ',' << T.npts_mean() << ','
                             << T.resid_sig() << ',' << T.lever_m() << ',' << T.angle_deg() << ','
                             << det.corners_in_fov << ',' << det.corners_accepted << ',' << det.rej_dist << '\n';
                    if ((nis_log_tick_ % 50) == 0) nis_csv_.flush();
                }
            }

            // ── The RAW per-candidate dump → tmp/corner_probe.csv ────────────────────────────
            // The frame file above is all means, and means are what sent the last three diagnoses to
            // the wrong term. This writes one row per candidate evaluated at the association gate,
            // with no aggregation at all: both lines' point counts, scatter, along-wall spread and
            // (φ,d) covariance, the Jacobian's lever and sin(θ), the three terms of S, and the
            // innovation. Capped at 300k rows (~40 MB) because it is a diagnostic, not a log.
            if (not det.probes.empty() and probe_rows_ < 300000u)
            {
                if (not probe_csv_.is_open())
                {
                    std::filesystem::create_directories("tmp");
                    probe_csv_.open("tmp/corner_probe.csv", std::ios::out | std::ios::trunc);
                    if (probe_csv_.is_open())
                    {
                        probe_csv_.imbue(std::locale::classic());   // es_ES would write commas (CLAUDE.md)
                        probe_csv_ << "frame,model_index,propagated,over_gate,d2,"
                                      "det_x,det_y,pred_x,pred_y,nu_x,nu_y,"
                                      "sdet_xx,sdet_xy,sdet_yy,sprd_xx,sprd_xy,sprd_yy,smodel_xx,smodel_yy,"
                                      "angle_deg,sin_theta,"
                                      "npts0,nraw0,rival0,rival_share0,ori0,resid_sig0,s_mean0,s_std0,s_span0,lever0,c00_0,c01_0,c11_0,"
                                      "npts1,nraw1,rival1,rival_share1,ori1,resid_sig1,s_mean1,s_std1,s_span1,lever1,c00_1,c01_1,c11_1\n";
                    }
                }
                if (probe_csv_.is_open())
                {
                    for (const auto& pr : det.probes)
                    {
                        probe_csv_ << probe_frame_ << ',' << pr.model_index << ','
                                   << pr.propagated << ',' << pr.over_gate << ',' << pr.d2 << ','
                                   << pr.det_x << ',' << pr.det_y << ',' << pr.pred_x << ',' << pr.pred_y << ','
                                   << pr.nu_x << ',' << pr.nu_y << ','
                                   << pr.sdet_xx << ',' << pr.sdet_xy << ',' << pr.sdet_yy << ','
                                   << pr.sprd_xx << ',' << pr.sprd_xy << ',' << pr.sprd_yy << ','
                                   << pr.smodel_xx << ',' << pr.smodel_yy << ','
                                   << pr.angle_deg << ',' << pr.sin_theta;
                        for (int k = 0; k < 2; ++k)
                            probe_csv_ << ',' << pr.npts[k] << ',' << pr.nraw[k] << ',' << pr.rival[k] << ',' << pr.rival_share[k] << ',' << pr.ori[k] << ',' << pr.resid_sig[k]
                                       << ',' << pr.s_mean[k] << ',' << pr.s_std[k] << ',' << pr.s_span[k]
                                       << ',' << pr.lever[k] << ',' << pr.c00[k] << ',' << pr.c01[k]
                                       << ',' << pr.c11[k];
                        probe_csv_ << '\n';
                        ++probe_rows_;
                    }
                    ++probe_frame_;
                    if ((probe_rows_ % 500u) < det.probes.size()) probe_csv_.flush();
                }
            }

            // Per-model-corner attribution → etc/corner_stats.csv. This is what decides whether a
            // given pillar earns its landmark status or should follow the trace artefacts out of
            // set_model_corners; the aggregate rej_* counters cannot say WHICH corner is at fault.
            if (params.corner_stats_csv)
                accumulate_corner_stats(det);

            // Acceptance-rate diagnostic removed from the terminal (2026-07-25). DetectionResult still
            // CARRIES every number it printed (corners_in_fov/detected/accepted, rej_*, soft_orient,
            // convex agree, merged_coincident, model_dup_dropped) — re-enabling is a print, not a re-derivation.
            // The residual-distribution line and the per-corner AMBIGUOUS dump that lived here were
            // one-off instrumentation for setting map_sigma from data (2026-07-20) and are removed now
            // that it is set. DetectionResult still CARRIES every one of those numbers — resid_mean/max,
            // resid_chi2_mean, corners_with_rival, runnerup_chi2_mean, min_assoc_prob, and per-match
            // assoc_prob / assoc_chi2_val / runnerup_chi2 — so re-enabling is a print, not a re-derivation.
            // (They were also mis-scoped: only the print above was inside the %20 throttle, so those two
            // fired EVERY frame.)

            // Store corner observations in the newest slot for the RFE loss
            auto& newest_slot = window_mgr_.newest();
            newest_slot.corner_obs.clear();
            for (const auto& m : det.matches)
            {
                if (m.model_index < 0 || m.model_index >= static_cast<int>(init_polygon_vertices_.size()))
                    continue;
                // Retired by the information-yield rule: still detected, still drawn (as retired),
                // but it must not vote. This is the ONLY place retirement touches inference.
                if (m.suppressed)
                    continue;
                // Contract check at the OPTIMIZER BOUNDARY (independent of the detector's own check —
                // this is the line the Hessian is actually built from). A non-finite observation, or a
                // zero-precision one, must never enter: it constrains nothing yet still participates in
                // the Hessian assembly, which is how min_ev → 0 (cond_num sentinel 1e8) produced NaN
                // losses and a NaN pose on 2026-07-21. Dropping it degrades to SDF-only, which is the
                // intended cascade.
                if (not m.detected.allFinite() or not m.information.allFinite())
                    continue;
                if (m.information.cwiseAbs().maxCoeff() <= 1e-9f)
                    continue;
                WindowSlot::CornerObs obs;
                obs.model_corner_world = init_polygon_vertices_[m.model_index];
                obs.detected_robot = m.detected;
                obs.information = m.information;   // graded Λ_det (robot frame) — used anisotropically by the loss
                newest_slot.corner_obs.push_back(obs);
            }
        }

        // Store validated object anchors (from the graph, set on the main thread) in the newest
        // slot so the RFE loss can add them as SE(2) pose-landmark factors.
        if (params.object_anchor.enable)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            window_mgr_.newest().object_anchors = latest_object_anchors_;
        }

        // The motion model's predictive covariance grows by this cycle's increment noise whether or not the
        // cycle is then corrected; score_surprise() scores and resets it at the next correction.
        // The scan-to-scan innovation's per-cycle inputs (motion_noise_innov.h). innov_ok_ is set again
        // only by a scan-only pose taken THIS cycle, so a cycle that ends on any other path breaks the chain.
        ++innov_cycle_;
        innov_ok_ = false;
        cyc_U_ok_ = false;
        cyc_odom_ = selected_prior.valid ? selected_prior.delta_pose : Eigen::Vector3f::Zero();
        cyc_dt_s_ = selected_prior.valid ? selected_prior.dt * 1e-3f : 0.f;   // OdometryPrior::dt is in ms
        if (selected_prior.valid and selected_prior.covariance_eigen.allFinite())
        {
            sur_P_pred_ += selected_prior.covariance_eigen; ++sur_open_cycles_;
            {   // the odometry's own signed motion over the stretch, for the drift monitor (drift_monitor.h)
                const Eigen::Vector3f &dp = selected_prior.delta_pose;
                const float th_mid = pred_theta - 0.5f * dp[2];
                sur_d_   += -std::sin(th_mid) * dp[0] + std::cos(th_mid) * dp[1];   // body forward = (-sin, cos)
                sur_phi_ += dp[2];
                sur_T_   += selected_prior.dt;
            }
            // The noise COMPONENTS this cycle added (motion_noise_vc.h), and their coefficient-weighted sum
            // as applied. Only a MEASURED, preintegrated prior carries them; a stretch with any other cycle
            // (command, fused, legacy diagonal) cannot say what the coefficients should be: it does not train.
            const auto &np = params.odom_preint_noise;
            if (motion_prior_selection.source == MotionPriorSource::Measured and selected_prior.has_preint
                and np.motion_proportional)
            {
                const std::array<float, rc::preint::Interval::NC> kk{np.k_long, np.k_lat, np.k_lat_turn,
                                                                     np.k_th_turn, np.k_t_trans, np.k_t_rot};
                for (int j = 0; j < rc::preint::Interval::NC; ++j)
                {
                    sur_U_[j]  += selected_prior.preint.unit[j];
                    sur_QK_    += kk[j] * selected_prior.preint.unit[j];
                    cyc_U_[j]   = selected_prior.preint.unit[j];
                }
                cyc_U_ok_ = cyc_odom_.allFinite();
            }
            else
                sur_mixed_ = true;
        }

        // ===== EARLY EXIT CHECK =====
        // ★NOT WHILE SEARCHING (09-17) — the relocaliser's rule (5ca93b4). The early exit trusts the prediction
        // against a FIXED map and skips the solve; while the layout is still estimated there is no fixed map. With
        // the room publishable on a parked full view, map_ready_ went true at t = 0 and 99-100% of solves were
        // skipped through the first 2.9 m of driving (pose = odometry, drifted 1.4 m, map collapsed).
        // While SEARCHING the early exit must not fire, but its MEASUREMENT still feeds the "pred (SDF)"
        // series — separate the measurement from the decision rather than starving the instrument.
        if (map_guided_checks_allowed() and searching())
            (void)try_prediction_early_exit(scan_pts, slot_odom_delta, selected_prior, lidar.second, true);
        if (auto early = (map_guided_checks_allowed() and not searching())
                             ? try_prediction_early_exit(scan_pts, slot_odom_delta, selected_prior, lidar.second)
                             : std::optional<UpdateResult>{})
        {
            // Store quality for future boundary prior gate (early exit = good pose).
            window_mgr_.newest().sdf_mse_final = early->sdf_mse;
            // Motion-model adaptation runs here too so prediction-mode frames count.
            service_calibration();
            early->corners_in_fov = res.corners_in_fov;
            early->corner_matches = std::move(res.corner_matches);
            early->lidar_scan = std::move(res.lidar_scan);
            // The MOUNT monitor runs on early-exit cycles too — pass 1 only, no second solve. This
            // is where ~99.75% of cycles end, so leaving it below the return starved the instrument
            // exactly in proportion to how good the prediction had become.
            // ── The ROW is written whenever the term runs; only the PROBE needs drive off ─────
            // ★ These two were one gate until 2026-08-29, so turning drive ON silently stopped
            //   etc/image_edge.csv — chi2_per_dof, r_rms_px, sum_gamma, n_used, info_ratio, every
            //   number that says whether the term is healthy — in exactly the arm where the term is
            //   actually steering the pose. A channel that logs its fit quality only while it cannot
            //   affect anything is instrumented backwards.
            // The twice-solve probe (dpose_*) genuinely does need drive off: with drive on the term
            // is already in the authority solve, so the "with/without" comparison has nothing left
            // to compare against and those columns are written zero.
            if (params.image_edge.enable and params.image_edge_shadow)
                run_image_edge_shadow(read_window_poses(), boundary_weight_now(), lidar.second,
                                      /*probe_pose=*/false);
            record_slam_shadow(*early);   // most LOCALIZING frames end here
            return *early;
        }

        // ===== OPTIMISATION =====
        {
            // Capture slot poses before optimisation for debug log (pose persistence check)
            if (params.debug_log_enabled)
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    const Eigen::Vector3f& a = window_mgr_.window[i].pose;
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << a[0] << ';' << a[1] << ';' << a[2];  // ';' avoids CSV column shift
                    s += ss.str();
                }
                slot_poses_pre_ = s.empty() ? "na" : s;
            }

            const auto t0 = std::chrono::high_resolution_clock::now();
            auto [last_loss, iterations] =
                  plain_slam_driving() ? std::pair<float, int>{plain_slam_.last_median_residual(),
                                                               std::max(1, plain_slam_.last_iterations())}   // the pose is plain SLAM's (above); NOT an early exit

                : run_gn_loop(selected_prior);
            last_t_adam_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            // RGB edge shadow: evaluated + logged, pose untouched. Inert unless enable && shadow,
            // and skipped entirely once the term is DRIVING (there would be nothing to compare to).
            if (params.image_edge.enable and params.image_edge_shadow)
                run_image_edge_shadow(read_window_poses(), boundary_weight_now(), lidar.second,
                                      /*probe_pose=*/not params.image_edge.drive);
            res.final_loss     = last_loss;
            res.iterations_used = iterations;

            // Capture slot poses after Adam for debug log
            if (params.debug_log_enabled)
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    const Eigen::Vector3f& a = window_mgr_.window[i].pose;
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << a[0] << ';' << a[1] << ';' << a[2];  // ';' avoids CSV column shift
                    s += ss.str();
                }
                slot_poses_post_ = s.empty() ? "na" : s;
            }
        }

        // ===== FE TERM BREAKDOWN (diagnostic log, no gradient needed) =====
        // Compute only every 5 frames to avoid re-running the full forward pass just for logging.
        // This saves ~4 ms per Adam run (≈5% of spike duration) at the cost of slightly stale
        // per-term breakdown values in the CSV.
        if (params.debug_log_enabled && (tracking_step_count_ % 5 == 0)) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            last_loss_breakdown_ = window_mgr_.compute_rfe_loss_breakdown(*model_, params);
            last_t_breakdown_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        } else {
            last_t_breakdown_ms_ = 0.f;
        }

        // ===== COVARIANCE UPDATE =====
        {
            const auto t0 = std::chrono::high_resolution_clock::now();
            auto [covariance, condition_number] = compute_posterior_covariance(scan_pts);
            if (plain_slam_driving())
            {
                covariance = plain_slam_.covariance();   // the posterior of the pose that was actually used
                current_covariance = covariance;
            }
            res.covariance = covariance;
            res.condition_number = condition_number;
            hess_pre_adaptive_ = res.covariance;   // before apply_adaptive_covariance floors it
            last_t_cov_ms_ = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        }

        // ===== WALL-SLAM: merge, derive the polygon, re-anchor once, fill the view =====
        if (estimating())
            wall_slam_after_solve(res);

        // ===== EXTRACT RESULT FROM NEWEST SLOT =====
        res.ok = true;

        {
            const Eigen::Vector3f p_acc = window_mgr_.newest().pose;
            float x = p_acc[0], y = p_acc[1], phi = p_acc[2];

            // ── DIVERGENCE RECOVERY: evidence is a cascade, never a cliff ────────────────────────
            // Corners are OPTIONAL evidence: none matched simply means the SDF term drives alone, and
            // that already happens naturally (loss_corner = 0). The failure that must never propagate
            // is the SDF optimization itself going non-finite — observed 2026-07-21: a singular Hessian
            // (cond_num hit its 1e8 sentinel) produced NaN losses, NaN was copied into the window pose
            // here, and from then on EVERY frame was NaN: the robot vanished from the canvas, the RT
            // edge published NaN to every peer, and corner detection silently reported in_fov=32 /
            // fewpoints=32 because NaN fails every comparison. Nothing recovered on its own.
            //
            // So: if the optimizer's output is not finite, drop one rung down the evidence ladder and
            // dead-reckon on odometry instead. The window slot is rewritten with the fallback so the
            // NaN cannot persist into the next frame's linearization point, and the covariance is
            // inflated because a dead-reckoned pose genuinely IS less certain — downstream consumers
            // then see the uncertainty grow rather than a confident lie.
            if (not std::isfinite(x) or not std::isfinite(y) or not std::isfinite(phi))
            {
                const bool have_odom = model_->has_prediction
                                       and model_->predicted_pos.allFinite()
                                       and std::isfinite(model_->predicted_theta);
                if (have_odom)   // rung 2: odometry dead-reckoning
                {
                    x   = model_->predicted_pos.x();
                    y   = model_->predicted_pos.y();
                    phi = model_->predicted_theta;
                }
                else if (last_good_pose_valid_)   // rung 3: hold the last pose we trusted
                {
                    x = last_good_pose_[0]; y = last_good_pose_[1]; phi = last_good_pose_[2];
                }
                else { x = 0.f; y = 0.f; phi = 0.f; }

                // Overwrite the poisoned slot so the next optimization starts from a finite point.
                window_mgr_.newest().pose = Eigen::Vector3f(x, y, phi);
                // Dead reckoning ⇒ uncertainty grows. Keep it finite and clearly worse than a fix.
                if (not current_covariance.allFinite())
                    current_covariance = Eigen::Matrix3f::Identity() * 0.5f;
                else
                    current_covariance *= 4.0f;
                res.diverged = true;
                rc::status::println("[SAFETY] SDF optimization produced a NON-FINITE pose — falling back to {} "
                             "(x={:.3f} y={:.3f} th={:.3f}); covariance inflated, corners bypassed",
                             have_odom ? "ODOMETRY" : (last_good_pose_valid_ ? "LAST GOOD POSE" : "ORIGIN"),
                             x, y, phi);
            }
            else
            {
                last_good_pose_ = Eigen::Vector3f{x, y, phi};
                last_good_pose_valid_ = true;
            }

            while (phi > M_PI) phi -= 2.0f * M_PI;
            while (phi < -M_PI) phi += 2.0f * M_PI;

            model_->robot_pos = {x, y};
            model_->robot_theta = phi;

            // Weighted by the doors: a return that came through an open doorway measured the next
            // room and must not inflate this room's reported fit. No open door ⇒ undefined weights ⇒
            // the plain median, byte-for-byte what this line produced before.
            res.sdf_mse = weighted_median_abs_sdf(model_->sdf(scan_pts),
                                                  door_point_weights(scan_pts));

            // Store localization quality so future frames can quality-gate the boundary prior
            // (Solutions B & C): when this slot becomes the oldest it carries its own sdf_mse.
            window_mgr_.newest().sdf_mse_final = res.sdf_mse;

            // ===== ONLINE MOTION MODEL ADAPTATION =====
            // Runs here so the newest slot's sdf_mse_final is valid for the quality gate.
            service_calibration();

            res.state << 2.f * model_->half_extents.x(), 2.f * model_->half_extents.y(), x, y, phi;

            // ── FLIP INSTRUMENTATION ──────────────────────────────────────────────────────────────
            // Detect a ~180° yaw jump vs the last optimized pose and dump WHY the optimizer chose it:
            //  • SDF at old vs new pose — Δ≈0 ⇒ the walls are INDIFFERENT (symmetric room), so the SDF
            //    can't prevent the flip; new≪old ⇒ the flip genuinely fits the walls better (old was wrong).
            //  • the FE terms at the settled pose — is corner/object DISAGREEING with the flip (high) but
            //    outweighed, or also indifferent (low)?  That says whether anything can break the tie.
            if (flip_prev_valid_)
            {
                const float dth = std::remainder(phi - flip_prev_th_, 2.0f * static_cast<float>(M_PI));
                if (std::abs(dth) > 2.0f)   // ~115°+ between consecutive optimized poses = a flip
                {
                    auto sdf_at = [&](float px, float py, float pth) {
                        return model_->sdf_at_pose(scan_pts, Eigen::Vector3f(px, py, pth)).cwiseAbs().mean();
                    };
                    const float sdf_old = sdf_at(flip_prev_x_, flip_prev_y_, flip_prev_th_);
                    const float sdf_new = sdf_at(x, y, phi);
                    // FRESH anchor whitened distance (σ) at old vs new pose: does the anchor SEE the flip?
                    // huge new / small old ⇒ the anchor discriminates strongly (should have vetoed it);
                    // both ~0 ⇒ the anchor isn't engaging (empty slot / wrong pin).
                    float anc_old = 0.f, anc_new = 0.f;
                    int   nanc = static_cast<int>(window_mgr_.newest().object_anchors.size());
                    for (const auto& a : window_mgr_.newest().object_anchors)
                    {
                        auto whit = [&](float px, float py, float pth) {
                            const float c = std::cos(pth), s = std::sin(pth);
                            const float dx = a.pose_world.x() - px, dy = a.pose_world.y() - py;
                            const float rx = a.obs_robot.x() - ( c * dx + s * dy);
                            const float ry = a.obs_robot.y() - (-s * dx + c * dy);
                            return std::sqrt(std::max(0.f,
                                rx * (a.information(0,0)*rx + a.information(0,1)*ry)
                              + ry * (a.information(1,0)*rx + a.information(1,1)*ry)));
                        };
                        anc_old = std::max(anc_old, whit(flip_prev_x_, flip_prev_y_, flip_prev_th_));
                        anc_new = std::max(anc_new, whit(x, y, phi));
                    }
                    // FRESH FE breakdown at the settled (flipped) pose (not the stale 5-frame cache).
                    const auto bd = window_mgr_.compute_rfe_loss_breakdown(*model_, params);
                    rc::status::print("[FLIP] dyaw={:+.0f}deg dxy=({:+.2f},{:+.2f}) | SDF old={:.3f} new={:.3f} | "
                               "anchorSD old={:.1f} new={:.1f} (n={}) | FE obs={:.2f} corner={:.2f} object={:.2f} | iters={}\n",
                               dth * 57.2958f, x - flip_prev_x_, y - flip_prev_y_, sdf_old, sdf_new,
                               anc_old, anc_new, nanc, bd.obs, bd.corner, bd.object, res.iterations_used);

                    // Also log one row per flip to a dedicated CSV (independent of DebugLog).
                    if (!flip_csv_open_attempted_)
                    {
                        flip_csv_open_attempted_ = true;
                        ::mkdir("tmp", 0755);
                        ::mkdir("tmp/sdf_localizer", 0755);
                        const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                        std::tm tm_local{}; localtime_r(&tt, &tm_local);
                        char ts_buf[32]; std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
                        flip_csv_.open(std::string("tmp/sdf_localizer/flips_") + ts_buf + ".csv",
                                       std::ios::out | std::ios::trunc);
                        if (flip_csv_.is_open())
                            flip_csv_ << "ts_ms,dyaw_deg,dx,dy,new_x,new_y,new_th,sdf_old,sdf_new,"
                                         "anchorSD_old,anchorSD_new,n_anchors,fe_obs,fe_corner,fe_object,iters\n";
                    }
                    if (flip_csv_.is_open())
                    {
                        flip_csv_ << res.timestamp_ms << ',' << dth * 57.2958f << ','
                                  << (x - flip_prev_x_) << ',' << (y - flip_prev_y_) << ','
                                  << x << ',' << y << ',' << phi << ','
                                  << sdf_old << ',' << sdf_new << ','
                                  << anc_old << ',' << anc_new << ',' << nanc << ','
                                  << bd.obs << ',' << bd.corner << ',' << bd.object << ','
                                  << res.iterations_used << '\n';
                        flip_csv_.flush();
                    }
                }
            }
            flip_prev_x_ = x; flip_prev_y_ = y; flip_prev_th_ = phi; flip_prev_valid_ = true;

            Eigen::Affine2f pose = Eigen::Affine2f::Identity();
            pose.translation() = Eigen::Vector2f{x, y};
            pose.linear() = Eigen::Rotation2Df(phi).toRotationMatrix();
            res.robot_pose = pose;
            record_slam_shadow(res);   // SHADOW: plain SLAM vs the SDF localiser (also on the early-exit path)

            if (model_->has_prediction)
            {
                res.innovation[0] = x - model_->predicted_pos.x();
                res.innovation[1] = y - model_->predicted_pos.y();
                float pred_th = model_->predicted_theta;
                res.innovation[2] = phi - pred_th;
                while (res.innovation[2] > M_PI) res.innovation[2] -= 2.0f * M_PI;
                while (res.innovation[2] < -M_PI) res.innovation[2] += 2.0f * M_PI;
                res.innovation_norm = std::sqrt(res.innovation[0]*res.innovation[0] +
                                                res.innovation[1]*res.innovation[1]);
                apply_adaptive_covariance(res);
            }
            // AFTER the adaptive floor, so `pub` is what actually leaves this agent. Logging it at
            // the covariance update instead recorded the recursion and called it the published value.
            log_hessian_check(res);
        }

        // ===== FINALIZE =====
        // Legacy boundary prior is recomputed post-optimization from the surviving oldest slot.
        // FEJ+Schur already folded the dropped slot into a frozen prior BEFORE the slide (above),
        // so skip this re-anchoring entirely when the flag is on.
        if (window_slid && window_mgr_.size() > 0 && not params.boundary_fej_schur)
            window_mgr_.recompute_boundary_prior(*model_, params);

        model_->robot_prev_pose = res.robot_pose;

        model_->prev_cov = res.covariance;

        model_->has_prediction = false;
        res.timestamp_ms = lidar.second;
        last_update_result = res;
        prev_sdf_mse_ = res.sdf_mse;   // track for boundary quality gate next frame
        // Hierarchical boundary precision (HIERARCHICAL_PRECISION.md): infer u_b_ (and the slow map_trust
        // state) from this frame's converged boundary residual, for NEXT frame's boundary_weight. Optimized
        // path only — the early-exit path skips optimization, so there is no fresh boundary evidence there.
        update_boundary_hyperprecision(res.covariance);

        // ===== DEBUG LOG =====
        if (debug_log_.is_open())
        {
            const auto wall_now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            const float vel_adv_y = velocity_history.empty() ? 0.f : velocity_history.back().adv_y;
            const float vel_rot   = velocity_history.empty() ? 0.f : velocity_history.back().rot;
            const float odom_adv  = odometry_history.empty() ? 0.f : odometry_history.back().adv;
            const float odom_rot  = odometry_history.empty() ? 0.f : odometry_history.back().rot;

            const float cmd_cov_xx = last_cmd_cov_(0,0);
            const float cmd_cov_tt = last_cmd_cov_(2,2);
            float meas_cov_xx = 0.f, meas_cov_tt = 0.f;
            if (last_measured_prior_.valid)
            {
                meas_cov_xx = last_measured_prior_.covariance_eigen(0, 0);
                meas_cov_tt = last_measured_prior_.covariance_eigen(2, 2);
            }
            const float sel_cov_xx = last_selected_prior_.covariance_eigen(0, 0);
            const float sel_cov_tt = last_selected_prior_.covariance_eigen(2, 2);

            std::string losses_str;
            for (size_t ai = 0; ai < last_adam_losses_.size(); ++ai)
            {
                if (ai > 0) losses_str += '|';
                std::ostringstream ss; ss << last_adam_losses_[ai];
                losses_str += ss.str();
            }
            if (losses_str.empty()) losses_str = "na";

            const float lr_eff = params.learning_rate_pos /
                std::sqrt(static_cast<float>(std::max(1, (int)window_mgr_.size())));
            const auto motion_ingress = get_motion_ingress_debug();

            debug_log_
                << lidar.second
                << ',' << wall_now_ms
                << ',' << selected_prior.dt
                << ',' << sampled_points.size()
                << ',' << vel_adv_y
                << ',' << vel_rot
                << ',' << odom_adv
                << ',' << odom_rot
                << ',' << motion_ingress.command_source
                << ',' << motion_ingress.command_adv_raw
                << ',' << motion_ingress.command_adv_normalized
                << ',' << motion_ingress.command_rot_raw
                << ',' << motion_ingress.command_rot_normalized
                << ',' << motion_ingress.command_ts_ms
                << ',' << motion_ingress.odom_source
                << ',' << motion_ingress.odom_adv_raw
                << ',' << motion_ingress.odom_adv_normalized
                << ',' << motion_ingress.odom_rot_raw
                << ',' << motion_ingress.odom_rot_normalized
                << ',' << motion_ingress.odom_ts_ms
                << ',' << (int)motion_prior_selection.command_prior.valid
                << ',' << (int)motion_prior_selection.command_prior.fresh
                << ',' << motion_prior_selection.command_prior.delta_pose[0]
                << ',' << motion_prior_selection.command_prior.delta_pose[1]
                << ',' << motion_prior_selection.command_prior.delta_pose[2]
                << ',' << cmd_cov_xx
                << ',' << cmd_cov_tt
                << ',' << (int)last_measured_prior_.valid
                << ',' << (int)last_measured_prior_.fresh
                << ',' << last_measured_prior_.delta_pose[0]
                << ',' << last_measured_prior_.delta_pose[1]
                << ',' << last_measured_prior_.delta_pose[2]
                << ',' << meas_cov_xx
                << ',' << meas_cov_tt
                << ',' << (int)last_selected_prior_.valid
                << ',' << (int)last_selected_prior_.fresh
                << ',' << motion_prior_source_name(last_motion_prior_source_)
                << ',' << last_selected_prior_.delta_pose[0]
                << ',' << last_selected_prior_.delta_pose[1]
                << ',' << last_selected_prior_.delta_pose[2]
                << ',' << sel_cov_xx
                << ',' << sel_cov_tt
                << ',' << pred_pos.x()
                << ',' << pred_pos.y()
                << ',' << pred_theta
                << ',' << slot_motion_cov(0,0)
                << ',' << slot_motion_cov(2,2)
                << ',' << 0                          // early_exit = 0 (Adam ran)
                << ',' << res.iterations_used
                << ',' << res.final_loss
                << ',' << lr_eff
                << ',' << res.state[2]
                << ',' << res.state[3]
                << ',' << res.state[4]
                << ',' << res.innovation[0]
                << ',' << res.innovation[1]
                << ',' << res.innovation[2]
                << ',' << res.innovation_norm
                << ',' << res.sdf_mse
                << ',' << res.covariance(0,0)
                << ',' << res.covariance(2,2)
                << ',' << res.condition_number
                << ',' << (int)window_mgr_.size()
                << ',' << tracking_step_count_
                << ',' << last_loss_breakdown_.boundary
                << ',' << last_loss_breakdown_.obs
                << ',' << last_loss_breakdown_.motion
                << ',' << last_loss_breakdown_.corner
                << ',' << last_loss_breakdown_.object
                << ',' << last_loss_init_
                << ',' << std::chrono::duration<float, std::milli>(
                               std::chrono::high_resolution_clock::now() - t_update_start_).count()
                << ',' << last_t_adam_ms_
                << ',' << last_t_cov_ms_
                << ',' << last_t_breakdown_ms_;

            // Per-slot pose persistence columns
            debug_log_ << ',' << slot_poses_pre_
                       << ',' << slot_poses_post_;

            // Per-slot sdf_mse_final (built here so newest slot's value is already set)
            {
                std::string s;
                for (size_t i = 0; i < window_mgr_.window.size(); ++i)
                {
                    if (i > 0) s += '|';
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(4)
                       << window_mgr_.window[i].sdf_mse_final;
                    s += ss.str();
                }
                debug_log_ << ',' << (s.empty() ? "na" : s);
            }

            // Boundary prior anchor
            {
                const auto& bp = window_mgr_.boundary_prior;
                debug_log_ << ',' << (int)bp.valid
                           << ',' << bp.mu[0]
                           << ',' << bp.mu[1]
                           << ',' << bp.mu[2];
            }

            debug_log_ << ',' << last_lbfgs_grad_norm_
                       << ',' << losses_str
                       << ",," 
                       << ',' << learned_odom_bias_.x()
                       << ',' << learned_odom_bias_.y()
                       << ',' << learned_odom_bias_.z();
            // last_early_exit_metric_, NOT res.early_exit_metric: res only receives it AFTER this
            // block (the assignment below), so this column logged the value from before the gate ran
            // — uniformly stale — and could never show the predicted-pose SDF that TRIGGERED this
            // optimization. That number is the margin by which the prediction missed the early-exit
            // threshold, i.e. the only direct measure of how close a rotating frame came to skipping
            // Adam. The early-exit path at the bottom of try_prediction_early_exit was already
            // correct (it assigns res.early_exit_metric before logging).
            debug_log_ << ',' << last_early_exit_metric_
                       << ',' << lost_logodds_
                       << ',' << rivals_.size()
                       << ',' << (grid_search_active_.load(std::memory_order_relaxed) ? 1 : 0);
            write_debug_tail();

            debug_log_ << '\n';
            debug_log_.flush();
        }

        // Adam path: expose the predicted-pose SDF that triggered this optimization (NaN if the
        // early-exit gate never evaluated it this frame — e.g. warmup / no odometry).
        res.early_exit_metric = last_early_exit_metric_;
        res.pred_sdf_median   = last_pred_sdf_median_;
        // While plain SLAM owns the pose, the "pred (SDF)" readout measured the prediction against the OLD SDF room
        // model, which does not follow the map layout — a large number about nothing. Show the SLAM's own fit instead:
        // the median distance of this scan's returns to the mapped surface at the solved pose.
        if (plain_slam_driving()) res.pred_sdf_median = plain_slam_.last_median_residual();
        res.pred_x = last_pred_pos_.x();
        res.slot_appended = params.window_stride_enabled and not rec_stride_replace_;
        res.pred_y = last_pred_pos_.y();
        res.pred_theta = last_pred_theta_;
        res.dx_local = cyc_dx_local_;
        res.dy_local = cyc_dy_local_;
        res.imu_dvx = cyc_imu_dvx_; res.imu_dvy = cyc_imu_dvy_;
        res.imu_dpx = cyc_imu_dpx_; res.imu_dpy = cyc_imu_dpy_;
        res.wheel_dvx = cyc_wheel_dvx_; res.wheel_dvy = cyc_wheel_dvy_;
        res.imu_lin_segs = cyc_imu_lin_segs_;
        res.imu_dtheta          = cyc_imu_dtheta_;
        res.wheel_dtheta        = cyc_wheel_dtheta_;
        res.heading_cov         = cyc_heading_cov_;
        res.gyro_weight         = cyc_heading_dt_ > 0.0
                                ? static_cast<float>(cyc_gyro_w_dt_ / cyc_heading_dt_) : -1.f;
        res.heading_diag        = cyc_hdiag_;
        res.heading_diag.nl_c0 = static_cast<float>(wheel_noise_.coeffs()[0]);
        res.heading_diag.nl_c1 = static_cast<float>(wheel_noise_.coeffs()[1]);
        res.heading_diag.nl_c2 = static_cast<float>(wheel_noise_.coeffs()[2]);
        res.heading_diag.nl_samples = wheel_noise_.samples();
        res.heading_diag.rest_gain_tr = params.zupt_on_prediction ? zupt_pred_gain_tr_ : -1.f;
        res.heading_diag.rest_gain_ro = params.zupt_on_prediction ? zupt_pred_gain_ro_ : -1.f;
        res.heading_diag.rest_dens_v  = rest_fwd_.density() * std::sqrt(2.f);   // as a |translation| density
        res.heading_diag.rest_dens_w  = rest_rot_.density();
        if (params.heading_fusion and cyc_hdiag_.total_dt > 0.f)
            res.heading_diag.dens_w = static_cast<float>(cyc_dens_w_dt_ / cyc_hdiag_.total_dt);
        if (params.heading_fusion and cyc_hdiag_.gyro_dt > 0.f)
            res.heading_diag.dens_g = static_cast<float>(cyc_dens_g_dt_ / cyc_hdiag_.gyro_dt);
        res.wheel_shadow_dtheta = cyc_wheel_shadow_dtheta_;
        res.imu_segs            = cyc_imu_segs_;
        res.wheel_segs          = cyc_wheel_segs_;
        // MUST come after the fields above: it reads dy_local/dx_local/imu_dtheta as the covariates
        // H. Called earlier it sees zeros, H -> 0, and the learner silently never learns anything.
        feed_motion_calibrator(res);
        score_surprise(res, true);   // after the adaptive floor: KL vs the PUBLISHED posterior; per-axis + learner vs the solver's
        // The scan-only pose re-taken at the CONVERGED pose (the step from there is ~0, so z is the
        // solver's scan-alone answer), for the noise learner (motion_noise_innov.h).
        if (params.motion_noise_innov and map_guided_checks_allowed() and not searching())
        {
            const float est_th = std::atan2(res.robot_pose.linear()(1, 0), res.robot_pose.linear()(0, 0));
            scan_only_pose(window_mgr_.newest().lidar_points, res.robot_pose.translation().x(),
                           res.robot_pose.translation().y(), est_th);
        }
        observe_innovation(res);
        return res;
    }

} // namespace rc
