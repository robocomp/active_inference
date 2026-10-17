// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// One localiser cycle: RoomConcept::update.

#include "room_concept.h"
#include "update_cycle.h"
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
        UpdateCycle c{lidar, velocity_history, odometry_history};
        auto& res = c.res;
        if (lidar.first.empty() || model_ == nullptr)
            return std::move(res);

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
            return std::move(res);
        }

        // The phases below were one 1014-line body until refactor step 2b (2026-10-17); each is the verbatim
        // block that sat under its `// =====` heading, and UpdateCycle carries the locals they share.
        update_predict(c);
        update_build_slot(c);
        update_carry_doors();
        update_observe_walls(c);
        update_observe_corners(c);
        update_open_cycle(c);
        if (auto early = update_early_exit(c))
            return *early;
        update_solve(c);
        update_extract(c);
        update_finalize(c);
        update_write_debug_row(c);
        update_publish_diagnostics(c);
        return std::move(res);
    }

    void RoomConcept::update_predict(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const auto& velocity_history = c.velocity_history;
        const auto& odometry_history = c.odometry_history;
        // ===== ORIENTATION SEARCH ON FIRST UPDATE =====
        if (needs_orientation_search_)
        {
            const auto state = model_->get_state();
            const float best_phi = find_best_initial_orientation(lidar.first, state[2], state[3], state[4]);
            model_->robot_theta = best_phi;
            needs_orientation_search_ = false;
        }

        // ===== PREDICTION =====
        c.motion_prior_selection = build_motion_prior_selection(velocity_history, odometry_history, lidar);
        const auto& motion_prior_selection = c.motion_prior_selection;
        const auto &selected_prior = motion_prior_selection.selected_prior;
        const PredictionState prediction = predict_step(model_, selected_prior, true);

        if (prediction.have_propagated)
            current_covariance = prediction.propagated_cov;

        const auto pred_pos = c.pred_pos = motion_prior_selection.predicted_pos;
        const auto pred_theta = c.pred_theta = motion_prior_selection.predicted_theta;
        last_pred_pos_ = pred_pos;      // kept for UpdateResult on BOTH return paths
        last_pred_theta_ = pred_theta;
    }

    void RoomConcept::update_build_slot(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const auto& selected_prior = c.selected_prior();
        const auto& pred_pos = c.pred_pos;
        const auto pred_theta = c.pred_theta;
        // ===== BUILD NEW WINDOW SLOT =====
        const auto& all_points = lidar.first;
        auto& sampled_points = c.sampled_points;
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
        auto& slot_odom_delta = c.slot_odom_delta;
        slot_odom_delta = Eigen::Vector3f::Zero();
        auto& slot_motion_cov = c.slot_motion_cov;
        slot_motion_cov = Eigen::Matrix3f::Identity() * params.default_slot_motion_cov;
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
        bool& window_slid = c.window_slid;
        window_slid = false;
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
    }

    void RoomConcept::update_carry_doors()
    {
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
                r.n_out = R_inv * r.n_out;   // a direction: rotated, not translated
                doors_robot_.push_back(std::move(r));
            }
        }
        window_mgr_.doors_newest_robot = &doors_robot_;
    }

    void RoomConcept::update_observe_walls(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const auto& sampled_points = c.sampled_points;
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
    }

    void RoomConcept::update_observe_corners(UpdateCycle& c)
    {
        const auto& all_points = c.lidar.first;
        auto& res = c.res;
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
    }

    void RoomConcept::update_open_cycle(UpdateCycle& c)
    {
        const auto& motion_prior_selection = c.motion_prior_selection;
        const auto& selected_prior = c.selected_prior();
        const auto pred_theta = c.pred_theta;
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
    }

} // namespace rc
