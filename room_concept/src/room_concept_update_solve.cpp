// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// One localiser cycle, second half: early exit, solve, result extraction, logs (RoomConcept::update_*).

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

    std::optional<RoomConcept::UpdateResult> RoomConcept::update_early_exit(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const ScanPoints& scan_pts = c.sampled_points;
        const auto& slot_odom_delta = c.slot_odom_delta;
        const auto& selected_prior = c.selected_prior();
        auto& res = c.res;
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
            return early;
        }
        return std::nullopt;
    }

    void RoomConcept::update_solve(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const ScanPoints& scan_pts = c.sampled_points;
        const auto& selected_prior = c.selected_prior();
        auto& res = c.res;
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
    }

    void RoomConcept::update_extract(UpdateCycle& c)
    {
        const ScanPoints& scan_pts = c.sampled_points;
        auto& res = c.res;
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
    }

    void RoomConcept::update_finalize(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const bool window_slid = c.window_slid;
        auto& res = c.res;
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
    }

    void RoomConcept::update_write_debug_row(UpdateCycle& c)
    {
        const auto& lidar = c.lidar;
        const auto& velocity_history = c.velocity_history;
        const auto& odometry_history = c.odometry_history;
        const auto& motion_prior_selection = c.motion_prior_selection;
        const auto& selected_prior = c.selected_prior();
        const auto& sampled_points = c.sampled_points;
        const auto& pred_pos = c.pred_pos;
        const auto pred_theta = c.pred_theta;
        const auto& slot_motion_cov = c.slot_motion_cov;
        auto& res = c.res;
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
    }

    void RoomConcept::update_publish_diagnostics(UpdateCycle& c)
    {
        auto& res = c.res;
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
    }

} // namespace rc
