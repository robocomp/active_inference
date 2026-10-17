// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Diagnostic logs and CSV sinks: the debug/recovery logs, Hessian check, corner stats, timing.

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

    void RoomConcept::init_debug_log()
    {
        if (!params.debug_log_enabled) return;

        ::mkdir("tmp", 0755);
        ::mkdir("tmp/sdf_localizer", 0755);

        // Build timestamped filename: log_YYYY-MM-DD_HH-MM-SS.csv
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(now);
            std::tm tm_local{};
            localtime_r(&tt, &tm_local);
            char ts_buf[32];
            std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
            debug_log_path_ = std::string("tmp/sdf_localizer/log_") + ts_buf + ".csv";
        }
        debug_log_.open(debug_log_path_, std::ios::out | std::ios::trunc);
        if (!debug_log_.is_open())
        {
            qWarning() << "Debug log could not be opened:" << QString::fromStdString(debug_log_path_);
            return;
        }

        // CSV header — one row per update() call
        debug_log_
            << "ts_ms"
            << ",wall_ms"
            << ",dt_ms"
            << ",n_lidar"
            << ",vel_adv_y"
            << ",vel_rot"
            << ",odom_adv"
            << ",odom_rot"
            << ",cmd_ingress_source,cmd_adv_raw,cmd_adv_norm,cmd_rot_raw,cmd_rot_norm,cmd_ingress_ts"
            << ",odom_ingress_source,odom_adv_raw,odom_adv_norm,odom_rot_raw,odom_rot_norm,odom_ingress_ts"
            << ",cmd_valid"
            << ",cmd_fresh"
            << ",cmd_dx,cmd_dy,cmd_dth"
            << ",cmd_cov_xx,cmd_cov_tt"
            << ",meas_valid"
            << ",meas_fresh"
            << ",meas_dx,meas_dy,meas_dth"
            << ",meas_cov_xx,meas_cov_tt"
            << ",sel_valid,sel_fresh,sel_source"
            << ",sel_dx,sel_dy,sel_dth"
            << ",sel_cov_xx,sel_cov_tt"
            << ",pred_x,pred_y,pred_theta"
            << ",slot_mcov_xx,slot_mcov_tt"
            << ",early_exit"
            << ",iters"
            << ",final_loss"
            << ",lr_eff"
            << ",res_x,res_y,res_theta"
            << ",innov_x,innov_y,innov_theta"
            << ",innov_norm"
            << ",sdf_mse"
            << ",cov_xx,cov_tt"
            << ",cond_num"
            << ",window_size"
            << ",tracking_steps"
            << ",loss_boundary"
            << ",loss_obs"
            << ",loss_motion"
            << ",loss_corner"
            << ",loss_object"
            << ",loss_init"
            << ",t_update_ms"
            << ",t_adam_ms"
            << ",t_cov_ms"
            << ",t_breakdown_ms"
            // ---- appended for loss/recovery analysis; APPEND ONLY, never insert ----
            // early_exit_metric was absent and had to be proxied through final_loss, which conflates
            // two different quantities across the two paths. recovery_bad/cooldown make the trigger's
            // internal state visible so an episode can be read forwards instead of inferred backwards
            // from window_size resets. search_active marks the frames a search was live on.
            //
            // ORDER FIXED 08-09: this group used to be declared HERE, before slot_poses_*, while both
            // writers emit it LAST. Every column from slot_poses_pre onward was therefore mislabelled
            // (slot-pose strings landed in the column called early_exit_metric). Columns 0..73, up to
            // t_breakdown_ms, were always correct. The header is what moved rather than the writers,
            // deliberately: the row layout below is what every log already on disk was written with,
            // so a reader keyed on this corrected header now reads the old files correctly too.
            << ",slot_poses_pre"
            << ",slot_poses_post"
            << ",slot_sdf_mse"
            << ",bp_valid,bp_x,bp_y,bp_theta"
            << ",lbfgs_grad_norm"
            << ",loss_curve"
            << ",ml_slip_k"
            << ",ml_odom_noise_trans"
            << ",ml_bias_x,ml_bias_y,ml_bias_theta"
            << ",early_exit_metric"
            // recovery_bad/recovery_cooldown RENAMED 09-16 (same positions): the RecoveryManager is gone.
            << ",lost_logodds,n_rivals,search_active"
            // ★ APPENDED 08-16, AT THE END, ON BOTH WRITERS TOGETHER. This file's history is the
            // reason for that discipline: a group once declared mid-header while both writers emitted
            // it LAST mislabelled every column from slot_poses_pre onward. New columns go at the tail
            // and nowhere else, and every one below is written on BOTH paths — a column that exists on
            // only one is worse than no column, because it silently restricts the analysis to the
            // frames that happened to take that path (which is how n_lidar came to read 0 on 98% of
            // rows). Old logs simply lack these fields; a reader keyed on the header handles that.
            //
            // preint_*   how many odometry samples the motion factor summarised and over how long —
            //            the direct witness of whether striding is chaining intervals as intended.
            // slot_mcov_ off-diagonals of the motion covariance. The whole argument for propagating it
            //   xy/xt/yt is that a heading error rotates subsequent translation; these are the terms
            //            that carries, and until now only the diagonal was logged, so the change's
            //            central claim was the one thing the log could not show.
            << ",preint_n,preint_T"
            << ",belief_age_s,belief_decay,aff_outcome,aff_completions,tgt_x,tgt_y,pub_tx,pub_ty,pub_ok"
            << ",slot_mcov_xy,slot_mcov_xt,slot_mcov_yt"
            // imu_cover fraction of the interval's segments whose heading came from the GYRO rather
            //            than the wheels. -1 when no segments were integrated.
            << ",imu_cover"
            // pub_cov_*  what the RT edge actually carried, AFTER the worker's kinematic clamp folded
            //            its un-applied residual in. cov_xx above is the PRE-clamp value, so these two
            //            differing is the clamp's contribution made visible instead of inferred.
            // clamp_hit  1 when the clamp fired on this frame, so its rate is measured not asserted.
            // wg_ratio   wheel/gyro heading ratio over IMU-covered segments; NaN until enough rotation
            //            has accumulated for the quotient to mean anything.
            << ",pub_cov_xx,pub_cov_tt,clamp_hit,wg_ratio"
            << "\n";
        debug_log_.flush();
        qInfo() << "Debug log writing to" << QString::fromStdString(debug_log_path_);
    }

    void RoomConcept::init_recovery_log()
    {
        // Loss/recovery episode log — one row per search. ALWAYS ON, deliberately NOT behind DebugLog: it
        // is low-volume (66 rows in a 708k-frame run) and it is the only record of what recovery did.
        // ★It used to be opened INSIDE init_debug_log(), after that function's `if (!debug_log_enabled)
        //  return;` — so with DebugLog=false (the committed default) it never opened, while this comment
        //  claimed the opposite. No episode was recorded from 2026-08-29 to 2026-09-16, the exact window
        //  in which relocalisation was reported as flipping the pose.
        ::mkdir("tmp", 0755);
        ::mkdir("tmp/sdf_localizer", 0755);
        const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm tm_local{};
        localtime_r(&tt, &tm_local);
        char ts_buf[32];
        std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d_%H-%M-%S", &tm_local);
        recovery_log_path_ = std::string("tmp/sdf_localizer/recovery_") + ts_buf + ".csv";
        recovery_log_.open(recovery_log_path_, std::ios::out | std::ios::trunc);
        if (not recovery_log_.is_open())
        {
            qWarning() << "Recovery episode log could not be opened:" << QString::fromStdString(recovery_log_path_);
            return;
        }
        recovery_log_.imbue(std::locale::classic());
        recovery_log_ << "ts_ms,trigger,n_lidar,"
                         "inc_x,inc_y,inc_theta,inc_loss,good_thr,"
                         "stage,best_x,best_y,best_theta,best_loss,"
                         "topk_best,topk_worst,ess,cov_xx,cov_yy,cov_tt,"
                         "n_evals,jump_m,jump_rad,success,duration_ms,beta,n_points,"
                         "n_modes,mode_entropy,top_weight,second_weight,n_yaws,n_segments,p_lost\n";
        recovery_log_.flush();
        qInfo() << "Recovery episode log writing to" << QString::fromStdString(recovery_log_path_);
    }

    // Fold one detection frame into the per-model-corner tallies. Cheap (a handful of map lookups per
    // frame); the expensive part is the rewrite below, which is throttled.
    void RoomConcept::accumulate_corner_stats(const CornerDetector::DetectionResult& det)
    {
        for (const int idx : det.in_fov_indices)
            corner_vertex_stats_[idx].in_fov++;
        for (const int idx : det.occluded_indices)
            corner_vertex_stats_[idx].occluded++;

        for (const auto& m : det.matches)
        {
            auto& s = corner_vertex_stats_[m.model_index];
            s.accepted++;
            s.last_yield  = m.yield;
            s.retired_now = m.suppressed;
            if (m.suppressed) s.suppressed++;
            s.assoc_prob_sum += m.assoc_prob;
            s.resid_sum      += (m.detected - m.predicted).norm();
            // Λ_det is 2×2 symmetric and ALREADY scaled by assoc_prob — i.e. exactly the precision the
            // loss sees. Its smallest eigenvalue is the number that matters: near zero means the two
            // wall fits were near-parallel and the corner constrains nothing along the bisector, which
            // is the pillar failure mode described at detect()'s per-corner band clamp.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
            if (es.info() == Eigen::Success)
            {
                s.lambda_min_sum += es.eigenvalues()(0);
                s.lambda_max_sum += es.eigenvalues()(1);
            }
            // A runner-up only exists when another model corner was inside the gate; the sentinel
            // means "no rival", and averaging it in is what made the old rival statistic meaningless.
            if (m.runnerup_chi2 < 1e8f)
            {
                s.rival_n++;
                s.runnerup_chi2_sum += m.runnerup_chi2;
            }
        }

        if (++corner_stats_frames_ % kCornerStatsRewritePeriod == 0)
            write_corner_stats_csv();
    }

    // Rewrite etc/corner_stats.csv from scratch — one row per model corner, current totals. Truncating
    // rather than appending keeps it directly readable (sort by accept_rate or lambda_min) without an
    // aggregation step. Columns chosen to answer one question per pair:
    //   accept_rate    — does this corner ever pay off, or is it pure cost?
    //   assoc_prob     — is it aliasing with a neighbour? (≪1 ⇒ the PDA is already muting it)
    //   runnerup_chi2  — how close was the rival? (near assoc χ² ⇒ coin flip ⇒ pose jumps)
    //   lambda_min     — does it constrain anything, or only one direction? (→0 ⇒ no information)
    //   resid_mean     — is the map honest here?
    void RoomConcept::write_corner_stats_csv()
    {
        // Reopen with trunc rather than rewinding: field widths vary between rewrites (a shrinking
        // average writes fewer characters), so seeking to 0 would leave the tail of the previous,
        // longer row behind. Throttled to every kCornerStatsRewritePeriod frames, so this is free.
        corner_stats_csv_.close();
        corner_stats_csv_.open("etc/corner_stats.csv", std::ios::out | std::ios::trunc);
        if (!corner_stats_csv_.is_open())
            return;

        corner_stats_csv_ << "# frames=" << corner_stats_frames_
                          << " wall_in/wall_out are the adjacent polygon edge lengths (m)\n"
                          << "vertex,wall_in,wall_out,in_fov,occluded,accepted,accept_rate,"
                             "assoc_prob,rival_n,runnerup_chi2,lambda_min,lambda_max,resid_mean,"
                             "yield,retired_now,suppressed,suppressed_frac\n";

        const auto& poly = init_polygon_vertices_;
        const int N = static_cast<int>(poly.size());
        for (const auto& [idx, s] : corner_vertex_stats_)
        {
            float w_in = 0.f, w_out = 0.f;
            if (N >= 3 && idx >= 0 && idx < N)
            {
                w_in  = (poly[idx] - poly[(idx + N - 1) % N]).norm();
                w_out = (poly[(idx + 1) % N] - poly[idx]).norm();
            }
            const double acc  = s.accepted > 0 ? static_cast<double>(s.accepted) : 0.0;
            const double rate = s.in_fov > 0 ? acc / s.in_fov : 0.0;
            const auto avg = [&](double sum) { return s.accepted > 0 ? sum / acc : 0.0; };
            corner_stats_csv_ << idx << ',' << w_in << ',' << w_out << ','
                              << s.in_fov << ',' << s.occluded << ',' << s.accepted << ',' << rate << ','
                              << avg(s.assoc_prob_sum) << ',' << s.rival_n << ','
                              << (s.rival_n > 0 ? s.runnerup_chi2_sum / s.rival_n : 0.0) << ','
                              << avg(s.lambda_min_sum) << ',' << avg(s.lambda_max_sum) << ','
                              << avg(s.resid_sum) << ',' << s.last_yield << ','
                              << (s.retired_now ? 1 : 0) << ',' << s.suppressed << ','
                              << (s.accepted > 0 ? s.suppressed / acc : 0.0) << '\n';
        }
        corner_stats_csv_.flush();
    }

    // Append one row to etc/hier_prec.csv (lazy-opened, truncate). Loc-thread only, no lock — same
    // discipline as opt_csv_. src labels the origin: "opt" (optimized boundary update), "ee"
    // (early-exit rotation surrogate), "reloc" (a map-trust relocalization just fired).
    void RoomConcept::log_hier_prec_row(const char *src, float r_b, float quad, float trace, bool reloc_fired)
    {
        if (!hier_prec_csv_open_attempted_)
        {
            hier_prec_csv_open_attempted_ = true;
            hier_prec_csv_.open("etc/hier_prec.csv", std::ios::out | std::ios::trunc);
            if (hier_prec_csv_.is_open())
                hier_prec_csv_ << "ts_ms,src,r_b,quad,trace,u_b,boundary_weight,map_trust_v,window_size,reloc_fired\n";
        }
        if (hier_prec_csv_.is_open())
        {
            hier_prec_csv_ << last_update_result.timestamp_ms << ',' << src << ',' << r_b << ',' << quad << ','
                           << trace << ',' << u_b_ << ',' << std::exp(u_b_) << ','
                           << map_trust_v_ << ',' << window_mgr_.size() << ',' << (reloc_fired ? 1 : 0) << '\n';
            hier_prec_csv_.flush();
        }
    }

    // =========================================================================
    //  Levenberg-Marquardt backend (analytic Jacobians — see room_gn_solver.h)
    // =========================================================================
// ── The published sigma, measured against what the window's own equations say ────────────────────
// Four precisions for the same newest pose:
//
//   marg  = H_nn − H_no H_oo⁻¹ H_on   the window's marginal. What the solve actually knows.
//   block = H_nn                      the same without the complement, i.e. pretending every other
//                                     window pose is known EXACTLY. Always the more confident.
//   rec   = current_covariance⁻¹      the filter's own posterior, before the adaptive floor.
//   pub   = res.covariance⁻¹          what leaves this agent, after apply_adaptive_covariance.
//
// ★ MEASURED 2026-08-29, first 79 optimised frames, and it corrected the guess that prompted it.
//   The guess was that sigma would prove OVER-confident because the filter has no prediction step.
//   It has one: predict_step() writes propagated_cov into current_covariance on every optimised
//   cycle (the polish-gated growth further down is the EARLY-EXIT path's, a different branch). And
//   the reading came out the other way — sigma_theta published 0.042 rad against the window's own
//   0.0034, i.e. 12x LOOSER, with x and y 7-8x. The filter is conservative, not over-confident, and
//   the ratio does NOT drift with the gap since the last solve (corr +0.05 over log gap), which is
//   what a working prediction step looks like.
// ★ block/marg came out at 0.79 median (0.51 at p10, on short windows): dropping the cross-terms
//   would claim ~21% tighter than the window supports, and up to 2x. That is the answer to "why
//   Schur-complement at all" in numbers.
// ★ WHAT IS STILL OPEN: whether 7-12x is the RIGHT amount of conservatism. It is set by the process
//   noise in propagated_cov, and sigma_pub varies far less than sigma_marg does (CV 0.16 vs 0.33 in
//   x), so the published number is carrying much more motion prior than fit quality. That matters
//   downstream — the speed governor and every concept agent's precision read it.
void RoomConcept::log_hessian_check(const UpdateResult& res)
{
    if (not params.hessian_check or not last_marg_ok_) return;
    const auto sig = [](const Eigen::Matrix3f& prec, int i) -> double
    {
        const Eigen::Matrix3f c = prec.inverse();
        return (c.allFinite() and c(i, i) > 0.f) ? std::sqrt(static_cast<double>(c(i, i))) : -1.0;
    };
    const auto sd = [](const Eigen::Matrix3f& c, int i) -> double
    {
        return (c.allFinite() and c(i, i) > 0.f) ? std::sqrt(static_cast<double>(c(i, i))) : -1.0;
    };
    static bool header = false;
    std::ofstream f("etc/hessian_check.csv", header ? std::ios::app : std::ios::trunc);
    if (not f) return;
    f.imbue(std::locale::classic());          // es_ES would write commas; see CLAUDE.md
    if (not header)
    {
        header = true;
        f << "t_ms,slots,dof_marginalised,"
             "sx_marg,sy_marg,st_marg,sx_block,sy_block,st_block,"
             "sx_rec,sy_rec,st_rec,sx_pub,sy_pub,st_pub,"
             "sx_pred,sy_pred,st_pred,sx_obs,sy_obs,st_obs,"
             "innov_x,innov_y,innov_t,nis,cov_accept,cov_det,cov_cond,"
             "ratio_x,ratio_y,ratio_t,block_over_marg_t,ms_since_last_opt\n";
    }
    const double sm[3] = {sig(last_marg_prec_, 0),  sig(last_marg_prec_, 1),  sig(last_marg_prec_, 2)};
    const double sb[3] = {sig(last_block_prec_, 0), sig(last_block_prec_, 1), sig(last_block_prec_, 2)};
    const double sr[3] = {sd(hess_pre_adaptive_, 0), sd(hess_pre_adaptive_, 1), sd(hess_pre_adaptive_, 2)};
    const double sp[3] = {sd(res.covariance, 0),     sd(res.covariance, 1),     sd(res.covariance, 2)};
    const auto ratio = [](double a, double b) { return (a > 0.0 and b > 0.0) ? a / b : -1.0; };
    f << std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch()).count()
      << ',' << last_gn_window_slots_ << ',' << last_marg_dof_;
    for (int i = 0; i < 3; ++i) f << ',' << sm[i];
    for (int i = 0; i < 3; ++i) f << ',' << sb[i];
    for (int i = 0; i < 3; ++i) f << ',' << sr[i];
    for (int i = 0; i < 3; ++i) f << ',' << sp[i];
    for (int i = 0; i < 3; ++i) f << ',' << sd(hess_pred_cov_, i);
    // The single slot's observation on its own, so "Q too big" and "H too small" stay separable.
    const Eigen::Matrix3f obs_cov = hess_obs_prec_.inverse();
    for (int i = 0; i < 3; ++i) f << ',' << sd(obs_cov, i);
    // ── NIS: the only reading that can SIZE the process noise without ground truth ───────────────
    // The innovation is the fit's correction to the prediction. Under a correct P_pred it is drawn
    // from it, so nu' P_pred^-1 nu has expectation 3 (its DOF). Below 3 ⇒ P_pred is bigger than the
    // corrections it is meant to cover, i.e. the process noise is too large, BY THAT FACTOR. This is
    // the number to set OdomNoiseScale from, and it is a measurement rather than a taste.
    // ⚠ Ignoring the observation noise makes this an UNDER-estimate of NIS, so it errs toward
    //   saying Q is fine — the conservative direction for a knob that would loosen safety margins.
    double nis = -1.0;
    if (hess_pred_cov_.allFinite() and res.innovation.allFinite())
    {
        const Eigen::Matrix3f P = hess_pred_cov_;
        if (std::abs(P.determinant()) > 1e-18f)
            nis = static_cast<double>(res.innovation.transpose() * P.inverse() * res.innovation);
    }
    for (int i = 0; i < 3; ++i) f << ',' << res.innovation[i];
    f << ',' << nis << ',' << hess_cov_accept_ << ',' << hess_cov_det_ << ',' << hess_cov_cond_;
    // marg / pub > 1 means the window is LESS certain than what we publish — the over-confidence.
    for (int i = 0; i < 3; ++i) f << ',' << ratio(sm[i], sp[i]);
    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count();
    f << ',' << ratio(sb[2], sm[2]) << ','
      << (hess_prev_opt_ms_ == 0 ? -1 : now_ms - hess_prev_opt_ms_) << '\n';
    hess_prev_opt_ms_ = now_ms;
    ++hess_check_rows_;

    if (now_ms - hess_check_last_log_ms_ > 5000)
    {
        hess_check_last_log_ms_ = now_ms;
        qInfo().nospace()
            << "[hess] sigma_theta  window(marg) " << sm[2] << "  window(block) " << sb[2]
            << "  filter " << sr[2] << "  published " << sp[2] << "  | marg/pub " << ratio(sm[2], sp[2])
            << " block/marg " << ratio(sb[2], sm[2])
            << " | NIS " << nis
            << " | cov update " << (hess_cov_accept_ == 1 ? "TAKEN"
                                  : hess_cov_accept_ == -2 ? "REFUSED: not positive definite"
                                  : hess_cov_accept_ == -3 ? "REFUSED: cond above condition_number_max"
                                  : hess_cov_accept_ == -1 ? "REFUSED: non-finite" : "?")
            << " det " << hess_cov_det_
            << " | slots " << last_gn_window_slots_ << " dof_marg " << last_marg_dof_
            << " rows " << hess_check_rows_;
    }
}

    /// One line every 5 s naming which of the three silences we are in.
    void RoomConcept::imgedge_health(std::int64_t timestamp_ms)
    {
        if (imgedge_health_last_ms_ == 0) { imgedge_health_last_ms_ = timestamp_ms; return; }
        if (timestamp_ms - imgedge_health_last_ms_ < 5000) return;
        qInfo().nospace().noquote()
            << "[triple] " << triple_rows_ << " points (" << triple_floor_ << " floor, "
            << triple_ceil_ << " ceiling) over " << triple_frames_ << " frames ("
            << QString::number(triple_frames_ ? double(triple_rows_) / triple_frames_ : 0.0, 'f', 2)
            << "/frame)"
            << (triple_rows_ == 0 and triple_frames_ > 0
                    ? "  <- frames ARE arriving and no vertex resolved: check that wall corners AND"
                      " floor junctions are both enabled, they are intersected in pairs"
                    : "")
            << (tps_chi2_n_ > 0
                    ? QString(" | POSE FACTOR (shadow, cannot drive): POST-fit chi2/dof %1 over"
                              " %2 frames — this is the gate, and it must approach 1")
                          .arg(tps_chi2_sum_ / tps_chi2_n_, 0, 'f', 2).arg(tps_frames_)
                    : QString());
        qInfo().nospace() << "[imgedge] " << imgedge_rows_ << " rows / " << imgedge_calls_
                          << " calls | no_window=" << imgedge_no_window_
                          << " no_obs=" << imgedge_no_obs_ << " (bound, nothing extracted)"
                          << " no_cam=" << imgedge_no_cam_ << " (no valid intrinsics/extrinsic)";
        imgedge_health_last_ms_ = timestamp_ms;
        imgedge_rows_ = imgedge_calls_ = imgedge_no_window_ = imgedge_no_obs_ = imgedge_no_cam_ = 0;
    }

    void RoomConcept::write_debug_tail()
    {
        if (not debug_log_.is_open())
            return;
        debug_log_ << ',' << last_preint_samples_
                   << ',' << last_preint_duration_s_
                   << ',' << last_belief_age_s_
                   << ',' << last_belief_decay_
                   << ',' << last_aff_outcome_
                   << ',' << last_aff_completions_
                   << ',' << last_tgt_x_
                   << ',' << last_tgt_y_
                   << ',' << last_pub_tx_
                   << ',' << last_pub_ty_
                   << ',' << last_pub_ok_
                   << ',' << last_slot_motion_cov_(0, 1)
                   << ',' << last_slot_motion_cov_(0, 2)
                   << ',' << last_slot_motion_cov_(1, 2)
                   << ',' << last_imu_cover_
                   << ',' << last_pub_cov_xx_
                   << ',' << last_pub_cov_tt_
                   << ',' << (last_clamp_hit_ ? 1 : 0)
                   << ',' << last_wheel_gyro_ratio_;
    }

    RoomConcept::OptTiming RoomConcept::take_optimizer_timing()
    {
        const auto n  = opt_total_count_.exchange(0, std::memory_order_relaxed);
        const auto ee = opt_earlyexit_count_.exchange(0, std::memory_order_relaxed);
        const auto us = opt_update_us_sum_.exchange(0, std::memory_order_relaxed);
        OptTiming t;
        t.count         = n;
        t.early_exits   = ee;
        t.avg_update_ms = (n > 0) ? (static_cast<double>(us) / 1000.0 / n) : 0.0;
        return t;
    }

} // namespace rc
