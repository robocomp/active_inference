// genericworker.h FIRST, exactly as specificworker.h has it: it pulls DSR's headers in the order the
// library expects, and dsr_signal_emitter.h fails to compile its own ThreadPool member otherwise.
#include <genericworker.h>

#include "pose_publisher.h"

#include "room_concept.h"
#include "room_scene_graph.h"
#include "room_viewer.h"
#include "../../common/robot_capability/robot_capability.h"

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDateTime>
#include <QDebug>
#include <QString>
#include <QThread>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <locale>

namespace rc
{
void PosePublisher::update_rt_rate_readout(std::int64_t now_ms, bool on_gui_thread)
{
    if (rt_rate_window_start_ms_ == 0)
    {
        rt_rate_window_start_ms_ = now_ms;
        return;
    }
    const std::int64_t elapsed = now_ms - rt_rate_window_start_ms_;
    if (elapsed < 1000)
        return;

    const float dt_s    = static_cast<float>(elapsed) * 0.001f;
    const float corr_hz = static_cast<float>(rt_corr_count_) / dt_s;   // RT publishes = corrected poses

    // Optimizer timing (loc thread): processing rate, mean cost/update, and early-exit fraction —
    // diagnoses whether localization is compute-bound (low Hz, high ms) and CUDA/window effects.
    const auto opt = room_concept_.take_optimizer_timing();
    const float opt_hz = static_cast<float>(opt.count) / dt_s;
    const float ee_pct = (opt.count > 0)
        ? 100.0f * static_cast<float>(opt.early_exits) / static_cast<float>(opt.count) : 0.0f;

    // Process CPU usage over the same window (sum across all threads — loc/GUI/ingest — so >100% on a
    // busy multithreaded frame is expected, exactly like `top`'s per-process %CPU). getrusage is portable
    // and needs no /proc parsing; delta CPU-seconds / wall-seconds × 100.
    float cpu_pct = 0.0f;
    {
        static double last_cpu_s = -1.0;
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0)
        {
            const double cpu_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6
                               + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
            if (last_cpu_s >= 0.0)
                cpu_pct = static_cast<float>(100.0 * (cpu_s - last_cpu_s) / dt_s);
            last_cpu_s = cpu_s;
        }
    }

    // ── Cost, logged rather than only displayed ──────────────────────────────────────────────────
    // The experiment compares configurations that differ in what they compute — a second camera
    // channel, a corner factor in the loss — so CPU is a RESULT, not housekeeping: an accuracy gain
    // bought with 40% more CPU is a different claim from one that is free.
    // ★ The active configuration is written into the header. Runs are compared days apart from
    //   archived files, and a cost figure whose condition has to be reconstructed from memory is
    //   worth very little — this file says what produced it.
    {
        static std::ofstream res_csv;
        if (not res_csv.is_open())
        {
            res_csv.open("etc/resource_usage.csv", std::ios::out | std::ios::trunc);
            if (res_csv.is_open())
            {
                res_csv.imbue(std::locale::classic());   // CLAUDE.md: never a comma decimal
                std::string cams;
                for (const auto &c : params.CALIB_CAMERAS)
                { if (not cams.empty()) cams += "+"; cams += c; }
                res_csv << "# camera=" << params.IMAGE_EDGE_CAMERA
                        << " drive=" << (params.IMAGE_EDGE_DRIVE ? 1 : 0)
                        << " shadow=" << (params.IMAGE_EDGE_SHADOW ? 1 : 0)
                        << " triple=" << (params.IMAGE_EDGE_USE_TRIPLE_POINTS ? 1 : 0)
                        << " calib_cameras=" << (cams.empty() ? "none" : cams)
                        << " enable=" << (params.IMAGE_EDGE_ENABLE ? 1 : 0) << '\n';
                res_csv << "ts_ms,cpu_pct,rss_mb,opt_hz,corr_hz,early_exit_pct,avg_update_ms\n";
            }
        }
        if (res_csv.is_open())
        {
            struct rusage ru2;
            const double rss_mb = (getrusage(RUSAGE_SELF, &ru2) == 0)
                                      ? ru2.ru_maxrss / 1024.0 : 0.0;   // ru_maxrss is KiB on Linux
            res_csv << now_ms << ',' << cpu_pct << ',' << rss_mb << ',' << opt_hz << ','
                    << corr_hz << ',' << ee_pct << ',' << opt.avg_update_ms << '\n';
            res_csv.flush();
        }
    }

    if (on_gui_thread && viewer())
    {
        // RT/opt rates → live plot (trend). The text readout keeps only the scalar optimizer-health
        // numbers that aren't plotted (ms per update, early-exit %, process CPU%).
        viewer()->add_rate_samples(corr_hz, opt_hz);
        viewer()->set_rt_rate_text(
            QString("%1 ms/upd · %2% early-exit · %3% CPU")
                .arg(opt.avg_update_ms, 0, 'f', 1)
                .arg(ee_pct, 0, 'f', 0)
                .arg(cpu_pct, 0, 'f', 0));
    }

    rt_corr_count_ = 0;
    rt_rate_window_start_ms_ = now_ms;
}

// Append one pose-trace row. type 0=corrected (optimizer, ~20 Hz), 1=predicted (dead-reckon, ~60 Hz).
// valid_ts_ms = the pose's validity stamp; wall_ms = QDateTime now. Lets us overlay the intermediate
// predicted poses on the corrected ones and see the dead-reckoning wander / correction snaps.
void PosePublisher::log_pose_trace(int type, std::int64_t valid_ts_ms,
                                    const Eigen::Affine2f& pose, float innov_norm)
{
    if (not pose_trace_open_attempted_)
    {
        pose_trace_open_attempted_ = true;
        pose_trace_.open("etc/pose_trace.csv", std::ios::out | std::ios::trunc);
        if (pose_trace_.is_open())
            pose_trace_ << "wall_ms,type,valid_ts_ms,x,y,theta,innov_norm\n";
    }
    if (not pose_trace_.is_open())
        return;
    const float x = pose.translation().x();
    const float y = pose.translation().y();
    const float th = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
    const std::int64_t wall_ms = QDateTime::currentMSecsSinceEpoch();
    pose_trace_ << wall_ms << ',' << type << ',' << valid_ts_ms << ','
                << x << ',' << y << ',' << th << ',' << innov_norm << '\n';
    pose_trace_.flush();

    // ── AND, IF THIS STEP WAS PHYSICALLY IMPOSSIBLE, RECORD IT SOMEWHERE THAT SURVIVES ──────────
    if (type < 0 or type > 1)
        return;
    TracePoint now{wall_ms, valid_ts_ms, x, y, th, innov_norm, true};
    const TracePoint before = prev_trace_[type];
    prev_trace_[type] = now;
    if (not before.set)
        return;

    const double dt_s = static_cast<double>(wall_ms - before.wall_ms) * 1e-3;
    if (dt_s <= 1e-3 or dt_s > 1.0)     // first row after a gap: nothing to compare against
        return;

    // The bound is what the BASE SAYS IT CAN DO, read off the robot node. No capability published ⇒
    // say so once and stay quiet, rather than inventing a number and calling its output evidence.
    rc::BaseCapability cap;
    if (G != nullptr)
        if (const auto robots = G->get_nodes_by_type("robot"); not robots.empty())
            cap = rc::read_base_capability(*G, robots.front().id());
    // ★EACH BOUND IS USED IF PUBLISHED, AND THE ABSENCE OF ONE DOES NOT DISABLE THE OTHER.
    // ⚠CORRECTION (2026-09-11): the commit that added this block, and an earlier version of this
    // comment, both asserted that the base declares no maximum linear speed. That was WRONG and the
    // error was mine: SVD48VBase config_diferential.toml has always carried `maxLinSpeed`, and a
    // grep pattern of mine failed to match the key. Both bounds have been published all along
    // (maxLinSpeed, now 1500 mm/s; maxRotSpeed 2 rad/s), so the linear check was never inert.
    // The independent-bounds design below is kept anyway, because it is right for a robot that
    // declares only one -- but it was not, as claimed, necessary here.
    // ★EITHER BOUND ALONE WOULD HAVE CAUGHT WHAT WAS SEEN. Both 2026-09-10 events carried 43.6 and
    // 89.2 degrees of heading step (89 deg in 34 ms is ~46 rad/s against a 2 rad/s base) as well as
    // impossible translation. The `trigger` column records which fired, so a reader never has to
    // assume both were checked.
    if (not cap.max_linear_speed_mps.has_value() and not cap.max_rot_speed_rps.has_value())
    {
        if (not pose_jump_no_capability_warned_)
        {
            pose_jump_no_capability_warned_ = true;
            qWarning() << "[pose_jumps] the base publishes NEITHER a linear nor a rotational speed"
                       << "capability, so no step can be called impossible — etc/pose_jumps.csv will"
                       << "stay empty. That is a MISSING CHANNEL, not an absence of jumps.";
        }
        return;
    }

    const double step_m      = std::hypot(now.x - before.x, now.y - before.y);
    const double implied_mps = step_m / dt_s;
    const double dtheta      = std::remainder(static_cast<double>(now.th - before.th), 2.0 * M_PI);
    const double implied_rps = std::abs(dtheta) / dt_s;
    const bool   too_fast    = cap.max_linear_speed_mps.has_value()
                               and implied_mps > cap.max_linear_speed_mps.value();
    const bool   too_spinny  = cap.max_rot_speed_rps.has_value()
                               and implied_rps > cap.max_rot_speed_rps.value();
    if (not too_fast and not too_spinny)
        return;
    const char *trigger = (too_fast and too_spinny) ? "both" : too_fast ? "linear" : "rotational";

    if (not pose_jump_log_attempted_)
    {
        pose_jump_log_attempted_ = true;
        pose_jump_run_id_ = wall_ms;
        // ★APPEND, never truncate. The header goes in only when the file is new, so a week of runs
        // accumulates into one file and run_id_ms separates them.
        const bool fresh = not std::filesystem::exists("etc/pose_jumps.csv")
                           or std::filesystem::file_size("etc/pose_jumps.csv") == 0;
        pose_jump_log_.open("etc/pose_jumps.csv", std::ios::out | std::ios::app);
        if (pose_jump_log_.is_open())
        {
            pose_jump_log_.imbue(std::locale::classic());   // never emit a decimal comma under es_ES
            if (fresh)
                pose_jump_log_ << "run_id_ms,wall_ms,type,dt_ms,trigger,"
                                  "x_before,y_before,th_before,valid_ts_before,innov_before,"
                                  "x_after,y_after,th_after,valid_ts_after,innov_after,"
                                  "step_m,implied_mps,dtheta_rad,implied_rps,cap_mps,cap_rps\n";
            qInfo() << "[pose_jumps] recording localiser discontinuities to etc/pose_jumps.csv"
                    << "(append-only; run id" << pose_jump_run_id_ << ")";
        }
    }
    if (not pose_jump_log_.is_open())
        return;

    // An unpublished bound is written EMPTY, never 0 — a zero here would read as a real capability
    // of zero, i.e. every step impossible, which is the opposite of what absence means.
    pose_jump_log_ << pose_jump_run_id_ << ',' << wall_ms << ',' << type << ','
                   << static_cast<std::int64_t>(dt_s * 1000.0) << ',' << trigger << ','
                   << before.x << ',' << before.y << ',' << before.th << ','
                   << before.valid_ts_ms << ',' << before.innov << ','
                   << now.x << ',' << now.y << ',' << now.th << ','
                   << now.valid_ts_ms << ',' << now.innov << ','
                   << step_m << ',' << implied_mps << ',' << dtheta << ',' << implied_rps << ',';
    if (cap.max_linear_speed_mps.has_value()) pose_jump_log_ << cap.max_linear_speed_mps.value();
    pose_jump_log_ << ',';
    if (cap.max_rot_speed_rps.has_value())    pose_jump_log_ << cap.max_rot_speed_rps.value();
    pose_jump_log_ << '\n';
    pose_jump_log_.flush();
}

bool PosePublisher::maybe_publish_corrected_pose()
{
    // Called from the LOCALIZER thread directly (on_result_ready, no marshal since the 2026-09-03
    // concurrency fix -- see set_on_result_ready) and, redundantly, is NOT also called from compute()
    // any more (see the FIX note there) precisely because two threads landing here would race on the
    // dedup state below and on the ofstream members inside dsr_update_calibration/dsr_update_affordance.
    // In PreserveBootstrapRoom mode the room is a static prior — the localizer still runs for the
    // viewer, but we never touch the graph.
    if (shutting_down_.load() || params.PRESERVE_BOOTSTRAP_ROOM)
        return false;

    // FIX 2026-09-04: held for the rest of this function. Shared with publish_predicted_tick(), which
    // runs on the imu-ingest thread and reads last_published_pose_/last_pub_adv_/_side_/_rot_/
    // last_published_cov_ -- all written below -- and calls into the same scene_graph_.
    std::lock_guard<std::mutex> publish_lock(publish_mutex_);

    // NOT const: get_last_result() returns a by-value copy, and the kinematic clamp below rewrites
    // robot_pose/covariance in place before this frame is handed to the scene graph.
    auto loc_res = room_concept_.get_last_result();
    if (!loc_res.has_value() || !loc_res->ok)
        return false;

    // Dedup by the lidar stamp so the compute() path and the immediate-publish path are idempotent
    // (whichever fires first for a given frame wins; the other no-ops). The throttle is a small
    // anti-burst floor only — publishing runs at the optimizer/lidar rate, not the compute rate.
    const auto now_ms = QDateTime::currentMSecsSinceEpoch();
    const bool fresh_correction = loc_res->timestamp_ms > 0
                                  && loc_res->timestamp_ms != last_dsr_published_ts_ms_;
    if (!fresh_correction
        || (last_dsr_publish_try_ms_ != 0 && now_ms - last_dsr_publish_try_ms_ < 15))
        return false;

    last_dsr_publish_try_ms_ = now_ms;

    // ---- KINEMATIC CLAMP on the published CORRECTION --------------------------------------------
    // The published stream must be a physically possible trajectory: between two published poses the
    // robot cannot have jumped further than its own speed limit allows. Measured on this agent's own
    // debug log (71k frames), it routinely did — 92.8% of frames take the prediction early-exit and
    // publish a drifting odometry prediction, then the optimizer runs on the remaining 7.2% and
    // discharges the whole accumulated error in ONE frame: |innovation| p99 96.7 mm, max 280.2 mm,
    // i.e. an implied 2.2 m/s at p99 and 7.2 m/s peak against a 0.70 m/s robot.
    //
    // This is not a tuning threshold — it is the support of the motion model. A delta implying 7.2 m/s
    // is not unlikely, it is impossible, so it cannot be a report about where the robot went.
    //
    // ★ WHAT IS BOUNDED IS THE CORRECTION, NOT THE POSE DELTA. The published delta is the sum of two
    // things with completely different supports: the motion the sensors MEASURED (bounded by the
    // robot's dynamics, and already a report about where it went), and the optimizer's correction on
    // top of it (a teleport, bounded by nothing). Bounding their SUM against W_MAX made the published
    // yaw rate a hard rate limiter at exactly W_MAX: measured 2026-08-28 on the Webots P3Bot, the
    // published |omega| maxed at 0.8003 rad/s across 60k frames while ground truth reached 3.51 rad/s,
    // 4.5% of frames sat pinned in [0.75, 0.80], and 2221 deg of real rotation was forbidden over one
    // run. The published pose then falls behind DURING a pivot and only catches up once the true rate
    // drops back under the cap — which is exactly the reported symptom: the controller's LiDAR cloud
    // and robot icon swing >100 deg off the walls while turning and snap back when the turn ends,
    // while room_concept's own viewer (which draws the UNCLAMPED last_result_) looks perfect
    // throughout. The clamp had zero headroom by construction, because W_MAX was set to the
    // controller's MaxRotSpeed — the very rate the robot is commanded to reach.
    //
    // So: let the measured motion (pred[k] - pred[k-1]) through untouched, and rate-limit only the
    // residual on top of it. A genuine relocalization still converges over a few frames; a pivot at
    // full speed is no longer misreported as a slower one.
    //
    // The correction itself is LEGITIMATE (real accumulated drift, not a bad fit), so it is not
    // rejected — only rate-limited.
    // Captured BEFORE the clamp rewrites robot_pose: this is the localiser's own estimate, which is
    // the base the NEXT prediction will be built on. It is what the next frame's measured-motion
    // difference must reference -- never the clamped pose, which no predictor ever sees.
    const Eigen::Affine2f est_pre_clamp = loc_res->robot_pose;

    bool clamp_fired = false;
    if (params.POSE_CLAMP_ENABLED and last_published_pose_.has_value()
        and last_published_ts_ms_ > 0 and loc_res->timestamp_ms > last_published_ts_ms_)
    {
        const float dt = static_cast<float>(loc_res->timestamp_ms - last_published_ts_ms_) / 1000.f;
        if (dt > 0.f and dt <= params.POSE_CLAMP_MAX_DT_S)
        {
            const auto& prev = last_published_pose_.value();
            const Eigen::Vector2f d_xy = loc_res->robot_pose.translation() - prev.translation();
            const float prev_th = std::atan2(prev.linear()(1, 0), prev.linear()(0, 0));
            const float new_th  = std::atan2(loc_res->robot_pose.linear()(1, 0),
                                             loc_res->robot_pose.linear()(0, 0));
            const float d_th = std::remainder(new_th - prev_th, 2.f * static_cast<float>(M_PI));

            // The motion the sensors MEASURED between the last published frame and this one.
            // build_motion_prior_selection() builds every prediction on last_update_result.robot_pose
            // — the previous CORRECTED estimate — so pred[k] - est[k-1] is exactly the preintegrated
            // increment for this interval, with nothing else in it. Verified on the live log after the
            // clamp fix: pred[k] - (est[k-1] + imu_dtheta) is 0.0002 deg at p50.
            //
            // ⚠ IT MUST BE est[k-1], NOT pred[k-1]. Differencing two consecutive PREDICTIONS looks
            // equivalent and is not: substituting pred[k-1] = est[k-2] + increment[k-1] gives
            // pred[k] - pred[k-1] = increment[k] + innovation[k-1], i.e. it hands the previous frame's
            // correction through the clamp UNBOUNDED, one frame late. That was the bug in the first
            // version of this fix, and it survived because a free-running predictor would make the two
            // forms identical — the log column it was checked against (est_theta) is written AFTER the
            // clamp, so it showed the clamp's own backlog and not the localiser's estimate.
            //
            // With no previous estimate to difference against (first publication after a reset) it is
            // zero, which degrades to the old total-delta behaviour for that single frame.
            //
            // Any correction the clamp did not apply on earlier frames is deliberately left INSIDE the
            // bounded part here, so a backlog still discharges at max_th per frame rather than
            // escaping as one jump.
            Eigen::Vector2f m_xy = Eigen::Vector2f::Zero();
            float           m_th = 0.f;
            if (last_published_est_.has_value())
            {
                const Eigen::Vector3f& pe = last_published_est_.value();
                m_xy = Eigen::Vector2f(loc_res->pred_x - pe.x(), loc_res->pred_y - pe.y());
                m_th = std::remainder(loc_res->pred_theta - pe.z(), 2.f * static_cast<float>(M_PI));
            }

            // What is left after the measured motion is accounted for IS the correction — the only
            // part of the published delta that has no dynamical support and can therefore be a
            // teleport. Bound that, and only that.
            const Eigen::Vector2f c_xy = d_xy - m_xy;
            const float           c_th = std::remainder(d_th - m_th, 2.f * static_cast<float>(M_PI));

            // Take the bound from the robot before using it. Cheap after the first success (one bool),
            // and placed HERE rather than in initialize() so it cannot silently fall back for the life
            // of the process because the robot node had not synced yet.
            apply_base_capability_to_pose_clamp();
            const float max_xy = params.POSE_CLAMP_V_MAX * dt;
            const float max_th = params.POSE_CLAMP_W_MAX * dt;
            const float n_xy = c_xy.norm();

            if (n_xy > max_xy or std::abs(c_th) > max_th)
            {
                const Eigen::Vector2f xy_c = m_xy + ((n_xy > max_xy and n_xy > 1e-9f)
                                             ? Eigen::Vector2f(c_xy * (max_xy / n_xy)) : c_xy);
                const float th_c = m_th + std::clamp(c_th, -max_th, max_th);

                // The part of the correction we did NOT apply is real, known error in the published
                // pose. Surfacing it as covariance is what keeps the clamp honest rather than a lie:
                // consumers learn the pose is deliberately behind the measurement, and it is the one
                // signal in this stream that is genuinely informative (it fires on ~1% of frames, so
                // it cannot shift the mean sigma the speed governor watches).
                const Eigen::Vector2f res_xy = d_xy - xy_c;
                const float res_th = d_th - th_c;
                loc_res->covariance(0, 0) += res_xy.x() * res_xy.x();
                loc_res->covariance(1, 1) += res_xy.y() * res_xy.y();
                loc_res->covariance(2, 2) += res_th * res_th;

                Eigen::Affine2f clamped = Eigen::Affine2f::Identity();
                clamped.translation() = prev.translation() + xy_c;
                clamped.linear() = Eigen::Rotation2Df(prev_th + th_c).toRotationMatrix();
                loc_res->robot_pose = clamped;

                clamp_fired = true;
                if (++pose_clamp_hits_ % 20 == 1)
                    qWarning() << "[pose-clamp] CORRECTION implied" << (n_xy / dt) << "m/s /"
                               << (std::abs(c_th) / dt) << "rad/s over dt" << dt << "s — limits"
                               << params.POSE_CLAMP_V_MAX << "/" << params.POSE_CLAMP_W_MAX
                               << "; measured motion passed through at" << (std::abs(m_th) / dt)
                               << "rad/s; carried" << res_xy.norm() * 1000.f
                               << "mm into covariance (" << pose_clamp_hits_ << " clamps)";
            }
        }
    }

    // Hand the ACTUALLY-PUBLISHED covariance back for the debug log. The clamp above runs downstream
    // of RoomConcept's own write, so cov_xx there is the PRE-clamp value while every consumer sees
    // this one; without recording it the published sigma is simply not observable from any log, which
    // is how a claim about it came to be made and then retracted.
    // ⚠ ONE-FRAME LAG BY CONSTRUCTION: RoomConcept writes its row before returning, so these values
    // land on the NEXT row. Join pub_cov_* to the frame BEFORE, not the one they appear on.
    room_concept_.note_published_covariance(loc_res->covariance(0, 0), loc_res->covariance(2, 2),
                                            clamp_fired);

    // FIX 2026-09-04: publish the CORRECTED twist, not the raw wheel-odometry snapshot.
    // last_robot_adv_speed_/_side_/_rot_ (kept as fallback below) is
    // -- the latest single sample of robot_concept's robot_current_speed attribute (wheel odometry,
    // ~10 Hz, uncorrected), cached in specificworker.cpp and forwarded here unchanged. Two problems
    // with that as the twist consumers extrapolate with (camera_visualizer.cpp, retina/scene_processor,
    // controller_obstacle_tracker): it can be up to ~100 ms stale (odometry at 10 Hz vs publish at
    // ~20 Hz), and it never received any of the corrections integrate_odometry_over_window() applies
    // to the localiser's OWN prediction -- gyro preference over wheel yaw (measured 8.2% over-report,
    // see room_concept.cpp:5991), the k_w/b_w scale+bias from the online calibrator, or the IMU linear
    // injection. This computes the same corrected quantity room_concept already trusts for its own
    // prediction, from the fields that quantity is already exposed through:
    //   loc_res->dy_local / dx_local -- BODY frame (dy=+Y forward, dx=+X lateral), the k_v/k_lat-scaled,
    //     IMU-injected translation accumulated THIS cycle (see room_concept.cpp:2916-2917/3512-3513).
    //   loc_res->imu_dtheta + wheel_dtheta -- the total rotation entered into the prior this cycle,
    //     whichever channel (gyro or wheel) supplied each segment (same fields motion_calib_.observe()
    //     already sums this way, room_concept.cpp:5732).
    // Divided by dt since the LAST PUBLISH (not a differentiated pose -- differentiating the published,
    // SDF-corrected pose is exactly the correction-induced velocity spike this twist channel was
    // introduced to avoid; see the frame-note in room_scene_graph.cpp). Falls back to the raw wheel
    // snapshot only when there is no previous publish to difference against (first frame) or the
    // interval is degenerate.
    float pub_adv = last_robot_adv_speed_, pub_side = last_robot_side_speed_, pub_rot = last_robot_rot_speed_;
    if (last_published_ts_ms_ > 0 and loc_res->timestamp_ms > last_published_ts_ms_)
    {
        const float dt_s = static_cast<float>(loc_res->timestamp_ms - last_published_ts_ms_) * 1e-3f;
        if (dt_s > 1e-4f)
        {
            pub_adv  = loc_res->dy_local / dt_s;                        // forward, body +Y
            pub_side = loc_res->dx_local / dt_s;                        // lateral, body +X
            pub_rot  = (loc_res->imu_dtheta + loc_res->wheel_dtheta) / dt_s;
        }
    }

    // Publish (corrected pose → robot↔room RT) at the optimizer rate.
    if (scene_graph_ == nullptr)
    {
        // Refuse rather than dereference. A missing hand-over is a wiring mistake, not a runtime
        // condition, so it must say so once and loudly instead of dying inside the callee.
        static bool warned = false;
        if (not warned)
        {
            warned = true;
            qCritical() << "[pose] no scene graph: set_scene_graph() was never called, so no pose can "
                           "be published. This is a construction-order bug in SpecificWorker::initialize().";
        }
        return false;
    }
    scene_graph_->update(*loc_res, pub_adv, pub_side, pub_rot);
    last_published_pose_ = loc_res->robot_pose;
    // Cache for publish_predicted_tick(): the twist and covariance an IMU-rate tick should extrapolate
    // with/report until the NEXT real publish replaces them. covariance is loc_res->covariance
    // (post-clamp, same one every other consumer of this publish sees -- see the note above on
    // note_published_covariance).
    last_pub_adv_       = pub_adv;
    last_pub_side_      = pub_side;
    last_pub_rot_        = pub_rot;
    last_published_cov_ = loc_res->covariance;
    // Anchor for the NEXT frame's measured-motion difference: the localiser's own estimate for this
    // frame, which is the base its next prediction is built on. Taken pre-clamp, so the difference
    // stays a pure sensor increment however hard the clamp bit here.
    last_published_est_ = Eigen::Vector3f(
        est_pre_clamp.translation().x(), est_pre_clamp.translation().y(),
        std::atan2(est_pre_clamp.linear()(1, 0), est_pre_clamp.linear()(0, 0)));
    last_published_ts_ms_ = loc_res->timestamp_ms;
    last_dsr_published_ts_ms_ = loc_res->timestamp_ms;

    ++rt_corr_count_;
    log_pose_trace(/*type=corrected*/0, loc_res->timestamp_ms,
                   loc_res->robot_pose, loc_res->innovation_norm);
    if (on_corrected_) on_corrected_(*loc_res);
    return true;
}

// FIX 2026-09-04: called from ImuIngestor's ingest thread, once per accepted IMU sample (~100 Hz on
// this robot, see imu_ingestor.cpp), so the graph is never more than one IMU period behind instead of
// waiting on the next lidar-paced correction (~20 Hz). Publishes a DEAD-RECKONED pose only -- it never
// touches last_published_pose_/_ts_ms_/_est_ (those anchor the kinematic clamp and the NEXT real
// prediction's measured-motion difference, and must only ever move on a real SDF-validated result) --
// so every tick extrapolates from the SAME last real publish with a growing dt, rather than compounding
// error by extrapolating from the previous tick's own guess.
//
// Deliberately narrow: calls dsr_publish_predicted_pose() (-> write_robot_room_rt(), the same
// NaN-guarded, ring-buffer-safe writer maybe_publish_corrected_pose() uses), NOT scene_graph_->update().
// update() also drives dsr_update_calibration()/dsr_update_affordance() -- CSV writers and DSR node
// churn sized for ~20 Hz -- which must stay on the real, lidar-paced path only.
void PosePublisher::publish_predicted_tick(std::int64_t imu_ts_ms)
{
    if (shutting_down_.load() || params.PRESERVE_BOOTSTRAP_ROOM || !scene_graph_)
        return;

    std::lock_guard<std::mutex> publish_lock(publish_mutex_);

    if (!last_published_pose_.has_value() || last_published_ts_ms_ <= 0 || imu_ts_ms <= last_published_ts_ms_)
        return;   // nothing real published yet, or this sample predates it

    const float dt_s = static_cast<float>(imu_ts_ms - last_published_ts_ms_) * 1e-3f;
    // Reuses POSE_CLAMP_MAX_DT_S rather than a new config key: same question ("how long a gap is
    // guessing still better than silence for"), same answer. Past it, a stalled lidar feed means the
    // cached twist is itself stale, and ticking anyway would run the extrapolation away unbounded at
    // ~100 Hz -- worse than just not publishing until a real correction arrives.
    if (dt_s <= 0.f || dt_s > params.POSE_CLAMP_MAX_DT_S)
        return;

    const Eigen::Affine2f& base = last_published_pose_.value();
    const float base_th = std::atan2(base.linear()(1, 0), base.linear()(0, 0));
    const float phi = last_pub_rot_ * dt_s;

    // Body-frame twist -> SE(2) chord. Same exponential map as
    // controller_obstacle_tracker.cpp::twist_delta() (this codebase's original, verified-correct
    // reference for this exact composition, 2026-08-04) -- the left Jacobian V turns v*dt into the
    // CHORD of the arc actually swept instead of cutting the corner. +Y forward = adv (last_pub_adv_),
    // +X lateral = side (last_pub_side_); see the frame note in room_scene_graph.cpp.
    const float vx = last_pub_side_ * dt_s;
    const float vy = last_pub_adv_  * dt_s;
    float dx = vx, dy = vy;
    if (std::abs(phi) > 1e-6f)
    {
        const float s = std::sin(phi), c = std::cos(phi);
        dx = (s * vx - (1.f - c) * vy) / phi;
        dy = ((1.f - c) * vx + s * vy) / phi;
    }

    Eigen::Affine2f predicted = Eigen::Affine2f::Identity();
    predicted.linear()      = Eigen::Rotation2Df(base_th + phi).toRotationMatrix();
    predicted.translation() = base.translation() + base.linear() * Eigen::Vector2f(dx, dy);

    // last_published_cov_ reused as-is (not grown with dt_s) -- a known simplification. A consumer
    // reading this between two real corrections currently sees the same sigma the last real one
    // reported, not an inflated one reflecting the extra dead-reckoned distance since. Revisit if a
    // consumer needs "how much am I trusting this specific sample" rather than just the freshest pose.
    if (scene_graph_ == nullptr) return;
    scene_graph_->dsr_publish_predicted_pose(predicted, last_published_cov_,
                                             static_cast<std::uint64_t>(imu_ts_ms));
    log_pose_trace(/*type=predicted*/1, imu_ts_ms, predicted, 0.f);
}

// Localiser pose beside the Webots supervisor pose, one row per published correction.
// robot_concept writes robot_gt_* onto the robot node ONLY while the producer reports
// simulated==true, so on real hardware the attributes are absent and this writes nothing and opens
// no file. Absence is the gate; there is no switch to misconfigure.
//
// Both poses are logged RAW, in their own frames — GT in world, the estimate in the room frame.
// They are deliberately NOT differenced here: the room frame's orientation is arbitrary, so a
// constant offset is expected and only its VARIATION is a defect. Fit offset+gain across the file
// and read the residual; a single pair of readings cannot tell those apart.
// Which convention is right is an EMPIRICAL question with a decisive answer: the correct one leaves
// a CONSTANT offset (the room frame's arbitrary orientation datum), the wrong one leaves an offset
// that swings with the robot's heading. Score both by circular concentration R = |mean unit vector|;
// R -> 1 is constant, R -> 0 is uniformly spread. Reported once, on the first few hundred samples.
// ── STAGE 2: pair each RGB triple point with the LiDAR corner of the SAME polygon vertex ─────────
// See mount_lidar_pair.h for why the residual has no pose in it and why that is the point.
// The camera<->LiDAR mount calibration that lived here — 691 lines, eight methods and their whole
// state — is now rc::MountCalibrator (src/mount_calibrator.{h,cpp}). It was never coordination: it is
// a self-contained measurement channel that happened to run on this thread.

void PosePublisher::apply_base_capability_to_pose_clamp()
{
    if (pose_clamp_from_capability_ or shutting_down_.load() or not G) return;
    const auto robots = G->get_nodes_by_type("robot");
    if (robots.empty()) return;                       // not synced yet: try again next cycle
    pose_clamp_from_capability_ = true;               // one shot, whatever it finds

    const auto cap = rc::read_base_capability(*G, robots.front().id());
    if (not cap.any())
    {
        std::cout << "[pose-clamp] the robot node publishes no base capability; keeping the configured "
                     "fallback (v " << params.POSE_CLAMP_V_MAX << " m/s, w " << params.POSE_CLAMP_W_MAX
                  << " rad/s). Producer: robot_concept + Agent.base_config_file." << std::endl;
        return;
    }
    // Absent stays absent: a field the base did not state leaves the fallback in force. Zero would
    // pin every correction to zero, which is not a conservative reading of "unknown".
    const float v_old = params.POSE_CLAMP_V_MAX, w_old = params.POSE_CLAMP_W_MAX;
    if (cap.max_linear_speed_mps) params.POSE_CLAMP_V_MAX = *cap.max_linear_speed_mps;
    if (cap.max_rot_speed_rps)    params.POSE_CLAMP_W_MAX = *cap.max_rot_speed_rps;
    std::cout << "[pose-clamp] bounded by what the ROBOT can do, not by a controller's preference: "
              << "v " << v_old << " -> " << params.POSE_CLAMP_V_MAX << " m/s, "
              << "w " << w_old << " -> " << params.POSE_CLAMP_W_MAX << " rad/s "
              << "(from robot_max_linear_speed / robot_max_rot_speed on '"
              << robots.front().name() << "')." << std::endl;
    if (params.POSE_CLAMP_W_MAX <= w_old)
        std::cout << "[pose-clamp] note: the capability is NOT above the previous value, so this clamp "
                     "was not the thing limiting the published yaw rate." << std::endl;
}
}   // namespace rc
