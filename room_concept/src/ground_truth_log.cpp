#include <genericworker.h>   // FIRST: DSR's signal emitter is not self-contained

#include "ground_truth_log.h"

#include <dsr/api/dsr_api.h>
#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <cmath>
#include <filesystem>
#include <locale>

namespace rc
{
void GroundTruthLog::gt_convention_report(float est_th, float gt_th_raw)
{
    const double d = est_th - gt_th_raw, u = est_th + gt_th_raw;
    gt_sum_diff_c_ += std::cos(d); gt_sum_diff_s_ += std::sin(d);
    gt_sum_sum_c_  += std::cos(u); gt_sum_sum_s_  += std::sin(u);
    if (++gt_n_ != gt_report_at_) return;
    gt_report_at_ *= 10;                                   // 200, 2000, 20000 -- three checks, then quiet
    const double n = static_cast<double>(gt_n_);
    const double Rd = std::hypot(gt_sum_diff_c_, gt_sum_diff_s_) / n;
    const double Ru = std::hypot(gt_sum_sum_c_,  gt_sum_sum_s_)  / n;
    const double od = std::atan2(gt_sum_diff_s_, gt_sum_diff_c_) * 180.0 / M_PI;
    const double ou = std::atan2(gt_sum_sum_s_,  gt_sum_sum_c_)  * 180.0 / M_PI;
    qInfo().nospace().noquote()
        << "[gt] convention check over " << gt_n_ << " samples: "
        << "est-gt R=" << QString::number(Rd, 'f', 4) << " (offset " << QString::number(od, 'f', 2)
        << " deg) | est+gt R=" << QString::number(Ru, 'f', 4) << " (offset "
        << QString::number(ou, 'f', 2) << " deg)  ->  "
        << (Ru > Rd ? "producer sign is INVERTED, local negation is CORRECT"
                    : "producer sign looks RIGHT -- robot_concept may have been fixed; REMOVE the "
                      "local negation in log_ground_truth or every heading comparison inverts");
}


void GroundTruthLog::log_ground_truth(const rc::RoomConcept::UpdateResult &res)
{
    if (shutting_down_.load() or not G)
        return;
    const auto robots = G->get_nodes_by_type("robot");
    if (robots.empty())
        return;
    const auto &rn = robots.front();
    const auto gx = G->get_attrib_by_name<robot_gt_x_att>(rn);
    const auto gy = G->get_attrib_by_name<robot_gt_y_att>(rn);
    const auto ga = G->get_attrib_by_name<robot_gt_angle_att>(rn);
    if (not gx.has_value() or not gy.has_value() or not ga.has_value())
        return;                                  // no ground truth -> real robot -> nothing to do

    if (not gt_csv_open_attempted_)
    {
        gt_csv_open_attempted_ = true;
        QDir().mkpath("tmp/sdf_localizer");
        // ★ ARCHIVE THE PREVIOUS RUN BEFORE TRUNCATING. This file is the only record of what the
        //   localiser actually did, and every restart used to erase it. That has already cost real
        //   evidence: an A/B comparison could not be re-checked because the baseline run's data was
        //   gone, and a stale copy of a DIFFERENT file was once analysed as if it were live. A
        //   comparison between two configurations is only possible if both survive.
        {
            const QString cur = "tmp/sdf_localizer/gt_error.csv";
            if (QFileInfo fi(cur); fi.exists() and fi.size() > 0)
            {
                const QString keep = QString("tmp/sdf_localizer/gt_error_%1.csv")
                                         .arg(fi.lastModified().toString("yyyy-MM-dd_HH-mm-ss"));
                if (QFile::rename(cur, keep))
                    qInfo() << "[gt] previous run archived as" << keep;
                else
                    qWarning() << "[gt] could NOT archive the previous run; it is about to be lost";
            }
        }
        gt_csv_.open("tmp/sdf_localizer/gt_error.csv", std::ios::out | std::ios::trunc);
        if (gt_csv_.is_open())
        {
            // Written through the CLASSIC locale: these machines run es_ES, where a comma is the
            // decimal separator, and a CSV whose fields contain commas is unparseable.
            gt_csv_.imbue(std::locale::classic());
            gt_csv_ << "ts_ms,gt_x,gt_y,gt_theta,est_x,est_y,est_theta,gt_theta_raw,"
                       "sdf_mse,iters,cov_tt,"
                       "imu_dtheta,wheel_dtheta,wheel_shadow_dtheta,imu_segs,wheel_segs,"
                       "pred_x,pred_y,pred_theta,dx_local,dy_local,"
                       "calib_k_v,calib_k_w,calib_yaw,calib_eps,calib_carried,calib_dropped,"
                       "calib_sig_kv,calib_sig_kw,calib_sig_yaw,calib_pos_var,"
                       "calib_b_omega,calib_informed,calib_cond,"
                       "imu_dvx,imu_dvy,wheel_dvx,wheel_dvy,imu_dpx,imu_dpy,imu_lin_segs,"
                       // ── FACTOR B of the camera-extrinsic experiment ─────────────────────────
                       // The same window solved twice in the shadow: under the mount as it is, and
                       // under the mount with the self-calibration removed. Both poses RAW and on
                       // the SAME ROW as the ground truth, so M1 is a subtraction here rather than
                       // a join across two files with two clocks. Never pre-differenced: a mean and
                       // its counterfactual travelling as one number is how a mismatch hides.
                       // fb_ts = 0 means the shadow did not produce a pair on this cycle.
                       "fb_ts,fb_cal_x,fb_cal_y,fb_cal_th,fb_nom_x,fb_nom_y,fb_nom_th,"
                       "fb_corr_pitch,fb_corr_height,fb_corr_yaw\n";
        }
        else
            qWarning() << "[gt] cannot open tmp/sdf_localizer/gt_error.csv";
    }
    if (not gt_csv_.is_open())
        return;

    const auto &p = res.robot_pose;
    const float est_th = std::atan2(p.linear()(1, 0), p.linear()(0, 0));
    // See the declaration in specificworker.h: the producer's sign is inverted, so the CSV carries
    // the corrected angle in gt_theta (what every downstream analysis wants) and the untouched value
    // in gt_theta_raw (so nothing is lost and the claim stays checkable from the file alone).
    const float gt_th_raw = ga.value();
    const float gt_th     = -gt_th_raw;
    gt_convention_report(est_th, gt_th_raw);
    // Fetched once and used raw: the pose error each implies is computed from this row offline,
    // because the subtraction is the analysis and not the measurement.
    const auto fb = room_concept_.get_factor_b();
    const std::int64_t fb_ts = fb.valid ? fb.ts_ms : 0;
    gt_csv_ << res.timestamp_ms
            << ',' << gx.value() << ',' << gy.value() << ',' << gt_th
            << ',' << p.translation().x() << ',' << p.translation().y() << ',' << est_th
            << ',' << gt_th_raw
            << ',' << res.sdf_mse << ',' << res.iterations_used
            << ',' << (res.covariance.rows() > 2 ? res.covariance(2, 2) : -1.f)
            // Heading-channel attribution: which sensor produced this cycle's predicted rotation.
            // imu_dtheta+wheel_dtheta is what entered the prior; wheel_shadow_dtheta is what the
            // wheels claimed over the SAME intervals the gyro overrode. With gt_theta alongside,
            // each channel's scale error can be regressed out separately instead of inferred from
            // their pooled effect.
            << ',' << res.imu_dtheta << ',' << res.wheel_dtheta << ',' << res.wheel_shadow_dtheta
            << ',' << res.imu_segs << ',' << res.wheel_segs
            // Pre-optimizer predicted pose: (est - pred) is the correction, i.e. one tooth measured
            // in the room frame with no GT and no frame fit. Equals est exactly when iters==0.
            << ',' << res.pred_x << ',' << res.pred_y << ',' << res.pred_theta
            // What the wheels claimed in the BODY frame: dx_local is lateral. Driving straight it
            // should be ~0; anything else is the predictor being told the robot slid sideways.
            << ',' << res.dx_local << ',' << res.dy_local
            // Learned motion-model parameters, so convergence is visible in the same file as the
            // error they are supposed to remove.
            << ',' << res.calib_k_v << ',' << res.calib_k_w << ',' << res.calib_yaw
            << ',' << res.calib_episodes
            // Spans that reached the trigger with nothing measured in them. See
            // CALIB_UNMEASURED_EPISODES.md: these used to be emitted as zeros.
            << ',' << res.calib_carried << ',' << res.calib_dropped
            // Sigmas + the R actually used: without these a replay of the filter from this file is
            // guesswork, and a parameter that stopped moving cannot be told from one never taught.
            << ',' << res.calib_sigma_k_v << ',' << res.calib_sigma_k_w
            << ',' << res.calib_sigma_yaw << ',' << res.calib_pos_var
            // Joint-solve outputs: the gyro bias it can now separate, which parameters this window
            // actually taught, and how collinear the window was.
            << ',' << res.calib_b_omega << ',' << res.calib_informed << ',' << res.calib_condition
            // Linear IMU channel. imu_dv vs wheel_dv is translation's first independent cross-check
            // logged before being fused, because a channel whose covariance is unknown (the ImuFrame
            // IDL has no acc_var) must be shown to agree with something before anything trusts it.
            << ',' << res.imu_dvx << ',' << res.imu_dvy
            << ',' << res.wheel_dvx << ',' << res.wheel_dvy
            << ',' << res.imu_dpx << ',' << res.imu_dpy << ',' << res.imu_lin_segs
            // ── Factor B: both poses RAW, on this row, never differenced here ────────────────────
            << ',' << fb_ts
            << ',' << fb.pose_calibrated.x() << ',' << fb.pose_calibrated.y() << ',' << fb.pose_calibrated.z()
            << ',' << fb.pose_nominal.x()    << ',' << fb.pose_nominal.y()    << ',' << fb.pose_nominal.z()
            << ',' << fb.correction.x() << ',' << fb.correction.y() << ',' << fb.correction.z()
            << '\n';
    gt_csv_.flush();
}


// ── RGB edge alignment: one extraction per compute() tick ────────────────────────────────────────
//
// Runs on the MAIN thread. That is safe and deliberate: everything it touches is plain data (a
// grayscale vector, the reduced CameraModel, the cached static extrinsic, the room polygon), so it
// makes no DSR call at all — in particular no room<-robot lookup, which is the circularity trap the
// whole design is built to avoid (see image_edge_source.h). The only graph access in this subsystem
// is CameraIngestor::bind_camera(), which is main-thread by construction because of the ts==0 cache.
// Every camera in ImageEdge.calibCameras that is NOT the driving one: bind, extract, pair against
// the same LiDAR corners, and feed its own evidence file. Deliberately a separate function from
// pump_image_edges(): the driving camera's path also produces the pose factor and the shadow, and
// none of that should run twice.
// The RGB channels that lived here — the driving camera's extraction, the calibration-only cameras,
// and the triple-point placement, 424 lines — are now rc::CalibChannels (src/calib_channels.{h,cpp}).

}   // namespace rc
