/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// WHAT A FRAME PUBLISHES (PosePublisher::maybe_publish_corrected_pose), as pure functions so a test can run the very
// code the agent runs: (1) during a hypothesis fit the survey's pose is substituted for the localiser's
// (room-transition C2.7), (2) the kinematic clamp bounds the CORRECTION, never the measured motion.
#pragma once

#include "update_result.h"
#include "gyro_integral.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace rc::publish
{
/// The survey's own pose while a remembered layout is judged (RoomConcept::survey_pose_during_fit): plain SLAM's shadow
/// pose, its PREDICTION for this frame (last shadow pose + this frame's odometry delta — what the clamp's measured-motion
/// term must difference), and the covariance published when the fit started.
struct SurveyPose
{
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();
    Eigen::Vector3f pred = Eigen::Vector3f::Zero();
    Eigen::Matrix3f cov  = Eigen::Matrix3f::Identity();
};

/// Make `r` the survey's frame: pose, state, covariance AND prediction (the clamp reads pred_x/y/theta: the fit tracker's
/// sits at the K-prior mode, metres away), and keep the epoch of the last publish (the fit's re-seeding is not a
/// relocalisation for consumers, so the clamp stays armed).
inline void substitute_survey(loc::UpdateResult& r, const SurveyPose& s, std::uint64_t last_published_epoch)
{
    r.robot_pose = Eigen::Translation2f(s.pose.head<2>()) * Eigen::Rotation2Df(s.pose.z());
    r.state[2] = s.pose.x(); r.state[3] = s.pose.y(); r.state[4] = s.pose.z();
    r.covariance = s.cov;
    r.pred_x = s.pred.x(); r.pred_y = s.pred.y(); r.pred_theta = s.pred.z();
    r.reloc_epoch = last_published_epoch;
}

struct Clamp
{
    Eigen::Affine2f pose = Eigen::Affine2f::Identity();   ///< what is published (== `now` when not fired)
    Eigen::Vector2f res_xy = Eigen::Vector2f::Zero();     ///< correction NOT applied (goes into covariance)
    float res_th = 0.f;
    bool  fired = false;
    float n_xy = 0.f, c_th = 0.f, m_th = 0.f;             ///< for the log
    bool  m_rejected = false;                             ///< the measured motion was outside the dynamics (ignored)
};

/// The kinematic clamp. `prev` = last published pose, `prev_est` = the localiser's estimate at that publish (pre-clamp),
/// `now` = this frame's pose, `pred` = this frame's prediction, max_xy/max_th = speed limits × dt.
inline Clamp clamp_correction(const Eigen::Affine2f& prev, const std::optional<Eigen::Vector3f>& prev_est,
                              const Eigen::Affine2f& now, const Eigen::Vector3f& pred, float max_xy, float max_th)
{
    Clamp out;
    out.pose = now;
    const Eigen::Vector2f d_xy = now.translation() - prev.translation();
    const float prev_th = std::atan2(prev.linear()(1, 0), prev.linear()(0, 0));
    const float new_th  = std::atan2(now.linear()(1, 0), now.linear()(0, 0));
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
    if (prev_est.has_value())
    {
        const Eigen::Vector3f& pe = prev_est.value();
        m_xy = Eigen::Vector2f(pred.x() - pe.x(), pred.y() - pe.y());
        m_th = std::remainder(pred.z() - pe.z(), 2.f * static_cast<float>(M_PI));
    }

    // ★ THE MEASURED MOTION HAS THE SAME SUPPORT AS THE CORRECTION (2026-11-01, warm21): a step the robot's dynamics cannot
    //   produce is not a measurement of where it went. A frame whose prediction was never made (the DEFAULT (0,0,0) after a
    //   model reset) "measured" -(the whole pose); passed through unbounded it re-based the published pose by about (-4.1,
    //   -6.3) m every frame — (-268,-458) after minutes, covariance 290+ m, the controller blind. Outside the support it
    //   counts as NO measured motion, and the whole step goes through the bounded correction below.
    if (not m_xy.allFinite() or not std::isfinite(m_th) or m_xy.norm() > max_xy or std::abs(m_th) > max_th)
    {
        out.m_rejected = true;
        m_xy = Eigen::Vector2f::Zero();
        m_th = 0.f;
    }

    // What is left after the measured motion is accounted for IS the correction — the only
    // part of the published delta that has no dynamical support and can therefore be a
    // teleport. Bound that, and only that.
    const Eigen::Vector2f c_xy = d_xy - m_xy;
    const float           c_th = std::remainder(d_th - m_th, 2.f * static_cast<float>(M_PI));
    const float n_xy = c_xy.norm();
    out.n_xy = n_xy; out.c_th = c_th; out.m_th = m_th;

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
        out.res_xy = d_xy - xy_c;
        out.res_th = d_th - th_c;
        out.pose = Eigen::Affine2f::Identity();
        out.pose.translation() = prev.translation() + xy_c;
        out.pose.linear() = Eigen::Rotation2Df(prev_th + th_c).toRotationMatrix();
        out.fired = true;
    }
    return out;
}

/// The PREDICTED pose PosePublisher::publish_predicted_tick writes: `base` (the last correction) carried `dt_s` forward by the
/// body twist cached at that correction (adv = +Y forward, side = +X lateral, rot = yaw rate), through the SE(2) exponential
/// map (left Jacobian V, so v*dt becomes the CHORD of the arc swept). One function for the tick AND, piece by piece
/// along the integrated gyro heading, for predict_to_newest_imu below (the correction's second RT block).
/// ★ It is a RIGHT composition with a body increment, so it commutes with any rigid change of the room frame on the left
///   (T_pub_int_, the proto-room's T_proto_room): propagating a pose already mapped into a published frame is the same as
///   mapping the propagated internal pose. write_robot_room_rt relies on that.
inline Eigen::Affine2f propagate_twist(const Eigen::Affine2f& base, float adv, float side, float rot, float dt_s)
{
    const float base_th = std::atan2(base.linear()(1, 0), base.linear()(0, 0));
    const float phi = rot * dt_s;
    const float vx = side * dt_s;
    const float vy = adv  * dt_s;
    float dx = vx, dy = vy;
    if (std::abs(phi) > 1e-6f)   // the limit phi -> 0 of the expressions below is (vx, vy): a removable singularity, not a gate
    {
        const float s = std::sin(phi), c = std::cos(phi);
        dx = (s * vx - (1.f - c) * vy) / phi;
        dy = ((1.f - c) * vx + s * vy) / phi;
    }
    Eigen::Affine2f out = Eigen::Affine2f::Identity();
    out.linear()      = Eigen::Rotation2Df(base_th + phi).toRotationMatrix();
    out.translation() = base.translation() + base.linear() * Eigen::Vector2f(dx, dy);
    return out;
}

/// ── EVERY CORRECTION IS WRITTEN AS TWO BLOCKS: ITSELF, AND ITSELF CARRIED TO THE NEWEST IMU SAMPLE ─────────────────
/// (owner, 2026-11-11). The correction for t_c lands ~40 ms after t_c; by then the IMU has samples up to t_imu > t_c.
/// write_robot_room_rt hands RT_API::rewrite_edge_RT_since {(t_c, corrected), (t_imu, this)}, which drops every block
/// stamped >= t_c, so a reader's newest block is never older than the last IMU sample at the time of the write, and no
/// prediction made from an older correction survives after t_c. (This replaces the old superseded_tail, which
/// re-propagated every IMU-rate tick stamped after t_c by the cached twist: ticks no longer write the edge by default.)
///
/// "Carried" = RE-INTEGRATED over the IMU samples in (t_c, t_imu]:
///   heading — the gyro, through the SAME integral as the motion prior (rc::imu::for_each_gyro_step, the body of
///             OdomSources::imu_dtheta) and the same calibration (dpsi = k_g * raw - b_g * dt; k_g/b_g are the
///             values the motion prior applied this frame, 1/0 when not acting);
///   translation — the correction's cached body twist (adv = +Y forward, side = +X lateral), composed piece by piece
///             along that heading with propagate_twist, each piece's rate being its own gyro increment / its length.
/// Clocks: `use_sim_clock` says the history's stamps are the producer's SIM clock (Webots), and then t_c_key is t_c
/// mapped onto it (SimClockMap::to_sim, as integrate_odometry_over_window does). The gyro increments are taken on
/// that clock; the twist is per WALL second (it was divided by the wall gap between lidar stamps), so each piece's
/// translation uses its length scaled by wall/key over the whole interval. On the real robot both clocks are wall and
/// the ratio is exactly 1. The block is STAMPED with the newest sample's WALL stamp, the corrections' clock.
/// Returns nullopt — and the caller writes the correction alone — when there is no sample newer than t_c on both
/// clocks, or the history does not bracket (t_c, t_imu] (the integral refuses partial coverage).
struct ImuCarried
{
    std::uint64_t   ts_ms = 0;   ///< WALL stamp of the newest IMU sample
    Eigen::Affine2f pose  = Eigen::Affine2f::Identity();
    double          dtheta = 0.0;   ///< calibrated heading increment integrated over (t_c, t_imu], rad
};
inline std::optional<ImuCarried>
predict_to_newest_imu(const Eigen::Affine2f& corrected, std::int64_t t_c_wall, std::int64_t t_c_key,
                      const std::vector<ImuReading>& imu, bool use_sim_clock,
                      float adv, float side, float k_g, float b_g)
{
    if (imu.size() < 2) return std::nullopt;
    const auto stamp = [use_sim_clock](const ImuReading& s) { return use_sim_clock ? s.integration_ts_ms() : s.effective_ts_ms(); };
    const ImuReading& newest = imu.back();
    const std::int64_t t_imu_wall = newest.effective_ts_ms();
    const std::int64_t t_imu_key  = stamp(newest);
    if (t_imu_wall <= t_c_wall or t_imu_key <= t_c_key) return std::nullopt;
    const double wall_per_key = static_cast<double>(t_imu_wall - t_c_wall) / static_cast<double>(t_imu_key - t_c_key);

    ImuCarried out;
    out.ts_ms = static_cast<std::uint64_t>(t_imu_wall);
    out.pose  = corrected;
    const bool ok = rc::imu::for_each_gyro_step(imu, stamp, t_c_key, t_imu_key,
        [&](std::int64_t s, std::int64_t e, double dth_raw)
        {
            const double dt_key  = static_cast<double>(e - s) * 1e-3;
            const double dth     = dth_raw * k_g - b_g * dt_key;
            const double dt_wall = dt_key * wall_per_key;
            out.dtheta += dth;
            out.pose = propagate_twist(out.pose, adv, side, static_cast<float>(dth / dt_wall), static_cast<float>(dt_wall));
        });
    if (not ok) return std::nullopt;
    return out;
}
}   // namespace rc::publish
