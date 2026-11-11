/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// WHAT A FRAME PUBLISHES (PosePublisher::maybe_publish_corrected_pose), as pure functions so a test can run the very
// code the agent runs: (1) during a hypothesis fit the survey's pose is substituted for the localiser's
// (room-transition C2.7), (2) the kinematic clamp bounds the CORRECTION, never the measured motion.
#pragma once

#include "update_result.h"

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

}   // namespace rc::publish
