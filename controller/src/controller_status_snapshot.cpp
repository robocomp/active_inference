// ControllerSession::status_snapshot — copy what the controller is doing into rc::ControllerStatusSnapshot for the window's
// status column (controller_status_snapshot.h). CONTROL thread, once per cycle, from SpecificWorker::update_custom_widget.
// Read-only and cheap: member copies plus one tangent/position lookup on the driven route — no search, no window test.
#include "controller_session.h"
#include "controller_status_snapshot.h"
#include "local_policy/local_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>

rc::ControllerStatusSnapshot ControllerSession::status_snapshot(const std::optional<ControllerRobotPose> &robot_pose,
                                                                std::uint64_t now_ms) const
{
    rc::ControllerStatusSnapshot s;
    s.valid = true;
    s.now_ms = now_ms;

    // who drives
    s.policy_enabled = policy_enabled();
    s.hybrid_on = chain_hybrid_on();
    s.hybrid_place = chain_hybrid_place_;
    s.chain_monitor_on = params_ != nullptr and params_->chain_monitor_enabled;
    s.marks_drive = chain_marks_drive();
    s.episode_active = chain_ep_.active;
    s.episode_final = chain_ep_.final;
    s.episode_why = chain_ep_.why;
    s.episode_sub_x = chain_ep_.sub.x();
    s.episode_sub_y = chain_ep_.sub.y();
    s.episode_s_end = chain_ep_.s_end;
    s.handoff_requested = chain_ep_request_ != nullptr ? std::string(chain_ep_request_) : std::string{};
    s.chain_wait = chain_wait_;
    s.marks_off = chain_marks_off_;
    s.clear_check = chain_clear_check_;
    {   // the monitor's verdict counts only while it is current (the chain tracking this cycle, give or take a second)
        const auto steady_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch()).count());
        const bool fresh = chain_last_monitor_ms_ != 0 and steady_ms - chain_last_monitor_ms_ < 1000;
        s.chain_verdict = fresh ? chain_last_verdict_ : -1;
        s.chain_ahead_m = chain_last_ahead_m_;
        s.chain_stall = fresh and chain_last_stall_;
    }
    s.marks = static_cast<int>(chain_marks_.marks.size());
    s.fail_marks = static_cast<int>(chain_marks_.fail.size());
    s.escape_active = escape_active_;
    s.backoff_active = returns_backoff_active_;
    s.lockon_active = lockon_.active();
    s.approach_order_active = approach_order_.active();
    s.policy_mode = policy_last_mode_;
    if (policy_last_mode_ >= 0)
        s.policy_mode_name = rc::policy::mode_name(static_cast<rc::policy::Decision::Mode>(policy_last_mode_));
    s.policy_hold_why = policy_last_why_;

    // the goal
    if (last_target_info_.has_value())
    {
        s.has_target = true;
        s.target_name = last_target_info_->node_name;
        s.target_from_affordance = last_target_info_->from_affordance;
        if (robot_pose.has_value()) s.target_straight_m = (last_target_info_->room_pos - robot_pose->pos).norm();
    }
    else if (current_target_room_.has_value())
    {
        s.has_target = true;
        if (robot_pose.has_value()) s.target_straight_m = (*current_target_room_ - robot_pose->pos).norm();
    }
    s.manual_target = manual_target_room_.has_value();
    s.orienting = orient_start_ms_.has_value() or orient_entry_ms_.has_value();

    // the route: progress is the chain monitor's projection when it ran, the follower's own on a mission route; nothing is
    // projected here (a projection is a search, and the panel must not cost the loop)
    if (const auto curve = driven_curve(); curve.valid())
    {
        s.route_valid = true;
        s.route_on_mission = curve.on_mission();
        s.route_len = curve.spline->length();
        if (chain_have_s_) { s.route_s_known = true; s.route_s = chain_s_prog_; }
        else if (curve.on_mission()) { s.route_s_known = true; s.route_s = route_.progress(); }
        if (s.route_s_known and robot_pose.has_value())
        {
            const float sp = std::clamp(s.route_s, 0.f, s.route_len);
            const float hd = curve.spline->heading_at(sp);
            const Eigen::Vector2f d = robot_pose->pos - curve.spline->position_at(sp);
            s.route_heading_err = std::remainder(robot_pose->theta - hd, 2.f * std::numbers::pi_v<float>);
            s.route_cross = std::cos(hd) * d.y() - std::sin(hd) * d.x();
        }
    }
    s.plan_gen = plan_spline_gen_;

    // monitors
    s.sigma_track = s.chain_monitor_on ? chain_sigma_track_now_ : -1.f;
    s.no_route_for_ms = no_route_since_ms_ != 0 and now_ms > no_route_since_ms_ ? float(now_ms - no_route_since_ms_) : 0.f;

    // localisation
    s.have_pose = robot_pose.has_value();
    s.room = current_room_name_;
    s.pose_xy_std = policy_pose_xy_std_;
    s.pose_theta_std = policy_pose_theta_std_;
    s.pose_age_ms = world_model_pose_stamp_age_ms_;

    // odometry (its stamp is the producer's WALL ms)
    s.odo_valid = odo_valid_;
    s.odo_v = odo_v_;
    s.odo_w = odo_w_;
    if (odo_stamp_ms_ != 0)
    {
        const auto wall_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        s.odo_age_ms = wall_now > odo_stamp_ms_ ? float(wall_now - odo_stamp_ms_) : 0.f;
    }
    s.odo_fresh_tau_ms = params_ != nullptr ? params_->command_freshness_tau_ms : 0.f;

    s.cross_status = cross_status_;
    return s;
}
