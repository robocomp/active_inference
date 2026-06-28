#include "controller_session.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <print>
#include <sstream>

void ControllerSession::set_params(const ControllerParams *params)
{
    params_ = params;
    if (params_)
    {
        // HOW (gains/caps/timing) — controller-owned. WHAT/WHEN (scalar_target, stable_n, timeout)
        // are per-affordance and come from the contract at begin() time.
        rc::LockOnConfig c;
        c.sweep_speed_mps = params_->lockon_sweep_speed_mps;
        c.sweep_range_m   = params_->lockon_sweep_range_m;
        c.k_yaw           = params_->lockon_k_yaw;
        c.max_yaw_rps     = params_->lockon_max_yaw_rps;
        c.dither_yaw_rps  = params_->lockon_dither_yaw_rps;
        c.settle_ms       = params_->lockon_settle_ms;
        c.step_ms         = params_->lockon_step_ms;
        c.offset_tol      = params_->lockon_offset_tol;
        c.max_attempts    = params_->lockon_max_attempts;
        lockon_.configure(c);
    }
}

void ControllerSession::set_graph(std::shared_ptr<DSR::DSRGraph> graph)
{
    graph_ = std::move(graph);
}

void ControllerSession::clear_tracking_state()
{
    last_mppi_trajectories_.clear();
    last_mppi_average_trajectory_.clear();
    last_best_mppi_trajectory_idx_ = -1;
    last_display_wp_index_ = 0;
}

bool ControllerSession::sync_world_state(std::uint64_t timestamp_ms,
                                         ControllerWorldModel &world_model,
                                         RoomPathPlanner &planner,
                                         ControllerObstacleTracker &obstacle_tracker,
                                         rc::TrajectoryController &path_controller,
                                         ControllerMotionCommander &motion_commander,
                                         ControllerDisplay &display)
{
    if (!world_model.refresh_graph_state())
    {
        room_polygon_.clear();
        inner_polygon_.clear();
        current_plan_.reset();
        if (!room_wait_logged_)
        {
            qInfo() << "Controller waiting for room and robot nodes in DSR";
            room_wait_logged_ = true;
        }
        update_display(std::nullopt,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        return false;
    }
    room_wait_logged_ = false;

    const auto room_polygon = world_model.read_room_polygon();
    if (!room_polygon.has_value() || room_polygon->size() < 3)
    {
        room_polygon_.clear();
        inner_polygon_.clear();
        current_plan_.reset();
        qInfo() << "Controller waiting for delimiting polygon attributes on room node";
        update_display(std::nullopt,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        return false;
    }

    room_polygon_ = room_polygon.value();
    inner_polygon_ = planner.compute_inner_polygon(room_polygon_);
    obstacle_tracker.update_active_obstacle_polygons(timestamp_ms, path_controller);
    return true;
}

std::optional<ControllerPlanningStep> ControllerSession::build_planning_step(std::uint64_t timestamp_ms,
                                                                             ControllerWorldModel &world_model,
                                                                             ControllerObstacleTracker &obstacle_tracker,
                                                                             rc::AffordanceManager &affordance_manager,
                                                                             RoomPathPlanner &planner,
                                                                             rc::TrajectoryController &path_controller,
                                                                             ControllerMotionCommander &motion_commander,
                                                                             ControllerDisplay &display)
{
    const auto robot_pose = world_model.read_robot_pose_in_room(timestamp_ms, obstacle_tracker.last_lidar_timestamp_ms());
    if (!robot_pose.has_value())
    {
        qInfo() << "Controller waiting for robot pose in room frame";
        update_display(std::nullopt,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        return std::nullopt;
    }

    update_base_speed(*robot_pose, timestamp_ms);   // base speed for the contract stillness gate
    overlay_now_ms_ = timestamp_ms;                 // overlay dead-reckoning target + base time
    overlay_lidar_ts_ms_ = obstacle_tracker.last_lidar_timestamp_ms();
    update_overlay_extrapolation(world_model, *robot_pose, timestamp_ms);

    ControllerPlanningStep step;
    step.robot_pose = *robot_pose;
    step.plan_origin = robot_pose->pos;

    if (manual_target_room_.has_value())
    {
        step.target.node_name = "mouse_target";
        step.target.room_pos = *manual_target_room_;
        current_target_room_ = step.target.room_pos;
        affordance_manager.clear_current();
        const bool use_snapped_manual_origin = manual_target_dirty_ && manual_target_origin_room_.has_value();
        if (use_snapped_manual_origin)
            step.plan_origin = *manual_target_origin_room_;
        step.target_changed = manual_target_dirty_ || !current_plan_.has_value();
        manual_target_dirty_ = false;
        last_target_info_.reset();
        active_target_id_ = 0;
        target_wait_logged_ = false;
        return step;
    }

    const auto target = world_model.read_target_in_room(timestamp_ms);
    if (!target.has_value())
    {
        if (!target_wait_logged_)
        {
            qInfo() << "Controller waiting for an affordance target in DSR";
            target_wait_logged_ = true;
        }
        current_plan_.reset();
        last_target_info_.reset();
        active_target_id_ = 0;
        current_target_room_.reset();
        affordance_manager.clear_current();
        update_display(robot_pose,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        return std::nullopt;
    }

    target_wait_logged_ = false;
    step.target = *target;

    // A producer's affordance viewpoint can land inside an obstacle footprint (its own object, or one
    // that grew/moved), which the planner can't reach → stall. Move ONLY the navigation target to the
    // nearest free point; the affordance's object/feedback (parent_node_id) is untouched, so the
    // contract still services the same object. Everything downstream (plan, arrival, display) then uses
    // the repaired target consistently.
    if (const auto safe = planner.repair_target(room_polygon_, inner_polygon_,
                                                obstacle_tracker.obstacle_polygons(), step.target.room_pos);
        safe.has_value() && (*safe - step.target.room_pos).squaredNorm() > 1e-6f)
    {
        qInfo() << "[controller] affordance target" << step.target.node_name.c_str()
                << "blocked → repaired to nearest free point ("
                << (*safe).x() << "," << (*safe).y() << ")";
        step.target.room_pos = *safe;

        // Re-aim the heading at the object now that the standpoint moved (the producer's yaw faced the
        // object from the original, now-discarded viewpoint).
        if (step.target.parent_node_id != 0)
            if (const auto obj = world_model.read_node_room_xy(step.target.parent_node_id, timestamp_ms);
                obj.has_value())
                step.target.yaw_rad = std::atan2(obj->y() - step.target.room_pos.y(),
                                                 obj->x() - step.target.room_pos.x());
    }

    current_target_room_ = step.target.room_pos;
    step.target_changed = !last_target_info_.has_value()
                       || !ControllerWorldModel::same_target_instance(*last_target_info_, step.target);
    last_target_info_ = step.target;
    active_target_id_ = target->node_id;
    return step;
}

bool ControllerSession::ensure_current_plan(const ControllerPlanningStep &step,
                                            RoomPathPlanner &planner,
                                            ControllerObstacleTracker &obstacle_tracker,
                                            rc::TrajectoryController &path_controller,
                                            ControllerMotionCommander &motion_commander,
                                            ControllerDisplay &display)
{
    if (step.target_changed || !current_plan_.has_value())
    {
        current_plan_ = planner.plan_path(room_polygon_,
                                          inner_polygon_,
                                          obstacle_tracker.obstacle_polygons(),
                                          step.plan_origin,
                                          step.target.room_pos);
    }

    if (!current_plan_.has_value() || current_plan_->room_path.empty())
    {
        qWarning() << "Controller could not produce a path to target" << step.target.node_name.c_str();
        update_display(step.robot_pose,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        return false;
    }

    if (step.target_changed || !path_controller.is_active())
    {
        path_controller.set_path(current_plan_->room_path);
        // Affordance targets carry a desired facing yaw (point AT the table); manual
        // mouse targets do not, so they keep the legacy stop-on-arrival behaviour.
        path_controller.set_goal_facing_yaw(step.target.from_affordance
                                                ? std::optional<float>(step.target.yaw_rad)
                                                : std::nullopt);
    }

    return true;
}

void ControllerSession::update_display(const std::optional<ControllerRobotPose> &robot_pose,
                                       ControllerDisplay &display,
                                       const ControllerObstacleVisuals &obstacle_polys,
                                       const ControllerPolygons &obstacle_rfe_points,
                                       int max_lidar_draw_points) const
{
    // Dead-reckoning of the DISPLAYED cloud + icon to "now" is computed once per cycle in
    // update_overlay_extrapolation (stored in overlay_icon_pose_ / overlay_correction_); here we
    // just apply it. The lidar buffer (obstacle detection) stays anchored at scan time.
    auto icon_pose = robot_pose;
    std::optional<Eigen::Affine2f> lidar_correction;
    if (robot_pose.has_value())
    {
        if (overlay_icon_pose_.has_value())
            icon_pose = overlay_icon_pose_;
        lidar_correction = overlay_correction_;
    }

    display.update(icon_pose,
                   room_polygon_,
                   inner_polygon_,
                   current_plan_,
                   obstacle_polys,
                   obstacle_rfe_points,
                   current_target_room_,
                   last_mppi_trajectories_,
                   last_mppi_average_trajectory_,
                   last_best_mppi_trajectory_idx_,
                   last_display_wp_index_,
                   max_lidar_draw_points,
                   lidar_correction);
}

void ControllerSession::execute_plan(const ControllerRobotPose &robot_pose,
                                     rc::TrajectoryController &path_controller,
                                     ControllerObstacleTracker &obstacle_tracker,
                                     ControllerMotionCommander &motion_commander,
                                     ControllerDisplay &display,
                                     rc::AffordanceManager &affordance_manager,
                                     const TimeSource &time_source)
{
    obstacle_tracker.refresh_temporary_lidar_obstacle(time_source(), robot_pose, path_controller);
    // Proactive: model anything the concept agents don't (throttled internally). Complements the
    // reactive blockage-driven creation below; both feed the same multi-instance tracker.
    obstacle_tracker.scan_for_unmodelled_obstacles(time_source(), robot_pose, path_controller);

    if (!current_plan_.has_value())
    {
        display.clear_robot_trajectory();
        clear_tracking_state();
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    // A lock-on micro-search in progress owns the base until it locks or gives up — bypass the path
    // follower so it isn't disturbed by goal_reached re-evaluation while we quasi-statically servo.
    if (lockon_.active())
    {
        if (step_lockon(motion_commander, time_source))
            finalize_reached(affordance_manager, path_controller, motion_commander, display);
        return;
    }

    const auto &boundary_polygon = inner_polygon_.empty() ? room_polygon_ : inner_polygon_;
    if (boundary_polygon.size() >= 3)
        path_controller.set_room_boundary(boundary_polygon);

    const auto control_output = path_controller.compute(robot_pose.as_transform());
    last_mppi_trajectories_ = control_output.trajectories_room;
    last_mppi_average_trajectory_ = control_output.average_trajectory_room;
    last_best_mppi_trajectory_idx_ = control_output.best_trajectory_idx;
    last_display_wp_index_ = std::max(0, control_output.current_wp_index);
    if (control_output.path_blocked)
    {
        clear_tracking_state();
        obstacle_tracker.create_temporary_lidar_obstacle(time_source(),
                                                         robot_pose,
                                                         control_output.blockage_center_room,
                                                         control_output.blockage_radius,
                                                         path_controller);
        current_plan_.reset();
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    if (control_output.goal_reached)
    {
        // Reached an affordance pose. Resolve its contract (type-level defaults + per-node
        // overrides). A Servo policy runs the lock-on micro-search to satisfy the contract's
        // completion predicate before consuming the target; Reach (and non-affordance targets like a
        // mouse click) finalise immediately.
        bool want_servo = false;
        if (params_ && params_->lockon_enabled && graph_
            && last_target_info_.has_value() && last_target_info_->from_affordance && active_target_id_ != 0)
        {
            if (const auto aff_node = graph_->get_node(active_target_id_); aff_node.has_value())
            {
                // Resolve the contract from the PARENT object's type (stable across affordance-node
                // renames on restart), not the affordance node name.
                std::string parent_type;
                if (last_target_info_->parent_node_id != 0)
                    if (const auto pn = graph_->get_node(last_target_info_->parent_node_id); pn.has_value())
                        parent_type = pn->type();
                active_contract_ = rc::affordance::read_contract(aff_node.value(), parent_type);
                feedback_node_id_ = active_contract_.feedback_node_id != 0
                                  ? active_contract_.feedback_node_id
                                  : last_target_info_->parent_node_id;   // default: the parent object
                want_servo = active_contract_.policy == rc::affordance::Policy::Servo
                          && feedback_node_id_ != 0;
            }
        }
        if (want_servo)
        {
            qInfo() << "[affordance]" << last_target_info_->node_name.c_str()
                    << "reached -> SERVO lock-on | feedback node" << feedback_node_id_
                    << "| goal clauses" << static_cast<int>(active_contract_.goal.size())
                    << "| timeout(ms)" << active_contract_.timeout_ms;
            lockon_.begin(time_source(), active_contract_.scalar_target,
                          active_contract_.stable_n, active_contract_.timeout_ms);
            path_controller.stop();
            if (step_lockon(motion_commander, time_source))
                finalize_reached(affordance_manager, path_controller, motion_commander, display);
            return;
        }
        qInfo() << "[affordance]" << (last_target_info_.has_value() ? last_target_info_->node_name.c_str() : "?")
                << "reached -> REACH (consume immediately)";
        finalize_reached(affordance_manager, path_controller, motion_commander, display);
        return;
    }

    if (!path_controller.is_active())
    {
        clear_tracking_state();
        motion_commander.stop_robot();
        return;
    }

    float adv_mps = control_output.adv;
    float side_mps = control_output.side;
    float rot_rps = -control_output.rot;
    motion_commander.apply_uncertainty_speed_limit(adv_mps, side_mps, rot_rps);

    const bool stalled_by_obstacle = control_output.blockage_detected_ahead
        && control_output.safety_guard_triggered
        && std::abs(adv_mps) < 5e-4f
        && std::abs(side_mps) < 5e-4f
        && std::abs(rot_rps) < 1e-3f;
    if (stalled_by_obstacle)
    {
        if (obstacle_tracker.create_temporary_lidar_obstacle(time_source(),
                                                             robot_pose,
                                                             control_output.blockage_center_room,
                                                             control_output.blockage_radius,
                                                             path_controller))
        {
            clear_tracking_state();
            current_plan_.reset();
            path_controller.stop();
            motion_commander.stop_robot();
            return;
        }
    }

    if (std::abs(adv_mps) < 5e-4f && std::abs(side_mps) < 5e-4f && std::abs(rot_rps) < 1e-3f)
    {
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    motion_commander.send_speed_command(adv_mps, side_mps, rot_rps);
}

// ─── Lock-on micro-search ─────────────────────────────────────────────────────

// Fill the normalised servo Reading from the contract-named feedback attributes (object-agnostic).
rc::LockOn::Reading ControllerSession::read_servo_reading(std::uint64_t feedback_node_id) const
{
    rc::LockOn::Reading r;
    if (!graph_ || feedback_node_id == 0)
        return r;
    const auto node = graph_->get_node(feedback_node_id);
    if (!node.has_value())
        return r;
    const auto &attrs = node->attrs();
    const auto &c = active_contract_;

    r.valid = c.valid_attr.empty();   // no gate attr ⇒ always valid
    if (!c.valid_attr.empty())
        if (const auto it = attrs.find(c.valid_attr); it != attrs.end())
            if (const auto v = rc::affordance::detail::attr_scalar(it->second)) r.valid = *v != 0.0f;
    if (!c.scalar_attr.empty())
        if (const auto it = attrs.find(c.scalar_attr); it != attrs.end())
            if (const auto v = rc::affordance::detail::attr_scalar(it->second)) r.scalar = *v;
    if (!c.err_vec_attr.empty())
        if (const auto it = attrs.find(c.err_vec_attr); it != attrs.end() && it->second.selected() == 3)
        {
            const auto &v = it->second.float_vec();
            if (v.size() >= 1) r.err_x = v[0];
            if (v.size() >= 2) r.err_y = v[1];
        }
    return r;
}

// Evaluate the contract's completion predicate against the current feedback-node attributes.
bool ControllerSession::goal_met(std::uint64_t feedback_node_id) const
{
    if (!graph_ || feedback_node_id == 0)
        return false;
    const auto node = graph_->get_node(feedback_node_id);
    if (!node.has_value())
        return false;
    // The look is "done" only when the contract clauses hold AND the base is quiet enough for a clean,
    // motion-free observation (no blur / pose smear). max_observe_*=0 → stillness not required.
    return rc::affordance::evaluate_goal(node.value(), active_contract_.goal) && robot_still();
}

void ControllerSession::update_base_speed(const ControllerRobotPose &pose, std::uint64_t timestamp_ms)
{
    constexpr float kTwoPi = 6.28318530718f;
    constexpr float kPosEps = 1e-3f;            // 1 mm — below this the pose value is "unchanged"
    constexpr float kThetaEps = 1e-3f;          // ~0.06°
    constexpr std::uint64_t kStaleMs = 400;     // no change for this long ⇒ treat the robot as stopped

    if (!prev_robot_pose_.has_value())
    {
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
        return;
    }

    const Eigen::Vector2f dpos = pose.pos - prev_robot_pose_->pos;
    const float dtheta = std::remainder(pose.theta - prev_robot_pose_->theta, kTwoPi);
    const bool changed = dpos.norm() > kPosEps || std::abs(dtheta) > kThetaEps;

    if (changed && timestamp_ms > prev_robot_ts_ms_)
    {
        // Velocity over the actual pose-CHANGE interval. The room←robot feed updates coarser than the
        // control loop, so per-cycle differencing would alternate real steps with zeros (bursty); we
        // only difference when the value moves, over the real elapsed time.
        const float dt = static_cast<float>(timestamp_ms - prev_robot_ts_ms_) * 1e-3f;
        if (dt > 1e-3f)
        {
            const Eigen::Vector2f v = dpos / dt;   // room-frame m/s
            const float w = dtheta / dt;
            // EMA so a single jittery sample doesn't spuriously trip the stillness gate / overlay.
            base_speed_lin_ = 0.5f * base_speed_lin_ + 0.5f * v.norm();
            base_speed_ang_ = 0.5f * base_speed_ang_ + 0.5f * std::abs(w);
            room_vel_.vx    = 0.5f * room_vel_.vx + 0.5f * v.x();
            room_vel_.vy    = 0.5f * room_vel_.vy + 0.5f * v.y();
            room_vel_.omega = 0.5f * room_vel_.omega + 0.5f * w;
        }
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
    }
    else if (timestamp_ms - last_pose_change_ms_ > kStaleMs)
    {
        // Pose value hasn't moved for a while ⇒ robot stopped: zero the velocity so the overlay
        // settles (and the pose-age clock restarts here, so a fresh pose isn't seen as ancient).
        base_speed_lin_ = 0.f;
        base_speed_ang_ = 0.f;
        room_vel_ = ControllerRoomVelocity{};
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
    }
}

void ControllerSession::update_overlay_extrapolation(const ControllerWorldModel &world_model,
                                                     const ControllerRobotPose &robot_pose,
                                                     std::uint64_t timestamp_ms)
{
    overlay_icon_pose_.reset();
    overlay_correction_.reset();
    if (!params_)
        return;
    if (!overlay_lidar_ts_ms_.has_value() || timestamp_ms <= *overlay_lidar_ts_ms_)
        return;

    const std::uint64_t gap_ms = timestamp_ms - *overlay_lidar_ts_ms_;   // lidar staleness (informational)
    // The room←robot value updates coarser than the lidar, so the cloud's anchor pose is as old as
    // the time since that value last changed — extrapolate by THAT, not just the lidar gap.
    const std::uint64_t pose_age_ms = timestamp_ms >= last_pose_change_ms_ ? timestamp_ms - last_pose_change_ms_ : 0;
    const float raw_dt = static_cast<float>(pose_age_ms) * 1e-3f;
    const float dt = std::min(raw_dt, std::max(0.f, params_->overlay_extrapolation_max_dt_s));
    const auto now_pose = extrapolate_room_pose(robot_pose, room_vel_, dt);

    // Apply the correction to the drawn cloud + icon only when extrapolating to "now". When the
    // cloud is drawn one frame old it is already exactly registered (the pose is bracketed, not
    // clamped), so the correction is left identity — pushing it forward would undo the exactness.
    // The diagnostics below run regardless, so the CSV still logs a baseline for A/B.
    if (params_->overlay_extrapolate_to_now && !params_->overlay_draw_one_frame_old)
    {
        overlay_icon_pose_ = now_pose;
        overlay_correction_ = now_pose.as_transform() * robot_pose.as_transform().inverse();
    }

    const Eigen::Vector2f disp = now_pose.pos - robot_pose.pos;
    const float dtheta = now_pose.theta - robot_pose.theta;

    // RT-staleness probe: the room←robot pose the tree returns at "now" vs at the lidar stamp. If
    // RTdelta≈0 while the robot moves, the upstream pose feed is stale/clamped — the real lag source.
    float rt_delta = -1.f;
    if (const auto pose_now_rt = world_model.read_robot_pose_in_room(timestamp_ms, std::nullopt);
        pose_now_rt.has_value())
        rt_delta = (pose_now_rt->pos - robot_pose.pos).norm();

    // Commanded (robot_ref_*) vs measured (robot_current_*) base velocity, read from the shared graph,
    // so the joystick/actuation sign can be checked against the robot's actual rotation. cmd_* is the
    // reference written by whoever commands (controller / joystick path); cur_* is the measured motion
    // (robot_concept writes it from the FullPose estimator). Same-sign rotation ⇒ consistent.
    float cmd_adv = 0.f, cmd_rot = 0.f, cur_adv = 0.f, cur_rot = 0.f;
    if (graph_)
    {
        if (const auto rid = world_model.graph_state().robot_id; rid != 0)
            if (auto robot_node = graph_->get_node(rid); robot_node.has_value())
            {
                cmd_adv = graph_->get_attrib_by_name<robot_ref_adv_speed_att>(*robot_node).value_or(0.f);
                cmd_rot = graph_->get_attrib_by_name<robot_ref_rot_speed_att>(*robot_node).value_or(0.f);
                cur_adv = graph_->get_attrib_by_name<robot_current_advance_speed_att>(*robot_node).value_or(0.f);
                cur_rot = graph_->get_attrib_by_name<robot_current_angular_speed_att>(*robot_node).value_or(0.f);
            }
    }

    // CSV (lazy-open + header, truncated each run), throttled to ~10 Hz.
    if (!params_->overlay_csv_path.empty() && timestamp_ms - overlay_csv_last_ms_ >= 100)
    {
        overlay_csv_last_ms_ = timestamp_ms;
        if (!overlay_csv_open_)
        {
            overlay_csv_.open(params_->overlay_csv_path, std::ios::out | std::ios::trunc);
            if (overlay_csv_.is_open())
                overlay_csv_ << "t_ms,lidar_ts,gap_ms,pose_age_ms,vx,vy,omega,extrap_dt_s,disp_m,dtheta_rad,RTdelta_m,"
                                "cmd_adv,cmd_rot,cur_adv,cur_rot\n";
            // Announce the resolved absolute path (it's a relative path → lands in the process CWD,
            // which is easy to miss), or the failure — so this is never silently a no-op again.
            std::error_code ec;
            const auto abs_path = std::filesystem::absolute(params_->overlay_csv_path, ec).string();
            std::cout << "[OverlayExtrap] CSV " << (overlay_csv_.is_open() ? "writing to " : "FAILED to open ")
                      << abs_path << std::endl;
            overlay_csv_open_ = true;
        }
        if (overlay_csv_.is_open())
        {
            overlay_csv_ << timestamp_ms << ',' << *overlay_lidar_ts_ms_ << ',' << gap_ms << ',' << pose_age_ms << ','
                         << room_vel_.vx << ',' << room_vel_.vy << ',' << room_vel_.omega << ','
                         << dt << ',' << disp.norm() << ',' << dtheta << ',' << rt_delta << ','
                         << cmd_adv << ',' << cmd_rot << ',' << cur_adv << ',' << cur_rot << '\n';
            overlay_csv_.flush();
        }
    }
}

bool ControllerSession::robot_still() const
{
    return rc::affordance::stillness_ok(base_speed_lin_, base_speed_ang_, active_contract_);
}

bool ControllerSession::step_lockon(ControllerMotionCommander &motion_commander, const TimeSource &time_source)
{
    const auto reading = read_servo_reading(feedback_node_id_);
    const bool met = goal_met(feedback_node_id_);
    const auto cmd = lockon_.update(reading, met, time_source());
    // Quasi-static: drive the (tiny, capped) nudge only during STEP; hold still while SETTLING/done.
    if (lockon_.active() && (cmd.adv != 0.0f || cmd.side != 0.0f || cmd.rot != 0.0f))
        motion_commander.send_speed_command(cmd.adv, cmd.side, cmd.rot);
    else
        motion_commander.stop_robot();
    if (lockon_.done())
        qInfo() << "[affordance] lock-on" << (lockon_.locked() ? "LOCKED" : "GIVE_UP")
                << "| node" << active_target_id_;
    return lockon_.done();
}

void ControllerSession::finalize_reached(rc::AffordanceManager &affordance_manager,
                                         rc::TrajectoryController &path_controller,
                                         ControllerMotionCommander &motion_commander,
                                         ControllerDisplay &display)
{
    if (graph_)
        affordance_manager.mark_reached(graph_);
    lockon_.reset();
    feedback_node_id_ = 0;
    clear_tracking_state();
    display.clear_robot_trajectory();
    current_plan_.reset();
    last_target_info_.reset();
    active_target_id_ = 0;
    current_target_room_.reset();
    manual_target_room_.reset();
    manual_target_origin_room_.reset();
    manual_target_dirty_ = false;
    path_controller.stop();
    motion_commander.stop_robot();
}

void ControllerSession::set_manual_target(const QPointF &point,
                                          ControllerWorldModel &world_model,
                                          ControllerObstacleTracker &obstacle_tracker,
                                          rc::AffordanceManager &affordance_manager,
                                          rc::TrajectoryController &path_controller,
                                          const TimeSource &time_source,
                                          const WakeCallback &wake_callback)
{
    manual_target_room_ = Eigen::Vector2f(static_cast<float>(point.x()), static_cast<float>(point.y()));
    current_target_room_ = manual_target_room_;
    manual_target_origin_room_.reset();
    if (world_model.graph_state().ready() || world_model.refresh_graph_state())
    {
        if (const auto robot_pose = world_model.read_robot_pose_in_room(time_source(), obstacle_tracker.last_lidar_timestamp_ms());
            robot_pose.has_value())
        {
            manual_target_origin_room_ = robot_pose->pos;
        }
    }
    affordance_manager.clear_current();
    manual_target_dirty_ = true;
    current_plan_.reset();
    last_target_info_.reset();
    active_target_id_ = 0;
    path_controller.stop();
    if (wake_callback)
        wake_callback();
}

void ControllerSession::clear_manual_target(rc::AffordanceManager &affordance_manager,
                                            rc::TrajectoryController &path_controller,
                                            ControllerMotionCommander &motion_commander,
                                            const WakeCallback &wake_callback)
{
    manual_target_room_.reset();
    current_target_room_.reset();
    affordance_manager.clear_current();
    manual_target_origin_room_.reset();
    manual_target_dirty_ = false;
    current_plan_.reset();
    last_target_info_.reset();
    active_target_id_ = 0;
    path_controller.stop();
    motion_commander.stop_robot();
    if (wake_callback)
        wake_callback();
}

void ControllerSession::stop(rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander)
{
    path_controller.stop();
    clear_tracking_state();
    motion_commander.stop_robot();
}