#include "controller_session.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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
    display.update(robot_pose,
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
                   max_lidar_draw_points);
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
    if (prev_robot_pose_.has_value() && timestamp_ms > prev_robot_ts_ms_)
    {
        const float dt = static_cast<float>(timestamp_ms - prev_robot_ts_ms_) * 1e-3f;
        if (dt > 1e-3f)
        {
            constexpr float kTwoPi = 6.28318530718f;
            const float lin = (pose.pos - prev_robot_pose_->pos).norm() / dt;
            const float ang = std::abs(std::remainder(pose.theta - prev_robot_pose_->theta, kTwoPi)) / dt;
            // EMA so a single jittery pose sample doesn't spuriously trip (or release) the gate.
            base_speed_lin_ = 0.5f * base_speed_lin_ + 0.5f * lin;
            base_speed_ang_ = 0.5f * base_speed_ang_ + 0.5f * ang;
        }
    }
    prev_robot_pose_  = pose;
    prev_robot_ts_ms_ = timestamp_ms;
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