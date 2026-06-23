#include "controller_session.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <print>
#include <sstream>

void ControllerSession::set_params(const ControllerParams *params)
{
    params_ = params;
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
        if (graph_)
            affordance_manager.mark_reached(graph_);
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