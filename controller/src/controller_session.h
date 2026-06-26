#pragma once

#include <QPointF>

#include <functional>
#include <memory>
#include <optional>

#include "controller_display.h"
#include "controller_lockon.h"
#include "../../common/affordance_protocol/affordance_protocol.h"
#include "controller_motion_commander.h"
#include "controller_obstacle_tracker.h"
#include "controller_runtime_types.h"
#include "controller_world_model.h"
#include "room_path_planner.h"
#include "trajectory_controller.h"
#include "../../common/affordance_manager/affordance_manager.h"

class ControllerSession
{
public:
    using WakeCallback = std::function<void()>;
    using TimeSource = std::function<std::uint64_t()>;

    void set_params(const ControllerParams *params);
    void set_graph(std::shared_ptr<DSR::DSRGraph> graph);

    bool sync_world_state(std::uint64_t timestamp_ms,
                          ControllerWorldModel &world_model,
                          RoomPathPlanner &planner,
                          ControllerObstacleTracker &obstacle_tracker,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display);

    std::optional<ControllerPlanningStep> build_planning_step(std::uint64_t timestamp_ms,
                                                              ControllerWorldModel &world_model,
                                                              ControllerObstacleTracker &obstacle_tracker,
                                                              rc::AffordanceManager &affordance_manager,
                                                              RoomPathPlanner &planner,
                                                              rc::TrajectoryController &path_controller,
                                                              ControllerMotionCommander &motion_commander,
                                                              ControllerDisplay &display);

    bool ensure_current_plan(const ControllerPlanningStep &step,
                             RoomPathPlanner &planner,
                             ControllerObstacleTracker &obstacle_tracker,
                             rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander,
                             ControllerDisplay &display);

    void update_display(const std::optional<ControllerRobotPose> &robot_pose,
                        ControllerDisplay &display,
                        const ControllerObstacleVisuals &obstacle_polys,
                        const ControllerPolygons &obstacle_rfe_points,
                        int max_lidar_draw_points) const;

    void execute_plan(const ControllerRobotPose &robot_pose,
                      rc::TrajectoryController &path_controller,
                      ControllerObstacleTracker &obstacle_tracker,
                      ControllerMotionCommander &motion_commander,
                      ControllerDisplay &display,
                      rc::AffordanceManager &affordance_manager,
                      const TimeSource &time_source);

    void set_manual_target(const QPointF &point,
                           ControllerWorldModel &world_model,
                           ControllerObstacleTracker &obstacle_tracker,
                           rc::AffordanceManager &affordance_manager,
                           rc::TrajectoryController &path_controller,
                           const TimeSource &time_source,
                           const WakeCallback &wake_callback);

    void clear_manual_target(rc::AffordanceManager &affordance_manager,
                             rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander,
                             const WakeCallback &wake_callback);

    void stop(rc::TrajectoryController &path_controller,
              ControllerMotionCommander &motion_commander);

private:
    void clear_tracking_state();

    // Contract-driven servo ("lock-on") on reaching an affordance pose. Returns true once finished
    // (LOCKED or GIVE_UP); drives the base directly via the motion commander.
    bool step_lockon(ControllerMotionCommander &motion_commander, const TimeSource &time_source);
    rc::LockOn::Reading read_servo_reading(std::uint64_t feedback_node_id) const;
    bool goal_met(std::uint64_t feedback_node_id) const;
    // Contract observation-stillness gate: track the base speed (finite-difference of the room-frame
    // robot pose → m/s, rad/s) and test it against the active contract's max_observe_vel/omega.
    void update_base_speed(const ControllerRobotPose &pose, std::uint64_t timestamp_ms);
    bool robot_still() const;
    void finalize_reached(rc::AffordanceManager &affordance_manager,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display);

    const ControllerParams *params_ = nullptr;
    rc::LockOn lockon_;
    rc::affordance::Contract active_contract_;     // resolved contract of the affordance in lock-on
    std::uint64_t feedback_node_id_ = 0;           // node carrying the contract's feedback attributes
    std::shared_ptr<DSR::DSRGraph> graph_;
    // Base speed (room frame) for the contract stillness gate, plus the previous pose it differences.
    float base_speed_lin_ = 0.0f;                  // m/s   (EMA-smoothed)
    float base_speed_ang_ = 0.0f;                  // rad/s (EMA-smoothed)
    std::optional<ControllerRobotPose> prev_robot_pose_;
    std::uint64_t prev_robot_ts_ms_ = 0;
    ControllerPolygon room_polygon_;
    ControllerPolygon inner_polygon_;
    std::optional<ControllerPathPlan> current_plan_;
    std::optional<Eigen::Vector2f> current_target_room_;
    std::optional<Eigen::Vector2f> manual_target_room_;
    std::optional<Eigen::Vector2f> manual_target_origin_room_;
    std::optional<ControllerTargetInfo> last_target_info_;
    std::vector<ControllerPolygon> last_mppi_trajectories_;
    ControllerPolygon last_mppi_average_trajectory_;
    int last_best_mppi_trajectory_idx_ = -1;
    int last_display_wp_index_ = 0;
    bool manual_target_dirty_ = false;
    std::uint64_t active_target_id_ = 0;
    bool room_wait_logged_ = false;
    bool target_wait_logged_ = false;
};