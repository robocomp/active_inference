#pragma once

#include <QPointF>

#include <fstream>
#include <functional>
#include <limits>
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
                             ControllerDisplay &display,
                             const TimeSource &time_source);

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
    // Contract-driven Orient (Policy::Orient): rotate IN PLACE toward `target_yaw` (no navigation) until the
    // contract's completion predicate fires (a depth detection arrives) or the contract times out. Returns
    // true once finished (LOOKED or GIVE_UP); drives the base rotation directly. Owns the base while active.
    bool step_orient(const ControllerRobotPose &robot_pose, ControllerMotionCommander &motion_commander,
                     const TimeSource &time_source, float target_yaw);
    std::optional<std::uint64_t> orient_start_ms_;   // rotate-to-look start stamp (for the contract timeout)
    int                          orient_stable_ = 0;  // consecutive goal-met measurements (→ stable_n = looked)
    // Contract observation-stillness gate: track the base speed (finite-difference of the room-frame
    // robot pose → m/s, rad/s) and test it against the active contract's max_observe_vel/omega.
    void update_base_speed(const ControllerRobotPose &pose, std::uint64_t timestamp_ms);
    // Dead-reckon the displayed cloud + icon from the last lidar stamp to "now" (stores the icon pose
    // and the cloud correction), and append a lag-diagnostics row to the overlay CSV.
    void update_overlay_extrapolation(const ControllerWorldModel &world_model,
                                      const ControllerRobotPose &robot_pose,
                                      std::uint64_t timestamp_ms);
    bool robot_still() const;
    void finalize_reached(rc::AffordanceManager &affordance_manager,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display);

    // ── Physical-wedge recovery ──────────────────────────────────────────────────────
    // Debounce over a per-cycle wedge signal supplied by the caller. A wedge is a PREDICTION ERROR: the
    // robot commands translation but the base doesn't achieve it (execute_plan compares commanded vs
    // measured base speed), or there is no route at all (ensure_current_plan → always stalled). A robot
    // that IS moving as commanded — detour, slow nav, arrival rotation, a still-sliding creep — is not
    // stalled, so none false-fire. Returns true once `stalled_this_cycle` has held for stuck_confirm_ms.
    bool detect_stuck(bool pursuing, bool stalled_this_cycle, std::uint64_t now_ms);
    // Begin an escape: choose turn direction from side clearance, drop a temp obstacle at
    // the stuck spot, reset the plan, and record the start pose/time.
    void begin_escape(const ControllerRobotPose &robot_pose,
                      ControllerObstacleTracker &obstacle_tracker,
                      rc::TrajectoryController &path_controller,
                      std::uint64_t now_ms);
    // Step the active escape (owns the base). Reverses with a slight turn (or rotates in
    // place if the rear is blocked) until distance/time bound is met, then stops + clears.
    void step_escape(const ControllerRobotPose &robot_pose,
                     rc::TrajectoryController &path_controller,
                     ControllerMotionCommander &motion_commander,
                     std::uint64_t now_ms);
    void reset_stuck_state();

    const ControllerParams *params_ = nullptr;
    rc::LockOn lockon_;
    rc::affordance::Contract active_contract_;     // resolved contract of the affordance in lock-on
    std::uint64_t feedback_node_id_ = 0;           // node carrying the contract's feedback attributes
    std::shared_ptr<DSR::DSRGraph> graph_;
    // Base speed (room frame) for the contract stillness gate, plus the previous pose it differences.
    float base_speed_lin_ = 0.0f;                  // m/s   (EMA-smoothed)
    float base_speed_ang_ = 0.0f;                  // rad/s (EMA-smoothed)
    ControllerRoomVelocity room_vel_;              // room-frame base velocity (EMA), for overlay dead-reckoning
    std::uint64_t overlay_now_ms_ = 0;             // current compute time (overlay extrapolation target)
    std::optional<std::uint64_t> overlay_lidar_ts_ms_;  // last lidar stamp (overlay extrapolation base time)
    std::optional<ControllerRobotPose> overlay_icon_pose_;   // dead-reckoned robot pose for the icon
    std::optional<Eigen::Affine2f> overlay_correction_;      // room(now)←room(scan) for the cloud
    std::ofstream overlay_csv_;                    // per-cycle overlay-lag diagnostics
    bool overlay_csv_open_ = false;
    std::uint64_t overlay_csv_last_ms_ = 0;        // throttle for CSV rows
    std::ofstream proximity_csv_;                  // near-obstacle black box ("why didn't it react")
    bool proximity_csv_open_ = false;
    std::uint64_t proximity_csv_last_ms_ = 0;      // throttle for proximity CSV rows
    std::optional<ControllerRobotPose> prev_robot_pose_;   // last pose at which the value actually changed
    std::uint64_t prev_robot_ts_ms_ = 0;                   // timestamp of that change (velocity dt base)
    std::uint64_t last_pose_change_ms_ = 0;                // = prev_robot_ts_ms_; pose-value age reference
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
    // Rate limit for the "target is boxed in" warning — the condition persists as long as the target does.
    std::uint64_t last_unreachable_log_ms_ = 0;

    // Physical-stuck recovery state.
    std::uint64_t stuck_since_ms_ = 0;          // start of the current wedge window (0 = not wedged)
    bool          escape_active_ = false;       // an escape maneuver currently owns the base
    std::uint64_t escape_start_ms_ = 0;         // escape start time (for the time bound)
    Eigen::Vector2f escape_start_pos_ = Eigen::Vector2f::Zero();  // pose at escape start (distance bound)
    float         escape_turn_sign_ = 1.0f;     // +1 / −1: rotation direction during escape
    int           escape_count_ = 0;            // consecutive escapes (alternating fallback for turn dir)
};