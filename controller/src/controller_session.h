#pragma once

#include <QPointF>

#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#include "controller_display.h"
#include "controller_lockon.h"
#include "controller_mission.h"
#include "../../common/affordance_protocol/affordance_protocol.h"
#include "controller_motion_commander.h"
#include "controller_obstacle_tracker.h"
#include "controller_runtime_types.h"
#include "controller_world_model.h"
#include "grid_planner.h"
#include "route_follower.h"
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
                          ControllerObstacleTracker &obstacle_tracker,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display);

    std::optional<ControllerPlanningStep> build_planning_step(std::uint64_t timestamp_ms,
                                                              ControllerWorldModel &world_model,
                                                              ControllerObstacleTracker &obstacle_tracker,
                                                              rc::AffordanceManager &affordance_manager,
                                                              rc::TrajectoryController &path_controller,
                                                              ControllerMotionCommander &motion_commander,
                                                              ControllerDisplay &display);

    bool ensure_current_plan(const ControllerPlanningStep &step,
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

    // The session is where targets are arbitrated (mouse > mission > affordance), so the mission runner
    // lives here rather than in the worker — the alternative is a fourth party that has to be consulted
    // by everyone who asks "what are we driving to".
    rc::MissionRunner &mission() { return mission_; }
    // Smooth the selected mission against the SAME grid + footprint predicate the planner drives with,
    // so a smoothed route cannot contain a pose the planner would then refuse. Returns waypoints moved.
    int smooth_selected_mission();
    const rc::MissionRunner &mission() const { return mission_; }

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
    // `arrived_at` / `now_ms` let a running mission close out the leg it just finished and step to the
    // next waypoint. The affordance path ignores them.
    void finalize_reached(rc::AffordanceManager &affordance_manager,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display,
                          const Eigen::Vector2f &arrived_at,
                          std::uint64_t now_ms);

    // ── Physical-wedge recovery ──────────────────────────────────────────────────────
    // Debounce over a per-cycle wedge signal supplied by the caller. A wedge is a PREDICTION ERROR and
    // NOTHING ELSE: the robot commands translation and the base doesn't achieve a healthy fraction of it
    // (execute_plan compares commanded vs measured base speed). A robot that IS moving as commanded —
    // detour, slow nav, arrival rotation, a still-sliding creep — is not stalled, so none false-fire.
    // Returns true once `stalled_this_cycle` has held for stuck_confirm_ms.
    //
    // PLANNER FAILURE IS NOT A WEDGE and no longer reaches here. It used to (ensure_current_plan passed
    // stalled=true unconditionally), which produced a closed loop: no route → escape → escape ends → still
    // no route → escape, with the base pinned at the escape constants indefinitely. Reversing changes the
    // robot's position, which is the right response to being physically trapped and no response at all to a
    // planner that cannot return a path. That branch now HOLDS and reports instead.
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
    // ROUTE EVENTS. One row per route build / repair, appended immediately. These are RARE events whose
    // diagnostics used to exist only on stdout, where they scroll away and cannot be compared between
    // runs — a number you cannot read later is not a measurement.
    std::ofstream route_events_csv_;
    // Both geometries, side by side: the A* polyline as planned and the smoothed curve as driven.
    // A deviation NUMBER says how far they differ; this says WHERE, which is the question when the
    // driven path visibly does not follow the planned one.
    std::ofstream route_geom_csv_;
    bool route_geom_csv_open_ = false;
    int  route_event_id_ = 0;
    void log_route_geometry();
    bool route_events_csv_open_ = false;
    void log_route_event(const char *event, bool ok, std::uint64_t t_ms,
                         const rc::TrajectoryController &path_controller,
                         float window_m);

    std::ofstream proximity_csv_;                  // near-obstacle black box ("why didn't it react")
    bool proximity_csv_open_ = false;
    std::uint64_t proximity_csv_last_ms_ = 0;      // throttle for proximity CSV rows
    std::optional<ControllerRobotPose> prev_robot_pose_;   // last pose at which the value actually changed
    std::uint64_t prev_robot_ts_ms_ = 0;                   // timestamp of that change (velocity dt base)
    std::uint64_t last_pose_change_ms_ = 0;                // = prev_robot_ts_ms_; pose-value age reference
    ControllerPolygon room_polygon_;
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
    // Rate limit for the planner-failure HOLD message. Planner failure is deliberately NOT routed into the
    // stuck/escape reflex (reversing cannot fix a planner), so this line is the only signal that it happened.
    std::uint64_t last_no_route_log_ms_ = 0;

    // Grid planner with EXACT robot-footprint collision. Replaces the visibility graph: it rasterises the same
    // obstacle polygons once into a fixed grid, so cost is independent of polygon COUNT (measured on the real
    // apartment: 24 vs 960 polygons both build in 0.3 ms and plan in <1 ms, where the visibility graph needed
    // ~1.2e8 segment tests at 154 polygons and stopped returning). Collision is the authored footprint, not an
    // inflated obstacle, so the six stacked C-space margins collapse to one explicit safety_margin_m.
    rc::GridPlanner grid_planner_;
    rc::MissionRunner mission_;
    // CONTINUOUS ROUTE MODE. The whole mission as one arc-length curve; no per-waypoint target, no
    // arrival test, no per-waypoint replan. Built once when a mission starts.
    rc::RouteFollower route_;
    bool route_active_ = false;
    bool waypoint_mode_logged_ = false;
    std::uint64_t last_route_build_ms_ = 0;
    bool build_route(const ControllerRobotPose &robot_pose);
    // Why the last hop the planner refused was refused. GridPlanner computes eight distinct reasons
    // (goal not footprint-feasible at any heading / start inside an obstacle / enclosed / expansion cap
    // / outside the grid ...) and the route builder was discarding all of them, which left "no path"
    // indistinguishable from every other cause. Recorded, not just printed.
    std::string last_plan_failure_;
    // Speed ceiling from the route's own curvature, looking ahead far enough to brake for what is coming.
    // Returns v_cap unchanged when there is no route (a click target has no curve).
    float route_speed_limit(float v_cap, float a_decel) const;
    // ROUTE REPAIR. Set by the recovery reflexes (visible blockage, or a wedge that dropped a virtual
    // disc) and consumed in the continuous-route branch of ensure_current_plan. Without it those
    // reflexes are inert in route mode: they create the obstacle and reset current_plan_, but the route
    // branch re-installs the SAME curve, so the robot backs off and drives at the blocker again.
    bool          route_repair_pending_ = false;
    std::uint64_t last_route_repair_ms_ = 0;
    int           route_repair_count_ = 0;
    // Fit the C2 curve to ANY planned polyline — a click target and an affordance target deserve the
    // same smooth path a mission gets. Returns the polyline unchanged if smoothing is off or fails.
    ControllerPolygon smooth_plan(const ControllerPolygon &poly) const;
    bool          escape_active_ = false;       // an escape maneuver currently owns the base
    std::uint64_t escape_start_ms_ = 0;         // escape start time (for the time bound)
    Eigen::Vector2f escape_start_pos_ = Eigen::Vector2f::Zero();  // pose at escape start (distance bound)
    float         escape_turn_sign_ = 1.0f;     // +1 / −1: rotation direction during escape
    int           escape_count_ = 0;            // consecutive escapes (alternating fallback for turn dir)
};