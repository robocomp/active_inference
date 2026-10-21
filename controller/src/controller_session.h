#pragma once

#include <QPointF>

#include <algorithm>
#include <deque>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

#include "controller_display.h"
#include "controller_lockon.h"
#include "controller_affordance_view.h"
#include "controller_mission.h"
#include "controller_door_passage.h"
#include "motion_verdict.h"
#include "controller_anchor_servo.h"
#include "lidar_guard_buffer.h"
#include "doorway_gap.h"
#include "../../common/interaction_anchor/interaction_anchor.h"
#include "../../common/affordance_protocol/affordance_protocol.h"
#include "../../common/robot_capability/robot_capability.h"
#include "controller_motion_commander.h"
#include "controller_obstacle_tracker.h"
#include "controller_runtime_types.h"
#include "controller_world_model.h"
#include "grid_planner.h"
#include "route_failure_memory.h"
#include "route_follower.h"
#include "stall_judge.h"
#include "trajectory_controller.h"
#include "../../common/affordance_manager/affordance_manager.h"
#include "local_policy/goal_adapter.h"
#include "local_policy/navigator.h"
#include "local_policy/unseen_contact.h"

class ControllerSession
{
public:
    using WakeCallback = std::function<void()>;
    // ── THE ROUTE THE ROBOT IS FOLLOWING HAS BEEN REPLACED ───────────────────────────────────────
    // Fired when the curve is re-authored: a full build (start of a run, or the route optimiser's
    // pass over the whole tour) and every in-place repair or reinstatement of a SECTION.
    //
    // ★WHY THE DISPLAY WANTS IT. The driven trace is the one overlay that accumulates, and its whole
    // meaning is "where the robot went on the route it is on". Re-author the route and that trace
    // becomes a picture of a curve nobody is driving any more, sitting on the canvas beside the one
    // that replaced it — two trajectories where there should be one, and more with every repair.
    //
    // ★A CALLBACK RATHER THAN A DISPLAY REFERENCE, because the session must not acquire one: this is
    // the only thing it would use it for, drive_mission_route (where both repair sites live) is
    // deliberately display-free, and threading a ControllerDisplay through it to erase a line would
    // put a GUI type in the middle of the control path. The worker owns the policy; the session only
    // reports the event.
    using RouteChangedCallback = std::function<void()>;
    void set_route_changed_callback(RouteChangedCallback cb) { route_changed_cb_ = std::move(cb); }
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

    // Dispatcher: what both modes share, then hand off. Neither helper below serves the other's mode.
    bool drive_mission_route(const ControllerPlanningStep &step,
                             rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander,
                             const TimeSource &time_source);
    bool drive_point_target(const ControllerPlanningStep &step,
                            ControllerObstacleTracker &obstacle_tracker,
                            rc::TrajectoryController &path_controller,
                            ControllerMotionCommander &motion_commander,
                            ControllerDisplay &display,
                            const TimeSource &time_source);
    // ── DOOR CROSS ON THE INTERACTION ANCHOR (2026-10-18) ────────────────────────────────────────────────────
    // While `aff_<door>_cross` is executed, door_concept hangs `door_N@robot` from the robot: the opening's pose in
    // the ROBOT frame. update_cross_anchor reads it (via the affordance's interaction_node_id, else the latched
    // anchor id once the affordance node is gone) into anchor_cross_; drive_point_target then skips the room route
    // and execute_plan drives the base from anchor_cross_command through step_anchor_cross — the same guard chain
    // (live clearance ceiling, uncertainty limiter, stall judge → escape) every other command passes.
    bool update_cross_anchor(const ControllerPlanningStep &step);
    struct AnchorCrossReading
    {
        std::uint64_t anchor_id = 0;
        std::string name;                       // e.g. door_1@robot
        rc::interaction::RelPose pose;          // the opening in the robot frame (+ covariance), as last PUBLISHED
        float width_m = 0.f;
        bool fresh = true;                      // Task 9: false = older than kAnchorStaleMs — wait for it, never drive on it
        std::int64_t age_ms = 0;                // now − its newest stamp
        int unseen_ms = 0;                      // fix round 1: the owner's last sighting this long ago (> 0 = PREDICTED pose)
        std::vector<rc::interaction::AnchorLeaf> leaves;   // Task 12 fix A: the swung leaf hypotheses, anchor frame (empty = none)
    };
    // The anchor of a `_cross` affordance (its interaction_node_id, else the anchor latched for that same affordance
    // name): fresh, or — with accept_stale (Task 9) — the stale one it last published (fresh = false), or nullopt.
    // Pure read; no state changes, no log.
    std::optional<AnchorCrossReading> read_cross_anchor(std::uint64_t affordance_id,
                                                         const std::string &affordance_name,
                                                         bool accept_stale = false) const;
    void step_anchor_cross(const ControllerRobotPose &robot_pose,
                           rc::TrajectoryController &path_controller,
                           ControllerObstacleTracker &obstacle_tracker,
                           ControllerMotionCommander &motion_commander,
                           ControllerDisplay &display,
                           rc::AffordanceManager &affordance_manager,
                           const TimeSource &time_source);
    // clearance_speed_limit's braking model along a robot-frame straight line (the servo's own direction of travel).
    float anchor_clearance_speed_limit(const Eigen::Vector2f &dir_robot, float v_cap,
                                       const rc::TrajectoryController &path_controller) const;
    // The braking deceleration both ceilings use: the tracker's cbf_max_decel, never more than the base delivers.
    float anchor_brake_decel(const rc::TrajectoryController &path_controller) const;
    // Task 14: the live returns in the ROBOT frame (the pose the field is built on), self-filtered by the BODY FOOTPRINT
    // grown by the range noise (lidar_body_guard.h) — not the ESDF's 0.405 m disc — as guard obstacles within `reach`.
    // The guard's whole input for one cycle: the returns plus the numbers its laws use, so every caller (the cross, the
    // tracker's command path, the escape, the census) applies the SAME net with the same constants.
    struct LiveReturns
    {
        rc::LidarGuardReturns r;
        float margin = 0.f, a_dec = 0.f, v_max = 0.f, w_max = 0.f, sigma = 0.f, reach = 0.f;
        int capture_self = 0;   // the newest sweep's returns dropped as the robot at capture (obstacle tracker, rigid frame)
        bool valid = false;   // a cloud was read
    };
    [[nodiscard]] LiveReturns live_returns(const ControllerRobotPose &pose, ControllerObstacleTracker &obstacle_tracker,
                                           const rc::TrajectoryController &path_controller) const;
    // Cap a command (robot-frame convention: side = +x right, rot CCW positive) by the live returns, unfloored.
    rc::LidarCapResult cap_by_returns(const LiveReturns &lr, float &adv, float &side, float &rot) const;
    // One census line (review item 4): what the net sees around the body right now.
    void log_returns_census(const char *when, const LiveReturns &lr) const;
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

    /// How the latest manual target ended — what a mission's `go_to` waits for. One epoch per set_manual_target, so
    /// a caller that remembers the epoch it started never reads the end of somebody else's target.
    enum class ManualTargetEnd { Pending, Reached, NoRoute, Cleared };
    struct ManualTargetStatus
    {
        std::uint64_t epoch = 0;
        ManualTargetEnd end = ManualTargetEnd::Pending;
    };
    [[nodiscard]] ManualTargetStatus manual_target_status() const { return manual_target_status_; }

    void clear_manual_target(rc::AffordanceManager &affordance_manager,
                             rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander,
                             const WakeCallback &wake_callback);

    void stop(rc::TrajectoryController &path_controller,
              ControllerMotionCommander &motion_commander,
              std::source_location loc = std::source_location::current());

    // ── Task 8: ONE LOCAL POLICY DRIVES EVERY MOTION (controller_policy_drive.cpp) ─────────────────────────────────────────
    // [Policy] Enabled: execute_plan hands the cycle to drive_with_policy BEFORE the escape / backoff / anchor servo /
    // lock-on / orient / band / tracker branches, which then do not run; ensure_current_plan skips escape and backoff. The
    // old chain is switched OFF, not deleted. The protocol halves (arrival → finalize_reached, the cross's through /
    // hand-back, the lock-on and orient completion predicates, Unreachable) are unchanged and shared.
    [[nodiscard]] bool policy_enabled() const { return params_ != nullptr and params_->policy_enabled; }
    // One policy tick toward `spec` (also the relative-motion path, SpecificWorker::step_relative_motion): builds the
    // input, keeps the Navigator's goal (set_goal only when the goal moved beyond its own tolerance), steps it, sends the
    // command, logs policy_decisions.csv and a [policy] line per mode change. `who` names the goal's owner in the logs.
    enum class PolicyVerdict { Driving, Arrived, Held, Unreachable };
    // ★ F1 (local_policy/reachability_belief.h): `stuck` is the tick's Stuck decision — a HOLD, never a hand back by itself;
    //   the verdict is Unreachable when the trapped field disconnects the goal OR the reachability belief hands it back
    //   (Navigator::Step::hand_back). Every hand-back site reads the verdict, nothing else.
    struct PolicyTick { PolicyVerdict verdict = PolicyVerdict::Held; std::string why; bool stuck = false;
                        float field_m = std::numeric_limits<float>::infinity();    // priced cost-to-go at the robot
                        float route_m = std::numeric_limits<float>::infinity(); };  // the cheapest route's metres (Live fix 3)
    // `also_free`: space the planner's map does not hold but the goal may be reached through (a doorway's measured corridor,
    // goal_adapter.h in_doorway_corridor); empty = the map alone. `jambs` (F4): the measured opening's jambs, whose contact
    // risk is added to the field's per-metre density over the corridor (doorway_corridor.h doorway_jamb_density); nullopt = none.
    PolicyTick policy_tick(const rc::policy::GoalSpec &spec, const std::string &who, const ControllerRobotPose &robot_pose,
                           ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                           std::uint64_t now_ms, std::function<bool(const Eigen::Vector2f &)> also_free = {},
                           std::function<bool(const Eigen::Vector2f &, float)> trap_veto = {},
                           std::optional<rc::policy::DoorwayJambs> jambs = std::nullopt);
    // The policy's state belongs to one goal owner: dropped on Stop / abort and when the owner ends.
    void policy_reset();
    // review I1: a cycle in which the policy did NOT drive although it owns the base (Pause / Halt / a no-step or hold / the
    // LiDAR-stall hold): the next tick drops the committed plan and the pending prediction unjudged.
    void policy_interrupt() { policy_gate_.on_hold(); }

    // ── WHY IS THE ROBOT NOT MOVING (motion_verdict.h) ── observability only: nothing that drives reads these.
    // The worker begin()s the evidence at the top of each control pass and selects the verdict at its end; every branch
    // of this class that ends a cycle without motion notes its cause here, with the numbers it decided on.
    [[nodiscard]] rc::why::Evidence &why() { return why_; }
    void note_why(rc::why::Cause c, std::string detail, std::source_location loc = std::source_location::current())
    {
        why_.note(c, std::move(detail), loc);
    }

    /// ABANDON the affordance being executed and let the planner choose the next one.
    /// The operator's override on the epistemic policy: the selector optimises expected information
    /// gain and cannot know that a particular look is hopeless — occluded, mis-posed, or simply not
    /// worth the wait — so there has to be a way to say so. It RETIRES the affordance (clears active +
    /// epistemic_pending on the node, exactly as a completed one) rather than merely dropping it: an
    /// affordance that is only released goes straight back into the pool and wins the very next
    /// selection, so the button would appear to do nothing. Returns false when nothing was running.
    bool skip_current_affordance(rc::AffordanceManager &affordance_manager,
                                 rc::TrajectoryController &path_controller,
                                 ControllerMotionCommander &motion_commander,
                                 ControllerDisplay &display,
                                 const TimeSource &time_source);

    /// ABORT: throw away the current activity entirely, leaving the session ready for a fresh Run.
    ///
    /// stop() halts the robot but keeps the route, the plan and the follower's progress, which is what
    /// a PAUSE wants. An abort must additionally drop the route, the current plan and any pending
    /// repair — otherwise the next Run re-installs the same curve at the same arc length and the robot
    /// carries on the old mission instead of starting the selected one.
    void abort(rc::TrajectoryController &path_controller,
               ControllerMotionCommander &motion_commander);

    // The session is where targets are arbitrated (mouse > mission > affordance), so the mission runner
    // lives here rather than in the worker — the alternative is a fourth party that has to be consulted
    // by everyone who asks "what are we driving to".
    // ROUTE mode: the tracker supplies its own turn rate from route curvature, so the curvature speed
    // limit must leave it headroom (see the note in route_speed_limit). Inert in PD/MPPI mode, which
    // have no feedforward to saturate.
    void set_route_tracker(bool active, float rot_headroom)
    { route_tracker_active_ = active; rot_headroom_ = std::clamp(rot_headroom, 0.1f, 1.0f); }

    void set_route_reverse(bool reverse) { route_reverse_ = reverse; }
    [[nodiscard]] bool route_reverse() const { return route_reverse_; }

    rc::DoorPassage &door_passage() { return door_passage_; }
    /// Mission (Command) mode: the open affordance contest runs only while an `explore` order holds it open.
    void set_command_explore(bool on) { command_explore_ = on; }
    rc::MissionRunner &mission() { return mission_; }
    // Smooth the selected mission against the SAME grid + footprint predicate the planner drives with,
    // so a smoothed route cannot contain a pose the planner would then refuse. Returns waypoints moved.
    int smooth_selected_mission();
    const rc::MissionRunner &mission() const { return mission_; }

    // ── Live tracking error, for the L-adaptation panel ────────────────────────────────────────────
    // The objective the panel plots, available in EVERY drive mode. It used to read MissionRunner's
    // summary directly, which is gated twice on things a click target and an affordance both lack — a
    // RUNNING mission, and a continuous ROUTE to measure deviation from — so the panel drew three flat
    // zeros through exactly the driving L is being adapted on. (Same absent-vs-zero confusion the
    // per-cycle CSV had before it was moved off the mission gate; see the note at the log site.)
    // The signal is the tracker's OWN cross-track (control_output.cross_track_m), the deviation from
    // whatever curve it is following, mission route or point plan alike.
    struct TrackingLive
    {
        float cross_rms_m = 0.f;
        float cross_max_m = 0.f;
        float rot_per_m   = 0.f;   // rotation effort per metre — the constraint series
        bool  on_mission  = false; // true while a mission runs: the numbers are then the GRADED ones
    };
    [[nodiscard]] TrackingLive live_tracking() const;

private:
    void clear_tracking_state();

    // Contract-driven servo ("lock-on") on reaching an affordance pose. Returns true once finished
    // (LOCKED or GIVE_UP); drives the base directly via the motion commander.
    bool step_lockon(const ControllerRobotPose &robot_pose, const rc::TrajectoryController &path_controller,
                     ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                     const TimeSource &time_source, bool drive_base = true);   // Task 8: false = protocol only
    // try104: the in-place turns and nudges that own the base (Orient, lock-on) go through the SAME live-returns net as
    // the tracker, the escape and the anchor servo — they used to reach the base uncapped. Sends what the net allows,
    // or stops when it holds the whole command.
    void send_through_returns(const ControllerRobotPose &robot_pose, const rc::TrajectoryController &path_controller,
                              ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                              float adv, float side, float rot, const char *who);
    rc::LockOn::Reading read_servo_reading(std::uint64_t feedback_node_id) const;
    // Perception producer's monotonic frame counter (mask_frame_id), or -1. See the definition.
    [[nodiscard]] int masks_frame_seq() const;
    bool goal_met(std::uint64_t feedback_node_id) const;
    // Contract-driven Orient (Policy::Orient): rotate IN PLACE toward `target_yaw` (no navigation) until the
    // contract's completion predicate fires (a depth detection arrives) or the contract times out. Returns
    // true once finished (LOOKED or GIVE_UP); drives the base rotation directly. Owns the base while active.
    bool step_orient(const ControllerRobotPose &robot_pose, const rc::TrajectoryController &path_controller,
                     ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                     const TimeSource &time_source, float target_yaw, bool drive_base = true);   // Task 8: false = protocol only
    // What the 2D view draws for a running Orient: the bearing asked for, and whether to draw it at
    // all. Latched where the program is built (which is the only place that knows the policy) and
    // pushed to the display beside the rest of the overlay.
    float orient_overlay_yaw_ = 0.f;
    bool  orient_overlay_visible_ = false;
    int  contract_for_epoch_ = -1;   // proposal the latched contract belongs to (see resolve_target_contract)
    int  affordance_epoch_ = -1;     // proposal the running view belongs to
    std::optional<std::uint64_t> orient_start_ms_;
    // The heading error the current turn STARTED from, so the chart can show a fraction rather than a
    // raw angle. Reset with orient_start_ms_, i.e. once per turn, not once per cycle.
    float orient_start_err_rad_ = 0.f;   // rotate-to-look start stamp (for the contract timeout)
    int                          orient_stable_ = 0;  // consecutive goal-met measurements (→ stable_n = looked)
    // Contract observation-stillness gate: track the base speed (finite-difference of the room-frame
    // robot pose → m/s, rad/s) and test it against the active contract's max_observe_vel/omega.
    void update_base_speed(const ControllerRobotPose &pose, std::uint64_t timestamp_ms);
    // Append a lag-diagnostics row to the overlay CSV (lidar staleness, pose age, RT ring state,
    // twist-correction and twist-accuracy columns). Diagnostics only — nothing here steers.
    // `rt_block_lead_ms` is ControllerObstacleTracker::rt_block_lead_ms() — signed ms by which the
    // newest room←robot RT block leads the registered scan. Logged beside RTdelta_m because it is
    // the column that disambiguates it (see update_rt_block_lead).
    void update_overlay_extrapolation(const ControllerWorldModel &world_model,
                                      const ControllerRobotPose &robot_pose,
                                      std::uint64_t timestamp_ms,
                                      const std::optional<std::int64_t> &rt_block_lead_ms,
                                      std::int64_t rt_twist_fix_dt_ms,
                                      const ControllerObstacleTracker &obstacle_tracker);
    bool robot_still() const;
    // `arrived_at` / `now_ms` let a running mission close out the leg it just finished and step to the
    // next waypoint. The affordance path ignores them.
    // ★WHICH CALLER completed it. Six call sites reach finalize_reached; three are outside the
    // arrival branch (lock-on, step_orient, teardown). mark_reached logs its own line, which is
    // inside finalize_reached and therefore identical for all six — useless for telling them apart.

    void finalize_reached(rc::AffordanceManager &affordance_manager,
                          rc::TrajectoryController &path_controller,
                          ControllerMotionCommander &motion_commander,
                          ControllerDisplay &display,
                          const Eigen::Vector2f &arrived_at,
                          std::uint64_t now_ms,
                          // A SKIP is not an arrival: there is no acquisition to inspect, so it must not
                          // arm the post-affordance dwell. Everything else about retiring the run is the
                          // same, which is why this is a flag rather than a second teardown path — two
                          // of those would drift apart and one of them would forget to clear something.
                          bool allow_dwell = true,
                          // Overrides the outcome derived from the contract. Used by the
                          // "already there" refusal, which is NOT an arrival and must not be
                          // reported to the producer as one.
                          std::optional<rc::affordance::Outcome> outcome_override = std::nullopt,
                          std::source_location floc = std::source_location::current());

    // ── Task 8: the policy drive (controller_policy_drive.cpp) ─────────────────────────────────────────────────────────
    void drive_with_policy(const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller,
                           ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                           ControllerDisplay &display, rc::AffordanceManager &affordance_manager, const TimeSource &time_source);
    // The ARRIVAL protocol of a point target (servo → lock-on, already-there, approach-only retry, REACH), extracted from
    // execute_plan so the tracker and the policy consume an arrival through the same code. Returns true when the cycle is
    // over (an arrival was consumed or a lock-on started). `drive_base` false: a lock-on started here does not move the
    // base (the policy drives its heading).
    bool consume_goal_reached(bool goal_reached, const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller,
                              ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                              ControllerDisplay &display, rc::AffordanceManager &affordance_manager, const TimeSource &time_source,
                              bool drive_base);
    // The cross's PROTOCOL half (step_anchor_cross's bookkeeping, shared with the policy): the measured gap, the near-side
    // gate, through / not-a-crossing / wait timeouts. `ended` = this cycle is over (finalized, handed back or stopped).
    struct CrossProtocol
    {
        bool ended = false, drive = false, on_gap = false;
        rc::interaction::RelPose drive_pose, pose;     // the pose driven on; the pose the guards / the wait-turn are asked about
        float drive_width = 0.f;
        std::string drive_name;
        std::vector<rc::interaction::AnchorLeaf> drive_leaves;
    };
    CrossProtocol cross_protocol(const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller,
                                 ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                                 ControllerDisplay &display, rc::AffordanceManager &affordance_manager, const TimeSource &time_source);
    void cross_hand_back(const std::string &why, bool timed_out, ControllerMotionCommander &motion_commander);
    static std::string contract_key_of(const DSR::Node &node);   // the affordance contract key of a parent object node
    void read_base_odometry(const ControllerWorldModel &world_model, std::uint64_t timestamp_ms);
    rc::policy::Navigator policy_nav_;
    std::optional<rc::policy::GoalSpec> policy_goal_;            // the goal the Navigator holds
    std::string policy_goal_who_;
    std::size_t policy_world_hash_ = 0;                          // the planner world the field was built on
    std::optional<rc::RobotFootprint> policy_body_;              // the BARE body (contact is against the bare hull)
    rc::policy::UnseenContact policy_unseen_;
    std::deque<rc::policy::SentCommand> policy_sent_;            // send history (s, steady) for the commands in force
    std::optional<rc::policy::Pose2> policy_odom_;               // odometry pose: integrated wheel speeds, never relocalised
    rc::policy::RoomFrameGuard policy_frame_;                    // Live fix 2: a room-frame change resets the policy
    rc::policy::PolicyTickGate policy_gate_;                     // review I1 / fix 2.1: holds and gaps interrupt the plan
    float odo_v_ = 0.f, odo_w_ = 0.f;                            // robot_current_advance / angular speed (CCW+), this cycle
    bool odo_valid_ = false;
    std::uint64_t odo_stamp_ms_ = 0;                             // robot_current_speed_timestamp (wall ms) — review M6
    rc::policy::OdoSignCheck policy_sign_;                       // review I5
    std::optional<float> policy_prev_yaw_;                       // review fix 2.2: the pose's last (time, heading) for I5
    std::optional<double> policy_prev_pose_t_;
    std::deque<rc::policy::OdoSample> policy_odo_hist_;          // odometry turn rate at its own times (steady s)
    int policy_last_mode_ = -1;
    std::string policy_last_why_;
    std::ofstream policy_csv_;
    std::uint64_t policy_no_field_since_ms_ = 0;
    // F1 fix round 1 (I4): an Unreachable verdict for the current policy goal (trapped field or reachability belief) is LATCHED
    // until the goal changes or the policy resets — the Navigator restarts its belief after a hand back, so the next tick is
    // no longer Unreachable, and it must not clear the no-route clock that reports a non-Reach target (Place / mouse)
    rc::policy::UnreachLatch policy_unreach_latch_;
    std::uint64_t policy_no_anchor_since_ms_ = 0;                // Live fix 4: a cross claim waiting for its anchor
    rc::policy::StaleAnchorWait policy_stale_wait_;              // Live fix 4: facing a never-published anchor
    // Task 8 ruling 1: a continuous-route tour driven by the policy — its waypoints as successive Place goals
    std::vector<rc::policy::RouteWaypoint> policy_route_wps_;
    int policy_route_laps_ = 1, policy_route_index_ = -1;   // -1 = no tour latched

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
    // Wedge = the robot was told to travel and did not GET anywhere. Judged on net displacement from
    // where the clock started, never on an instantaneous speed reading (pose jitter reads as speed).
    // Returns true ONLY for a WEDGE, which is the only verdict an escape answers. A ThrottleStall is
    // handled inside — reported and counted, never escaped — because the cure for "our limiter is
    // holding the base at 15%" is not a manoeuvre. See stall_judge.h for why the two must not be
    // conflated, and why judging on the post-throttle command alone made the second one invisible.
    // spin_is_not_a_wedge (Task 9, fix C): the caller's command is rotation-dominant and its own (the anchor servo's), so
    // a SPINNING verdict is reported and returned false — no escape, no hand-back. A Wedge still escapes.
    bool detect_stuck(bool pursuing, float asked_lin_mps, float cmd_lin_mps, float pose_sigma_m,
                      const Eigen::Vector2f &pos_room, float heading_rad, float commanded_rot_rps, std::uint64_t now_ms,
                      bool spin_is_not_a_wedge = false, bool returns_hold = false);
    void reset_stuck_window();
    // Begin an escape: choose turn direction from side clearance, drop a temp obstacle at
    // the stuck spot, reset the plan, and record the start pose/time.
    void begin_escape(const ControllerRobotPose &robot_pose,
                      ControllerObstacleTracker &obstacle_tracker,
                      rc::TrajectoryController &path_controller,
                      std::uint64_t now_ms);
    // Step the active escape (owns the base). Reverses with a slight turn (or rotates in
    // place if the rear is blocked) until distance/time bound is met, then stops + clears.
    // Task 14 fix round 4: one cycle of the returns' backoff on the tracker path. true = it commanded the base this cycle;
    // false = clearance is open (or nothing opens it): the backoff ended, the plan was dropped so the planner re-plans.
    bool step_returns_backoff(const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller,
                              ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                              const char *why);
    // Are real returns inside the band the net acts on (the robot hemmed in)?
    [[nodiscard]] bool hemmed_in_by_returns(const LiveReturns &lr) const;
    void step_escape(const ControllerRobotPose &robot_pose,
                     rc::TrajectoryController &path_controller,
                     ControllerObstacleTracker &obstacle_tracker,
                     ControllerMotionCommander &motion_commander,
                     std::uint64_t now_ms);
    void reset_stuck_state();

    const ControllerParams *params_ = nullptr;
    // ── AFFORDANCE PANEL (display only) ──────────────────────────────────────────────────────────
    // view_contract_ is a SEPARATE, display-only read of the same node. active_contract_ is still
    // resolved exactly where it always was (on arrival, in execute_plan) — moving that would change
    // when a policy takes effect, and this window is not worth a behaviour change.
    rc::AffordanceExecution affordance_view_;
    rc::affordance::Contract target_contract_;   // resolved when the target is SELECTED (see resolve_)
    bool target_contract_known_ = false;
    std::uint64_t contract_for_node_id_ = 0;     // which affordance target_contract_ belongs to
    [[nodiscard]] bool wants_final_facing(const ControllerTargetInfo &target) const;
    // Is a standpoint chosen earlier still usable against the map as it stands NOW? The residual field
    // fills in as the robot approaches, so an answer computed once is an answer frozen at the moment of
    // least information.
    [[nodiscard]] bool fix_still_good(const Eigen::Vector2f &pos, const ControllerTargetInfo &target) const;
    void resolve_target_contract(const ControllerTargetInfo &target);
    std::vector<std::string> affordance_recent_;
    // The protocol conversation. Bounded: this runs for the life of the agent, and an unbounded log
    // in a GUI struct is a slow leak that only shows up after an hour of driving.
    std::vector<rc::AffordanceExecution::ProtocolLine> affordance_transcript_;
    // ★ONE KEY PER CANDIDATE. A single string here was compared against EVERY candidate in the loop
    // below and then overwritten from the FIRST one, so any candidate that was not first mismatched
    // on every cycle and spoke on every cycle. Measured: 'afford_room on offer' at 5 Hz for the whole
    // run, which flushed the 200-line ring in ~20 s and buried every real transition in it.
    std::map<std::string, std::string> affordance_last_state_;   // node name -> last protocol state
    // Last line said by each SIDE. note_protocol used to dedup against the immediately previous entry
    // only, which two alternating speakers defeat completely — each differs from the other, so both
    // pass every cycle.
    std::map<int, std::string> protocol_last_by_side_;
    std::string affordance_last_target_;
    std::ofstream protocol_log_;
    bool          protocol_log_open_ = false;
    void note_protocol(rc::AffordanceExecution::ProtocolLine::Side side, std::uint64_t t_ms,
                       std::string text);
    std::uint64_t affordance_started_ms_ = 0;
    std::uint64_t affordance_step_since_ms_ = 0;
    std::string affordance_prev_step_;
    float affordance_nav_total_m_ = 1.f;
    // Draw whoever is commanding the base on the velocity panel, from the shared graph's robot_ref_*.
    // Only takes effect when our own output loop is silent — see ControllerDisplay::
    // update_velocity_trace_external. `kRefSpeedStaleMs` bounds how old a reference may be and still
    // count as a command: a leftover from a commander that has stopped is not one, and drawing it would
    // flat-line the panel at whatever was left behind instead of showing that nothing is driving.
    static constexpr std::uint64_t kRefSpeedStaleMs = 500;
    void feed_external_velocity_trace(const ControllerWorldModel &world_model,
                                      ControllerDisplay &display,
                                      std::uint64_t timestamp_ms);

    void update_affordance_view(const ControllerRobotPose &robot_pose,
                                const rc::TrajectoryController::ControlOutput &o,
                                bool output_enabled, float align_tol_rad, std::uint64_t now_ms);
    std::optional<float> feedback_scalar(std::uint64_t node_id, const std::string &attr) const;
public:
    [[nodiscard]] const rc::AffordanceExecution &affordance_view() const { return affordance_view_; }
private:

    rc::LockOn lockon_;
    rc::affordance::Contract active_contract_;     // resolved contract of the affordance in lock-on
    std::uint64_t feedback_node_id_ = 0;           // node carrying the contract's feedback attributes
    std::shared_ptr<DSR::DSRGraph> graph_;
    // Base speed (room frame) for the contract stillness gate, plus the previous pose it differences.
    float base_speed_lin_ = 0.0f;                  // m/s   (EMA-smoothed)
    float base_speed_ang_ = 0.0f;                  // rad/s (EMA-smoothed)
    bool  route_tracker_active_ = false;   // see set_route_tracker
    float rot_headroom_ = 0.70f;
    ControllerRoomVelocity room_vel_;              // room-frame base velocity (EMA), for overlay dead-reckoning
    std::uint64_t overlay_now_ms_ = 0;             // current compute time (overlay extrapolation target)
    std::optional<std::uint64_t> overlay_lidar_ts_ms_;  // last lidar stamp (overlay extrapolation base time)
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
    // ── WHAT THE PLANNER'S WORLD CONTAINS, OVER TIME ─────────────────────────────────────────────
    // With retina down, residual_concept's occupancy grid is the ONLY channel that can put a real
    // obstacle in front of A*, and nothing recorded whether it was doing so. See the definition.
    // Cloud-vs-icon pose pairing — see the block in build_planning_step.
    std::ofstream overlay_pairing_csv_;
    bool overlay_pairing_csv_open_ = false;
    std::uint64_t overlay_pairing_last_ms_ = 0;

    std::ofstream residual_world_csv_;
    bool residual_world_csv_open_ = false;
    std::uint64_t residual_world_last_ms_ = 0;
    void log_residual_world(std::uint64_t t_ms, const ControllerObstacleTracker &obstacle_tracker,
                            const ControllerRobotPose &robot_pose);
    // Fits the drawn cloud onto the room polygon and records the yaw error against omega — the
    // symptom itself, rather than another suspected cause of it. See the definition.
    std::ofstream cloud_fit_csv_;
    bool cloud_fit_csv_open_ = false;
    std::uint64_t cloud_fit_last_ms_ = 0;
    void log_cloud_wall_fit(std::uint64_t t_ms, ControllerObstacleTracker &obstacle_tracker,
                            const ControllerRobotPose &robot_pose);
    // WORLD SNAPSHOT at route-build time: the planner's raster, the waypoints (as recorded AND as
    // repaired), and every parameter the build consumed. It exists so the route can be rebuilt offline
    // by tools/route_bench against exactly this world — the route optimiser has weights whose effect is
    // GEOMETRIC, and measuring a geometric effect by driving one lap per weight costs a run apiece
    // against a 14.5%-noise reversal count. Written on every build; one file, last build wins.
    void dump_route_world(const Eigen::Vector2f &start,
                          const std::vector<Eigen::Vector2f> &raw,
                          const std::vector<Eigen::Vector2f> &repaired,
                          int laps,
                          const rc::RouteOptimizerConfig &opt) const;
    bool route_events_csv_open_ = false;
    // Rate limit for the deferred-waypoint re-offer (see drive_mission_route). ~1 Hz: the test is cheap,
    // but a recovery re-authors a route window and that is not.
    std::uint64_t last_reinstate_ms_ = 0;
    // The planner callbacks the route path hands to RouteFollower. ONE spelling: they were written out
    // character-for-character at the repair site and again at the reinstate site.
    rc::RouteFollower::PlanFn route_plan_fn();
    rc::RouteFollower::FreeFn route_free_fn();
    // THE INVARIANT after anything re-authors the curve under the follower: count it, tell the mission,
    // and make the follower re-install. It was five statements copied twice; a missed line means the
    // robot keeps driving the old curve, which is not a failure that announces itself.
    void on_route_reauthored(const char *event, float window_m, rc::TrajectoryController &path_controller,
                             std::uint64_t now_ms);
    void log_route_event(const char *event, bool ok, std::uint64_t t_ms,
                         const rc::TrajectoryController &path_controller,
                         float window_m);

    // THE PER-CYCLE CONTROL RECORD (tracker_diag.csv). One row per control cycle: the command issued,
    // the clearance it was issued against, the tracker's own Frenet error and arc length, the safety
    // gate's verdict, the pose it was computed from, and the actuation path's timing.
    // ★It began as the sampler's black box — temperature, cost spread, effective sample size, which
    // term owned the cost — to answer "is the optimiser choosing or averaging?". Those 17 columns went
    // with the sampler on 2026-09-09. What survives is the question that outlived it: a robot that
    // creeps looks identical from outside whatever the cause, so the cause has to be written down per
    // cycle. Cheap (a few floats at 10 Hz) and only while a mission is running.
    std::ofstream tracker_csv_;
    bool tracker_csv_open_ = false;
    void log_tracker_diagnostics(std::uint64_t t_ms, const rc::TrajectoryController::ControlOutput &o,
                              float commanded_adv, float measured_speed,
                              float path_kappa, float track_s, float measured_rot,
                              float pose_xy_std, float pose_theta_std,
                              const ControllerRobotPose &robot_pose,
                              const ControllerMotionCommander::OutputRateStats &ors,
                              float pose_stamp_age);
    float world_model_pose_stamp_age_ms_ = -1.f;

    // Closest the BODY has come to an obstacle this run; the cycle that beats it gets snapshotted.
    float tightest_cycle_clearance_ = std::numeric_limits<float>::max();
    // Sign of the last commanded rotation outside the deadband, and whether a reversal has already been
    // snapshotted this run (the first is enough; later ones would overwrite the file).
    int  prev_cmd_rot_sign_ = 0;
    bool reversal_captured_ = false;

    std::ofstream proximity_csv_;                  // near-obstacle black box ("why didn't it react")

    // ── FINAL-APPROACH BLACK BOX ─────────────────────────────────────────────────────────────────
    // The last metre into an affordance standpoint, and the terminal rotation. That rotation is driven
    // by TrajectoryController's align branch, which returns BEFORE every safety stage — no ESDF, no
    // footprint test — so nothing in the running system could say whether the robot had room to turn.
    // This records it per cycle instead of inferring it after the collision.
    std::ofstream approach_csv_;
    bool approach_csv_open_ = false;
    bool approach_active_ = false;                 // are we inside a logged approach right now
    std::string approach_target_;                  // which standpoint this approach belongs to
    float approach_min_clear_ = 0.f;               // tightest body clearance seen during the approach
    float approach_min_clear_align_ = 0.f;         // ... and during the rotation alone
    std::uint64_t approach_start_ms_ = 0;
    std::string approach_warned_;                  // standpoint we have already warned about
    void log_approach_diagnostics(std::uint64_t t_ms,
                                  const rc::TrajectoryController::ControlOutput &o,
                                  const ControllerRobotPose &robot_pose,
                                  const rc::TrajectoryController &path_controller);
    bool proximity_csv_open_ = false;
    std::uint64_t proximity_csv_last_ms_ = 0;      // throttle for proximity CSV rows
    std::optional<ControllerRobotPose> prev_robot_pose_;   // last pose at which the value actually changed
    std::uint64_t prev_robot_ts_ms_ = 0;                   // timestamp of that change (velocity dt base)
    // Stamped when the pose VALUE moves by more than an epsilon. Feeds the overlay's
    // pose_unchanged_ms column, which is time-since-the-estimate-moved and NOT pose latency —
    // it reads largest when the robot is nearly still. See the note at the computation.
    std::uint64_t last_pose_change_ms_ = 0;
    ControllerPolygon room_polygon_;
    std::optional<ControllerPathPlan> current_plan_;
    std::optional<Eigen::Vector2f> current_target_room_;
    std::optional<Eigen::Vector2f> manual_target_room_;
    std::optional<Eigen::Vector2f> manual_target_origin_room_;
    std::optional<ControllerTargetInfo> last_target_info_;
    int last_display_wp_index_ = 0;
    bool manual_target_dirty_ = false;
    ManualTargetStatus manual_target_status_;
    /// The manual target ended this way, if one was pending (the first ending is the one reported).
    void end_manual_target(ManualTargetEnd end)
    {
        if (manual_target_room_.has_value() and manual_target_status_.end == ManualTargetEnd::Pending)
            manual_target_status_.end = end;
    }
    std::uint64_t active_target_id_ = 0;
    // ★ "ALREADY THERE" IS NOT AN ARRIVAL. Set on the first cycle after adopting a target; if the
    // goal already reads reached on that cycle, no approach happened and nothing was observed —
    // consuming it as a completion is what produced the dwell loop (reach instantly, hold 3 s,
    // get re-offered, repeat, indistinguishable from a hang). Needs no distance threshold: the
    // question is whether an approach OCCURRED, which the first cycle answers exactly.
    bool target_is_new_ = false;
    bool room_wait_logged_ = false;
    // Latched on the MODE, not a bool: the idle notice must speak again after a mode change, and once
    // per process is how a Target-mode idle stayed invisible.
    std::optional<rc::DriveMode> target_wait_logged_mode_;
    bool target_wait_logged_ = false;

    // ── Post-affordance dwell (see ControllerParams::affordance_dwell_ms) ─────────────────────────
    // Set at finalize_reached; until it passes, build_planning_step holds the base still and does NOT
    // consult the affordance planner. Gating BEFORE select_target matters: merely refusing to drive
    // would still let the manager claim the next affordance and publish an execution edge, so the
    // graph would say the robot is servicing an affordance it has not started.
    std::uint64_t affordance_dwell_until_ms_ = 0;   // the CLOCK part of the wait
    std::uint64_t affordance_dwell_deadline_ms_ = 0;   // the hard bound on the whole wait
    bool          dwell_logged_ = false;
    // ── CONFIRMING LOOKS (see ControllerParams::affordance_dwell_mask_hits) ───────────────────────
    // How many separate producer frames have carried a mask of the affordance's object since the dwell
    // began. Counted off the "masks" node directly rather than through the panel's composer: the
    // composer only runs while the affordance WINDOW IS OPEN, and behaviour that depends on whether a
    // window is open is not behaviour, it is a coincidence.
    int           dwell_mask_hits_ = 0;
    int           dwell_last_mask_frame_ = -1;
    // Whether THIS dwell is waiting on an acquisition at all — decided once, when it is armed, from the
    // finished affordance's contract and whether its look succeeded. Latched rather than recomputed:
    // the contract is re-resolved as soon as the next target is picked, so asking again mid-dwell would
    // answer about a different affordance.
    bool          dwell_wants_mask_ = false;
    // WHAT counts as a confirming look, captured when the dwell is armed: the finished contract's own
    // completion clauses, and the feedback node they are evaluated on. Using the producer's own
    // detection predicate — the very signal the servo used to declare success — instead of re-deriving
    // "is my object in the masks" by matching a YOLO class label against a graph node NAME. That match
    // can never succeed for an affordance whose object is not named after a COCO class (a room, a
    // metaconcept), so the wait it gated was unsatisfiable BY CONSTRUCTION, not merely unlucky.
    std::vector<rc::affordance::GoalClause> dwell_goal_;
    std::uint64_t dwell_feedback_node_ = 0;
    std::uint64_t dwell_last_log_ms_ = 0;
    // Did the last affordance's look actually SUCCEED (locked / detection fired), as opposed to timing
    // out? Set by the two executors that know — step_lockon and step_orient.
    bool          last_look_succeeded_ = false;
    // Count the looks and report the outcome, so the log distinguishes "acquired" from "gave up on".
    void count_dwell_mask_hits();
    // What the dwell is showing off — the object of the affordance that just finished. Carried only so
    // the camera/masks view can highlight the masks that action was TAKEN TO ACQUIRE.
    std::string   dwell_object_;
public:
    // The object whose masks the panel should highlight: the live affordance's, or during the dwell the
    // one that just finished. Empty when neither. Read by the worker each cycle.
    [[nodiscard]] const std::string &attention_object() const { return attention_object_; }
    // The standpoint that object is being observed FROM — the affordance's own target pose. Mirrors
    // attention_object() exactly (live affordance, or the just-finished one through the dwell) so the
    // overlay can never mark a standpoint belonging to a different affordance than the object it
    // highlights. Empty when no affordance is in play, or when the target is a clicked point (which
    // designs no standpoint at all — it IS the point).
    [[nodiscard]] const std::optional<ControllerStandpoint> &attention_standpoint() const
    { return attention_standpoint_; }
    // Seconds left of the post-affordance dwell (0 when not dwelling) — shown in the affordance panel.
    // Two phases: the CLOCK floor, then the bounded wait for the acquisition. Once the clock has passed
    // and the dwell is still armed, the number that means something is how long the bound will keep
    // waiting for the mask — a countdown that hit zero while the robot was still standing there would
    // be the panel contradicting the robot.
    [[nodiscard]] float dwell_left_s(std::uint64_t now_ms) const
    {
        if (affordance_dwell_until_ms_ == 0) return 0.f;
        if (affordance_dwell_until_ms_ > now_ms)
            return static_cast<float>(affordance_dwell_until_ms_ - now_ms) / 1000.f;
        return affordance_dwell_deadline_ms_ > now_ms
                   ? static_cast<float>(affordance_dwell_deadline_ms_ - now_ms) / 1000.f : 0.f;
    }
    // Confirming looks counted so far, and how many are wanted (0 = the dwell is the clock alone).
    [[nodiscard]] int dwell_mask_hits() const { return dwell_mask_hits_; }

    /// Is the affordance at a step whose completion actually READS the masks?
    /// True during the servo lock-on, during an Orient glance, and through a dwell that is counting
    /// confirming looks — the three places the detection predicate is evaluated. False while claiming,
    /// navigating or aligning, where the masks are on screen but nothing is being decided by them.
    /// The affordance view uses it to decide whether to draw the YOLO ROIs at all: silhouettes shown
    /// during the drive are clutter over the one thing worth watching then (where the belief puts the
    /// object), and worse, they invite reading a detection as progress at a step that ignores it.
    [[nodiscard]] bool look_step_active() const
    {
        return lockon_.active() or orient_start_ms_.has_value()
            or (affordance_dwell_until_ms_ != 0 and dwell_wants_mask_);
    }
private:
    std::string attention_object_;
    std::optional<ControllerStandpoint> attention_standpoint_;
    // The standpoint of the target currently in force, or nothing when it is not an affordance's.
    // ONE derivation, called from every place attention_object_ is written, so the pair cannot drift.
    [[nodiscard]] std::optional<ControllerStandpoint> current_standpoint() const;
    // Mirror of AffordanceManager::suppressed_name(), pushed into the view. Held on the session because
    // the panel snapshot is assembled here and the manager is not reachable from the publish site.
    std::string suppressed_affordance_;
    // Names the branch that ended a cycle without sending a speed command (see the .cpp).
    void note_no_command(std::source_location loc = std::source_location::current());
    rc::why::Evidence why_;                 // this cycle's evidence (motion_verdict.h)
    rc::why::AffordanceMemory why_aff_memory_;   // the last sighting of each door-passage affordance, kept past deletion
    void observe_door_passage_why(std::uint64_t now_ms);   // passage note + memory, once per cycle (sync_world_state)
    // The name of what the robot is being driven to, for the "MOVING — to …" detail.
    [[nodiscard]] std::string why_target_name() const;

    // Rate limit for the "target is boxed in" warning — the condition persists as long as the target does.
    std::uint64_t last_unreachable_log_ms_ = 0;

    // ── HOW FAR OUR OWN REPAIR MOVED THE STANDPOINT, THIS CYCLE ──────────────────────────────────
    // The arrival test measures against the REPAIRED target, so a repair that relocates the goal onto
    // the robot satisfies it without the robot going anywhere: measured 2026-08-19, 140 of 163 accepted
    // arrivals were a median 2.86 m from the cell the producer published, each one reported to room as
    // SATISFIED. Carrying the displacement lets the arrival test ask the only question that matters —
    // am I at the standpoint that was ASKED FOR — while still allowing the honest few-centimetre repair.
    float last_repair_applied_m_ = 0.f;
    // Rate limit for "the repair would have substituted a different standpoint".
    std::uint64_t last_repair_reject_log_ms_ = 0;
    std::uint64_t last_pocket_dump_ms_ = 0;   // rate limit for unreachable_world.txt
    // ★THE CURRENT TARGET IS AN APPROACH, NOT THE STANDPOINT ROOM ASKED FOR. Set when the published
    // cell is unroutable and we drive to the closest reachable pose instead: the drive is real work,
    // but the completion must say `unreachable`, or room books an observation from a vantage the robot
    // never stood at. It is the difference between "I could not get there" and "I was there".
    bool target_is_approach_only_ = false;

    // Physical-stuck recovery state. The DECISION lives in rc::StallJudge (stall_judge.h) — pure, and
    // therefore testable without a graph or a robot, which is why the defect it now fixes could survive
    // here for as long as it did. This class only feeds it numbers and acts on the verdict.
    rc::StallJudge stall_judge_;
    // Throttle stalls: the robot brought to a standstill by our OWN speed limiter. Reported, never
    // escaped — reversing does not fix a limiter. Accumulated so the end-of-run line can say how much of
    // the run went this way, which is the number that would have made 2026-08-16 a glance.
    float stall_throttled_s_ = 0.f;
    int   stall_throttled_windows_ = 0;
    std::ofstream stall_events_csv_;
    bool  stall_events_csv_open_ = false;
    void  log_stall_event(const char *verdict, const rc::StallJudge::Report &r,
                          const Eigen::Vector2f &pos, float sigma, std::uint64_t t_ms);
    // Rate limit for the planner-failure HOLD message. Planner failure is deliberately NOT routed into the
    // stuck/escape reflex (reversing cannot fix a planner), so this line is the only signal that it happened.
    std::uint64_t last_no_route_log_ms_ = 0;
    std::uint64_t no_route_since_ms_    = 0;   ///< first cycle of the current no-route HOLD (0 = routing fine)
    std::uint64_t short_path_since_ms_  = 0;   ///< first cycle the path ENDED short of an affordance standpoint
    std::string   reach_unreachable_why_;      ///< non-empty ⇒ report this Reach standpoint Unreachable next cycle
    bool          route_dropped_ = false;      ///< an Unreachable was just reported: stop and drop the old route
    std::ofstream plan_track_csv_;             ///< plan_track.csv: per-cycle plan geometry + tracking errors
    /// A Reach standpoint (exploration: where exactly does not matter) the body cannot take — occupied, or wedged
    /// on the way in. Flag it; report_unreachable_if_no_route() tells the producer, which picks another.
    bool flag_reach_unreachable(std::string why)
    {
        if (not last_target_info_.has_value() or not last_target_info_->from_affordance) return false;
        if (not target_contract_known_ or target_contract_.policy != rc::affordance::Policy::Reach) return false;
        if (reach_unreachable_why_.empty()) reach_unreachable_why_ = std::move(why);
        return true;
    }
public:
    /// No route for > 5 s ⇒ report Unreachable to the affordance producer. Called on EVERY cycle the worker
    /// fails to plan (ensure_current_plan false returns before execute_plan) and from execute_plan.
    void report_unreachable_if_no_route(rc::AffordanceManager &affordance_manager, const TimeSource &time_source);
private:

    // Grid planner with EXACT robot-footprint collision. Replaces the visibility graph: it rasterises the same
    // obstacle polygons once into a fixed grid, so cost is independent of polygon COUNT (measured on the real
    // apartment: 24 vs 960 polygons both build in 0.3 ms and plan in <1 ms, where the visibility graph needed
    // ~1.2e8 segment tests at 154 polygons and stopped returning). Collision is the authored footprint, not an
    // inflated obstacle, so the six stacked C-space margins collapse to one explicit safety_margin_m.
    rc::GridPlanner grid_planner_;
    // Session totals: metres driven and wall time since startup, across every mission, target and
    // affordance (see the cycle). Distinct from MissionRunner's, which counts only while a mission runs.
    float session_distance_m_ = 0.f;
    std::optional<Eigen::Vector2f> odo_last_pos_;
    std::uint64_t odo_last_ms_ = 0;
    std::uint64_t session_start_ms_ = 0;

    // Log dedup: repair and unreachability are both DETERMINISTIC and re-evaluated every cycle, so an
    // unchanged answer must not reprint. Steady state should look steady in the log.
    std::optional<Eigen::Vector2f> last_repair_;
    std::string last_repair_name_;
    // The target ensure_current_plan could not route to. Reachability is GLOBAL, so the repair stage
    // cannot discover it on its own — this is how the search tells it to widen the question.
    // ── WHERE ROUTING FAILED, NOT WHICH NODE ─────────────────────────────────────────────────────
    // ★★★THIS WAS A std::string OF THE NODE NAME, SET ONCE AND NEVER CLEARED (fixed 2026-08-20).
    // Every standpoint room publishes rides the same node, `afford_room`, so the first routing
    // failure of the process latched "unroutable" onto the NODE and every later cell inherited it —
    // for the life of the agent. Measured: the controller reported `unreachable` for a cell 0.62 m
    // away that the same planner, replayed on the same captured world, routed to in 0.64 m with three
    // turning points; and the failure string it quoted was EMPTY, because the plan that actually
    // failed had happened minutes earlier and its message was long overwritten. Downstream this was
    // the engine of the phantom arrivals: routing_failed_here sent every target through
    // nearest_reachable, which relocated it onto the robot.
    // ★A ROUTE FAILURE IS A FACT ABOUT (goal, robot pose, map) — the same triple the map-verdict cache
    // is keyed on, and for the same reason: it cannot outlive any of the three. Quantised to 5 cm,
    // which is finer than the arrival band and coarser than a stationary robot's jitter.
    // ★★★AND NOT THE MAP EITHER (2026-10-19, try96 leg 3). The triple included world_hash(), and the
    // live obstacle layer changes the hash on every set_world — so the record written by a failed
    // plan() never matched the next cycle, nearest_reachable never ran, and a door approach with a
    // reachable pose 0.6 m short was reported Unreachable after the 5 s no-route liveness. Keyed now on
    // (node name, PRODUCER's raw pose, robot) — see route_failure_memory.h. A map that starts routing
    // again needs no key: plan() is always tried first, and nearest_reachable returns a reachable goal
    // exactly.
    rc::RouteFailureMemory route_failure_;
    // The soft/hard residual census line (sync_world_state): logged on change, at most every 10 s.
    int last_residual_soft_logged_ = -1, last_residual_hard_logged_ = -1;
    std::uint64_t last_residual_split_log_ms_ = 0;

    // ── THE RESOLVED STANDPOINT, HELD FOR THE LIFE OF THE TARGET ─────────────────────────────────
    // ★★★A TARGET THAT MOVES WHEN THE MAP MOVES IS NOT A TARGET (2026-08-20). The clearance search
    // (in-cell, then widened to the closest clear spot) is deterministic given a grid — but the grid
    // is rebuilt every cycle from a belief field that flickers, so re-running the searchevery cycle
    // returned a different pose: measured, one published cell resolved 89 times to 14 DISTINCT
    // standpoints spread over 0.99 m. same_target_instance calls a target changed at 5 cm, so every
    // one of those was a replan: a fresh curve, arc length back to zero, and a tracker that never got
    // past acquisition — 432 consecutive cycles with rotation saturated at the cap, alternating sign
    // every half second, and track_s pinned at 0.07 m. That is the "repaired target chased the robot"
    // failure this file already documents for unroutable_fix_, reintroduced by a newer search that
    // was not given the same defence.
    // ★HELD, BUT REVALIDATED: the cheap half of the question (is this pose still footprint-free, can
    // the body still turn there) is a local lookup, so it is asked every cycle; failing it drops the
    // hold and the search runs ONCE more. That is the difference between caching an answer and
    // freezing it.
    // ── WHICH CURVE IS THE ROBOT BEING MEASURED AGAINST? ─────────────────────────────────────────
    // ★Cross-track is the quantity every safety layer is computed against, and it is meaningless
    // across a REPLAN: the robot is on its new path and metres from the old one, which reads as an
    // excursion it never made. Measured 2026-08-20: a 1.02 m "departure" with the carrot 14 cm ahead
    // and DEAD AHEAD (bearing -0.2 deg), no rotation saturation, driving calmly at 0.33 m/s — a
    // tracker following something faithfully, and nothing in the log to say the something had just
    // been replaced. One counter settles it: a sample whose generation differs from its neighbour's
    // spans a path change and is not evidence of anything.
    std::uint32_t path_generation_ = 0;
    std::uint64_t last_carrot_collapse_log_ms_ = 0;   // rate limit for the collapse report

    std::optional<Eigen::Vector2f> resolved_standpoint_;
    Eigen::Vector2f                resolved_for_cell_ = Eigen::Vector2f::Zero();
    // The reachability repair, computed ONCE for a given raw standpoint and held. Recomputing it per
    // cycle makes the target follow the robot (nearest_reachable is measured FROM the robot).
    std::optional<Eigen::Vector2f> unroutable_fix_;

    // ── PER-CONSUMER POSE (see ControllerWorldModel's two readers) ───────────────────────────────
    // The freshest pose the RT tree holds, resolved once per cycle and given ONLY to the control law.
    // Everything else keeps the scan-aligned pose it already had, because the split is the point: the
    // tracker's e_y/e_psi are about where the robot IS, while anything reasoning about the obstacle set
    // alongside the instant it was captured wants the pose that goes with it. Empty ⇒ the tree had
    // nothing fresher and the scan-aligned pose is used, which is exactly the previous behaviour.
    std::optional<ControllerRobotPose> tracker_pose_;
    // How far the fresh pose sits from the scan-aligned one, i.e. the correction actually applied. This
    // is the number that says whether the split bought anything, and it costs one subtraction.
    float tracker_pose_lead_m_ = 0.f;

    // ── YAW-CORRECTION MONITOR (temporary; delete with GridPlanner::pose_free_legacy) ─────────────
    // The footprint used to be rasterised 90 degrees from its direction of travel. Correcting it moves a
    // boundary the whole stack stands on, and the effect is invisible from behaviour — a robot that no
    // longer scrapes looks exactly like a robot that never did. So it is MEASURED: once, over the whole
    // C-space, for how much room the correction took and gave; and then continuously, at the only two
    // poses the controller actually commits to — where the robot IS and where it is going — for how
    // often the two answers differ in practice. A change nobody can see the effect of is a change
    // nobody can defend.
    bool census_logged_ = false;
    long yawfix_cycles_ = 0;
    long yawfix_robot_now_blocked_ = 0;    // corrected refuses the robot's own pose; legacy allowed it
    long yawfix_robot_now_free_ = 0;       // ...and the reverse: room the correction gave back
    long yawfix_target_now_blocked_ = 0;
    long yawfix_target_now_free_ = 0;
    std::uint64_t yawfix_log_ms_ = 0;
    void monitor_footprint_orientation(const ControllerPlanningStep &step, std::uint64_t timestamp_ms);

    // ── FINAL-APPROACH RE-CHECK (see recheck_standpoint_on_approach) ─────────────────────────────
    // Where the live LiDAR moved the standpoint to, and which target it belongs to. HELD once taken,
    // for the same reason unroutable_fix_ is: a displacement recomputed from scratch every cycle
    // against a flickering cloud is a target that jitters, and a target that moves when the evidence
    // blinks is not a target. It is dropped the moment it stops being admissible, never on a whim.
    // ONE optional, not three members synced by hand. They were cleared in pairs at four sites and
    // approach_fix_anchor_ was cleared at none of them — a sync that was already partial. Bundled, every
    // one of those sites is a single .reset() and the inconsistency is unrepresentable.
    struct ApproachFix
    {
        Eigen::Vector2f pos;
        std::string name;      // the target it belongs to; the identity that survives a repair
        Eigen::Vector2f anchor;   // the standpoint it was derived FROM (see the anchor-moved guard)
        // ★THE HEADING THE FIX WAS JUDGED AT, frozen when it was taken. The admissibility of a FIXED
        // pose must not depend on where the observer currently stands, and it did: the re-test derived
        // the arrival heading from `centre - plan_origin`, so every metre the robot advanced rotated
        // the footprint at the held standpoint, swept a different rectangle, and pulled returns that
        // were outside the body inside it. The pose was then declared "no longer clear" and re-solved
        // FURTHER OUT, with nothing in the world having changed. Measured 2026-08-19 over 139 s:
        // 57 relocations, the standpoint fleeing 0.084 m per cycle against 0.027 m of robot closure —
        // 3.1x faster than the robot could close — 77% of them in the robot's own travel direction,
        // and NOT ONE arrival in the whole run (d_target never below 0.60 m, 0 `reached` rows).
        // A carrot on a stick, manufactured entirely by re-judging a static pose from a moving eye.
        float arrival_yaw;
        // ★AND THE FACING, FROZEN WITH IT. `aim_at_object` re-derives the commanded yaw from the
        // object's LIVE position every cycle, so freezing only the standpoint's position left the
        // pose half-static: the robot stood still and kept being told to face somewhere new.
        // Measured 2026-08-19 on the first run with the static standpoint: 0.73 rad of commanded
        // turning PER METRE travelled, |cmd_rot| at its cap on 70% of approach cycles, 6.3 whole
        // revolutions over one file, and not one `ALIGN` or `reached` row in 1478.
        // A pose is a position AND a heading. Freezing half of one is not freezing it.
        float object_yaw;
    };
    // `anchor` is the standpoint the fix was derived FROM. A producer is free to republish its viewpoint
    // somewhere else under the same name, and a fix that outlived the pose it repaired would silently
    // ignore that — the robot driving to a correction for a problem that has moved.
    std::optional<ApproachFix> approach_fix_;

    // ── AN APPROACH IS A COMMITMENT, AND THE PRODUCER DOES NOT GET TO REVOKE IT MID-FLIGHT ───────
    // ★★★MEASURED 2026-08-23, one 11.4 min affordance run: `afford_room`'s standpoint was republished
    // MORE THAN A METRE AWAY 39 times — a republish every 9.4 s, the closest pair 3.0 s apart, single
    // jumps of 2.2, 3.1, 4.0 and 4.5 m — and 36 of the 39 forced a whole new path generation. Every one
    // of the 74 target moves in that file happened with the robot INSIDE 1 m of the target, median
    // 0.318 m. `d_target_m` bottomed out at 0.253 m against a 0.25 m goal_threshold: the robot was
    // walked to within THREE MILLIMETRES of arrival and the goalposts moved, over and over.
    // ZERO of 2235 rows were inside the threshold and `goal_reached` was never once true.
    // ★THIS IS THE PRODUCER, NOT US, and approach_diag.csv's raw/tgt pair is what proves it — the
    // discriminator that file was built for. All 41 of the >1 m moves moved `raw_tgt` as well, so they
    // came down the wire; our own repair (fix_held) accounts only for the small ones.
    // ★SAME DISEASE AS ApproachFix ABOVE, ONE AGENT OVER. That comment records a standpoint fleeing
    // 0.084 m/cycle against 0.027 m of robot closure with not one arrival in 139 s, and the cure was to
    // FREEZE the pose being approached. It froze the half we author. This freezes the half we are told.
    // ★WHY A LATCH AND NOT A BETTER STANDPOINT. A viewpoint chosen for information gain is SATISFIED by
    // the robot arriving, so arriving is exactly what destroys it and the next-best-view legitimately
    // moves. Chasing it is a carrot on a stick: the sequence never terminates and the affordance is
    // never actually executed. A republish is not wrong — it is EARLY. It belongs to the next approach.
    // ★NOT A TIMER AND NOT A DISTANCE GATE. The commitment lasts exactly one approach: taken when a
    // standpoint is first accepted for an affordance node, released when that approach concludes
    // (arrival, refusal, unreachable, or the target being dropped — all of which clear
    // last_target_info_). The producer's newest offer is simply the one the NEXT epoch starts from.
    // ★IT DOES NOT BLOCK OUR OWN REPAIRS. It is applied where the raw pose lands, BEFORE the
    // reachability repair, the boxed-in search and recheck_standpoint_on_approach — so a committed
    // standpoint that turns out to be occupied or unroutable is still moved by the machinery that
    // exists for that, which reads the world rather than the wire.
    struct ApproachCommitment
    {
        std::uint64_t node_id = 0;          // the affordance this commitment belongs to
        int epoch = 0;                      // WHICH producer proposal this is — echoed on the wire
        Eigen::Vector2f room_pos{0.f, 0.f}; // the standpoint we accepted and will drive to
        float yaw_rad = 0.f;                // and the facing that came with it — a pose is both
        Eigen::Vector2f last_offer{0.f, 0.f};          // the most recent thing the producer published
        Eigen::Vector2f last_offer_counted{0.f, 0.f};  // the last offer the counter below has counted
        int deferred = 0;                   // republishes held off — REPUBLISHES, not cycles
        bool edge_seen = false;             // our `executing` edge has been observed in the graph
    };
    // ── WHY THE PIPELINE REFUSED TO PLAN THIS CYCLE ──────────────────────────────────────────────
    // ★ensure_current_plan() returning false is the one outcome with NO diagnosis attached, and it is
    // also the one that freezes the canvas: compute() returns on it WITHOUT updating the display, so
    // exactly when something has gone wrong the view stops telling you anything. Each false return now
    // names itself, and the worker prints it throttled.
    std::string hold_reason_;
    void note_hold(std::string why) { hold_reason_ = std::move(why); }

public:
    [[nodiscard]] const std::string &hold_reason() const { return hold_reason_; }
    // For the relative-motion executor (controller_relative_motion.h): the SAME footprint predicate the planner
    // plans with, and the room polygon it was rasterised from — both refreshed every cycle in sync_world_state.
    [[nodiscard]] const rc::GridPlanner &grid_planner() const { return grid_planner_; }
    [[nodiscard]] const ControllerPolygon &room_polygon() const { return room_polygon_; }
private:

    RouteChangedCallback route_changed_cb_;
    // One place to fire it, so a future re-authoring site cannot forget: everything that replaces the
    // curve already funnels through on_route_reauthored() or build_route().
    void note_route_changed() { if (route_changed_cb_) route_changed_cb_(); }

    std::optional<ApproachCommitment> approach_commit_;
    std::uint64_t approach_commit_log_ms_ = 0;   // rate limit for the deferral line
    // Applied where the producer's pose lands, before every repair. Returns nothing: it edits
    // `step.target` in place, which is the only way a caller can be sure it cannot be forgotten.
    void hold_approach_commitment(ControllerPlanningStep &step, std::uint64_t timestamp_ms);
    void apply_anchor_standpoint(ControllerPlanningStep &step, const ControllerRobotPose &robot_pose);   // Task 12 fix C
    struct AnchorStandpointLatch { std::uint64_t node_id = 0; int epoch = 0; Eigen::Vector2f room_pos = Eigen::Vector2f::Zero(); };
    std::optional<AnchorStandpointLatch> anchor_standpoint_latch_;   // fix round 1 (I-5): one fused standpoint per proposal

    std::uint64_t approach_blocked_log_ms_ = 0;   // rate limit for "found nowhere better"
    // The body used by the approach re-check, held rather than rebuilt per call (shadow() allocates).
    // ★THE ONE BODY. Built once from the mesh the graph names (controller_robot_body.h) and pushed into the
    // grid planner and the trajectory controller; every other consumer reads it from here. It was previously
    // a fresh RobotFootprint::shadow() temporary at eleven call sites, which is why a per-robot shape could
    // not be delivered at all: a static factory has nowhere to put "which robot".
    rc::RobotFootprint approach_body_ = rc::RobotFootprint::shadow();
    // Task 14 fix round 5: the returns net's two footprints (lidar_body_guard.h): the RETURNS body (planning hull ∪ the
    // mesh outline — what a return can hit) and the SELF footprint (hull ∪ mesh ⊕ σ_r — what the LiDAR sees of the robot).
    // Empty until the body loads ⇒ the planning hull.
    std::vector<Eigen::Vector2f> returns_body_, self_body_;
    // try104: the room-frame cells that held a WORLD return in the last sweep the net read — the evidence that keeps a wall
    // the robot is touching from being deleted as "the robot" (rc::lidar_guard_returns_from). Written by live_returns
    // (const: it is the net's memory of the sweep, not session state anyone else reads).
    mutable rc::LidarWorldMemory returns_world_memory_;
    bool tracker_turn_room_logged_ = false;   // try104: the tracker's turn is being given its room (logged on transitions)
    [[nodiscard]] const std::vector<Eigen::Vector2f> &returns_body() const
    {
        return returns_body_.empty() ? approach_body_.polygon() : returns_body_;
    }
    bool anchor_returns_binding_logged_ = false;   // Task 14: the returns' ceiling binds (logged on transitions)
    bool returns_hold_in_window_ = false;   // Task 14 fix 1: the live returns held the command in the open judge window
    bool returns_backoff_logged_ = false;   // Task 14 fix 1: the returns' backoff is driving (logged on transitions)
    bool startup_census_done_ = false;      // Task 14 fix 1: the parked start-up census line, once
    std::string anchor_census_for_;         // Task 14 fix 1: the cross whose first-cycle census was logged
    std::string anchor_band_logged_for_;    // Task 14 fix 3: the cross whose first band return was logged
    bool body_loaded_ = false;                 // one-shot guard: load when the robot node first appears
    // ★WHAT THIS BASE CAN DO, as opposed to what we have chosen to ask of it. Read once beside the body
    // (the robot node states both), and held rather than re-read because it is a property of the hardware,
    // not a signal. Every field is optional and absent means "the base did not say — keep our constant",
    // never zero. Nothing here clamps anything: see common/robot_capability/robot_capability.h on why the
    // audit reports and does not retune.
    rc::BaseCapability base_capability_;
    // Re-ask the standpoint against the LIVE return cloud once the robot is close enough for that cloud
    // to carry information about it, and move it if it is occupied by something the grid has forgotten.
    // Mutates step.target (position and, when it moves, the facing yaw that aims at the object).
    void recheck_standpoint_on_approach(ControllerPlanningStep &step,
                                        ControllerWorldModel &world_model,
                                        ControllerObstacleTracker &obstacle_tracker,
                                        const rc::TrajectoryController &path_controller,
                                        std::uint64_t timestamp_ms);

    // Drive the selected tour in REVERSE waypoint order. A property of the RUN, set when Run is
    // pressed, honoured by build_route — which reverses a local copy, so the recorded mission on disk is
    // never touched and the flag cannot leak into missions.toml.
    bool route_reverse_ = false;

    rc::MissionRunner mission_;
    rc::DoorPassage door_passage_;                                  // controller_door_passage.h
    std::string door_required_;                                     // its required affordance this cycle ("" = none)
    bool command_explore_ = false;                                  // Mission mode: an `explore` order is active
    std::optional<Eigen::Vector2f> passage_door_centre_;            // the passage's door node, current room frame
    std::vector<std::vector<Eigen::Vector2f>> door_passages_;       // its free corridor this cycle (room frame)
    std::optional<AnchorCrossReading> anchor_cross_;                // THIS cycle's anchor, fresh or WAITED for (nullopt = room route)
    // Task 9, fix A: while a Cross step is anchored it never falls back to the room route. engaged = this Cross step has
    // driven on a FRESH anchor (a stale one then means STOP and wait); before that, a named anchor is waited for while
    // turning toward its last pose (anchor_wait_pose_, carried by the robot's own turn since anchor_wait_theta_). The wait
    // is bounded by the cross contract's own timeout (target_contract_.timeout_ms), then handed back Unreachable — which
    // the door passage counts as an attempt.
    bool anchor_cross_engaged_ = false;
    std::uint64_t anchor_wait_since_ms_ = 0;                        // first stale cycle of the current wait (0 = none)
    std::int64_t anchor_wait_stamp_ = 0;                            // the stale pose's stamp anchor_wait_pose_ holds
    rc::interaction::RelPose anchor_wait_pose_;                     // that pose, in the robot frame at anchor_wait_theta_
    float anchor_wait_theta_ = 0.f;                                 // the robot's room heading when it was latched
    bool anchor_wait_blocked_ = false;                              // Task 11 fix 3: this wait's turn was judged SPINNING
    // Task 9, fix C: a SPINNING verdict on a rotation-dominant servo command is not a wedge; since when it has held.
    std::uint64_t anchor_spin_since_ms_ = 0;
    bool anchor_backoff_logged_ = false;                             // the jamb-deadlock backoff: log on change
    bool anchor_turn_room_logged_ = false;                           // try101: the servo's turn is making its room (log on change)
    bool anchor_leaf_logged_ = false;
    bool anchor_carried_logged_ = false;                             // Task 13: driving on the carried pose past the switch (log on change)
    bool anchor_through_ = false;                                    // Task 12 Fix E: the servo reached `through` in this claim
    std::optional<ControllerTargetInfo> clearing_target_;                // Task 12 Fix E: the cross being executed, kept past the room switch                                // Task 12 fix A: the anchor carries a leaf — log on change
    bool anchor_predicted_logged_ = false;                           // fix round 1: driving on a PREDICTED anchor: log on change
    // Fix round 1: the cross's engaged / wait / spin state belongs to ONE claim — reset when the target (node, epoch)
    // changes (resolve_target_contract) and when a claim of it is handed back (report_unreachable_if_no_route).
    void reset_anchor_cross_state();
    rc::StallVerdict last_judged_verdict_ = rc::StallVerdict::None;  // the stall judge's last closed window
    std::uint64_t anchor_latched_id_ = 0;                           // last anchor read, kept while the cross lasts
    std::string anchor_latched_for_;                                // the cross affordance it was read for
    int anchor_mode_logged_ = -1;                                   // -1 none, 0 room route, 1 anchor, 2 waiting: log on change
    // Task 7, fix 3: the anchor DRAWN on the 2-D view — set only by step_anchor_cross (the cross is engaged AND this
    // cycle read a fresh anchor), cleared at every cycle's start and whenever update_cross_anchor finds none.
    // update_display re-checks its age against kAnchorStaleMs, so a cycle that returns early never shows a stale one.
    std::optional<AnchorCrossReading> anchor_drawn_;
    // ── Task 18: THE DOORWAY AS THE LIDAR MEASURES IT (doorway_gap.h) ──────────────────────────────────────────────────
    // From the first cycle the through-span of the anchor's wall is SEEN, the cross steers on it (re-measured every cycle;
    // carried by the robot's own motion — room-frame pose deltas within one room — while it is not seen). The anchor stays
    // the prior until then and says where along which wall to look. Reset with every claim (reset_anchor_cross_state).
    std::optional<rc::interaction::RelPose> gap_line_;          // the gap's line, robot frame (anchor convention), seen or carried
    float gap_width_ = 0.f;                                     // its clear width as last MEASURED
    bool gap_seen_now_ = false;                                 // this cycle measured it (else carried)
    std::optional<ControllerRobotPose> gap_carry_pose_;         // the pose gap_line_ is expressed at
    std::string gap_carry_room_;                                // ...and its room frame
    bool gap_steering_logged_ = false, gap_lost_logged_ = false;
    // try89 fix: `through` needs the robot OBSERVED on the near side (s < −body half-length, on a seen gap or a seen anchor)
    // earlier in this claim; a claim that never was cannot complete as through.
    rc::CrossThroughGate cross_gate_;
    // try90 fix: the last anchor reading of this cross — once the anchor is REMOVED, a cross that holds a gap and has seen its
    // near side goes on with it (rc::cross_continues_without_anchor): the reading is kept only for its name and door width.
    std::optional<AnchorCrossReading> anchor_last_reading_;
    bool anchor_gone_logged_ = false;
    [[nodiscard]] bool gap_carries_cross(const std::string &cross_name) const
    {
        return anchor_latched_for_ == cross_name and anchor_last_reading_.has_value()
           and rc::cross_continues_without_anchor(gap_line_.has_value(), cross_gate_);
    }
    Eigen::Vector3f gap_logged_{0.f, 0.f, 0.f};                 // e, (unused), width at the last "[gap] seen" line
    float gap_logged_heading_ = 0.f;
    std::string current_room_name_;                             // the room frame robot poses are read in (this cycle)
    // try105 leg 5: the yaw the room polygon's wall gave the anchor's prior at acquisition (doorway_gap.h
    // doorway_gap_fuse_wall_yaw): the correction applied (rad) and the wall's yaw σ, for the "[gap] seen" line.
    float gap_room_yaw_fix_ = 0.f, gap_room_yaw_sigma_ = -1.f;
    // One gap cycle: carry, measure on the prior (the carried gap, else the anchor — its yaw fused with the room wall's), log.
    // `heading_var`: the localiser's heading variance (rad²; < 0 = unknown ⇒ no room yaw). True = steer on gap_line_.
    bool update_doorway_gap(const AnchorCrossReading &a, const ControllerRobotPose &pose, ControllerObstacleTracker &obstacle_tracker,
                            float heading_var);
    bool anchor_hoist_logged_ = false;                              // build_planning_step's bypass: log on change
    // CONTINUOUS ROUTE MODE. The whole mission as one arc-length curve; no per-waypoint target, no
    // arrival test, no per-waypoint replan. Built once when a mission starts.
    // The fitted curve for a NON-mission plan (affordance target, click target). Missions keep theirs
    // inside route_; this is the same object for everything else, retained for the same reason — so the
    // local elastic band can deform it every cycle instead of it being a one-shot fit.
    rc::RouteSpline plan_spline_;
    bool  plan_spline_valid_ = false;
    float plan_progress_s_ = 0.f;      // robot's arc length along plan_spline_, forward-only
    rc::RouteFollower route_;
public:
    [[nodiscard]] float session_distance_m() const { return session_distance_m_; }
    [[nodiscard]] float session_elapsed_s() const
    {
        return session_start_ms_ == 0 or overlay_now_ms_ < session_start_ms_
             ? 0.f : static_cast<float>(overlay_now_ms_ - session_start_ms_) * 1e-3f;
    }
private:
    // ── Local elastic band (see step_route_band / ControllerParams::band_*) ──
    // Deform the installed route in a window ahead of the robot against the live ESDF, every cycle.
    // Per-solve evidence lives in band_diag.csv (one row per ATTEMPT, so an inert band reads as inert);
    // there is deliberately no in-memory aggregate — it would be a second, unreadable copy of the file.
    // ── WHO OWNS THE CURVE BEING DRIVEN ──────────────────────────────────────────────────────────
    // ONE decision, made in ONE place, then CONSULTED. It used to be re-derived independently by
    // ensure_current_plan and by step_route_band from the same raw flags
    // (route_active_ && mission_.running() && route_.valid(), plus plan_spline_valid_ &&
    // plan_spline_.valid()) — two answers to one question, which is a disagreement waiting to happen.
    // It happened: the band deformed a curve in place while another path toggled that curve's validity,
    // and a consumer keyed on pointer identity saw the pointer alternate and reset itself every other
    // cycle. A mission route and a point-target plan are the SAME TYPE reached by different paths, so
    // nothing in the type system was ever going to catch that. This seam is what catches it: no consumer
    // may pick a curve, so no consumer can pick a different one.
    enum class DriveOwner { None, MissionRoute, PointPlan };
    struct DrivenCurve
    {
        DriveOwner owner = DriveOwner::None;
        const rc::RouteSpline *spline = nullptr;
        const std::vector<Eigen::Vector2f> *samples = nullptr;   // deformed in place; stays current
        std::size_t control_count = 0;
        [[nodiscard]] bool valid() const { return owner != DriveOwner::None and spline != nullptr; }
        [[nodiscard]] bool on_mission() const { return owner == DriveOwner::MissionRoute; }
    };
    [[nodiscard]] DrivenCurve driven_curve() const;

    void step_route_band(const DrivenCurve &curve, const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller);
    // Truncate band_diag.csv for THIS run, whether or not the band is enabled. Called before any early
    // return in step_route_band: a disabled run that never opens the file leaves the previous run's
    // rows at the fixed path for archive_on_stop to copy under this run's stamp.
    void ensure_band_csv(bool band_enabled);
    void log_band_diagnostics(std::uint64_t t_ms, const rc::RouteOptimizerReport &rep,
                              std::size_t freeze_before, std::size_t freeze_after, std::size_t ctrl_count);
    std::ofstream band_csv_;
    bool band_csv_open_ = false;
    long long band_cycle_ = 0;
    bool route_active_ = false;
    // ── Live tracking accumulator (see live_tracking) ──────────────────────────────────────────────
    // Deliberately NOT hosted in MissionRunner. Sampling that outside a run would fold idle affordance
    // driving into the stats the run is graded on, and its clearance vector — copied and SORTED by
    // summary() — would grow without bound between missions. This is three sums and a max, reset on
    // each rising edge of "the controller is driving", so every target gets its own curve.
    double live_ct_sq_sum_ = 0.0;
    double live_rot_effort_rad_ = 0.0;
    double live_dist_m_ = 0.0;
    std::size_t live_ct_n_ = 0;
    float live_ct_max_m_ = 0.f;
    bool live_active_ = false;
    std::optional<Eigen::Vector2f> live_last_pos_;
    std::uint64_t live_last_ms_ = 0;
    void sample_live_tracking(bool driving, const ControllerRobotPose &robot_pose, float cross_track_m,
                              float rot_rps, std::uint64_t now_ms);
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
    // ── SPEED FROM CLEARANCE: THE BRAKING-DISTANCE MODEL (2026-10-09) ─────────────────────────────
    // The plain tracker queries no field by design (it is built on PathWorld), so "slow down for what
    // the LiDAR sees" has to reach it as a CEILING, exactly as the curvature limit does. For a route
    // sample at arc distance ds ahead whose live clearance is c (robot-frame ESDF, 1-Lipschitz), the
    // body cannot touch anything before ds + (c - support), so v <= sqrt(2 a_dec (ds + c - support)).
    // The minimum over the braking horizon is the fastest speed from which the robot can still stop
    // short of the first certified contact. No gate and no cutoff: it is the same v_stop law the
    // tracker already applies to the END of the route, applied wherever the field says the route ends
    // early. Measured need (tracker_diag 2026-10-09 19:06:29-33): cmd_adv held at 0.35 while min_esdf
    // fell 1.34 -> 0.20 m, then 3.5 s of commanded 0.33 at zero measured speed before any verdict.
    // Until plain SLAM, the pose-covariance ramp did this job by accident (see TRACKER_REVIEW history
    // in config.toml: "the broken bound had been holding the system together"); with mm sigma it is
    // released, and nothing else bounds speed by what is in front of the robot.
    [[nodiscard]] std::optional<float> clearance_speed_limit(const DrivenCurve &curve,
                                                             const ControllerRobotPose &robot_pose,
                                                             const rc::TrajectoryController &path_controller) const;
    // Nearest live return to a probe point (ROOM frame), from the RAW buffer with the body hull as the
    // only self-filter. The ESDF deletes every return inside max(0.40, circumscribed + 0.08) = 0.405 m
    // of the centre (read_lidar_points_robot), so it is blind to exactly the obstacles a wedged body
    // is touching; the escape must not decide "may I reverse" from it.
    [[nodiscard]] float raw_return_clearance(const Eigen::Vector2f &probe_room, const ControllerRobotPose &pose,
                                             ControllerObstacleTracker &tracker) const;
    // ROUTE REPAIR. Set by the recovery reflexes (visible blockage, or a wedge that dropped a virtual
    // disc) and consumed in the continuous-route branch of ensure_current_plan. Without it those
    // reflexes are inert in route mode: they create the obstacle and reset current_plan_, but the route
    // branch re-installs the SAME curve, so the robot backs off and drives at the blocker again.
    bool          route_repair_pending_ = false;
    std::uint64_t last_route_repair_ms_ = 0;
    int           route_repair_count_ = 0;
    // Fit the C2 curve to ANY planned polyline — a click target and an affordance target deserve the
    // same smooth path a mission gets. Returns the polyline unchanged if smoothing is off or fails.
    // One place builds the optimiser's configuration; the mission route and every other planned path
    // both use it, so a clearance preference cannot apply to one and silently not the other.
    rc::RouteOptimizerConfig make_route_optimizer_config() const;
    // NOT const: it RETAINS the fitted spline so the band can keep deforming it. A plan that is only a
    // list of samples cannot be re-optimised — the control polygon is the decision variable, and
    // throwing it away is what made an affordance route a one-shot fit.
    ControllerPolygon smooth_plan(const ControllerPolygon &poly);
    // Task 14 fix round 4 (try79): the RETURNS' BACKOFF on the tracker path — owns the base while it opens clearance from the
    // live returns, then hands back to the planner (re-plan). `returns_held_since_ms_`: the net has held the tracker's whole
    // command since then (0 = not held). `returns_block_pending_`: the stall judge called a window BLOCKED BY A REAL RETURN.
    // `returns_backoff_failed_`: the last backoff found nothing that opens clearance — not re-triggered from the same hold
    // until the target changes or a plan drives again.
    bool          returns_backoff_active_ = false;
    std::uint64_t returns_held_since_ms_ = 0;
    bool          returns_block_pending_ = false;
    bool          returns_backoff_failed_ = false;
    bool          escape_active_ = false;       // an escape maneuver currently owns the base
    std::uint64_t escape_start_ms_ = 0;         // escape start time (for the time bound)
    Eigen::Vector2f escape_start_pos_ = Eigen::Vector2f::Zero();  // pose at escape start (distance bound)
    float         escape_turn_sign_ = 1.0f;     // +1 / −1: rotation direction during escape
    float         escape_rear_raw_m_ = 100.f;   // raw-return clearance behind the body at escape start
    int           escape_count_ = 0;            // consecutive escapes (alternating fallback for turn dir)
    // ── REJECT RATHER THAN STRUGGLE ──────────────────────────────────────────────────────────────
    // Escapes charged to the affordance they happened under, because escape_count_ is a session-lifetime
    // counter and says nothing about WHICH target the robot keeps failing to reach. Measured live: 12
    // escapes in 3 minutes, all against one refrigerator standpoint, while three other affordances sat
    // Offered — the robot achieving 2-19% of every commanded metre. An affordance the robot cannot
    // physically reach is worth abandoning, not grinding at; the gain it advertises is unreachable gain.
    std::uint64_t escapes_target_id_ = 0;       // which affordance the count below belongs to
    int           escapes_on_target_ = 0;
    std::uint64_t reject_affordance_id_ = 0;    // set by begin_escape, acted on where the manager is reachable
    std::string   reject_affordance_name_;
    // ── SPOTS THE ROBOT HAS ALREADY FAILED TO REACH ──────────────────────────────────────────────
    // Remembered by POSITION, not by affordance id: the producer is not told that the approach failed,
    // so it re-publishes the same standpoint the moment the suppression lapses — and a re-created node
    // would carry a new id anyway. The position is the thing that was actually useless.
    // ★NOT revalidated against the map, deliberately. These spots failed PHYSICALLY: the grid said
    // 0.31-0.38 m of clearance while the base achieved 2-19% of every commanded metre. pose_free() would
    // therefore call them fine and forget them instantly, and the robot would go back and wedge again.
    // The map is not the arbiter of a failure the map cannot see. They live for the session.
    // ★A CONSUMER MEMORY MUST NOT OUTLIVE THE PRODUCER'S. Without `when_ms` this was a PERMANENT
    // blacklist: each entry vetoes a 0.30 m disc for ever, coverage only grows, and the skip that reads
    // it costs no physical time — so once enough discs accumulate, offer→instant-skip→re-offer closes
    // into a loop at software speed. Measured 2026-08-19, completions per minute: +2,+2,+4,+2 for four
    // minutes (the physical rate), then +103, then ~+150 sustained — 1694 in 15 min, all Refused, robot
    // pinned, zero net closure. A phase transition, not a drift.
    // ★The bound is the PRODUCER's recovery time (room IorDecayTime = 120 s): a consumer memory that
    // outlives it permanently refuses what the producer legitimately re-offers, and the two can never
    // settle. It also restores the rule stated everywhere else here — "no blacklist; a cell refused
    // only because the robot happened to be parked on it comes back into play on its own".
    struct UselessSpot
    {
        Eigen::Vector2f pos = Eigen::Vector2f::Zero();
        std::string name;
        int hits = 0;
        std::uint64_t when_ms = 0;
    };
    std::vector<UselessSpot> useless_spots_;
    std::optional<Eigen::Vector2f> last_raw_target_pos_;   // as PUBLISHED, before our repair moved it
    std::uint64_t last_useless_log_ms_ = 0;
    // Written from build_planning_step, before any early return, so a freeze RECORDS ITSELF.
    // The candidates as CONVERSATION — producer offers and the selector's choice. Separate from the
    // JSONL dump below and deliberately unthrottled: see the note at its definition.
    void note_selection_protocol(std::uint64_t t_ms, const rc::AffordanceManager &affordance_manager);
    void log_selection_json(std::uint64_t t_ms, const rc::AffordanceManager &affordance_manager,
                            const std::optional<Eigen::Vector2f> &robot_xy, const char *stage);
    unsigned last_finalize_line_ = 0;   // which caller completed the affordance
    // ★★★THE STANDPOINT THE INSTANT-ARRIVAL BRANCH LAST COMPLETED. That branch fires on
    // `target_is_new_ && d <= goal_threshold`, and finalize_reached CLEARS last_target_info_ — which
    // is exactly what makes the next adoption look new. So completing re-arms the condition that
    // completes: measured 4522 calls at an unchanging d = 0.18 m, at full loop rate, while room's
    // actual offer sat 3.96 m away and the robot never moved. The completion is legitimate ONCE per
    // standpoint; the loop is it firing for the same one for ever.
    std::optional<Eigen::Vector2f> instant_completed_at_;
    std::ofstream select_json_;            // per-cycle "why is nothing being executed" record
    bool          select_json_open_ = false;
    std::uint64_t last_select_json_ms_ = 0;
    [[nodiscard]] const UselessSpot *known_useless_spot(const Eigen::Vector2f &pos,
                                                       std::uint64_t now_ms) const;
    void remember_useless_spot(const Eigen::Vector2f &pos, const std::string &name,
                               std::uint64_t now_ms);
};