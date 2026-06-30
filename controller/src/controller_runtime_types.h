#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../../common/robust_metrics/robust_metrics.h"
#include "room_path_planner.h"

using ControllerPolygon = std::vector<Eigen::Vector2f>;
using ControllerPolygons = std::vector<ControllerPolygon>;
using ControllerPathPlan = RoomPathPlanner::PathPlan;

enum class ControllerObstacleKind
{
    Object,
    Obstacle,
    Temporary
};

struct ControllerObstacleVisual
{
    ControllerPolygon polygon;
    ControllerObstacleKind kind = ControllerObstacleKind::Obstacle;
    std::string label;   // short tag drawn on the footprint: t_1/c_1/b_1 (objects), o_1/o_2 (obstacles)
};

using ControllerObstacleVisuals = std::vector<ControllerObstacleVisual>;

struct ControllerParams
{
    float clearance_m = 0.4f;
    float grid_resolution_m = 0.35f;
    float connection_radius_m = 1.2f;
    float waypoint_tolerance_m = 0.25f;
    float max_adv_speed_mps = 0.7f;
    float max_rot_speed_rps = 0.8f;
    float pos_gain = 1.2f;
    float rot_gain = 1.5f;
    bool interpolate_rt = true;
    // Dead-reckon the DISPLAYED lidar cloud + robot icon forward from the last lidar
    // timestamp to "now" using the measured base velocity, so the overlay tracks the robot
    // instead of trailing it by the lidar/pose latency. Display-only — the obstacle buffer
    // stays anchored at scan time.
    bool overlay_extrapolate_to_now = true;
    float overlay_extrapolation_max_dt_s = 0.4f;   // clamp the extrapolation horizon
    // Draw the cloud (and read the pose) one lidar frame OLD — query the RT at the PREVIOUS scan's
    // stamp, which room_concept has had a full frame to publish, so InterpolatedRT brackets it
    // exactly instead of clamping at the leading edge. Exact registration, no extrapolation, at the
    // cost of ~one lidar period (~60 ms) of overlay age. Takes precedence over the extrapolation
    // (which would push the exact past pose back toward now and defeat the point).
    bool overlay_draw_one_frame_old = true;
    // When non-empty, append per-cycle overlay-lag diagnostics to this CSV (for plotting the lag /
    // velocity / RT-staleness evolution). Empty = disabled. Relative → lands in the launch CWD; the
    // resolved absolute path is printed once on first write so it's findable.
    std::string overlay_csv_path = "overlay_lag_eval.csv";
    int max_lidar_draw_points = 600;
    std::string lidar_name = "lidar3D";
    // Zero-copy media plane (LiDAR). When lidar_use_media is true, the LiDAR point
    // cloud is drained from the DDS media plane instead of the DSR laser_* attrs.
    // The DDS domain + topic are NOT configured here: they are read from the media
    // descriptor JSON attribute authored by the producer on the lidar_name node, so
    // the consumer always uses the producer's actual (dedicated) domain. The
    // subscriber is created lazily, only once that node + descriptor exist.
    bool lidar_use_media = true;
    // Stream watchdog: if no fresh LiDAR frame arrives for this long while operating,
    // the controller enters a local emergency hold (stops the robot, waits for the
    // stream to recover) instead of planning on stale perception.
    int lidar_stall_timeout_ms = 2000;
    std::string target_edge_type = "target";
    float pose_xy_std_slow_m = 0.03f;
    float pose_xy_std_stop_m = 0.12f;
    float pose_theta_std_slow_rad = 0.04f;
    float pose_theta_std_stop_rad = 0.20f;
    float min_adv_speed_scale = 0.15f;
    float min_rot_speed_scale = 0.05f;
    float uncertainty_prediction_horizon_s = 0.4f;
    float pose_xy_std_growth_per_mps = 0.5f;
    float pose_theta_std_growth_per_rps = 1.0f;
    float adv_rotation_coupling_exponent = 0.5f;
    float temporary_obstacle_front_distance_m = 1.8f;
    float temporary_obstacle_half_width_m = 0.9f;
    float temporary_obstacle_cluster_margin_m = 0.35f;
    float temporary_obstacle_padding_m = 0.18f;
    float temporary_obstacle_occlusion_depth_m = 0.28f;
    RobustLossType temporary_obstacle_robust_loss = RobustLossType::Huber;
    float temporary_obstacle_robust_loss_scale_m = 0.08f;
    int temporary_obstacle_min_points = 12;
    int temporary_obstacle_history_scans = 5;
    std::uint64_t temporary_obstacle_ttl_ms = 3000;
    float temporary_obstacle_existence_init_log_odds = 1.6f;
    float temporary_obstacle_existence_min_log_odds = -4.f;
    float temporary_obstacle_existence_max_log_odds = 4.f;
    float temporary_obstacle_existence_remove_threshold_log_odds = -1.8f;
    float temporary_obstacle_existence_observation_bias = 0.30f;
    float temporary_obstacle_existence_support_gain = 0.06f;
    float temporary_obstacle_existence_remembered_gain = 0.02f;
    float temporary_obstacle_existence_weak_miss_penalty = 0.05f;
    float temporary_obstacle_existence_absence_penalty = 0.75f;
    float goal_clearance_relax_dist_m = 0.6f;
    float goal_obstacle_margin_m = 0.08f;
    float goal_clearance_min_ratio = 0.85f;
    float straight_speed_heading_threshold_rad = 0.08f;
    float straight_speed_clearance_margin_m = 0.20f;
    float straight_speed_min_goal_dist_m = 1.5f;

    // Affordance servo ("lock-on") executor — the HOW (gains/caps/timing). The WHAT/WHEN
    // (scalar_target, completion predicate, stable_n, timeout) is per-affordance and comes from the
    // affordance_protocol Contract. Off by default. See controller_lockon.h / affordance_protocol.h.
    bool  lockon_enabled         = false;
    float lockon_sweep_speed_mps = 0.12f;   // PRIMARY: distance-sweep advance speed
    float lockon_sweep_range_m   = 0.45f;   // oscillate ± this far from the arrival pose
    float lockon_offset_tol      = 0.15f;
    float lockon_k_yaw           = 0.8f;
    float lockon_max_yaw_rps     = 0.12f;
    float lockon_dither_yaw_rps  = 0.10f;
    float lockon_settle_ms       = 400.0f;
    float lockon_step_ms         = 400.0f;
    int   lockon_max_attempts    = 30;

    // Physical-stuck detection + reverse-and-turn escape. Distinct from the MPPI's
    // geometric path_blocked (which predicts the planned path runs through obstacles):
    // this fires when the base is COMMANDED to move but is NOT actually making progress
    // (wedged below the lidar plane, scraping a frame, wheel slip, oscillating in place).
    // Commanded vs measured (room-frame EMA base speed) over a window → escape maneuver:
    // reverse with a slight turn toward the side with more ESDF clearance, drop a temp
    // obstacle at the stuck spot, then replan. On by default.
    bool  stuck_recovery_enabled = true;
    float stuck_cmd_lin_eps      = 0.05f;   // m/s — commanded linear above this = "trying to move"
    float stuck_cmd_rot_eps      = 0.15f;   // rad/s — commanded rotation above this counts too
    float stuck_meas_lin_eps     = 0.02f;   // m/s — measured below this = "not moving"
    float stuck_meas_rot_eps     = 0.08f;   // rad/s
    float stuck_confirm_ms       = 1500.0f; // sustained no-progress before escape fires
    float escape_adv_speed_mps   = 0.15f;   // reverse speed (issued negative)
    float escape_rot_speed_rps   = 0.35f;   // slight turn rate during escape
    float escape_distance_m      = 0.30f;   // back up at most this far …
    float escape_max_ms          = 1500.0f; // … or this long, whichever comes first
    float escape_side_probe_m    = 0.50f;   // lateral ESDF probe for turn-direction choice
    float escape_rear_probe_m    = 0.45f;   // rear ESDF probe distance
    float escape_rear_min_m      = 0.30f;   // rear clearance below this → rotate-in-place, no reverse
};

struct ControllerGraphState
{
    std::uint64_t room_id = 0;
    std::uint64_t robot_id = 0;
    std::string room_name;
    std::string robot_name;

    bool ready() const
    {
        return room_id != 0 && robot_id != 0 && !room_name.empty() && !robot_name.empty();
    }
};

struct ControllerTargetInfo
{
    std::uint64_t node_id = 0;
    std::string node_name;
    Eigen::Vector2f room_pos = Eigen::Vector2f::Zero();
    float yaw_rad = 0.f;
    float epistemic_gain = 0.f;
    bool epistemic_pending = false;
    bool from_affordance = false;
    std::uint64_t parent_node_id = 0;   // object the affordance hangs from (carries feedback attrs)
};

struct ControllerRobotPose
{
    Eigen::Vector2f pos = Eigen::Vector2f::Zero();
    float theta = 0.f;

    Eigen::Affine2f as_transform() const
    {
        Eigen::Affine2f tf = Eigen::Affine2f::Identity();
        tf.translation() = pos;
        tf.linear() = Eigen::Rotation2Df(theta).toRotationMatrix();
        return tf;
    }
};

// Robot base velocity expressed in the ROOM frame (so extrapolation is a plain Euler step,
// no body-frame rotation needed). Measured by differencing consecutive room poses.
struct ControllerRoomVelocity
{
    float vx = 0.f;     // m/s
    float vy = 0.f;     // m/s
    float omega = 0.f;  // rad/s
};

// Constant-velocity dead-reckoning of a room pose over dt seconds.
inline ControllerRobotPose extrapolate_room_pose(const ControllerRobotPose &pose,
                                                 const ControllerRoomVelocity &vel,
                                                 float dt_s)
{
    ControllerRobotPose out;
    out.pos = pose.pos + Eigen::Vector2f(vel.vx, vel.vy) * dt_s;
    out.theta = pose.theta + vel.omega * dt_s;
    return out;
}

struct ControllerPlanningStep
{
    ControllerRobotPose robot_pose;
    ControllerTargetInfo target;
    Eigen::Vector2f plan_origin = Eigen::Vector2f::Zero();
    bool target_changed = false;
};

struct ControllerPoseUncertainty
{
    float xy_std_m = 0.f;
    float theta_std_rad = 0.f;
};