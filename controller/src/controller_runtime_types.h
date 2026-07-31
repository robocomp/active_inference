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
    Temporary,
    GridOccupancy   // residual_concept occupancy-grid hulls: what the CONTROLLER treats as occupied space
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
    // THE standoff. Exactly one number now expresses "how much room do we want beyond the robot's actual
    // shape", replacing six independent C-space margins that were spread across three agents and summed to
    // ~0.95 m for a robot that physically passes 0.461 m. Because feasibility is the exact footprint test and
    // this is only a preference on top of it, tuning it for comfort can no longer make a reachable goal
    // unreachable — which is what the old stack could do, and did.
    float footprint_safety_margin_m = 0.05f;
    float clearance_m = 0.4f;
    float grid_resolution_m = 0.35f;
    float connection_radius_m = 1.2f;
    float waypoint_tolerance_m = 0.25f;
    float max_adv_speed_mps = 0.7f;
    float max_rot_speed_rps = 0.8f;
    float pos_gain = 1.2f;
    float rot_gain = 1.5f;
    // ── Fixed-rate velocity output (see controller_motion_commander.h) ──
    // The base is commanded from a dedicated thread at this period instead of from compute(), whose measured
    // cadence was median 108 ms but mean 212, p99 1.26 s and max 9.5 s — that variance IS the stutter. The
    // command is re-sent every tick even when unchanged (the old code skipped identical commands entirely, so
    // the base could hear nothing for seconds during steady driving).
    float velocity_output_period_ms = 50.f;
    // Authority of a command decays as 1/(1+(age/τ)²) so a stalled planner coasts to a stop rather than
    // driving blind on a stale command. τ well above the healthy cycle time ⇒ no penalty in normal operation
    // (≈0.95 at 108 ms), strong attenuation by ~1 s. Not a watchdog cutoff — a continuous precision term.
    float command_freshness_tau_ms = 500.f;
    // ── Output-stage acceleration limit ──
    // Measured on a live run: the commanded rotation reverses at nearly full scale inside one cycle
    // (+0.456 → −0.457 rad/s in 100 ms; |Δrot| > 0.1 rad/s on 12.7% of cycles, > 0.5 on 1.2%). A base cannot
    // follow a step like that — its own controller saturates and the chassis lurches, which is the stutter you
    // feel. Bounding the SLEW here converts the chatter into a signal the base can actually track, and because
    // the chatter is near-symmetric (±0.45) the limiter also averages it toward zero instead of executing both
    // extremes in turn. Applied in the fixed-rate output thread, where dt is exact — unlike the upstream
    // per-iteration EMA, whose effective time constant swings with the 105–1470 ms compute cadence.
    // This is an actuator constraint, not a tuning gate; it is deliberately NOT a substitute for giving the
    // MPPI a control-continuity cost, which is where the oscillation actually comes from.
    // ASYMMETRIC: slowing down is always safe, so braking gets a much higher limit than accelerating —
    // a hold/stop still takes effect in ~0.2 s while acceleration stays smooth.
    float max_lin_accel_mps2 = 1.5f;
    float max_lin_decel_mps2 = 3.0f;
    float max_rot_accel_rps2 = 4.0f;
    float max_rot_decel_rps2 = 8.0f;
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
    // Near-obstacle "black box": when the robot comes within proximity_log_distance_m of a tracked
    // obstacle (or the trajectory ESDF drops that low), append a CSV row capturing what the controller
    // saw that cycle (min_esdf, safety-guard/blockage flags, commanded velocity, geometric nearest
    // obstacle). Lets a later collision be explained — e.g. nearest_obst_m small but min_esdf large ⇒
    // the ESDF never saw the obstacle (self-occlusion). Off by default.
    bool proximity_log_enabled = false;
    std::string proximity_csv_path = "proximity_obstacles.csv";
    float proximity_log_distance_m = 0.6f;
    int max_lidar_draw_points = 600;
    std::string lidar_name = "lidar3D";
    // Per-device high/low LiDAR planes (lidar3d_dds): points arrive in the DEVICE frame (metres) and
    // are transformed to the robot frame via each sensor's static mount RT edge (robot<-helios /
    // robot<-bpearl), then MERGED into one scan per cycle. Preferred over the fused lidar_name plane;
    // the controller falls back to lidar_name (already robot-frame) only while neither is live.
    std::string lidar_helios_name = "helios";
    std::string lidar_bpearl_name = "bpearl";
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
    // Controller-side LiDAR obstacle CREATION (reactive blockage/stall temp obstacles + refresh). Set false
    // to make the dedicated `residual_concept` agent the SOLE obstacle source: the controller then only
    // CONSUMES graph "obstacle" nodes (read_obstacle_polygons) and no longer creates its own from LiDAR.
    // The physical-STUCK recovery (wedged on something invisible to the LiDAR) is a separate reflex and is
    // NOT gated by this flag.
    // DEFAULT OFF (2026-07-29). Every controller-side local obstacle is now gated by this, including the
    // stuck-recovery virtual disc that previously escaped it. residual_concept is the sole obstacle source:
    // it runs a proper inverse-sensor-model occupancy grid with ray carving and a Beta posterior, whereas the
    // controller's version fitted padded boxes to a few LiDAR returns with an existence belief that saturated
    // on first sight and could not be disconfirmed by free space. Two independent obstacle representations
    // feeding one ESDF, each with its own inflation and neither able to retire the other, is what produced
    // duplicated and immortal geometry. Flip back on only if residual_concept is down.
    bool obstacle_creation_enabled = false;
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
    // LiDAR height band (metres, robot frame ≈ height above floor since robot_from_lidar is identity
    // and the source cloud is already floor-referenced). The min is a floor-plane reject cutoff: the
    // lowest lidar beam grazes the floor into a dense ring at ~0.11–0.18 m (confirmed with an EMPTY
    // Webots room), so it MUST sit above that ring — 0.20 is the known-good value. Lower it only if the
    // [LidarZ] histogram shows the floor ring is lower AND you need to catch sub-0.20 m obstacles.
    float temporary_obstacle_min_height_m = 0.20f;
    float temporary_obstacle_max_height_m = 1.8f;
    // Same idea for the proactive unmodelled-obstacle scan (room frame, z = height above floor).
    float unmodelled_scan_min_z_m = 0.10f;
    float unmodelled_scan_max_z_m = 1.50f;
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

    // Physical-WEDGE detection + reverse-and-turn escape. Distinct from the MPPI's geometric
    // path_blocked (a VISIBLE obstacle on the planned path, handled by modelling + replanning): a wedge
    // is a PREDICTION ERROR — the robot commands translation but the base achieves less than
    // stuck_slip_ratio of it (blocked by something the planner can't see: below the lidar plane, a
    // scraped frame, wheel slip, or boxed in with no route). This is the ONLY thing that is really
    // "stuck": a robot moving as commanded — detour, slow nav, arrival rotation, a still-sliding creep —
    // is fine, and earlier signal versions (straight-line goal distance, then waypoint-index progress)
    // false-fired on exactly those, dropping recovery discs mid-nav / at the target. When the base fails
    // to achieve its commanded translation for stuck_confirm_ms → escape: reverse with a slight turn
    // toward the side with more ESDF clearance, drop a marker at the wedge spot, then replan. On by default.
    bool  stuck_recovery_enabled = true;
    float stuck_cmd_lin_eps      = 0.05f;   // m/s — LEGACY (no longer gates detect_stuck; kept for config back-compat)
    float stuck_cmd_rot_eps      = 0.15f;   // rad/s — LEGACY (see above)
    float stuck_meas_lin_eps     = 0.02f;   // m/s — LEGACY (base-speed gate removed; kept for config back-compat)
    float stuck_meas_rot_eps     = 0.08f;   // rad/s — LEGACY (see above)
    float stuck_slip_ratio       = 0.25f;   // wedge = measured base speed < this × commanded (prediction error)
    float stuck_confirm_ms       = 1500.0f; // sustained command-without-motion before escape fires
    float escape_adv_speed_mps   = 0.15f;   // reverse speed (issued negative)
    float escape_rot_speed_rps   = 0.35f;   // slight turn rate during escape
    float escape_distance_m      = 0.30f;   // back up at most this far …
    float escape_max_ms          = 1500.0f; // … or this long, whichever comes first
    float escape_side_probe_m    = 0.50f;   // lateral ESDF probe for turn-direction choice
    float escape_rear_probe_m    = 0.45f;   // rear ESDF probe distance
    float escape_rear_min_m      = 0.30f;   // rear clearance below this → rotate-in-place, no reverse
    // On stuck, drop a LOCAL-ONLY virtual obstacle at the wedge spot: visible to the planner/MPPI
    // (set_static_obstacles) but NEVER uploaded to DSR. Covers the case where MPPI wedges on
    // something residual_concept never modelled (e.g. below the LiDAR plane) — the LiDAR-based
    // temp obstacle would find no points and create nothing, so replanning alone can't help. The
    // disc ages out on this TTL so it doesn't permanently pollute the map (a dynamic obstacle that
    // moved on is forgotten); if the robot re-wedges, stuck recovery drops a fresh one.
    float         stuck_virtual_obstacle_radius_m = 0.30f;   // half-extent of the virtual disc
    float         stuck_virtual_obstacle_forward_m = 0.40f;  // placed this far ahead of the robot
    std::uint64_t stuck_virtual_obstacle_ttl_ms   = 5000;    // lifetime before it ages out
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