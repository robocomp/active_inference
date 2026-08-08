#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../../common/robust_metrics/robust_metrics.h"

using ControllerPolygon = std::vector<Eigen::Vector2f>;
using ControllerPolygons = std::vector<ControllerPolygon>;

// The route the grid planner returned, already reduced to its turning points. It used to carry a second
// `graph_nodes` vector — the visibility graph's vertices — which the grid planner could only fill with a
// verbatim copy of the path. Turning points ARE the waypoints, so one vector says it.
struct ControllerPathPlan
{
    std::vector<Eigen::Vector2f> room_path;
};

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
    // ── ROUND FOOTPRINT (display only) ────────────────────────────────────────────────────────────
    // A round table's belief has no yaw and no width/depth — it has a RADIUS — and drawing it as the
    // axis-aligned box its w/h attributes imply asserts corners and an orientation the agent never
    // inferred. When the concept agent says the shape is round (table_concept picks round_table.obj
    // over table.obj by free-energy model evidence), the canvas draws the disc instead.
    // DISPLAY ONLY: `polygon` is untouched and remains what the planner avoids, which stays the
    // conservative circumscribing box. Changing what the robot plans around is a separate decision
    // from changing what the operator is shown.
    bool  round = false;
    float round_radius_m = 0.f;   // max(width, depth) / 2 — the disc spans the same extent as the box
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
    // Planning grid resolution. Independent of the residual's evidence grid: the planner does not need
    // centimetre fidelity, and cell count drives both memory and search time quadratically.
    float planner_cell_size_m = 0.06f;
    // A* CLEARANCE PREFERENCE — see GridPlanner::Params::clearance_weight. The search cost was pure
    // shortest-path, so it hugged anything the footprint test would legally allow and the route
    // optimiser downstream had to undo it. This makes tight cells cost more inside the search itself.
    // It is a PREFERENCE: it never makes a passable gap unplannable. 0 = the old behaviour.
    float planner_clearance_weight = 1.5f;
    float planner_clearance_pref_m = 0.9f;
    // Preferred standoff BEYOND the robot's real extent, handed to the MPPI as d_safe. Pure comfort: the
    // hard constraint is the footprint test, so raising this can slow the robot near obstacles but can never
    // make a reachable goal unreachable. It used to be `clearance_m` and did three unrelated jobs at once —
    // it also shrank the navigable room polygon by its own value and set the MPPI's disc radius to half of it.
    float comfort_standoff_m = 0.5f;
    // Cross-cycle control-continuity cost in the MPPI (see TrajectoryController::Params).
    // 0 = off, which is the pre-existing behaviour and the baseline condition.
    // How many waypoints BEYOND the current one the follower's path should extend. 1 = the previous
    // behaviour (path ends at the next waypoint, so every waypoint is an arrival and a deceleration).
    int path_horizon_waypoints = 1;
    // true  = the mission is driven as ONE continuous curve (RouteFollower): no waypoint targets, no
    //         arrival radius, no per-waypoint replan, curvature-continuous.
    // false = the previous waypoint-by-waypoint behaviour.
    bool route_continuous = false;
    // Fit the C2 curve to EVERY planned path — click targets and affordance targets too, not only
    // missions. Independent of route_continuous, which is about how a MISSION is driven.
    bool smooth_planned_path = true;
    float route_spacing_m = 0.05f;
    float route_smoothing_m = 0.40f;
    // Variationally optimise the route's control polygon before it is driven (route_optimizer.h).
    // ★ON. Re-enabled after the term balance was fixed; measured on this exact tour with
    // tools/route_bench against the world snapshot the controller writes (route_world.txt): it removes
    // 3 of the 4 places where the route demanded a turn radius below the robot's 0.23 m inscribed (all
    // three planner artefacts around (0,-1.9)) and raises min clearance 0.334 -> 0.431 m, costing 0.06 m
    // of waypoint fidelity and no traversal time. The 4th is AUTHORED (wp21/wp22 are 0.35 m apart with a
    // 104° turn between them) and no optimiser can remove it without abandoning the waypoints.
    // ★HISTORY, kept because the failure mode is worth recognising: this was DISABLED 2026-08-01 after
    // it wrecked a live route — max control-point movement 24.8 m, min clearance 0.020 -> 0.000 m (it
    // made its OWN safety term worse), 913 of 1324 samples rejected by the feasibility pass, a route
    // that doubled back around the table. The objective was ~69 with the clearance term bounded above
    // by 1, so the anchor term swamped everything — the dilution failure the MPPI's G_info also had.
    // ★KEEP THIS SWITCHABLE. An optimised route CHANGES THE STIMULUS: a run under this flag measures
    // the route, not the controller, so isolating a controller change means being able to turn it off.
    // (This default read `false` with the disabling note above until 2026-08-04, while etc/config.toml
    // had said `true` for some time — the config won every run, and the stale default misled a reader
    // into believing the feature was dead. Defaults and config must agree or the comment is a trap.)
    bool route_optimize = true;
    // ── LOCAL ELASTIC BAND (route deformed at control rate against the LIVE field) ────────────────
    // The route optimiser above runs ONCE, when the route is installed, against GridPlanner's static
    // room-frame EDT — polygons only. The band re-runs the SAME objective every cycle on a window of
    // control points ahead of the robot, against the MPPI's live robot-frame ESDF (rebuilt each compute
    // from LiDAR), so the geometry keeps absorbing what the world actually says. This is Quinlan/
    // Brock-Khatib with the B-spline control polygon as the decision variable, which is what
    // route_optimizer already is — only the field and the variable window differ.
    // ★OFF by default. It deforms the route the robot is driving; it must be earned on measurements.
    bool band_enabled = false;
    // Gauss-Newton steps per cycle. This is a WARM-STARTED incremental solve, not a fresh one: the
    // polygon already sits near its optimum, so a handful of steps tracks a moving field. 30 (the build
    // value) at control rate would be re-solving from scratch 10 times a second.
    int   band_iterations = 4;
    // Metres of route AHEAD of the robot left frozen before the window opens. The curve must not move
    // under the robot, and the follower's carrot looks ahead — deforming inside that lookahead steps the
    // command. Everything behind the robot is frozen too (arc length is measured from s=0).
    float band_lead_m = 1.0f;
    // Length of the deformable window beyond the lead. Bounded because a band is a LOCAL edit by
    // definition, and because the live ESDF is an 8x8 m robot-frame box — asking about geometry outside
    // it returns "no obstacle" (100 m), which is honest but carries no information.
    float band_window_m = 4.0f;
    // Run the band every N control cycles. 1 = every cycle.
    int   band_period_cycles = 1;
    // Route optimiser: speed (0) <-> safety (1). Trades PRECISION between the clearance preference and
    // the curvature prior (see RouteOptimizerConfig::safety_bias). 0.5 is exactly the written weights, so
    // it is inert until moved. Live from the UI slider; applies to the next route build or repair.
    float route_safety_bias = 0.5f;
    float lambda_continuity = 0.0f;
    float continuity_rot_factor = 1.0f;
    // One row of continuous trajectory statistics per completed run is appended here. Empty disables the file (the console
    // summary is printed either way).
    std::string mission_csv_path = "mission_metrics.csv";
    // One JSON per completed run, under <dir>/<mission>/. Empty disables.
    std::string mission_run_dir = "etc/runs";
    float max_adv_speed_mps = 0.7f;
    // Lateral-acceleration budget. ONE number with two consumers, deliberately: it sets the route
    // optimiser's curvature normaliser (rho = v_max^2 / a_lat) and the controller's curvature speed
    // ceiling (v = sqrt(a_lat / kappa)). If they disagreed, the route would be shaped for a comfort
    // level the controller then refused to drive at.
    float max_lateral_accel_mps2 = 1.0f;
    float max_rot_speed_rps = 0.8f;
    float pos_gain = 1.2f;
    float rot_gain = 1.5f;
    // ── Fixed-rate velocity output (see controller_motion_commander.h) ──
    // The base is commanded from a dedicated thread at this period instead of from compute(), whose measured
    // cadence was median 108 ms but mean 212, p99 1.26 s and max 9.5 s — that variance IS the stutter. The
    // command is re-sent every tick even when unchanged (the old code skipped identical commands entirely, so
    // the base could hear nothing for seconds during steady driving).
    float velocity_output_period_ms = 50.f;
    // ── DATA-DRIVEN CONTROL ──────────────────────────────────────────────────────────────────────
    // false = the pipeline runs when the presence state machine's on_operating_loop hook sets a flag,
    //         i.e. at Period.Compute on the GUI THREAD. Lidar is drained at the control thread's own
    //         wake rate and mostly discarded, and whichever scan survives to the next gate opening is
    // How often the control thread wakes to look for a new scan. Only an upper bound on how late a
    // fresh scan can be noticed; it is not the control rate, which is the scan rate.
    float control_poll_ms = 20.f;
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
    // Draw the cloud (and read the pose) one lidar frame OLD — query the RT at the PREVIOUS scan's
    // Compose the room←robot pose with room_concept's published body twist when the RT query fell
    // OUTSIDE the ring and the tree could only clamp to an end block. That clamp is the normal case
    // for a freshly-arrived scan — the pose is derived FROM a scan, so it always trails one — and it
    // registers the whole cloud with a pose |Δt| ms off, which shows up as a bulk ω·Δt rotation the
    // moment the robot turns and vanishes at rest. See ControllerObstacleTracker::twist_corrected.
    // It repairs the clamp the RT ring cannot avoid:
    // the correction is exact where the buffer was merely patient, and drops a full lidar period of
    // latency from the obstacle path. Turn OFF to A/B against the buffered-only behaviour.
    bool rt_twist_compensation = true;
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
    // The camera node whose rgb media descriptor backs the affordance panel's picture. DISPLAY ONLY —
    // nothing the robot does depends on it, so an absent node costs a backdrop and nothing else.
    std::string camera_node_name = "zed";
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

    // Honour an affordance's commanded facing yaw at arrival. true = the follower rotates in place at
    // the goal until it faces target.yaw_rad, and only then reports goal_reached (needed when the
    // point of the affordance is to LOOK at something). false = arrival is judged on POSITION alone
    // and the terminal rotation never happens — the right setting for pure navigation runs, where
    // turning on the spot at every waypoint is wasted motion and one more place to hang.
    bool  goal_facing_yaw_enabled = true;

    // Affordance servo ("lock-on") executor — the HOW (gains/caps/timing). The WHAT/WHEN
    // (scalar_target, completion predicate, stable_n, timeout) is per-affordance and comes from the
    // affordance_protocol Contract. Off by default. See controller_lockon.h / affordance_protocol.h.
    bool  lockon_enabled         = false;
    // ★HALVED 2026-08-07. The search moves to CHANGE THE VIEW, and every centimetre of that motion
    // costs mask quality: the producer's own ego-motion channel (mask_motion_var / motion_dotd) grows
    // with base speed, so a fast sweep degrades the very evidence it is sweeping to collect. Measured
    // symptom: a mask plainly visible to the eye that the loop never captured.
    float lockon_sweep_speed_mps = 0.06f;   // PRIMARY: distance-sweep advance speed
    float lockon_sweep_range_m   = 0.45f;   // oscillate ± this far from the arrival pose
    float lockon_offset_tol      = 0.15f;
    float lockon_k_yaw           = 0.8f;
    float lockon_max_yaw_rps     = 0.06f;
    float lockon_dither_yaw_rps  = 0.05f;
    float lockon_settle_ms       = 900.0f;   // MINIMUM still-window before a measurement
    float lockon_settle_max_ms   = 2500.0f;  // bound on waiting for post-stop evidence
    int   lockon_settle_new_frames = 2;      // producer frames to wait for after the base stops
    float lockon_step_ms         = 300.0f;
    int   lockon_max_attempts    = 30;

    // ── DWELL AFTER AN AFFORDANCE ─────────────────────────────────────────────────────────────────
    // Hold still for this long after finishing one affordance before the planner is allowed to select
    // the next. An affordance is an EPISTEMIC action: the robot goes somewhere to LOOK. Its result is
    // whatever perception acquired at the final pose — masks, and the belief update they drive — and
    // the moment the next affordance is selected the base turns away and that result is gone before
    // anyone (or the mask pipeline, which runs at camera rate with its own latency) has seen it. The
    // dwell is where the affordance panel's camera view is worth looking at.
    // NOT a settling gate on the servo: the contract's own stable_n already decides when the LOOK is
    // finished. This is time held afterwards, for the observer and for the producer's latency.
    // 0 disables it and the old back-to-back behaviour returns.
    float affordance_dwell_ms = 3000.0f;
    // ── AND WAIT FOR THE ACQUISITION, NOT JUST FOR THE CLOCK ──────────────────────────────────────
    // A fixed dwell is a bet that the mask will arrive inside it. When the affordance exists to acquire
    // a mask of a particular object, the honest end condition is the ACQUISITION: hold until that
    // object's mask has been seen in this many separate producer frames. One sighting proves nothing —
    // YOLO flickers, and a single frame is exactly what an intermittent detection can produce by
    // accident — so the count is what makes it evidence rather than a coincidence.
    // 0 disables it and the dwell is the clock alone.
    int   affordance_dwell_mask_hits = 5;
    // BOUND on the whole wait. A mask that is never going to arrive must not park the robot forever:
    // past this the dwell ends regardless and the log says the acquisition failed, which is a result.
    float affordance_dwell_max_ms = 12000.0f;

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

// ── WHERE THE AFFORDANCE SAID TO STAND ───────────────────────────────────────────────────────────
// The affordance's own target: the pose it designed for the ROBOT to occupy in order to observe the
// object it services. Carried out to the camera overlay, which marks THIS — the target of the action —
// rather than the object, which is what the action is ABOUT. The two are metres apart by construction
// (the standpoint sits at the sensor's stand-off distance off a chosen face), so a marker on the object
// centre was never the target it claimed to be.
// ★This is the standpoint IN FORCE, i.e. after the reachability/rotatability repair in
// select_target — what the robot is actually driving to. Marking the pre-repair value would put the
// cross where nobody is going.
struct ControllerStandpoint
{
    Eigen::Vector2f room_pos = Eigen::Vector2f::Zero();
    float yaw_rad = 0.f;
    // Did the CONTRACT design a final orientation (Servo/Orient), or is this a plain Reach? Only a
    // designed facing is ever drawn — see the same rule in ControllerSession::wants_final_facing.
    bool  has_facing = false;
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