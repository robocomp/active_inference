#pragma once

#include <vector>
#include <optional>
#include <random>
#include <string>
#include <Eigen/Dense>

#include "../../common/robot_footprint/robot_footprint.h"

#include "lidar_buffer_types.h"

namespace rc
{
/**
 * TrajectoryController — MPPI-based local controller with ESDF.
 *
 * Proper MPPI: warm-start + Gaussian perturbations + AR(1) noise.
 * All K samples are perturbations of the previous optimal sequence.
 * Weighted average over the FULL T-step sequence (not just first step).
 */
class TrajectoryController
{
public:
    enum class ControlMode { MPPI, PD };

    struct Params
    {
        // Superparameter in [0,1]: 0 = calm, 0.5 = neutral, 1 = excited
        float mood = 0.5f;
        bool  enable_mood = true;

        // Mood gains (higher values increase the effect of mood on each family)
        float mood_speed_gain = 0.35f;        // max_adv, max_rot, carrot lookahead
        float mood_reactivity_gain = 0.35f;   // warm-start inertia, output smoothing, brake
        float mood_caution_gain = 0.30f;      // calm-side boost for d_safe and obstacle conservatism
        // Should mood widen the ARRIVAL RADIUS? No — and this exists to keep it that way.
        // goal_threshold is a tolerance, not a fluidity knob (the arm classes its REACH_TOLERANCE_M as
        // fixed for the same reason, FSM.md:198-224). Letting a learned confidence widen it means each
        // leg physically ENDS EARLIER, so duration_s and path_length_m fall — a 5-10% "improvement" on
        // a 1.2 m leg that is really just a shorter leg, and indistinguishable from a real one after
        // the fact. A mechanism must not be able to move the ruler it is measured with.
        bool  mood_scales_goal_threshold = false;

        // Kinematic limits (differential drive: adv + rot only)
        float max_adv   = 0.8f;   // m/s forward
        float max_back_adv = 0.20f; // m/s backward (for evasion maneuvers)
        float max_rot   = 0.7f;   // rad/s

        // Safety.
        // d_safe is a PREFERRED standoff — comfort, not geometry. The hard limit is the footprint itself
        // (see body_extent_* below), so d_safe can be tuned freely without ever making a reachable goal
        // unreachable. There is deliberately NO robot_radius here: a disc radius is a guess, and six of them
        // stacked across this stack once demanded ~0.95 m of gap for a robot that passes 0.461 m.
        float d_safe       = 0.35f; // preferred ESDF standoff BEYOND the body extent
        float safety_priority_scale = 1.0f; // always-on safety boost applied before mood (>=1 recommended)

        // Carrot / path following
        float carrot_lookahead = 2.0f;
        // How far the chord to the carrot may deviate from the ROUTE FRAGMENT it spans, in metres.
        // Pure pursuit cuts any route feature whose radius is small against the lookahead: with a 2 m
        // lookahead, a tight curve or a spur can sit centimetres away in SPACE while being metres away
        // in ARC LENGTH, so the robot drives straight at the carrot and never makes the excursion — it
        // does not go where the route says to go. Obstacle-freedom does not catch this: the shortcut is
        // usually through open floor. This is a FIDELITY limit (how much of the route may be skipped),
        // not a safety one, which is why it is a stated allowance rather than derived.
        float carrot_max_route_cut_m = 0.15f;
        bool  carrot_curve_adaptation_enabled = false; // temporary inhibit switch for curve-based carrot adaptation
        float carrot_curve_lookahead_min = 0.9f;   // minimum lookahead used only in tight curves
        float carrot_curve_min_heading_change = 0.08f; // rad; below this, path is considered straight
        float carrot_curve_release_heading_change = 0.04f; // hysteresis release threshold (straight again)
        float goal_threshold   = 0.25f;

        // Final in-place alignment: after reaching the goal position, rotate the
        // base to face the target's commanded yaw (so e.g. the robot looks AT the
        // table at an epistemic affordance) before declaring the goal reached.
        float align_yaw_tol_rad = 0.06f;   // |yaw error| under which alignment is done (~3.4 deg)
        float align_kp          = 1.5f;    // P gain on yaw error -> rot command
        float align_min_rot     = 0.10f;   // rad/s floor to overcome stiction while aligning

        // MPPI sampling — initial / baseline values (adapted by ESS)
        int   num_samples      = 100;       // K baseline
        int   trajectory_steps = 50;       // T baseline
        float trajectory_dt    = 0.1f;     // dt per step

        // Command floors to avoid degenerate zero-motion plans
        float min_adv_cmd = 0.05f;
        // Three deterministic seeds — stop, and pivot in place either way — added because the sampler
        // cannot otherwise propose them (adv is clamped at >= 0 around a forward nominal, so "stop" needs
        // ~12 consecutive noise draws to land on the clamp). ★DEFAULT OFF: they were added mid-session on
        // an argument, they did not change either offline snapshot (one did not need them, the other was
        // already overlapping), and the first lap that carried them came back slower. Unmeasured code
        // does not get to ride along in the baseline. Switch on to measure.
        bool  enable_stop_pivot_seeds = false;

        // Discount and numerical epsilons
        float cost_discount = 0.95f;          // per-step discount in rollout scoring
        float weights_epsilon = 1e-10f;       // minimum normalized-weight sum / weight threshold
        float ess_den_epsilon = 1e-20f;       // ESS denominator guard

        // Debug/diagnostics
        int debug_print_period = 50;          // print MPPI diagnostics every N compute cycles

        // ESS-based adaptive ranges
        int   K_min = 20,  K_max = 300;    // adaptive K bounds
        int   T_min = 15,  T_max = 120;     // adaptive T bounds
        float lambda_min = 1.0f, lambda_max = 500.0f;  // adaptive λ bounds
        // DIAGNOSTIC ONLY (default off): bypass `lambda_used = max(adaptive, cost_range/5)` and use the
        // configured mppi_lambda directly. The floor makes the temperature a function of whichever term
        // has the widest spread, so no lambda sweep can ever be run while it is active — measured, a
        // 30x sweep of mppi_lambda changed the closed-loop result by nothing at all. This exists to ask
        // one question: is the range floor amplifying a term's instability, or merely reporting it?
        bool  lambda_fixed = false;
        // ── "NAV2-STYLE" SCORING (default off) ────────────────────────────────────────────────────
        // Nav2's MPPI is a configuration that demonstrably works, and it is coherent in a way ours is
        // not: every critic is bounded and O(1), the temperature is a small FIXED number (0.3), and the
        // collision cost is a huge finite value. Those three facts are one mechanism — exp(-1e5/0.3) is
        // zero, so a big collision cost IS hard rejection. Copying any one of them alone breaks: our
        // measured cost spread is ~1400, so their lambda would put ESS at 0.03 (measured: 23-37 rotation
        // flips per 10 s and 0.93 m covered in ten seconds).
        // So this flag changes the SCALES, which is the precondition for a small fixed lambda to mean
        // anything: every per-step term is clamped to obstacle_cost_cap, and the accumulated terms are
        // divided by the number of steps actually simulated, so a total is a per-step average and does
        // not grow with the horizon. Set with lambda_fixed + a small mppi_lambda to get nav2's regime.
        bool  bounded_costs = false;
        // Nav2 has no structured exploration seeds — only i.i.d. noise around the warm start. Ours
        // inject 6 fixed wide-angle turns, and the closed-loop trace showed the winning seed alternating
        // between those and the random ones (28 changes in 59 cycles), with the commanded rotation
        // disagreeing in SIGN with the best rollout. They are a manufactured second mode.
        bool  enable_injection_seeds = true;
        float ess_smoothing = 0.25f;       // EMA alpha for ESS (faster response to drops)
        float ess_initial_ratio = 0.5f;    // initial ESS / K used when a new path starts

        // MPPI temperature (initial, adapted by ESS)
        float mppi_lambda       = 8.0f;

        // Noise standard deviations (for Gaussian perturbations)
        float sigma_adv   = 0.12f;
        float sigma_rot   = 0.15f;

        // AR(1) temporal noise correlation
        // Adaptive sigma limits
        float sigma_min_adv   = 0.04f;
        float sigma_min_rot   = 0.08f;
        float sigma_max_adv   = 0.25f;
        float sigma_max_rot   = 0.25f;

        // Nominal control blending (warm start weights)
        float warm_start_adv_weight = 0.5f;
        float warm_start_rot_weight = 0.3f;

        // Gradient optimization (post-processing refinement)
        int   optim_iterations = 0;
        float optim_lr         = 0.05f;
        float optimize_goal_pull_dist_cap = 1.0f;     // cap for goal-pull distance in seed refinement
        float optimize_obstacle_cap_ratio = 2.0f;     // max obstacle correction / goal correction ratio
        float optimize_goal_min_norm = 1e-4f;         // minimum goal correction norm for ratio gating
        float optimize_remaining_cap_steps = 3.0f;    // cap remaining-time Jacobian factor in dt units

        // EFE weights (scoring)
        float lambda_goal      = 5.0f;
        float lambda_obstacle  = 8.0f;   // global obstacle weight (single multiplier)
        float lambda_smooth    = 0.1f;  // smoothness cost multiplier (penalizes control changes between steps)
        float lambda_velocity  = 0.01f;  // velocity cost multiplier (penalizes excessive speed, encourages earlier arrival) 
        float lambda_delta_vel = 0.18f;  // extra penalty on changes in velocity (oscillation penalty)
        float lambda_heading   = 0.6f;   // heading term multiplier relative to lambda_goal
        // ── CONTROL CONTINUITY (cross-cycle) ──────────────────────────────────────────────────────
        // Penalises how far a rollout's FIRST control sits from the command actually EXECUTED last
        // cycle. This is the term the measured stutter says is missing: the documented failure is a
        // near-full-scale reversal BETWEEN cycles (+0.456 -> -0.457 rad/s in 100 ms), and nothing in
        // the objective sees that. G_smooth looks like it should, but it references
        // prev_optimal_[0] — the previously PLANNED first step, not what the base was told (the
        // executed command is a 3-step mean, then possibly the straight-speed override, then the
        // output smoother, then the slew limiter) — and at lambda_smooth = 0.1 a full reversal costs
        // ~0.08 against goal/obstacle terms in the 5-400 range, i.e. nothing.
        //
        // Deviations are NORMALISED by the kinematic limits, so the two channels are commensurate and
        // this weight is scale-free (a full-scale reversal costs exactly lambda_continuity per channel)
        // rather than depending on the units the base happens to use.
        //
        // DEFAULT 0 = OFF. The A/B must be exactly one change, and a baseline recorded before this
        // existed has to stay comparable.
        // ⚠ This buys smoothness with REACTION LAG: a high weight makes the robot reluctant to change
        // what it is doing, including when it should. The CBF, the safety guard and the footprint test
        // are all downstream of it and unaffected, so the failure mode is sluggishness, not collision —
        // but that is the axis to watch when raising it.
        float lambda_continuity = 0.0f;
        float continuity_rot_factor = 1.0f;   // relative weight of the rotation channel (where the stutter is)
        float lambda_progress  = 1.0f;   // moving-away penalty multiplier relative to lambda_goal

        // Obstacle model (simple 2-stage quadratic)
        float close_obstacle_margin = 0.02f;  // hard zone width over robot radius
        float close_obstacle_gain   = 3.0f;   // hard-zone multiplier (inside close_obstacle_margin)
        float obstacle_cost_cap     = 8.0f;   // per-step obstacle cost cap (prevents ESS collapse in narrow passages)

        // Lateral clearance shaping (pre-SG): penalize trajectories that run
        // too close to side obstacles, helping recentring in narrow passages.
        float lambda_lateral_clearance = 5.0f;
        float lateral_probe_offset = 0.22f;         // side probe offset from trajectory centerline
        float lateral_probe_front_offset = 0.22f;   // forward longitudinal probe station
        float lateral_probe_rear_offset = 0.18f;    // rear longitudinal probe station
        float lateral_clearance_margin = 0.25;     // desired extra side clearance over robot radius
        float lateral_balance_gain = 1.5f;          // penalize left-right side imbalance while passing obstacles
        float lateral_bumper_margin = 0.14f;        // hard side repulsion band over robot radius
        float lambda_lateral_bumper = 18.0f;        // hard penalty inside side repulsion band
        float lateral_corner_bias_gain = 2.5f;      // emphasize front-side grazing near corners
        float lateral_closing_gain = 0.0f;          // extra penalty when side clearance is decreasing
                                                    // (disabled: replaced by CBF term below)

        // Control Barrier Function (CBF) — velocity-aware safety barrier
        // h(x,v) = d_ESDF - r_robot - v²/(2*a_max)
        // Penalises rollout steps where ḣ + α·h < 0 (barrier decreasing too fast)
        bool  enable_cbf          = true;
        float lambda_cbf          = 6.0f;   // weight for CBF violation cost
        float cbf_alpha           = 1.5f;   // class-K decay rate  (larger = more conservative)
        float cbf_max_decel       = 1.0f;   // assumed max braking deceleration  (m/s²)
        float cbf_cost_cap        = 10.0f;  // per-step CBF cost cap

        // Continuous clearance relaxation near final goal (no hard switch):
        // relax only in the last segment of the approach and never below a
        // fixed fraction of d_safe, so the robot does not shave obstacles.
        float goal_clearance_relax_dist = 0.6f;
        float goal_obstacle_margin = 0.08f;
        float goal_clearance_min_ratio = 0.85f;

        // Collision and velocity-shape penalties in rollout score
        float collision_penalty = 400.0f;
        float hard_collision_horizon_s = 1.2f;  // only collisions within this lookahead are treated as hard-infeasible
        float far_collision_penalty_scale = 0.5f; // extra soft penalty for collisions beyond hard horizon
        float rot_cost_factor = 5.0f;         // relative cost multiplier for rotational effort

        // Exploration gating by Safety-Guard proximity (sigmoid on frontal distance)
        float sg_explore_pre_distance = 0.40f; // start increasing exploration this much before SG activation distance
        float sg_explore_sigmoid_width = 0.12f; // transition softness (meters)

        // Nominal and injected seed shaping
        float nominal_alignment_floor = 0.1f; // minimum forward alignment when turning toward carrot
        float nominal_goal_dist_scale = 1.0f; // distance at which nominal speed reaches full scale
        float injection_adv_scale = 0.7f;     // forward speed scale used by structured injection seeds
        float straight_speed_heading_threshold = 0.08f; // rad; below this, treat path as straight
        float straight_speed_clearance_margin = 0.20f; // extra ESDF clearance over d_safe to allow max speed
        float straight_speed_min_goal_dist = 1.5f;     // only force max speed while still far from goal

        // Structured exploration offsets (radians)
        float inject_offset_30 = 0.5f;
        float inject_offset_60 = 1.05f;
        float inject_offset_90 = 1.57f;

        // ESS adaptation thresholds and gains

        // Low-ESS decisiveness: blend weighted MPPI command with best seed
        // when ESS ratio collapses, preventing over-conservative averaging.

        // ESDF grid
        float grid_resolution  = 0.05f;
        float grid_half_size   = 4.0f;
        float esdf_unknown_distance = 100.0f;   // distance returned when ESDF query is outside the grid
        float esdf_init_distance = 9999.0f;     // initial large value used during ESDF passes
        float esdf_diag_step = 1.414f;          // diagonal neighbor cost in grid units
        float esdf_grad_min_norm = 1e-4f;       // gradient norm threshold to normalize ESDF gradient
        float esdf_self_filter_radius = 0.40f;  // ignore lidar returns this close to robot center
        float esdf_self_filter_half_width = 0.32f; // ignore returns inside the robot body width
        float esdf_self_filter_front = 0.42f;   // ignore returns inside the robot body forward extent
        float esdf_self_filter_rear = 0.24f;    // ignore returns inside the robot body rear extent

        // Path progression heuristics
        float waypoint_advance_lookahead_factor = 0.5f;
        float segment_length_epsilon = 1e-3f;

        // Goal/heading numerical thresholds
        float heading_norm_epsilon = 0.01f;

        // Output smoothing
        float velocity_smoothing = 0.60f;

        // PD carrot-follower gains
        float pd_Kp_rot = 2.0f;         // proportional gain for angular error
        float pd_Kd_rot = 0.3f;         // derivative gain for angular error
        float pd_speed_cos_power = 1.0f; // adv = max_adv * cos^power(angle_err)
        // Cross-track feedback for the PD tracker (Stanley's second term). 0 disables it, which is pure
        // pursuit and cuts corners — see compute_pd. Units: the gain is 1/s (it divides a metre by a
        // speed), the softening constant is m/s and only sets how the term behaves near a standstill.
        float pd_cross_track_gain = 1.0f;
        float pd_cross_track_soft_mps = 0.30f;

        // Visualization

        // Gaussian brake for high rotation (to prevent oscillation)
        // Reduced from 1.5: lambda_delta_vel now handles oscillation in scoring,
        // and nominal cos(angle_err) already reduces speed when turning.
        float gauss_k = 0.5f;

        // Path blockage detection
        float blockage_esdf_threshold = 0.2f;   // ESDF below this on path = blocked (well under d_safe)
        int   blockage_min_waypoints = 4;        // need N consecutive blocked waypoints
        float blockage_lookahead_m = 1.5f;       // only check waypoints within this distance ahead
        // ★compute() runs at Period.Compute = 100 ms (10 Hz), NOT 20 Hz — the old comment here said
        // "~0.75s at 20Hz" and was wrong by 2x. 15 cycles is 1.5 s of confirmation, which at 0.375 m/s
        // is 0.56 m of travel: with a 1.5 m lookahead the robot still has ~0.9 m in hand when it fires.
        int   blockage_confirm_cycles = 15;      // consecutive cycles before declaring blockage (1.5 s at 10 Hz)
        int   blockage_cooldown_cycles = 100;    // minimum cycles between replan triggers
    };

    struct ControlOutput
    {
        float adv  = 0.f;
        float side = 0.f;
        float rot  = 0.f;
        bool  safety_guard_triggered = false;
        // ── Safety-gate record, per cycle ──
        // In MPPI mode the gate is a backstop behind a controller that scores its own rollouts. In PD
        // mode it is the ONLY thing between the tracker and an obstacle, and none of it was observable:
        // safety_guard_triggered is a bool and the run JSON keeps only a total, so a lap the gate saved
        // forty times and a lap where it never fired read identically. These say what it actually did.
        float gate_speed_scale = 1.f;    // fraction of the commanded adv the gate allowed through (1 = untouched)
        float gate_horizon_s = 0.f;      // lookahead it used — speed-dependent, so it varies per cycle
        float gate_min_esdf = -1.f;      // worst clearance along the PREDICTED arc (-1 = gate did not run)
        bool  gate_hard_stop = false;    // even adv=0 was unsafe ⇒ fell through to rotate-away
        bool  gate_hard_collision = false;
        bool  goal_reached = false;
        // Position reached; rotating IN PLACE to the target facing yaw (arrival maneuver, adv=side=0).
        // goal_reached is still false until aligned. Consumers must NOT treat this as a wedge — it makes
        // no waypoint progress by design.
        bool  aligning = false;
        float dist_to_goal = 0.f;
        // Signed angular error to the commanded facing yaw, when the target carries one. This is the exact
        // quantity the arrival test compares against align_yaw_tol_rad — surfaced so the UI can show what
        // arrival is actually waiting on, instead of it only being inferable from the robot's behaviour.
        std::optional<float> goal_yaw_err_rad;
        float min_esdf = 0.f;
        // Signed lateral offset of the PATH in the robot frame (+ = path lies to the right, so the robot
        // has drifted left). The session computes a cross-track RMS for the run summary, but nothing
        // recorded it per cycle, so "it wanders off the route" could not be turned into a number.
        float cross_track_m = 0.f;

        // ESDF-input diagnostics: how many RAW lidar points actually reached build_esdf this cycle
        // (after the self-filter) and the nearest of them to the robot center (robot frame). Lets a
        // consumer tell whether an obstacle that IS in the cloud got dropped before the ESDF.
        int   n_esdf_points = 0;
        float nearest_esdf_point_m = -1.f;   // -1 when no points survived the self-filter

        // ESS diagnostics for UI
        float ess = 0.f;          // current ESS value
        int   ess_K = 1;          // current K (to compute ratio)
        float explore = 0.f;      // exploration signal [0,1]

        // ── WHY THE SOFTMAX CHOSE WHAT IT CHOSE ───────────────────────────────────────────────────
        // ESS alone says the weights are flat but not WHY. These say whether the temperature was the
        // configured one or the adaptive floor, how much cost spread there was to discriminate on, and
        // which term owns the cost — which is the difference between "the optimiser is averaging" and
        // "every rollout really is equally good". Diagnostics only: nothing reads them to decide.
        float lambda_used = 0.f;      // temperature actually applied in the softmax
        float lambda_adaptive = 0.f;  // the controller's own adaptive temperature, before the floor
        float cost_range = 0.f;       // g_max - g_min over non-colliding rollouts
        float cost_best = 0.f;        // G_total of the best rollout
        // Per-term cost of the BEST rollout, so "which term dominates" is answerable per cycle.
        float g_goal = 0.f, g_obs = 0.f, g_vel = 0.f, g_smooth = 0.f, g_lat = 0.f, g_cbf = 0.f;
        int   n_collisions = 0;
        // Rollout-set shape, from the block that used to print all of this to a terminal every second and
        // was commented out (removed 2026-08-01 — mppi_diag.csv records it per cycle to a file instead).
        // p_free is the fraction of rollouts that survive the feasibility test; steering_concentration is
        // how aligned the surviving ones are. Together they say whether there IS a dominant way through,
        // which is a different question from whether the softmax can tell (that is ESS).
        float p_free = 0.f;
        float steering_concentration = 0.f;
        // The lateral balance term's OWN INPUT, measured at the robot's current pose: the left/right
        // difference of the normalised side-clearance deficits, in [-1,1]. Positive means the left side is
        // the tighter one. g_lat is a total and cannot say whether the centring sub-term is what is
        // steering; this can. If the commanded rotation tracks this with a lag, the weave is a centring
        // servo with gain and delay — a control problem, not a cost-structure one.
        float side_asymmetry = 0.f;
        // WHICH SEED WON, as an index into the sampled set — not into the drawn/ranked list, which is
        // what best_trajectory_idx reports and which is always 0 by construction. Seeds are generated in
        // a fixed order (nominal, 6 structured injections, then the random ones), so the index says which
        // FAMILY the winner came from, and watching it across cycles says whether the solver is switching
        // between families or refining one.
        int   best_seed_idx = -1;
        // Rotation of that winning seed's first step. If the commanded rotation is a weighted mean that
        // sits far from the best rollout's own value, the mean is not representing any single plan.
        float best_seed_rot = 0.f;

        Eigen::Vector2f carrot_room = Eigen::Vector2f::Zero();
        int current_wp_index = 0;

        // All candidate trajectories in room frame (fresh every tick)
        std::vector<std::vector<Eigen::Vector2f>> trajectories_room;
        std::vector<Eigen::Vector2f> average_trajectory_room;
        int best_trajectory_idx = -1;

        // Path blockage detection output
        bool  blockage_detected_ahead = false;   // true when a blocked path segment is visible but not yet confirmed
        bool  path_blocked = false;              // true when persistent blockage detected
        Eigen::Vector2f blockage_center_room = Eigen::Vector2f::Zero(); // center of blocked region in room frame
        float blockage_radius = 0.f;             // approximate radius of blocked region
    };

    TrajectoryController() = default;

    void set_path(const std::vector<Eigen::Vector2f>& path_room);
    /// As set_path, but WITHOUT relax_path/smooth_path_spline. For a path that is already smooth and
    /// already footprint-checked (RouteSpline): re-running the elastic band would push samples around
    /// with its own clearance heuristic and could undo the feasibility pass, and re-smoothing a C2 curve
    /// with a C1 spline would put back the curvature steps the whole exercise removes.
    void set_path_presmoothed(const std::vector<Eigen::Vector2f>& path_room);

    /// Swap the path GEOMETRY without resetting the follower — for a continuously deformed route
    /// (the local elastic band), which hands over a slightly different curve every cycle.
    ///
    /// set_path/set_path_presmoothed both go through reset_mppi_state, which is correct for a NEW route
    /// and catastrophic at control rate: it clears prev_optimal_ (the warm start the whole sampler leans
    /// on), the adaptive lambda/K/T, smoothed_vel_, last_cmd_valid_ and carrot_seg_hint_. Calling it
    /// every cycle would restart the optimisation from scratch 10 times a second.
    ///
    /// SAFE ONLY FOR A DEFORMATION, not a re-route: the caller must guarantee the new curve is the same
    /// route, still starts behind the robot, and has not moved under it — which is what freezing the
    /// control points before the robot buys. carrot_seg_hint_ is kept and merely clamped, because with a
    /// frozen prefix the arc length behind the robot is unchanged, so the index still means what it did.
    void update_path_geometry(const std::vector<Eigen::Vector2f>& path_room);
    // Optional room-frame heading (atan2(dy,dx) direction) the robot's forward axis
    // should face once the goal position is reached. Empty = legacy behaviour:
    // declare the goal reached immediately on arrival, no final rotation.
    void set_goal_facing_yaw(std::optional<float> yaw_rad);

    /// Where ARRIVAL is judged, when that is not the end of the path.
    ///
    /// The follower shapes speed against the end of its path: the nominal seed scales adv down inside
    /// `nominal_goal_dist_scale`, the straight-speed override switches off inside
    /// `straight_speed_min_goal_dist`, and `effective_d_safe_for_goal_dist` relaxes the standoff. All of
    /// that is correct behaviour for ARRIVING and wrong for PASSING THROUGH. Handing the follower a path
    /// that ends at the next waypoint therefore made every waypoint a deceleration — on a 30-waypoint
    /// tour with 1.28 m legs, the robot was inside the arrival regime for essentially the whole run and
    /// never reached cruise (measured mean 0.30 m/s against a 0.7 m/s limit).
    ///
    /// So the two questions are separated: the path may run several waypoints ahead, which is what the
    /// SPEED terms see, while arrival is judged against this near point. Unset ⇒ the path end, i.e. the
    /// previous behaviour exactly.
    /// `outgoing_dir_room` is the direction the route CONTINUES in after this point. With it, arrival
    /// also fires on PASSAGE — once the robot is past the plane through the point normal to that
    /// direction — not only on proximity.
    ///
    /// This is not optional once the path runs past the arrival point. The carrot sits
    /// `carrot_lookahead` (2 m) ahead ALONG the path, so with a 4 m horizon it aims well beyond the
    /// waypoint and the robot deliberately cuts the corner — often never entering the 0.25 m arrival
    /// radius at all. Measured: the robot drove 6.03 m on a 0.41 m leg, ended 1.57 m away, and the leg
    /// never completed; the follower then ran out of path, went inactive, the session re-issued the same
    /// path and it turned around and drove it again. Cutting the corner is the POINT of the horizon, so
    /// arrival has to be a question about passing, not about touching.
    void set_arrival_point(std::optional<Eigen::Vector2f> room_pos,
                           std::optional<Eigen::Vector2f> outgoing_dir_room = std::nullopt)
    { arrival_point_room_ = room_pos; arrival_outgoing_room_ = outgoing_dir_room; }

    /// WHO OWNS ARRIVAL. The endpoint test above is EUCLIDEAN: "am I near path_room_.back()?". That is a
    /// correct question for a path that goes from here to somewhere else, and a WRONG one for a closed
    /// route, where the end IS the beginning: at s=0 the robot stands within the goal radius of the
    /// endpoint it has not driven a single metre toward, so arrival fires on the first cycle and the tour
    /// never starts. Proximity alone cannot distinguish "not yet departed" from "returned" — only arc
    /// length can, and the RouteFollower already carries it (forward-only `progress()`), so a continuous
    /// route judges its own completion and switches this test OFF. Two arrival definitions for one motion
    /// is the defect; this leaves exactly one in force at a time.
    void set_endpoint_arrival(bool on) { endpoint_arrival_ = on; }

    // CURVATURE-LIMITED SPEED. A ceiling on max_adv for this cycle, supplied by whoever owns the
    // reference curve. The MPPI cannot derive it: its horizon is ~1.4 s of travel and its nominal seed
    // scales speed by carrot DISTANCE and alignment, neither of which knows how sharp the turn ahead is.
    // The route does know — it is C2, so kappa(s) is continuous — and v = sqrt(a_lat_max / kappa) is the
    // physical relation between lateral acceleration and turn radius, not a tuning threshold.
    //
    // It must be a CEILING ON THE PLAN, not a clamp on the output: clamping the command after the fact
    // would leave the MPPI planning trajectories it is not allowed to execute, and the resulting
    // commanded-vs-measured gap is exactly what the wedge detector reads as being stuck.
    // nullopt clears it (a click target has no curve, so it keeps the full speed envelope).
    void set_speed_limit(std::optional<float> v_max_mps) { speed_limit_ = v_max_mps; }

    // Seed the carrot's forward-only anchor when a path is (re)installed mid-route — after a repair the
    // robot is somewhere in the middle, and starting the search from segment 0 would aim it back at the
    // route's beginning. set_path resets this to 0, which is correct for a path that starts at the robot.
    void set_carrot_hint(int segment_index) { carrot_seg_hint_ = std::max(0, segment_index); }

    ControlOutput compute(const Eigen::Affine2f& robot_pose);
    /// Same cycle, with the cloud supplied instead of read from the buffer — the replay path.
    ControlOutput compute(const Eigen::Affine2f& robot_pose,
                          const std::vector<Eigen::Vector3f>& lidar_points);

    /// Seed the sampler. The default seeds from std::random_device, so two runs with IDENTICAL inputs
    /// produce DIFFERENT commands — which is correct for a robot and useless for a comparison: it puts
    /// sampling noise inside every A/B, on top of the 14.5% the world already contributes. Setting a seed
    /// makes a cycle reproducible, which is what tools/mppi_bench needs to answer "what would this cost
    /// change have done" without driving.
    void set_seed(std::uint32_t seed) { rng_.seed(seed); }

    /// Write everything the NEXT compute() consumes to `path`: pose, the lidar points it actually used,
    /// the obstacle and boundary polygons, the path, and the parameters. Replaying it rebuilds the ESDF
    /// with the SAME build_esdf on the SAME inputs, so nothing is re-derived by a second implementation
    /// (the trap route_bench avoids by recording the planner's raster).
    /// ★Snapshots the INPUTS, not the ESDF: here the producer and the replayer are the same function, so
    /// the inputs are the smaller, more honest artefact.
    void request_snapshot(std::string path) { snapshot_path_ = std::move(path); }

    /// Restore a snapshot: parameters, path, obstacle and boundary points are set on THIS controller,
    /// and the cycle's pose and cloud are handed back so the caller can replay it with
    /// compute(pose, lidar). Returns false on a malformed stream.
    bool load_snapshot(std::istream &is, Eigen::Affine2f &pose_out,
                       std::vector<Eigen::Vector3f> &lidar_out);
    void stop();
    void set_lidar_buffer(LidarPointBuffer *buffer) { lidar_buffer_ = buffer; }

    void set_control_mode(ControlMode mode) { control_mode_ = mode; }
    ControlMode control_mode() const { return control_mode_; }

    /// Set static obstacle polygons (furniture). Their edges are sampled into points
    /// and injected into every ESDF build, transformed to robot frame.
    void set_static_obstacles(const std::vector<std::vector<Eigen::Vector2f>>& obstacles_room,
                              float sample_spacing = 0.05f);

    /// Set room boundary polygon. Edges are sampled into points used by relax_path
    /// to push waypoints away from walls toward the center of free space.
    void set_room_boundary(const std::vector<Eigen::Vector2f>& polygon,
                           float sample_spacing = 0.05f);

    bool is_active() const { return active_ && !path_room_.empty(); }
    int current_waypoint_index() const { return wp_index_; }
    const std::vector<Eigen::Vector2f>& get_path() const { return path_room_; }
    std::optional<Eigen::Vector2f> current_waypoint_room() const;

    /// Signed-distance-to-obstacle (m) at a point given in the ROBOT frame, sampled from
    /// the ESDF built during the most recent compute(). Used by recovery to probe side /
    /// rear clearance. Returns the unknown-distance sentinel if no ESDF exists yet.
    ///
    /// FRAME: the same one the rollouts integrate in — **+Y is FORWARD, +X is RIGHT**
    /// (`x += adv·sin θ`, `y += adv·cos θ`). This is NOT the OmniRobot command frame
    /// (adv/side/rot, +side = left). So a LEFT probe is (−d, 0), RIGHT is (+d, 0),
    /// FRONT is (0, +d) and REAR is (0, −d).
    ///
    /// ⚠ ONLY VALID ON A CYCLE WHERE compute() RAN. The grid is rebuilt by build_esdf()
    /// inside compute(); nothing else refreshes it. During an escape/recovery manoeuvre
    /// compute() is not called at all, so this keeps returning the field as it was FROZEN at
    /// the moment the manoeuvre began — still expressed in robot-frame coordinates — while
    /// the robot is reversing and rotating out from under it. Treat readings taken during
    /// such a manoeuvre as a snapshot of the situation at its START, not as live clearance.
    float clearance_at(float rx, float ry) const { return query_esdf(rx, ry); }
    /// Gradient of the same field (robot frame), for a caller that wants to DESCEND it rather than
    /// merely read it — the local elastic band. Outside the grid clearance_at returns
    /// esdf_unknown_distance (100 m), so a one-sided clearance term sees no deficit and therefore no
    /// force: the band cannot be pulled by geometry the live field does not cover.
    Eigen::Vector2f clearance_gradient_at(float rx, float ry) const { return query_esdf_gradient(rx, ry); }

    // What the room-boundary injection did on the last build_esdf: how many cells it contributed, and
    // whether it was rejected as implausible. Recorded rather than printed so it can be compared per run.
    int  esdf_boundary_cells() const { return esdf_boundary_cells_; }
    bool esdf_boundary_rejected() const { return esdf_boundary_rejected_; }

    /// Body reach toward whatever is nearest RIGHT NOW (robot frame, heading 0) — the number to
    /// subtract from clearance_at(0,0) to get the true gap between the body and the world.
    float body_extent_here() const;

    /// The robot's real silhouette. Exposed so callers ask THIS object what the body reaches instead of
    /// keeping their own radius — that habit is what produced six independent margins across three agents.
    const RobotFootprint& footprint() const { return footprint_; }

    Params params;

private:
    Params active_params_;

    bool active_ = false;
    ControlMode control_mode_ = ControlMode::MPPI;
    // Where the robot projects onto the path, from the last compute_carrot. Kept because compute_carrot
    // already computes it and the tracker needs it; recomputing would risk a second, disagreeing answer.
    Eigen::Vector2f last_path_proj_room_ = Eigen::Vector2f::Zero();
    bool last_path_proj_valid_ = false;
    std::vector<Eigen::Vector2f> path_room_;
    int wp_index_ = 0;
    std::optional<float> goal_facing_yaw_;
    std::optional<float> speed_limit_;   // see set_speed_limit
    int carrot_seg_hint_ = 0;            // forward-only anchor for compute_carrot; see the comment there
    std::optional<Eigen::Vector2f> arrival_point_room_;
    std::optional<Eigen::Vector2f> arrival_outgoing_room_;  // desired room-frame facing dir after arrival
    bool endpoint_arrival_ = true;   // see set_endpoint_arrival — off while a continuous route owns arrival
    // The robot's real shape — the SAME polygon the global planner collides (common/robot_footprint), so the
    // two layers cannot disagree about what fits. Previously the MPPI used a 0.25 m disc while the planner
    // used the footprint, which meant the planner could route through a 0.46 m gap the MPPI would then refuse.
    RobotFootprint footprint_ = RobotFootprint::shadow();
    // The command actually returned last cycle — the reference for the continuity cost. Not
    // prev_optimal_[0], which is what was planned rather than what was executed.
    float last_cmd_adv_ = 0.f, last_cmd_rot_ = 0.f;
    bool  last_cmd_valid_ = false;
    bool aligning_ = false;                 // true while doing the final in-place rotation
    // Worst-case time the base may keep executing one alignment command before we can revise it. Used to bound
    // the in-place rotation so it cannot overshoot the tolerance band (see the arrival block). Measured p99 of
    // the compute cadence is ~1 s with a tail to 1.5 s, so this is deliberately pessimistic — being slow to
    // finish an alignment is harmless; overshooting it means never finishing at all.
    float align_worst_cycle_s_ = 1.0f;

    // Pending snapshot request (see request_snapshot). Written at the END of compute(), when the cycle's
    // inputs and its outcome are both known.
    std::string snapshot_path_;
    void write_snapshot(const Eigen::Affine2f &robot_pose,
                        const std::vector<Eigen::Vector3f> &lidar_points) const;

    // ---- ESDF ----
    std::vector<float> esdf_data_;
    int esdf_N_ = 0;
    LidarPointBuffer *lidar_buffer_ = nullptr;

    // Static obstacles (furniture) — pre-sampled points in room frame
    std::vector<Eigen::Vector2f> static_obstacle_points_room_;

    // Room boundary walls — pre-sampled points in room frame (for path relaxation)
    std::vector<Eigen::Vector2f> room_boundary_points_room_;
    // Cells the boundary pass turned on this cycle — kept so a rejected boundary can be rolled back
    // without erasing obstacles the LiDAR or furniture had already marked at the same cells.
    std::vector<int> boundary_cells_;
    int  esdf_boundary_cells_ = 0;
    bool esdf_boundary_rejected_ = false;

    // Output smoothing
    Eigen::Vector3f smoothed_vel_ = Eigen::Vector3f::Zero();
    bool has_prev_vel_ = false;
    float prev_angle_err_ = 0.f;  // for PD derivative term

    // ---- MPPI state ----
    // Previous optimal control sequence (T steps of [adv, rot])
    // This is the core warm-start: each cycle shifts it and samples around it
    struct ControlStep { float adv = 0.f; float rot = 0.f; };
    std::vector<ControlStep> prev_optimal_;

    // Adaptive noise sigmas
    float adaptive_sigma_adv_;
    float adaptive_sigma_rot_;

    // Adaptive MPPI state
    int   adaptive_K_;               // current number of samples
    int   adaptive_T_;               // current horizon length
    float adaptive_lambda_;          // current MPPI temperature
    float ess_smooth_ = 0.f;        // EMA-smoothed ESS
    float dominance_smooth_ = 0.5f; // EMA-smoothed dominance in [0,1]
    float explore_ = 0.f;           // continuous exploration signal [0,1] = 1 - dominance
    float sg_explore_gate_smooth_ = 0.f; // EMA-smoothed Safety-Guard gating factor in [0,1]
    float last_mppi_ms_ = 0.f;      // last MPPI wall-clock time
    int   safety_guard_mood_cooldown_ = 0; // cycles until next mood bump allowed
    bool  carrot_curve_active_ = false; // hold reduced carrot lookahead while inside a curve

    // Path blockage detection state
    int   blockage_streak_ = 0;          // consecutive cycles with blockage detected
    int   blockage_cooldown_ = 0;        // cycles remaining before next replan allowed

    // Compute ESS for diagnostics and adapt from dominance
    float compute_ess(const std::vector<float>& weights, int K) const;
    void adapt_from_dominance(float dominance, int K, float sg_gate);
    void refresh_active_params();

    // Installs `path_room` and resets ALL per-path controller state (MPPI warm start, adaptive
    // K/T/λ/σ, ESS, smoothing, blockage, alignment) WITHOUT any geometry conditioning. Both
    // set_path and set_path_presmoothed go through this; only set_path then relaxes + splines.
    // Does NOT set wp_index_ — the two callers legitimately differ, so each sets its own.
    void reset_mppi_state(const std::vector<Eigen::Vector2f>& path_room);
    // Path-blockage detector. Called from BOTH control modes (the PD branch returns before step 15).
    void detect_path_blockage(ControlOutput& out, const Eigen::Affine2f& robot_pose);

    // Elastic-band path relaxation: push waypoints toward center of free space
    void relax_path(int iterations = 20);

    // Catmull-Rom spline smoothing: replace polyline with smooth curve
    void smooth_path_spline();

    // RNG
    std::mt19937 rng_{std::random_device{}()};
    std::normal_distribution<float> normal_{0.f, 1.f};

    // A trajectory sample: sequence of (adv, rot) commands
    struct Seed
    {
        std::vector<float> adv;
        std::vector<float> rot;
    };

    // Result of simulating a seed
    struct SimResult
    {
        std::vector<Eigen::Vector2f> positions;
        float G_total = 0.f;
        float G_goal = 0.f;
        float G_obs = 0.f;
        float G_lat = 0.f;
        float G_cbf = 0.f;
        float G_smooth = 0.f;
        float G_vel = 0.f;
        float min_esdf = 1e9f;
        bool  collides = false;
    };

    // ---- Methods ----
    void build_esdf(const std::vector<Eigen::Vector3f>& lidar_points,
                    const Eigen::Affine2f& robot_pose);
    std::vector<Eigen::Vector3f> read_lidar_points_robot(const Eigen::Affine2f& robot_pose) const;
    float query_esdf(float rx, float ry) const;
    Eigen::Vector2f query_esdf_gradient(float rx, float ry) const;

    Eigen::Vector2f compute_carrot(const Eigen::Affine2f& robot_pose);
    void advance_waypoints(const Eigen::Affine2f& robot_pose);

    // Nominal control toward carrot (initial guess for warm-start)
    Seed compute_nominal(const Eigen::Vector2f& carrot_robot, int steps) const;

    // MPPI sampling: generate K perturbations around the nominal
    std::vector<Seed> sample_trajectories(const Eigen::Vector2f& carrot_robot,
                                          const Seed& nominal);

    // Takes no sampling-mean argument: the only thing that ever needed it was the
    // information-theoretic correction, deleted on purpose (see the block comment in
    // simulate_and_score for why it must not come back).
    SimResult simulate_and_score(const Seed& seed,
                                 const Eigen::Vector2f& carrot_robot,
                                 const Eigen::Vector2f& goal_robot);
    void optimize_seed(Seed& seed, const Eigen::Vector2f& carrot_robot);

    // Obstacle scoring helpers (single-weight 2-stage quadratic model)
    float effective_d_safe_for_goal_dist(float goal_dist) const;
    // Robot extent toward the nearest obstacle at a pose — the footprint's support function along the
    // ESDF's negative gradient. Replaces the constant robot_radius disc in the obstacle terms.
    float body_extent_toward_obstacle(float rx, float ry, float theta) const;
    // Robot extent toward a KNOWN direction, for a body at heading `heading`. Use this wherever the thing
    // being tested has a bearing — a LiDAR return, a nearest obstacle point, a path tangent — because then
    // the exact answer is available and a disc is pure pessimism. `dir` and `heading` must be in the SAME
    // frame; which frame does not matter, only that they agree.
    //
    // ONE negation, in ONE place: this controller integrates x += adv·sin θ, y += adv·cos θ, so its heading
    // is measured from +y toward +x — CLOCKWISE — while RobotFootprint rotates counter-clockwise. The Shadow
    // hull is very nearly x-symmetric, so mixing the two costs only millimetres, which is precisely why it
    // would never be caught by watching the robot drive. Keep the conversion here and it cannot drift.
    float body_extent(const Eigen::Vector2f& dir, float heading = 0.f) const
    { return footprint_.support_radius(-heading, dir); }
    // Worst-case extent over all bearings. The ONLY legitimate fallback: score normalisations and scale
    // constants, where no bearing exists and being conservative costs nothing but a slightly wider ramp.
    float body_extent_max() const { return footprint_.circumscribed_radius(); }
    // Can the robot get from HERE to `carrot_robot` by heading straight at it? The MPPI pulls toward the
    // carrot in a straight line, so a carrot the robot cannot cut to directly is not a target, it is a
    // trap. Samples the chord against the live ESDF with the body's real reach at the chord heading.
    bool chord_admits_body(const Eigen::Vector2f& carrot_robot) const;
    // The furthest point along the chord that IS admissible, so the carrot can be pulled back to it.
    Eigen::Vector2f clip_carrot_to_reachable(const Eigen::Vector2f& carrot_robot) const;
    float obstacle_step_cost(float esdf_val, float d_safe_eff, float body_r) const;
    float obstacle_repulsion_strength(float esdf_val, float d_safe_eff, float body_r) const;

    // PD carrot-follower (alternative to MPPI)
    ControlOutput compute_pd(ControlOutput& out,
                             const Eigen::Vector2f& carrot_robot,
                             const std::vector<Eigen::Vector3f>& lidar_points,
                             const Eigen::Affine2f& robot_pose);

    static float clamp01(float x);
    static float smoothstep01(float x);
    static Eigen::Vector2f room_to_robot(const Eigen::Vector2f& p_room,
                                          const Eigen::Affine2f& robot_pose);
};

} // namespace rc

