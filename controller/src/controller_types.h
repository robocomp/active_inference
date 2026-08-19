#ifndef RC_CONTROLLER_TYPES_H
#define RC_CONTROLLER_TYPES_H

// Tuning parameters and the per-cycle control record, hoisted OUT of TrajectoryController so that a
// tracker can live in its own file without including — or depending on — the whole controller.
// ★TrajectoryController keeps `using Params = ...` / `using ControlOutput = ...` aliases, so every
// existing `TrajectoryController::Params` call site still compiles unchanged. The hoist is a
// structural change with deliberately zero churn at the ~60 places that name these types.
//
// Named TrackerParams rather than Params because `rc::Params` at namespace scope is too generic a
// name to claim in a namespace this shared.

#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <cstdint>

namespace rc
{

struct TrackerParams
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
    // ── CARROT RATE LIMIT ────────────────────────────────────────────────────────────────────
    // How far the carrot may move IN THE ROOM FRAME in one control cycle, as a multiple of the
    // distance the robot itself could have travelled (max_adv * dt). The carrot slides along the
    // route at roughly the robot's own speed, so anything far above that is not the carrot
    // advancing — it is the PROJECTION jumping because the pose jumped.
    // ★Measured 2026-08-02: room_concept fails its odometry prediction, snaps to a scan match, and
    // the pose moves further in one cycle than the robot can drive — 3.0% of cycles show a
    // pose-differenced speed above the commanded LIMIT. The carrot then teleported up to 1.79 m in
    // a single cycle and its bearing swung +-58 deg (1 sd), which both control modes faithfully
    // steered at. That is why the weave survived replacing the entire controller.
    // ★This bounds the CARROT's contribution. It cannot undo the pose jump's direct effect on the
    // geometry — if the estimate moves 0.5 m sideways, the bearing to a perfectly steady carrot
    // changes too. The real fix is that the localiser should not jump; this stops the steering
    // target from amplifying it. 0 disables.
    float carrot_rate_limit_factor = 3.0f;
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
    // How close a lidar return must be for a room-frame MODEL point (furniture, room polygon) to be
    // considered already measured and therefore dropped. It is the scale of pose disagreement you are
    // willing to attribute to LOCALISATION rather than to a genuine second obstacle — measured
    // lateral pose snaps are ~75 mm, so 0.12 m covers them with margin while still admitting a real
    // object 12 cm from a wall. 0 restores the old union behaviour (model always painted).
    float model_merge_radius_m = 0.12f;
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

    // ── THE PLAIN TRACKER (ControlMode::PLAIN) — see trackers/plain_tracker.h ─────────────
    // omega = g_dc * v * [ kappa_bar(s + v*T_lag) - (2/L)*e_psi - (1/L^2)*e_y ]
    //
    // The curvature term is the ONLY one that reads the route, so feedback sees a zero-mean
    // residual and cannot double-count the nominal turn — the failure that killed the 2026-08-04
    // attempt to ADD a feedforward to pure pursuit (Kp*carrot_angle already IS one).
    // Feedback is critically damped in the ARC-LENGTH domain: e_y'' + (2/L)e_y' + (1/L^2)e_y = 0,
    // so an error decays over L metres at any speed and the time-domain bandwidth v/L falls as the
    // robot slows — always on the safe side of the 0.36 Hz lag ceiling.
    float plain_T_lag = 0.42f;    // EXACT: identified tau 0.213-0.236 + delay 0.20, r^2 0.94-0.95
    float plain_g_dc  = 1.124f;   // EXACT: 1 / 0.89 identified DC gain
    float plain_W     = 0.40f;    // EXACT: the route's own smoothing scale (kappa_avg window)
    // ★MEASURED IN tools/tracker_sim, NOT DERIVED. The margin rule L >= 3*T_lag*v_max said 1.0 and
    // predicted that a laggier robot would need MORE. Both halves were wrong: on the identified
    // plant L=0.60 beats 1.0/1.5/2.0, and on a slow plant (tau 0.35, delay 0.30, gain 0.75) the
    // rule's own prescription of 1.37 gave 380 mm rms against 100 mm at 0.60. The rule's LOWER
    // bound survives — the sweep does degrade below ~0.45 m — so 0.60 sits just above it.
    // ★L is a constant of the CONTROLLER, not of the robot: it is deliberately NOT adapted.
    float plain_L = 0.50f;
    // ★THE TERM THE DESIGN FORGOT. The curvature speed limit permits v = omega_max/kappa, which
    // hands the FEEDFORWARD the entire rotation budget — and g_dc pushes it past the clamp — so on
    // every curve the command saturates and the feedback loop is effectively open. Measured in
    // tracker_sim: reserving headroom moves rms 154 -> 94 -> 75 mm at 1.00 -> 0.70 -> 0.55, and
    // corr(e,kappa) from -0.160 (riding OUTSIDE the curve) to ~0. Costs speed: TV(v)/m 0.72 -> 1.13.
    float plain_rot_headroom = 0.70f;
    // ── THE HAIRPIN PAIR. Neither works without the other. ───────────────────────────────────────
    // This tour turns 178 degrees at s=24.18 (kappa_avg 6.81). Two separate things stopped the robot
    // driving it, and both are fixed here.
    //
    // 1. plain_brake_k — an EXPONENTIAL brake on advance against the DEMANDED turn rate:
    //        v *= exp(-k * (omega_ccw/max_rot)^2)   // omega_ccw = the CLAMPED rate
    //    The ratio coupling alone brakes only when omega SATURATES, so at omega_want just under the cap
    //    the robot drove at FULL speed while turning at maximum rate — precisely the approach to a
    //    hairpin. This brakes continuously and reaches zero, which is what makes a true point turn
    //    possible. ★PdArm has had exactly this as gauss_k = 0.5 since long before; the PD tracker
    //    survives this corner and the plain one did not.
    //    ★2.0 is MEASURED, not chosen (tools/tracker_sim, pinned world, projection window 0.60):
    //        k    0.0    0.5    1.0    2.0    3.0    5.0
    //        s   24.2   24.2   61.0  73.70  73.70  73.70   (route is 73.85 m)
    //      rms  1543   1584   1930   74.0   80.6   86.0    mm
    //    Below 2.0 it stalls at the hairpin outright. 2.0 is the knee: through, best rms, fastest.
    //    ⚠PD's 0.5 is NOT enough here — PD gets away with it because its carrot cuts the corner, while
    //    this tracker has to actually turn.
    float plain_brake_k = 0.25f;
    // 2. plain_proj_window — how far FORWARD IN ARC LENGTH the projection may search.
    //    The route is a DIRECTED curve, so arc length only moves forward and a projection cannot
    //    legitimately jump 2.5 m in one cycle however close the two points are in space. It did, because
    //    the window was 2 m and the hairpin's fold sits inside it: the search snapped to the RETURNING
    //    leg, the tip was never projected onto, and the robot turned around where the two legs touch.
    //    Measured: 3 cycles skipping 3.52 m of arc, 9.2% of the route, largest leap 2.52 m at s=24.2.
    //    The robot covers ~0.035 m per cycle, so 2 m was ~57x what TRACKING needs — the width existed
    //    for catch-up, which reset()'s whole-route re-acquire now handles separately.
    //    ⚠THE TWO ARE A PAIR. Tightening this without the brake STALLS the robot at the hairpin (it is
    //    then held to the curve and cannot execute the reversal); adding the brake without tightening
    //    this changes nothing, because the search still leaps the fold and skips the corner entirely.
    float plain_proj_window = 2.00f;
    // ★NOTE FOR ANYONE TEMPTED TO ADD ONE HERE. Between them, plain_align_power (a cos^5 pivot),
    // plain_offset_gate_ref (an offset speed gate) and plain_ff_denom_min (a feedforward denominator
    // floor) were added across four laps in one session and then ALL THREE DELETED on 2026-08-05, along
    // with a hidden 0.15 m/s steering floor. They were four approximations of one exact statement —
    // never command a turn rate the robot cannot deliver — which plain_tracker.cpp now writes directly
    // using only max_rot and plain_L, both of which already existed. The tracker is back to the five
    // parameters it started with, three of which (T_lag, g_dc, W) are IDENTIFIED constants of the robot
    // and the route rather than choices. Adding a knob here is nearly always a sign that a law is wrong
    // somewhere else.
    // Cross-track feedback for the PD tracker (Stanley's second term). 0 disables it, which is pure
    // pursuit and cuts corners — see compute_pd. Units: the gain is 1/s (it divides a metre by a
    // speed), the softening constant is m/s and only sets how the term behaves near a standstill.
    float pd_cross_track_gain = 1.0f;
    float pd_cross_track_soft_mps = 0.30f;
    // ── LATERAL BUMPER for the PD tracker ────────────────────────────────────────────────────
    // In MPPI mode lambda_lateral_bumper pushes the body off things it is passing too close to. In
    // PD mode that term does not run: the band shapes the route DELIBERATIVELY and the safety gate
    // only BRAKES, so nothing pushes sideways. With measured p05 body clearance ~0.08 m and mean
    // tracking error ~0.08 m the robot rides the margin and has no way to recover from it — which
    // is what "still too close to thin walls" is. The route cannot fix this: the error is the
    // tracker's, not the route's.
    //
    // ONE-SIDED AND SELF-EXTINGUISHING, in the same shape as the route optimiser's clearance term:
    //     deficit_side = max(0, bumper_dist - gap_side)
    //     push         = (deficit_left - deficit_right) / bumper_dist        // in [-1, 1]
    // Both sides clear ⇒ both deficits zero ⇒ exactly zero push and zero gradient. There is no
    // "if too close" switch; the term simply has no value where there is room. A wall on the LEFT
    // makes deficit_left large ⇒ push positive ⇒ steer right, away from it.
    // Fed into the SAME angular error the cross-track term uses, through the same bounded atan
    // form, so it cannot demand a rate the base does not have and it strengthens as speed drops.
    // 0 disables.
    float pd_bumper_gain = 1.0f;
    // Gap (m, measured from the BODY, not the centre) below which a side starts pushing back.
    float pd_bumper_dist_m = 0.25f;

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
    // ★★★THE VALUE THE ARRIVAL TEST ACTUALLY COMPARED — dist_to_goal is NOT it.
    // compute() writes the Euclidean distance to the arrival point into dist_to_goal, tests it against
    // goal_threshold, and RETURNS if it fires. If it does not fire, the PLAIN tracker runs afterwards
    // and overwrites dist_to_goal with the REMAINING ARC LENGTH along the spline. So on every
    // not-arrived cycle the field a log reads holds arc length, while the test compared a chord — two
    // different quantities in one column, changing meaning between phases. approach_diag.csv was read
    // against goal_threshold for a whole day on that basis. This field is written once, at the test,
    // and no tracker touches it.
    float dist_to_arrival_pt = 0.f;
    // Why the arrival test did or did not run, recorded at the moment it was evaluated rather than
    // inferred afterwards from follower state that later cycles may have changed.
    bool  arrival_test_ran = false;   // execution reached the goal check at all (not an early return)
    bool  arrival_endpoint_on = false;// endpoint_arrival_ as seen BY THE TEST, not as polled later
    bool  arrival_passed_pt = false;  // fired via passed_arrival (path continues past the point)
    bool  arrival_by_recession = false;// fired via passed_by_recession (receded from a close approach)
    // Signed angular error to the commanded facing yaw, when the target carries one. This is the exact
    // quantity the arrival test compares against align_yaw_tol_rad — surfaced so the UI can show what
    // arrival is actually waiting on, instead of it only being inferable from the robot's behaviour.
    std::optional<float> goal_yaw_err_rad;
    float min_esdf = 0.f;
    // Signed lateral offset of the PATH in the robot frame (+ = path lies to the right, so the robot
    // has drifted left). The session computes a cross-track RMS for the run summary, but nothing
    // recorded it per cycle, so "it wanders off the route" could not be turned into a number.
    float cross_track_m = 0.f;
    // Lateral bumper: signed push in [-1,1] (+ = pushed right, i.e. something close on the left)
    // and the two body-to-obstacle gaps it was computed from. Recorded because "it still clips the
    // wall" and "the bumper never engaged" look identical without them.
    // The STEERING TARGET itself, in the robot frame. Both control modes steer at the carrot, and
    // it is passed through clip_carrot_to_reachable against the LIVE ESDF every cycle — so a clip
    // that engages and disengages between cycles moves the target the robot is chasing, and nothing
    // recorded it. The weave survived replacing the whole controller, so the cause is in something
    // SHARED by both, and this is the main shared thing that was invisible.
    float carrot_bearing_rad = 0.f;   // atan2(x, y): 0 = dead ahead, + = to the right
    float carrot_dist_m = 0.f;        // after clipping — a chattering clip shows up here
    float pd_bumper_push = 0.f;
    float pd_gap_left_m = -1.f, pd_gap_right_m = -1.f;

    // ESDF-input diagnostics: how many RAW lidar points actually reached build_esdf this cycle
    // (after the self-filter) and the nearest of them to the robot center (robot frame). Lets a
    // consumer tell whether an obstacle that IS in the cloud got dropped before the ESDF.
    int   n_esdf_points = 0;
    float nearest_esdf_point_m = -1.f;   // -1 when no points survived the self-filter
    // Room-frame MODEL points (furniture, room polygon) dropped this build because the lidar already
    // reported an obstacle within model_merge_radius_m. Steady = model and measurement AGREE, which
    // is the normal case. A sharp CHANGE means they have started to disagree — i.e. the pose moved
    // under the model. That makes this a pose-jump detector independent of the reported covariance,
    // which is measurably blind to jumps (sigma ratio 1.12 at jumps, 2026-08-02).
    int   esdf_model_dropped = 0;

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

}   // namespace rc

#endif
