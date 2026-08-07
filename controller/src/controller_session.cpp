#include <numbers>
#include "controller_session.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>

namespace
{
// Short CSV tag for the obstacle layer that owns a polygon — the whole point of Stage 0 is that
// "an obstacle is 0.13 m away" is useless until you know WHICH agent put it there.
const char *obstacle_kind_tag(ControllerObstacleKind kind)
{
    switch (kind)
    {
        case ControllerObstacleKind::Object:        return "object";    // a concept agent's box
        case ControllerObstacleKind::Obstacle:      return "obstacle";  // a graph "obstacle" node
        case ControllerObstacleKind::Temporary:     return "temp";      // controller-born (LiDAR or escape disc)
        case ControllerObstacleKind::GridOccupancy: return "grid";      // residual_concept occupancy hull
    }
    return "?";
}
}   // namespace
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <iostream>
#include <limits>
#include <print>
#include <sstream>

namespace
{
// Contract-resolution key for an affordance's parent object node. Post graph-schema migration the
// node type is the generic "object" and the class lives in the object_subtype attribute; prefer it
// so rc::affordance::default_contract_for() recovers the class-specific default (table/chair/…)
// rather than the generic Reach fallback. Per-node baked aff_* overrides still win on top. Falls
// back to type() for legacy typed nodes (e.g. bottle_concept's "cylinder").
std::string parent_contract_key(const DSR::Node &node)
{
    if (const auto subtype = node.attrs().find("object_subtype"); subtype != node.attrs().end())
        if (subtype->second.selected() == 0 and not subtype->second.str().empty())
            return subtype->second.str();
    return node.type();
}
}  // namespace

void ControllerSession::set_params(const ControllerParams *params)
{
    params_ = params;
    if (params_)
    {
        // HOW (gains/caps/timing) — controller-owned. WHAT/WHEN (scalar_target, stable_n, timeout)
        // are per-affordance and come from the contract at begin() time.
        rc::LockOnConfig c;
        c.sweep_speed_mps = params_->lockon_sweep_speed_mps;
        c.sweep_range_m   = params_->lockon_sweep_range_m;
        c.k_yaw           = params_->lockon_k_yaw;
        c.max_yaw_rps     = params_->lockon_max_yaw_rps;
        c.dither_yaw_rps  = params_->lockon_dither_yaw_rps;
        c.settle_ms       = params_->lockon_settle_ms;
        c.step_ms         = params_->lockon_step_ms;
        c.offset_tol      = params_->lockon_offset_tol;
        c.max_attempts    = params_->lockon_max_attempts;
        lockon_.configure(c);
    }
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
                                         ControllerObstacleTracker &obstacle_tracker,
                                         rc::TrajectoryController &path_controller,
                                         ControllerMotionCommander &motion_commander,
                                         ControllerDisplay &display)
{
    // End-to-end perception latency, sampled once per cycle where the world model is in scope. Measured
    // against the RT edge's own validity stamp (the lidar scan behind the pose), so it spans capture ->
    // room_concept -> DSR -> this read. -1 when the edge carries no stamp history.
    if (const auto age = world_model.pose_stamp_age_ms(timestamp_ms); age.has_value())
        world_model_pose_stamp_age_ms_ = static_cast<float>(*age);
    else
        world_model_pose_stamp_age_ms_ = -1.f;

    if (!world_model.refresh_graph_state())
    {
        room_polygon_.clear();
        current_plan_.reset();
        plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
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
        current_plan_.reset();
        plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
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
    obstacle_tracker.update_active_obstacle_polygons(timestamp_ms, path_controller);
    // Rasterise the SAME obstacle set the visibility graph used into the footprint planner's grid. Cheap
    // enough to redo every cycle (0.3 ms on the apartment) and independent of polygon count, so the obstacle
    // set can be as detailed as perception makes it without the planner degrading.
    if (params_)
    {
        grid_planner_.params.cell_size_m = std::max(0.02f, params_->planner_cell_size_m);
        grid_planner_.params.safety_margin_m = params_->footprint_safety_margin_m;
        grid_planner_.params.clearance_weight = params_->planner_clearance_weight;
        grid_planner_.params.clearance_pref_m = params_->planner_clearance_pref_m;
    }
    grid_planner_.set_world(room_polygon_, obstacle_tracker.obstacle_polygons());
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

    update_base_speed(*robot_pose, timestamp_ms);   // base speed for the contract stillness gate
    overlay_now_ms_ = timestamp_ms;                 // overlay dead-reckoning target + base time
    overlay_lidar_ts_ms_ = obstacle_tracker.last_lidar_timestamp_ms();
    update_overlay_extrapolation(world_model, *robot_pose, timestamp_ms, obstacle_tracker.rt_block_lead_ms(),
                                 obstacle_tracker.rt_twist_fix_dt_ms(), obstacle_tracker);

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

    // The mission no longer supplies per-waypoint targets: it IS the route, driven in arc-length
    // coordinates by the RouteFollower in ensure_current_plan. Nothing to select here — but a running
    // mission still owns the base, so the affordance planner must not be consulted.
    if (mission_.running())
    {
        affordance_manager.clear_current();
        active_target_id_ = 0;
        target_wait_logged_ = false;
        step.target.node_name = "mission:" + mission_.selected_name();
        step.target.room_pos = robot_pose->pos;
        step.target_changed = false;
        last_target_info_ = step.target;
        return step;
    }

    // THE DRIVE MODE SAYS WHAT DRIVES, and that has to hold when a target is CONSUMED, not just when one is
    // chosen. Reaching a clicked point used to fall straight through to this branch: the affordance planner
    // handed over a target on the very next cycle and the robot drove off again, which looked like it was
    // returning to where it started. A one-click target ENDS at the point clicked.
    // The same reasoning retires a mission: in "Mission" the selector must not quietly become "Affordances"
    // the moment a tour finishes.
    const auto target = rc::uses_mission(mission_.mode()) or mission_.mode() == rc::DriveMode::Target
                            ? std::nullopt
                            : world_model.read_target_in_room(timestamp_ms);
    if (!target.has_value())
    {
        // ── IDLE FOR WANT OF A TARGET, AND SAY SO — IN EVERY MODE ────────────────────────────────
        // This spoke only in AffordancesOnly. In Target mode it stopped the base and returned in total
        // silence, writing not even a diagnostic row, which is indistinguishable from a hang: the CSV
        // freezes, the [CTRL] fps line keeps ticking, and the robot sits there.
        // It is reachable in Target mode by an ordinary sequence: finishing an affordance runs
        // finalize_reached, which clears manual_target_room_ along with everything else it retires, so
        // switching to Target afterwards finds no clicked point and idles — correctly, but mutely.
        // Latched on the MODE, so it speaks again after a mode change rather than once per process.
        if (target_wait_logged_mode_ != mission_.mode())
        {
            target_wait_logged_mode_ = mission_.mode();
            if (mission_.mode() == rc::DriveMode::Target)
                std::println("[controller] IDLE in Target mode — no point has been clicked. "
                             "(Finishing an affordance clears the previous one.) Click to drive.");
            else
                std::println("[controller] IDLE in {} mode — waiting for a target.",
                             rc::to_string(mission_.mode()));
        }
        current_plan_.reset();
        plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
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
    // The contract has to be known HERE, before the repair below, because the policy now decides both
    // whether there will be a terminal rotation and therefore whether the standpoint needs room for
    // one. active_contract_ is resolved at ARRIVAL and stays that way — it is what the servo executor
    // uses, and moving it would change when a policy takes effect mid-run.
    resolve_target_contract(step.target);

    // TARGET REPAIR ON THE PLANNER'S OWN PREDICATE.
    // A producer's affordance viewpoint can land inside an obstacle footprint (its own object, or one that grew
    // or moved), which the planner cannot reach. Only the NAVIGATION target is moved; the affordance's object
    // and feedback node (parent_node_id) are untouched, so the contract still services the same object.
    //
    // The repair now asks grid_planner_ itself for the nearest footprint-feasible pose, so "repaired" and
    // "plannable" are the SAME predicate and cannot drift apart. They used to be two numbers in two files —
    // repair guaranteed 0.20 m of clearance while the controller enforced 0.425 m at the goal — which left a
    // 0.225 m band where repair reported success and the robot then hunted forever at a target it was never
    // permitted to reach.
    // ── REACHABILITY IS PART OF "REPAIRED" ────────────────────────────────────────────────────────
    // nearest_free asks "does the footprint FIT here", which is purely LOCAL: it cannot tell clear floor
    // from clear floor sealed behind a wall. So for a target in a pocket it returned the same pose every
    // cycle, the planner correctly said "no route", and the robot held forever — the repair line and the
    // hold line alternating at 20 Hz with no way out. That is the whole bug.
    // The flood fill runs ONLY after a route has actually failed for this target (ensure_current_plan
    // records it), so the fast path still pays for nothing but the ring search.
    // ★KEYED BY NODE NAME, not by position. It was a position, and that made the whole fallback DEAD
    // CODE: ensure_current_plan records the target it tried to route to, which is the pose AFTER this
    // repair moved it, while the test here runs BEFORE the repair, against the raw DSR pose. The two
    // are 0.60 m apart for aff_refrigerator_1 — the repair's own displacement — so the match never
    // fired and nearest_reachable was never called. The name is the identity that survives the repair,
    // and it needs no tolerance to compare.
    // ★STICKY FOR THE LIFE OF THE TARGET. Clearing it on a successful plan OSCILLATES: the success is
    // itself the product of this repair, so clearing re-enables nearest_free, which moves the target
    // back into the pocket, which fails again — a good plan and a failed plan on alternate cycles. It
    // costs nothing to hold: once the goal IS reachable, nearest_reachable returns it exactly.
    const bool routing_failed_here = not unroutable_target_name_.empty()
                                     and unroutable_target_name_ == step.target.node_name;
    // ★COMPUTED ONCE AND HELD. nearest_reachable takes the ROBOT's position as its origin, so
    // re-running it every cycle returns a DIFFERENT point every cycle as the robot moves — and
    // same_target_instance calls a target "changed" once it shifts 5 cm. The repaired target therefore
    // chased the robot: replan every cycle, a fresh curve every cycle, the tracker's arc length pinned
    // at 0 (measured: track_s never exceeded 0.05 m over 300 cycles) while it orbited its own start at
    // 0.55 m/s with rot saturated. A target that moves when the robot moves is not a target.
    // Held for as long as the sticky flag is — same key, no second notion of identity and no new number.
    //
    // ★BUT REVALIDATED AGAINST THE LIVE GRID EVERY CYCLE. Computing it once and holding it froze the
    // answer at the instant the robot knew LEAST: the residual field fills in as the robot approaches
    // and sees the space, so a standpoint that was genuinely the closest reachable one from three
    // metres away can be inside an obstacle by the time the robot arrives — and nothing would have
    // re-asked. The cheap half of the question (is the footprint still free there, can it still turn
    // there) is a local lookup, so it is asked every cycle; failing it drops the cache and the flood
    // fill runs ONCE more against what is now known. That is the difference between caching an answer
    // and freezing it.
    // Recompute-on-invalidation only — never per cycle — because per cycle is what made the target
    // chase the robot.
    if (not routing_failed_here) unroutable_fix_.reset();
    else
    {
        if (unroutable_fix_.has_value() and not fix_still_good(*unroutable_fix_, step.target))
        {
            std::println("[controller] '{}' — the repaired standpoint ({:.2f},{:.2f}) is no longer "
                         "usable against the current map; re-solving.",
                         step.target.node_name, unroutable_fix_->x(), unroutable_fix_->y());
            unroutable_fix_.reset();
        }
        if (not unroutable_fix_.has_value())
            unroutable_fix_ = grid_planner_.nearest_reachable(step.plan_origin, step.target.room_pos);
    }
    // ── THE STANDPOINT MUST BE SOMEWHERE THE ROBOT CAN TURN, NOT JUST STAND ──────────────────────
    // With GoalFacingYawEnabled the robot performs a terminal rotation IN PLACE here, and that rotation
    // runs with no obstacle check of any kind — the align branch returns ahead of every safety stage.
    // nearest_free only ever asked "does the footprint fit at the facing heading", which is one heading
    // out of the whole arc the body sweeps; for a 0.46 x 0.65 body the swept width peaks at the DIAGONAL
    // (0.796 m), so a spot that is fine to stand in can be impossible to turn in. Ask for the stronger
    // property, and prefer the roomiest spot that has it.
    // Falls back to nearest_free rather than failing: a standpoint that is merely reachable still beats
    // no standpoint, and this is a preference for clearance, not a new precondition for servicing an
    // object. When facing yaw is off there is no terminal rotation, so the old question is the right one.
    const bool will_rotate_here = wants_final_facing(step.target);
    const auto safe = routing_failed_here
                    ? unroutable_fix_
                    : will_rotate_here
                        ? [&]() -> std::optional<Eigen::Vector2f>
                          {
                              if (const auto r = grid_planner_.nearest_rotatable(step.target.room_pos);
                                  r.has_value())
                                  return r;
                              return grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad);
                          }()
                        : grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad);
    if (safe.has_value() && (*safe - step.target.room_pos).squaredNorm() > 1e-6f)
    {
        // Logged only when the ANSWER changes. Repair is deterministic and runs every cycle, so an
        // unchanged repair printed an identical line at 20 Hz — which buried the one line that mattered
        // (the hold) and made a steady state look like thrashing.
        if (not last_repair_.has_value() or (*last_repair_ - *safe).squaredNorm() > 1e-6f
            or last_repair_name_ != step.target.node_name)
        {
            last_repair_ = *safe;
            last_repair_name_ = step.target.node_name;
            std::println("[controller] target '{}' {} ({:.2f},{:.2f}) → ({:.2f},{:.2f}), {:.2f} m short{}",
                         step.target.node_name,
                         routing_failed_here ? "NOT REACHABLE → closest reachable"
                                            : will_rotate_here ? "→ moved to a spot it can TURN in"
                                                               : "blocked → repaired",
                         step.target.room_pos.x(), step.target.room_pos.y(), safe->x(), safe->y(),
                         (*safe - step.target.room_pos).norm(),
                         routing_failed_here ? " [" + grid_planner_.last_failure() + "]" : "");
        }
        step.target.room_pos = *safe;

        // Re-aim the heading at the object now that the standpoint moved (the producer's yaw faced the
        // object from the original, now-discarded viewpoint).
        if (step.target.parent_node_id != 0)
            if (const auto obj = world_model.read_node_room_xy(step.target.parent_node_id, timestamp_ms);
                obj.has_value())
                step.target.yaw_rad = std::atan2(obj->y() - step.target.room_pos.y(),
                                                 obj->x() - step.target.room_pos.x());
    }
    else if (not safe.has_value())
    {
        // No point within the search radius satisfies the clearance, so we are about to hand the planner a
        // target it cannot reach — it will fail, the no-progress clock will run, and stuck-recovery will start
        // reversing and dropping virtual obstacles. That was previously SILENT, which made it look like a
        // planner or perception fault rather than an unreachable goal. Raising the clearance to match the
        // controller makes this outcome more likely, so it must be visible. Rate-limited: the condition
        // persists for as long as the target does, and one line per cycle would bury everything else.
        if (timestamp_ms - last_unreachable_log_ms_ >= 3000)
        {
            last_unreachable_log_ms_ = timestamp_ms;
            std::println("[controller] target '{}' at ({:.2f},{:.2f}) is BOXED IN — no footprint-feasible pose "
                         "within reach. Planning will fail until the obstacles change.",
                         step.target.node_name, step.target.room_pos.x(), step.target.room_pos.y());
        }
    }

    current_target_room_ = step.target.room_pos;
    step.target_changed = !last_target_info_.has_value()
                       || !ControllerWorldModel::same_target_instance(*last_target_info_, step.target);
    last_target_info_ = step.target;
    active_target_id_ = target->node_id;
    return step;
}

// The optimiser's configuration, in ONE place. Every constant here is read off a measured physical
// quantity of THIS robot, not tuned:
//   d_target = what the body actually occupies (worst-case reach, since the field carries no bearing)
//              plus the same comfort standoff the MPPI prefers,
//   rho      = v_max^2 / a_lat_max, the radius below which a turn stops being drivable at speed,
//   sigma_a  = a stated fidelity allowance — how far the route may drift from what was clicked — in
//              the same sense as carrot_max_route_cut_m, not a safety number.
// The distance field is EXACT (GridPlanner's EDT): an optimiser follows the gradient of whatever it is
// handed, so a chamfer's direction-dependent error would be baked into the route's SHAPE.
rc::RouteOptimizerConfig ControllerSession::make_route_optimizer_config() const
{
    rc::RouteOptimizerConfig opt;
    opt.enabled = params_ == nullptr or params_->route_optimize;
    opt.distance = [this](const Eigen::Vector2f &p) { return grid_planner_.distance_at(p); };
    opt.distance_gradient = [this](const Eigen::Vector2f &p) { return grid_planner_.distance_gradient_at(p); };
    opt.d_target = rc::RobotFootprint::shadow().circumscribed_radius()
                 + (params_ ? params_->comfort_standoff_m : 0.35f);
    const float v_max = params_ ? params_->max_adv_speed_mps : 0.7f;
    opt.rho = v_max * v_max / std::max(0.05f, params_ ? params_->max_lateral_accel_mps2 : 1.0f);
    opt.sigma_a = 0.30f;
    opt.clearance_floor = rc::RobotFootprint::shadow().circumscribed_radius();
    opt.iterations = 30;
    opt.safety_bias = params_ ? params_->route_safety_bias : 0.5f;
    return opt;
}

ControllerPolygon ControllerSession::smooth_plan(const ControllerPolygon &poly)
{
    plan_spline_valid_ = false;
    plan_progress_s_ = 0.f;
    // ★THE CURVE IS BUILT EVEN WHEN SMOOTHING IS OFF. PlainTracker steers at a curve, so plan_spline_
    // is what makes it able to drive a click target at all; SmoothPlannedPath governs only whether the
    // smoothed SAMPLES replace the polyline handed downstream. Conflating the two made a display-level
    // preference silently decide whether the robot moves.
    if (poly.size() < 2 or params_ == nullptr) return poly;
    const bool replace_samples = params_->smooth_planned_path;
    rc::RouteSpline &spline = plan_spline_;
    // ★OPTIMISE THIS PATH TOO. It used to pass no optimiser at all, so an AFFORDANCE or click target
    // got A* + a C2 fit and nothing else: the clearance term never ran, the safety slider did nothing,
    // and BandEnabled did nothing either (step_route_band returns unless a mission is running). Which
    // is exactly what "I don't see any movement in the path" was. A driven path is a driven path — the
    // clearance preference should not depend on whether a MISSION happens to be the thing driving.
    const rc::RouteOptimizerConfig opt = make_route_optimizer_config();
    if (not spline.build(poly, params_->route_spacing_m,
                         [this](const Eigen::Vector2f &p, float h) { return grid_planner_.pose_free(p, h); },
                         params_->route_smoothing_m, &opt))
        return poly;   // smoothing is an improvement, never a precondition: fall back to the polyline
    plan_spline_valid_ = true;
    return replace_samples ? spline.samples() : poly;
}

void ControllerSession::log_route_geometry()
{
    if (!route_geom_csv_open_)
    {
        route_geom_csv_.open("route_geometry.csv", std::ios::out | std::ios::trunc);
        if (route_geom_csv_.is_open()) route_geom_csv_ << "event_id,kind,i,x,y\n";
        route_geom_csv_open_ = true;
    }
    if (!route_geom_csv_.is_open()) return;
    const auto &poly = route_.polyline();
    for (std::size_t i = 0; i < poly.size(); ++i)
        route_geom_csv_ << route_event_id_ << ",astar," << i << ',' << poly[i].x() << ',' << poly[i].y() << '\n';
    const auto &curve = route_.path();
    for (std::size_t i = 0; i < curve.size(); ++i)
        route_geom_csv_ << route_event_id_ << ",smoothed," << i << ',' << curve[i].x() << ',' << curve[i].y() << '\n';
    route_geom_csv_.flush();
}

void ControllerSession::dump_route_world(const Eigen::Vector2f &start,
                                         const std::vector<Eigen::Vector2f> &raw,
                                         const std::vector<Eigen::Vector2f> &repaired,
                                         int laps,
                                         const rc::RouteOptimizerConfig &opt) const
{
    std::ofstream f("route_world.txt", std::ios::out | std::ios::trunc);
    if (not f.is_open()) return;
    // Locale-proof the writer (CLAUDE.md): these machines run es_ES, where the decimal separator is a
    // COMMA. std::ofstream formats through the C++ global locale, which stays "C" unless someone calls
    // std::locale::global — but this file is parsed back by tools that read with the C library, and a
    // comma here would silently truncate every coordinate to its integer part. Cheap insurance.
    f.imbue(std::locale::classic());
    f << std::setprecision(9);
    f << "# route world snapshot — ControllerSession::build_route. Replay with tools/route_bench.\n";
    f << "version 1\n";
    f << "mission " << mission_.selected_name() << '\n';
    f << "laps " << laps << '\n';
    f << "start " << start.x() << ' ' << start.y() << '\n';
    // Both waypoint sets: the tour as RECORDED and as REPAIRED to feasible poses. The bench must drive
    // the repaired ones (they are what was planned through) but anchor-fidelity is only meaningful
    // against what was actually asked for, and the two differ by however much the world moved.
    for (const auto &p : raw)      f << "wp_raw "  << p.x() << ' ' << p.y() << '\n';
    for (const auto &p : repaired) f << "wp_safe " << p.x() << ' ' << p.y() << '\n';
    if (params_ != nullptr)
        f << "fit " << params_->route_spacing_m << ' ' << params_->route_smoothing_m << ' '
          << params_->max_adv_speed_mps << ' ' << params_->max_lateral_accel_mps2 << ' '
          << params_->comfort_standoff_m << '\n';
    // EVERY field the solve consumed, so the bench replays rather than infers. New fields go on the END:
    // the reader takes them in order and leaves anything missing at its default, so an older snapshot
    // stays readable instead of becoming a file that parses into a subtly different optimiser.
    f << "opt " << opt.d_target << ' ' << opt.rho << ' ' << opt.sigma_a << ' ' << opt.clearance_floor << ' '
      << opt.w_kappa << ' ' << opt.w_clear << ' ' << opt.w_gauge << ' ' << opt.clear_peak << ' '
      << opt.anchor_huber << ' ' << opt.iterations << ' ' << opt.kappa_peak << ' '
      << opt.safety_bias << '\n';
    grid_planner_.write_grid(f);
}

void ControllerSession::log_mppi_diagnostics(std::uint64_t t_ms,
                                             const rc::TrajectoryController::ControlOutput &o,
                                             float commanded_adv, float measured_speed,
                                             float path_kappa, float track_s, float measured_rot,
                                             float pose_xy_std, float pose_theta_std,
                                             const ControllerRobotPose &robot_pose,
                                             const ControllerMotionCommander::OutputRateStats &ors,
                                             float pose_stamp_age)
{
    if (!mppi_csv_open_)
    {
        mppi_csv_.open("mppi_diag.csv", std::ios::out | std::ios::trunc);
        if (mppi_csv_.is_open())
            mppi_csv_ << "# per-cycle control record. The ess/lambda/g_* columns describe the MPPI SAMPLER\n"
                         "# and are ZERO when ControlMode=pd — that is the sampler not running, not a bug.\n"
                         "# The gate_* columns are the SAFETY GATE. In pd mode it is the ONLY thing between\n"
                         "# the tracker and an obstacle, and all six are populated. In mppi mode the gate has a\n"
                         "# different shape (a ladder plus backup manoeuvres, and it is ARMED by a frontal-lidar\n"
                         "# cone, so it does not run every cycle) — there only sg_trig, gate_horizon, gate_min_esdf\n"
                         "# and gate_hard_coll are written; gate_scale/gate_hard_stop keep their defaults.\n"
                         "#   gate_scale   = fraction of commanded adv it let through (1 = untouched, pd only)\n"
                         "#   gate_horizon = its lookahead this cycle (speed-dependent: v/a_decel + 0.15)\n"
                         "#   gate_min_esdf= worst clearance along the PREDICTED arc; -1 = gate did not run\n"
                         "#   gate_hard_stop = even adv=0 was unsafe, so it rotated away instead (pd only)\n"
                         "# track_s = the TRACKER's OWN arc length (m), route_length - dist_to_goal.\n"
                         "#   DISTINCT from profile.csv's route_s_m, which is RouteFollower::progress() —\n"
                         "#   a different projection with a different search window. They can disagree, and\n"
                         "#   only THIS one drives the control law. -1 = no continuous route.\n"
                         "# path_kappa = SIGNED route curvature at the robot's projection (1/m). Sentinel\n"
                         "#   -999 = no continuous route, which is NOT the same as a straight (kappa=0).\n"
                         "# pd_cross_err_m = the cross-track error THE PD LAW SAW: signed lateral offset of the\n"
                         "#   path in the ROBOT frame (+ = path to the right), from the polyline projection.\n"
                         "#   ★NOT the run JSON's cross_track_rms_m, which the session computes against the\n"
                         "#   SPLINE with the OPPOSITE sign and for both modes. Two different estimators; do\n"
                         "#   not mix them. This one is 0 in mppi mode (the law does not run).\n"
                         "#   With path_kappa this is the pair a gain self-tuner needs: under-gain shows\n"
                         "#   as e correlated with kappa, over-gain as e oscillating about zero.\n"
                         "# pose_stamp_age = END-TO-END perception latency in ms: wall clock now minus the\n"
                         "#   VALIDITY STAMP on the room<-robot RT edge, which room_concept sets from the\n"
                         "#   lidar scan that produced the pose. So it spans lidar capture -> room_concept\n"
                         "#   -> DSR -> this read. -1 = the edge carries no stamp history, which is not\n"
                         "#   the same as zero latency. This replaces a residual with a measurement.\n"
                         "# out_* / ice_* = the ACTUATION path, per control cycle (the same numbers the\n"
                         "#   5 s [vel-out] line prints — it is per-cycle, not per-5s, because the stats\n"
                         "#   reset on read and only the PRINT is throttled). ice_ms is the synchronous\n"
                         "#   setSpeedBase RPC to the bridge, a HARD FLOOR on the output period: measured\n"
                         "#   0.2-41.3 ms, so the delivery delay swings ~40 ms cycle to cycle. Variable\n"
                         "#   actuation delay costs more phase margin than fixed delay of the same size.\n"
                         "#   Lagged one control cycle (cached, so reading it cannot steal the window).\n"
                         "# model_dropped = room-frame MODEL points (furniture, room polygon) the ESDF\n"
                         "#   dropped because the lidar already reported an obstacle there. STEADY means\n"
                         "#   model and measurement agree — the normal case, NOT a fault. A sharp CHANGE\n"
                         "#   means they disagree, i.e. the pose moved under the model, so this is a\n"
                         "#   pose-jump detector that does not depend on the (blind) reported sigma.\n"
                         "# pose_x/pose_y/pose_th = the RAW room-frame pose this cycle was computed from,\n"
                         "#   logged so a localisation JUMP can be seen directly rather than inferred from\n"
                         "#   its consequences. A jump is |d(pose)| far larger than the commanded speed can\n"
                         "#   account for over the elapsed time. NOTE the covariance channel cannot show\n"
                         "#   this: a localiser that snaps to a scan match is CONFIDENT about the new pose,\n"
                         "#   so pose_xy_std stays small while the estimate teleports.\n"
                         "# carrot_bear/carrot_dist = the STEERING TARGET both modes chase, AFTER\n"
                         "#   clip_carrot_to_reachable. If this oscillates while cross-track is smooth, the\n"
                         "#   target is moving and neither control law is at fault.\n"
                         "# pose_xy_std / pose_theta_std = the localiser's own covariance, i.e. HOW NOISY\n"
                         "#   THE POSE IS. -1 = the RT edge carried no covariance, so the value is absent,\n"
                         "#   not zero. Lagged by one cycle (~100 ms) against a ~5 Hz localiser.\n"
                         "# bump_push = PD lateral bumper, signed, in [-1,1]: + = pushed RIGHT because\n"
                         "#   something is close on the LEFT. 0 = both sides clear (the term is one-sided,\n"
                         "#   so 0 means no deficit, NOT that the bumper is disabled). gap_l/gap_r are the\n"
                         "#   body-to-obstacle gaps it read; -1 = the bumper did not run.\n"
                         "# meas_rot = SIGNED measured angular rate (rad/s), EMA-smoothed and differenced\n"
                         "#   from the localiser pose (~5 Hz) — so it LAGS. Adequate to identify a plant lag\n"
                         "#   of order 0.2-0.5 s; do not read faster dynamics out of it.\n"
                         "t_ms,ess,ess_K,ess_ratio,lambda_used,lambda_adaptive,cost_range,cost_best,"
                         "g_goal,g_obs,g_vel,g_smooth,g_lat,g_cbf,n_collisions,"
                         "cmd_adv,cmd_rot,meas_speed,min_esdf,explore,p_free,steer_conc,side_asym,"
                         "sg_trig,gate_scale,gate_horizon,gate_min_esdf,gate_hard_stop,gate_hard_coll,"
                         "pd_cross_err_m,path_kappa,track_s,meas_rot,bump_push,gap_l,gap_r,pose_xy_std,"
                         "pose_theta_std,carrot_bear,carrot_dist,pose_x,pose_y,pose_th,model_dropped,"
                         "out_ticks,out_period_ms,out_period_max,ice_ms,ice_max,cmd_age_max,fresh_min,"
                         "pose_stamp_age\n";
        mppi_csv_open_ = true;
    }
    if (!mppi_csv_.is_open()) return;
    const float ratio = o.ess_K > 0 ? o.ess / static_cast<float>(o.ess_K) : 0.f;
    mppi_csv_ << t_ms << ',' << o.ess << ',' << o.ess_K << ',' << ratio << ','
              << o.lambda_used << ',' << o.lambda_adaptive << ',' << o.cost_range << ',' << o.cost_best << ','
              << o.g_goal << ',' << o.g_obs << ',' << o.g_vel << ',' << o.g_smooth << ','
              << o.g_lat << ',' << o.g_cbf << ',' << o.n_collisions << ','
              << commanded_adv << ',' << o.rot << ',' << measured_speed << ',' << o.min_esdf << ',' << o.explore << ','
              << o.p_free << ',' << o.steering_concentration << ',' << o.side_asymmetry << ','
              << (o.safety_guard_triggered ? 1 : 0) << ',' << o.gate_speed_scale << ','
              << o.gate_horizon_s << ',' << o.gate_min_esdf << ','
              << (o.gate_hard_stop ? 1 : 0) << ',' << (o.gate_hard_collision ? 1 : 0) << ','
              << o.cross_track_m << ',' << path_kappa << ',' << track_s << ',' << measured_rot << ','
              << o.pd_bumper_push << ',' << o.pd_gap_left_m << ',' << o.pd_gap_right_m << ','
              << pose_xy_std << ',' << pose_theta_std << ','
              << o.carrot_bearing_rad << ',' << o.carrot_dist_m << ','
              << robot_pose.pos.x() << ',' << robot_pose.pos.y() << ',' << robot_pose.theta << ','
              << o.esdf_model_dropped << ','
              << ors.ticks << ',' << ors.period_mean_ms << ',' << ors.period_max_ms << ','
              << ors.ice_mean_ms << ',' << ors.ice_max_ms << ','
              << ors.cmd_age_max_ms << ',' << ors.scale_min << ','
              << pose_stamp_age << '\n';
}

void ControllerSession::log_route_event(const char *event, bool ok, std::uint64_t t_ms,
                                        const rc::TrajectoryController &path_controller,
                                        float window_m)
{
    if (!route_events_csv_open_)
    {
        route_events_csv_.open("route_events.csv", std::ios::out | std::ios::trunc);
        if (route_events_csv_.is_open())
            route_events_csv_ << "t_ms,event,ok,mission,route_len_m,samples,corrections,window_m,"
                                 "opt_ran,opt_rejected,opt_iters,cost_before,cost_after,"
                                 "e_kappa,e_clear,e_anchor,e_gauge,clear_before_m,clear_after_m,max_move_m,"
                                 "esdf_boundary_cells,esdf_boundary_rejected,"
                                 "max_dev_m,mean_dev_m,detail\n";
        route_events_csv_open_ = true;
    }
    if (!route_events_csv_.is_open()) return;

    const auto &o = route_.spline().last_optimizer_report();
    route_events_csv_ << t_ms << ',' << event << ',' << (ok ? 1 : 0) << ','
                      << mission_.selected_name() << ','
                      << route_.length() << ',' << route_.path().size() << ',' << route_.corrections() << ','
                      << window_m << ','
                      << (o.ran ? 1 : 0) << ',' << (o.rejected ? 1 : 0) << ',' << o.iterations << ','
                      << o.cost_before << ',' << o.cost_after << ','
                      << o.e_kappa << ',' << o.e_clear << ',' << o.e_anchor << ',' << o.e_gauge << ','
                      << o.min_clearance_before << ',' << o.min_clearance_after << ',' << o.max_move_m << ','
                      << path_controller.esdf_boundary_cells() << ','
                      << (path_controller.esdf_boundary_rejected() ? 1 : 0) << ','
                      << route_.spline().max_deviation_m() << ',' << route_.spline().mean_deviation_m() << ','
                      // The planner's reason contains commas AND is only meaningful for the event that
                      // produced it. Sanitised so a naive splitter cannot be shifted a column, and
                      // consumed so it cannot go stale on the next row.
                      << [this]
                         {
                             std::string d = last_plan_failure_;
                             std::replace(d.begin(), d.end(), ',', ';');
                             last_plan_failure_.clear();
                             return d;
                         }() << '\n';
    route_events_csv_.flush();
    ++route_event_id_;
    log_route_geometry();
}

bool ControllerSession::build_route(const ControllerRobotPose &robot_pose)
{
    const auto *m = mission_.selected_mission();
    if (m == nullptr or m->waypoints.size() < 2) return false;

    // REPAIR EVERY WAYPOINT FIRST. A recorded waypoint is a DESIRE; what can actually be driven is the
    // nearest footprint-feasible pose to it. The world moves between recording and running — an obstacle
    // appears, a wall is re-estimated — so a raw waypoint is not guaranteed drivable, and planning
    // straight to it fails outright.
    // This is not a workaround: the per-waypoint mode did exactly this (nearest_free, below) on every
    // target, which is why it drove this same tour for five clean laps while the route builder could not
    // plan past waypoint 6. Moving the planning into the route builder dropped the repair with it.
    // Yaw is the direction of travel toward the NEXT waypoint — the pose the robot will actually present
    // there, and the one the planner searches under.
    std::vector<Eigen::Vector2f> wps, raw_wps;
    wps.reserve(m->waypoints.size());
    raw_wps.reserve(m->waypoints.size());
    int repaired = 0, skipped = 0;
    const int n = static_cast<int>(m->waypoints.size());
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f raw = m->waypoints[i].pos;
        const Eigen::Vector2f next = m->waypoints[(i + 1) % n].pos;
        const Eigen::Vector2f dir = next - raw;
        const float yaw = dir.squaredNorm() > 1e-9f ? std::atan2(dir.y(), dir.x()) : 0.f;
        const auto safe = grid_planner_.nearest_free(raw, yaw);
        if (not safe.has_value())
        {
            // Boxed in beyond the repair radius. Dropping one waypoint keeps the tour drivable, which is
            // better than refusing to move at all — but it CHANGES THE ROUTE, so it is said out loud
            // rather than swallowed: a benchmark whose stimulus quietly differs is worse than no run.
            std::println("[route] waypoint {} at ({:.2f},{:.2f}) is BOXED IN — dropping it from the route. "
                         "The driven route no longer matches the recorded one.", i + 1, raw.x(), raw.y());
            ++skipped;
            continue;
        }
        if ((*safe - raw).squaredNorm() > 1e-6f) ++repaired;
        wps.push_back(*safe);
        raw_wps.push_back(raw);
    }
    if (wps.size() < 2)
    {
        std::println("[route] fewer than 2 drivable waypoints remain — cannot build a route.");
        return false;
    }
    if (repaired > 0 or skipped > 0)
        std::println("[route] {} waypoint(s) moved to the nearest feasible pose, {} dropped.",
                     repaired, skipped);

    // ── VARIATIONAL ROUTE OPTIMISATION ──
    // Every constant here is read off a measured physical quantity of THIS robot, not tuned:
    //   d_target = what the body actually occupies (worst-case reach, since the field carries no bearing)
    //              plus the same comfort standoff the MPPI prefers,
    //   rho      = v_max^2 / a_lat_max, the radius below which a turn stops being drivable at speed,
    //   sigma_a  = a stated fidelity allowance — how far the route may drift from what was clicked —
    //              in the same sense as carrot_max_route_cut_m, not a safety number.
    // The distance field is EXACT (GridPlanner's EDT): an optimiser follows the gradient of whatever it
    // is handed, so a chamfer's direction-dependent error would be baked into the route's shape.
    {
        const rc::RouteOptimizerConfig opt = make_route_optimizer_config();
        route_.set_optimizer(opt);
        dump_route_world(robot_pose.pos, raw_wps, wps, mission_.laps_remaining(), opt);
    }

    last_plan_failure_.clear();
    const bool built = route_.build(
        robot_pose.pos, wps, mission_.laps_remaining(),
        [this](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
        {
            auto r = grid_planner_.plan(a, b);
            if (not r.has_value()) last_plan_failure_ = grid_planner_.last_failure();
            return r;
        },
        [this](const Eigen::Vector2f &p, float h) { return grid_planner_.pose_free(p, h); },
        params_ ? params_->route_spacing_m : 0.05f,
        params_ ? params_->route_smoothing_m : 0.40f);
    route_active_ = built;
    if (not built and not last_plan_failure_.empty())
        std::println("[route] the planner refused that hop: {}", last_plan_failure_);
    return built;
}

float ControllerSession::route_speed_limit(float v_cap, float a_decel) const
{
    if (not route_active_ or not route_.valid() or params_ == nullptr) return v_cap;

    const float a_lat = std::max(0.05f, params_->max_lateral_accel_mps2);
    const float a_dec = std::max(0.05f, a_decel);
    const float s_now = route_.progress();
    // Look ahead by the distance needed to stop from the cap, plus a little: anything closer than that
    // is something we must ALREADY be slowing for. Beyond it, no amount of braking is required yet.
    const float horizon = v_cap * v_cap / (2.f * a_dec) + 1.0f;

    // ── The curvature the ROTATION limit must use is not the one the lateral limit uses ──
    // omega = v*kappa <= max_rot gives v <= max_rot/kappa, which is 1/kappa — it AMPLIFIES an
    // overestimate of curvature, where the lateral limit's sqrt(a_lat/kappa) damps it. Fed the point
    // value of curvature_at (a second difference, and the comment below says why that is noisy at this
    // spacing) it throttled 15% of a lap below 0.40 m/s and cost 22% of the lap time, for curvature
    // structure finer than the 0.40 m scale the route was actually fitted at — i.e. mostly fitting noise.
    //
    // What a differential drive must physically deliver is the NET HEADING CHANGE over the stretch it is
    // about to drive, so the right quantity is the AVERAGE curvature over that stretch:
    //     kappa_avg = |psi(s+W) - psi(s)| / W      (the mean value theorem, applied honestly)
    // Two things make this the better estimator, not merely a smoother one:
    //   • heading_at is a FIRST difference of positions; curvature_at is a SECOND difference. One order
    //     less differentiation is one order less noise amplification.
    //   • a lone 5 cm curvature spike contributes almost nothing to the net heading change, so it stops
    //     mattering by construction rather than by being filtered out — while a sustained tight curve
    //     accumulates its full turn and still binds.
    // W is the route's own smoothing scale: the curve was fitted with control points that far apart, so
    // curvature structure finer than W is a property of the fit, not of the route.
    const float w_kappa = std::max(0.10f, params_->route_smoothing_m);
    // ONE estimator, shared with the tracker: RouteSpline::kappa_avg is CENTRED, so the forward window
    // [s, s+W] this limit wants is the centred window at s + W/2. Expressing it that way keeps a single
    // implementation instead of two that can drift apart, and is exactly the previous arithmetic.
    const auto kappa_avg_at = [this, w_kappa](float s)
    { return std::abs(route_.spline().kappa_avg(s + 0.5f * w_kappa, w_kappa)); };

    float v = v_cap;
    // The tightest rotation-limited speed the scan finds. The floor at the bottom may not exceed it —
    // see the note there; it is what makes a cusp followable at all.
    float v_rot_min = v_cap;
    // Sampled at 10 cm against a 5 cm curve — deliberately coarser than the curve's own spacing, because
    // curvature_at is a second difference of the samples and is therefore noisiest at that scale.
    for (float ds = 0.f; ds <= horizon; ds += 0.10f)
    {
        const float k = std::abs(route_.spline().curvature_at(s_now + ds));
        if (k < 1e-3f) continue;                       // straight: no constraint from here
        // TWO limits, and only one of them was here. v^2·kappa = a_lat is the COMFORT limit — how much
        // lateral acceleration the payload will accept, on the POINT curvature as before. omega = v·kappa
        // <= max_rot is the KINEMATIC one: a differential drive physically cannot hold curvature kappa
        // faster than max_rot/kappa, whatever the lateral budget says — and it takes the AVERAGED
        // curvature, for the reasons given above the loop.
        // ★Measured at the authored wp21/wp22 hairpin (0.35 m across a 104 deg turn): sqrt(a_lat/kappa)
        // permits 0.41 m/s, which demands omega = 2.4 rad/s against a max_rot of 0.8. The robot was being
        // allowed a speed it could not turn at, so it left the route — reported as "cuts quite a bit on
        // the hairpin". The MPPI never showed this because its rollouts integrate the real kinematics
        // with rot clamped, so a too-fast rollout visibly fails to track and is scored down; a geometric
        // tracker has no such foresight and simply cannot comply.
        const float v_lat = std::sqrt(a_lat / k);      // v^2·kappa = a_lat — comfort, POINT curvature
        const float k_avg = kappa_avg_at(s_now + ds);  // net heading change / arc length — see above
        // ★HEADROOM. omega_max/kappa hands the FEEDFORWARD the entire rotation budget, so on a curve
        // the command saturates and the feedback loop is effectively open — measured in tracker_sim as
        // a systematic UNDER-turn (the robot rides outside the curve, corr(e,kappa) -0.160). Reserving
        // a fraction for feedback moved rms 154 -> 94 mm at 0.70 and corr to ~0. Inert for the PD
        // tracker, which has no feedforward to saturate, so it is applied in ROUTE mode only.
        const float rot_budget = std::max(0.05f, params_->max_rot_speed_rps)
                               * (route_tracker_active_ ? rot_headroom_ : 1.0f);
        const float v_rot = k_avg > 1e-3f ? rot_budget / k_avg : v_cap;
        v_rot_min = std::min(v_rot_min, v_rot);
        const float v_here = std::min(v_lat, v_rot);
        // The bound is on the speed we may hold NOW: we must be able to shed the difference over ds.
        const float v_allowed = std::sqrt(v_here * v_here + 2.f * a_dec * ds);
        v = std::min(v, v_allowed);
    }
    // A floor purely against numerical noise in the curvature estimate: a spurious kappa spike must not
    // be able to command a standstill.
    // ★2026-08-05 — IT MUST NEVER OVERRULE THE ROTATION BUDGET, and until now it did. The old form was a
    // flat clamp to 0.15 m/s, justified by "a differential drive has no minimum turn radius — it can
    // rotate in place — so a sharp corner is handled by the rotation". That is true of the ROBOT and
    // FALSE of the TRACKER: the plain tracker's omega = g_dc*v*kappa is PROPORTIONAL to v, so it cannot
    // rotate in place, and flooring v is therefore the same as demanding a turn rate.
    // Measured on this tour (tools/tracker_sim): the route reaches |kappa_avg| = 7.79 1/m — radius
    // 0.13 m, against a 0.325 m circumscribed body — at s=23.9, with a 52.5 degree heading step between
    // adjacent 5 cm samples, i.e. a cusp. At the 0.15 floor that demands omega = 1.17 rad/s against a
    // 0.8 limit, so the command saturates, the robot under-turns, leaves the route and the Frenet
    // feedback diverges: 4675 mm rms on the two-lap tour against 157 mm on one lap, with a PERFECT pose
    // and no obstacles. The robot's lap 5 carried the same signature — 1.03 m rms, 4.32 m max, and 31%
    // extra distance driven.
    // Flooring at min(0.15, v_rot_min) keeps the noise protection everywhere it was doing a job — on a
    // straight v_rot is metres per second, so the floor is unchanged at 0.15 — while letting a genuine
    // corner slow to the speed the robot can actually turn through it (about 0.07 m/s at that cusp).
    return std::clamp(v, std::min(0.15f, v_rot_min), v_cap);
}

// ── DRIVE-MODE SEPARATION ────────────────────────────────────────────────────────────────────────
// ensure_current_plan is a DISPATCHER now, nothing more. It used to hold both regimes back to back in
// one 260-line function, sharing current_plan_, plan_spline_valid_ and route_repair_pending_ between
// them, so a reader had to track which of the two any given line belonged to — and a mission-side
// mechanism could reach a point target's geometry without anything looking wrong. The two answer
// different questions: "keep driving the curve I already have" versus "plan me a path to there".
// What stays here is only what genuinely belongs to BOTH: the escape maneuver owns the base regardless
// of mode, and a fresh target clears the stuck clock regardless of mode.
bool ControllerSession::ensure_current_plan(const ControllerPlanningStep &step,
                                            ControllerObstacleTracker &obstacle_tracker,
                                            rc::TrajectoryController &path_controller,
                                            ControllerMotionCommander &motion_commander,
                                            ControllerDisplay &display,
                                            const TimeSource &time_source)
{
    // An escape maneuver (physical-stuck recovery) owns the base and must run even when NO plan
    // exists — e.g. the robot is boxed in and plan_path keeps failing. Step it here, before any
    // (re)planning, so the reverse-out completes. begin_escape reset current_plan_, so once the
    // escape finishes the next plan_path routes around the temp obstacle dropped at the wedge spot.
    if (escape_active_)
    {
        display.set_stuck_active(true);   // this path returns before compute()'s update_custom_widget
        step_escape(step.robot_pose, path_controller, motion_commander, time_source());
        return false;
    }

    if (step.target_changed)
        reset_stuck_state();   // a fresh target = fresh navigation; don't inherit a stale stuck clock

    // ── CONTINUOUS ROUTE ──
    // Built once, driven in arc-length coordinates. Nothing below this block runs in that mode: there is
    // no target to replan to, and re-issuing a path is exactly what destroys the follower's continuity.
    if (params_ and params_->route_continuous and mission_.running())
        return drive_mission_route(step, path_controller, motion_commander, time_source);
    return drive_point_target(step, obstacle_tracker, path_controller, motion_commander,
                             display, time_source);
}

// ── MISSION ROUTE ────────────────────────────────────────────────────────────────────────────────
// Built once and driven in ARC-LENGTH coordinates. There is no target to replan to here, and
// re-issuing a path is exactly what destroys the follower's continuity — so this regime repairs
// its curve in place and never re-plans. Nothing in drive_point_target runs in this mode.
bool ControllerSession::drive_mission_route(const ControllerPlanningStep &step,
                                           rc::TrajectoryController &path_controller,
                                           ControllerMotionCommander &motion_commander,
                                           const TimeSource &time_source)
{
    // A failed build must not be retried every cycle: that is ~30 A* calls and a screenful of log
    // at 10 Hz, which buries the one line that says why. Try, then wait before trying again.
    if (not route_active_ and time_source() - last_route_build_ms_ >= 2000)
    {
        last_route_build_ms_ = time_source();
        const bool ok = build_route(step.robot_pose);
        log_route_event("build", ok, time_source(), path_controller, 0.f);
    }
    if (not route_active_)
    {
        path_controller.stop();
        motion_commander.stop_robot();
        if (time_source() - last_no_route_log_ms_ >= 3000)
        {
            last_no_route_log_ms_ = time_source();
            std::println("[controller] HOLDING — the mission route could not be built.");
        }
        return false;
    }
    // ── LOCAL REPAIR ──
    // A recovery reflex fired and put a new obstacle in the planner's world. In leg mode the next
    // cycle simply replanned to the current target; here there is no target to replan to, so the
    // route itself has to be re-authored around the blocker or nothing changes at all. Rate-limited
    // for the same reason the build is: a repair is ~one A* call plus a refit, and retrying it every
    // cycle would bury the one line that says whether it worked.
    if (route_repair_pending_ and route_active_
        and time_source() - last_route_repair_ms_ >= 1500)
    {
        last_route_repair_ms_ = time_source();
        // How much route may be re-authored. Behind: the robot has just reversed out, so the detour
        // must start somewhere it can still reach. Ahead: far enough to clear the blocker and rejoin.
        constexpr float kRepairBackM = 1.0f;
        constexpr float kRepairAheadM = 4.0f;
        const auto result = route_.repair(step.robot_pose.pos, kRepairBackM, kRepairAheadM,
                          [this](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
                          {
                              auto r = grid_planner_.plan(a, b);
                              if (not r.has_value()) last_plan_failure_ = grid_planner_.last_failure();
                              return r;
                          },
                          [this](const Eigen::Vector2f &p, float hdg) { return grid_planner_.pose_free(p, hdg); });

        using RR = rc::RouteFollower::RepairResult;
        if (result == RR::NotNeeded)
        {
            // The reflex fired but the route across the window is still footprint-feasible: whatever
            // was seen is not on our path. Clear the request and KEEP DRIVING — the expensive mistake
            // here is not a missed repair, it is re-authoring a good route and stopping to do it.
            route_repair_pending_ = false;
            log_route_event("repair_skipped", true, time_source(), path_controller, kRepairBackM + kRepairAheadM);
        }
        else if (result == RR::Repaired)
        {
            route_repair_pending_ = false;
            ++route_repair_count_;
            mission_.note_replan();   // count the repair that HAPPENED, not the reflex that asked for one
            // Force the repaired curve to be installed: the follower is still holding the old one.
            path_controller.stop();
            current_plan_.reset();
            plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
            log_route_event("repair", true, time_source(), path_controller, kRepairBackM + kRepairAheadM);
        }
        else
        {
            // HOLD, and stay pending. The obstacle carries a TTL, so the retry after the cooldown
            // either finds a detour or finds the blocker gone. Rebuilding the whole route would get
            // the robot moving again, but it resets progress_ — laps and finished() would believe the
            // robot was back at the start, and the run's metrics would silently stop meaning anything.
            mission_.note_replan();
            path_controller.stop();
            motion_commander.stop_robot();
            log_route_event("repair_failed", false, time_source(), path_controller, kRepairBackM + kRepairAheadM);
            return false;
        }
    }
    if (not path_controller.is_active())
    {
        // set_path_presmoothed: the curve is already C2 and already footprint-checked, so the
        // elastic band and the C1 spline inside set_path would only undo both.
        path_controller.set_path_presmoothed(route_.path());
    // The ROUTE tracker reads s, psi(s) and kappa_avg(s) from the curve itself. Non-owning: the band
    // deforms this same spline in place just before compute, so the tracker sees the deformed curve
    // without a state reset — which is the whole reason update_path_geometry exists for the polyline.
    path_controller.set_route(&route_.spline());
    // ── WHAT THIS ROUTE MAKES UNAVOIDABLE ───────────────────────────────────────────────────────
    // Computed ONCE per route, from the spline and the identified plant. It is what makes J_route
    // route-independent: the run's totals are divided by these, so each term reads ">= 1, where 1 is as
    // well as the plant permits HERE" instead of being dominated by how much this particular tour turns.
    // Same inputs route_speed_limit uses, so v* is the profile the robot will actually be held to.
    {
    const float v_cap = params_ ? params_->max_adv_speed_mps : 0.7f;
    const float a_lat = params_ ? params_->max_lateral_accel_mps2 : 1.0f;
    const float w_max = params_ ? params_->max_rot_speed_rps : 0.8f;
    const auto &tp = path_controller.params;
    const rc::RouteIdeal ideal = rc::route_ideal(route_.spline(), v_cap, a_lat, tp.cbf_max_decel,
                                                 w_max, tp.plain_W, tp.plain_T_lag,
                                                 route_tracker_active_ ? rot_headroom_ : 1.0f);
    mission_.set_route_ideal(ideal.tv_v, ideal.tv_w, ideal.rms_e, ideal.valid);
    std::println("[route] ideal floor: TV(v*)={:.2f} m/s  TV(w*)={:.2f} rad/s  rms(e*)={:.4f} m "
                 "over {:.1f} m ({:.0f}% of the route contributes to TV(w*)){}",
                 ideal.tv_v, ideal.tv_w, ideal.rms_e, ideal.length_m,
                 ideal.length_m > 0.f ? 100.f * ideal.w_span / ideal.length_m : 0.f,
                 ideal.valid ? "" : "   ⚠INVALID — J_route will be NaN");
    }
        // Seed the carrot's forward-only anchor at the robot's own arc length. After a repair the
        // route is re-installed mid-drive, and a hint of 0 would aim the carrot at the route's start.
        path_controller.set_carrot_hint(
            static_cast<int>(route_.progress() / std::max(0.01f, route_.spline().spacing())));
        path_controller.set_goal_facing_yaw(std::nullopt);
        path_controller.set_arrival_point(std::nullopt);
        // A TOUR ENDS WHERE IT BEGAN, so the follower's euclidean "am I near the last point?" test is
        // true before the robot has moved: the route is installed, arrival fires on cycle 1, the
        // session consumes it as an affordance arrival ("reached -> REACH") and stops. Measured:
        // 09:34 completed the tour (progress 35.15/35.33 m); the next two runs recorded progress
        // 0.00 m and 0.10 m of motion, because the robot was now parked on the endpoint. The mission
        // is ended by arc length below (route_.finished()), which knows the difference between not
        // yet departed and returned — so that is the only arrival test left running here.
        path_controller.set_endpoint_arrival(false);
        current_plan_ = ControllerPathPlan{.room_path = route_.path()};
    }
    return true;
}

// ── POINT TARGET ─────────────────────────────────────────────────────────────────────────────────
// A click target or an affordance standpoint: plan a path to a POINT, smooth it, install it, and
// hold with a reason if no path exists. Never touches route_ — a mission owns that.
bool ControllerSession::drive_point_target(const ControllerPlanningStep &step,
                                          ControllerObstacleTracker &obstacle_tracker,
                                          rc::TrajectoryController &path_controller,
                                          ControllerMotionCommander &motion_commander,
                                          ControllerDisplay &display,
                                          const TimeSource &time_source)
{
    if (mission_.running() and params_ and not params_->route_continuous and not waypoint_mode_logged_)
    {
        waypoint_mode_logged_ = true;
        std::println("[route] mission running in WAYPOINT mode — RouteContinuous is false.");
    }
    if (step.target_changed || !current_plan_.has_value())
    {
        // FOOTPRINT PLANNER. The robot's real shape is tested against the grid; obstacles are NOT inflated.
        // Falls back to nothing — if this cannot find a route, the HOLD branch below reports precisely why.
        mission_.note_replan();   // counted whether or not it succeeds — a failed replan is the worse one
        if (const auto route = grid_planner_.plan(step.plan_origin, step.target.room_pos); route.has_value())
        {
            ControllerPathPlan plan{.room_path = *route};
            // Smooth it the same way a mission route is smoothed. A click target should not get a
            // visibly worse path than a tour does, and the follower's steering command follows curvature
            // either way — a C1 polyline steps it at every turning point regardless of who asked.
            plan.room_path = smooth_plan(plan.room_path);
            current_plan_ = std::move(plan);
        }
        else
        {
            // BRACES MATTER HERE: without them the invalidation below runs on the SUCCESS path too,
            // immediately undoing the smooth_plan() that just fitted the curve — and the band would
            // then never deform a plan, silently, while every flag said it was enabled.
            current_plan_.reset();
            plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
            // Tell the REPAIR stage that this exact target could not be routed. It cannot be discovered
            // there — reachability is global, and only the search knows it — so the next cycle's repair
            // asks for the nearest REACHABLE pose instead of merely the nearest free one.
            unroutable_target_name_ = step.target.node_name;
        }
    }

    if (!current_plan_.has_value() || current_plan_->room_path.empty())
    {
        update_display(step.robot_pose,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        // ── PLANNER FAILURE → HOLD. Deliberately NOT the escape reflex. ──
        // This branch used to feed the no-progress clock and, after stuck_confirm_ms, reverse+turn out. That
        // conflates two unrelated faults. A WEDGE is a prediction error — we command translation and the base
        // does not achieve it — and reversing is the right answer because backing off physically changes the
        // situation. "plan_path returned nothing" is not that: the robot may be standing in clear floor,
        // perfectly free to move, with the planner simply unable to produce a route. Reversing does not fix a
        // planner, so the reflex fires, ends, finds no route again, and fires again — a closed loop. Measured
        // live: the base sat at exactly the escape constants (−0.150 m/s, ±0.350 rad/s) for half a 35 s run,
        // beginning 1.3 s after start and never leaving.
        //
        // So: hold the base, and SAY SO. A stationary robot with a clear message is diagnosable; one thrashing
        // backwards looks like a perception or traction fault and hides the real cause. The stuck clock is
        // reset here so a genuine wedge later starts from a clean window rather than inheriting this one.
        reset_stuck_state();
        path_controller.stop();
        motion_commander.stop_robot();
        if (time_source() - last_no_route_log_ms_ >= 3000)
        {
            last_no_route_log_ms_ = time_source();
            std::println("[controller] HOLDING — no route to '{}': {}. Not escaping: the robot is not wedged, "
                         "the planner produced no path, and reversing cannot fix a planner.",
                         step.target.node_name, grid_planner_.last_failure());
        }
        return false;
    }

    if (step.target_changed || !path_controller.is_active())
    {
        // Presmoothed: smooth_plan already fitted the C2 curve AND checked every sample against the
        // footprint, so set_path's elastic band and C1 spline would only undo both.
        if (params_ and params_->smooth_planned_path)
            path_controller.set_path_presmoothed(current_plan_->room_path);
        else
            path_controller.set_path(current_plan_->room_path);
        // Arrival is judged at the waypoint we are actually going to, not at the end of the extended
        // path. Without this the mission would skip every waypoint but the last one on the horizon.
        // Affordance targets carry a desired facing yaw (point AT the table); manual
        // mouse targets do not, so they keep the legacy stop-on-arrival behaviour.
        // GoalFacingYawEnabled=false drops the facing requirement for affordances too, so arrival is
        // judged on POSITION alone and the robot never turns in place at the goal. Wanted for pure
        // navigation runs, where the terminal rotation is wasted motion and an extra way to hang at a
        // target; object affordances that must actually look at their object need it back on.
        // ★DERIVED FROM THE CONTRACT'S POLICY, not from a global switch alone — see
        // wants_final_facing(). GoalFacingYawEnabled remains a kill switch, but it is no longer the
        // whole answer: it used to be `from_affordance and <flag>`, so turning it on for an object
        // affordance also turned it on for every room waypoint, which is a Reach and has no use for
        // an orientation at all.
        const bool want_facing = wants_final_facing(step.target);
        path_controller.set_goal_facing_yaw(want_facing
                                                ? std::optional<float>(step.target.yaw_rad)
                                                : std::nullopt);
        // A point target DOES end at its endpoint — restore the follower's own arrival test, which a
        // continuous route switched off (it may have been the previous thing installed).
        path_controller.set_endpoint_arrival(true);
    }

    return true;
}


// ── FINAL-APPROACH BLACK BOX ─────────────────────────────────────────────────────────────────────
// WHY THIS EXISTS. An affordance standpoint is verified footprint-feasible at exactly ONE heading —
// the facing yaw, by the repair — and the plan is verified at the TRAVEL heading. Neither verifies the
// arc between them, and the terminal rotation sweeps precisely that arc. The body is 0.65 m wide with
// a 0.46 m inscribed width, so "feasible at both ends" does not imply "feasible in the middle", and a
// spot that is fine to stand in can be impossible to turn in.
//
// Worse, TrajectoryController's align branch sets adv=side=0, computes rot, and RETURNS — ahead of
// out.min_esdf and ahead of every safety stage. So while the robot turns at a standpoint it is running
// open-loop against the world: no ESDF, no footprint test, no guard of any kind. Nothing in the running
// system could answer "did it have room to turn", which is why this is a per-cycle record and not a
// post-hoc reconstruction.
//
// Three poses matter and they are NOT the same pose:
//   the TARGET      — the standpoint, the only one anything ever checked
//   where it STOPS  — anywhere within goal_threshold (0.25 m) of it, or beyond via passed_arrival
//   the SWEEP       — every heading the body passes through while turning, at wherever it stopped
// The columns below carry all three so a collision can be attributed to one of them.
void ControllerSession::log_approach_diagnostics(std::uint64_t t_ms,
                                                 const rc::TrajectoryController::ControlOutput &o,
                                                 const ControllerRobotPose &robot_pose)
{
    if (not last_target_info_.has_value() or not last_target_info_->from_affordance) { approach_active_ = false; return; }
    const auto &tgt = *last_target_info_;

    // The zone: the last metre in, plus the whole rotation regardless of distance. One metre is the
    // reporting window, not a control decision — nothing branches on it.
    const float d_target = (robot_pose.pos - tgt.room_pos).norm();
    const bool in_zone = o.aligning or d_target < 1.0f;
    if (not in_zone)
    {
        if (approach_active_) approach_active_ = false;
        return;
    }

    // Body theta -> the FORWARD heading the planner and the footprint both speak (+X right, +Y forward).
    const float facing_now = std::remainder(robot_pose.theta + static_cast<float>(M_PI_2), 2.f * static_cast<float>(M_PI));
    const float facing_des = tgt.yaw_rad;

    const float clear_now  = grid_planner_.pose_clearance(robot_pose.pos, facing_now);
    const float clear_des  = grid_planner_.pose_clearance(robot_pose.pos, facing_des);
    const bool  free_now   = grid_planner_.pose_free(robot_pose.pos, facing_now);
    const bool  free_des   = grid_planner_.pose_free(robot_pose.pos, facing_des);
    // ★The measurement that did not exist: can it turn where it ACTUALLY IS, not where it was aimed.
    const auto sweep       = grid_planner_.rotation_sweep(robot_pose.pos, facing_now, facing_des);
    // ... and the same question asked at the standpoint itself, which is what the repair should have
    // checked. Comparing the two separates "bad standpoint" from "stopped in a bad place".
    const auto sweep_tgt   = grid_planner_.rotation_sweep(tgt.room_pos, facing_now, facing_des);

    if (not approach_active_)
    {
        approach_active_ = true;
        approach_target_ = tgt.node_name;
        approach_start_ms_ = t_ms;
        approach_min_clear_ = std::numeric_limits<float>::max();
        approach_min_clear_align_ = std::numeric_limits<float>::max();
        // Say it BEFORE the robot commits, once per standpoint. This is the line that predicts the
        // collision instead of describing it: the sweep is evaluated at the standpoint, so it is known
        // as soon as the approach starts, a metre out, while there is still room to do something.
        if (approach_warned_ != tgt.node_name and not sweep_tgt.feasible)
            std::println("[approach] ⚠ '{}' — the standpoint ({:.2f},{:.2f}) is NOT rotatable: turning to "
                         "face {:.0f}° passes through {:.0f}° where the footprint does NOT fit "
                         "(tightest body clearance {:.3f} m). The terminal rotation runs with NO obstacle "
                         "check, so this is where a wall gets hit.",
                         tgt.node_name, tgt.room_pos.x(), tgt.room_pos.y(),
                         facing_des * 180.f / static_cast<float>(M_PI),
                         sweep_tgt.worst_heading_rad * 180.f / static_cast<float>(M_PI),
                         sweep_tgt.min_clearance_m);
        approach_warned_ = tgt.node_name;
    }
    approach_min_clear_ = std::min(approach_min_clear_, clear_now);
    if (o.aligning) approach_min_clear_align_ = std::min(approach_min_clear_align_, clear_now);

    if (not approach_csv_open_)
    {
        approach_csv_.open("approach_diag.csv", std::ios::out | std::ios::trunc);
        approach_csv_.imbue(std::locale::classic());   // decimal POINT regardless of LANG (see CLAUDE.md)
        approach_csv_open_ = approach_csv_.is_open();
        if (approach_csv_open_)
            approach_csv_
                << "# FINAL APPROACH to an affordance standpoint. One row per cycle inside 1 m, plus every\n"
                   "# cycle of the terminal rotation (which has NO obstacle check of its own).\n"
                   "# phase        = approach | ALIGN (rotating in place) | reached\n"
                   "# d_target_m   = robot to the STANDPOINT.  d_arrival_m = what the arrival test compares\n"
                   "#                against goal_threshold. They differ once the path is extended past it.\n"
                   "# clear_now/des_m = footprint slack at the CURRENT / DESIRED heading, at the robot's\n"
                   "#                actual pose. free_now/des = the same as the planner's yes/no predicate.\n"
                   "# sweep_min_m / sweep_ok = tightest slack over the headings the rotation passes through,\n"
                   "#                at the robot's ACTUAL pose. sweep_ok=0 means it cannot turn here.\n"
                   "# tgt_sweep_min_m / tgt_sweep_ok = the same asked at the STANDPOINT. tgt_sweep_ok=0 and\n"
                   "#                sweep_ok=0 => bad standpoint; tgt_sweep_ok=1 and sweep_ok=0 => it\n"
                   "#                stopped somewhere worse than where it was sent.\n"
                   "# min_esdf_m   = the LIVE field (residual + dynamic obstacles), which the model layers\n"
                   "#                above do not include. A gap between it and clear_now is a moving thing.\n"
                   "#                ★nan DURING ALIGN, AND THAT IS THE POINT: the align branch returns\n"
                   "#                before out.min_esdf is ever assigned, so the controller does not even\n"
                   "#                SAMPLE the field while turning. A 0 here would read as 'no clearance';\n"
                   "#                nan says 'not measured'. clear_now_m is the model-side answer instead.\n"
                   "t_ms,target,phase,tgt_x,tgt_y,tgt_facing_deg,rob_x,rob_y,rob_facing_deg,"
                   "d_target_m,d_arrival_m,yaw_err_deg,cmd_adv,cmd_rot,"
                   "clear_now_m,clear_des_m,free_now,free_des,sweep_min_m,sweep_ok,sweep_worst_deg,"
                   "tgt_sweep_min_m,tgt_sweep_ok,min_esdf_m,goal_reached\n";
    }
    if (not approach_csv_open_) return;

    constexpr float kDeg = 180.f / static_cast<float>(M_PI);
    approach_csv_ << t_ms << ',' << tgt.node_name << ','
                  << (o.goal_reached ? "reached" : (o.aligning ? "ALIGN" : "approach")) << ','
                  << tgt.room_pos.x() << ',' << tgt.room_pos.y() << ',' << facing_des * kDeg << ','
                  << robot_pose.pos.x() << ',' << robot_pose.pos.y() << ',' << facing_now * kDeg << ','
                  << d_target << ',' << o.dist_to_goal << ','
                  << (o.goal_yaw_err_rad.has_value() ? *o.goal_yaw_err_rad * kDeg : 0.f) << ','
                  << o.adv << ',' << o.rot << ','
                  << clear_now << ',' << clear_des << ',' << (free_now ? 1 : 0) << ',' << (free_des ? 1 : 0) << ','
                  << sweep.min_clearance_m << ',' << (sweep.feasible ? 1 : 0) << ','
                  << sweep.worst_heading_rad * kDeg << ','
                  << sweep_tgt.min_clearance_m << ',' << (sweep_tgt.feasible ? 1 : 0) << ','
                  << (o.aligning ? std::numeric_limits<float>::quiet_NaN() : o.min_esdf) << ','
                  << (o.goal_reached ? 1 : 0) << '\n';

    // On arrival, one console line that says how it actually went — the three quantities you would
    // otherwise reconstruct from the CSV by hand every time.
    if (o.goal_reached)
    {
        std::println("[approach] '{}' reached in {:.1f} s — stopped {:.3f} m from the standpoint, "
                     "yaw err {:.1f}°, tightest body clearance {:.3f} m (during rotation {:.3f} m), "
                     "live min_esdf {}{}",
                     tgt.node_name, static_cast<float>(t_ms - approach_start_ms_) * 1e-3f,
                     d_target, o.goal_yaw_err_rad.has_value() ? *o.goal_yaw_err_rad * kDeg : 0.f,
                     approach_min_clear_ == std::numeric_limits<float>::max() ? 0.f : approach_min_clear_,
                     approach_min_clear_align_ == std::numeric_limits<float>::max()
                         ? 0.f : approach_min_clear_align_,
                     o.aligning ? std::string("not sampled while aligning")
                                : std::format("{:.3f} m", o.min_esdf),
                     sweep.feasible ? "" : "  ⚠ROTATED THROUGH AN INFEASIBLE HEADING");
        approach_csv_.flush();
        approach_active_ = false;
    }
}


// One scalar off the feedback node, by contract-declared NAME. Uses the protocol's own attr_scalar so
// the panel and the executor read an attribute the same way.

// ── DOES THIS TARGET END WITH A TERMINAL ROTATION? ───────────────────────────────────────────────
// ONE rule, consulted by the two places that care: the executor (does it rotate at the goal?) and the
// target repair (does the standpoint need room to rotate?). They were computing it separately, and the
// repair's copy had already drifted — it was moving ROOM WAYPOINTS to spots they could turn around in,
// for a rotation those targets never perform.
//
// It comes from the contract's POLICY, because the policy is where the semantics already live:
//   Reach  — "navigate to the pose, then consume". Purely positional; nothing in a Reach contract
//            mentions an orientation and nothing downstream consumes one. A room_concept waypoint is
//            a Reach, and a terminal rotation there is wasted motion at every waypoint of every lap.
//   Servo  — the rotation has a real job: it aims the camera so the servo STARTS with valid feedback.
//            Without it valid_attr is false on arrival and LockOn dithers to reacquire.
//   Orient — rotating to the yaw IS the affordance.
//
// GoalFacingYawEnabled stays, as a kill switch over all of it, which is what a global flag is for. It
// was previously the WHOLE answer, ANDed only with "is this an affordance" — so enabling it for the
// refrigerator enabled it for every room waypoint too.

// Is a previously-chosen standpoint STILL good, against the map as it stands now? Deliberately the
// cheap half of the question — footprint feasibility, plus room to turn when this target ends in a
// rotation. Both are local lookups. Reachability is not re-tested here: that is a flood fill, and the
// planner already reports it immediately (a failed plan sets unroutable_target_name_ on the very next
// cycle), so paying for it every cycle would buy nothing.
bool ControllerSession::fix_still_good(const Eigen::Vector2f &pos, const ControllerTargetInfo &target) const
{
    return wants_final_facing(target) ? grid_planner_.can_turn_here(pos)
                                      : grid_planner_.pose_free(pos, target.yaw_rad);
}

bool ControllerSession::wants_final_facing(const ControllerTargetInfo &target) const
{
    if (not target.from_affordance) return false;                       // a clicked point has no facing
    if (params_ != nullptr and not params_->goal_facing_yaw_enabled) return false;
    if (not target_contract_known_) return false;   // unknown policy ⇒ do not invent a rotation
    using P = rc::affordance::Policy;
    return target_contract_.policy == P::Servo or target_contract_.policy == P::Orient;
}

// Resolved once per target, when it is SELECTED. Same read the executor performs at arrival, done
// early because the policy now decides behaviour before the robot gets there.
void ControllerSession::resolve_target_contract(const ControllerTargetInfo &target)
{
    if (target.node_id == contract_for_node_id_) return;
    contract_for_node_id_ = target.node_id;
    target_contract_ = {};
    target_contract_known_ = false;
    if (not graph_ or not target.from_affordance or target.node_id == 0) return;
    const auto n = graph_->get_node(target.node_id);
    if (not n.has_value()) return;
    std::string parent_type;
    if (target.parent_node_id != 0)
        if (const auto pn = graph_->get_node(target.parent_node_id); pn.has_value())
            parent_type = parent_contract_key(pn.value());
    target_contract_ = rc::affordance::read_contract(n.value(), parent_type);
    target_contract_known_ = true;
}

std::optional<float> ControllerSession::feedback_scalar(std::uint64_t node_id, const std::string &attr) const
{
    if (not graph_ or node_id == 0) return std::nullopt;
    const auto node = graph_->get_node(node_id);
    if (not node.has_value()) return std::nullopt;
    const auto &attrs = node->attrs();
    const auto it = attrs.find(attr);
    if (it == attrs.end()) return std::nullopt;
    return rc::affordance::detail::attr_scalar(it->second);
}

namespace
{
const char *compare_symbol(rc::affordance::CompareOp op)
{
    using CO = rc::affordance::CompareOp;
    switch (op) { case CO::LE: return "<="; case CO::EQ: return "=="; case CO::NE: return "!="; default: return ">="; }
}
}   // namespace

// ── THE AFFORDANCE PROGRAM, AS DATA ──────────────────────────────────────────────────────────────
// Built fresh each cycle from state that already exists — nothing here drives anything, and nothing
// here is allowed to. In particular the contract is read into a DISPLAY-ONLY copy: active_contract_ is
// still resolved exactly where it always was (on arrival, in execute_plan), because moving that would
// change when a policy takes effect. The panel just wants to show the program BEFORE it starts, so it
// reads the same node itself.
void ControllerSession::update_affordance_view(const ControllerRobotPose &robot_pose,
                                               const rc::TrajectoryController::ControlOutput &o,
                                               bool output_enabled, float align_tol_rad,
                                               std::uint64_t now_ms)
{
    using Step = rc::AffordanceStepView;
    using S = rc::AffordanceStepView::State;
    rc::AffordanceExecution v;
    v.recent = affordance_recent_;

    const bool on_affordance = last_target_info_.has_value() and last_target_info_->from_affordance;
    if (not on_affordance)
    {
        // Retire the live run into `recent` once, then keep showing the last steps: a window opened
        // after the fact must still say what happened.
        if (affordance_view_.active and not affordance_view_.affordance.empty())
        {
            affordance_recent_.insert(affordance_recent_.begin(),
                std::format("{} {:.1f}s {}", affordance_view_.affordance, affordance_view_.elapsed_s,
                            lockon_.locked() ? "locked" : lockon_.phase() == rc::LockOn::Phase::GiveUp
                                                            ? "gave up" : "reached"));
            if (affordance_recent_.size() > 4) affordance_recent_.resize(4);
        }
        affordance_view_.active = false;
        affordance_view_.recent = affordance_recent_;
        return;
    }

    if (affordance_view_.affordance != last_target_info_->node_name)
    {
        affordance_started_ms_ = now_ms;
        affordance_step_since_ms_ = now_ms;
        affordance_prev_step_.clear();
        affordance_nav_total_m_ = std::max(0.05f, o.dist_to_goal);   // the distance this run began with
        // The contract is already resolved — build_planning_step does it when the target is selected,
        // because the policy decides real behaviour now and not just what this window draws.
    }

    v.active = true;
    v.affordance = last_target_info_->node_name;
    v.contract_known = target_contract_known_;
    v.policy = std::string(rc::affordance::to_string(target_contract_.policy));
    v.elapsed_s = static_cast<float>(now_ms - affordance_started_ms_) * 1e-3f;
    if (graph_ and last_target_info_->parent_node_id != 0)
        if (const auto pn = graph_->get_node(last_target_info_->parent_node_id); pn.has_value())
            v.object = pn->name();

    // ★THE FEEDBACK NODE, RESOLVED THE WAY THE CONTRACT SAYS — not taken from feedback_node_id_.
    // That member is only set at ARRIVAL (execute_plan), so gating the clause rows on it meant they
    // could not populate during the approach at all, and worse: it is not cleared between affordances,
    // so a leftover value would have shown this affordance's clauses evaluated against the PREVIOUS
    // object's node. read_contract already defaults feedback_node_id to the affordance's parent, which
    // is the same rule the executor applies — so applying it here needs no new convention.
    const std::uint64_t fb = target_contract_.feedback_node_id != 0 ? target_contract_.feedback_node_id
                                                                  : last_target_info_->parent_node_id;

    const bool servo = target_contract_.policy == rc::affordance::Policy::Servo;
    const bool orient = target_contract_.policy == rc::affordance::Policy::Orient;
    // The timeout clock only means anything once the servo loop owns it.
    if (servo and lockon_.active()) v.timeout_s = target_contract_.timeout_ms * 1e-3f;

    // The one reason that is worth saying on EVERY row it applies to: a command computed and discarded.
    // This is the failure that looks like every other failure from the outside.
    const bool cmd_nonzero = std::abs(o.adv) > 1e-3f or std::abs(o.rot) > 1e-3f;
    const std::string disarmed = (not output_enabled and cmd_nonzero)
        ? std::format("BASE OUTPUT DISARMED — commanding {:.2f} m/s, {:.2f} rad/s, all discarded",
                      o.adv, -o.rot)
        : std::string{};

    const bool navigating = not o.aligning and not o.goal_reached and not lockon_.active();
    const bool aligning = o.aligning;

    // 1. CLAIM — by the time anything else runs, this is behind us.
    // The yaw is only worth showing when something ACTS on it. A Reach carries a yaw in the target
    // struct like every other target does, but nothing consumes it — printing "facing 156 deg" for a
    // room waypoint states a requirement that does not exist.
    v.steps.push_back({.label = "claim affordance", .state = S::Done, .progress = -1.f,
                       .detail = wants_final_facing(*last_target_info_)
                           ? std::format("({:.2f},{:.2f}) facing {:.0f} deg",
                                         last_target_info_->room_pos.x(), last_target_info_->room_pos.y(),
                                         last_target_info_->yaw_rad * 180.f / static_cast<float>(M_PI))
                           : std::format("({:.2f},{:.2f}) — orientation not used",
                                         last_target_info_->room_pos.x(), last_target_info_->room_pos.y())});

    // 2. NAVIGATE — skipped by Orient, which is a rotation in place and has no (x,y) target at all.
    {
        Step s{.label = "navigate to standpoint"};
        if (orient) { s.state = S::Skipped; s.detail = "orient policy — no standpoint"; }
        else
        {
            const float remaining = o.dist_to_goal;
            s.state = navigating ? S::Active : S::Done;
            s.progress = std::clamp(1.f - remaining / affordance_nav_total_m_, 0.f, 1.f);
            s.detail = std::format("{:.2f} m to go of {:.2f}", remaining, affordance_nav_total_m_);
            if (navigating and not disarmed.empty()) s.blocked_why = disarmed;
        }
        v.steps.push_back(std::move(s));
    }

    // 3. ALIGN — and this is where the rotation the controller performs WITHOUT ANY OBSTACLE CHECK
    // becomes visible. rotation_sweep asks the question the arrival path never asks: is there room to
    // turn HERE, at the pose the robot actually stopped in, through every heading on the way.
    {
        Step s{.label = "align to facing yaw"};
        const float tol = align_tol_rad;
        const float err = o.goal_yaw_err_rad.value_or(0.f);
        // ★SKIPPED, NOT PENDING, when the executor is configured never to run it. want_facing in
        // ensure_current_plan is `from_affordance and goal_facing_yaw_enabled`, so with
        // GoalFacingYawEnabled=false the controller is handed no facing yaw and NEVER aligns. Showing
        // that as a pending step reads as "stuck here", which is the opposite of the truth.
        // Same rule the executor uses — wants_final_facing() — not a third copy of it. The reason is
        // worth distinguishing: "this policy has no final orientation" is a property of the affordance,
        // while the kill switch being off is a property of the run.
        const bool facing_enabled = wants_final_facing(*last_target_info_);
        if (not facing_enabled)
        {
            s.state = S::Skipped;
            const bool killed = params_ != nullptr and not params_->goal_facing_yaw_enabled;
            s.detail = killed
                ? "GoalFacingYawEnabled=false — disabled for this run"
                : std::format("policy '{}' has no final orientation", v.policy);
            v.steps.push_back(std::move(s));
        }
        else
        {
            s.state = aligning ? S::Active : (o.goal_reached or lockon_.active()) ? S::Done : S::Pending;
            if (o.goal_yaw_err_rad.has_value())
            {
                s.progress = std::clamp(1.f - std::abs(err) / std::max(1e-3f, std::abs(err) + tol), 0.f, 1.f);
                s.detail = std::format("yaw err {:.1f} deg (tol {:.1f})",
                                       err * 180.f / static_cast<float>(M_PI),
                                       tol * 180.f / static_cast<float>(M_PI));
            }
            if (aligning)
            {
                if (not disarmed.empty()) s.blocked_why = disarmed;
                else
                {
                    const float facing_now = std::remainder(robot_pose.theta + static_cast<float>(M_PI_2),
                                                            2.f * static_cast<float>(M_PI));
                    const auto sweep = grid_planner_.rotation_sweep(robot_pose.pos, facing_now,
                                                                    last_target_info_->yaw_rad);
                    if (not sweep.feasible)
                        s.blocked_why = std::format("NO ROOM TO TURN HERE — footprint does not fit at "
                                                    "{:.0f} deg on the way (tightest {:.3f} m). The terminal "
                                                    "rotation runs with no obstacle check.",
                                                    sweep.worst_heading_rad * 180.f / static_cast<float>(M_PI),
                                                    sweep.min_clearance_m);
                }
            }
            v.steps.push_back(std::move(s));
        }
    }

    // 4. SERVO — the contract's micro-search, only under a Servo policy.
    if (servo)
    {
        Step s{.label = "servo lock-on", .kind = Step::Kind::ServoLoop};
        const auto ph = lockon_.phase();
        v.phase = ph == rc::LockOn::Phase::Settle ? "settle" : ph == rc::LockOn::Phase::Step ? "step"
                : ph == rc::LockOn::Phase::Locked ? "locked" : ph == rc::LockOn::Phase::GiveUp ? "gave up"
                                                             : "idle";
        s.state = lockon_.locked() ? S::Done
                : ph == rc::LockOn::Phase::GiveUp ? S::Failed
                : lockon_.active() ? S::Active : S::Pending;
        if (fb != 0 and lockon_.active())
        {
            const auto r = read_servo_reading(fb);
            s.detail = std::format("err_x {:+.3f}  scalar {:.3f} -> {:.3f}{}",
                                   r.err_x, r.scalar, target_contract_.scalar_target,
                                   r.valid ? "" : "   [feedback INVALID — dithering to reacquire]");
            if (not disarmed.empty()) s.blocked_why = disarmed;
        }
        else if (s.state == S::Pending)
            s.detail = "waiting for arrival";
        v.steps.push_back(std::move(s));
    }
    else
        v.phase = o.goal_reached ? "reached" : aligning ? "aligning" : "driving";

    // 5. THE COMPLETION CLAUSES — one row each, straight from the contract, because they are per
    // affordance and a hardcoded list would describe a different program.
    if (target_contract_known_ and fb != 0)
    {
        for (const auto &c : target_contract_.goal)
        {
            Step s{.label = std::format("{} {} {:.3f}", c.attr, compare_symbol(c.op), c.value),
                   .kind = Step::Kind::Clause};
            const auto now = feedback_scalar(fb, c.attr);
            if (now.has_value())
            {
                const bool holds = rc::affordance::clause_ok(*now, c.op, c.value);
                s.state = holds ? S::Done : S::Active;
                s.detail = std::format("now {:.3f}{}", *now, holds ? "  OK" : "");
            }
            else
            {
                s.state = S::Pending;
                s.blocked_why = std::format("'{}' is not published on the feedback node", c.attr);
            }
            v.steps.push_back(std::move(s));
        }
        // The stability requirement is a step of its own: a predicate that keeps flickering true is not
        // the same as one that HOLDS, and without this row the difference is invisible.
        if (not target_contract_.goal.empty())
        {
            Step s{.label = "hold stable", .kind = Step::Kind::Stable};
            s.state = lockon_.locked() ? S::Done : lockon_.active() ? S::Active : S::Pending;
            s.progress = target_contract_.stable_n > 0
                       ? std::clamp(static_cast<float>(lockon_.stable()) /
                                    static_cast<float>(target_contract_.stable_n), 0.f, 1.f) : -1.f;
            s.detail = std::format("{} of {} consecutive", lockon_.stable(), target_contract_.stable_n);
            v.steps.push_back(std::move(s));
        }
    }

    // Per-step elapsed: time since the ACTIVE row last changed, so a stall reads as a growing number on
    // the row that is stalling.
    std::string active_label;
    for (const auto &s : v.steps) if (s.state == S::Active) { active_label = s.label; break; }
    if (active_label != affordance_prev_step_) { affordance_prev_step_ = active_label; affordance_step_since_ms_ = now_ms; }
    const float in_step = static_cast<float>(now_ms - affordance_step_since_ms_) * 1e-3f;
    for (auto &s : v.steps)
        s.elapsed_s = s.state == S::Active ? in_step : (s.state == S::Done ? 0.f : 0.f);

    affordance_view_ = std::move(v);
}

void ControllerSession::update_display(const std::optional<ControllerRobotPose> &robot_pose,
                                       ControllerDisplay &display,
                                       const ControllerObstacleVisuals &obstacle_polys,
                                       const ControllerPolygons &obstacle_rfe_points,
                                       int max_lidar_draw_points) const
{
    // The displayed cloud and icon are drawn at the scan's own pose. Display-side dead-reckoning to
    // "now" was removed 2026-08-04: it only ever applied when overlay_draw_one_frame_old was OFF, and
    // that A/B settled ON (see etc/config.toml) — the cloud is registered exactly at its scan stamp,
    // so pushing it forward would undo that.
    const auto &icon_pose = robot_pose;
    const std::optional<Eigen::Affine2f> lidar_correction;

    display.update(icon_pose,
                   room_polygon_,
                   current_plan_,
                   obstacle_polys,
                   obstacle_rfe_points,
                   current_target_room_,
                   last_mppi_trajectories_,
                   last_mppi_average_trajectory_,
                   last_best_mppi_trajectory_idx_,
                   last_display_wp_index_,
                   max_lidar_draw_points,
                   lidar_correction);

    // Light the toolbar stuck indicator while an escape maneuver owns the base (robot is
    // reversing/turning out of a wedge). Pushed every cycle; the widget dedups same-state calls.
    display.set_stuck_active(escape_active_);
}

ControllerSession::DrivenCurve ControllerSession::driven_curve() const
{
    // A MISSION route lives in route_; every point target — click or affordance — lives in plan_spline_.
    // Mission wins when it is running, which is the same precedence every consumer used to implement
    // for itself.
    if (route_active_ and mission_.running() and route_.valid())
        return {DriveOwner::MissionRoute, &route_.spline(), &route_.path(), route_.control_count()};
    if (plan_spline_valid_ and plan_spline_.valid())
        return {DriveOwner::PointPlan, &plan_spline_, &plan_spline_.samples(),
                plan_spline_.control_points().size()};
    return {};
}

void ControllerSession::step_route_band(const DrivenCurve &curve,
                                        const ControllerRobotPose &robot_pose,
                                        rc::TrajectoryController &path_controller)
{
    // Truncate the diagnostic FIRST, before any early return. Otherwise a run with the band OFF never
    // opens the file, archive_on_stop copies the PREVIOUS run's rows under this run's stamp, and the
    // archive claims the band solved 993 times during a lap where it never ran. Observed exactly that on
    // 20260802-135917, whose band_diag.csv was byte-identical to 20260802-135128's.
    ensure_band_csv(params_ != nullptr and params_->band_enabled);
    if (params_ == nullptr or not params_->band_enabled) return;
    if (not path_controller.is_active()) return;
    // The band no longer decides WHAT it may deform — it is told. Narrow spaces and corners are exactly
    // where a one-shot fit is weakest, so this is where the continuous version earns its keep, on either
    // owner's curve.
    if (not curve.valid()) return;
    const bool on_mission = curve.on_mission();
    const rc::RouteSpline &spline = *curve.spline;
    const std::size_t M = curve.control_count;
    if (M < 8) return;                       // nothing a window can be carved out of
    if (params_->band_period_cycles > 1 and (band_cycle_++ % params_->band_period_cycles) != 0) return;

    // ── The window, in control-point indices ──
    // Control points sit every h along the POLYLINE; progress() is arc length along the CURVE, which is
    // shorter (smoothing cuts corners). The two metrics are not interchangeable — conflating them is what
    // deleted 15 m of a 3-lap tour once (see fit_from_polyline). A PROPORTIONAL map is used instead of
    // either metric: it needs no correspondence, is monotone, and its error is absorbed by the lead
    // margin, which exists to be conservative. It is only choosing which points to unfreeze.
    const float len = std::max(0.01f, spline.length());
    // Where the robot is along the curve. A mission tracks this itself; a plan does not, so it is
    // projected here with a FORWARD-ONLY hint for the same reason the follower does — a path that
    // crosses itself would otherwise teleport progress to the wrong branch.
    if (not on_mission)
        plan_progress_s_ = spline.project(robot_pose.pos, plan_progress_s_, 3.0f);
    const float progress = on_mission ? route_.progress() : plan_progress_s_;
    const float frac = std::clamp(progress / len, 0.f, 1.f);
    const float h = std::max(0.05f, spline.control_spacing());
    const auto i_robot = static_cast<std::size_t>(frac * static_cast<float>(M));
    const auto lead = static_cast<std::size_t>(std::ceil(std::max(0.f, params_->band_lead_m) / h));
    const auto span = static_cast<std::size_t>(std::ceil(std::max(h, params_->band_window_m) / h));
    const std::size_t freeze_before = std::min(i_robot + lead, M);
    const std::size_t freeze_after  = std::min(freeze_before + span, M);
    if (freeze_after <= freeze_before + 1) return;    // window collapsed (route nearly finished)

    // ── The live field, room frame ──
    // The ESDF is robot-frame, so each query is transformed in and each gradient transformed back out.
    // Outside its 8x8 m box clearance_at returns 100 m: a one-sided clearance term sees no deficit there
    // and applies no force, so the band simply cannot be driven by geometry the field does not cover.
    const Eigen::Affine2f pose = robot_pose.as_transform();
    const Eigen::Matrix2f R = pose.linear();
    const Eigen::Matrix2f Rt = R.transpose();
    const Eigen::Vector2f t = pose.translation();
    auto distance = [&path_controller, Rt, t](const Eigen::Vector2f &p) -> float
    {
        const Eigen::Vector2f r = Rt * (p - t);
        return path_controller.clearance_at(r.x(), r.y());
    };
    auto gradient = [&path_controller, Rt, t, R](const Eigen::Vector2f &p) -> Eigen::Vector2f
    {
        const Eigen::Vector2f r = Rt * (p - t);
        return R * path_controller.clearance_gradient_at(r.x(), r.y());
    };

    rc::RouteOptimizerReport rep;
    if (on_mission)
    {
        rep = route_.deform_window(distance, gradient, freeze_before, freeze_after,
                                   params_->band_iterations);
    }
    else
    {
        // A plan has no authored waypoints, so it carries NO anchor likelihood — and it does not need
        // one: optimize_route always pins the endpoints, which is exactly the requirement ("still start
        // where I am and still end at the target"). Everything between is free to answer to clearance
        // and bending alone, which is what makes this useful in a corner.
        rc::RouteOptimizerConfig cfg = make_route_optimizer_config();
        cfg.enabled = true;
        cfg.distance = distance;
        cfg.distance_gradient = gradient;
        cfg.freeze_before = freeze_before;
        cfg.freeze_after = freeze_after;
        cfg.iterations = std::max(1, params_->band_iterations);
        cfg.verbose = false;          // this runs at control rate; per-cycle logging is noise
        rep = plan_spline_.deform(cfg);
    }
    // Logged BEFORE the early return, so a rejected or no-op solve leaves a row too. A band that is on
    // and doing nothing must be visible as such — "enabled" and "working" are different claims, and a
    // run where every solve moved 0.000 m would otherwise look exactly like a working one.
    log_band_diagnostics(overlay_now_ms_, rep, freeze_before, freeze_after, M);
    if (not rep.ran or rep.rejected) return;

    // Hand the follower the deformed geometry WITHOUT resetting it — see update_path_geometry. The
    // prefix is frozen, so arc length behind the robot is unchanged and progress()/waypoint arc lengths
    // still mean what they did.
    const auto &deformed = *curve.samples;
    path_controller.update_path_geometry(deformed);
    current_plan_ = ControllerPathPlan{.room_path = deformed};
}

void ControllerSession::ensure_band_csv(bool band_enabled)
{
    if (band_csv_open_) return;
    band_csv_.open("band_diag.csv", std::ios::out | std::ios::trunc);
    band_csv_open_ = true;
    if (!band_csv_.is_open()) return;
    if (not band_enabled)
    {
        // A file that says so, rather than no file — which archive_on_stop would fill with the last
        // run's contents. An absent measurement and a stale one are not the same thing.
        band_csv_ << "# BAND DISABLED for this run (BandEnabled=false) — no solves were attempted.\n"
                     "# This file is deliberately header-only: it records the ABSENCE of band activity,\n"
                     "# which is what a control arm of the A/B needs to be able to state.\n";
        band_csv_.flush();
        return;
    }
    band_csv_ << "# local elastic band — ONE ROW PER SOLVE ATTEMPT, including the ones that did\n"
                         "# nothing: ran=0 means the solve was refused (degenerate window / no field),\n"
                         "# rejected=1 means it solved and the acceptance test reverted it. A band that is\n"
                         "# enabled and inert must be readable as such.\n"
                         "# win_lo/win_hi = deformable control-point index range; ctrl_n = polygon size.\n"
                         "# clr_before/clr_after are the OPTIMISER's min clearance over the route, in the\n"
                         "#   LIVE robot-frame field — not comparable with the run JSON's body clearance.\n"
                         "t_ms,ran,rejected,uvd_violated,iterations,win_lo,win_hi,ctrl_n,"
                         "max_move_m,cost_before,cost_after,clr_before,clr_after,"
                 "e_kappa,e_clear,e_anchor,e_gauge\n";
}

void ControllerSession::log_band_diagnostics(std::uint64_t t_ms, const rc::RouteOptimizerReport &rep,
                                             std::size_t freeze_before, std::size_t freeze_after,
                                             std::size_t ctrl_count)
{
    if (!band_csv_.is_open()) return;
    band_csv_ << t_ms << ',' << (rep.ran ? 1 : 0) << ',' << (rep.rejected ? 1 : 0) << ','
              << (rep.uvd_violated ? 1 : 0) << ',' << rep.iterations << ','
              << freeze_before << ',' << freeze_after << ',' << ctrl_count << ','
              << rep.max_move_m << ',' << rep.cost_before << ',' << rep.cost_after << ','
              << rep.min_clearance_before << ',' << rep.min_clearance_after << ','
              << rep.e_kappa << ',' << rep.e_clear << ',' << rep.e_anchor << ',' << rep.e_gauge << '\n';
}

void ControllerSession::execute_plan(const ControllerRobotPose &robot_pose,
                                     rc::TrajectoryController &path_controller,
                                     ControllerObstacleTracker &obstacle_tracker,
                                     ControllerMotionCommander &motion_commander,
                                     ControllerDisplay &display,
                                     rc::AffordanceManager &affordance_manager,
                                     const TimeSource &time_source)
{
    if (!params_ || params_->obstacle_creation_enabled)
        obstacle_tracker.refresh_temporary_lidar_obstacle(time_source(), robot_pose, path_controller);
    // Proactive scene-level "model anything the concept agents don't" is now owned by the dedicated
    // `residual_concept` agent (the residual/null concept: LiDAR residual-filter → 3D DBSCAN → box belief
    // → "obstacle" nodes). Its obstacles arrive via the graph and are consumed by read_obstacle_polygons
    // exactly like the reactive ones, so the planner still avoids them. HYBRID phase: the controller keeps
    // only the fast in-loop REACTIVE blockage reflex below (create_temporary_lidar_obstacle) for
    // collision safety with zero DDS latency; the proactive full-scene scan is DISABLED here. Re-enable
    // by uncommenting the call if residual_concept is not running.
    // obstacle_tracker.scan_for_unmodelled_obstacles(time_source(), robot_pose, path_controller);

    // An escape maneuver (physical-stuck recovery) owns the base until it finishes backing
    // out — bypass the planner/follower entirely, just like the lock-on micro-search below.
    // It survives the current_plan_ reset done in begin_escape(), so it's gated FIRST.
    if (escape_active_)
    {
        step_escape(robot_pose, path_controller, motion_commander, time_source());
        return;
    }

    if (!current_plan_.has_value())
    {
        display.clear_robot_trajectory();
        clear_tracking_state();
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    // A lock-on micro-search in progress owns the base until it locks or gives up — bypass the path
    // follower so it isn't disturbed by goal_reached re-evaluation while we quasi-statically servo.
    if (lockon_.active())
    {
        if (step_lockon(motion_commander, time_source))
            finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
        return;
    }

    // Orient affordance (Policy::Orient): rotate IN PLACE toward the target bearing — no navigation. Owns
    // the base like the lock-on above; only for a from-affordance target whose contract is Orient (a
    // bearing-only hypothesis, Part D). Read the contract each cycle so it activates as soon as selected.
    if (params_ && params_->lockon_enabled && graph_ && last_target_info_.has_value()
        && last_target_info_->from_affordance && active_target_id_ != 0)
    {
        if (const auto aff = graph_->get_node(active_target_id_); aff.has_value())
        {
            std::string parent_type;
            if (last_target_info_->parent_node_id != 0)
                if (const auto pn = graph_->get_node(last_target_info_->parent_node_id); pn.has_value())
                    parent_type = parent_contract_key(pn.value());
            const auto contract = rc::affordance::read_contract(aff.value(), parent_type);
            if (contract.policy == rc::affordance::Policy::Orient)
            {
                active_contract_  = contract;
                feedback_node_id_ = contract.feedback_node_id != 0 ? contract.feedback_node_id
                                                                   : last_target_info_->parent_node_id;
                path_controller.stop();
                if (step_orient(robot_pose, motion_commander, time_source, last_target_info_->yaw_rad))
                    finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
                return;
            }
        }
    }

    // The MPPI gets the room's TRUE boundary. It used to get a copy shrunk inward by clearance_m (0.5 m),
    // which is a C-space margin: it silently deleted a half-metre band of real floor along every wall, on top
    // of the footprint test that already keeps the body inside. The body's extent is applied where it belongs
    // — in the obstacle terms, per pose, per bearing.
    if (room_polygon_.size() >= 3)
        path_controller.set_room_boundary(room_polygon_);

    // LOCAL ELASTIC BAND: let the route absorb what the live field says, before the follower measures
    // itself against it. Runs on the ESDF built by the PREVIOUS compute — one cycle old, which is the
    // same age as everything else here and self-guarding on cycle 1, where the field is empty and every
    // query returns esdf_unknown_distance ⇒ no deficit ⇒ no force ⇒ the polygon does not move.
    step_route_band(driven_curve(), robot_pose, path_controller);

    // ── SESSION ODOMETER ────────────────────────────────────────────────────────────────────────
    // Metres actually driven since the agent started — every mission, target and affordance, not per
    // run. MissionRunner integrates the same quantity but only while a mission is RUNNING, so nothing
    // counted a click target or the drive back to a start point.
    // The step is rejected when it implies a speed the base cannot produce: a localization jump is not
    // travel, and one 2 m re-anchor would silently add 2 m to the odometer. The bound is the machine's
    // own envelope, not a tuned number.
    {
        if (session_start_ms_ == 0) session_start_ms_ = overlay_now_ms_;
        const Eigen::Vector2f &pos = robot_pose.pos;
        if (odo_last_pos_.has_value() and overlay_now_ms_ > odo_last_ms_)
        {
            const float step = (pos - *odo_last_pos_).norm();
            const float dt = static_cast<float>(overlay_now_ms_ - odo_last_ms_) * 1e-3f;
            if (step <= (params_ ? params_->max_adv_speed_mps : 0.7f) * dt * 2.f)
                session_distance_m_ += step;
        }
        odo_last_pos_ = pos;
        odo_last_ms_ = overlay_now_ms_;
    }

    // Curvature-limited speed. Only a continuous ROUTE carries a speed profile; a plan keeps the full
    // envelope, and the ceiling is cleared rather than left stale from a previous mission.
    if (route_active_ and mission_.running())
        path_controller.set_speed_limit(route_speed_limit(params_ ? params_->max_adv_speed_mps : 0.7f,
                                                          path_controller.params.cbf_max_decel));
    else
        path_controller.set_speed_limit(std::nullopt);

    const auto control_output = path_controller.compute(robot_pose.as_transform());
    // Surface what the ARRIVAL test is waiting on, every cycle, before any of the branches below can return
    // early — otherwise the readout would freeze exactly in the states worth watching (aligning, blocked).
    // The affordance program, rebuilt from state that already exists. Runs EVERY cycle an affordance is
    // live, not only while the window is open: these runs last seconds, and a view you must open in time
    // to catch a failure is one that never catches it.
    update_affordance_view(robot_pose, control_output, motion_commander.output_enabled(),
                           path_controller.params.align_yaw_tol_rad, overlay_now_ms_);
    display.set_affordance_execution(affordance_view_);
    display.set_session_totals(session_distance_m_, session_elapsed_s());
    display.set_goal_distance(control_output.dist_to_goal, control_output.goal_yaw_err_rad,
                              control_output.aligning);
    last_mppi_trajectories_ = control_output.trajectories_room;
    last_mppi_average_trajectory_ = control_output.average_trajectory_room;
    last_best_mppi_trajectory_idx_ = control_output.best_trajectory_idx;
    last_display_wp_index_ = std::max(0, control_output.current_wp_index);

    // Mission instrumentation. Sampled from the SAME control output the robot is about to execute, so the
    // metrics describe what happened rather than what was planned. The clearance recorded is the gap between
    // the BODY and the nearest obstacle — the ESDF measures from the rotation centre, and reporting that
    // would flatter every run by one body radius.
    if (route_active_ and not mission_.running()) route_active_ = false;
    // One row per cycle while the robot is being DRIVEN — not only while a MISSION runs.
    // ★It used to be gated on mission_.running(), which silently switched off cross_track, the safety
    // gate, the bumper and path_kappa for every affordance and click target: the file simply stopped
    // being written and kept the previous mission's contents, timestamped an hour earlier. Asked to
    // diagnose a bad affordance trajectory, the log on disk described a different run entirely — the
    // same absent-vs-stale confusion that band_diag.csv had, in the other direction.
    // is_active() is the honest condition: there is a path being followed. It still keeps an idle agent
    // from writing forever, which is all the mission gate was for.
    if (path_controller.is_active())
    {
        // Signed curvature at the robot's own projection on the route. -999 marks "no continuous
        // route" rather than 0, which is a real curvature (a straight) — an absent value and a
        // meaningful one must not share an encoding.
        // ★Same quantity the mission sampler already derives below (ref_kappa); computed here rather
        // than hoisted because that block sits inside a different guard. If these ever disagree, one of
        // them is wrong — they read the same spline at the same arc length.
        const float kappa_here = (route_active_ and route_.valid())
                               ? route_.spline().curvature_at(route_.progress()) : -999.f;
        // Localisation sigma, so "is the pose noisy right now" is a MEASUREMENT and not an inference.
        // It only lived in profile.csv, which is gated on a mission running, so during affordance
        // driving — the mode being developed — it was invisible exactly when it was being blamed.
        // ★One cycle old: apply_uncertainty_speed_limit runs later in this same function, so this is
        // the previous cycle's read. The localiser publishes at ~5 Hz, so a 100 ms lag is well inside
        // one update and cannot change any conclusion drawn from it.
        const auto ud = motion_commander.last_uncertainty_diag();
        // track_s: the TRACKER's own arc length, route_length - dist_to_goal. DISTINCT from
        // profile.csv's route_s_m (RouteFollower::progress()) — different projection, different window.
        // Only this one drives the control law, and it had never been recorded, so every "projection
        // jump" measured before now described the session's projection instead of the tracker's.
        // ★Read from the tracker, not reconstructed. It used to be route_length - dist_to_goal, which is
        // s only because PlainTracker happens to put s_remaining in dist_to_goal; in PD/MPPI that field
        // is a EUCLIDEAN norm, so the same CSV column meant two different things by mode.
        const float track_s = path_controller.tracker_arc_length().value_or(-1.f);
        log_approach_diagnostics(overlay_now_ms_, control_output, robot_pose);
        log_mppi_diagnostics(overlay_now_ms_, control_output, control_output.adv, base_speed_lin_,
                             kappa_here, track_s, room_vel_.omega, ud.xy_std_m, ud.theta_std_rad, robot_pose,
                             motion_commander.last_output_rate_stats(),
                             world_model_pose_stamp_age_ms_);
        // CAPTURE THE HARDEST CYCLE OF THE RUN for offline replay (tools/mppi_bench). "Hardest" is where
        // the controller had the least room to choose: every rollout infeasible, or the tightest the
        // horizon ever got. A snapshot from open floor proves nothing — measured, a cost term that is
        // load-bearing near contact is completely inert two metres away, so a comfortable cycle replays
        // identically under every setting and answers no question at all.
        const bool all_infeasible = control_output.ess_K > 0
                                and control_output.n_collisions >= control_output.ess_K;
        if (all_infeasible and tightest_cycle_clearance_ > -1.f)
        {
            tightest_cycle_clearance_ = -1.f;      // nothing beats this; stop re-requesting
            path_controller.request_snapshot("mppi_cycle.txt");
        }

        // ── CAPTURE A REVERSAL ITSELF ─────────────────────────────────────────────────────────────
        // The reversal count is the loudest defect in this stack and three hypotheses for it have now
        // been falsified by measuring PROXIES: route geometry (the optimiser removed 3 of 4 tight corners
        // and the count did not move), sampling dither (at an open cycle the command is 65x its own
        // seed-to-seed noise), and mode averaging (which predicts flips clustered at obstacles, while
        // measurement shows them spread over 83% of the lap). So capture the EVENT, not a proxy for it:
        // the cycle on which the commanded rotation actually changes sign. Replaying that cycle with
        // several seeds says immediately whether the flip was noise, a mode swap, or a real decision.
        // ★Same deadband as TrajectoryStats::rot_reversals (0.05 rad/s) — a different one would count a
        // different thing and the snapshot would not correspond to the metric it is meant to explain.
        // ★OFF BY ONE CYCLE, deliberately not fixed: a request is served at the end of the NEXT compute,
        // so the snapshot is the cycle AFTER the flip (100 ms later), not the flip itself. Capturing the
        // exact cycle would mean buffering every cycle's cloud on the chance it turns out interesting.
        // For the question being asked — is this command noise-dominated — the successor cycle answers it
        // just as well, because the rollouts are resampled from scratch either way. If the question ever
        // becomes "what CHANGED between the two cycles", this is no longer good enough and the buffering
        // has to be built.
        constexpr float kRotDeadband = 0.05f;
        const int rot_sign = control_output.rot > kRotDeadband ? 1
                           : (control_output.rot < -kRotDeadband ? -1 : 0);
        if (rot_sign != 0)
        {
            if (prev_cmd_rot_sign_ != 0 and rot_sign != prev_cmd_rot_sign_ and not reversal_captured_)
            {
                reversal_captured_ = true;         // the first one; later ones would overwrite it
                path_controller.request_snapshot("mppi_reversal.txt");
            }
            prev_cmd_rot_sign_ = rot_sign;
        }
    }

    if (route_active_ and mission_.running())
    {
        // Progress is a scalar that only increases. Legs are read OFF it rather than driven by it.
        route_.advance(robot_pose.pos);
        mission_.note_progress(route_.progress(), route_.length(), route_.laps_done());
        if (route_.finished(robot_pose.pos))
        {
            mission_.stop("completed", time_source());
            route_active_ = false;
            path_controller.stop();
            motion_commander.stop_robot();
            return;
        }
    }

    if (mission_.running())
    {
        const float esdf_here = path_controller.clearance_at(0.f, 0.f);
        // Subtract the body's reach TOWARD THE OBSTACLE, not its worst-case disc. Using the
        // circumscribed radius (0.325 m) where the true reach at that bearing may be 0.230 m understates
        // the gap by up to 9.5 cm — which is the same disc-for-footprint substitution this controller was
        // rewritten to remove, reintroduced in the measurement instead of the model. It made every
        // recorded run report 1-5 cm of clearance and fail the safety constraint.
        const float body_clearance = esdf_here - path_controller.body_extent_here();
        // SNAPSHOT THE CYCLE WHERE THE ROBOT WAS CLOSEST TO SOMETHING — this quantity, the gap between the
        // BODY and the nearest obstacle right now, and not min_esdf. min_esdf is the minimum over the best
        // rollout's WHOLE horizon, which is dominated by the 5 s tail every plan has; triggering on it
        // selected the cycle whose prediction dipped lowest rather than the cycle where the robot was
        // actually in trouble, and the snapshot it produced had 0.40 m of room. A trigger has to mean what
        // its name says, for the same reason a metric does.
        if (mission_.running() and body_clearance < tightest_cycle_clearance_)
        {
            tightest_cycle_clearance_ = body_clearance;
            path_controller.request_snapshot("mppi_cycle.txt");
        }
        // Deviation from the reference curve — the continuous tracking signal. NaN when there is no
        // reference (a click target has no route), and the stats skip it rather than inventing a zero.
        float cross_track = std::numeric_limits<float>::quiet_NaN();
        float heading_err = 0.f, ref_kappa = 0.f;
        // ★SCORE THE RUN ON THE PROJECTION THAT STEERED IT. PlainTracker publishes the Frenet pair it
        // actually used (out.cross_track_m, out.carrot_bearing_rad at its own s_hint_); recomputing it
        // here at route_.progress() scored the run on the OTHER projection — the one measured jumping
        // 2.98 m against the tracker's 0.73 m. cross_track_rms is the objective tools/adapt_L.py
        // minimises, so it must be the tracker's error, not a bystander's.
        if (route_tracker_active_ and route_active_ and std::isfinite(control_output.cross_track_m))
        {
            cross_track = control_output.cross_track_m;
            heading_err = control_output.carrot_bearing_rad;
            ref_kappa = route_.valid() ? route_.spline().curvature_at(route_.progress()) : 0.f;
        }
        else if (route_active_)
        {
            const float s_now = route_.progress();
            const Eigen::Vector2f ref = route_.spline().position_at(s_now);
            const float ref_h = route_.spline().heading_at(s_now);
            const Eigen::Vector2f d = robot_pose.pos - ref;
            cross_track = -std::sin(ref_h) * d.x() + std::cos(ref_h) * d.y();   // signed, left positive
            heading_err = std::atan2(std::sin(robot_pose.theta - ref_h), std::cos(robot_pose.theta - ref_h));
            ref_kappa = route_.spline().curvature_at(s_now);
        }
        mission_.sample(robot_pose.pos, control_output.rot, std::abs(control_output.adv),
                        body_clearance, control_output.safety_guard_triggered,
                        cross_track, heading_err, ref_kappa, time_source());
    }

    // ── Near-obstacle black box. Fires BEFORE any branch below, so it captures the cycle whether the
    //    controller reacts (blocked/stalled) or keeps driving. It gates on the RAW lidar cloud (the
    //    thing that's "perfectly visible"), NOT on the tracked obstacles/ESDF — those are exactly what
    //    fails, so gating on them would miss the episode. The smoking gun for a missed reaction is a
    //    small nearest_lidar_m (something IS there) paired with a large nearest_obst_m (never tracked)
    //    and/or a large min_esdf (ESDF never saw it). cmd_* are the pre-limit MPPI velocities (rot sign
    //    is flipped before sending). Off unless enabled. ──
    if (params_ && params_->proximity_log_enabled)
    {
        const Eigen::Vector2f rp = robot_pose.pos;
        const float gate = params_->proximity_log_distance_m;

        // (a) Nearest RAW lidar return (room frame, same band-filtered buffer the viewer draws). Ignore
        //     hits closer than kSelfR to skip the robot's own body returns — these extend to ~0.35 m
        //     (observed: the column pinned at 0.15/0.33), so 0.40 clears them and reveals the real
        //     approaching object in the 0.40–0.80 m window (contact is inside the footprint anyway).
        constexpr float kSelfR = 0.40f;
        // NEAR-SHELL CHARACTERISATION (which self-return escapes the source footprint disc?). The 0.55 m
        // population that dominates this log is a FIXED RADIUS travelling with the robot, i.e. the cut edge
        // of lidar3d_dds's [Footprint] radius=0.55, not an obstacle. These columns say WHICH body part it is
        // and therefore WHICH knob fixes it, without guessing a new radius:
        //   shell_sectors ≈ 12 (returns all around)  ⇒ 360° body/base ring   ⇒ widen [Footprint] radius
        //   shell_sectors small + bearing clustered  ⇒ arm/protrusion        ⇒ targeted envelope, not a disc
        //   near_lidar_z  in [0.11,0.35]/[0.35,0.60] ⇒ raise the [Skirt] z_max (bpearl / helios band)
        //   near_lidar_z  in [0.60,1.35]             ⇒ mid-band disc between [Skirt] and [Footprint]
        //   near_lidar_z  > 1.35                     ⇒ ABOVE [Footprint] z_max: the disc never applied
        // min_lidar_all_m drops the kSelfR cut entirely: it says whether anything at all survives inside
        // 0.40 m (if it pins at 0.55 too, the source disc — not the controller — is what sets the floor).
        constexpr float kShellOuterM = 0.70f;   // outer edge of the near shell we characterise
        constexpr int   kShellSectors = 12;     // 30° azimuth bins → sector coverage = "is it a full ring?"
        float nearest_lidar = std::numeric_limits<float>::infinity();
        float nearest_lidar_bearing_deg = 0.f;   // robot frame, 0 = straight ahead (+y), + = to the right (+x)
        float nearest_lidar_z = 0.f;             // room-frame z of that return ≈ height above the floor
        float min_lidar_all = std::numeric_limits<float>::infinity();
        int shell_points = 0;
        std::uint16_t shell_sector_mask = 0;
        // ── NEAR-BODY CENSUS ──────────────────────────────────────────────────────────────────────────
        // Everything above measures distance from the robot's ORIGIN in PLAN VIEW, which cannot answer the
        // question that matters: is a surviving return actually ON the robot? A downward sensor puts floor
        // returns at plan-view radius ~0 (measured: min_lidar_all median 0.0012 m on every cycle) and those
        // are perfectly legitimate. What is NOT legitimate is a return inside the body's own silhouette AT
        // BODY HEIGHT — that is a self-hit the mesh filter failed to remove, and until now nothing recorded
        // it, so three separate diagnoses tonight were argued from a 2-D proxy instead of measured.
        // Signed clearance to the body SURFACE, using the same support function the collision test uses:
        // negative ⇒ the return is inside the footprint column.
        float body_clear_min = std::numeric_limits<float>::infinity();
        float body_clear_z = 0.f, body_clear_bearing = 0.f;
        int n_in_footprint = 0;      // inside the silhouette, any height
        int n_in_body = 0;           // ...and at body height ⇒ a self-return that got through
        int n_under_floor = 0;       // ...and below the body ⇒ floor seen under the robot (expected)
        constexpr float kBodyZLo = 0.05f;   // above this is the body, below it is the floor beneath us
        constexpr float kBodyZHi = 1.45f;   // the mesh tops out at 1.42 m
        int n_in_hull = 0;           // ...and inside the body's radius AT ITS OWN HEIGHT — the honest count
        // ── THE ROBOT IS NOT A COLUMN ────────────────────────────────────────────────────────────────
        // RobotFootprint is the 2-D PROJECTION, so it reports 0.31 m at every height. The real silhouette
        // varies by a factor of 2.4: 0.311 at the base, 0.150 at the waist, 0.255 at the shoulder. A return
        // beside the waist at 0.25 m is INSIDE the projection and 10 cm CLEAR of the body — and counting it
        // as a self-hit is what sent me chasing a filter bug that was not there.
        // Max radius per 0.1 m band, measured from shadow.obj recentred by +0.0534 m (the same mesh and the
        // same placement the driver's self-filter uses). ★If that mesh changes this table must be regenerated
        // — which is precisely why the real fix is a height-banded footprint derived from the mesh at load
        // time rather than a copy living here. This is a DIAGNOSTIC, and it is labelled as one.
        static constexpr float kHullR[15] = {0.311f, 0.309f, 0.304f, 0.269f, 0.205f, 0.150f, 0.129f, 0.178f,
                                             0.255f, 0.174f, 0.170f, 0.168f, 0.174f, 0.168f, 0.134f};
        const auto hull_radius_at = [](float z)
        {
            const int i = std::clamp(static_cast<int>(std::floor(z / 0.1f)), 0, 14);
            return kHullR[i];
        };
        if (auto *lb = obstacle_tracker.lidar_buffer())
        {
            const auto [cloud_opt] = lb->read_last();
            if (cloud_opt.has_value())
            {
                const auto &[lxs, lys, lzs] = cloud_opt.value();
                const Eigen::Affine2f robot_from_room = robot_pose.as_transform().inverse();
                const std::size_t nn = std::min({lxs.size(), lys.size(), lzs.size()});
                for (std::size_t i = 0; i < nn; ++i)
                {
                    const float d = std::hypot(lxs[i] - rp.x(), lys[i] - rp.y());
                    min_lidar_all = std::min(min_lidar_all, d);

                    // Census FIRST: it must see the returns the kSelfR gate below discards, since those are
                    // precisely the ones in question.
                    {
                        const Eigen::Vector2f q = robot_from_room * Eigen::Vector2f(lxs[i], lys[i]);
                        if (d < 1.0f)   // cheap reject; the body reaches 0.325 m
                        {
                            const auto &fp = path_controller.footprint();
                            const Eigen::Vector2f dir = d > 1e-6f ? Eigen::Vector2f(q.x() / d, q.y() / d)
                                                                 : Eigen::Vector2f(0.f, 1.f);
                            // theta = 0: q is already in the robot frame, so the footprint needs no rotation.
                            const float reach = fp.support_radius(0.f, dir);
                            if (const float clear = d - reach; clear < body_clear_min)
                            {
                                body_clear_min = clear;
                                body_clear_z = lzs[i];
                                body_clear_bearing = std::atan2(q.x(), q.y()) * 180.f / static_cast<float>(M_PI);
                            }
                            if (fp.contains(q))
                            {
                                ++n_in_footprint;
                                if (lzs[i] >= kBodyZLo and lzs[i] <= kBodyZHi)
                                {
                                    ++n_in_body;
                                    if (d < hull_radius_at(lzs[i])) ++n_in_hull;   // genuinely inside the body
                                }
                                else if (lzs[i] < kBodyZLo)                        ++n_under_floor;
                            }
                        }
                    }

                    if (d < kSelfR) continue;
                    const Eigen::Vector2f p_robot = robot_from_room * Eigen::Vector2f(lxs[i], lys[i]);
                    // Bearing in the ROBOT frame (x right, y forward — the convention the obstacle tracker
                    // uses): a body-fixed return keeps a CONSTANT bearing no matter where the robot drives
                    // or how it turns, which is exactly what separates a self-hit from a world obstacle.
                    const float bearing_deg = std::atan2(p_robot.x(), p_robot.y()) * 180.f / static_cast<float>(M_PI);
                    if (d < nearest_lidar)
                    {
                        nearest_lidar = d;
                        nearest_lidar_bearing_deg = bearing_deg;
                        nearest_lidar_z = lzs[i];
                    }
                    if (d <= kShellOuterM)
                    {
                        ++shell_points;
                        int sector = static_cast<int>(std::floor((bearing_deg + 180.f) * kShellSectors / 360.f));
                        shell_sector_mask |= static_cast<std::uint16_t>(1u << std::clamp(sector, 0, kShellSectors - 1));
                    }
                }
            }
        }
        const float nearest_lidar_out = std::isfinite(nearest_lidar) ? nearest_lidar : -1.f;
        const float min_lidar_all_out = std::isfinite(min_lidar_all) ? min_lidar_all : -1.f;
        const int shell_sectors_out = std::popcount(shell_sector_mask);
        const float body_clear_out = std::isfinite(body_clear_min) ? body_clear_min : 99.f;
        if (!std::isfinite(nearest_lidar)) { nearest_lidar_bearing_deg = 0.f; nearest_lidar_z = 0.f; }

        // (b) Nearest TRACKED obstacle polygon edge (what the controller actually reasons about).
        float nearest = std::numeric_limits<float>::infinity();
        const auto &polys = obstacle_tracker.obstacle_polygons();
        for (const auto &poly : polys)
        {
            const std::size_t n = poly.size();
            for (std::size_t i = 0; i < n; ++i)   // min point-to-edge distance robot→polygon
            {
                const Eigen::Vector2f &a = poly[i];
                const Eigen::Vector2f &b = poly[(i + 1) % n];
                const Eigen::Vector2f ab = b - a;
                const float len2 = ab.squaredNorm();
                const float t = len2 > 1e-9f ? std::clamp((rp - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                nearest = std::min(nearest, (rp - (a + t * ab)).norm());
            }
        }
        const float nearest_out = std::isfinite(nearest) ? nearest : -1.f;

        // Same nearest polygon, but ATTRIBUTED to the layer that owns it, plus the raw (pre-z-band)
        // cloud proximity. Together these answer the two questions the old columns could not: which
        // agent's geometry is squeezing the robot, and whether "no LiDAR support" just meant the
        // support was below the controller's own height band.
        const auto near_obst = obstacle_tracker.nearest_obstacle_info(rp, robot_pose.theta);
        const auto &raw_prox = obstacle_tracker.raw_cloud_proximity();

        // Gate on the RAW cloud first (fires even when nothing is tracked), plus the two model layers.
        const bool near = (nearest_lidar_out >= 0.f && nearest_lidar_out <= gate)
                       || (nearest_out >= 0.f && nearest_out <= gate)
                       || (control_output.min_esdf <= gate);
        const std::uint64_t t_ms = time_source();
        if (near && t_ms - proximity_csv_last_ms_ >= 100)
        {
            proximity_csv_last_ms_ = t_ms;
            if (!proximity_csv_open_)
            {
                proximity_csv_.open(params_->proximity_csv_path, std::ios::out | std::ios::trunc);
                if (proximity_csv_.is_open())
                    proximity_csv_ << "t_ms,rx,ry,rtheta,vx,vy,omega,cmd_adv,cmd_side,cmd_rot,min_esdf,"
                                      "n_esdf_pts,nearest_esdf_pt_m,nearest_lidar_m,nearest_obst_m,n_obst,"
                                      "safety_guard,blockage_ahead,path_blocked,blk_x,blk_y,blk_r,dist_to_goal,"
                                      // Self-stuck diagnostics: split the nearest MODELLED obstacle by source
                                      // (temp-LiDAR vs virtual escape disc), plus the temp obstacle's health.
                                      // Signed distances (robot centre → edge): NEGATIVE ⇒ robot INSIDE it
                                      // (self-collision trap). near_temp with LOW log_odds / HIGH missed / HIGH
                                      // age + a large nearest_lidar_m = an unsupported phantom the robot is
                                      // stuck on. escape_active=1 ⇒ a recovery maneuver owns the base.
                                      "stuck_ms,escape_active,n_temp,n_virtual,near_temp_m,near_virtual_m,"
                                      // WEDGE signal = cmd_lin (commanded translation) vs meas_lin (measured
                                      // base speed): meas_lin < StuckSlipRatio×cmd_lin ⇒ wedged this cycle.
                                      // aligning=1 ⇒ arrived, rotating in place to face target (turn-around).
                                      "near_temp_logodds,near_temp_missed,near_temp_age_ms,cmd_lin,meas_lin,aligning,"
                                      // SELF-RETURN characterisation (see the block that computes these).
                                      // near_lidar_bearing_deg: robot frame, 0 = ahead, + = right. CONSTANT
                                      // across the room ⇒ body-fixed ⇒ self-hit. near_lidar_z: room-frame z
                                      // ≈ height above floor → picks which lidar3d_dds z-band knob applies.
                                      // shell_sectors: of 12 30°-bins in [0.40,0.70] m, how many hold a
                                      // return — 12 ⇒ full 360° ring (base/tray), few ⇒ a local protrusion.
                                      "near_lidar_bearing_deg,near_lidar_z,min_lidar_all_m,n_shell_pts,shell_sectors,"
                                      "body_clear_m,body_clear_z,body_clear_bearing_deg,n_in_footprint,n_in_body,n_in_hull,n_under_floor,"
                                      // ATTRIBUTION: which of the four obstacle layers owns the nearest
                                      // polygon (object = concept agent, grid = residual_concept hull,
                                      // temp/virtual = ours), its label, bearing, and whether the robot
                                      // centre is INSIDE it. nearest_obst_m alone could never say this.
                                      "near_obst_kind,near_obst_label,near_obst_bearing_deg,near_obst_inside,"
                                      // RAW cloud, measured BEFORE the [0.20,1.8] z-band cut: everything
                                      // else on this row reads the filtered buffer, so "no LiDAR support"
                                      // has meant "none in-band". raw_below_band_n counts returns under the
                                      // band within 1 m — evidence residual sees and this side discards.
                                      "raw_nearest_m,raw_nearest_z,raw_nearest_bearing_deg,raw_below_band_n\n";
                proximity_csv_open_ = true;
            }
            if (proximity_csv_.is_open())
            {
                const auto od = obstacle_tracker.obstacle_proximity_diag(rp, t_ms);
                proximity_csv_ << t_ms << ',' << rp.x() << ',' << rp.y() << ',' << robot_pose.theta << ','
                               << room_vel_.vx << ',' << room_vel_.vy << ',' << room_vel_.omega << ','
                               << control_output.adv << ',' << control_output.side << ',' << control_output.rot << ','
                               << control_output.min_esdf << ','
                               << control_output.n_esdf_points << ',' << control_output.nearest_esdf_point_m << ','
                               << nearest_lidar_out << ',' << nearest_out << ',' << polys.size() << ','
                               << (control_output.safety_guard_triggered ? 1 : 0) << ','
                               << (control_output.blockage_detected_ahead ? 1 : 0) << ','
                               << (control_output.path_blocked ? 1 : 0) << ','
                               << control_output.blockage_center_room.x() << ','
                               << control_output.blockage_center_room.y() << ','
                               << control_output.blockage_radius << ',' << control_output.dist_to_goal << ','
                               // No-progress clock (ms): 0 = fresh/moving, climbs toward stuck_confirm_ms.
                               // The escape fires when this crosses the threshold — the last row before the
                               // "[recovery] STUCK -> escape" line is the smoking gun (cmd_* ~0, this ~confirm_ms).
                               << (stuck_since_ms_ != 0 ? t_ms - stuck_since_ms_ : 0) << ','
                               << (escape_active_ ? 1 : 0) << ',' << od.n_temp << ',' << od.n_virtual << ','
                               << od.near_temp_m << ',' << od.near_virtual_m << ','
                               << od.near_temp_log_odds << ',' << od.near_temp_missed << ','
                               << od.near_temp_age_ms << ','
                               << std::hypot(control_output.adv, control_output.side) << ','
                               << base_speed_lin_ << ','
                               << (control_output.aligning ? 1 : 0) << ','
                               << nearest_lidar_bearing_deg << ',' << nearest_lidar_z << ','
                               << min_lidar_all_out << ',' << shell_points << ',' << shell_sectors_out << ','
                               << body_clear_out << ',' << body_clear_z << ',' << body_clear_bearing << ','
                               << n_in_footprint << ',' << n_in_body << ',' << n_in_hull << ',' << n_under_floor << ','
                               << obstacle_kind_tag(near_obst.kind) << ','
                               << (near_obst.label.empty() ? "-" : near_obst.label) << ','
                               << near_obst.bearing_deg << ',' << (near_obst.inside ? 1 : 0) << ','
                               << raw_prox.distance_m << ',' << raw_prox.z_m << ','
                               << raw_prox.bearing_deg << ',' << raw_prox.below_band_within_1m << '\n';
                proximity_csv_.flush();
            }
        }
    }

    if (control_output.path_blocked)
    {
        clear_tracking_state();
        if (!params_ || params_->obstacle_creation_enabled)
            obstacle_tracker.create_temporary_lidar_obstacle(time_source(),
                                                             robot_pose,
                                                             control_output.blockage_center_room,
                                                             control_output.blockage_radius,
                                                             path_controller);
        current_plan_.reset();
        plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
        // In leg mode resetting the plan WAS the replan trigger. A continuous route has no target to
        // replan to, so the request has to be explicit or the route branch just re-installs the same
        // curve and the robot drives at the blocker again.
        route_repair_pending_ = true;
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    if (control_output.goal_reached)
    {
        // Reached an affordance pose. Resolve its contract (type-level defaults + per-node
        // overrides). A Servo policy runs the lock-on micro-search to satisfy the contract's
        // completion predicate before consuming the target; Reach (and non-affordance targets like a
        // mouse click) finalise immediately.
        bool want_servo = false;
        if (params_ && params_->lockon_enabled && graph_
            && last_target_info_.has_value() && last_target_info_->from_affordance && active_target_id_ != 0)
        {
            if (const auto aff_node = graph_->get_node(active_target_id_); aff_node.has_value())
            {
                // Resolve the contract from the PARENT object's type (stable across affordance-node
                // renames on restart), not the affordance node name.
                std::string parent_type;
                if (last_target_info_->parent_node_id != 0)
                    if (const auto pn = graph_->get_node(last_target_info_->parent_node_id); pn.has_value())
                        parent_type = parent_contract_key(pn.value());
                active_contract_ = rc::affordance::read_contract(aff_node.value(), parent_type);
                feedback_node_id_ = active_contract_.feedback_node_id != 0
                                  ? active_contract_.feedback_node_id
                                  : last_target_info_->parent_node_id;   // default: the parent object
                want_servo = active_contract_.policy == rc::affordance::Policy::Servo
                          && feedback_node_id_ != 0;
            }
        }
        if (want_servo)
        {
            qInfo() << "[affordance]" << last_target_info_->node_name.c_str()
                    << "reached -> SERVO lock-on | feedback node" << feedback_node_id_
                    << "| goal clauses" << static_cast<int>(active_contract_.goal.size())
                    << "| timeout(ms)" << active_contract_.timeout_ms;
            lockon_.begin(time_source(), active_contract_.scalar_target,
                          active_contract_.stable_n, active_contract_.timeout_ms);
            path_controller.stop();
            if (step_lockon(motion_commander, time_source))
                finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
            return;
        }
        qInfo() << "[affordance]" << (last_target_info_.has_value() ? last_target_info_->node_name.c_str() : "?")
                << "reached -> REACH (consume immediately)";
        finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
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
    // Publish what the limiter did into the actuation stream. It sits between the MPPI's command and the
    // wheels, and until now a lap could show a 17% gap between the two with no way to say whether this
    // was the cause or whether it was inert for want of a covariance on the RT edge.
    {
        const auto ud = motion_commander.last_uncertainty_diag();
        mission_.note_uncertainty_limit(ud.valid, ud.xy_std_m, ud.theta_std_rad, ud.adv_scale, ud.rot_scale);
    }

    // ARRIVAL ROTATION: position reached, the controller is rotating IN PLACE to the target facing yaw
    // (adv=side=0, goal_reached still false). This is a controller-owned maneuver that makes NO waypoint
    // progress by design — turning around at a target. It must bypass the stuck/escape logic entirely,
    // else it reads as a wedge and drops a recovery disc right at the target. Just issue the rotation.
    if (control_output.aligning)
    {
        reset_stuck_state();   // not stuck — clear any no-advance window carried over from the approach
        motion_commander.send_speed_command(adv_mps, side_mps, rot_rps);
        return;
    }

    // Physical-WEDGE check. We are PURSUING an active, unreached target (goal_reached, no-plan, lock-on,
    // orient, path_blocked and arrival-rotation all returned earlier). A wedge is a PREDICTION ERROR: the
    // robot commands translation but the base doesn't achieve a healthy fraction of it. This is the ONLY
    // thing that is really "stuck" — a robot moving as commanded (detour, slow nav, a still-sliding
    // creep) is fine, and a VISIBLE blockage was already modelled + replanned by the path_blocked branch
    // above; this catches the INVISIBLE wedge the planner can't see. Sustained → escape: reverse+turn out
    // and drop a marker so the next plan routes around. base_speed_lin_ is the EMA measured base speed.
    const float cmd_lin = std::hypot(adv_mps, side_mps);
    if (detect_stuck(/*pursuing=*/true, cmd_lin, robot_pose.pos, time_source()))
    {
        begin_escape(robot_pose, obstacle_tracker, path_controller, time_source());
        step_escape(robot_pose, path_controller, motion_commander, time_source());
        return;
    }

    // MPPI produced no usable motion this cycle but we are not yet confirmed-wedged: hold the base.
    if (std::abs(adv_mps) < 5e-4f && std::abs(side_mps) < 5e-4f && std::abs(rot_rps) < 1e-3f)
    {
        path_controller.stop();
        motion_commander.stop_robot();
        return;
    }

    motion_commander.send_speed_command(adv_mps, side_mps, rot_rps);
}

// ─── Lock-on micro-search ─────────────────────────────────────────────────────

// Fill the normalised servo Reading from the contract-named feedback attributes (object-agnostic).
rc::LockOn::Reading ControllerSession::read_servo_reading(std::uint64_t feedback_node_id) const
{
    rc::LockOn::Reading r;
    if (!graph_ || feedback_node_id == 0)
        return r;
    const auto node = graph_->get_node(feedback_node_id);
    if (!node.has_value())
        return r;
    const auto &attrs = node->attrs();
    const auto &c = active_contract_;

    r.valid = c.valid_attr.empty();   // no gate attr ⇒ always valid
    if (!c.valid_attr.empty())
        if (const auto it = attrs.find(c.valid_attr); it != attrs.end())
            if (const auto v = rc::affordance::detail::attr_scalar(it->second)) r.valid = *v != 0.0f;
    if (!c.scalar_attr.empty())
        if (const auto it = attrs.find(c.scalar_attr); it != attrs.end())
            if (const auto v = rc::affordance::detail::attr_scalar(it->second)) r.scalar = *v;
    if (!c.err_vec_attr.empty())
        if (const auto it = attrs.find(c.err_vec_attr); it != attrs.end() && it->second.selected() == 3)
        {
            const auto &v = it->second.float_vec();
            if (v.size() >= 1) r.err_x = v[0];
            if (v.size() >= 2) r.err_y = v[1];
        }
    return r;
}

// Evaluate the contract's completion predicate against the current feedback-node attributes.
bool ControllerSession::goal_met(std::uint64_t feedback_node_id) const
{
    if (!graph_ || feedback_node_id == 0)
        return false;
    const auto node = graph_->get_node(feedback_node_id);
    if (!node.has_value())
        return false;
    // The look is "done" only when the contract clauses hold AND the base is quiet enough for a clean,
    // motion-free observation (no blur / pose smear). max_observe_*=0 → stillness not required.
    return rc::affordance::evaluate_goal(node.value(), active_contract_.goal) && robot_still();
}

void ControllerSession::update_base_speed(const ControllerRobotPose &pose, std::uint64_t timestamp_ms)
{
    constexpr float kTwoPi = 6.28318530718f;
    constexpr float kPosEps = 1e-3f;            // 1 mm — below this the pose value is "unchanged"
    constexpr float kThetaEps = 1e-3f;          // ~0.06°
    constexpr std::uint64_t kStaleMs = 400;     // no change for this long ⇒ treat the robot as stopped

    if (!prev_robot_pose_.has_value())
    {
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
        return;
    }

    const Eigen::Vector2f dpos = pose.pos - prev_robot_pose_->pos;
    const float dtheta = std::remainder(pose.theta - prev_robot_pose_->theta, kTwoPi);
    const bool changed = dpos.norm() > kPosEps || std::abs(dtheta) > kThetaEps;

    if (changed && timestamp_ms > prev_robot_ts_ms_)
    {
        // Velocity over the actual pose-CHANGE interval. The room←robot feed updates coarser than the
        // control loop, so per-cycle differencing would alternate real steps with zeros (bursty); we
        // only difference when the value moves, over the real elapsed time.
        const float dt = static_cast<float>(timestamp_ms - prev_robot_ts_ms_) * 1e-3f;
        if (dt > 1e-3f)
        {
            const Eigen::Vector2f v = dpos / dt;   // room-frame m/s
            const float w = dtheta / dt;
            // EMA so a single jittery sample doesn't spuriously trip the stillness gate / overlay.
            base_speed_lin_ = 0.5f * base_speed_lin_ + 0.5f * v.norm();
            base_speed_ang_ = 0.5f * base_speed_ang_ + 0.5f * std::abs(w);
            room_vel_.vx    = 0.5f * room_vel_.vx + 0.5f * v.x();
            room_vel_.vy    = 0.5f * room_vel_.vy + 0.5f * v.y();
            room_vel_.omega = 0.5f * room_vel_.omega + 0.5f * w;
        }
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
    }
    else if (timestamp_ms - last_pose_change_ms_ > kStaleMs)
    {
        // Pose value hasn't moved for a while ⇒ robot stopped: zero the velocity so the overlay
        // settles (and the pose-age clock restarts here, so a fresh pose isn't seen as ancient).
        base_speed_lin_ = 0.f;
        base_speed_ang_ = 0.f;
        room_vel_ = ControllerRoomVelocity{};
        prev_robot_pose_  = pose;
        prev_robot_ts_ms_ = timestamp_ms;
        last_pose_change_ms_ = timestamp_ms;
    }
}

void ControllerSession::update_overlay_extrapolation(const ControllerWorldModel &world_model,
                                                     const ControllerRobotPose &robot_pose,
                                                     std::uint64_t timestamp_ms,
                                                     const std::optional<std::int64_t> &rt_block_lead_ms,
                                                     std::int64_t rt_twist_fix_dt_ms,
                                                     const ControllerObstacleTracker &obstacle_tracker)
{
    if (!params_)
        return;
    if (!overlay_lidar_ts_ms_.has_value() || timestamp_ms <= *overlay_lidar_ts_ms_)
        return;

    const std::uint64_t gap_ms = timestamp_ms - *overlay_lidar_ts_ms_;   // lidar staleness
    // The room←robot value updates coarser than the lidar, so the cloud's anchor pose is as old as
    // the time since that value last changed. Reported, not acted on.
    const std::uint64_t pose_age_ms = timestamp_ms >= last_pose_change_ms_ ? timestamp_ms - last_pose_change_ms_ : 0;

    // RT-staleness probe: the room←robot pose the tree returns at "now" vs at the lidar stamp. If
    // RTdelta≈0 while the robot moves, the upstream pose feed is stale/clamped — the real lag source.
    float rt_delta = -1.f;
    if (const auto pose_now_rt = world_model.read_robot_pose_in_room(timestamp_ms, std::nullopt);
        pose_now_rt.has_value())
        rt_delta = (pose_now_rt->pos - robot_pose.pos).norm();

    // Commanded (robot_ref_*) vs measured (robot_current_*) base velocity, read from the shared graph,
    // so the joystick/actuation sign can be checked against the robot's actual rotation. cmd_* is the
    // reference written by whoever commands (controller / joystick path); cur_* is the measured motion
    // (robot_concept writes it from the FullPose estimator). Same-sign rotation ⇒ consistent.
    float cmd_adv = 0.f, cmd_rot = 0.f, cur_adv = 0.f, cur_rot = 0.f;
    if (graph_)
    {
        if (const auto rid = world_model.graph_state().robot_id; rid != 0)
            if (auto robot_node = graph_->get_node(rid); robot_node.has_value())
            {
                cmd_adv = graph_->get_attrib_by_name<robot_ref_adv_speed_att>(*robot_node).value_or(0.f);
                cmd_rot = graph_->get_attrib_by_name<robot_ref_rot_speed_att>(*robot_node).value_or(0.f);
                cur_adv = graph_->get_attrib_by_name<robot_current_advance_speed_att>(*robot_node).value_or(0.f);
                cur_rot = graph_->get_attrib_by_name<robot_current_angular_speed_att>(*robot_node).value_or(0.f);
            }
    }

    // CSV (lazy-open + header, truncated each run), throttled to ~10 Hz.
    if (!params_->overlay_csv_path.empty() && timestamp_ms - overlay_csv_last_ms_ >= 100)
    {
        overlay_csv_last_ms_ = timestamp_ms;
        if (!overlay_csv_open_)
        {
            overlay_csv_.open(params_->overlay_csv_path, std::ios::out | std::ios::trunc);
            if (overlay_csv_.is_open())
                overlay_csv_ << "t_ms,lidar_ts,gap_ms,pose_age_ms,vx,vy,omega,RTdelta_m,"
                                "cmd_adv,cmd_rot,cur_adv,cur_rot,rt_lead_ms,rt_fix_dt_ms,"
                                "twist_pred_dt_ms,twist_pred_err_m,twist_pred_err_deg\n";
            // Announce the resolved absolute path (it's a relative path → lands in the process CWD,
            // which is easy to miss), or the failure — so this is never silently a no-op again.
            std::error_code ec;
            const auto abs_path = std::filesystem::absolute(params_->overlay_csv_path, ec).string();
            std::cout << "[OverlayExtrap] CSV " << (overlay_csv_.is_open() ? "writing to " : "FAILED to open ")
                      << abs_path << std::endl;
            overlay_csv_open_ = true;
        }
        if (overlay_csv_.is_open())
        {
            overlay_csv_ << timestamp_ms << ',' << *overlay_lidar_ts_ms_ << ',' << gap_ms << ',' << pose_age_ms << ','
                         << room_vel_.vx << ',' << room_vel_.vy << ',' << room_vel_.omega << ','
                         << rt_delta << ','
                         << cmd_adv << ',' << cmd_rot << ',' << cur_adv << ',' << cur_rot << ','
                         // empty (not 0) when the ring carries no timestamps at all — that is a
                         // different failure from "the feed is level with the scan".
                         << (rt_block_lead_ms.has_value() ? std::to_string(*rt_block_lead_ms) : std::string{}) << ','
                         // signed ms the twist walked the pose onto the scan (0 = query was inside
                         // the ring, nothing to repair). |rt_fix_dt_ms| * omega is the bulk cloud
                         // rotation this removed.
                         << rt_twist_fix_dt_ms << ','
                         // Twist-vs-RT residual over one lidar period: what dropping the one-frame
                         // buffer would cost in registration accuracy. Empty when the probe could
                         // not run (ring too short to hold both ends).
                         << obstacle_tracker.twist_pred_dt_ms() << ','
                         // ★ NOT std::to_string for floats: it formats through the C locale, which is
                         // es_ES on this machine, so it emits a decimal COMMA — the field splits in two,
                         // the row grows past the header and every parse downstream is silently wrong
                         // (hit exactly that on 2026-08-04). Stream insertion uses the ofstream's own
                         // locale, same as every other float column in this row.
                         ;
            if (obstacle_tracker.twist_pred_err_m().has_value())
                overlay_csv_ << *obstacle_tracker.twist_pred_err_m();
            overlay_csv_ << ',';
            if (obstacle_tracker.twist_pred_err_deg().has_value())
                overlay_csv_ << *obstacle_tracker.twist_pred_err_deg();
            overlay_csv_ << '\n';
            overlay_csv_.flush();
        }
    }
}

bool ControllerSession::robot_still() const
{
    return rc::affordance::stillness_ok(base_speed_lin_, base_speed_ang_, active_contract_);
}

bool ControllerSession::step_lockon(ControllerMotionCommander &motion_commander, const TimeSource &time_source)
{
    const auto reading = read_servo_reading(feedback_node_id_);
    const bool met = goal_met(feedback_node_id_);
    const auto cmd = lockon_.update(reading, met, time_source());
    // Quasi-static: drive the (tiny, capped) nudge only during STEP; hold still while SETTLING/done.
    if (lockon_.active() && (cmd.adv != 0.0f || cmd.side != 0.0f || cmd.rot != 0.0f))
        motion_commander.send_speed_command(cmd.adv, cmd.side, cmd.rot);
    else
        motion_commander.stop_robot();
    if (lockon_.done())
        qInfo() << "[affordance] lock-on" << (lockon_.locked() ? "LOCKED" : "GIVE_UP")
                << "| node" << active_target_id_;
    return lockon_.done();
}

bool ControllerSession::step_orient(const ControllerRobotPose &robot_pose,
                                    ControllerMotionCommander &motion_commander,
                                    const TimeSource &time_source, float target_yaw)
{
    const std::uint64_t now = time_source();
    if (!orient_start_ms_)
        orient_start_ms_ = now;

    // Completion: the contract's goal predicate (e.g. chair_detection_alive) held stable_n measurements =
    // the glance paid off; or the contract timed out = give up. Consume the affordance either way.
    if (goal_met(feedback_node_id_)) ++orient_stable_; else orient_stable_ = 0;
    const bool looked    = orient_stable_ >= std::max(1, active_contract_.stable_n);
    const bool timed_out = static_cast<double>(now - *orient_start_ms_) > active_contract_.timeout_ms;
    if (looked || timed_out)
    {
        motion_commander.stop_robot();
        qInfo() << "[affordance] orient" << (looked ? "LOOKED (detection)" : "GIVE_UP (timeout)")
                << "| node" << active_target_id_;
        orient_start_ms_.reset();
        orient_stable_ = 0;
        return true;
    }

    // Rotate the base toward the target bearing (capped). Once nearly aligned, HOLD STILL so the look is
    // motion-free (the Orient contract's .still asks for a quiet capture) and wait for the detection.
    const float yaw_err = std::atan2(std::sin(target_yaw - robot_pose.theta),
                                     std::cos(target_yaw - robot_pose.theta));
    const float k   = params_ ? params_->lockon_k_yaw       : 0.8f;
    const float cap = params_ ? params_->lockon_max_yaw_rps : 0.12f;
    float rot = std::clamp(k * yaw_err, -cap, cap);
    if (std::abs(yaw_err) < 0.05f)
        rot = 0.0f;
    if (rot != 0.0f) motion_commander.send_speed_command(0.0f, 0.0f, rot);
    else             motion_commander.stop_robot();
    return false;
}

void ControllerSession::finalize_reached(rc::AffordanceManager &affordance_manager,
                                         rc::TrajectoryController &path_controller,
                                         ControllerMotionCommander &motion_commander,
                                         ControllerDisplay &display,
                                         const Eigen::Vector2f &arrived_at,
                                         std::uint64_t now_ms)
{
    // A mission waypoint is reached the same way any target is; stepping the mission here means arrival
    // logic exists in exactly one place and a mission cannot drift out of sync with what the robot did.
    const bool mission_continues = false;   // a mission is ended by arc length, never by an arrival
    if (graph_)
        affordance_manager.mark_reached(graph_);
    lockon_.reset();
    orient_start_ms_.reset();
    orient_stable_ = 0;
    reset_stuck_state();
    feedback_node_id_ = 0;
    current_plan_.reset();
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    last_target_info_.reset();
    active_target_id_ = 0;
    current_target_room_.reset();
    manual_target_room_.reset();
    manual_target_origin_room_.reset();
    manual_target_dirty_ = false;
    // The clicked point has been reached, so its marker stops being the thing the robot is driving to.
    // Leaving it drawn would claim an active target that is already history.
    mission_.set_click_target(std::nullopt);

    if (mission_continues)
    {
        // ── FLY-THROUGH ──
        // An intermediate waypoint is a place to PASS, not a place to arrive. Stopping at each one would
        // measure the start/stop transient N times instead of the trajectory, and it is not what the tour
        // is asking for: the mission already has somewhere else to be.
        //
        // So: keep the base moving. current_plan_ is reset so the next cycle plans to the new waypoint, and
        // the target name changed, so ensure_current_plan hands the follower a fresh path — none of which
        // needs the base stopped first. Not calling stop_robot() means the motion commander keeps issuing
        // the last command (under its usual freshness decay) across the one cycle it takes to replan,
        // instead of a zero that the robot would have to accelerate out of again.
        // The MPPI's warm start is deliberately left intact for the same reason: it is the continuity.
        //
        // The corner-cutting this allows is bounded by the follower's goal threshold, and it is MEASURED —
        // arrival_error_m records exactly how close to each waypoint the robot actually came.
        return;
    }

    clear_tracking_state();
    display.clear_robot_trajectory();
    path_controller.stop();
    motion_commander.stop_robot();
}

int ControllerSession::smooth_selected_mission()
{
    if (grid_planner_.width() == 0 or grid_planner_.height() == 0)
    {
        std::println("[controller] cannot smooth: no occupancy grid yet — wait for the room and obstacles.");
        return 0;
    }
    return mission_.smooth_selected(
        [this](const Eigen::Vector2f &p, float heading) { return grid_planner_.pose_free(p, heading); });
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
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
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
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    last_target_info_.reset();
    active_target_id_ = 0;
    path_controller.stop();
    motion_commander.stop_robot();
    if (wake_callback)
        wake_callback();
}

// ─── Physical-stuck recovery ──────────────────────────────────────────────────────────

void ControllerSession::reset_stuck_state()
{
    reset_stuck_window();   // the anchor and the commanded-travel accumulator go with the clock
    escape_active_ = false;
}

bool ControllerSession::detect_stuck(bool pursuing, float cmd_lin_mps,
                                     const Eigen::Vector2f &pos_room, std::uint64_t now_ms)
{
    if (!params_ || !params_->stuck_recovery_enabled) { reset_stuck_window(); return false; }
    // Not commanding translation ⇒ nothing is being predicted, so nothing can be contradicted.
    if (!pursuing || cmd_lin_mps <= 0.f) { reset_stuck_window(); return false; }

    // ── A WEDGE IS UNACHIEVED NET DISPLACEMENT, NOT A LOW SPEED READING ──────────────────────────
    // This used to compare base_speed_lin_ against stuck_slip_ratio * commanded. base_speed_lin_ is an
    // EMA of |dpos|/dt evaluated ONLY on cycles where the pose changed, dividing by the pose-CHANGE
    // interval — a design that is right for a smooth velocity readout and structurally unable to express
    // a wedge. A robot jittering +-3 cm about a fixed point produces large |dpos| over a short dt, so it
    // reports a healthy speed while going nowhere. MEASURED on the run that prompted this: commanded
    // 10.33 m of travel over 20 s, path length 2.13 m, NET DISPLACEMENT 0.008 m — eight millimetres —
    // with the base pinned at 0.55 m/s and 0.80 rad/s, no safety guard firing, and the wedge silent.
    //
    // So compare the two quantities the prediction is actually about: how far the robot was TOLD to
    // travel since the clock started, against how far it ACTUALLY GOT from where it started. Net
    // displacement is immune to oscillation by construction — jitter contributes nothing to it, which is
    // the whole point — and no new parameter appears: stuck_slip_ratio and stuck_confirm_ms keep their
    // meanings, applied to distance instead of to speed.
    if (stuck_since_ms_ == 0)
    {
        stuck_since_ms_ = now_ms;
        stuck_anchor_pos_ = pos_room;
        stuck_cmd_travel_m_ = 0.f;
        stuck_last_ms_ = now_ms;
        return false;
    }
    const float dt = static_cast<float>(now_ms - stuck_last_ms_) * 1e-3f;
    stuck_last_ms_ = now_ms;
    if (dt > 0.f) stuck_cmd_travel_m_ += cmd_lin_mps * dt;

    // ★JUDGED AT THE END OF A FULL WINDOW, NOT CONTINUOUSLY. Testing the ratio on every cycle looks
    // natural and is wrong: early in the window the commanded travel is a few centimetres, so ANY pose
    // jitter clears stuck_slip_ratio * that, restarts the clock, and the window can never mature. (My
    // first version did exactly this and the replay below never fired.) Over a full stuck_confirm_ms the
    // commanded travel is ~0.8 m at cruise, so the bar is ~0.2 m — far above jitter, and far below what
    // a healthy robot covers. Same two parameters, and stuck_confirm_ms now also means what it says:
    // how long the robot is watched before being judged.
    if (now_ms - stuck_since_ms_ <= static_cast<std::uint64_t>(params_->stuck_confirm_ms)) return false;

    const float achieved_m = (pos_room - stuck_anchor_pos_).norm();
    if (achieved_m >= params_->stuck_slip_ratio * stuck_cmd_travel_m_)
    {
        // It went where it was told. Start the next window HERE, so each verdict is about one stretch.
        stuck_since_ms_ = now_ms;
        stuck_anchor_pos_ = pos_room;
        stuck_cmd_travel_m_ = 0.f;
        return false;
    }
    std::println("[controller] WEDGE — commanded {:.2f} m of travel over {:.1f} s and achieved {:.3f} m "
                 "net ({:.0f}% of it, floor {:.0f}%). Escaping.",
                 stuck_cmd_travel_m_, static_cast<float>(now_ms - stuck_since_ms_) * 1e-3f, achieved_m,
                 stuck_cmd_travel_m_ > 1e-6f ? 100.f * achieved_m / stuck_cmd_travel_m_ : 0.f,
                 100.f * params_->stuck_slip_ratio);
    return true;
}

void ControllerSession::reset_stuck_window()
{
    stuck_since_ms_ = 0;
    stuck_cmd_travel_m_ = 0.f;
}

void ControllerSession::begin_escape(const ControllerRobotPose &robot_pose,
                                     ControllerObstacleTracker &obstacle_tracker,
                                     rc::TrajectoryController &path_controller,
                                     std::uint64_t now_ms)
{
    // Choose the turn direction toward whichever side has more ESDF clearance.
    //
    // ⚠ TWO DIFFERENT ROBOT FRAMES ARE IN PLAY HERE — they are NOT the same one.
    //   • The COMMAND frame (OmniRobot / send_speed_command): adv = forward, side = +y left.
    //   • The ESDF frame that clearance_at(rx, ry) queries: +Y IS FORWARD, +X IS RIGHT.
    //     That is the frame TrajectoryController integrates its rollouts in
    //     (trajectory_controller.cpp: `x += adv*sin(theta)`, `y += adv*cos(theta)`, with the
    //     explicit comment "Differential-drive kinematics (Y+ = forward, X+ = right)").
    // So a SIDE probe varies rx, and a FRONT/REAR probe varies ry. This block previously
    // probed (0, +probe)/(0, −probe) and called them left/right — they were front and rear,
    // which meant the turn direction was picked by comparing FRONT clearance against REAR,
    // and since a wedge means something is in front, the sign came out the same nearly every
    // time. Do not "simplify" these back to the y axis.
    const float probe = params_ ? params_->escape_side_probe_m : 0.5f;
    const float cl = path_controller.clearance_at(-probe, 0.f);   // left clearance  (−x = left)
    const float cr = path_controller.clearance_at(+probe, 0.f);   // right clearance (+x = right)
    if (std::abs(cl - cr) > 0.05f)
        escape_turn_sign_ = (cl > cr) ? +1.f : -1.f;
    else
        escape_turn_sign_ = (escape_count_ % 2 == 0) ? +1.f : -1.f;

    // Mark the stuck spot just ahead of the robot so the replanner routes around whatever we
    // wedged on. We drop a LOCAL-ONLY VIRTUAL obstacle (not the LiDAR-observed temp obstacle):
    // the whole point of a stuck event is that the blocker is invisible to the pipeline — the
    // LiDAR path would find no points and create nothing, so the planner would just reproduce the
    // same blocked route. The virtual disc is geometric, always succeeds, is visible to the
    // planner/MPPI, and is NOT uploaded to DSR. It ages out on its TTL so a since-moved obstacle
    // is forgotten; if we re-wedge, another one is dropped.
    const float vrad        = params_ ? params_->stuck_virtual_obstacle_radius_m  : 0.30f;
    const float fwd_off_cfg = params_ ? params_->stuck_virtual_obstacle_forward_m : 0.40f;
    // Worst-case body extent: the disc is dropped ahead of the robot before it turns, so it must clear the
    // body at any heading. Asked of the footprint, not of a config knob that also meant three other things.
    const float body_radius = path_controller.footprint().circumscribed_radius();
    // Push the disc far enough ahead that its NEAR edge clears the robot footprint. Otherwise the marker
    // overlaps the body (default 0.40 ahead − 0.30 radius = 0.10 m from centre, well inside clearance),
    // the planner/MPPI reads the robot as "already colliding" with its own marker, no rollout escapes,
    // and we re-stick on ourselves. Near edge = fwd_off − vrad ≥ body_radius (+ a small margin).
    const float fwd_off = std::max(fwd_off_cfg, body_radius + vrad + 0.05f);
    const Eigen::Vector2f fwd(std::cos(robot_pose.theta), std::sin(robot_pose.theta));
    const Eigen::Vector2f stuck_center = robot_pose.pos + fwd_off * fwd;
    // Gated with the rest of controller-side obstacle creation. This disc was the one source NOT covered by
    // obstacle_creation_enabled, and it is the one that litters the map: every wedge drops a fresh 0.30 m disc
    // ~0.85 m ahead, and the robot has been wedging constantly (the no-route branch counts as a wedge, so an
    // unreachable target alone produces a disc every few seconds along the whole approach). The escape
    // maneuver itself — reverse and turn out — is a genuinely useful reflex and still runs; only the map
    // pollution is removed. Re-enable with Controller.ObstacleCreationEnabled once residual_concept is trusted
    // as the sole obstacle source.
    if (!params_ || params_->obstacle_creation_enabled)
        obstacle_tracker.add_virtual_obstacle(now_ms, stuck_center, vrad);
    // An escape is an exceptional state transition, so it gets one line — but only the facts that identify
    // the episode. The geometry dump that used to live here existed to prove the disc was not trapping the
    // robot on itself; that is now guaranteed by construction (fwd_off clears the footprint above).
    std::println("[controller] ESCAPE #{} — wedged, reversing and turning {}. Side clearance L={:.2f} R={:.2f} m.",
                 escape_count_ + 1, escape_turn_sign_ > 0 ? "left" : "right", cl, cr);

    mission_.note_escape();
    escape_active_   = true;
    escape_start_ms_ = now_ms;
    escape_start_pos_ = robot_pose.pos;
    ++escape_count_;
    stuck_since_ms_ = 0;

    clear_tracking_state();
    current_plan_.reset();
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    // Same reason as the visible-blockage path: the virtual disc just dropped ahead of the robot is only
    // useful if something re-plans against it, and in continuous-route mode nothing does unless asked.
    route_repair_pending_ = true;
    path_controller.stop();

    qInfo() << "[recovery] STUCK -> escape | reverse + turn"
            << (escape_turn_sign_ > 0 ? "LEFT" : "RIGHT")
            << "| clearance L" << cl << "R" << cr << "| attempt" << escape_count_;
}

void ControllerSession::step_escape(const ControllerRobotPose &robot_pose,
                                    rc::TrajectoryController &path_controller,
                                    ControllerMotionCommander &motion_commander,
                                    std::uint64_t now_ms)
{
    const float backed = (robot_pose.pos - escape_start_pos_).norm();
    const float max_ms = params_ ? params_->escape_max_ms : 1500.f;
    const float max_dist = params_ ? params_->escape_distance_m : 0.30f;
    if (now_ms - escape_start_ms_ > static_cast<std::uint64_t>(max_ms) || backed > max_dist)
    {
        escape_active_ = false;
        motion_commander.stop_robot();   // plan already reset in begin_escape → next cycle replans
        qInfo() << "[recovery] escape done | backed" << backed << "m";
        return;
    }

    // Rear-clearance guard: don't reverse into a wall. If the space behind is tight, escape
    // by rotating in place only (adv = 0).
    //
    // ⚠ clearance_at() is in the ESDF/rollout frame: +Y FORWARD, +X RIGHT (see begin_escape
    // above for why this is not the OmniRobot command frame). REAR is therefore −y, not −x.
    // This read used to be clearance_at(−rear_probe, 0) — that is the LEFT side, so the guard
    // was gating "may I reverse?" on whether the robot's LEFT was clear, and would happily
    // reverse into a wall behind it any time the left flank was open.
    const float rear_probe = params_ ? params_->escape_rear_probe_m : 0.45f;
    const float rear_min   = params_ ? params_->escape_rear_min_m   : 0.30f;
    const float rear = path_controller.clearance_at(0.f, -rear_probe);
    const float adv = (rear < rear_min) ? 0.f : -(params_ ? params_->escape_adv_speed_mps : 0.15f);
    const float rot = escape_turn_sign_ * (params_ ? params_->escape_rot_speed_rps : 0.35f);
    motion_commander.send_speed_command(adv, 0.f, rot);
}

void ControllerSession::stop(rc::TrajectoryController &path_controller,
                             ControllerMotionCommander &motion_commander)
{
    path_controller.stop();
    reset_stuck_state();
    clear_tracking_state();
    motion_commander.stop_robot();
}

void ControllerSession::abort(rc::TrajectoryController &path_controller,
                              ControllerMotionCommander &motion_commander)
{
    stop(path_controller, motion_commander);
    // Everything that would let the next Run resume the OLD activity rather than start a new one.
    route_active_ = false;
    route_repair_pending_ = false;
    current_plan_.reset();
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    active_target_id_ = 0;
    last_target_info_.reset();
    // Band bookkeeping belongs to the route that is going away.
    band_cycle_ = 0;
}