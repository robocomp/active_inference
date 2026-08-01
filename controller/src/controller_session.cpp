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
    if (!world_model.refresh_graph_state())
    {
        room_polygon_.clear();
        current_plan_.reset();
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
    update_overlay_extrapolation(world_model, *robot_pose, timestamp_ms);

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
        if (!target_wait_logged_ and mission_.mode() == rc::DriveMode::AffordancesOnly)
        {
            qInfo() << "Controller waiting for an affordance target in DSR";
            target_wait_logged_ = true;
        }
        current_plan_.reset();
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
    const auto safe = grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad);
    if (safe.has_value() && (*safe - step.target.room_pos).squaredNorm() > 1e-6f)
    {
        std::println("[controller] target '{}' blocked → repaired ({:.2f},{:.2f}) → ({:.2f},{:.2f}){}",
                     step.target.node_name, step.target.room_pos.x(), step.target.room_pos.y(),
                     safe->x(), safe->y(), "");
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

ControllerPolygon ControllerSession::smooth_plan(const ControllerPolygon &poly) const
{
    if (poly.size() < 2 or params_ == nullptr or not params_->smooth_planned_path) return poly;
    rc::RouteSpline spline;
    if (not spline.build(poly, params_->route_spacing_m,
                         [this](const Eigen::Vector2f &p, float h) { return grid_planner_.pose_free(p, h); },
                         params_->route_smoothing_m))
        return poly;   // smoothing is an improvement, never a precondition: fall back to the polyline
    return spline.samples();
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
    std::vector<Eigen::Vector2f> wps;
    wps.reserve(m->waypoints.size());
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
        route_.set_optimizer(opt);
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

    float v = v_cap;
    // Sampled at 10 cm against a 5 cm curve — deliberately coarser than the curve's own spacing, because
    // curvature_at is a second difference of the samples and is therefore noisiest at that scale.
    for (float ds = 0.f; ds <= horizon; ds += 0.10f)
    {
        const float k = std::abs(route_.spline().curvature_at(s_now + ds));
        if (k < 1e-3f) continue;                       // straight: no constraint from here
        const float v_here = std::sqrt(a_lat / k);     // v^2·kappa = a_lat, the whole model
        // The bound is on the speed we may hold NOW: we must be able to shed the difference over ds.
        const float v_allowed = std::sqrt(v_here * v_here + 2.f * a_dec * ds);
        v = std::min(v, v_allowed);
    }
    // A floor purely against numerical noise in the curvature estimate, not a behavioural knob: a
    // spurious kappa spike must not be able to command a standstill. A differential drive has no minimum
    // turn radius — it can rotate in place — so a genuinely sharp corner is handled by the rotation, and
    // the forward speed never needs to reach zero for geometric reasons.
    return std::clamp(v, 0.15f, v_cap);
}

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
            current_plan_.reset();
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
        path_controller.set_goal_facing_yaw(step.target.from_affordance
                                                ? std::optional<float>(step.target.yaw_rad)
                                                : std::nullopt);
        // A point target DOES end at its endpoint — restore the follower's own arrival test, which a
        // continuous route switched off (it may have been the previous thing installed).
        path_controller.set_endpoint_arrival(true);
    }

    return true;
}

void ControllerSession::update_display(const std::optional<ControllerRobotPose> &robot_pose,
                                       ControllerDisplay &display,
                                       const ControllerObstacleVisuals &obstacle_polys,
                                       const ControllerPolygons &obstacle_rfe_points,
                                       int max_lidar_draw_points) const
{
    // Dead-reckoning of the DISPLAYED cloud + icon to "now" is computed once per cycle in
    // update_overlay_extrapolation (stored in overlay_icon_pose_ / overlay_correction_); here we
    // just apply it. The lidar buffer (obstacle detection) stays anchored at scan time.
    auto icon_pose = robot_pose;
    std::optional<Eigen::Affine2f> lidar_correction;
    if (robot_pose.has_value())
    {
        if (overlay_icon_pose_.has_value())
            icon_pose = overlay_icon_pose_;
        lidar_correction = overlay_correction_;
    }

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

    // Curvature-limited speed. Only a continuous route carries kappa(s); a click target has no curve, so
    // it keeps the full envelope and the ceiling is cleared rather than left stale from a previous mission.
    if (route_active_ and mission_.running())
        path_controller.set_speed_limit(route_speed_limit(params_ ? params_->max_adv_speed_mps : 0.7f,
                                                          path_controller.params.cbf_max_decel));
    else
        path_controller.set_speed_limit(std::nullopt);

    const auto control_output = path_controller.compute(robot_pose.as_transform());
    // Surface what the ARRIVAL test is waiting on, every cycle, before any of the branches below can return
    // early — otherwise the readout would freeze exactly in the states worth watching (aligning, blocked).
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
        // Deviation from the reference curve — the continuous tracking signal. NaN when there is no
        // reference (a click target has no route), and the stats skip it rather than inventing a zero.
        float cross_track = std::numeric_limits<float>::quiet_NaN();
        float heading_err = 0.f, ref_kappa = 0.f;
        if (route_active_)
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
    const bool wedged_this_cycle = base_speed_lin_ < params_->stuck_slip_ratio * cmd_lin;
    if (detect_stuck(/*pursuing=*/true, wedged_this_cycle, time_source()))
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
                                                     std::uint64_t timestamp_ms)
{
    overlay_icon_pose_.reset();
    overlay_correction_.reset();
    if (!params_)
        return;
    if (!overlay_lidar_ts_ms_.has_value() || timestamp_ms <= *overlay_lidar_ts_ms_)
        return;

    const std::uint64_t gap_ms = timestamp_ms - *overlay_lidar_ts_ms_;   // lidar staleness (informational)
    // The room←robot value updates coarser than the lidar, so the cloud's anchor pose is as old as
    // the time since that value last changed — extrapolate by THAT, not just the lidar gap.
    const std::uint64_t pose_age_ms = timestamp_ms >= last_pose_change_ms_ ? timestamp_ms - last_pose_change_ms_ : 0;
    const float raw_dt = static_cast<float>(pose_age_ms) * 1e-3f;
    const float dt = std::min(raw_dt, std::max(0.f, params_->overlay_extrapolation_max_dt_s));
    const auto now_pose = extrapolate_room_pose(robot_pose, room_vel_, dt);

    // Apply the correction to the drawn cloud + icon only when extrapolating to "now". When the
    // cloud is drawn one frame old it is already exactly registered (the pose is bracketed, not
    // clamped), so the correction is left identity — pushing it forward would undo the exactness.
    // The diagnostics below run regardless, so the CSV still logs a baseline for A/B.
    if (params_->overlay_extrapolate_to_now && !params_->overlay_draw_one_frame_old)
    {
        overlay_icon_pose_ = now_pose;
        overlay_correction_ = now_pose.as_transform() * robot_pose.as_transform().inverse();
    }

    const Eigen::Vector2f disp = now_pose.pos - robot_pose.pos;
    const float dtheta = now_pose.theta - robot_pose.theta;

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
                overlay_csv_ << "t_ms,lidar_ts,gap_ms,pose_age_ms,vx,vy,omega,extrap_dt_s,disp_m,dtheta_rad,RTdelta_m,"
                                "cmd_adv,cmd_rot,cur_adv,cur_rot\n";
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
                         << dt << ',' << disp.norm() << ',' << dtheta << ',' << rt_delta << ','
                         << cmd_adv << ',' << cmd_rot << ',' << cur_adv << ',' << cur_rot << '\n';
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
    stuck_since_ms_ = 0;
    escape_active_ = false;
}

bool ControllerSession::detect_stuck(bool pursuing, bool stalled_this_cycle, std::uint64_t now_ms)
{
    if (!params_ || !params_->stuck_recovery_enabled)
    {
        stuck_since_ms_ = 0;
        return false;
    }
    // Pure debounce over the caller's per-cycle wedge signal. `pursuing` means an active, unreached
    // target (goal_reached / lock-on / orient / path_blocked all returned before this). `stalled_this_
    // cycle` is a PREDICTION ERROR: the robot is commanding translation but the base isn't achieving it
    // (execute_plan), or there is no route at all (ensure_current_plan). A robot that IS moving as
    // commanded — detour, slow nav, arrival rotation, a creep that is still sliding — is not stalled, so
    // none of those false-fire. Only a command-without-motion sustained for stuck_confirm_ms is a wedge.
    if (!pursuing || !stalled_this_cycle)
    {
        stuck_since_ms_ = 0;
        return false;
    }
    if (stuck_since_ms_ == 0)
    {
        stuck_since_ms_ = now_ms;   // start the wedge clock (a brief slip/hesitation is not a wedge)
        return false;
    }
    return now_ms - stuck_since_ms_ > static_cast<std::uint64_t>(params_->stuck_confirm_ms);
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