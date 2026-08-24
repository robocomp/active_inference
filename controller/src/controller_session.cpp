#include <numbers>
#include "controller_session.h"
#include "corner_visibility.h"   // point_in_polygon: is the standpoint in the room at all

#include <algorithm>
#include <ranges>
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
// ── HOW LONG A REFUSED AFFORDANCE STAYS OUT OF CONTENTION ────────────────────────────────────────
// Only long enough for the selector to pick something ELSE this round. It is NOT the memory of a bad
// standpoint — that is known_useless_spot(), which is keyed on the POSITION and consulted before the
// robot drives anywhere, so a re-offered bad pose costs nothing however short this is.
// ★SMALL ON PURPOSE, and room is why. Suppression is by NODE id, and room_concept publishes every
// standpoint it will ever propose through ONE `afford_room` node — so a long suppression there does not
// retire a bad spot, it halts room exploration entirely, including standpoints across the apartment that
// are perfectly reachable. It was 400 rounds (~20 s at 20 Hz), which in a room-only run leaves the robot
// with nothing to do at all. A few cycles breaks the immediate selection loop; the position memory does
// the actual remembering.
constexpr int kUnreachableRounds = 3;
// How long a "the robot failed here" memory may veto a standpoint. Capped at the producer's recovery
// time (room_concept IorDecayTime = 120 s); see UselessSpot in the header for the measurement.
constexpr std::uint64_t kUselessSpotMemoryMs = 120000;

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
        c.settle_max_ms   = params_->lockon_settle_max_ms;
        c.settle_new_frames = params_->lockon_settle_new_frames;
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
    grid_planner_.set_world(room_polygon_, obstacle_tracker.obstacle_polygons(),
                            {.centres = obstacle_tracker.residual_cells(),
                             .cell_size_m = obstacle_tracker.residual_cell_size_m()});
    return true;
}


// ★★★THE RECORD MUST SURVIVE THE FAILURE IT EXPLAINS. The first version of this logged from inside
// execute_plan, so when the selector found nothing there was no plan, execute_plan never ran, and the
// log went SILENT for the whole freeze — a 55 s hole where the evidence should have been. A diagnostic
// that stops writing exactly when the thing goes wrong is worse than none, because the gap reads as
// "nothing happened". So it is written from build_planning_step, before any early return can skip it.
namespace
{
// ── STANDPOINT AUDIT ────────────────────────────────────────────────────────────────────────────
// ★WRITTEN UNCONDITIONALLY, BECAUSE THE PRINT IT REPLACES IS DEDUPED. The repair's own [controller]
// line only speaks when the ANSWER changes, so a repair that keeps returning the same point while the
// robot stands still is silent for as long as the failure lasts — which is exactly the interval that
// needs explaining (measured 2026-08-19: 112 completions, robot < 5 cm, not one repair line). This
// row carries the three poses side by side — what the PRODUCER published, what we drive to after the
// repair, and where the robot IS — so "arrived" can never again be read without seeing which target
// it arrived at. Locale-pinned: this file is read back by tools (CLAUDE.md's from_chars rule).
void audit_standpoint(std::string_view event, std::uint64_t t_ms,
                      const Eigen::Vector2f &wire, const Eigen::Vector2f &tgt,
                      const Eigen::Vector2f &rob, std::string_view branch, std::string_view detail,
                      int target_new, int goal_reached, int arrival_real)
{
    static std::ofstream f;
    static bool ok = false;
    if (not ok)
    {
        f.open("standpoint_audit.jsonl", std::ios::out | std::ios::trunc);
        f.imbue(std::locale::classic());
        ok = f.is_open();
    }
    if (not ok) return;
    f << std::format(R"({{"t_ms":{},"event":"{}","wire_x":{:.3f},"wire_y":{:.3f},)"
                     R"("tgt_x":{:.3f},"tgt_y":{:.3f},"rob_x":{:.3f},"rob_y":{:.3f},)"
                     R"("d_wire":{:.3f},"d_tgt":{:.3f},"repair_m":{:.3f},)"
                     R"("branch":"{}","detail":"{}","target_new":{},"goal_reached":{},"arrival_real":{}}})" "\n",
                     t_ms, event, wire.x(), wire.y(), tgt.x(), tgt.y(), rob.x(), rob.y(),
                     (wire - rob).norm(), (tgt - rob).norm(), (tgt - wire).norm(),
                     branch, detail, target_new, goal_reached, arrival_real);
    f.flush();
}
}   // namespace

// ── WHICH BRANCH STOPPED THE BASE ───────────────────────────────────────────────────────────────
// ★"Stalled" has meant three different things in this run's diagnostics, and only one of them is a
// zero command: the robot standing still because execute_plan returned before reaching the speed
// funnel is invisible in base_commands.jsonl, which only records commands that were SENT. Ten
// branches can end the cycle that way. source_location names the one that did, at 5 Hz, into the same
// audit file the standpoint rows go to — so a freeze can be read off one file instead of inferred.
void ControllerSession::note_no_command(std::source_location loc)
{
    static std::uint64_t last_ms = 0;
    const auto now = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    if (now - last_ms < 200) return;
    last_ms = now;
    const Eigen::Vector2f here = current_target_room_.value_or(Eigen::Vector2f::Zero());
    audit_standpoint("no-command", now, here, here, here, "stopped",
                     std::format("line {}", loc.line()), -1, -1, -1);
}

void ControllerSession::log_selection_json(std::uint64_t t_ms,
                                           const rc::AffordanceManager &affordance_manager,
                                           const std::optional<Eigen::Vector2f> &robot_xy,
                                           const char *stage)
{
    if (not select_json_open_)
    {
        select_json_.open("affordance_select.jsonl", std::ios::out | std::ios::trunc);
        select_json_open_ = select_json_.is_open();
    }
    if (not select_json_open_ or (t_ms - last_select_json_ms_) < 200) return;
    last_select_json_ms_ = t_ms;
    std::string cands;
    for (const auto &c : affordance_manager.last_candidates())
        cands += std::format(R"({}{{"name":"{}","state":"{}","eligible":{},"gain":{:.4f}}})",
                             cands.empty() ? "" : ",", c.node_name, c.state, c.eligible ? 1 : 0, c.gain);

    // ── THE SAME CANDIDATES, AS CONVERSATION ──────────────────────────────────────────────────────
    // A producer speaks by putting a node on offer; that is the only voice it has here. So an offer
    // appearing IS the producer's line, and the selector's choice among the offers is its own.
    // ★THE LINE WORTH HAVING IS THE ONE WHERE THE HIGHEST-SCORING CANDIDATE IS NOT TAKEN. That is
    // never wrong on its own -- JustCompleted is deliberately unclaimable -- but it is invisible in a
    // status readout, and it is what cost an entire competing traversal per pivot step.
    for (const auto &c : affordance_manager.last_candidates())
    {
        const std::string key = c.node_name + "/" + c.state;
        if (key == affordance_last_state_) continue;
        if (c.state == "Offered")
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Producer, t_ms,
                          std::format("'{}' on offer, {:.3f} nats", c.node_name, c.gain));
    }
    if (not affordance_manager.last_candidates().empty())
    {
        const auto &cs = affordance_manager.last_candidates();
        affordance_last_state_ = cs.front().node_name + "/" + cs.front().state;
        const std::string chosen = last_target_info_.has_value() ? last_target_info_->node_name
                                                                 : std::string{};
        if (not chosen.empty() and chosen != affordance_last_target_)
        {
            affordance_last_target_ = chosen;
            // Was anything scored higher and passed over? Say so, and say why.
            const auto *best = &cs.front();
            for (const auto &c : cs) if (c.gain > best->gain) best = &c;
            if (best->node_name != chosen and not best->eligible)
                note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Selector, t_ms,
                    std::format("chose '{}' -- '{}' scored higher ({:.3f}) but is {} and cannot be claimed",
                                chosen, best->node_name, best->gain, best->state));
            else
                note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Selector, t_ms, std::format("chose '{}'", chosen));
        }
    }
    select_json_ << std::format(
        R"({{"t_ms":{},"stage":"{}","rob_x":{:.3f},"rob_y":{:.3f},"has_target":{},"target":"{}",)"
        R"("tgt_x":{:.3f},"tgt_y":{:.3f},"d_target":{:.3f},"raw_x":{:.3f},"raw_y":{:.3f},)"
        R"("repair_m":{:.3f},"has_plan":{},"reject":"{}","suppressed":"{}","candidates":[{}]}})" "\n",
        t_ms, stage,
        robot_xy ? robot_xy->x() : 0.f, robot_xy ? robot_xy->y() : 0.f,
        last_target_info_.has_value() ? 1 : 0,
        last_target_info_.has_value() ? last_target_info_->node_name : std::string{},
        // ★THE THREE POSES THAT MATTER, TOGETHER. `tgt` is what the controller is driving to, `raw` is
        // what the PRODUCER published, and `repair_m` is how far our own standpoint repair moved it.
        // Without these the log cannot answer "is the refusal about the producer's cell or ours?", which
        // is exactly the question a refusal reading "already at this standpoint" raises when the
        // published cell is 1.55 m away.
        last_target_info_.has_value() ? last_target_info_->room_pos.x() : 0.f,
        last_target_info_.has_value() ? last_target_info_->room_pos.y() : 0.f,
        (last_target_info_.has_value() and robot_xy)
            ? (last_target_info_->room_pos - *robot_xy).norm() : -1.f,
        last_raw_target_pos_ ? last_raw_target_pos_->x() : 0.f,
        last_raw_target_pos_ ? last_raw_target_pos_->y() : 0.f,
        (last_target_info_.has_value() and last_raw_target_pos_)
            ? (last_target_info_->room_pos - *last_raw_target_pos_).norm() : -1.f,
        current_plan_.has_value() ? 1 : 0,
        affordance_manager.last_reject_reason(), suppressed_affordance_, cands);
    select_json_.flush();
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
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                      std::format("no planning step (line {})", 327));
        return std::nullopt;
    }

    // ── NO POSE-JUMP GATE HERE, DELIBERATELY ─────────────────────────────────────────────────────
    // The measurement stands and is worth keeping: 2172 of 122841 localiser cycles in one run (1.77%)
    // imply a speed above 1 m/s on a base that tops out near 0.6, 35 of them are steps over 0.30 m,
    // the worst 11.63 m in 50 ms — and the covariance does not see it (sigma after the six worst:
    // 1.023, 1.583, 7.164, 1.164, 0.086, 3.076). Everything downstream inherits it: cross-track of
    // 4.98 m against a path built before the jump, `start was NOT footprint-feasible` storms, a
    // ten-minute stall. See the memory note localiser-pose-jumps-sigma-blind.
    // ★WHAT WAS TRIED AND REMOVED: a gate rejecting a step above `2*v_max*dt + 3*sigma + 0.10`. Three
    // invented numbers, and a hard cutoff on a continuous quantity — the same shape as the carrot gate
    // removed just below, which stopped the robot twice in one hour.
    // ★THE FORM THIS SHOULD TAKE: the disagreement between the displacement the localiser reports and
    // the one the robot's own motion predicts IS a standard deviation, in metres, of the pose estimate
    // — and the speed limiter already consumes a pose sigma continuously. Feed it there and a robot
    // whose pose is in doubt slows and stops through the mechanism that exists, with nothing to tune
    // and no cliff to fall off. That is a real change, and it needs a bench and a watched run before
    // it goes anywhere near the robot.
    // ── THE CONTROL LAW'S OWN POSE, RESOLVED HERE AND USED ONLY BY IT ────────────────────────────
    // `robot_pose` above is SCAN-ALIGNED: read_robot_pose_in_room pins the query to the last LiDAR
    // stamp so the pose goes with the observations the obstacle set was built from. That is right for
    // the obstacle side and wrong for the tracker, whose e_y and e_psi are about where the robot IS —
    // and the cost is not hypothetical: the newest RT block already leads that scan by 33 ms at the
    // median and 68 ms at p90 (measured over 6000 cycles of overlay_lag_eval.csv), so the loop was
    // declining a fresher pose that already existed. At 0.35 m/s that is 12-24 mm of cross-track error
    // the feedback then demanded 0.034-0.067 rad/s to correct, on evidence it did not need to be
    // holding. No extrapolation and no model: this is a real RT block, just a later one.
    // ★Guarded so this cannot silently become the only pose. If the tree has nothing fresher the
    // optional stays empty and the tracker gets the scan-aligned pose — the previous behaviour exactly.
    tracker_pose_.reset();
    tracker_pose_lead_m_ = 0.f;
    if (params_ == nullptr or params_->tracker_uses_latest_pose)
        if (const auto fresh = world_model.read_robot_pose_latest(timestamp_ms); fresh.has_value())
        {
            tracker_pose_ = *fresh;
            tracker_pose_lead_m_ = (fresh->pos - robot_pose->pos).norm();
        }

    update_base_speed(*robot_pose, timestamp_ms);   // base speed for the contract stillness gate
    overlay_now_ms_ = timestamp_ms;                 // overlay dead-reckoning target + base time
    overlay_lidar_ts_ms_ = obstacle_tracker.last_lidar_timestamp_ms();
    update_overlay_extrapolation(world_model, *robot_pose, timestamp_ms, obstacle_tracker.rt_block_lead_ms(),
                                 obstacle_tracker.rt_twist_fix_dt_ms(), obstacle_tracker);

    // ── THE VELOCITY PANEL MUST SHOW WHOEVER IS DRIVING, NOT ONLY US ─────────────────────────────
    // The trace is fed from our motion commander's output loop, so it went blank the moment anything
    // else commanded the base — a joystick with this controller halted being the obvious case, and the
    // one that prompted this. robot_ref_* in the shared graph is the agreed channel for exactly that.
    // Placed HERE rather than in update_overlay_extrapolation, which returns early when there is no
    // fresh lidar stamp and so is not called every cycle — a diagnostic that reports intermittently is
    // worse than one that reports not at all, because the gaps read as the robot having stopped.
    feed_external_velocity_trace(world_model, display, timestamp_ms);

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
    // ── POST-AFFORDANCE DWELL ─────────────────────────────────────────────────────────────────────
    // Stand still and look at what was just acquired before going anywhere else. This sits BEFORE
    // read_target_in_room deliberately: that call is what runs the epistemic selection and CLAIMS the
    // winner (publishing an execution edge into the shared graph), so gating only the driving would
    // leave the graph asserting an affordance that has not begun. Returning nullopt here is the same
    // "nothing to drive to" the idle branch below handles, minus its teardown — the plan is already
    // cleared by finalize_reached, and the base is already stopped.
    // ★THE CLOCK IS THE FLOOR, THE ACQUISITION IS THE CONDITION. When the affordance went to look at a
    // named object, the dwell holds until that object's mask has come back in `affordance_dwell_mask_hits`
    // SEPARATE producer frames — one sighting is what an intermittent detector produces by accident, and
    // a count is what turns it into evidence. Bounded by affordance_dwell_max_ms so a mask that is never
    // coming ends the wait with a failure in the log instead of parking the robot.
    const int hits_wanted = params_ ? params_->affordance_dwell_mask_hits : 0;
    // Decided when the dwell was armed, from the finished affordance's CONTRACT — not from whether an
    // object happens to be named, which was true of every affordance and so made this wait universal.
    const bool wants_mask = hits_wanted > 0 and dwell_wants_mask_ and not dwell_object_.empty();
    if (affordance_dwell_until_ms_ != 0 and timestamp_ms < affordance_dwell_deadline_ms_)
        count_dwell_mask_hits();
    const bool acquisition_done = not wants_mask or dwell_mask_hits_ >= hits_wanted;
    const bool dwelling = affordance_dwell_until_ms_ != 0
                      and timestamp_ms < affordance_dwell_deadline_ms_
                      and (timestamp_ms < affordance_dwell_until_ms_ or not acquisition_done);
    if (not dwelling and affordance_dwell_until_ms_ != 0)
    {
        // Report the OUTCOME on the way out, because "held 3 s and saw it 5 times" and "held 12 s and
        // never saw it" are the two different things this wait exists to tell apart.
        if (wants_mask)
            std::println("[affordance] dwell over: '{}' mask seen {}/{} times{}",
                         dwell_object_, dwell_mask_hits_, hits_wanted,
                         acquisition_done ? "" : " — GAVE UP (bounded wait elapsed)");
        affordance_dwell_until_ms_ = 0;
        affordance_dwell_deadline_ms_ = 0;
    }
    if (dwelling)
    {
        if (not dwell_logged_)
        {
            dwell_logged_ = true;
            std::println("[affordance] DWELL after '{}' — holding still for {:.1f} s{} (masks arrive at "
                         "camera rate, not at arrival).",
                         dwell_object_.empty() ? std::string("—") : dwell_object_,
                         static_cast<double>(affordance_dwell_until_ms_ - timestamp_ms) / 1000.0,
                         wants_mask ? std::format(" and until its mask returns {}x", hits_wanted)
                                    : std::string{});
        }
        // HEARTBEAT, because a robot standing still for up to twelve seconds must be able to say what
        // it is standing still FOR. One line every two seconds naming the predicate, the node it is
        // read on and the count so far — the previous version logged once at the start and then went
        // silent, which is how a wait that could never be satisfied looked exactly like a wait that was
        // nearly over.
        if (wants_mask and timestamp_ms - dwell_last_log_ms_ >= 2000)
        {
            dwell_last_log_ms_ = timestamp_ms;
            std::println("[affordance] dwell waiting: {}/{} confirming looks of '{}' "
                         "(predicate '{}' on node {}), {:.1f} s of bound left",
                         dwell_mask_hits_, hits_wanted,
                         dwell_object_.empty() ? std::string("—") : dwell_object_,
                         dwell_goal_.empty() ? std::string("—") : dwell_goal_.front().attr,
                         dwell_feedback_node_,
                         static_cast<double>(affordance_dwell_deadline_ms_ - timestamp_ms) / 1000.0);
        }
        update_display(robot_pose,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        // THE PANEL HAS TO KEEP TICKING THROUGH THE DWELL. Everything that publishes the affordance
        // view lives past this early return, so without republishing here the window would freeze on
        // the last cycle before the dwell — showing a countdown that never counts and a highlight
        // aimed at whatever was live before. A viewer that stops updating during the ONE interval it
        // exists to cover is worse than no viewer, because it looks like it is still reporting.
        attention_object_ = dwell_object_;
        // The finished affordance's standpoint is where the robot is STANDING right now — which is
        // exactly what makes the dwell readable: the marker is underfoot, not ahead.
        attention_standpoint_ = current_standpoint();
        affordance_view_.active = false;              // it FINISHED; the dwell is not execution
        affordance_view_.dwell_left_s = dwell_left_s(timestamp_ms);
        affordance_view_.dwell_mask_hits = dwell_mask_hits_;
        affordance_view_.dwell_mask_needed = wants_mask ? hits_wanted : 0;
        affordance_view_.suppressed = suppressed_affordance_;
        display.set_affordance_execution(affordance_view_);
        stop(path_controller, motion_commander);
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                      std::format("no planning step (line {})", 505));
        return std::nullopt;
    }
    dwell_logged_ = false;

    // ── ACT ON A PENDING REJECTION, BEFORE THE NEXT SELECTION ────────────────────────────────────
    // Set by begin_escape after kMaxEscapesPerAffordance wedges against one standpoint. Done here
    // because this is where the manager is reachable, and it must land BEFORE read_target_in_room or
    // the very next selection hands the same unreachable affordance straight back.
    if (reject_affordance_id_ != 0)
    {
        std::println("[controller] ABANDONING '{}' after {} escapes without reaching it — releasing the "
                     "claim so another affordance can win. An affordance the body cannot reach advertises "
                     "gain it cannot deliver; grinding at it serves nothing.",
                     reject_affordance_name_, escapes_on_target_);
        affordance_manager.suppress_target(graph_, reject_affordance_id_, kUnreachableRounds);
        affordance_manager.release_execution_claim(graph_);
        // Remember the spot as PUBLISHED, so the same proposal is recognised next time it arrives.
        if (last_raw_target_pos_.has_value())
            remember_useless_spot(*last_raw_target_pos_, reject_affordance_name_, timestamp_ms);
        reject_affordance_id_ = 0;
        reject_affordance_name_.clear();
        escapes_on_target_ = 0;
        escapes_target_id_ = 0;
        // Drop every trace of the abandoned goal so the next selection starts clean rather than
        // re-planning to the pose we just gave up on.
        last_target_info_.reset();
        current_target_room_.reset();
        active_target_id_ = 0;
        current_plan_.reset();
        plan_spline_valid_ = false;
    }

    const auto target = rc::uses_mission(mission_.mode()) or mission_.mode() == rc::DriveMode::Target
                            ? std::nullopt
                            : world_model.read_target_in_room(timestamp_ms);
    // Read straight after the selection that produced it — it is a per-call report, and any later read
    // would be describing a different cycle's contest.
    // The planner grid's identity, stamped before selection reads it: a verdict this side reached
    // from the map is remembered against that map, and lapses when it is rebuilt differently.
    affordance_manager.set_map_identity(grid_planner_.world_hash());
    suppressed_affordance_ = affordance_manager.suppressed_name();
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
        // ★PUBLISH THE IDLE STATE. This branch returned without touching the affordance view, so the
        // panel kept the LAST thing it was told — the final cycle of the dwell — and went on claiming
        // "DWELL 0.0 s, holding still until the acquisition is confirmed" while the robot was in fact
        // idle with nothing eligible to do. A window that freezes on its last frame does not merely
        // stop informing you, it actively misreports, and it misreports the state you most need to
        // recognise: stuck.
        affordance_view_.active = false;
        affordance_view_.dwell_left_s = 0.f;
        affordance_view_.dwell_mask_needed = 0;
        affordance_view_.suppressed = suppressed_affordance_;
        affordance_view_.phase = "idle — no eligible affordance";
        display.set_affordance_execution(affordance_view_);
        attention_object_.clear();
        attention_standpoint_.reset();
        update_display(robot_pose,
                       display,
                       obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        stop(path_controller, motion_commander);
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                      std::format("no planning step (line {})", 592));
        return std::nullopt;
    }

    target_wait_logged_ = false;

    // ★Record the selection outcome BEFORE any early return below can skip it.
    log_selection_json(timestamp_ms, affordance_manager,
                       std::optional<Eigen::Vector2f>{robot_pose->pos.head<2>().cast<float>()},
                       target.has_value() ? "selected" : "no-target");

    // ★★★SAY SO WHEN WE STOP EXECUTING. The consumer owns the fact "am I executing this"; the producer
    // must not have to INFER it from a timeout. Measured 2026-08-19: 39% of cycles read Executing on the
    // wire while the consumer held no target and no plan — it had dropped both (no route) and left the
    // claim standing, so room waited on an affordance nobody was driving. The execution lease does
    // recover it, but at 45 s — a guessed constant that converts a stall into a shorter stall instead of
    // preventing one. Releasing is free, immediate, and needs no timer, which puts the lease back to
    // being a backstop for a CRASHED consumer rather than the primary recovery path.
    // ★release_execution_claim is idempotent: it clears `active` only when the node really reads
    // Executing, so calling it on any targetless cycle is safe and does nothing the rest of the time.
    if (not target.has_value() and not current_plan_.has_value() and graph_)
        if (affordance_manager.release_execution_claim(graph_))
        {
            std::println("[affordance] no target and no plan — releasing the execution claim so the "
                         "producer can offer again (it should not have to wait out a lease).");
            std::fflush(stdout);
        }

    // ── A SPOT WE ALREADY FAILED AT, RECOGNISED BEFORE DRIVING TO IT AGAIN ───────────────────────
    // The producer does not know the approach failed — nothing tells it — so it re-publishes the same
    // standpoint as soon as the suppression lapses, and without this we would re-learn it the expensive
    // way every time: drive over, wedge three times, abandon, repeat. Matched on the RAW published pose,
    // because that is what the producer will send again; the repaired pose is ours, not theirs.
    if (target->from_affordance)
        if (const auto *spot = known_useless_spot(target->room_pos, timestamp_ms); spot != nullptr)
        {
            // Rate-limited: the producer re-offers continuously, and one line per cycle would bury the run.
            if (timestamp_ms - last_useless_log_ms_ >= 5000)
            {
                last_useless_log_ms_ = timestamp_ms;
                std::println("[controller] '{}' is offering ({:.2f},{:.2f}) again — the robot already failed "
                             "to reach that spot {}x. Skipping it without driving there.",
                             target->node_name, target->room_pos.x(), target->room_pos.y(), spot->hits);
            }
            // ★★★AND SAY SO ON THE WIRE. Skipping in silence is exactly the failure the fact channel
            // was built to end: the producer offers, the consumer quietly declines, and both wait —
            // "no affordance arriving", with the base stopped and nothing in any log to say why. The
            // spot is remembered for kUselessSpotMemoryMs (120 s), so a silent skip could hold the pair
            // for two minutes on a cell room believed was still on offer. Reporting it costs nothing
            // and lets room de-prioritise this cell and pick another on its next cycle.
            // ★ONCE PER (cell, pose, map), like every other verdict this side reaches. Making the
            // silent skip SPEAK was right; making it speak twelve times a second was not — measured
            // 2026-08-20: 379 identical reports for one cell in five minutes, p50 0.1 s apart, each
            // one a completion the producer consumed and re-offered. Recording the verdict makes the
            // SELECTOR skip this candidate until the robot moves or the map changes, so the producer
            // is told exactly once and nothing has to be re-decided. No timer: that would be the
            // eleventh, and this file spent the day deleting them.
            affordance_manager.note_map_verdict(target->room_pos, robot_pose->pos.head<2>().cast<float>());
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          "-> Unreachable: no route from here to that pose");
            affordance_manager.mark_reached(graph_, rc::affordance::Outcome::Unreachable);
            audit_standpoint("fact-unreachable", timestamp_ms, target->room_pos, target->room_pos,
                             robot_pose->pos.head<2>().cast<float>(), "known-useless",
                             std::format("failed here {}x before", spot->hits), -1, -1, 0);
            current_plan_.reset();
            plan_spline_valid_ = false;
            stop(path_controller, motion_commander);
            update_display(robot_pose, display, obstacle_tracker.display_obstacle_polygons(),
                           obstacle_tracker.temporary_obstacle_rfe_points(),
                           params_ ? params_->max_lidar_draw_points : 0);
            return std::nullopt;   // next cycle the selector picks something else
        }

    step.target = *target;
    // Remembered for the rejection path: what the PRODUCER published, before our repair moves it.
    last_raw_target_pos_ = target->room_pos;
    // ★HOLD THE STANDPOINT WE COMMITTED TO. Runs HERE — on the raw published pose, before the
    // reachability repair, the boxed-in search and the close-in re-check — so those still act on
    // whatever we are actually driving to. See ApproachCommitment in the header for the measurement.
    hold_approach_commitment(step, timestamp_ms);
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
    const auto same_5cm = [](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
    { return (a - b).squaredNorm() < 0.05f * 0.05f; };
    const bool routing_failed_here =
        unroutable_at_.has_value()
        and same_5cm(unroutable_at_->goal, step.target.room_pos)
        and same_5cm(unroutable_at_->robot, step.robot_pose.pos.head<2>().cast<float>())
        and unroutable_at_->map == grid_planner_.world_hash();
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

    // ── THE CELL IS THE PRODUCER'S, SO WE REPORT FACTS AND EDIT NOTHING ─────────────────────────
    // ★★★NO REPAIR OF AN AFFORDANCE STANDPOINT (2026-08-19). room_concept chose that cell by
    // maximising expected information gain over its own grid — visibility, yaw, what the vantage can
    // see. This process knows none of that and cannot recompute it; all it knows is reachability. So
    // moving the cell "a little" is not a small correction, it is answering a question we were not
    // asked: half a metre can put a wall corner between the camera and the thing the gain was computed
    // from, and the producer is never told the pose it reasoned about is not the pose that happened.
    // ★The repair existed because the wire had no way to say "I cannot". It does now — Infeasible and
    // Unreachable are FACTS this side alone can establish — so the honest answer replaces the edit.
    // Measured on the loop this deletes: 208 nearest_reachable repairs, median displacement 1.80 m,
    // 139 of them landing within 0.21 m of the ROBOT, and 140 of 163 accepted arrivals a median 2.86 m
    // from the published cell, every one reported to room as SATISFIED.
    // ★A CLICKED TARGET IS NOT A PRODUCER'S DECISION VARIABLE — it is this agent's own goal, so the
    // repair below still applies to it. Ownership is the discriminator, not the geometry.
    // ── AN ORIENT AFFORDANCE DOES NOT NAVIGATE, SO EVERY QUESTION BELOW IS ABOUT THE WRONG POSE ──
    // Policy::Orient is "rotate in place toward the bearing": the executor stops the follower and turns
    // where the robot STANDS (step_orient), and the published (x,y) is never driven to. So resolving
    // that (x,y) into a standpoint, routing to it, and reporting outside_room / infeasible /
    // unreachable about it describe a place the robot was never going — facts that are true of a point
    // and false of the action.
    // ★AND THE ONE QUESTION THAT DOES MATTER WAS NEVER ASKED. A rotation in place sweeps the body's
    // DIAGONAL (0.796 m for this 0.46 x 0.65 base, against 0.46 m standing), and it runs with NO
    // obstacle check of any kind — the align branch returns ahead of every safety stage. Whether that
    // sweep fits is a fact about where the robot IS, this side is the only one that can establish it,
    // and the answer is reported rather than repaired: there is nowhere to move the target TO, because
    // the target is not a place. A producer told `infeasible` here knows to wait until the robot has
    // been carried somewhere with room, which is the whole basis of an opportunistic manoeuvre.
    const bool orient_in_place = step.target.from_affordance and target_contract_known_
                             and target_contract_.policy == rc::affordance::Policy::Orient;
    // ★THE CONSUMER'S READING OF THE CONTRACT, ON DISK. Whether it knows this is an Orient decides
    // every branch below, and from outside the two failures look identical: a robot that will not
    // turn. Deduplicated by note_protocol, so this is one line per change of reading, not per cycle.
    if (step.target.from_affordance)
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
            std::format("reading '{}': contract {}, policy {}", step.target.node_name,
                        target_contract_known_ ? "known" : "NOT KNOWN",
                        target_contract_known_
                            ? (target_contract_.policy == rc::affordance::Policy::Orient  ? "Orient"
                             : target_contract_.policy == rc::affordance::Policy::Servo   ? "Servo"
                                                                                          : "Reach")
                            : "-"));
    if (orient_in_place and grid_planner_.has_world())
    {
        const auto robot_xy = step.robot_pose.pos.head<2>().cast<float>();
        if (const auto blk = grid_planner_.why_cannot_turn(robot_xy); blk.blocked)
        {
            if (timestamp_ms - last_repair_reject_log_ms_ >= 3000)
            {
                last_repair_reject_log_ms_ = timestamp_ms;
                std::println("[controller] '{}' is an orient affordance and the body cannot turn "
                             "all the way round at ({:.2f},{:.2f}), where the robot stands — "
                             "reporting infeasible. Nothing to repair: an orient has no standpoint.",
                             step.target.node_name, robot_xy.x(), robot_xy.y());
                std::println("[controller]   because: {}/{} headings blocked by {} cell(s){}; nearest "
                             "at ({:.2f},{:.2f}) = {:.2f} m, bearing {:.0f} deg",
                             blk.headings_blocked, 8, blk.cells_blocked,
                             blk.off_map ? ", footprint leaves the map" : "",
                             blk.nearest_cell.x(), blk.nearest_cell.y(), blk.nearest_m,
                             blk.nearest_bearing_deg);
            }
            // ★THE SHAPE OF THE OBSTRUCTION, NOT JUST ITS EXISTENCE. A refusal the producer must
            // believe and wait on deserves better than a verdict: one stray cell 15 cm from the body
            // centre and a wall across three headings are the same word from outside and call for
            // opposite responses. Recorded on the audit line so it survives the terminal.
            audit_standpoint("fact-orient-cannot-turn", timestamp_ms,
                             last_raw_target_pos_.value_or(step.target.room_pos), robot_xy, robot_xy,
                             "can_turn_here",
                             std::format("{}/8 headings blocked by {} cell(s){}; nearest {:.2f} m at "
                                         "bearing {:.0f} deg, cell ({:.2f},{:.2f})",
                                         blk.headings_blocked, blk.cells_blocked,
                                         blk.off_map ? ", off-map" : "", blk.nearest_m,
                                         blk.nearest_bearing_deg, blk.nearest_cell.x(),
                                         blk.nearest_cell.y()),
                             -1, -1, 0);
            affordance_manager.note_map_verdict(robot_xy, robot_xy);
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          "-> Infeasible: the body cannot sweep its diagonal where it stands");
            affordance_manager.mark_reached(graph_, rc::affordance::Outcome::Infeasible);
            current_plan_.reset();
            plan_spline_valid_ = false;
            last_repair_applied_m_ = 0.f;
            stop(path_controller, motion_commander);
            update_display(robot_pose, display, obstacle_tracker.display_obstacle_polygons(),
                           obstacle_tracker.temporary_obstacle_rfe_points(),
                           params_ ? params_->max_lidar_draw_points : 0);
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          std::format("no planning step (line {})", 822));
            return std::nullopt;
        }
        // Turnable here: there is no standpoint to resolve and no route worth repairing, so the whole
        // block below is skipped rather than run against a pose nothing will drive to.
    }
    else if (step.target.from_affordance and grid_planner_.has_world())
    {
        // ★has_world() FIRST. pose_free answers false for every pose before the grid is rasterised, so
        // without it the first cycles of a run would report an infeasible standpoint when the only true
        // statement is "I do not know yet" — a fact channel that lies on startup is worse than none.
        // ── WHAT THE PRODUCER ACTUALLY DECIDED IS A CELL, NOT A POINT ───────────────────────────
        // ★★★MEASURED IMMEDIATELY AFTER DELETING THE REPAIR OUTRIGHT (2026-08-19, 22:21): the pair
        // went honest and IDLE — cell after cell reported infeasible, the base stopped for ~20 s at a
        // time. Deleting the repair was half right. room's exploration grid has 0.5 m cells
        // (EpistemicPlanner::ior_cell_size) and it publishes the cell CENTRE; the information gain was
        // computed for the CELL. So moving inside that cell is not editing anyone's decision, it is
        // resolving a cell into a pose the body fits in — which is this side's job and nobody else's.
        // Moving OUTSIDE it picks a different cell, which is the producer's decision and not ours.
        // ★THE BUDGET IS THE CELL, NOT A TUNING KNOB: half a cell diagonal is the furthest a point can
        // be from the centre and still be in the same cell. Nothing to tune; if room changes its grid
        // this must follow, which is why the number is derived here from the cell size and named.
        target_is_approach_only_ = false;
        // ── 1. IS IT EVEN IN THE ROOM? ──────────────────────────────────────────────────────────
        // ★THE FIRST QUESTION, BEFORE ANY GEOMETRY. A standpoint outside the room polygon cannot be
        // stood at, cannot be routed to, and cannot be repaired into one that can — every later test
        // would answer "no" for a reason that mis-describes what happened, and the producer would be
        // told "the body does not fit" about a place that is not in its own layout. Rejected with its
        // own word so a run full of these points at the exploration grid's extent, not at the planner.
        if (room_polygon_.size() >= 3 and not rc::corner_visibility::point_in_polygon(step.target.room_pos, room_polygon_))
        {
            const Eigen::Vector2f cell = step.target.room_pos;
            if (timestamp_ms - last_repair_reject_log_ms_ >= 3000)
            {
                last_repair_reject_log_ms_ = timestamp_ms;
                std::println("[controller] '{}' at ({:.2f},{:.2f}) is OUTSIDE the room polygon — "
                             "reporting outside_room. Not a navigation failure: that cell is not in "
                             "the layout the producer itself published.",
                             step.target.node_name, cell.x(), cell.y());
            }
            audit_standpoint("fact-outside-room", timestamp_ms, last_raw_target_pos_.value_or(cell),
                             cell, step.robot_pose.pos.head<2>().cast<float>(), "room-polygon",
                             "outside the room layout", -1, -1, 0);
            affordance_manager.note_map_verdict(cell, step.robot_pose.pos.head<2>().cast<float>());
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          "-> OutsideRoom: that pose is not inside the producer's own layout");
            affordance_manager.mark_reached(graph_, rc::affordance::Outcome::OutsideRoom);
            current_plan_.reset();
            plan_spline_valid_ = false;
            last_repair_applied_m_ = 0.f;
            stop(path_controller, motion_commander);
            update_display(robot_pose, display, obstacle_tracker.display_obstacle_polygons(),
                           obstacle_tracker.temporary_obstacle_rfe_points(),
                           params_ ? params_->max_lidar_draw_points : 0);
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          std::format("no planning step (line {})", 875));
            return std::nullopt;
        }
        const float producer_cell_m = params_ ? params_->producer_cell_size_m : 0.5f;
        const float in_cell_m = 0.5f * std::numbers::sqrt2_v<float> * producer_cell_m;
        std::optional<rc::affordance::Outcome> fact;
        std::string why;
        std::optional<Eigen::Vector2f> in_cell_pose;

        // ── REUSE THE STANDPOINT WE ALREADY RESOLVED FOR THIS CELL ──────────────────────────────
        // Held for the life of the target and revalidated cheaply — see resolved_standpoint_. Without
        // this the search re-answers every cycle against a grid that flickers, the target walks, and
        // every step of it is a replan the tracker has to start over from.
        const bool same_cell = (step.target.room_pos - resolved_for_cell_).squaredNorm() < 0.05f * 0.05f;
        if (not same_cell) resolved_standpoint_.reset();
        if (resolved_standpoint_.has_value())
        {
            const bool still_good = grid_planner_.pose_free(*resolved_standpoint_, step.target.yaw_rad)
                and (not will_rotate_here or grid_planner_.can_turn_here(*resolved_standpoint_));
            if (still_good) in_cell_pose = resolved_standpoint_;
            else            resolved_standpoint_.reset();
        }
        if (not in_cell_pose.has_value() and will_rotate_here)
        {
            in_cell_pose = grid_planner_.nearest_rotatable(step.target.room_pos, in_cell_m);
            if (not in_cell_pose.has_value())
                in_cell_pose = grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad, in_cell_m);
        }
        else if (not in_cell_pose.has_value())
            in_cell_pose = grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad, in_cell_m);
        // ★THE RING SEARCH IS QUANTISED TO THE PLANNER'S 0.10 m GRID, so `max_radius_m` bounds the
        // search, not the answer: measured p50 0.403 m and max 0.455 m against a 0.354 m half-diagonal,
        // i.e. up to one planner cell outside the producer's cell. A bound that is checked only inside
        // the search is a bound in name; enforce it on the RESULT, which is the thing being claimed.
        if (in_cell_pose.has_value()
            and (*in_cell_pose - step.target.room_pos).norm() > in_cell_m)
            in_cell_pose.reset();
        // ── 2b. NOTHING IN THE CELL FITS → TAKE THE CLOSEST CLEAR SPOT ANYWHERE ─────────────────
        // ★The cell is the producer's preference and is tried first, so a standpoint that merely needs
        // a few centimetres keeps its gain intact. But refusing outright when the whole cell is
        // occupied leaves the robot idle beside a perfectly good vantage half a metre further out, so
        // widen the search rather than give up — and RECORD how far it went, because the arrival test
        // and the audit both need to know this is no longer the cell that was asked for.
        if (not in_cell_pose.has_value())
        {
            in_cell_pose = will_rotate_here
                             ? grid_planner_.nearest_rotatable(step.target.room_pos)
                             : grid_planner_.nearest_free(step.target.room_pos, step.target.yaw_rad);
            // ★THROTTLED LIKE THE OTHERS: the resolution is deterministic and re-derived at 20 Hz, so
            // one target produced 355 identical rows in 40 s. One row per changed answer.
            static std::uint64_t last_widen_ms = 0;
            static Eigen::Vector2f last_widen{std::numeric_limits<float>::infinity(),
                                              std::numeric_limits<float>::infinity()};
            const bool widen_is_news = in_cell_pose.has_value()
                and (timestamp_ms - last_widen_ms >= 500 or (*in_cell_pose - last_widen).squaredNorm() > 1e-4f);
            if (widen_is_news) { last_widen_ms = timestamp_ms; last_widen = *in_cell_pose; }
            if (widen_is_news)
                audit_standpoint("widened", timestamp_ms, last_raw_target_pos_.value_or(step.target.room_pos),
                                 *in_cell_pose, step.robot_pose.pos.head<2>().cast<float>(),
                                 will_rotate_here ? "nearest_rotatable" : "nearest_free",
                                 std::format("whole {:.2f} m cell occupied; closest clear spot is {:.2f} m out",
                                             producer_cell_m, (*in_cell_pose - step.target.room_pos).norm()),
                                 -1, -1, -1);
        }
        if (not in_cell_pose.has_value())
        { fact = rc::affordance::Outcome::Infeasible;
          why = std::format("no pose the body fits in anywhere within reach of the {:.2f} m cell",
                            producer_cell_m); }
        else if (routing_failed_here)
        {
            // ── GO AS CLOSE AS THE MAP ALLOWS — AND SAY THAT IS WHAT HAPPENED ───────────────────
            // ★Driving to the closest reachable pose is RIGHT: it makes progress, it changes what the
            // sensors can see, and a cell that is unroutable from here is often routable from there.
            // What was wrong was never the drive — it was reporting the arrival as SATISFIED, which
            // told room its cell had been observed when the robot had stopped metres short (140 of 163
            // arrivals, median 2.86 m). So: still approach, but the target is flagged an APPROACH and
            // the completion says `unreachable`, so nothing about room's cell is marked observed.
            // ★If the closest reachable pose is where the robot already stands, there is no approach to
            // make — that is the sealed-pocket case — and the fact goes out immediately instead of
            // driving zero metres and calling it an attempt.
            if (not unroutable_fix_.has_value())
                unroutable_fix_ = grid_planner_.nearest_reachable(step.plan_origin, step.target.room_pos);
            const float band = path_controller.goal_threshold() + approach_body_.circumscribed_radius();
            const auto robot_xy = step.robot_pose.pos.head<2>().cast<float>();
            if (unroutable_fix_.has_value()
                and (*unroutable_fix_ - robot_xy).norm() > band
                and (*unroutable_fix_ - step.target.room_pos).norm()
                        < (robot_xy - step.target.room_pos).norm())
            {
                audit_standpoint("approach", timestamp_ms, last_raw_target_pos_.value_or(step.target.room_pos),
                                 *unroutable_fix_, robot_xy, "nearest_reachable",
                                 "closest reachable — will report unreachable on arrival", -1, -1, -1);
                step.target.room_pos = *unroutable_fix_;
                target_is_approach_only_ = true;
                in_cell_pose = step.target.room_pos;     // already feasible by construction
            }
            else
            {
                fact = rc::affordance::Outcome::Unreachable;
                why = "no route from here, and no closer pose the robot can reach ["
                    + grid_planner_.last_failure() + "]";
            }
        }
        if (fact.has_value())
        {
            const Eigen::Vector2f cell = step.target.room_pos;
            if (timestamp_ms - last_repair_reject_log_ms_ >= 3000)
            {
                last_repair_reject_log_ms_ = timestamp_ms;
                std::println("[controller] '{}' at ({:.2f},{:.2f}): {} — reporting {} to the producer. "
                             "Not moving the standpoint: the cell is room's to choose.",
                             step.target.node_name, cell.x(), cell.y(), why,
                             rc::affordance::to_string(*fact));
            }
            // Remembered against (cell, pose, map) so it is decided ONCE from here — see
            // AffordanceManager::note_map_verdict for the two rate limits this replaces.
            affordance_manager.note_map_verdict(cell, step.robot_pose.pos.head<2>().cast<float>());
            audit_standpoint(*fact == rc::affordance::Outcome::Infeasible ? "fact-infeasible"
                                                                         : "fact-unreachable",
                             timestamp_ms, last_raw_target_pos_.value_or(cell), cell,
                             step.robot_pose.pos.head<2>().cast<float>(),
                             "no-repair", why, -1, -1, 0);
            // ★WHEN THE FREE SPACE IS DISCONNECTED, DUMP THE WORLD THAT DISCONNECTED IT. "no route:
            // 78106 expansions over 16246 free cells" says the search exhausted a large free space and
            // still could not reach a cell 1.60 m away — the robot is in a POCKET, and which polygon
            // seals it is not something to guess at. One snapshot per 20 s: room polygon, every
            // obstacle the planner rasterised, the live LiDAR points, and the start/goal pair.
            if (*fact == rc::affordance::Outcome::Unreachable
                and timestamp_ms - last_pocket_dump_ms_ >= 20000)
            {
                last_pocket_dump_ms_ = timestamp_ms;
                std::ofstream f("unreachable_world.txt", std::ios::out | std::ios::trunc);
                if (f.is_open())
                {
                    f.imbue(std::locale::classic());     // es_ES writes commas; tools read points
                    f << std::setprecision(9);
                    f << "# the world the planner could not cross. " << why << "\n";
                    f << "t_ms " << timestamp_ms << '\n';
                    const auto r = step.robot_pose.pos.head<2>().cast<float>();
                    f << "start " << r.x() << ' ' << r.y() << '\n';
                    f << "goal " << cell.x() << ' ' << cell.y() << '\n';
                    for (const auto &p : room_polygon_) f << "room " << p.x() << ' ' << p.y() << '\n';
                    int i = 0;
                    for (const auto &o : obstacle_tracker.display_obstacle_polygons())
                    {
                        // The LABEL and the node name go with the polygon: "which polygon seals the
                        // pocket" is only actionable if it says WHOSE it is (a table's box, a residual
                        // hull, a temporary LiDAR blocker this agent dropped itself).
                        for (const auto &p : o.polygon)
                            f << "obs " << i << ' ' << p.x() << ' ' << p.y() << ' '
                              << (o.label.empty() ? "-" : o.label) << ' '
                              << (o.color_key.empty() ? "-" : o.color_key) << '\n';
                        ++i;
                    }
                    // temporary_obstacle_rfe_points() returns POLYGONS (the virtual blockers this agent
                    // drops when a path is blocked), not points — the name says points, the type says
                    // otherwise, and the compiler is the only reason that did not become another
                    // "a name is a claim" entry. Written as polygons, which is what they are.
                    int k = 0;
                    for (const auto &poly : obstacle_tracker.temporary_obstacle_rfe_points())
                    {
                        for (const auto &p : poly) f << "rfe " << k << ' ' << p.x() << ' ' << p.y() << '\n';
                        ++k;
                    }
                    // ★AND THE RESIDUAL CELLS THEMSELVES — what the planner now marks directly, and
                    // the only way tools/replay_world can tell a world that FLICKERS from one that is
                    // genuinely blocked. Replaying the polygon-only dump found a clean 4.03 m route
                    // through a world the live planner had just called unroutable, which says the two
                    // were different frames, not different answers.
                    f << "cell_size " << obstacle_tracker.residual_cell_size_m() << '\n';
                    for (const auto &c : obstacle_tracker.residual_cells())
                        f << "cell " << c.x() << ' ' << c.y() << '\n';
                }
            }
            // ★NOT REMEMBERED LOCALLY. The 120 s useless-spot memory exists for a DRIVE failure — the
            // robot went there and wedged — where re-driving is expensive and the evidence is not
            // otherwise recoverable. A PLANNING failure is not that: the planner re-answers it in a few
            // milliseconds on every offer, so a local veto adds nothing except a second authority with
            // its own clock, holding a cell for two minutes while room's own attempt suppressor is
            // already decaying on a different one. That is the "two authorities, different timers"
            // pattern that has caused most of the stalls in this pair; one guessed constant is enough.
            // ★THE FACT GOES ON THE WIRE, AND THE CLAIM COMES OFF WITH IT. mark_reached writes the
            // outcome and clears epistemic_pending/active in ONE node write, so the producer cannot
            // observe the release and read the previous outcome. It then de-prioritises the cell
            // itself through note_attempt/attempt_suppressor — a decaying score term, not a blacklist,
            // and NOT a stamp in its visit grid: nothing was observed, so nothing about the belief
            // may change. That is rule 5, and it is the producer's to apply, not ours.
            // ★We deliberately do NOT call suppress_target here: writing epistemic_refused into the
            // producer's node is the same ownership inversion in the opposite direction — the consumer
            // deciding what the producer may offer.
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          std::format("-> {}", rc::affordance::to_string(*fact)));
            affordance_manager.mark_reached(graph_, *fact);
            current_plan_.reset();
            plan_spline_valid_ = false;
            last_repair_applied_m_ = 0.f;
            stop(path_controller, motion_commander);
            update_display(robot_pose, display, obstacle_tracker.display_obstacle_polygons(),
                           obstacle_tracker.temporary_obstacle_rfe_points(),
                           params_ ? params_->max_lidar_draw_points : 0);
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                          std::format("no planning step (line {})", 1074));
            return std::nullopt;
        }
        // Feasible somewhere in the producer's cell: take that pose and drive. Recorded as the repair
        // it is, so the arrival test's allowance (last_repair_applied_m_) stays exact.
        if (in_cell_pose.has_value())
        {
            resolved_standpoint_ = in_cell_pose;          // hold it: the next cycle must not re-answer
            resolved_for_cell_ = step.target.room_pos;
            last_repair_applied_m_ = (*in_cell_pose - step.target.room_pos).norm();
            if (last_repair_applied_m_ > 1e-3f)
            {
                // ★ONE ROW WHEN THE ANSWER CHANGES, not one per cycle: the resolution is deterministic
                // and the robot re-derives it at 20 Hz, which produced 692 identical rows in 40 s.
                static std::uint64_t last_ic_ms = 0;
                static Eigen::Vector2f last_ic{std::numeric_limits<float>::infinity(),
                                               std::numeric_limits<float>::infinity()};
                if (timestamp_ms - last_ic_ms >= 500 or (*in_cell_pose - last_ic).squaredNorm() > 1e-4f)
                {
                    last_ic_ms = timestamp_ms; last_ic = *in_cell_pose;
                    audit_standpoint("in-cell", timestamp_ms, last_raw_target_pos_.value_or(step.target.room_pos),
                                     *in_cell_pose, step.robot_pose.pos.head<2>().cast<float>(),
                                     will_rotate_here ? "nearest_rotatable" : "nearest_free",
                                     "within the producer's cell", -1, -1, -1);
                }
                step.target.room_pos = *in_cell_pose;
                if (step.target.parent_node_id != 0)
                    if (const auto obj = world_model.read_node_room_xy(step.target.parent_node_id, timestamp_ms);
                        obj.has_value())
                        step.target.yaw_rad = std::atan2(obj->y() - step.target.room_pos.y(),
                                                         obj->x() - step.target.room_pos.x());
            }
            recheck_standpoint_on_approach(step, world_model, obstacle_tracker, path_controller, timestamp_ms);
            current_target_room_ = step.target.room_pos;
            step.target_changed = !last_target_info_.has_value()
                               || !ControllerWorldModel::same_target_instance(*last_target_info_, step.target);
            last_target_info_ = step.target;
            active_target_id_ = target->node_id;
            target_is_new_ = step.target_changed;
            return step;
        }
    }
    // From here down the repair applies to targets this agent owns (a click, a mission waypoint) and,
    // until the grid exists, to an affordance too — with the budget below still bounding it.
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
    // ★AUDIT THE REPAIR EVERY TIME IT IS ASKED, not only when its answer changes. Throttled to 2 Hz
    // and to real movement, so a steady state costs a row every half second instead of twenty.
    {
        static std::uint64_t last_audit_ms = 0;
        static Eigen::Vector2f last_audit_tgt{std::numeric_limits<float>::infinity(),
                                              std::numeric_limits<float>::infinity()};
        const Eigen::Vector2f repaired = safe.value_or(step.target.room_pos);
        if (timestamp_ms - last_audit_ms >= 500
            or (repaired - last_audit_tgt).squaredNorm() > 1e-4f)
        {
            last_audit_ms = timestamp_ms;
            last_audit_tgt = repaired;
            audit_standpoint("repair", timestamp_ms,
                             last_raw_target_pos_.value_or(step.target.room_pos),
                             repaired, step.robot_pose.pos.head<2>().cast<float>(),
                             routing_failed_here ? "nearest_reachable"
                                                 : will_rotate_here ? "nearest_rotatable" : "nearest_free",
                             safe.has_value() ? grid_planner_.last_failure() : std::string("no-repair-found"),
                             -1, -1, -1);
        }
    }
    // ── A REPAIR IS A CORRECTION, NOT A SUBSTITUTION ────────────────────────────────────────────
    // ★★★MEASURED 2026-08-19: 208 repairs went through nearest_reachable with a MEDIAN displacement of
    // 1.80 m (max 3.54 m), and 139 of them pulled the standpoint from a median 2.86 m away to within
    // 0.21 m of the ROBOT. That is not repairing a viewpoint, it is answering a different question:
    // "the cell you asked for is unreachable, so here is where you already are". The robot then passes
    // the arrival test without moving, reports SATISFIED, and the producer stamps a cell visited that
    // was never observed — 20 completions a minute with the base commanded to zero.
    // ★THE BUDGET IS THE ARRIVAL BAND ITSELF, no new constant: a repair the robot could not tell apart
    // from an arrival is one the PRODUCER cannot tell apart from its own cell. Beyond it the honest
    // answer is the one this function already gives for a boxed-in target — say so, suppress it for a
    // few rounds so the producer's next offer gets a turn, and drive nowhere.
    // ★Suppression here FAILS OPEN by construction: unreachable_rounds_ expires on its own, and the
    // no-two-in-a-row yield in AffordanceManager still hands the affordance back when it is the only
    // one on offer. The rule may delay a standpoint; it can never silence the channel.
    const float repair_budget = path_controller.goal_threshold() + approach_body_.circumscribed_radius();
    const Eigen::Vector2f published = last_raw_target_pos_.value_or(step.target.room_pos);
    last_repair_applied_m_ = 0.f;
    if (safe.has_value() and (*safe - published).norm() > repair_budget)
    {
        if (timestamp_ms - last_repair_reject_log_ms_ >= 3000)
        {
            last_repair_reject_log_ms_ = timestamp_ms;
            std::println("[controller] '{}' at ({:.2f},{:.2f}) is NOT REACHABLE — the closest reachable pose "
                         "({:.2f},{:.2f}) is {:.2f} m away, past the {:.2f} m repair budget, and {:.2f} m from "
                         "the robot. Driving there would be arriving somewhere else. Reporting unreachable.",
                         step.target.node_name, published.x(), published.y(), safe->x(), safe->y(),
                         (*safe - published).norm(), repair_budget,
                         (*safe - step.robot_pose.pos.head<2>().cast<float>()).norm());
        }
        audit_standpoint("repair-rejected", timestamp_ms, published, *safe,
                         step.robot_pose.pos.head<2>().cast<float>(),
                         routing_failed_here ? "nearest_reachable"
                                             : will_rotate_here ? "nearest_rotatable" : "nearest_free",
                         "over-budget", -1, -1, -1);
        // Remembered on the PUBLISHED pose, because that is what the producer will offer again.
        remember_useless_spot(published, step.target.node_name, timestamp_ms);
        affordance_manager.suppress_target(graph_, target->node_id, kUnreachableRounds);
        current_plan_.reset();
        plan_spline_valid_ = false;
        stop(path_controller, motion_commander);
        update_display(robot_pose, display, obstacle_tracker.display_obstacle_polygons(),
                       obstacle_tracker.temporary_obstacle_rfe_points(),
                       params_ ? params_->max_lidar_draw_points : 0);
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                      std::format("no planning step (line {})", 1192));
        return std::nullopt;
    }
    if (safe.has_value() && (*safe - step.target.room_pos).squaredNorm() > 1e-6f)
    {
        last_repair_applied_m_ = (*safe - step.target.room_pos).norm();
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

    // ── AND ONE MORE TIME, CLOSE IN, WITH EVIDENCE THE GRID DOES NOT CARRY ───────────────────────
    // Everything above this line reads the planner grid, and the grid forgets: residual_concept's hulls
    // decay, so a standpoint inside a real object reads free and keeps reading free no matter how often
    // it is re-tested. This asks the last metres against the live return cloud instead. It runs AFTER
    // the repair on purpose — the pose it must judge is the one the robot is actually driving to, not
    // the one the producer published.
    recheck_standpoint_on_approach(step, world_model, obstacle_tracker, path_controller, timestamp_ms);


    current_target_room_ = step.target.room_pos;
    // ── INV-3: SAY WHAT WE ARE DRIVING TO, AFTER EVERY REPAIR, BEFORE WE DRIVE ───────────────────
    // HERE and not at the commitment, because this is the first line at which step.target.room_pos is
    // FINAL — the reachability repair, the boxed-in search and recheck_standpoint_on_approach have all
    // had their say. Those moved the standpoint by up to 0.3 m in ordinary running and 5.5 m in the
    // no-path cases, and the producer has never once been told. Publishing the producer's pose here
    // instead of ours would restore the whole defect one level down: a claim that names a pose nobody
    // is driving to is the same lie as a claim that names none.
    // ★Idempotent (publish_executing compares first), so the 20 Hz call rate costs no wire traffic.
    if (step.target.from_affordance and step.target.node_id != 0 and graph_)
    {
        const int claimed_epoch = approach_commit_.has_value()
                                ? approach_commit_->epoch
                                : rc::AffordanceManager::producer_epoch(graph_, step.target.node_id).value_or(0);
        // epoch 0 = a pre-rollout producer. We still drive; we just cannot name a proposal, and saying
        // nothing is honest where saying "epoch 0" would look like a real claim to a rebuilt peer.
        if (claimed_epoch != 0)
            affordance_manager.publish_executing(graph_, step.target.node_id, claimed_epoch,
                                                 step.target.room_pos.x(), step.target.room_pos.y(),
                                                 step.target.yaw_rad);
    }
    step.target_changed = !last_target_info_.has_value()
                       || !ControllerWorldModel::same_target_instance(*last_target_info_, step.target);
    if (step.target.from_affordance and step.target_changed)
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, timestamp_ms,
                      std::format("planning step built for '{}'", step.target.node_name));
    last_target_info_ = step.target;
    active_target_id_ = target->node_id;
    target_is_new_ = step.target_changed;   // consumed by the goal_reached branch, see target_is_new_
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
    // THE EXACT TEST, not a disc. The acceptance guard asks "does the body still fit where it fit
    // before", answered by the footprint's own support function at the route's heading — the same
    // machinery robot_footprint.h was written to provide, and the reason it exists ("a rectangle
    // reaches further along its diagonal than across its width, and a disc model has to assume the
    // worst case in every direction").
    // ★THE SAME QUARTER TURN THE GRID PLANNER WAS MISSING. RouteOptimizer computes this heading as
    // atan2(tangent.y, tangent.x) — a room YAW — while RobotFootprint::support_radius takes its own
    // theta, whose forward axis is +y. Passed raw, the guard tested a body lying 90 degrees across the
    // route it was checking: 4.2 cm too narrow where the route runs beside something and 4.2 cm too
    // long where it approaches one. Same defect, same fix, and support_radius_yaw exists to state it.
    opt.support_radius = [](float heading_yaw, const Eigen::Vector2f &dir_world)
    { return rc::RobotFootprint::shadow().support_radius_yaw(heading_yaw, dir_world); };
    // Only used if support_radius is ever unset. INSCRIBED, not circumscribed: below the inscribed
    // radius the body cannot fit at ANY heading, so it is a true bound; the circumscribed radius is the
    // worst case over all headings at once and sits above every clearance this apartment affords.
    opt.clearance_floor = rc::RobotFootprint::shadow().inscribed_radius();
    opt.iterations = 30;
    opt.safety_bias = params_ ? params_->route_safety_bias : 0.5f;
    opt.w_jerk = params_ ? params_->route_jerk_weight : 0.f;
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
        route_geom_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
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
      << opt.safety_bias << ' ' << opt.w_jerk << '\n';
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
        // ★RENAMED 2026-08-19: this is the TRACKER's per-cycle log, and the tracker is PLAIN, not MPPI.
        // The old name cost a real misdiagnosis — a 19 s off-path excursion with 39 rotation-cap
        // sign-flips was attributed to "MPPI mode-flipping between two near-equal-cost rollouts" purely
        // because the file said mppi. A filename is a claim like any column name, and this one was
        // false. (Five columns misread the same way today: yaw_err_deg, min_esdf vs clear_now,
        // rob_facing_deg, d_arrival vs the gate's operand, path_kappa's -999 sentinel.)
        mppi_csv_.open("tracker_diag.csv", std::ios::out | std::ios::trunc);
        mppi_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
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
                         "pose_stamp_age,path_gen\n";
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
              << pose_stamp_age << ',' << path_generation_ << '\n';
}

void ControllerSession::log_route_event(const char *event, bool ok, std::uint64_t t_ms,
                                        const rc::TrajectoryController &path_controller,
                                        float window_m)
{
    if (!route_events_csv_open_)
    {
        route_events_csv_.open("route_events.csv", std::ios::out | std::ios::trunc);
        route_events_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
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
    // ── THE AUTHORED ORDER, POSSIBLY BACKWARDS ───────────────────────────────────────────────────
    // A LOCAL copy, reversed here and nowhere else: the recorded mission must not change because a run
    // was driven the other way round, or the reversal would be written back to missions.toml the next
    // time the library is saved. Reversing the list rather than indexing backwards also means the yaw
    // below — "face the NEXT waypoint" — keeps working with no second spelling: in a reversed tour the
    // next waypoint IS the previous one, and that is what the robot will actually be driving toward.
    std::vector<Eigen::Vector2f> authored;
    authored.reserve(m->waypoints.size());
    for (const auto &w : m->waypoints) authored.push_back(w.pos);
    if (route_reverse_)
    {
        std::ranges::reverse(authored);
        std::println("[route] REVERSED: driving '{}' in reverse waypoint order. This is a DIFFERENT "
                     "stimulus from the forward tour — recorded as params.reversed in the run JSON so "
                     "the two cannot be compared by accident.", mission_.selected_name());
    }
    for (int i = 0; i < n; ++i)
    {
        const Eigen::Vector2f raw = authored[i];
        const Eigen::Vector2f next = authored[(i + 1) % n];
        const Eigen::Vector2f dir = next - raw;
        const float yaw = dir.squaredNorm() > 1e-9f ? std::atan2(dir.y(), dir.x()) : 0.f;
        const auto safe = grid_planner_.nearest_free(raw, yaw);
        if (not safe.has_value())
        {
            // Boxed in beyond the repair radius. It CHANGES THE ROUTE, so it is said out loud rather
            // than swallowed: a benchmark whose stimulus quietly differs is worse than no run.
            // ★HANDED ON RAW RATHER THAN DROPPED HERE (2026-08-17). There used to be two independent,
            // permanent drop sites — this one and RouteFollower's — and neither remembered anything.
            // Passing the unrepaired waypoint through means its hop simply fails in the builder, which
            // DEFERS it, so both cases funnel into the one mechanism that can give it back. The
            // immediate behaviour is unchanged (an infeasible pose cannot be planned to either way);
            // what changes is that the tour can recover it when the robot gets there.
            std::println("[route] waypoint {} at ({:.2f},{:.2f}) is BOXED IN — deferring it. The driven "
                         "route does not match the recorded one until it comes back.", i + 1, raw.x(), raw.y());
            ++skipped;
            wps.push_back(raw);
            raw_wps.push_back(raw);
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
        std::println("[route] {} waypoint(s) moved to the nearest feasible pose, {} deferred.",
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
        // ★SHARPNESS. The budget above is a CONSTANT, so v_rot = h*w_max/kappa scales a hairpin the
        // robot can barely fit through exactly as it scales a gentle bend. That is the whole speed law
        // in every turn — measured, the comfort limit stops binding above kappa = 0.3 1/m and the
        // rotational-acceleration limit never binds at all — so if the robot is to take a REALLY sharp
        // turn more carefully than a wide one, the budget itself has to know how sharp the turn is.
        // See ControllerRuntimeParams::sharp_turn_slowdown for the measured sweep; q = 0 is the old law.
        const float kr = k_avg * rc::RobotFootprint::shadow().circumscribed_radius();
        const float sharp = 1.f + std::max(0.f, params_->sharp_turn_slowdown) * kr * kr;
        const float rot_budget = std::max(0.05f, params_->max_rot_speed_rps)
                               * (route_tracker_active_ ? rot_headroom_ : 1.0f) / sharp;
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
    // Measure what the yaw correction did, at the two poses the controller commits to. HERE and not in
    // build_planning_step because that function returns early for a manual target and for a running
    // mission — and a mission is precisely where corridors are driven, so monitoring only the affordance
    // path would have watched the one mode with the least of what it is looking for. Temporary; see the
    // function. Cheap: four grid lookups, plus one whole-map census on the first world.
    monitor_footprint_orientation(step, time_source());

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
rc::RouteFollower::PlanFn ControllerSession::route_plan_fn()
{
    return [this](const Eigen::Vector2f &a, const Eigen::Vector2f &b)
    {
        auto r = grid_planner_.plan(a, b);
        if (not r.has_value()) last_plan_failure_ = grid_planner_.last_failure();
        return r;
    };
}

rc::RouteFollower::FreeFn ControllerSession::route_free_fn()
{
    return [this](const Eigen::Vector2f &p, float hdg) { return grid_planner_.pose_free(p, hdg); };
}

void ControllerSession::on_route_reauthored(const char *event, float window_m,
                                           rc::TrajectoryController &path_controller,
                                           std::uint64_t now_ms)
{
    ++route_repair_count_;
    mission_.note_replan();   // count what HAPPENED, not the reflex that asked for it
    // Force the new curve to be installed: the follower is still holding the old one.
    path_controller.stop();
    current_plan_.reset();
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    log_route_event(event, true, now_ms, path_controller, window_m);
}

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
    // ── SECOND CHANCE FOR THE WAYPOINTS THE BUILD COULD NOT REACH ────────────────────────────────
    // A waypoint blocked when the route was built used to be gone for the whole mission, and that drop
    // was SELF-FULFILLING: out of the route means the robot never drives toward it, so the close-range
    // evidence that would free it is never gathered. Now it is deferred and re-offered here.
    // Rate-limited to ~1 Hz because a successful recovery re-authors a window; the test itself is nearly
    // free, since reinstate_deferred only plans for a waypoint the robot is BOTH near and still short of.
    // Costs exactly nothing on the overwhelmingly common route with nothing deferred.
    if (route_active_ and route_.deferred_count() > 0
        and time_source() - last_reinstate_ms_ >= 1000)
    {
        last_reinstate_ms_ = time_source();
        // The same window a repair re-authors: it is how far ahead the route may be re-planned, so it is
        // also how far ahead a waypoint can be put back into one. No second allowance to keep in step.
        constexpr float kReinstateAheadM = 4.0f;
        const auto rr = route_.reinstate_deferred(step.robot_pose.pos, kReinstateAheadM,
                                                  route_plan_fn(), route_free_fn());
        // The curve changed under the follower — same obligation as a repair, same one place.
        if (rr.recovered > 0)
            on_route_reauthored("reinstate", kReinstateAheadM, path_controller, time_source());
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
                                          route_plan_fn(), route_free_fn());

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
            on_route_reauthored("repair", kRepairBackM + kRepairAheadM, path_controller, time_source());
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
        ++path_generation_;      // route re-authored — same reasoning as above
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
                                                 route_tracker_active_ ? rot_headroom_ : 1.0f,
                                                 params_ ? params_->sharp_turn_slowdown : 0.f,
                                                 rc::RobotFootprint::shadow().circumscribed_radius());
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
    // ★★★WHO OWNS ARRIVAL IS A PROPERTY OF WHAT IS BEING DRIVEN, NOT OF WHEN A PATH WAS LAST INSTALLED
    // — AND NOT OF WHETHER THIS CYCLE HAPPENED TO SUCCEED. Set FIRST, before every early return.
    // A continuous route switches this off on purpose (drive_mission_route: a tour ends by arc length,
    // and its euclidean endpoint test is true before the robot has moved). It has to come back on for a
    // point target, and the two places it was tried before were both too late:
    //   1. inside `if (target_changed or not is_active())` — so it was re-armed only when a path was
    //      RE-INSTALLED. While the target churned every few seconds that happened constantly and armed
    //      the flag BY ACCIDENT; once this session made the target stable (gain out of the identity
    //      test, standpoint and facing frozen), the accidental re-arm went with it. That was 13c8bc3.
    //   2. at the END of the function — which the `HOLDING — no route to '<t>'` early return above
    //      skips entirely. And that return is likeliest EXACTLY where it hurts: a few centimetres from
    //      the goal, where plan_path has nothing left to plan. So the flag stayed off through the whole
    //      terminal approach, which is the one stretch where it decides anything.
    // Measured with (2) live, run of 11:42: 20 approach rows inside the 0.25 m goal_threshold —
    // d_arrival down to 0.196 m at 0.63 m/s — with ZERO `reached` and ZERO `ALIGN` rows in 202.
    // ★The lesson is the placement, not the line: a fact about WHAT WE ARE DOING must be asserted on
    // entry, not as a side effect of a code path that can fail.
    path_controller.set_endpoint_arrival(true);

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
            unroutable_at_ = UnroutableAt{.goal = step.target.room_pos,
                                          .robot = step.robot_pose.pos.head<2>().cast<float>(),
                                          .map = grid_planner_.world_hash()};
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
        ++path_generation_;      // a new curve: cross-track before and after are different questions
        if (params_ and params_->smooth_planned_path)
            path_controller.set_path_presmoothed(current_plan_->room_path);
        else
            path_controller.set_path(current_plan_->room_path);
        // ── HAND THE POINT-TARGET CURVE TO PLAIN ─────────────────────────────────────────────────
        // set_route() had exactly ONE caller — the mission branch — so `route_spline_` was null for every
        // affordance and every click, and the PLAIN dispatch fell through to PD. Invisible until the
        // fallback got a counter: 901 consecutive cycles on PD, zero recoveries, i.e. the whole run.
        // ★WIRED, REVERTED, AND NOW RE-WIRED — the middle step matters. Wiring it on 2026-08-09 made the
        // robot drive THROUGH its targets ("jumps over the target and moves forward until it hits an
        // obstacle"): PLAIN's stop taper had a floor made of the spline's own sampling grid, so it never
        // reached zero. That is fixed at the taper (see plain_tracker.cpp) and measured on
        // tracker_sim --stop-test: PLAIN now comes to rest 0.063 m from the endpoint against PD's
        // 0.124 m, having overshot by 0.056 m, and it tracks 4.7x tighter on this geometry (19 vs 91 mm
        // rms on a 2.5 m hop). Do not re-wire this without that fix present.
        // driven_curve() already states the precedence (mission route, else point plan); this installs
        // what it resolves rather than restating the rule. force_reset because the curve is new even
        // though plan_spline_'s address is not — see set_route().
        path_controller.set_route(plan_spline_valid_ and plan_spline_.valid() ? &plan_spline_ : nullptr,
                                  /*force_reset=*/true);
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
                                                 const ControllerRobotPose &robot_pose,
                                                 const rc::TrajectoryController &path_controller)
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
        // ★ARCHIVE IT. This was the ONLY diagnostic in the controller that was truncated every run and
        // never kept — and it is the one that settled who was at fault for the affordance stall, by
        // aligning its `tgt_*` against room_concept's `pub_*` on one wall clock. Two runs in a row were
        // compared against a file that no longer existed for the earlier one, and the 2026-08-19 09:05
        // baseline was lost to the very next start.
        // ★Registered HERE, at open, and not next to the others in specificworker.cpp — that is the
        // stall_events.csv pattern (:3921) and it is the correct one for a file that only some runs
        // write. Registering it up front would make archive_on_stop copy the PREVIOUS run's rows under
        // THIS run's stamp on any run that never approaches an affordance, which is exactly the lie
        // band_diag.csv told on 20260802-135917 (see ensure_band_csv).
        if (approach_csv_open_) mission_.archive_on_stop("approach_diag.csv");
        if (approach_csv_open_)
            approach_csv_
                << "# FINAL APPROACH to an affordance standpoint. One row per cycle inside 1 m, plus every\n"
                   "# cycle of the terminal rotation (which has NO obstacle check of its own).\n"
                   "# phase        = approach | ALIGN (rotating in place) | reached\n"
                   "# d_target_m   = robot to the STANDPOINT.  d_arrival_m = the EUCLIDEAN distance to the\n"
                   "#                arrival point that the arrival test COMPARED against goal_thr, captured at\n"
                   "#                the test itself. It used to be read back out of dist_to_goal, which the PLAIN\n"
                   "#                tracker overwrites with REMAINING ARC LENGTH on every not-arrived cycle — so\n"
                   "#                the column silently changed meaning between approach and reached rows, and a\n"
                   "#                whole day of reading it against goal_thr compared two different quantities.\n"
                   "# arr_ran      = the goal check was REACHED this cycle (0 = compute() early-returned before it,\n"
                   "#                so end_arr/goal_thr describe a test that never ran). arr_end_on = endpoint_arrival_\n"
                   "#                AS SEEN BY THE TEST; if it disagrees with end_arr (polled after compute) the\n"
                   "#                flag moved during the cycle. arr_passed/arr_recede = the two non-proximity\n"
                   "#                arrival routes. d_arrival_m < goal_thr with arr_ran=1 and goal_reached=0 is a\n"
                   "#                CONTRADICTION — that is the thing to chase.\n"
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
                   "# raw_tgt_x/y  = the standpoint AS PUBLISHED by the producer; tgt_x/y is what the\n"
                   "#                controller is actually driving to, and fix_held = 1 when a standpoint\n"
                   "#                repair is in force — so the two differ BY OUR DOING.\n"
                   "#                ★'Who moved the target' used to require aligning this file against\n"
                   "#                room_concept's log on one wall clock, and the answer changed the whole\n"
                   "#                diagnosis. It is a local question now: raw moving = producer, tgt\n"
                   "#                moving while raw stands still = us.\n"
                   "#                ⚠These three lines were first written INTO the column row itself,\n"
                   "# end_arr/goal_thr = whether the follower's ARRIVAL TEST IS RUNNING AT ALL, and the\n"
                   "#                radius it uses. end_arr=0 ⇒ every arrival is skipped: the robot\n"
                   "#                drives past its goal and keeps going, which from outside looks like\n"
                   "#                a tracking or planning fault and is neither.\n"
                   "#                ⚠⚠AND I SPLIT THE HEADER A SECOND TIME ADDING THESE, in the very\n"
                   "#                commit whose message said every legend line belongs above the column\n"
                   "#                row. The column row is ONE logical line assembled from adjacent\n"
                   "#                string literals; inserting a literal between them splices text into\n"
                   "#                the middle of it. Append legend lines HERE, never below.\n"
                   "#                which split the header across two lines and made the file unreadable\n"
                   "#                by its own header. Every legend line belongs above the column row.\n"
                   "t_ms,target,phase,tgt_x,tgt_y,tgt_facing_deg,rob_x,rob_y,rob_facing_deg,"
                   "d_target_m,d_arrival_m,yaw_err_deg,cmd_adv,cmd_rot,"
                   "clear_now_m,clear_des_m,free_now,free_des,sweep_min_m,sweep_ok,sweep_worst_deg,"
                   "tgt_sweep_min_m,tgt_sweep_ok,min_esdf_m,goal_reached,raw_tgt_x,raw_tgt_y,fix_held,"
                   "end_arr,goal_thr,arr_ran,arr_end_on,arr_passed,arr_recede\n";
    }
    if (not approach_csv_open_) return;

    constexpr float kDeg = 180.f / static_cast<float>(M_PI);
    approach_csv_ << t_ms << ',' << tgt.node_name << ','
                  << (o.goal_reached ? "reached" : (o.aligning ? "ALIGN" : "approach")) << ','
                  << tgt.room_pos.x() << ',' << tgt.room_pos.y() << ',' << facing_des * kDeg << ','
                  << robot_pose.pos.x() << ',' << robot_pose.pos.y() << ',' << facing_now * kDeg << ','
                  << d_target << ',' << o.dist_to_arrival_pt << ','
                  << (o.goal_yaw_err_rad.has_value() ? *o.goal_yaw_err_rad * kDeg : 0.f) << ','
                  << o.adv << ',' << o.rot << ','
                  << clear_now << ',' << clear_des << ',' << (free_now ? 1 : 0) << ',' << (free_des ? 1 : 0) << ','
                  << sweep.min_clearance_m << ',' << (sweep.feasible ? 1 : 0) << ','
                  << sweep.worst_heading_rad * kDeg << ','
                  << sweep_tgt.min_clearance_m << ',' << (sweep_tgt.feasible ? 1 : 0) << ','
                  << (o.aligning ? std::numeric_limits<float>::quiet_NaN() : o.min_esdf) << ','
                  << (o.goal_reached ? 1 : 0) << ','
                  // The producer's own standpoint, before any repair of ours, and whether a repair is
                  // in force. Attribution of every target move, without a second agent's clock.
                  << (last_raw_target_pos_ ? last_raw_target_pos_->x()
                                           : std::numeric_limits<float>::quiet_NaN()) << ','
                  << (last_raw_target_pos_ ? last_raw_target_pos_->y()
                                           : std::numeric_limits<float>::quiet_NaN()) << ','
                  << (approach_fix_.has_value() ? 1 : 0) << ','
                  << (path_controller.endpoint_arrival() ? 1 : 0) << ','
                  << path_controller.goal_threshold() << ','
                  << (o.arrival_test_ran ? 1 : 0) << ',' << (o.arrival_endpoint_on ? 1 : 0) << ','
                  << (o.arrival_passed_pt ? 1 : 0) << ',' << (o.arrival_by_recession ? 1 : 0) << '\n';

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

// ── WHAT THE YAW CORRECTION ACTUALLY DID, MEASURED ───────────────────────────────────────────────
// Two readings, because they answer two different questions.
//
// THE CENSUS is the whole C-space, once, on the first real world: how much of it changed hands.
// ★It does NOT shrink. A quarter turn is exactly two of the eight heading buckets, so the corrected
// rasterisation at heading h is the old one at heading h-2 — summed over all headings both cover the
// identical set of body placements, and the free totals come out equal to the state. Measured on the
// recorded apartment: 74385 free states before, 74385 after, with 2882 of them (3.9%) swapping sides.
// That is the honest answer to "does this cost us space": no, it MOVES it. A gap running east-west
// stops admitting east-west travel — the body is 0.543 m across, and the old orientation asked that gap
// for 0.460 — and starts admitting north-south, which is the 4.2 cm of head-on room the old
// orientation was spending on the body's length. So expect both: corridors that close, and standpoints
// that stop reporting BOXED IN. Costs one cycle (~25 ms on this map) and never runs again.
//
// THE COUNTERS are the live poses the controller commits to — where the robot is, and where it is going.
// The census says how much space moved; these say whether any of it was space the robot was USING. That
// distinction is the whole question: a correction that removes ten thousand states the robot never
// visits costs nothing, and one that removes the standpoint it is driving to costs a run.
void ControllerSession::monitor_footprint_orientation(const ControllerPlanningStep &step,
                                                      std::uint64_t timestamp_ms)
{
    if (not census_logged_ and grid_planner_.width() > 0 and grid_planner_.occupied_cells() > 0)
    {
        census_logged_ = true;
        const auto c = grid_planner_.orientation_census();
        std::println("[footprint] YAW CORRECTION census over the live map: {} (cell,heading) states, "
                     "{} free — unchanged in TOTAL, as it must be (a quarter turn is 2 of 8 buckets, so "
                     "the two rasterisations cover the same placements). {} of them ({:.1f}% of the free "
                     "C-space) CHANGED HANDS: that many headings the body may no longer hold where it "
                     "stands, and as many it now may. Expect corridors that close and standpoints that "
                     "stop reporting BOXED IN — not less room overall.{}",
                     c.states, c.free_now, c.lost,
                     c.free_legacy > 0 ? 100.0 * c.lost / c.free_legacy : 0.0,
                     c.lost == 0 ? "   ⚠ZERO — the correction did not land." : "");
    }

    // The robot's own pose, at the heading it is actually holding (body theta -> yaw).
    const float robot_yaw = step.robot_pose.theta + static_cast<float>(M_PI_2);
    ++yawfix_cycles_;
    const bool r_now = grid_planner_.pose_free(step.robot_pose.pos, robot_yaw);
    const bool r_was = grid_planner_.pose_free_legacy(step.robot_pose.pos, robot_yaw);
    yawfix_robot_now_blocked_ += (r_was and not r_now);
    yawfix_robot_now_free_ += (r_now and not r_was);
    const bool t_now = grid_planner_.pose_free(step.target.room_pos, step.target.yaw_rad);
    const bool t_was = grid_planner_.pose_free_legacy(step.target.room_pos, step.target.yaw_rad);
    yawfix_target_now_blocked_ += (t_was and not t_now);
    yawfix_target_now_free_ += (t_now and not t_was);

    // Report only when the two rasterisations DISAGREED about something the robot cared about. Steady
    // state should look steady: on an open floor they agree every cycle and this stays silent.
    const long disagreements = yawfix_robot_now_blocked_ + yawfix_robot_now_free_
                             + yawfix_target_now_blocked_ + yawfix_target_now_free_;
    if (disagreements > 0 and timestamp_ms - yawfix_log_ms_ >= 10000)
    {
        yawfix_log_ms_ = timestamp_ms;
        std::println("[footprint] yaw-correction monitor, {} cycles: at the ROBOT's own pose the "
                     "corrected body was refused {}x where the old one passed (and freed {}x); at the "
                     "TARGET, refused {}x / freed {}x. A 'refused at the robot' is the interesting one — "
                     "it means the body was already somewhere the old rasterisation thought was fine.",
                     yawfix_cycles_, yawfix_robot_now_blocked_, yawfix_robot_now_free_,
                     yawfix_target_now_blocked_, yawfix_target_now_free_);
    }
}

// See the header. One get_node plus three attribute reads per cycle, all mutex-guarded; the display
// decides whether the values are actually drawn, because only it knows whether our own higher-rate feed
// has spoken since the last cycle.
void ControllerSession::feed_external_velocity_trace(const ControllerWorldModel &world_model,
                                                    ControllerDisplay &display,
                                                    std::uint64_t timestamp_ms)
{
    float ref_adv = 0.f, ref_rot = 0.f, meas_adv = 0.f, meas_rot = 0.f;
    bool fresh = false;
    if (graph_)
        if (const auto rid = world_model.graph_state().robot_id; rid != 0)
            if (const auto robot_node = graph_->get_node(rid); robot_node.has_value())
            {
                ref_adv = graph_->get_attrib_by_name<robot_ref_adv_speed_att>(*robot_node).value_or(0.f);
                ref_rot = graph_->get_attrib_by_name<robot_ref_rot_speed_att>(*robot_node).value_or(0.f);
                // ★AND WHAT THE BASE ACTUALLY DID. robot_ref_* turns out to be written by THIS AGENT
                // only — measured 2026-08-18, the robot drove at 0.640 m/s with the reference at exactly
                // 0.000 — so a panel fed from the reference alone is blank for a whole manual drive.
                // robot_current_* is written by robot_concept whoever is driving, and is the only
                // universally available witness. The display keeps it on its OWN series because it is
                // MEASURED, not commanded, and folding it in would change what the panel means.
                meas_adv = graph_->get_attrib_by_name<robot_current_advance_speed_att>(*robot_node).value_or(0.f);
                meas_rot = graph_->get_attrib_by_name<robot_current_angular_speed_att>(*robot_node).value_or(0.f);
                // Staleness against WHEN THE REFERENCE WAS WRITTEN, not against whether it is non-zero:
                // a commander holding still writes a legitimate 0, and that must not read as nobody driving.
                if (const auto ts = graph_->get_attrib_by_name<robot_ref_speed_timestamp_att>(*robot_node);
                    ts.has_value() and timestamp_ms >= *ts)
                    fresh = timestamp_ms - *ts <= kRefSpeedStaleMs;
            }
    display.update_velocity_trace_external(ref_adv, ref_rot, fresh, meas_adv, meas_rot);
}

// ── AN APPROACH IS A COMMITMENT ──────────────────────────────────────────────────────────────────
// The producer may republish an affordance's standpoint whenever it likes, and for a viewpoint chosen
// by information gain it WILL: arriving is what satisfies the gain, so arriving is what moves the next
// best view. Following each republish makes the target a carrot on a stick — measured, 39 republishes
// over a metre in 11.4 minutes, the robot never once closing the last 3 mm to its own goal threshold.
// So the standpoint is latched for the duration of ONE approach. The newest offer is not discarded, it
// is simply what the next approach will start from.
// ★Released by the target's own lifetime, with no timer: `last_target_info_` is cleared at every place
// an approach concludes (finalize_reached, the unreachable/useless-spot refusal, a manual take-over),
// and a different affordance carries a different node_id. Both are checked here, so there is no second
// notion of "the approach is over" to fall out of step with the first.
// ★Deliberately NOT applied to a manual click target: a click IS the operator moving the goal, and
// refusing to follow it would be the controller ignoring the person driving it.
void ControllerSession::hold_approach_commitment(ControllerPlanningStep &step, std::uint64_t timestamp_ms)
{
    if (not step.target.from_affordance or step.target.node_id == 0)
    { approach_commit_.reset(); return; }

    // ── THE EPOCH IS WHAT MAKES THIS A POSITION RATHER THAN A UNILATERAL ACT ─────────────────────
    // ★★★DEGRADATION FIRST, because it is the half that is easy to forget and it breaks the fleet if
    // it is wrong. nullopt means the producer has not been rebuilt against the new cortex header yet,
    // so it cannot publish an epoch and CANNOT SEE our executing edge either. Holding a commitment it
    // has no way to observe is exactly the silent divergence this whole change exists to abolish — so
    // with no epoch we do what the controller did before any of this: follow every republish.
    // Worse behaviour, honestly arrived at, and it disappears the moment that agent is rebuilt.
    const auto epoch = rc::AffordanceManager::producer_epoch(graph_, step.target.node_id);
    if (not epoch.has_value())
    {
        if (approach_commit_.has_value())
        {
            std::println("[approach] '{}' publishes no proposal epoch — a pre-rollout producer cannot see "
                         "our claim, so deferring would be invisible to it. Following its republishes "
                         "until it is rebuilt.", step.target.node_name);
            std::fflush(stdout);
        }
        approach_commit_.reset();
        return;
    }

    // A different affordance, or an approach that concluded since the last cycle, is a new epoch.
    if (approach_commit_.has_value()
        and (approach_commit_->node_id != step.target.node_id or not last_target_info_.has_value()))
        approach_commit_.reset();

    // A newer proposal than the one we hold is not a reason to reset the commitment — it is exactly the
    // case the commitment exists for. But an OLDER or equal epoch with a moved pose would mean the
    // producer rewrote content without bumping, which is a producer bug; take the new pose in that case
    // rather than pin the robot to something nobody is offering.
    if (approach_commit_.has_value() and approach_commit_->epoch > epoch.value())
        approach_commit_.reset();

    if (not approach_commit_.has_value())
    {
        approach_commit_ = ApproachCommitment{.node_id = step.target.node_id,
                                              .epoch = epoch.value(),
                                              .room_pos = step.target.room_pos,
                                              .yaw_rad = step.target.yaw_rad,
                                              .last_offer = step.target.room_pos,
                                              .deferred = 0};
        return;                                   // the first offer IS the commitment
    }

    // ★THE SAME EPSILON same_target_instance USES, and for the same reason: below it the two poses are
    // the same standpoint and there is nothing to defer. A second number here would let a move be
    // "different enough to replan" and "not different enough to defer" at once.
    constexpr float pos_eps_m = 0.05f;
    const Eigen::Vector2f offered = step.target.room_pos;
    const bool moved = (offered - approach_commit_->room_pos).cwiseAbs().maxCoeff() >= pos_eps_m;
    approach_commit_->last_offer = offered;
    if (not moved) return;

    // ★COUNT REPUBLISHES, NOT CYCLES. The first version incremented every cycle the offer differed, so
    // ONE republish held for two minutes at 20 Hz printed "DEFERRED (2407x)" and read as 2407 separate
    // events. A counter that conflates a duration with a count is not a measurement.
    if ((offered - approach_commit_->last_offer_counted).cwiseAbs().maxCoeff() >= pos_eps_m)
    {
        ++approach_commit_->deferred;
        approach_commit_->last_offer_counted = offered;
    }
    step.target.room_pos = approach_commit_->room_pos;
    step.target.yaw_rad  = approach_commit_->yaw_rad;
    // Rate-limited: the producer re-offers continuously and one line per cycle would bury the run.
    if (timestamp_ms - approach_commit_log_ms_ >= 3000)
    {
        approach_commit_log_ms_ = timestamp_ms;
        std::println("[approach] '{}' is offering epoch {} at ({:.2f},{:.2f}), {:.2f} m from epoch {} which "
                     "this approach is executing ({:.2f},{:.2f}) — HOLDING ({} republishes). We publish what "
                     "we are driving to on the `executing` edge, so the producer can see the disagreement "
                     "and decide: wait, or withdraw. It is no longer ours to settle alone.",
                     step.target.node_name, epoch.value(), offered.x(), offered.y(),
                     (offered - approach_commit_->room_pos).norm(), approach_commit_->epoch,
                     approach_commit_->room_pos.x(), approach_commit_->room_pos.y(),
                     approach_commit_->deferred);
        std::fflush(stdout);
    }
}

// ── THE STANDPOINT, RE-ASKED IN THE LAST METRES AGAINST EVIDENCE THAT HAS NOT FADED ──────────────
// WHAT WAS ALREADY THERE, so this is not mistaken for a check that did not exist: the standpoint IS
// re-tested every cycle. set_world() re-rasterises the whole obstacle set each compute, and the repair
// in build_planning_step re-runs nearest_free / nearest_rotatable on the raw published pose. What was
// missing is not FREQUENCY, it is EVIDENCE. Every one of those tests reads the planner grid, and the
// grid's occupancy is rasterised from beliefs that DECAY — residual_concept's occupancy hulls above
// all. When the hull over a real object fades, the cells under it read free, `pose_free` answers "fine"
// and goes on answering "fine" however many times it is asked, and the robot drives at a standpoint
// inside something the map has forgotten. Re-asking a stale question faster cannot fix that.
//
// The live return cloud does not decay: it is re-measured every sweep and nothing has to believe in it.
// So in the last metres — where those returns actually carry information about the standpoint, rather
// than about whatever occludes it — the standpoint is asked again with the cloud admitted as evidence,
// and moved if the body placed there would be in contact.
//
// Three properties keep this from becoming its own fault:
//   ANCHORED at the standpoint, never at the robot, so the search is deterministic and the target
//     cannot chase the robot the way a robot-relative repair once did.
//   HELD once taken (approach_fix_), so a cloud that flickers at the boundary cannot make the goal
//     jitter — the fix is dropped when it stops being admissible, not when the evidence blinks.
//   BOUNDED by the same window it was noticed in: an affordance viewpoint dragged further than that is
//     no longer a viewpoint of that object, and a target we cannot repair is reported, not invented.
void ControllerSession::recheck_standpoint_on_approach(ControllerPlanningStep &step,
                                                       ControllerWorldModel &world_model,
                                                       ControllerObstacleTracker &obstacle_tracker,
                                                       const rc::TrajectoryController &path_controller,
                                                       std::uint64_t timestamp_ms)
{
    const float window = params_ != nullptr ? params_->affordance_approach_recheck_m : 0.f;
    if (window <= 0.f or not step.target.from_affordance) { approach_fix_.reset(); return; }
    // Keyed by NAME, like the reachability repair: the name is the identity that survives every repair
    // that moves the pose, and it needs no tolerance to compare.
    if (approach_fix_.has_value() and approach_fix_->name != step.target.node_name)
        approach_fix_.reset();

    // The anchor is the standpoint as the grid repair left it — that is the pose the robot is actually
    // driving to, and searching around anything else would be answering about a pose already rejected.
    const Eigen::Vector2f anchor = step.target.room_pos;
    // A held fix belongs to the pose it repaired. If that pose has moved further than the fix itself was
    // allowed to move, it is a different standpoint and the old correction is about a problem elsewhere.
    if (approach_fix_.has_value() and (anchor - approach_fix_->anchor).squaredNorm() > window * window)
        approach_fix_.reset();

    // ★★★ALREADY STANDING THERE ⇒ THERE IS NOTHING TO APPROACH, AND NOTHING TO REPAIR. Leave the target
    // exactly as the producer published it and let the ordinary arrival path have it.
    // ★THE LIVELOCK THIS BREAKS, from a live log 2026-08-19:
    //     [approach] 'afford_room' — 0.09 m out, the live LiDAR says (-1.50,-3.88) is OCCUPIED ...
    //                Standpoint moved to (-1.71,-3.64), 0.31 m away.
    //     [affordance] afford_room REFUSED: already at this standpoint on the first cycle
    //     [aff-select] 'afford_room' was the only affordance on offer — taking it again ...
    // ...repeating forever, with the robot stationary and both agents busy.
    // The evidence that condemned (-1.50,-3.88) is the robot's OWN blind spot: the self-return filter
    // must delete every point inside the body, which leaves a body-shaped hole guaranteed to contain no
    // returns whatever is really there. Standing 0.09 m away, the robot IS that hole.
    // ★The existing guard below — reject a candidate that lands inside the body — was aimed at exactly
    // this and does not reach it. It only stops the search landing ON the robot, so the search lands
    // just OUTSIDE it instead (0.31 m, here), which is the same loop one step removed. Its comment
    // claims "a normal approach relocates nothing and never reaches this line"; a robot already parked
    // on the viewpoint reaches it every cycle. Rejecting candidates is the wrong altitude — the
    // question is whether this function should be running at all.
    // ★Threshold-free in the sense that matters: this is not a tuned margin, it is the SAME predicate
    // the arrival test uses (goal_threshold). "Close enough to have arrived" and "close enough that
    // there is nothing left to approach" are one question, so they must not be two numbers.
    if ((step.robot_pose.pos - anchor).norm() <= path_controller.params.goal_threshold)
    {
        approach_fix_.reset();
        return;
    }
    // ★MEASURED TO WHAT WE ARE DRIVING TO, not to the anchor. Once a fix is taken the robot approaches
    // the MOVED pose, so a window measured against the anchor walks out of range exactly because the fix
    // worked — which would drop it and put the target straight back where the returns said it must not
    // be, once per lap of that loop, forever.
    // Measured to the pose actually being driven to, so a held fix does not walk itself out of the
    // window and get dropped. ★The gate applies WHETHER OR NOT a fix is held: the earlier spelling
    // (`not approach_fix_.has_value() and ...`) disabled it outright once a fix existed, so the full
    // cloud read and scan below then ran every cycle for the rest of the target's life however far away
    // the robot was. That is the opposite of what the comment above claims. Caught by review 2026-08-17.
    const Eigen::Vector2f driving_to = approach_fix_.has_value() ? approach_fix_->pos : anchor;
    if ((step.plan_origin - driving_to).squaredNorm() > window * window) return;

    auto *lidar = obstacle_tracker.lidar_buffer();
    if (lidar == nullptr) return;
    const auto [cloud_opt] = lidar->read_last();
    if (not cloud_opt.has_value()) return;
    const auto &[xs, ys, zs] = cloud_opt.value();
    const std::size_t n_pts = std::min({xs.size(), ys.size(), zs.size()});
    if (n_pts == 0) return;

    // The SAME standoff the planner obeys — footprint_safety_margin_m, the one number robot_footprint.h
    // exists to keep from multiplying. Nothing here introduces a second one.
    // ★HELD AS A MEMBER, not rebuilt per call: shadow() allocates an 18-vertex polygon, and this runs on
    // the control thread. The margin is re-set only when the configured one moves.
    const float margin = params_ != nullptr ? params_->footprint_safety_margin_m : 0.f;
    if (std::abs(approach_body_.safety_margin() - margin) > 1e-6f)
        approach_body_.set_safety_margin(margin);
    const auto &body = approach_body_;
    // ★NOT `+ safety_margin()`. circumscribed_radius() ALREADY adds the margin (robot_footprint.cpp:48),
    // so adding it again stacked a second 0.05 m — 0.375 m where the body reaches 0.325 — which is the
    // exact multiplying-margin failure robot_footprint.h was written to abolish, reintroduced inside the
    // file that quotes it. Caught by review 2026-08-17.
    const float reach = body.circumscribed_radius();
    const float reach2 = reach * reach;

    // ── THE STANDPOINT MUST BE CLEAR WITH THE ROOM THE ARRIVAL ITSELF NEEDS ──────────────────────
    // ★★★A POSE VALIDATED AS CLEAR IS NOT WHERE THE ROBOT STOPS. Arrival completes anywhere within
    // goal_threshold of the standpoint, in any direction, so a body that merely FITS at the validated
    // pose can be a quarter of a metre deeper into the furniture when the robot actually halts — which
    // is the reported behaviour: targets that drive the robot into contact with the table.
    // ★THE MARGIN IS THE ARRIVAL BAND, not a comfort preference. ComfortStandoff (0.6 m) is the wrong
    // instrument: a standpoint exists to SEE something, and holding it two thirds of a metre off every
    // return would refuse most of the cells worth standing in. goal_threshold is the honest number
    // because it is exactly how far the arrival test permits the robot to be from the pose we checked.
    // Nothing new is introduced — the same quantity the arrival gate already compares.
    rc::RobotFootprint standoff_body = approach_body_;
    standoff_body.set_safety_margin(margin + path_controller.params.goal_threshold);
    const float standoff_reach2 = standoff_body.circumscribed_radius() * standoff_body.circumscribed_radius();

    // Returns that could touch ANY candidate the search may return, with the robot's own body removed.
    // Self-returns must go first: the wheels sit at 0.25 m and the robot is by definition standing next
    // to the standpoint by the time this runs, so leaving them in would condemn every standpoint the
    // moment the robot got close enough to check it.
    constexpr float kBodyZLo = 0.05f;   // above this is the body, below it is the floor beneath us
    constexpr float kBodyZHi = 1.45f;   // the mesh tops out at 1.42 m
    const float robot_yaw = step.robot_pose.theta + static_cast<float>(M_PI_2);
    std::vector<Eigen::Vector2f> returns;
    returns.reserve(256);                    // typical near-anchor count; avoids ~10 reallocations
    for (std::size_t i = 0; i < n_pts; ++i)
    {
        if (zs[i] < kBodyZLo or zs[i] > kBodyZHi) continue;
        const Eigen::Vector2f p{xs[i], ys[i]};
        if ((p - anchor).squaredNorm() > (window + reach) * (window + reach)) continue;
        if (body.contains_yaw(p, step.robot_pose.pos, robot_yaw)) continue;   // the robot seeing itself
        returns.push_back(p);
    }
    if (returns.empty()) return;   // no evidence is not the same as clear, and it licenses nothing

    // WHICH HEADINGS THE BODY WILL ACTUALLY HOLD THERE decides the shape of the test, because a rigid
    // footprint is not rotation-invariant and assuming a disc is what the footprint planner exists to
    // stop doing. A target that ends in a terminal rotation sweeps every heading, so it must clear the
    // circumscribed disc; one that does not is tested at the two headings it really occupies — the
    // arrival heading (it drives in along the bearing from wherever it is) and the commanded facing.
    const bool rotates = wants_final_facing(step.target);
    // ★THE ARRIVAL HEADING IS AN ARGUMENT NOW, NOT A RE-DERIVATION. Deriving it inside from
    // `centre - plan_origin` made the admissibility of a FIXED pose a function of where the robot
    // currently stands, which is what produced the fleeing standpoint (see ApproachFix::arrival_yaw).
    // Choosing a NEW fix still asks the question from where the robot is — that is the heading it will
    // really arrive on — but RE-TESTING a held one must ask it at the heading the fix was chosen for.
    const auto arrival_yaw_from = [&](const Eigen::Vector2f &centre)
    {
        const Eigen::Vector2f approach_dir = centre - step.plan_origin;
        return approach_dir.squaredNorm() > 1e-8f
             ? std::atan2(approach_dir.y(), approach_dir.x())
             : step.target.yaw_rad;
    };
    const auto clear_of_returns_at = [&](const Eigen::Vector2f &centre, const float arrival_yaw)
    {
        // ★★★ THE PATCH UNDER THE ROBOT IS A BLIND SPOT, NOT FREE SPACE. The self-return filter above
        // deletes every point inside the robot's own body — it must, or the wheels at 0.25 m would
        // condemn every standpoint the moment the robot got close enough to test it. But that leaves a
        // body-shaped hole GUARANTEED to hold no returns whatever is really there, and
        // nearest_free_where, searching outward from an anchor the object itself occupies, lands on it
        // almost every time. Live 2026-08-19, robot at (-0.81,-2.26):
        //     [approach] (-1.00,-2.38) is OCCUPIED ... Standpoint moved to (-0.81,-2.26)
        //     [affordance] REFUSED: already at this standpoint on the first cycle
        // — the standpoint relocated onto the robot ITSELF, was instantly refused as already-reached,
        // and the pair looped there indefinitely.
        // ★This file already states the rule, about an empty cloud: "no evidence is not the same as
        // clear, and it licenses nothing". Same rule, applied to the hole we punch in the cloud
        // ourselves. A viewpoint the robot already occupies also yields exactly the image it already
        // has, so there is nothing to be gained by moving there even if it IS free.
        // ★★NOT AN ARRIVAL TEST, and that is the whole point. This rejects a candidate RELOCATION and
        // never a completion, so it cannot turn a legitimate arrival into a refusal — which is what
        // forced the revert of 31faa35 (af7ba55). A normal approach relocates nothing and never
        // reaches this line.
        if (body.contains_yaw(centre, step.robot_pose.pos, robot_yaw)) return false;
        for (const auto &p : returns)
        {
            // ★JUDGED WITH THE ARRIVAL BAND ADDED, so the pose stays clear wherever inside that band
            // the robot actually stops. The self-return filter above still uses the TRUE body: growing
            // it there would delete returns from the very furniture this is meant to keep away from.
            if ((p - centre).squaredNorm() > standoff_reach2) continue;   // out of reach at any heading
            if (rotates) return false;                                   // ...and it turns through all
            if (standoff_body.contains_yaw(p, centre, arrival_yaw)) return false;
            if (standoff_body.contains_yaw(p, centre, step.target.yaw_rad)) return false;
        }
        return true;
    };
    // The predicate nearest_free_where takes: a CANDIDATE is judged at the heading the robot would
    // really arrive on, which is the honest question when choosing where to go.
    const auto clear_of_returns = [&](const Eigen::Vector2f &centre)
    { return clear_of_returns_at(centre, arrival_yaw_from(centre)); };

    const auto aim_at_object = [&]()
    {
        // The producer's yaw faced the object from the viewpoint it published; the standpoint has moved.
        if (step.target.parent_node_id == 0) return;
        if (const auto obj = world_model.read_node_room_xy(step.target.parent_node_id, timestamp_ms);
            obj.has_value())
            step.target.yaw_rad = std::atan2(obj->y() - step.target.room_pos.y(),
                                             obj->x() - step.target.room_pos.x());
    };

    // ★A FIX, ONCE TAKEN, IS STATIC FOR THE LIFE OF THE TARGET. Validated ONCE against the live LiDAR
    // at the moment it is chosen (that is what `clear_of_returns` below does), and then held — no
    // re-test, no re-solve.
    // ★★WHY THE RE-TEST IS GONE (user, watching a live run, 2026-08-19: "you are moving the target
    // before the robot can reach it" / "just leave it static in a place that the local lidar validates
    // as reachable"). This block used to re-ask, every cycle, whether the held standpoint was still
    // clear, and re-solve when the answer came back no. Freezing the arrival heading removed the
    // systematic FLEE (measured: 77% of moves in the robot's own travel direction, down to 57% with
    // 50% being no correlation at all) but not the MOTION: 197 target moves over 5 cm in one 120 s
    // run, p50 0.30 m. A standpoint that is re-decided is not a standpoint.
    // ★The re-test could never be a stable question, because its evidence is not stable. The LiDAR
    // returns it judges against are re-gathered every cycle from a moving robot, at a viewpoint that
    // is by construction getting CLOSER to the standpoint — so occlusion, incidence and density at
    // that patch of floor all change under it, and a boundary case flickers. Holding a flickering
    // predicate's FIRST answer is not worse than holding its latest one; it is merely honest about
    // which answer it is holding.
    // ★What replaces it is not nothing. If the held standpoint really is unstandable, the robot fails
    // to arrive, the wedge or Spinning verdict fires, and after kMaxEscapesPerAffordance the affordance
    // is given up and the claim released (begin_escape). That path reports the failure to the producer
    // instead of silently re-siting its viewpoint — which is what room_concept asked for on 2026-08-19
    // ("refuse, never move": a relocated-then-completed standpoint makes the producer record an
    // observation that never happened, corrupting its exploration drive).
    // The lifecycle guards above still drop the fix when it stops belonging to this target: a changed
    // node name (:1751) and a producer that republishes its viewpoint elsewhere (:1759).
    if (approach_fix_.has_value())
    {
        step.target.room_pos = approach_fix_->pos;
        // ★THE FROZEN FACING, not a fresh aim. See ApproachFix::object_yaw: re-aiming every cycle at a
        // live object estimate is a second moving target wearing the first one's clothes.
        step.target.yaw_rad  = approach_fix_->object_yaw;
        return;
    }

    // One search, both questions: the grid (which knows about walls, and about obstacles no longer in
    // view) AND the cloud (which knows about what is there now). nearest_free_where returns the anchor
    // itself when the anchor already satisfies both, so this both tests and repairs in one call.
    const auto fix = grid_planner_.nearest_free_where(anchor, step.target.yaw_rad, clear_of_returns, window);
    if (not fix.has_value())
    {
        // Nowhere within the window is clear under both. Say so and leave the target alone: inventing a
        // standpoint outside the window would service the affordance from somewhere it cannot see its
        // object, and driving on silently is what this whole function exists to stop.
        if (timestamp_ms - approach_blocked_log_ms_ >= 3000)
        {
            approach_blocked_log_ms_ = timestamp_ms;
            std::println("[approach] ⚠ '{}' at ({:.2f},{:.2f}) is occupied by {} live LiDAR return(s) the "
                         "planner grid no longer carries, and NOTHING within {:.2f} m is clear under both. "
                         "Holding the target as published — it will drive into that.",
                         step.target.node_name, anchor.x(), anchor.y(), returns.size(), window);
        }
        return;
    }
    if ((*fix - anchor).squaredNorm() <= 1e-6f) return;   // the standpoint is clear; nothing to do


    step.target.room_pos = *fix;      // aim_at_object reads room_pos, so the move lands first
    aim_at_object();                  // ONE aim, at the moment of the repair, and it is kept
    approach_fix_ = ApproachFix{.pos = *fix, .name = step.target.node_name, .anchor = anchor,
                               .arrival_yaw = arrival_yaw_from(*fix),
                               .object_yaw = step.target.yaw_rad};
    std::println("[approach] '{}' — {:.2f} m out, the live LiDAR says ({:.2f},{:.2f}) is OCCUPIED by "
                 "something the grid has forgotten (residual decays; returns do not). Standpoint moved to "
                 "({:.2f},{:.2f}), {:.2f} m away.",
                 step.target.node_name, (step.plan_origin - anchor).norm(),
                 anchor.x(), anchor.y(), fix->x(), fix->y(), (*fix - anchor).norm());
    // (room_pos and yaw_rad were both set above, before the fix was recorded, so that the frozen
    // object_yaw is exactly the heading this run of the function chose — not one re-derived later.)
}

bool ControllerSession::wants_final_facing(const ControllerTargetInfo &target) const
{
    if (not target.from_affordance) return false;                       // a clicked point has no facing
    if (params_ != nullptr and not params_->goal_facing_yaw_enabled) return false;
    if (not target_contract_known_) return false;   // unknown policy ⇒ do not invent a rotation
    using P = rc::affordance::Policy;
    return target_contract_.policy == P::Servo or target_contract_.policy == P::Orient;
}

// Same spot, within the width of the robot: a standpoint half a body away from one that wedged is the
// same approach through the same gap, not a new opportunity.
const ControllerSession::UselessSpot *ControllerSession::known_useless_spot(const Eigen::Vector2f &pos,
                                                                           const std::uint64_t now_ms) const
{
    constexpr float kSameSpotM = 0.30f;
    for (const auto &s : useless_spots_)
    {
        // ★EXPIRED EVIDENCE IS NOT EVIDENCE. The failure was about a situation — an obstacle field, an
        // approach bearing, a pose — and those change. Past the producer's recovery time the veto must
        // be allowed to be wrong, or it outlives everything that justified it.
        if (now_ms >= s.when_ms and now_ms - s.when_ms > kUselessSpotMemoryMs) continue;
        if ((s.pos - pos).norm() <= kSameSpotM)
            return &s;
    }
    return nullptr;
}

void ControllerSession::remember_useless_spot(const Eigen::Vector2f &pos, const std::string &name,
                                             const std::uint64_t now_ms)
{
    std::erase_if(useless_spots_, [&](const UselessSpot &s)
                  { return now_ms >= s.when_ms and now_ms - s.when_ms > kUselessSpotMemoryMs; });
    for (auto &s : useless_spots_)
        if ((s.pos - pos).norm() <= 0.30f) { ++s.hits; s.when_ms = now_ms; return; }   // re-failed: restamp
    useless_spots_.push_back({pos, name, 1, now_ms});
    std::println("[controller] remembering ({:.2f},{:.2f}) as unreachable — {} spot(s) now known bad. "
                 "The producer is not told, so it will offer this again; we will not drive there again.",
                 pos.x(), pos.y(), useless_spots_.size());
}

// ── THE AFFORDANCE'S TARGET, FOR THE OVERLAY ─────────────────────────────────────────────────────
// last_target_info_ carries the standpoint AFTER select_target's repair, which is the one the robot is
// driving to and therefore the only one worth marking; the pre-repair value is a pose the controller
// has already rejected. The facing comes from wants_final_facing() rather than a second reading of the
// contract, so the overlay draws a heading exactly when the executor will turn to it.
std::optional<ControllerStandpoint> ControllerSession::current_standpoint() const
{
    if (not last_target_info_.has_value() or not last_target_info_->from_affordance)
        return std::nullopt;
    return ControllerStandpoint{.room_pos = last_target_info_->room_pos,
                                .yaw_rad = last_target_info_->yaw_rad,
                                .has_facing = wants_final_facing(*last_target_info_)};
}

// Resolved once per target, when it is SELECTED. Same read the executor performs at arrival, done
// early because the policy now decides behaviour before the robot gets there.
void ControllerSession::resolve_target_contract(const ControllerTargetInfo &target)
{
    if (target.node_id == contract_for_node_id_) return;
    contract_for_node_id_ = target.node_id;
    target_contract_ = {};
    target_contract_known_ = false;
    // ★A NEW AFFORDANCE HAS NOT LOOKED AT ANYTHING YET. Reset here, at the ONE place a new affordance
    // becomes current, rather than only where a look begins: a REACH affordance is consumed on arrival
    // and never runs a look at all, so a leftover `true` from the PREVIOUS affordance's successful look
    // made its dwell wait for confirming masks of an object it had never even turned toward. That is
    // the "arrives, does not look, waits for five masks that never come" failure — a flag whose reset
    // lived on a path the failing case does not take.
    last_look_succeeded_ = false;
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
// Append one protocol line, bounded. Deduplicated on the text itself: the caller runs every cycle
// and most cycles say nothing new, so a transcript that recorded them all would bury the transitions
// it exists to show.
void ControllerSession::note_protocol(rc::AffordanceExecution::ProtocolLine::Side side,
                                      std::uint64_t t_ms, std::string text)
{
    if (text.empty()) return;
    if (not affordance_transcript_.empty() and affordance_transcript_.back().text == text) return;
    // ★ALSO TO DISK. A transcript that exists only behind a window dies with the window, and these
    // exchanges are seconds long -- by the time anyone opens the panel the interesting part is over.
    // It is also the only way to read the conversation when the panel says nothing, which is exactly
    // the case that needs explaining.
    if (not protocol_log_open_)
    {
        protocol_log_.open("affordance_protocol.log", std::ios::out | std::ios::trunc);
        protocol_log_open_ = protocol_log_.is_open();
    }
    if (protocol_log_open_)
    {
        using Side = rc::AffordanceExecution::ProtocolLine::Side;
        protocol_log_ << std::format("{} {:<9} {}\n", t_ms,
            side == Side::Producer ? "producer" : side == Side::Consumer ? "consumer" : "selector",
            text);
        protocol_log_.flush();
    }
    affordance_transcript_.push_back({t_ms, side, std::move(text)});
    // ★BOUNDED. This lives for the life of the agent; an unbounded vector behind a GUI is a leak that
    // only shows after an hour of driving. 200 lines is several minutes of a busy exchange.
    if (affordance_transcript_.size() > 200)
        affordance_transcript_.erase(affordance_transcript_.begin(),
                                     affordance_transcript_.begin() + 100);
}

void ControllerSession::update_affordance_view(const ControllerRobotPose &robot_pose,
                                               const rc::TrajectoryController::ControlOutput &o,
                                               bool output_enabled, float align_tol_rad,
                                               std::uint64_t now_ms)
{
    using Step = rc::AffordanceStepView;
    using S = rc::AffordanceStepView::State;
    rc::AffordanceExecution v;
    v.recent = affordance_recent_;
    v.transcript = affordance_transcript_;

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
        orient_overlay_visible_ = false;   // nothing is turning; do not leave a bearing on the map
        affordance_view_.recent = affordance_recent_;
        affordance_view_.transcript = affordance_transcript_;
        return;
    }

    if (affordance_view_.affordance != last_target_info_->node_name)
    {
        note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, now_ms,
                      std::format("claimed '{}'", last_target_info_->node_name));
        affordance_started_ms_ = now_ms;
        affordance_step_since_ms_ = now_ms;
        affordance_prev_step_.clear();
        affordance_nav_total_m_ = std::max(0.05f, o.dist_to_goal);   // the distance this run began with
        // The contract is already resolved — build_planning_step does it when the target is selected,
        // because the policy decides real behaviour now and not just what this window draws.
    }

    orient_overlay_visible_ = false;   // re-armed below only while an Orient is the running policy
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
    // ── AN ORIENT IS A DIFFERENT PROGRAM, NOT A REACH WITH HOLES IN IT ────────────────────────────
    // The pipeline below is a Reach: claim, navigate, align, arrive. An Orient never navigates and has
    // no standpoint, so rendering it there produced a chart that was mostly Skipped boxes and said
    // almost nothing about what the affordance actually does. The steps a program shows should come
    // from the CONTRACT, since that is where the producer wrote down what it wants done.
    //
    // What an Orient actually is: turn to a bearing, at a rate the producer may have named, and hold
    // there long enough to count. The completion is the arrival at the bearing when there is no
    // predicate -- which is exactly the case the calibration pivot uses, and the case that once
    // completed on cycle one having turned nothing.
    if (orient)
    {
        const float target_yaw = last_target_info_->yaw_rad;
        const float yaw_err = std::atan2(std::sin(target_yaw - robot_pose.theta),
                                         std::cos(target_yaw - robot_pose.theta));
        constexpr float kAligned = 0.05f;                  // the executor's own band
        const bool aligned = std::abs(yaw_err) < kAligned;
        orient_overlay_yaw_ = target_yaw;
        orient_overlay_visible_ = true;

        v.steps.push_back({.label = "claim affordance", .kind = Step::Kind::Pipeline,
                           .state = S::Done, .progress = -1.f,
                           .detail = std::format("rotate in place to {:.0f} deg",
                                                 target_yaw * 180.f / static_cast<float>(M_PI))});

        {
            // The rate is the producer's if it named one, ours otherwise -- and which of those it is
            // matters, because a manoeuvre capped by a servo tuning it was never meant for is the
            // defect this field was added to fix.
            const float asked = target_contract_.max_yaw_rate;
            const float cap   = asked > 0.f ? asked
                                            : (params_ ? params_->lockon_max_yaw_rps : 0.12f);
            Step st{.label = "turn to bearing", .kind = Step::Kind::Pipeline};
            st.state = aligned ? S::Done : S::Active;
            // Fraction of the way there, from wherever the turn started.
            st.progress = orient_start_err_rad_ > 1e-3f
                        ? std::clamp(1.f - std::abs(yaw_err) / orient_start_err_rad_, 0.f, 1.f) : -1.f;
            st.detail = std::format("{:+.0f} deg to go, {:.2f} rad/s cap ({})",
                                    yaw_err * 180.f / static_cast<float>(M_PI), cap,
                                    asked > 0.f ? "producer's rate" : "our servo cap");
            v.steps.push_back(std::move(st));
        }

        {
            Step st{.label = "hold aligned", .kind = Step::Kind::Stable};
            const int need = std::max(1, target_contract_.stable_n);
            st.state = not aligned ? S::Pending
                     : orient_stable_ >= need ? S::Done : S::Active;
            st.progress = std::clamp(static_cast<float>(orient_stable_) / static_cast<float>(need),
                                     0.f, 1.f);
            st.detail = std::format("{}/{} cycles inside {:.0f} deg", orient_stable_, need,
                                    kAligned * 180.f / static_cast<float>(M_PI));
            v.steps.push_back(std::move(st));
        }

        {
            // ★NAMED, because an Orient with NO predicate is the case that silently completed on the
            // first cycle standing still. A chart that does not say which rule is deciding cannot show
            // the difference between "arrived" and "never had to".
            Step st{.label = target_contract_.goal.empty() ? "complete: the bearing IS the goal"
                                                           : "complete: predicate holds",
                    .kind = Step::Kind::Terminal};
            st.state = S::Pending;
            st.detail = target_contract_.goal.empty()
                      ? "no completion predicate — arriving at the bearing completes it"
                      : std::format("{} clause(s) on the feedback node", target_contract_.goal.size());
            v.steps.push_back(std::move(st));
        }

        v.transcript = affordance_transcript_;
        affordance_view_ = v;
        return;
    }

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
            // ★THE REQUIREMENT COMES FROM THE LOOP ITSELF once it is running. Before arrival there is no
            // loop to ask, so the selection-time contract is the best available answer — but the moment
            // the servo begins, the only number that means anything is the one it is counting against.
            const int needed = lockon_.active() or lockon_.done() ? lockon_.stable_needed()
                                                                  : target_contract_.stable_n;
            s.progress = needed > 0 ? std::clamp(static_cast<float>(lockon_.stable()) /
                                                 static_cast<float>(needed), 0.f, 1.f) : -1.f;
            s.detail = std::format("{} of {} consecutive", lockon_.stable(), needed);
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
    band_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
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

    // ── THE PANEL MUST NAME THE AFFORDANCE IN EVERY STATE, INCLUDING THE ONES THAT RETURN EARLY ──
    // ★★★MEASURED 2026-08-24: the window's first row read "no affordance yet" for the whole of an
    // `afford_calib` run, while affordance_select.jsonl showed it selected on 404 of 407 cycles,
    // state Executing, gain 50.0, zero rejections. The view was not wrong about its data — it was
    // never reached.
    // ★ROOT CAUSE IS ORDERING, NOT DATA. update_affordance_view sits ~100 lines below, and its own
    // comment says it "runs EVERY cycle an affordance is live, not only while the window is open".
    // Three returns above it break that, and one of them is fatal for an ORIENT affordance
    // specifically: `afford_calib` publishes THE ROBOT'S OWN POSE as its standpoint (an orient does
    // not navigate — room_scene_graph.cpp), so there is no plan, `!current_plan_` fires, and both the
    // Orient branch below AND the panel are skipped. The one affordance whose whole behaviour is
    // "turn in place" was structurally invisible to the window built to watch it.
    // ★A ZEROED ControlOutput IS HONEST HERE. These three states own the base and are not tracking a
    // curve, so there is no dist_to_goal to report; the fields that describe navigation are absent
    // because navigation is absent, which is exactly what the panel should convey. What must NOT be
    // absent is the identity — which affordance this is — and that comes from last_target_info_.
    // ★★★AND IT HAS TO REACH THE DISPLAY, NOT JUST THE MEMBER. This lambda recomputed
    // affordance_view_ and stopped there. The only push to the panel is at the END of execute_plan,
    // and every branch that uses this lambda RETURNS before reaching it — so the four states that OWN
    // the base (escape, no-plan, lock-on, orient) were precisely the four the window could not see.
    // What it showed instead was whatever the last full cycle had left in it: the previous Reach, or
    // at startup the empty view that reads "no affordance yet". Recomputing a view and not delivering
    // it is indistinguishable from not recomputing it, and worse, because the code reads as if it had
    // been handled. The bearing overlay goes out here for the same reason.
    const auto refresh_view_only = [&]
    {
        const rc::TrajectoryController::ControlOutput idle{};
        update_affordance_view(robot_pose, idle, motion_commander.output_enabled(),
                               path_controller.params.align_yaw_tol_rad, overlay_now_ms_);
        affordance_view_.dwell_left_s = dwell_left_s(overlay_now_ms_);
        affordance_view_.dwell_mask_hits = dwell_mask_hits_;
        affordance_view_.suppressed = suppressed_affordance_;
        display.set_affordance_execution(affordance_view_);
        display.set_orient_overlay(robot_pose.pos.x(), robot_pose.pos.y(), orient_overlay_yaw_,
                                   robot_pose.theta, orient_overlay_visible_);
    };

    // An escape maneuver (physical-stuck recovery) owns the base until it finishes backing
    // out — bypass the planner/follower entirely, just like the lock-on micro-search below.
    // It survives the current_plan_ reset done in begin_escape(), so it's gated FIRST.
    if (escape_active_)
    {
        step_escape(robot_pose, path_controller, motion_commander, time_source());
        refresh_view_only();
        return;
    }

    if (!current_plan_.has_value())
    {
        display.clear_robot_trajectory();
        clear_tracking_state();
        path_controller.stop();
        note_no_command(); motion_commander.stop_robot();
        refresh_view_only();
        return;
    }

    // A lock-on micro-search in progress owns the base until it locks or gives up — bypass the path
    // follower so it isn't disturbed by goal_reached re-evaluation while we quasi-statically servo.
    if (lockon_.active())
    {
        if (step_lockon(motion_commander, time_source))
            finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
        refresh_view_only();
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
                // BEFORE the step, so the view describes the state being acted on — the same ordering
                // the full cycle uses — and the final cycle of a turn is shown as a turn rather than
                // as the idle that finalize_reached leaves behind.
                refresh_view_only();
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

    // ★THE ONE CONSUMER THAT GETS THE FRESH POSE. compute() serves the control law AND expresses the
    // room-frame cloud in the robot frame for the ESDF, and BOTH want "where the robot is now": the
    // cloud is a map registered at its own per-plane stamps, so bringing it into the robot frame with a
    // stale pose displaces every obstacle by the robot's motion since the scan — pushing them behind
    // where they really are relative to the body, which is the unsafe direction. So the freshest pose is
    // the right argument for the whole call, not merely tolerable for half of it.
    // Everything else in this function keeps `robot_pose`, the scan-aligned one.
    const auto &control_pose = tracker_pose_.has_value() ? *tracker_pose_ : robot_pose;
    const auto control_output = path_controller.compute(control_pose.as_transform());
    // Surface what the ARRIVAL test is waiting on, every cycle, before any of the branches below can return
    // early — otherwise the readout would freeze exactly in the states worth watching (aligning, blocked).
    // The affordance program, rebuilt from state that already exists. Runs EVERY cycle an affordance is
    // live, not only while the window is open: these runs last seconds, and a view you must open in time
    // to catch a failure is one that never catches it.
    update_affordance_view(robot_pose, control_output, motion_commander.output_enabled(),
                           path_controller.params.align_yaw_tol_rad, overlay_now_ms_);
    // What the camera/masks view should highlight. During the DWELL the affordance is already retired,
    // so it is the object of the one that just finished — which is the whole point of dwelling: the
    // acquisition being inspected belongs to an action that has ended.
    attention_object_ = affordance_view_.active ? affordance_view_.object
                      : (overlay_now_ms_ < affordance_dwell_until_ms_ ? dwell_object_ : std::string{});
    attention_standpoint_ = attention_object_.empty() ? std::nullopt : current_standpoint();
    affordance_view_.dwell_left_s = dwell_left_s(overlay_now_ms_);
    affordance_view_.dwell_mask_hits = dwell_mask_hits_;
    affordance_view_.suppressed = suppressed_affordance_;
    display.set_affordance_execution(affordance_view_);
    display.set_orient_overlay(robot_pose.pos.x(), robot_pose.pos.y(), orient_overlay_yaw_,
                               robot_pose.theta, orient_overlay_visible_);
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
    // The L-adaptation panel's objective, on the SAME gate as the CSV below and for the same reason:
    // the deviation a click target or an affordance is driving with is the deviation L is adapted on.
    sample_live_tracking(path_controller.is_active(), robot_pose, control_output.cross_track_m,
                         control_output.rot, overlay_now_ms_);
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
        log_approach_diagnostics(overlay_now_ms_, control_output, robot_pose, path_controller);
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
            note_no_command(); motion_commander.stop_robot();
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
                proximity_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
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
                               << (stall_judge_.since_ms() != 0 ? t_ms - stall_judge_.since_ms() : 0) << ','
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
        note_no_command(); motion_commander.stop_robot();
        return;
    }

    // ★★★A "REACHED" WITH NO PATH IS NOT AN ARRIVAL. compute()'s FIRST line returns
    //     if (!active_ || path_room_.empty()) { active_ = false; out.goal_reached = true; return out; }
    // so a target adopted this cycle — before any path is installed — reports reached. Consuming that
    // told the producer a cell was visited while the robot stood 3.11 m away: 383 completions, robot
    // stationary, room re-offering an unvisited cell and the consumer answering "I just did that".
    // ★THE DISCRIMINATOR IS WHERE THE ROBOT ACTUALLY IS. A genuine arrival has the body overlapping
    // the standpoint (measured live: 0.18 m of a 6.85 m approach); the degenerate one had the robot
    // 3.11 m away. The bound is the arrival band plus the body radius — both already in the system,
    // so nothing new is introduced, and it is deliberately generous: it exists to reject a metres-away
    // "arrival", not to re-judge a close one. Being slightly outside the band is still an arrival.
    const float d_here =
        last_target_info_.has_value()
            ? (last_target_info_->room_pos - robot_pose.pos.head<2>().cast<float>()).norm()
            : std::numeric_limits<float>::infinity();
    // ★★★AND IT MUST BE THE STANDPOINT THAT WAS ASKED FOR. d_here measures against OUR target — the
    // pose after the repair — so a repair that relocated the goal onto the robot satisfies this test
    // without the robot moving: 140 of 163 accepted arrivals were a median 2.86 m from the published
    // cell (measured 2026-08-19, standpoint_audit.jsonl). The producer is then told a cell was visited
    // that nothing ever looked at, its neglect drops, and the standpoint is never offered again — the
    // one failure the whole protocol exists to prevent, reported as success.
    // ★The allowance is the repair WE applied, so an honest few-centimetre correction still arrives and
    // a substitution cannot. No new constant: the band is the arrival band, the slack is our own edit.
    // ★DELIBERATELY NOT KEYED ON "a path was active". Measured: 163 of 163 arrivals — the genuine ones
    // included — report no active path, because the follower deactivates itself on reaching the goal.
    // The empty-path degenerate and a real arrival are indistinguishable there; only distance separates
    // them, which is why this test is a distance and not a state.
    const float d_published =
        (last_target_info_.has_value() and last_raw_target_pos_.has_value())
            ? (*last_raw_target_pos_ - robot_pose.pos.head<2>().cast<float>()).norm()
            : d_here;
    const float arrival_band = path_controller.goal_threshold() + approach_body_.circumscribed_radius();
    // ★AN APPROACH IS JUDGED AGAINST THE APPROACH POSE, and completed as `unreachable`. Without the
    // first half the robot would arrive at the closest reachable pose and the guard would reject it
    // (the published cell is metres away, correctly), leaving the claim standing for ever; without the
    // second half we would be back to reporting an arrival at a cell nothing ever looked at.
    const bool arrival_is_real =
        not last_target_info_.has_value() or not last_target_info_->from_affordance
        or (target_is_approach_only_ ? d_here <= arrival_band
                                     : (d_here <= arrival_band
                                        and d_published <= arrival_band + last_repair_applied_m_));
    // ★AUDIT EVERY ARRIVAL DECISION, both verdicts. "Reached" is the claim the producer trusts, so the
    // operands that produced it have to be on disk: which pose the test measured against (the REPAIRED
    // target, not the published cell), how far the robot was from EACH, and whether a path existed at
    // all — compute() reports goal_reached with an empty path, which is not an arrival.
    if (control_output.goal_reached)
        audit_standpoint("arrival", time_source(),
                         last_raw_target_pos_.value_or(
                             last_target_info_.has_value() ? last_target_info_->room_pos
                                                           : robot_pose.pos.head<2>().cast<float>()),
                         last_target_info_.has_value() ? last_target_info_->room_pos
                                                       : robot_pose.pos.head<2>().cast<float>(),
                         robot_pose.pos.head<2>().cast<float>(),
                         path_controller.is_active() ? "path-active" : "no-path",
                         current_plan_.has_value() ? "plan" : "no-plan",
                         target_is_new_ ? 1 : 0, 1, arrival_is_real ? 1 : 0);
    if (control_output.goal_reached and arrival_is_real)
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
            last_look_succeeded_ = false;   // this look has not happened yet; do not inherit the last one's
            lockon_.begin(time_source(), active_contract_.scalar_target,
                          active_contract_.stable_n, active_contract_.timeout_ms);
            path_controller.stop();
            if (step_lockon(motion_commander, time_source))
                finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source());
            return;
        }
        // ★★★THE "ALREADY THERE ⇒ REFUSE" RULE IS GONE (2026-08-19). Introduced by 9b3e5a4 for a real
        // problem — a standpoint the robot already occupies yields no new image, so consuming it as an
        // arrival and dwelling 3 s for an acquisition that cannot come was a loop — but reporting it as
        // a REFUSAL made the producer and consumer livelock instead, and that turned out to be far
        // worse. It was the proximate cause of every stall measured today:
        //   · ~104 completions/min with the robot standing still (refuse → re-offer → refuse)
        //   · the take→refuse cycle at exactly the retry period, target never held long enough to plan
        //   · "REFUSED: already at this standpoint" logged with the published cell 1.55 m away
        // ★A refusal is a statement to the PRODUCER that its cell is not standable. "I am already here"
        // is not that: the cell was fine, the robot simply arrived early. Reporting it as SATISFIED is
        // both true and terminating — room stamps the cell visited (mark_and_refresh writes the robot's
        // own pose into the visit grid), its neglect drops, and the next selection goes elsewhere. That
        // is the behaviour that ran for weeks before this rule existed.
        // ★9b3e5a4's ACTUAL insight is kept: there is no acquisition to wait for, so skip the dwell.
        // That removes the loop it was written to fix without inventing a refusal to do it.
        // ★★★AND IT MUST ACTUALLY BE THERE. `goal_reached` is ALSO returned true by the follower's
        // first line when it has no path yet (`if (!active_ || path_room_.empty()) { goal_reached =
        // true; return; }`), so a target adopted this cycle reads "reached" BEFORE any path exists.
        // Reporting that as Satisfied told the producer a cell had been visited that the robot was
        // 3.11 m away from: 383 completions, robot stationary, room re-offering the same unvisited
        // cell and the consumer answering "I just did that" — both frozen for 40 s at a time.
        // ★So require the robot to be within the arrival band of the standpoint. Same quantity the
        // arrival gate itself compares; no new constant. A degenerate "reached" with no path fails it.
        const float d_to_standpoint =
            last_target_info_.has_value()
                ? (last_target_info_->room_pos - robot_pose.pos.head<2>().cast<float>()).norm()
                : std::numeric_limits<float>::infinity();
        // ★ONCE PER STANDPOINT. Without this the branch re-fires every cycle (see instant_completed_at_).
        // Cleared when the robot leaves, so a genuine later re-offer of the same cell still works.
        const bool already_instant_done =
            instant_completed_at_.has_value()
            and (last_target_info_.has_value()
                 and (last_target_info_->room_pos - *instant_completed_at_).norm() < 0.30f);
        if (instant_completed_at_.has_value()
            and (robot_pose.pos.head<2>().cast<float>() - *instant_completed_at_).norm() > 0.60f)
            instant_completed_at_.reset();        // moved away: the cell may legitimately return
        if (target_is_new_ and last_target_info_.has_value() and last_target_info_->from_affordance
            and d_to_standpoint <= path_controller.goal_threshold()
            and not already_instant_done)
        {
            instant_completed_at_ = last_target_info_->room_pos;
            qInfo() << "[affordance]" << last_target_info_->node_name.c_str()
                    << "reached on the first cycle at" << d_to_standpoint
                    << "m — already at this standpoint. Completing as SATISFIED without the dwell.";
            finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source(), /*allow_dwell=*/false);
            return;
        }
        qInfo() << "[affordance]" << (last_target_info_.has_value() ? last_target_info_->node_name.c_str() : "?")
                << (target_is_approach_only_ ? "approach ended -> reporting UNREACHABLE (got as close as the map allows)"
                                             : "reached -> REACH (consume immediately)");
        audit_standpoint(target_is_approach_only_ ? "approach-end" : "arrival-end", time_source(),
                         last_raw_target_pos_.value_or(robot_pose.pos.head<2>().cast<float>()),
                         last_target_info_.has_value() ? last_target_info_->room_pos
                                                       : robot_pose.pos.head<2>().cast<float>(),
                         robot_pose.pos.head<2>().cast<float>(),
                         target_is_approach_only_ ? "approach" : "standpoint", "", -1, 1, 1);
        finalize_reached(affordance_manager, path_controller, motion_commander, display,
                             robot_pose.pos, time_source(), /*allow_dwell=*/not target_is_approach_only_,
                             target_is_approach_only_
                                 ? std::optional<rc::affordance::Outcome>(rc::affordance::Outcome::Unreachable)
                                 : std::nullopt);
        return;
    }

    if (!path_controller.is_active())
    {
        clear_tracking_state();
        note_no_command(); motion_commander.stop_robot();
        return;
    }

    // ── THE CARROT GUARD WAS REMOVED, AND WHY IT SHOULD NOT COME BACK AS A GATE ──────────────────
    // The observation behind it is real: clip_carrot_to_reachable can pull the carrot onto the robot,
    // and the bearing to a 4 cm carrot is pose noise, which is what saturates the rotation and walks
    // the robot 0.73 m off the corridor. The RESPONSE was wrong three times over, live, in one hour:
    //   · `d < 3 sigma, floor 0.08 m` — sigma read invalid on this path, so only the floor survived
    //     and it fired 7 times a second, stopping the robot outright ("not working now");
    //   · `d > 0` added to exclude the never-computed carrot — still 3.3 times a second;
    //   · with sigma valid at 0.072 m, 3 sigma = 0.217 m while carrots legitimately sit at 0.13 m,
    //     so the gate condemns ordinary driving.
    // ★THE LESSON IS THE HOUSE RULE, RELEARNED THE EXPENSIVE WAY: this is a continuous quantity and a
    // hard cutoff on it is a threshold wearing a derivation. The AI-aligned form is a PRECISION — the
    // disagreement between what the localiser says the robot moved and what its own motion predicts is
    // itself a standard deviation, and it belongs in the pose uncertainty the speed limiter already
    // consumes continuously. Inflate that, and a robot whose pose is in doubt slows and stops on the
    // mechanism that already exists, with nothing new to tune. Until that is built, the tracker keeps
    // its own behaviour: better an excursion that is measured than a gate that stops the robot.
    float adv_mps = control_output.adv;
    float side_mps = control_output.side;
    float rot_rps = -control_output.rot;
    // ★WHAT THE TRACKER ASKED FOR, CAPTURED BEFORE THE LIMITER OVERWRITES IT. The stall judge needs both
    // sides of this line: the ask expresses INTENT (and so decides whether anything is being predicted at
    // all), the post-limiter value is what the base was actually told. Reading only the second is what
    // made a robot frozen by its own speed limit indistinguishable from a robot with nowhere to go —
    // see stall_judge.h.
    const float asked_lin_mps = std::hypot(adv_mps, side_mps);
    motion_commander.apply_uncertainty_speed_limit(adv_mps, side_mps, rot_rps);
    // Publish what the limiter did into the actuation stream. It sits between the MPPI's command and the
    // wheels, and until now a lap could show a 17% gap between the two with no way to say whether this
    // was the cause or whether it was inert for want of a covariance on the RT edge.
    {
        const auto ud = motion_commander.last_uncertainty_diag();
        mission_.note_uncertainty_limit(ud.valid, ud.xy_std_m, ud.theta_std_rad, ud.adv_scale, ud.rot_scale);
        // ── AND ONTO THE VELOCITY PLOT, SO THE CAUSE SITS BESIDE THE EFFECT ──────────────────────
        // Mapped HERE because this is where the knees are: sigma is scaled so PoseXYStdStop lands at
        // max_adv, i.e. the line touching the top of the velocity band reads "at or past the stop knee,
        // throttle floored". Clamped there so it can never blow the plot's auto-scale and squash the adv
        // and rot traces it is meant to be compared against. Negative when no covariance reached the
        // limiter — the plot breaks the line rather than drawing a confident zero.
        const float v_max = params_ != nullptr ? params_->max_adv_speed_mps : 0.7f;
        const float stop = params_ != nullptr ? params_->pose_xy_std_stop_m : 0.12f;
        display.set_uncertainty_trace_value(
            ud.valid and stop > 1e-6f ? std::min(v_max, ud.xy_std_m * v_max / stop) : -1.f);
    }

    // ARRIVAL ROTATION: position reached, the controller is rotating IN PLACE to the target facing yaw
    // (adv=side=0, goal_reached still false). This is a controller-owned maneuver that makes NO waypoint
    // progress by design — turning around at a target. It must bypass the stuck/escape logic entirely,
    // else it reads as a wedge and drops a recovery disc right at the target. Just issue the rotation.
    if (control_output.aligning)
    {
        // ★★THE ARRIVAL ROTATION IS NO LONGER EXEMPT FROM THE STALL DETECTOR — measured 2026-08-19.
        // This used to call reset_stuck_state() every aligning cycle and return, so the detector was
        // structurally blind for the whole of a manoeuvre that has no timeout of its own. Live: FIVE
        // episodes in one 545 s run where the loop stopped reaching the MPPI stage for 19, 22, 39, 53
        // and 54 seconds, each entered from healthy driving (adv ~0.6, track_s advancing) and each
        // resuming with adv exactly 0.000, rot at the +-0.800 cap and track_s back at 0.00. From
        // outside it is a robot that has stopped; from inside it was "turning around at a target",
        // for the better part of a minute, with nothing able to contradict it.
        // ★THE EXEMPTION WAS RIGHT ABOUT THE OLD QUESTION AND WRONG ABOUT THE NEW ONE. Its reason —
        // "makes NO waypoint progress by design, so it reads as a wedge" — is entirely true of the
        // TRANSLATION test, which is why the ask passed here is zero: a turn in place must never be
        // called a wedge. But Spinning asks the other question, the one that IS meaningful during a
        // rotation: you were told to sweep this much heading, did you end up anywhere new? A
        // converging arrival rotation nets what it sweeps and stays silent. One that sweeps radians
        // and nets nothing is not turning around at a target, it is stuck at one, and it needs exactly
        // what a wedge needs — something to change the pose and force a replan.
        // ★NO TIMEOUT, DELIBERATELY. A bound on the manoeuvre would be a threshold standing in for
        // this question, and it would have to be loose enough for the slowest legitimate turn, which
        // is most of the 19 s. The convergence test needs no such number.
        if (detect_stuck(/*pursuing=*/true, /*asked_lin_mps=*/0.f, /*cmd_lin_mps=*/0.f,
                         motion_commander.last_uncertainty_diag().valid
                             ? motion_commander.last_uncertainty_diag().xy_std_m : 0.f,
                         robot_pose.pos, robot_pose.theta, rot_rps, time_source()))
        {
            begin_escape(robot_pose, obstacle_tracker, path_controller, time_source());
            step_escape(robot_pose, path_controller, motion_commander, time_source());
            return;
        }
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
    // The sigma the limiter ITSELF used this cycle, not a second reading of the graph — so the judge and
    // the throttle can never disagree about how well the robot is localised. `valid` false means no
    // covariance reached the limiter, so it did not throttle; 0 then makes the sigma clause a no-op,
    // which is right because the branch that consults it is unreachable when nothing was throttled.
    const auto unc = motion_commander.last_uncertainty_diag();
    const float pose_sigma_m = unc.valid ? unc.xy_std_m : 0.f;
    // ★ Heading and COMMANDED rotation, so the judge can see a robot that obeys every order and still
    // arrives nowhere: adv 0, rot at its cap, heading sweeping. Passing zeros here would compile and
    // silently leave the Spinning verdict unreachable — the detector would exist and never fire.
    if (detect_stuck(/*pursuing=*/true, asked_lin_mps, cmd_lin, pose_sigma_m,
                     robot_pose.pos, robot_pose.theta, rot_rps, time_source()))
    {
        begin_escape(robot_pose, obstacle_tracker, path_controller, time_source());
        step_escape(robot_pose, path_controller, motion_commander, time_source());
        return;
    }

    // MPPI produced no usable motion this cycle but we are not yet confirmed-wedged: hold the base.
    // ★★REVERTED 2026-08-19, SAME DAY, ON A LIVE STALL. Plan §1.2 removed the `path_controller.stop()`
    // here, on the argument that a single quiet cycle should not destroy a whole traversal — which is
    // true, and is still one of the three churn mechanisms. But the safety argument attached to it was
    // WRONG, and this is the exact hole:
    //   this branch fires when adv AND rot are BOTH ~0, and StallJudge opens no window in that case by
    //   design — `not (asked_mps > 0 or |commanded_rot| > 0)` resets it and returns (stall_judge.h).
    //   A robot commanded EXACTLY NOTHING is invisible to the stall detector, deliberately: nothing is
    //   predicted, so nothing can be contradicted. So neither the wedge nor Spinning can fire here, and
    //   the commit message claiming Spinning covers it was simply mistaken. Removing stop() removed the
    //   ONLY thing that broke a zero-command state, and `epistemic_gain` leaving the identity test
    //   (same commit) removed the accidental second rescue — a per-cycle replan.
    // The user reports the robot stalled with zero velocities commanded on the first run with this
    // build, on a controller that had run for weeks. Restored to the known-good behaviour.
    // ★The churn this re-introduces is real and still wants fixing — but it wants a BOUNDED version
    // (retire the plan after N consecutive quiet cycles, not after one), designed and benched offline,
    // not a valve removed live. Do not re-remove this line without that.
    if (std::abs(adv_mps) < 5e-4f && std::abs(side_mps) < 5e-4f && std::abs(rot_rps) < 1e-3f)
    {
        path_controller.stop();
        note_no_command(); motion_commander.stop_robot();
        return;
    }

    motion_commander.send_speed_command(adv_mps, side_mps, rot_rps);
}

// ─── Live tracking error ──────────────────────────────────────────────────────
// One cumulative RMS of the TRACKER's own cross-track error, plus the rotation effort per metre it
// spent holding it. Accumulated whenever the controller is driving, so the L-adaptation panel reads
// the same objective under a mission, a clicked target and an affordance — the deviation is a
// property of following a curve, and every one of those hands the tracker a curve.
void ControllerSession::sample_live_tracking(bool driving, const ControllerRobotPose &robot_pose,
                                             float cross_track_m, float rot_rps, std::uint64_t now_ms)
{
    if (not driving)                     // idle: keep the last curve on screen, arm the next reset
    {
        live_active_ = false;
        return;
    }
    if (not live_active_)                // rising edge — a new target is a new curve, not a continuation
    {
        live_active_ = true;
        live_ct_sq_sum_ = live_rot_effort_rad_ = live_dist_m_ = 0.0;
        live_ct_n_ = 0;
        live_ct_max_m_ = 0.f;
        live_last_pos_.reset();
        live_last_ms_ = now_ms;
    }

    // Same integration guard as MissionRunner::sample: after a stall, integrating a stale omega over a
    // large dt fabricates turning that never happened.
    const float dt = std::clamp(static_cast<float>(now_ms - live_last_ms_) / 1000.f, 0.f, 0.5f);
    live_last_ms_ = now_ms;
    if (live_last_pos_.has_value()) live_dist_m_ += (robot_pose.pos - *live_last_pos_).norm();
    live_last_pos_ = robot_pose.pos;
    live_rot_effort_rad_ += std::abs(rot_rps) * dt;

    // NaN is "no projection this cycle", not "no error" — skipped, never counted as a zero.
    if (std::isfinite(cross_track_m))
    {
        const float e = std::abs(cross_track_m);
        live_ct_sq_sum_ += static_cast<double>(e) * e;
        ++live_ct_n_;
        live_ct_max_m_ = std::max(live_ct_max_m_, e);
    }
}

ControllerSession::TrackingLive ControllerSession::live_tracking() const
{
    TrackingLive t;
    // While a mission runs, report the GRADED accumulator: the live curve and the number the run is
    // scored with at STOP then cannot disagree, which is the whole point of plotting it live.
    if (mission_.running())
    {
        const auto js = mission_.summary();
        t.cross_rms_m = js.cross_track_rms_m;
        t.cross_max_m = js.cross_track_max_m;
        t.rot_per_m   = js.distance_m > 1.f ? js.rot_effort_rad / js.distance_m : 0.f;
        t.on_mission  = true;
        return t;
    }
    if (live_ct_n_ > 0)
        t.cross_rms_m = static_cast<float>(std::sqrt(live_ct_sq_sum_ / static_cast<double>(live_ct_n_)));
    t.cross_max_m = live_ct_max_m_;
    // Same 1 m floor as the mission's: below it the ratio is dominated by the first few centimetres of
    // a start-up pivot and reads as a spike that means nothing.
    t.rot_per_m = live_dist_m_ > 1.0 ? static_cast<float>(live_rot_effort_rad_ / live_dist_m_) : 0.f;
    return t;
}

// ─── Lock-on micro-search ─────────────────────────────────────────────────────

// The perception producer's monotonic frame counter, so the servo can wait for looks taken AFTER the
// base stopped instead of guessing the pipeline's latency with a fixed dwell. -1 when unavailable, and
// the servo then falls back to that dwell — the controller must not need the retina to run.
// A COUNTER, not a timestamp, on purpose: mask_timestamp_ms is the camera's clock and this is the
// controller's, and nothing in the fleet guarantees the two agree. A counter needs no such agreement.
// mask_frame_id advances on every published camera frame even when the scene is empty, so "2 new
// frames" means two genuine looks, not two detections.
// One pass over the newest masks frame: did it carry a mask of the object this dwell is waiting on?
// Counts at most once per PRODUCER FRAME (guarded on mask_frame_id), so the number means "separate
// looks that confirmed it" and not "control cycles that happened to run while one frame sat there" —
// at 20 Hz against a ~15 Hz producer the second would reach 5 off a single sighting.
void ControllerSession::count_dwell_mask_hits()
{
    if (not graph_ or dwell_feedback_node_ == 0 or dwell_goal_.empty())
        return;
    // PACED BY THE PRODUCER'S FRAME COUNTER, JUDGED BY THE CONTRACT'S OWN PREDICATE.
    // The counter supplies the rhythm — one look per published camera frame, so the number means
    // "separate looks that confirmed it" and not "control cycles that ran while one frame sat there"
    // (at 20 Hz against a ~15 Hz producer the second reaches 5 off a single sighting).
    // The predicate supplies the verdict, and it is the producer's, not ours: `<class>_detection_alive`
    // on the object's own node, the same clause the servo used to declare the look successful. The
    // previous version matched YOLO class labels against the graph node NAME, which is unsatisfiable by
    // construction whenever the object is not named after a COCO class — so the robot stood waiting for
    // a confirmation that could not arrive while the mask it wanted was plainly on screen.
    const auto masks = graph_->get_node("masks");
    if (not masks.has_value())
        return;
    const auto frame = graph_->get_attrib_by_name<mask_frame_id_att>(masks.value());
    if (not frame.has_value() or frame.value() == dwell_last_mask_frame_)
        return;
    dwell_last_mask_frame_ = frame.value();

    const auto feedback = graph_->get_node(dwell_feedback_node_);
    if (not feedback.has_value())
        return;
    if (rc::affordance::evaluate_goal(feedback.value(), dwell_goal_))
        ++dwell_mask_hits_;
}

int ControllerSession::masks_frame_seq() const
{
    if (not graph_)
        return -1;
    const auto node = graph_->get_node("masks");
    if (not node.has_value())
        return -1;
    const auto id = graph_->get_attrib_by_name<mask_frame_id_att>(node.value());
    return id.has_value() ? id.value() : -1;
}

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
            overlay_csv_.imbue(std::locale::classic());  // decimal POINT regardless of LANG (CLAUDE.md)
            if (overlay_csv_.is_open())
                overlay_csv_ << "t_ms,lidar_ts,gap_ms,pose_age_ms,vx,vy,omega,RTdelta_m,"
                                "cmd_adv,cmd_rot,cur_adv,cur_rot,rt_lead_ms,rt_fix_dt_ms,"
                                "twist_pred_dt_ms,twist_pred_err_m,twist_pred_err_deg,"
                                // ★THE PER-CONSUMER POSE SPLIT, MEASURED. How far the FRESH pose given
                                // to the control law sits from the SCAN-ALIGNED one everything else
                                // uses — i.e. the correction the split actually applies, in metres.
                                // 0 means the tree had nothing fresher and the two are the same pose,
                                // which is also what this column reads when TrackerUsesLatestPose is
                                // false. Multiply by the (1/L^2)=2.78 rad/s per metre feedback gain to
                                // read it as the demand the loop is no longer making on stale evidence.
                                "tracker_pose_lead_m\n";
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
            overlay_csv_ << ',' << tracker_pose_lead_m_;
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
    const auto cmd = lockon_.update(reading, met, time_source(), masks_frame_seq());
    // Quasi-static: drive the (tiny, capped) nudge only during STEP; hold still while SETTLING/done.
    if (lockon_.active() && (cmd.adv != 0.0f || cmd.side != 0.0f || cmd.rot != 0.0f))
        motion_commander.send_speed_command(cmd.adv, cmd.side, cmd.rot);
    else
        motion_commander.stop_robot();
    if (lockon_.done())
    {
        last_look_succeeded_ = lockon_.locked();
        qInfo() << "[affordance] lock-on" << (lockon_.locked() ? "LOCKED" : "GIVE_UP")
                << "| node" << active_target_id_;
        // ★STAMP THE OUTCOME INTO THE RETAINED VIEW. The panel is built at the TOP of the cycle from
        // the previous cycle's lock-on state, and the caller finalizes (and lockon_.reset()s) the
        // moment this returns — so the last snapshot ever written showed the servo still running, and
        // "hold stable" could never render as anything but Active however well the look went. The row
        // that reports the outcome was structurally incapable of reporting it.
        using S = rc::AffordanceStepView::State;
        const bool ok = lockon_.locked();
        affordance_view_.phase = ok ? "locked" : "gave up";
        for (auto &s : affordance_view_.steps)
            if (s.kind == rc::AffordanceStepView::Kind::Stable
                or s.kind == rc::AffordanceStepView::Kind::ServoLoop)
            {
                s.state = ok ? S::Done : S::Failed;
                s.progress = ok ? 1.f : s.progress;
                if (ok and s.kind == rc::AffordanceStepView::Kind::Stable)
                    s.detail = std::format("{} of {} consecutive — HELD",
                                           lockon_.stable(), lockon_.stable_needed());
                if (not ok and s.blocked_why.empty())
                    s.blocked_why = "timed out before the predicate held";
            }
    }
    return lockon_.done();
}

bool ControllerSession::step_orient(const ControllerRobotPose &robot_pose,
                                    ControllerMotionCommander &motion_commander,
                                    const TimeSource &time_source, float target_yaw)
{
    const std::uint64_t now = time_source();
    if (!orient_start_ms_)
        {
            orient_start_ms_ = now;
            const float e0 = std::atan2(std::sin(target_yaw - robot_pose.theta),
                                        std::cos(target_yaw - robot_pose.theta));
            orient_start_err_rad_ = std::abs(e0);
        }

    // ★ONE DEFINITION OF "POINTING THERE", NOT TWO. This band already lived further down as the point
    // where the base stops rotating and holds still so the capture is quiet; making it also the
    // completion test for a bearing-only Orient keeps a single number instead of a second one free to
    // drift away from it.
    constexpr float kOrientAlignedRad = 0.05f;
    const float yaw_err = std::atan2(std::sin(target_yaw - robot_pose.theta),
                                     std::cos(target_yaw - robot_pose.theta));
    const bool aligned = std::abs(yaw_err) < kOrientAlignedRad;

    // ★WHAT COMPLETES AN ORIENT DEPENDS ON WHETHER IT ASKED FOR ANYTHING BEYOND THE ROTATION.
    // With a predicate — a glance, "turn that way until the detector fires" — the predicate is the
    // completion and the rotation merely serves it; that is the only Orient anyone has authored so far
    // and its behaviour is unchanged.
    // With NO predicate the rotation IS the affordance, and the completion is arriving at the bearing.
    // ★This is a fix, not a new mode: evaluate_goal on an empty clause list returns true, so before
    // this an empty-predicate Orient satisfied its goal on the FIRST cycle, standing still, and
    // reported LOOKED without having turned. A producer sequencing turns off those completions would
    // have counted a whole pivot in half a second and believed the odometry closed perfectly.
    const bool bearing_is_the_goal = active_contract_.goal.empty();
    if (bearing_is_the_goal ? aligned : goal_met(feedback_node_id_)) ++orient_stable_; else orient_stable_ = 0;
    const bool looked    = orient_stable_ >= std::max(1, active_contract_.stable_n);
    const bool timed_out = static_cast<double>(now - *orient_start_ms_) > active_contract_.timeout_ms;
    if (looked || timed_out)
    {
        motion_commander.stop_robot();
        last_look_succeeded_ = looked;
        qInfo() << "[affordance] orient" << (looked ? "LOOKED (detection)" : "GIVE_UP (timeout)")
                << "| node" << active_target_id_;
        orient_start_ms_.reset();
        orient_stable_ = 0;
        return true;
    }

    // Rotate the base toward the target bearing (capped). Once nearly aligned, HOLD STILL so the look is
    // motion-free (the Orient contract's .still asks for a quiet capture) and wait for the detection.
    const float k = params_ ? params_->lockon_k_yaw : 0.8f;
    // ★THE PRODUCER MAY DECLARE THE RATE, because it knows what the manoeuvre is FOR and we do not.
    // Our own lockon_max_yaw_rps was tuned for the LockOn micro-search, where creeping protects the
    // masks being collected; a calibration pivot observes nothing while it turns, so the same cap is
    // pure cost there. Both manoeuvres used to share this constant, and the pivot's producer -- with
    // no way to say otherwise -- priced a 50 s detour that actually took 7 minutes.
    // ★THE CEILING IS THE BASE'S LIMIT, NOT THE SERVO'S TUNING -- and that distinction is the actual
    // bug. lockon_max_yaw_rps is a CHOICE (creep so the masks stay sharp); max_rot_speed_rps is what
    // the machine can safely do. Clamping a producer's request to the servo tuning would keep every
    // manoeuvre at 0.06 and change nothing; clamping it to nothing at all would let a contract
    // command a base past its limits. So: the producer names a rate, we clamp it to the HARDWARE
    // ceiling, and our servo tuning applies only when no rate was asked for.
    const float servo_cap = params_ ? params_->lockon_max_yaw_rps : 0.12f;
    const float base_cap  = params_ ? params_->max_rot_speed_rps  : 0.8f;
    const float asked     = active_contract_.max_yaw_rate;
    const float cap       = (asked > 0.f) ? std::min(asked, base_cap) : servo_cap;
    float rot = std::clamp(k * yaw_err, -cap, cap);
    if (aligned)
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
                                         std::uint64_t now_ms,
                                         bool allow_dwell,
                                         std::optional<rc::affordance::Outcome> outcome_override,
                                         std::source_location floc)
{
    last_finalize_line_ = floc.line();
    // ★★★WHO COMPLETED IT, AND FROM HOW FAR. Six callers reach here; three are outside the arrival
    // branch. The starvation returns because something completes an affordance the robot never drove
    // to, which then makes the (correct) just-completed suppression block the very cell it should
    // take: 78% of cycles rejected, 92 s with no base command. This names the caller and the distance.
    {
        static std::ofstream fj;
        static bool ok = false;
        if (not ok) { fj.open("finalize_sites.jsonl", std::ios::out | std::ios::trunc);
                      fj.imbue(std::locale::classic()); ok = fj.is_open(); }
        if (ok)
        {
            // `arrived_at` is the pose the caller says the robot completed at — the honest source here.
            const float d = last_target_info_.has_value()
                ? (last_target_info_->room_pos - arrived_at).norm() : -1.f;
            fj << std::format(R"({{"caller_line":{},"d_to_target":{:.3f},"target":"{}","dwell":{}}})" "\n",
                              floc.line(), d,
                              last_target_info_.has_value() ? last_target_info_->node_name : std::string{},
                              allow_dwell ? 1 : 0);
            fj.flush();
        }
    }
    // A mission waypoint is reached the same way any target is; stepping the mission here means arrival
    // logic exists in exactly one place and a mission cannot drift out of sync with what the robot did.
    const bool mission_continues = false;   // a mission is ended by arc length, never by an arrival
    // ARM THE DWELL BEFORE mark_reached CLEARS THE AFFORDANCE — after it there is nothing left to say
    // which object this was. Only for an affordance: a clicked target is not an epistemic action and
    // has no acquisition to inspect, so holding the robot after one would just be a delay.
    const bool was_affordance = affordance_manager.has_current();
    // ── DOES THIS AFFORDANCE ACTUALLY NEED A MASK? ────────────────────────────────────────────────
    // Ask the CONTRACT, do not assume. Every look-shaped contract in the fleet states the requirement
    // outright — its completion predicate contains a `<class>_detection_alive` clause, which is the
    // producer saying "this affordance is finished when a detection of my object is live". An
    // affordance with no such clause (a Reach, a pose-only move) has no acquisition to wait for, and
    // making it wait anyway just parks the robot for the bound — which is exactly what it did.
    // The match is on the established attribute vocabulary rather than a dedicated flag; when the
    // contract grows an explicit one, this is the single line that moves to it.
    const bool contract_wants_detection =
        std::ranges::any_of(active_contract_.goal, [](const rc::affordance::GoalClause &c)
                            { return c.attr.find("detection") != std::string::npos; });
    if (allow_dwell and was_affordance and params_ and params_->affordance_dwell_ms > 0.f)
    {
        affordance_dwell_until_ms_ = now_ms + static_cast<std::uint64_t>(params_->affordance_dwell_ms);
        // The bound is on the WHOLE wait, and never shorter than the clock — a max below the floor
        // would silently cancel the dwell it is supposed to bound.
        affordance_dwell_deadline_ms_ =
            std::max(affordance_dwell_until_ms_,
                     now_ms + static_cast<std::uint64_t>(std::max(params_->affordance_dwell_max_ms,
                                                                  params_->affordance_dwell_ms)));
        dwell_logged_ = false;
        dwell_object_ = affordance_view_.object;
        dwell_mask_hits_ = 0;
        dwell_last_mask_frame_ = -1;
        // ★AND ONLY WHEN THE LOOK ACTUALLY SUCCEEDED. A contract that GAVE UP gave up because the
        // detection never held; standing there waiting for five of them afterwards waits for the one
        // thing just demonstrated not to be coming, and burns the whole bound doing it.
        // Captured BEFORE the teardown below clears feedback_node_id_ and last_target_info_.
        dwell_goal_ = active_contract_.goal;
        dwell_feedback_node_ = feedback_node_id_ != 0
                                   ? feedback_node_id_
                                   : (last_target_info_.has_value() ? last_target_info_->parent_node_id : 0);
        dwell_wants_mask_ = contract_wants_detection and last_look_succeeded_
                        and dwell_feedback_node_ != 0 and not dwell_goal_.empty();
        dwell_last_log_ms_ = now_ms;
    }
    if (graph_)
    {
        // ★ WHAT ACTUALLY HAPPENED, not "we got here". last_look_succeeded_ is the contract's own
        // verdict — the completion predicate held for stable_n cycles — and it is already what decides
        // whether the dwell is worth arming just above. Arriving at the standpoint is not the same as
        // observing anything: a contract that ran to timeout arrives too, and used to be reported to
        // the producer identically to one that succeeded.
        // ★ ASK WHETHER THERE WAS A PREDICATE TO FAIL. last_look_succeeded_ is set ONLY by the servo
        // lock-on and orient paths; a Reach contract never touches it, so it is false on every
        // successful Reach arrival. Room affordances are policy=Reach with an EMPTY goal — arriving IS
        // the completion, `evaluate_goal` returns true trivially for an empty clause set — and reading
        // the look flag for them reported `timeout` for four consecutive arrivals, one of them logged
        // at d=0.23 m. A contract with no predicate cannot fail one.
        // ★AN ORIENT ALWAYS HAS A PREDICATE: THE BEARING. The empty-goal exemption above is right for a
        // Reach -- arriving IS the completion and there is nothing that could have failed. An Orient
        // with no goal clause is NOT that case: step_orient evaluates `aligned` as its predicate and
        // writes last_look_succeeded_ from it, so it can and does fail -- it can run out of patience
        // short of the bearing. Falling into `not had_predicate` reported those as `satisfied`, which
        // is the one thing the producer must not be told: the calibration pivot advances its sequence
        // on that word, and a step that never turned would be logged as a step that did.
        const bool had_predicate = not active_contract_.goal.empty()
                                or active_contract_.policy == rc::affordance::Policy::Orient;
        // outcome_override wins: a refusal is a statement about the APPROACH, and no reading of the
        // contract's predicate can express it — there was no approach to judge.
        {
            const auto oc = outcome_override.value_or(
                (not had_predicate or last_look_succeeded_) ? rc::affordance::Outcome::Satisfied
                                                            : rc::affordance::Outcome::Timeout);
            note_protocol(rc::AffordanceExecution::ProtocolLine::Side::Consumer, now_ms,
                          std::format("-> {} after {:.1f} s{}", rc::affordance::to_string(oc),
                                      (now_ms - affordance_started_ms_) / 1000.0,
                                      had_predicate ? "" : " (no predicate: arriving IS the goal)"));
        }
        affordance_manager.mark_reached(graph_,
                                        outcome_override.value_or(
                                            (not had_predicate or last_look_succeeded_)
                                                ? rc::affordance::Outcome::Satisfied
                                                : rc::affordance::Outcome::Timeout));
    }
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
    // The final-approach correction belongs to the standpoint just retired. Re-selecting the same
    // affordance later must re-measure, not inherit an answer taken against a scan that is now old.
    approach_fix_.reset();
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

bool ControllerSession::skip_current_affordance(rc::AffordanceManager &affordance_manager,
                                               rc::TrajectoryController &path_controller,
                                               ControllerMotionCommander &motion_commander,
                                               ControllerDisplay &display,
                                               const TimeSource &time_source)
{
    // A DWELL IS SKIPPABLE. It is the state most likely to be stuck — standing still waiting for a
    // confirmation that may never arrive — so the escape hatch has to reach it. Cancelling the wait is
    // the whole of the work here; the affordance it belonged to is already retired.
    if (affordance_dwell_until_ms_ != 0)
    {
        std::println("[affordance] dwell CANCELLED by operator after {}/{} confirming looks",
                     dwell_mask_hits_, params_ ? params_->affordance_dwell_mask_hits : 0);
        affordance_dwell_until_ms_ = 0;
        affordance_dwell_deadline_ms_ = 0;
        dwell_wants_mask_ = false;
        dwell_logged_ = false;
        affordance_view_.dwell_left_s = 0.f;
        affordance_view_.dwell_mask_needed = 0;
        display.set_affordance_execution(affordance_view_);
        return true;
    }

    const bool on_affordance = last_target_info_.has_value() and last_target_info_->from_affordance;
    if (not affordance_manager.has_current() and not on_affordance)
    {
        std::println("[affordance] SKIP ignored — no affordance is executing.");
        return false;
    }

    const std::string name = not affordance_view_.affordance.empty()
                                 ? affordance_view_.affordance
                                 : affordance_manager.current_name();
    std::println("[affordance] SKIPPED by operator: '{}' after {:.1f} s — retiring it and selecting the next.",
                 name.empty() ? std::string("?") : name, static_cast<double>(affordance_view_.elapsed_s));

    // Record the outcome HERE, and blank the live run so the automatic retirement in
    // update_affordance_view cannot file a second row for it — and cannot file it as "reached", which
    // is the one thing this outcome is not.
    affordance_recent_.insert(affordance_recent_.begin(),
        std::format("{} {:.1f}s skipped", name.empty() ? std::string("?") : name,
                    affordance_view_.elapsed_s));
    if (affordance_recent_.size() > 4) affordance_recent_.resize(4);
    affordance_view_.active = false;
    affordance_view_.phase = "skipped";
    affordance_view_.affordance.clear();
    affordance_view_.dwell_left_s = 0.f;
    affordance_view_.dwell_mask_needed = 0;
    affordance_view_.recent = affordance_recent_;
    display.set_affordance_execution(affordance_view_);

    lockon_.reset();
    // Same teardown a finished affordance gets — including mark_reached, which is what stops the
    // selector handing the identical affordance straight back — but WITHOUT the dwell: a skip means
    // "there is nothing here worth looking at", so holding still to look at it would invert the intent.
    // ★AND SAY WHAT IT WAS. Without an explicit outcome this falls through to the predicate reading,
    // which for an affordance whose predicate never ran reports whatever the LAST one left behind.
    // `afford_calib` is one node reused for every step of its pivot, so resolve_target_contract's
    // reset (keyed on node id) does not fire between steps: a skipped step would have inherited the
    // previous step's `true` and been reported SATISFIED, advancing a twelve-step sequence by a step
    // that never happened. Abandoned is the word for this and it already exists — "an operator or a
    // higher-priority interrupt ended it".
    finalize_reached(affordance_manager, path_controller, motion_commander, display,
                     Eigen::Vector2f::Zero(), time_source(), /*allow_dwell=*/false,
                     rc::affordance::Outcome::Abandoned);
    return true;
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

bool ControllerSession::detect_stuck(bool pursuing, float asked_lin_mps, float cmd_lin_mps,
                                    float pose_sigma_m, const Eigen::Vector2f &pos_room,
                                    float heading_rad, float commanded_rot_rps,
                                    std::uint64_t now_ms)
{
    if (!params_ || !params_->stuck_recovery_enabled) { reset_stuck_window(); return false; }

    // ── THE WINDOW IS OPENED BY THE ASK, AND THE VERDICT NAMES THE CAUSE ─────────────────────────
    // Both halves used to be one number and one branch, taken AFTER the pose-covariance limiter had
    // already scaled the command down — so the harder the limiter held the robot, the smaller the bar it
    // had to clear, and a robot frozen at 0.011 m/s was certified healthy every window. See stall_judge.h
    // for the measurement and for why "just use the pre-throttle ask" is not on its own a fix.
    const rc::StallJudge::Params jp{.slip_ratio = params_->stuck_slip_ratio,
                                    .confirm_ms = params_->stuck_confirm_ms};
    const auto r = stall_judge_.note(pursuing, asked_lin_mps, cmd_lin_mps, pose_sigma_m,
                                     pos_room, heading_rad, commanded_rot_rps, now_ms, jp);
    if (r.verdict == rc::StallVerdict::None) return false;

    if (r.verdict == rc::StallVerdict::Wedge)
    {
        std::println("[controller] WEDGE — commanded {:.2f} m of travel over {:.1f} s and achieved {:.3f} m "
                     "net ({:.0f}% of it, floor {:.0f}%). Escaping.",
                     r.commanded_m, r.window_s, r.achieved_m,
                     r.asked_m > 1e-6f ? 100.f * r.achieved_m / r.asked_m : 0.f,
                     100.f * params_->stuck_slip_ratio);
        log_stall_event("wedge", r, pos_room, pose_sigma_m, now_ms);
        return true;
    }

    // ── SPINNING: IT OBEYS EVERY ORDER, SWEEPS RADIANS, AND ARRIVES NOWHERE ──────────────────────
    // ★MEASURED, and the reason this branch exists rather than the verdict merely being defined:
    // without it a Spinning report FELL THROUGH to the throttle-stall block below and was written to
    // stall_events.csv as `throttle_stall` — which by design never escapes, so nothing recovered it.
    // Run 2026-08-19 09:05: 9 of the 12 logged "throttle_stall" events were this verdict wearing the
    // wrong label (all of them at (-0.31,-2.38) over 20.3 s, every one with asked_m = 0 and
    // delivered = 1 — a translation stall that asked for no translation is a contradiction on its
    // face). They also inflated `stall_throttled_s_` to 18.5 s, of which ~14 s never happened.
    // The robot sat there for 74 s of a 144 s run, commanded rotation at its cap on 80% of cycles,
    // sweeping 3-6 rad per 6 s window and netting 0.07-0.87 of it (2-29%).
    // It is a LIVELOCK, not an obstruction — but the response a wedge gets is the one it needs: change
    // the pose and force a replan. And on an affordance the escape is already capped
    // (kMaxEscapesPerAffordance = 3, see begin_escape), after which the standpoint is given up and the
    // selector moves on — which is the correct end for a place the body cannot turn in.
    if (r.verdict == rc::StallVerdict::Spinning)
    {
        std::println("[controller] SPINNING — commanded {:.2f} rad of heading over {:.1f} s and netted "
                     "{:.2f} rad of it ({:.0f}%, floor {:.0f}%), while moving {:.3f} m. Obeying every "
                     "order and arriving nowhere: escaping.",
                     r.asked_rot_rad, r.window_s, r.achieved_rot_rad,
                     r.asked_rot_rad > 1e-6f ? 100.f * r.achieved_rot_rad / r.asked_rot_rad : 0.f,
                     100.f * params_->stuck_slip_ratio, r.achieved_m);
        log_stall_event("spin", r, pos_room, pose_sigma_m, now_ms);
        return true;
    }

    // ── THROTTLE STALL: OUR OWN LIMITER STOPPED THE ROBOT ────────────────────────────────────────
    // Reported, never escaped. Reversing does not raise a speed limiter, so an escape here would fire,
    // end, find the limiter still floored and fire again — with a virtual obstacle dropped at every
    // round — which is the same closed loop "planner failure is not a wedge" exists to prevent.
    // It re-reports once per window on purpose: fifteen seconds of frozen must not read like one bad
    // window, and the accumulated total is what the end-of-run line quotes.
    stall_throttled_s_ += r.window_s;
    ++stall_throttled_windows_;
    std::println("[controller] ⚠STALLED BY OUR OWN SPEED LIMIT — the tracker asked for {:.2f} m over "
                 "{:.1f} s, the pose-covariance limiter let through {:.3f} m ({:.0f}% of it), and the robot "
                 "moved {:.3f} m net, which is inside its own position sigma of {:.3f} m. NOT escaping: "
                 "reversing cannot raise a speed limit. Total this run: {:.1f} s over {} window(s). "
                 "sigma is above PoseXYStdStop ⇒ look at the localiser, not at the controller.",
                 r.asked_m, r.window_s, r.commanded_m, 100.f * r.delivered, r.achieved_m, pose_sigma_m,
                 stall_throttled_s_, stall_throttled_windows_);
    log_stall_event("throttle_stall", r, pos_room, pose_sigma_m, now_ms);
    return false;
}

// One row per verdict. These are RARE events whose diagnosis previously existed only on stdout, where it
// scrolls away and cannot be compared between runs — the same reason route_events.csv exists. Archived
// with the run, so "how often did this happen, and where" becomes answerable across runs instead of
// requiring the afternoon it took the first time.
void ControllerSession::log_stall_event(const char *verdict, const rc::StallJudge::Report &r,
                                        const Eigen::Vector2f &pos, float sigma, std::uint64_t t_ms)
{
    if (not stall_events_csv_open_)
    {
        stall_events_csv_.open("stall_events.csv", std::ios::out | std::ios::trunc);
        // Locale-independent output: the C++ global locale is "C" today, but a single std::locale::global
        // anywhere in the process would start emitting decimal COMMAS into a file every reader parses as
        // points. Cheap insurance, applied to every data file this agent writes.
        stall_events_csv_.imbue(std::locale::classic());
        // ★asked_rot/achieved_rot are what makes a `spin` row READABLE. Without them a spin logs as
        // asked_m = 0, achieved_m ~ 0, delivered = 1 — numbers that carry no evidence at all, which is
        // how 9 of these hid inside the throttle-stall rows of the 2026-08-19 run.
        stall_events_csv_ << "t_ms,verdict,window_s,asked_m,commanded_m,achieved_m,delivered,"
                             "asked_rot_rad,achieved_rot_rad,pose_sigma_m,x,y,total_throttled_s\n";
        stall_events_csv_open_ = true;
        mission_.archive_on_stop("stall_events.csv");
    }
    if (not stall_events_csv_.is_open()) return;
    stall_events_csv_ << t_ms << ',' << verdict << ',' << r.window_s << ',' << r.asked_m << ','
                      << r.commanded_m << ',' << r.achieved_m << ',' << r.delivered << ','
                      << r.asked_rot_rad << ',' << r.achieved_rot_rad << ','
                      << sigma << ',' << pos.x() << ',' << pos.y() << ',' << stall_throttled_s_ << '\n';
    stall_events_csv_.flush();
}

void ControllerSession::reset_stuck_window()
{
    stall_judge_.reset();
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
    reset_stuck_window();

    // ── CHARGE THIS ESCAPE TO THE AFFORDANCE IT HAPPENED UNDER ───────────────────────────────────
    // An escape is a reflex, not a plan: it reverses out and lets the planner try again. That is right
    // ONCE. Repeated against the same standpoint it becomes a livelock — wedge, escape, retry, wedge —
    // and the robot spends minutes achieving a few centimetres while other affordances go unserved.
    // After kMaxEscapesPerAffordance the honest conclusion is that this approach is not available to
    // this body right now, so give the affordance up and let the selector choose another.
    if (last_target_info_.has_value() and last_target_info_->from_affordance and active_target_id_ != 0)
    {
        constexpr int kMaxEscapesPerAffordance = 3;
        if (active_target_id_ != escapes_target_id_)
        {
            escapes_target_id_ = active_target_id_;
            escapes_on_target_ = 0;
        }
        if (++escapes_on_target_ >= kMaxEscapesPerAffordance)
        {
            reject_affordance_id_ = active_target_id_;
            reject_affordance_name_ = last_target_info_->node_name;
        }
    }

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
    // The dwell included: it is a wait belonging to an affordance the operator has just abandoned, and
    // leaving it armed would hold the base still for seconds after an explicit Stop.
    affordance_dwell_until_ms_ = 0;
    affordance_dwell_deadline_ms_ = 0;
    dwell_mask_hits_ = 0;
    dwell_logged_ = false;
    route_active_ = false;
    route_repair_pending_ = false;
    current_plan_.reset();
    plan_spline_valid_ = false;   // the fitted curve belongs to the plan being dropped
    active_target_id_ = 0;
    last_target_info_.reset();
    // Band bookkeeping belongs to the route that is going away.
    band_cycle_ = 0;
}