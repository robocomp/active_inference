// controller_policy_drive.cpp — Task 8 of the local-motion policy: ONE authority drives the base.
//
// With [Policy] Enabled, execute_plan hands every cycle to drive_with_policy BEFORE the old chain (escape, returns backoff,
// anchor servo + jamb guards + make-room, lock-on and orient motion, the band, the tracker's command path with its caps,
// the uncertainty limiter and the clearance ceiling); none of those run, so none of them can write the base. What stays is
// the PROTOCOL: which goal the task layer asks for, when an arrival / a crossing is complete (consume_goal_reached,
// cross_protocol — the same code the old chain uses), and Unreachable through the affordance protocol.
//
// The policy itself is rc::policy::Navigator (local_policy/navigator.h): the SAME API the Task-6 closed-loop harness
// (tests/test_policy_scenarios.cpp) drives — set_goal on a new goal, step() every cycle with the returns, the base limits,
// the pose uncertainty and the clock; it commits to plans, judges them, lays traps only on failed predictions and reports
// Unreachable. This file builds its input from the controller's world (conversions: local_policy/goal_adapter.h) and sends
// its (v, w).
#include "controller_session.h"

#include <chrono>
#include <cmath>
#include <format>
#include <numbers>
#include <print>

namespace
{
double steady_now_s()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::uint64_t steady_now_ms()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
const char *goal_kind_name(rc::policy::Goal::Kind k)
{
    switch (k)
    {
        case rc::policy::Goal::Kind::Place: return "place";
        case rc::policy::Goal::Kind::Doorway: return "doorway";
        case rc::policy::Goal::Kind::Heading: return "heading";
    }
    return "?";
}
}  // namespace

// The base's measured motion (robot_concept writes robot_current_* from the base's FullPose odometry; rot is CCW-positive:
// room_concept's prediction integrates it as such). Read once per cycle where the world model is in scope.
void ControllerSession::read_base_odometry(const ControllerWorldModel &world_model, std::uint64_t)
{
    odo_valid_ = false;
    if (graph_ == nullptr) return;
    const auto rid = world_model.graph_state().robot_id;
    if (rid == 0) return;
    if (const auto node = graph_->get_node(rid); node.has_value())
    {
        const auto v = graph_->get_attrib_by_name<robot_current_advance_speed_att>(*node);
        const auto w = graph_->get_attrib_by_name<robot_current_angular_speed_att>(*node);
        if (v.has_value() and w.has_value() and std::isfinite(*v) and std::isfinite(*w))
        {
            odo_v_ = *v;
            odo_w_ = *w;
            odo_valid_ = true;
            // the producer's wall stamp of this sample (robot_concept: FullPose timestamp, wall ms — rt_twist.h)
            odo_stamp_ms_ = static_cast<std::uint64_t>(graph_->get_attrib_by_name<robot_current_speed_timestamp_att>(*node).value_or(0));
        }
    }
}

void ControllerSession::policy_reset()
{
    policy_goal_.reset();
    policy_goal_who_.clear();
    policy_unseen_.reset();
    policy_sent_.clear();
    policy_last_mode_ = -1;
    policy_last_why_.clear();
    policy_no_field_since_ms_ = 0;
    policy_gate_.reset();
    policy_sign_.reset();          // an operator Stop / a new activity re-arms the check (it is latched otherwise)
    policy_prev_yaw_.reset();
    policy_prev_pose_t_.reset();
    policy_odo_hist_.clear();
}

ControllerSession::PolicyTick ControllerSession::policy_tick(const rc::policy::GoalSpec &spec, const std::string &who,
                                                             const ControllerRobotPose &robot_pose,
                                                             ControllerObstacleTracker &obstacle_tracker,
                                                             ControllerMotionCommander &motion_commander, std::uint64_t now_ms,
                                                             std::function<bool(const Eigen::Vector2f &)> also_free,
                                                             std::function<bool(const Eigen::Vector2f &, float)> trap_veto)
{
    using rc::policy::Decision;
    PolicyTick out;
    const double now_s = steady_now_s();
    const auto &prm = params_ != nullptr ? *params_ : ControllerParams{};
    auto hold = [&](const std::string &why)
    {
        out.verdict = PolicyVerdict::Held;
        out.why = why;
        note_why(rc::why::Cause::PolicyHold, std::format("{}: {}", who, why));
        note_no_command(); motion_commander.stop_robot();
        policy_sent_.push_back({now_s, 0.f, 0.f});
        policy_gate_.on_hold();   // review fix 2.1: a hold — however short — is an INTERRUPTION of the committed plan
    };
    if (not grid_planner_.has_world()) { hold("no planner map yet — no cost-to-go"); return out; }
    // ★ Live fix 2 — A ROOM-FRAME CHANGE (policy_tick_gate.h RoomFrameGuard): everything held is in the old frame — reset
    //   it (goal, field, plan, evidence), and hold until a scan accepted after the change (registered in the new frame).
    if (policy_frame_.on_room(current_room_name_, steady_now_ms()))
    {
        std::println("[policy] {}: the room frame changed to '{}' — goal, field, plan and evidence reset; waiting for a scan "
                     "registered in the new frame", who, current_room_name_);
        policy_reset();
    }
    if (not policy_frame_.ready(obstacle_tracker.last_scan_accept_steady_ms()))
    {
        hold(std::format("Blind — the room frame changed to '{}'; the contact returns are still the old frame's", current_room_name_));
        return out;
    }
    // ★ review M6: ODOMETRY FRESHNESS — v_now / w_now and the unseen-contact evidence come from robot_current_*; older than
    //   Controller.CommandFreshnessTauMs (the age past which the base itself treats a command as stale) they describe a
    //   motion that is over: Blind, hold.
    {
        const auto wall_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const float tau = std::max(1.f, prm.command_freshness_tau_ms);
        if (not odo_valid_ or odo_stamp_ms_ == 0 or (wall_now > odo_stamp_ms_ and float(wall_now - odo_stamp_ms_) > tau))
        {
            const std::string why = not odo_valid_ or odo_stamp_ms_ == 0 ? "Blind — odometry stale (no robot_current_* speeds)"
                : std::format("Blind — odometry stale ({} ms old > CommandFreshnessTauMs {:.0f})", wall_now - odo_stamp_ms_, tau);
            if (why.substr(0, 30) != policy_last_why_.substr(0, 30)) std::println("[policy] {}: {}", who, why);
            policy_last_why_ = why;
            policy_last_mode_ = -1;
            hold(why);
            return out;
        }
    }
    // ★ review I5: a turn sign the pose contradicts (OdoSignCheck, latched) — Blind: do not drive on a mirrored world
    if (policy_sign_.contradicted())
    {
        const std::string why = "Blind — odometry turn sign contradicts the pose (robot_current_angular_speed vs the heading change, "
                                "3 turning intervals in a row) — check the base/bridge sign";
        if (why != policy_last_why_) std::println("[policy] {}: {}", who, why);
        policy_last_why_ = why;
        policy_last_mode_ = -1;
        hold(why);
        return out;
    }

    // ── the BODY: contact is judged against the BARE hull (contact.cpp); the planner's safety margin is a route preference
    policy_body_ = approach_body_;
    policy_body_->set_safety_margin(0.f);
    const rc::RobotFootprint &body = *policy_body_;

    // ── the pose (the freshest one, as the tracker's control law used) — the ONLY quarter turn is to_policy_pose
    // ★ review C1: the pose is the CALLER's — the freshest it has (drive_with_policy: tracker_pose_ else the scan-aligned one;
    //   a move order: read_robot_pose_latest this cycle). Never a member that only some paths refresh.
    const rc::policy::Pose2 pose = rc::policy::to_policy_pose(robot_pose);

    // ── base limits: the policy's speeds never above the capability; accelerations from it; the deadband measured
    rc::policy::BaseLimitsInputs bi;
    bi.v_max_policy = prm.max_adv_speed_mps;
    bi.w_max_policy = prm.max_rot_speed_rps;
    bi.cap_v_max = base_capability_.max_linear_speed_mps;
    bi.cap_w_max = base_capability_.max_rot_speed_rps;
    bi.cap_a_lin = base_capability_.max_linear_accel_mps2;
    bi.cap_d_lin = base_capability_.max_linear_decel_mps2;
    bi.axes_length_m = base_capability_.axes_length_m;
    bi.cap_v_min = base_capability_.min_linear_speed;
    bi.cap_w_min = base_capability_.min_rot_speed;
    bi.cfg_v_min = prm.policy_min_linear_speed_mps;
    bi.cfg_w_min = prm.policy_min_rot_speed_rps;
    if (prm.policy_wheel_decel_mps2 > 0.f) bi.measured_wheel_decel = prm.policy_wheel_decel_mps2;
    const rc::policy::BaseLimits limits = rc::policy::base_limits(bi);

    rc::policy::PolicyParams pp;
    pp.horizon_s = prm.policy_horizon_s;
    pp.risk_eps = prm.policy_risk_eps;
    pp.release_risk_max = prm.policy_release_risk_max;
    pp.progress_scale_m = prm.policy_progress_scale_m;

    // ── the cost-to-go: free = the planner's distance field > the body's inscribed radius (the centre can be there at
    //    some heading); extra = the residual belief as a per-METRE density: residual_risk_weight x the nats the BODY covers
    //    (GridPlanner::body_risk_at — the A*'s own per-metre charge; see grid_planner.h for the conversion)
    const float r_ins = body.inscribed_radius();
    auto free_fn = [this, r_ins, also_free](const Eigen::Vector2f &p)
    { return rc::policy::field_free_from_distance(grid_planner_.distance_at(p), grid_planner_.cell_size(), r_ins) or (also_free and also_free(p)); };
    auto extra_fn = [this](const Eigen::Vector2f &p)
    { return std::max(0.f, grid_planner_.params.residual_risk_weight) * grid_planner_.body_risk_at(p); };
    const bool new_goal = not policy_goal_.has_value() or policy_goal_who_ != who or not rc::policy::same_goal(*policy_goal_, spec);
    if (new_goal)
    {
        // The field's grid: the planner's extent at the harness's validated 0.05 m resolution (the field's own resolution
        // bounds what the progress monitor can call progress — CostToGo::resolution_m).
        constexpr float kFieldCell = 0.05f;
        const float span_x = static_cast<float>(grid_planner_.width()) * grid_planner_.cell_size();
        const float span_y = static_cast<float>(grid_planner_.height()) * grid_planner_.cell_size();
        const Eigen::Vector2f origin = grid_planner_.grid_origin() - 0.5f * Eigen::Vector2f::Constant(grid_planner_.cell_size());
        const rc::policy::CostToGo::Grid g{origin, kFieldCell, static_cast<int>(std::ceil(span_x / kFieldCell)) + 1,
                                           static_cast<int>(std::ceil(span_y / kFieldCell)) + 1};
        policy_nav_.set_goal(spec.goal, g, free_fn, extra_fn, spec.points, body.circumscribed_radius());
        policy_goal_ = spec;
        policy_goal_who_ = who;
        policy_world_hash_ = grid_planner_.world_hash();
        // (the unseen-contact evidence is KEPT across goals: it is about what the body touched, in the odometry frame)
        policy_last_mode_ = -1;
        std::println("[policy] goal for {}: {} at ({:.2f},{:.2f}){} — field built in {:.1f} ms, {:.2f} m from the robot", who,
                     goal_kind_name(spec.goal.kind), spec.goal.p.x(), spec.goal.p.y(),
                     spec.goal.yaw ? std::format(" facing {:.0f} deg", *spec.goal.yaw * 57.29578f) : std::string(),
                     policy_nav_.last_build_ms(), policy_nav_.field().at(pose.p));
    }
    // ★ Live fix 2 ruling 4 — DIAGNOSTIC: a new goal whose field at the robot exceeds 10x the straight-line distance is
    //   dumped (free mask summary, traps, residual density stats, goal points) to a file named in the log.
    if (new_goal)
    {
        float straight = std::numeric_limits<float>::infinity();
        for (const auto &q : spec.points) straight = std::min(straight, (q - pose.p).norm());
        const float f_robot = policy_nav_.field().at(pose.p);
        if (std::isfinite(straight) and (not std::isfinite(f_robot) or f_robot > 10.f * std::max(straight, 0.05f)))
        {
            const std::string fname = std::format("policy_field_dump_{}.txt", now_ms);
            std::ofstream f(fname, std::ios::out | std::ios::trunc);
            f.imbue(std::locale::classic());
            const float span_x = static_cast<float>(grid_planner_.width()) * grid_planner_.cell_size();
            const float span_y = static_cast<float>(grid_planner_.height()) * grid_planner_.cell_size();
            const Eigen::Vector2f o = grid_planner_.grid_origin() - 0.5f * Eigen::Vector2f::Constant(grid_planner_.cell_size());
            long n = 0, n_free = 0, n_risk = 0; double sum_risk = 0.0; float max_risk = 0.f;
            for (float y = o.y(); y <= o.y() + span_y; y += 0.05f)
                for (float x = o.x(); x <= o.x() + span_x; x += 0.05f)
                {
                    const Eigen::Vector2f q(x, y);
                    ++n;
                    if (not free_fn(q)) continue;
                    ++n_free;
                    const float e = extra_fn(q);
                    if (e > 0.f) { ++n_risk; sum_risk += e; max_risk = std::max(max_risk, e); }
                }
            f << "who " << who << "\nroom " << current_room_name_ << "\nrobot " << pose.p.x() << ' ' << pose.p.y() << ' ' << pose.yaw
              << "\nfield_at_robot " << f_robot << "\nstraight_m " << straight
              << "\ngoal_kind " << goal_kind_name(spec.goal.kind) << " p " << spec.goal.p.x() << ' ' << spec.goal.p.y()
              << "\ngrid origin " << o.x() << ' ' << o.y() << " span " << span_x << ' ' << span_y << " cell 0.05"
              << "\nfree_cells " << n_free << " of " << n << " ; free_at_robot " << (free_fn(pose.p) ? 1 : 0)
              << " ; also_free_at_robot " << (also_free and also_free(pose.p) ? 1 : 0)
              << "\nrisk_density cells>0 " << n_risk << " mean " << (n_risk > 0 ? sum_risk / double(n_risk) : 0.0) << " max " << max_risk
              << " (metres per metre, residual_risk_weight x body_risk_at)\n";
            for (const auto &q : spec.points)
                f << "goal_point " << q.x() << ' ' << q.y() << " free " << (free_fn(q) ? 1 : 0) << " field " << policy_nav_.field().at(q) << '\n';
            for (const auto &t : policy_nav_.traps()) f << "trap " << t.c.x() << ' ' << t.c.y() << " r " << t.r << '\n';
            // Live fix 3: the density where it matters, and the straight line robot -> nearest goal point every 0.10 m
            // (free flag, residual density, field) — enough to tell a disconnected free mask from a priced region
            f << "extra_at_robot " << extra_fn(pose.p) << " ; planner_distance_at_robot " << grid_planner_.distance_at(pose.p)
              << " ; residual_risk_weight " << grid_planner_.params.residual_risk_weight << '\n';
            Eigen::Vector2f gq = spec.points.front();
            for (const auto &q : spec.points) if ((q - pose.p).norm() < (gq - pose.p).norm()) gq = q;
            f << "extra_at_goal " << extra_fn(gq) << " ; planner_distance_at_goal " << grid_planner_.distance_at(gq) << '\n';
            const float len = (gq - pose.p).norm();
            for (float u = 0.f; u <= len + 1e-4f; u += 0.10f)
            {
                const Eigen::Vector2f q = pose.p + (len > 0.f ? u / len : 0.f) * (gq - pose.p);
                f << "line " << u << ' ' << q.x() << ' ' << q.y() << " free " << (free_fn(q) ? 1 : 0) << " dist " << grid_planner_.distance_at(q)
                  << " extra " << extra_fn(q) << " field " << policy_nav_.field().at(q) << '\n';
            }
            std::println("[policy] {}: the field at the robot ({:.2f} m) is > 10x the straight distance ({:.2f} m) — field inputs "
                         "dumped to {}", who, f_robot, straight, fname);
        }
    }
    policy_nav_.set_trap_veto(trap_veto);   // Live fix 2 ruling 2 (after set_goal, which clears it); empty = none
    if (new_goal) {}
    else if (grid_planner_.world_hash() != policy_world_hash_)
    {
        // the planner's map moved under the goal: applied by the Navigator between plans (navigator.h update_world)
        policy_nav_.update_world(free_fn, extra_fn);
        policy_world_hash_ = grid_planner_.world_hash();
    }

    // ── odometry: the wheels' measured speeds integrated (never relocalised) — the frame UnseenContact keeps its evidence in
    // ★ A GAP IN THE POLICY'S OWN TICKS (Halted / Pause / a move order / a probe owned the base) breaks the send history: what
    //   reached the wheels then is not in policy_sent_ (a Pause stops the base without telling the policy), so the "commands
    //   in force" would read a stopped robot as BLOCKED. Beyond the longest interval a tick can legitimately take (0.5 s, the
    //   clamp of the odometry step) the history restarts: no unseen-contact update this tick, the odometry re-anchored — and
    //   the evidence kept in the old odometry frame dropped with it (the base may have moved unseen by this integration).
    // ★ review I1: an explicit INTERRUPTION (policy_interrupt(): Pause / Halt / a hold / the LiDAR-stall hold) is the same
    //   event, and the Navigator's committed plan and pending prediction are dropped unjudged (Navigator::interrupt).
    const double prev_tick_s = policy_gate_.last_tick_s();
    const bool resumed = policy_gate_.on_tick(now_s);   // a hold / Halt / stall noted, the first tick, or a gap (policy_tick_gate.h)
    const float dt_tick = resumed ? 0.f : static_cast<float>(std::max(0.0, now_s - prev_tick_s));
    const float v_meas = odo_valid_ ? odo_v_ : 0.f, w_meas = odo_valid_ ? odo_w_ : 0.f;
    if (resumed)
    {
        policy_nav_.interrupt();   // drop the committed plan and its prediction UNJUDGED (an interruption is not evidence)
        policy_sent_.clear(); policy_odom_.reset(); policy_unseen_.reset();
        policy_odo_hist_.clear(); policy_prev_pose_t_.reset();
    }
    // ── I5 evidence, TIME-ALIGNED (review fix 2.2): the odometry sample describes the base at now - its age (wall now minus
    //    robot_current_speed_timestamp); the pose describes it at now - the pose age. The pose's heading change between its
    //    last two times is compared with the odometry INTEGRATED over that same interval (OdoSignCheck::update_interval).
    // Live fix 2 ruling 3: the TRUE pose age (no clamp) — tourP3 had 30 % of rows >= 0.5 s, hidden by the old 0.5 s clamp
    const float pose_age_s = world_model_pose_stamp_age_ms_ >= 0.f ? world_model_pose_stamp_age_ms_ * 1e-3f : 0.f;
    {
        const auto wall_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const double odo_age_s = odo_stamp_ms_ != 0 and wall_now > odo_stamp_ms_ ? double(wall_now - odo_stamp_ms_) * 1e-3 : 0.0;
        const double t_odo = now_s - odo_age_s;
        if (policy_odo_hist_.empty() or t_odo > policy_odo_hist_.back().t) policy_odo_hist_.push_back({t_odo, w_meas});
        while (policy_odo_hist_.size() > 2 and policy_odo_hist_[1].t < now_s - 2.0) policy_odo_hist_.pop_front();
        const double t_pose = now_s - pose_age_s;
        if (policy_prev_pose_t_.has_value() and t_pose > *policy_prev_pose_t_)
        {
            const std::vector<rc::policy::OdoSample> hist(policy_odo_hist_.begin(), policy_odo_hist_.end());
            const float dt_pose = static_cast<float>(t_pose - *policy_prev_pose_t_);
            // sigma: the relative yaw sigma (per 1 s) as a random walk over the interval
            policy_sign_.update_interval(hist, *policy_prev_pose_t_, t_pose,
                                         std::remainder(pose.yaw - policy_prev_yaw_.value_or(pose.yaw), 2.f * std::numbers::pi_v<float>),
                                         limits.w_min_eff, prm.policy_sigma_rel_yaw_rad * std::sqrt(dt_pose));
        }
        if (not policy_prev_pose_t_.has_value() or t_pose > *policy_prev_pose_t_) { policy_prev_pose_t_ = t_pose; policy_prev_yaw_ = pose.yaw; }
    }
    if (not policy_odom_.has_value()) policy_odom_ = pose;
    const rc::policy::Pose2 odom_before = *policy_odom_;
    {
        const float yaw_mid = policy_odom_->yaw + 0.5f * w_meas * dt_tick;
        policy_odom_->p += v_meas * dt_tick * Eigen::Vector2f(std::cos(yaw_mid), std::sin(yaw_mid));
        policy_odom_->yaw += w_meas * dt_tick;
    }
    // ── unseen contact: the PREVIOUS tick's commands in force (send history shifted by the command latency) against what
    //    the wheels measured over it
    const float cmd_latency = std::max(0.f, prm.policy_command_latency_s);
    if (dt_tick > 0.f)
    {
        const std::vector<rc::policy::SentCommand> hist(policy_sent_.begin(), policy_sent_.end());
        const auto in_force = rc::policy::commands_in_force(hist, now_s, dt_tick, cmd_latency);
        policy_unseen_.update(in_force, v_meas, w_meas, v_meas * dt_tick, w_meas * dt_tick, odom_before, limits, body,
                              prm.policy_unseen_odo_sigma_m);
    }
    while (not policy_sent_.empty() and policy_sent_.front().t < now_s - 2.0) policy_sent_.pop_front();

    // ── the returns: the policy's CONTACT band (Lidar.ContactMinHeightM up; the helios far floor cut by source at capture),
    //    room frame as registered at capture, re-expressed on the current pose by being compared with it; sigma_r =
    //    LidarRangeSigmaM; existence 1 (the live net's rule today). Only what a two-step plan can reach can matter: the body's
    //    circumscribed radius + v_max over (2 horizons + latency) + the brake + 6 sigma — beyond it a return is dropped
    //    (the contact model would prune it anyway; this keeps the denoising cheap).
    std::vector<rc::policy::Return> returns;
    const float latency_s = pose_age_s + cmd_latency;
    const float sigma_r = prm.lidar_range_sigma_m;
    const float reach = body.circumscribed_radius() + limits.v_max * (2.f * pp.horizon_s + latency_s)
                      + limits.v_max * limits.v_max / (2.f * std::max(0.05f, limits.d_lin)) + 6.f * sigma_r;
    if (auto *buf = obstacle_tracker.contact_buffer(); buf != nullptr)
    {
        const auto [cloud_opt] = buf->read_last();
        if (cloud_opt.has_value())
        {
            const auto &[xs, ys, zs] = cloud_opt.value();
            const std::size_t n = std::min(xs.size(), ys.size());
            returns.reserve(n / 4);
            for (std::size_t i = 0; i < n; ++i)
            {
                const Eigen::Vector2f q(xs[i], ys[i]);
                if (not q.allFinite() or (q - pose.p).squaredNorm() > reach * reach) continue;
                returns.push_back({.p = q, .sigma = sigma_r, .p_exist = 1.f});
            }
        }
    }
    const std::size_t n_lidar = returns.size();
    for (const auto &h : policy_unseen_.hypotheses(pose, *policy_odom_)) returns.push_back(h);   // odometry -> room frame
    const std::uint64_t accepted = obstacle_tracker.last_scan_accept_steady_ms();
    const float returns_age_s = accepted == 0 ? 1e3f : static_cast<float>(steady_now_ms() - accepted) * 1e-3f;

    rc::policy::PolicyInput in;
    in.pose = pose;
    in.v_now = v_meas;
    in.w_now = w_meas;
    in.sigma_rel_xy = prm.policy_sigma_rel_xy_m;
    in.sigma_rel_yaw = prm.policy_sigma_rel_yaw_rad;
    in.pose_drift_sigma_m = prm.policy_pose_drift_sigma_m;
    in.pose_drift_time_s = prm.policy_pose_drift_time_s;
    in.returns_age_s = returns_age_s;
    in.returns = returns;
    in.body = &body;
    in.base = limits;
    in.latency_s = latency_s;
    in.now_s = now_s;

    const auto t0 = std::chrono::steady_clock::now();
    const rc::policy::Navigator::Step st = policy_nav_.step(in, pp);
    const double step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const Decision &d = st.d;
    const float f_here = policy_nav_.field().at(pose.p);

    // ── act
    const bool moving = d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release;
    if (st.unreachable)
    {
        out.verdict = PolicyVerdict::Unreachable;
        out.why = std::format("the goal is no longer reachable from ({:.2f},{:.2f}) after a trap at ({:.2f},{:.2f})", pose.p.x(),
                              pose.p.y(), st.trap ? st.trap->c.x() : 0.f, st.trap ? st.trap->c.y() : 0.f);
        note_why(rc::why::Cause::PolicyHold, std::format("{}: {}", who, out.why));
        note_no_command(); motion_commander.stop_robot();
        policy_sent_.push_back({now_s, 0.f, 0.f});
    }
    else if (moving)
    {
        const auto c = rc::policy::to_session_command(d.v, d.w);
        why_.set_driving_detail(std::format("{} ({} {}): {} v {:+.2f} m/s w {:+.2f} rad/s, P(contact) {:.3f}, field {:.2f} m", who,
                                            goal_kind_name(spec.goal.kind), rc::policy::mode_name(d.mode), d.why, d.v, d.w,
                                            1.f - std::exp(d.log_p_safe), f_here));
        motion_commander.send_speed_command(c.adv, c.side, c.rot);
        policy_sent_.push_back({now_s, d.v, d.w});
        out.verdict = PolicyVerdict::Driving;
    }
    else
    {
        out.verdict = d.mode == Decision::Mode::Arrived ? PolicyVerdict::Arrived : PolicyVerdict::Held;
        // Live fix 2: a sensing contradiction (returns inside the body) is a HOLD, not a stuck verdict — never a hand back
        out.stuck = d.mode == Decision::Mode::Stuck and not d.sensing_contradiction;
        out.why = d.why;
        if (d.mode == Decision::Mode::Arrived) note_why(rc::why::Cause::Arrived, std::format("{}: the policy reached the {} goal", who, goal_kind_name(spec.goal.kind)));
        else note_why(rc::why::Cause::PolicyHold, std::format("{}: {} — {}", who, rc::policy::mode_name(d.mode), d.why));
        note_no_command(); motion_commander.stop_robot();
        policy_sent_.push_back({now_s, 0.f, 0.f});
    }

    // ── one [policy] line per MODE CHANGE (and on a trap / an unreachable verdict)
    const int mode_i = static_cast<int>(d.mode);
    if (st.trap_vetoed)
        std::println("[policy] {}: a failed prediction's trap would block the measured doorway corridor — not laid (the Release "
                     "stands)", who);
    if (mode_i != policy_last_mode_ or st.trap or st.unreachable)
    {
        std::println("[policy] {} -> {} ({}): {} | v {:+.2f} w {:+.2f} P(contact) {:.3f} cont {:.2f} prog {:+.3f} m eligible {} | "
                     "field {:.2f} m, {} returns + {} unseen, age {:.2f} s{}{}",
                     who, rc::policy::mode_name(d.mode), goal_kind_name(spec.goal.kind), d.why, d.v, d.w, 1.f - std::exp(d.log_p_safe),
                     d.log_p_cont, d.progress_m, d.eligible, f_here, n_lidar, returns.size() - n_lidar, returns_age_s,
                     st.trap ? std::format(" | TRAP at ({:.2f},{:.2f}) r {:.2f}", st.trap->c.x(), st.trap->c.y(), st.trap->r) : std::string(),
                     st.unreachable ? " | UNREACHABLE" : "");
        policy_last_mode_ = mode_i;
    }

    // ── policy_decisions.csv: one row per cycle (classic locale: decimal POINTS whatever LANG says — CLAUDE.md)
    if (not policy_csv_.is_open())
    {
        policy_csv_.open("policy_decisions.csv", std::ios::out | std::ios::trunc);
        policy_csv_.imbue(std::locale::classic());
        if (policy_csv_.is_open())
            policy_csv_ << "t_ms,who,goal_kind,mode,v,w,log_p_safe,log_p_cont,progress_m,eligible,replanned,plan_remaining_p,"
                           "prediction_failed,trap,unreachable,world_applied,x,y,yaw,v_meas,w_meas,odo_valid,field_m,n_returns,n_unseen,"
                           "returns_age_s,latency_s,step_ms,why\n";
    }
    if (policy_csv_.is_open())
    {
        std::string why = d.why;
        for (char &ch : why) if (ch == ',' or ch == '\n') ch = ';';
        std::string w = who;
        for (char &ch : w) if (ch == ',') ch = ';';
        policy_csv_ << now_ms << ',' << w << ',' << goal_kind_name(spec.goal.kind) << ',' << rc::policy::mode_name(d.mode) << ','
                    << d.v << ',' << d.w << ',' << d.log_p_safe << ',' << d.log_p_cont << ',' << d.progress_m << ',' << d.eligible << ','
                    << (st.replanned ? 1 : 0) << ',' << st.plan_remaining_p << ',' << (d.prediction_failed ? 1 : 0) << ','
                    << (st.trap ? 1 : 0) << ',' << (st.unreachable ? 1 : 0) << ',' << (st.world_applied ? 1 : 0) << ','
                    << pose.p.x() << ',' << pose.p.y() << ',' << pose.yaw << ',' << v_meas << ',' << w_meas << ',' << (odo_valid_ ? 1 : 0) << ','
                    << f_here << ',' << n_lidar << ',' << (returns.size() - n_lidar) << ',' << returns_age_s << ',' << latency_s << ','
                    << step_ms << ',' << why << '\n';
        policy_csv_.flush();
    }
    if (out.why.empty()) out.why = d.why;
    out.field_m = f_here;
    return out;
}

// ── THE CYCLE, WITH THE POLICY DRIVING ────────────────────────────────────────────────────────────────────────────────────
void ControllerSession::drive_with_policy(const ControllerRobotPose &robot_pose, rc::TrajectoryController &path_controller,
                                          ControllerObstacleTracker &obstacle_tracker, ControllerMotionCommander &motion_commander,
                                          ControllerDisplay &display, rc::AffordanceManager &affordance_manager,
                                          const TimeSource &time_source)
{
    const std::uint64_t now = time_source();
    const ControllerRobotPose &cpose = tracker_pose_.has_value() ? *tracker_pose_ : robot_pose;
    const float standoff = params_ != nullptr ? params_->anchor_cross_standoff_m : 2.0f;
    rc::RobotFootprint bare = approach_body_;
    bare.set_safety_margin(0.f);
    rc::policy::GoalAdapterParams gp;
    gp.tol_xy = path_controller.goal_threshold();
    gp.tol_yaw = path_controller.params.align_yaw_tol_rad;
    // the cross's goal line lies 2 tol_xy BEYOND the standoff (review I3)
    gp.past_m = standoff + 2.f * gp.tol_xy;   // review I3: real slack — the protocol's through (s >= standoff) is the ONE
                                              // definition of arrival; decide() never reports Arrived for a Doorway
    gp.inscribed_radius_m = bare.inscribed_radius();
    const auto stop_here = [&](rc::why::Cause c, const std::string &why)
    {
        note_why(c, why);
        note_no_command(); motion_commander.stop_robot();
    };

    // ── 0. A CONTINUOUS MISSION ROUTE (Task 8 ruling 1): its waypoints are successive Place goals for the Navigator
    //       (goal_adapter.h route_goal_step) — the authored order (reversed with the run), laps as the mission counts them,
    //       each repaired to the nearest feasible pose as build_route repairs them; the next one on Arrived; the final one
    //       keeps the route's end heading if it has one. A waypoint the policy cannot reach (Stuck / Unreachable) is
    //       skipped with a line, as the route builder defers one. The tour ends by the mission's own stop.
    if (params_ != nullptr and params_->route_continuous and mission_.running())
    {
        if (policy_route_index_ < 0)
        {
            policy_route_wps_.clear();
            if (const auto *m = mission_.running_mission(); m != nullptr)
                for (const auto &w : m->waypoints) policy_route_wps_.push_back({w.pos, w.yaw_rad});
            if (route_reverse_) std::ranges::reverse(policy_route_wps_);
            const int n = static_cast<int>(policy_route_wps_.size());
            for (int i = 0; i < n; ++i)
            {
                auto &w = policy_route_wps_[static_cast<std::size_t>(i)];
                const Eigen::Vector2f d = policy_route_wps_[static_cast<std::size_t>((i + 1) % n)].pos - w.pos;
                if (const auto safe = grid_planner_.nearest_free(w.pos, d.squaredNorm() > 1e-9f ? std::atan2(d.y(), d.x()) : 0.f);
                    safe.has_value()) w.pos = *safe;
            }
            policy_route_laps_ = std::max(1, mission_.laps_remaining());
            policy_route_index_ = 0;
            std::println("[policy] mission '{}': {} waypoints x {} lap(s) as successive place goals", mission_.selected_name(), n,
                         policy_route_laps_);
        }
        auto step = rc::policy::route_goal_step(policy_route_wps_, policy_route_laps_, policy_route_index_, false, gp.tol_xy, gp.tol_yaw);
        if (not step.finished and step.goal.has_value())
        {
            const std::string who = std::format("mission '{}' wp {}", mission_.selected_name(), step.index + 1);
            const auto tick = policy_tick(*step.goal, who, cpose, obstacle_tracker, motion_commander, now);
            const bool skip = tick.verdict == PolicyVerdict::Unreachable or (tick.verdict == PolicyVerdict::Held and tick.stuck);
            if (skip) std::println("[policy] {}: cannot be reached ({}) — skipped", who, tick.why);
            if (tick.verdict == PolicyVerdict::Arrived or skip)
                step = rc::policy::route_goal_step(policy_route_wps_, policy_route_laps_, policy_route_index_, true, gp.tol_xy, gp.tol_yaw);
            policy_route_index_ = step.index;
        }
        if (step.finished)
        {
            mission_.stop("completed", time_source());
            route_active_ = false;
            policy_route_index_ = -1;
            policy_reset();
            stop_here(rc::why::Cause::MissionDone, std::format("mission '{}' completed (local policy)", mission_.selected_name()));
        }
        return;
    }
    policy_route_index_ = -1;   // no tour running: the next one latches afresh

    // ── 1. THE CROSS: the protocol (gap, near side, through / not-a-crossing / waits) is shared; the motion is a Doorway goal
    //       on the MEASURED gap (the anchor while no gap is held), or — waiting for an anchor never seen — a Heading goal
    //       toward where it was last published
    if (anchor_cross_.has_value())
    {
        const CrossProtocol cp = cross_protocol(robot_pose, path_controller, obstacle_tracker, motion_commander, display,
                                                affordance_manager, time_source);
        if (not cp.ended)
        {
            const std::string who = std::format("cross {}", anchor_latched_for_);
            rc::policy::GoalSpec spec;
            std::function<bool(const Eigen::Vector2f &)> corridor;
            std::function<bool(const Eigen::Vector2f &, float)> veto;
            if (cp.drive)
            {
                const auto gap = rc::policy::gap_in_room(cp.drive_pose, cp.drive_width, cpose);
                spec = rc::policy::doorway_goal(gap, gp);
                // the far side is not in the planner's map: the measured opening's corridor joins the near room to the goal
                // line (1 m back into the room just left; one field cell past the line)
                const float back = 1.0f, ahead = gp.past_m + 0.05f, r_ins = gp.inscribed_radius_m;
                corridor = [gap, back, ahead, r_ins](const Eigen::Vector2f &p) { return rc::policy::in_doorway_corridor(gap, p, back, ahead, r_ins); };
                // Live fix 2 ruling 2: the measured opening (at least the body's width) is evidence the body fits there — no
                // trap may block it; the bare body's full width = its lateral support both sides
                const float body_w = bare.support_radius(0.f, Eigen::Vector2f(1.f, 0.f)) + bare.support_radius(0.f, Eigen::Vector2f(-1.f, 0.f));
                veto = [gap, back, ahead, r_ins, body_w](const Eigen::Vector2f &c, float r)
                { return rc::policy::disc_meets_doorway_corridor(gap, c, r, back, ahead, r_ins, body_w); };
            }
            else
            {
                const float half_len = approach_body_.support_radius(0.f, Eigen::Vector2f(0.f, 1.f));
                spec = rc::policy::heading_goal(cpose.pos, rc::policy::to_policy_pose(cpose).yaw + rc::anchor_wait_aim(cp.pose, half_len),
                                                gp.tol_yaw);
            }
            const auto tick = policy_tick(spec, who + (cp.drive ? "" : " (waiting)"), cpose, obstacle_tracker, motion_commander, now,
                                          corridor, veto);
            if (tick.verdict == PolicyVerdict::Unreachable or (tick.verdict == PolicyVerdict::Held and tick.stuck))
                cross_hand_back(std::format("the local policy cannot drive it ({})", tick.why), /*timed_out=*/false, motion_commander);

        }
        anchor_cross_.reset();
        return;
    }

    // review I4: gated on a TARGET (and, below, on the field), not on an A* route
    if (not last_target_info_.has_value())
    {
        display.clear_robot_trajectory();
        clear_tracking_state();
        stop_here(rc::why::Cause::NoTarget, "no target (the policy has no goal)");
        policy_reset();
        return;
    }

    // ── 2. A LOCK-ON in progress: its predicate machinery runs as before (step_lockon, protocol only); its motion is a
    //       Heading goal toward the object
    if (lockon_.active() and last_target_info_.has_value())
    {
        note_why(rc::why::Cause::LockOn, std::format("lock-on micro-search on '{}' (policy heading)", why_target_name()));
        if (step_lockon(robot_pose, path_controller, obstacle_tracker, motion_commander, time_source, /*drive_base=*/false))
        {
            finalize_reached(affordance_manager, path_controller, motion_commander, display, robot_pose.pos, time_source());
            policy_reset();
            return;
        }
        policy_tick(rc::policy::lockon_heading_goal(*last_target_info_, cpose, gp), std::format("lock-on {}", last_target_info_->node_name),
                    cpose, obstacle_tracker, motion_commander, now);
        return;
    }

    // ── 3. AN ORIENT contract: step_orient's completion (looked / timed out) as before, the turn as a Heading goal
    if (params_ != nullptr and params_->lockon_enabled and graph_ and last_target_info_.has_value()
        and last_target_info_->from_affordance and active_target_id_ != 0)
    {
        if (const auto aff = graph_->get_node(active_target_id_); aff.has_value())
        {
            std::string parent_type;
            if (last_target_info_->parent_node_id != 0)
                if (const auto pn = graph_->get_node(last_target_info_->parent_node_id); pn.has_value())
                    parent_type = contract_key_of(pn.value());
            const auto contract = rc::affordance::read_contract(aff.value(), parent_type);
            if (contract.policy == rc::affordance::Policy::Orient)
            {
                active_contract_ = contract;
                feedback_node_id_ = contract.feedback_node_id != 0 ? contract.feedback_node_id : last_target_info_->parent_node_id;
                note_why(rc::why::Cause::Orienting, std::format("orient '{}': a heading goal for the policy", last_target_info_->node_name));
                if (step_orient(robot_pose, path_controller, obstacle_tracker, motion_commander, time_source,
                                last_target_info_->yaw_rad, /*drive_base=*/false))
                {
                    finalize_reached(affordance_manager, path_controller, motion_commander, display, robot_pose.pos, time_source());
                    policy_reset();
                    return;
                }
                if (const auto spec = rc::policy::goal_for_target(*last_target_info_, cpose, std::nullopt, rc::policy::GoalMode::Orient,
                                                                  false, gp))
                    policy_tick(*spec, std::format("orient {}", last_target_info_->node_name), cpose, obstacle_tracker,
                                motion_commander, now);
                return;
            }
        }
    }

    // ── 4. A PLACE: a reach standpoint, a click target, a mission waypoint
    if (not last_target_info_.has_value())
    {
        stop_here(rc::why::Cause::NoTarget, "a plan is held but no target names where it goes");
        return;
    }
    const auto spec = rc::policy::goal_for_target(*last_target_info_, cpose, std::nullopt, rc::policy::GoalMode::Place,
                                                  wants_final_facing(*last_target_info_), gp);
    if (not spec.has_value()) { stop_here(rc::why::Cause::PolicyHold, "no goal could be built"); return; }
    const std::string who = last_target_info_->node_name.empty() ? std::string("target") : last_target_info_->node_name;
    const auto tick = policy_tick(*spec, who, cpose, obstacle_tracker, motion_commander, now);
    // ARRIVAL: the same protocol the tracker's arrival goes through (servo → lock-on, already there, approach-only, REACH)
    if (consume_goal_reached(tick.verdict == PolicyVerdict::Arrived, robot_pose, path_controller, obstacle_tracker,
                             motion_commander, display, affordance_manager, time_source, /*drive_base=*/false))
    {
        if (not lockon_.active()) policy_reset();
        return;
    }
    // NO FIELD ROUTE: the goal is not reachable on the map from here — the existing no-route clock reports it (5 s, the same
    // liveness bound the planner's no-route hold has: report_unreachable_if_no_route)
    // review I4 / M4: "no route" = the FIELD at the robot is non-finite; the clock runs on that and nothing else resets it
    if (not std::isfinite(tick.field_m))
    {
        if (no_route_since_ms_ == 0) no_route_since_ms_ = now;
        note_why(rc::why::Cause::NoRoute, std::format("{}: the cost-to-go is infinite at the robot — no way to the goal on the map", who));
    }
    else if (tick.verdict != PolicyVerdict::Unreachable and not tick.stuck)
        no_route_since_ms_ = 0;
    // UNREACHABLE / STUCK: handed back through the protocol (a mouse target is dropped by the no-route clock)
    if (tick.verdict == PolicyVerdict::Unreachable or (tick.verdict == PolicyVerdict::Held and tick.stuck))
    {
        // a Reach is handed back at once (the producer offers another); any other contract / a mouse target goes through
        // the 5-s no-route clock (report_unreachable_if_no_route), which reports or drops it
        if (not flag_reach_unreachable(std::format("local policy: {}", tick.why)) and no_route_since_ms_ == 0)
            no_route_since_ms_ = now;
    }
}
