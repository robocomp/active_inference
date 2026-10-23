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

// ★ Task O fix round 1: the active contract's precondition probability (policy-drive helpers)
std::optional<rc::policy::OrientPrecondition> ControllerSession::contract_precondition(const rc::affordance::Contract &c) const
{
    const std::uint64_t fb = c.feedback_node_id != 0 ? c.feedback_node_id
                           : feedback_node_id_ != 0 ? feedback_node_id_
                           : last_target_info_.has_value() ? last_target_info_->parent_node_id : 0;
    if (not graph_ or fb == 0) return std::nullopt;
    const auto node = graph_->get_node(fb);
    if (not node.has_value()) return std::nullopt;
    return rc::policy::precondition_from_clauses(c.pre, [&](const std::string &attr) -> std::optional<float>
    {
        if (attr == "door_reach_prob") return graph_->get_attrib_by_name<door_reach_prob_att>(node.value());
        return std::nullopt;
    });
}

// Is the published point usable as the goal? Unknown grid / blocked pose => the repaired point stands in.
bool ControllerSession::published_target_admissible(const ControllerTargetInfo &t) const
{
    if (not last_raw_target_pos_.has_value()) return false;
    return grid_planner_.pose_free(*last_raw_target_pos_, t.yaw_rad);
}

void ControllerSession::policy_reset()
{
    orient_entry_ms_.reset();
    place_entry_ms_.reset();
    policy_goal_.reset();
    policy_goal_who_.clear();
    policy_unseen_.reset();
    policy_sent_.clear();
    policy_last_mode_ = -1;
    policy_last_why_.clear();
    policy_no_field_since_ms_ = 0;
    policy_unreach_latch_.on_new_goal();
    policy_gate_.reset();
    // Task 3d fix round 2 (CRITICAL-2): the turn-sign belief and its EMERGENCY are about the BASE, not the activity — a reset (run by
    // a hand-back, a new goal, a room-frame change) keeps them; only the operator clears them (operator_clear_turn_sign_fault)
    policy_prev_yaw_.reset();
    policy_prev_pose_t_.reset();
    policy_prev_pose_p_.reset();
    policy_odo_hist_.clear();
    policy_clock_prev_ms_ = 0;      // Task 3d fix round 1 (M-1): a reset clock has no previous tick
    policy_probe_t0_ = -1.0;
    policy_leaf_ = {};
    policy_leaf_room_.clear();
    policy_leaf_evidence_.reset();
}

void ControllerSession::operator_clear_turn_sign_fault()
{
    if (policy_sign_safety_.active() or policy_sign_.flipped())
        std::println("[policy] turn-sign EMERGENCY / Blind cleared by the operator (P was {:.4f})", policy_sign_.p_flipped());
    policy_sign_safety_.operator_clear();
    policy_sign_.reset();
    policy_probe_t0_ = -1.0;
}

ControllerSession::PolicyTick ControllerSession::policy_tick(const rc::policy::GoalSpec &spec, const std::string &who,
                                                             const ControllerRobotPose &robot_pose,
                                                             ControllerObstacleTracker &obstacle_tracker,
                                                             ControllerMotionCommander &motion_commander, std::uint64_t now_ms,
                                                             std::function<bool(const Eigen::Vector2f &)> also_free,
                                                             std::function<bool(const Eigen::Vector2f &, float)> trap_veto,
                                                             std::optional<rc::policy::DoorwayRisk> jambs)
{
    using rc::policy::Decision;
    PolicyTick out;
    if (not jambs.has_value() and policy_held_active_) jambs = policy_held_jambs_;   // Task 3f r1 (I-1): the held doorway's jambs
    const double now_s = steady_now_s();
    const auto &prm = params_ != nullptr ? *params_ : ControllerParams{};
    auto hold = [&](const std::string &why)
    {
        out.verdict = PolicyVerdict::Held;
        out.why = why;
        note_why(rc::why::Cause::PolicyHold, std::format("{}: {}", who, why));
        note_no_command(); motion_commander.stop_robot();
        policy_sent_.push_back({now_s, 0.f, 0.f});
        // Task 3d: a hold that returns before the tick's own 2-s trim (the latched turn-sign EMERGENCY holds every tick) must not
        // grow the send history without bound
        while (not policy_sent_.empty() and policy_sent_.front().t < now_s - 2.0) policy_sent_.pop_front();
        policy_gate_.on_hold();   // review fix 2.1: a hold — however short — is an INTERRUPTION of the committed plan
    };
    // ★ Task 3d fix round 2 (CRITICAL-2): the latched turn-sign EMERGENCY holds the base idle whatever the goal — no hand-back here
    //   (it was handed back ONCE on entry), no field evaluated, so the no-route clock does not advance
    if (policy_sign_safety_.active())
    {
        hold(std::format("EMERGENCY — odometry turn sign flipped (P={:.4f}); held until the operator clears it (Stop)", policy_sign_.p_flipped()));
        return out;
    }
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
    // ★ review I5 → Task 3d (rtD1e leg 5): the odometry TURN SIGN is a BELIEF (goal_adapter.h TurnSignBelief), folded here — BEFORE
    //   the Blind check, so a Blind policy keeps observing and Blind ENDS when the belief drops (folded after it, as it was, a
    //   Blind tick never observed again: a second latch). TIME-ALIGNED (review fix 2.2): the odometry sample describes the base at
    //   now - its age (wall now minus robot_current_speed_timestamp); the pose describes it at now - the pose age; the pose's
    //   heading change AND displacement between its last two times are compared with the odometry INTEGRATED over that same
    //   interval. The history restarts only on a real gap in ticks (PolicyTickGate::kMaxTickGapS) — never on a hold: a Blind hold
    //   is itself a hold, and clearing on it would never collect the two samples that end it.
    {
        const rc::policy::Pose2 sp = rc::policy::to_policy_pose(robot_pose);
        const float v_s = odo_valid_ ? odo_v_ : 0.f, w_s = odo_valid_ ? odo_w_ : 0.f;
        const float pose_age_s = world_model_pose_stamp_age_ms_ >= 0.f ? world_model_pose_stamp_age_ms_ * 1e-3f : 0.f;
        const auto wall_now = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        const double odo_age_s = odo_stamp_ms_ != 0 and wall_now > odo_stamp_ms_ ? double(wall_now - odo_stamp_ms_) * 1e-3 : 0.0;
        const double t_odo = now_s - odo_age_s;
        if (not policy_odo_hist_.empty() and t_odo - policy_odo_hist_.back().t > rc::policy::PolicyTickGate::kMaxTickGapS)
        { policy_odo_hist_.clear(); policy_prev_pose_t_.reset(); policy_prev_pose_p_.reset(); }
        if (policy_odo_hist_.empty() or t_odo > policy_odo_hist_.back().t) policy_odo_hist_.push_back({t_odo, w_s, v_s});
        while (policy_odo_hist_.size() > 2 and policy_odo_hist_[1].t < now_s - 2.0) policy_odo_hist_.pop_front();
        const double t_pose = now_s - pose_age_s;
        if (policy_prev_pose_t_.has_value() and t_pose > *policy_prev_pose_t_)
        {
            const std::vector<rc::policy::OdoSample> hist(policy_odo_hist_.begin(), policy_odo_hist_.end());
            const float dt_pose = static_cast<float>(t_pose - *policy_prev_pose_t_);
            // sigmas: the relative heading sigma (per 1 s) as a random walk over the interval; the odometry's displacement sigma
            // (UnseenOdoSigmaM per tick) with the relative xy sigma over the interval; a correction's size = room_concept's
            // published pose xy std, else the pose's drift sigma
            const float s_yaw = prm.policy_sigma_rel_yaw_rad * std::sqrt(dt_pose);
            const float s_p = std::hypot(prm.policy_unseen_odo_sigma_m, prm.policy_sigma_rel_xy_m * std::sqrt(dt_pose));
            const float s_corr = policy_pose_xy_std_ > 0.f ? policy_pose_xy_std_ : prm.policy_pose_drift_sigma_m;
            // fix round 1 (review C-1/b): the POSE's own heading noise — room_concept's published heading std, the larger of the
            // interval's two ends (it rises to 0.16-0.18 rad at a relocalisation, rtD1e). 0 = not published: only the odometry's
            // relative sigma remains, and a yaw-only correction is then explained only through the translation residual (flagged)
            const float s_pose_yaw = std::max(policy_pose_theta_std_, policy_prev_theta_std_);
            policy_sign_.update_interval(hist, *policy_prev_pose_t_, t_pose,
                                         std::remainder(sp.yaw - policy_prev_yaw_.value_or(sp.yaw), 2.f * std::numbers::pi_v<float>),
                                         (sp.p - policy_prev_pose_p_.value_or(sp.p)).norm(), s_yaw, s_pose_yaw, s_p, s_corr);
        }
        if (not policy_prev_pose_t_.has_value() or t_pose > *policy_prev_pose_t_)
        { policy_prev_pose_t_ = t_pose; policy_prev_yaw_ = sp.yaw; policy_prev_pose_p_ = sp.p; policy_prev_theta_std_ = policy_pose_theta_std_; }
    }
    // Blind is decided below, once the contact returns are assembled: the policy may PROBE (fix round 1, review C-1/c)
    const bool sign_blind = policy_sign_.flipped();

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
    // ★ Task 3f (b) (tour3A leg 1: the doorway goal line 2.4 m past the door lay beyond the newborn room's planner raster ⇒ the
    //   field was +inf everywhere for 9 min). The field's grid covers the planner's extent ∪ the goal points ∪ the robot
    //   (local_policy/field_extent.h); a cell OUTSIDE the planner's raster is UNSEEN — its distance to a known obstacle is the
    //   raster's border value plus the way out (extended_distance: an UPPER bound on the clearance), its price the honest prior for unseen space: ln 2 nats per
    //   planner cell the body covers (residual_risk.h's surprisal at a symmetric no-evidence prior, Jeffreys Beta(½,½)), not free and not a wall.
    const float planner_cell = grid_planner_.cell_size();
    const Eigen::Vector2f ext_lo = grid_planner_.grid_origin() - 0.5f * Eigen::Vector2f::Constant(planner_cell);
    const rc::policy::PlannerExtent ext{ext_lo, ext_lo + planner_cell * Eigen::Vector2f(float(grid_planner_.width()), float(grid_planner_.height()))};
    const float body_area = [&body]
    {
        const auto &poly = body.polygon();
        double a = 0.0;
        for (std::size_t i = 0, n = poly.size(); i < n; ++i)
            a += double(poly[i].x()) * poly[(i + 1) % n].y() - double(poly[(i + 1) % n].x()) * poly[i].y();
        return static_cast<float>(0.5 * std::abs(a));
    }();
    const float unseen_density = rc::policy::unseen_prior_density(body_area, planner_cell,
                                                                    std::max(0.f, grid_planner_.params.residual_risk_weight));
    auto dist_fn = [this, ext](const Eigen::Vector2f &p)
    { return rc::policy::extended_distance(ext, p, [this](const Eigen::Vector2f &q) { return grid_planner_.distance_at(q); }); };
    auto free_fn = [this, r_ins, also_free, dist_fn](const Eigen::Vector2f &p)
    { return rc::policy::field_free_from_distance(dist_fn(p), grid_planner_.cell_size(), r_ins) or (also_free and also_free(p)); };
    // ★ F4: with a measured opening, its jambs' contact risk joins the density over the doorway's corridor — the SAME weight
    //   turns its nats into metres, so F1's route surprisal reads it as nats too (a crossing at offset e adds R(e) nats:
    //   ~0 on the centre line of a passable opening, so a passable door does not lean Unreachable; see doorway_corridor.h)
    // ★ Task 3b: the door's LEAF (door_concept's hypotheses, room frame) is a known obstacle: with the hypothesis-dependent
    //   excess of the soft residual cells it explains this cycle (sync_world_state), one mixture over its hypotheses prices the
    //   body's centre at p (door_leaf.h door_leaf_density, on the bare inscribed radius free_fn uses). Every goal adds it: the
    //   planner's prices this cycle are the share all hypotheses agree on.
    auto extra_fn = [this, jambs, leaf = policy_leaf_evidence_, ext, unseen_density, dist_fn](const Eigen::Vector2f &p)
    {
        const float w = std::max(0.f, grid_planner_.params.residual_risk_weight);
        return (ext.contains(p) ? w * grid_planner_.body_risk_at(p) : unseen_density)   // Task 3f (b): unseen = the prior
               + w * ((jambs ? rc::policy::doorway_risk_density(*jambs, p, [&](const Eigen::Vector2f &q) { return rc::policy::field_free_from_distance(dist_fn(q), grid_planner_.cell_size(), jambs->anchor.inscribed_radius_m); }) : 0.f)
                      + (leaf ? rc::policy::door_leaf_density(*leaf, p) : 0.f));
    };
    const bool new_goal = not policy_goal_.has_value() or policy_goal_who_ != who or not rc::policy::same_goal(*policy_goal_, spec);
    if (new_goal)
    {
        // The field's grid: the planner's extent ∪ the goal points ∪ the robot (Task 3f (b): a goal is never off the grid it
        // seeds), at the harness's validated 0.05 m resolution (the field's own resolution bounds what the progress monitor
        // can call progress — CostToGo::resolution_m).
        constexpr float kFieldCell = 0.05f;
        const rc::policy::CostToGo::Grid g = rc::policy::field_grid(ext, spec.points, pose.p, body.circumscribed_radius() + kFieldCell,
                                                                    kFieldCell);
        // ★ F1: the navigator turns the field's priced-minus-length back into nats with the SAME weight extra_fn used.
        policy_nav_.set_route_risk_weight(std::max(0.f, grid_planner_.params.residual_risk_weight));
        policy_nav_.set_goal(spec.goal, g, free_fn, extra_fn, spec.points, body.circumscribed_radius());
        policy_goal_ = spec;
        policy_goal_who_ = who;
        policy_unreach_latch_.on_new_goal();   // F1 (I4): a new goal is a new question
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
            f << "leaf_hypotheses " << (policy_leaf_evidence_ ? policy_leaf_evidence_->leaf.hyps.size() : 0u) << " ; leaf_excess_cells "
              << (policy_leaf_evidence_ ? policy_leaf_evidence_->cells.size() : 0u) << " ; leaf_hard_in_swept "
              << (policy_leaf_evidence_ ? policy_leaf_evidence_->hard_in_swept : 0) << " ; leaf_nats_at_robot "
              << (policy_leaf_evidence_ ? rc::policy::door_leaf_density(*policy_leaf_evidence_, pose.p) : 0.f) << '\n';
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
        // (Task 3d: the turn-sign history is NOT cleared on a hold — see the belief's fold above)
    }
    // Live fix 2 ruling 3: the TRUE pose age (no clamp) — tourP3 had 30 % of rows >= 0.5 s, hidden by the old 0.5 s clamp
    const float pose_age_s = world_model_pose_stamp_age_ms_ >= 0.f ? world_model_pose_stamp_age_ms_ * 1e-3f : 0.f;
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
    in.camera_half_fov_rad = prm.policy_camera_half_fov_rad;       // Task R
    in.camera_range_m = prm.policy_camera_range_m;
    in.affordance_facing = prm.policy_affordance_facing;
    in.affordance_reach_m = prm.policy_affordance_reach_m;
    in.pose_drift_sigma_m = prm.policy_pose_drift_sigma_m;
    in.pose_drift_time_s = prm.policy_pose_drift_time_s;
    in.returns_age_s = returns_age_s;
    in.returns = returns;
    in.body = &body;
    in.base = limits;
    in.latency_s = latency_s;
    in.now_s = now_s;

    // ★ Task 3d fix round 1 (c) — BLIND WHILE THE TURN SIGN IS IN DOUBT, AND AN EPISTEMIC WAY OUT. The goal is not driven
    //   (the world may be mirrored); instead the policy turns in place at the base's smallest executable rate if the contact
    //   model clears BOTH directions (local_policy.h sign_probe) — the turn's evidence settles the sign either way: a false Blind
    //   ends, a true flip stays Blind. The probe alternates direction every horizon so the net rotation stays ~0. No safe probe:
    //   held, and reported. Any of these interrupts the Navigator's plan (a hold).
    if (sign_blind and policy_sign_safety_.on_tick(policy_sign_.probe_decisive(pp.horizon_s)) == rc::policy::TurnSignSafety::Tick::Entered)
    {
        // ★ (ruling + fix round 2) EMERGENCY ENTRY — the probe's own evidence has answered "flipped": one more probe buys nothing.
        //   Stop, report ONCE, hand the active goal back ONCE (this tick's Unreachable), then the latch holds idle (top of the tick).
        std::println("[policy] EMERGENCY: odometry turn sign flipped (P={:.4f}) — the probe confirmed it; goal '{}' handed back, robot "
                     "held until the operator clears it (Stop)", policy_sign_.p_flipped(), who);
        policy_sign_.set_probing(false);
        policy_probe_t0_ = -1.0;
        hold(std::format("EMERGENCY — odometry turn sign flipped (P={:.4f}), confirmed by the probe: goal handed back", policy_sign_.p_flipped()));
        out.verdict = PolicyVerdict::Unreachable;
        return out;
    }
    if (not sign_blind) policy_sign_.set_probing(false);
    if (sign_blind)
    {
        const std::string head = std::format("Blind — odometry turn sign contradicts the pose (P(flipped) {:.2f}: robot_current_angular_speed "
                                             "vs the heading change, relocaliser corrections discounted) — check the base/bridge sign",
                                             policy_sign_.p_flipped());
        if (const auto wp = rc::policy::sign_probe(in, pp); wp.has_value())
        {
            if (policy_probe_t0_ < 0.0 or now_s - policy_probe_t0_ >= pp.horizon_s) { policy_probe_t0_ = now_s; policy_probe_sign_ = -policy_probe_sign_; }
            const float w = policy_probe_sign_ * *wp;
            policy_sign_.set_probing(true);   // its evidence is the probe's own (probe_decisive)
            const std::string why = head + std::format("; probing in place at w {:+.2f} rad/s to settle it", w);
            if (why.substr(0, 40) != policy_last_why_.substr(0, 40)) std::println("[policy] {}: {}", who, why);
            policy_last_why_ = why; policy_last_mode_ = -1;
            out.verdict = PolicyVerdict::Held; out.why = why;
            note_why(rc::why::Cause::PolicyHold, std::format("{}: {}", who, why));
            const auto c = rc::policy::to_session_command(0.f, w);
            motion_commander.send_speed_command(c.adv, c.side, c.rot);
            policy_sent_.push_back({now_s, 0.f, w});
            policy_gate_.on_hold();
            return out;
        }
        const std::string why = head + "; no probe turn clears the contact model here — held";
        if (why.substr(0, 40) != policy_last_why_.substr(0, 40)) std::println("[policy] {}: {}", who, why);
        policy_last_why_ = why; policy_last_mode_ = -1;
        policy_probe_t0_ = -1.0;
        policy_sign_.set_probing(false);
        hold(why);
        return out;
    }
    policy_probe_t0_ = -1.0;

    const auto t0 = std::chrono::steady_clock::now();
    const rc::policy::Navigator::Step st = policy_nav_.step(in, pp);
    const double step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const Decision &d = st.d;
    const float f_here = policy_nav_.field().at(pose.p);             // the PRICED cost-to-go (route choice, reachability)
    const float route_here = policy_nav_.field().at_length(pose.p);  // the cheapest route's METRES (progress; Live fix 3)

    // ── act
    const bool moving = d.mode == Decision::Mode::Drive or d.mode == Decision::Mode::Release;
    // ★ F1: Unreachable is the trapped field's (st.unreachable) or the reachability belief's (st.hand_back: the place's log-odds
    //   that no motion escapes exceeds log(route / (v_max dt)) — reachability_belief.h). A Stuck alone is a hold (below).
    if (st.unreachable or st.hand_back)
    {
        out.verdict = PolicyVerdict::Unreachable;
        policy_unreach_latch_.on_verdict(true);   // F1 (I4): reported by the protocol even after the belief restarts
        out.why = st.unreachable
            ? std::format("the goal is no longer reachable from ({:.2f},{:.2f}) after a trap at ({:.2f},{:.2f})", pose.p.x(),
                          pose.p.y(), st.trap ? st.trap->c.x() : 0.f, st.trap ? st.trap->c.y() : 0.f)
            : std::format("no motion escapes from ({:.2f},{:.2f}): reachability log-odds {:.2f} > bar {:.2f} (the {:.2f} m route vs "
                          "{:.2f} s between decisions) — last decision: {}", pose.p.x(), pose.p.y(), st.unreach_L, st.unreach_bar, route_here, st.unreach_dt_obs, d.why);
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
        // Live fix 2: a sensing contradiction (returns inside the body) is a HOLD, not a stuck verdict — never a hand back.
        // F1: a Stuck is a HOLD too — the next tick decides again; the hand back is the belief's (above)
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
    if (mode_i != policy_last_mode_ or st.trap or st.unreachable or st.hand_back)
    {
        std::println("[policy] {} -> {} ({}): {} | v {:+.2f} w {:+.2f} P(contact) {:.3f} cont {:.2f} prog {:+.3f} m eligible {} | "
                     "field {:.2f} m (route {:.2f} m), {} returns + {} unseen, age {:.2f} s | reachability L {:+.2f} bar {:.2f}{}{}",
                     who, rc::policy::mode_name(d.mode), goal_kind_name(spec.goal.kind), d.why, d.v, d.w, 1.f - std::exp(d.log_p_safe),
                     d.log_p_cont, d.progress_m, d.eligible, f_here, route_here, n_lidar, returns.size() - n_lidar, returns_age_s,
                     st.unreach_L, st.unreach_bar,
                     st.trap ? std::format(" | TRAP at ({:.2f},{:.2f}) r {:.2f}", st.trap->c.x(), st.trap->c.y(), st.trap->r) : std::string(),
                     st.unreachable ? " | UNREACHABLE" : st.hand_back ? " | UNREACHABLE (reachability belief)" : "");
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
                           "returns_age_s,latency_s,step_ms,route_m,unreach_L,unreach_bar,hand_back,why\n";
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
                    << step_ms << ',' << route_here << ',' << st.unreach_L << ',' << st.unreach_bar << ',' << (st.hand_back ? 1 : 0) << ','
                    << why << '\n';
        policy_csv_.flush();
    }
    if (out.why.empty()) out.why = d.why;
    out.field_evaluated = true;   // Task 3d: the field was built and read at the robot this tick
    out.field_m = f_here;
    out.route_m = route_here;
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
    // Task 3b: the leaf is held while the cross below holds it — and (Task 3f r1, I-1) after it, while the body is still within
    // the doorway's reach (door_leaf.h body_within_door_zone): the goal after a passage must still route around the leaf and
    // the jambs it just went past. Released once outside, or on a room-frame change (the held geometry is the old frame's).
    policy_leaf_ = {};
    policy_held_active_ = false;
    if (policy_held_jambs_.has_value() and policy_held_room_ == current_room_name_
        and rc::policy::body_within_door_zone(cpose.pos, approach_body_.circumscribed_radius(), policy_held_jambs_->anchor.gap.centre,
                                              0.5f * policy_held_jambs_->anchor.gap.width, policy_held_leaf_))
    {
        policy_leaf_ = policy_held_leaf_;
        policy_leaf_room_ = current_room_name_;
        policy_held_active_ = true;
    }
    else if (policy_held_jambs_.has_value())
    {
        std::println("[policy] the held doorway ({} leaf hypotheses + jambs) is released — the body is outside its reach{}",
                     policy_held_leaf_.hyps.size(), policy_held_room_ == current_room_name_ ? "" : " (the room frame changed)");
        policy_held_jambs_.reset();
        policy_held_leaf_ = {};
        policy_held_room_.clear();
    }
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
            // F1: a Stuck is a hold; only the Unreachable verdict (trapped field or reachability belief) skips the waypoint
            const bool skip = tick.verdict == PolicyVerdict::Unreachable;
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
        policy_no_anchor_since_ms_ = 0;
        const CrossProtocol cp = cross_protocol(robot_pose, path_controller, obstacle_tracker, motion_commander, display,
                                                affordance_manager, time_source);
        if (not cp.ended)
        {
            const std::string who = std::format("cross {}", anchor_latched_for_);
            rc::policy::GoalSpec spec;
            std::function<bool(const Eigen::Vector2f &)> corridor;
            std::function<bool(const Eigen::Vector2f &, float)> veto;
            std::optional<rc::policy::DoorwayRisk> jambs;
            // ★ Task 3b: the LEAF door_concept models (anchor frame, rigid with the anchor's pose) in the room frame — a known
            //   obstacle in the field (policy_tick) and the explanation of its residual cells (sync_world_state, next cycle)
            policy_leaf_ = {};
            if (cp.anchor_pose.x.allFinite() and not anchor_cross_->leaves.empty())
                policy_leaf_ = rc::policy::door_leaf_in_room(cp.anchor_pose, anchor_cross_->leaves, cpose);
            policy_leaf_room_ = current_room_name_;
            if (cp.drive)
            {
                // ★ Task 3b (rtD1c leg 5: a 0.03-0.3-score gap locked on the open leaf set the goal line and the veto): the
                //   measured gap enters through F4's risk mixture and the corridor union below, and the trap veto is that union's
                //   (doorway_corridor.h disc_meets_doorway_risk_corridor — fix round 1, I-2: no mean of a bimodal mixture). The
                //   goal line was the ANCHOR's always; since Task 3e it is the MAP line below (a low-score leaf gap still loses).
                // ★ Task 3e (1) (rtD1h leg 5: an anchor aliased ~1.9 m / 70° off the door replaced a 0.99-score gap on the true
                //   opening and pulled the robot back out of the doorway): the goal line is the MAP of two hypotheses — "the
                //   anchor is the door" (its own P(not lost) × its lateral coverage) vs "the gap is the door" (its score × its
                //   coverage) — a SELECTION, never their average (doorway_hypotheses.h says why the disagreement cannot choose).
                //   Fix round 1: the anchor's block is AGED by the motion since its stamp (I-1: one missed block widens it, it is
                //   never zeroed; r2: the motion actually made, and a parked / not-drivable block raises λ), and the choice is the MAP of a RECURSIVE belief (I-2: evidence accumulates per cycle under a
                //   switching hazard — inertia without a hysteresis band). The risk mixture and the veto below carry BOTH (F4/3b).
                const float body_half_w = 0.5f * (bare.support_radius(0.f, Eigen::Vector2f(1.f, 0.f))
                                                  + bare.support_radius(0.f, Eigen::Vector2f(-1.f, 0.f)));
                const double lambda = anchor_owner_p_lost().value_or(anchor_lost_.lambda);   // ⚑ D-B hook: the owner's P(lost)
                // fix round 2 (c): aged by the motion ACTUALLY made since the block's stamp (the localiser's pose then vs now)
                const ControllerRobotPose then = policy_pose_at(cp.anchor_pose.stamp_ms).value_or(cpose);
                const auto aged = rc::policy::aged_anchor(cp.anchor_pose, double((cpose.pos - then.pos).norm()),
                                                          double(std::remainder(cpose.theta - then.theta, 2.f * std::numbers::pi_v<float>)));
                const rc::policy::DoorwayLine a_line{aged, cp.anchor_width, 1.0 - lambda};
                const std::optional<rc::policy::DoorwayLine> g_line =
                    cp.on_gap ? std::optional<rc::policy::DoorwayLine>(rc::policy::DoorwayLine{cp.drive_pose, cp.drive_width, cp.gap_score})
                              : std::nullopt;
                const auto choice = rc::policy::choose_doorway_line(a_line, g_line, body_half_w);
                const double t_now = steady_now_s();
                const double dt_b = doorway_belief_t_ < 0.0 ? 0.0 : t_now - doorway_belief_t_;
                // the claim's first cycle has no interval to weigh: the tie (the anchor) until the next one
                if (g_line.has_value()) doorway_belief_.step(choice.pi_anchor, choice.pi_gap, dt_b);
                else doorway_belief_.predict(dt_b);
                doorway_belief_t_ = t_now;
                const bool on_gap_line = g_line.has_value() and doorway_belief_.gap();
                if (not policy_goal_line_known_ or on_gap_line != policy_goal_on_gap_logged_)
                {
                    policy_goal_line_known_ = true;
                    policy_goal_on_gap_logged_ = on_gap_line;
                    std::println("[policy] cross goal line: the {} (P(gap is the door) {:.3f}; this cycle pi_anchor {:.2f} = (1-lambda "
                                 "{:.3f}) x coverage {:.2f} [block {} ms old]; pi_gap {:.2f} = score {:.2f} x coverage {:.2f}; "
                                 "P(both the same opening) {:.2f}, d2 {:.1f})",
                                 on_gap_line ? "measured GAP" : "ANCHOR", doorway_belief_.p_gap(), choice.pi_anchor, lambda,
                                 choice.c_anchor, cp.anchor_age_ms, choice.pi_gap, cp.on_gap ? cp.gap_score : 0.f, choice.c_gap,
                                 choice.p_both_right, choice.d2);
                }
                spec = on_gap_line ? rc::policy::doorway_goal(rc::policy::gap_in_room(cp.drive_pose, cp.drive_width, cpose), gp)
                                   : rc::policy::doorway_goal(rc::policy::gap_in_room(cp.anchor_pose, cp.anchor_width, cpose), gp);
                // the far side is not in the planner's map: the measured opening's corridor joins the near room to the goal
                // line (1 m back into the room just left; one field cell past the line)
                const float back = 1.0f, ahead = gp.past_m + 0.05f;
                // ★ F4: the jambs in the route's risk. Each opening's along-wall / along-normal sigmas from its own pose, the
                //   contact model's shared offset (sigma_rel_xy with the body-geometry sigma PolicyInput carries — the sigma the
                //   policy's rollouts use), each jamb edge measured with the LiDAR's range sigma.
                const float g_body = rc::policy::PolicyInput{}.body_geometry_sigma_m;
                const float sigma_s = std::hypot(params_ != nullptr ? params_->policy_sigma_rel_xy_m : 0.f, g_body);
                const float sigma_r = params_ != nullptr ? params_->lidar_range_sigma_m : ControllerParams{}.lidar_range_sigma_m;
                auto jambs_of = [&](const rc::interaction::RelPose &rp, float width)
                {
                    const auto geo = rc::anchor_cross_geometry(rp);
                    const float sigma_n = std::sqrt(std::max(0.f, geo.n.dot(rp.P.topLeftCorner<2, 2>() * geo.n)));
                    return rc::policy::doorway_jambs(rc::policy::gap_in_room(rp, width, cpose), geo.sigma_lat, sigma_n, sigma_s,
                                                     sigma_r, bare, back, ahead);
                };
                // ★ rtD1c leg 5: the [gap] detector locked onto the open LEAF (0.40-0.52 m, score 0.03-0.3, against an 0.87 m
                //   door) and the corridor followed it. The ANCHOR's opening is the door; the gap measurement is a second
                //   hypothesis weighted by its own score (doorway_corridor.h DoorwayRisk) — and the corridor is either one's.
                rc::policy::DoorwayRisk risk{jambs_of(cp.anchor_pose, cp.anchor_width), std::nullopt, 0.f};
                if (cp.on_gap) { risk.gap = jambs_of(cp.drive_pose, cp.drive_width); risk.p_gap = cp.gap_score; }
                jambs = risk;
                // Task 3f r1 (I-1): held for the goals after this cross while the body is within the doorway's reach
                policy_held_jambs_ = risk;
                policy_held_leaf_ = policy_leaf_;
                policy_held_room_ = current_room_name_;
                corridor = [risk](const Eigen::Vector2f &p) { return rc::policy::in_doorway_risk_corridor(risk, p); };
                // Live fix 2 ruling 2: the measured opening (at least the body's width) is evidence the body fits there — no
                // trap may block it; the bare body's full width = its lateral support both sides
                const float body_w = bare.support_radius(0.f, Eigen::Vector2f(1.f, 0.f)) + bare.support_radius(0.f, Eigen::Vector2f(-1.f, 0.f));
                veto = [risk, body_w](const Eigen::Vector2f &c, float r)
                { return rc::policy::disc_meets_doorway_risk_corridor(risk, c, r, body_w); };
            }
            else
            {
                // a waiting cycle carries no evidence about which line is the door: the hazard alone, so the next drive cycle
                // weighs only its own interval (Task 3e r1)
                if (doorway_belief_t_ >= 0.0)
                {
                    const double t_now = steady_now_s();
                    doorway_belief_.predict(t_now - doorway_belief_t_);
                    doorway_belief_t_ = t_now;
                }
                const float half_len = approach_body_.support_radius(0.f, Eigen::Vector2f(0.f, 1.f));
                spec = rc::policy::heading_goal(cpose.pos, rc::policy::to_policy_pose(cpose).yaw + rc::anchor_wait_aim(cp.pose, half_len),
                                                gp.tol_yaw);
            }
            const auto tick = policy_tick(spec, who + (cp.drive ? "" : " (waiting)"), cpose, obstacle_tracker, motion_commander, now,
                                          corridor, veto, jambs);
            // F1: a Stuck is a hold; the crossing is handed back only on the Unreachable verdict (trapped field / belief)
            // ★ Task 3f (c): past the room switch a field BUILT and READ non-finite at the robot for the no-route clock's span
            //   gives the clearing up (policy_tick_gate.h ClearingNoRoute) — the passage concludes by its room-switch rule
            bool clearing_no_route = false;
            if (door_passage_.clearing())
            {
                using O = rc::policy::NoRouteObs;
                const O obs = not tick.field_evaluated ? O::NotEvaluated : std::isfinite(tick.field_m) ? O::Finite : O::Infinite;
                clearing_no_route = policy_clearing_no_route_.on_tick(now, obs, tick.verdict == PolicyVerdict::Unreachable, tick.stuck,
                                                                      policy_unreach_latch_);
            }
            else
                policy_clearing_no_route_.reset();
            if (tick.verdict == PolicyVerdict::Unreachable)
                cross_hand_back(std::format("the local policy cannot drive it ({})", tick.why), /*timed_out=*/false, motion_commander);
            else if (clearing_no_route)
                cross_hand_back(std::format("the local policy has no route to the goal line for {:.1f} s", rc::policy::kNoRouteReportMs * 1e-3),
                                /*timed_out=*/false, motion_commander);
            // ★ Live fix 4 ruling 4: waiting for an anchor never seen, FACING where it was published, for one plan duration
            //   (2 holds) — nothing at this pose changes it: hand the cross back and let the passage reposition (approach).
            else if (not cp.drive and policy_stale_wait_.on_tick(tick.verdict == PolicyVerdict::Arrived, steady_now_s(),
                                                                2.0 * (params_ != nullptr ? params_->policy_horizon_s : 1.0)))
            {
                cross_hand_back("the anchor stays unpublished from where the robot stands (facing it for a plan duration)",
                                /*timed_out=*/false, motion_commander);
                door_passage_.reposition("the cross's anchor stays unpublished from here");
                policy_stale_wait_.reset();
            }
            if (cp.drive) policy_stale_wait_.reset();

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
                // Live fix 4 ruling 1: the orient goes to its standpoint first (a Place with yaw); its protocol (look / timeout)
                // runs only once there — on the way, an incidental heading match is not the look
                // ★ Task O: the standpoint is the PUBLISHED target (our repair had put it 0.02 m from the robot, 0.18 m short of the
                //   offer), and "there" is the affordance's OWN precondition: the `*_reach_prob >=` clause of ITS contract against
                //   the feedback node (fix round 1, I3 — no sibling lookup); no such clause or no measurement: the distance tolerance.
                ControllerTargetInfo orient_t = *last_target_info_;
                orient_t.room_pos = rc::policy::goal_target_pos(last_raw_target_pos_, orient_t.room_pos,
                                                                published_target_admissible(orient_t));
                const auto pre = contract_precondition(contract);
                // ★ I4: the bound runs from the STEP'S ENTRY with the contract's own timeout (nothing else started a clock before the
                //   robot was settled, so an unmet precondition hovered for ever); the same give-up the protocol's timeout reports
                if (rc::policy::step_entry_expired(orient_entry_ms_, orient_entry_node_, active_target_id_, now, contract.timeout_ms))
                {
                    motion_commander.stop_robot();
                    last_look_succeeded_ = false;
                    qInfo() << "[affordance] orient GIVE_UP (timeout from step entry) | node" << active_target_id_;
                    orient_entry_ms_.reset();
                    finalize_reached(affordance_manager, path_controller, motion_commander, display, robot_pose.pos, time_source());
                    policy_reset();
                    return;
                }
                const bool at_standpoint = rc::policy::orient_settled(cpose, orient_t, gp, pre);
                note_why(rc::why::Cause::Orienting, std::format("orient '{}': {}", last_target_info_->node_name,
                                                                at_standpoint ? "turning in place" : "driving to its standpoint"));
                if (at_standpoint and step_orient(robot_pose, path_controller, obstacle_tracker, motion_commander, time_source,
                                                  last_target_info_->yaw_rad, /*drive_base=*/false))
                {
                    finalize_reached(affordance_manager, path_controller, motion_commander, display, robot_pose.pos, time_source());
                    policy_reset();
                    return;
                }
                if (const auto spec = rc::policy::goal_for_target(orient_t, cpose, std::nullopt, rc::policy::GoalMode::Orient,
                                                                  false, gp, pre))
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
    // ★ Live fix 4 ruling 2: a CROSS whose anchor door_concept has not hung yet is WAITED for, still — never driven as a room
    //   route to the far room's through-pose (tourP8: `back up v=-0.35` toward a target 6.8 m away for a cycle per claim).
    //   door_concept hangs one on every cross claim (the cycle after it sees it); bounded by the cross contract's timeout.
    if (last_target_info_->from_affordance and last_target_info_->node_name.ends_with("_cross"))
    {
        if (policy_no_anchor_since_ms_ == 0) policy_no_anchor_since_ms_ = now;
        if (target_contract_known_ and double(now - policy_no_anchor_since_ms_) > double(target_contract_.timeout_ms))
        {
            cross_hand_back(std::format("no anchor was hung for '{}' within the cross contract's timeout", last_target_info_->node_name),
                            /*timed_out=*/true, motion_commander);
            policy_no_anchor_since_ms_ = 0;
            return;
        }
        stop_here(rc::why::Cause::AnchorWaiting, std::format("'{}': waiting for door_concept to hang the cross's anchor ({:.1f} s)",
                                                             last_target_info_->node_name, (now - policy_no_anchor_since_ms_) * 1e-3));
        return;
    }
    policy_no_anchor_since_ms_ = 0;
    // ★ Task O fix round 1 (I2): toward the PUBLISHED target (the repair is an admissible substitute only), and — when the active
    //   contract states a precondition probability — completed by IT, not by arriving within tol_xy
    ControllerTargetInfo place_t = *last_target_info_;
    std::optional<rc::policy::OrientPrecondition> place_pre;
    if (last_target_info_->from_affordance)
    {
        place_t.room_pos = rc::policy::goal_target_pos(last_raw_target_pos_, place_t.room_pos, published_target_admissible(place_t));
        if (target_contract_known_) place_pre = contract_precondition(target_contract_);
    }
    // ★ fix round 2 (b): a Place that waits on the precondition probability is bounded from its entry by the contract's own timeout
    if (place_pre.has_value() and place_pre->p < place_pre->complete_at
        and rc::policy::step_entry_expired(place_entry_ms_, place_entry_node_, active_target_id_, now, target_contract_.timeout_ms))
    {
        motion_commander.stop_robot();
        qInfo() << "[affordance] place GIVE_UP (precondition unmet, timeout from step entry) | node" << active_target_id_;
        place_entry_ms_.reset();
        finalize_reached(affordance_manager, path_controller, motion_commander, display, robot_pose.pos, time_source());
        policy_reset();
        return;
    }
    const auto spec = rc::policy::goal_for_target(place_t, cpose, std::nullopt, rc::policy::GoalMode::Place,
                                                  wants_final_facing(place_t), gp, place_pre);
    if (not spec.has_value()) { stop_here(rc::why::Cause::PolicyHold, "no goal could be built"); return; }
    const std::string who = last_target_info_->node_name.empty() ? std::string("target") : last_target_info_->node_name;
    const auto tick = policy_tick(*spec, who, cpose, obstacle_tracker, motion_commander, now);
    const bool precondition_met = place_pre.has_value() and place_pre->p >= place_pre->complete_at;
    // ARRIVAL: the same protocol the tracker's arrival goes through (servo → lock-on, already there, approach-only, REACH)
    if (consume_goal_reached(precondition_met or (not place_pre.has_value() and tick.verdict == PolicyVerdict::Arrived), robot_pose, path_controller, obstacle_tracker,
                             motion_commander, display, affordance_manager, time_source, /*drive_base=*/false))
    {
        if (not lockon_.active()) policy_reset();
        return;
    }
    // NO FIELD ROUTE: the goal is not reachable on the map from here — the existing no-route clock reports it (5 s, the same
    // liveness bound the planner's no-route hold has: report_unreachable_if_no_route)
    // review I4 / M4 + Task 3d: "no route" = the FIELD, BUILT and READ at the robot, is non-finite; a tick that evaluated no field
    // (Blind / stale / frame wait / no map) neither starts nor advances the clock (policy_tick_gate.h advance_no_route_clock)
    {
        using O = rc::policy::NoRouteObs;
        // an Unreachable VERDICT on a tick that evaluated no field (the turn-sign EMERGENCY's entry tick) is a statement, not an
        // absence of evidence: it starts the clock; the held ticks after it evaluate nothing and do not advance it
        const O obs = not tick.field_evaluated ? (tick.verdict == PolicyVerdict::Unreachable ? O::Infinite : O::NotEvaluated)
                    : std::isfinite(tick.field_m) ? O::Finite : O::Infinite;
        no_route_since_ms_ = rc::policy::advance_no_route_clock(no_route_since_ms_, now, policy_clock_prev_ms_, obs,
                                                                tick.verdict == PolicyVerdict::Unreachable, tick.stuck, policy_unreach_latch_);
        policy_clock_prev_ms_ = now;
        if (obs == O::Infinite)
            note_why(rc::why::Cause::NoRoute, std::format("{}: the cost-to-go is infinite at the robot — no way to the goal on the map", who));
    }
    // UNREACHABLE: handed back through the protocol (a mouse target is dropped by the no-route clock). F1: a Stuck is a hold —
    // the reachability belief's verdict, not a single Stuck tick, is what hands back
    if (tick.verdict == PolicyVerdict::Unreachable)
    {
        // a Reach is handed back at once (the producer offers another); any other contract / a mouse target goes through
        // the 5-s no-route clock (report_unreachable_if_no_route), which reports or drops it
        if (not flag_reach_unreachable(std::format("local policy: {}", tick.why)) and no_route_since_ms_ == 0)
            no_route_since_ms_ = now;
    }
}
