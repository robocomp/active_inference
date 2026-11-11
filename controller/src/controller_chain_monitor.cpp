// THE CHAIN'S MODEL-BASED MONITORS, LIVE — STEP 2 OF THE PORT: LOGGING ONLY (2026-11-11).
// STEP 3 (2026-11-11, owner "go ahead with step 3"): while the CHAIN drives (Policy.Enabled = false) the marks enter the planner's
// world (sync_world_state), a plan-first verdict re-plans (the return value), and sigma_track is the posterior over the measured
// cross-track error. The hand-off verdict (4) stays logging only until step 5; WAIT is step 4.
// Fable's porting order: (1) grid_planner speed-up, (2) window test + plan-first + marks LOGGING ONLY, (3) marks with sigma +
// clearing driving the planner, (4) WAIT, (5) episodes/passability/extension. This file is (2): every cycle the spline chain
// drives, the shared monitors (chain_monitor/chain_monitor.h — the SAME code the 2-D mock runs) judge what the chain WOULD
// do, and chain_monitor.csv records it. Nothing here changes a command, a plan or the planner's world.
// What the log is for: the live sigma_track (cross-track rms), the false-unmapped rate while parked (marks that would be
// drawn under the pose covariance), the window/plan-first verdict rates on real routes, and this step's own cost.
#include "controller_session.h"
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include "local_policy/contact.h"
#include "local_policy/goal_adapter.h"

ControllerSession::ChainAction ControllerSession::step_chain_monitor(const ControllerRobotPose &robot_pose, const rc::ControlOutput &out,
                                           ControllerObstacleTracker &obstacle_tracker,
                                           const rc::TrajectoryController &path_controller)
{
    if (params_ == nullptr or not params_->chain_monitor_enabled) return ChainAction::None;
    const auto t_start = std::chrono::steady_clock::now();
    const DrivenCurve curve = driven_curve();
    const std::uint64_t now_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch()).count());
    if (not chain_csv_.is_open())
    {
        chain_csv_.open("chain_monitor.csv", std::ios::out | std::ios::trunc);
        chain_csv_.imbue(std::locale::classic());
        if (chain_csv_.is_open())
            chain_csv_ << "# chain_monitor.csv — the chain's monitors in LOGGING-ONLY mode (controller_chain_monitor.cpp)\n"
                          "# verdict: 0 clear, 1 failing stretch seen not yet reached, 2 plan-first (unmapped cause), 3 plan-first (impassable), 4 hand-off to the policy\n"
                          "t_ms,owner,s_prog,len,cross_m,xtrack_rms,sigma_track,pose_xy_std,pose_age_ms,v,w,n_returns,n_unmapped_2m,"
                          "win_lp,win_first_bad,s_bad,reach_m,verdict,pass_blocked_at,marks,fail_marks,marks_added,marks_cleared,"
                          "mon_pred,mon_real,mon_resn,mon_H,mon_fail,mon_idle,mon_censored,cost_ms,sigma_yaw,carry_m,replan,marks_in_world,bel_x,bel_y,bel_yaw,odo_x,odo_y,odo_yaw,dt_pose,odo_period\n";
    }
    const rc::policy::Pose2 bel = rc::policy::to_policy_pose(robot_pose);
    const float cell = grid_planner_.params.cell_size_m;
    const auto &body = policy_body_.has_value() ? *policy_body_ : approach_body_;
    const float risk_eps = params_->policy_risk_eps;
    const float k_eps = rc::chain::two_sided_quantile(risk_eps);
    const float sigma_r = params_->lidar_range_sigma_m;
    const float now_s = float(double(now_ms % 100000000ULL) * 1e-3);
    const float dt = chain_last_t_s_ > 0.f and now_s > chain_last_t_s_ ? now_s - chain_last_t_s_ : 0.05f;
    chain_last_t_s_ = now_s;
    // THE MARKS' FRAME. Marks live in the room frame they were drawn in; a room switch drops them (another frame).
    if (current_room_name_ != chain_marks_room_)
    { chain_marks_ = {}; chain_marks_room_ = current_room_name_; chain_marks_off_ = false; chain_have_prev_ = false; }
    // ★ MARKS FOLLOW RELOCALISATIONS, NOT ODOMETRY DRIFT (as the mock): odometry integrated from the measured speeds carries
    //   a mark room_prev -> odom -> room_now; a move beyond k_eps x slip sigma (and half a cell) is the localiser's step, and
    //   every mark goes with it. Only with valid odometry — stale speeds would read as a jump and drag the marks along.
    //   ★ ON THE POSE'S OWN CLOCK (live chain3, 2026-11-11): integrated over the WALL interval, the comparison with the
    //   scan-stamped pose was off by (stamp jitter) x (v + w r) — at 1 rad/s, 2 m lever, 40 ms: 8 cm — and "relocalisations"
    //   fired on half the cycles, dragging every mark through every turn. The increment is now integrated over the interval
    //   between the believed pose's own stamps (the LiDAR scan it is pinned to), and the slip sigma carries the integration's timing
    //   uncertainty: speeds are known only at their sample instants, sigma_t = (odometry period)/sqrt(12), measured.
    float carry = -1.f;
    // robot_pose is SCAN-ALIGNED (read_robot_pose_in_room pins it to the last LiDAR stamp): its clock is the scan's stamp, not
    // the RT edge's newest block (pose_stamp_age_ms measures that one — chain4 integrated over it and still fired 127 carries)
    // ★ AND NOT PAST THE RT EDGE'S NEWEST BLOCK (live chain6/7, 2026-11-11): in 15-16 % of moving cycles the scan is newer than
    //   every block, the read CLAMPS to the newest one, and robot_pose repeats last cycle's value exactly — with or without
    //   room_concept's late-correction rewrite (A/B). Its true stamp is then the newest block's (now - pose age), not the scan's.
    const double newest_block_ms = world_model_pose_stamp_age_ms_ >= 0.f ? double(overlay_now_ms_) - double(world_model_pose_stamp_age_ms_)
                                                                         : std::numeric_limits<double>::infinity();
    const double pose_stamp_ms = overlay_lidar_ts_ms_.has_value() ? std::min(double(*overlay_lidar_ts_ms_), newest_block_ms)
                                                                  : (std::isfinite(newest_block_ms) ? newest_block_ms : double(now_ms));
    if (odo_valid_ and odo_stamp_ms_ > chain_prev_odo_stamp_ms_ and chain_prev_odo_stamp_ms_ != 0)
    {
        chain_odo_period_s_ += (double(odo_stamp_ms_ - chain_prev_odo_stamp_ms_) * 1e-3 - chain_odo_period_s_) / (chain_odo_period_n_ + 1.0);
        chain_odo_period_n_ = std::min(chain_odo_period_n_ + 1.0, 200.0);   // a running mean over the recent samples
    }
    if (odo_stamp_ms_ != 0) chain_prev_odo_stamp_ms_ = odo_stamp_ms_;
    const float dt_pose = chain_have_prev_ ? float(std::max(0.0, pose_stamp_ms - chain_prev_pose_stamp_ms_) * 1e-3) : 0.f;
    if (odo_valid_)
    {
        chain_odom_.p += dt_pose * odo_v_ * Eigen::Vector2f(std::cos(chain_odom_.yaw + 0.5f * odo_w_ * dt_pose), std::sin(chain_odom_.yaw + 0.5f * odo_w_ * dt_pose));
        chain_odom_.yaw += dt_pose * odo_w_;
        if (params_->chain_marks_carry and chain_have_prev_ and not chain_marks_.empty())
        {
            const auto rot = [](float a, const Eigen::Vector2f &v) { return Eigen::Vector2f(Eigen::Rotation2Df(a) * v); };
            const auto bp = chain_prev_bel_, op = chain_prev_odom_, oc = chain_odom_;
            const auto transport = [&](const Eigen::Vector2f &q)
            { const Eigen::Vector2f qo = op.p + rot(op.yaw - bp.yaw, q - bp.p); return Eigen::Vector2f(bel.p + rot(bel.yaw - oc.yaw, qo - oc.p)); };
            const float dp = (oc.p - op.p).norm(), dth = std::abs(std::remainder(oc.yaw - op.yaw, 2.f * std::numbers::pi_v<float>));
            const float slip = params_->chain_slip_sigma, sig_t = float(chain_odo_period_s_ / std::sqrt(12.0));
            const float av = std::abs(odo_v_), aw = std::abs(odo_w_);
            carry = chain_marks_.carry_if_relocalised(transport, [&](const Eigen::Vector2f &c)
                    { const float r = (c - bel.p).norm(); return slip * (dp + dth * r) + sig_t * (av + aw * r) + 1e-4f; }, k_eps, cell);
            if (carry >= 0.f) std::println("[chain] relocalisation step: {} marks carried ({:.2f} m)", chain_marks_.marks.size(), carry);
        }
        chain_prev_odom_ = chain_odom_; chain_prev_bel_ = bel; chain_have_prev_ = true; chain_prev_pose_stamp_ms_ = pose_stamp_ms;
    }
    else chain_have_prev_ = false;
    if (not curve.valid())
    {
        chain_progress_.clear();
        chain_have_s_ = false;
        return ChainAction::None;
    }
    const rc::RouteSpline &spline = *curve.spline;
    const float sigma_loc = policy_pose_xy_std_;   // room_concept's published pose xy std (0 = none: sigma_map carries it)

    // projection, cross-track and its running rms (the live sigma_track)
    // THE ROUTE'S OWN PROGRESS. A mission route is projected by its follower (route_.progress()). A point plan lives in ONE
    // reused object (plan_spline_), so a re-plan is invisible by address: live chain1 (2026-11-11) kept projecting from the
    // OLD arc length onto the new curve — |cross| 2.3 m, most hand-off verdicts and monitor fails were that artefact.
    // The plan's generation counter says when the curve is new; the projection then starts again from the route start.
    const bool mission = curve.on_mission();
    const bool fresh = not chain_have_s_ or chain_last_owner_ != int(mission) or (not mission and chain_last_gen_ != plan_spline_gen_);
    if (fresh) chain_progress_.clear();   // the monitor's history is arc length along the OLD curve
    if (mission) chain_s_prog_ = route_.progress();
    else chain_s_prog_ = spline.project(bel.p, fresh ? 0.f : chain_s_prog_, 3.0f);
    chain_have_s_ = true; chain_last_gen_ = plan_spline_gen_; chain_last_owner_ = int(mission);
    const float s_prog = chain_s_prog_;
    const float hd = spline.heading_at(s_prog);
    const Eigen::Vector2f d = bel.p - spline.position_at(s_prog);
    const float cross = std::cos(hd) * d.y() - std::sin(hd) * d.x();
    // ★ sigma_track = THE POSTERIOR over the measured tracking error (step 3): the prior sigma is worth chain_sigma_track_prior_m
    //   metres of driving; every cycle the tracker drives the route adds its cross-track and heading error weighted by the
    //   distance driven (a pivot drives none and so says nothing about lateral tracking — no speed gate needed).
    const float yaw_err = std::remainder(bel.yaw - hd, 2.f * std::numbers::pi_v<float>);
    if (path_controller.is_active() and not fresh and odo_valid_)
    {
        const double ds = double(std::max(0.f, odo_v_)) * dt;
        chain_xt_w2_ += double(cross) * cross * ds; chain_yaw_w2_ += double(yaw_err) * yaw_err * ds; chain_xt_m_ += ds;
    }
    const float xtrack_rms = chain_xt_m_ > 0.0 ? float(std::sqrt(chain_xt_w2_ / chain_xt_m_)) : -1.f;
    const double m0 = params_->chain_sigma_track_prior_m, s0 = params_->chain_sigma_track, y0 = params_->chain_sigma_track_yaw;
    const float sigma_track = float(std::sqrt((m0 * s0 * s0 + chain_xt_w2_) / (m0 + chain_xt_m_)));
    // ★ The HEADING sigma stays the prior (live chain3: learned, it rose to 0.27 rad and failed the window at its pivot on 82 % of
    //   cycles). The heading error is NOT noise about the tangent: a pursuit tracker steers off it on purpose entering a route
    //   and through a bend. Logged (sigma_yaw = measured rms) for the record, never fed to the window.
    const float sigma_track_yaw = float(y0);
    const float yaw_rms = chain_xt_m_ > 0.0 ? float(std::sqrt(chain_yaw_w2_ / chain_xt_m_)) : -1.f;

    // the live returns and this sweep's look of the marks' belief (chain_observe, controller_chain_hybrid.cpp)
    chain_sigma_track_now_ = sigma_track;
    const ChainObs obs = chain_observe(bel, obstacle_tracker);
    const auto &cells = obs.cells;
    const rc::chain::UnmappedTest unmapped{&grid_planner_, k_eps, sigma_r, params_->chain_map_sigma};
    int n_unmapped_2m = 0;
    for (const auto &c : cells)
        if ((c.p - bel.p).norm() <= 2.f and unmapped(c.p, sigma_loc)) ++n_unmapped_2m;

    // (a) the window ahead and the verdict the hybrid would draw
    const auto win = rc::chain::window_test(body, spline, s_prog, bel, cells, sigma_track, sigma_track_yaw, risk_eps);
    const float reach_m = std::abs(odo_v_) * path_controller.params.plain_T_lag + 0.5f * path_controller.params.plain_W
                        + body.circumscribed_radius();
    const float s_bad = rc::chain::window_s_bad(win, s_prog);
    int verdict = 0; float pass_blocked = -1.f; std::size_t added = 0; bool replan = false; const char *handoff = nullptr;
    if (win.first_bad >= 0)
    {
        if (s_bad - s_prog > reach_m) verdict = 1;
        else
        {
            const auto &wb = win.poses[std::size_t(win.first_bad)];
            bool unmapped_cause = false;
            for (const auto &c : cells)
                if (rc::policy::hull_distance(body, wb, c.p) <= 3.f * sigma_track + cell and unmapped(c.p, sigma_loc)) { unmapped_cause = true; break; }
            if (not unmapped_cause)
                pass_blocked = rc::chain::stretch_blocked_at(body, spline, grid_planner_, s_prog, std::min(spline.length(), s_prog + 4.0f),
                                                             cells, bel, risk_eps, k_eps, [&](const Eigen::Vector2f &) { return sigma_loc; },
                                                             params_->chain_map_sigma).blocked_at;
            verdict = unmapped_cause ? 2 : (pass_blocked >= 0.f ? 3 : 4);
            // PLAN-FIRST (the mock's replan_live): mark the unmapped cells and re-plan on map + marks, once per body radius of
            // travel (it did not help last time). Step 2 drew them as shadows; step 3 hands them to the planner and re-plans.
            if ((verdict == 2 or verdict == 3) and (bel.p - chain_last_mark_at_).norm() > body.circumscribed_radius())
            {
                chain_last_mark_at_ = bel.p;
                for (const auto &c : cells) if (unmapped(c.p, sigma_loc) and chain_marks_.add(c.p, cell)) ++added;
                replan = chain_marks_drive();
            }
            // ★ STEP 5: tight but passable (4), or a plan-first that did not help (the last re-plan was within a body radius): the
            //   stretch is the POLICY's — an episode (the mock's start_episode "window")
            else handoff = "window";
        }
    }
    // (b) the marks' belief looked on this sweep in chain_observe; a cleared mark re-plans only if it could have shaped the
    // route: within the planner's own clearance reach of it (the body plus the clearance preference the A* cost prices). Live
    // chain2 cleared ~1 mark/s — re-planning on every one would stop-start the robot for nothing. And even then only for a
    // SHORTER route (drive_point_target, next cycle, once the planner's world has lost the cleared mark: rc::chain::shorter_route).
    const std::size_t cleared = obs.cleared;
    {
        const float shaped = body.circumscribed_radius() + grid_planner_.params.clearance_pref_m;
        for (const auto &c : obs.cleared_at)
            if ((c - spline.position_at(spline.project(c, s_prog, spline.length()))).norm() <= shaped)
            { chain_clear_check_ = chain_marks_drive(); chain_last_mark_at_ = Eigen::Vector2f(1e9f, 1e9f); break; }
    }
    if (added > 0 or cleared > 0) chain_marks_off_ = false;   // the marks changed: the planner tries them again

    // (c) the progress monitor (the tracker's own horizon, the pose's own age)
    const float pose_age_s = world_model_pose_stamp_age_ms_ >= 0.f ? world_model_pose_stamp_age_ms_ * 1e-3f : 0.f;
    rc::chain::ProgressMonitor::In mi;
    mi.t = now_s; mi.s_prog = s_prog; mi.dt = dt;
    mi.T_lag = path_controller.params.plain_T_lag; mi.W = path_controller.params.plain_W;
    mi.v_max = params_->max_adv_speed_mps; mi.v = odo_v_; mi.w = odo_w_; mi.age = pose_age_s; mi.sigma_pose = sigma_loc;
    mi.circumscribed_r = body.circumscribed_radius();
    if (last_target_info_.has_value())
    { mi.goal_applies = true; mi.goal_dist = (last_target_info_->room_pos - robot_pose.pos).norm(); mi.goal_tol = path_controller.goal_threshold(); }
    const auto mv = chain_progress_.judge(mi);
    if (mv.fail and handoff == nullptr) handoff = "stall";   // ★ STEP 5: the chain's own progress prediction failed
    chain_progress_.record(now_s, out.adv, s_prog, out.rot);   // adv: the tracker's forward command (its own frame)

    const float cost_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    if (chain_csv_.is_open())
    {
        chain_csv_ << now_ms << ',' << (curve.on_mission() ? 1 : 0) << ',' << s_prog << ',' << spline.length() << ',' << cross << ','
                   << xtrack_rms << ',' << sigma_track << ',' << sigma_loc << ',' << pose_age_s * 1e3f << ',' << odo_v_ << ',' << odo_w_ << ','
                   << cells.size() << ',' << n_unmapped_2m << ',' << win.lp_worst << ',' << win.first_bad << ',' << s_bad << ',' << reach_m << ','
                   << verdict << ',' << pass_blocked << ',' << chain_marks_.marks.size() << ',' << chain_marks_.fail.size() << ','
                   << added << ',' << cleared << ',' << mv.pred << ',' << mv.realised << ',' << mv.resn << ',' << mv.H << ','
                   << (mv.fail ? 1 : 0) << ',' << (mv.idle ? 1 : 0) << ',' << (mv.censored ? 1 : 0) << ',' << cost_ms << ','
                   << yaw_rms << ',' << carry << ',' << (replan ? 1 : 0) << ',' << (chain_marks_in_world_ ? 1 : 0) << ','
                   << bel.p.x() << ',' << bel.p.y() << ',' << bel.yaw << ',' << chain_odom_.p.x() << ',' << chain_odom_.p.y() << ','
                   << chain_odom_.yaw << ',' << dt_pose << ',' << chain_odo_period_s_ << '\n';
        chain_csv_.flush();
    }
    chain_last_verdict_ = verdict; chain_last_ahead_m_ = win.first_bad >= 0 ? s_bad - s_prog : 0.f; chain_last_stall_ = mv.fail;
    chain_last_monitor_ms_ = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch()).count());
    if (replan) return ChainAction::Replan;
    if (handoff != nullptr and chain_hybrid_on())
    {
        chain_ep_request_ = handoff;
        chain_ep_first_bad_ = std::string_view(handoff) == "window" ? std::max(0, win.first_bad) : 0;
        chain_progress_.clear();
        return ChainAction::Episode;
    }
    return ChainAction::None;
}
