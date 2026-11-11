// THE HYBRID — PORT STEPS 4 + 5 (2026-11-11; the 2-D mock's run_chain, tests/test_policy_scenarios.cpp, is the reference).
//   Policy.Enabled + Policy.HybridChain: a plain PLACE is the spline chain's (planner -> spline -> PLAIN tracker, with the
//   monitors of controller_chain_monitor.cpp). The chain hands a stretch to the POLICY as an EPISODE when its window fails within
//   reach and the stretch is tight but passable, when its progress prediction fails, when the tracker reports a blockage, a
//   wedge or a spin, or stops short of the standpoint. The episode drives toward the first route point past the failing stretch
//   that stays clear for a body length (or the target itself), on a field whose free space includes the ROUTE CORRIDOR (clean38:
//   the policy's own free map cut the robot off the goal). It ends when the robot is past that point AND the chain's window
//   passes there (else it extends); an impassable stretch or the policy's hand-back re-plans on map + live marks, then puts
//   FAILURE MARKS on the failed section, and with no way left the robot WAITS (step 4) if the map alone routes.
#include "controller_session.h"
#include <cmath>
#include <numbers>
#include "local_policy/contact.h"
#include "local_policy/goal_adapter.h"
#include <chrono>

namespace
{
double steady_now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}

ControllerSession::ChainObs ControllerSession::chain_observe(const rc::policy::Pose2 &bel, ControllerObstacleTracker &obstacle_tracker)
{
    ChainObs o;
    if (params_ == nullptr) return o;
    const auto &body = policy_body_.has_value() ? *policy_body_ : approach_body_;
    const float sigma_r = params_->lidar_range_sigma_m, cell = grid_planner_.params.cell_size_m;
    std::vector<rc::policy::Return> returns;
    std::vector<std::pair<float, float>> bearing_range;
    if (auto *buf = obstacle_tracker.contact_buffer(); buf != nullptr)
        if (const auto [cloud_opt] = buf->read_last(); cloud_opt.has_value())
        {
            const auto &[xs, ys, zs] = cloud_opt.value();
            const std::size_t n = std::min(xs.size(), ys.size());
            const float reach = 4.0f + body.circumscribed_radius() + 6.f * sigma_r;   // the window's span
            for (std::size_t i = 0; i < n; ++i)
            {
                const Eigen::Vector2f q(xs[i], ys[i]);
                if (not q.allFinite()) continue;
                const Eigen::Vector2f r = q - bel.p;
                bearing_range.emplace_back(std::atan2(r.y(), r.x()), r.norm());
                if (r.squaredNorm() <= reach * reach) returns.push_back({.p = q, .sigma = sigma_r, .p_exist = 1.f});
            }
        }
    o.cells = rc::policy::denoise_returns(returns);
    if (not chain_marks_.marks.empty() and not bearing_range.empty())
    {
        auto look = chain_marks_.observe(bel.p, bearing_range, cell, body.circumscribed_radius(), params_->policy_risk_eps,
                                         2.f * body.circumscribed_radius());
        o.cleared = look.cleared;
        o.cleared_at = std::move(look.cleared_at);
    }
    if (o.cleared > 0)
    {
        chain_marks_off_ = false;   // the marks changed: the planner tries them again
        if (chain_wait_)
        {   // STEP 4: the way may be open — the planner tries map + marks again
            chain_wait_ = false;
            current_plan_.reset();
            plan_spline_valid_ = false;
            std::println("[chain] WAIT over: {} mark(s) seen gone, re-planning on map + {} marks", o.cleared, chain_marks_.marks.size());
        }
    }
    return o;
}

void ControllerSession::chain_start_episode(const rc::policy::Pose2 &bel, const std::vector<rc::policy::Return> &cells, int first_bad,
                                            const char *why)
{
    const auto &body = policy_body_.has_value() ? *policy_body_ : approach_body_;
    const bool have_route = plan_spline_valid_ and plan_spline_.valid();
    const float risk_eps = params_->policy_risk_eps, lp_ok = std::log1p(-risk_eps);
    const float r_c = body.circumscribed_radius();
    ChainEpisode ep;
    ep.active = true;
    ep.why = why;
    ep.final = true;   // default: the target itself
    ep.next_pass_s = steady_now_s();
    const float s_prog = have_route ? plan_spline_.project(bel.p, chain_s_prog_, 3.0f) : 0.f;
    ep.s_end = have_route ? std::min(plan_spline_.length(), s_prog + 4.0f) : 0.f;
    // the subgoal: the first route point past the failing stretch that stays clear for a body length (the mock's start_episode);
    // "finish" (the tracker stopped short) hands over the target itself
    if (have_route and std::string_view(why) != "finish")
    {
        const float s_bad = s_prog + 0.10f * float(std::max(0, first_bad));
        float clear_run = 0.f;
        for (float s = s_bad; s + 0.10f <= plan_spline_.length(); s += 0.10f)
        {
            const rc::policy::Pose2 seg[2] = {{plan_spline_.position_at(s), plan_spline_.heading_at(s)},
                                              {plan_spline_.position_at(s + 0.10f), plan_spline_.heading_at(s + 0.10f)}};
            if (rc::policy::log_p_no_contact_cells(body, seg, cells, chain_sigma_track_now_, params_->chain_sigma_track_yaw, 0.01f) >= lp_ok)
                clear_run += 0.10f;
            else clear_run = 0.f;
            if (clear_run >= 2.f * r_c) { ep.sub = seg[0].p; ep.final = false; ep.s_end = s; break; }
        }
    }
    // THE ROUTE CORRIDOR on the planner's raster: a disc of the planner's clearance at every route sample, and one at the robot
    // reaching its projection on the route — connected by construction wherever the route runs
    ep.corr_origin = grid_planner_.grid_origin();
    ep.corr_cell = grid_planner_.cell_size();
    ep.corr_w = grid_planner_.width();
    ep.corr_h = grid_planner_.height();
    ep.corr = std::make_shared<std::vector<char>>(std::size_t(std::max(0, ep.corr_w * ep.corr_h)), 0);
    const auto disc = [&ep](const Eigen::Vector2f &c, float r)
    {
        const int i0 = int(std::floor((c.x() - r - ep.corr_origin.x()) / ep.corr_cell)), i1 = int(std::ceil((c.x() + r - ep.corr_origin.x()) / ep.corr_cell));
        const int j0 = int(std::floor((c.y() - r - ep.corr_origin.y()) / ep.corr_cell)), j1 = int(std::ceil((c.y() + r - ep.corr_origin.y()) / ep.corr_cell));
        for (int j = std::max(0, j0); j <= std::min(ep.corr_h - 1, j1); ++j)
            for (int i = std::max(0, i0); i <= std::min(ep.corr_w - 1, i1); ++i)
                if ((ep.corr_origin + Eigen::Vector2f(float(i), float(j)) * ep.corr_cell - c).norm() <= r)
                    (*ep.corr)[std::size_t(j * ep.corr_w + i)] = 1;
    };
    if (have_route)
    {
        for (float s = 0.f; s <= plan_spline_.length(); s += ep.corr_cell)
        { const Eigen::Vector2f q = plan_spline_.position_at(s); disc(q, std::max(ep.corr_cell, grid_planner_.distance_at(q))); }
        disc(bel.p, std::max(grid_planner_.distance_at(bel.p), (bel.p - plan_spline_.position_at(s_prog)).norm() + ep.corr_cell));
    }
    else disc(bel.p, std::max(ep.corr_cell, grid_planner_.distance_at(bel.p)));
    chain_ep_ = std::move(ep);
    chain_ep_marks_ = chain_marks_.polygons(grid_planner_.params.cell_size_m);   // frozen in the planner world for the episode
    policy_reset();   // a new goal for the policy (its field, plan and evidence belong to the previous one)
    if (chain_ep_.final)
        std::println("[chain] HAND-OFF ({}) -> the policy drives to the target itself", why);
    else
        std::println("[chain] HAND-OFF ({}) -> the policy drives to ({:.2f},{:.2f}), route s {:.2f}", why, chain_ep_.sub.x(),
                     chain_ep_.sub.y(), chain_ep_.s_end);
}

void ControllerSession::chain_episode_failed(const rc::policy::Pose2 &bel, const std::vector<rc::policy::Return> &cells, float s0,
                                             float s1, const char *why)
{
    const auto &body = policy_body_.has_value() ? *policy_body_ : approach_body_;
    const float cell = grid_planner_.params.cell_size_m;
    chain_ep_ = {};
    policy_reset();
    chain_progress_.clear();
    // (1) re-plan on map + LIVE marks (the mock's replan_live) — unless the last re-plan was within a body radius: it did not help
    if ((bel.p - chain_last_replan_at_).norm() > body.circumscribed_radius())
    {
        chain_last_replan_at_ = bel.p;
        const rc::chain::UnmappedTest unmapped{&grid_planner_, rc::chain::two_sided_quantile(params_->policy_risk_eps),
                                               params_->lidar_range_sigma_m, params_->chain_map_sigma};
        std::size_t added = 0;
        for (const auto &c : cells) if (unmapped(c.p, policy_pose_xy_std_) and chain_marks_.add(c.p, cell)) ++added;
        chain_marks_off_ = false;
        current_plan_.reset();
        plan_spline_valid_ = false;
        std::println("[chain] EPISODE FAILED ({}): re-plan on map + live ({} new marks, {} total)", why, added, chain_marks_.marks.size());
        return;
    }
    // (2) the failure itself is the evidence: close the route's centre line over the failed section (failure marks) and re-plan;
    //     no way left -> the planner's marks fallback -> WAIT if the map alone routes (step 4), else the no-route clock
    if (plan_spline_valid_ and plan_spline_.valid())
        for (float s = std::max(0.f, s0); s <= std::min(plan_spline_.length(), s1); s += cell)
            chain_marks_.add_fail(plan_spline_.position_at(s));
    chain_marks_off_ = false;
    current_plan_.reset();
    plan_spline_valid_ = false;
    std::println("[chain] EPISODE FAILED ({}) again here: failure marks on route s {:.2f}..{:.2f} ({} total), re-plan", why, s0, s1,
                 chain_marks_.fail.size());
}

bool ControllerSession::chain_drive_episode(const rc::policy::GoalSpec &final_spec, const std::string &who,
                                            const ControllerRobotPose &robot_pose, const ControllerRobotPose &cpose,
                                            rc::TrajectoryController &path_controller, ControllerObstacleTracker &obstacle_tracker,
                                            ControllerMotionCommander &motion_commander, ControllerDisplay &display,
                                            rc::AffordanceManager &affordance_manager, const TimeSource &time_source)
{
    const rc::policy::Pose2 bel = rc::policy::to_policy_pose(robot_pose);
    const auto &body = policy_body_.has_value() ? *policy_body_ : approach_body_;
    const float risk_eps = params_->policy_risk_eps, k_eps = rc::chain::two_sided_quantile(risk_eps);
    const float r_c = body.circumscribed_radius();
    const ChainObs obs = chain_observe(bel, obstacle_tracker);
    const bool have_route = plan_spline_valid_ and plan_spline_.valid();
    const float s_prog = have_route ? plan_spline_.project(bel.p, chain_s_prog_, 3.0f) : 0.f;
    if (have_route) chain_s_prog_ = s_prog;
    const double now_s = steady_now_s();
    // PASSABILITY of the handed-over stretch, twice a second (the LiDAR's own refresh of what lies ahead is not faster)
    if (have_route and now_s >= chain_ep_.next_pass_s)
    {
        chain_ep_.next_pass_s = now_s + 0.5;
        const float blocked = rc::chain::stretch_blocked_at(body, plan_spline_, grid_planner_, s_prog, std::max(s_prog, chain_ep_.s_end),
                                                            obs.cells, bel, risk_eps, k_eps,
                                                            [this](const Eigen::Vector2f &) { return policy_pose_xy_std_; },
                                                            params_->chain_map_sigma).blocked_at;
        if (blocked >= 0.f)
        {
            note_no_command(); motion_commander.stop_robot();
            chain_episode_failed(bel, obs.cells, blocked - 0.10f, blocked + 0.10f, "the stretch is impassable");
            return true;
        }
    }
    // the policy, toward the episode's goal; its field's free space = the map's ∪ the route corridor
    const auto corr = chain_ep_.corr;
    const Eigen::Vector2f co = chain_ep_.corr_origin;
    const float cc = chain_ep_.corr_cell;
    const int cw = chain_ep_.corr_w, ch = chain_ep_.corr_h;
    auto also_free = [corr, co, cc, cw, ch](const Eigen::Vector2f &p)
    {
        const int i = int(std::lround((p.x() - co.x()) / cc)), j = int(std::lround((p.y() - co.y()) / cc));
        return corr != nullptr and i >= 0 and j >= 0 and i < cw and j < ch and (*corr)[std::size_t(j * cw + i)] != 0;
    };
    const rc::policy::GoalSpec spec = chain_ep_.final ? final_spec
        : rc::policy::place_goal(chain_ep_.sub, std::nullopt, final_spec.goal.tol_xy, final_spec.goal.tol_yaw);
    const auto tick = policy_tick(spec, std::format("{} (episode: {})", who, chain_ep_.why), cpose, obstacle_tracker,
                                  motion_commander, time_source(), also_free);
    if (chain_ep_.final)
    {
        if (tick.verdict == PolicyVerdict::Arrived)
        {
            chain_ep_ = {};
            if (consume_goal_reached(true, robot_pose, path_controller, obstacle_tracker, motion_commander, display, affordance_manager,
                                     time_source, /*drive_base=*/false))
                if (not lockon_.active()) policy_reset();
            return true;
        }
    }
    else
    {
        // past the stretch along the route — and the chain can take over HERE (its window from this pose, over the tracker's
        // reach plus a body length, passes): the episode is over; else it EXTENDS to the next clear route point
        bool past = false;
        if (have_route and s_prog >= chain_ep_.s_end)
        {
            const float reach_m = std::abs(odo_v_) * path_controller.params.plain_T_lag + 0.5f * path_controller.params.plain_W + r_c;
            const auto win = rc::chain::window_test(body, plan_spline_, s_prog, bel, obs.cells, chain_sigma_track_now_,
                                                    params_->chain_sigma_track_yaw, risk_eps, reach_m + 2.f * r_c);
            if (win.first_bad < 0) past = true;
            else
            {
                std::println("[chain] EPISODE EXTENDED past route s {:.2f}", s_prog);
                chain_start_episode(bel, obs.cells, 0, "extend");
                return true;
            }
        }
        if (past or tick.verdict == PolicyVerdict::Arrived)
        {
            std::println("[chain] EPISODE OVER ({}): the chain re-plans from here (route s {:.2f})", chain_ep_.why, s_prog);
            chain_ep_ = {};
            policy_reset();
            chain_progress_.clear();
            current_plan_.reset();
            plan_spline_valid_ = false;
            return true;
        }
    }
    // the policy cannot reach its stretch: evidence for the planner, not the end of the goal (Fable §1.4)
    if (tick.verdict == PolicyVerdict::Unreachable)
    {
        note_no_command(); motion_commander.stop_robot();
        chain_episode_failed(bel, obs.cells, s_prog, s_prog + 2.f * r_c, "the policy handed back");
    }
    return true;
}
