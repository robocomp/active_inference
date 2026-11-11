// The status column's words (controller_status_text.h). GUI thread only; composes from copies, keeps no controller state.
#include "controller_status_text.h"
#include "controller_mission.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <string_view>
#include <vector>

namespace rc::status_text
{
namespace
{
constexpr std::size_t kMaxEvents = 6;

float deg(float rad) { return rad * 180.f / std::numbers::pi_v<float>; }

std::string ago(std::uint64_t now_ms, std::uint64_t t_ms)
{
    const std::uint64_t d = now_ms > t_ms ? now_ms - t_ms : 0;
    if (d < 60'000) return std::format("{} s ago", d / 1000);
    if (d < 3'600'000) return std::format("{} min ago", d / 60'000);
    return std::format("{} h ago", d / 3'600'000);
}

std::string age_text(float ms)
{
    if (ms < 0.f) return "unknown age";
    if (ms < 1000.f) return std::format("{:.0f} ms old", ms);
    return std::format("{:.1f} s old", ms * 1e-3f);
}

std::string plural(int n, std::string_view one, std::string_view many) { return std::format("{} {}", n, n == 1 ? one : many); }

/// why a stretch was handed to the policy, in words (chain_ep_.why / chain_ep_request_)
std::string handoff_words(std::string_view why)
{
    if (why == "window") return "the path ahead is tight but passable";
    if (why == "stall") return "the robot was not advancing as predicted";
    if (why == "blocked") return "the tracker found the path blocked";
    if (why == "finish") return "the tracker stopped short of the goal";
    if (why == "wedge") return "the robot got wedged";
    if (why == "spin") return "the robot was spinning in place";
    if (why == "extend") return "the tight stretch goes on";
    return std::string(why);
}

/// the motion verdict's cause as a plain sentence fragment
std::string cause_words(rc::why::Cause c)
{
    using C = rc::why::Cause;
    switch (c)
    {
        case C::PresenceDegraded: return "a required agent is missing";
        case C::LidarStalled:     return "no fresh LiDAR data";
        case C::Halted:           return "driving is switched off";
        case C::Disarmed:         return "the base output is disarmed";
        case C::NoWorld:          return "the room or robot is not in the graph yet";
        case C::NoPose:           return "no robot pose in the room";
        case C::RelativeMotion:   return "carrying out a move/turn order";
        case C::Escape:           return "backing out after getting stuck";
        case C::ReturnsBackoff:   return "backing away from obstacles too close";
        case C::PassageWaiting:   return "the door passage waits for its next step";
        case C::PassageClearing:  return "driving on through the doorway";
        case C::AnchorWaiting:    return "waiting to see the door again";
        case C::AnchorHandBack:   return "the door crossing was given back";
        case C::MissionWaiting:   return "the mission says wait";
        case C::Dwell:            return "looking at what it just reached";
        case C::TargetRejected:   return "the goal was refused";
        case C::NoRoute:          return "no route to the goal";
        case C::Blockage:         return "the path is blocked, re-routing";
        case C::PathShort:        return "the path ends short of the goal";
        case C::ReturnsHold:      return "slowed by nearby obstacles";
        case C::PolicyHold:       return "the local policy is holding";
        case C::LockOn:           return "fine alignment (lock-on)";
        case C::Orienting:        return "turning to face its goal";
        case C::Arrived:          return "arrived";
        case C::MissionDone:      return "the mission is complete";
        case C::NoTarget:         return "nothing to do";
        case C::TrackerZero:      return "the tracker asked to stop";
        case C::Unexplained:      return "stopped, no reason given";
        case C::Driving:          return "driving to its goal";
    }
    return "?";
}

std::string speed_words(float adv, float rot)
{
    std::string s = std::abs(adv) < 0.005f ? std::string("standing")
                  : std::format("{:.2f} m/s {}", std::abs(adv), adv > 0.f ? "forward" : "backward");
    if (std::abs(rot) >= 0.005f)
        s += std::format(", turning {} {:.0f}°/s", rot > 0.f ? "left" : "right", std::abs(deg(rot)));
    return s;
}
}   // namespace

// ── RECENT HAPPENINGS ────────────────────────────────────────────────────────────────────────────────────────────────
void StatusEvents::push(std::uint64_t t, std::string what)
{
    events_.push_front({t, std::move(what)});
    while (events_.size() > kMaxEvents) events_.pop_back();
}

void StatusEvents::observe(const ControllerStatusSnapshot &cur, std::uint64_t now_ms)
{
    if (not cur.valid) return;
    if (not prev_.has_value()) { prev_ = cur; return; }
    const auto &p = *prev_;
    if (cur.episode_active and (not p.episode_active or cur.episode_why != p.episode_why))
    {
        push(now_ms, "handed a stretch to the policy: " + handoff_words(cur.episode_why));
        if (cur.episode_why == "stall") stall_ms_ = now_ms;
    }
    else if (not cur.episode_active and p.episode_active)
        push(now_ms, "the policy's stretch ended; the chain drives again");
    if (cur.chain_wait and not p.chain_wait)
        push(now_ms, std::format("stopped to WAIT: {} not on the map block the way", plural(cur.marks, "obstacle", "obstacles")));
    else if (not cur.chain_wait and p.chain_wait)
        push(now_ms, "the wait is over: something moved, re-planning");
    if (cur.marks > p.marks)
    {
        mark_added_ms_ = now_ms;
        push(now_ms, std::format("marked {} not on the map; re-planning around", plural(cur.marks - p.marks, "obstacle", "obstacles")));
    }
    else if (cur.marks < p.marks)
    {
        mark_cleared_ms_ = now_ms;
        push(now_ms, std::format("{} not on the map seen gone", plural(p.marks - cur.marks, "obstacle", "obstacles")));
    }
    if (cur.fail_marks > p.fail_marks)
        push(now_ms, "the policy could not pass a stretch: closed it, re-planning");
    if (cur.target_name != p.target_name and not cur.target_name.empty())
        push(now_ms, "new goal: " + cur.target_name);
    else if (cur.plan_gen != p.plan_gen and cur.route_valid)
        push(now_ms, std::format("new route planned ({:.1f} m)", cur.route_len));
    prev_ = cur;
}

// ── THE COLUMN ───────────────────────────────────────────────────────────────────────────────────────────────────────
std::string compose(const Inputs &in, const StatusEvents &events, std::uint64_t now_ms)
{
    const auto &c = in.ctl;
    std::vector<std::string> L;
    const auto section = [&L](std::string_view title) { if (not L.empty()) L.emplace_back(); L.emplace_back(title); };

    // ── headline: moving or not, and why, since when
    if (const auto &v = in.verdict; v.valid)
    {
        L.push_back(std::format("{} — {} ({})", v.driving ? "▶ MOVING" : "■ STOPPED", cause_words(v.cause),
                                rc::why::format_duration(v.duration_ms)));
        if (not v.detail.empty()) L.push_back(v.detail);
    }

    // ── who drives
    section("WHO DRIVES");
    const bool policy_driving = c.policy_enabled and (not c.hybrid_on or c.episode_active or not c.hybrid_place);
    if (not in.mission.driving)
        L.push_back(std::format("Nobody: driving is {} (press Run or click a target).", in.mission.paused ? "PAUSED" : "off"));
    else if (not c.armed)
        L.emplace_back("Nobody: the base output is DISARMED; commands are not sent.");
    else if (c.relative_motion_active)
        L.push_back("A move/turn order" + (c.relative_motion_progress.empty() ? std::string(".") : ": " + c.relative_motion_progress));
    else if (c.escape_active or in.stuck_active)
        L.emplace_back("Escape manoeuvre: the robot got stuck and is backing out.");
    else if (c.backoff_active)
        L.emplace_back("Backing away from obstacles that hem the robot in.");
    else if (c.lockon_active)
        L.emplace_back("Lock-on: small corrective moves to line up with the target.");
    else if (c.chain_wait)
        L.push_back(std::format("WAITING: the way is blocked by {} not on the map. Watching for one to move.",
                                plural(c.marks, "obstacle", "obstacles")));
    else if (c.hybrid_on and c.episode_active)
    {
        L.push_back("The local policy, for a short episode, because " + handoff_words(c.episode_why) + ".");
        L.push_back(c.episode_final ? std::string("It drives to the goal itself.")
                                    : std::format("It drives to ({:.1f}, {:.1f}) and hands back past {:.1f} m along the route.",
                                                  c.episode_sub_x, c.episode_sub_y, c.episode_s_end));
    }
    else if (c.hybrid_on and c.hybrid_place)
        L.emplace_back("The spline chain (route → spline → tracker); the policy stands by.");
    else if (c.policy_enabled)
        L.emplace_back("The local policy.");
    else
        L.push_back(std::string("The spline chain (route → spline → tracker)")
                    + (c.marks_drive ? "; obstacles not on the map are marked for the planner." : "."));
    if (not c.handoff_requested.empty() and not c.episode_active)
        L.push_back("Handing over to the policy next cycle: " + handoff_words(c.handoff_requested) + ".");
    if (not in.hold_reason.empty() and not c.chain_wait)
        L.push_back("Holding: " + in.hold_reason);
    if (policy_driving and in.mission.driving)
    {
        if (c.policy_mode >= 0) L.push_back("Policy decision: " + c.policy_mode_name + ".");
        else if (not c.policy_hold_why.empty()) L.push_back("Policy holds: " + c.policy_hold_why);
    }

    // ── the goal
    section("GOAL");
    {
        std::string what;
        const auto &a = in.affordance;
        const auto &dp = in.door;
        const bool passage = not dp.door.empty() and dp.step != "idle" and dp.step != "done";
        if (c.relative_motion_active) what = "Move order";
        else if (passage) what = std::format("Cross door {} (step: {}{})", dp.door, dp.step,
                                             dp.step_state.empty() ? std::string{} : ", " + dp.step_state);
        else if (c.lockon_active) what = "Line up with " + (c.target_name.empty() ? a.affordance : c.target_name);
        else if (c.orienting) what = "Turn to face " + (c.target_name.empty() ? std::string("the goal") : c.target_name);
        else if (a.active)
        {
            const std::string_view kind = a.policy == "Orient" ? "Turn towards" : a.policy == "Servo" ? "Approach and look at"
                                                                                                   : "Go to";
            what = std::format("{} {}", kind, a.affordance);
            if (not a.object.empty()) what += " (" + a.object + ")";
        }
        else if (c.approach_order_active) what = "Approach order: " + c.target_name;
        else if (c.manual_target) what = "Clicked place";
        else if (in.mission.running) what = "Mission route";
        else if (c.has_target) what = c.target_name.empty() ? std::string("A place") : c.target_name;
        if (what.empty()) L.emplace_back("No goal.");
        else
        {
            L.push_back(what);
            std::string dist;
            if (in.goal_dist_m.has_value()) dist = std::format("{:.2f} m to go", *in.goal_dist_m);
            else if (c.target_straight_m >= 0.f) dist = std::format("{:.2f} m away (straight line)", c.target_straight_m);
            if (in.goal_aligning) dist += dist.empty() ? "aligning" : ", aligning";
            if (in.goal_yaw_err_rad.has_value()) dist += std::format("{}facing {:.0f}° off", dist.empty() ? "" : ", ", std::abs(deg(*in.goal_yaw_err_rad)));
            if (not dist.empty()) L.push_back(dist);
            if (c.route_valid)
            {
                if (c.route_s_known and c.route_len > 0.f)
                {
                    const float left = std::max(0.f, c.route_len - c.route_s);
                    std::string r = std::format("Route: {:.1f} of {:.1f} m ({:.0f} %)", c.route_s, c.route_len,
                                                100.f * std::clamp(c.route_s / c.route_len, 0.f, 1.f));
                    if (c.odo_valid and c.odo_v > 0.05f) r += std::format(", about {:.0f} s left", left / c.odo_v);
                    L.push_back(std::move(r));
                }
                else L.push_back(std::format("Route: {:.1f} m{}", c.route_len, c.route_on_mission ? " (mission)" : ""));
            }
            else if (c.no_route_for_ms > 0.f)
                L.push_back(std::format("No route found for {:.1f} s.", c.no_route_for_ms * 1e-3f));
        }
    }

    // ── motion
    section("MOTION");
    L.push_back("Asked: " + (c.cmd_stop ? std::string("stop") : speed_words(c.cmd_adv, c.cmd_rot)));
    if (c.odo_valid) L.push_back("Measured: " + speed_words(c.odo_v, c.odo_w));
    else L.emplace_back("Measured: no odometry.");
    if (std::isfinite(c.route_heading_err) and std::isfinite(c.route_cross))
        L.push_back(std::format("{:.0f}° off the route's direction, {:.0f} cm to its {}", std::abs(deg(c.route_heading_err)),
                                std::abs(c.route_cross) * 100.f, c.route_cross >= 0.f ? "left" : "right"));

    // ── safety / monitors
    section("SAFETY");
    if (not c.chain_monitor_on) L.emplace_back("Route monitor: off.");
    else if (c.episode_active) L.push_back("Ahead: a problem; the policy has the stretch (" + handoff_words(c.episode_why) + ").");
    else if (not c.handoff_requested.empty()) L.push_back("Ahead: a problem; handing over to the policy.");
    else if (c.chain_wait) L.emplace_back("Ahead: blocked by obstacles not on the map; waiting.");
    else if (c.marks_off) L.emplace_back("Ahead: no way around the marked obstacles; trying the map alone.");
    else if (events.last_mark_added_ms() != 0 and now_ms - events.last_mark_added_ms() < 3000)
        L.emplace_back("Ahead: re-planning around an obstacle not on the map.");
    else if (c.clear_check) L.emplace_back("Ahead: an obstacle left; checking for a shorter way.");
    else if (c.chain_verdict == 1) L.push_back(std::format("Ahead: something tight {:.1f} m along the route — still beyond the tracker's reach.", c.chain_ahead_m));
    else if (c.chain_verdict == 2) L.push_back(std::format("Ahead: an obstacle not on the map {:.1f} m along the route.", c.chain_ahead_m));
    else if (c.chain_verdict == 3) L.push_back(std::format("Ahead: no way through {:.1f} m along the route.", c.chain_ahead_m));
    else if (c.chain_verdict == 4) L.push_back(std::format("Ahead: tight but passable {:.1f} m along the route — the policy's.", c.chain_ahead_m));
    else if (c.route_valid) L.emplace_back("Ahead: nothing that needs action.");
    else L.emplace_back("Ahead: no route to watch.");
    if (c.sigma_track >= 0.f) L.push_back(std::format("Tracking error: about ±{:.0f} cm", c.sigma_track * 100.f));
    if (c.marks > 0 or c.fail_marks > 0 or events.last_mark_cleared_ms() != 0)
    {
        std::string m = c.marks == 0 ? std::string("No obstacles marked off the map")
                                     : plural(c.marks, "obstacle", "obstacles") + " not on the map";
        if (c.fail_marks > 0) m += std::format("; {} where the policy could not pass", plural(c.fail_marks, "spot", "spots"));
        if (events.last_mark_cleared_ms() != 0) m += "; last one seen gone " + ago(now_ms, events.last_mark_cleared_ms());
        L.push_back(m + ".");
    }
    if (c.chain_monitor_on)
        L.push_back(c.chain_stall ? std::string("Progress: the route is not advancing as the commands predict (stall).")
                  : events.last_stall_ms() != 0 ? "Progress: a stall was flagged " + ago(now_ms, events.last_stall_ms()) + "."
                                                : std::string("Progress: no stall flagged."));

    // ── localisation
    section("WHERE");
    if (not c.have_pose) L.emplace_back("No pose in the room.");
    else
    {
        L.push_back("In " + (c.room.empty() ? std::string("an unnamed room") : c.room));
        if (c.pose_xy_std > 0.f)
            L.push_back(std::format("Pose ±{:.0f} cm, ±{:.1f}°, {}", c.pose_xy_std * 100.f, deg(c.pose_theta_std), age_text(c.pose_age_ms)));
        else
            L.push_back("Pose: no uncertainty published, " + age_text(c.pose_age_ms));
    }

    // ── health
    section("HEALTH");
    if (in.control_hz > 0.f)
        L.push_back(std::format("Control {:.1f} Hz (slowest cycle {:.0f} ms)", in.control_hz, in.control_worst_ms));
    if (in.lidar_stalled) L.push_back(std::format("LiDAR STALLED for {:.1f} s", in.lidar_stall_s));
    else
    {
        std::string l = in.lidar_hz >= 0.f ? std::format("LiDAR {:.1f} Hz", in.lidar_hz) : std::string("LiDAR");
        l += c.lidar_age_ms >= 0.f ? ", last sweep " + age_text(c.lidar_age_ms) : std::string(", no sweep yet");
        L.push_back(std::move(l));
    }
    if (c.odo_age_ms < 0.f) L.emplace_back("Odometry: none yet");
    else
        L.push_back(std::format("Odometry {} ({})", c.odo_fresh_tau_ms > 0.f and c.odo_age_ms > c.odo_fresh_tau_ms ? "STALE" : "fresh",
                                age_text(c.odo_age_ms)));

    // ── task: mission, affordance, door passage
    section("TASK");
    {
        const auto mode = rc::from_index(in.mission.mode_index);
        L.push_back(std::format("Mode {}, {}", rc::to_string(mode), in.mission.driving ? "running" : in.mission.paused ? "paused" : "stopped"));
        if ((in.mission.running or in.mission.recording) and not in.mission.status.empty()) L.push_back("Mission: " + in.mission.status);
        if (not c.mission_step.empty()) L.push_back("Mission step: " + c.mission_step);
        const auto &a = in.affordance;
        if (a.active)
        {
            std::string s = "Affordance " + a.affordance;
            if (not a.phase.empty()) s += ", " + a.phase;
            s += a.timeout_s > 0.f ? std::format(", {:.0f} of {:.0f} s", a.elapsed_s, a.timeout_s) : std::format(", {:.0f} s", a.elapsed_s);
            L.push_back(std::move(s));
        }
        else if (a.dwell_left_s > 0.f)
            L.push_back(std::format("Looking at {} for {:.1f} s more", a.affordance, a.dwell_left_s));
        if (not a.recent.empty()) L.push_back("Last outcome: " + a.recent.front());
        if (const auto &dp = in.door; not dp.door.empty() and dp.step != "idle")
        {
            std::string s = "Door " + dp.door + ": " + (dp.step == "done" ? std::string("passed") : dp.step);
            if (dp.attempts > 0) s += std::format(" ({} failed)", plural(dp.attempts, "attempt", "attempts"));
            L.push_back(std::move(s));
            if (not dp.cross.empty()) L.push_back("Crossing: " + dp.cross);
            else if (not c.cross_status.empty()) L.push_back("Crossing: " + c.cross_status);
        }
    }

    // ── recent happenings
    if (not events.events().empty())
    {
        section("RECENTLY");
        for (const auto &e : events.events()) L.push_back(ago(now_ms, e.t_ms) + ": " + e.what);
    }

    std::string out;
    for (const auto &l : L)
    {
        if (not out.empty()) out += '\n';
        out += l;
    }
    return out;
}

}   // namespace rc::status_text
