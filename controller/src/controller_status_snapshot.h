#pragma once
/*
 * controller_status_snapshot.h — WHAT THE CONTROLLER IS DOING, AS PLAIN VALUES, for the status column of the window.
 *
 * Filled on the CONTROL thread once per cycle (ControllerSession::status_snapshot, controller_status_snapshot.cpp, plus a
 * few worker-level fields SpecificWorker adds), copied into the display's staged snapshot under its mutex, and turned into
 * words on the GUI thread at the panel's own refresh rate (controller_status_text.h). Read-only: nothing here drives, and
 * every value is a copy of state that already exists — no new estimate is made for the panel.
 *
 * Plain std types only, so the struct can travel across the thread boundary by value and the header stays light enough
 * for controller_session.h to include.
 */

#include <cstdint>
#include <limits>
#include <string>

namespace rc
{

struct ControllerStatusSnapshot
{
    bool valid = false;
    std::uint64_t now_ms = 0;                 ///< the worker's clock when this was taken (current_time_ms)

    // ── WHO DRIVES ──────────────────────────────────────────────────────────────────────────────────────────────────
    bool policy_enabled = false;              ///< [Policy] Enabled
    bool hybrid_on = false;                   ///< [Policy] HybridChain + chain monitor + policy: a PLACE is the chain's
    bool hybrid_place = false;                ///< the last drive_with_policy cycle reached its PLACE branch (the chain plans it)
    bool chain_monitor_on = false;            ///< [Chain] Monitor enabled
    bool marks_drive = false;                 ///< the chain drives with the policy off and its marks act on the planner
    bool episode_active = false;              ///< the policy drives a handed-over stretch
    bool episode_final = false;               ///< ... whose goal is the target itself
    std::string episode_why;                  ///< window / stall / blocked / finish / wedge / spin / extend
    float episode_sub_x = 0.f, episode_sub_y = 0.f, episode_s_end = 0.f;
    std::string handoff_requested;            ///< a hand-over asked for this cycle, served next cycle ("" = none)
    bool chain_wait = false;                  ///< WAIT: blocked by what the map does not have, watching the marks
    bool marks_off = false;                   ///< no route with the marks: planning on the map alone
    bool clear_check = false;                 ///< a mark near the route cleared: looking for a shorter route
    int   chain_verdict = -1;                 ///< the window's last verdict (0 clear .. 4 hand-off), -1 = none recent
    float chain_ahead_m = 0.f;                ///< how far ahead along the route the failing stretch starts
    bool  chain_stall = false;                ///< the progress monitor's last judgement failed
    int marks = 0, fail_marks = 0;            ///< obstacles not on the map; failure marks (where the policy could not pass)
    bool escape_active = false;               ///< the physical-stuck escape owns the base
    bool backoff_active = false;              ///< backing off live returns that hem the body in
    bool lockon_active = false;               ///< the lock-on micro-search owns the base
    bool approach_order_active = false;       ///< an "approach <object>" order is running
    int policy_mode = -1;                     ///< rc::policy::Decision::Mode as int; -1 = none / a hold, -2 = rate hold
    std::string policy_mode_name;             ///< "" when policy_mode < 0
    std::string policy_hold_why;              ///< the policy's last hold reason (only meaningful when policy_mode < 0)

    // ── THE GOAL ────────────────────────────────────────────────────────────────────────────────────────────────────
    bool has_target = false;
    std::string target_name;
    bool target_from_affordance = false;
    bool manual_target = false;               ///< a clicked / ordered place
    bool orienting = false;                   ///< an orient step (turn in place) is running
    float target_straight_m = -1.f;           ///< straight-line distance robot→target (-1 = unknown)

    bool route_valid = false;                 ///< a route (mission route or point plan) is being driven
    bool route_on_mission = false;
    bool route_s_known = false;
    float route_s = 0.f, route_len = 0.f;     ///< progress along the route / its length, metres
    float route_heading_err = std::numeric_limits<float>::quiet_NaN();   ///< robot yaw − route tangent, rad
    float route_cross = std::numeric_limits<float>::quiet_NaN();         ///< signed distance beside the route, m (left +)
    std::uint64_t plan_gen = 0;               ///< bumps on every new point plan (a re-plan)

    // ── MONITORS ────────────────────────────────────────────────────────────────────────────────────────────────────
    float sigma_track = -1.f;                 ///< the chain's tracking-error posterior, m (-1 = monitor off)
    float no_route_for_ms = 0.f;              ///< how long the planner has found no route (0 = routing fine)

    // ── LOCALISATION ────────────────────────────────────────────────────────────────────────────────────────────────
    bool have_pose = false;
    std::string room;
    float pose_xy_std = 0.f, pose_theta_std = 0.f;   ///< room_concept's published pose std (0 = none published)
    float pose_age_ms = -1.f;                 ///< age of the pose at its own stamp (-1 = unknown)

    // ── ODOMETRY (robot_current_*) ──────────────────────────────────────────────────────────────────────────────────
    bool odo_valid = false;
    float odo_v = 0.f, odo_w = 0.f;           ///< measured advance m/s, turn rad/s (CCW +)
    float odo_age_ms = -1.f;                  ///< age of the sample (wall clock), -1 = unknown
    float odo_fresh_tau_ms = 0.f;             ///< Controller.CommandFreshnessTauMs: older than this = stale

    std::string cross_status;                 ///< the door cross's own words (ControllerSession::cross_status_)

    // ── WORKER-LEVEL (filled by SpecificWorker::update_custom_widget) ───────────────────────────────────────────────
    bool armed = true;                        ///< base output armed
    bool cmd_stop = true;                     ///< the command in force is a stop
    float cmd_adv = 0.f, cmd_rot = 0.f;       ///< the command in force, m/s and rad/s
    std::string cmd_site;                     ///< which code line sent it
    float lidar_age_ms = -1.f;                ///< since the last accepted LiDAR sweep (-1 = never)
    bool relative_motion_active = false;      ///< a "move <m>" / "turn <deg>" order owns the base
    std::string relative_motion_progress;
    std::string mission_step;                 ///< the running mission interpreter step ("" = none)
};

}   // namespace rc
