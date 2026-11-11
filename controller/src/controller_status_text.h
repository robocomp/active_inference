#pragma once
/*
 * controller_status_text.h — the status column's words. GUI THREAD ONLY.
 *
 * ControllerDisplay::present() gathers everything already staged for the window (the motion verdict, the mission readout,
 * the door passage, the running affordance, the goal distance, the rates) plus the control thread's
 * rc::ControllerStatusSnapshot, and compose() turns it into a short block of plain sentences for a person watching the
 * robot: who drives, the goal, the motion, what the safety monitors say, where it thinks it is, and whether the inputs are
 * healthy. StatusEvents keeps a few recent happenings (a hand-over to the policy, a wait, an obstacle marked or seen gone,
 * a new route) by comparing consecutive snapshots, so a moment that lasted one second is still readable a minute later.
 *
 * No state of the controller is kept here and nothing here is ever read by the control thread.
 */

#include <cstdint>
#include <deque>
#include <optional>
#include <string>

#include "controller_status_snapshot.h"
#include "controller_status_line.h"        // rc::DoorPassageStatus
#include "controller_mission_panel.h"      // rc::MissionPanel::View
#include "controller_affordance_view.h"    // rc::AffordanceExecution
#include "motion_verdict.h"                // rc::why::View

namespace rc::status_text
{

struct Inputs
{
    ControllerStatusSnapshot ctl;
    rc::why::View verdict;
    rc::MissionPanel::View mission;
    rc::DoorPassageStatus door;
    rc::AffordanceExecution affordance;
    std::optional<float> goal_dist_m, goal_yaw_err_rad;
    bool goal_aligning = false;
    bool stuck_active = false;
    bool lidar_stalled = false;
    float lidar_stall_s = 0.f;
    std::string hold_reason;
    float control_hz = 0.f, control_worst_ms = 0.f;
    float lidar_hz = -1.f;
};

/// Recent happenings, detected as changes between consecutive snapshots (GUI thread). Newest first, a handful kept.
class StatusEvents
{
public:
    struct Event { std::uint64_t t_ms = 0; std::string what; };
    void observe(const ControllerStatusSnapshot &cur, std::uint64_t now_ms);
    [[nodiscard]] const std::deque<Event> &events() const { return events_; }
    /// when an obstacle not on the map was last marked / last seen gone (0 = never)
    [[nodiscard]] std::uint64_t last_mark_added_ms() const { return mark_added_ms_; }
    [[nodiscard]] std::uint64_t last_mark_cleared_ms() const { return mark_cleared_ms_; }
    [[nodiscard]] std::uint64_t last_stall_ms() const { return stall_ms_; }

private:
    void push(std::uint64_t t, std::string what);
    std::optional<ControllerStatusSnapshot> prev_;
    std::deque<Event> events_;
    std::uint64_t mark_added_ms_ = 0, mark_cleared_ms_ = 0, stall_ms_ = 0;
};

/// The whole column, one item per line. `now_ms` is the same clock StatusEvents was fed with.
[[nodiscard]] std::string compose(const Inputs &in, const StatusEvents &events, std::uint64_t now_ms);

}   // namespace rc::status_text
