// policy_tick_gate.h — when does a policy tick RESUME after the policy did not drive? (Task 8 review fixes I1 / fix 2.1)
// Pure (std only). The session notes every cycle in which the policy owned the base but did not drive it — its own early
// holds (no map, odometry stale, the odometry-sign Blind), a move order's no-pose / Pause holds, Halt, a hold in the
// planner stage, the LiDAR-stall hold — with on_hold(); on_tick(now) then says whether this tick must call
// Navigator::interrupt() (drop the committed plan and its pending prediction unjudged, restart the command history): after
// any hold, on the very first tick, or after a gap in ticks longer than kMaxTickGapS (the longest interval one control
// tick can legitimately span; the odometry step clamp). An interruption is not evidence about the world.
#pragma once

#include <cstdint>
#include <string>

namespace rc::policy
{
class PolicyTickGate
{
public:
    static constexpr double kMaxTickGapS = 0.5;
    void on_hold() { held_ = true; }
    bool on_tick(double now_s)
    {
        const bool resume = held_ or last_ < 0.0 or now_s - last_ > kMaxTickGapS;
        held_ = false;
        last_ = now_s;
        return resume;
    }
    double last_tick_s() const { return last_; }
    void reset() { held_ = false; last_ = -1.0; }
private:
    bool held_ = false;
    double last_ = -1.0;
};
// ★ Live fix 2 (tourP3, 2026-10-20) — A ROOM-FRAME CHANGE. room_concept moves the robot's RT into the next room's frame
//   once it is past the plane (live: pose (0.60, 2.57) -> (-3.64, 0.41) in one cycle). Everything the policy holds is in
//   the OLD frame: the goal and its field (the field read 365-728 m at the new pose: at()'s clamp-to-grid + nearest-finite
//   fallback evaluated off its own frame), the odometry anchor, and the contact buffer, whose sweeps were registered in the
//   old frame and, read against the new pose, fell INSIDE the body ("9 returns inside the body" -> Stuck -> hand back).
//   on_room() says the frame changed (the caller resets the policy: goal, field, plan, evidence); ready(last_scan_ms) is
//   false until a scan accepted AFTER the change — the first one registered in the new frame. Clocks: steady ms, the same
//   the obstacle tracker stamps accepted scans with.
class RoomFrameGuard
{
public:
    bool on_room(const std::string& room, std::uint64_t now_ms)
    {
        if (room.empty() or room == room_) return false;
        const bool changed = not room_.empty();
        room_ = room;
        if (changed) switched_ms_ = now_ms;
        return changed;
    }
    bool ready(std::uint64_t last_scan_accept_ms) const { return switched_ms_ == 0 or last_scan_accept_ms > switched_ms_; }
    const std::string& room() const { return room_; }
private:
    std::string room_;
    std::uint64_t switched_ms_ = 0;
};
}  // namespace rc::policy
