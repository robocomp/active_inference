// policy_tick_gate.h — when does a policy tick RESUME after the policy did not drive? (Task 8 review fixes I1 / fix 2.1)
// Pure (std only). The session notes every cycle in which the policy owned the base but did not drive it — its own early
// holds (no map, odometry stale, the odometry-sign Blind), a move order's no-pose / Pause holds, Halt, a hold in the
// planner stage, the LiDAR-stall hold — with on_hold(); on_tick(now) then says whether this tick must call
// Navigator::interrupt() (drop the committed plan and its pending prediction unjudged, restart the command history): after
// any hold, on the very first tick, or after a gap in ticks longer than kMaxTickGapS (the longest interval one control
// tick can legitimately span; the odometry step clamp). An interruption is not evidence about the world.
#pragma once

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
}  // namespace rc::policy
