#pragma once
// RoomHold — "no room resolved yet" is NOT YET, not "wrong room".
//
// A mission step scoped to a room (explore R, go_to a pose in R, cross/open a passage from R) needs the robot's current
// room. Right after start-up the room node may not exist yet (room_concept still has to birth it): the controller then
// reads room_name == "" . Failing `wrong_room` on that killed the mission before the room was born. Only a RESOLVED,
// DIFFERENT room is wrong_room; an empty one makes the step wait, with its own budget (kHoldBudgetMs) so it cannot hang.
#include <cstdint>
#include <optional>
#include <string>

#include "../../common/mission/mission_ir.h"

namespace rc
{
class RoomHold
{
public:
    /// How long a step may wait for the room to resolve. A budget (like the interpreter's budget_s), not a gate: when it
    /// runs out the step fails "room_unresolved", the honest word for "never learnt where I am".
    static constexpr std::uint64_t kHoldBudgetMs = 60'000;

    enum class Verdict { Waiting, Ready, TimedOut };

    /// Does this step need the current room to start?
    static bool scoped(const mission::Step &s)
    {
        if (s.op == "explore" or s.op == "go_to") return true;
        if (s.op == "cross" or s.op == "open")
        {
            const auto d = s.ref("door");
            return d.has_value() and d->kind == mission::Ref::Kind::Passage;
        }
        return false;
    }

    void hold(const mission::Step &s, std::uint64_t now_ms) { pending_ = s; since_ = now_ms; }
    bool holding() const { return pending_.has_value(); }
    const mission::Step &step() const { return *pending_; }
    void clear() { pending_.reset(); }

    /// Ready (and the hold is released) once `here` is resolved; TimedOut when the budget ran out first.
    Verdict check(const std::string &here, std::uint64_t now_ms)
    {
        if (not here.empty()) { pending_.reset(); return Verdict::Ready; }
        if (now_ms - since_ >= kHoldBudgetMs) { pending_.reset(); return Verdict::TimedOut; }
        return Verdict::Waiting;
    }

private:
    std::optional<mission::Step> pending_;
    std::uint64_t since_ = 0;
};
}   // namespace rc
