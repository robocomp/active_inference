#pragma once
// ApproachHold — a hand-back of the door's approach is NOT a step failure.
//
// gold13 leg 2: the approach standpoint sat against a bed; the controller handed it back (Unreachable) and the mission
// poll failed the cross step on the FIRST attempt ("approach_unreachable"), although door_concept re-sites the standpoint
// around the remembered hand-back (door_approach_pose.h, lateral +0.81 within seconds). The passage keeps the step on
// `approach` and waits for the re-offer ("waiting for it to be offered again"); the poll must wait too.
// The wait has its own budget (like RoomHold::kHoldBudgetMs): it ends in failure when the budget runs out or when the
// producer withdraws the approach altogether (no node to re-offer).
#include <cstdint>

namespace rc
{
class ApproachHold
{
public:
    /// ⚑ FLAGGED CONSTANT: how long a handed-back approach may wait for its re-offer when the mission step carries no
    /// budget_s of its own. Equal to door_concept's approach_refusal_memory_s (60 s): the horizon over which the producer
    /// is still re-solving around that hand-back. No model-level term exists for "how long to wait for a peer".
    static constexpr std::uint64_t kHoldBudgetMs = 60'000;

    enum class Verdict { NotHeld, Waiting, Failed };

    /// attempts > 0 on the approach step = a hand-back happened. `present` = the producer still has the approach node; `executing` = it has been re-claimed.
    Verdict check(bool step_is_approach, int attempts, bool present, bool executing, std::uint64_t now_ms)
    {
        if (not step_is_approach or attempts <= 0) { since_ = 0; return Verdict::NotHeld; }
        if (not present) { since_ = 0; return Verdict::Failed; }          // withdrawn by the producer
        if (executing) { since_ = 0; return Verdict::Waiting; }           // the re-offer is being driven: progress, budget restarts
        if (since_ == 0) since_ = now_ms;
        if (now_ms - since_ >= kHoldBudgetMs) { since_ = 0; return Verdict::Failed; }
        return Verdict::Waiting;
    }
    void reset() { since_ = 0; }

private:
    std::uint64_t since_ = 0;
};
}   // namespace rc
