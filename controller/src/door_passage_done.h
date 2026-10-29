#pragma once
// The DONE decision of a door passage (protocol fact 3, spec 2026-10-29 C1.6/C2.10): the passage is through when the robot's
// resolved room ID changed (room_concept switched the robot's RT parent), never merely because the robot reached the point
// past the door. Ids are compared, never names: a same-id rename is not a change. id 0 = unknown.
#include <cstdint>

namespace rc
{
/// `captured` = the room id the Cross step started in (0 = unknown then; filled in by the FIRST known id seen before P_post);
/// `now` = the id resolved this cycle (track_room / graph_state().room_id). True = the room switch happened.
/// If no id was known before P_post, the first known id after it counts as the switch (it happened while unknown).
[[nodiscard]] inline bool passage_room_switched(std::uint64_t& captured, std::uint64_t now, bool post_reached)
{
    if (now == 0) return false;
    if (captured == 0)
    {
        if (post_reached) return true;
        captured = now;
        return false;
    }
    return now != captured;
}
}  // namespace rc
