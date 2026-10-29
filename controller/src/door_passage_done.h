#pragma once
// The DONE decision of a door passage (protocol fact 3, spec 2026-10-29 C1.6/C2.10): the passage is through when the robot's
// resolved room id CHANGED (room_concept switched the robot's RT parent), never merely because the robot reached the point
// past the door. A same-id rename is not a change: the ids are compared, nothing else.
#include <string>

namespace rc
{
/// `cross_room` = the room id when the Cross step began (empty = unknown); `room_frame` = the id resolved now.
[[nodiscard]] inline bool passage_room_switched(const std::string& cross_room, const std::string& room_frame)
{
    return not cross_room.empty() and room_frame != cross_room;
}
/// Reaching P_post decides nothing by itself: the passage is done only on the room switch; otherwise it holds, waiting.
[[nodiscard]] inline bool passage_done(bool reached_p_post, const std::string& cross_room, const std::string& room_frame)
{
    (void)reached_p_post;
    return passage_room_switched(cross_room, room_frame);
}
}  // namespace rc
