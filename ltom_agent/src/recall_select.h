// Which remembered objects ltom should offer as recall candidates in the current room (pure, no DSR, no Qt).
#pragma once
#include "repertory.h"

namespace ltom
{
struct Recall { std::uint64_t uid = 0; std::string cls; Keyframe kf; std::optional<float> exist_prob; };   // exist_prob: the entry's

// An entry is recalled when its class is listed, its last life is not alive (death, owner_exit and gap all mean
// "not on the graph now"), its uid is not on the graph (`present`), and its MOST RECENT keyframe is in `room_key`
// (that keyframe is the recalled pose). Ordered by uid.
std::vector<Recall> select_recalls(const Repertory& r, std::int64_t room_key,
                                   const std::set<std::uint64_t>& present, std::span<const std::string> classes);
}
