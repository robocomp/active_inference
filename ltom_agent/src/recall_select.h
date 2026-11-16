// Which remembered objects ltom should offer as recall candidates in the current room (pure, no DSR, no Qt).
#pragma once
#include "repertory.h"

namespace ltom
{
struct Recall
{
    std::uint64_t uid = 0; std::string cls; Keyframe kf; std::optional<float> exist_prob;   // exist_prob: the entry's
    std::string color_name; float color_prob = 0.f;                                          // the remembered colour
    std::uint64_t same_as_uid = 0; float same_as_prob = 0.f;   // a linked memory: the live uid it continues as (1 = confirmed)
};

// An entry is recalled when its class is listed, its last life is not alive (death, owner_exit and gap all mean
// "not on the graph now"), its uid is not on the graph (`present`), and its MOST RECENT keyframe is in `room_key`
// (that keyframe is the recalled pose). Ordered by uid.
// Changed object (2026-11-15): a memory marked GONE from that room since its last keyframe there is not offered; a
// memory that continues as a live object is offered only while that object is present, carrying its uid, so the
// concept agent keeps the link and never resurrects it at a birth.
std::vector<Recall> select_recalls(const Repertory& r, std::int64_t room_key,
                                   const std::set<std::uint64_t>& present, std::span<const std::string> classes);
}
