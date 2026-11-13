#pragma once
// passage_refs.h -- a mission's passage refs into REMEMBERED rooms, rewritten as room refs before the card (chat_e2e11).
//
// The controller resolves `cross passage:ID@ROOM` ONLY through ltsm's LIVE passage node, which exists only while the
// room beyond is proto (controller specificworker_mission.cpp resolve_door); after an eviction it is gone and the step
// is refused `passage_unresolved` ("go to my chair" from room_2, chat_e2e11). `cross <room>` resolves through the
// `recall_<room>` child ltsm posts under the door that leads to that remembered room (mission_door_relation.h), and
// holds until it appears. The model keeps WRITING passages: it copies them 10/10 in the planner eval, while given
// `cross room_3` it wrote `cross door_1` again (4/10). So the translation is done here, in code, on the checked IR.
// Limit: two passages joining the same two rooms are not told apart (the controller takes the first door by name).
#include "mission_plan.h"
#include "topology.h"

#include <algorithm>
#include <variant>
#include <vector>

namespace rc::chat
{
/// Rewrites, in place and recursively, every Passage ref whose passage is in `t` and whose other end (seen from the
/// ref's room) is a remembered room into a Node ref naming that room. Returns how many were rewritten.
[[nodiscard]] inline int passages_to_rooms(std::vector<rc::mission::Step>& steps, const rc::topo::Topology& t)
{
    using rc::mission::Ref;
    int n = 0;
    for (auto& s : steps)
    {
        for (auto& [key, v] : s.args)
            if (Ref* r = std::get_if<Ref>(&v); r and r->kind == Ref::Kind::Passage)
            {
                const auto p = std::ranges::find(t.passages, r->name, &rc::topo::TopoPassageInfo::id);
                if (p == t.passages.end()) continue;
                const std::string other = topo_other_room(*p, r->from);
                if (other.empty() or std::ranges::none_of(t.rooms, [&](const auto& room) { return room.name == other; }))
                    continue;
                *r = Ref{.kind = Ref::Kind::Node, .name = other};
                ++n;
            }
        n += passages_to_rooms(s.steps, t);
    }
    return n;
}
}  // namespace rc::chat
