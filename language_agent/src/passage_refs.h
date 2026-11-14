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
#include <map>
#include <optional>
#include <string>
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
namespace passage_refs_detail
{
inline bool is_door_ref(const rc::mission::Step& s, std::string& name)
{
    const auto it = s.args.find("door");
    if (it == s.args.end()) return false;
    const auto* r = std::get_if<rc::mission::Ref>(&it->second);
    if (not r or r->kind != rc::mission::Ref::Kind::Node or not r->name.starts_with("door")) return false;
    name = r->name;
    return true;
}
}  // namespace passage_refs_detail

/// A door ref that is NOT live in the current room cannot resolve when its step starts: door nodes are re-made on every
/// entry, so a name from an earlier visit (or another room) is a stale graph slot (chat_e2e15: `cross door_24
/// expect=room_1` from room_2, "door 'door_24' is not in the graph"). With an expect=ROOM on its `cross`, the step means
/// "the door that leads to ROOM": rewritten to `cross ROOM`, which the controller resolves (and waits for) through
/// ltsm's recall_ROOM under the door. An `open` of the same door in the same list follows its cross. A door with no
/// expect is left as written. Recursive; returns how many refs were rewritten.
///   ★A door step AFTER any earlier `cross` is stale too, even if its name is live now: the robot is in another room by
///   then (chat_e2e19: `cross door_1 bind=a; cross door_1 expect=room_1` — door_1 was live at the card, gone in room_2).
namespace passage_refs_detail
{
inline int doors_to_rooms(std::vector<rc::mission::Step>& steps, const std::vector<std::string>& live_doors,
                          const std::map<std::string, std::string>& leads_to, bool& crossed)
{
    using rc::mission::Ref;
    int n = 0;
    for (auto& s : steps)
    {
        if (std::string d; (s.op == "cross" or s.op == "open") and is_door_ref(s, d)
                           and (crossed or std::ranges::find(live_doors, d) == live_doors.end()))
        {
            std::optional<std::string> room;
            if (const auto e = s.string("expect"); s.op == "cross" and e.has_value() and e->starts_with("room")) room = *e;
            else if (const auto it = leads_to.find(d); it != leads_to.end()) room = it->second;
            if (room.has_value())
            {
                s.args["door"] = Ref{.kind = Ref::Kind::Node, .name = *room};
                ++n;
            }
        }
        n += doors_to_rooms(s.steps, live_doors, leads_to, crossed);
        if (s.op == "cross") crossed = true;   // every later door name belongs to a room not entered yet
    }
    return n;
}
inline void collect_leads_to(const std::vector<rc::mission::Step>& steps, std::map<std::string, std::string>& leads_to)
{
    for (const auto& s : steps)
    {
        if (std::string d; s.op == "cross" and is_door_ref(s, d))
            if (const auto e = s.string("expect"); e.has_value() and e->starts_with("room")) leads_to.emplace(d, *e);
        collect_leads_to(s.steps, leads_to);
    }
}
}  // namespace passage_refs_detail

[[nodiscard]] inline int doors_to_rooms(std::vector<rc::mission::Step>& steps, const std::vector<std::string>& live_doors)
{
    std::map<std::string, std::string> leads_to;   // door name -> the room its cross expects (this list)
    passage_refs_detail::collect_leads_to(steps, leads_to);
    bool crossed = false;
    return passage_refs_detail::doors_to_rooms(steps, live_doors, leads_to, crossed);
}
}  // namespace rc::chat
