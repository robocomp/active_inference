// Topology pure side (src/topology.h): decode of ltsm's ltsm_topology summary and the cheapest route over it.
#include "../src/topology.h"
#include "../../ltsm_agent/src/topology_digest.h"

#include <cassert>
#include <print>

using namespace rc::chat;
using rc::ltsm::TopoPassage;
using rc::ltsm::TopoRoom;

namespace
{
TopoPassage passage(std::string id, std::string a, std::string b, double pa = 1, double pb = 1)
{
    TopoPassage p;
    p.id = std::move(id); p.room_a = std::move(a); p.face_a = "n"; p.room_b = std::move(b); p.face_b = "s";
    p.pass_alpha = pa; p.pass_beta = pb;
    return p;
}
TopoPassageInfo info(std::string id, std::string a, std::string b, double pa = 1, double pb = 1)
{
    TopoPassageInfo p;
    p.id = std::move(id); p.room_a = std::move(a); p.room_b = std::move(b); p.pass_alpha = pa; p.pass_beta = pb;
    return p;
}
}   // namespace

int main()
{
    // encode -> decode, field for field
    {
        auto p = passage("p0", "room_1", "room_2", 4.5, 1.25);
        p.crossings = 3; p.open_alpha = 2; p.open_beta = 0.5; p.live_door = "door_1";
        const auto t = decode_topology(rc::ltsm::encode_topology({{"room_1", true}, {"room_2", false}}, {p}));
        assert(t.has_value());
        assert(t->rooms.size() == 2 and t->rooms[0].name == "room_1" and t->rooms[0].live and not t->rooms[1].live);
        assert(t->passages.size() == 1);
        const auto& q = t->passages[0];
        assert(q.id == "p0" and q.room_a == "room_1" and q.face_a == "n" and q.room_b == "room_2" and q.face_b == "s");
        assert(q.crossings == 3 and q.pass_alpha == 4.5 and q.pass_beta == 1.25);
        assert(q.open_alpha == 2 and q.open_beta == 0.5);
        assert(q.live_door == "door_1");
        const auto t2 = decode_topology(rc::ltsm::encode_topology({}, {passage("p1", "a", "b")}));
        assert(t2.has_value() and not t2->passages[0].live_door.has_value());
    }
    assert(not decode_topology("not json").has_value());
    assert(not decode_topology("[1]").has_value());

    // chain room_1 - room_2 - room_3, plus an island
    {
        Topology t;
        t.passages = {info("p0", "room_1", "room_2"), info("p1", "room_2", "room_3")};
        t.rooms = {{"room_1", true}, {"room_2", false}, {"room_3", false}, {"room_9", false}};
        assert((route(t, "room_1", "room_3") == std::vector<std::string>{"p0", "p1"}));
        assert((route(t, "room_3", "room_1") == std::vector<std::string>{"p1", "p0"}));   // undirected
        assert(route(t, "room_1", "room_9").empty());
        assert(route(t, "room_1", "room_1").empty());
    }
    // parallel passages: the reliable one wins, whichever order they are listed in
    {
        Topology t;
        t.passages = {info("bad", "a", "b", 1, 9), info("good", "a", "b", 9, 1)};
        assert((route(t, "a", "b") == std::vector<std::string>{"good"}));
        std::ranges::reverse(t.passages);
        assert((route(t, "a", "b") == std::vector<std::string>{"good"}));
    }
    // cost is 1/E[p_pass]: two reliable hops beat one unreliable direct passage
    {
        Topology t;
        t.passages = {info("direct", "a", "c", 1, 9), info("h1", "a", "b", 9, 1), info("h2", "b", "c", 9, 1)};
        assert((route(t, "a", "c") == std::vector<std::string>{"h1", "h2"}));
    }
    // tool answers
    {
        Topology t;
        t.rooms = {{"room_1", true}, {"room_2", false}, {"room_3", false}, {"room_9", false}};
        t.passages = {info("p0", "room_1", "room_2"), info("p1", "room_2", "room_3")};
        t.passages[0].live_door = "door_1";
        t.passages[0].crossings = 3;
        assert(answer_route_to(t, "room_1", "room_1") == "You are already in room_1.");
        assert(answer_route_to(t, "room_1", "room_9").starts_with("No remembered route"));
        assert(answer_route_to(t, "room_1", "room_7").starts_with("No remembered room"));
        const auto r = answer_route_to(t, "room_1", "room_3");
        assert(r.contains("cross passage:p0@room_1 expect=room_2 (live door there: door_1)"));
        assert(r.contains("cross passage:p1@room_2 expect=room_3"));
        assert(answer_known_rooms(t, "room_1").contains("room_1 (you are here)"));
        assert(answer_known_rooms(Topology{}, "room_1").starts_with("No rooms"));
        // the planner's STATE line: every reachable remembered room, its legs in order; the first leg by its live door
        assert(ways_line(t, "room_1") == "ways from here (each leg in order): room_2 = cross door_1 expect=room_2; "
                                         "room_3 = cross door_1 expect=room_2, cross passage:p1@room_2 expect=room_3");
        assert(ways_line(t, "room_3") == "ways from here (each leg in order): room_1 = cross passage:p1@room_3 "
                                         "expect=room_2, cross passage:p0@room_2 expect=room_1; room_2 = cross "
                                         "passage:p1@room_3 expect=room_2");   // no live door at the first leg
        // names a mission may use from memory: remembered rooms (recall re-creates them by name) and passage ids
        assert(remembered_name(t, "room_3") and remembered_name(t, "p1"));
        assert(not remembered_name(t, "door_1"));    // a live door seen from memory is not a name of the memory
        assert(not remembered_name(t, "room_7") and not remembered_name(Topology{}, "room_1"));
        assert(ways_line(t, "room_9").empty());           // nothing reachable from an island
        assert(ways_line(Topology{}, "room_1").empty());
        const auto line = remembered_rooms_line(t, "room_1");
        assert(line.starts_with("rooms in long-term memory (call route_to for the way): room_2 (behind door_1, crossed 3x), room_3 (2 passages away"));
        assert(line.contains("room_9 (no known route)"));
        assert(remembered_rooms_line(Topology{}, "room_1").empty());
    }
    // the planner's STATE (final review I6): no "(call route_to …)" — the planner has no tools — and the ways line
    // appended. ★ Pinned to the EXACT text tests/check_llm_missions.py's remembered-room cases send (STATE_REMEMBERED):
    // the eval's STATE is this function's output for the tools eval's topology (room_2 behind door_1, room_3 beyond).
    {
        Topology t;
        t.rooms = {{"room_1", true}, {"room_2", false}, {"room_3", false}};
        t.passages = {info("passage_0", "room_1", "room_2"), info("passage_1", "room_2", "room_3")};
        t.passages[0].live_door = "door_1";
        t.passages[0].crossings = 3;
        const std::string line = remembered_rooms_line(t, "room_1");
        assert(line == "rooms in long-term memory (call route_to for the way): room_2 (behind door_1, crossed 3x), "
                       "room_3 (2 passages away, first behind door_1, crossed 3x)");   // = check_llm_tools.py STATE
        const std::string chat = "STATE:\ncurrent room: room_1\n" + line + "\nlast move/turn sent from this chat: none\n";
        const std::string planner = planner_state_text(chat, ways_line(t, "room_1"));
        assert(planner == "STATE:\ncurrent room: room_1\n"
                          "rooms in long-term memory: room_2 (behind door_1, crossed 3x), room_3 (2 passages away, first "
                          "behind door_1, crossed 3x)\nlast move/turn sent from this chat: none\n"
                          "ways from here (each leg in order): room_2 = cross door_1 expect=room_2; room_3 = cross door_1 "
                          "expect=room_2, cross passage:passage_1@room_2 expect=room_3\n");
        assert(not planner.contains("route_to"));
        assert(planner_state_text("STATE:\nx\n", "") == "STATE:\nx\n");   // nothing remembered: unchanged
    }
    std::println("test_topology: ok");
}
