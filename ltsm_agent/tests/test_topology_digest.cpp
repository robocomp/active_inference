// g++ -std=c++23 -O0 -g -Wall -Wextra ltsm_agent/tests/test_topology_digest.cpp -o /tmp/test_topology_digest && /tmp/test_topology_digest
//
// The ltsm_topology summary's encoder (src/topology_digest.h), pinned without a graph: field names, nesting, optional
// live_door, locale-proof numbers. The decode side is checked in language_agent/tests/test_topology.cpp (round trip).
#include "../src/topology_digest.h"

#include <cassert>
#include <print>

using namespace rc::ltsm;

int main()
{
    const std::vector<TopoRoom> rooms{{"room_1", true}, {"room_2", false}};
    TopoPassage p;
    p.id = "p0"; p.room_a = "room_1"; p.face_a = "east"; p.room_b = "room_2"; p.face_b = "west";
    p.crossings = 3; p.pass_alpha = 4.5; p.pass_beta = 1; p.live_door = "door_1";
    const auto j = encode_topology(rooms, {p});
    const std::string want =
        R"({"rooms":[{"name":"room_1","live":true},{"name":"room_2","live":false}],)"
        R"("passages":[{"id":"p0","a":{"room":"room_1","face":"east"},"b":{"room":"room_2","face":"west"},)"
        R"("crossings":3,"p_pass":{"alpha":4.5,"beta":1},"p_open":{"alpha":1,"beta":1},"live_door":"door_1"}]})";
    assert(j == want);

    p.live_door.reset();
    assert(encode_topology(rooms, {p}).find("live_door") == std::string::npos);

    TopoPassage q; q.id = "a\"b";                       // quotes are escaped
    assert(encode_topology({}, {q}).find(R"("id":"a\"b")") != std::string::npos);
    assert(encode_topology({}, {}) == R"({"rooms":[],"passages":[]})");
    assert(live_name("r12_room_1") == "room_1" and live_name("r1_door_3") == "door_3");
    assert(live_name("room_1") == "room_1" and live_name("r_x") == "r_x" and live_name("rx1_a") == "rx1_a");
    assert(live_name("r") == "r" and live_name("") == "");
    std::println("test_topology_digest: ok");
}
