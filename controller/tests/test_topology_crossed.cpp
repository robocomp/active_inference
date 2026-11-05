// test_topology_crossed — rc::topo::crossed_doors (common/ltsm_topology/topology_decode.h): the doors the LTSM remembers
// as crossed, so the exploration drive's "crossed" survives a restart (spec 2026-11-05, "Calls made").
// Build + run (from controller/):
//   g++ -std=c++23 -O1 -fPIC -I../common $(pkg-config --cflags Qt6Core) tests/test_topology_crossed.cpp -o /tmp/ttc \
//       $(pkg-config --libs Qt6Core) && /tmp/ttc
#include <print>

#include "ltsm_topology/topology_decode.h"

int main()
{
    int fails = 0, checks = 0;
    const auto check = [&](bool ok, const char* what) { ++checks; if (not ok) { ++fails; std::println("FAIL: {}", what); } };
    check(rc::topo::crossed_doors("").empty(), "topology_missing_is_empty (empty string)");
    check(rc::topo::crossed_doors("{").empty(), "topology_missing_is_empty (malformed)");
    const char* json = R"({"rooms":[{"name":"room_1","live":true},{"name":"room_2","live":false}],
      "passages":[{"id":"passage_1","a":{"room":"room_1","face":"r0_door_3"},"b":{"room":"room_2","face":"r1_door_9"},
                   "crossings":2,"p_pass":{"alpha":3,"beta":1},"p_open":{"alpha":1,"beta":1},"live_door":"door_9"},
                  {"id":"passage_2","a":{"room":"room_1","face":"door_4"},"b":{"room":"room_3","face":"door_5"},
                   "crossings":0,"p_pass":{"alpha":1,"beta":1},"p_open":{"alpha":1,"beta":1}}]})";
    const auto c = rc::topo::crossed_doors(json);
    check(c.contains("door_3") and c.contains("door_9"), "crossed_faces_and_live_door (memory prefix stripped)");
    check(not c.contains("door_4") and not c.contains("door_5"), "a passage never crossed contributes nothing");
    check(c.size() == 2, "exactly the crossed passage's names");
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
