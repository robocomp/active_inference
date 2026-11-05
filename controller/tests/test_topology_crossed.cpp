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
    // final review C1: a re-entered room's remembered doors
    const char* j2 = R"({"rooms":[],"passages":[
      {"id":"passage_1","a":{"room":"room_1","face":"r1_door_1"},"b":{"room":"room_2","face":"r2_door_2"},"crossings":1,
       "p_pass":{"alpha":1,"beta":1},"p_open":{"alpha":1,"beta":1},"live_door":"door_8"},
      {"id":"passage_3","a":{"room":"room_1","face":"r1_door_9"},"b":{"room":"room_4","face":"r4_door_10"},"crossings":0,
       "p_pass":{"alpha":1,"beta":1},"p_open":{"alpha":1,"beta":1}}]})";
    check(rc::topo::remembered_uncrossed(j2, "room_1") == 1, "remembered_uncrossed counts the uncrossed passage not in view");
    check(rc::topo::remembered_uncrossed(j2, "room_2") == 0, "a crossed passage does not count");
    check(rc::topo::remembered_uncrossed("", "room_1") == 0, "no summary: nothing remembered");
    check(rc::topo::live_door_for(j2, "door_1") == std::optional<std::string>("door_8"), "live_door_for resolves a stale face name");
    check(not rc::topo::live_door_for(j2, "door_9").has_value(), "no live door in view: nullopt");
    // explore_tour6: right after a crossing the passage's live_door still named the crossed face itself — never return it
    check(not rc::topo::live_door_for(j2, "door_8").has_value(), "live_door_for never returns the name it was asked about");
    std::println("{}/{} checks passed", checks - fails, checks);
    return fails == 0 ? 0 : 1;
}
