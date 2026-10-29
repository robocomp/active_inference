// passage_done (src/door_passage_done.h): a passage completes on the ROOM SWITCH, never on reaching P_post alone.
//   g++ -std=c++23 -O0 -g -o /tmp/tpd test_passage_done.cpp && /tmp/tpd
#include "../src/door_passage_done.h"
#include <print>
int main()
{
    int f = 0;
    auto chk = [&](bool ok, const char* w) { if (not ok) { ++f; std::println("FAIL {}", w); } };
    chk(not rc::passage_done(true, "room_1", "room_1"), "reached P_post + same room => not done");
    chk(rc::passage_done(false, "room_1", "room_3"), "room id changed => done");
    chk(rc::passage_done(true, "room_1", "room_3"), "P_post + room changed => done");
    chk(not rc::passage_done(false, "room_1", "room_1"), "same id (a rename keeps the id) => not done");
    chk(not rc::passage_done(true, "", "room_1"), "unknown start room => not done");
    std::println("test_passage_done: {}", f == 0 ? "all pass" : "FAILURES");
    return f;
}
