// passage_room_switched (src/door_passage_done.h): a passage completes on the ROOM-ID switch, never on reaching P_post alone.
//   g++ -std=c++23 -O0 -g -o /tmp/tpd test_passage_done.cpp && /tmp/tpd
#include "../src/door_passage_done.h"
#include <print>
int main()
{
    int f = 0;
    auto chk = [&](bool ok, const char* w) { if (not ok) { ++f; std::println("FAIL {}", w); } };
    std::uint64_t c = 1;
    chk(not rc::passage_room_switched(c, 1, true), "P_post + same id => not done");
    chk(rc::passage_room_switched(c, 3, false), "id changed => done");
    c = 1;
    chk(not rc::passage_room_switched(c, 1, false), "same id (a rename keeps the id) => not done");
    c = 0;
    chk(not rc::passage_room_switched(c, 0, false) and c == 0, "unknown stays unknown");
    chk(not rc::passage_room_switched(c, 5, false) and c == 5, "first known id before P_post is captured");
    chk(rc::passage_room_switched(c, 6, false), "then a different id => done");
    c = 0;
    chk(not rc::passage_room_switched(c, 0, true), "unknown at P_post => holds");
    chk(rc::passage_room_switched(c, 4, true), "none known before P_post, first known after => done");
    std::println("test_passage_done: {}", f == 0 ? "all pass" : "FAILURES");
    return f;
}
