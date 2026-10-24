// RoomHold (src/mission_room_hold.h): an unresolved current room is NOT YET, not wrong_room.
//   g++ -std=c++23 -O0 -g -o /tmp/tmrh test_mission_room_hold.cpp && /tmp/tmrh
#include "../src/mission_room_hold.h"

#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(x) check((x), __LINE__, #x)
using rc::RoomHold;
using namespace rc::mission;

Step explore(const std::string &room)
{
    Step s{.id = "1", .op = "explore"};
    s.args["room"] = Ref{.kind = Ref::Kind::Node, .name = room};
    return s;
}
}   // namespace

int main()
{
    // which steps are room-scoped
    CHECK(RoomHold::scoped(explore("room_1")));
    CHECK(RoomHold::scoped(Step{.op = "go_to"}));
    Step cross{.op = "cross"};
    cross.args["door"] = Ref{.kind = Ref::Kind::Passage, .name = "passage_0", .from = "room_1"};
    CHECK(RoomHold::scoped(cross));
    cross.args["door"] = Ref{.kind = Ref::Kind::Node, .name = "door_3"};
    CHECK(not RoomHold::scoped(cross));
    CHECK(not RoomHold::scoped(Step{.op = "wait"}));
    CHECK(not RoomHold::scoped(Step{.op = "move"}));

    // empty room: waits (does not fail), then is released the moment the room resolves
    RoomHold h;
    h.hold(explore("room_1"), 1000);
    CHECK(h.holding());
    CHECK(h.check("", 1000) == RoomHold::Verdict::Waiting);
    CHECK(h.check("", 30'000) == RoomHold::Verdict::Waiting);
    CHECK(h.holding());
    CHECK(h.check("room_1", 31'000) == RoomHold::Verdict::Ready);
    CHECK(not h.holding());

    // a resolved room, whichever it is, releases the hold: the caller then compares (only a DIFFERENT room is wrong_room)
    h.hold(explore("room_1"), 0);
    CHECK(h.check("room_2", 5) == RoomHold::Verdict::Ready);

    // the room never resolves: the hold has its own budget and ends in TimedOut
    h.hold(explore("room_1"), 5000);
    CHECK(h.check("", 5000 + RoomHold::kHoldBudgetMs - 1) == RoomHold::Verdict::Waiting);
    CHECK(h.check("", 5000 + RoomHold::kHoldBudgetMs) == RoomHold::Verdict::TimedOut);
    CHECK(not h.holding());

    // cancel drops the hold
    h.hold(explore("room_1"), 0);
    h.clear();
    CHECK(not h.holding());

    std::println("{} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
