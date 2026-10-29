// DoorRelation (src/mission_door_relation.h): cross <room> / cross new pick a door by relation; the hold is lazy.
//   g++ -std=c++23 -O0 -g -o /tmp/tmdr test_mission_door_relation.cpp && /tmp/tmdr
#include "../src/mission_door_relation.h"
#include "../src/mission_room_hold.h"

#include <print>

namespace
{
int failures = 0;
void check(bool ok, int line, const char *what) { if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
using namespace rc;
mission::Ref node(const std::string &n) { return {.kind = mission::Ref::Kind::Node, .name = n}; }
DoorFacts door(std::string n, std::set<std::string> recalls = {}, bool match = false, bool looked = false, double d = 5.0)
{ return {.name = std::move(n), .recall_rooms = std::move(recalls), .has_match = match, .looked = looked, .dist = d}; }
}   // namespace

int main()
{
    CHECK(relation_of(node("new")) == DoorRelationKind::New);
    CHECK(relation_of(node("room_2")) == DoorRelationKind::Room);
    CHECK(relation_of(node("door_3")) == DoorRelationKind::None);

    // by recall child
    std::vector<DoorFacts> ds{door("door_1", {"room_2"}), door("door_4", {"room_3"}), door("door_9")};
    CHECK(pick_door(node("room_2"), ds) == "door_1");
    CHECK(pick_door(node("room_3"), ds) == "door_4");
    CHECK(not pick_door(node("room_7"), ds).has_value());

    // new: excludes recall and passage-match doors
    CHECK(pick_door(node("new"), ds) == "door_9");
    ds.push_back(door("door_11", {}, /*match*/ true));
    CHECK(pick_door(node("new"), ds) == "door_9");
    ds = {door("door_1", {"room_2"}), door("door_11", {}, true)};
    CHECK(not pick_door(node("new"), ds).has_value());

    // new among several: looked first, else nearest, else name
    ds = {door("door_5", {}, false, false, 1.0), door("door_6", {}, false, false, 3.0), door("door_7", {}, false, true, 9.0)};
    CHECK(pick_door(node("new"), ds) == "door_7");
    ds[2].looked = false;
    CHECK(pick_door(node("new"), ds) == "door_5");
    ds[1].dist = 1.0;
    CHECK(pick_door(node("new"), ds) == "door_5");

    // room-scoped: a relation waits for the room
    mission::Step cross{.id = "1", .op = "cross"};
    cross.args["door"] = node("new");
    CHECK(RoomHold::scoped(cross));
    cross.args["door"] = node("door_3");
    CHECK(not RoomHold::scoped(cross));

    // lazy hold, then resolve
    cross.args["door"] = node("room_2");
    RelationHold h;
    h.hold(cross, 1000);
    CHECK(h.holding());
    CHECK(h.check(std::nullopt, 1500) == RelationHold::Verdict::Waiting and h.holding());
    ds = {door("door_1")};
    CHECK(not pick_door(node("room_2"), ds).has_value());   // not yet
    ds[0].recall_rooms.insert("room_2");                    // ltsm posts the recall
    CHECK(h.check(pick_door(node("room_2"), ds), 2000) == RelationHold::Verdict::Ready and not h.holding());
    // budget
    h.hold(cross, 0);
    CHECK(h.check(std::nullopt, RelationHold::kHoldBudgetMs - 1) == RelationHold::Verdict::Waiting);
    CHECK(h.check(std::nullopt, RelationHold::kHoldBudgetMs) == RelationHold::Verdict::TimedOut and not h.holding());
    // the step's own budget_s wins when smaller
    cross.args["budget_s"] = 10.0;
    h.hold(cross, 0);
    CHECK(h.check(std::nullopt, 10'000) == RelationHold::Verdict::TimedOut);

    std::println("{}", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
