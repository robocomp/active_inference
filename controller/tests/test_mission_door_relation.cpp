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
    {   // a rename under the same id is not a room change
        std::uint64_t id = 7; std::string name = "room_4";
        CHECK(track_room(id, name, 7, "room_2") == RoomEvent::Renamed and name == "room_2");
        CHECK(track_room(id, name, 7, "room_2") == RoomEvent::None);
        CHECK(track_room(id, name, 9, "room_5") == RoomEvent::Changed and name == "room_5" and id == 9);
    }
    {   // a mission's room is the room BY ID: survives a same-id rename, ends on an id change
        std::uint64_t id = 7; std::string name = "room_4";
        RoomAliases al;
        CHECK(robot_in_room(id, name, al, 7, "room_4"));                       // explore room_4 starts: live node of that name
        track_room(id, name, 7, "room_2"); al["room_4"] = id;            // room_concept adopts room_2 in place
        CHECK(robot_in_room(id, name, al, std::nullopt, "room_4"));            // the step still holds: same id
        CHECK(robot_in_room(id, name, al, 7, "room_2"));
        CHECK(not robot_in_room(id, name, al, 9, "room_5"));                   // another live room
        track_room(id, name, 9, "room_5");                               // the robot really leaves
        CHECK(not robot_in_room(id, name, al, std::nullopt, "room_4"));        // id differs -> outside
        CHECK(robot_in_room(id, name, al, 9, "room_5"));
        CHECK(not robot_in_room(0, "", al, std::nullopt, "room_4"));           // unresolved is not "in"
        CHECK(room_id_named(al, 3, "room_4") == 3u);                           // a reused name is its live node's
    }
    {   // the explore poll decision
        CHECK(explore_room_verdict(7, 7, "room_4", "room_4") == ExploreRoom::Inside);
        CHECK(explore_room_verdict(7, 7, "room_4", "room_2") == ExploreRoom::Renamed);
        CHECK(explore_room_verdict(7, 9, "room_4", "room_5") == ExploreRoom::Outside);
        CHECK(explore_room_verdict(7, 0, "room_4", "") == ExploreRoom::Inside);
        // a rename keeps the claim: only the names swap (the session's scope-change path is not taken)
        std::string er = "room_4", scope = "room_4";
        CHECK(rename_scope(er, scope, "room_2") and er == "room_2" and scope == "room_2");
        std::string er2 = "room_4", none;   // a door passage stands: no scope, none invented
        CHECK(not rename_scope(er2, none, "room_2") and none.empty());
        // cross room_4 finds recall_room_2 (same id) and vice versa
        RoomAliases ids{{"room_4", 7}, {"room_2", 7}, {"room_9", 3}};
        std::vector<DoorFacts> ds{door("door_1", {"room_2"}), door("door_2", {"room_9"})};
        CHECK(not pick_door(node("room_4"), ds).has_value());
        expand_recalls(ds, ids, "room_4");
        CHECK(pick_door(node("room_4"), ds) == std::optional<std::string>("door_1"));
    }
    {   // poll decision + session scope step together, with an EXECUTING claim: a same-id rename must not release it
        bool claim = true, lockon = true, plan = true;
        auto release = [&] { claim = lockon = plan = false; };
        ScopeCycle sc{.explore_room = "room_4", .manager_scope = "", .pending_rename = ""};
        auto cycle = [&](bool cmd) { auto o = scope_cycle(sc, cmd, ""); if (o.release) release(); return o; };
        CHECK(cycle(true).release);                                    // entering the scope releases (by design)...
        claim = lockon = plan = true;                                  // ...then a drive starts and executes
        std::string room_name = "room_4";
        const std::uint64_t id = 7;
        // room_concept renames room_4 -> room_2 under the same id; the poll sees it
        CHECK(explore_room_verdict(id, id, room_name, "room_2") == ExploreRoom::Renamed);
        room_name = "room_2"; sc.pending_rename = room_name;           // rename_command_explore_room
        const auto o = cycle(true);
        CHECK(not o.release and claim and lockon and plan);            // survives
        CHECK(o.changed == false and sc.manager_scope == "room_2" and sc.explore_room == "room_2");
        CHECK(explore_room_verdict(id, id, room_name, "room_2") == ExploreRoom::Inside);
        // a genuinely different explore (set_command_explore clears the queued rename) still releases
        sc.pending_rename.clear(); sc.explore_room = "room_5";
        CHECK(cycle(true).release and not claim);
    }
    {   // what the session would set on the REAL manager (it feeds the real scope in and reconciles to outcome.manager_scope)
        std::string real;   // the manager's scope
        auto run = [&](std::string er, std::string pend, bool cmd, std::string_view req) {
            ScopeCycle sc{.explore_room = er, .manager_scope = real, .pending_rename = pend};
            const auto o = scope_cycle(sc, cmd, req);
            real = o.manager_scope;                                // session: set_room_scope(outcome.manager_scope)
            return o;
        };
        auto o = run("room_4", "", true, "");                      // enter
        CHECK(o.release and real == "room_4");
        o = run("room_4", "room_2", true, "");                     // same-id rename: names only
        CHECK(not o.release and not o.changed and real == "room_2");
        o = run("room_2", "", true, "door_1");                     // a passage overrides: clear, no release
        CHECK(not o.release and o.changed and real.empty());
    }
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
    // …and a door a crossing fact names (the face just come through, before ltsm marks it with a recall)
    {
        auto entry = door("door_49");
        entry.crossed = true;
        std::vector<DoorFacts> fresh{entry};
        CHECK(not pick_door(node("new"), fresh).has_value());
        fresh.push_back(door("door_50"));
        CHECK(pick_door(node("new"), fresh) == "door_50");
    }

    // new among several: looked first, else nearest, else name
    ds = {door("door_5", {}, false, false, 1.0), door("door_6", {}, false, false, 3.0), door("door_7", {}, false, true, 9.0)};
    CHECK(pick_door(node("new"), ds) == "door_7");
    ds[2].looked = false;
    CHECK(pick_door(node("new"), ds) == "door_5");
    ds[1].dist = 1.0;
    CHECK(pick_door(node("new"), ds) == "door_5");

    // the closest OPEN door first (user 2026-10-31): open beats shut-but-looked-and-near; among open, the nearest
    {
        auto shut_near = door("door_2", {}, false, true, 1.0);
        auto open_far = door("door_1", {}, false, false, 6.0);
        open_far.open = true;
        auto open_mid = door("door_7", {}, false, false, 3.0);
        open_mid.open = true;
        CHECK(pick_door(node("new"), std::vector<DoorFacts>{shut_near, open_far}) == "door_1");
        CHECK(pick_door(node("new"), std::vector<DoorFacts>{shut_near, open_far, open_mid}) == "door_7");
        CHECK(pick_door(node("new"), std::vector<DoorFacts>{shut_near}) == "door_2");   // none open: as before
    }

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
