// exit_fact.h: the crossing chain on a real DSR graph (private DDS domain). Links 1-3, the guard of link 4, and every edge
// dying with its source node.
//   g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 -DSHADOW_JSON="\"$PWD/../../../robot_concept/shadow.json\"" \
//       $(pkg-config --cflags Qt6Core) -o /tmp/tef test_exit_fact.cpp -ldsr_api -ldsr_core -lfastcdr -lfastdds \
//       $(pkg-config --libs Qt6Core) && /tmp/tef
#include "../exit_fact.h"

#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)
}  // namespace

int main()
{
    using namespace rc::exit_fact;
    auto G = std::make_shared<DSR::DSRGraph>("exit_fact_test", 990, std::string(SHADOW_JSON), true, int8_t(90), DSR::Queue);
    const auto add = [&](DSR::Node n) { return G->insert_node(n).value_or(0); };
    const auto door = [&](const char* name)   // door_concept's face: an `object` with object_subtype "door"
    {
        auto n = DSR::Node::create<object_node_type>(name);
        G->add_or_modify_attrib_local<object_subtype_att>(n, std::string("door"));
        return add(n);
    };
    const auto robot = G->get_nodes_by_type("robot").front().id();
    const auto room_1 = add(DSR::Node::create<room_node_type>("room_1"));
    const auto room_2 = add(DSR::Node::create<room_node_type>("room_2"));
    const auto d13 = door("door_13");   // room_1's face of DOOR_1
    const auto d20 = door("door_20");   // another door of room_1
    const auto d49 = door("door_49");   // room_2's face of DOOR_1 (link 3 makes it)

    CHECK(not crossed_through(*G, {d13, d20}).has_value());
    CHECK(not is_crossed_face(*G, d13));

    // link 1, then a second crossing replaces it (one crossing at a time)
    CHECK(write_crossed(*G, robot, d20, 0));
    CHECK(write_crossed(*G, robot, d13, 7));
    const auto c = crossed_through(*G, {d13, d20});
    CHECK(c.has_value() and c->face == d13 and c->predicted_room_id == 7);
    CHECK(not G->get_edge(robot, d20, "exit").has_value());
    CHECK(is_crossed_face(*G, d13) and not is_crossed_face(*G, d20));
    CHECK(not crossed_through(*G, {d49}).has_value());   // a fact about another room's faces is not about this one

    // link 2
    CHECK(not entered_through(*G, room_2).has_value());
    CHECK(write_entered(*G, d13, room_2));
    CHECK(entered_through(*G, room_2) == d13);
    CHECK(not entered_through(*G, room_1).has_value());
    CHECK(not twin_written(*G, room_2, {d49}));          // link 4 must wait

    // link 3
    CHECK(write_twin(*G, d49, d13));
    CHECK(twin_of(*G, d49) == d13 and not twin_of(*G, d13).has_value());
    CHECK(is_crossed_face(*G, d49));                     // the entry face is never `cross new`
    CHECK(twin_written(*G, room_2, {d49}));              // link 4 may archive room_1

    // link 4: ltsm archives room_1 (room -> room `exit`), deleting nothing; it is not mistaken for links 2/3
    CHECK(not is_archived(*G, room_1));
    CHECK(write_archived(*G, room_1, room_2));
    CHECK(is_archived(*G, room_1) and not is_archived(*G, room_2));
    CHECK(archived_rooms(*G) == std::vector<std::uint64_t>{room_1});
    CHECK(entered_through(*G, room_2) == d13);           // a room->room edge is not a door->room one
    CHECK(twin_of(*G, d49) == d13);
    CHECK(room_of(*G, d13) == 0);                        // (no RT chain built in this test)
    // link 5: door_concept deletes OLD_FACE (links 1 and 2 die with it), room_concept the room (link 4 dies with it)
    CHECK(G->delete_node(d13));
    CHECK(not crossed_through(*G, {d13}).has_value());
    CHECK(not entered_through(*G, room_2).has_value());
    CHECK(not twin_of(*G, d49).has_value());
    CHECK(G->delete_node(room_1));
    CHECK(archived_rooms(*G).empty());

    std::println("test_exit_fact: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
