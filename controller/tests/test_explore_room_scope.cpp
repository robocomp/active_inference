// Mission `explore room_X` confines the affordance contest to room_X (local-policy plan Task 4, user 2026-10-21).
// Part A pins the pure rule (common/affordance_manager/affordance_room_scope.h); part B runs the REAL selector
// (AffordanceManager::select_target) on a real DSR graph with two rooms, the way the controller calls it.
// Build+run (a real DSR graph on a private DDS domain, like test_door_passage):
//   g++ -std=c++23 -O0 -g -fPIC -I/usr/include/eigen3 -I../../common -DSHADOW_JSON="\"$PWD/../../robot_concept/shadow.json\"" \
//       $(pkg-config --cflags Qt6Core) -o /tmp/ters test_explore_room_scope.cpp ../../common/affordance_manager/affordance_manager.cpp \
//       -ldsr_api -ldsr_core -lfastcdr -lfastdds $(pkg-config --libs Qt6Core) && /tmp/ters
#include "../../common/affordance_manager/affordance_room_scope.h"
#include "../../common/affordance_manager/affordance_manager.h"

#include <dsr/api/dsr_api.h>

#include <map>
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

using namespace rc::affordance;

// ── A: the pure rule ─────────────────────────────────────────────────────────────────────────────────────────────
void pure_rule()
{
    // who scopes the contest: only an explore order with a room, and never over a door passage's required entrant
    CHECK(contest_room_scope(true, "", "room_1") == "room_1");
    CHECK(contest_room_scope(false, "", "room_1").empty());                       // no mission: unchanged
    CHECK(contest_room_scope(true, "aff_door_1_cross", "room_1").empty());       // a cross step: unchanged
    CHECK(contest_room_scope(false, "aff_door_1_cross", "").empty());

    // the parent chain: affordance → door → wall → room
    std::map<std::uint64_t, ChainNode> tree{
        {1, {"room", "room_1", std::nullopt}},
        {2, {"wall", "wall_1", 1}},
        {3, {"door", "door_5", 2}},
        {4, {"room", "room_2", std::nullopt}},
        {5, {"table", "table_9", 4}},
        {7, {"object", "loop_a", 8}},   // a malformed cycle
        {8, {"object", "loop_b", 7}},
    };
    const auto lookup = [&](std::uint64_t id) -> std::optional<ChainNode>
    {
        if (const auto it = tree.find(id); it != tree.end()) return it->second;
        return std::nullopt;
    };
    CHECK(room_of_chain(1, lookup) == std::optional<std::string>("room_1"));   // parented to the room itself
    CHECK(room_of_chain(3, lookup) == std::optional<std::string>("room_1"));   // door → wall → room
    CHECK(room_of_chain(5, lookup) == std::optional<std::string>("room_2"));
    CHECK(not room_of_chain(std::nullopt, lookup).has_value());                 // no parent
    CHECK(not room_of_chain(99, lookup).has_value());                           // dangling parent
    CHECK(not room_of_chain(7, lookup).has_value());                            // a cycle terminates, unresolved

    // the partition
    struct C { std::string name; std::optional<std::string> room; };
    const auto name_of = [](const C &c) { return c.name; };
    const auto room_of = [](const C &c) { return c.room; };
    const std::vector<C> all{{"afford_room", "room_1"}, {"aff_door_1", "room_1"}, {"aff_door_look_door_5", "room_1"},
                             {"aff_table_9", "room_2"}, {"aff_orphan", std::nullopt}, {"aff_chair_2", "room_1"}};
    {
        auto v = all;
        const auto ex = scope_to_room(v, "room_1", name_of, room_of);
        CHECK(v.size() == 4);
        CHECK(ex.size() == 2);
        CHECK(ex.size() == 2 and ex[0].name == "aff_table_9" and ex[0].room == std::optional<std::string>("room_2"));
        CHECK(ex.size() == 2 and ex[1].name == "aff_orphan" and not ex[1].room.has_value());
        CHECK(describe_excluded(ex) == " 'aff_table_9'(room_2) 'aff_orphan'(unresolved)");
    }
    {
        auto v = all;
        const auto ex = scope_to_room(v, "", name_of, room_of);   // no order: nothing filtered
        CHECK(v.size() == all.size() and ex.empty());
    }
}

// ── B: the real selector, two rooms ──────────────────────────────────────────────────────────────────────────────
struct World
{
    std::shared_ptr<DSR::DSRGraph> G;
    std::uint64_t room_1 = 0, room_2 = 0, door_5 = 0, table_9 = 0;
    // producers own their nodes for the test's lifetime
    std::vector<std::unique_ptr<rc::AffordanceManager>> producers;
};

std::uint64_t insert(DSR::DSRGraph &G, DSR::Node n, std::optional<std::uint64_t> parent)
{
    if (parent.has_value()) G.add_or_modify_attrib_local<parent_att>(n, *parent);
    G.add_or_modify_attrib_local<level_att>(n, 1);
    return G.insert_node(n).value_or(0);
}

World make_world(int k)
{
    World w;
    w.G = std::make_shared<DSR::DSRGraph>("explore_scope_test_" + std::to_string(k), 980 + k, std::string(SHADOW_JSON),
                                          true, static_cast<int8_t>(80 + k), DSR::Queue);
    auto &G = *w.G;
    w.room_1 = insert(G, DSR::Node::create<room_node_type>("room_1"), std::nullopt);
    w.room_2 = insert(G, DSR::Node::create<room_node_type>("room_2"), std::nullopt);
    const auto wall = insert(G, DSR::Node::create<wall_node_type>("wall_1"), w.room_1);
    w.door_5 = insert(G, DSR::Node::create<door_node_type>("door_5"), wall);
    w.table_9 = insert(G, DSR::Node::create<table_node_type>("table_9"), w.room_2);
    const auto offer = [&](const std::string &name, std::uint64_t parent, float x, float gain)
    {
        auto p = std::make_unique<rc::AffordanceManager>(name);
        CHECK(p->publish_target(w.G, parent, x, 0.f, 0.f, gain));
        w.producers.push_back(std::move(p));
    };
    offer("afford_room", w.room_1, 1.f, 1.0f);
    offer("aff_door_5", w.door_5, 2.f, 1.5f);
    offer("aff_door_look_door_5", w.room_1, 3.f, 0.5f);
    offer("aff_table_9", w.table_9, 4.f, 9.0f);   // the biggest gain on offer — and in room_2
    return w;
}

void real_selector()
{
    {   // explore room_1: the room_2 table is never selected, however big its gain; the best of room_1 wins
        auto w = make_world(0);
        rc::AffordanceManager controller;
        controller.set_room_scope(contest_room_scope(true, "", "room_1"));
        const auto t = controller.select_target(w.G);
        CHECK(t.has_value() and t->node_name == "aff_door_5");
        bool table_seen = false;
        for (const auto &c : controller.last_candidates()) table_seen = table_seen or c.node_name == "aff_table_9";
        CHECK(not table_seen);                                                  // not even a contestant
        CHECK(controller.last_room_excluded().size() == 1);
        CHECK(not controller.last_room_excluded().empty() and controller.last_room_excluded()[0].name == "aff_table_9");
        // completes, re-selects: still never the table — the rest of room_1 in turn
        controller.mark_reached(w.G, Outcome::Satisfied);
        const auto t2 = controller.select_target(w.G);
        CHECK(t2.has_value() and t2->node_name != "aff_table_9");
        CHECK(t2.has_value() and t2->node_name == "afford_room");
    }
    {   // explore room_2: only the table
        auto w = make_world(1);
        rc::AffordanceManager controller;
        controller.set_room_scope(contest_room_scope(true, "", "room_2"));
        const auto t = controller.select_target(w.G);
        CHECK(t.has_value() and t->node_name == "aff_table_9");
        CHECK(controller.last_room_excluded().size() == 3);
    }
    {   // no mission: the open contest, unchanged — the highest gain wins wherever it is
        auto w = make_world(2);
        rc::AffordanceManager controller;
        controller.set_room_scope(contest_room_scope(false, "", "room_1"));
        const auto t = controller.select_target(w.G);
        CHECK(t.has_value() and t->node_name == "aff_table_9");
        CHECK(controller.last_room_excluded().empty());
    }
    {   // an unresolvable target (no parent anywhere) is not in room_1
        auto w = make_world(3);
        DSR::Node orphan = DSR::Node::create<affordance_node_type>("aff_orphan");
        w.G->add_or_modify_attrib_local<epistemic_target_x_m_att>(orphan, 5.f);
        w.G->add_or_modify_attrib_local<epistemic_target_y_m_att>(orphan, 0.f);
        w.G->add_or_modify_attrib_local<epistemic_target_yaw_rad_att>(orphan, 0.f);
        w.G->add_or_modify_attrib_local<epistemic_gain_att>(orphan, 20.f);
        w.G->add_or_modify_attrib_local<epistemic_pending_att>(orphan, true);
        w.G->add_or_modify_attrib_local<active_att>(orphan, false);
        w.G->add_or_modify_attrib_local<level_att>(orphan, 1);
        CHECK(w.G->insert_node(orphan).has_value());
        rc::AffordanceManager controller;
        controller.set_room_scope("room_1");
        const auto t = controller.select_target(w.G);
        CHECK(t.has_value() and t->node_name == "aff_door_5");
        bool orphan_out = false;
        for (const auto &e : controller.last_room_excluded()) orphan_out = orphan_out or (e.name == "aff_orphan" and not e.room);
        CHECK(orphan_out);
    }
    {   // a cross step: the required entrant scopes the contest; the room scope stays off
        auto w = make_world(4);
        rc::AffordanceManager controller;
        controller.set_required_name("aff_table_9");
        controller.set_room_scope(contest_room_scope(true, controller.required_name(), "room_1"));
        const auto t = controller.select_target(w.G);
        CHECK(t.has_value() and t->node_name == "aff_table_9");
    }
}
}   // namespace

int main()
{
    pure_rule();
    real_selector();
    std::println("test_explore_room_scope: {}/{} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
