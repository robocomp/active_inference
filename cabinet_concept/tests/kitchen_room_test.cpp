/*
 * kitchen_room_test — the kitchen model is released when the robot leaves its room (cabinet_kitchen_room.h).
 *
 * Replays the c3a_1 sequence against the agent's own rules with a fake graph (id → parent room):
 *   build in room_1 → publish runs → the robot crosses to room_2 → (room_2 outline not usable yet) → build in room_2.
 * Checks: on the crossing every run under room_1 is deleted and the model is reset; while unbound nothing is
 * published; no run is ever parented to a room the model was not built over (the c3a_1 re-hang).
 *
 *   cmake -S . -B build && make -C build kitchen_room_test && ./build/kitchen_room_test
 */
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>

#include "cabinet_kitchen_room.h"

namespace
{
int fails = 0;
void check(const char* what, bool ok)
{
    if (not ok) { std::printf("FAIL %s\n", what); ++fails; }
}

constexpr std::uint64_t room_1 = 11, room_2 = 22;

// A fake agent: the graph is id → parent; `built` stands for kitchen_mgr_.built(); publish mirrors
// publish_kitchen_boxes() (existing node: re-assign the RT from the parent the rule returns; new: insert under it).
struct FakeAgent
{
    std::map<std::uint64_t, std::uint64_t>          graph;   // node id → parent room
    std::unordered_map<std::string, std::uint64_t>  nodes;   // cell → node id
    std::uint64_t bound = 0, working = 0, next_id = 100;
    bool built = false;
    int resets = 0;

    void build_if_possible(bool outline_usable)
    {
        if (not built and outline_usable) { built = true; bound = working; }
    }
    void publish(std::initializer_list<const char*> cells)
    {
        const auto parent = rc::kitchen_room::publish_room(bound, working);
        if (not built or not parent) return;
        for (const char* c : cells)
        {
            if (auto it = nodes.find(c); it != nodes.end()) graph[it->second] = *parent;   // insert_or_assign_edge_RT
            else { nodes[c] = next_id; graph[next_id++] = *parent; }
        }
    }
    std::optional<int> release()
    {
        return rc::kitchen_room::release_if_left_behind(bound, working, nodes,
            [this](const std::string&, std::uint64_t id) { graph.erase(id); },
            [this] { built = false; ++resets; });
    }
    int under(std::uint64_t room) const
    {
        int n = 0;
        for (const auto& [id, p] : graph) n += (p == room);
        return n;
    }
};
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: match the agent's locale (nothing is parsed here)

    using rc::kitchen_room::publish_room;
    check("publish: not built -> nothing", not publish_room(0, room_1).has_value());
    check("publish: built here -> this room", publish_room(room_1, room_1) == std::optional<std::uint64_t>{room_1});
    check("publish: built for the room just left -> nothing", not publish_room(room_1, room_2).has_value());

    FakeAgent a;
    a.working = room_1;
    check("no room change -> no release", not a.release().has_value());
    a.build_if_possible(true);
    a.publish({"w13_base", "w14_base", "w14_up", "peninsula"});
    check("4 runs published under room_1", a.under(room_1) == 4);
    check("same room next cycle -> no release", not a.release().has_value() and a.under(room_1) == 4);

    a.working = 0;   // nobody has said this cycle: never a release
    check("working room unknown -> no release", not a.release().has_value() and a.under(room_1) == 4);

    // The crossing. Order as in compute(): follow the room, release, build, publish.
    a.working = room_2;
    const auto released = a.release();
    check("crossing -> released 4 runs", released == std::optional<int>{4});
    check("crossing -> nothing left under room_1", a.under(room_1) == 0);
    check("crossing -> model reset and unbound", not a.built and a.bound == 0 and a.resets == 1 and a.nodes.empty());

    a.build_if_possible(false);   // room_2 has no usable outline yet: births WAIT
    a.publish({"w13_base", "w14_base", "w14_up", "peninsula"});
    check("unbound -> nothing published under room_2", a.under(room_2) == 0 and a.graph.empty());
    check("released once only", not a.release().has_value() and a.resets == 1);

    a.build_if_possible(true);    // room_2 outline arrives: rebuilt over room_2's walls
    check("rebuilt -> bound to room_2", a.built and a.bound == room_2);
    a.publish({"w5_base"});
    check("a run seen in room_2 goes under room_2", a.under(room_2) == 1 and a.under(room_1) == 0);

    std::printf("kitchen_room_test: %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
